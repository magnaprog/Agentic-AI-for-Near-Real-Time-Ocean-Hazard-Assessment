"""Pipeline worker service entrypoint.

Consumes records from Kafka ``raw.observations`` topic, buffers them in
time windows, and drives ingest, QC audit metadata, anomaly scoring, FSM
transitions, and fail-closed ABSTAIN routing. Live Scenario inversion,
Verification, and Report generation are deferred (the worker abstains at
the assessment stages; the offline scripts run the full pipeline).

When Kafka is not configured, validates the pipeline graph spec and
waits for shutdown (development/test mode).

Data flow::

    Kafka -> poll_batch -> classify by source_type -> StationBufferManager
      -> per-station anomaly scoring (AnomalyAgent.process_station_data)
      -> build AnomalyAssessment dict -> run_pipeline_sync(FSM)
      -> build + persist OceanEvidenceAssessment (active event only)
      -> commit offsets

Seismic events are handled separately: they update the FSM via
``evaluate_seismic_trigger()`` and refresh the AnomalyAgent's
seismic context for Rayleigh wave discrimination.

Usage::

    python -m hazard_assessment.workers.pipeline_runner

"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final
from uuid import UUID, uuid4

import numpy as np

from hazard_assessment import __version__
from hazard_assessment.audit.logger import AuditEntry
from hazard_assessment.messaging.consumer import KafkaConsumer
from hazard_assessment.schemas.observation import is_dart_station_id
from hazard_assessment.schemas.ocean_evidence import (
    ASSESSMENT_SCHEMA_VERSION,
    CheckpointSource,
    StationScoringStatus,
)
from hazard_assessment.schemas.ocean_evidence_hashing import (
    KafkaMessageCoordinate,
    TransportProvenance,
    derive_live_checkpoint_id,
    finalize_assessment_hashes,
    transport_provenance_hash,
)
from hazard_assessment.workers.assessment_builder import (
    StationAttemptResult,
    build_detector_config,
    build_ocean_evidence_assessment,
    classify_calibration_status,
)
from hazard_assessment.workers.reviewer_packet import (
    RENDERER_VERSION,
    render_reviewer_packet,
)
from hazard_assessment.workers.station_buffer import RetainedSampleQC

if TYPE_CHECKING:
    from hazard_assessment.orchestrator.states import SeismicIdentity

logger = logging.getLogger("hazard_assessment.workers.pipeline")

RAW_OBSERVATIONS_TOPIC = "raw.observations"

# Tolerance for source clocks running ahead of this host. Observation and
# seismic timestamps describe events that have already happened, so a record
# dated beyond this is not a late arrival, it is a wrong clock or a corrupted
# row: a .dart line whose year field reads "99" parses as 2099, and the parser
# accepts two-digit years by design. Fifteen minutes covers real skew and the
# longest normal sample interval.
FUTURE_TOLERANCE_SEC: Final[int] = 900

# Buffer window: seconds to accumulate records before processing.
# Standard mode: 60s aligns with DART standard-mode polling.
# Event mode: 15s for rapid assessment during event-mode sampling.
BUFFER_WINDOW_STANDARD_SEC = 60
BUFFER_WINDOW_EVENT_SEC = 15

# Raw-payload provenance hashes are recorded only when they match the canonical
# 64-hex SHA-256 form (the same pattern InputRef.sha256 enforces), so a
# malformed value never reaches the escalation packet.
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")

# External seismic provider bound to event identity. The only
# deployed seismic connector polls the USGS FDSN event feed
# (ingest/seismic.py::SeismicIngestConnector); its records carry no provider
# field, so the worker names the provider here. A second provider would need
# its own connector and a provider field on the record.
_SEISMIC_PROVIDER = "usgs"

# Cap on distinct observation provenance entries recorded per event. A long
# multi-hour event at high cadence would otherwise accumulate thousands of
# input_provenance entries; this bounds the audit-trail volume (and the per-event
# dedup set) and matches the escalation packet's read cap, beyond which more refs
# add no review value.
_MAX_PROVENANCE_PER_EVENT = 1000

# Pacific tsunamigenic zones for seismic trigger evaluation.
# Uses zone identifiers that map to geographic bounding boxes
# in classify_seismic_zone().  This is the set used
# for Pacific basin operations.
PACIFIC_TSUNAMIGENIC_ZONES: set[str] = {
    "cascadia", "alaska_aleutian", "japan", "kuril", "kamchatka",
    "maule", "peru_chile", "central_america",
    "solomon_islands", "tonga_kermadec", "new_zealand", "philippines",
    "indonesia", "pacific_rim",
}


def classify_seismic_zone(lat: float, lon: float) -> str:
    """Map earthquake epicenter to a tsunamigenic zone identifier.

    Uses a simple geographic bounding-box classifier for the Pacific
    basin.  This is intentionally coarse - the zone identifier is only
    used to gate ``evaluate_seismic_trigger()`` (which checks magnitude
    and zone membership).  NOTE: a misclassified zone drops the trigger
    entirely.  The seismic trigger is the ONLY path out of IDLE - there
    is no independent DART-only or anomaly-only FSM entry (anomaly
    scores are ignored in IDLE and the DART event-mode latch requires
    an active event context) - so an event outside these boxes is not
    tracked at all.  This is accepted because the system is explicitly
    Pacific-scoped; extending coverage means extending the zone boxes,
    not relying on a non-existent fallback.

    Returns:
        Zone identifier string matching PACIFIC_TSUNAMIGENIC_ZONES.
    """
    # Japan / Tohoku / Kuril
    if 30.0 <= lat <= 50.0 and 130.0 <= lon <= 160.0:
        return "japan"
    if 44.0 <= lat <= 56.0 and 145.0 <= lon <= 165.0:
        return "kuril"
    # Kamchatka
    if 50.0 <= lat <= 62.0 and 155.0 <= lon <= 175.0:
        return "kamchatka"
    # Alaska / Aleutian
    if 50.0 <= lat <= 65.0 and (-180.0 <= lon <= -140.0 or 170.0 <= lon <= 180.0):
        return "alaska_aleutian"
    # Cascadia / Pacific NW
    if 40.0 <= lat <= 52.0 and -130.0 <= lon <= -120.0:
        return "cascadia"
    # Peru / Chile (Maule region more specific)
    if -45.0 <= lat <= -30.0 and -80.0 <= lon <= -68.0:
        return "maule"
    if -30.0 <= lat <= 0.0 and -85.0 <= lon <= -68.0:
        return "peru_chile"
    # Central America
    if 5.0 <= lat <= 20.0 and -110.0 <= lon <= -75.0:
        return "central_america"
    # Tonga / Kermadec
    if -35.0 <= lat <= -15.0 and -180.0 <= lon <= -170.0:
        return "tonga_kermadec"
    # Solomon Islands
    if -12.0 <= lat <= -4.0 and 150.0 <= lon <= 165.0:
        return "solomon_islands"
    # Indonesia
    if -10.0 <= lat <= 5.0 and 95.0 <= lon <= 140.0:
        return "indonesia"
    # Philippines
    if 5.0 <= lat <= 20.0 and 118.0 <= lon <= 130.0:
        return "philippines"
    # New Zealand
    if -50.0 <= lat <= -33.0 and 165.0 <= lon <= 180.0:
        return "new_zealand"
    # Generic Pacific Rim fallback for anything near the Ring of Fire
    if abs(lat) <= 60.0 and (lon > 100.0 or lon < -60.0):
        return "pacific_rim"
    return "unknown"


@dataclass
class CheckpointTransport:
    """Kafka transport metadata accumulated since the last checkpoint.

    Holds the coordinates of every message consumed since the previous
    ``_process_buffer`` call: decoded envelopes and transport-rejected
    (undecodable) messages alike, so checkpoint identity covers the
    complete consumed offset manifest.
    Rejected-only polls leave the record buffer empty, so their
    coordinates simply fold into the next processed checkpoint; no
    bound is placed on that accumulation because sustained
    rejected-only consumption means total ingest-pipeline failure,
    which is already logged per message.
    """

    consumer_group: str
    messages: list[KafkaMessageCoordinate] = field(default_factory=list)

    def rejected_markers(self) -> list[tuple[str, int, int]]:
        return [
            (m.topic, m.partition, m.offset)
            for m in self.messages
            if m.transport_rejected
        ]

    def offset_ranges(self) -> list[tuple[str, int, int, int]]:
        """Per-(topic, partition) consumed offset ranges, decoded plus rejected."""
        bounds: dict[tuple[str, int], tuple[int, int]] = {}
        for m in self.messages:
            key = (m.topic, m.partition)
            lo, hi = bounds.get(key, (m.offset, m.offset))
            bounds[key] = (min(lo, m.offset), max(hi, m.offset))
        return [
            (topic, partition, lo, hi)
            for (topic, partition), (lo, hi) in bounds.items()
        ]

    def checkpoint_id(self) -> str | None:
        """Deterministic live checkpoint identity, or None when empty."""
        if not self.messages:
            return None
        return derive_live_checkpoint_id(
            self.consumer_group,
            self.offset_ranges(),
            self.rejected_markers(),
        )


class PipelineWorkerState:
    """Mutable state shared across the pipeline worker's lifecycle.

    Encapsulates the AnomalyAgent, FSM, CalibrationManager, and
    StationBufferManager so ``_process_buffer()`` can be a pure
    function of ``(kafka_buffer, worker_state)``.
    """

    def __init__(
        self,
        calibration_dir: str | None = None,
        db_client: Any | None = None,
    ) -> None:
        from hazard_assessment.agents.anomaly_agent import AnomalyAgent
        from hazard_assessment.agents.qc_agent import QCAgent
        from hazard_assessment.audit.logger import AuditLogger
        from hazard_assessment.config.settings import ThresholdSettings
        from hazard_assessment.orchestrator.states import FSMOrchestrator

        from .calibration import CalibrationManager
        from .station_buffer import StationBufferManager

        self.agent = AnomalyAgent()
        self.qc_agent = QCAgent()
        self.threshold_settings = ThresholdSettings()
        self.db_client = db_client
        # One DB-backed audit logger serves both the FSM's transition audit
        # (write-audit-before-state-change) and the pipeline-node audit
        # entries.  Without an audit_writer the FSM skips the transition
        # audit entirely, leaving worker-driven escalations with no trail.
        self.audit_logger = AuditLogger(db_client=db_client)
        # Event whose reviewer packet has been confirmed present, so the
        # presence query stops repeating on every later checkpoint.
        self.packet_confirmed_event_id: Any | None = None
        # Event whose missing reviewer packet has already been disclosed, so
        # the audit trail gets one entry rather than one per checkpoint.
        self.packet_gap_disclosed_event_id: Any | None = None
        self.fsm = FSMOrchestrator(
            thresholds=self.threshold_settings.to_threshold_config(),
            audit_writer=self.audit_logger,
            db_client=db_client,
        )
        self.station_buffers = StationBufferManager()
        self.calibration = CalibrationManager()
        self.seismic_events: list[Any] = []

        # Per-event dedup for observation input-provenance, so the rolling
        # station buffer's repeated records do not spam the audit trail with the
        # same hash. Reset when the active event changes.
        self.recorded_provenance_hashes: set[str] = set()
        self.provenance_event_id: Any = None
        self.provenance_capped: bool = False

        # Per-event accumulation of DART stations whose event-mode records were
        # accepted while THIS event was active (drives dart_confirmation and the
        # context station lists). Reset when the active event changes, so a
        # prior event's activations cannot leak into a new event.
        self.event_mode_station_set: set[str] = set()
        self.event_mode_event_id: Any = None

        # Identity of the most recently processed checkpoint, derived from the
        # batch's Kafka transport metadata. None when the batch
        # carried no transport metadata (direct _process_buffer callers).
        self.last_checkpoint_id: str | None = None

        # Identity of THIS worker process, embedded in assessment transport
        # provenance and checkpoint-attempt records so crash-and-restart
        # sequences stay distinguishable in the durable trail.
        self.run_id = uuid4()

        # Recover FSM state from DB if available
        if db_client is not None and self.fsm.recover_from_db():
            recovered_ctx = self.fsm.event_context
            if recovered_ctx is not None:
                self.event_mode_station_set = set(
                    recovered_ctx.stations_in_event_mode
                )
                self.event_mode_event_id = recovered_ctx.event_id

        # Load calibration data if directory provided
        if calibration_dir:
            cal_path = Path(calibration_dir)
            n_loaded = self.calibration.load_directory(cal_path)
            if n_loaded > 0:
                self.calibration.apply_to_agent(self.agent)
            logger.info("Calibration: %d stations loaded from %s", n_loaded, cal_path)


async def _run() -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    # Validate that the pipeline graph specification loads without errors.
    from hazard_assessment.orchestrator.pipeline import build_pipeline_graph

    graph = build_pipeline_graph()
    n_static = len(graph.get("edges", []))
    n_conditional = sum(
        len(v.get("branches", {}))
        for v in graph.get("conditional_edges", {}).values()
    )
    logger.info(
        "Pipeline graph loaded: %d nodes, %d edges (%d static + %d conditional)",
        len(graph.get("nodes", [])),
        n_static + n_conditional,
        n_static,
        n_conditional,
    )

    # Initialize DB client for FSM state persistence (optional)
    db_client = None
    db_host = os.getenv("DB_HOST", "").strip()
    if db_host:
        try:
            from hazard_assessment.storage.client import ClientConfig, DatabaseClient

            # Migration 009 provisions the dedicated pipeline_worker role
            # (FSM state, audit, lineage, assessment, and attempt-table
            # writes) and revokes the worker-facing lineage INSERT from
            # orchestrator_writer, so the worker connects as its own role.
            config = ClientConfig.from_env(role="pipeline_worker")
            db_client = DatabaseClient(config)
            if not db_client.is_connected:
                db_client = None
        except Exception:
            logger.exception("Failed to connect to database")

    # Initialize pipeline worker state (agent, FSM, calibration, buffers).
    # db_client is passed to FSM for state persistence.
    calibration_dir = os.getenv("CALIBRATION_DIR", "").strip() or None
    worker = PipelineWorkerState(
        calibration_dir=calibration_dir,
        db_client=db_client,
    )

    # Initialize Kafka consumer
    consumer = KafkaConsumer(topics=[RAW_OBSERVATIONS_TOPIC])

    if not consumer.is_connected:
        logger.info("Pipeline worker ready - no Kafka broker, awaiting shutdown")
        await stop.wait()
        logger.info("Pipeline worker shutting down")
        return

    logger.info("Pipeline worker consuming from %s", RAW_OBSERVATIONS_TOPIC)

    buffer: dict[str, list[dict[str, Any]]] = {}  # keyed by source_type:station_id
    transport = CheckpointTransport(consumer_group=consumer.group_id)
    window_start = time.monotonic()
    buffer_window = BUFFER_WINDOW_STANDARD_SEC

    try:
        while not stop.is_set():
            # Poll for records. Decoded and transport-rejected coordinates are
            # both retained so the checkpoint's consumed offset manifest stays
            # complete.
            batch = consumer.poll_batch(timeout=1.0, max_records=50)
            for rec in batch.records:
                value = rec.get("value", {})
                key = rec.get("key", "unknown")
                buffer.setdefault(key, []).append(value)
                message_id = value.get("message_id", "") if isinstance(value, dict) else ""
                transport.messages.append(KafkaMessageCoordinate(
                    topic=rec["topic"],
                    partition=rec["partition"],
                    offset=rec["offset"],
                    timestamp_type=rec.get("timestamp_type", ""),
                    timestamp_ms=(
                        rec["timestamp"] if rec.get("timestamp", -1) >= 0 else None
                    ),
                    application_message_id=str(message_id or ""),
                    transport_rejected=False,
                ))
            for rej in batch.rejected:
                transport.messages.append(KafkaMessageCoordinate(
                    topic=rej.topic,
                    partition=rej.partition,
                    offset=rej.offset,
                    transport_rejected=True,
                ))

            # Keep retained station windows wall-clock current before choosing
            # the buffering cadence. Otherwise a stale DART event-mode flag can
            # keep the worker on the faster cadence until the next processed
            # batch, even though no current evidence remains.
            worker.station_buffers.trim_all(now_epoch=time.time())

            # Coverage is evaluated on every tick, not only after a processed
            # batch: total data silence is the worst coverage case and produces
            # no batch at all, so a batch-only evaluation would leave the flag
            # reading healthy exactly when the network went dark. trim_all
            # above has just aged the windows, so this reads current coverage.
            # The flag never gates FSM transitions.
            worker.fsm.evaluate_coverage(_count_usable_dart_stations(worker))

            # Switch to event-mode buffer window if any DART station
            # is in event mode (faster assessment cadence)
            if worker.station_buffers.stations_in_event_mode():
                buffer_window = BUFFER_WINDOW_EVENT_SEC
            else:
                buffer_window = BUFFER_WINDOW_STANDARD_SEC

            # Check the FSM monitor timeout on every quiet tick: it must fire
            # during data silence (the docstring contract is "called
            # periodically"), so a seismic record arriving after a long quiet
            # MONITOR finds the already-timed-out IDLE state instead of being
            # dropped against a stale MONITOR. Skipped while records are
            # buffered: timing out BEFORE scoring would discard buffered
            # evidence at the boundary (an above-T1 wave observation would be
            # scored in IDLE and ignored); the post-processing check below
            # covers that path.
            _check_monitor_timeout_when_quiet(worker, buffer)

            # Check if buffer window has elapsed
            elapsed = time.monotonic() - window_start
            if elapsed >= buffer_window and buffer:
                total_records = sum(len(v) for v in buffer.values())
                n_stations = len(buffer)
                logger.info(
                    "Processing buffer: %d records from %d station(s) "
                    "(window=%.0fs)",
                    total_records,
                    n_stations,
                    elapsed,
                )

                # Use one wall-clock time for stale trimming and stale
                # current-batch observation gating inside _process_buffer.
                # This batch's records are still only in the Kafka buffer dict,
                # so stale retained windows can be removed without discarding
                # current evidence.
                now_epoch = time.time()
                _process_buffer(
                    buffer, worker, now_epoch=now_epoch, transport=transport
                )

                # Commit offsets after successful processing
                consumer.commit()

                # Timeout check AFTER buffered evidence was scored (the
                # quiet-tick check above is skipped while records buffer).
                worker.fsm.check_monitor_timeout()

                # Reset Kafka batch buffer and transport accumulator; the
                # next checkpoint starts a fresh consumed offset manifest.
                buffer.clear()
                transport = CheckpointTransport(consumer_group=consumer.group_id)
                window_start = time.monotonic()

            # Brief sleep to avoid busy-loop when no records
            await asyncio.sleep(0.1)

    finally:
        consumer.close()
        if db_client is not None:
            db_client.close()

    logger.info("Pipeline worker shut down cleanly")


def _count_usable_dart_stations(worker: PipelineWorkerState) -> int:
    """Count DART stations currently carrying QC-usable data.

    A station counts when its retained window holds at least one sample whose
    per-record QC verdict was usable. Samples buffered with ``qc=None`` are
    QC-unevaluated, and this follows the rule the assessment side already
    applies to them (see ``assessment_builder``): unevaluated is not counted as
    usable, so coverage is never inferred from records QC never saw.

    Only DART is counted. The flag it feeds is about the triangulation
    minimum, which coastal gauges do not contribute to.
    """
    usable = 0
    for source_type, station_id in worker.station_buffers.station_keys():
        if source_type != "dart":
            continue
        window = worker.station_buffers.get_window(station_id, source_type)
        if window is None:
            continue
        if any(
            sample.qc is not None and sample.qc.usable
            for sample in window.retained_samples()
        ):
            usable += 1
    return usable


def _unwrap_record(payload: dict[str, Any]) -> dict[str, Any]:
    """Return the canonical record from a Kafka envelope.

    The ingest producer wraps each record as
    ``{"schema_version", "timestamp", "source_type", "record": {...}}``
    (see ``messaging/producer.py`` and ``workers/ingest_runner.py``); the
    consumer buffers the whole envelope, so the observation fields live
    under the ``"record"`` key.  Return that inner dict, falling back to the
    payload unchanged when it is already flat (so unit tests that feed flat
    records, and any future flat producer, keep working).
    """
    inner = payload.get("record")
    return inner if isinstance(inner, dict) else payload


def _build_seismic_identity(
    record: dict[str, Any],
    *,
    now_epoch: float | None = None,
    kafka_positions: dict[str, tuple[int, int]] | None = None,
) -> SeismicIdentity | None:
    """Build the SeismicIdentity of one unwrapped seismic record.

    Returns None when the record carries no external event ID (identity
    cannot bind or match without one). The provider update time is kept
    only when it parses as a timezone-aware timestamp no later than local
    receipt (``now_epoch`` when the live worker supplies it, wall clock
    otherwise); a missing, malformed, naive, or post-receipt-future value
    becomes None, which the FSM treats as provenance that can never
    silently supersede the latest valid revision. The revision ID is the
    connector's deterministic per-revision source_id (external event ID
    plus provider update time); the Kafka receipt position is resolved
    from this batch's transport coordinates via that same ID.
    """
    from hazard_assessment.orchestrator.states import SeismicIdentity
    from hazard_assessment.schemas.ocean_evidence import SeismicContextClass

    external_event_id = record.get("event_id")
    if not isinstance(external_event_id, str) or not external_event_id:
        return None

    revision_id = record.get("source_id")
    if not isinstance(revision_id, str):
        revision_id = ""

    sha = record.get("payload_sha256")
    revision_sha256 = (
        sha if isinstance(sha, str) and _SHA256_RE.match(sha) else ""
    )

    provider_updated: datetime | None = None
    raw_updated = record.get("updated_timestamp")
    if isinstance(raw_updated, str) and raw_updated:
        try:
            parsed = datetime.fromisoformat(raw_updated)
        except ValueError:
            parsed = None
        if parsed is not None and parsed.tzinfo is not None:
            parsed = parsed.astimezone(UTC)
            receipt_epoch = (
                now_epoch if now_epoch is not None
                else datetime.now(UTC).timestamp()
            )
            if parsed.timestamp() <= receipt_epoch:
                provider_updated = parsed

    position = (
        kafka_positions.get(revision_id)
        if kafka_positions is not None and revision_id
        else None
    )

    return SeismicIdentity(
        provider=_SEISMIC_PROVIDER,
        external_event_id=external_event_id,
        revision_id=revision_id,
        revision_sha256=revision_sha256,
        provider_updated_utc=provider_updated,
        kafka_partition=position[0] if position is not None else None,
        kafka_offset=position[1] if position is not None else None,
        context_class=SeismicContextClass.LIVE_RECEIPT_ORDERED.value,
    )


def _ingest_seismic_record(
    record: dict[str, Any],
    worker: PipelineWorkerState,
    *,
    now_epoch: float | None = None,
    trace_id: Any = None,
    kafka_positions: dict[str, tuple[int, int]] | None = None,
) -> None:
    """Process a seismic event record: update agent context and trigger FSM.

    Seismic records are handled differently from DART/CO-OPS - they don't
    go through anomaly scoring.  Instead they:
    1. Offer the record to the FSM as a revision of the active external
       event; matching and admissibility ordering live in
       ``update_seismic_revision``
    2. Update the AnomalyAgent's seismic event list (for Rayleigh filtering)
    3. Potentially trigger FSM IDLE -> MONITOR transition, binding the
       external identity to the new event context

    When ``now_epoch`` is provided by the live worker, a backlog record older
    than the 6h seismic-context horizon is NOT added to the anomaly agent's
    Rayleigh context, but the FSM trigger is still evaluated regardless of
    age (deliberate fail-safe; see the policy comment at the trigger call).
    """
    from hazard_assessment.agents.anomaly_detection import SeismicEvent

    record = _unwrap_record(record)
    magnitude = record.get("magnitude")
    latitude = record.get("latitude")
    longitude = record.get("longitude")
    depth_km = record.get("depth_km")
    event_id = record.get("event_id", "unknown")

    # Revision handling comes BEFORE the trigger-field guard: a validated,
    # locally received revision of the ACTIVE external event advances the
    # latest admissible revision even when it lacks magnitude or
    # coordinates, which are trigger inputs rather than identity inputs.
    # With no active or matching event this is a no-op.
    identity = _build_seismic_identity(
        record, now_epoch=now_epoch, kafka_positions=kafka_positions
    )
    if identity is not None:
        worker.fsm.update_seismic_revision(identity)

    if magnitude is None or latitude is None or longitude is None:
        logger.debug("Skipping seismic record with missing fields: %s", event_id)
        return

    # Parse origin time from source_timestamp
    origin_time_str = record.get("source_timestamp")
    if origin_time_str is None:
        return
    try:
        origin_time = datetime.fromisoformat(origin_time_str)
    except (ValueError, TypeError):
        logger.warning("Invalid seismic origin time: %s", origin_time_str)
        return

    # Ensure timezone-aware (SeismicEvent.__post_init__ requires it), then
    # normalize to UTC: trigger_time_utc is serialized and rendered as UTC
    # downstream (Mission Control appends "Z"), so an aware non-UTC offset
    # must not survive this boundary.
    if origin_time.tzinfo is None:
        origin_time = origin_time.replace(tzinfo=UTC)
    else:
        origin_time = origin_time.astimezone(UTC)

    # A record from the future is as malformed as one with an unparseable
    # time, and is rejected here rather than reaching the FSM. Two things went
    # wrong without this. The prune filter keeps events whose age is under six
    # hours, and a future origin has a negative age, so the event was never
    # pruned and polluted the Rayleigh discrimination context until the worker
    # restarted. And the DART event-mode latch only accepts records timestamped
    # at or after the origin, so a future-dated trigger suppressed the evidence
    # latch for every real event-mode row that followed: fail-dangerous, in a
    # path whose whole purpose is to push toward human review.
    reference_epoch = now_epoch if now_epoch is not None else datetime.now(UTC).timestamp()
    if origin_time.timestamp() > reference_epoch + FUTURE_TOLERANCE_SEC:
        logger.warning(
            "Rejecting seismic record %s: origin %s is more than %ds ahead of this host",
            event_id,
            origin_time.isoformat(),
            FUTURE_TOLERANCE_SEC,
        )
        return

    # Create SeismicEvent and add to agent's list
    try:
        event = SeismicEvent(
            event_id=event_id,
            magnitude=magnitude,
            origin_time=origin_time,
            latitude=latitude,
            longitude=longitude,
        )
    except (ValueError, TypeError):
        logger.warning("Invalid seismic event data for %s", event_id)
        return

    # Update seismic events list (keep events from last 6 hours). One time
    # reference for both pruning and the stale-append gate: the live worker's
    # now_epoch when provided, wall clock otherwise, so the two checks cannot
    # disagree about what "stale" means.
    ref_epoch = now_epoch if now_epoch is not None else datetime.now(UTC).timestamp()
    worker.seismic_events = [
        e for e in worker.seismic_events
        if (ref_epoch - e.origin_time.timestamp()) < 6 * 3600
    ]
    # On the live path, do not add a backlog record already older than the
    # same 6h horizon: Rayleigh-wave discrimination only makes sense for
    # quakes whose surface waves could still be arriving, and the event would
    # be pruned on the next seismic message anyway.
    if now_epoch is None or (now_epoch - origin_time.timestamp()) < 6 * 3600:
        worker.seismic_events.append(event)
    # Sync the agent's private context copy UNCONDITIONALLY: pruning alone
    # changes the list, and skipping the sync in the stale-skip branch would
    # leave the anomaly agent's Rayleigh/quiet checks reading a cached event
    # the worker already pruned.
    worker.agent.update_seismic_events(worker.seismic_events)

    logger.info(
        "Seismic event: M%.1f at (%.2f, %.2f) - %s",
        magnitude, latitude, longitude, event_id,
    )

    # A further revision of the ACTIVE external event is not a concurrent-
    # event trigger: the identity handling above already latched whatever
    # was admissible, and evaluate_seismic_trigger would be a guaranteed
    # no-op (FSM is non-IDLE whenever a context exists) that logs a
    # misleading concurrent-event WARNING for every provider revision of a
    # live event. Skip it so that warning keeps meaning what it says.
    # Records matching no active identity (new events, unrelated events,
    # events created without identity) still evaluate normally.
    active_ctx = worker.fsm.event_context
    if (
        identity is not None
        and active_ctx is not None
        and active_ctx.seismic_provider == identity.provider
        and active_ctx.external_event_id == identity.external_event_id
    ):
        logger.debug(
            "Seismic record %s is a revision of the active event %s; "
            "trigger evaluation skipped",
            identity.revision_id,
            identity.external_event_id,
        )
        return

    # Evaluate FSM seismic trigger (IDLE -> MONITOR), deliberately with NO age
    # gate: a large shallow quake from a downtime backlog (e.g. a 13h-old M8)
    # escalates straight to a reviewable ESCALATE even though its origin is
    # past the monitor-timeout window. This is the fail-safe policy choice:
    # far-field tsunami waves can still be propagating 12-24h after origin
    # (cross-basin travel times), so forcing human review of an old major
    # event is safer than silently dropping it. A moderate old event enters
    # MONITOR and the origin-based monitor timeout returns it to IDLE on the
    # next check. Pinned by test_stale_seismic_backlog_policy.
    zone = classify_seismic_zone(latitude, longitude)
    transition = worker.fsm.evaluate_seismic_trigger(
        magnitude=magnitude,
        region=zone,
        epicenter_lat=latitude,
        epicenter_lon=longitude,
        tsunamigenic_zones=PACIFIC_TSUNAMIGENIC_ZONES,
        depth_km=depth_km,
        origin_time_utc=origin_time,
        trace_id=trace_id,
        # Binds provider, external event ID, and the immutable trigger
        # revision at event creation. None only when the
        # record carried no external event ID.
        seismic_identity=identity,
    )

    # Record raw-input provenance for the escalation packet. Only when this
    # record actually created the event (a transition fired, so the event
    # context is the one this record triggered) and the payload hash is a real
    # canonical SHA-256: link (source, source record id, hash) to the FSM event
    # id so generate_escalation_packet can assemble real input_refs from the
    # durable audit trail. No fabrication - a missing/malformed hash is skipped.
    ctx = worker.fsm.event_context
    payload_sha256 = record.get("payload_sha256")
    if (
        transition is not None
        and ctx is not None
        and isinstance(payload_sha256, str)
        and _SHA256_RE.match(payload_sha256)
    ):
        worker.audit_logger.append(AuditEntry(
            event_id=ctx.event_id,
            # Distinct event_type so the escalation packet ALWAYS includes the
            # seismic trigger's provenance (the decision-critical input) and it
            # is never pushed out of the capped observation-provenance read.
            trace_id=trace_id,
            event_type="seismic_provenance",
            producer="ingest_seismic",
            data={
                "source": "seismic",
                "record_id": str(event_id),
                "sha256": payload_sha256,
            },
        ))


def _emit_seismic_only_abstain(
    worker: PipelineWorkerState, *, trace_id: Any = None
) -> None:
    """Emit a fail-closed ABSTAIN artifact for a seismic-only FSM transition.

    A seismic trigger can move the FSM to MONITOR or ESCALATE with no scored
    station window (DART/CO-OPS data arrives 15-30 min after the quake). The
    deployed worker does not run live scenario inversion or verification, so it
    must abstain here rather than synthesize an under-validated assessment.
    Without this the transition produced only state_transition audit entries and
    no reviewable ABSTAIN artifact.

    The artifact is Tier-1 ABSTAIN (non-distributable by construction) and is
    written to the shared audit trail, so it surfaces via /api/audit and
    /api/lineage exactly like a pipeline-driven ABSTAIN.
    """
    ctx = worker.fsm.event_context
    event_id = ctx.event_id if ctx is not None else None
    fsm_state = worker.fsm.state.value
    reason = (
        f"Seismic-triggered transition to {fsm_state} with no scored station "
        "window yet (DART/CO-OPS records may be present but insufficient to "
        "score). The deployed worker does not run live scenario inversion or "
        "verification, so it abstains and escalates for human situational "
        "awareness while awaiting sufficient waveform data."
    )

    status = "ABSTAIN"
    try:
        from hazard_assessment.agents.assessment_formatter import format_abstain
        assessment = format_abstain(
            event_id=event_id,
            fsm_state=fsm_state,
            abstain_reason=reason,
            producer="pipeline_worker",
        )
        status = assessment.status.value
    except Exception:
        # Formatting the typed artifact must never suppress the audit record of
        # the ABSTAIN decision, which is the durable, reviewable output.
        logger.exception("Failed to format seismic-only ABSTAIN artifact")

    worker.audit_logger.append(AuditEntry(
        event_id=event_id,
        trace_id=trace_id,
        event_type="abstain_triggered",
        producer="pipeline_worker",
        data={
            "reason": reason,
            "fsm_state": fsm_state,
            "trigger": "seismic_only",
            "status": status,
        },
    ))
    from hazard_assessment.telemetry.metrics import record_abstain
    record_abstain()
    logger.info(
        "Seismic-only transition to %s with no scored window: emitted ABSTAIN",
        fsm_state,
    )


def _check_monitor_timeout_when_quiet(
    worker: PipelineWorkerState, buffer: dict[str, list[dict[str, Any]]]
) -> None:
    """Fire the FSM monitor timeout only when no records are buffered.

    During data silence the timeout must still fire (so a later seismic
    trigger is not dropped against a stale MONITOR), but while records are
    buffered the event must not be timed out before that evidence is scored:
    an above-T1 wave observation arriving just before the boundary would
    otherwise be evaluated in IDLE and ignored. The caller re-checks the
    timeout after each processed batch.
    """
    if not buffer:
        worker.fsm.check_monitor_timeout()


def _record_observation_provenance(
    worker: PipelineWorkerState,
    source: str,
    station_id: str,
    payload_sha256: Any,
    source_id: Any = None,
) -> None:
    """Record raw-input provenance for an observation during an active event.

    When the FSM has an active event, link this observation's real payload hash
    (source, station id, sha256) to the event id so generate_escalation_packet
    can assemble input_refs from the durable audit trail. Deduped per event so
    the rolling buffer's repeated records do not spam the audit trail; a missing
    or malformed hash is skipped (no fabrication). Observations outside an active
    event are not provenance for any escalation and are not recorded.
    """
    ctx = worker.fsm.event_context
    if ctx is None:
        return
    if not isinstance(payload_sha256, str) or not _SHA256_RE.match(payload_sha256):
        return
    if worker.provenance_event_id != ctx.event_id:
        # New active event: reset the per-event dedup set + cap marker.
        worker.provenance_event_id = ctx.event_id
        worker.recorded_provenance_hashes = set()
        worker.provenance_capped = False
    # Dedup is content-addressed by payload hash, deliberately: two records
    # with the same hash carry byte-identical raw content (the DART raw line
    # includes its timestamp), so one reference suffices for lineage.
    if payload_sha256 in worker.recorded_provenance_hashes:
        return
    if len(worker.recorded_provenance_hashes) >= _MAX_PROVENANCE_PER_EVENT:
        # Cap reached: record a one-time marker so the escalation packet can
        # disclose that observation provenance is incomplete for this event
        # (a read-side check cannot detect it, since the worker drops the excess
        # before it ever reaches the audit trail).
        if not worker.provenance_capped:
            worker.provenance_capped = True
            worker.audit_logger.append(AuditEntry(
                event_id=ctx.event_id,
                event_type="provenance_capped",
                producer=f"ingest_{source}",
                data={"cap": _MAX_PROVENANCE_PER_EVENT},
            ))
        return
    worker.recorded_provenance_hashes.add(payload_sha256)
    # record_id should identify the raw RECORD (the connector's source_id
    # carries station + timestamp + measurement type), not just the station;
    # fall back to the station id when a record arrives without one. The
    # InputRef schema pattern re-validates it at packet-build time, so a
    # malformed value is skipped there rather than fabricated here.
    record_id = source_id if isinstance(source_id, str) and source_id else station_id
    worker.audit_logger.append(AuditEntry(
        event_id=ctx.event_id,
        event_type="input_provenance",
        producer=f"ingest_{source}",
        data={
            "source": source,
            "record_id": record_id,
            "station_id": station_id,
            "sha256": payload_sha256,
        },
    ))


def _ingest_observation_records(
    key: str,
    records: list[dict[str, Any]],
    worker: PipelineWorkerState,
    *,
    now_epoch: float | None = None,
    qc_by_hash: dict[str, RetainedSampleQC] | None = None,
    admission_counts: dict[tuple[str, str], list[int]] | None = None,
) -> set[str]:
    """Ingest DART or CO-OPS observation records into station buffers.

    Parses Kafka envelope records and appends them to the appropriate
    StationWindow in the StationBufferManager. Returns the set of DART station
    ids that had a VALIDLY INGESTED event-mode record in this batch (so the
    caller can latch dart_confirmation only on accepted data, not on malformed
    records that were skipped before they entered the buffer/QC/scoring path).
    When ``now_epoch`` is provided by the live worker, records older than the
    wall-clock rolling window are skipped before buffering, provenance, or
    scoring.

    ``qc_by_hash`` maps a record's payload SHA-256 to the per-record QC
    already computed for this batch (runs QC before admission), so
    each accepted sample can carry its QC metadata into the retained
    window. Records without a canonical hash, or hashes QC never produced
    a report for, are buffered with ``qc=None``; the assessment side later
    accounts for them as QC-unevaluated rather than silently usable.

    ``admission_counts`` (when supplied) accumulates ``[attempted,
    admitted]`` record counts per (source_type, station_id) for
    assessment construction. Attempted covers every
    record offered under this station key, including records that fail
    parsing or the stale-window gate; admitted counts only the samples
    the buffer accepted.
    """
    parts = key.split(":", 1)
    source_type = parts[0] if len(parts) >= 2 else "unknown"
    station_id = parts[1] if len(parts) >= 2 else key
    # Station IDs are validated syntactically (five ASCII digits), NOT against
    # a station inventory. Deliberate: a hard-coded inventory goes stale as
    # NDBC deploys new buoys, and silently dropping a real station's data is
    # fail-dangerous, while an unknown-but-well-formed ID only pushes toward
    # human review and is processed without Rayleigh coordinates (graceful
    # None fallback). Kafka is an internal bus fed by validating connectors;
    # an attacker who can publish to it could use a real station ID anyway,
    # so an inventory check adds no security at this boundary.
    if source_type == "dart" and not is_dart_station_id(station_id):
        logger.warning("Skipping DART records with invalid station identifier")
        return set()

    counts: list[int] | None = None
    if admission_counts is not None and source_type in ("dart", "coops"):
        counts = admission_counts.setdefault((source_type, station_id), [0, 0])
        counts[0] += len(records)

    # Event-time relevance gate for the dart_confirmation latch: an event-mode
    # observation timestamped BEFORE the current event's seismic origin cannot
    # be this event's evidence (DART event mode physically follows its
    # triggering quake), so a delayed row from a prior event arriving after a
    # new trigger must not latch. The sample itself is still buffered for
    # scoring; only the event-mode evidence is gated. This is timestamp
    # scoping, not causal attribution: a prior event still emitting
    # POST-origin event-mode rows latches for the current event, deliberately
    # fail-safe (such rows are current physical state and only push toward
    # human review). True source association needs multi-event tracking.
    ctx = worker.fsm.event_context
    trigger_epoch = ctx.trigger_time_utc.timestamp() if ctx is not None else None

    event_mode_stations: set[str] = set()
    future_dated = 0
    for record in records:
        record = _unwrap_record(record)
        ts_str = record.get("source_timestamp")
        if ts_str is None:
            continue
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue
        # A naive timestamp would make .timestamp() interpret it in the HOST
        # local timezone, shifting the epoch by the host's UTC offset; treat
        # naive as UTC at this boundary (same rule as the seismic path).
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=UTC)
        epoch_sec = ts.timestamp()

        if now_epoch is not None:
            cutoff_epoch = now_epoch - worker.station_buffers.window_sec
            if epoch_sec < cutoff_epoch:
                continue
            # The window is bounded on both sides. A future-dated sample was
            # admitted before this check and became the window's latest_epoch,
            # which is the clock the assessment and the Rayleigh timing read,
            # and it survived every trim until wall-clock time caught up with
            # it. One row dated 2099 pinned a station for 73 years.
            if epoch_sec > now_epoch + FUTURE_TOLERANCE_SEC:
                future_dated += 1
                continue

        # Values come straight off Kafka, so type-check before any numeric
        # path (np.isfinite raises TypeError on a non-numeric value, which
        # would otherwise kill the consumer loop). bool is excluded because
        # JSON true/false would silently become a 1.0/0.0 sample.
        if source_type == "dart":
            height_m = record.get("height_m")
            if not isinstance(height_m, int | float) or isinstance(height_m, bool):
                continue
            # event_mode must be a real JSON boolean; any other type is
            # treated as False rather than skipping the record, because a
            # well-formed pressure sample still has scoring value and a
            # malformed flag must not produce a spurious confirmation latch.
            event_mode = record.get("event_mode") is True
            sha = record.get("payload_sha256")
            payload_hash = sha if isinstance(sha, str) and _SHA256_RE.match(sha) else None
            # DART measurement type (1, 2, or 3); anything malformed rides
            # as None rather than skipping a well-formed pressure sample.
            mt = record.get("measurement_type")
            measurement_type = (
                mt if isinstance(mt, int) and not isinstance(mt, bool) else None
            )
            qc = (
                qc_by_hash.get(payload_hash)
                if qc_by_hash is not None and payload_hash is not None
                else None
            )
            accepted = worker.station_buffers.append_dart(
                station_id, epoch_sec, float(height_m), event_mode=event_mode,
                payload_hash=payload_hash,
                measurement_type=measurement_type, qc=qc,
            )
            # Latch event-mode evidence only from samples that actually
            # entered the anomaly window: a missing-data sentinel (9999.0)
            # or non-finite value is not valid signal evidence even when
            # the record carries event_mode=True. The sample must also be
            # timestamped at or after the current event's seismic origin
            # (see the relevance gate above).
            if (
                event_mode
                and accepted
                and (trigger_epoch is None or epoch_sec >= trigger_epoch)
            ):
                event_mode_stations.add(station_id)
        elif source_type == "coops":
            water_level_m = record.get("water_level_m")
            if not isinstance(water_level_m, int | float) or isinstance(water_level_m, bool):
                continue
            sha = record.get("payload_sha256")
            payload_hash = sha if isinstance(sha, str) and _SHA256_RE.match(sha) else None
            raw_product = record.get("product")
            product = raw_product if isinstance(raw_product, str) and raw_product else None
            qc = (
                qc_by_hash.get(payload_hash)
                if qc_by_hash is not None and payload_hash is not None
                else None
            )
            accepted = worker.station_buffers.append_coops(
                station_id, epoch_sec, float(water_level_m),
                payload_hash=payload_hash,
                product=product, qc=qc,
            )
        else:
            continue

        if not accepted:
            continue
        if counts is not None:
            counts[1] += 1

        # The buffered record contributes to this station's anomaly window;
        # record event provenance when an event is active. The payload hash
        # rides WITH the sample (threaded into append above) so lineage rows
        # can reference every retained observation that gets scored.
        _record_observation_provenance(
            worker,
            source_type,
            station_id,
            record.get("payload_sha256"),
            source_id=record.get("source_id"),
        )
    if future_dated:
        logger.warning(
            "%s: dropped %d observation(s) dated more than %ds ahead of this host; "
            "check the source clock",
            key,
            future_dated,
            FUTURE_TOLERANCE_SEC,
        )

    return event_mode_stations


def _run_qc(
    key: str,
    records: list[dict[str, Any]],
    worker: PipelineWorkerState,
    *,
    trace_id: Any = None,
    companion_failures: list[str] | None = None,
) -> dict[str, RetainedSampleQC]:
    """Run QARTOD QC on a station's record batch and audit a summary.

    QC is audit metadata only: the anomaly agent intentionally processes raw
    values regardless of QC flags (genuine tsunami signals trip the
    rate-of-change and spike checks; see agents/qc_checks.py), so QC must NOT
    filter records out of the anomaly buffer.

    Returns per-record QC keyed by the record's payload SHA-256 (the stable
    source-record identity), computed once here so accepted samples
    can carry it into the station buffer without rerunning stateful QC at
    assessment time. Every QCReport carries a schema-validated canonical
    provenance hash, so the join is total over QC'd records; a record with
    a malformed or missing payload hash fails report construction, QC for
    the whole batch degrades to no metadata (empty map, pre-existing
    best-effort behavior), and ingestion continues with ``qc=None``
    samples. Duplicate hashes keep the last report; a duplicated hash is
    the same source record, and the buffer rejects duplicate timestamps
    anyway.

    When the live worker supplies the batch trace_id, the qc_complete audit
    entry carries a handoff_id and the station's accepted payload hashes,
    and a matching qc_report row is persisted to processed_features
    (best-effort, DB-optional), completing the get_provenance() lineage
    chain for QC output.

    ``companion_failures`` (when supplied) collects a label for each
    qc_report row that failed to persist, so the checkpoint's assessment
    can disclose incomplete companion lineage.
    """
    from hazard_assessment.agents.qc_agent import qc_observation_from_dict

    parts = key.split(":", 1)
    source_type = parts[0] if len(parts) >= 2 else "unknown"
    station_id = parts[1] if len(parts) >= 2 else key
    if source_type not in ("dart", "coops"):
        return {}
    if source_type == "dart" and not is_dart_station_id(station_id):
        logger.warning("Skipping DART QC with invalid station identifier")
        return {}

    observations: list[Any] = []
    qc_hashes: list[str] = []
    for record in records:
        record = _unwrap_record(record)
        if record.get("source_timestamp") is None:
            continue
        try:
            observations.append(
                qc_observation_from_dict(source_type, station_id, record)
            )
        except (ValueError, TypeError):
            logger.debug("QC: skipping unparseable %s record for %s", source_type, station_id)
            continue
        # Lineage hashes for the records QC ACTUALLY summarized (which can
        # include records the anomaly buffer rejects, e.g. sentinels).
        sha = record.get("payload_sha256")
        if isinstance(sha, str) and _SHA256_RE.match(sha):
            qc_hashes.append(sha)
    if not observations:
        return {}

    try:
        reports = worker.qc_agent.process_qc_observations(observations)
    except Exception:
        # QC is best-effort audit metadata; never let it break ingestion.
        logger.exception("QC failed for %s; continuing without a QC summary", key)
        return {}

    # Per-record QC keyed by payload hash. Reports come back in the QC
    # agent's internal sort order, so join on the report's provenance hash
    # (the observation's payload SHA-256, schema-validated canonical form)
    # rather than input position.
    qc_by_hash: dict[str, RetainedSampleQC] = {}
    for r in reports:
        qc_by_hash[r.provenance_hash] = RetainedSampleQC(
            usable=r.record_usable,
            flags=tuple(sorted(
                (name, int(flag))
                for name, flag in r.qartod_flags.model_dump().items()
            )),
            confidence=r.station_confidence,
            n_checks_evaluated=r.n_checks_evaluated,
        )

    n_usable = sum(1 for r in reports if r.record_usable)
    min_confidence = min(r.station_confidence for r in reports)
    summary = {
        "station_id": station_id,
        "source_type": source_type,
        "n_records": len(reports),
        "n_usable": n_usable,
        "min_station_confidence": round(min_confidence, 4),
        "n_zero_coverage": sum(1 for r in reports if r.n_checks_evaluated == 0),
    }
    ctx = worker.fsm.event_context
    event_id = ctx.event_id if ctx is not None else None
    if trace_id is None:
        worker.audit_logger.append(AuditEntry(
            event_type="qc_complete",
            producer="qc_agent",
            data=summary,
        ))
        return qc_by_hash
    # Live path: one handoff per station batch links the audit entry to the
    # processed_features row, and input_hashes link both to raw_observations.
    handoff_id = uuid4()
    input_hashes = qc_hashes
    worker.audit_logger.append(AuditEntry(
        event_id=event_id,
        trace_id=trace_id,
        event_type="qc_complete",
        producer="qc_agent",
        data={**summary, "handoff_id": str(handoff_id), "input_hashes": input_hashes},
    ))
    if worker.db_client is not None:
        from hazard_assessment.telemetry.metrics import record_lineage_persist_failure
        ok = worker.db_client.insert_processed_feature(
            feature_type="qc_report",
            producer_agent="qc_agent",
            handoff_id=handoff_id,
            trace_id=trace_id,
            payload=summary,
            source_refs=[{"sha256": h} for h in input_hashes],
            event_id=event_id,
            station_id=station_id,
        )
        if not ok:
            record_lineage_persist_failure()
            if companion_failures is not None:
                companion_failures.append(
                    f"qc_report:{source_type}:{station_id}"
                )
    return qc_by_hash


def _score_station_attempt(
    station_key: tuple[str, str],
    worker: PipelineWorkerState,
    *,
    n_records_attempted: int = 0,
    n_records_admitted: int = 0,
) -> tuple[StationAttemptResult, dict[str, Any] | None, bool]:
    """Score one (source_type, station_id) window and describe the attempt.

    Returns ``(attempt, assessment_dict, spatial_ran)``: every considered
    station yields one StationAttemptResult, and
    the AnomalyAssessment dict for run_pipeline_sync() exists only when
    scoring succeeded. ``spatial_ran`` reports whether the scorer
    produced a spatial-consistency result. Never raises: a scoring
    exception becomes a SCORING_FAILED attempt.
    """
    source_type, station_id = station_key
    window = worker.station_buffers.get_window(station_id, source_type)

    # Attempt-level calibration facts, source-gated exactly like the
    # scoring path: calibration CSVs are DART pressure series, so a
    # CO-OPS station with an equal identifier must not inherit a DART
    # station's tidal fit.
    cal = worker.calibration.get(station_id) if source_type == "dart" else None
    cal_span_minutes = cal.span_days * 1440.0 if cal is not None else None
    cal_sha256 = cal.source_sha256 if cal is not None else ""

    # The Rayleigh-wave check needs the station position; the registry
    # holds DART positions only, so the lookup is gated by the qualified
    # key's source. rayleigh_inputs_available describes input
    # availability for every attempt outcome, not whether the check ran.
    from hazard_assessment.data.station_coordinates import station_coordinates
    coords = station_coordinates(station_id) if source_type == "dart" else None
    rayleigh_inputs_available = coords is not None and worker.agent.has_seismic_context

    if window is None:
        # The station appeared in this batch but no record survived into
        # a retained window. The schema requires an empty snapshot here.
        return (
            StationAttemptResult(
                source=source_type,
                station_id=station_id,
                scoring_status=StationScoringStatus.NO_RETAINED_DATA,
                calibration_status=classify_calibration_status(
                    source=source_type,
                    scores=None,
                    calibration_span_minutes=cal_span_minutes,
                ),
                n_records_attempted=n_records_attempted,
                n_records_admitted=n_records_admitted,
                calibration_sha256=cal_sha256,
                rayleigh_inputs_available=rayleigh_inputs_available,
            ),
            None,
            False,
        )

    retained = tuple(window.retained_samples())
    arrays = window.to_arrays()
    if arrays is None:
        logger.debug("Station %s: insufficient data for scoring", station_id)
        return (
            StationAttemptResult(
                source=source_type,
                station_id=station_id,
                scoring_status=StationScoringStatus.INSUFFICIENT_RETAINED_DATA,
                calibration_status=classify_calibration_status(
                    source=source_type,
                    scores=None,
                    calibration_span_minutes=cal_span_minutes,
                ),
                n_records_attempted=n_records_attempted,
                n_records_admitted=n_records_admitted,
                retained_samples=retained,
                dart_window_event_mode=window.event_mode,
                calibration_sha256=cal_sha256,
                rayleigh_inputs_available=rayleigh_inputs_available,
            ),
            None,
            False,
        )

    times_hours, values, sampling_sec, event_t0 = arrays

    fit_times: np.ndarray | None = None
    fit_values: np.ndarray | None = None
    if cal is not None:
        fit_times = cal.times_hours
        fit_values = cal.values
        # Shift event times to the calibration epoch so harmonic phase
        # prediction is consistent.  Without this, the tidal coefficients
        # (fit on calibration epoch) produce wrong phase when applied to
        # event times measured from a different origin.
        offset_hours = (event_t0 - cal.t0_epoch) / 3600.0
        times_hours = times_hours + offset_hours

    # Determine if FSM is in MONITOR+ state for threshold suppression
    from hazard_assessment.orchestrator.states import SystemState
    fsm_monitoring = worker.fsm.state in (
        SystemState.MONITOR, SystemState.INVESTIGATE,
        SystemState.ASSESS, SystemState.ESCALATE,
    )

    # Unknown stations get (0.0, 0.0), which disables the Rayleigh check
    # for them rather than scoring it against a fabricated position.
    origin_lat, origin_lon = coords if coords is not None else (0.0, 0.0)

    # Use the latest observation time (not wall-clock) as the processing/spike
    # time so the Rayleigh arrival-timing check and the seismic-quiet window are
    # evaluated against data time. This also keeps the path replay-deterministic
    # and avoids Kafka/batch-delay skew.
    last_sample_utc = (
        datetime.fromtimestamp(window.latest_epoch, tz=UTC)
        if window.latest_epoch is not None
        else None
    )

    score_start = time.monotonic()
    try:
        scores, spatial_result = worker.agent.process_station_data(
            station_id=station_id,
            times_hours=times_hours,
            values=values,
            sampling_interval_sec=sampling_sec,
            source_type=window.source_type,
            fit_times_hours=fit_times,
            fit_values=fit_values,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
            processing_time=last_sample_utc,
            fsm_monitoring=fsm_monitoring,
        )
    except Exception as exc:
        logger.exception("Anomaly scoring failed for station %s", station_id)
        return (
            StationAttemptResult(
                source=source_type,
                station_id=station_id,
                scoring_status=StationScoringStatus.SCORING_FAILED,
                calibration_status=classify_calibration_status(
                    source=source_type,
                    scores=None,
                    calibration_span_minutes=cal_span_minutes,
                ),
                n_records_attempted=n_records_attempted,
                n_records_admitted=n_records_admitted,
                retained_samples=retained,
                failure_reason=f"{type(exc).__name__}: {exc}"[:500],
                dart_window_event_mode=window.event_mode,
                calibration_sha256=cal_sha256,
                rayleigh_inputs_available=rayleigh_inputs_available,
            ),
            None,
            False,
        )

    from hazard_assessment.telemetry.metrics import (
        observe_station_scoring_duration,
        record_anomaly_score,
    )
    observe_station_scoring_duration(time.monotonic() - score_start)
    record_anomaly_score()

    # Build the AnomalyAssessment envelope
    assessment = worker.agent.build_assessment(
        station_ids=[station_id],
        scores=scores,
        spatial_result=spatial_result,
    )
    attempt = StationAttemptResult(
        source=source_type,
        station_id=station_id,
        scoring_status=StationScoringStatus.SCORING_SUCCEEDED,
        calibration_status=classify_calibration_status(
            source=source_type,
            scores=scores,
            calibration_span_minutes=cal_span_minutes,
        ),
        n_records_attempted=n_records_attempted,
        n_records_admitted=n_records_admitted,
        retained_samples=retained,
        scores=scores,
        dart_window_event_mode=window.event_mode,
        calibration_sha256=cal_sha256,
        rayleigh_inputs_available=rayleigh_inputs_available,
    )
    return attempt, assessment.model_dump(), spatial_result is not None


def _score_station(
    station_key: tuple[str, str],
    worker: PipelineWorkerState,
) -> dict[str, Any] | None:
    """Run anomaly detection on one (source_type, station_id) window.

    Returns an AnomalyAssessment dict suitable for run_pipeline_sync(),
    or None if insufficient data. Thin wrapper over
    _score_station_attempt for direct callers and tests.
    """
    _, assessment, _ = _score_station_attempt(station_key, worker)
    return assessment


def _persist_anomaly_features(
    worker: PipelineWorkerState,
    assessments: list[tuple[tuple[str, str], dict[str, Any]]],
    trace_id: Any,
) -> list[str]:
    """Persist scored anomaly assessments as processed_features lineage rows.

    One row and one anomaly_scored audit entry per scored station (keyed by
    ``(source_type, station_id)``), sharing the assessment envelope's
    handoff_id and the batch trace_id. Best-effort at every step; never
    raises into the pipeline. Returns one label per anomaly_score row or
    audit entry that failed to persist, for the checkpoint assessment's
    companion-persistence disclosure.
    """
    failures: list[str] = []
    ctx = worker.fsm.event_context
    event_id = ctx.event_id if ctx is not None else None
    for (source_type, station_id), assessment in assessments:
        try:
            handoff_id = str(assessment.get("handoff_id") or uuid4())
            window = worker.station_buffers.get_window(station_id, source_type)
            # The scorer consumed the whole retained window, so lineage must
            # reference every retained sample (each carries its raw record's
            # payload hash), not just this batch's records.
            input_hashes = window.sample_hashes() if window is not None else []
            worker.audit_logger.append(AuditEntry(
                event_id=event_id,
                trace_id=trace_id,
                event_type="anomaly_scored",
                producer="anomaly_agent",
                data={
                    "station_id": station_id,
                    "anomaly_score": assessment.get("anomaly_score"),
                    "handoff_id": handoff_id,
                    "input_hashes": input_hashes,
                },
            ))
            if worker.db_client is not None:
                from hazard_assessment.telemetry.metrics import (
                    record_lineage_persist_failure,
                )
                ok = worker.db_client.insert_processed_feature(
                    feature_type="anomaly_score",
                    producer_agent="anomaly_agent",
                    handoff_id=handoff_id,
                    trace_id=trace_id,
                    payload=assessment,
                    source_refs=[{"sha256": h} for h in input_hashes],
                    event_id=event_id,
                    station_id=station_id,
                )
                if not ok:
                    record_lineage_persist_failure()
                    failures.append(
                        f"anomaly_score:{source_type}:{station_id}"
                    )
        except Exception as exc:
            logger.exception(
                "Failed to persist anomaly lineage for station %s", station_id
            )
            failures.append(
                f"anomaly_score:{source_type}:{station_id}: "
                f"{type(exc).__name__}"
            )
    return failures


def _reconcile_fsm_with_db(worker: PipelineWorkerState) -> None:
    """Adopt a durable event closure before processing a batch.

    The worker is the sole current runtime writer. This defensive path exists
    for a future trusted event-disposition writer: if that writer closes an
    event to IDLE, the worker must not retain stale in-memory ESCALATE state and
    drop the next seismic trigger. Caller-asserted assessment review does not
    write FSM state.

    This only acts when the durable state is IDLE while the worker is mid-event:
    it then reloads from the DB (recover_from_db, which restores IDLE with a
    cleared context). It never overwrites an event the worker is driving (that
    event shows non-IDLE in the DB), so it cannot clobber an in-memory latch.
    No-op without a DB.
    """
    from hazard_assessment.orchestrator.states import SystemState

    if worker.db_client is None or worker.fsm.state == SystemState.IDLE:
        return
    try:
        row = worker.db_client.load_fsm_state()
    except Exception:
        logger.exception("Failed to read FSM state for reconciliation")
        return
    if row is not None and row.get("current_state") == "IDLE":
        logger.info(
            "FSM resolved to IDLE in the database; resetting worker from %s",
            worker.fsm.state.value,
        )
        worker.fsm.recover_from_db()



def _transport_provenance(
    worker: PipelineWorkerState, transport: CheckpointTransport
) -> TransportProvenance:
    """Sorted, deduplicated transport provenance for this batch.

    Coordinates are deduplicated by (topic, partition, offset) and sorted
    because TransportProvenance validates exactly that ordering. The poll
    loop appends coordinates in receipt order, which a multi-partition
    assignment can interleave.
    """
    unique = {(m.topic, m.partition, m.offset): m for m in transport.messages}
    return TransportProvenance(
        run_id=str(worker.run_id),
        consumer_group=transport.consumer_group,
        messages=sorted(
            unique.values(), key=lambda m: (m.topic, m.partition, m.offset)
        ),
    )


def _record_reviewer_packet_gap(
    worker: PipelineWorkerState,
    *,
    noticed_at_checkpoint_id: str,
    trace_id: Any,
    event_id: Any,
    reason: str,
) -> None:
    """Disclose that an escalated event has no durable reviewer packet.

    Deliberately not ``_record_assessment_gap``: that one means the
    checkpoint's science ran but no assessment row exists, and it drives the
    assessment-gap counter. Here the assessment persisted fine and only the
    packet is missing, so reusing that channel would blur two failure classes
    an operator has to tell apart.

    ``noticed_at_checkpoint_id`` is named for what it is. The checkpoint that
    failed to write the packet is the one that entered ESCALATE, which is not
    necessarily the checkpoint running now.
    """
    worker.audit_logger.append(AuditEntry(
        event_id=event_id,
        trace_id=trace_id,
        event_type="reviewer_packet_gap",
        producer="pipeline_worker",
        data={"noticed_at_checkpoint_id": noticed_at_checkpoint_id, "reason": reason},
    ))


def _record_assessment_gap(
    worker: PipelineWorkerState,
    *,
    checkpoint_id: str,
    trace_id: Any,
    event_id: Any,
    reason: str,
) -> None:
    """Disclose an assessment gap: operator metric plus audit entry.

    A gap means this checkpoint's deterministic science ran but no
    assessment row exists for it. The checkpoint stays committable; the
    disclosure is what makes the gap reviewable instead of silent.
    """
    from hazard_assessment.telemetry.metrics import record_assessment_gap
    record_assessment_gap()
    worker.audit_logger.append(AuditEntry(
        event_id=event_id,
        trace_id=trace_id,
        event_type="assessment_gap",
        producer="pipeline_worker",
        data={"checkpoint_id": checkpoint_id, "reason": reason},
    ))
    logger.error("Assessment gap for checkpoint %s: %s", checkpoint_id, reason)


def _handle_redelivered_checkpoint(
    worker: PipelineWorkerState,
    checkpoint_id: str,
    existing_row: dict[str, Any],
    transport: CheckpointTransport,
) -> None:
    """Record a redelivered checkpoint without re-running the FSM.

    An existing assessment under this checkpoint key means a prior worker
    pass already processed this exact consumed-offset manifest through
    assessment persistence and then failed to commit offsets (the crash
    window). The FSM effects of the original pass live in the
    durable state this process recovered from, so re-evaluating the
    forward FSM here would double-apply the same evidence. The in-memory
    station windows are deliberately NOT rebuilt from the redelivered
    records: the windows that produced the existing assessment died with
    the original process, and replaying one batch would fabricate a
    partial window state that never existed.

    Repair side effects are an idempotent reviewer-packet check, one
    redelivery attempt record, and one audit disclosure. All are best-effort
    and offset commit proceeds anyway.
    """
    event_id = existing_row.get("event_id")
    handoff_id = str(existing_row.get("handoff_id") or "")
    logger.info(
        "Checkpoint %s already holds assessment %s; recording redelivery "
        "and skipping forward evaluation",
        checkpoint_id,
        handoff_id,
    )
    tp_hash: str | None = None
    try:
        tp_hash = transport_provenance_hash(
            _transport_provenance(worker, transport)
        )
    except Exception:
        logger.exception("Failed to hash redelivery transport provenance")
    if worker.db_client is not None:
        _persist_reviewer_packet(
            worker,
            db=worker.db_client,
            assessment_row=existing_row,
            checkpoint_id=checkpoint_id,
            trace_id=None,
        )
        worker.db_client.append_assessment_checkpoint_attempt(
            checkpoint_id=checkpoint_id,
            schema_version=ASSESSMENT_SCHEMA_VERSION,
            attempt_kind="redelivery",
            outcome="existing",
            event_id=event_id,
            worker_run_id=worker.run_id,
            transport_provenance_hash=tp_hash,
            detail="existing assessment found before forward evaluation",
        )
    worker.audit_logger.append(AuditEntry(
        event_id=event_id,
        event_type="assessment_redelivery",
        producer="pipeline_worker",
        data={
            "checkpoint_id": checkpoint_id,
            "handoff_id": handoff_id,
            "note": (
                "redelivered checkpoint: existing assessment stands; "
                "forward FSM evaluation skipped"
            ),
        },
    ))


def _build_and_persist_assessment(
    *,
    worker: PipelineWorkerState,
    checkpoint_id: str | None,
    transport: CheckpointTransport | None,
    trace_id: Any,
    station_attempts: list[StationAttemptResult],
    fsm_state_before: str,
    pipeline_outcome_field: str | None,
    seismic_only_no_score: bool,
    spatial_analysis_ran: bool,
    companion_failures: list[str],
    now_epoch: float | None,
) -> None:
    """revalidate, construct, hash, persist, audit.

    Produces one OceanEvidenceAssessment for a live checkpoint processed
    while an event is active, including seismic-only and
    insufficient-data checkpoints (their station attempts record
    unavailable ocean evidence). Never raises into the pipeline: every
    failure becomes an assessment gap (metric, audit entry, attempt
    record) and the checkpoint stays committable.

    Step 17 side effects are the attempt record, audit entries, and,
    for the checkpoint that enters ESCALATE, the durable reviewer
    packet rendered from the just-persisted row.
    """
    ctx = worker.fsm.event_context
    if ctx is None:
        # without an active event the worker only buffers
        # and scores; no assessment identity exists to persist.
        return
    if checkpoint_id is None or transport is None:
        # Direct callers (unit tests, future replay drivers) carry no
        # live checkpoint identity; the replay path derives its own
        # checkpoint ids in a later stage.
        return
    db = worker.db_client
    if db is None:
        # DB-less runtime keeps deterministic science and in-memory audit
        # behavior, but an assessment is by definition a durable
        # idempotent row: with no database there is nothing to
        # persist against or look up on redelivery.
        logger.debug(
            "No database: skipping assessment for checkpoint %s", checkpoint_id
        )
        return

    event_id = ctx.event_id

    # Step 14: revalidate durable event identity before construction.
    # A future trusted disposition writer may have closed the event (durable
    # IDLE), or the durable row may name a different event; persisting under
    # the stale in-memory identity
    # would attribute evidence to a resolved or foreign event. A failed
    # read proceeds: persist_assessment is idempotent and its conflict
    # check compares event identity again at insert time.
    try:
        row = db.load_fsm_state()
    except Exception:
        logger.exception("Failed to re-read durable FSM state before persist")
        row = None
    if row is not None:
        durable_state = row.get("current_state")
        durable_ctx = row.get("event_context")
        durable_event_id = (
            str(durable_ctx.get("event_id", ""))
            if isinstance(durable_ctx, dict)
            else ""
        )
        if durable_state == "IDLE" or (
            durable_event_id and durable_event_id != str(event_id)
        ):
            reason = (
                "durable event identity diverged before persist "
                f"(durable_state={durable_state}, "
                f"durable_event_id={durable_event_id or 'none'})"
            )
            db.append_assessment_checkpoint_attempt(
                checkpoint_id=checkpoint_id,
                schema_version=ASSESSMENT_SCHEMA_VERSION,
                attempt_kind="original",
                outcome="conflict",
                event_id=event_id,
                trace_id=trace_id,
                worker_run_id=worker.run_id,
                detail=reason,
            )
            _record_assessment_gap(
                worker,
                checkpoint_id=checkpoint_id,
                trace_id=trace_id,
                event_id=event_id,
                reason=reason,
            )
            return

    # Step 15: construct the draft assessment and finalize its hashes.
    produced_at = (
        datetime.fromtimestamp(now_epoch, tz=UTC)
        if now_epoch is not None
        else datetime.now(UTC)
    )
    thresholds = worker.threshold_settings
    try:
        assessment = build_ocean_evidence_assessment(
            checkpoint_id=checkpoint_id,
            checkpoint_source=CheckpointSource.LIVE_KAFKA,
            event_id=event_id,
            event_context=ctx,
            trace_id=trace_id,
            produced_at_utc=produced_at,
            station_attempts=station_attempts,
            detector_config=build_detector_config(
                thresholds.t1, thresholds.t2, thresholds.t3
            ),
            fsm_state_before=fsm_state_before,
            fsm_state_after=worker.fsm.state.value,
            fsm_transition_ref=(
                str(trace_id)
                if worker.fsm.state.value != fsm_state_before
                else ""
            ),
            dart_event_mode_stations_since_event_origin=(
                sorted(worker.event_mode_station_set)
                if worker.event_mode_event_id == event_id
                else []
            ),
            pipeline_outcome_field=pipeline_outcome_field,
            seismic_only_no_score=seismic_only_no_score,
            spatial_analysis_ran=spatial_analysis_ran,
            database_available=True,
            companion_persistence_failures=companion_failures,
            code_version=__version__,
        )
        assessment = finalize_assessment_hashes(
            assessment, _transport_provenance(worker, transport)
        )
    except Exception as exc:
        # Covers AssessmentConstructionError and pydantic validation
        # failures alike: any construction failure is a gap.
        detail = f"{type(exc).__name__}: {exc}"[:500]
        logger.exception("Assessment construction failed")
        db.append_assessment_checkpoint_attempt(
            checkpoint_id=checkpoint_id,
            schema_version=ASSESSMENT_SCHEMA_VERSION,
            attempt_kind="original",
            outcome="build_failed",
            event_id=event_id,
            trace_id=trace_id,
            worker_run_id=worker.run_id,
            detail=detail,
        )
        _record_assessment_gap(
            worker,
            checkpoint_id=checkpoint_id,
            trace_id=trace_id,
            event_id=event_id,
            reason=f"assessment build failed: {detail}",
        )
        return

    # Step 16: idempotent persist keyed on (checkpoint_id, schema_version).
    payload = assessment.model_dump(mode="json")
    result = db.persist_assessment(
        checkpoint_id=checkpoint_id,
        schema_version=ASSESSMENT_SCHEMA_VERSION,
        event_id=event_id,
        producer_agent=assessment.producer,
        handoff_id=assessment.handoff_id,
        trace_id=trace_id,
        payload=payload,
        input_manifest_hash=assessment.input_manifest_hash,
        scientific_content_hash=assessment.scientific_content_hash,
        transport_provenance_hash=assessment.transport_provenance_hash or None,
        source_refs=[{"sha256": ref.sha256} for ref in assessment.input_refs],
        code_version=__version__,
    )

    # Step 17: attempt record plus audit side effects. Best-effort; the
    # client logs its own failures and nothing here blocks offset commit.
    db.append_assessment_checkpoint_attempt(
        checkpoint_id=checkpoint_id,
        schema_version=ASSESSMENT_SCHEMA_VERSION,
        attempt_kind="original",
        outcome={
            "inserted": "inserted",
            "existing": "existing",
            "conflict": "conflict",
        }.get(result.status, "persist_failed"),
        event_id=event_id,
        trace_id=trace_id,
        worker_run_id=worker.run_id,
        transport_provenance_hash=assessment.transport_provenance_hash or None,
        detail=result.detail,
    )
    if result.status == "inserted":
        worker.audit_logger.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="assessment_persisted",
            producer="pipeline_worker",
            data={
                "checkpoint_id": checkpoint_id,
                "handoff_id": str(assessment.handoff_id),
                "assessment_schema_version": ASSESSMENT_SCHEMA_VERSION,
                "input_manifest_hash": assessment.input_manifest_hash,
                "scientific_content_hash": assessment.scientific_content_hash,
                "pipeline_outcome": assessment.pipeline_outcome.value,
            },
        ))
    elif result.status == "existing":
        # A row appeared although the step-3 lookup found none: another
        # pass through the crash window between the original insert and
        # its offset commit, rebuilt from scratch by this worker. The
        # original row stands; disclose the duplicate pass.
        worker.audit_logger.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="assessment_redelivery",
            producer="pipeline_worker",
            data={
                "checkpoint_id": checkpoint_id,
                "handoff_id": str((result.row or {}).get("handoff_id") or ""),
                "note": "existing assessment row adopted after rebuild",
            },
        ))
    else:
        _record_assessment_gap(
            worker,
            checkpoint_id=checkpoint_id,
            trace_id=trace_id,
            event_id=event_id,
            reason=f"assessment persist {result.status}: {result.detail}",
        )

    if result.status in ("inserted", "existing"):
        _persist_reviewer_packet(
            worker,
            db=db,
            assessment_row=result.row,
            checkpoint_id=checkpoint_id,
            trace_id=trace_id,
        )


def _persist_reviewer_packet(
    worker: PipelineWorkerState,
    *,
    db: Any,
    assessment_row: dict[str, Any] | None,
    checkpoint_id: str,
    trace_id: Any,
) -> None:
    """Ensure the entering-ESCALATE row has its durable reviewer packet.

    Both payload and identity come from the committed assessment row. This
    matters when ``persist_assessment`` adopts an existing row: a newly
    rebuilt payload is not the packet-of-record source. The idempotent call
    also runs from the early redelivery short-circuit, repairing a crash
    between assessment and packet persistence without re-running the FSM.
    """
    row = assessment_row or {}
    payload = row.get("payload")
    row_id = row.get("id")
    event_id = row.get("event_id")
    if row_id is None or event_id is None or not isinstance(payload, dict):
        logger.error(
            "Reviewer packet skipped: committed assessment row is missing "
            "id, event_id, or payload for checkpoint %s",
            checkpoint_id,
        )
        return
    if str(payload.get("event_id") or "") != str(event_id):
        logger.error(
            "Reviewer packet skipped: assessment row/payload event mismatch "
            "for checkpoint %s",
            checkpoint_id,
        )
        return
    if payload.get("fsm_state_after") != "ESCALATE":
        return
    if payload.get("fsm_state_before") == "ESCALATE":
        # Only the checkpoint that enters ESCALATE carries the packet of
        # record; later ESCALATE checkpoints persist assessments only. Their
        # evidence is a different assessment row, so re-rendering from one
        # would silently bind the reviewer to evidence the FSM did not
        # escalate on.
        #
        # They are still the place to notice that the packet never landed. If
        # the entering checkpoint's render or persist failed, review of this
        # event is impossible for its whole life, and nothing else looks
        # wrong. Disclose that once per event rather than staying silent.
        if worker.packet_confirmed_event_id != event_id:
            try:
                existing = db.get_escalation_packet_for_event(event_id)
            except Exception:
                logger.exception(
                    "Reviewer packet presence check raised for event %s", event_id
                )
                return
            if existing is not None:
                # Confirmed present: stop querying for this event.
                worker.packet_confirmed_event_id = event_id
            elif worker.packet_gap_disclosed_event_id != event_id:
                # Absent, as far as this query can tell.
                # get_escalation_packet_for_event returns None both when the
                # packet is missing and when the query itself failed, so a
                # transient database fault reads the same as a real gap. Latch
                # only the disclosure, not the query: the audit trail gets one
                # entry, and a later checkpoint can still confirm the packet
                # and stop reporting. The wording says "not found" rather than
                # asserting review is impossible, for the same reason.
                worker.packet_gap_disclosed_event_id = event_id
                _record_reviewer_packet_gap(
                    worker,
                    noticed_at_checkpoint_id=checkpoint_id,
                    trace_id=trace_id,
                    event_id=event_id,
                    reason=(
                        "no durable reviewer packet found for an escalated "
                        "event; human review needs the packet of record"
                    ),
                )
        return
    try:
        packet, content_sha256 = render_reviewer_packet(
            assessment_payload=payload, assessment_row_id=int(row_id)
        )
        result = db.persist_escalation_packet(
            assessment_row_id=int(row_id),
            event_id=event_id,
            renderer_version=RENDERER_VERSION,
            packet=packet,
            content_sha256=content_sha256,
        )
    except Exception:
        logger.exception(
            "Reviewer packet render or persist raised for checkpoint %s",
            checkpoint_id,
        )
        return
    if result.status == "inserted":
        worker.audit_logger.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="escalation_packet_persisted",
            producer="pipeline_worker",
            data={
                "checkpoint_id": checkpoint_id,
                "assessment_row_id": int(row_id),
                "renderer_version": RENDERER_VERSION,
                "content_sha256": content_sha256,
            },
        ))
    elif result.status == "conflict":
        # Same (assessment_row_id, renderer_version) key with a different
        # hash: renderer nondeterminism or row tampering. Disclose loudly.
        worker.audit_logger.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="escalation_packet_conflict",
            producer="pipeline_worker",
            data={
                "checkpoint_id": checkpoint_id,
                "assessment_row_id": int(row_id),
                "renderer_version": RENDERER_VERSION,
                "detail": result.detail,
            },
        ))
    elif result.status == "error":
        # A log line alone left this invisible: no audit entry, no metric, and
        # no later checkpoint retries, so an event became permanently
        # un-reviewable while the worker reported nothing unusual.
        logger.error(
            "Reviewer packet persist failed for checkpoint %s: %s",
            checkpoint_id,
            result.detail,
        )
        worker.packet_gap_disclosed_event_id = event_id
        _record_reviewer_packet_gap(
            worker,
            noticed_at_checkpoint_id=checkpoint_id,
            trace_id=trace_id,
            event_id=event_id,
            reason=f"reviewer packet persist failed: {result.detail}",
        )


@dataclass(frozen=True)
class CheckpointSummary:
    """Intermediates that one ``_process_buffer()`` checkpoint computed.

    The live worker persists its assessment internally, so live callers
    ignore this return. An offline
    observation-time replay driver runs the same checkpoint flow without
    Kafka transport and needs these intermediates to construct a
    REPLAY-checkpoint assessment under its own derived identity;
    returning them keeps the checkpoint orchestration single-sourced
    instead of duplicated in such a driver.
    """

    trace_id: UUID
    fsm_state_before: str
    station_attempts: tuple[StationAttemptResult, ...]
    pipeline_outcome_field: str | None
    n_scored_assessments: int
    seismic_transitioned: bool
    spatial_analysis_ran: bool
    companion_failures: tuple[str, ...]


def _process_buffer(
    buffer: dict[str, list[dict[str, Any]]],
    worker: PipelineWorkerState,
    *,
    now_epoch: float | None = None,
    transport: CheckpointTransport | None = None,
) -> CheckpointSummary | None:
    """Process one checkpoint of buffered records through the pipeline.

    Flow:
    1. Reconcile FSM state with the durable row
    2. Derive checkpoint identity from the batch's Kafka metadata
    3. Redelivered checkpoint (assessment already persisted): record the
       redelivery and return without forward evaluation
    4. Ingest seismic events (updates FSM + agent context)
    5. Run QC as metadata (never filtering), then ingest DART/CO-OPS
       observations into station buffers
    6. Record one scoring attempt per source-qualified station
    7. Select the highest-scoring station and run the FSM pipeline
    8. Build, hash, and idempotently persist the checkpoint's
       OceanEvidenceAssessment when an event is active; failures become
       operator-visible assessment gaps and never block offset commit

    ``transport`` carries the consumed Kafka coordinates for this batch;
    direct callers (tests, future replay drivers) may omit it, in which
    case no live checkpoint identity exists for the batch.
    """
    # Observe any future cross-process event closure before handling this
    # batch, so the worker cannot retain stale ESCALATE state after durable
    # state returns to IDLE.
    _reconcile_fsm_with_db(worker)

    # Derive the deterministic checkpoint identity for this batch before
    # any forward evaluation, so later slices can look up an
    # existing assessment for redelivered coordinates before mutating state.
    worker.last_checkpoint_id = (
        transport.checkpoint_id() if transport is not None else None
    )

    # Flow step 3: look up an existing assessment for this checkpoint key
    # BEFORE any forward FSM evaluation. A hit means Kafka redelivered a
    # batch whose original processing already persisted its assessment
    # (crash between persist and offset commit); re-running the FSM on it
    # would double-apply the same evidence. A failed lookup returns None
    # and the batch proceeds: the later persist is idempotent.
    if (
        worker.last_checkpoint_id is not None
        and transport is not None
        and worker.db_client is not None
    ):
        existing = worker.db_client.get_assessment_by_checkpoint(
            worker.last_checkpoint_id, ASSESSMENT_SCHEMA_VERSION
        )
        if existing is not None:
            _handle_redelivered_checkpoint(
                worker, worker.last_checkpoint_id, existing, transport
            )
            # No forward evaluation ran, so there is no checkpoint summary.
            return None

    if now_epoch is not None:
        worker.station_buffers.trim_all(now_epoch=now_epoch)

    # A seismic-only batch carries no observation evidence to score, so a
    # MONITOR that already crossed its timeout can be timed out NOW: otherwise
    # the first non-empty poll after downtime/backlog (when no quiet tick ever
    # ran) processes the new seismic record against the stale MONITOR and
    # drops it, and the post-batch timeout clears the old event only after
    # the new trigger was lost. When the batch DOES carry observations, the
    # timeout stays deferred until after they are scored (they may lift the
    # score above T1 and legitimately keep the event alive); a new seismic
    # record in that same mixed batch can still be dropped against the stale
    # event - a known narrow limitation of the single-event FSM.
    if all(key.startswith("seismic:") for key in buffer):
        worker.fsm.check_monitor_timeout()

    total = sum(len(v) for v in buffer.values())
    sources = set()
    for key in buffer:
        parts = key.split(":", 1)
        if parts:
            sources.add(parts[0])

    logger.info(
        "Pipeline batch: %d records, sources=%s, stations=%d",
        total,
        sorted(sources),
        len(buffer),
    )

    # Flow step 4: process seismic records first (updates FSM context).
    # Each record is handled independently so a malformed record cannot
    # prevent processing of DART/CO-OPS observations in the same batch.
    # The seismic loop only calls evaluate_seismic_trigger, which transitions
    # the FSM solely from IDLE, so any state change across the loop is a fresh
    # seismic transition this batch (used below to decide whether to emit a
    # seismic-only ABSTAIN when no station window is scored).
    # One trace id identifies this batch's pipeline RUN. It tags the
    # run-scoped audit entries: seismic-triggered FSM transitions, the
    # seismic trigger's seismic_provenance, the seismic-only ABSTAIN, QC,
    # anomaly scoring, the lineage rows, and the pipeline-node entries, so
    # /api/lineage/{trace_id} returns them together. The separate
    # get_provenance(trace_id) SQL path returns only the processed_features
    # lineage rows (QC/anomaly outputs) that share this trace, not the audit
    # entries.
    # input_provenance and provenance_capped are deliberately NOT trace-tagged:
    # they are per-EVENT observation provenance accumulated across many batches
    # (deduped by event id) and consumed by the escalation packet via
    # event_id, not trace_id.
    batch_trace_id = uuid4()
    # Kafka receipt position per application message ID, used as the
    # seismic-revision ordering tiebreak. Ingest message IDs are
    # the connectors' deterministic source_ids, so a seismic record's
    # source_id resolves its own coordinates. First occurrence wins: with
    # producer-side duplicates the earliest receipt is the honest local
    # receipt position.
    seismic_positions: dict[str, tuple[int, int]] = {}
    if transport is not None:
        for coord in transport.messages:
            if coord.transport_rejected or not coord.application_message_id:
                continue
            seismic_positions.setdefault(
                coord.application_message_id, (coord.partition, coord.offset)
            )
    fsm_state_before_seismic = worker.fsm.state
    for key, records in buffer.items():
        if key.startswith("seismic:"):
            for record in records:
                try:
                    _ingest_seismic_record(
                        record, worker, now_epoch=now_epoch,
                        trace_id=batch_trace_id,
                        kafka_positions=seismic_positions,
                    )
                except Exception:
                    logger.exception("Failed to process seismic record: %s", key)
    seismic_transitioned = worker.fsm.state != fsm_state_before_seismic

    # Flow step 5: per station batch, run QARTOD QC on the parseable records
    # FIRST, then ingest into station buffers so accepted
    # samples carry their per-record QC, measurement type or product, and
    # payload hash. QC is metadata only; the anomaly agent
    # processes raw values regardless, so QC never filters records out of
    # the buffer. Collect the stations that had a validly INGESTED
    # event-mode record this batch (used for dart_confirmation below, so
    # the latch only fires on accepted data).
    batch_event_mode_stations: set[str] = set()
    admission_counts: dict[tuple[str, str], list[int]] = {}
    companion_failures: list[str] = []
    for key, records in buffer.items():
        if not key.startswith("seismic:"):
            qc_by_hash = _run_qc(
                key, records, worker, trace_id=batch_trace_id,
                companion_failures=companion_failures,
            )
            batch_event_mode_stations |= _ingest_observation_records(
                key, records, worker, now_epoch=now_epoch,
                qc_by_hash=qc_by_hash, admission_counts=admission_counts,
            )

    # Update DART confirmation on the FSM when one or more stations had an
    # accepted event-mode record THIS batch while an event is active. This
    # deliberately ratchets: once any DART enters event mode during an event the
    # flag stays set in the EventContext, and there is intentionally no path
    # that resets it to False while the event is active. A DART that tripped
    # into event mode observed a real pressure perturbation, and the wave then
    # continues toward the coast, so the buoy returning to standard cadence must
    # NOT relax the seismic override and let the FSM de-escalate. The latch is
    # bounded per event: the FSM clears the whole EventContext
    # (dart_confirmation included) on resolve_event / monitor timeout, and the
    # worker's per-event station accumulation resets when the event id changes,
    # so it never leaks across events.
    #
    # Latching uses ONLY batch_event_mode_stations (accepted event-mode records
    # in this batch), never station_buffers.stations_in_event_mode(): the
    # buffer's per-window flags are not event-scoped, so a resolved event's
    # stale flag would otherwise attach dart_confirmation to a later, unrelated
    # event. Nothing is lost by this: DART event mode physically FOLLOWS a
    # triggering quake and event-mode rows keep arriving for hours, so any
    # genuine activation during the event latches from its own rows, and the
    # FSM latch persists for the rest of the event once set.
    #
    # Detection is batch-aware: a station counts if ANY validly ingested record
    # in this batch was event-mode, not just the latest, so an out-of-order or
    # trailing standard-mode record in the same batch cannot mask an event-mode
    # signal (the less-safe direction, since the latch can only push toward
    # escalation / human review). Only accepted records count, so a malformed
    # record that never entered the buffer/QC/scoring path cannot latch.
    if batch_event_mode_stations:
        ctx = worker.fsm.event_context
        if ctx is not None:
            if worker.event_mode_event_id != ctx.event_id:
                worker.event_mode_event_id = ctx.event_id
                worker.event_mode_station_set = set()
            worker.event_mode_station_set |= batch_event_mode_stations
            worker.fsm.update_dart_confirmation(
                dart_confirmation=True,
                stations_in_event_mode=sorted(worker.event_mode_station_set),
            )

    # Flow step 6: one StationAttemptResult per
    # source-qualified station considered this checkpoint. That is every
    # station holding a retained window, plus every batch station whose
    # records were all rejected before buffering; the latter become
    # NO_RETAINED_DATA attempts instead of silently vanishing from the
    # assessment.
    scope: dict[tuple[str, str], None] = dict.fromkeys(
        worker.station_buffers.station_keys()
    )
    for batch_station_key in admission_counts:
        scope.setdefault(batch_station_key, None)

    assessments: list[tuple[tuple[str, str], dict[str, Any]]] = []
    station_attempts: list[StationAttemptResult] = []
    spatial_analysis_ran = False
    for station_key in scope:
        station_counts = admission_counts.get(station_key)
        attempt, assessment, spatial_ran = _score_station_attempt(
            station_key,
            worker,
            n_records_attempted=station_counts[0] if station_counts else 0,
            n_records_admitted=station_counts[1] if station_counts else 0,
        )
        station_attempts.append(attempt)
        spatial_analysis_ran = spatial_analysis_ran or spatial_ran
        if assessment is not None:
            assessments.append((station_key, assessment))

    pipeline_outcome_field: str | None = None
    if not assessments:
        if seismic_transitioned:
            # A seismic trigger moved the FSM (to MONITOR, or ESCALATE for a
            # large shallow quake) but no station window has been scored yet -
            # DART/CO-OPS data arrives 15-30 min after origin. Emit a
            # fail-closed ABSTAIN artifact so the transition is reviewable,
            # instead of leaving only the state_transition audit entries.
            _emit_seismic_only_abstain(worker, trace_id=batch_trace_id)
        else:
            logger.debug("No stations had sufficient data for scoring")
    else:
        # Flow step 7: select the highest-scoring assessment and drive the
        # FSM. The FSM evaluates one score per call; the maximum ensures
        # the most concerning signal drives state transitions.
        best_key, best_assessment = max(
            assessments, key=lambda x: x[1].get("anomaly_score", 0.0),
        )

        logger.info(
            "Best anomaly score: station=%s:%s score=%.4f (of %d scored)",
            best_key[0],
            best_key[1],
            best_assessment.get("anomaly_score", 0.0),
            len(assessments),
        )

        # SCOPE: the deployed worker runs anomaly detection + FSM only.  It
        # does not pre-populate scenario_assessment or verification_result,
        # so once the FSM reaches ASSESS/ESCALATE the pipeline fail-closes
        # to ABSTAIN (the correct safe behavior).  Live scenario inversion,
        # verification, and report generation require the unit-source/
        # propagation database and a VerificationInput assembly layer; they
        # are intentionally deferred to avoid emitting under-validated
        # science.  The assessments validated in the paper come from the
        # offline scripts (scripts/validate_*.py).
        from hazard_assessment.orchestrator.nodes import run_pipeline_sync
        from hazard_assessment.orchestrator.pipeline import PipelineState
        from hazard_assessment.orchestrator.states import SystemState

        state: PipelineState = {
            "anomaly_assessment": best_assessment,
            "trace_id": str(batch_trace_id),
        }

        result = run_pipeline_sync(
            state,
            fsm=worker.fsm,
            audit_logger=worker.audit_logger,
        )

        # Persist every scored assessment as an anomaly_score lineage row
        # plus a companion anomaly_scored audit entry sharing the batch
        # trace and the envelope's handoff_id, with the station's accepted
        # payload hashes as input_hashes. This is what makes
        # get_provenance(trace_id) walk feature -> audit entry ->
        # raw_observations with real data. Best-effort: lineage persistence
        # must never break the pipeline; failures are disclosed in the
        # checkpoint assessment instead.
        companion_failures.extend(
            _persist_anomaly_features(worker, assessments, batch_trace_id)
        )

        if worker.fsm.state in (SystemState.ASSESS, SystemState.ESCALATE):
            logger.info(
                "FSM at %s: the deployed worker abstains here (scenario/"
                "verification/report are not computed in the worker; full "
                "assessment runs via the offline pipeline). Operator review "
                "required.",
                worker.fsm.state.value,
            )

        final = result.get("final_assessment", {})
        pipeline_outcome_field = final.get("outcome")
        logger.info(
            "Pipeline complete: fsm_state=%s, outcome=%s, status=%s",
            result.get("fsm_state", "?"),
            final.get("outcome", "?"),
            final.get("pipeline_status", result.get("pipeline_status", "?")),
        )

    # Flow step 8: assemble and idempotently persist
    # this checkpoint's OceanEvidenceAssessment. Runs for every live
    # checkpoint processed while an event is active, including seismic-only
    # and insufficient-data batches, whose assessments record unavailable
    # ocean evidence. Failures inside never block deterministic science or
    # the offset commit.
    _build_and_persist_assessment(
        worker=worker,
        checkpoint_id=worker.last_checkpoint_id,
        transport=transport,
        trace_id=batch_trace_id,
        station_attempts=station_attempts,
        fsm_state_before=fsm_state_before_seismic.value,
        pipeline_outcome_field=pipeline_outcome_field,
        seismic_only_no_score=seismic_transitioned and not assessments,
        spatial_analysis_ran=spatial_analysis_ran,
        companion_failures=companion_failures,
        now_epoch=now_epoch,
    )

    return CheckpointSummary(
        trace_id=batch_trace_id,
        fsm_state_before=fsm_state_before_seismic.value,
        station_attempts=tuple(station_attempts),
        pipeline_outcome_field=pipeline_outcome_field,
        n_scored_assessments=len(assessments),
        seismic_transitioned=seismic_transitioned,
        spatial_analysis_ran=spatial_analysis_ran,
        companion_failures=tuple(companion_failures),
    )


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("APP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )
    metrics_port = os.getenv("METRICS_PORT", "").strip()
    if metrics_port.isdigit():
        from hazard_assessment.telemetry.metrics import start_metrics_exporter
        start_metrics_exporter(int(metrics_port))
    # The worker wraps every pipeline run in pipeline_span (nodes.py); without
    # a configured provider those spans are silent no-ops. No-op when
    # OTLP_ENDPOINT is unset.
    from hazard_assessment.telemetry.tracing import configure_tracer_provider
    configure_tracer_provider(os.getenv("OTLP_ENDPOINT"))
    asyncio.run(_run())


if __name__ == "__main__":
    main()
