"""QC Agent - applies QARTOD-aligned quality checks to observation records."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.agents.qc_checks import (
    CONFIDENCE_EXCLUSION_THRESHOLD,
    QCObservation,
    compute_station_confidence,
    count_evaluated_checks,
    prune_station_history,
    run_all_checks,
    sort_observations,
)
from hazard_assessment.ingest.coops import (
    COOPS_PRODUCT_EXPECTED_INTERVAL_SEC,
    CoopsRecord,
)
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.schemas.envelope import DataSource, InputRef
from hazard_assessment.schemas.qc import EVENT_MODE_NOTE, DataMode, QCReport

_MANIFEST = AgentManifest(
    name="qc_agent",
    version="1.0.0",
    capabilities=[
        AgentCapability.READ_DATA,
        AgentCapability.WRITE_DATA,
        AgentCapability.WRITE_AUDIT,
        AgentCapability.PRODUCE_KAFKA,
    ],
    description="Applies QARTOD-aligned quality checks to incoming observations",
)


# DART *measurement* cadences per measurement_type.  These are distinct from
# the satellite *delivery* cadence (DART_STANDARD_DATA_CADENCE_SEC = 6 h) used
# by the ingest layer for stale detection.  QC timing-gap checks need the
# sensor sampling interval, not the batch delivery interval.
# Ref: NDBC DART description - BPR subsamples every 15 min (type 1),
#      event mode transmits 1-min averages (type 2) or 15-sec samples (type 3).
_DART_MEASUREMENT_CADENCE_SEC: dict[int, float] = {
    1: 900.0,   # 15-min standard-mode subsamples (one 15-sec sample per 15 min)
    2: 60.0,    # 1-min event-mode averages (four 15-sec samples averaged)
    3: 15.0,    # 15-sec event-mode samples
}


def _dart_expected_interval(measurement_type: int) -> float:
    """Return expected measurement interval in seconds for DART data."""
    return _DART_MEASUREMENT_CADENCE_SEC.get(measurement_type, 900.0)


def _coops_expected_interval(product: str) -> float:
    """Return expected interval in seconds for a CO-OPS product."""
    return float(COOPS_PRODUCT_EXPECTED_INTERVAL_SEC.get(product, 60))


def dart_record_to_qc_obs(record: DartRecord) -> QCObservation:
    """Convert a DartRecord to a QCObservation for QC checks."""
    return QCObservation(
        source_type="dart",
        station_id=record.station_id,
        source_timestamp=record.source_timestamp,
        value_m=record.height_m,
        measurement_type=record.measurement_type,
        event_mode=record.event_mode,
        expected_interval_sec=_dart_expected_interval(record.measurement_type),
        payload_sha256=record.payload_sha256,
    )


def coops_record_to_qc_obs(record: CoopsRecord) -> QCObservation:
    """Convert a CoopsRecord to a QCObservation for QC checks."""
    return QCObservation(
        source_type="coops",
        station_id=record.station_id,
        source_timestamp=record.source_timestamp,
        value_m=record.water_level_m,
        measurement_type=None,
        event_mode=False,
        expected_interval_sec=_coops_expected_interval(record.product),
        payload_sha256=record.payload_sha256,
    )


def qc_observation_from_dict(
    source_type: str,
    station_id: str,
    record: dict[str, Any],
) -> QCObservation:
    """Build a QCObservation from a worker-side record dict.

    The pipeline worker holds JSON-decoded record dicts (post-Kafka) rather
    than typed DartRecord/CoopsRecord objects, so this mirrors
    ``dart_record_to_qc_obs`` / ``coops_record_to_qc_obs`` for that shape.
    """
    ts = record["source_timestamp"]
    source_timestamp = ts if isinstance(ts, datetime) else datetime.fromisoformat(ts)
    if source_type == "dart":
        measurement_type = record.get("measurement_type")
        return QCObservation(
            source_type="dart",
            station_id=station_id,
            source_timestamp=source_timestamp,
            value_m=record.get("height_m"),
            measurement_type=measurement_type,
            # Same strict-bool rule as worker ingestion: a malformed flag
            # ("false", 1) must not read as event mode in QC audit metadata.
            event_mode=record.get("event_mode") is True,
            expected_interval_sec=_dart_expected_interval(
                measurement_type if isinstance(measurement_type, int) else 1
            ),
            payload_sha256=str(record.get("payload_sha256", "")),
        )
    return QCObservation(
        source_type="coops",
        station_id=station_id,
        source_timestamp=source_timestamp,
        value_m=record.get("water_level_m"),
        measurement_type=None,
        event_mode=False,
        expected_interval_sec=_coops_expected_interval(str(record.get("product", ""))),
        payload_sha256=str(record.get("payload_sha256", "")),
    )


def _source_for(source_type: Literal["dart", "coops"]) -> DataSource:
    if source_type == "dart":
        return DataSource.DART
    return DataSource.COOPS


def process_observations(
    observations: list[QCObservation],
    station_history: dict[str, list[QCObservation]] | None = None,
    *,
    processing_time: datetime | None = None,
) -> list[QCReport]:
    """Run QC checks on a batch of observations and produce QCReports.

    Args:
        observations: Unsorted batch of observations (sorted internally).
        station_history: Optional pre-existing per-station history for
            flat-line detection across batches, keyed by the
            source-qualified station key ``"{source_type}:{station_id}"``
            so equal identifiers from different sources cannot share
            sequential-check state.
        processing_time: Optional fixed timestamp for report production and
            history pruning.  Defaults to ``datetime.now(UTC)`` when *None*.
            Inject a fixed value for deterministic tests and replay.

    Returns:
        List of QCReport envelopes, one per observation.
    """
    if station_history is None:
        station_history = {}

    # deterministic sort for out-of-order arrivals
    sorted_obs = sort_observations(observations)

    # History-retention cutoff. Derive from the newest observation rather than
    # wall-clock when possible: replaying historical records must not prune the
    # very history the flat-line check needs (a wall-clock cutoff is decades
    # after replayed timestamps and would wipe everything).
    now = (
        processing_time
        or max((o.source_timestamp for o in sorted_obs), default=None)
        or datetime.now(UTC)
    )

    # Track the previous observation per station for sequential checks
    prev_by_station: dict[str, QCObservation] = {}

    # Seed prev from history tails
    for sid, hist in station_history.items():
        if hist:
            prev_by_station[sid] = hist[-1]

    reports: list[QCReport] = []

    for obs in sorted_obs:
        sid = f"{obs.source_type}:{obs.station_id}"
        prev = prev_by_station.get(sid)
        history = station_history.get(sid, [])

        flags = run_all_checks(obs, prev, history)
        confidence = compute_station_confidence(flags)
        record_usable = confidence >= CONFIDENCE_EXCLUSION_THRESHOLD

        data_mode = (
            DataMode.EVENT if obs.event_mode else DataMode.STANDARD
        )
        event_note = EVENT_MODE_NOTE if obs.event_mode else ""

        report = QCReport(
            producer="qc_agent",
            produced_at_utc=now,
            input_refs=[
                InputRef(
                    source=_source_for(obs.source_type),
                    # Station-level identifier: QCObservation does not carry the
                    # connector's per-record source_id. The sha256 is still the
                    # raw record's payload hash, so the reference is
                    # content-precise even though the id is station-coarse.
                    record_id=f"{obs.source_type}:{obs.station_id}",
                    sha256=obs.payload_sha256,
                ),
            ],
            station_id=obs.station_id,
            observed_at_utc=obs.source_timestamp,
            measurement_type=obs.measurement_type,  # type: ignore[arg-type]
            data_mode=data_mode,
            event_mode_note=event_note,
            record_usable=record_usable,
            qartod_flags=flags,
            station_confidence=confidence,
            n_checks_evaluated=count_evaluated_checks(flags),
            provenance_hash=obs.payload_sha256,
        )
        reports.append(report)

        # Update state for next iteration
        prev_by_station[sid] = obs
        station_history.setdefault(sid, []).append(obs)

    # Prune old history to prevent unbounded memory growth
    prune_station_history(station_history, now)

    return reports


class QCAgent(BaseAgent):
    """Quality Control Agent.

    Applies timing, range, spike, rate-of-change, and flat-line checks
    to each incoming observation record. Produces QCReport envelopes.
    Maintains per-station history for flat-line and continuity checks.
    """

    def __init__(self) -> None:
        super().__init__(manifest=_MANIFEST)
        self._station_history: dict[str, list[QCObservation]] = {}

    def process_records(
        self,
        records: list[DartRecord | CoopsRecord],
        *,
        processing_time: datetime | None = None,
    ) -> list[QCReport]:
        """Process a batch of ingest records and return QCReports.

        This is the primary entry point for the QC Agent. It converts
        ingest records to QCObservations, runs all QARTOD checks, and
        returns QCReport envelopes.

        Args:
            records: Ingest records (DART or CO-OPS).
            processing_time: Optional fixed timestamp for deterministic
                output.  Passed through to :func:`process_observations`.
        """
        observations: list[QCObservation] = []
        for record in records:
            if isinstance(record, DartRecord):
                observations.append(dart_record_to_qc_obs(record))
            elif isinstance(record, CoopsRecord):
                observations.append(coops_record_to_qc_obs(record))
            else:
                raise TypeError(f"Unsupported record type: {type(record).__name__}")

        return process_observations(
            observations, self._station_history, processing_time=processing_time
        )

    def process_qc_observations(
        self,
        observations: list[QCObservation],
        *,
        processing_time: datetime | None = None,
    ) -> list[QCReport]:
        """Run QC on pre-built QCObservations using this agent's station history.

        Used by the live pipeline worker, which holds JSON-decoded record dicts
        rather than typed DartRecord/CoopsRecord objects.
        """
        return process_observations(
            observations, self._station_history, processing_time=processing_time
        )
