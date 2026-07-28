"""Deterministic FSM state definitions for the orchestrator.

The system operates through five states. All transitions are governed by
deterministic threshold comparisons on the anomaly ensemble score and
seismic context. No LLM is in the decision path; template-based
reasoning text is generated downstream by the Report Agent.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol
from uuid import UUID, uuid4

from hazard_assessment.schemas.observation import is_dart_station_id

if TYPE_CHECKING:
    from hazard_assessment.storage.client import DatabaseClient

logger = logging.getLogger(__name__)


def _sanitize_dart_station_ids(station_ids: list[str]) -> list[str]:
    valid = [station_id for station_id in station_ids if is_dart_station_id(station_id)]
    if len(valid) != len(station_ids):
        logger.warning("Dropped invalid DART station identifier from FSM context")
    return valid


class SystemState(StrEnum):
    """FSM states for the hazard assessment orchestrator."""

    IDLE = "IDLE"
    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    ASSESS = "ASSESS"
    ESCALATE = "ESCALATE"


# Valid state transitions (from -> set of allowed destinations)
VALID_TRANSITIONS: dict[SystemState, set[SystemState]] = {
    SystemState.IDLE: {SystemState.MONITOR},
    SystemState.MONITOR: {SystemState.IDLE, SystemState.INVESTIGATE, SystemState.ESCALATE},
    SystemState.INVESTIGATE: {SystemState.MONITOR, SystemState.ASSESS},
    SystemState.ASSESS: {SystemState.INVESTIGATE, SystemState.ESCALATE},
    # ESCALATE -> IDLE requires an explicit resolve_event() call. That method
    # does not authenticate authority, and no current API route invokes it.
    # Caller-asserted assessment review never changes FSM state. A future
    # trusted event-disposition path must validate authority before calling it.
    SystemState.ESCALATE: {SystemState.IDLE},
}


@dataclass(frozen=True)
class ThresholdConfig:
    """Per-basin threshold configuration for state transitions.

    Thresholds are calibrated against historical replay data.
    Initial values are defaults; divergence expected after calibration.
    """

    basin: str
    t1: float  # MONITOR -> INVESTIGATE
    t2: float  # INVESTIGATE -> ASSESS
    t3: float  # ASSESS -> ESCALATE
    seismic_min_magnitude: float = 6.0
    # M7.5 aligns with PTWC operational criteria for automatic Tsunami Watch
    # issuance (M>=7.5 in tsunamigenic zones; M>=7.0 in the Aleutians).
    escalation_magnitude: float = 7.5
    # 12-hour timeout: applies only when anomaly score stays below T1 (no
    # evidence of ocean disturbance). If any DART/CO-OPS station shows an
    # elevated score, the FSM advances to INVESTIGATE and the timeout does
    # not fire. The 12-hour window accommodates the maximum trans-Pacific
    # tsunami travel time (~22 hours Chile -> Japan), with margin for the
    # nearest DART station to detect the wave. For most Pacific subduction
    # zone sources, the nearest DART station detects within ~2-4 hours,
    # but sparse DART coverage in the South Pacific and Indian Ocean means
    # some source-station pairs can exceed 6 hours. The 12-hour default
    # balances false-negative avoidance (missed slow-arriving tsunamis)
    # against system availability (the single-event FSM blocks new events
    # while in MONITOR). Operators can adjust via THRESHOLD_MONITOR_TIMEOUT_HOURS.
    monitor_timeout_hours: float = 12.0
    # Depth threshold for seismic-only escalation: earthquakes shallower than
    # this depth (in km) with M >= escalation_magnitude trigger immediate
    # MONITOR -> ESCALATE without requiring DART confirmation. Aligns with
    # PTWC/JMA practice of issuing initial warnings on seismic data alone
    # for large shallow tsunamigenic events. Deep earthquakes (> 100 km) are
    # far less tsunamigenic and should not bypass the DART confirmation gate.
    seismic_escalation_depth_km: float = 100.0

    def __post_init__(self) -> None:
        if not (0.0 < self.t1 < self.t2 < self.t3 <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < t1 < t2 < t3 <= 1, "
                f"got t1={self.t1}, t2={self.t2}, t3={self.t3}"
            )


DEFAULT_THRESHOLDS = ThresholdConfig(
    basin="pacific",
    t1=0.35,
    t2=0.60,
    t3=0.85,
)


@dataclass(frozen=True)
class TransitionRecord:
    """Immutable record of a state transition for the audit trail."""

    transition_id: UUID = field(default_factory=uuid4)
    timestamp_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    event_id: UUID | None = None
    trace_id: UUID | None = None
    from_state: SystemState = SystemState.IDLE
    to_state: SystemState = SystemState.IDLE
    trigger_reason: str = ""
    anomaly_score: float | None = None
    seismic_magnitude: float | None = None
    thresholds_used: ThresholdConfig = field(default_factory=lambda: DEFAULT_THRESHOLDS)

    def __post_init__(self) -> None:
        if self.timestamp_utc.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware")


class AuditWriter(Protocol):
    """Protocol for writing transition records to the audit trail."""

    def write_transition(self, record: TransitionRecord) -> None: ...


@dataclass(frozen=True)
class SeismicIdentity:
    """External identity of one received seismic revision.

    Built by the pipeline worker from a validated, locally received seismic
    record. ``provider_updated_utc`` must be None when the provider update
    time is missing, malformed, or in the future relative to local receipt;
    such a revision is retained as provenance (raw record and audit trail)
    but can never silently supersede the latest valid revision.
    ``kafka_partition``/``kafka_offset`` are the receipt coordinates used as
    the ordering tiebreak; None when the transport did not resolve them.
    """

    provider: str
    external_event_id: str
    revision_id: str
    revision_sha256: str  # "" when the record carried no canonical hash
    provider_updated_utc: datetime | None
    kafka_partition: int | None
    kafka_offset: int | None
    context_class: str  # SeismicContextClass value; live worker passes LIVE_RECEIPT_ORDERED

    def __post_init__(self) -> None:
        if (
            self.provider_updated_utc is not None
            and self.provider_updated_utc.tzinfo is None
        ):
            raise ValueError("provider_updated_utc must be timezone-aware")


@dataclass
class EventContext:
    """Mutable context for an active seismic event being tracked."""

    event_id: UUID = field(default_factory=uuid4)
    seismic_magnitude: float = 0.0
    seismic_region: str = ""
    epicenter_lat: float = 0.0
    epicenter_lon: float = 0.0
    trigger_time_utc: datetime = field(default_factory=lambda: datetime.now(UTC))
    latest_anomaly_score: float = 0.0

    # Hypocentral depth in km from the seismic trigger, or None when the
    # source (e.g. a preliminary USGS solution) did not report one. The
    # seismic-only escalation path requires depth < 100 km, so a None depth
    # never fires it.
    depth_km: float | None = None

    # DART event-mode flag. Set to True when one or more DART stations have
    # entered event-mode transmission (measurement types 2 or 3), indicating
    # the onboard BPR detected a pressure perturbation exceeding its ~3 cm
    # trigger threshold. This is a necessary but NOT sufficient condition
    # for tsunami confirmation - event-mode can also be triggered by seismic
    # Rayleigh waves or calibration pulses. The anomaly scoring pipeline
    # performs the actual tsunami signal discrimination via detiding, bandpass
    # filtering, and ensemble scoring. The FSM uses this flag together with
    # M>=7.5 as a seismic override to bypass the T3 anomaly threshold, per
    # PTWC practice of issuing Tsunami Watch on M>=7.5 with DART activation.
    #
    # Ratchets True for the lifetime of an event once set (see the worker's
    # update_dart_confirmation call site in pipeline_runner.py): it is never
    # reset to False while the event is active, only cleared with the whole
    # EventContext at resolve_event / monitor timeout.
    dart_confirmation: bool = False

    active_dart_stations: list[str] = field(default_factory=list)
    stations_in_event_mode: list[str] = field(default_factory=list)

    # External seismic identity. Empty strings and None
    # mean "not bound": events created by callers that supply no
    # SeismicIdentity (offline scripts, old durable rows) carry no external
    # identity, and the assessment builder must disclose that. The trigger
    # fields are immutable by construction: only evaluate_seismic_trigger
    # sets them, at event creation. The latest_revision_* fields track the
    # latest admissible revision of the SAME external event and are updated
    # only through update_seismic_revision(). latest_revision_updated_utc
    # None means the incumbent revision had no valid provider update time
    # (possible only while the incumbent is the trigger itself).
    seismic_provider: str = ""
    external_event_id: str = ""
    trigger_revision_id: str = ""
    trigger_revision_sha256: str = ""
    latest_revision_id: str = ""
    latest_revision_sha256: str = ""
    latest_revision_updated_utc: datetime | None = None
    latest_revision_kafka_partition: int | None = None
    latest_revision_kafka_offset: int | None = None
    seismic_context_class: str = ""

    def __post_init__(self) -> None:
        if self.trigger_time_utc.tzinfo is None:
            raise ValueError("trigger_time_utc must be timezone-aware")
        if (
            self.latest_revision_updated_utc is not None
            and self.latest_revision_updated_utc.tzinfo is None
        ):
            raise ValueError("latest_revision_updated_utc must be timezone-aware")
        self.active_dart_stations = _sanitize_dart_station_ids(self.active_dart_stations)
        self.stations_in_event_mode = _sanitize_dart_station_ids(self.stations_in_event_mode)
        # Do not clear dart_confirmation when station lists are empty. The
        # event-mode list can legitimately expire while the per-event boolean
        # remains a one-way latch. Sanitization protects emitted identifiers.


class FSMOrchestrator:
    """Deterministic finite-state machine for hazard assessment orchestration.

    **Thread safety:** This class is NOT thread-safe internally. Callers
    must provide external synchronization (e.g., ``threading.Lock``) when
    the FSM is accessed from multiple threads.  The FastAPI app uses
    ``_fsm_lock`` for this purpose.  Any future consumer (pipeline
    runner, CLI tool) must do the same.

    **Database persistence:** When ``db_client`` is provided, FSM state
    is persisted to the ``fsm_current_state`` table in TimescaleDB.
    On startup, ``recover_from_db()`` loads the last known state.

    **Important - cross-process write serialization:** The Postgres advisory
    lock in ``upsert_fsm_state`` serializes concurrent DB *writes* (no torn
    state row; validated against PostgreSQL 16 - ``pg_advisory_xact_lock`` blocks
    a second acquirer until the holder commits). A *lock-spanning*
    read-modify-write (one advisory lock held across reload -> evaluate ->
    persist) is deliberately NOT implemented because the pipeline worker is
    the sole runtime FSM writer in the current architecture. The API reloads
    durable state for reads; caller-asserted assessment review does not mutate
    the FSM. Lock-spanning serialization and explicit ownership are required
    before introducing another worker or a trusted event-disposition writer.
    No public alert is gated on FSM state and the audit trail is append-only.
    """

    MAX_HISTORY = 1000

    def __init__(
        self,
        thresholds: ThresholdConfig = DEFAULT_THRESHOLDS,
        audit_writer: AuditWriter | None = None,
        db_client: DatabaseClient | None = None,
    ) -> None:
        self._state = SystemState.IDLE
        self._thresholds = thresholds
        self._audit_writer = audit_writer
        self._db_client = db_client
        self._event_context: EventContext | None = None
        self._transition_history: list[TransitionRecord] = []
        self._sensor_degraded = False
        # Set to True by recover_from_db() on deserialization failure.
        # Callers and health-check endpoints should inspect this flag and
        # surface it via the operator dashboard rather than relying on logs.
        self._recovery_failed = False

    def recover_from_db(self) -> bool:
        """Load FSM state from the database on startup.

        Returns True if state was recovered, False if no DB or no state found.
        """
        if self._db_client is None:
            return False

        row = self._db_client.load_fsm_state()
        if row is None:
            return False

        try:
            self._state = SystemState(row["current_state"])
            self._sensor_degraded = row.get("sensor_degraded", False)
            ctx_json = row.get("event_context")
            if self._state == SystemState.IDLE:
                # An IDLE state means no active event. Never reconstruct an event
                # context for it, even if the row carries one - that defends
                # against a stale IDLE-with-context row written by an older
                # version before _transition cleared the context before persist.
                if ctx_json:
                    logger.warning(
                        "Recovered an IDLE FSM row with a non-null event_context; "
                        "dropping the stale context."
                    )
                self._event_context = None
            elif ctx_json:
                ctx_data = ctx_json if isinstance(ctx_json, dict) else json.loads(ctx_json)
                # Normalize an aware non-UTC offset to UTC: the value is
                # serialized and rendered as UTC downstream. A naive string
                # stays naive so EventContext rejects it (recovery failure).
                trigger_time = datetime.fromisoformat(ctx_data["trigger_time_utc"])
                if trigger_time.tzinfo is not None:
                    trigger_time = trigger_time.astimezone(UTC)
                # Seismic identity fields. Rows written
                # before these fields existed recover as "identity not
                # bound" via the same defaults EventContext uses. A naive
                # latest_revision_updated_utc string fails EventContext
                # validation (recovery failure), like trigger_time_utc.
                latest_rev_updated: datetime | None = None
                latest_rev_updated_raw = ctx_data.get("latest_revision_updated_utc")
                if latest_rev_updated_raw is not None:
                    latest_rev_updated = datetime.fromisoformat(latest_rev_updated_raw)
                    if latest_rev_updated.tzinfo is not None:
                        latest_rev_updated = latest_rev_updated.astimezone(UTC)
                self._event_context = EventContext(
                    event_id=UUID(ctx_data["event_id"]),
                    seismic_magnitude=ctx_data.get("seismic_magnitude", 0.0),
                    seismic_region=ctx_data.get("seismic_region", ""),
                    epicenter_lat=ctx_data.get("epicenter_lat", 0.0),
                    epicenter_lon=ctx_data.get("epicenter_lon", 0.0),
                    depth_km=ctx_data.get("depth_km"),
                    trigger_time_utc=trigger_time,
                    latest_anomaly_score=ctx_data.get("latest_anomaly_score", 0.0),
                    dart_confirmation=ctx_data.get("dart_confirmation", False),
                    active_dart_stations=ctx_data.get("active_dart_stations", []),
                    stations_in_event_mode=ctx_data.get("stations_in_event_mode", []),
                    seismic_provider=ctx_data.get("seismic_provider", ""),
                    external_event_id=ctx_data.get("external_event_id", ""),
                    trigger_revision_id=ctx_data.get("trigger_revision_id", ""),
                    trigger_revision_sha256=ctx_data.get(
                        "trigger_revision_sha256", ""
                    ),
                    latest_revision_id=ctx_data.get("latest_revision_id", ""),
                    latest_revision_sha256=ctx_data.get(
                        "latest_revision_sha256", ""
                    ),
                    latest_revision_updated_utc=latest_rev_updated,
                    latest_revision_kafka_partition=ctx_data.get(
                        "latest_revision_kafka_partition"
                    ),
                    latest_revision_kafka_offset=ctx_data.get(
                        "latest_revision_kafka_offset"
                    ),
                    seismic_context_class=ctx_data.get("seismic_context_class", ""),
                )
            else:
                # A non-IDLE durable row MUST carry an event context: every
                # non-IDLE persist writes one (_transition clears context only
                # on IDLE). Without it the FSM would sit in MONITOR/ESCALATE
                # dropping new seismic triggers, with a monitor timeout that
                # no-ops on the missing context. Treat as recovery failure.
                raise ValueError(
                    f"non-IDLE durable state {self._state.value} has no event_context"
                )
            logger.info(
                "FSM state recovered from database: state=%s, has_event=%s",
                self._state.value,
                self._event_context is not None,
            )
            return True
        except Exception:
            # Determine what state we were attempting to restore (for operator context)
            stale_state = (
                row.get("current_state", "UNKNOWN") if isinstance(row, dict) else "UNKNOWN"
            )
            logger.critical(
                "FSM STATE RECOVERY FAILED: falling back to IDLE. "
                "The database indicated prior state=%s. "
                "If the system was tracking an active event (MONITOR/INVESTIGATE/"
                "ASSESS/ESCALATE), that event context is LOST. "
                "The system resumes from IDLE and will accept new triggers; "
                "operator MUST manually verify no active tsunami event is in "
                "progress (this flag is an alarm, not an interlock).",
                stale_state,
            )
            logger.exception("Recovery failure details")
            # Actually fall back: _state was already assigned from the row
            # before context parsing failed, so without this reset the FSM
            # would start in ESCALATE/MONITOR with no event context - an
            # inconsistent operator state the docstring promises never exists.
            self._state = SystemState.IDLE
            self._event_context = None
            # Durable record on the FIRST failure in this process: the sticky
            # flag is process-local, so a worker-only failure would otherwise
            # be invisible to the API/dashboard once the worker overwrites the
            # corrupt row. Duck-typed (the AuditWriter protocol only promises
            # write_transition); best-effort, never blocks the fallback.
            if not self._recovery_failed and self._audit_writer is not None:
                recorder = getattr(self._audit_writer, "record_recovery_failure", None)
                if recorder is not None:
                    try:
                        recorder(stale_state)
                    except Exception:
                        logger.exception("Failed to record recovery-failure audit entry")
            # Flag the failure so health-check endpoints and the operator dashboard
            # can surface it explicitly rather than relying on log visibility.
            self._recovery_failed = True
            return False


    def _persist_state(self) -> None:
        """Persist current FSM state to the database (fire-and-forget)."""
        if self._db_client is None:
            return

        ctx_dict: dict[str, Any] | None = None
        if self._event_context is not None:
            ctx_dict = {
                "event_id": str(self._event_context.event_id),
                "seismic_magnitude": self._event_context.seismic_magnitude,
                "seismic_region": self._event_context.seismic_region,
                "epicenter_lat": self._event_context.epicenter_lat,
                "epicenter_lon": self._event_context.epicenter_lon,
                "depth_km": self._event_context.depth_km,
                "trigger_time_utc": self._event_context.trigger_time_utc.isoformat(),
                "latest_anomaly_score": self._event_context.latest_anomaly_score,
                "dart_confirmation": self._event_context.dart_confirmation,
                "active_dart_stations": self._event_context.active_dart_stations,
                "stations_in_event_mode": self._event_context.stations_in_event_mode,
                # External seismic identity: both trigger
                # and latest identities must survive observation-only
                # checkpoints and restart.
                "seismic_provider": self._event_context.seismic_provider,
                "external_event_id": self._event_context.external_event_id,
                "trigger_revision_id": self._event_context.trigger_revision_id,
                "trigger_revision_sha256": (
                    self._event_context.trigger_revision_sha256
                ),
                "latest_revision_id": self._event_context.latest_revision_id,
                "latest_revision_sha256": (
                    self._event_context.latest_revision_sha256
                ),
                "latest_revision_updated_utc": (
                    self._event_context.latest_revision_updated_utc.isoformat()
                    if self._event_context.latest_revision_updated_utc is not None
                    else None
                ),
                "latest_revision_kafka_partition": (
                    self._event_context.latest_revision_kafka_partition
                ),
                "latest_revision_kafka_offset": (
                    self._event_context.latest_revision_kafka_offset
                ),
                "seismic_context_class": (
                    self._event_context.seismic_context_class
                ),
            }

        # The client swallows SQL and connection errors and reports False, so
        # the caller has to read the result: without this the only signal for a
        # failed durable write is a log line inside the client, and the
        # divergence below (worker in one state, database in another) goes
        # unannounced until a restart recovers the stale state.
        persisted = self._db_client.upsert_fsm_state(
            state=self._state.value,
            event_context=ctx_dict,
            sensor_degraded=self._sensor_degraded,
        )
        if persisted is False:
            logger.error(
                "FSM PERSISTENCE FAILED: in-memory state is %s but the durable "
                "row was not updated. The API reads durable state, so review "
                "and recovery will disagree with this worker. Operator "
                "attention required.",
                self._state.value,
            )

    @property
    def state(self) -> SystemState:
        return self._state

    @property
    def thresholds(self) -> ThresholdConfig:
        return self._thresholds

    @property
    def transition_history(self) -> list[TransitionRecord]:
        return list(self._transition_history)

    @property
    def event_context(self) -> EventContext | None:
        """Return a deep copy of the event context to prevent external mutation.

        EventContext is a mutable dataclass with list fields
        (active_dart_stations, stations_in_event_mode). Returning the
        internal reference would let callers corrupt FSM state.
        """
        if self._event_context is None:
            return None
        return copy.deepcopy(self._event_context)

    @property
    def sensor_degraded(self) -> bool:
        """True when DART station coverage is below the minimum for triangulation.

        This is a boolean flag, not a separate FSM state.
        The flag is set by evaluate_coverage() and logged to the audit
        trail. It does not trigger FSM state transitions - active event
        review (INVESTIGATE/ASSESS/ESCALATE) must not be interrupted by
        sensor coverage transients.
        """
        return self._sensor_degraded

    @property
    def recovery_failed(self) -> bool:
        """True if ANY recover_from_db() call failed in this process's lifetime.

        Deliberately STICKY: a later successful recovery does not clear it,
        because the API refreshes FSM state from the DB on every read
        endpoint, and clear-on-success would let a single corrupt-row
        failure (event context LOST) vanish from the dashboard as soon as
        the worker writes a fresh row - before any operator saw it. The
        alarm clears on process restart; an explicit operator
        acknowledgement path is future work.

        On recovery failure the FSM falls back to IDLE and the event
        context is lost. The flag is surfaced via /status, /api/fsm, and
        the Mission Control banner; processing continues from IDLE (the
        flag is an alarm, not an interlock).
        """
        return self._recovery_failed

    def evaluate_coverage(self, n_usable_stations: int) -> None:
        """Update sensor_degraded flag based on usable DART station count.

        Called after QC processing. If n_usable_stations < 2 (minimum
        for triangulation), sets sensor_degraded=True and logs a warning.
        Restoring coverage clears the flag.

        This replaces the original SENSOR_FAULT FSM state design.
        A separate state had an unresolved conflict: no transition path
        from SENSOR_FAULT to INVESTIGATE/ASSESS/ESCALATE, meaning a
        genuine event during sensor degradation could be lost.

        Args:
            n_usable_stations: Count of DART stations with QC result
                PASS or SUSPECT (usable for analysis).
        """
        was_degraded = self._sensor_degraded
        self._sensor_degraded = n_usable_stations < 2

        if self._sensor_degraded and not was_degraded:
            logger.warning(
                "SENSOR COVERAGE DEGRADED: only %d usable DART station(s) "
                "(minimum 2 for triangulation). FSM state: %s",
                n_usable_stations,
                self._state.value,
            )
        elif not self._sensor_degraded and was_degraded:
            logger.info(
                "Sensor coverage restored: %d usable DART station(s). FSM state: %s",
                n_usable_stations,
                self._state.value,
            )

        if self._sensor_degraded != was_degraded:
            # Persist on change only. The API and Mission Control read durable
            # state, and the other persist sites all hang off score updates or
            # transitions: in IDLE with no active event neither fires, which is
            # exactly the case where a dark sensor network matters most. Best
            # effort, like the transition persist: coverage is an alarm flag,
            # not an interlock, so a database failure must not stop processing.
            try:
                self._persist_state()
            except Exception:
                logger.error(
                    "Failed to persist sensor coverage change (degraded=%s)",
                    self._sensor_degraded,
                    exc_info=True,
                )

    def initialize_post_hoc_replay_event(
        self,
        *,
        origin_time_utc: datetime,
        seismic_identity: SeismicIdentity,
        trace_id: UUID | None = None,
    ) -> TransitionRecord:
        """Open a replay event using identity and origin alignment only.

        This path exists for retrospective replay of archived events,
        where the archived seismic record is a final post-hoc product. It
        deliberately accepts no magnitude, depth, region, or epicenter, so
        those final fields cannot trigger seismic escalation or enter the
        replay FSM state. Ocean scores may advance the event after MONITOR.
        """
        if self._state is not SystemState.IDLE:
            raise InvalidTransitionError(
                "Post-hoc replay event can only be initialized from IDLE"
            )
        if origin_time_utc.tzinfo is None:
            raise ValueError("origin_time_utc must be timezone-aware")
        if seismic_identity.context_class != "POST_HOC_FINAL_PRODUCT":
            raise ValueError(
                "Post-hoc replay initialization requires "
                "context_class=POST_HOC_FINAL_PRODUCT"
            )
        if not (
            seismic_identity.provider
            and seismic_identity.external_event_id
            and seismic_identity.revision_id
            and seismic_identity.revision_sha256
        ):
            raise ValueError("Post-hoc replay seismic identity is incomplete")

        previous_context = self._event_context
        context = EventContext(trigger_time_utc=origin_time_utc)
        context.seismic_provider = seismic_identity.provider
        context.external_event_id = seismic_identity.external_event_id
        context.trigger_revision_id = seismic_identity.revision_id
        context.trigger_revision_sha256 = seismic_identity.revision_sha256
        context.latest_revision_id = seismic_identity.revision_id
        context.latest_revision_sha256 = seismic_identity.revision_sha256
        context.latest_revision_updated_utc = seismic_identity.provider_updated_utc
        context.latest_revision_kafka_partition = seismic_identity.kafka_partition
        context.latest_revision_kafka_offset = seismic_identity.kafka_offset
        context.seismic_context_class = seismic_identity.context_class
        self._event_context = context

        try:
            return self._transition(
                to_state=SystemState.MONITOR,
                reason=(
                    "Post-hoc replay origin boundary: event identity and "
                    "origin alignment only"
                ),
                anomaly_score=None,
                seismic_magnitude=None,
                trace_id=trace_id,
            )
        except Exception:
            self._event_context = previous_context
            raise

    def evaluate_seismic_trigger(
        self,
        magnitude: float,
        region: str,
        epicenter_lat: float,
        epicenter_lon: float,
        tsunamigenic_zones: set[str],
        depth_km: float | None = None,
        origin_time_utc: datetime | None = None,
        trace_id: UUID | None = None,
        seismic_identity: SeismicIdentity | None = None,
    ) -> TransitionRecord | None:
        """Evaluate whether a seismic event triggers IDLE -> MONITOR.

        For large shallow earthquakes (M >= escalation_magnitude AND
        depth < seismic_escalation_depth_km), a chained MONITOR -> ESCALATE
        transition fires immediately, bypassing the DART anomaly score
        requirement. This aligns with PTWC/JMA operational practice of
        issuing initial warnings on seismic parameters alone; DART data
        refines the assessment when it arrives 15-30 minutes later.

        When ``seismic_identity`` is supplied (the live worker always
        supplies it), the created event context binds the external provider,
        event ID, and trigger revision identity at construction, and the
        latest admissible revision starts equal to the trigger. The trigger
        revision is immutable: nothing after event creation can rewrite it.
        Callers that pass no identity (offline
        scripts, unit tests) create an event with no external identity
        bound.

        Returns a TransitionRecord if a transition occurred, None otherwise.
        The returned record is the LAST transition (ESCALATE if seismic-only
        escalation fired, MONITOR otherwise).
        """
        if self._state != SystemState.IDLE:
            logger.warning(
                "SEISMIC TRIGGER DROPPED: M%.1f event (region=%s) received while FSM "
                "is in %s state. Single-event FSM cannot track concurrent events.",
                magnitude,
                region,
                self._state.value,
            )
            return None

        if magnitude < self._thresholds.seismic_min_magnitude:
            return None

        if region not in tsunamigenic_zones:
            return None

        prev_context = self._event_context
        # trigger_time_utc carries the seismic ORIGIN time when the caller
        # provides one (the worker always does), so DART event-mode evidence
        # can be scoped to this event: an observation timestamped before the
        # quake's origin cannot be this event's evidence. Falls back to wall
        # clock when no origin is supplied (callers that never gate on it).
        if origin_time_utc is not None:
            new_context = EventContext(
                seismic_magnitude=magnitude,
                seismic_region=region,
                epicenter_lat=epicenter_lat,
                epicenter_lon=epicenter_lon,
                depth_km=depth_km,
                trigger_time_utc=origin_time_utc,
            )
        else:
            new_context = EventContext(
                seismic_magnitude=magnitude,
                seismic_region=region,
                epicenter_lat=epicenter_lat,
                epicenter_lon=epicenter_lon,
                depth_km=depth_km,
            )
        if seismic_identity is not None:
            # Bind the external identity at event creation, before the
            # transition persists the context. The latest admissible
            # revision starts as the trigger revision itself.
            new_context.seismic_provider = seismic_identity.provider
            new_context.external_event_id = seismic_identity.external_event_id
            new_context.trigger_revision_id = seismic_identity.revision_id
            new_context.trigger_revision_sha256 = seismic_identity.revision_sha256
            new_context.latest_revision_id = seismic_identity.revision_id
            new_context.latest_revision_sha256 = seismic_identity.revision_sha256
            new_context.latest_revision_updated_utc = (
                seismic_identity.provider_updated_utc
            )
            new_context.latest_revision_kafka_partition = (
                seismic_identity.kafka_partition
            )
            new_context.latest_revision_kafka_offset = seismic_identity.kafka_offset
            new_context.seismic_context_class = seismic_identity.context_class
        self._event_context = new_context

        try:
            record = self._transition(
                to_state=SystemState.MONITOR,
                reason=f"Seismic trigger: M{magnitude} at {region}",
                anomaly_score=None,
                seismic_magnitude=magnitude,
                trace_id=trace_id,
            )
        except Exception:
            self._event_context = prev_context
            raise

        # Seismic-only escalation: large shallow earthquake in tsunamigenic
        # zone -> immediate ESCALATE without waiting for DART confirmation.
        # Fail-safe: depth_km must be known (not None) and < threshold.
        if (
            magnitude >= self._thresholds.escalation_magnitude
            and depth_km is not None
            and depth_km < self._thresholds.seismic_escalation_depth_km
        ):
            try:
                record = self._transition(
                    to_state=SystemState.ESCALATE,
                    reason=(
                        f"Seismic-only escalation: M{magnitude} at "
                        f"depth {depth_km:.0f} km in {region} "
                        f"(PTWC criteria: M>={self._thresholds.escalation_magnitude}, "
                        f"depth<{self._thresholds.seismic_escalation_depth_km:.0f} km). "
                        f"No DART event-mode activation required."
                    ),
                    anomaly_score=None,
                    seismic_magnitude=magnitude,
                    trace_id=trace_id,
                )
            except Exception:
                # If escalation fails, FSM remains in MONITOR - acceptable
                # degraded behavior. The normal DART-based path can still work.
                logger.warning(
                    "Seismic-only escalation failed for M%.1f; "
                    "FSM remains in MONITOR.",
                    magnitude,
                )

        return record

    def check_monitor_timeout(
        self, now: datetime | None = None
    ) -> TransitionRecord | None:
        """Check if MONITOR state has exceeded its timeout duration.

        Should be called periodically (e.g., on each polling cycle).
        If the time since the seismic trigger exceeds monitor_timeout_hours
        and the anomaly score is still below T1, returns to IDLE so the
        system can accept new seismic triggers.

        When the caller supplied origin_time_utc to evaluate_seismic_trigger
        (the worker always does), trigger_time_utc is the seismic ORIGIN
        time, so the timeout measures from the quake's origin rather than
        ingestion wall-clock time. A replayed or long-delayed feed whose
        origin is already past the timeout window times out on the first
        check.

        Returns a TransitionRecord if a timeout transition occurred, None otherwise.
        """
        if self._state != SystemState.MONITOR:
            return None

        if self._event_context is None:
            return None

        if now is None:
            now = datetime.now(UTC)
        elif now.tzinfo is None:
            raise ValueError("now must be timezone-aware")

        elapsed_hours = (
            now - self._event_context.trigger_time_utc
        ).total_seconds() / 3600.0

        if elapsed_hours < self._thresholds.monitor_timeout_hours:
            return None

        if self._event_context.latest_anomaly_score >= self._thresholds.t1:
            return None  # Score is elevated; don't timeout

        record = self._transition(
            to_state=SystemState.IDLE,
            reason=(
                f"Monitor timeout after {elapsed_hours:.1f}h "
                f"(limit: {self._thresholds.monitor_timeout_hours}h); "
                f"anomaly score {self._event_context.latest_anomaly_score:.3f} "
                f"below T1 ({self._thresholds.t1})"
            ),
            anomaly_score=self._event_context.latest_anomaly_score,
            seismic_magnitude=self._event_context.seismic_magnitude,
        )
        # _transition clears _event_context on the IDLE transition (before it
        # persists), so the durable row is consistent.
        return record

    def update_dart_confirmation(
        self,
        dart_confirmation: bool,
        stations_in_event_mode: list[str] | None = None,
    ) -> None:
        """Update DART confirmation status on the active event context.

        Called when one or more DART stations are in event mode. This is the
        only sanctioned way to set dart_confirmation; the event_context property
        returns a copy to prevent external mutation.

        Only meaningful in MONITOR, INVESTIGATE, ASSESS, or ESCALATE states.
        In IDLE the call is a no-op (no active event context).
        ESCALATE updates are allowed so the human reviewer sees the latest
        active DART stations.

        The flag is a one-way latch within an event: it is OR-ed with the
        current value here (never cleared), enforcing the per-event ratchet in
        code rather than by caller convention. It is reset only when the whole
        EventContext is cleared on resolve_event / monitor timeout. It is
        persisted with the current event-mode station list but CONDITIONALLY
        (only if the durable row is still this same event and not IDLE), so a
        stale worker whose event was already resolved cannot resurrect it, while
        the current worker makes the latch survive a restart.

        Station identifiers are sanitized to canonical five-digit DART IDs
        before being stored, and when a station list is supplied a NEW
        confirmation latches only if at least one valid station remains
        (an already-latched confirmation is never cleared).
        """
        if self._event_context is None:
            return
        if self._state == SystemState.IDLE:
            return
        valid_stations = None
        if stations_in_event_mode is not None:
            valid_stations = _sanitize_dart_station_ids(stations_in_event_mode)
            self._event_context.stations_in_event_mode = valid_stations
            self._event_context.active_dart_stations = valid_stations
            dart_confirmation = dart_confirmation and bool(valid_stations)
        self._event_context.dart_confirmation = (
            self._event_context.dart_confirmation or dart_confirmation
        )
        # Durably latch the ratchet for THIS event only (conditional on event_id
        # + non-IDLE in the durable row), so it survives a worker restart without
        # letting a stale worker resurrect a resolved event. Best-effort; no-op
        # without a DB.
        if self._event_context.dart_confirmation and self._db_client is not None:
            try:
                self._db_client.persist_dart_confirmation(
                    self._event_context.event_id,
                    list(self._event_context.stations_in_event_mode),
                )
            except Exception:
                logger.exception("Failed to persist dart_confirmation")

    def update_seismic_revision(self, identity: SeismicIdentity) -> bool:
        """Consider one received seismic revision for the active event.

        This is the only sanctioned way to advance the latest admissible
        revision. The revision supersedes the current
        latest only when ALL of the following hold:

        - an event is active with a bound external identity, and the
          revision's provider and external event ID match it exactly (an
          unrelated earthquake never replaces the active identity);
        - the revision carries a valid provider update time
          (``provider_updated_utc`` is not None; the worker nulls missing,
          malformed, and post-receipt-future values, which remain
          provenance in the raw archive but cannot silently supersede); and
        - it orders strictly after the incumbent by (provider update time,
          Kafka position, payload hash). An incumbent with no valid update
          time (possible only while the incumbent is the trigger itself)
          is superseded by any valid matching revision.

        Kafka position is compared as (partition, offset). Revisions of one
        external event share a message key and therefore a partition, so
        the offset comparison is well defined; a cross-partition comparison
        (possible only if the topic was repartitioned mid-event) is
        arbitrary but deterministic. Unknown coordinates compare as (-1,-1),
        so a known position wins at equal update times. An exactly equal
        ordering tuple (an at-least-once redelivery) does not supersede.

        The trigger revision fields are never touched. The in-memory update
        is authoritative for this process; durability is best-effort and
        CONDITIONAL (same event, non-IDLE durable row), mirroring
        persist_dart_confirmation, so a stale worker cannot resurrect a
        resolved event.

        Returns True when the revision superseded the incumbent.
        """
        ctx = self._event_context
        if ctx is None or self._state == SystemState.IDLE:
            return False
        if not ctx.seismic_provider or not ctx.external_event_id:
            # Event has no bound external identity (offline caller or a
            # pre-identity durable row): matching is impossible, so no
            # revision can attach to it.
            return False
        if (
            identity.provider != ctx.seismic_provider
            or identity.external_event_id != ctx.external_event_id
        ):
            return False
        updated = identity.provider_updated_utc
        if updated is None:
            return False

        candidate = (
            updated,
            identity.kafka_partition if identity.kafka_partition is not None else -1,
            identity.kafka_offset if identity.kafka_offset is not None else -1,
            identity.revision_sha256,
        )
        if ctx.latest_revision_updated_utc is not None:
            incumbent = (
                ctx.latest_revision_updated_utc,
                (
                    ctx.latest_revision_kafka_partition
                    if ctx.latest_revision_kafka_partition is not None
                    else -1
                ),
                (
                    ctx.latest_revision_kafka_offset
                    if ctx.latest_revision_kafka_offset is not None
                    else -1
                ),
                ctx.latest_revision_sha256,
            )
            if candidate <= incumbent:
                return False

        ctx.latest_revision_id = identity.revision_id
        ctx.latest_revision_sha256 = identity.revision_sha256
        ctx.latest_revision_updated_utc = updated
        ctx.latest_revision_kafka_partition = identity.kafka_partition
        ctx.latest_revision_kafka_offset = identity.kafka_offset
        logger.info(
            "Seismic revision superseded latest for event %s: %s",
            ctx.external_event_id,
            identity.revision_id,
        )
        if self._db_client is not None:
            try:
                self._db_client.persist_seismic_revision(
                    ctx.event_id,
                    {
                        "latest_revision_id": ctx.latest_revision_id,
                        "latest_revision_sha256": ctx.latest_revision_sha256,
                        "latest_revision_updated_utc": updated.isoformat(),
                        "latest_revision_kafka_partition": (
                            ctx.latest_revision_kafka_partition
                        ),
                        "latest_revision_kafka_offset": (
                            ctx.latest_revision_kafka_offset
                        ),
                    },
                )
            except Exception:
                logger.exception("Failed to persist seismic revision identity")
        return True

    def evaluate_anomaly_score(
        self, score: float, *, trace_id: UUID | None = None,
    ) -> TransitionRecord | None:
        """Evaluate anomaly score against thresholds for state transitions.

        Returns a TransitionRecord if a transition occurred, None otherwise.

        If the transition fails (e.g. audit write error), the
        ``latest_anomaly_score`` mutation is rolled back to prevent
        stale scores from affecting timeout logic or dashboard state.
        """
        if not (0.0 <= score <= 1.0) or math.isnan(score):
            raise ValueError(f"anomaly score must be in [0, 1] and finite, got {score}")
        prev_score: float | None = None
        if self._event_context is not None:
            prev_score = self._event_context.latest_anomaly_score
            self._event_context.latest_anomaly_score = score

        new_state: SystemState | None = None
        reason = ""

        if self._state == SystemState.IDLE:
            logger.debug(
                "Anomaly score %.3f received in IDLE state (no active event); ignoring.",
                score,
            )
            return None

        if self._state == SystemState.ESCALATE:
            logger.debug(
                "Anomaly score %.3f received in ESCALATE state "
                "(awaiting human decision); score updated but no transition.",
                score,
            )
            self._persist_state()
            return None

        if self._state == SystemState.MONITOR:
            if score >= self._thresholds.t1:
                new_state = SystemState.INVESTIGATE
                reason = f"Anomaly score {score:.3f} >= T1 ({self._thresholds.t1})"
            elif (
                # Seismic override in MONITOR: without this branch the
                # M7.5+ / DART event-mode override is unreachable from
                # MONITOR (the INVESTIGATE and ASSESS overrides require
                # the score to have crossed T1 at least once). A large
                # event whose seismic-only escalation did not fire (depth
                # unknown or >= 100 km) but whose DART stations switched
                # to event mode must not sit in MONITOR on a sub-T1
                # score: advance one state per evaluation (MONITOR ->
                # INVESTIGATE -> ASSESS -> ESCALATE), each step audited,
                # mirroring the override in the branches below.
                self._event_context is not None
                and self._event_context.seismic_magnitude
                >= self._thresholds.escalation_magnitude
                and self._event_context.dart_confirmation
            ):
                new_state = SystemState.INVESTIGATE
                reason = (
                    f"Anomaly score {score:.3f} below T1 ({self._thresholds.t1}) "
                    f"but M{self._event_context.seismic_magnitude} + DART event-mode "
                    f"activation prevents holding in MONITOR; advancing to "
                    f"INVESTIGATE for seismic override"
                )

        elif self._state == SystemState.INVESTIGATE:
            if score >= self._thresholds.t2:
                new_state = SystemState.ASSESS
                reason = f"Anomaly score {score:.3f} >= T2 ({self._thresholds.t2})"
            elif (
                # Seismic override, evaluated for any sub-T2 score and ahead
                # of the de-escalation test. It must stay at this level: nested
                # inside "score < t1" it would leave [T1, T2) uncovered, and an
                # M7.5+ event with DART event-mode activation would take no
                # branch at all and stall in INVESTIGATE, which has no timeout
                # to release it and which blocks the single-event FSM. The
                # physical measurement trumps a low anomaly score, so advance
                # to ASSESS and let the ASSESS override carry it onward.
                # Mirrors the ASSESS branch below.
                self._event_context is not None
                and self._event_context.seismic_magnitude
                >= self._thresholds.escalation_magnitude
                and self._event_context.dart_confirmation
            ):
                new_state = SystemState.ASSESS
                reason = (
                    f"Anomaly score {score:.3f} below T2 ({self._thresholds.t2}) "
                    f"but M{self._event_context.seismic_magnitude} + DART event-mode "
                    f"activation prevents holding in INVESTIGATE; advancing to ASSESS "
                    f"for seismic override"
                )
            elif score < self._thresholds.t1:
                new_state = SystemState.MONITOR
                reason = f"Anomaly score {score:.3f} dropped below T1 ({self._thresholds.t1})"

        elif self._state == SystemState.ASSESS:
            if score >= self._thresholds.t3:
                new_state = SystemState.ESCALATE
                reason = f"Anomaly score {score:.3f} >= T3 ({self._thresholds.t3})"
            elif (
                # Seismic override: large earthquakes with DART confirmation
                # bypass the T3 anomaly threshold. This covers the scenario
                # where deep-ocean confirmation is strong but the anomaly
                # ensemble hasn't converged above T3 yet.
                #
                # DESIGN DECISION: This elif intentionally takes priority
                # over the de-escalation check (score < T2) below. If a
                # DART buoy has entered high-cadence event mode (an
                # activation, not an independent waveform confirmation;
                # see EventContext.dart_confirmation) and the earthquake
                # is M7.5+, escalating toward human review is the
                # fail-safe choice regardless of the ensemble score.
                self._event_context is not None
                and self._event_context.seismic_magnitude >= self._thresholds.escalation_magnitude
                and self._event_context.dart_confirmation
            ):
                new_state = SystemState.ESCALATE
                reason = (
                    f"M{self._event_context.seismic_magnitude} >= "
                    f"{self._thresholds.escalation_magnitude} with DART event-mode activation"
                )
            elif score < self._thresholds.t2:
                new_state = SystemState.INVESTIGATE
                reason = f"Anomaly score {score:.3f} dropped below T2 ({self._thresholds.t2})"

        if new_state is None:
            # Keep API/BFF readers aligned with the worker's latest evaluated
            # score even when no state transition occurs.
            self._persist_state()
            return None

        try:
            return self._transition(
                to_state=new_state,
                reason=reason,
                anomaly_score=score,
                seismic_magnitude=(
                    self._event_context.seismic_magnitude if self._event_context else None
                ),
                trace_id=trace_id,
            )
        except Exception:
            # Rollback: restore previous score on failed transition
            if self._event_context is not None and prev_score is not None:
                self._event_context.latest_anomaly_score = prev_score
            raise

    def resolve_event(self, *, trace_id: UUID | None = None) -> TransitionRecord | None:
        """Low-level transition from ESCALATE to IDLE.

        This method does not establish caller identity or authority. No current
        API route invokes it; any future event-disposition path must enforce
        trusted-human authorization before calling it.
        """
        if self._state != SystemState.ESCALATE:
            return None

        record = self._transition(
            to_state=SystemState.IDLE,
            reason="Event closed by explicit disposition",
            anomaly_score=None,
            seismic_magnitude=None,
            trace_id=trace_id,
        )
        # _transition clears _event_context on the IDLE transition (before it
        # persists), so the durable row is consistent.
        return record

    def _transition(
        self,
        to_state: SystemState,
        reason: str,
        anomaly_score: float | None,
        seismic_magnitude: float | None,
        trace_id: UUID | None = None,
    ) -> TransitionRecord:
        """Execute a state transition and record it.

        Validates that the transition is allowed before executing.
        """
        allowed = VALID_TRANSITIONS.get(self._state, set())
        if to_state not in allowed:
            raise InvalidTransitionError(
                f"Transition from {self._state} to {to_state} is not allowed. "
                f"Valid transitions: {allowed}"
            )

        record = TransitionRecord(
            event_id=self._event_context.event_id if self._event_context else None,
            trace_id=trace_id,
            from_state=self._state,
            to_state=to_state,
            trigger_reason=reason,
            anomaly_score=anomaly_score,
            seismic_magnitude=seismic_magnitude,
            thresholds_used=self._thresholds,
        )

        # Write audit BEFORE changing state. If the audit write fails,
        # the transition is aborted and the FSM stays in its current state.
        if self._audit_writer is not None:
            self._audit_writer.write_transition(record)

        self._state = to_state
        self._transition_history.append(record)

        # Observability only: a metrics failure must never disrupt an FSM
        # transition (otherwise it would propagate into the caller's
        # score-rollback path in evaluate_anomaly_score).
        try:
            from hazard_assessment.telemetry.metrics import record_fsm_transition
            record_fsm_transition(to_state.value)
        except Exception:  # pragma: no cover - defensive
            logger.debug("FSM transition metric failed", exc_info=True)

        if len(self._transition_history) > self.MAX_HISTORY:
            self._transition_history = self._transition_history[-self.MAX_HISTORY :]

        # Returning to IDLE means the event is over: clear the event context
        # here, before persisting, so the durable fsm_current_state row never
        # stores IDLE alongside a stale active event_context. Without this,
        # recover_from_db would reload an inconsistent IDLE-with-active-event
        # (and leak the resolved event's dart_confirmation across resolution).
        # The TransitionRecord above already captured the event_id, so the audit
        # entry still links to the resolved event.
        if to_state == SystemState.IDLE:
            self._event_context = None

        # Persist state to DB after successful transition.
        # Wrapped in try/except to prevent DB failures from aborting the
        # in-memory transition. The in-memory state is authoritative during
        # runtime; the DB is best-effort persistence for crash recovery.
        try:
            self._persist_state()
        except Exception:
            logger.error(
                "FSM PERSISTENCE FAILED: in-memory state is %s but database "
                "may still reflect %s. On restart, recover_from_db() will "
                "load the stale DB state. Operator attention required.",
                to_state.value,
                record.from_state.value,
                exc_info=True,
            )

        return record


class InvalidTransitionError(Exception):
    """Raised when an invalid state transition is attempted."""
