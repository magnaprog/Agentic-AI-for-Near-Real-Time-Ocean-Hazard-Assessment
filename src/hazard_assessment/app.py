"""FastAPI application entry point for the hazard assessment system.

Provides HTTP endpoints for health checks, system status, and
internal data access used by the Mission Control dashboard.
The API is internal-only. It does not serve public-facing
assessment products.

Note on policy enforcement: The ``/api/policy/check`` endpoint
provides a passive query interface. No agent or pipeline node calls
it, so the permission matrix documents the intended capability
envelope rather than enforcing it. Active enforcement (intercepting
every agent action) would require orchestrator integration; this
endpoint is the building block for that, not the mechanism itself.

EscalationPacket generator and routing.
Caller-gated assessment review bound to durable evidence.

"""

from __future__ import annotations

import logging
import os
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from hmac import compare_digest
from typing import Any, Literal
from uuid import UUID, uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, ValidationError

from hazard_assessment import __version__
from hazard_assessment.agents.base import AgentCapability
from hazard_assessment.audit.logger import AuditEntry, AuditLogger
from hazard_assessment.config.settings import ThresholdSettings
from hazard_assessment.orchestrator.states import (
    EventContext,
    FSMOrchestrator,
    SystemState,
    TransitionRecord,
)
from hazard_assessment.policy.approval import (
    check_policy,
    denial_to_response,
    load_permission_matrix,
    log_denial,
    log_policy_result,
)
from hazard_assessment.policy.guardrails import scan_text
from hazard_assessment.schemas.envelope import DataSource, InputRef
from hazard_assessment.schemas.escalation import EscalationPacket
from hazard_assessment.schemas.human_decision import (
    AssessmentReviewDecision,
    IdentityAssurance,
    ReviewDecision,
)
from hazard_assessment.workers.reviewer_packet import canonical_packet_hash

logger = logging.getLogger(__name__)

API_KEY_HEADER_NAME = "X-Hazard-Api-Key"
REVIEWER_ID_HEADER_NAME = "X-Reviewer-Id"

#: Bound on the caller-asserted reviewer identity. The value is written to the
#: append-only audit trail as the record's producer and echoed into logs, so
#: it must be bounded and filtered: an unbounded field accepts a
#: 5000-character identity, and an unfiltered one accepts ANSI escape
#: sequences that render as control codes in any terminal reading the trail.
#: The identity is only caller-asserted; this bounds its shape, not its
#: truthfulness.
REVIEWER_ID_MAX_LENGTH = 128
# Row cap for the trace/event-scoped lineage and activity-report queries. A
# single pipeline run produces well under this many audit entries, so the cap is
# effectively unreachable for trace scope; the responses still expose a
# ``truncated`` flag so a caller can detect the rare overflow (e.g. a long-lived
# event with many runs) rather than mistaking a capped count for the true total.
_LINEAGE_QUERY_LIMIT = 1000
_api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)

# Set during lifespan startup; empty string signals "not yet initialized".
_HAZARD_API_KEY: str = ""


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Validate runtime requirements and initialize persistence on startup.

    When DB_HOST is set, connects to TimescaleDB for persistent FSM state
    and audit trail. Otherwise falls back to in-memory (development/test).
    """
    global _HAZARD_API_KEY, _audit, _fsm, _db_client
    api_key = os.getenv("HAZARD_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "HAZARD_API_KEY is required for protected API endpoints"
        )
    _HAZARD_API_KEY = api_key

    thresholds = ThresholdSettings().to_threshold_config()
    _db_client = None
    _audit = AuditLogger()
    _fsm = FSMOrchestrator(thresholds=thresholds, audit_writer=_audit)

    # Initialize database persistence if configured
    db_host = os.getenv("DB_HOST", "").strip()
    if db_host:
        try:
            from hazard_assessment.storage.client import ClientConfig, DatabaseClient

            config = ClientConfig.from_env(role="orchestrator_writer")
            _db_client = DatabaseClient(config)
            if _db_client.is_connected:
                _audit = AuditLogger(db_client=_db_client)
                _fsm = FSMOrchestrator(
                    thresholds=thresholds,
                    audit_writer=_audit,
                    db_client=_db_client,
                )
                _fsm.recover_from_db()
                logger.info("Database persistence enabled (%s)", db_host)
            else:
                logger.warning("Database connection failed - using in-memory state")
                _db_client = None
        except Exception:
            logger.exception("Failed to initialize database - using in-memory state")
            _db_client = None

    from hazard_assessment.telemetry.tracing import configure_tracer_provider

    configure_tracer_provider(os.getenv("OTLP_ENDPOINT"))
    yield
    # Shutdown: close database pools. The investigator's pool is opened lazily
    # on first use of /api/investigate, so it may or may not exist.
    global _investigator_db
    if _db_client is not None:
        _db_client.close()
    with _investigator_db_lock:
        if _investigator_db is not None:
            _investigator_db.close()
            _investigator_db = None


app = FastAPI(
    title="Agentic AI for Near-Real-Time Ocean Hazard Assessment",
    version=__version__,
    description="Internal API for the near-real-time ocean hazard assessment pipeline",
    docs_url=None,
    redoc_url=None,
    openapi_url=None,
    lifespan=_lifespan,
)


def _api_key_matches(provided: str, expected: str) -> bool:
    """Compare API keys in constant time without raising on any input.

    ``hmac.compare_digest`` rejects ``str`` arguments holding non-ASCII
    characters with a TypeError rather than returning False. Header values
    arrive decoded as latin-1, so a single high byte in the key made this
    check raise and turned a 401 into an unhandled 500. Comparing the UTF-8
    encodings keeps the comparison constant-time and makes every invalid key
    an ordinary mismatch.
    """
    return compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def require_internal_api_key(
    x_hazard_api_key: str | None = Security(_api_key_header),
) -> None:
    if (
        not _HAZARD_API_KEY
        or x_hazard_api_key is None
        or not _api_key_matches(x_hazard_api_key, _HAZARD_API_KEY)
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")

# Module-level instances. When DB_HOST is set, these are replaced during
# lifespan startup with DB-backed instances.
_db_client: Any = None  # Optional[DatabaseClient], typed as Any to avoid import
#: Opened on first use of /api/investigate. Separate from _db_client because
#: findings must be written as investigator_writer, a role the API's own
#: orchestrator_writer identity is deliberately not granted.
_investigator_db: Any | None = None
_investigator_db_lock = threading.Lock()
_audit = AuditLogger()
_fsm = FSMOrchestrator(audit_writer=_audit)
_permission_matrix = load_permission_matrix()

# Guards mutable FSM, audit, and escalation-packet state against concurrent
# requests.  FastAPI runs sync endpoints in a threadpool, so a threading.Lock
# is required (not asyncio.Lock).
_fsm_lock = threading.Lock()

# Active escalation packet, set when FSM enters ESCALATE.
# Only one escalation can be active at a time (single-event FSM).
_active_escalation_packet: EscalationPacket | None = None


# ---------- Shared helpers ----------


def _parse_uuid_param(value: str | None, field_name: str) -> UUID | None:
    """Parse a UUID string from a request parameter, raising 400 on invalid input."""
    if not value:
        return None
    try:
        return UUID(value)
    except ValueError:
        raise HTTPException(
            status_code=400, detail=f"Invalid UUID for {field_name}: {value}"
        ) from None


def _serialize_audit_entry(entry: AuditEntry) -> dict[str, Any]:
    """Serialize an AuditEntry to a JSON-safe dict for API responses."""
    return {
        "entry_id": str(entry.entry_id),
        "timestamp_utc": entry.timestamp_utc.isoformat(),
        "event_id": str(entry.event_id) if entry.event_id else None,
        "trace_id": str(entry.trace_id) if entry.trace_id else None,
        "event_type": entry.event_type,
        "producer": entry.producer,
        "data": entry.data,
    }


# ---------- Agent registry (read-only) ----------

def _load_agent_manifests() -> tuple[dict[str, Any], ...]:
    from hazard_assessment.agents.anomaly_agent import _MANIFEST as ANOMALY_MANIFEST
    from hazard_assessment.agents.qc_agent import _MANIFEST as QC_MANIFEST
    from hazard_assessment.agents.report_agent import _MANIFEST as REPORT_MANIFEST
    from hazard_assessment.agents.scenario_agent import _MANIFEST as SCENARIO_MANIFEST
    from hazard_assessment.agents.verification_agent import _MANIFEST as VERIFY_MANIFEST

    manifests = [
        (QC_MANIFEST, "LIVE_WORKER"),
        (ANOMALY_MANIFEST, "LIVE_WORKER"),
        (SCENARIO_MANIFEST, "OFFLINE_EVALUATION_ONLY"),
        (VERIFY_MANIFEST, "OFFLINE_EVALUATION_ONLY"),
        (REPORT_MANIFEST, "OFFLINE_EVALUATION_ONLY"),
    ]
    return tuple(
        {
            "name": manifest.name,
            "version": manifest.version,
            "execution_path": execution_path,
            "description": manifest.description,
        }
        for manifest, execution_path in manifests
    )


_agent_manifests = _load_agent_manifests()


# ---------- EscalationPacket generator ----------


def generate_escalation_packet(
    ctx: EventContext,
    transition: TransitionRecord | None,
    audit_logger: AuditLogger,
    escalation_magnitude: float = 7.5,
    t3_threshold: float = 0.85,
) -> EscalationPacket:
    """Generate an EscalationPacket when the FSM enters ESCALATE.

    Assembles the evidence bundle from the event context, transition
    record, and the anomaly timeline from the audit trail. Callers cannot
    supply scenario or verification dictionaries: the live
    worker fail-closes those stages at ESCALATE, so no trusted source for
    that content exists and the corresponding packet fields stay None.

    Target: packet created within 60 seconds of escalation trigger.
    """
    # Query the durable state_transition audit entries once, sorted oldest-first
    # (query_entries returns newest-first). Used both to recover a worker-driven
    # transition and to build the chronological anomaly timeline. With a DB this
    # includes transitions written by the worker process.
    transition_entries = sorted(
        audit_logger.query_entries(
            event_id=ctx.event_id,
            event_type="state_transition",
            limit=_LINEAGE_QUERY_LIMIT,
        ),
        key=lambda e: e.timestamp_utc,
    )

    # Recover the triggering ESCALATE transition from the audit trail when the
    # API has no in-memory transition (a worker-driven ESCALATE: recover_from_db
    # restores state but not transition_history). This gives the packet the real
    # trigger reason, time, and trace_id rather than unknown / now / None. Note a
    # seismic-only transition is not a pipeline run, so its state_transition entry
    # carries no trace_id; such a packet has event lineage (event_id) but no
    # trace lineage, which is correct (there is no pipeline execution to trace).
    recovered_trigger: str | None = None
    recovered_trace_id: UUID | None = None
    recovered_time: datetime | None = None
    if transition is None:
        for e in transition_entries:  # ascending: keep the latest ESCALATE
            if e.data.get("to_state") == "ESCALATE":
                recovered_trigger = e.data.get("trigger_reason") or None
                recovered_trace_id = e.trace_id
                recovered_time = e.timestamp_utc

    # Build criticality reasons
    reasons: list[str] = []
    trigger = "unknown"
    if transition and transition.trigger_reason:
        trigger = transition.trigger_reason
        reasons.append(trigger)
    elif recovered_trigger:
        trigger = recovered_trigger
        reasons.append(trigger)
    if ctx.seismic_magnitude >= escalation_magnitude:
        reasons.append(f"Major earthquake M{ctx.seismic_magnitude}")
    if ctx.dart_confirmation:
        reasons.append("DART event-mode activation")
    if ctx.latest_anomaly_score >= t3_threshold:
        reasons.append(f"Anomaly score {ctx.latest_anomaly_score:.3f} above T3")
    if not reasons:
        reasons.append("Escalation triggered by FSM transition")

    # Build the chronological anomaly timeline from the sorted entries.
    anomaly_timeline: list[dict[str, Any]] = []
    for entry in transition_entries:
        if entry.data.get("anomaly_score") is not None:
            anomaly_timeline.append({
                "timestamp_utc": entry.timestamp_utc.isoformat(),
                "anomaly_score": entry.data["anomaly_score"],
                "from_state": entry.data.get("from_state", ""),
                "to_state": entry.data.get("to_state", ""),
            })

    # Build recommended action based on context
    recommended_action = "Human review required"
    if ctx.seismic_magnitude >= 8.0 and ctx.dart_confirmation:
        recommended_action = (
            "URGENT: Major tsunami-generating earthquake with DART event-mode activation. "
            "Review scenario assessment and coastal proxies immediately."
        )
    elif ctx.dart_confirmation:
        recommended_action = (
            "DART event-mode detected. Review verification results and "
            "scenario inversion before deciding."
        )

    # P2 enforcement: scan human-readable text fields for prohibited terms.
    # Uses .violations check only (same pattern as format_abstain/format_human_decision).
    # Escalation packets are internal evidence bundles; no disclaimer is required.
    # Covers every human-readable prose field: reasons, the recommended
    # action, and the seismic region. Identifier fields (station IDs,
    # InputRef.record_id) are deliberately not term-scanned; they are
    # shape-constrained identifiers rather than prose.
    scannable = " ".join(reasons + [recommended_action, ctx.seismic_region])
    scan_result = scan_text(scannable)
    from hazard_assessment.telemetry.metrics import record_guardrail_scan
    record_guardrail_scan(passed=not scan_result.violations)
    if scan_result.violations:
        terms = [v.term for v in scan_result.violations]
        raise ValueError(
            f"Escalation packet contains prohibited alert terminology: {terms}"
        )

    # Assemble raw-input provenance from the audit entries the worker recorded
    # for this event (query_entries reads the durable audit_events table when a
    # DB is configured, so worker-process entries are visible). Deduped by hash;
    # a malformed entry is skipped so one bad row cannot fail the packet
    # (InputRef.sha256 is regex-strict). The seismic trigger's provenance is
    # recorded under a distinct event_type and ALWAYS included (the decision-
    # critical input, never pushed out of the capped observation read).
    # NOTE: the SQL get_provenance() lineage path is a SEPARATE mechanism that
    # joins audit_events.input_hashes; it is populated by the worker's
    # processed_features producer (qc_report / anomaly_score rows) and exposed
    # via /api/lineage/provenance/{trace_id}, not by these packet entries.
    input_refs: list[InputRef] = []
    seen_hashes: set[str] = set()

    def _append_provenance(entries: list[AuditEntry]) -> None:
        for prov in entries:
            sha = prov.data.get("sha256")
            if not isinstance(sha, str) or sha in seen_hashes:
                continue
            try:
                input_refs.append(
                    InputRef(
                        source=DataSource(prov.data.get("source", "internal")),
                        record_id=str(prov.data.get("record_id", "")),
                        sha256=sha,
                    )
                )
                seen_hashes.add(sha)
            except (ValidationError, ValueError):
                logger.warning(
                    "Skipping malformed provenance entry for event %s",
                    ctx.event_id,
                )

    # Seismic trigger provenance first (always included, never capped).
    _append_provenance(
        audit_logger.query_entries(
            event_id=ctx.event_id,
            event_type="seismic_provenance",
            limit=_LINEAGE_QUERY_LIMIT,
        )
    )
    # Then capped observation provenance (DART/CO-OPS).
    _append_provenance(
        audit_logger.query_entries(
            event_id=ctx.event_id,
            event_type="input_provenance",
            limit=_LINEAGE_QUERY_LIMIT,
        )
    )
    # Disclose truncation precisely: the worker records a one-time
    # "provenance_capped" marker when it drops observation provenance beyond its
    # per-event cap (a read-side count cannot detect it, since the excess never
    # reaches the audit trail).
    input_refs_truncated = bool(
        audit_logger.query_entries(
            event_id=ctx.event_id, event_type="provenance_capped", limit=1
        )
    )

    # Use the transition timestamp as the escalation time when available
    # (in-memory, else recovered from the audit trail), rather than the model
    # construction time (which is slightly later).
    escalation_time = (
        transition.timestamp_utc
        if transition
        else (recovered_time or datetime.now(UTC))
    )

    packet = EscalationPacket(
        producer="escalation_generator",
        event_id=ctx.event_id,
        trace_id=transition.trace_id if transition else recovered_trace_id,
        escalation_trigger=trigger,
        escalation_time_utc=escalation_time,
        criticality_reasons=reasons,
        anomaly_timeline=anomaly_timeline,
        seismic_magnitude=ctx.seismic_magnitude,
        seismic_region=ctx.seismic_region,
        epicenter_lat=ctx.epicenter_lat,
        epicenter_lon=ctx.epicenter_lon,
        latest_anomaly_score=ctx.latest_anomaly_score,
        dart_confirmation=ctx.dart_confirmation,
        active_dart_stations=list(ctx.active_dart_stations),
        recommended_action=recommended_action,
        input_refs=input_refs,
        input_refs_truncated=input_refs_truncated,
    )

    # Log escalation packet generation to audit trail
    audit_logger.append(
        AuditEntry(
            event_id=ctx.event_id,
            trace_id=transition.trace_id if transition else recovered_trace_id,
            event_type="escalation_packet_generated",
            producer="escalation_generator",
            data={
                "packet_id": str(packet.handoff_id),
                "packet_hash": packet.packet_hash,
                "criticality_reasons": reasons,
                "trigger": trigger,
            },
        )
    )

    return packet


# ---------- Request models ----------

class ReviewRequest(BaseModel):
    """Caller-gated review of one durable assessment packet.

    Reviewer identity comes from ``X-Reviewer-Id``. Packet row identity and
    canonical hash prove which immutable evidence was submitted for review;
    the API derives and records the bound assessment ID and scientific hash
    from that packet rather than trusting caller-supplied assessment fields.
    """

    event_id: str
    decision: Literal["APPROVE", "REJECT", "DEFER"]
    decision_reason: str = Field(min_length=1, max_length=5000)
    escalation_packet_row_id: int = Field(ge=1)
    escalation_packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    trace_id: str | None = Field(
        default=None,
        description="Pipeline trace_id (UUID) for lineage correlation.",
    )


class PolicyCheckRequest(BaseModel):
    """Payload for validating an agent action against the permission matrix."""

    agent_name: str
    capability: str
    human_decision_present: bool = False


# ---------- State and audit endpoints ----------


@app.get("/health")
async def health_check() -> dict[str, str]:
    """Liveness probe for container orchestration.

    Declared async deliberately. Every other route here is a sync def, so it
    runs in Starlette's bounded threadpool, and nine of them call
    _refresh_fsm_from_db, which blocks for the pool timeout when the database
    is unreachable. With Mission Control polling and Prometheus scraping on
    timers, a database outage fills that threadpool and a sync liveness probe
    queues behind the very failure it exists to report independently. This
    body touches nothing, so running it on the event loop keeps it answering.
    """
    return {"status": "healthy", "version": __version__}


def _refresh_fsm_from_db() -> None:
    """Reload FSM state from the database so reads reflect worker transitions.

    Best-effort and a no-op without a database. Must be called while holding
    ``_fsm_lock``. Durable transition history is read separately from audit.
    """
    if _db_client is not None:
        try:
            _fsm.recover_from_db()
        except Exception:
            logger.exception("Failed to refresh FSM state from database")


def _event_transition_history(event_id: UUID | None) -> list[dict[str, Any]]:
    """Return observed transitions, using durable worker audit when configured."""
    if _audit.durable_persistence_configured:
        try:
            durable_entries = _audit.query_entries(
                event_id=event_id,
                event_type="state_transition",
                limit=_LINEAGE_QUERY_LIMIT,
                raise_on_error=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Durable FSM transition history is unavailable",
            ) from exc
        entries = sorted(
            durable_entries,
            key=lambda entry: entry.timestamp_utc,
        )
        return [
            {
                "transition_id": str(
                    entry.data.get("transition_id") or entry.entry_id
                ),
                "event_id": str(entry.event_id) if entry.event_id else None,
                "timestamp_utc": entry.timestamp_utc.isoformat(),
                "from_state": str(entry.data.get("from_state", "")).rsplit(".", 1)[-1],
                "to_state": str(entry.data.get("to_state", "")).rsplit(".", 1)[-1],
                "trigger_reason": str(entry.data.get("trigger_reason", "")),
                "anomaly_score": entry.data.get("anomaly_score"),
                "seismic_magnitude": entry.data.get("seismic_magnitude"),
            }
            for entry in entries
        ]

    return [
        {
            "transition_id": str(transition.transition_id),
            "event_id": str(transition.event_id) if transition.event_id else None,
            "timestamp_utc": transition.timestamp_utc.isoformat(),
            "from_state": transition.from_state.value,
            "to_state": transition.to_state.value,
            "trigger_reason": transition.trigger_reason,
            "anomaly_score": transition.anomaly_score,
            "seismic_magnitude": transition.seismic_magnitude,
        }
        for transition in _fsm.transition_history
        if event_id is None or transition.event_id == event_id
    ]


@app.get("/status", dependencies=[Depends(require_internal_api_key)])
def system_status() -> dict[str, str | bool]:
    """Return current FSM state and event tracking status."""
    with _fsm_lock:
        _refresh_fsm_from_db()
        ctx = _fsm.event_context
        return {
            "fsm_state": _fsm.state.value,
            "has_active_event": ctx is not None,
            "event_id": str(ctx.event_id) if ctx else "",
            "recovery_failed": _fsm.recovery_failed,
            "timestamp_utc": datetime.now(UTC).isoformat(),
        }


# ---------- Mission Control data endpoints ----------


@app.get("/api/fsm", dependencies=[Depends(require_internal_api_key)])
def get_fsm_state() -> dict[str, Any]:
    """Return full FSM snapshot: state, event context, thresholds, history."""
    with _fsm_lock:
        _refresh_fsm_from_db()
        ctx = _fsm.event_context
        thresholds = _fsm.thresholds
        return {
            "fsm_state": _fsm.state.value,
            "sensor_degraded": _fsm.sensor_degraded,
            "recovery_failed": _fsm.recovery_failed,
            "has_active_event": ctx is not None,
            "event_context": (
                {
                    "event_id": str(ctx.event_id),
                    "seismic_magnitude": ctx.seismic_magnitude,
                    "seismic_region": ctx.seismic_region,
                    "epicenter_lat": ctx.epicenter_lat,
                    "epicenter_lon": ctx.epicenter_lon,
                    "trigger_time_utc": ctx.trigger_time_utc.isoformat(),
                    "latest_anomaly_score": ctx.latest_anomaly_score,
                    "dart_confirmation": ctx.dart_confirmation,
                    "active_dart_stations": ctx.active_dart_stations,
                    "stations_in_event_mode": ctx.stations_in_event_mode,
                }
                if ctx
                else None
            ),
            "thresholds": {
                "basin": thresholds.basin,
                "t1": thresholds.t1,
                "t2": thresholds.t2,
                "t3": thresholds.t3,
            },
            "transition_history": _event_transition_history(
                ctx.event_id if ctx is not None else None
            ),
        }


@app.get("/api/agents", dependencies=[Depends(require_internal_api_key)])
def get_agents() -> list[dict[str, Any]]:
    """Return component manifests and their implemented execution path."""
    return list(_agent_manifests)


@app.get("/api/audit", dependencies=[Depends(require_internal_api_key)])
def get_audit(
    event_id: str | None = Query(None),
    event_type: str | None = Query(None),
    trace_id: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> list[dict[str, Any]]:
    """Return recent audit trail entries."""
    parsed_event_id = _parse_uuid_param(event_id, "event_id")
    parsed_trace_id = _parse_uuid_param(trace_id, "trace_id")

    with _fsm_lock:
        try:
            entries = _audit.query_entries(
                event_id=parsed_event_id,
                event_type=event_type,
                trace_id=parsed_trace_id,
                raise_on_error=_audit.durable_persistence_configured,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Durable audit history is unavailable",
            ) from exc
        recent = sorted(entries, key=lambda e: e.timestamp_utc, reverse=True)[:limit]
        return [_serialize_audit_entry(e) for e in recent]


# ---------- Escalation packet endpoint ----------


@app.get("/api/escalation", dependencies=[Depends(require_internal_api_key)])
def get_escalation_packet() -> dict[str, Any]:
    """Return the compatibility-only process-memory escalation packet.

    This legacy packet is not accepted by ``/api/review``. Review clients must
    fetch ``/api/escalation/packet-of-record`` and submit its durable row ID and
    canonical SHA-256.
    """
    with _fsm_lock:
        _refresh_fsm_from_db()
        if _fsm.state != SystemState.ESCALATE:
            raise HTTPException(
                status_code=404,
                detail="No active escalation packet: FSM is not in ESCALATE state",
            )
        if _active_escalation_packet is None:
            # The FSM is in ESCALATE (possibly a worker-driven transition
            # surfaced by the read-through), but no packet has been generated in
            # this process yet.
            raise HTTPException(
                status_code=404,
                detail=(
                    "FSM is in ESCALATE but no escalation packet has been "
                    "generated yet; POST /api/escalation/generate to create one"
                ),
            )
        ctx = _fsm.event_context
        if ctx is None or _active_escalation_packet.event_id != ctx.event_id:
            # The active packet is for a different (earlier) event than the one
            # the FSM is currently escalated on - never serve stale evidence.
            raise HTTPException(
                status_code=404,
                detail=(
                    "Active escalation packet is for a different event; POST "
                    "/api/escalation/generate to rebuild it for the current event"
                ),
            )
        return _active_escalation_packet.model_dump(mode="json")


@app.post("/api/escalation/generate", dependencies=[Depends(require_internal_api_key)])
def trigger_escalation_packet_generation() -> dict[str, Any]:
    """Generate a compatibility-only process-memory escalation packet.

    The current worker independently persists the packet of record from the
    committed entering-ESCALATE assessment. This legacy endpoint accepts no
    request body and its output is not accepted by ``/api/review``.
    """
    global _active_escalation_packet

    with _fsm_lock:
        # Surface a worker-driven ESCALATE via the read-through so a packet can
        # be generated for it, not only for an API-process transition. The
        # restored event_context provides ctx; the triggering TransitionRecord
        # is not in this process's in-memory history (recover_from_db does not
        # rebuild it), so the packet is built with transition=None - functional,
        # but without the in-memory trace linkage. The anomaly timeline is still
        # reconstructed from the durable audit trail (query_entries).
        _refresh_fsm_from_db()
        if _fsm.state != SystemState.ESCALATE:
            raise HTTPException(
                status_code=409,
                detail="FSM is not in ESCALATE state",
            )
        ctx = _fsm.event_context
        if ctx is None:
            raise HTTPException(
                status_code=409,
                detail="No active event context",
            )

        # Find the transition that brought us to ESCALATE
        transition: TransitionRecord | None = None
        for t in reversed(_fsm.transition_history):
            # Require the transition to belong to the CURRENT event: after a
            # read-through to a worker-driven event, this process's in-memory
            # history may hold an ESCALATE from a prior event. A mismatch leaves
            # transition=None, so the packet recovers from the durable audit
            # trail instead of attaching a stale trigger/time/trace.
            if t.to_state == SystemState.ESCALATE and t.event_id == ctx.event_id:
                transition = t
                break

        thresholds = _fsm.thresholds
        try:
            _active_escalation_packet = generate_escalation_packet(
                ctx=ctx,
                transition=transition,
                audit_logger=_audit,
                escalation_magnitude=thresholds.escalation_magnitude,
                t3_threshold=thresholds.t3,
            )
        except ValueError as exc:
            # P2 guardrail violation: scan_text() found prohibited alert
            # terminology in the escalation packet text fields.
            raise HTTPException(
                status_code=400,
                detail="Escalation packet failed guardrail validation",
            ) from exc

        return {
            "status": "generated",
            "packet_id": str(_active_escalation_packet.handoff_id),
            "packet_hash": _active_escalation_packet.packet_hash,
        }


@app.get(
    "/api/escalation/packet-of-record",
    dependencies=[Depends(require_internal_api_key)],
)
def get_escalation_packet_of_record() -> dict[str, Any]:
    """Return the durable reviewer packet for the active event.

    The pipeline worker renders this packet from the persisted assessment
    row of the checkpoint that entered ESCALATE and stores it in the
    append-only ``escalation_packets`` table with a canonical content
    hash and renderer version. This endpoint retrieves that immutable row
    from storage, so it survives API restart and never re-assembles
    evidence from in-process state. Selection is deterministic: the
    packet bound to the earliest assessment row for the event, never an
    unspecified latest assessment.
    """
    with _fsm_lock:
        _refresh_fsm_from_db()
        if _db_client is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    "No database configured: durable reviewer packets "
                    "are unavailable in this runtime"
                ),
            )
        ctx = _fsm.event_context
        if ctx is None:
            raise HTTPException(status_code=404, detail="No active event")
        row = _db_client.get_escalation_packet_for_event(ctx.event_id)
    if row is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No durable reviewer packet persisted for the active "
                "event; the pipeline worker writes it at the checkpoint "
                "that enters ESCALATE"
            ),
        )
    created_at = row.get("created_at")
    return {
        "packet_row_id": row.get("id"),
        "assessment_row_id": row.get("assessment_row_id"),
        "event_id": str(row.get("event_id")),
        "renderer_version": row.get("renderer_version"),
        "content_sha256": row.get("content_sha256"),
        "created_at": (
            created_at.isoformat()
            if isinstance(created_at, datetime)
            else created_at
        ),
        "packet": row.get("packet"),
    }


# ---------- Enhanced human review endpoint ----------


@app.post("/api/review", dependencies=[Depends(require_internal_api_key)])
def submit_review(
    req: ReviewRequest,
    x_reviewer_id: str | None = Header(default=None, alias=REVIEWER_ID_HEADER_NAME),
) -> dict[str, Any]:
    """Record a caller-gated assessment review against durable evidence.

    Packet row ID and canonical hash must match the active event's immutable
    packet of record. Assessment identity and scientific hash are derived
    from that packet and enter the decision hash and audit record.

    Current identity remains CALLER_ASSERTED. Therefore every decision,
    including APPROVE, records assessment review only: it does not authorize
    distribution, close the event, or change FSM state. Trusted human
    authentication and separate event disposition remain future work.
    """

    # Auth placeholder: require the caller to assert an identity in the header.
    # A real auth system would validate credentials and extract identity from a
    # JWT or session token.
    reviewer_id = (x_reviewer_id or "").strip()
    if reviewer_id == "":
        raise HTTPException(
            status_code=400,
            detail="X-Reviewer-Id header is required for decision provenance",
        )
    if len(reviewer_id) > REVIEWER_ID_MAX_LENGTH:
        raise HTTPException(
            status_code=400,
            detail=(
                f"X-Reviewer-Id must be at most {REVIEWER_ID_MAX_LENGTH} characters"
            ),
        )
    # Control characters would be written into the append-only audit trail and
    # echoed into logs. A newline is already refused by the HTTP layer; an
    # escape sequence is not, and it renders as control codes in a terminal
    # reading the record back.
    if any(ch < " " or "\x7f" <= ch <= "\x9f" for ch in reviewer_id):
        raise HTTPException(
            status_code=400,
            detail="X-Reviewer-Id must not contain control characters",
        )

    parsed_id = _parse_uuid_param(req.event_id, "event_id")
    if parsed_id is None:
        raise HTTPException(status_code=400, detail="event_id is required")
    parsed_trace_id = _parse_uuid_param(req.trace_id, "trace_id")

    with _fsm_lock:
        # Refresh from the DB first so a worker-driven ESCALATE is reviewable and
        # the resolve acts on the current cross-process state (read-through;
        # no-op without a DB).
        _refresh_fsm_from_db()
        # Guard: all review decisions must target the active escalated event.
        ctx = _fsm.event_context
        if ctx is None or _fsm.state != SystemState.ESCALATE:
            raise HTTPException(
                status_code=409,
                detail="No active escalation: FSM is not in ESCALATE state",
            )
        if parsed_id != ctx.event_id:
            raise HTTPException(
                status_code=409,
                detail="Event ID does not match the active escalation event",
            )

        if _db_client is None:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Durable reviewer storage is required before an "
                    "assessment review can be recorded"
                ),
            )
        packet_row = _db_client.get_escalation_packet_for_event(ctx.event_id)
        if packet_row is None:
            raise HTTPException(
                status_code=409,
                detail="No durable reviewer packet exists for the active event",
            )
        if packet_row.get("id") != req.escalation_packet_row_id:
            raise HTTPException(
                status_code=409,
                detail=(
                    "escalation_packet_row_id does not match the active "
                    "event's packet of record"
                ),
            )
        stored_packet_hash = str(packet_row.get("content_sha256") or "")
        if not compare_digest(stored_packet_hash, req.escalation_packet_hash):
            raise HTTPException(
                status_code=409,
                detail=(
                    "escalation_packet_hash does not match the active "
                    "event's packet of record"
                ),
            )
        packet = packet_row.get("packet")
        if not isinstance(packet, dict) or not compare_digest(
            canonical_packet_hash(packet), stored_packet_hash
        ):
            raise HTTPException(
                status_code=409,
                detail="Stored reviewer packet failed its canonical hash check",
            )
        if str(packet_row.get("event_id")) != str(ctx.event_id) or str(
            packet.get("event_id")
        ) != str(ctx.event_id):
            raise HTTPException(
                status_code=409,
                detail="Reviewer packet does not belong to the active event",
            )
        assessment_row_id = packet_row.get("assessment_row_id")
        if packet.get("assessment_row_id") != assessment_row_id:
            raise HTTPException(
                status_code=409,
                detail="Reviewer packet assessment-row binding is inconsistent",
            )
        assessment = packet.get("assessment")
        if not isinstance(assessment, dict) or str(
            assessment.get("event_id")
        ) != str(ctx.event_id):
            raise HTTPException(
                status_code=409,
                detail="Reviewer packet assessment binding is invalid",
            )
        try:
            assessment_id = UUID(str(assessment.get("handoff_id")))
            parsed_assessment_row_id = int(assessment_row_id)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=409,
                detail="Reviewer packet assessment identity is invalid",
            ) from exc
        assessment_hash = str(
            assessment.get("scientific_content_hash") or ""
        )
        if len(assessment_hash) != 64 or any(
            char not in "0123456789abcdef" for char in assessment_hash
        ):
            raise HTTPException(
                status_code=409,
                detail="Reviewer packet assessment scientific hash is invalid",
            )

        # The reviewer's reason becomes part of the immutable audit record;
        # the USER_MANUAL tells reviewers reserved alert terms are blocked
        # here, so enforce that promise before anything is persisted.
        reason_scan = scan_text(req.decision_reason)
        if reason_scan.violations:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Decision reason contains prohibited alert terminology: "
                    f"{sorted({v.term for v in reason_scan.violations})}"
                ),
            )

        # Single source of truth for review binding and decision hash.
        now = datetime.now(UTC)
        decision_record = AssessmentReviewDecision(
            producer=reviewer_id,
            event_id=parsed_id,
            reviewer_id=reviewer_id,
            identity_assurance=IdentityAssurance.CALLER_ASSERTED,
            decision=ReviewDecision(req.decision),
            decision_reason=req.decision_reason,
            decided_at_utc=now,
            escalation_packet_row_id=req.escalation_packet_row_id,
            escalation_packet_hash=stored_packet_hash,
            assessment_row_id=parsed_assessment_row_id,
            assessment_id=assessment_id,
            assessment_scientific_content_hash=assessment_hash,
        )
        decision_hash = decision_record.decision_hash

        # A successful authority-bearing response requires a confirmed durable
        # append. Process-memory audit alone is not an immutable review record.
        review_entry = AuditEntry(
            event_id=parsed_id,
            trace_id=parsed_trace_id,
            event_type="assessment_review_decision",
            producer=reviewer_id,
            data={
                "decision": req.decision,
                "decision_reason": req.decision_reason,
                "event_id": str(parsed_id),
                "escalation_packet_row_id": req.escalation_packet_row_id,
                "escalation_packet_hash": stored_packet_hash,
                "assessment_row_id": parsed_assessment_row_id,
                "assessment_id": str(assessment_id),
                "assessment_scientific_content_hash": assessment_hash,
                "decision_hash": decision_hash,
                "decided_at_utc": now.isoformat(),
                "identity_assurance": decision_record.identity_assurance.value,
                "distribution_authorized": False,
                "event_disposition_recorded": False,
            },
        )
        if not _audit.append_durable(review_entry):
            raise HTTPException(
                status_code=503,
                detail="Review record could not be committed to durable audit storage",
            )

        return {
            "status": "recorded",
            "decision": req.decision,
            "event_id": str(parsed_id),
            "decision_hash": decision_hash,
            "identity_assurance": decision_record.identity_assurance.value,
            "distribution_authorized": False,
            "event_disposition_recorded": False,
            "fsm_state": _fsm.state.value,
        }


# ---------- E8: Lineage API ----------


@app.get("/api/lineage/{trace_id}", dependencies=[Depends(require_internal_api_key)])
def get_lineage_by_trace(trace_id: str) -> dict[str, Any]:
    """Return the audit lineage for a single pipeline execution.

    Each pipeline run gets a unique trace_id (UUID4). This endpoint
    returns all audit entries produced during that run, ordered chronologically.
    """
    parsed = _parse_uuid_param(trace_id, "trace_id")
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid UUID for trace_id: (empty)")

    with _fsm_lock:
        entries = _audit.query_entries(trace_id=parsed, limit=_LINEAGE_QUERY_LIMIT)

    if not entries:
        raise HTTPException(status_code=404, detail="No entries found for trace_id")

    entries.sort(key=lambda e: e.timestamp_utc)
    return {
        "trace_id": trace_id,
        "entry_count": len(entries),
        "truncated": len(entries) >= _LINEAGE_QUERY_LIMIT,
        "entries": [_serialize_audit_entry(e) for e in entries],
    }


@app.get("/api/lineage/event/{event_id}", dependencies=[Depends(require_internal_api_key)])
def get_lineage_by_event(event_id: str) -> dict[str, Any]:
    """Return audit lineage for all pipeline runs of a seismic event.

    Groups entries by trace_id so the caller can distinguish
    between different pipeline executions for the same event.
    """
    parsed = _parse_uuid_param(event_id, "event_id")
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid UUID for event_id: (empty)")

    with _fsm_lock:
        entries = _audit.query_entries(event_id=parsed, limit=_LINEAGE_QUERY_LIMIT)

    if not entries:
        raise HTTPException(status_code=404, detail="No entries found for event_id")

    # Group by trace_id
    groups: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(entries, key=lambda e: e.timestamp_utc):
        key = str(e.trace_id) if e.trace_id else "__no_trace__"
        groups.setdefault(key, []).append(_serialize_audit_entry(e))

    return {
        "event_id": event_id,
        "total_entries": len(entries),
        "truncated": len(entries) >= _LINEAGE_QUERY_LIMIT,
        "traces": groups,
    }


@app.get(
    "/api/lineage/provenance/{trace_id}",
    dependencies=[Depends(require_internal_api_key)],
)
def get_provenance_by_trace(trace_id: str) -> dict[str, Any]:
    """Return the raw-input provenance chain for one pipeline execution.

    Walks the SQL get_provenance() join: processed_features rows for the
    trace, their companion audit entries (matched on handoff_id), and the
    raw_observations rows referenced by the audit entries' input_hashes.
    The rows are produced by the live worker's lineage persistence
    (qc_report and anomaly_score features). Requires a configured database;
    the in-memory fallback has no processed_features store.
    """
    parsed = _parse_uuid_param(trace_id, "trace_id")
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid UUID for trace_id: (empty)")

    if _db_client is None:
        raise HTTPException(
            status_code=503,
            detail="Provenance lineage requires a configured database",
        )

    rows = _db_client.query_lineage(parsed)
    if rows is None:
        # Query failure (permission, broken function, connection): not the
        # same as an empty result, and must not masquerade as 404.
        raise HTTPException(status_code=503, detail="Lineage query failed")
    if not rows:
        raise HTTPException(status_code=404, detail="No provenance found for trace_id")

    def _json_safe(value: Any) -> Any:
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, list):
            return [_json_safe(v) for v in value]
        if isinstance(value, dict):
            return {k: _json_safe(v) for k, v in value.items()}
        return str(value)

    return {
        "trace_id": trace_id,
        "row_count": len(rows),
        "rows": [_json_safe(dict(r)) for r in rows],
    }


# ---------- Policy check endpoint ----------


@app.post("/api/policy/check", dependencies=[Depends(require_internal_api_key)])
def policy_check(req: PolicyCheckRequest) -> dict[str, Any]:
    """Validate an agent action against the permission matrix.

    Returns a structured policy result. When the action is denied the
    response body contains a ``policy_violation`` object, not a bare
    HTTP 403, so callers can inspect the reason programmatically.
    """
    try:
        cap = AgentCapability(req.capability)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown capability code: {req.capability}",
        )

    with _fsm_lock:
        fsm_state = _fsm.state.value

        result = check_policy(
            agent_name=req.agent_name,
            capability=cap,
            fsm_state=fsm_state,
            human_decision_present=req.human_decision_present,
            matrix=_permission_matrix,
        )

        ctx = _fsm.event_context
        event_id = ctx.event_id if ctx else None

        # Record ALL permission checks for activity analysis
        log_policy_result(
            result,
            _audit,
            agent_name=req.agent_name,
            capability=req.capability,
            fsm_state=fsm_state,
            event_id=event_id,
        )

        if not result.allowed:
            # Invariant: PolicyCheckResult.__post_init__ guarantees denial
            # is set when allowed is False.
            if result.denial is None:
                raise RuntimeError("PolicyCheckResult.allowed=False but denial is None")
            log_denial(result.denial, _audit, event_id=event_id)
            return denial_to_response(result.denial)

        return {"status": "allowed"}


# ---------- After-action analysis endpoint ----------


class AfterActionRequest(BaseModel):
    """Payload for triggering after-action analysis."""

    event_id: str = Field(description="Event UUID to analyze")


def _open_investigator_db() -> Any | None:
    """Return a client connected as investigator_writer, or None.

    Opened on first use and reused, because the investigator runs rarely and a
    second pool held open for the life of the process would cost connections
    for nothing. None means the role cannot connect, which the caller turns
    into a 503 rather than falling back to a role that should not be writing
    these rows.
    """
    global _investigator_db
    if _investigator_db is not None:
        return _investigator_db
    if not os.getenv("DB_HOST", "").strip():
        return None
    # Sync endpoints run in Starlette's threadpool, so two callers can arrive
    # here at once. Without the lock both would build a pool and one would be
    # dropped on the floor still holding its connections.
    with _investigator_db_lock:
        if _investigator_db is not None:
            return _investigator_db
        try:
            from hazard_assessment.storage.client import ClientConfig, DatabaseClient

            candidate = DatabaseClient(ClientConfig.from_env(role="investigator_writer"))
            if not candidate.is_connected:
                candidate.close()
                return None
            _investigator_db = candidate
        except Exception:
            logger.exception("Could not open a connection as investigator_writer")
            return None
    return _investigator_db


@app.post("/api/investigate", dependencies=[Depends(require_internal_api_key)])
def investigate_active_event() -> dict[str, Any]:
    """Investigate the evidence behind the currently active event.

    This is the counterpart of /api/after-action, and its gates are the
    mirror image: after-action refuses the active event, this one requires it.
    The model chooses which read-only audit queries to run for each issue and
    reports what the records support.

    What it cannot do, by construction rather than by instruction:
    - Findings are written to evidence_issue_results, which migration 009
      grants to investigator_writer alone. pipeline_worker, which drives the
      FSM, holds neither INSERT nor SELECT on it, so a finding can be neither
      written nor read by the code that escalates.
    - It runs only once the worker has persisted a checkpoint, and binds its
      findings to that row while reading the event's audit trail for
      evidence. Either way it sits off the detection path and cannot delay or
      block ingest, scoring, or a transition.
    - Every finding is guardrail-scanned before persistence, and one that
      reaches for reserved alert wording is dropped rather than stored.

    Contract enforced here:
    - 409 when no event is active, since there is nothing to investigate.
    - 503 when durable storage is not configured at all, since the normal path
      has to be able to record what advice was given and on what evidence.
      A single insert that then fails is reported per finding as
      persistence="error" and the finding is still returned: losing it would
      be worse than returning it with the record plainly marked missing.
    - 404 when the active event has no persisted assessment checkpoint yet,
      which is the normal state early in an event.
    - Re-invocation for the same checkpoint, issue, model and prompt version
      collides on a deterministic invocation id and is reported as "existing"
      rather than stored twice.
    """
    try:
        # investigator imports langchain only inside its functions, so importing
        # it proves nothing about whether the provider stack is installed. The
        # factory does import langchain_core at module level, so naming it here
        # is what makes this 501 real rather than decorative: without it a
        # missing dependency surfaced as three separately failed issues and an
        # HTTP 200.
        from hazard_assessment.agents.llm_advisory.factory import (  # noqa: F401
            build_chat_model,
        )
        from hazard_assessment.agents.llm_advisory.investigator import (
            ISSUE_NAMES,
            investigate_assessment,
        )
        from hazard_assessment.config.settings import LLMSettings
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="LLM dependencies not installed",
        ) from exc

    settings = LLMSettings()
    if not settings.is_enabled:
        raise HTTPException(
            status_code=501,
            detail=(
                "LLM layer not configured. Set LLM_API_KEY for a hosted "
                "provider, or LLM_BASE_URL for an endpoint you run yourself."
            ),
        )

    with _fsm_lock:
        _refresh_fsm_from_db()
        active_ctx = _fsm.event_context
        if active_ctx is None:
            raise HTTPException(
                status_code=409,
                detail="No event is currently active",
            )
        event_id = active_ctx.event_id
        if _db_client is None or not _audit.durable_persistence_configured:
            raise HTTPException(
                status_code=503,
                detail="Investigation requires durable storage for its findings",
            )
        # Snapshot under the lock: the tool loop is long-running and
        # AuditLogger is not thread-safe (see audit/logger.py).
        try:
            audit_snapshot = _audit.snapshot(event_id=event_id, raise_on_error=True)
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Durable audit history is unavailable",
            ) from exc

    # Findings are written through the investigator's own database identity.
    # The API connects as orchestrator_writer, which migration 009 gives no
    # INSERT on evidence_issue_results; only investigator_writer has it. That
    # separation is the mechanism keeping a finding out of the decision path,
    # so the code has to honor it rather than widen the API's grant.
    investigator_db = _open_investigator_db()
    if investigator_db is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Cannot reach the database as investigator_writer. Set "
                "DB_INVESTIGATOR_WRITER_PASSWORD (or DB_PASSWORD) for that role."
            ),
        )

    assessment = _db_client.get_latest_assessment_for_event(event_id)
    if assessment is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "The active event has no persisted assessment checkpoint yet; "
                "there is no evidence row to investigate"
            ),
        )
    assessment_row_id = int(assessment["id"])

    findings = investigate_assessment(
        settings,
        audit_snapshot,
        event_id=event_id,
        assessment_row_id=assessment_row_id,
    )

    from hazard_assessment.telemetry.metrics import record_guardrail_scan

    recorded: list[dict[str, Any]] = []
    for finding in findings:
        record_guardrail_scan(passed=not finding.guardrail_violations)
        status = investigator_db.insert_evidence_issue_result(
            invocation_id=finding.invocation_id,
            assessment_row_id=assessment_row_id,
            issue_name=finding.issue_name,
            result=finding.to_result_payload(),
            result_sha256=finding.result_sha256,
        )
        recorded.append({
            "issue_name": finding.issue_name,
            "invocation_id": finding.invocation_id,
            "persistence": status,
            "tool_calls": finding.tool_calls,
            "guardrail_violations": finding.guardrail_violations,
            "finding": finding.finding,
        })

    # Naming the issues that produced nothing keeps a partial run legible; the
    # investigator keeps going when one issue fails.
    missing = sorted(set(ISSUE_NAMES) - {r["issue_name"] for r in recorded})
    # Each insert is its own transaction, so a run can leave one issue on record
    # and another not. Reporting the counts stops a caller reading HTTP 200 and
    # a populated issues_recorded as "everything was saved".
    not_stored = sorted(
        str(r["issue_name"]) for r in recorded if r["persistence"] not in ("inserted", "existing")
    )

    # An investigation is a model acting on a live event, so the fact that it
    # ran belongs in the audit trail whether or not the findings landed. Without
    # this a failed set of inserts left no record anywhere that anyone asked.
    _audit.append(
        AuditEntry(
            event_id=event_id,
            event_type="evidence_investigation",
            producer="evidence_investigator",
            data={
                "assessment_row_id": assessment_row_id,
                "issues_recorded": [str(r["issue_name"]) for r in recorded],
                "issues_failed": missing,
                "issues_not_stored": not_stored,
                "guardrail_withheld": sorted(
                    str(r["issue_name"]) for r in recorded if r["guardrail_violations"]
                ),
            },
        )
    )

    return {
        "event_id": str(event_id),
        "assessment_row_id": assessment_row_id,
        "issues_recorded": recorded,
        "issues_failed": missing,
        "issues_not_stored": not_stored,
        # Everything on record for this checkpoint, including findings from an
        # earlier run. Without this the table is write-only from the API's side,
        # and a "conflict" or "existing" result would name a row the caller
        # could not see.
        "on_record": investigator_db.get_evidence_issue_results(assessment_row_id),
        "non_authoritative": True,
    }


@app.post("/api/after-action", dependencies=[Depends(require_internal_api_key)])
def run_after_action(req: AfterActionRequest) -> dict[str, Any]:
    """Run LLM-powered analysis for a nonactive event with audit history.

    Uses a 3-node LangGraph graph with tool use (timeline -> gaps -> draft).
    The LLM decides which audit queries to run via bound tools. This is the
    counterpart of /api/investigate, which handles the active event.
    The current system has no trusted event-disposition record, so this gate
    proves only that the requested event is not active in the current FSM and
    has audit history. It does not prove event closure.

    Contract enforced here:
    - The currently active event is rejected with 409, and an event with no
      audit records at all with 404.
    - Every tool call the LLM makes is recorded and returned, including
      unknown-tool requests, tool errors, and loop non-convergence.
    - Tool results carry explicit truncation flags (see tools.py).
    - The result (post-guardrail text plus the tool-call log) is
      persisted to the audit trail as an after_action_report entry,
      durable in audit_events when a database is configured.

    This is a sync endpoint; FastAPI runs it in a thread pool automatically.

    Returns the after-action report or an error if LLM is not configured.
    """
    parsed_event_id = _parse_uuid_param(req.event_id, "event_id")
    if parsed_event_id is None:
        raise HTTPException(status_code=400, detail="event_id is required")

    # Lazy import to avoid loading LLM dependencies when not configured
    try:
        from hazard_assessment.agents.llm_advisory.after_action import (
            build_after_action_graph,
        )
        from hazard_assessment.config.settings import LLMSettings
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="LLM dependencies not installed",
        ) from exc

    settings = LLMSettings()
    if not settings.is_enabled:
        raise HTTPException(
            status_code=501,
            detail=(
                "LLM layer not configured. Set LLM_API_KEY for a hosted "
                "provider, or LLM_BASE_URL for an endpoint you run yourself."
            ),
        )
    # Nonactive-event gate: refuse the current event and events the audit
    # trail has never seen. This does not establish trusted event closure.
    with _fsm_lock:
        _refresh_fsm_from_db()
        active_ctx = _fsm.event_context
        if active_ctx is not None and active_ctx.event_id == parsed_event_id:
            raise HTTPException(
                status_code=409,
                detail="Event is still active and cannot be analyzed after action",
            )
        if not _audit.durable_persistence_configured:
            raise HTTPException(
                status_code=503,
                detail="After-action analysis requires durable audit storage",
            )
        # Snapshot audit entries under the lock so the long-running LLM
        # graph operates on an immutable copy (AuditLogger is not
        # thread-safe - see audit/logger.py docstring).
        try:
            audit_snapshot = _audit.snapshot(
                event_id=parsed_event_id,
                raise_on_error=True,
            )
        except Exception as exc:
            raise HTTPException(
                status_code=503,
                detail="Durable audit history is unavailable",
            ) from exc
    if audit_snapshot.count == 0:
        raise HTTPException(
            status_code=404,
            detail="No audit records exist for this event",
        )
    tool_call_log: list[dict[str, Any]] = []
    try:
        graph = build_after_action_graph(
            settings,
            audit_snapshot,
            pinned_event_id=parsed_event_id,
            tool_call_log=tool_call_log,
        )
        result = graph.invoke(
            {"event_id": str(parsed_event_id)},
            config={"configurable": {"thread_id": str(parsed_event_id)}},
        )
    except Exception as exc:
        logger.exception("After-action analysis failed for event %s", req.event_id)
        raise HTTPException(
            status_code=500,
            detail="After-action analysis failed",
        ) from exc

    # Guardrail scan: after-action output is raw LLM text and must pass
    # the same guardrail scanner as synthesis narratives.  Uses
    # .violations check (no disclaimer required for internal reports).
    from hazard_assessment.telemetry.metrics import record_guardrail_scan

    text_fields = {
        "timeline": result.get("timeline"),
        "gaps": result.get("gaps"),
        "draft_report": result.get("draft_report"),
    }
    for field_name, text in text_fields.items():
        if text:
            field_scan = scan_text(str(text))
            record_guardrail_scan(passed=not field_scan.violations)
            if field_scan.violations:
                terms = [v.term for v in field_scan.violations]
                logger.warning(
                    "After-action %s field failed guardrail scan: %s",
                    field_name, terms,
                )
                text_fields[field_name] = (
                    f"(redacted: contained {len(terms)} prohibited alert term(s))"
                )

    # Persist the post-guardrail text and complete tool-call log. Success is
    # returned only after the durable audit append is confirmed.
    report_correlation_id = uuid4()
    report_entry = AuditEntry(
        event_id=parsed_event_id,
        event_type="after_action_report",
        producer="after_action_graph",
        data={
            "timeline": text_fields["timeline"],
            "gaps": text_fields["gaps"],
            "draft_report": text_fields["draft_report"],
            "tool_calls": tool_call_log,
            "n_tool_calls": len(tool_call_log),
            "report_correlation_id": str(report_correlation_id),
        },
    )
    with _fsm_lock:
        persisted = _audit.append_durable(report_entry)
    if not persisted:
        raise HTTPException(
            status_code=503,
            detail="After-action report could not be committed to durable audit storage",
        )

    return {
        "event_id": req.event_id,
        **text_fields,
        "tool_calls": tool_call_log,
        "report_correlation_id": str(report_correlation_id),
    }


# ---------- Activity report endpoint ----------


@app.get("/api/activity-report/{event_id}", dependencies=[Depends(require_internal_api_key)])
def get_activity_report(event_id: str) -> dict[str, Any]:
    """Return structured activity report for paper analysis.

    Groups audit entries by type: FSM transitions, permission checks, LLM
    calls and guardrail scans. ``entries_by_type`` carries every type present,
    so it is not limited to the ones the summary counts.
    """
    parsed = _parse_uuid_param(event_id, "event_id")
    if parsed is None:
        raise HTTPException(status_code=400, detail="Invalid event_id")

    with _fsm_lock:
        entries = _audit.query_entries(event_id=parsed, limit=_LINEAGE_QUERY_LIMIT)

    if not entries:
        raise HTTPException(status_code=404, detail="No entries found for event_id")

    # Group by event_type
    grouped: dict[str, list[dict[str, Any]]] = {}
    for e in sorted(entries, key=lambda e: e.timestamp_utc):
        grouped.setdefault(e.event_type, []).append(_serialize_audit_entry(e))

    # Compute summary statistics
    llm_calls = grouped.get("llm_call", [])
    permission_checks = grouped.get("permission_check", [])
    guardrail_scans = grouped.get("guardrail_scan", [])

    return {
        "event_id": event_id,
        "total_entries": len(entries),
        "truncated": len(entries) >= _LINEAGE_QUERY_LIMIT,
        "summary": {
            "fsm_transitions": len(grouped.get("state_transition", [])),
            "llm_calls": len(llm_calls),
            "llm_total_latency_ms": sum(
                c.get("data", {}).get("latency_ms", 0) for c in llm_calls
            ),
            "guardrail_scans": len(guardrail_scans),
            "guardrail_violations": sum(
                1 for s in guardrail_scans if not s.get("data", {}).get("passed", True)
            ),
            "permission_checks_total": len(permission_checks),
            "permission_checks_allowed": sum(
                1 for p in permission_checks if p.get("data", {}).get("allowed", False)
            ),
            "permission_checks_denied": sum(
                1 for p in permission_checks if not p.get("data", {}).get("allowed", True)
            ),
            # Two counters used to sit here, station_coverage_reports and
            # tool_invocations, each counting an audit event type nothing
            # writes: coverage surfaces as the sensor_degraded flag on
            # /api/fsm, and tool calls are recorded inside the entry of the
            # path that made them, the after-action report or the findings
            # payload, rather than as separate rows. Both could only ever read
            # zero, and a field with one possible value is not information.
        },
        "entries_by_type": grouped,
    }


# ---------- Prometheus metrics endpoint ----------


@app.get("/metrics")
def prometheus_metrics() -> Any:
    """Expose Prometheus metrics for scraping."""
    try:
        from hazard_assessment.telemetry.metrics import generate_metrics_response

        with _fsm_lock:
            _refresh_fsm_from_db()
            fsm_state = _fsm.state.value
        return generate_metrics_response(fsm_state)
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="prometheus-client not installed",
        )
