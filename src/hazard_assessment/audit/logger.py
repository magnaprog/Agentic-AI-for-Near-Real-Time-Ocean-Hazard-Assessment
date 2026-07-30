"""Append-only audit logger for the hazard assessment system.

All state transitions, agent handoffs, and human decisions are recorded
in an append-only audit trail. Records are never modified or deleted.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4, uuid5

from pydantic import BaseModel, Field

from hazard_assessment.schemas.envelope import AwareDatetime

if TYPE_CHECKING:
    from hazard_assessment.orchestrator.states import TransitionRecord
    from hazard_assessment.storage.client import DatabaseClient

_logger = logging.getLogger(__name__)


class AuditEntry(BaseModel):
    """A single audit trail entry.

    Immutable once persisted. The entry_id and timestamp are
    generated at creation time and cannot be modified.
    """

    entry_id: UUID = Field(default_factory=uuid4)
    timestamp_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC)
    )
    event_id: UUID | None = Field(
        default=None,
        description="Event ID linking entries for the same seismic event",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Trace ID correlating entries from a single pipeline execution",
    )
    event_type: str = Field(description="Type of event (e.g., 'state_transition', 'qc_complete')")
    producer: str = Field(description="Name of the agent or component that produced this entry")
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Structured payload for this audit entry",
    )

    model_config = {"extra": "forbid", "frozen": True}


# Fixed namespace for deriving a stable AuditEntry.entry_id from a DB row id
# (the audit_events table has no entry_id column of its own).
_AUDIT_ROW_NAMESPACE = UUID("a7c3e2b1-0000-4000-8000-000000000001")


def _row_to_audit_entry(row: dict[str, Any]) -> AuditEntry:
    """Map a raw ``audit_events`` DB row to an :class:`AuditEntry`.

    The DB schema shreds an entry across columns (``agent_name`` -> producer,
    ``action`` -> event_type, ``recorded_at`` -> timestamp, and
    ``state_before`` / ``state_after`` / ``decision_basis`` / ``metadata`` ->
    ``data``). Reassemble them so the API serializer treats DB-sourced entries
    identically to in-memory ones. ``entry_id`` is derived deterministically
    from the row id.
    """

    def _as_uuid(value: Any) -> UUID | None:
        if value is None:
            return None
        return value if isinstance(value, UUID) else UUID(str(value))

    ts = row.get("recorded_at")
    if not isinstance(ts, datetime):
        ts = datetime.fromisoformat(str(ts))
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=UTC)
    else:
        ts = ts.astimezone(UTC)

    meta = row.get("metadata")
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except ValueError:
            meta = {}
    data: dict[str, Any] = dict(meta) if isinstance(meta, dict) else {}
    # When the metadata JSONB is present it is a faithful ``json.dumps(entry.data)``
    # round-trip (see storage/client.py append_audit), so it already restores the
    # original payload exactly. Only reconstruct ``data`` from the shredded
    # columns when metadata is absent (e.g. a row written without it). Overlaying
    # the columns otherwise injects keys the in-memory entry never had - notably
    # ``decision_basis``, whose column stores an event_type fallback for every
    # non-transition entry - which would make DB-sourced entries diverge from
    # their in-memory equivalents.
    if not data:
        for col, key in (
            ("state_before", "from_state"),
            ("state_after", "to_state"),
            ("decision_basis", "decision_basis"),
            ("reasoning_trace", "reasoning_trace"),
            ("model_version", "model_version"),
            ("handoff_id", "handoff_id"),
        ):
            value = row.get(col)
            if value is not None:
                data[key] = str(value) if isinstance(value, UUID) else value

    row_id = row.get("id")
    entry_id = (
        uuid5(_AUDIT_ROW_NAMESPACE, f"audit:{row_id}")
        if row_id is not None
        else uuid4()
    )
    return AuditEntry(
        entry_id=entry_id,
        timestamp_utc=ts,
        event_id=_as_uuid(row.get("event_id")),
        trace_id=_as_uuid(row.get("trace_id")),
        event_type=str(row.get("action") or ""),
        producer=str(row.get("agent_name") or ""),
        data=data,
    )


class AuditLogger:
    """Append-only audit logger with optional TimescaleDB persistence.

    When ``db_client`` is provided, entries are written to the
    ``audit_events`` table in TimescaleDB in addition to the in-memory
    buffer. When ``db_client`` is None, falls back to in-memory only
    (keeps tests working without a database).

    The ``max_entries`` cap prevents unbounded memory growth in
    long-running processes. When the cap is reached, the oldest entries
    are evicted (FIFO). With DB persistence enabled, the full history
    is always available via database queries.

    **Thread safety:** This class is NOT thread-safe internally.
    Callers must provide external synchronization when the logger
    is accessed from multiple threads. The FastAPI app uses
    ``_fsm_lock`` for this purpose.
    """

    MAX_ENTRIES = 10_000

    def __init__(
        self,
        max_entries: int = MAX_ENTRIES,
        db_client: DatabaseClient | None = None,
    ) -> None:
        self._buffer: list[AuditEntry] = []
        self._max_entries = max_entries
        self._db_client = db_client

    def append(self, entry: AuditEntry) -> None:
        """Append an entry to the audit trail. Entries are never modified.

        Stores a deep copy so callers cannot mutate persisted entries
        by retaining a reference to the original object.

        When the buffer exceeds ``max_entries``, the oldest entries
        are evicted to prevent unbounded memory growth.

        If a db_client is configured, also persists to TimescaleDB.
        """
        self._buffer.append(entry.model_copy(deep=True))
        if len(self._buffer) > self._max_entries:
            self._buffer = self._buffer[-self._max_entries :]

        if self._db_client is not None:
            try:
                if not self._db_client.append_audit(entry):
                    _logger.error("Failed to persist audit entry to database")
            except Exception:
                _logger.exception("Failed to persist audit entry to database")

    @property
    def durable_persistence_configured(self) -> bool:
        """Whether this logger has a database client for durable writes."""
        return self._db_client is not None

    def append_durable(self, entry: AuditEntry) -> bool:
        """Persist an entry before adding it to process memory.

        Authority-bearing API endpoints use this path so a successful response
        means the database confirmed the append. Ordinary pipeline audit calls
        retain best-effort ``append()`` semantics.
        """
        if self._db_client is None:
            return False
        try:
            if not self._db_client.append_audit(entry):
                return False
        except Exception:
            _logger.exception("Failed to persist required audit entry")
            return False

        self._buffer.append(entry.model_copy(deep=True))
        if len(self._buffer) > self._max_entries:
            self._buffer = self._buffer[-self._max_entries :]
        return True

    def write_transition(self, record: TransitionRecord) -> None:
        """Write an FSM transition record to the audit trail.

        Satisfies the AuditWriter protocol defined in
        orchestrator.states, converting TransitionRecord fields
        into an AuditEntry.
        """
        entry = AuditEntry(
            event_id=record.event_id,
            trace_id=record.trace_id,
            event_type="state_transition",
            producer="orchestrator",
            data={
                "transition_id": str(record.transition_id),
                "from_state": str(record.from_state),
                "to_state": str(record.to_state),
                "trigger_reason": record.trigger_reason,
                "anomaly_score": record.anomaly_score,
                "seismic_magnitude": record.seismic_magnitude,
                "thresholds": {
                    "basin": record.thresholds_used.basin,
                    "t1": record.thresholds_used.t1,
                    "t2": record.thresholds_used.t2,
                    "t3": record.thresholds_used.t3,
                },
            },
        )
        self.append(entry)

    @property
    def count(self) -> int:
        """Number of entries in the audit trail."""
        return len(self._buffer)

    def get_entries(
        self,
        event_id: UUID | None = None,
        event_type: str | None = None,
        trace_id: UUID | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> list[AuditEntry]:
        """Retrieve audit entries with optional filtering.

        Returns deep copies to prevent external mutation of the
        internal buffer. Uses in-memory buffer (DB queries available
        via ``query_db`` for paginated access).
        """
        results = self._buffer
        if event_id is not None:
            results = [e for e in results if e.event_id == event_id]
        if event_type is not None:
            results = [e for e in results if e.event_type == event_type]
        if trace_id is not None:
            results = [e for e in results if e.trace_id == trace_id]
        results = results[offset:offset + limit]
        return [entry.model_copy(deep=True) for entry in results]

    def query_db(
        self,
        event_id: UUID | None = None,
        event_type: str | None = None,
        trace_id: UUID | None = None,
        producer: str | None = None,
        limit: int = 100,
        offset: int = 0,
        raise_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Query audit entries from the database with pagination.

        Returns raw dicts from the database. Falls back to empty list
        if no db_client is configured.
        """
        if self._db_client is None:
            return []
        return self._db_client.query_audit(
            event_id=event_id,
            event_type=event_type,
            trace_id=trace_id,
            producer=producer,
            limit=limit,
            offset=offset,
            raise_on_error=raise_on_error,
        )

    def query_entries(
        self,
        event_id: UUID | None = None,
        event_type: str | None = None,
        trace_id: UUID | None = None,
        limit: int = 200,
        offset: int = 0,
        raise_on_error: bool = False,
    ) -> list[AuditEntry]:
        """Return audit entries as :class:`AuditEntry` objects.

        When a database is configured, read the durable ``audit_events`` table
        so entries written by other processes (e.g. the pipeline worker) are
        visible; otherwise fall back to the in-memory buffer. Used by the API
        audit/lineage endpoints.
        """
        if self._db_client is None:
            # Match the DB branch's "most recent `limit`" window: query_audit
            # uses ORDER BY recorded_at DESC LIMIT/OFFSET, whereas get_entries
            # returns the oldest-first front slice. Without this, a trace with
            # more than `limit` entries returns the START of history in-memory
            # but the END with a DB configured, and /api/audit ("recent") would
            # miss the newest entries on the in-memory path. Fetch the full
            # filtered set, then take the newest `limit` after skipping `offset`
            # from the newest end, returned newest-first so the in-memory and DB
            # branches mirror ORDER BY recorded_at DESC exactly - same set AND
            # order - giving query_entries a deployment-independent contract.
            # (Every current caller re-sorts, so the order also does not leak.)
            matching = self.get_entries(
                event_id=event_id,
                event_type=event_type,
                trace_id=trace_id,
                limit=len(self._buffer) + 1,
                offset=0,
            )
            end = len(matching) - offset
            if end <= 0:
                return []
            return matching[max(0, end - limit):end][::-1]
        rows = self.query_db(
            event_id=event_id,
            event_type=event_type,
            trace_id=trace_id,
            limit=limit,
            offset=offset,
            raise_on_error=raise_on_error,
        )
        return [_row_to_audit_entry(row) for row in rows]

    # ---- Agentic activity recording methods ----

    def log_llm_call(
        self,
        event_id: UUID | None,
        agent: str,
        model: str,
        prompt_tokens: int | None,
        response_tokens: int | None,
        latency_ms: float,
        success: bool,
        trace_id: UUID | None = None,
    ) -> None:
        """Record an LLM synthesis or after-action invocation."""
        self.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="llm_call",
            producer=agent,
            data={
                # audit_events carries an llm_invoked column that the storage
                # client fills from this key. Nothing set it, so the column read
                # FALSE on every row including the model calls themselves, which
                # made a column named for the question unable to answer it.
                "llm_invoked": True,
                "model": model,
                "prompt_tokens": prompt_tokens,
                "response_tokens": response_tokens,
                "latency_ms": latency_ms,
                "success": success,
            },
        ))

    def record_recovery_failure(self, durable_state: str) -> None:
        """Append an audit entry for an FSM recovery failure.

        A recovery failure destroys the in-memory event context and the
        sticky recovery_failed flag is process-local, so without an audit
        record a worker-only failure could be invisible to the API/dashboard
        (the worker overwrites the corrupt row on its next transition).
        Persisted best-effort via the normal append path.
        """
        self.append(AuditEntry(
            event_type="fsm_recovery_failed",
            producer="fsm_orchestrator",
            data={"durable_state": durable_state},
        ))

    def log_guardrail_scan(
        self,
        event_id: UUID | None,
        agent: str,
        text_length: int,
        violations: list[str],
        passed: bool,
        trace_id: UUID | None = None,
    ) -> None:
        """Record guardrail scanner execution and results."""
        self.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="guardrail_scan",
            producer=agent,
            data={
                "text_length": text_length,
                "violation_count": len(violations),
                "violations": violations,
                "passed": passed,
            },
        ))

    def log_permission_check(
        self,
        event_id: UUID | None,
        agent: str,
        capability: str,
        allowed: bool,
        reason: str | None = None,
        trace_id: UUID | None = None,
    ) -> None:
        """Record ALL permission checks (not just denials)."""
        self.append(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="permission_check",
            producer=agent,
            data={
                "capability": capability,
                "allowed": allowed,
                "reason": reason or "",
            },
        ))

    def snapshot(
        self,
        event_id: UUID | None = None,
        *,
        raise_on_error: bool = False,
    ) -> AuditLogger:
        """Return a shallow copy of this logger scoped to *event_id*.

        The returned AuditLogger holds deep-copied entries and is safe
        to read from a different thread without the caller's lock.
        Used by the after-action endpoint to avoid holding ``_fsm_lock``
        during slow LLM operations.

        Reads through query_entries so that, with a database configured,
        durable entries written by other processes (e.g. the pipeline
        worker) are included; without one this falls back to the in-memory
        buffer. Reversed back to oldest-first to preserve the buffer's
        chronological order for the after-action tools.
        """
        snap = AuditLogger(max_entries=self._max_entries)
        snap._buffer = list(reversed(self.query_entries(
            event_id=event_id,
            limit=self._max_entries,
            raise_on_error=raise_on_error,
        )))
        return snap
