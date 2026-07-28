"""Database client with connection pooling for TimescaleDB.

Provides typed methods for all application-level database operations:
raw observation inserts, audit trail writes, FSM state persistence,
and lineage queries.

Falls back gracefully when no database is available (returns None / empty
lists) so that tests and development environments can run without a DB.
"""

from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterator

    from hazard_assessment.ingest.validation import QuarantinedRecord
from uuid import UUID

import psycopg
from psycopg.conninfo import make_conninfo
from psycopg.rows import dict_row

from hazard_assessment.audit.logger import AuditEntry

logger = logging.getLogger(__name__)

# Advisory lock key for serializing FSM mutations.
# Different from the migration lock (provision.py) to avoid contention.
FSM_LOCK_KEY = 1742910700

# feature_type value for ocean evidence assessment rows in
# processed_features (migration 009).
OCEAN_EVIDENCE_FEATURE_TYPE = "ocean_evidence_assessment"


@dataclass(frozen=True)
class AssessmentPersistResult:
    """Outcome of an idempotent assessment insert.

    status:
        "inserted"  - this call created the row.
        "existing"  - a row with the same checkpoint key already existed and
                      its event identity, input-manifest hash, and scientific
                      hash all match; ``row`` is that row.
        "conflict"  - a row exists under the same checkpoint key but at least
                      one of event identity, input-manifest hash, or
                      scientific hash differs. Hard failure: the caller must
                      not treat either assessment as authoritative for this
                      checkpoint without disclosure.
        "error"     - the insert or read-back failed (no pool, SQL error).
    """

    status: str
    row: dict[str, Any] | None = None
    detail: str = ""


@dataclass(frozen=True)
class ClientConfig:
    """Connection parameters for a specific database role."""

    host: str
    port: int
    dbname: str
    user: str
    password: str
    connect_timeout: int = 10
    pool_min_size: int = 2
    pool_max_size: int = 10

    @classmethod
    def from_env(
        cls,
        role: str = "orchestrator_writer",
    ) -> ClientConfig:
        """Build config from environment, using role-specific password."""
        role_env = role.upper()
        default_pw = os.getenv("DB_DEFAULT_ROLE_PASSWORD", os.getenv("DB_PASSWORD", ""))
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            dbname=os.getenv("DB_NAME", "hazard_assessment"),
            user=role,
            password=os.getenv(f"DB_{role_env}_PASSWORD") or default_pw,
            connect_timeout=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
        )

    @property
    def conninfo(self) -> str:
        # make_conninfo quotes and escapes each value. Interpolating by hand
        # breaks on any password containing a space, quote, or backslash, and
        # lets a crafted value inject additional libpq keywords.
        return make_conninfo(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            connect_timeout=self.connect_timeout,
        )


class DatabaseClient:
    """Synchronous database client with connection pooling.

    Uses psycopg3 ConnectionPool for thread-safe connection reuse.
    FastAPI runs sync endpoints in a threadpool, so sync connections
    are appropriate.

    When ``config`` is None, the client operates in no-op mode -
    all writes are silently discarded and all reads return empty
    results. This keeps tests working without a database.
    """

    def __init__(self, config: ClientConfig | None = None) -> None:
        self._config = config
        self._pool: Any = None
        if config is not None:
            try:
                from psycopg_pool import ConnectionPool

                self._pool = ConnectionPool(
                    conninfo=config.conninfo,
                    min_size=config.pool_min_size,
                    max_size=config.pool_max_size,
                    max_lifetime=300.0,
                    max_idle=60.0,
                    timeout=5.0,
                    kwargs={"row_factory": dict_row},
                )
                logger.info(
                    "Database pool created: %s@%s:%d/%s (pool %d-%d)",
                    config.user,
                    config.host,
                    config.port,
                    config.dbname,
                    config.pool_min_size,
                    config.pool_max_size,
                )
            except Exception:
                logger.warning(
                    "Failed to create database pool - running in no-op mode",
                    exc_info=True,
                )
                self._pool = None

    @property
    def is_connected(self) -> bool:
        return self._pool is not None

    @contextmanager
    def _conn(self) -> Iterator[psycopg.Connection[dict[str, Any]]]:
        """Yield a connection from the pool, or raise if no pool."""
        if self._pool is None:
            raise RuntimeError("No database connection available")
        with self._pool.connection() as conn:
            yield conn

    def close(self) -> None:
        """Close the connection pool."""
        if self._pool is not None:
            self._pool.close()
            self._pool = None

    # ---- Raw observations ----

    def insert_observations(
        self,
        records: list[dict[str, Any]],
        source_type: str,
    ) -> int:
        """Insert raw observations one-by-one. Returns count of new rows inserted.

        Uses ON CONFLICT DO NOTHING for idempotent inserts (dedup on
        station_id, observed_at, payload_hash).  Per-row execution is
        intentional: we need individual rowcount to track inserts vs
        conflict-skipped duplicates.
        """
        if not self._pool or not records:
            return 0

        inserted = 0
        try:
            with self._conn() as conn:
                for rec in records:
                    result = conn.execute(
                        """
                        INSERT INTO raw_observations
                            (station_id, source_type, observed_at, raw_payload, payload_hash)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT DO NOTHING
                        """,
                        (
                            rec["station_id"],
                            source_type,
                            rec["observed_at"],
                            json.dumps(rec.get("payload", {})),
                            rec["payload_hash"],
                        ),
                    )
                    if result.rowcount and result.rowcount > 0:
                        inserted += 1
                conn.commit()
        except Exception:
            # Roll back so a mid-batch failure never reports uncommitted rows
            # as inserted. The pool also resets the connection on return.
            logger.exception("Failed to insert observations")
            return 0
        return inserted

    def insert_processed_feature(
        self,
        *,
        feature_type: str,
        producer_agent: str,
        handoff_id: UUID | str,
        trace_id: UUID | str,
        payload: dict[str, Any],
        source_refs: list[dict[str, Any]] | None = None,
        event_id: UUID | str | None = None,
        station_id: str | None = None,
        code_version: str | None = None,
    ) -> bool:
        """Persist one agent output envelope into processed_features.

        This is the lineage producer for get_provenance(): the row's
        handoff_id and trace_id must match a companion audit entry whose
        input_hashes carry the raw payload hashes, so the SQL join can walk
        feature -> audit entry -> raw_observations. Best-effort and a no-op
        when no database is configured; never raises.
        """
        if not self._pool:
            return False
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO processed_features
                        (feature_type, event_id, station_id, producer_agent,
                         source_refs, handoff_id, trace_id, payload,
                         code_version)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        feature_type,
                        str(event_id) if event_id else None,
                        station_id,
                        producer_agent,
                        json.dumps(source_refs or [], default=str),
                        str(handoff_id),
                        str(trace_id),
                        json.dumps(payload, default=str),
                        code_version,
                    ),
                )
                conn.commit()
            return True
        except Exception:
            logger.exception("Failed to insert processed feature")
            return False

    def insert_quarantined_record(self, record: QuarantinedRecord) -> bool:
        """Persist a quarantined (validation-failed) record. Returns True on
        insert.

        Best-effort and a no-op when no database is configured.
        """
        if not self._pool:
            return False
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO quarantined_records
                        (source_id, source_type, reason_code, reason_detail,
                         raw_fields, quarantined_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        record.source_id,
                        record.source_type,
                        str(record.reason_code),
                        record.reason_detail,
                        json.dumps(record.raw_fields, default=str),
                        record.quarantined_at,
                    ),
                )
                conn.commit()
            return True
        except Exception:
            logger.exception("Failed to insert quarantined record")
            return False

    # ---- Audit trail ----

    def append_audit(self, entry: AuditEntry) -> bool:
        """Insert an audit entry into the database. Returns True on success."""
        if not self._pool:
            return False

        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events
                        (agent_name, action, event_id, trace_id, handoff_id,
                         state_before, state_after, decision_basis,
                         input_hashes, llm_invoked, reasoning_trace, recorded_at,
                         metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        entry.producer,  # AuditEntry.producer -> agent_name
                        entry.event_type,  # AuditEntry.event_type -> action
                        str(entry.event_id) if entry.event_id else None,
                        str(entry.trace_id) if entry.trace_id else None,
                        str(entry.data.get("handoff_id", ""))
                        if entry.data.get("handoff_id")
                        else None,
                        entry.data.get("from_state"),
                        entry.data.get("to_state"),
                        (
                            entry.data.get("trigger_reason")
                            or (
                                json.dumps(entry.data.get("decision_basis"))
                                if entry.data.get("decision_basis")
                                else None
                            )
                            or entry.event_type  # fallback to action name
                        ),
                        entry.data.get("input_hashes"),
                        entry.data.get("llm_invoked", False),
                        entry.data.get("reasoning_trace"),
                        entry.timestamp_utc,
                        json.dumps(entry.data) if entry.data else None,
                    ),
                )
                conn.commit()
            return True
        except Exception:
            logger.exception("Failed to append audit entry")
            return False

    def query_audit(
        self,
        event_id: UUID | None = None,
        event_type: str | None = None,
        trace_id: UUID | None = None,
        producer: str | None = None,
        limit: int = 100,
        offset: int = 0,
        raise_on_error: bool = False,
    ) -> list[dict[str, Any]]:
        """Query audit entries, optionally surfacing storage failures."""
        if not self._pool:
            if raise_on_error:
                raise RuntimeError("audit database pool is unavailable")
            return []

        conditions: list[str] = []
        params: list[Any] = []

        if event_id is not None:
            conditions.append("event_id = %s")
            params.append(str(event_id))
        if event_type is not None:
            conditions.append("action = %s")  # AuditEntry.event_type -> action
            params.append(event_type)
        if trace_id is not None:
            conditions.append("trace_id = %s")
            params.append(str(trace_id))
        if producer is not None:
            conditions.append("agent_name = %s")  # AuditEntry.producer -> agent_name
            params.append(producer)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        params.extend([limit, offset])

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    f"""
                    SELECT * FROM audit_events
                    {where}
                    ORDER BY recorded_at DESC, id DESC
                    LIMIT %s OFFSET %s
                    """,
                    params,
                ).fetchall()
                return list(rows)
        except Exception:
            logger.exception("Failed to query audit entries")
            if raise_on_error:
                raise
            return []

    def query_lineage(self, trace_id: UUID) -> list[dict[str, Any]] | None:
        """Query provenance chain for a trace ID using the get_provenance function.

        Returns the rows on success (possibly empty), or None when the query
        itself failed (no pool, permission error, broken function), so the
        caller can distinguish "no provenance" from "could not query".
        """
        if not self._pool:
            return None

        try:
            with self._conn() as conn:
                rows = conn.execute(
                    "SELECT * FROM get_provenance(%s)",
                    (str(trace_id),),
                ).fetchall()
                return list(rows)
        except Exception:
            logger.exception("Failed to query lineage")
            return None

    # ---- FSM state ----

    def load_fsm_state(self) -> dict[str, Any] | None:
        """Load the current FSM state from the database.

        Returns a dict with current_state, event_context, sensor_degraded,
        updated_at - or None if no database or no row found.
        """
        if not self._pool:
            return None

        try:
            with self._conn() as conn:
                row = conn.execute(
                    "SELECT current_state, event_context, sensor_degraded, updated_at "
                    "FROM fsm_current_state WHERE id = 1"
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            logger.exception("Failed to load FSM state")
            return None

    def upsert_fsm_state(
        self,
        state: str,
        event_context: dict[str, Any] | None = None,
        sensor_degraded: bool = False,
    ) -> bool:
        """Persist the current FSM state to the database.

        Uses advisory lock to serialize concurrent FSM mutations
        from api-server and pipeline-worker.
        """
        if not self._pool:
            return False

        try:
            with self._conn() as conn:
                # Transaction-scoped lock: released automatically on commit/rollback
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (FSM_LOCK_KEY,))
                conn.execute(
                    """
                    UPDATE fsm_current_state
                    SET current_state = %s,
                        event_context = %s,
                        sensor_degraded = %s,
                        updated_at = NOW()
                    WHERE id = 1
                    """,
                    (
                        state,
                        json.dumps(event_context) if event_context else None,
                        sensor_degraded,
                    ),
                )
                conn.commit()  # lock released here
            return True
        except Exception:
            logger.exception("Failed to upsert FSM state")
            return False

    def persist_dart_confirmation(
        self,
        event_id: UUID | str,
        stations_in_event_mode: list[str] | None = None,
    ) -> bool:
        """Durably latch confirmation and current event-mode stations.

        The UPDATE only matches when the durable row is still the SAME event and
        not IDLE, so a stale worker (whose event was already resolved to IDLE, or
        replaced by a different event) cannot resurrect it - the condition fails
        and no row is written. The current worker makes the latch survive a
        restart (without this, only the value from the last _transition is
        durable). Returns True if a row was updated.

        Held under the same advisory lock as upsert_fsm_state to serialize with
        concurrent FSM writes.
        """
        if not self._pool:
            return False
        try:
            with self._conn() as conn:
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (FSM_LOCK_KEY,))
                station_json = (
                    json.dumps(stations_in_event_mode)
                    if stations_in_event_mode is not None
                    else None
                )
                cur = conn.execute(
                    """
                    UPDATE fsm_current_state
                    SET event_context = CASE
                            WHEN %s::jsonb IS NULL THEN
                                jsonb_set(
                                    event_context,
                                    '{dart_confirmation}',
                                    'true'::jsonb
                                )
                            ELSE
                                jsonb_set(
                                    jsonb_set(
                                        jsonb_set(
                                            event_context,
                                            '{dart_confirmation}',
                                            'true'::jsonb
                                        ),
                                        '{stations_in_event_mode}',
                                        %s::jsonb
                                    ),
                                    '{active_dart_stations}',
                                    %s::jsonb
                                )
                            END,
                        updated_at = NOW()
                    WHERE id = 1
                      AND current_state <> 'IDLE'
                      AND event_context ->> 'event_id' = %s
                    """,
                    (station_json, station_json, station_json, str(event_id)),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            logger.exception("Failed to persist dart_confirmation")
            return False

    def persist_seismic_revision(
        self, event_id: UUID | str, revision: dict[str, Any]
    ) -> bool:
        """Durably merge the latest admissible seismic revision, conditionally.

        Same conditional pattern as persist_dart_confirmation: the UPDATE only
        matches while the durable row is still the SAME event and not IDLE, so
        a stale worker cannot write revision identity onto a resolved or
        replaced event. ``revision`` holds the latest_revision_* keys of the
        event-context JSON (see FSMOrchestrator.update_seismic_revision); the
        jsonb merge leaves every other context key, including the immutable
        trigger revision, untouched. Returns True if a row was updated.

        Held under the same advisory lock as upsert_fsm_state to serialize
        with concurrent FSM writes.
        """
        if not self._pool:
            return False
        try:
            with self._conn() as conn:
                conn.execute("SELECT pg_advisory_xact_lock(%s)", (FSM_LOCK_KEY,))
                cur = conn.execute(
                    """
                    UPDATE fsm_current_state
                    SET event_context = event_context || %s::jsonb,
                        updated_at = NOW()
                    WHERE id = 1
                      AND current_state <> 'IDLE'
                      AND event_context ->> 'event_id' = %s
                    """,
                    (json.dumps(revision), str(event_id)),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            logger.exception("Failed to persist seismic revision identity")
            return False

    # ---- Ocean evidence assessments ----

    def persist_assessment(
        self,
        *,
        checkpoint_id: str,
        schema_version: int,
        event_id: UUID | str,
        producer_agent: str,
        handoff_id: UUID | str,
        trace_id: UUID | str,
        payload: dict[str, Any],
        input_manifest_hash: str,
        scientific_content_hash: str,
        transport_provenance_hash: str | None = None,
        source_refs: list[dict[str, Any]] | None = None,
        code_version: str | None = None,
    ) -> AssessmentPersistResult:
        """Insert an assessment row, or return the existing identical one.

        Uses the partial unique index on (checkpoint_id,
        assessment_schema_version) as the idempotency key. When a row
        already exists under that key, it is returned as "existing" only
        when its event identity, input-manifest hash, and scientific hash
        all match this call; any mismatch is a "conflict" (hard failure
        under one logical checkpoint key). Assessment rows are
        immutable (migration 009), so this method never updates.
        """
        if not self._pool:
            return AssessmentPersistResult(
                status="error", detail="no database connection available"
            )
        try:
            with self._conn() as conn:
                inserted_row = conn.execute(
                    """
                    INSERT INTO processed_features
                        (feature_type, event_id, station_id, producer_agent,
                         source_refs, handoff_id, trace_id, payload,
                         code_version, checkpoint_id,
                         assessment_schema_version, input_manifest_hash,
                         scientific_content_hash, transport_provenance_hash)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
                            %s, %s)
                    ON CONFLICT (checkpoint_id, assessment_schema_version)
                        WHERE feature_type = 'ocean_evidence_assessment'
                    DO NOTHING
                    RETURNING *
                    """,
                    (
                        OCEAN_EVIDENCE_FEATURE_TYPE,
                        str(event_id),
                        None,
                        producer_agent,
                        json.dumps(source_refs or [], default=str),
                        str(handoff_id),
                        str(trace_id),
                        json.dumps(payload, default=str),
                        code_version,
                        checkpoint_id,
                        schema_version,
                        input_manifest_hash,
                        scientific_content_hash,
                        transport_provenance_hash,
                    ),
                ).fetchone()
                if inserted_row is not None:
                    conn.commit()
                    return AssessmentPersistResult(
                        status="inserted", row=dict(inserted_row)
                    )

                existing = conn.execute(
                    """
                    SELECT * FROM processed_features
                    WHERE feature_type = %s
                      AND checkpoint_id = %s
                      AND assessment_schema_version = %s
                    """,
                    (OCEAN_EVIDENCE_FEATURE_TYPE, checkpoint_id, schema_version),
                ).fetchone()
                conn.commit()
                if existing is None:
                    # DO NOTHING fired but the winning row is not visible:
                    # only possible if it was rolled back after blocking us.
                    return AssessmentPersistResult(
                        status="error",
                        detail=(
                            "insert skipped on conflict but no committed row "
                            "found for this checkpoint key"
                        ),
                    )
                existing_dict = dict(existing)
                mismatches: list[str] = []
                if str(existing_dict.get("event_id")) != str(event_id):
                    mismatches.append("event_id")
                if existing_dict.get("input_manifest_hash") != input_manifest_hash:
                    mismatches.append("input_manifest_hash")
                if (
                    existing_dict.get("scientific_content_hash")
                    != scientific_content_hash
                ):
                    mismatches.append("scientific_content_hash")
                if mismatches:
                    return AssessmentPersistResult(
                        status="conflict",
                        row=existing_dict,
                        detail="mismatched fields: " + ", ".join(mismatches),
                    )
                return AssessmentPersistResult(status="existing", row=existing_dict)
        except Exception as exc:
            logger.exception("Failed to persist assessment")
            return AssessmentPersistResult(status="error", detail=str(exc))

    def get_assessment_by_checkpoint(
        self, checkpoint_id: str, schema_version: int
    ) -> dict[str, Any] | None:
        """Return the assessment row for a checkpoint key, or None.

        None means either no row or a failed query. A caller using this
        for redelivery detection may safely treat a failed query as "no
        existing assessment": the subsequent persist_assessment call is
        idempotent and will surface the existing row.
        """
        if not self._pool:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM processed_features
                    WHERE feature_type = %s
                      AND checkpoint_id = %s
                      AND assessment_schema_version = %s
                    """,
                    (OCEAN_EVIDENCE_FEATURE_TYPE, checkpoint_id, schema_version),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get assessment by checkpoint")
            return None

    def get_assessment_by_handoff(
        self, handoff_id: UUID | str
    ) -> dict[str, Any] | None:
        """Return the assessment row with this handoff (assessment) ID."""
        if not self._pool:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM processed_features
                    WHERE feature_type = %s AND handoff_id = %s
                    """,
                    (OCEAN_EVIDENCE_FEATURE_TYPE, str(handoff_id)),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get assessment by handoff ID")
            return None

    def get_latest_assessment_for_event(
        self, event_id: UUID | str
    ) -> dict[str, Any] | None:
        """Return the most recently produced assessment row for an event."""
        if not self._pool:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM processed_features
                    WHERE feature_type = %s AND event_id = %s
                    ORDER BY produced_at DESC, id DESC
                    LIMIT 1
                    """,
                    (OCEAN_EVIDENCE_FEATURE_TYPE, str(event_id)),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get latest assessment for event")
            return None

    def append_assessment_checkpoint_attempt(
        self,
        *,
        checkpoint_id: str,
        schema_version: int,
        attempt_kind: str,
        outcome: str,
        event_id: UUID | str | None = None,
        trace_id: UUID | str | None = None,
        worker_run_id: UUID | str | None = None,
        transport_provenance_hash: str | None = None,
        detail: str = "",
    ) -> bool:
        """Append one original or redelivery attempt record.

        The attempt table is append-only; the assessment row's transport
        hash describes its creation attempt, and later attempts land here
        instead of mutating it. Best-effort: returns False on failure.
        """
        if not self._pool:
            return False
        try:
            with self._conn() as conn:
                conn.execute(
                    """
                    INSERT INTO assessment_checkpoint_attempt
                        (checkpoint_id, assessment_schema_version,
                         attempt_kind, outcome, event_id, trace_id,
                         worker_run_id, transport_provenance_hash, detail)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        checkpoint_id,
                        schema_version,
                        attempt_kind,
                        outcome,
                        str(event_id) if event_id else None,
                        str(trace_id) if trace_id else None,
                        str(worker_run_id) if worker_run_id else None,
                        transport_provenance_hash,
                        detail,
                    ),
                )
                conn.commit()
            return True
        except Exception:
            logger.exception("Failed to append assessment checkpoint attempt")
            return False

    def persist_escalation_packet(
        self,
        *,
        assessment_row_id: int,
        event_id: UUID | str,
        renderer_version: str,
        packet: dict[str, Any],
        content_sha256: str,
    ) -> AssessmentPersistResult:
        """Insert a reviewer packet row, or return the existing identical one.

        Idempotency key is UNIQUE (assessment_row_id, renderer_version)
        from migration 009. Reuses AssessmentPersistResult semantics:
        an existing row is "existing" only when its event identity and
        content hash match this call; any mismatch is a "conflict".
        Packet rows are append-only (migration 009 triggers), so this
        method never updates.
        """
        if not self._pool:
            return AssessmentPersistResult(
                status="error", detail="no database connection available"
            )
        try:
            with self._conn() as conn:
                inserted_row = conn.execute(
                    """
                    INSERT INTO escalation_packets
                        (assessment_row_id, event_id, renderer_version,
                         packet, content_sha256)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (assessment_row_id, renderer_version)
                    DO NOTHING
                    RETURNING *
                    """,
                    (
                        assessment_row_id,
                        str(event_id),
                        renderer_version,
                        json.dumps(packet, default=str),
                        content_sha256,
                    ),
                ).fetchone()
                if inserted_row is not None:
                    conn.commit()
                    return AssessmentPersistResult(
                        status="inserted", row=dict(inserted_row)
                    )

                existing = conn.execute(
                    """
                    SELECT * FROM escalation_packets
                    WHERE assessment_row_id = %s AND renderer_version = %s
                    """,
                    (assessment_row_id, renderer_version),
                ).fetchone()
                conn.commit()
                if existing is None:
                    return AssessmentPersistResult(
                        status="error",
                        detail=(
                            "insert skipped on conflict but no committed "
                            "packet row found for this assessment key"
                        ),
                    )
                existing_dict = dict(existing)
                mismatches: list[str] = []
                if str(existing_dict.get("event_id")) != str(event_id):
                    mismatches.append("event_id")
                if existing_dict.get("content_sha256") != content_sha256:
                    mismatches.append("content_sha256")
                if mismatches:
                    return AssessmentPersistResult(
                        status="conflict",
                        row=existing_dict,
                        detail="mismatched fields: " + ", ".join(mismatches),
                    )
                return AssessmentPersistResult(
                    status="existing", row=existing_dict
                )
        except Exception as exc:
            logger.exception("Failed to persist escalation packet")
            return AssessmentPersistResult(status="error", detail=str(exc))

    def get_escalation_packet_for_event(
        self, event_id: UUID | str
    ) -> dict[str, Any] | None:
        """Return the packet of record for an event, or None.

        Deterministic selection rule: the packet bound to the earliest
        assessment row for the event (the checkpoint that entered
        ESCALATE persists first under the single-writer worker), and the
        newest renderer version of that row if several exist. Never an
        unspecified latest assessment. renderer_version is a TEXT column
        holding integer strings; ordering by length before value gives
        numeric order ("10" > "2") without a cast.
        """
        if not self._pool:
            return None
        try:
            with self._conn() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM escalation_packets
                    WHERE event_id = %s
                    ORDER BY assessment_row_id ASC,
                             length(renderer_version) DESC,
                             renderer_version DESC
                    LIMIT 1
                    """,
                    (str(event_id),),
                ).fetchone()
                return dict(row) if row else None
        except Exception:
            logger.exception("Failed to get escalation packet for event")
            return None

    # ---- Audit count (for UI) ----

    def count_audit(
        self,
        event_id: UUID | None = None,
        event_type: str | None = None,
    ) -> int:
        """Count audit entries matching filters."""
        if not self._pool:
            return 0

        conditions: list[str] = []
        params: list[Any] = []

        if event_id is not None:
            conditions.append("event_id = %s")
            params.append(str(event_id))
        if event_type is not None:
            conditions.append("action = %s")  # AuditEntry.event_type -> action
            params.append(event_type)

        where = "WHERE " + " AND ".join(conditions) if conditions else ""

        try:
            with self._conn() as conn:
                row = conn.execute(
                    f"SELECT COUNT(*) AS cnt FROM audit_events {where}",
                    params,
                ).fetchone()
                return row["cnt"] if row else 0
        except Exception:
            logger.exception("Failed to count audit entries")
            return 0
