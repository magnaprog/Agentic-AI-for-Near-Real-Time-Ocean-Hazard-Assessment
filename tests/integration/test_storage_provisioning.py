"""Integration test for TimescaleDB schema provisioning."""

from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

import pytest

try:
    import psycopg
    from psycopg import sql
except ModuleNotFoundError:  # pragma: no cover - exercised in minimal test envs
    psycopg = None  # type: ignore[assignment]
    sql = None  # type: ignore[assignment]

from hazard_assessment.storage.provision import apply_migrations, discover_migrations

pytestmark = pytest.mark.skipif(psycopg is None, reason="psycopg is not installed")

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "hazard_assessment"
    / "storage"
    / "migrations"
)

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", "5432"))
DB_ADMIN_USER = os.getenv("DB_ADMIN_USER", "hazard_admin")
DB_ADMIN_PASSWORD = os.getenv("DB_ADMIN_PASSWORD", "")
DB_ADMIN_NAME = os.getenv("DB_ADMIN_NAME", "postgres")


@pytest.fixture(scope="module")
def provisioned_test_database() -> str:
    db_name = f"hazard_e2t1_{uuid4().hex[:10]}"

    try:
        with psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_ADMIN_NAME,
            user=DB_ADMIN_USER,
            password=DB_ADMIN_PASSWORD,
            connect_timeout=3,
            autocommit=True,
        ) as admin_conn:
            admin_conn.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(db_name))
            )
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL is not reachable for integration tests: {exc}")
    except psycopg.Error as exc:
        if exc.sqlstate == "42501":
            pytest.skip(
                "Integration tests require CREATE DATABASE privilege for the admin role."
            )
        raise

    yield db_name

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_ADMIN_NAME,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
        autocommit=True,
    ) as admin_conn:
        admin_conn.execute(
            sql.SQL("DROP DATABASE IF EXISTS {} WITH (FORCE)").format(
                sql.Identifier(db_name)
            )
        )


def test_migration_applies_twice_and_enforces_access_controls(
    provisioned_test_database: str,
) -> None:
    migrations = discover_migrations(MIGRATIONS_DIR)

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        first = apply_migrations(conn, migrations)
        second = apply_migrations(conn, migrations)

        # Derive the expected applied-version list from the discovered
        # migrations rather than hardcoding it: a hardcoded duplicate desyncs
        # the moment a migration file is added (the root cause of the 008
        # incident). The role/grant assertions below independently verify the
        # critical migrations (001/003/008) actually ran.
        all_versions = [m.version for m in migrations]
        assert first.applied_versions == all_versions
        assert first.skipped_versions == []
        assert second.applied_versions == []
        assert second.skipped_versions == all_versions

        required_roles = {"ingest_writer", "agent_writer", "agent_reader", "audit_reader"}
        role_rows = conn.execute(
            """
            SELECT rolname
            FROM pg_roles
            WHERE rolname = ANY(%s)
            """,
            (list(required_roles),),
        ).fetchall()
        assert {str(row[0]) for row in role_rows} == required_roles

        # Role memberships for the four application roles should remain empty:
        # each service logs in directly as its least-privilege role.
        membership_rows = conn.execute(
            """
            SELECT member.rolname, role.rolname
            FROM pg_auth_members m
            JOIN pg_roles role ON role.oid = m.roleid
            JOIN pg_roles member ON member.oid = m.member
            WHERE member.rolname = ANY(%s)
            """,
            (list(required_roles),),
        ).fetchall()
        assert membership_rows == []

        limited_role_rows = conn.execute(
            """
            SELECT
                rolname, rolsuper, rolinherit, rolcreatedb,
                rolcreaterole, rolreplication, rolcanlogin
            FROM pg_roles
            WHERE rolname = ANY(%s)
            """,
            (list(required_roles),),
        ).fetchall()
        role_flags = {
            str(row[0]): tuple(bool(value) for value in row[1:])
            for row in limited_role_rows
        }
        for role_name in required_roles:
            assert role_flags[role_name] == (False, False, False, False, False, True)

        hazard_app_count = conn.execute(
            "SELECT count(*) FROM pg_roles WHERE rolname = 'hazard_app'"
        ).fetchone()
        assert hazard_app_count is not None
        assert int(hazard_app_count[0]) == 0

        trigger_rows = conn.execute(
            """
            SELECT tgname
            FROM pg_trigger
            WHERE tgrelid = 'audit_events'::regclass
              AND NOT tgisinternal
            """
        ).fetchall()
        trigger_names = {str(row[0]) for row in trigger_rows}
        assert "audit_events_block_update" in trigger_names
        assert "audit_events_block_delete" in trigger_names

        # 004: quarantined_records is append-only via block update/delete triggers.
        quarantine_trigger_rows = conn.execute(
            """
            SELECT tgname
            FROM pg_trigger
            WHERE tgrelid = 'quarantined_records'::regclass
              AND NOT tgisinternal
            """
        ).fetchall()
        quarantine_trigger_names = {str(row[0]) for row in quarantine_trigger_rows}
        assert "quarantined_records_block_update" in quarantine_trigger_names
        assert "quarantined_records_block_delete" in quarantine_trigger_names

        # 005: get_provenance runs as SECURITY DEFINER so the EXECUTE-granted
        # roles (which lack uniform SELECT on the underlying tables) can call it.
        secdef_row = conn.execute(
            "SELECT prosecdef FROM pg_proc WHERE proname = 'get_provenance'"
        ).fetchone()
        assert secdef_row is not None and bool(secdef_row[0]) is True

        # 006: EXECUTE is revoked from PUBLIC (a SECURITY DEFINER function must
        # not be runnable by unintended roles), and re-granted to the intended
        # roles only.
        public_exec = conn.execute(
            "SELECT has_function_privilege('public', 'get_provenance(uuid)', 'EXECUTE')"
        ).fetchone()
        assert public_exec is not None and bool(public_exec[0]) is False
        for role in ("agent_reader", "audit_reader", "orchestrator_writer"):
            role_exec = conn.execute(
                "SELECT has_function_privilege(%s, 'get_provenance(uuid)', 'EXECUTE')",
                (role,),
            ).fetchone()
            assert role_exec is not None and bool(role_exec[0]) is True

        # 007: pg_temp is pinned LAST in the function's search_path. An
        # unlisted pg_temp is searched FIRST for relation names, so an
        # EXECUTE-granted role with TEMP privilege (PUBLIC default) could
        # otherwise shadow the tables this SECURITY DEFINER function reads
        # with a temp object whose expressions would evaluate with definer
        # rights.
        proconfig_row = conn.execute(
            "SELECT proconfig FROM pg_proc WHERE proname = 'get_provenance'"
        ).fetchone()
        assert proconfig_row is not None and proconfig_row[0] is not None
        sp = next(
            (c for c in proconfig_row[0] if str(c).startswith("search_path=")),
            None,
        )
        assert sp is not None
        sp_parts = [p.strip() for p in str(sp).removeprefix("search_path=").split(",")]
        assert sp_parts == ["pg_catalog", "public", "pg_temp"]

        # Behavioral check: a temp-table shadow of processed_features planted
        # by an EXECUTE-granted role must NOT be read by the function. The
        # function LEFT-JOINs from processed_features, so if temp resolution
        # won, the planted row would come back; with pg_temp pinned last the
        # real (empty) table wins and the result is empty.
        shadow_trace = uuid4()
        conn.execute("SET ROLE orchestrator_writer")
        try:
            conn.execute(
                """
                CREATE TEMP TABLE processed_features (
                    id BIGINT,
                    feature_type TEXT,
                    event_id UUID,
                    trace_id UUID,
                    handoff_id UUID,
                    produced_at TIMESTAMPTZ,
                    producer_agent TEXT,
                    source_refs JSONB,
                    code_version TEXT
                )
                """
            )
            conn.execute(
                """
                INSERT INTO pg_temp.processed_features VALUES
                    (1, 'shadow', gen_random_uuid(), %s, gen_random_uuid(),
                     NOW(), 'attacker', '{}'::jsonb, 'x')
                """,
                (shadow_trace,),
            )
            shadow_rows = conn.execute(
                "SELECT * FROM get_provenance(%s)", (shadow_trace,)
            ).fetchall()
        finally:
            conn.execute("RESET ROLE")
            conn.execute("DROP TABLE IF EXISTS pg_temp.processed_features")
        assert shadow_rows == []

        # 004: quarantined_records is append-only - UPDATE and DELETE raise.
        # Commit the row so it persists across both per-row trigger checks.
        conn.execute(
            """
            INSERT INTO quarantined_records
                (source_id, source_type, reason_code, reason_detail)
            VALUES ('s1', 'dart', 'schema_validation_failed', 'x')
            """
        )
        conn.commit()
        for stmt in (
            "UPDATE quarantined_records SET source_id = 's2' WHERE source_id = 's1'",
            "DELETE FROM quarantined_records WHERE source_id = 's1'",
        ):
            # Match the message, not only the error class. Migration 004 raises
            # the trigger with ERRCODE 42501, the same SQLSTATE PostgreSQL uses
            # for a missing grant, so the class alone cannot say which layer
            # stopped the write. This connection is the table owner and holds
            # UPDATE and DELETE, so today only the trigger can; matching the
            # message keeps that true if the roles are ever changed. The
            # audit_events assertions below already work this way.
            with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
                conn.execute(stmt)
            conn.rollback()

        grant_rows = conn.execute(
            """
            SELECT grantee, table_name, privilege_type
            FROM information_schema.role_table_grants
            WHERE grantee = ANY(%s)
              AND table_name = ANY(%s)
            """,
            (
                ["ingest_writer", "agent_writer", "agent_reader", "audit_reader"],
                ["raw_observations", "processed_features", "audit_events"],
            ),
        ).fetchall()
        grants: dict[tuple[str, str], set[str]] = {}
        for grantee, table_name, privilege_type in grant_rows:
            key = (str(grantee), str(table_name))
            grants.setdefault(key, set()).add(str(privilege_type))

        assert "INSERT" in grants[("ingest_writer", "raw_observations")]
        assert "INSERT" in grants[("ingest_writer", "audit_events")]

        # Migration 012 removed agent_writer's INSERT on processed_features.
        # That table holds the append-only ocean-evidence assessments, whose
        # rows nobody can correct once written, so a role no application
        # connects as must not be able to claim a checkpoint key.
        assert ("agent_writer", "processed_features") not in grants
        assert "INSERT" in grants[("agent_writer", "audit_events")]

        assert "SELECT" in grants[("agent_reader", "raw_observations")]
        assert "SELECT" in grants[("agent_reader", "processed_features")]

        assert "SELECT" in grants[("audit_reader", "audit_events")]
        assert "SELECT" in grants[("audit_reader", "raw_observations")]
        assert "SELECT" in grants[("audit_reader", "processed_features")]

        # 001 created provenance_chain and 002 rebuilt it; 003 replaced it with
        # get_provenance() because the view's LATERAL join defeats chunk
        # exclusion, but left the view in place. 013 drops it, so a fully
        # migrated database offers one provenance path rather than a fast
        # function beside the slow view it superseded.
        assert conn.execute(
            "SELECT COUNT(*) FROM information_schema.views "
            "WHERE table_name = 'provenance_chain'"
        ).fetchone()[0] == 0, "provenance_chain should be dropped by 013"
        assert conn.execute(
            "SELECT COUNT(*) FROM pg_proc WHERE proname = 'get_provenance'"
        ).fetchone()[0] == 1, "get_provenance() is the surviving provenance path"

        no_update_delete_rows = conn.execute(
            """
            SELECT grantee, table_name, privilege_type
            FROM information_schema.role_table_grants
            WHERE grantee = ANY(%s)
              AND table_name = ANY(%s)
              AND privilege_type IN ('UPDATE', 'DELETE')
            """,
            (
                ["ingest_writer", "agent_writer", "agent_reader", "audit_reader"],
                ["audit_events", "raw_observations"],
            ),
        ).fetchall()
        assert no_update_delete_rows == []

        inserted_id_row = conn.execute(
            """
            INSERT INTO audit_events (agent_name, action, decision_basis)
            VALUES (%s, %s, %s)
            RETURNING id
            """,
            ("test_agent", "test_action", "integration-test"),
        ).fetchone()
        assert inserted_id_row is not None
        inserted_id = int(inserted_id_row[0])
        conn.commit()

        with pytest.raises(psycopg.Error, match="append-only"):
            conn.execute(
                "UPDATE audit_events SET action = %s WHERE id = %s",
                ("mutated_action", inserted_id),
            )
        conn.rollback()

        with pytest.raises(psycopg.Error, match="append-only"):
            conn.execute(
                "DELETE FROM audit_events WHERE id = %s",
                (inserted_id,),
            )
        conn.rollback()


def test_fsm_persistence_round_trip_and_advisory_lock(
    provisioned_test_database: str,
) -> None:
    """Real-DB FSM persistence: the DatabaseClient round-trip preserves
    dart_confirmation, an IDLE upsert stores a NULL context,
    persist_dart_confirmation stores the latch and station list conditionally
    on event_id + non-IDLE, and the advisory lock serializes connections.
    """
    from uuid import uuid4

    from hazard_assessment.storage.client import (
        FSM_LOCK_KEY,
        ClientConfig,
        DatabaseClient,
    )

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        ev_a, ev_b = uuid4(), uuid4()

        def _ctx(ev: object, dart: bool = False) -> dict[str, object]:
            return {
                "event_id": str(ev),
                "seismic_magnitude": 9.1,
                "seismic_region": "japan",
                "epicenter_lat": 38.3,
                "epicenter_lon": 142.37,
                "depth_km": 29.0,
                "trigger_time_utc": "2011-03-11T05:46:24+00:00",
                "latest_anomaly_score": 0.0,
                "dart_confirmation": dart,
                "active_dart_stations": [],
                "stations_in_event_mode": [],
            }

        # Round-trip preserves dart_confirmation.
        db.upsert_fsm_state(state="ESCALATE", event_context=_ctx(ev_a, dart=True))
        row = db.load_fsm_state()
        assert row is not None
        assert row["current_state"] == "ESCALATE"
        assert row["event_context"]["dart_confirmation"] is True

        # IDLE stores a NULL context.
        db.upsert_fsm_state(state="IDLE", event_context=None)
        row = db.load_fsm_state()
        assert row is not None
        assert row["current_state"] == "IDLE"
        assert row["event_context"] is None

        # Conditional persist: no-op after resolution; latches on the current event.
        assert db.persist_dart_confirmation(ev_a) is False  # row is IDLE
        db.upsert_fsm_state(state="MONITOR", event_context=_ctx(ev_b, dart=False))
        assert db.persist_dart_confirmation(ev_a) is False  # different event
        assert db.persist_dart_confirmation(
            ev_b, ["21418", "46403"]
        ) is True  # current event
        row = db.load_fsm_state()
        assert row is not None
        assert row["event_context"]["dart_confirmation"] is True
        assert row["event_context"]["stations_in_event_mode"] == ["21418", "46403"]
        assert row["event_context"]["active_dart_stations"] == ["21418", "46403"]
    finally:
        db.close()

    # Advisory-lock mutual exclusion across two connections.
    a = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
        autocommit=False,
    )
    b = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
        autocommit=True,
    )
    try:
        a.execute("SELECT pg_advisory_xact_lock(%s)", (FSM_LOCK_KEY,))
        held = b.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (FSM_LOCK_KEY,)
        ).fetchone()
        assert held is not None and held[0] is False
        b.execute("SELECT pg_advisory_unlock_all()")
        a.commit()
        freed = b.execute(
            "SELECT pg_try_advisory_xact_lock(%s)", (FSM_LOCK_KEY,)
        ).fetchone()
        assert freed is not None and freed[0] is True
    finally:
        a.close()
        b.close()


def test_processed_feature_provenance_join_round_trip(
    provisioned_test_database: str,
) -> None:
    """Real-DB lineage chain: a processed_features row plus a companion audit
    entry (shared handoff_id and trace_id, input_hashes carrying the raw
    payload hash) joins through get_provenance() back to the
    raw_observations row. This is the end-to-end contract behind
    /api/lineage/provenance/{trace_id} and the worker's lineage producer.
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from hazard_assessment.audit.logger import AuditEntry
    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        trace_id = uuid4()
        handoff_id = uuid4()
        event_id = uuid4()
        payload_hash = uuid4().hex + uuid4().hex  # 64 hex chars

        inserted = db.insert_observations(
            [
                {
                    "station_id": "21418",
                    "observed_at": datetime.now(UTC),
                    "payload": {"height_m": 5500.0},
                    "payload_hash": payload_hash,
                },
            ],
            source_type="dart",
        )
        assert inserted == 1

        assert db.append_audit(AuditEntry(
            event_id=event_id,
            trace_id=trace_id,
            event_type="anomaly_scored",
            producer="anomaly_agent",
            data={
                "station_id": "21418",
                "handoff_id": str(handoff_id),
                "input_hashes": [payload_hash],
            },
        )) is True

        assert db.insert_processed_feature(
            feature_type="anomaly_score",
            producer_agent="anomaly_agent",
            handoff_id=handoff_id,
            trace_id=trace_id,
            payload={"anomaly_score": 0.42},
            source_refs=[{"sha256": payload_hash}],
            event_id=event_id,
            station_id="21418",
        ) is True

        rows = db.query_lineage(trace_id)
        assert rows is not None
        assert len(rows) >= 1
        row = dict(rows[0])
        assert row["feature_type"] == "anomaly_score"
        assert str(row["handoff_id"]) == str(handoff_id)
        assert row["raw_payload_hash"] == payload_hash
        assert row["raw_station_id"] == "21418"

        # The pipeline worker connects as pipeline_worker since migration 009
        # (008's grant to orchestrator_writer was a stopgap that 009 revokes):
        # the worker role can insert lineage rows, the API-facing role cannot.
        with psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=provisioned_test_database,
            user=DB_ADMIN_USER,
            password=DB_ADMIN_PASSWORD,
            connect_timeout=3,
            autocommit=True,
        ) as conn:
            conn.execute("SET ROLE pipeline_worker")
            conn.execute(
                """
                INSERT INTO processed_features
                    (feature_type, producer_agent, source_refs, handoff_id,
                     trace_id, payload)
                VALUES ('anomaly_score', 'anomaly_agent', '[]', %s, %s, '{}')
                """,
                (str(uuid4()), str(uuid4())),
            )
            conn.execute("RESET ROLE")
            conn.execute("SET ROLE orchestrator_writer")
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                conn.execute(
                    """
                    INSERT INTO processed_features
                        (feature_type, producer_agent, source_refs, handoff_id,
                         trace_id, payload)
                    VALUES ('anomaly_score', 'anomaly_agent', '[]', %s, %s, '{}')
                    """,
                    (str(uuid4()), str(uuid4())),
                )
            conn.execute("RESET ROLE")
    finally:
        db.close()


def _hex64(n: int) -> str:
    return f"{n:064x}"


def test_assessment_persistence_idempotency_and_retrieval(
    provisioned_test_database: str,
) -> None:
    """Real-DB assessment persistence (migration 009 + client):
    insert-or-return-existing keyed on (checkpoint_id, schema_version),
    hard conflict on content divergence under one checkpoint key, malformed
    identity rejected by the CHECK constraint, retrieval methods, and the
    append-only attempt audit including its attempt_kind CHECK."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        event_id = uuid4()
        checkpoint = _hex64(0xA1)
        base = dict(
            checkpoint_id=checkpoint,
            schema_version=1,
            event_id=event_id,
            producer_agent="pipeline_worker",
            handoff_id=uuid4(),
            trace_id=uuid4(),
            payload={"type": "OceanEvidenceAssessment", "stations": []},
            input_manifest_hash=_hex64(0xB1),
            scientific_content_hash=_hex64(0xC1),
            transport_provenance_hash=_hex64(0xD1),
            source_refs=[{"sha256": _hex64(0xE1)}],
            code_version="test",
        )

        first = db.persist_assessment(**base)
        assert first.status == "inserted"
        assert first.row is not None
        row_id = first.row["id"]

        # Identical redelivery: the existing row is adopted, not duplicated.
        again = db.persist_assessment(**base)
        assert again.status == "existing"
        assert again.row is not None
        assert again.row["id"] == row_id

        # Same checkpoint key, diverged content: hard conflict.
        diverged = dict(base, scientific_content_hash=_hex64(0xC2))
        conflict = db.persist_assessment(**diverged)
        assert conflict.status == "conflict"
        assert "scientific_content_hash" in conflict.detail

        # A second schema version under the same checkpoint id is a
        # distinct idempotency key.
        v2 = db.persist_assessment(
            **dict(base, schema_version=2, handoff_id=uuid4(), trace_id=uuid4())
        )
        assert v2.status == "inserted"

        # Malformed checkpoint identity never lands: the row CHECK rejects
        # it and the client reports an error status.
        bad = db.persist_assessment(
            **dict(base, checkpoint_id="not-a-hash", handoff_id=uuid4())
        )
        assert bad.status == "error"

        # Retrieval methods.
        got = db.get_assessment_by_checkpoint(checkpoint, 1)
        assert got is not None and got["id"] == row_id
        assert db.get_assessment_by_checkpoint(_hex64(0xFF), 1) is None
        by_handoff = db.get_assessment_by_handoff(base["handoff_id"])
        assert by_handoff is not None and by_handoff["id"] == row_id

        second = db.persist_assessment(
            **dict(
                base,
                checkpoint_id=_hex64(0xA2),
                handoff_id=uuid4(),
                trace_id=uuid4(),
            )
        )
        assert second.status == "inserted"
        assert second.row is not None
        latest = db.get_latest_assessment_for_event(event_id)
        assert latest is not None
        assert latest["id"] == second.row["id"]

        # Attempt audit: appends succeed, malformed attempt_kind is
        # rejected by the table CHECK and surfaces as False.
        assert db.append_assessment_checkpoint_attempt(
            checkpoint_id=checkpoint,
            schema_version=1,
            attempt_kind="redelivery",
            outcome="existing",
            event_id=event_id,
            worker_run_id=uuid4(),
            transport_provenance_hash=_hex64(0xD2),
            detail="integration test",
        ) is True
        assert db.append_assessment_checkpoint_attempt(
            checkpoint_id=checkpoint,
            schema_version=1,
            attempt_kind="bogus",
            outcome="existing",
        ) is False
    finally:
        db.close()

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        n = conn.execute(
            "SELECT count(*) FROM assessment_checkpoint_attempt "
            "WHERE checkpoint_id = %s",
            (checkpoint,),
        ).fetchone()
        assert n is not None and int(n[0]) == 1


def test_escalation_packet_persistence_round_trip(
    provisioned_test_database: str,
) -> None:
    """Real-DB reviewer packet contract (migration 009 + client):
    idempotent insert keyed on (assessment_row_id, renderer_version), hard
    conflict on content divergence, earliest-assessment-row retrieval
    independent of insertion order, hash recomputable from the stored
    JSONB via canonical re-serialization, hex CHECK on the hash column,
    and append-only immutability."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient
    from hazard_assessment.workers.reviewer_packet import (
        RENDERER_VERSION,
        canonical_packet_hash,
        render_reviewer_packet,
    )

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        event_id = uuid4()

        def _assessment(checkpoint_hex: int, state_after: str) -> tuple[int, dict]:
            payload = {
                "checkpoint_id": _hex64(checkpoint_hex),
                "event_id": str(event_id),
                "fsm_state_before": "ASSESS",
                "fsm_state_after": state_after,
                "pipeline_outcome": "ABSTAIN",
                "stations": [],
            }
            result = db.persist_assessment(
                checkpoint_id=_hex64(checkpoint_hex),
                schema_version=1,
                event_id=event_id,
                producer_agent="pipeline_worker",
                handoff_id=uuid4(),
                trace_id=uuid4(),
                payload=payload,
                input_manifest_hash=_hex64(checkpoint_hex + 1),
                scientific_content_hash=_hex64(checkpoint_hex + 2),
            )
            assert result.status == "inserted"
            assert result.row is not None
            return int(result.row["id"]), payload

        row1_id, payload1 = _assessment(0x6A1, "ESCALATE")
        row2_id, payload2 = _assessment(0x6B1, "ESCALATE")
        assert row1_id < row2_id

        packet1, hash1 = render_reviewer_packet(
            assessment_payload=payload1, assessment_row_id=row1_id
        )
        packet2, hash2 = render_reviewer_packet(
            assessment_payload=payload2, assessment_row_id=row2_id
        )

        # Insert LATER assessment's packet first: retrieval must still
        # return the packet bound to the earliest assessment row.
        later = db.persist_escalation_packet(
            assessment_row_id=row2_id,
            event_id=event_id,
            renderer_version=RENDERER_VERSION,
            packet=packet2,
            content_sha256=hash2,
        )
        assert later.status == "inserted"
        first = db.persist_escalation_packet(
            assessment_row_id=row1_id,
            event_id=event_id,
            renderer_version=RENDERER_VERSION,
            packet=packet1,
            content_sha256=hash1,
        )
        assert first.status == "inserted"
        assert first.row is not None
        packet_row_id = int(first.row["id"])

        # Identical redelivery: existing row adopted, not duplicated.
        again = db.persist_escalation_packet(
            assessment_row_id=row1_id,
            event_id=event_id,
            renderer_version=RENDERER_VERSION,
            packet=packet1,
            content_sha256=hash1,
        )
        assert again.status == "existing"
        assert again.row is not None
        assert int(again.row["id"]) == packet_row_id

        # Same (assessment_row_id, renderer_version), diverged hash:
        # disclosed as a hard conflict, never silently adopted.
        conflict = db.persist_escalation_packet(
            assessment_row_id=row1_id,
            event_id=event_id,
            renderer_version=RENDERER_VERSION,
            packet=packet1,
            content_sha256=_hex64(0x6F1),
        )
        assert conflict.status == "conflict"
        assert "content_sha256" in (conflict.detail or "")

        # Earliest-assessment-row rule plus JSONB hash recomputability.
        got = db.get_escalation_packet_for_event(event_id)
        assert got is not None
        assert int(got["assessment_row_id"]) == row1_id
        assert got["renderer_version"] == RENDERER_VERSION
        assert got["content_sha256"] == hash1
        assert canonical_packet_hash(got["packet"]) == hash1
        assert db.get_escalation_packet_for_event(uuid4()) is None

        # Malformed hash never lands: rejected by the 64-hex CHECK.
        bad = db.persist_escalation_packet(
            assessment_row_id=row2_id,
            event_id=event_id,
            renderer_version="2",
            packet=packet2,
            content_sha256="not-a-hash",
        )
        assert bad.status == "error"
    finally:
        db.close()

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        for stmt in (
            "UPDATE escalation_packets SET content_sha256 = %s WHERE id = %s",
            "DELETE FROM escalation_packets WHERE id = %s",
        ):
            params = (
                (_hex64(0x6E1), packet_row_id)
                if "UPDATE" in stmt
                else (packet_row_id,)
            )
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(stmt, params)
            conn.rollback()


def test_assessment_and_attempt_rows_are_immutable(
    provisioned_test_database: str,
) -> None:
    """Migration 009 conditional immutability: assessment rows can be
    neither updated, deleted, nor forged from a non-assessment row; the
    attempt table is fully append-only. This is also the retention
    protection for review-bound assessments."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        checkpoint = _hex64(0x1A1)
        result = db.persist_assessment(
            checkpoint_id=checkpoint,
            schema_version=1,
            event_id=uuid4(),
            producer_agent="pipeline_worker",
            handoff_id=uuid4(),
            trace_id=uuid4(),
            payload={},
            input_manifest_hash=_hex64(0x1B1),
            scientific_content_hash=_hex64(0x1C1),
        )
        assert result.status == "inserted"
        assert result.row is not None
        row_id = result.row["id"]
        assert db.append_assessment_checkpoint_attempt(
            checkpoint_id=checkpoint,
            schema_version=1,
            attempt_kind="original",
            outcome="inserted",
        ) is True
    finally:
        db.close()

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        for stmt, params in (
            (
                "UPDATE processed_features SET payload = '{\"x\": 1}' "
                "WHERE id = %s",
                (row_id,),
            ),
            ("DELETE FROM processed_features WHERE id = %s", (row_id,)),
        ):
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(stmt, params)
            conn.rollback()

        # Forge protection: re-typing an ordinary lineage row INTO an
        # assessment is blocked by the NEW-side trigger condition.
        lineage = conn.execute(
            """
            INSERT INTO processed_features
                (feature_type, producer_agent, source_refs, handoff_id,
                 trace_id, payload)
            VALUES ('anomaly_score', 'anomaly_agent', '[]', %s, %s, '{}')
            RETURNING id
            """,
            (str(uuid4()), str(uuid4())),
        ).fetchone()
        assert lineage is not None
        conn.commit()
        with pytest.raises(psycopg.Error, match="append-only"):
            conn.execute(
                "UPDATE processed_features "
                "SET feature_type = 'ocean_evidence_assessment' WHERE id = %s",
                (int(lineage[0]),),
            )
        conn.rollback()

        attempt_id = conn.execute(
            "SELECT id FROM assessment_checkpoint_attempt "
            "WHERE checkpoint_id = %s",
            (_hex64(0x1A1),),
        ).fetchone()
        assert attempt_id is not None
        for stmt in (
            "UPDATE assessment_checkpoint_attempt SET outcome = 'x' "
            f"WHERE id = {int(attempt_id[0])}",
            "DELETE FROM assessment_checkpoint_attempt "
            f"WHERE id = {int(attempt_id[0])}",
        ):
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(stmt)
            conn.rollback()


def test_pipeline_worker_and_investigator_role_grants(
    provisioned_test_database: str,
) -> None:
    """Migration 009 role separation: the dedicated worker role holds
    exactly the worker write surface plus assessment and reviewer-packet
    insertion, the investigator writer can persist only its own results,
    and the API-facing role can only read reviewer packets."""
    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        new_roles = {"pipeline_worker", "investigator_writer"}
        flag_rows = conn.execute(
            """
            SELECT
                rolname, rolsuper, rolinherit, rolcreatedb,
                rolcreaterole, rolreplication, rolcanlogin
            FROM pg_roles
            WHERE rolname = ANY(%s)
            """,
            (list(new_roles),),
        ).fetchall()
        flags = {
            str(row[0]): tuple(bool(v) for v in row[1:]) for row in flag_rows
        }
        assert set(flags) == new_roles
        for role in new_roles:
            assert flags[role] == (False, False, False, False, False, True)

        def priv(role: str, table: str, privilege: str) -> bool:
            got = conn.execute(
                "SELECT has_table_privilege(%s, %s, %s)",
                (role, table, privilege),
            ).fetchone()
            assert got is not None
            return bool(got[0])

        # pipeline_worker: worker write surface plus assessment insertion.
        assert priv("pipeline_worker", "processed_features", "INSERT")
        assert priv("pipeline_worker", "processed_features", "SELECT")
        assert priv("pipeline_worker", "assessment_checkpoint_attempt", "INSERT")
        assert priv("pipeline_worker", "assessment_checkpoint_attempt", "SELECT")
        assert priv("pipeline_worker", "fsm_current_state", "UPDATE")
        assert priv("pipeline_worker", "audit_events", "INSERT")
        assert not priv("pipeline_worker", "evidence_issue_results", "INSERT")

        # orchestrator_writer lost the 008 stopgap lineage insert.
        assert not priv("orchestrator_writer", "processed_features", "INSERT")

        # investigator_writer: own results only, read-only elsewhere.
        assert priv("investigator_writer", "evidence_issue_results", "INSERT")
        assert priv("investigator_writer", "evidence_issue_results", "SELECT")
        assert priv("investigator_writer", "processed_features", "SELECT")
        assert not priv("investigator_writer", "processed_features", "INSERT")
        assert not priv(
            "investigator_writer", "assessment_checkpoint_attempt", "INSERT"
        )

        # Reviewer packets: only the pipeline worker writes them;
        # the API-facing role reads them; other roles get nothing.
        assert priv("pipeline_worker", "escalation_packets", "INSERT")
        assert priv("pipeline_worker", "escalation_packets", "SELECT")
        assert priv("orchestrator_writer", "escalation_packets", "SELECT")
        assert not priv("orchestrator_writer", "escalation_packets", "INSERT")
        for role in ("investigator_writer", "agent_writer", "ingest_writer"):
            assert not priv(role, "escalation_packets", "INSERT")
            assert not priv(role, "escalation_packets", "SELECT")


def test_evidence_issue_results_uniqueness_and_immutability(
    provisioned_test_database: str,
) -> None:
    """Migration 009 evidence_issue_results contract: deterministic
    invocation IDs are unique, hashes are format-checked, rows are
    immutable, and the investigator role can write them under the real
    grant while the FK ties each result to a durable assessment row."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        result = db.persist_assessment(
            checkpoint_id=_hex64(0x2A1),
            schema_version=1,
            event_id=uuid4(),
            producer_agent="pipeline_worker",
            handoff_id=uuid4(),
            trace_id=uuid4(),
            payload={},
            input_manifest_hash=_hex64(0x2B1),
            scientific_content_hash=_hex64(0x2C1),
        )
        assert result.status == "inserted"
        assert result.row is not None
        assessment_row_id = int(result.row["id"])
    finally:
        db.close()

    invocation = _hex64(0x3A1)
    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
        autocommit=True,
    ) as conn:
        conn.execute("SET ROLE investigator_writer")
        try:
            conn.execute(
                """
                INSERT INTO evidence_issue_results
                    (invocation_id, assessment_row_id, issue_name, result,
                     result_sha256)
                VALUES (%s, %s, 'detiding_quality', '{}', %s)
                """,
                (invocation, assessment_row_id, _hex64(0x3B1)),
            )
            # Repeat-run identity: the same deterministic invocation ID
            # cannot land twice, even with a different result hash.
            with pytest.raises(psycopg.errors.UniqueViolation):
                conn.execute(
                    """
                    INSERT INTO evidence_issue_results
                        (invocation_id, assessment_row_id, issue_name, result,
                         result_sha256)
                    VALUES (%s, %s, 'detiding_quality', '{}', %s)
                    """,
                    (invocation, assessment_row_id, _hex64(0x3B2)),
                )
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(
                    """
                    INSERT INTO evidence_issue_results
                        (invocation_id, assessment_row_id, issue_name, result,
                         result_sha256)
                    VALUES (%s, %s, 'detiding_quality', '{}', 'nothex')
                    """,
                    (_hex64(0x3A2), assessment_row_id),
                )
        finally:
            conn.execute("RESET ROLE")

        for stmt in (
            "UPDATE evidence_issue_results SET issue_name = 'x' "
            "WHERE invocation_id = %s",
            "DELETE FROM evidence_issue_results WHERE invocation_id = %s",
        ):
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(stmt, (invocation,))


def test_escalation_packets_uniqueness_and_immutability(
    provisioned_test_database: str,
) -> None:
    """Migration 009 escalation_packets contract: one packet per
    (assessment, renderer version), immutable once written."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        event_id = uuid4()
        result = db.persist_assessment(
            checkpoint_id=_hex64(0x4A1),
            schema_version=1,
            event_id=event_id,
            producer_agent="pipeline_worker",
            handoff_id=uuid4(),
            trace_id=uuid4(),
            payload={},
            input_manifest_hash=_hex64(0x4B1),
            scientific_content_hash=_hex64(0x4C1),
        )
        assert result.status == "inserted"
        assert result.row is not None
        assessment_row_id = int(result.row["id"])
    finally:
        db.close()

    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
        autocommit=True,
    ) as conn:
        conn.execute(
            """
            INSERT INTO escalation_packets
                (assessment_row_id, event_id, renderer_version, packet,
                 content_sha256)
            VALUES (%s, %s, '1.0.0', '{}', %s)
            """,
            (assessment_row_id, str(event_id), _hex64(0x4D1)),
        )
        with pytest.raises(psycopg.errors.UniqueViolation):
            conn.execute(
                """
                INSERT INTO escalation_packets
                    (assessment_row_id, event_id, renderer_version, packet,
                     content_sha256)
                VALUES (%s, %s, '1.0.0', '{}', %s)
                """,
                (assessment_row_id, str(event_id), _hex64(0x4D2)),
            )
        for stmt in (
            "UPDATE escalation_packets SET renderer_version = 'x' "
            "WHERE assessment_row_id = %s",
            "DELETE FROM escalation_packets WHERE assessment_row_id = %s",
        ):
            with pytest.raises(psycopg.Error, match="append-only"):
                conn.execute(stmt, (assessment_row_id,))


def test_persist_seismic_revision_round_trip(
    provisioned_test_database: str,
) -> None:
    """Real-DB seismic revision identity persistence: the merge
    lands only on the active event's non-IDLE row and is a no-op after
    resolution or for a different event."""
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        event_id = uuid4()
        db.upsert_fsm_state(
            state="MONITOR",
            event_context={
                "event_id": str(event_id),
                "seismic_magnitude": 8.8,
                "seismic_region": "maule",
                "epicenter_lat": -35.85,
                "epicenter_lon": -72.72,
                "depth_km": 22.9,
                "trigger_time_utc": "2010-02-27T06:34:11+00:00",
                "latest_anomaly_score": 0.0,
                "dart_confirmation": False,
                "active_dart_stations": [],
                "stations_in_event_mode": [],
                "latest_revision_id": "seismic:us2010chile:1",
            },
        )
        revision = {
            "latest_revision_id": "seismic:us2010chile:2",
            "latest_revision_sha256": "b" * 64,
            "latest_revision_updated_utc": "2010-02-27T06:50:00+00:00",
        }
        assert db.persist_seismic_revision(event_id, revision) is True
        row = db.load_fsm_state()
        assert row is not None
        ctx = row["event_context"]
        assert ctx["latest_revision_id"] == "seismic:us2010chile:2"
        assert ctx["latest_revision_sha256"] == "b" * 64
        # Untouched fields survive the JSONB merge.
        assert ctx["seismic_magnitude"] == 8.8

        # Different event: no-op.
        assert db.persist_seismic_revision(uuid4(), revision) is False

        # Resolved (IDLE): no-op.
        db.upsert_fsm_state(state="IDLE", event_context=None)
        assert db.persist_seismic_revision(event_id, revision) is False
    finally:
        db.close()


def test_configure_limited_roles_password_sql_is_server_accepted() -> None:
    """configure_limited_roles emits ALTER ROLE ... PASSWORD as a quoted literal,
    not a bound parameter (PostgreSQL rejects "$1" there). Prove the emitted
    form is accepted by a real server. Uses a throwaway role so the dev DB's
    app-role passwords are never rewritten (which would break sibling tests).
    """
    from psycopg import sql as _sql

    role = f"tmp_prov_{uuid4().hex[:12]}"
    # A password exercising quote-escaping (single quote -> '').
    password = "pw-x'quote"
    try:
        with psycopg.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_ADMIN_NAME,
            user=DB_ADMIN_USER,
            password=DB_ADMIN_PASSWORD,
            connect_timeout=3,
            autocommit=True,
        ) as conn:
            conn.execute(
                _sql.SQL("CREATE ROLE {} WITH LOGIN").format(_sql.Identifier(role))
            )
            # Mirror the exact statements configure_limited_roles emits for a
            # non-empty password (provision.py): constraint ALTER, then the
            # password ALTER with sql.Literal. Server must accept both.
            conn.execute(
                _sql.SQL(
                    "ALTER ROLE {role} WITH LOGIN NOINHERIT "
                    "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
                ).format(role=_sql.Identifier(role))
            )
            conn.execute(
                _sql.SQL("ALTER ROLE {role} WITH PASSWORD {password}").format(
                    role=_sql.Identifier(role),
                    password=_sql.Literal(password),
                )
            )
            # The password actually round-trips: a login with it succeeds.
            with psycopg.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_ADMIN_NAME,
                user=role,
                password=password,
                connect_timeout=3,
            ) as role_conn:
                row = role_conn.execute("SELECT current_user").fetchone()
                assert row is not None and row[0] == role
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL is not reachable for integration tests: {exc}")
    finally:
        # Best effort. An unconditional reconnect here raises
        # OperationalError when the server is unreachable, and an exception
        # from a finally block replaces the pytest.skip raised above, so the
        # whole suite reported a failure instead of a skip on any machine
        # without PostgreSQL. If the server is unreachable there is also no
        # throwaway role to drop.
        try:
            with psycopg.connect(
                host=DB_HOST,
                port=DB_PORT,
                dbname=DB_ADMIN_NAME,
                user=DB_ADMIN_USER,
                password=DB_ADMIN_PASSWORD,
                connect_timeout=3,
                autocommit=True,
            ) as conn:
                conn.execute(
                    _sql.SQL("DROP ROLE IF EXISTS {}").format(_sql.Identifier(role))
                )
        except psycopg.OperationalError:
            pass


def test_assessment_identity_check_rejects_null_identity(
    provisioned_test_database: str,
) -> None:
    """Migration 010: the assessment identity CHECK must be NULL-safe.

    As written in 009 the constraint read `feature_type <> '...' OR (checkpoint_id
    ~ '...' AND ...)`. A CHECK passes when its expression is true OR NULL, so a
    NULL checkpoint_id made the regex NULL, the conjunction NULL, and the row was
    accepted. Such a row would also escape the partial unique index (NULLs are
    distinct) and the ON CONFLICT idempotency in persist_assessment, and the
    append-only triggers would then make it permanently undeletable.
    """
    from uuid import uuid4

    import psycopg

    good = _hex64(0xABC)
    dsn = (
        f"host={DB_HOST} port={DB_PORT} dbname={provisioned_test_database} "
        f"user={DB_ADMIN_USER} password={DB_ADMIN_PASSWORD} connect_timeout=3"
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        columns = (
            "(feature_type, event_id, producer_agent, handoff_id, trace_id, "
            "payload, checkpoint_id, assessment_schema_version, "
            "input_manifest_hash, scientific_content_hash)"
        )
        insert = (
            f"INSERT INTO processed_features {columns} "
            "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)"
        )

        def assessment_row(checkpoint, version, manifest, scientific):
            return (
                "ocean_evidence_assessment",
                str(uuid4()),
                "pipeline_worker",
                str(uuid4()),
                str(uuid4()),
                "{}",
                checkpoint,
                version,
                manifest,
                scientific,
            )

        # Every identity column NULL: must be refused by the named constraint.
        with pytest.raises(psycopg.errors.CheckViolation) as excinfo:
            conn.execute(insert, assessment_row(None, None, None, None))
        assert (
            excinfo.value.diag.constraint_name
            == "processed_features_assessment_identity_chk"
        )

        # One NULL among otherwise valid identity columns is still refused.
        for missing in range(4):
            values = [good, 1, good, good]
            values[missing] = None
            with pytest.raises(psycopg.errors.CheckViolation):
                conn.execute(insert, assessment_row(*values))

        # A fully populated assessment row is still accepted.
        conn.execute(insert, assessment_row(good, 1, good, good))

        # Non-assessment rows are unaffected: they legitimately carry NULLs.
        conn.execute(
            "INSERT INTO processed_features "
            "(feature_type, event_id, producer_agent, handoff_id, trace_id, payload) "
            "VALUES ('qc_report', %s, 'qc_agent', %s, %s, '{}'::jsonb)",
            (str(uuid4()), str(uuid4()), str(uuid4())),
        )


def test_query_path_indexes_serve_audit_and_packet_lookups(
    provisioned_test_database: str,
) -> None:
    """Migration 011: the unfiltered audit listing and the packet-of-record
    lookup have supporting indexes.

    The audit listing orders by (recorded_at DESC, id DESC) with no filter;
    before 011 no index led with recorded_at, so the planner sequential-scanned
    the append-only table and sorted on every unfiltered dashboard refresh.
    This asserts the indexes exist and that the planner actually chooses the
    audit index for the hot query once the table is large enough to matter.
    """
    from uuid import uuid4

    import psycopg

    dsn = (
        f"host={DB_HOST} port={DB_PORT} dbname={provisioned_test_database} "
        f"user={DB_ADMIN_USER} password={DB_ADMIN_PASSWORD} connect_timeout=3"
    )
    with psycopg.connect(dsn, autocommit=True) as conn:
        names = {
            r[0]
            for r in conn.execute(
                "SELECT indexname FROM pg_indexes "
                "WHERE tablename IN ('audit_events', 'escalation_packets')"
            ).fetchall()
        }
        assert "idx_audit_recorded_at" in names
        assert "idx_escalation_packets_event_id" in names

        # Seed enough rows that the planner prefers the index, then confirm
        # the unfiltered listing plan uses it (a seq scan + sort here is the
        # regression this migration exists to prevent).
        conn.execute(
            "INSERT INTO audit_events "
            "(event_id, agent_name, action, decision_basis, recorded_at) "
            "SELECT %s, 'seed', 'anomaly_scored', 'x', "
            "now() - (g || ' seconds')::interval "
            "FROM generate_series(1, 5000) g",
            (str(uuid4()),),
        )
        conn.execute("ANALYZE audit_events")
        plan = "\n".join(
            r[0]
            for r in conn.execute(
                "EXPLAIN SELECT * FROM audit_events "
                "ORDER BY recorded_at DESC, id DESC LIMIT 100"
            ).fetchall()
        )
        assert "idx_audit_recorded_at" in plan, plan


def test_agent_writer_cannot_write_processed_features(
    provisioned_test_database: str,
) -> None:
    """Migration 012: a role no application connects as must not be able to
    claim an ocean-evidence assessment key.

    processed_features holds the assessments, whose identity is
    (checkpoint_id, assessment_schema_version) and whose rows are append-only
    by trigger. With INSERT, agent_writer could squat a checkpoint key before
    the worker, which makes the genuine assessment persist as a conflict and
    leaves a row not even the table owner can delete or correct. The role also
    shares DB_PASSWORD with the ingest containers by default.

    The lineage privileges it does use are asserted here too, so a blanket
    revoke would fail this test rather than pass it.
    """
    with psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    ) as conn:
        row = conn.execute(
            """
            SELECT
                has_table_privilege('agent_writer', 'processed_features', 'INSERT'),
                has_sequence_privilege(
                    'agent_writer', 'processed_features_id_seq', 'USAGE'
                ),
                has_table_privilege('agent_writer', 'raw_observations', 'SELECT'),
                has_table_privilege('agent_writer', 'audit_events', 'INSERT'),
                has_table_privilege('pipeline_worker', 'processed_features', 'INSERT')
            """
        ).fetchone()
        assert row is not None
        pf_insert, seq_usage, raw_select, audit_insert, worker_insert = row

        assert pf_insert is False, "agent_writer must not insert assessments"
        assert seq_usage is False, "without INSERT the sequence grant is dead"
        assert raw_select is True, "lineage reads must survive the revoke"
        assert audit_insert is True, "audit append must survive the revoke"
        assert worker_insert is True, "the pipeline worker must still write"


def test_insert_evidence_issue_result_is_idempotent(
    provisioned_test_database: str,
) -> None:
    """DatabaseClient.insert_evidence_issue_result against the real table.

    The investigator derives a deterministic invocation ID, so re-running the
    same issue over the same assessment must be recognized as already recorded
    rather than raise or append a second opinion. Migration 009 makes these
    rows append-only, so "existing" is the only correct answer for a repeat.
    """
    from uuid import uuid4

    from hazard_assessment.storage.client import ClientConfig, DatabaseClient

    cfg = ClientConfig(
        host=DB_HOST,
        port=DB_PORT,
        dbname=provisioned_test_database,
        user=DB_ADMIN_USER,
        password=DB_ADMIN_PASSWORD,
        connect_timeout=3,
    )
    db = DatabaseClient(cfg)
    try:
        persisted = db.persist_assessment(
            checkpoint_id=_hex64(0x5A1),
            schema_version=1,
            event_id=uuid4(),
            producer_agent="pipeline_worker",
            handoff_id=uuid4(),
            trace_id=uuid4(),
            payload={},
            input_manifest_hash=_hex64(0x5B1),
            scientific_content_hash=_hex64(0x5C1),
        )
        assert persisted.status == "inserted"
        assert persisted.row is not None
        row_id = int(persisted.row["id"])

        from hazard_assessment.agents.llm_advisory.investigator import (
            compute_invocation_id,
        )

        invocation = compute_invocation_id(
            assessment_row_id=row_id,
            issue_name="station_agreement",
            model="test-model",
        )
        payload = {"issue_name": "station_agreement", "finding": "one station only"}

        first = db.insert_evidence_issue_result(
            invocation_id=invocation,
            assessment_row_id=row_id,
            issue_name="station_agreement",
            result=payload,
            result_sha256=_hex64(0x5D1),
        )
        assert first == "inserted"

        # The identical investigation again is recognized, not duplicated.
        assert db.insert_evidence_issue_result(
            invocation_id=invocation,
            assessment_row_id=row_id,
            issue_name="station_agreement",
            result=payload,
            result_sha256=_hex64(0x5D1),
        ) == "existing"

        # Same invocation id but a different result must surface as a conflict.
        # Migration 009 states the contract: "a repeated run with a different
        # result hash fails the unique constraint and is surfaced as a conflict
        # instead of silently coexisting." Returning "existing" here would hand
        # a caller one finding while the database kept another, which is exactly
        # what happens when a first run fails to converge, claims the identity,
        # and the real finding follows.
        assert db.insert_evidence_issue_result(
            invocation_id=invocation,
            assessment_row_id=row_id,
            issue_name="station_agreement",
            result={"issue_name": "station_agreement", "finding": "revised"},
            result_sha256=_hex64(0x5D2),
        ) == "conflict"

        # The conflict left the original on record rather than overwriting it.
        stored_first = db.get_evidence_issue_results(row_id)[0]
        assert stored_first["result_sha256"] == _hex64(0x5D1)
        assert stored_first["result"]["finding"] == "one station only"

        # A different issue over the same assessment is a distinct finding.
        other = db.insert_evidence_issue_result(
            invocation_id=compute_invocation_id(
                assessment_row_id=row_id,
                issue_name="evidence_gaps",
                model="test-model",
            ),
            assessment_row_id=row_id,
            issue_name="evidence_gaps",
            result={"issue_name": "evidence_gaps"},
            result_sha256=_hex64(0x5D3),
        )
        assert other == "inserted"

        stored = db.get_evidence_issue_results(row_id)
        assert [r["issue_name"] for r in stored] == [
            "station_agreement",
            "evidence_gaps",
        ]
        assert stored[0]["result"]["finding"] == "one station only"

        # A malformed digest must be refused by the column CHECK, surfaced as
        # "error" rather than an exception escaping the client.
        assert db.insert_evidence_issue_result(
            invocation_id=_hex64(0x5A9),
            assessment_row_id=row_id,
            issue_name="station_agreement",
            result={},
            result_sha256="not-a-digest",
        ) == "error"
    finally:
        db.close()
