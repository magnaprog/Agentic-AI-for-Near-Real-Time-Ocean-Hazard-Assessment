"""Unit tests for E8 trace propagation and lineage API.

Covers:
- B1: trace_id on BaseEnvelope and AuditEntry
- B2: trace_id propagation through PipelineState and nodes
- B3: OTel tracing wrapper (no-op span)
- B4: Lineage API endpoints
- B5: SQL migration file existence
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from hazard_assessment.audit.logger import AuditEntry, AuditLogger
from hazard_assessment.orchestrator.nodes import _parse_trace_id, run_pipeline_sync
from hazard_assessment.orchestrator.pipeline import PipelineState
from hazard_assessment.orchestrator.states import FSMOrchestrator, SystemState
from hazard_assessment.schemas.envelope import BaseEnvelope
from tests._review_support import install_durable_review_packet

# ============================================================
# B1: trace_id on BaseEnvelope and AuditEntry
# ============================================================


class TestBaseEnvelopeTraceId:
    def test_trace_id_defaults_to_none(self) -> None:
        env = BaseEnvelope(producer="test")
        assert env.trace_id is None

    def test_trace_id_accepts_uuid(self) -> None:
        tid = uuid4()
        env = BaseEnvelope(producer="test", trace_id=tid)
        assert env.trace_id == tid

    def test_trace_id_round_trips_through_json(self) -> None:
        tid = uuid4()
        env = BaseEnvelope(producer="test", trace_id=tid)
        dumped = env.model_dump(mode="json")
        restored = BaseEnvelope.model_validate(dumped)
        assert restored.trace_id == tid

    def test_trace_id_absent_from_json_is_none(self) -> None:
        data = {"producer": "test", "schema_version": "1.0"}
        env = BaseEnvelope.model_validate(data)
        assert env.trace_id is None


class TestAuditEntryTraceId:
    def test_trace_id_defaults_to_none(self) -> None:
        entry = AuditEntry(event_type="test", producer="test")
        assert entry.trace_id is None

    def test_trace_id_accepts_uuid(self) -> None:
        tid = uuid4()
        entry = AuditEntry(event_type="test", producer="test", trace_id=tid)
        assert entry.trace_id == tid

    def test_frozen_prevents_mutation(self) -> None:
        tid = uuid4()
        entry = AuditEntry(event_type="test", producer="test", trace_id=tid)
        with pytest.raises(ValidationError):
            entry.trace_id = uuid4()  # type: ignore[misc]

    def test_extra_forbid_still_enforced(self) -> None:
        with pytest.raises(ValidationError):
            AuditEntry(
                event_type="test",
                producer="test",
                trace_id=uuid4(),
                unexpected_field="bad",  # type: ignore[call-arg]
            )


class TestAuditLoggerTraceFilter:
    def test_filter_by_trace_id(self) -> None:
        logger = AuditLogger()
        tid1 = uuid4()
        tid2 = uuid4()

        logger.append(AuditEntry(event_type="a", producer="x", trace_id=tid1))
        logger.append(AuditEntry(event_type="b", producer="x", trace_id=tid2))
        logger.append(AuditEntry(event_type="c", producer="x", trace_id=tid1))

        results = logger.get_entries(trace_id=tid1)
        assert len(results) == 2
        assert all(e.trace_id == tid1 for e in results)

    def test_filter_by_trace_and_event_type(self) -> None:
        logger = AuditLogger()
        tid = uuid4()

        logger.append(AuditEntry(event_type="a", producer="x", trace_id=tid))
        logger.append(AuditEntry(event_type="b", producer="x", trace_id=tid))

        results = logger.get_entries(trace_id=tid, event_type="a")
        assert len(results) == 1
        assert results[0].event_type == "a"


# ============================================================
# B2: trace_id in PipelineState and node propagation
# ============================================================


class TestParseTraceId:
    def test_returns_none_for_missing(self) -> None:
        state: PipelineState = {}
        assert _parse_trace_id(state) is None

    def test_returns_none_for_empty_string(self) -> None:
        state: PipelineState = {"trace_id": ""}
        assert _parse_trace_id(state) is None

    def test_parses_valid_uuid_string(self) -> None:
        tid = uuid4()
        state: PipelineState = {"trace_id": str(tid)}
        assert _parse_trace_id(state) == tid

    def test_returns_none_for_invalid_string(self) -> None:
        state: PipelineState = {"trace_id": "not-a-uuid"}
        assert _parse_trace_id(state) is None


class TestPipelineSyncTraceId:
    def test_generates_trace_id_when_absent(self) -> None:
        audit = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=audit)
        state: PipelineState = {
            "anomaly_assessment": {"anomaly_score": 0.1},
        }
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)
        assert "trace_id" in result
        # Must be a valid UUID
        UUID(result["trace_id"])

    def test_preserves_caller_trace_id(self) -> None:
        audit = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=audit)
        tid = str(uuid4())
        state: PipelineState = {
            "trace_id": tid,
            "anomaly_assessment": {"anomaly_score": 0.1},
        }
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)
        assert result["trace_id"] == tid

    def test_audit_entries_carry_trace_id(self) -> None:
        """When a transition occurs, the audit entry must carry the trace_id."""
        audit = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=audit)
        tid = str(uuid4())

        # Trigger a seismic event so MONITOR->INVESTIGATE transition can fire
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="test",
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            tsunamigenic_zones={"test"},
        )

        state: PipelineState = {
            "trace_id": tid,
            "anomaly_assessment": {"anomaly_score": 0.36},
        }
        run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

        trace_entries = audit.get_entries(trace_id=UUID(tid))
        assert len(trace_entries) == 1  # Exactly one MONITOR->INVESTIGATE transition
        assert trace_entries[0].event_type == "state_transition"
        assert trace_entries[0].data["from_state"] == "MONITOR"
        assert trace_entries[0].data["to_state"] == "INVESTIGATE"
        assert trace_entries[0].trace_id == UUID(tid)


# ============================================================
# B3: OTel tracing wrapper
# ============================================================


class TestOTelTracing:
    def test_pipeline_span_no_op(self) -> None:
        """pipeline_span works without a configured exporter (no-op spans)."""
        from hazard_assessment.telemetry.tracing import pipeline_span

        tid = str(uuid4())
        with pipeline_span(tid) as span:
            # No-op span still returns a span object
            assert span is not None

    def test_configure_tracer_provider_noop(self) -> None:
        """configure_tracer_provider(None) does nothing (no crash)."""
        from hazard_assessment.telemetry.tracing import configure_tracer_provider

        configure_tracer_provider(None)  # Should not raise


# ============================================================
# B4: Lineage API endpoints
# ============================================================


AUTH = {"X-Hazard-Api-Key": "test-key"}


@contextmanager
def _app_client(
    api_key: str = "test-key",
) -> Generator[tuple[Any, TestClient], None, None]:
    """Context manager that starts the app with the given API key."""
    import hazard_assessment.app as app_module
    from hazard_assessment.audit.logger import AuditLogger
    from hazard_assessment.orchestrator.states import FSMOrchestrator

    saved_env = {
        k: os.environ.get(k) for k in ("HAZARD_API_KEY", "APP_ENVIRONMENT")
    }
    os.environ["HAZARD_API_KEY"] = api_key
    os.environ["APP_ENVIRONMENT"] = "development"

    fresh_audit = AuditLogger()
    fresh_fsm = FSMOrchestrator(audit_writer=fresh_audit)
    old_audit, old_fsm = app_module._audit, app_module._fsm
    old_api_key = app_module._HAZARD_API_KEY
    old_packet = app_module._active_escalation_packet
    old_db_client = app_module._db_client
    app_module._audit = fresh_audit
    app_module._fsm = fresh_fsm
    app_module._active_escalation_packet = None
    app_module._db_client = None

    try:
        with TestClient(app_module.app) as client:
            yield app_module, client
    finally:
        app_module._audit = old_audit
        app_module._fsm = old_fsm
        app_module._HAZARD_API_KEY = old_api_key
        app_module._active_escalation_packet = old_packet
        app_module._db_client = old_db_client
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class TestAuditEndpointTraceFilter:
    def test_audit_returns_trace_id_field(self) -> None:
        with _app_client() as (app_module, client):
            tid = uuid4()
            app_module._audit.append(AuditEntry(
                event_type="test", producer="unit", trace_id=tid,
            ))
            resp = client.get("/api/audit", headers=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["trace_id"] == str(tid)

    def test_audit_filter_by_trace_id(self) -> None:
        with _app_client() as (app_module, client):
            tid = uuid4()
            other = uuid4()
            app_module._audit.append(AuditEntry(
                event_type="a", producer="x", trace_id=tid,
            ))
            app_module._audit.append(AuditEntry(
                event_type="b", producer="x", trace_id=other,
            ))
            resp = client.get(
                "/api/audit", headers=AUTH, params={"trace_id": str(tid)}
            )
            assert resp.status_code == 200
            data = resp.json()
            assert len(data) == 1
            assert data[0]["event_type"] == "a"

    def test_audit_rejects_invalid_trace_uuid(self) -> None:
        with _app_client() as (_, client):
            resp = client.get(
                "/api/audit", headers=AUTH, params={"trace_id": "not-valid"}
            )
            assert resp.status_code == 400


class TestLineageByTrace:
    def test_returns_entries_for_trace(self) -> None:
        with _app_client() as (app_module, client):
            tid = uuid4()
            app_module._audit.append(AuditEntry(
                event_type="a", producer="x", trace_id=tid,
            ))
            app_module._audit.append(AuditEntry(
                event_type="b", producer="y", trace_id=tid,
            ))
            resp = client.get(f"/api/lineage/{tid}", headers=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert data["trace_id"] == str(tid)
            assert data["entry_count"] == 2
            assert len(data["entries"]) == 2
            # Entries include trace_id for consistent shape across endpoints
            assert all(e["trace_id"] == str(tid) for e in data["entries"])

    def test_404_for_unknown_trace(self) -> None:
        with _app_client() as (_, client):
            resp = client.get(f"/api/lineage/{uuid4()}", headers=AUTH)
            assert resp.status_code == 404

    def test_400_for_invalid_uuid(self) -> None:
        with _app_client() as (_, client):
            resp = client.get("/api/lineage/not-a-uuid", headers=AUTH)
            assert resp.status_code == 400

    def test_requires_auth(self) -> None:
        with _app_client() as (_, client):
            resp = client.get(f"/api/lineage/{uuid4()}")
            assert resp.status_code == 401


class TestLineageByEvent:
    def test_groups_by_trace_id(self) -> None:
        with _app_client() as (app_module, client):
            eid = uuid4()
            tid1 = uuid4()
            tid2 = uuid4()
            app_module._audit.append(AuditEntry(
                event_id=eid, event_type="a", producer="x", trace_id=tid1,
            ))
            app_module._audit.append(AuditEntry(
                event_id=eid, event_type="b", producer="y", trace_id=tid2,
            ))
            resp = client.get(f"/api/lineage/event/{eid}", headers=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert data["event_id"] == str(eid)
            assert data["total_entries"] == 2
            assert str(tid1) in data["traces"]
            assert str(tid2) in data["traces"]
            # Entries include trace_id for consistent shape across endpoints
            entry1 = data["traces"][str(tid1)][0]
            assert entry1["trace_id"] == str(tid1)

    def test_404_for_unknown_event(self) -> None:
        with _app_client() as (_, client):
            resp = client.get(f"/api/lineage/event/{uuid4()}", headers=AUTH)
            assert resp.status_code == 404

    def test_400_for_invalid_uuid(self) -> None:
        with _app_client() as (_, client):
            resp = client.get("/api/lineage/event/not-a-uuid", headers=AUTH)
            assert resp.status_code == 400

    def test_entries_without_trace_grouped_under_no_trace(self) -> None:
        with _app_client() as (app_module, client):
            eid = uuid4()
            app_module._audit.append(AuditEntry(
                event_id=eid, event_type="legacy", producer="old",
            ))
            resp = client.get(f"/api/lineage/event/{eid}", headers=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            assert "__no_trace__" in data["traces"]
            orphans = data["traces"]["__no_trace__"]
            assert len(orphans) == 1
            assert orphans[0]["event_type"] == "legacy"
            assert orphans[0]["producer"] == "old"


# ============================================================
# B5: SQL migration file existence
# ============================================================


class TestSQLMigration:
    def test_002_migration_exists(self) -> None:
        migration = Path(
            "src/hazard_assessment/storage/migrations/002_trace_lineage.sql"
        )
        assert migration.exists(), "002_trace_lineage.sql migration not found"

    def test_002_migration_contains_trace_id_column(self) -> None:
        migration = Path(
            "src/hazard_assessment/storage/migrations/002_trace_lineage.sql"
        )
        content = migration.read_text()
        assert "trace_id UUID" in content
        assert "idx_audit_trace_id" in content
        assert "idx_audit_input_hashes_gin" in content
        assert "provenance_chain" in content

    def test_002_migration_view_has_order_by(self) -> None:
        migration = Path(
            "src/hazard_assessment/storage/migrations/002_trace_lineage.sql"
        )
        content = migration.read_text()
        assert "ORDER BY" in content, "provenance_chain view must have ORDER BY for DISTINCT ON"


# ============================================================
# Review fix tests: trace_id in escalation + human decision + resolve
# ============================================================


class TestEscalationPacketTraceId:
    """Escalation packet AuditEntry must carry trace_id from transition."""

    def test_escalation_audit_entry_carries_trace_id(self) -> None:
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.orchestrator.states import (
            EventContext,
            TransitionRecord,
        )

        audit = AuditLogger()
        tid = uuid4()
        ctx = EventContext(
            seismic_magnitude=7.5,
            seismic_region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
        )
        transition = TransitionRecord(
            event_id=ctx.event_id,
            trace_id=tid,
            from_state=SystemState.ASSESS,
            to_state=SystemState.ESCALATE,
            trigger_reason="anomaly_score >= T3",
            anomaly_score=0.90,
            seismic_magnitude=7.5,
        )

        generate_escalation_packet(
            ctx=ctx,
            transition=transition,
            audit_logger=audit,
        )

        entries = audit.get_entries(event_type="escalation_packet_generated")
        assert len(entries) == 1
        assert entries[0].trace_id == tid

    def test_escalation_audit_entry_none_trace_when_no_transition(self) -> None:
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=7.5,
            seismic_region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
        )

        generate_escalation_packet(
            ctx=ctx,
            transition=None,
            audit_logger=audit,
        )

        entries = audit.get_entries(event_type="escalation_packet_generated")
        assert len(entries) == 1
        assert entries[0].trace_id is None

    def test_packet_assembles_input_refs_from_provenance(self) -> None:
        """generate_escalation_packet builds real input_refs from the
        input_provenance audit entries recorded for the event (deduped by
        hash)."""
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditEntry
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=9.1,
            seismic_region="japan_trench",
            epicenter_lat=38.3,
            epicenter_lon=142.37,
        )
        sha = "b" * 64
        # Two entries with the same hash must dedupe to one ref.
        for _ in range(2):
            audit.append(
                AuditEntry(
                    event_id=ctx.event_id,
                    event_type="input_provenance",
                    producer="ingest_seismic",
                    data={"source": "seismic", "record_id": "us-tohoku", "sha256": sha},
                )
            )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        assert len(packet.input_refs) == 1
        assert packet.input_refs[0].sha256 == sha
        assert packet.input_refs[0].record_id == "us-tohoku"
        assert packet.input_refs[0].source.value == "seismic"

    def test_packet_input_refs_empty_without_provenance(self) -> None:
        """No provenance entries -> empty input_refs (current observation-only
        escalation behavior), not an error."""
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=7.5,
            seismic_region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
        )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        assert packet.input_refs == []

    def test_worker_driven_packet_recovers_trigger_and_trace_from_audit(self) -> None:
        """When the API has no in-memory transition (worker-driven ESCALATE), the
        packet recovers the trigger reason and trace_id from the durable
        state_transition audit entry instead of trigger='unknown' / trace=None."""
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditEntry
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        tid = uuid4()
        ctx = EventContext(
            seismic_magnitude=9.1,
            seismic_region="japan_trench",
            epicenter_lat=38.3,
            epicenter_lon=142.37,
        )
        audit.append(
            AuditEntry(
                event_id=ctx.event_id,
                trace_id=tid,
                event_type="state_transition",
                producer="orchestrator",
                data={
                    "from_state": "MONITOR",
                    "to_state": "ESCALATE",
                    "trigger_reason": "Seismic-only escalation: M9.1",
                    "anomaly_score": 0.0,
                },
            )
        )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        assert packet.trace_id == tid
        assert "Seismic-only escalation" in packet.escalation_trigger

    def test_packet_anomaly_timeline_is_chronological(self) -> None:
        """The packet anomaly_timeline must be oldest-first even though
        query_entries returns newest-first."""
        from datetime import UTC, datetime, timedelta

        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditEntry
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=8.0,
            seismic_region="x",
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        base = datetime(2011, 3, 11, 6, 0, tzinfo=UTC)
        for i in (2, 0, 1):  # appended out of order
            audit.append(
                AuditEntry(
                    event_id=ctx.event_id,
                    event_type="state_transition",
                    producer="orchestrator",
                    timestamp_utc=base + timedelta(minutes=i),
                    data={
                        "from_state": "MONITOR",
                        "to_state": "INVESTIGATE",
                        "anomaly_score": float(i),
                    },
                )
            )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        ts = [e["timestamp_utc"] for e in packet.anomaly_timeline]
        assert ts == sorted(ts)

    def test_packet_marks_input_refs_truncated_when_worker_capped(self) -> None:
        """The packet discloses input_refs_truncated when the worker recorded a
        provenance_capped marker (it dropped observation provenance beyond its
        per-event cap), so a reviewer knows the list is incomplete."""
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditEntry
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=8.0,
            seismic_region="x",
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        audit.append(
            AuditEntry(
                event_id=ctx.event_id,
                event_type="input_provenance",
                producer="ingest_dart",
                data={"source": "dart", "record_id": "21418", "sha256": "a" * 64},
            )
        )
        audit.append(
            AuditEntry(
                event_id=ctx.event_id,
                event_type="provenance_capped",
                producer="ingest_dart",
                data={"cap": 1000},
            )
        )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        assert packet.input_refs_truncated is True
        assert len(packet.input_refs) == 1

    def test_seismic_provenance_always_included(self) -> None:
        """The seismic trigger provenance (distinct event_type) is included in the
        packet input_refs, independent of the capped observation read."""
        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditEntry
        from hazard_assessment.orchestrator.states import EventContext

        audit = AuditLogger()
        ctx = EventContext(
            seismic_magnitude=9.1,
            seismic_region="japan",
            epicenter_lat=38.3,
            epicenter_lon=142.37,
        )
        seismic_sha = "b" * 64
        audit.append(
            AuditEntry(
                event_id=ctx.event_id,
                event_type="seismic_provenance",
                producer="ingest_seismic",
                data={
                    "source": "seismic",
                    "record_id": "us-tohoku",
                    "sha256": seismic_sha,
                },
            )
        )

        packet = generate_escalation_packet(
            ctx=ctx, transition=None, audit_logger=audit
        )

        assert any(r.sha256 == seismic_sha for r in packet.input_refs)
        assert packet.input_refs_truncated is False


class TestHumanDecisionTraceId:
    """Human decision AuditEntry must carry trace_id when provided."""

    def test_review_with_trace_id(self) -> None:
        with _app_client() as (app_module, client):
            # Set up FSM in ESCALATE with an active escalation packet
            fsm = app_module._fsm
            fsm.evaluate_seismic_trigger(
                magnitude=7.5, region="pacific_nw",
                epicenter_lat=46.0, epicenter_lon=-130.0,
                tsunamigenic_zones={"pacific_nw"},
            )
            fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
            fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
            fsm.evaluate_anomaly_score(0.90)  # -> ESCALATE

            event_id = str(fsm.event_context.event_id)
            packet_fields, _ = install_durable_review_packet(
                app_module, event_id
            )
            tid = str(uuid4())

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "Confirmed by duty scientist",
                    **packet_fields,
                    "trace_id": tid,
                },
            )
            assert resp.status_code == 200

            entries = app_module._audit.get_entries(
                event_type="assessment_review_decision", trace_id=UUID(tid),
            )
            assert len(entries) == 1
            assert entries[0].trace_id == UUID(tid)

    def test_review_without_trace_id(self) -> None:
        with _app_client() as (app_module, client):
            fsm = app_module._fsm
            fsm.evaluate_seismic_trigger(
                magnitude=7.5, region="pacific_nw",
                epicenter_lat=46.0, epicenter_lon=-130.0,
                tsunamigenic_zones={"pacific_nw"},
            )
            fsm.evaluate_anomaly_score(0.40)
            fsm.evaluate_anomaly_score(0.65)
            fsm.evaluate_anomaly_score(0.90)

            event_id = str(fsm.event_context.event_id)
            packet_fields, _ = install_durable_review_packet(
                app_module, event_id
            )

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "DEFER",
                    "decision_reason": "Need more data",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200

            entries = app_module._audit.get_entries(
                event_type="assessment_review_decision"
            )
            assert len(entries) == 1
            assert entries[0].trace_id is None

    def test_review_invalid_trace_id_returns_400(self) -> None:
        with _app_client() as (_, client):
            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": str(uuid4()),
                    "decision": "APPROVE",
                    "decision_reason": "Test",
                    "escalation_packet_row_id": 1,
                    "escalation_packet_hash": "a" * 64,
                    "trace_id": "not-a-uuid",
                },
            )
            assert resp.status_code == 400
            assert "trace_id" in resp.json()["detail"]


class TestResolveEventTraceId:
    """resolve_event() must propagate trace_id to TransitionRecord."""

    def test_resolve_event_carries_trace_id(self) -> None:
        audit = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=audit)
        tid = uuid4()

        fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
        )
        fsm.evaluate_anomaly_score(0.40)
        fsm.evaluate_anomaly_score(0.65)
        fsm.evaluate_anomaly_score(0.90)

        record = fsm.resolve_event(trace_id=tid)
        assert record is not None
        assert record.trace_id == tid

        # The transition audit entry should also carry trace_id
        entries = audit.get_entries(trace_id=tid)
        assert len(entries) >= 1
        resolve_entries = [
            e for e in entries
            if e.data.get("to_state") == "IDLE"
            and e.data.get("from_state") == "ESCALATE"
        ]
        assert len(resolve_entries) == 1

    def test_resolve_event_without_trace_id(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
        )
        fsm.evaluate_anomaly_score(0.40)
        fsm.evaluate_anomaly_score(0.65)
        fsm.evaluate_anomaly_score(0.90)

        record = fsm.resolve_event()
        assert record is not None
        assert record.trace_id is None


class TestEventLineageResponseShape:
    """Event lineage entries must include event_id for consistency."""

    def test_event_lineage_entries_include_event_id(self) -> None:
        with _app_client() as (app_module, client):
            eid = uuid4()
            tid = uuid4()
            app_module._audit.append(AuditEntry(
                event_id=eid, trace_id=tid,
                event_type="test", producer="unit",
            ))
            resp = client.get(f"/api/lineage/event/{eid}", headers=AUTH)
            assert resp.status_code == 200
            data = resp.json()
            entry = data["traces"][str(tid)][0]
            assert "event_id" in entry
            assert entry["event_id"] == str(eid)
