"""Unit tests for core API authentication and review-gate behavior.

Review endpoint binds decisions to durable packet row identity and hash.
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tests._review_support import DurableReviewDb, install_durable_review_packet


@contextmanager
def _app_client(
    api_key: str = "test-key",
) -> Generator[tuple[Any, TestClient], None, None]:
    """Context manager that starts the app with the given API key.

    Uses the lifespan-aware TestClient so startup validation runs correctly.
    Resets module-level FSM/audit/escalation singletons on entry and restores
    env vars on exit to ensure test isolation.
    """
    import hazard_assessment.app as app_module
    from hazard_assessment.audit.logger import AuditLogger
    from hazard_assessment.orchestrator.states import FSMOrchestrator

    saved_env = {
        k: os.environ.get(k) for k in ("HAZARD_API_KEY", "APP_ENVIRONMENT")
    }
    os.environ["HAZARD_API_KEY"] = api_key
    os.environ["APP_ENVIRONMENT"] = "development"

    # Fresh singletons per test to prevent state leaking between tests.
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


AUTH = {"X-Hazard-Api-Key": "test-key"}


def _escalate_and_install_packet(
    app_module: Any, _client: TestClient
) -> tuple[str, dict[str, Any]]:
    """Drive FSM to ESCALATE and install its durable packet of record."""
    app_module._fsm.evaluate_seismic_trigger(
        magnitude=7.2,
        region="test-zone",
        epicenter_lat=0.0,
        epicenter_lon=0.0,
        tsunamigenic_zones={"test-zone"},
    )
    app_module._fsm.evaluate_anomaly_score(0.36)  # MONITOR -> INVESTIGATE
    app_module._fsm.evaluate_anomaly_score(0.61)  # INVESTIGATE -> ASSESS
    app_module._fsm.evaluate_anomaly_score(0.86)  # ASSESS -> ESCALATE

    event_id = str(app_module._fsm.event_context.event_id)
    request_fields, _ = install_durable_review_packet(app_module, event_id)
    return event_id, request_fields


def test_app_startup_fails_without_hazard_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HAZARD_API_KEY", raising=False)
    monkeypatch.setenv("APP_ENVIRONMENT", "development")

    import hazard_assessment.app as app_module

    with pytest.raises(RuntimeError, match="HAZARD_API_KEY is required"):
        with TestClient(app_module.app):
            pass


def test_app_startup_succeeds_in_production_without_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Environment label does not block process-local development mode.

    This proves startup only, not coherent live multi-process operation.
    """
    monkeypatch.setenv("HAZARD_API_KEY", "test-key")
    monkeypatch.setenv("APP_ENVIRONMENT", "production")
    monkeypatch.delenv("DB_HOST", raising=False)

    import hazard_assessment.app as app_module

    with TestClient(app_module.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 200


def test_health_is_public() -> None:
    with _app_client() as (_, client):
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json()["status"] == "healthy"


def test_api_fsm_reports_configured_thresholds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("THRESHOLD_T1", "0.21")
    monkeypatch.setenv("THRESHOLD_T2", "0.51")
    monkeypatch.setenv("THRESHOLD_T3", "0.81")

    with _app_client() as (_, client):
        response = client.get("/api/fsm", headers=AUTH)

        assert response.status_code == 200
        assert response.json()["thresholds"] == {
            "basin": "pacific",
            "t1": 0.21,
            "t2": 0.51,
            "t3": 0.81,
        }


def test_status_requires_api_key() -> None:
    with _app_client() as (_, client):
        response = client.get("/status")

        assert response.status_code == 401


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_api_schema_documentation_is_disabled(path: str) -> None:
    with _app_client() as (_, client):
        assert client.get(path).status_code == 404


def test_agents_reports_execution_paths_not_runtime_status() -> None:
    with _app_client() as (_, client):
        response = client.get("/api/agents", headers=AUTH)

        assert response.status_code == 200
        agents = response.json()
        assert {agent["execution_path"] for agent in agents} == {
            "LIVE_WORKER",
            "OFFLINE_EVALUATION_ONLY",
        }
        assert all("status" not in agent for agent in agents)


def test_api_fsm_accepts_valid_api_key() -> None:
    with _app_client() as (_, client):
        response = client.get("/api/fsm", headers=AUTH)

        assert response.status_code == 200
        payload = response.json()
        assert "fsm_state" in payload
        assert "transition_history" in payload
        assert payload["recovery_failed"] is False


def test_status_surfaces_recovery_failed_flag() -> None:
    """A destructive recovery fallback must be operator-visible, not just
    a log line: /status and /api/fsm expose recovery_failed."""
    with _app_client() as (_, client):
        response = client.get("/status", headers=AUTH)
        assert response.status_code == 200
        assert response.json()["recovery_failed"] is False


def test_provenance_endpoint_requires_database() -> None:
    """The SQL get_provenance() lineage path has no in-memory fallback."""
    from uuid import uuid4

    with _app_client() as (_, client):
        response = client.get(f"/api/lineage/provenance/{uuid4()}", headers=AUTH)
        assert response.status_code == 503


def test_provenance_endpoint_serializes_rows() -> None:
    """With a database configured, the endpoint returns the get_provenance()
    rows JSON-safe (UUIDs and timestamps stringified)."""
    from datetime import UTC, datetime
    from uuid import uuid4

    trace = uuid4()

    class _StubDb:
        is_connected = True

        def query_lineage(self, trace_id: Any) -> list[dict[str, Any]]:
            assert str(trace_id) == str(trace)
            return [
                {
                    "feature_type": "anomaly_score",
                    "handoff_id": uuid4(),
                    "produced_at": datetime.now(UTC),
                    "raw_payload_hash": "a" * 64,
                },
            ]

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        try:
            response = client.get(f"/api/lineage/provenance/{trace}", headers=AUTH)
            assert response.status_code == 200
            payload = response.json()
            assert payload["row_count"] == 1
            row = payload["rows"][0]
            assert row["feature_type"] == "anomaly_score"
            assert row["raw_payload_hash"] == "a" * 64
            assert isinstance(row["handoff_id"], str)

            missing = client.get("/api/lineage/provenance/not-a-uuid", headers=AUTH)
            assert missing.status_code == 400
        finally:
            app_module._db_client = None


def test_provenance_endpoint_distinguishes_failure_from_empty() -> None:
    """A failed lineage query (permission/function/connection error) is 503,
    not a misleading 404; an empty successful query is 404."""
    from uuid import uuid4

    class _FailingDb:
        is_connected = True

        def query_lineage(self, _trace_id: Any) -> None:
            return None  # query failure contract

    class _EmptyDb:
        is_connected = True

        def query_lineage(self, _trace_id: Any) -> list[dict[str, Any]]:
            return []

    with _app_client() as (app_module, client):
        try:
            app_module._db_client = _FailingDb()
            assert (
                client.get(f"/api/lineage/provenance/{uuid4()}", headers=AUTH).status_code
                == 503
            )
            app_module._db_client = _EmptyDb()
            assert (
                client.get(f"/api/lineage/provenance/{uuid4()}", headers=AUTH).status_code
                == 404
            )
        finally:
            app_module._db_client = None


def test_api_review_requires_caller_asserted_reviewer_identity() -> None:
    with _app_client() as (app_module, client):
        event_id, packet_fields = _escalate_and_install_packet(app_module, client)

        response = client.post(
            "/api/review",
            headers=AUTH,
            json={
                "event_id": event_id,
                "decision": "REJECT",
                "decision_reason": "insufficient evidence",
                **packet_fields,
            },
        )

        assert response.status_code == 400
        assert "X-Reviewer-Id" in response.json()["detail"]


def test_api_review_bounds_the_caller_asserted_identity() -> None:
    """The identity is written to the append-only audit trail as producer.

    It must be bounded and filtered: unbounded, the field accepts a
    5000-character value, and unfiltered it accepts an ANSI escape sequence,
    which renders as control codes in any terminal reading the record back. A
    newline is already refused by the HTTP layer; an escape sequence is not.
    """
    with _app_client() as (app_module, client):
        event_id, packet_fields = _escalate_and_install_packet(app_module, client)
        body = {
            "event_id": event_id,
            "decision": "REJECT",
            "decision_reason": "insufficient evidence",
            **packet_fields,
        }

        too_long = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "a" * 5000},
            json=body,
        )
        assert too_long.status_code == 400
        assert "at most" in too_long.json()["detail"]

        control = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "duty\x1b[31mscientist"},
            json=body,
        )
        assert control.status_code == 400
        assert "control characters" in control.json()["detail"]

        # An ordinary identity at the bound is still accepted.
        ok = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "d" * 128},
            json=body,
        )
        assert ok.status_code != 400, ok.json()


def test_api_review_validates_uuid_after_auth() -> None:
    with _app_client() as (_, client):
        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": "not-a-uuid",
                "decision": "APPROVE",
                "decision_reason": "ok",
                "escalation_packet_row_id": 1,
                "escalation_packet_hash": "a" * 64,
            },
        )

        assert response.status_code == 400
        assert "Invalid UUID" in response.json()["detail"]


def test_api_review_rejects_non_active_event_for_all_decisions() -> None:
    with _app_client() as (_, client):
        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": "11111111-1111-1111-1111-111111111111",
                "decision": "REJECT",
                "decision_reason": "not in escalation",
                "escalation_packet_row_id": 1,
                "escalation_packet_hash": "a" * 64,
            },
        )

        assert response.status_code == 409
        assert "active escalation" in response.json()["detail"]


def test_api_review_non_active_event_does_not_write_audit_entry() -> None:
    with _app_client() as (_, client):
        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": "11111111-1111-1111-1111-111111111111",
                "decision": "DEFER",
                "decision_reason": "not active",
                "escalation_packet_row_id": 1,
                "escalation_packet_hash": "a" * 64,
            },
        )
        assert response.status_code == 409

        audit_resp = client.get("/api/audit", headers=AUTH)
        assert audit_resp.status_code == 200
        assert audit_resp.json() == []


def test_api_review_records_reject_for_active_escalated_event() -> None:
    with _app_client() as (app_module, client):
        event_id, packet_fields = _escalate_and_install_packet(app_module, client)

        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": event_id,
                "decision": "REJECT",
                "decision_reason": "needs more evidence",
                **packet_fields,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"
        assert response.json()["decision"] == "REJECT"

        audit_resp = client.get("/api/audit", headers=AUTH)
        assert audit_resp.status_code == 200
        # Find the assessment_review_decision entry (skip escalation_packet_generated entries)
        human_entries = [
            e for e in audit_resp.json() if e["event_type"] == "assessment_review_decision"
        ]
        assert len(human_entries) >= 1
        assert human_entries[0]["producer"] == "operator-1"


def test_api_review_accepts_equivalent_uuid_case_for_active_event() -> None:
    with _app_client() as (app_module, client):
        event_id, packet_fields = _escalate_and_install_packet(app_module, client)
        # Use uppercased UUID to test case-insensitivity
        event_id_upper = event_id.upper()

        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": event_id_upper,
                "decision": "DEFER",
                "decision_reason": "waiting on additional inputs",
                **packet_fields,
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"
        assert response.json()["decision"] == "DEFER"


def test_api_review_fails_when_durable_audit_commit_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _app_client() as (app_module, client):
        event_id, packet_fields = _escalate_and_install_packet(app_module, client)
        db = app_module._db_client
        monkeypatch.setattr(db, "append_audit", lambda _entry: False)

        response = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "operator-1"},
            json={
                "event_id": event_id,
                "decision": "DEFER",
                "decision_reason": "Durability failure test",
                **packet_fields,
            },
        )

        assert response.status_code == 503
        assert "durable audit" in response.json()["detail"].lower()


def test_api_fsm_reflects_db_state_read_through() -> None:
    """With a database configured, /api/fsm reloads state from the DB so it
    reflects transitions made by other processes (e.g. the pipeline worker)
    instead of serving a stale startup snapshot.
    """
    from uuid import uuid4

    from hazard_assessment.audit.logger import AuditEntry, AuditLogger
    from hazard_assessment.orchestrator.states import FSMOrchestrator

    event_id = uuid4()

    class _StubDb(DurableReviewDb):
        def load_fsm_state(self) -> dict[str, Any]:
            # A valid worker-written row: non-IDLE rows always carry a
            # context (a null context is now treated as corrupt and falls
            # back to IDLE; see test_recover_from_db_fails_on_active_state_
            # without_context in test_fsm.py).
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": {
                    "event_id": str(event_id),
                    "seismic_magnitude": 9.1,
                    "seismic_region": "japan",
                    "epicenter_lat": 38.3,
                    "epicenter_lon": 142.37,
                    "depth_km": 29.0,
                    "trigger_time_utc": "2011-03-11T05:46:24+00:00",
                    "latest_anomaly_score": 0.97,
                    "dart_confirmation": True,
                    "active_dart_stations": ["21418"],
                    "stations_in_event_mode": ["21418"],
                },
            }

    with _app_client() as (app_module, client):
        db = _StubDb(None)
        app_module._db_client = db
        app_module._fsm = FSMOrchestrator(db_client=db)
        app_module._audit = AuditLogger(db_client=db)
        app_module._audit.append(
            AuditEntry(
                event_id=event_id,
                event_type="state_transition",
                producer="pipeline-worker",
                data={
                    "transition_id": str(uuid4()),
                    "from_state": "MONITOR",
                    "to_state": "ESCALATE",
                    "trigger_reason": "worker transition",
                },
            )
        )
        try:
            response = client.get("/api/fsm", headers=AUTH)
            assert response.status_code == 200
            assert response.json()["fsm_state"] == "ESCALATE"
            assert response.json()["transition_history"][0]["trigger_reason"] == (
                "worker transition"
            )
        finally:
            app_module._db_client = None


def test_api_fsm_fails_loud_when_durable_history_query_fails() -> None:
    from hazard_assessment.audit.logger import AuditLogger

    class _FailingDb:
        is_connected = True

        def load_fsm_state(self) -> None:
            return None

        def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("audit unavailable")

        def close(self) -> None:
            return None

    with _app_client() as (app_module, client):
        db = _FailingDb()
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)

        response = client.get("/api/fsm", headers=AUTH)

        assert response.status_code == 503
        assert "transition history" in response.json()["detail"].lower()

        audit_response = client.get("/api/audit", headers=AUTH)
        assert audit_response.status_code == 503
        assert "audit history" in audit_response.json()["detail"].lower()


def test_api_audit_surfaces_db_written_entries() -> None:
    """With a database configured, /api/audit reads the durable audit_events
    table so entries written by other processes (e.g. the pipeline worker) are
    visible, instead of only the API process's in-memory buffer. Verifies the
    DB-row -> AuditEntry mapping: the metadata JSONB is a faithful round-trip of
    the original ``data`` dict (the shredded-column fallback for metadata-less
    rows is covered in test_audit.py::TestRowToAuditEntry).
    """
    from datetime import UTC, datetime
    from uuid import uuid4

    from hazard_assessment.audit.logger import AuditLogger

    ev = uuid4()
    tr = uuid4()
    db_row = {
        "id": 42,
        "event_id": ev,
        "trace_id": tr,
        "recorded_at": datetime(2026, 1, 1, 0, 0, tzinfo=UTC),
        "agent_name": "orchestrator",
        "action": "state_transition",
        "state_before": "MONITOR",
        "state_after": "INVESTIGATE",
        "decision_basis": "score >= T1",
        # Real append_audit writes metadata = json.dumps(entry.data); for a
        # transition that data dict already carries from_state/to_state, so the
        # round-trip (not the shredded columns) is what surfaces them.
        "metadata": {
            "from_state": "MONITOR",
            "to_state": "INVESTIGATE",
            "trigger_reason": "score >= T1",
            "anomaly_score": 0.42,
        },
    }

    class _StubDb:
        is_connected = True

        def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
            return [db_row]

        def append_audit(self, _entry: Any) -> None:
            pass

    with _app_client() as (app_module, client):
        app_module._audit = AuditLogger(db_client=_StubDb())
        response = client.get("/api/audit", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert len(body) == 1
        entry = body[0]
        assert entry["producer"] == "orchestrator"
        assert entry["event_type"] == "state_transition"
        assert entry["event_id"] == str(ev)
        assert entry["trace_id"] == str(tr)
        assert entry["data"]["from_state"] == "MONITOR"
        assert entry["data"]["to_state"] == "INVESTIGATE"
        assert entry["data"]["anomaly_score"] == 0.42


def test_api_audit_returns_most_recent_on_in_memory_path() -> None:
    """Regression: /api/audit must return the NEWEST entries even when more than
    the internal fetch window (200) match, on the in-memory (no-DB) path.

    A prior bug had query_entries front-slice the append-ordered buffer, so with
    >200 matching entries it returned the OLDEST 200 and the endpoint's "recent"
    contract silently missed the most recent entries. Explicit increasing
    timestamps make the DESC ordering deterministic (no now() ties).
    """
    from datetime import UTC, datetime, timedelta

    from hazard_assessment.audit.logger import AuditEntry

    ev = uuid4()
    base = datetime(2026, 1, 1, tzinfo=UTC)
    with _app_client() as (app_module, client):
        for i in range(250):
            app_module._audit.append(
                AuditEntry(
                    event_id=ev,
                    event_type="handoff",
                    producer=f"p{i}",
                    timestamp_utc=base + timedelta(seconds=i),
                    data={"i": i},
                )
            )
        response = client.get("/api/audit?limit=5", headers=AUTH)
        assert response.status_code == 200
        body = response.json()
        assert [e["data"]["i"] for e in body] == [249, 248, 247, 246, 245]


def test_worker_driven_escalate_is_reviewable() -> None:
    """A worker-driven ESCALATE, surfaced via the DB read-through, must be
    reviewable. GET /api/escalation reports "in ESCALATE but no packet yet",
    and POST /api/escalation/generate builds a packet for it. Both endpoints
    must refresh FSM state from the DB first, or the API process's in-memory
    FSM is stale and they answer 404/409."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from hazard_assessment.orchestrator.states import FSMOrchestrator

    ev = uuid4()
    ctx_json = {
        "event_id": str(ev),
        "seismic_magnitude": 9.1,
        "seismic_region": "japan_trench",
        "epicenter_lat": 38.3,
        "epicenter_lon": 142.37,
        "depth_km": 29.0,
        "trigger_time_utc": datetime(2011, 3, 11, 5, 46, tzinfo=UTC).isoformat(),
        "latest_anomaly_score": 0.0,
        "dart_confirmation": False,
        "active_dart_stations": [],
        "stations_in_event_mode": [],
    }

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, Any]:
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": ctx_json,
            }

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        app_module._fsm = FSMOrchestrator(db_client=app_module._db_client)
        try:
            # In ESCALATE via read-through, but no packet generated yet.
            pre = client.get("/api/escalation", headers=AUTH)
            assert pre.status_code == 404
            assert "no escalation packet" in pre.json()["detail"].lower()

            # A duty scientist generates the packet for the worker-driven event.
            gen = client.post("/api/escalation/generate", headers=AUTH, json={})
            assert gen.status_code == 200

            # Now it is reviewable and reflects the worker's event.
            got = client.get("/api/escalation", headers=AUTH)
            assert got.status_code == 200
            body = got.json()
            assert body["event_id"] == str(ev)
            assert body["seismic_magnitude"] == 9.1
        finally:
            app_module._db_client = None


def test_escalation_generate_ignores_stale_transition_from_prior_event() -> None:
    """POST /api/escalation/generate must not attach trigger/time/trace
    metadata from an in-memory ESCALATE transition that belongs to a PRIOR
    event. With the DB read-through showing event B while this process's
    in-memory history still holds an ESCALATE for old event A, the packet must
    fall through to transition=None (audit-trail recovery or 'unknown'), never
    reusing event A's trigger."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from hazard_assessment.orchestrator.states import (
        FSMOrchestrator,
        SystemState,
        TransitionRecord,
    )

    ev_a = uuid4()
    ev_b = uuid4()
    ctx_json = {
        "event_id": str(ev_b),
        "seismic_magnitude": 9.0,
        "seismic_region": "chile_trench",
        "epicenter_lat": -35.85,
        "epicenter_lon": -72.72,
        "depth_km": 20.0,
        "trigger_time_utc": datetime(2010, 2, 27, 6, 34, tzinfo=UTC).isoformat(),
        "latest_anomaly_score": 0.0,
        "dart_confirmation": False,
        "active_dart_stations": [],
        "stations_in_event_mode": [],
    }

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, Any]:
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": ctx_json,
            }

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        app_module._fsm = FSMOrchestrator(db_client=app_module._db_client)
        # Stale in-memory ESCALATE from a prior event this process once drove.
        app_module._fsm._transition_history.append(
            TransitionRecord(
                event_id=ev_a,
                from_state=SystemState.ASSESS,
                to_state=SystemState.ESCALATE,
                trigger_reason="STALE TRIGGER FROM PRIOR EVENT A",
                seismic_magnitude=8.8,
            )
        )
        try:
            gen = client.post("/api/escalation/generate", headers=AUTH, json={})
            assert gen.status_code == 200

            body = client.get("/api/escalation", headers=AUTH).json()
            assert body["event_id"] == str(ev_b)
            assert body["escalation_trigger"] != "STALE TRIGGER FROM PRIOR EVENT A"
        finally:
            app_module._db_client = None


def test_review_decision_reason_rejects_prohibited_terms() -> None:
    """Reserved alert terminology is rejected before review persistence."""
    with _app_client() as (app_module, client):
        event_id = _escalate_in_memory(app_module)
        packet_fields, _ = install_durable_review_packet(app_module, event_id)

        resp = client.post(
            "/api/review",
            headers={**AUTH, "X-Reviewer-Id": "scientist-1"},
            json={
                "event_id": event_id,
                "decision": "REJECT",
                "decision_reason": "Looks like an official Tsunami Warning case",
                **packet_fields,
            },
        )
        assert resp.status_code == 400
        assert "prohibited alert terminology" in resp.json()["detail"]


def test_stale_escalation_packet_rejected_for_different_event() -> None:
    """GET /api/escalation must not serve a packet whose event differs from the
    FSM's current escalated event (stale evidence from a prior event)."""
    from datetime import UTC, datetime
    from uuid import uuid4

    from hazard_assessment.orchestrator.states import FSMOrchestrator

    state = {"event_id": str(uuid4())}

    def _ctx_json() -> dict[str, Any]:
        return {
            "event_id": state["event_id"],
            "seismic_magnitude": 9.1,
            "seismic_region": "japan_trench",
            "epicenter_lat": 38.3,
            "epicenter_lon": 142.37,
            "depth_km": 29.0,
            "trigger_time_utc": datetime(2011, 3, 11, 5, 46, tzinfo=UTC).isoformat(),
            "latest_anomaly_score": 0.0,
            "dart_confirmation": False,
            "active_dart_stations": [],
            "stations_in_event_mode": [],
        }

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, Any]:
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": _ctx_json(),
            }

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        app_module._fsm = FSMOrchestrator(db_client=app_module._db_client)
        try:
            # Generate and serve a packet for the current event.
            assert (
                client.post(
                    "/api/escalation/generate", headers=AUTH, json={}
                ).status_code
                == 200
            )
            assert client.get("/api/escalation", headers=AUTH).status_code == 200

            # The FSM now escalates on a DIFFERENT event; the stale packet must
            # not be served.
            state["event_id"] = str(uuid4())
            resp = client.get("/api/escalation", headers=AUTH)
            assert resp.status_code == 404
            assert "different event" in resp.json()["detail"].lower()
        finally:
            app_module._db_client = None


def _escalate_in_memory(app_module: Any) -> str:
    """Drive the in-process FSM to ESCALATE; return the event id."""
    app_module._fsm.evaluate_seismic_trigger(
        magnitude=7.8,
        region="pacific_nw",
        epicenter_lat=46.0,
        epicenter_lon=-130.0,
        tsunamigenic_zones={"pacific_nw"},
    )
    app_module._fsm.evaluate_anomaly_score(0.36)
    app_module._fsm.evaluate_anomaly_score(0.61)
    app_module._fsm.evaluate_anomaly_score(0.86)
    ctx = app_module._fsm.event_context
    assert ctx is not None
    return str(ctx.event_id)


def test_packet_of_record_requires_database() -> None:
    """Without a database there is no durable reviewer packet to serve."""
    with _app_client() as (app_module, client):
        _escalate_in_memory(app_module)
        resp = client.get("/api/escalation/packet-of-record", headers=AUTH)
        assert resp.status_code == 404
        assert "no database" in resp.json()["detail"].lower()


def test_packet_of_record_requires_active_event() -> None:
    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> None:
            return None

        def get_escalation_packet_for_event(self, _event_id: Any) -> None:
            raise AssertionError("must not be queried without an active event")

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        try:
            resp = client.get("/api/escalation/packet-of-record", headers=AUTH)
            assert resp.status_code == 404
            assert "no active event" in resp.json()["detail"].lower()
        finally:
            app_module._db_client = None


def test_packet_of_record_serves_durable_row() -> None:
    """The endpoint returns the immutable escalation_packets row for the
    active event: identity columns plus the packet body, with created_at
    serialized to ISO 8601."""
    from datetime import UTC, datetime

    queried: list[Any] = []

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> None:
            return None

        def get_escalation_packet_for_event(
            self, event_id: Any
        ) -> dict[str, Any] | None:
            queried.append(event_id)
            return {
                "id": 3,
                "assessment_row_id": 41,
                "event_id": event_id,
                "renderer_version": "1",
                "content_sha256": "c" * 64,
                "created_at": datetime(2026, 7, 17, 1, 2, 3, tzinfo=UTC),
                "packet": {"kind": "escalation_reviewer_packet"},
            }

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        try:
            event_id = _escalate_in_memory(app_module)
            resp = client.get("/api/escalation/packet-of-record", headers=AUTH)
            assert resp.status_code == 200
            body = resp.json()
            assert body["packet_row_id"] == 3
            assert body["assessment_row_id"] == 41
            assert body["event_id"] == event_id
            assert body["renderer_version"] == "1"
            assert body["content_sha256"] == "c" * 64
            assert body["created_at"] == "2026-07-17T01:02:03+00:00"
            assert body["packet"] == {"kind": "escalation_reviewer_packet"}
            assert [str(q) for q in queried] == [event_id]
        finally:
            app_module._db_client = None


def test_packet_of_record_404_when_no_durable_row() -> None:
    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> None:
            return None

        def get_escalation_packet_for_event(self, _event_id: Any) -> None:
            return None

    with _app_client() as (app_module, client):
        app_module._db_client = _StubDb()
        try:
            _escalate_in_memory(app_module)
            resp = client.get("/api/escalation/packet-of-record", headers=AUTH)
            assert resp.status_code == 404
            assert "no durable reviewer packet" in resp.json()["detail"].lower()
        finally:
            app_module._db_client = None


# ---------------------------------------------------------------------------
# After-action analysis endpoint
# ---------------------------------------------------------------------------


def test_after_action_rejects_active_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Event closure gate: the currently active event cannot be analyzed."""
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    with _app_client() as (app_module, client):
        event_id = _escalate_in_memory(app_module)
        resp = client.post(
            "/api/after-action", headers=AUTH, json={"event_id": event_id}
        )
        assert resp.status_code == 409
        assert "still active" in resp.json()["detail"].lower()


def test_after_action_rejects_unknown_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An event with no audit records at all is rejected, not analyzed."""
    from hazard_assessment.audit.logger import AuditLogger

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    with _app_client() as (app_module, client):
        db = DurableReviewDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        resp = client.post(
            "/api/after-action", headers=AUTH, json={"event_id": str(uuid4())}
        )
        assert resp.status_code == 404
        assert "no audit records" in resp.json()["detail"].lower()


def test_after_action_requires_durable_audit_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hazard_assessment.audit.logger import AuditEntry

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    with _app_client() as (app_module, client):
        event_id = uuid4()
        app_module._audit.append(
            AuditEntry(
                event_id=event_id,
                event_type="state_transition",
                producer="orchestrator",
                data={"from_state": "ASSESS", "to_state": "IDLE"},
            )
        )

        resp = client.post(
            "/api/after-action", headers=AUTH, json={"event_id": str(event_id)}
        )

        assert resp.status_code == 503
        assert "durable audit storage" in resp.json()["detail"].lower()


def test_after_action_fails_when_durable_history_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hazard_assessment.audit.logger import AuditLogger

    class _FailingAuditDb(DurableReviewDb):
        def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
            raise RuntimeError("audit unavailable")

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    with _app_client() as (app_module, client):
        db = _FailingAuditDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)

        resp = client.post(
            "/api/after-action", headers=AUTH, json={"event_id": str(uuid4())}
        )

        assert resp.status_code == 503
        assert "audit history" in resp.json()["detail"].lower()


def test_after_action_fails_when_report_commit_is_not_confirmed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import hazard_assessment.agents.llm_advisory.after_action as aa_module
    from hazard_assessment.audit.logger import AuditEntry, AuditLogger

    class _StubGraph:
        def invoke(self, _state: Any, config: Any = None) -> dict[str, str]:
            return {"timeline": "timeline", "gaps": "gaps", "draft_report": "report"}

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setattr(
        aa_module,
        "build_after_action_graph",
        lambda *_args, **_kwargs: _StubGraph(),
    )

    with _app_client() as (app_module, client):
        event_id = uuid4()
        db = DurableReviewDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        app_module._audit.append(
            AuditEntry(
                event_id=event_id,
                event_type="state_transition",
                producer="orchestrator",
                data={"from_state": "ASSESS", "to_state": "IDLE"},
            )
        )
        monkeypatch.setattr(db, "append_audit", lambda _entry: False)

        resp = client.post(
            "/api/after-action", headers=AUTH, json={"event_id": str(event_id)}
        )

        assert resp.status_code == 503
        assert "durable audit storage" in resp.json()["detail"].lower()


def test_after_action_withholds_a_tool_log_using_reserved_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reserved terminology in the tool-call log must not reach the operator.

    The narrative fields were scanned but the tool-call log was not, and the
    log is model-authored: an unknown-tool request echoes back the name the
    model asked for, and a recorded call echoes its arguments. That put a
    reserved term in front of a duty scientist, and into the audit trail,
    while the three narrative fields came back clean. The investigator already
    scans its own log for exactly this reason.
    """
    import hazard_assessment.agents.llm_advisory.after_action as aa_module
    from hazard_assessment.audit.logger import AuditEntry, AuditLogger

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    # The model asks for a tool whose name carries an official NOAA product
    # term. The name is echoed verbatim into the log by resolve_tool_calls.
    poisoned_call = {
        "node": "timeline",
        "tool": "issue_tsunami_warning",
        "args": {"text": "All Clear"},
        "error": "unknown_tool",
    }

    class _StubGraph:
        def invoke(self, _state: Any, config: Any = None) -> dict[str, Any]:
            return {
                "timeline": "clean timeline",
                "gaps": "clean gaps",
                "draft_report": "clean report",
            }

    def _stub_builder(
        _settings: Any,
        _audit_snapshot: Any,
        *,
        pinned_event_id: Any,
        tool_call_log: list[dict[str, Any]] | None = None,
    ) -> _StubGraph:
        assert tool_call_log is not None
        tool_call_log.append(poisoned_call)
        return _StubGraph()

    monkeypatch.setattr(aa_module, "build_after_action_graph", _stub_builder)

    with _app_client() as (app_module, client):
        closed_event = uuid4()
        db = DurableReviewDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        app_module._audit.append(AuditEntry(
            event_id=closed_event,
            event_type="state_transition",
            producer="orchestrator",
            data={"from_state": "ASSESS", "to_state": "IDLE"},
        ))
        resp = client.post(
            "/api/after-action",
            headers=AUTH,
            json={"event_id": str(closed_event)},
        )
        assert resp.status_code == 200
        body = resp.json()

        # The clean narrative is untouched: only the log was poisoned.
        assert body["timeline"] == "clean timeline"

        # Withheld whole rather than edited, because the log is the record of
        # what the model did. The count survives; the model's text does not.
        assert body["tool_calls"] != [poisoned_call]
        assert len(body["tool_calls"]) == 1
        assert body["tool_calls"][0]["n_calls"] == 1
        assert "withheld" in body["tool_calls"][0]

        returned = json.dumps(body)
        for term in ("issue_tsunami_warning", "All Clear"):
            assert term not in returned, f"{term!r} reached the operator"

        # The same must hold for what was written to the audit trail.
        persisted = app_module._audit.get_entries(
            event_id=closed_event, event_type="after_action_report"
        )
        assert len(persisted) == 1
        stored = json.dumps(persisted[0].data, default=str)
        for term in ("issue_tsunami_warning", "All Clear"):
            assert term not in stored, f"{term!r} was persisted to the audit trail"


def test_after_action_returns_and_persists_tool_call_log(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """the endpoint returns the tool-call log and persists the
    result (text plus log) as an after_action_report audit entry."""
    import hazard_assessment.agents.llm_advisory.after_action as aa_module
    from hazard_assessment.audit.logger import AuditEntry, AuditLogger

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")

    recorded_call = {
        "node": "timeline",
        "tool": "query_audit_trail",
        "args": {"event_type": ""},
        "n_total_matching": 1,
        "n_returned": 1,
        "truncated": False,
    }

    class _StubGraph:
        def invoke(self, _state: Any, config: Any = None) -> dict[str, Any]:
            return {
                "timeline": "clean timeline",
                "gaps": "clean gaps",
                "draft_report": "clean report",
            }

    def _stub_builder(
        _settings: Any,
        _audit_snapshot: Any,
        *,
        pinned_event_id: Any,
        tool_call_log: list[dict[str, Any]] | None = None,
    ) -> _StubGraph:
        assert tool_call_log is not None
        tool_call_log.append(recorded_call)
        return _StubGraph()

    monkeypatch.setattr(aa_module, "build_after_action_graph", _stub_builder)

    with _app_client() as (app_module, client):
        closed_event = uuid4()
        db = DurableReviewDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        # A nonactive event with durable audit history.
        app_module._audit.append(AuditEntry(
            event_id=closed_event,
            event_type="state_transition",
            producer="orchestrator",
            data={"from_state": "ASSESS", "to_state": "IDLE"},
        ))
        resp = client.post(
            "/api/after-action",
            headers=AUTH,
            json={"event_id": str(closed_event)},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["timeline"] == "clean timeline"
        assert body["draft_report"] == "clean report"
        assert body["tool_calls"] == [recorded_call]
        assert body["report_correlation_id"]

        persisted = app_module._audit.get_entries(
            event_id=closed_event, event_type="after_action_report"
        )
        assert len(persisted) == 1
        entry = persisted[0]
        assert entry.data["report_correlation_id"] == body["report_correlation_id"]
        assert entry.producer == "after_action_graph"
        assert entry.data["draft_report"] == "clean report"
        assert entry.data["tool_calls"] == [recorded_call]
        assert entry.data["n_tool_calls"] == 1


def test_status_rejects_non_ascii_api_key_with_401_not_500() -> None:
    """A non-ASCII key must fail authentication, not crash the handler.

    ``hmac.compare_digest`` raises TypeError on ``str`` arguments holding
    non-ASCII characters instead of returning False. Header values reach the
    app decoded as latin-1, so one high byte used to escape the dependency as
    an unhandled exception and surface as a 500. The header is sent as raw
    bytes because HTTP clients refuse to encode a non-ASCII ``str`` header.
    """
    with _app_client() as (_, client):
        response = client.get(
            "/status",
            headers={b"X-Hazard-Api-Key": b"\xff\xfe"},
        )

        assert response.status_code == 401


# ---------------------------------------------------------------------------
# Active-event investigation endpoint
# ---------------------------------------------------------------------------
#
# The gates are the mirror image of /api/after-action: that one refuses the
# active event, this one requires it. They are pinned here because the endpoint
# writes findings under a role the API is deliberately not granted, so a gate
# quietly regressing would either lose the separation or fail in production
# rather than in a test.


def test_investigate_requires_an_active_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with _app_client() as (_app_module, client):
        resp = client.post("/api/investigate", headers=AUTH)

    assert resp.status_code == 409
    assert "no event is currently active" in resp.json()["detail"].lower()


def test_investigate_requires_the_llm_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Config failure is reported before event state, and names both routes."""
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    with _app_client() as (_app_module, client):
        resp = client.post("/api/investigate", headers=AUTH)

    assert resp.status_code == 501
    detail = resp.json()["detail"]
    assert "LLM_API_KEY" in detail
    assert "LLM_BASE_URL" in detail


def test_investigate_requires_durable_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An in-memory audit trail leaves no record of the advice given."""
    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with _app_client() as (app_module, client):
        _escalate_in_memory(app_module)
        resp = client.post("/api/investigate", headers=AUTH)

    assert resp.status_code == 503
    assert "durable storage" in resp.json()["detail"].lower()


def test_investigate_requires_a_persisted_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Normal early in an event: nothing to investigate until one is written."""
    from hazard_assessment.audit.logger import AuditLogger

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    with _app_client() as (app_module, client):
        db = DurableReviewDb(None)
        db.get_latest_assessment_for_event = lambda event_id: None  # type: ignore[method-assign]
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        # Stand in for the investigator's own connection so the test exercises
        # the checkpoint gate rather than the role-connection gate.
        app_module._investigator_db = db
        try:
            _escalate_in_memory(app_module)
            resp = client.post("/api/investigate", headers=AUTH)
        finally:
            app_module._investigator_db = None

    assert resp.status_code == 404
    assert "assessment checkpoint" in resp.json()["detail"].lower()


def test_investigate_refuses_without_the_investigator_role(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Findings must not be written through the API's own database identity.

    migration 009 grants INSERT on evidence_issue_results to
    investigator_writer alone, so failing to reach the database as that role is
    a refusal rather than a reason to fall back to orchestrator_writer.
    """
    from hazard_assessment.audit.logger import AuditLogger

    monkeypatch.setenv("LLM_API_KEY", "test-llm-key")
    monkeypatch.setenv("LLM_MODEL", "test-model")
    monkeypatch.delenv("DB_HOST", raising=False)
    with _app_client() as (app_module, client):
        db = DurableReviewDb(None)
        app_module._db_client = db
        app_module._audit = AuditLogger(db_client=db)
        app_module._investigator_db = None
        _escalate_in_memory(app_module)
        resp = client.post("/api/investigate", headers=AUTH)

    assert resp.status_code == 503
    assert "investigator_writer" in resp.json()["detail"]


def test_investigate_requires_the_api_key() -> None:
    with _app_client() as (_app_module, client):
        assert client.post("/api/investigate").status_code == 401
