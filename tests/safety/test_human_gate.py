"""Safety tests for human gate enforcement.

These tests verify that the review gate cannot be bypassed:
1. Review requires a durable packet of record.
2. Packet row ID and canonical hash must match.
3. Review requires an active ESCALATE event.
4. Review must target the active event.
5. Every review remains caller-asserted.
6. No caller-asserted decision changes FSM state or authorizes distribution.
7. Decision hash binds packet and assessment identities and hashes.
8. Review requires an explicit caller-asserted reviewer ID.
9. Empty decision_reason is rejected.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from tests._review_support import (
    DurableReviewDb,
    install_durable_review_packet,
)


@contextmanager
def _app_client(
    api_key: str = "test-key",
) -> Generator[tuple[Any, TestClient], None, None]:
    """Context manager that starts the app with the given API key.

    Resets module-level FSM/audit/escalation singletons on entry to
    ensure test isolation.
    """
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


def _escalate_fsm(app_module: Any) -> str:
    """Drive the FSM to ESCALATE state and return the active event_id."""
    app_module._fsm.evaluate_seismic_trigger(
        magnitude=7.8,
        region="pacific_nw",
        epicenter_lat=46.0,
        epicenter_lon=-130.0,
        tsunamigenic_zones={"pacific_nw"},
    )
    app_module._fsm.evaluate_anomaly_score(0.36)  # MONITOR -> INVESTIGATE
    app_module._fsm.evaluate_anomaly_score(0.61)  # INVESTIGATE -> ASSESS
    app_module._fsm.evaluate_anomaly_score(0.86)  # ASSESS -> ESCALATE

    ctx = app_module._fsm.event_context
    assert ctx is not None
    return str(ctx.event_id)


AUTH = {"X-Hazard-Api-Key": "test-key"}


def _generate_packet(client: TestClient) -> str:
    """Generate a legacy in-memory packet for generator-specific tests."""
    resp = client.post(
        "/api/escalation/generate",
        headers=AUTH,
        json={},
    )
    assert resp.status_code == 200
    return resp.json()["packet_id"]


def _install_packet(app_module: Any, event_id: str) -> dict[str, Any]:
    fields, _ = install_durable_review_packet(app_module, event_id)
    return fields


# ---------------------------------------------------------------------------
# Decision CANNOT be submitted without escalation packet
# ---------------------------------------------------------------------------


class TestHumanGateCannotBypass:
    """Safety tests: verify no path exists to bypass the human review gate."""

    def test_decision_rejected_when_no_packet_exists(self) -> None:
        """A decision CANNOT be submitted if no escalation packet has been
        generated - this enforces that the reviewer has viewed the evidence."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            app_module._db_client = DurableReviewDb(None)
            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "looks good",
                    "escalation_packet_row_id": 1,
                    "escalation_packet_hash": "a" * 64,
                },
            )
            assert resp.status_code == 409
            assert "reviewer packet" in resp.json()["detail"].lower()

    def test_decision_rejected_with_wrong_packet_id(self) -> None:
        """A decision with a fabricated packet ID is rejected."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    **packet_fields,
                    "escalation_packet_row_id": 999,
                },
            )
            assert resp.status_code == 409
            assert "does not match" in resp.json()["detail"].lower()

    def test_decision_rejected_with_wrong_packet_hash(self) -> None:
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)
            packet_fields["escalation_packet_hash"] = "f" * 64

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    **packet_fields,
                },
            )
            assert resp.status_code == 409
            assert "packet_hash" in resp.json()["detail"]

    def test_decision_rejected_when_stored_packet_fails_hash_check(self) -> None:
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields, row = install_durable_review_packet(
                app_module, event_id
            )
            row["packet"]["assessment"][
                "scientific_content_hash"
            ] = "f" * 64

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    **packet_fields,
                },
            )
            assert resp.status_code == 409
            assert "canonical hash" in resp.json()["detail"].lower()

    def test_decision_rejected_when_not_in_escalate_state(self) -> None:
        """A decision is rejected if the FSM is not in ESCALATE state."""
        with _app_client() as (app_module, client):
            # FSM is in IDLE - no escalation
            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": str(uuid4()),
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    "escalation_packet_row_id": 1,
                    "escalation_packet_hash": "a" * 64,
                },
            )
            assert resp.status_code == 409
            assert "active escalation" in resp.json()["detail"].lower()

    def test_decision_rejected_for_wrong_event_id(self) -> None:
        """A decision targeting a non-active event is rejected."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": str(uuid4()),  # wrong event
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    **packet_fields,
                },
            )
            assert resp.status_code == 409
            assert "event id" in resp.json()["detail"].lower()

    def test_decision_requires_durable_packet_binding_fields(self) -> None:
        """Omitting durable packet row/hash fields is a validation error."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            _generate_packet(client)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "approving",
                    # missing packet row ID and hash
                },
            )
            assert resp.status_code == 422  # FastAPI validation error

    def test_empty_decision_reason_rejected(self) -> None:
        """Empty decision_reason is rejected (min_length=1)."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "",
                    **packet_fields,
                },
            )
            assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Caller-asserted reviews never change FSM state
# ---------------------------------------------------------------------------


class TestDecisionOutcomes:
    """Verify decision outcomes match specification."""

    def test_reject_stays_in_escalate(self) -> None:
        """REJECT keeps FSM in ESCALATE - does NOT return to IDLE."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "REJECT",
                    "decision_reason": "insufficient evidence",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "recorded"
            assert resp.json()["decision"] == "REJECT"

            # FSM must still be in ESCALATE
            fsm_resp = client.get("/api/fsm", headers=AUTH)
            assert fsm_resp.json()["fsm_state"] == "ESCALATE"

            # Escalation packet must still be accessible after REJECT
            packet_resp = client.get(
                "/api/escalation/packet-of-record", headers=AUTH
            )
            assert packet_resp.status_code == 200
            assert (
                packet_resp.json()["packet_row_id"]
                == packet_fields["escalation_packet_row_id"]
            )

    def test_defer_stays_in_escalate(self) -> None:
        """DEFER keeps FSM in ESCALATE."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "DEFER",
                    "decision_reason": "need more data",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "recorded"

            fsm_resp = client.get("/api/fsm", headers=AUTH)
            assert fsm_resp.json()["fsm_state"] == "ESCALATE"

            # Escalation packet must still be accessible after DEFER
            packet_resp = client.get(
                "/api/escalation/packet-of-record", headers=AUTH
            )
            assert packet_resp.status_code == 200
            assert (
                packet_resp.json()["packet_row_id"]
                == packet_fields["escalation_packet_row_id"]
            )

    def test_caller_asserted_approve_does_not_resolve_event(self) -> None:
        """Caller-asserted APPROVE records review but cannot close event."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "confirmed tsunami signal",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "recorded"
            assert resp.json()["distribution_authorized"] is False
            assert resp.json()["event_disposition_recorded"] is False

            fsm_resp = client.get("/api/fsm", headers=AUTH)
            assert fsm_resp.json()["fsm_state"] == "ESCALATE"

    def test_multiple_reviews_bind_same_packet_without_disposition(self) -> None:
        """Subsequent review records may bind the same immutable packet."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            # First: REJECT
            client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "REJECT",
                    "decision_reason": "need review",
                    **packet_fields,
                },
            )

            # Then: APPROVE (still in ESCALATE)
            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-2"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "additional evidence confirms",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["status"] == "recorded"
            assert resp.json()["fsm_state"] == "ESCALATE"


# ---------------------------------------------------------------------------
# Decision hash and audit trail
# ---------------------------------------------------------------------------


class TestDecisionIntegrity:
    """Verify decision hash and audit persistence."""

    def test_decision_hash_is_returned(self) -> None:
        """Every decision response includes a decision_hash."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "REJECT",
                    "decision_reason": "needs review",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert "decision_hash" in resp.json()
            assert len(resp.json()["decision_hash"]) == 64  # SHA-256 hex

    def test_decision_persisted_to_audit_with_hash(self) -> None:
        """Decision is persisted to audit trail with all required fields."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "scientist-42"},
                json={
                    "event_id": event_id,
                    "decision": "DEFER",
                    "decision_reason": "waiting for DART",
                    **packet_fields,
                },
            )

            audit_resp = client.get(
                "/api/audit",
                params={"event_type": "assessment_review_decision"},
                headers=AUTH,
            )
            entries = audit_resp.json()
            assert len(entries) == 1
            entry = entries[0]
            assert entry["producer"] == "scientist-42"
            assert entry["data"]["decision"] == "DEFER"
            assert entry["data"]["decision_reason"] == "waiting for DART"
            assert (
                entry["data"]["escalation_packet_row_id"]
                == packet_fields["escalation_packet_row_id"]
            )
            assert (
                entry["data"]["escalation_packet_hash"]
                == packet_fields["escalation_packet_hash"]
            )
            assert "assessment_id" in entry["data"]
            assert "assessment_scientific_content_hash" in entry["data"]
            assert "decision_hash" in entry["data"]
            assert "decided_at_utc" in entry["data"]

    def test_api_hash_matches_schema_hash(self) -> None:
        """API hash matches AssessmentReviewDecision._compute_hash()."""
        from hazard_assessment.schemas.human_decision import (
            AssessmentReviewDecision,
        )

        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "REJECT",
                    "decision_reason": "insufficient data",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            api_hash = resp.json()["decision_hash"]

            # Retrieve the decided_at_utc from audit to reconstruct
            audit_resp = client.get(
                "/api/audit",
                params={"event_type": "assessment_review_decision"},
                headers=AUTH,
            )
            entry = audit_resp.json()[0]["data"]

            record = AssessmentReviewDecision(
                producer="operator-1",
                event_id=event_id,
                reviewer_id="operator-1",
                decision="REJECT",
                decision_reason="insufficient data",
                decided_at_utc=entry["decided_at_utc"],
                escalation_packet_row_id=(
                    entry["escalation_packet_row_id"]
                ),
                escalation_packet_hash=entry["escalation_packet_hash"],
                assessment_row_id=entry["assessment_row_id"],
                assessment_id=entry["assessment_id"],
                assessment_scientific_content_hash=(
                    entry["assessment_scientific_content_hash"]
                ),
            )
            assert record.decision_hash == api_hash

    def test_review_records_are_honest_about_identity(self) -> None:
        """every decision the current API records carries
        identity_assurance=CALLER_ASSERTED in both the response and the
        audit record. The endpoint has no path to claim an authenticated
        human principal."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "confirmed by independent analysis",
                    **packet_fields,
                },
            )
            assert resp.status_code == 200
            assert resp.json()["identity_assurance"] == "CALLER_ASSERTED"

            audit_resp = client.get(
                "/api/audit",
                params={"event_type": "assessment_review_decision"},
                headers=AUTH,
            )
            entry = audit_resp.json()[0]
            assert entry["data"]["identity_assurance"] == "CALLER_ASSERTED"

    def test_identity_assurance_is_hash_bound(self) -> None:
        """Tampering with the recorded assurance level changes the
        decision hash, so an after-the-fact upgrade to
        TRUSTED_HUMAN_PRINCIPAL is detectable."""
        from datetime import UTC, datetime
        from uuid import uuid4

        from hazard_assessment.schemas.human_decision import (
            AssessmentReviewDecision,
            IdentityAssurance,
        )

        common = dict(
            producer="operator-1",
            event_id=uuid4(),
            reviewer_id="operator-1",
            decision="APPROVE",
            decision_reason="confirmed",
            decided_at_utc=datetime(2026, 7, 17, tzinfo=UTC),
            escalation_packet_row_id=3,
            escalation_packet_hash="a" * 64,
            assessment_row_id=41,
            assessment_id=uuid4(),
            assessment_scientific_content_hash="b" * 64,
        )
        asserted = AssessmentReviewDecision(
            identity_assurance=IdentityAssurance.CALLER_ASSERTED, **common
        )
        trusted = AssessmentReviewDecision(
            identity_assurance=IdentityAssurance.TRUSTED_HUMAN_PRINCIPAL,
            **common,
        )
        assert asserted.decision_hash != trusted.decision_hash

    def test_reviewer_id_header_is_required(self) -> None:
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            resp = client.post(
                "/api/review",
                headers=AUTH,
                json={
                    "event_id": event_id,
                    "decision": "REJECT",
                    "decision_reason": "reviewing",
                    **packet_fields,
                },
            )
            assert resp.status_code == 400
            assert "X-Reviewer-Id" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Escalation packet generation and content
# ---------------------------------------------------------------------------


class TestEscalationPacket:
    """Verify escalation packet generation and content."""

    def test_packet_generated_on_escalate(self) -> None:
        """An escalation packet can be generated when FSM is in ESCALATE."""
        with _app_client() as (app_module, client):
            _escalate_fsm(app_module)

            resp = client.post(
                "/api/escalation/generate",
                headers=AUTH,
                json={},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "generated"
            assert "packet_id" in data
            assert "packet_hash" in data
            assert len(data["packet_hash"]) == 64

    def test_packet_not_generated_when_not_escalated(self) -> None:
        """Packet generation is rejected when FSM is not in ESCALATE."""
        with _app_client() as (_, client):
            resp = client.post(
                "/api/escalation/generate",
                headers=AUTH,
                json={},
            )
            assert resp.status_code == 409

    def test_packet_contains_required_fields(self) -> None:
        """Generated packet contains all required evidence fields."""
        with _app_client() as (app_module, client):
            _escalate_fsm(app_module)
            _generate_packet(client)

            resp = client.get("/api/escalation", headers=AUTH)
            assert resp.status_code == 200
            packet = resp.json()

            # Required fields
            assert "escalation_trigger" in packet
            assert "criticality_reasons" in packet
            assert len(packet["criticality_reasons"]) >= 1
            assert "recommended_action" in packet
            assert "anomaly_timeline" in packet
            assert "packet_hash" in packet
            assert "handoff_id" in packet
            assert "event_id" in packet
            assert "seismic_magnitude" in packet
            assert "seismic_region" in packet
            assert "latest_anomaly_score" in packet
            assert "input_refs" in packet  # provenance references

    def test_packet_not_available_when_not_escalated(self) -> None:
        """Escalation packet endpoint returns 404 when not in ESCALATE."""
        with _app_client() as (_, client):
            resp = client.get("/api/escalation", headers=AUTH)
            assert resp.status_code == 404

    def test_packet_logged_to_audit_trail(self) -> None:
        """Escalation packet generation is logged in audit trail."""
        with _app_client() as (app_module, client):
            _escalate_fsm(app_module)
            _generate_packet(client)

            audit_resp = client.get(
                "/api/audit",
                params={"event_type": "escalation_packet_generated"},
                headers=AUTH,
            )
            entries = audit_resp.json()
            assert len(entries) == 1
            assert "packet_id" in entries[0]["data"]
            assert "packet_hash" in entries[0]["data"]

    def test_caller_asserted_approve_preserves_durable_packet(self) -> None:
        """Review cannot delete or replace immutable packet of record."""
        with _app_client() as (app_module, client):
            event_id = _escalate_fsm(app_module)
            packet_fields = _install_packet(app_module, event_id)

            client.post(
                "/api/review",
                headers={**AUTH, "X-Reviewer-Id": "operator-1"},
                json={
                    "event_id": event_id,
                    "decision": "APPROVE",
                    "decision_reason": "confirmed",
                    **packet_fields,
                },
            )

            resp = client.get(
                "/api/escalation/packet-of-record", headers=AUTH
            )
            assert resp.status_code == 200
            assert (
                resp.json()["packet_row_id"]
                == packet_fields["escalation_packet_row_id"]
            )


# ---------------------------------------------------------------------------
# Schema-level safety invariants
# ---------------------------------------------------------------------------


class TestSchemaInvariants:
    """Verify schema-level safety invariants for escalation and decision."""

    def test_escalation_packet_requires_criticality_reasons(self) -> None:
        """EscalationPacket requires at least one criticality reason."""
        from pydantic import ValidationError

        from hazard_assessment.schemas.escalation import EscalationPacket

        with pytest.raises(ValidationError):
            EscalationPacket(
                producer="test",
                escalation_trigger="test",
                criticality_reasons=[],  # min_length=1 violation
            )

    def test_human_decision_requires_escalation_packet_id(self) -> None:
        """HumanDecision requires escalation_packet_id."""
        from datetime import UTC, datetime

        from pydantic import ValidationError

        from hazard_assessment.schemas.human_decision import HumanDecision

        with pytest.raises(ValidationError):
            HumanDecision(
                producer="test",
                reviewer_id="operator-1",
                decision="APPROVE",
                decision_reason="ok",
                decided_at_utc=datetime.now(UTC),
                # missing escalation_packet_id
            )

    def test_human_decision_computes_hash(self) -> None:
        """HumanDecision auto-computes a decision hash."""
        from datetime import UTC, datetime

        from hazard_assessment.schemas.human_decision import HumanDecision

        decision = HumanDecision(
            producer="test",
            reviewer_id="operator-1",
            decision="APPROVE",
            decision_reason="confirmed",
            decided_at_utc=datetime.now(UTC),
            escalation_packet_id=uuid4(),
        )
        assert len(decision.decision_hash) == 64

    def test_escalation_packet_computes_hash(self) -> None:
        """EscalationPacket auto-computes a packet hash."""
        from hazard_assessment.schemas.escalation import EscalationPacket

        packet = EscalationPacket(
            producer="test",
            escalation_trigger="test trigger",
            criticality_reasons=["reason 1"],
        )
        assert len(packet.packet_hash) == 64

    def test_human_decision_requires_nonempty_reason(self) -> None:
        """HumanDecision rejects empty decision_reason."""
        from datetime import UTC, datetime

        from pydantic import ValidationError

        from hazard_assessment.schemas.human_decision import HumanDecision

        with pytest.raises(ValidationError):
            HumanDecision(
                producer="test",
                reviewer_id="operator-1",
                decision="APPROVE",
                decision_reason="",  # min_length=1 violation
                decided_at_utc=datetime.now(UTC),
                escalation_packet_id=uuid4(),
            )


# ---------------------------------------------------------------------------
# Guardrail scanner blocks prohibited terms in ALL output paths
# ---------------------------------------------------------------------------


class TestGuardrailsAllOutputPaths:
    """Guardrail scanner blocks prohibited terms in all output paths.

    Tests report, ABSTAIN, and escalation packet paths.
    """

    def test_report_path_rejects_prohibited_terms(self) -> None:
        """scan_text detects prohibited terms in report-like text."""
        from hazard_assessment.policy.guardrails import scan_text

        result = scan_text("Tsunami Warning issued for the Pacific coast.")
        assert len(result.violations) > 0
        assert any(v.term == "Warning" for v in result.violations)

    def test_abstain_path_rejects_prohibited_terms(self) -> None:
        """format_abstain() rejects abstain_reason containing prohibited terms."""
        from hazard_assessment.agents.assessment_formatter import format_abstain

        with pytest.raises(ValueError, match="prohibited alert terminology"):
            format_abstain(
                event_id=None,
                fsm_state="ASSESS",
                abstain_reason="Tsunami Warning detected in source data",
            )

    def test_escalation_packet_rejects_prohibited_terms(self) -> None:
        """generate_escalation_packet() rejects prohibited terms in reasons."""

        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditLogger
        from hazard_assessment.orchestrator.states import (
            EventContext,
            SystemState,
            TransitionRecord,
        )

        ctx = EventContext(
            seismic_magnitude=8.0,
            seismic_region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
        )
        transition = TransitionRecord(
            from_state=SystemState.ASSESS,
            to_state=SystemState.ESCALATE,
            trigger_reason="Tsunami Warning threshold exceeded",
            anomaly_score=0.90,
        )

        with pytest.raises(ValueError, match="prohibited alert terminology"):
            generate_escalation_packet(
                ctx=ctx,
                transition=transition,
                audit_logger=AuditLogger(),
            )

    def test_escalation_packet_rejects_prohibited_terms_in_region(self) -> None:
        """Prohibited terms in the seismic region string are rejected.

        Caller-supplied scenario and verification dictionaries were removed
        from the packet generator, so the remaining prose paths
        are the criticality reasons, the recommended action, and the
        seismic region.
        """

        from hazard_assessment.app import generate_escalation_packet
        from hazard_assessment.audit.logger import AuditLogger
        from hazard_assessment.orchestrator.states import (
            EventContext,
            SystemState,
            TransitionRecord,
        )

        ctx = EventContext(
            seismic_magnitude=8.0,
            seismic_region="Warning region",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
        )
        transition = TransitionRecord(
            from_state=SystemState.ASSESS,
            to_state=SystemState.ESCALATE,
            trigger_reason="Anomaly score 0.90 >= T3",
            anomaly_score=0.90,
        )

        with pytest.raises(ValueError, match="prohibited alert terminology"):
            generate_escalation_packet(
                ctx=ctx,
                transition=transition,
                audit_logger=AuditLogger(),
            )
