"""Unit tests for the permission-matrix policy check.

Validates what check_policy returns: that a capability outside an agent's
declared set is reported as denied, that EMIT_REPORT in ESCALATE is reported
as denied without a human decision, and that denials are structured and
auditable.

These are tests of a query function. The function is not wired into any
execution path, so nothing here shows that an action was prevented. What
actually holds the bounds is tested elsewhere: database grants in
tests/integration, terminology in test_guardrails.py, and ABSTAIN routing
and the review gate in tests/safety.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hazard_assessment.agents.base import AgentCapability
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.policy.approval import (
    DenialReason,
    PolicyCheckResult,
    PolicyDenial,
    check_policy,
    denial_to_response,
    load_permission_matrix,
    log_denial,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_matrix(
    agents: dict | None = None,
    permissions: list | None = None,
) -> dict:
    """Build a minimal permission matrix dict for testing."""
    return {
        "schema_version": "1.0",
        "permissions": permissions or [],
        "agents": agents or {},
    }


# ---------------------------------------------------------------------------
# Permission matrix loading
# ---------------------------------------------------------------------------


class TestLoadPermissionMatrix:
    def test_loads_default_matrix(self) -> None:
        matrix = load_permission_matrix()
        assert "agents" in matrix
        assert "permissions" in matrix
        assert "schema_version" in matrix

    def test_default_matrix_contains_known_agents(self) -> None:
        matrix = load_permission_matrix()
        assert "report_agent" in matrix["agents"]
        assert "human_review_gate" in matrix["agents"]
        assert "orchestrator" in matrix["agents"]

    def test_report_agent_has_emit_report(self) -> None:
        matrix = load_permission_matrix()
        assert "ER" in matrix["agents"]["report_agent"]["allowed"]

    def test_qc_agent_does_not_have_emit_report(self) -> None:
        matrix = load_permission_matrix()
        assert "ER" not in matrix["agents"]["qc_agent"]["allowed"]


# ---------------------------------------------------------------------------
# Capability checks - allowed
# ---------------------------------------------------------------------------


class TestCapabilityAllowed:
    def test_allowed_capability_passes(self) -> None:
        matrix = _make_matrix(agents={
            "qc_agent": {"allowed": ["RD", "WD", "WA"]},
        })
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.READ_DATA,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert result.allowed
        assert result.denial is None

    def test_write_audit_allowed_for_qc(self) -> None:
        matrix = _make_matrix(agents={
            "qc_agent": {"allowed": ["RD", "WD", "WA", "PK", "CK"]},
        })
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.WRITE_AUDIT,
            fsm_state="MONITOR",
            matrix=matrix,
        )
        assert result.allowed

    def test_emit_report_allowed_in_non_escalate_state(self) -> None:
        """EMIT_REPORT is allowed when not in ESCALATE, even without human decision."""
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ASSESS",
            matrix=matrix,
        )
        assert result.allowed


# ---------------------------------------------------------------------------
# Capability checks - denied
# ---------------------------------------------------------------------------


class TestCapabilityDenied:
    def test_unregistered_agent_denied(self) -> None:
        matrix = _make_matrix(agents={})
        result = check_policy(
            agent_name="rogue_agent",
            capability=AgentCapability.READ_DATA,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial is not None
        assert result.denial.reason == DenialReason.AGENT_NOT_REGISTERED

    def test_capability_not_in_allowed_set(self) -> None:
        matrix = _make_matrix(agents={
            "qc_agent": {"allowed": ["RD", "WD"]},
        })
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.INVOKE_LLM,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial is not None
        assert result.denial.reason == DenialReason.CAPABILITY_NOT_ALLOWED
        assert "INVOKE_LLM" in result.denial.detail

    def test_modify_state_denied_for_qc(self) -> None:
        matrix = _make_matrix(agents={
            "qc_agent": {"allowed": ["RD", "WD", "WA", "PK", "CK"]},
        })
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.MODIFY_STATE,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.CAPABILITY_NOT_ALLOWED


# ---------------------------------------------------------------------------
# Human approval gating - EMIT_REPORT in ESCALATE
# ---------------------------------------------------------------------------


class TestHumanApprovalGating:
    """Core acceptance criteria: EMIT_REPORT in ESCALATE without human decision => denied."""

    def test_emit_report_escalate_no_human_decision_denied(self) -> None:
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            human_decision_present=False,
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial is not None
        assert result.denial.reason == DenialReason.HUMAN_APPROVAL_REQUIRED
        assert "ESCALATE" in result.denial.detail

    def test_emit_report_escalate_with_human_decision_allowed(self) -> None:
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            human_decision_present=True,
            matrix=matrix,
        )
        assert result.allowed
        assert result.denial is None

    def test_non_gated_capability_in_escalate_allowed(self) -> None:
        """READ_DATA in ESCALATE does not require human approval."""
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.READ_DATA,
            fsm_state="ESCALATE",
            human_decision_present=False,
            matrix=matrix,
        )
        assert result.allowed

    def test_emit_report_idle_no_human_decision_allowed(self) -> None:
        """EMIT_REPORT in IDLE does not require human approval."""
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="IDLE",
            human_decision_present=False,
            matrix=matrix,
        )
        assert result.allowed

    def test_emit_report_assess_no_human_decision_allowed(self) -> None:
        """EMIT_REPORT in ASSESS does not require human approval."""
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ASSESS",
            matrix=matrix,
        )
        assert result.allowed

    def test_default_human_decision_is_false(self) -> None:
        """When human_decision_present is not passed, it defaults to False."""
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.HUMAN_APPROVAL_REQUIRED


# ---------------------------------------------------------------------------
# Denial precedence - earlier checks take priority
# ---------------------------------------------------------------------------


class TestDenialPrecedence:
    def test_unregistered_agent_takes_priority_over_human_gate(self) -> None:
        """An unregistered agent is denied even if it would also need human approval."""
        matrix = _make_matrix(agents={})
        result = check_policy(
            agent_name="unknown_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.AGENT_NOT_REGISTERED

    def test_missing_capability_takes_priority_over_human_gate(self) -> None:
        """Capability denied takes priority over human-approval check."""
        matrix = _make_matrix(agents={
            "qc_agent": {"allowed": ["RD"]},
        })
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            matrix=matrix,
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.CAPABILITY_NOT_ALLOWED


# ---------------------------------------------------------------------------
# Denial fields are correctly populated
# ---------------------------------------------------------------------------


class TestDenialFields:
    def test_denial_contains_agent_name(self) -> None:
        matrix = _make_matrix(agents={})
        result = check_policy(
            agent_name="test_agent",
            capability=AgentCapability.READ_DATA,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert result.denial.agent_name == "test_agent"

    def test_denial_contains_capability(self) -> None:
        matrix = _make_matrix(agents={
            "agent_a": {"allowed": ["RD"]},
        })
        result = check_policy(
            agent_name="agent_a",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="IDLE",
            matrix=matrix,
        )
        assert result.denial.capability == "ER"

    def test_denial_contains_fsm_state(self) -> None:
        matrix = _make_matrix(agents={
            "report_agent": {"allowed": ["RD", "WD", "WA", "ER"]},
        })
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            matrix=matrix,
        )
        assert result.denial.fsm_state == "ESCALATE"


# ---------------------------------------------------------------------------
# Structured denial response
# ---------------------------------------------------------------------------


class TestDenialResponse:
    def test_denial_to_response_structure(self) -> None:
        denial = PolicyDenial(
            agent_name="report_agent",
            capability="ER",
            fsm_state="ESCALATE",
            reason=DenialReason.HUMAN_APPROVAL_REQUIRED,
            detail="Human approval needed.",
        )
        resp = denial_to_response(denial)
        assert resp["status"] == "denied"
        assert "policy_violation" in resp
        pv = resp["policy_violation"]
        assert pv["agent_name"] == "report_agent"
        assert pv["capability"] == "ER"
        assert pv["fsm_state"] == "ESCALATE"
        assert pv["reason"] == "HUMAN_APPROVAL_REQUIRED"
        assert pv["detail"] == "Human approval needed."

    def test_denial_to_response_is_not_just_403(self) -> None:
        """The response is a structured dict, not a bare HTTP status code."""
        denial = PolicyDenial(
            agent_name="x",
            capability="ER",
            fsm_state="ESCALATE",
            reason=DenialReason.HUMAN_APPROVAL_REQUIRED,
            detail="detail",
        )
        resp = denial_to_response(denial)
        assert isinstance(resp, dict)
        assert "status" in resp
        assert "policy_violation" in resp


# ---------------------------------------------------------------------------
# Audit logging of denials
# ---------------------------------------------------------------------------


class TestDenialAuditLogging:
    def test_log_denial_appends_to_audit(self) -> None:
        audit = AuditLogger()
        denial = PolicyDenial(
            agent_name="report_agent",
            capability="ER",
            fsm_state="ESCALATE",
            reason=DenialReason.HUMAN_APPROVAL_REQUIRED,
            detail="Human approval required.",
        )
        log_denial(denial, audit)
        assert audit.count == 1
        entries = audit.get_entries(event_type="policy_denial")
        assert len(entries) == 1
        assert entries[0].producer == "policy_check_endpoint"
        assert entries[0].data["reason"] == "HUMAN_APPROVAL_REQUIRED"

    def test_log_denial_includes_event_id(self) -> None:
        audit = AuditLogger()
        eid = uuid4()
        denial = PolicyDenial(
            agent_name="agent_a",
            capability="RD",
            fsm_state="IDLE",
            reason=DenialReason.AGENT_NOT_REGISTERED,
            detail="Not registered.",
        )
        log_denial(denial, audit, event_id=eid)
        entries = audit.get_entries(event_id=eid)
        assert len(entries) == 1
        assert entries[0].event_id == eid

    def test_log_denial_data_fields(self) -> None:
        audit = AuditLogger()
        denial = PolicyDenial(
            agent_name="qc_agent",
            capability="IL",
            fsm_state="MONITOR",
            reason=DenialReason.CAPABILITY_NOT_ALLOWED,
            detail="Not allowed.",
        )
        log_denial(denial, audit)
        data = audit.get_entries()[0].data
        assert data["agent_name"] == "qc_agent"
        assert data["capability"] == "IL"
        assert data["fsm_state"] == "MONITOR"
        assert data["detail"] == "Not allowed."

    def test_multiple_denials_logged_independently(self) -> None:
        audit = AuditLogger()
        for i in range(3):
            denial = PolicyDenial(
                agent_name=f"agent_{i}",
                capability="RD",
                fsm_state="IDLE",
                reason=DenialReason.AGENT_NOT_REGISTERED,
                detail=f"Denial {i}",
            )
            log_denial(denial, audit)
        assert audit.count == 3


# ---------------------------------------------------------------------------
# PolicyCheckResult timezone validation
# ---------------------------------------------------------------------------


class TestPolicyCheckResultTimezone:
    def test_naive_checked_at_utc_rejected(self) -> None:
        naive = datetime(2026, 3, 4, 12, 0, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            PolicyCheckResult(allowed=True, checked_at_utc=naive)

    def test_aware_checked_at_utc_accepted(self) -> None:
        aware = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)
        result = PolicyCheckResult(allowed=True, checked_at_utc=aware)
        assert result.checked_at_utc == aware

    def test_default_checked_at_utc_is_aware(self) -> None:
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.READ_DATA,
            fsm_state="IDLE",
            matrix=_make_matrix(agents={"qc_agent": {"allowed": ["RD"]}}),
        )
        assert result.checked_at_utc.tzinfo is not None


# ---------------------------------------------------------------------------
# Integration with real permission matrix
# ---------------------------------------------------------------------------


class TestRealPermissionMatrix:
    """Tests against the actual permissions.yaml shipped with the package."""

    def test_report_agent_emit_report_escalate_denied_without_approval(self) -> None:
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            human_decision_present=False,
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.HUMAN_APPROVAL_REQUIRED

    def test_report_agent_emit_report_escalate_allowed_with_approval(self) -> None:
        result = check_policy(
            agent_name="report_agent",
            capability=AgentCapability.EMIT_REPORT,
            fsm_state="ESCALATE",
            human_decision_present=True,
        )
        assert result.allowed

    def test_orchestrator_can_modify_state(self) -> None:
        result = check_policy(
            agent_name="orchestrator",
            capability=AgentCapability.MODIFY_STATE,
            fsm_state="MONITOR",
        )
        assert result.allowed

    def test_qc_agent_cannot_modify_state(self) -> None:
        result = check_policy(
            agent_name="qc_agent",
            capability=AgentCapability.MODIFY_STATE,
            fsm_state="MONITOR",
        )
        assert not result.allowed
        assert result.denial.reason == DenialReason.CAPABILITY_NOT_ALLOWED

    def test_human_review_gate_can_approve_output(self) -> None:
        result = check_policy(
            agent_name="human_review_gate",
            capability=AgentCapability.APPROVE_OUTPUT,
            fsm_state="ESCALATE",
        )
        assert result.allowed

    def test_anomaly_agent_cannot_invoke_llm(self) -> None:
        result = check_policy(
            agent_name="anomaly_agent",
            capability=AgentCapability.INVOKE_LLM,
            fsm_state="INVESTIGATE",
        )
        assert not result.allowed

    def test_all_agents_can_write_audit(self) -> None:
        matrix = load_permission_matrix()
        for agent_name in matrix["agents"]:
            result = check_policy(
                agent_name=agent_name,
                capability=AgentCapability.WRITE_AUDIT,
                fsm_state="IDLE",
                matrix=matrix,
            )
            assert result.allowed, f"{agent_name} should be allowed to write audit"


class TestPolicyCheckResultInvariant:
    """Verify PolicyCheckResult rejects inconsistent allowed/denial combinations."""

    def test_denied_without_denial_raises(self) -> None:
        with pytest.raises(ValueError, match="denial must be set"):
            PolicyCheckResult(allowed=False, denial=None)

    def test_allowed_with_no_denial_is_valid(self) -> None:
        result = PolicyCheckResult(allowed=True)
        assert result.denial is None

    def test_denied_with_denial_is_valid(self) -> None:
        denial = PolicyDenial(
            agent_name="test",
            capability="RD",
            fsm_state="IDLE",
            reason=DenialReason.AGENT_NOT_REGISTERED,
            detail="test",
        )
        result = PolicyCheckResult(allowed=False, denial=denial)
        assert not result.allowed


class TestApprovalStateCoupling:
    """Verify that string literals in approval.py match FSM enum values.

    approval.py deliberately uses string literals to avoid importing the FSM
    module (see comment at line 26). These tests catch drift if the FSM enum
    values are ever renamed.
    """

    def test_human_approval_states_match_fsm_enum(self) -> None:
        from hazard_assessment.orchestrator.states import SystemState
        from hazard_assessment.policy.approval import _HUMAN_APPROVAL_REQUIRED_STATES

        valid_state_values = {s.value for s in SystemState}
        for state_str in _HUMAN_APPROVAL_REQUIRED_STATES:
            assert state_str in valid_state_values, (
                f"'{state_str}' in _HUMAN_APPROVAL_REQUIRED_STATES is not a valid "
                f"SystemState value. Valid values: {valid_state_values}"
            )

    def test_escalate_is_in_approval_required_states(self) -> None:
        """ESCALATE must always require human approval - safety invariant."""
        from hazard_assessment.policy.approval import _HUMAN_APPROVAL_REQUIRED_STATES

        assert "ESCALATE" in _HUMAN_APPROVAL_REQUIRED_STATES


class TestProhibitedTermPolicyMatchesEnforcement:
    """The P2 policy term list must match the terms the guardrail actually blocks.

    permissions.yaml documents the reserved-language policy but the scanner in
    policy/guardrails.py is what enforces it. If the two drift, the policy
    record misstates what the system blocks. This pins them together.
    """

    def test_p2_terms_match_guardrail(self) -> None:
        from hazard_assessment.policy.guardrails import PROHIBITED_TERMS

        matrix = load_permission_matrix()
        p2 = next(a for a in matrix["prohibited_actions"] if a["id"] == "P2")
        assert set(p2["prohibited_terms"]) == set(PROHIBITED_TERMS)
