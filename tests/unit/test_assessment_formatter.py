"""Tests for the Assessment Formatter.

Verifies format_human_decision() status transitions, decision trace
appending, field preservation, guardrail defense-in-depth, and
format_abstain() typed ABSTAIN envelope construction.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hazard_assessment.agents.assessment_formatter import (
    format_abstain,
    format_human_decision,
)
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.envelope import DecisionStep, StepResult
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.human_decision import HumanDecision, ReviewDecision

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_provisional_assessment(
    *,
    event_id=None,
    tier: int = 1,
    summary: str | None = None,
) -> FinalAssessment:
    """Build a PROVISIONAL FinalAssessment for testing."""
    return FinalAssessment(
        producer="test_report_agent",
        event_id=event_id,
        status=AssessmentStatus.PROVISIONAL,
        report_tier=tier,
        summary=summary or (
            "TECHNICAL BRIEF\n"
            f"{NON_AUTHORITATIVE_DISCLAIMER}\n\n"
            "Test assessment summary."
        ),
        uncertainty=UncertaintyInfo(
            confidence_level=ConfidenceLevel.MODERATE,
            key_uncertainties=["Station coverage limited"],
        ),
        provenance_bundle_id=uuid4(),
        decision_trace=[
            DecisionStep(
                step="template_rendering",
                result=StepResult.PASS,
                evidence="Tier 1 template rendered successfully",
            ),
            DecisionStep(
                step="guardrail_scan",
                result=StepResult.PASS,
                evidence="No prohibited NOAA alert terminology detected",
            ),
        ],
    )


def _make_human_decision(
    decision: ReviewDecision = ReviewDecision.APPROVE,
    reviewer_id: str = "alice",
    reason: str = "Assessment looks correct.",
) -> HumanDecision:
    """Build a HumanDecision for testing."""
    return HumanDecision(
        producer="human_review_node",
        reviewer_id=reviewer_id,
        decision=decision,
        decision_reason=reason,
        decided_at_utc=datetime.now(UTC),
        escalation_packet_id=uuid4(),
    )


# ===================================================================
# TestFormatHumanDecisionApprove
# ===================================================================


class TestFormatHumanDecisionApprove:
    """APPROVE transitions PROVISIONAL -> APPROVED_INTERNAL."""

    def test_approve_sets_approved_internal(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.status == AssessmentStatus.APPROVED_INTERNAL

    def test_approve_adds_decision_trace_step(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        # Original had 2 steps; result should have 3
        assert len(result.decision_trace) == len(fa.decision_trace) + 1
        last = result.decision_trace[-1]
        assert last.step == "human_review"
        assert last.result == StepResult.PASS
        assert "APPROVE" in last.evidence
        assert decision.reviewer_id in last.evidence

    def test_approve_preserves_summary(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.summary == fa.summary

    def test_approve_preserves_tier(self) -> None:
        fa = _make_provisional_assessment(tier=2)
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.report_tier == 2

    def test_approve_preserves_uncertainty(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.uncertainty.confidence_level == fa.uncertainty.confidence_level
        assert result.uncertainty.key_uncertainties == fa.uncertainty.key_uncertainties

    def test_approve_preserves_provenance(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.provenance_bundle_id == fa.provenance_bundle_id

    def test_approve_preserves_handoff_id(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        result = format_human_decision(fa, decision)
        assert result.handoff_id == fa.handoff_id


# ===================================================================
# TestFormatHumanDecisionRejectDefer
# ===================================================================


class TestFormatHumanDecisionRejectDefer:
    """REJECT/DEFER keep PROVISIONAL status with INFO trace step."""

    def test_reject_stays_provisional(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.REJECT)
        result = format_human_decision(fa, decision)
        assert result.status == AssessmentStatus.PROVISIONAL

    def test_reject_adds_info_trace_step(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(
            decision=ReviewDecision.REJECT,
            reason="Coastal proxy values look anomalous.",
        )
        result = format_human_decision(fa, decision)
        last = result.decision_trace[-1]
        assert last.step == "human_review"
        assert last.result == StepResult.INFO
        assert "REJECT" in last.evidence

    def test_defer_stays_provisional(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(decision=ReviewDecision.DEFER)
        result = format_human_decision(fa, decision)
        assert result.status == AssessmentStatus.PROVISIONAL

    def test_defer_adds_info_trace_step(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(
            decision=ReviewDecision.DEFER,
            reason="Awaiting additional DART data.",
        )
        result = format_human_decision(fa, decision)
        last = result.decision_trace[-1]
        assert last.step == "human_review"
        assert last.result == StepResult.INFO
        assert "DEFER" in last.evidence


# ===================================================================
# TestFormatHumanDecisionErrors
# ===================================================================


class TestFormatHumanDecisionErrors:
    """Guard clauses and defense-in-depth."""

    def test_non_provisional_raises_value_error(self) -> None:
        # Construct an ABSTAIN assessment via format_abstain
        abstain_fa = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        decision = _make_human_decision(decision=ReviewDecision.APPROVE)
        with pytest.raises(ValueError, match="PROVISIONAL"):
            format_human_decision(abstain_fa, decision)

    def test_prohibited_term_in_reason_raises_value_error(self) -> None:
        fa = _make_provisional_assessment()
        decision = _make_human_decision(
            decision=ReviewDecision.APPROVE,
            reason="The tsunami Warning was appropriate.",
        )
        with pytest.raises(ValueError, match="prohibited"):
            format_human_decision(fa, decision)


# ===================================================================
# TestFormatAbstain
# ===================================================================


class TestFormatAbstain:
    """format_abstain() produces a typed FinalAssessment(status=ABSTAIN)."""

    def test_produces_final_assessment(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Model fit quality below threshold",
        )
        assert isinstance(result, FinalAssessment)

    def test_status_is_abstain(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        assert result.status == AssessmentStatus.ABSTAIN

    def test_tier_is_1(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        assert result.report_tier == 1

    def test_confidence_is_low(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        assert result.uncertainty.confidence_level == ConfidenceLevel.LOW

    def test_summary_passes_guardrails(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Model fit quality below threshold",
        )
        scan_result = scan_text(result.summary)
        assert scan_result.passed

    def test_summary_contains_disclaimer(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        assert NON_AUTHORITATIVE_DISCLAIMER in result.summary

    def test_summary_contains_reason(self) -> None:
        reason = "RMSE 6.0 cm exceeds threshold"
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason=reason,
        )
        assert reason in result.summary

    def test_provenance_auto_generated(self) -> None:
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        assert result.provenance_bundle_id is not None

    def test_decision_trace_populated(self) -> None:
        reason = "Model fit quality below threshold"
        result = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason=reason,
        )
        assert len(result.decision_trace) == 1
        step = result.decision_trace[0]
        assert step.step == "verification_outcome"
        assert step.result == StepResult.FAIL
        assert step.evidence == reason

    def test_prohibited_term_in_reason_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="prohibited"):
            format_abstain(
                event_id=None,
                fsm_state="ASSESS",
                abstain_reason="The tsunami Warning was appropriate.",
            )
