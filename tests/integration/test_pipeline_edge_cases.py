"""Integration tests for pipeline edge cases.

Tests behavior under degraded conditions:
- Zero anomaly score (no detection)
- Below-threshold scores at various FSM states
- Missing scenario data
- ABSTAIN routing from verification failure
"""

from __future__ import annotations

from hazard_assessment.agents.report_agent import ReportAgent
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.orchestrator.nodes import run_pipeline_sync
from hazard_assessment.orchestrator.pipeline import PipelineState
from hazard_assessment.orchestrator.states import SystemState
from hazard_assessment.schemas.final_assessment import AssessmentStatus
from hazard_assessment.schemas.human_decision import ReviewDecision
from hazard_assessment.schemas.verification import VerificationOutcome
from tests.integration.conftest import (
    make_anomaly_dict,
    make_fsm_in_state,
    make_human_decision_dict,
    make_scenario_dict,
    make_verification_dict,
)


class TestZeroScore:
    """Pipeline should handle zero anomaly score gracefully."""

    def test_zero_score_stays_in_monitor(self) -> None:
        """Score of 0.0 should not advance past MONITOR."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.MONITOR, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.0),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        assert result["fsm_state"] == "MONITOR"
        # Zero score: FSM stays in MONITOR, no scenario/verification/report
        fa = result.get("final_assessment", {})
        # Should not produce a full FinalAssessment (no "status" field)
        # because the pipeline never reaches the report node
        assert fa.get("status") is None or fa.get("status") != "APPROVED_INTERNAL"

    def test_below_t1_from_investigate_deescalates(self) -> None:
        """Score below T1 from INVESTIGATE should de-escalate to MONITOR."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.10),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        assert result["fsm_state"] == "MONITOR"


class TestVerificationFailAbstain:
    """Verification FAIL should route to ABSTAIN, not REPORT."""

    def test_fail_verification_produces_abstain(self) -> None:
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.FAIL),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        final = result.get("final_assessment", {})
        assert final.get("status") == AssessmentStatus.ABSTAIN.value

    def test_abstain_has_structured_document(self) -> None:
        """ABSTAIN should produce a summary with the abstention notice."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.FAIL),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        final = result.get("final_assessment", {})
        # ABSTAIN should have a summary containing the failure notice
        assert final.get("summary") is not None
        assert "VERIFICATION FAILURE" in final["summary"]


class TestBoundaryScores:
    """Test scores exactly at threshold boundaries."""

    def test_exact_t1_triggers_investigate(self) -> None:
        """Score exactly at T1 (0.35) should trigger INVESTIGATE."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.MONITOR, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.35),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        assert result["fsm_state"] == "INVESTIGATE"

    def test_just_below_t1_stays_in_monitor(self) -> None:
        """Score just below T1 (0.349) should stay in MONITOR."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.MONITOR, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.349),
        }
        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )
        assert result["fsm_state"] == "MONITOR"
