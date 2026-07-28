"""Integration tests for pipeline orchestration and handoff schemas.

Tests full simulated event paths through the pipeline:
- PASS path (anomaly -> assess -> scenario -> verify -> report -> human_review)
- FAIL path (verify FAIL -> abstain -> human_review)
- Insufficient evidence (below ASSESS threshold)
- De-escalation paths
- Multi-step escalation
- Schema validation at every handoff
- Audit trail completeness

Note: The FSM transitions one state per evaluate_anomaly_score() call.
To reach ASSESS, start FSM in INVESTIGATE. To reach ESCALATE, start in ASSESS.
"""

from __future__ import annotations

from hazard_assessment.agents.report_agent import ReportAgent
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.orchestrator.nodes import run_pipeline_sync
from hazard_assessment.orchestrator.pipeline import PipelineState
from hazard_assessment.orchestrator.states import SystemState
from hazard_assessment.schemas.anomaly import AnomalyAssessment
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    FinalAssessment,
)
from hazard_assessment.schemas.human_decision import ReviewDecision
from hazard_assessment.schemas.verification import VerificationOutcome
from tests.integration.conftest import (
    make_anomaly_dict,
    make_fsm_in_state,
    make_human_decision_dict,
    make_scenario_dict,
    make_verification_dict,
)


class TestFullPassPath:
    """Test 1: Full PASS path with schema validation at every handoff."""

    def test_full_pass_path_with_schema_validation(self) -> None:
        """FSM in INVESTIGATE, score >= T2 -> ASSESS -> scenario -> verify(PASS)
        -> report -> human_review(APPROVE) -> APPROVED_INTERNAL."""
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }

        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )

        # FSM should have reached ASSESS
        assert result["fsm_state"] == "ASSESS"

        # AnomalyAssessment annotated by orchestrate_node
        aa = result["anomaly_assessment"]
        assert aa["current_state"] == "ASSESS"
        assert aa["state_changed"] is True
        AnomalyAssessment.model_validate(aa)

        # FinalAssessment should be APPROVED_INTERNAL
        fa = result["final_assessment"]
        assert fa["status"] == "APPROVED_INTERNAL"
        validated_fa = FinalAssessment.model_validate(fa)
        assert validated_fa.report_tier == 1

        # Audit trail completeness
        entry_types = {e.event_type for e in audit.get_entries()}
        assert "state_transition" in entry_types
        assert "verification_complete" in entry_types
        assert "report_generated" in entry_types
        assert "assessment_formatted" in entry_types


class TestFullFailPath:
    """Test 2: Verification FAIL -> ABSTAIN path."""

    def test_full_fail_path_verification_to_abstain(self) -> None:
        """FSM in INVESTIGATE -> ASSESS -> verify(FAIL) -> abstain."""
        fsm = make_fsm_in_state(SystemState.INVESTIGATE)
        agent = ReportAgent()
        audit = AuditLogger()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.FAIL),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }

        result = run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )

        assert result.get("abstain_triggered") is True
        fa = result["final_assessment"]
        assert fa["status"] == "ABSTAIN"
        # Backwards-compat keys (outcome, pipeline_status) are injected
        # post-model_dump by abstain_node - strip them for schema validation
        fa_clean = {k: v for k, v in fa.items() if k not in ("outcome", "pipeline_status")}
        validated = FinalAssessment.model_validate(fa_clean)
        assert validated.status == AssessmentStatus.ABSTAIN

        entry_types = {e.event_type for e in audit.get_entries()}
        assert "abstain_triggered" in entry_types


class TestInsufficientEvidence:
    """Test 3: Below ASSESS threshold -> insufficient evidence."""

    def test_insufficient_evidence_below_assess(self) -> None:
        """anomaly(0.20) -> stays MONITOR -> insufficient_evidence."""
        fsm = make_fsm_in_state(SystemState.MONITOR)
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.20),
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        fa = result["final_assessment"]
        assert fa["outcome"] == "insufficient_evidence"
        assert result.get("abstain_triggered") is not True


class TestEscalateWithHumanReject:
    """Test 4: ESCALATE -> verify(PASS) -> report -> REJECT -> PROVISIONAL."""

    def test_escalate_with_human_reject(self) -> None:
        fsm = make_fsm_in_state(SystemState.ASSESS)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.90),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
            "human_decision": make_human_decision_dict(ReviewDecision.REJECT),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        assert result["fsm_state"] == "ESCALATE"
        fa = result["final_assessment"]
        validated = FinalAssessment.model_validate(fa)
        assert validated.status == AssessmentStatus.PROVISIONAL


class TestEscalateWithHumanDefer:
    """Test 5: Same as Test 4 with DEFER -> PROVISIONAL."""

    def test_escalate_with_human_defer(self) -> None:
        fsm = make_fsm_in_state(SystemState.ASSESS)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.90),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
            "human_decision": make_human_decision_dict(
                ReviewDecision.DEFER, reason="Need more data"
            ),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        fa = result["final_assessment"]
        validated = FinalAssessment.model_validate(fa)
        assert validated.status == AssessmentStatus.PROVISIONAL


class TestDeescalationAssessToInvestigate:
    """Test 6: ASSESS -> lower score -> INVESTIGATE -> insufficient_evidence."""

    def test_deescalation_assess_to_investigate(self) -> None:
        """Start in ASSESS, score < T2 -> INVESTIGATE."""
        fsm = make_fsm_in_state(SystemState.ASSESS)
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.50),
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["final_assessment"]["outcome"] == "insufficient_evidence"


class TestDeescalationInvestigateToMonitor:
    """Test 7: INVESTIGATE -> low score -> MONITOR."""

    def test_deescalation_investigate_to_monitor(self) -> None:
        fsm = make_fsm_in_state(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": make_anomaly_dict(0.20)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["final_assessment"]["outcome"] == "insufficient_evidence"


class TestMissingVerificationFailClosed:
    """Test 8: ASSESS path, no verification -> fail-closed ABSTAIN."""

    def test_missing_verification_failclosed_abstain(self) -> None:
        fsm = make_fsm_in_state(SystemState.INVESTIGATE)
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            # No verification_result -> route_after_verify returns ABSTAIN
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is True


class TestPassWithConcernsTakesReportPath:
    """Test 9: PASS_WITH_CONCERNS -> report path, not abstain."""

    def test_pass_with_concerns_takes_report_path(self) -> None:
        fsm = make_fsm_in_state(SystemState.INVESTIGATE)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(
                VerificationOutcome.PASS_WITH_CONCERNS
            ),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        assert result.get("abstain_triggered") is not True
        fa = result["final_assessment"]
        validated = FinalAssessment.model_validate(fa)
        assert validated.status == AssessmentStatus.PROVISIONAL
        assert validated.report_tier == 1


class TestMultiStepEscalation:
    """Test 10: 4 runs across same FSM: MONITOR -> INVESTIGATE -> ASSESS -> ESCALATE."""

    def test_multi_step_escalation_across_runs(self) -> None:
        fsm = make_fsm_in_state(SystemState.MONITOR)
        agent = ReportAgent()

        # Run 1: MONITOR -> INVESTIGATE (score >= T1=0.35)
        state1: PipelineState = {"anomaly_assessment": make_anomaly_dict(0.40)}
        result1 = run_pipeline_sync(state1, fsm=fsm)
        assert result1["fsm_state"] == "INVESTIGATE"
        AnomalyAssessment.model_validate(result1["anomaly_assessment"])

        # Run 2: INVESTIGATE -> ASSESS (score >= T2=0.60)
        state2: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
        }
        result2 = run_pipeline_sync(state2, fsm=fsm, report_agent=agent)
        assert result2["fsm_state"] == "ASSESS"

        # Run 3: ASSESS -> ESCALATE (score >= T3=0.85)
        state3: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.90),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
        }
        result3 = run_pipeline_sync(state3, fsm=fsm, report_agent=agent)
        assert result3["fsm_state"] == "ESCALATE"
        fa = result3["final_assessment"]
        FinalAssessment.model_validate(fa)


class TestHandoffSchemaRoundtrip:
    """Test 11: Build each schema, serialize, feed through pipeline, validate."""

    def test_handoff_schema_roundtrip_all_transitions(self) -> None:
        """Schema roundtrip: construct -> model_dump -> pipeline -> model_validate."""
        fsm = make_fsm_in_state(SystemState.INVESTIGATE)
        agent = ReportAgent()

        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        # Every handoff schema round-trips through pipeline
        AnomalyAssessment.model_validate(result["anomaly_assessment"])
        fa = FinalAssessment.model_validate(result["final_assessment"])

        assert fa.status == AssessmentStatus.APPROVED_INTERNAL
        assert fa.report_tier == 1
        assert fa.provenance_bundle_id is not None


class TestAuditTrailCompleteness:
    """Test 12: Full PASS path, verify all audit entry types."""

    def test_audit_trail_completeness(self) -> None:
        audit = AuditLogger()
        fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": make_anomaly_dict(0.65),
            "scenario_assessment": make_scenario_dict(),
            "verification_result": make_verification_dict(VerificationOutcome.PASS),
            "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
        }

        run_pipeline_sync(
            state, fsm=fsm, report_agent=agent, audit_logger=audit,
        )

        entry_types = [e.event_type for e in audit.get_entries()]
        assert "state_transition" in entry_types
        assert "verification_complete" in entry_types
        assert "report_generated" in entry_types
        assert "assessment_formatted" in entry_types
        assert len(entry_types) >= 4


class TestThreeStepDeescalation:
    """Test 13: ASSESS -> INVESTIGATE -> MONITOR across 3 runs."""

    def test_three_step_deescalation(self) -> None:
        fsm = make_fsm_in_state(SystemState.ASSESS)

        # Run 1: ASSESS -> INVESTIGATE (score < T2=0.60)
        state1: PipelineState = {"anomaly_assessment": make_anomaly_dict(0.50)}
        result1 = run_pipeline_sync(state1, fsm=fsm)
        assert result1["fsm_state"] == "INVESTIGATE"

        # Run 2: INVESTIGATE -> MONITOR (score < T1=0.35)
        state2: PipelineState = {"anomaly_assessment": make_anomaly_dict(0.20)}
        result2 = run_pipeline_sync(state2, fsm=fsm)
        assert result2["fsm_state"] == "MONITOR"
        assert result2["final_assessment"]["outcome"] == "insufficient_evidence"
