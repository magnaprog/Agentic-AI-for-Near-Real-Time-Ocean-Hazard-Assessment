"""Unit tests for pipeline node functions.

Tests cover:
- orchestrate_node: FSM threshold transitions, de-escalation, state annotation
- AnomalyAssessment handoff schema carries current_state and state_changed
- final_node: insufficient-evidence, abstain, and incomplete_report outputs
- scenario_node, verify_node, abstain_node, report_node, human_review_node
- run_pipeline_sync: end-to-end pipeline execution
- Audit logging on state transitions, verification, and ABSTAIN
- report_node with ReportAgent generates FinalAssessment
- abstain_node typed envelope, human_review_node formatting
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from hazard_assessment.agents.report_agent import ReportAgent
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.orchestrator.nodes import (
    abstain_node,
    final_node,
    human_review_node,
    orchestrate_node,
    report_node,
    run_pipeline_sync,
    verify_node,
)
from hazard_assessment.orchestrator.pipeline import PipelineState
from hazard_assessment.orchestrator.states import (
    FSMOrchestrator,
    SystemState,
    ThresholdConfig,
)
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER
from hazard_assessment.schemas.envelope import DecisionStep, StepResult
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.human_decision import HumanDecision, ReviewDecision
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)

# Fixed values for deterministic test fixtures
_FIXED_DECISION_TIME = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
_FIXED_ESCALATION_PACKET_ID = UUID("b2c3d4e5-f6a7-8901-bcde-f12345678901")

THRESHOLDS = ThresholdConfig(basin="pacific", t1=0.35, t2=0.60, t3=0.85)

SAMPLE_ASSESSMENT = {
    "type": "AnomalyAssessment",
    "schema_version": "1.0",
    "producer": "anomaly_agent",
    "anomaly_score": 0.40,
    "score_components": {"threshold": 0.5, "statistical": 0.3, "ml": None},
    "triggering_stations": ["21413"],
    "spatial_confirmations": [],
    "seismic_quiet": False,
    "meteotsunami_score": 0.0,
    "stations_offline": [],
    "coverage_note": "",
    "reasoning_trace": "test",
    "current_state": "",
    "state_changed": False,
}


def _make_fsm(
    initial_state: SystemState = SystemState.MONITOR,
    thresholds: ThresholdConfig = THRESHOLDS,
    audit_writer: AuditLogger | None = None,
) -> FSMOrchestrator:
    """Create an FSM in the specified state with an event context."""
    fsm = FSMOrchestrator(thresholds=thresholds, audit_writer=audit_writer)
    if initial_state != SystemState.IDLE:
        # Transition IDLE -> MONITOR via seismic trigger
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
        )
        if initial_state == SystemState.INVESTIGATE:
            fsm.evaluate_anomaly_score(0.40)  # >= T1 (0.35)
        elif initial_state == SystemState.ASSESS:
            fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
            fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        elif initial_state == SystemState.ESCALATE:
            fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
            fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
            fsm.evaluate_anomaly_score(0.90)  # -> ESCALATE
    return fsm


def _assessment_with_score(score: float) -> dict:
    """Return a sample anomaly assessment dict with the given score."""
    a = dict(SAMPLE_ASSESSMENT)
    a["anomaly_score"] = score
    return a


class TestOrchestrateNode:
    """Wire deterministic FSM to real anomaly scores."""

    def test_monitor_to_investigate_on_t1(self) -> None:
        """Score >= T1 triggers MONITOR -> INVESTIGATE."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is True
        assert fsm.state == SystemState.INVESTIGATE

    def test_investigate_to_assess_on_t2(self) -> None:
        """Score >= T2 triggers INVESTIGATE -> ASSESS."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.65)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "ASSESS"
        assert result["state_changed"] is True

    def test_assess_to_escalate_on_t3(self) -> None:
        """Score >= T3 triggers ASSESS -> ESCALATE."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.90)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "ESCALATE"
        assert result["state_changed"] is True

    def test_investigate_deescalation_below_t1(self) -> None:
        """Score < T1 in INVESTIGATE triggers de-escalation to MONITOR."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is True

    def test_assess_deescalation_below_t2(self) -> None:
        """Score < T2 in ASSESS triggers de-escalation to INVESTIGATE."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.50)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is True

    def test_no_transition_within_threshold_band(self) -> None:
        """Score between T1 and T2 in INVESTIGATE doesn't transition."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.50)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is False

    def test_no_transition_monitor_below_t1(self) -> None:
        """Score below T1 in MONITOR doesn't transition."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is False

    def test_missing_assessment_returns_current_state(self) -> None:
        """Missing anomaly_assessment keeps FSM state unchanged."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is False


class TestOrchestrateNodeAnnotation:
    """AnomalyAssessment handoff schema annotation."""

    def test_assessment_annotated_with_fsm_state(self) -> None:
        """The orchestrate node annotates current_state on the assessment."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        result = orchestrate_node(state, fsm=fsm)

        assessment = result["anomaly_assessment"]
        assert assessment["current_state"] == "INVESTIGATE"
        assert assessment["state_changed"] is True

    def test_assessment_annotated_no_transition(self) -> None:
        """When no transition occurs, state_changed is False."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        result = orchestrate_node(state, fsm=fsm)

        assessment = result["anomaly_assessment"]
        assert assessment["current_state"] == "MONITOR"
        assert assessment["state_changed"] is False

    def test_assessment_not_mutated_in_place(self) -> None:
        """The original assessment dict is not mutated."""
        fsm = _make_fsm(SystemState.MONITOR)
        original = _assessment_with_score(0.40)
        state: PipelineState = {"anomaly_assessment": original}

        orchestrate_node(state, fsm=fsm)

        # Original should still have empty current_state
        assert original["current_state"] == ""

    def test_event_id_propagated_from_fsm(self) -> None:
        """Event ID is propagated from FSM context to pipeline state."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        result = orchestrate_node(state, fsm=fsm)

        assert "event_id" in result
        assert result["event_id"] != ""


class TestOrchestrateNodeAudit:
    """Audit logging on state transitions."""

    def test_audit_entry_on_transition(self) -> None:
        """A transition produces an audit entry."""
        audit = AuditLogger()
        fsm = _make_fsm(SystemState.MONITOR, audit_writer=audit)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        orchestrate_node(state, fsm=fsm, audit_logger=audit)

        assert audit.count >= 1
        entries = audit.get_entries(event_type="state_transition")
        assert len(entries) >= 1
        entry = entries[-1]
        assert entry.data["from_state"] == "MONITOR"
        assert entry.data["to_state"] == "INVESTIGATE"

    def test_no_audit_entry_without_transition(self) -> None:
        """No transition means no audit entry from orchestrate_node."""
        fsm = _make_fsm(SystemState.MONITOR)
        audit = AuditLogger()
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        orchestrate_node(state, fsm=fsm, audit_logger=audit)

        entries = audit.get_entries(event_type="state_transition")
        assert len(entries) == 0


class TestFinalNode:
    """Test the final pipeline node."""

    def test_insufficient_evidence_output(self) -> None:
        """Non-ASSESS states produce insufficient_evidence."""
        state: PipelineState = {
            "fsm_state": "MONITOR",
            "anomaly_assessment": _assessment_with_score(0.20),
        }

        result = final_node(state)

        assert "final_assessment" in result
        fa = result["final_assessment"]
        assert fa["outcome"] == "insufficient_evidence"
        assert fa["fsm_state"] == "MONITOR"

    def test_preserves_existing_final_assessment(self) -> None:
        """If final_assessment already exists, final_node doesn't overwrite."""
        existing = {"outcome": "approved", "detail": "Human approved"}
        state: PipelineState = {
            "fsm_state": "ESCALATE",
            "final_assessment": existing,
        }

        result = final_node(state)

        # Should return empty update (preserving existing)
        assert result == {}

    def test_missing_state_handled(self) -> None:
        """Empty state produces a reasonable default."""
        state: PipelineState = {}

        result = final_node(state)

        assert result["final_assessment"]["outcome"] == "insufficient_evidence"


class TestRunPipelineSync:
    """End-to-end pipeline execution tests."""

    def test_monitor_score_below_t1_goes_to_final(self) -> None:
        """Low score in MONITOR routes to final with insufficient evidence."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is False
        assert result["final_assessment"]["outcome"] == "insufficient_evidence"

    def test_monitor_to_investigate_pipeline(self) -> None:
        """Score >= T1 transitions to INVESTIGATE and routes to final."""
        fsm = _make_fsm(SystemState.MONITOR)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is True
        assert result["final_assessment"]["outcome"] == "insufficient_evidence"

    def test_assess_without_verification_triggers_abstain(self) -> None:
        """ASSESS state without verification_result -> ABSTAIN (fail-closed)."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.65)}

        result = run_pipeline_sync(state, fsm=fsm)

        # Score between T2-T3 keeps ASSESS
        assert result["fsm_state"] == "ASSESS"
        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_escalate_without_verification_triggers_abstain(self) -> None:
        """ESCALATE without verification_result -> ABSTAIN (fail-closed)."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.90)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "ESCALATE"
        assert result["state_changed"] is True
        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_full_escalation_path(self) -> None:
        """MONITOR -> INVESTIGATE -> ASSESS -> ESCALATE across multiple runs."""
        audit = AuditLogger()
        fsm = _make_fsm(SystemState.MONITOR, audit_writer=audit)

        _pass_verification = {
            "overall": VerificationOutcome.PASS,
            "abstain_required": False,
        }

        # Step 1: MONITOR -> INVESTIGATE (score >= T1)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)
        assert result["fsm_state"] == "INVESTIGATE"

        # Step 2: INVESTIGATE -> ASSESS (score >= T2)
        # Provide verification_result so pipeline takes report path, not abstain
        state = {
            "anomaly_assessment": _assessment_with_score(0.65),
            "verification_result": _pass_verification,
        }
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)
        assert result["fsm_state"] == "ASSESS"
        assert result.get("abstain_triggered") is not True

        # Step 3: ASSESS -> ESCALATE (score >= T3)
        state = {
            "anomaly_assessment": _assessment_with_score(0.90),
            "verification_result": _pass_verification,
        }
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)
        assert result["fsm_state"] == "ESCALATE"
        assert result["state_changed"] is True
        assert result.get("abstain_triggered") is not True
        assert result["final_assessment"]["outcome"] == "verified_pending_report"

        # Verify 4 transitions: IDLE->MONITOR (setup) + 3 pipeline transitions
        transitions = audit.get_entries(event_type="state_transition")
        assert len(transitions) == 4

    def test_deescalation_path(self) -> None:
        """INVESTIGATE -> MONITOR when score drops below T1."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.20)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is True
        assert result["final_assessment"]["outcome"] == "insufficient_evidence"

    def test_assess_deescalation_to_investigate(self) -> None:
        """ASSESS -> INVESTIGATE when score drops below T2."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.50)}

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is True

    def test_pipeline_with_audit_logging(self) -> None:
        """Pipeline produces audit entries on transitions."""
        audit = AuditLogger()
        fsm = _make_fsm(SystemState.MONITOR, audit_writer=audit)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.40)}

        run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

        entries = audit.get_entries(event_type="state_transition")
        # 2 entries: IDLE->MONITOR (from _make_fsm setup) + MONITOR->INVESTIGATE
        assert len(entries) == 2
        assert entries[-1].data["from_state"] == "MONITOR"
        assert entries[-1].data["to_state"] == "INVESTIGATE"


class TestSeismicOverrideIntegration:
    """Test seismic override path through the pipeline."""

    def test_seismic_override_escalation(self) -> None:
        """M7.5+ with DART confirmation bypasses T3 in ASSESS."""
        fsm = _make_fsm(SystemState.ASSESS)
        # Upgrade magnitude and set DART confirmation
        fsm._event_context.seismic_magnitude = 8.0
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21413"])

        # Score below T3 but seismic override should escalate
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.70)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "ESCALATE"
        assert result["state_changed"] is True
        assessment = result["anomaly_assessment"]
        assert assessment["current_state"] == "ESCALATE"


class TestBidirectionalTransitions:
    """Verify all bidirectional de-escalation transitions work through pipeline."""

    def test_investigate_to_monitor_boundary(self) -> None:
        """Score exactly at T1 stays in INVESTIGATE (>= T1 went up)."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.35)}

        result = orchestrate_node(state, fsm=fsm)

        # score == T1 means >= T1 is False for de-escalation (score < T1 required)
        # But evaluate_anomaly_score checks `score < t1` for de-escalation
        # 0.35 < 0.35 is False, so no transition
        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is False

    def test_investigate_just_below_t1(self) -> None:
        """Score just below T1 triggers de-escalation."""
        fsm = _make_fsm(SystemState.INVESTIGATE)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.349)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is True

    def test_assess_to_investigate_boundary(self) -> None:
        """Score exactly at T2 stays in ASSESS."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.60)}

        result = orchestrate_node(state, fsm=fsm)

        # 0.60 < 0.60 is False, so no de-escalation
        assert result["fsm_state"] == "ASSESS"
        assert result["state_changed"] is False

    def test_assess_just_below_t2(self) -> None:
        """Score just below T2 triggers de-escalation."""
        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {"anomaly_assessment": _assessment_with_score(0.599)}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "INVESTIGATE"
        assert result["state_changed"] is True


# ---------------------------------------------------------------------------
# New node functions and ABSTAIN enforcement
# ---------------------------------------------------------------------------

ASSESS_ASSESSMENT = {
    "type": "AnomalyAssessment",
    "schema_version": "1.0",
    "producer": "anomaly_agent",
    "anomaly_score": 0.65,
    "score_components": {"threshold": 0.5, "statistical": 0.3, "ml": None},
    "triggering_stations": ["21413"],
    "spatial_confirmations": [],
    "seismic_quiet": False,
    "meteotsunami_score": 0.0,
    "stations_offline": [],
    "coverage_note": "",
    "reasoning_trace": "test",
    "current_state": "",
    "state_changed": False,
}


def _make_assess_fsm() -> FSMOrchestrator:
    """Create an FSM in ASSESS state for E6 tests."""
    fsm = FSMOrchestrator(thresholds=THRESHOLDS)
    fsm.evaluate_seismic_trigger(
        magnitude=7.0,
        region="pacific_nw",
        epicenter_lat=46.0,
        epicenter_lon=-130.0,
        tsunamigenic_zones={"pacific_nw"},
    )
    fsm.evaluate_anomaly_score(0.40)  # MONITOR -> INVESTIGATE
    fsm.evaluate_anomaly_score(0.65)  # INVESTIGATE -> ASSESS
    return fsm


class TestVerifyNode:
    """verify_node audit logging."""

    def test_returns_empty_update(self) -> None:
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }
        assert verify_node(state) == {}

    def test_audit_logging(self) -> None:
        audit = AuditLogger()
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }

        verify_node(state, audit_logger=audit)

        entries = audit.get_entries(event_type="verification_complete")
        assert len(entries) == 1
        assert entries[0].data["overall"] == VerificationOutcome.PASS
        assert entries[0].data["abstain_required"] is False

    def test_no_audit_without_verification_result(self) -> None:
        audit = AuditLogger()
        state: PipelineState = {}

        verify_node(state, audit_logger=audit)

        assert audit.count == 0


class TestReportNode:
    """report_node sets incomplete status."""

    def test_sets_incomplete_e7_status(self) -> None:
        state: PipelineState = {}
        result = report_node(state)
        assert result["pipeline_status"] == "incomplete_report"


def _make_provisional_fa_dict() -> dict:
    """Build a PROVISIONAL FinalAssessment dict for pipeline state."""
    fa = FinalAssessment(
        producer="test_report_agent",
        status=AssessmentStatus.PROVISIONAL,
        report_tier=1,
        summary=(
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
        ],
    )
    return fa.model_dump()


def _make_human_decision_dict(
    decision: ReviewDecision = ReviewDecision.APPROVE,
    reason: str = "Assessment looks correct.",
) -> dict:
    """Build a HumanDecision dict for pipeline state."""
    hd = HumanDecision(
        producer="human_review_node",
        reviewer_id="alice",
        decision=decision,
        decision_reason=reason,
        decided_at_utc=_FIXED_DECISION_TIME,
        escalation_packet_id=_FIXED_ESCALATION_PACKET_ID,
    )
    return hd.model_dump()


class TestHumanReviewNode:
    """human_review_node formatting and pass-through."""

    def test_returns_empty_update(self) -> None:
        state: PipelineState = {}
        assert human_review_node(state) == {}

    def test_approve_updates_status(self) -> None:
        state: PipelineState = {
            "final_assessment": _make_provisional_fa_dict(),
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }
        result = human_review_node(state)
        assert result["final_assessment"]["status"] == "APPROVED_INTERNAL"

    def test_reject_keeps_provisional(self) -> None:
        state: PipelineState = {
            "final_assessment": _make_provisional_fa_dict(),
            "human_decision": _make_human_decision_dict(ReviewDecision.REJECT),
        }
        result = human_review_node(state)
        assert result["final_assessment"]["status"] == "PROVISIONAL"

    def test_defer_keeps_provisional(self) -> None:
        state: PipelineState = {
            "final_assessment": _make_provisional_fa_dict(),
            "human_decision": _make_human_decision_dict(ReviewDecision.DEFER),
        }
        result = human_review_node(state)
        assert result["final_assessment"]["status"] == "PROVISIONAL"

    def test_no_assessment_is_passthrough(self) -> None:
        state: PipelineState = {
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }
        assert human_review_node(state) == {}

    def test_abstain_assessment_is_passthrough(self) -> None:
        """ABSTAIN status -> no formatting (not PROVISIONAL)."""
        from hazard_assessment.agents.assessment_formatter import format_abstain

        fa = format_abstain(
            event_id=None,
            fsm_state="ASSESS",
            abstain_reason="Verification failed",
        )
        state: PipelineState = {
            "final_assessment": fa.model_dump(),
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }
        assert human_review_node(state) == {}

    def test_approve_audit_logging(self) -> None:
        """APPROVE produces an audit entry with decision metadata."""
        audit = AuditLogger()
        state: PipelineState = {
            "final_assessment": _make_provisional_fa_dict(),
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }

        human_review_node(state, audit_logger=audit)

        entries = audit.get_entries(event_type="assessment_formatted")
        assert len(entries) == 1
        assert entries[0].data["decision"] == "APPROVE"
        assert entries[0].data["reviewer_id"] == "alice"
        assert entries[0].data["new_status"] == "APPROVED_INTERNAL"


class TestAbstainNode:
    """abstain_node flag setting and audit logging."""

    def test_sets_abstain_flags(self) -> None:
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "coverage: n_stations=1",
            },
        }

        result = abstain_node(state)

        assert result["abstain_triggered"] is True
        assert result["abstain_reason"] == "coverage: n_stations=1"
        # abstain_node now produces typed FinalAssessment
        assert "final_assessment" in result
        assert result["final_assessment"]["status"] == "ABSTAIN"
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_default_reason_when_missing(self) -> None:
        state: PipelineState = {}

        result = abstain_node(state)

        assert result["abstain_triggered"] is True
        assert result["abstain_reason"] == "Verification failed or missing"

    def test_audit_logging(self) -> None:
        audit = AuditLogger()
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "model_fit_quality: rmse=6.00",
            },
        }

        abstain_node(state, audit_logger=audit)

        entries = audit.get_entries(event_type="abstain_triggered")
        assert len(entries) == 1
        assert entries[0].data["reason"] == "model_fit_quality: rmse=6.00"

    def test_abstain_produces_typed_envelope(self) -> None:
        """The dict (minus legacy keys) round-trips through FinalAssessment."""
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "coverage: n_stations=1",
            },
        }

        result = abstain_node(state)
        fa_dict = dict(result["final_assessment"])
        # Strip backwards-compat keys before validation
        fa_dict.pop("outcome", None)
        fa_dict.pop("pipeline_status", None)
        fa = FinalAssessment.model_validate(fa_dict)
        assert fa.status == AssessmentStatus.ABSTAIN
        assert fa.report_tier == 1


class TestFinalNodeE6:
    """final_node ABSTAIN and incomplete_report branches."""

    def test_abstain_output(self) -> None:
        state: PipelineState = {
            "fsm_state": "ASSESS",
            "abstain_triggered": True,
            "abstain_reason": "coverage: n_stations=1",
        }

        result = final_node(state)

        fa = result["final_assessment"]
        assert fa["outcome"] == "abstain"
        assert fa["pipeline_status"] == "abstain"
        assert fa["abstain_reason"] == "coverage: n_stations=1"

    def test_incomplete_e7_output(self) -> None:
        state: PipelineState = {
            "fsm_state": "ASSESS",
            "pipeline_status": "incomplete_report",
        }

        result = final_node(state)

        fa = result["final_assessment"]
        assert fa["outcome"] == "verified_pending_report"
        assert "no report agent" in fa["detail"]


class TestPipelineSyncE6:
    """E6 end-to-end pipeline tests through run_pipeline_sync."""

    def test_fail_verification_to_abstain(self) -> None:
        """FAIL verification -> abstain_triggered=True."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "model_fit_quality: rmse=6.00 cm",
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_pass_verification_to_report(self) -> None:
        """PASS verification -> no abstain, report path."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is not True
        assert result["final_assessment"]["outcome"] != "abstain"

    def test_missing_verification_to_abstain(self) -> None:
        """Missing verification_result -> ABSTAIN (fail-closed)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"


# ---------------------------------------------------------------------------
# Report node with ReportAgent
# ---------------------------------------------------------------------------


_REPORT_EVENT_ID = uuid4()


def _make_scenario_dict() -> dict:
    """Construct a ScenarioAssessment, then return its dict form."""
    scenario = ScenarioAssessment(
        producer="scenario_agent",
        event_id=_REPORT_EVENT_ID,
        method="NNLS_UNIT_SOURCE",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        dart_stations_used=["D21414"],
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=1800,
        top_scenarios=[
            RankedScenario(
                unit_source_ids=["SRC_A"],
                weights=[1.0],
                waveform_rmse_cm=1.0,
                mw_equivalent=7.8,
                rank=1,
                posterior_weight=0.8,
            ),
        ],
        coastal_proxies=[],
        ensemble_spread=EnsembleSpread.LOW,
        bilateral_rupture_evaluated=False,
        limiting_assumptions=["Flat bathymetry assumed"],
    )
    return scenario.model_dump()


def _make_verification_dict(
    outcome: VerificationOutcome = VerificationOutcome.PASS,
) -> dict:
    """Construct a VerificationResult, then return its dict form."""
    checks = [
        VerificationCheck(
            name="holdout_station", result=CheckResult.PASS, evidence="OK"
        ),
    ]
    abstain_required = outcome == VerificationOutcome.FAIL
    abstain_reason = "Verification failed" if abstain_required else None
    vr = VerificationResult(
        producer="verification_agent",
        event_id=_REPORT_EVENT_ID,
        overall=outcome,
        checks=checks,
        abstain_required=abstain_required,
        abstain_reason=abstain_reason,
    )
    return vr.model_dump()


class TestReportNodeE7:
    """report_node with ReportAgent."""

    def test_with_agent_generates_report(self) -> None:
        """When report_agent is provided, report_node generates FinalAssessment."""
        agent = ReportAgent()
        state: PipelineState = {
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(),
        }

        result = report_node(state, report_agent=agent)

        assert result["pipeline_status"] == "report_generated"
        assert "final_assessment" in result
        fa = result["final_assessment"]
        assert fa["status"] == "PROVISIONAL"
        assert fa["report_tier"] == 1

    def test_without_agent_returns_incomplete(self) -> None:
        """Without report_agent, report_node returns incomplete_report."""
        state: PipelineState = {}
        result = report_node(state)
        assert result["pipeline_status"] == "incomplete_report"
        assert "final_assessment" not in result

    def test_missing_scenario_returns_incomplete(self) -> None:
        """With agent but no scenario_assessment, returns incomplete_report."""
        agent = ReportAgent()
        state: PipelineState = {
            "verification_result": _make_verification_dict(),
        }

        result = report_node(state, report_agent=agent)

        assert result["pipeline_status"] == "incomplete_report"

    def test_report_audit_logged(self) -> None:
        """Audit entry is created on successful report generation."""
        agent = ReportAgent()
        audit = AuditLogger()
        state: PipelineState = {
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(),
        }

        report_node(state, report_agent=agent, audit_logger=audit)

        entries = audit.get_entries(event_type="report_generated")
        assert len(entries) == 1
        assert entries[0].data["tier"] == 1
        assert entries[0].data["confidence"] == "HIGH"


class TestPipelineSyncE7:
    """run_pipeline_sync with ReportAgent."""

    def test_pass_verification_generates_report(self) -> None:
        """Full pipeline with agent produces FinalAssessment with PROVISIONAL."""
        fsm = _make_assess_fsm()
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(VerificationOutcome.PASS),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        assert result.get("abstain_triggered") is not True
        assert result["pipeline_status"] == "report_generated"
        fa = result["final_assessment"]
        assert fa["status"] == "PROVISIONAL"

    def test_pass_verification_without_agent_gives_incomplete(self) -> None:
        """Without agent, verified path gives verified_pending_report."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is not True
        assert result["final_assessment"]["outcome"] == "verified_pending_report"

    def test_pipeline_with_human_approve(self) -> None:
        """APPROVE human decision transitions PROVISIONAL -> APPROVED_INTERNAL."""
        fsm = _make_assess_fsm()
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(VerificationOutcome.PASS),
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        assert result.get("abstain_triggered") is not True
        fa = result["final_assessment"]
        assert fa["status"] == "APPROVED_INTERNAL"

    def test_pipeline_with_human_reject(self) -> None:
        """REJECT human decision keeps PROVISIONAL status."""
        fsm = _make_assess_fsm()
        agent = ReportAgent()
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(VerificationOutcome.PASS),
            "human_decision": _make_human_decision_dict(ReviewDecision.REJECT),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=agent)

        assert result.get("abstain_triggered") is not True
        fa = result["final_assessment"]
        assert fa["status"] == "PROVISIONAL"


# ---------------------------------------------------------------------------
# Coverage gap tests for nodes.py
# ---------------------------------------------------------------------------


class TestOrchestrateNodeMissingScore:
    """orchestrate_node when anomaly_score is None."""

    def test_none_anomaly_score_returns_fail_safe(self) -> None:
        """Assessment present but anomaly_score=None -> fail-safe, no FSM update."""
        fsm = _make_fsm(SystemState.MONITOR)
        assessment = dict(SAMPLE_ASSESSMENT)
        assessment["anomaly_score"] = None
        state: PipelineState = {"anomaly_assessment": assessment}

        result = orchestrate_node(state, fsm=fsm)

        assert result["fsm_state"] == "MONITOR"
        assert result["state_changed"] is False


class TestReportNodeSynthesisException:
    """report_node when synthesize() raises."""

    def test_synthesis_exception_returns_incomplete(self) -> None:
        """If ReportAgent.synthesize() raises, report_node returns incomplete_report."""

        class _FailingAgent:
            def synthesize(self, scenario, verification):
                raise RuntimeError("Simulated synthesis failure")

        agent = _FailingAgent()
        state: PipelineState = {
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(),
        }

        result = report_node(state, report_agent=agent)

        assert result["pipeline_status"] == "incomplete_report"
        assert "final_assessment" not in result


class TestHumanReviewNodeErrors:
    """human_review_node error handling."""

    def test_malformed_final_assessment_returns_empty(self) -> None:
        """Malformed final_assessment -> ValidationError catch -> {}."""
        state: PipelineState = {
            "final_assessment": {"status": "PROVISIONAL", "bad_field": True},
            "human_decision": _make_human_decision_dict(ReviewDecision.APPROVE),
        }
        result = human_review_node(state)
        assert result == {}

    def test_malformed_human_decision_returns_empty(self) -> None:
        """Malformed human_decision -> ValidationError catch -> {}."""
        state: PipelineState = {
            "final_assessment": _make_provisional_fa_dict(),
            "human_decision": {"decision": "APPROVE"},  # missing required fields
        }
        result = human_review_node(state)
        assert result == {}


class TestRunPipelineSyncUnexpectedVerifyRoute:
    """run_pipeline_sync unexpected verify route -> ABSTAIN."""

    def test_unexpected_verify_route_falls_back_to_abstain(self) -> None:
        """When route_after_verify returns unexpected value, fall back to ABSTAIN."""
        from unittest.mock import patch

        fsm = _make_fsm(SystemState.ASSESS)
        state: PipelineState = {
            "anomaly_assessment": ASSESS_ASSESSMENT,
            "scenario_assessment": _make_scenario_dict(),
            "verification_result": _make_verification_dict(VerificationOutcome.PASS),
        }

        with patch(
            "hazard_assessment.orchestrator.pipeline.route_after_verify",
            return_value="UNEXPECTED_ROUTE",
        ):
            result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is True
