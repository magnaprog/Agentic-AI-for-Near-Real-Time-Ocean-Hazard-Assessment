"""Shared fixtures for integration tests.

Builds schema-validated objects for pipeline integration testing.
Reuses patterns from tests/unit/test_pipeline_nodes.py.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from hazard_assessment.orchestrator.states import (
    AuditWriter,
    FSMOrchestrator,
    SystemState,
    ThresholdConfig,
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
_FIXED_ESCALATION_PACKET_ID = UUID("8a729518-4413-4043-9791-135864832890")

THRESHOLDS = ThresholdConfig(basin="pacific", t1=0.35, t2=0.60, t3=0.85)
SHARED_EVENT_ID = uuid4()


def make_anomaly_dict(score: float) -> dict:
    """Return a sample anomaly assessment dict with the given score."""
    return {
        "type": "AnomalyAssessment",
        "schema_version": "1.0",
        "producer": "anomaly_agent",
        "anomaly_score": score,
        "score_components": {"threshold": 0.5, "statistical": 0.3, "ml": None},
        "triggering_stations": ["21413"],
        "spatial_confirmations": [],
        "seismic_quiet": False,
        "meteotsunami_score": 0.0,
        "stations_offline": [],
        "coverage_note": "",
        "reasoning_trace": "integration test",
        "current_state": "",
        "state_changed": False,
    }


def make_fsm_in_state(
    target: SystemState = SystemState.MONITOR,
    thresholds: ThresholdConfig = THRESHOLDS,
    audit_writer: AuditWriter | None = None,
) -> FSMOrchestrator:
    """Create an FSM in the specified state with an event context."""
    fsm = FSMOrchestrator(thresholds=thresholds, audit_writer=audit_writer)
    if target != SystemState.IDLE:
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
        )
        if target == SystemState.INVESTIGATE:
            fsm.evaluate_anomaly_score(0.40)
        elif target == SystemState.ASSESS:
            fsm.evaluate_anomaly_score(0.40)
            fsm.evaluate_anomaly_score(0.65)
        elif target == SystemState.ESCALATE:
            fsm.evaluate_anomaly_score(0.40)
            fsm.evaluate_anomaly_score(0.65)
            fsm.evaluate_anomaly_score(0.90)
    return fsm


def make_scenario_dict(event_id: UUID | None = None) -> dict:
    """Construct a ScenarioAssessment dict."""
    eid = event_id or SHARED_EVENT_ID
    scenario = ScenarioAssessment(
        producer="scenario_agent",
        event_id=eid,
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


def make_verification_dict(
    outcome: VerificationOutcome = VerificationOutcome.PASS,
    event_id: UUID | None = None,
) -> dict:
    """Construct a VerificationResult dict."""
    eid = event_id or SHARED_EVENT_ID
    checks = [
        VerificationCheck(
            name="holdout_station", result=CheckResult.PASS, evidence="OK"
        ),
    ]
    if outcome == VerificationOutcome.PASS_WITH_CONCERNS:
        checks.append(
            VerificationCheck(
                name="data_coverage",
                result=CheckResult.CONCERN,
                evidence="Only 65% coverage",
            )
        )
    elif outcome == VerificationOutcome.FAIL:
        checks.append(
            VerificationCheck(
                name="data_coverage",
                result=CheckResult.FAIL,
                evidence="Insufficient constraining-station coverage",
            )
        )
    abstain_required = outcome == VerificationOutcome.FAIL
    abstain_reason = "Verification failed" if abstain_required else None
    vr = VerificationResult(
        producer="verification_agent",
        event_id=eid,
        overall=outcome,
        checks=checks,
        abstain_required=abstain_required,
        abstain_reason=abstain_reason,
    )
    return vr.model_dump()


def make_human_decision_dict(
    decision: ReviewDecision = ReviewDecision.APPROVE,
    reason: str = "Assessment looks correct.",
) -> dict:
    """Build a HumanDecision dict."""
    hd = HumanDecision(
        producer="human_review_node",
        reviewer_id="alice",
        decision=decision,
        decision_reason=reason,
        decided_at_utc=_FIXED_DECISION_TIME,
        escalation_packet_id=_FIXED_ESCALATION_PACKET_ID,
    )
    return hd.model_dump()
