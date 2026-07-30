"""Safety tests for bounded agency enforcement.

These tests verify that the system's safety invariants hold:
1. Agent manifests are strict and declare their capabilities
2. FAIL verification always triggers ABSTAIN
3. Alert-language guardrails block prohibited terminology
4. All envelopes reject unknown fields
5. Pipeline execution enforces ABSTAIN on verification fail
6. Verification and ABSTAIN decisions are logged to audit trail
7. ABSTAIN output contains no probabilistic coastal guidance
8. Verification error paths route to ABSTAIN
9. No distributable output without human decision
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.audit.logger import AuditLogger
from hazard_assessment.orchestrator.nodes import human_review_node, run_pipeline_sync
from hazard_assessment.orchestrator.pipeline import (
    PipelineNode,
    PipelineState,
    route_after_verify,
)
from hazard_assessment.orchestrator.states import (
    FSMOrchestrator,
    ThresholdConfig,
)
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.envelope import BaseEnvelope, DecisionStep, StepResult
from hazard_assessment.schemas.verification import (
    CheckResult,
    PrerequisiteStatus,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)


class TestBoundedAgencyManifest:
    """Verify that agent manifests are strict and declare capabilities.

    These assert the manifest model's own strictness, not that any capability
    is enforced at an action site. Nothing consults the declared set at run
    time.
    """

    def test_manifest_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            AgentManifest(
                name="rogue_agent",
                capabilities=[AgentCapability.READ_DATA],
                unknown_field="should_fail",
            )

    def test_manifest_requires_capabilities(self) -> None:
        with pytest.raises(ValidationError):
            AgentManifest(name="no_caps_agent")

    def test_agent_exposes_manifest(self) -> None:
        class TestAgent(BaseAgent):
            def process(self, envelope: BaseEnvelope) -> BaseEnvelope:
                return envelope

        manifest = AgentManifest(
            name="test_agent",
            capabilities=[AgentCapability.READ_DATA],
        )
        agent = TestAgent(manifest=manifest)
        assert agent.name == "test_agent"
        assert AgentCapability.READ_DATA in agent.manifest.capabilities


class TestFailVerdictSafety:
    """Verify that FAIL verification always triggers ABSTAIN - the core safety invariant."""

    def test_schema_enforces_fail_requires_abstain(self) -> None:
        """Schema layer: FAIL without abstain_required=True is rejected."""
        with pytest.raises(ValidationError, match="abstain_required=False"):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.FAIL,
                checks=[
                    VerificationCheck(
                        name="coverage",
                        result=CheckResult.FAIL,
                        evidence="Only 1 DART station",
                    )
                ],
                abstain_required=False,
            )

    def test_schema_enforces_abstain_needs_reason(self) -> None:
        """Schema layer: abstain_required=True without reason is rejected."""
        with pytest.raises(ValidationError, match="abstain_reason is required"):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.FAIL,
                checks=[
                    VerificationCheck(
                        name="coverage",
                        result=CheckResult.FAIL,
                        evidence="Only 1 DART station",
                    )
                ],
                abstain_required=True,
                abstain_reason=None,
            )

    def test_router_enforces_fail_routes_to_abstain(self) -> None:
        """Pipeline layer: FAIL outcome routes to ABSTAIN node."""
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_router_enforces_abstain_flag_routes_to_abstain(self) -> None:
        """Pipeline layer: abstain_required=True routes to ABSTAIN even if
        outcome field is somehow not FAIL (defense-in-depth)."""
        state: PipelineState = {
            "verification_result": {
                "overall": "PASS_WITH_CONCERNS",
                "abstain_required": True,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN

    def test_router_pass_routes_to_report(self) -> None:
        """Pipeline layer: PASS routes to REPORT node."""
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            }
        }
        assert route_after_verify(state) == PipelineNode.REPORT

    def test_schema_enforces_incomplete_requires_abstain(self) -> None:
        """Schema layer: INCOMPLETE without abstain_required=True is
        rejected. A verification that could not evaluate its required
        checks must never present as distributable."""
        with pytest.raises(ValidationError, match="abstain_required=False"):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.INCOMPLETE,
                checks=[
                    VerificationCheck(
                        name="coverage",
                        result=CheckResult.NOT_EVALUATED,
                        evidence="station geometry unavailable",
                        prerequisite=PrerequisiteStatus.MISSING,
                    )
                ],
                abstain_required=False,
            )

    def test_router_enforces_incomplete_routes_to_abstain(self) -> None:
        """Pipeline layer: INCOMPLETE outcome routes to ABSTAIN node."""
        state: PipelineState = {
            "verification_result": {
                "overall": VerificationOutcome.INCOMPLETE,
                "abstain_required": True,
            }
        }
        assert route_after_verify(state) == PipelineNode.ABSTAIN


class TestAlertLanguageSafety:
    """Verify that alert-language guardrails block all prohibited terms."""

    @pytest.mark.parametrize(
        "prohibited_term",
        ["Warning", "Advisory", "Watch", "Information Statement"],
    )
    def test_blocks_prohibited_terms(self, prohibited_term: str) -> None:
        text = f"Tsunami {prohibited_term} issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == prohibited_term for v in result.violations)

    def test_requires_disclaimer(self) -> None:
        text = "Safe assessment text without disclaimer."
        result = scan_text(text)
        assert not result.passed
        assert not result.has_disclaimer

    def test_clean_text_with_disclaimer_passes(self) -> None:
        text = f"Elevated anomaly detected. {NON_AUTHORITATIVE_DISCLAIMER}"
        result = scan_text(text)
        assert result.passed


class TestEnvelopeStrictness:
    """Verify that all envelopes reject unknown fields (extra: forbid)."""

    def test_base_envelope_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            BaseEnvelope(producer="test", rogue_field="bad")

    def test_verification_rejects_extra(self) -> None:
        with pytest.raises(ValidationError):
            VerificationResult(
                producer="test",
                overall=VerificationOutcome.PASS,
                checks=[
                    VerificationCheck(
                        name="basic_check",
                        result=CheckResult.PASS,
                        evidence="test evidence",
                    )
                ],
                abstain_required=False,
                extra_field="bad",
            )


# ---------------------------------------------------------------------------
# ABSTAIN enforcement through pipeline execution
# ---------------------------------------------------------------------------

_THRESHOLDS = ThresholdConfig(basin="pacific", t1=0.35, t2=0.60, t3=0.85)

_SAMPLE_ASSESSMENT = {
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
    """Create an FSM in ASSESS state for safety tests."""
    fsm = FSMOrchestrator(thresholds=_THRESHOLDS)
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


class TestAbstainPipelineEnforcement:
    """Verify ABSTAIN enforcement at the pipeline execution layer.

    These 6 tests cover every routing case through run_pipeline_sync.
    All use FSM in ASSESS state to reach the scenario path.
    """

    def test_fail_verification_triggers_abstain(self) -> None:
        """FAIL verification -> abstain_triggered=True, outcome=abstain."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "model_fit_quality: rmse=6.00 cm",
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["abstain_reason"] == "model_fit_quality: rmse=6.00 cm"
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_missing_verification_triggers_abstain(self) -> None:
        """Missing verification_result -> abstain (fail-closed)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            # No verification_result - simulates verify node error/timeout
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_pass_routes_to_report(self) -> None:
        """PASS verification -> no abstain, report path taken."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is not True
        assert result["final_assessment"]["outcome"] != "abstain"

    def test_pass_with_concerns_routes_to_report(self) -> None:
        """PASS_WITH_CONCERNS -> no abstain, report path taken."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS_WITH_CONCERNS,
                "abstain_required": False,
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result.get("abstain_triggered") is not True
        assert result["final_assessment"]["outcome"] != "abstain"

    def test_abstain_required_overrides_outcome(self) -> None:
        """abstain_required=True on PASS_WITH_CONCERNS -> ABSTAIN (defense-in-depth)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS_WITH_CONCERNS,
                "abstain_required": True,
                "abstain_reason": "Manual override",
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_unknown_outcome_triggers_abstain(self) -> None:
        """Unknown verification outcome -> ABSTAIN (fail-closed)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": "GARBAGE",
                "abstain_required": False,
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_empty_verification_dict_triggers_abstain(self) -> None:
        """Empty verification_result dict -> ABSTAIN (fail-closed)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {},
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"

    def test_verification_missing_overall_key_triggers_abstain(self) -> None:
        """verification_result without 'overall' -> ABSTAIN (fail-closed)."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "checks": [{"name": "holdout", "result": "PASS", "evidence": "ok"}],
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        assert result["abstain_triggered"] is True
        assert result["final_assessment"]["outcome"] == "abstain"


# ---------------------------------------------------------------------------
# Verification and ABSTAIN audit trail
# ---------------------------------------------------------------------------


class TestAbstainAuditTrace:
    """Verify audit logging of verification and ABSTAIN decisions."""

    def test_verification_logged_on_pass(self) -> None:
        """A PASS verification produces a verification_complete audit entry."""
        fsm = _make_assess_fsm()
        audit = AuditLogger()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.PASS,
                "abstain_required": False,
            },
        }

        run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

        entries = audit.get_entries(event_type="verification_complete")
        assert len(entries) == 1
        assert entries[0].data["overall"] == "PASS"
        assert entries[0].data["abstain_required"] is False

    def test_abstain_logged_on_fail(self) -> None:
        """A FAIL verification produces both verification_complete and
        abstain_triggered audit entries."""
        fsm = _make_assess_fsm()
        audit = AuditLogger()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "coverage: n_stations=1",
            },
        }

        run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

        verify_entries = audit.get_entries(event_type="verification_complete")
        assert len(verify_entries) == 1

        abstain_entries = audit.get_entries(event_type="abstain_triggered")
        assert len(abstain_entries) == 1
        assert abstain_entries[0].data["reason"] == "coverage: n_stations=1"

    def test_no_audit_without_logger(self) -> None:
        """Pipeline runs without crash when audit_logger is None."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "test",
            },
        }

        # Should not raise
        result = run_pipeline_sync(state, fsm=fsm, audit_logger=None)
        assert result["abstain_triggered"] is True


# ---------------------------------------------------------------------------
# ABSTAIN output contains NO probabilistic coastal guidance
# ---------------------------------------------------------------------------


class TestAbstainOutputContent:
    """ABSTAIN output must not contain probabilistic coastal guidance.

    Verifies the rendered ABSTAIN summary text has no amplitude values,
    arrival times, or scenario rankings.
    """

    # Patterns that indicate coastal guidance content.
    _COASTAL_PATTERNS = [
        re.compile(r"\d+\.?\d*\s*(?:m|cm|mm)\b"),  # amplitude values
        re.compile(r"arrival\s+time", re.IGNORECASE),
        re.compile(r"\bETA\b"),
        re.compile(r"scenario\s+rank", re.IGNORECASE),
        re.compile(r"coastal\s+proxy", re.IGNORECASE),
        re.compile(r"wave\s+height", re.IGNORECASE),
    ]

    def test_abstain_summary_has_no_coastal_guidance_patterns(self) -> None:
        """ABSTAIN summary from pipeline contains no coastal guidance."""
        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "verification_result": {
                "overall": VerificationOutcome.FAIL,
                "abstain_required": True,
                "abstain_reason": "coverage: n_stations=1",
            },
        }

        result = run_pipeline_sync(state, fsm=fsm)

        fa = result["final_assessment"]
        assert fa["outcome"] == "abstain"

        # Check the summary text for coastal guidance patterns
        summary = fa.get("summary", "")
        for pattern in self._COASTAL_PATTERNS:
            match = pattern.search(summary)
            assert match is None, (
                f"ABSTAIN summary contains coastal guidance pattern: {match.group()!r}"
            )


# ---------------------------------------------------------------------------
# No distributable output without human decision
# ---------------------------------------------------------------------------


def _make_provisional_fa() -> dict:
    """Build a PROVISIONAL FinalAssessment dict for safety tests."""
    from hazard_assessment.schemas.final_assessment import (
        AssessmentStatus,
        ConfidenceLevel,
        FinalAssessment,
        UncertaintyInfo,
    )

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


class TestNonDistributableWithoutHumanDecision:
    """No pipeline path produces distributable output without human decision.

    APPROVED_INTERNAL is the only distributable status.
    report_node produces PROVISIONAL; only APPROVE transitions it.
    """

    def test_report_output_is_provisional(self) -> None:
        """report_node with ReportAgent produces PROVISIONAL, not distributable."""
        from hazard_assessment.agents.report_agent import ReportAgent
        from hazard_assessment.schemas.scenario import (
            ConstraintStage,
            EnsembleSpread,
            RankedScenario,
            ScenarioAssessment,
        )

        event_id = uuid4()
        scenario = ScenarioAssessment(
            producer="scenario_agent",
            event_id=event_id,
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
        verification = VerificationResult(
            producer="verification_agent",
            event_id=event_id,
            overall=VerificationOutcome.PASS,
            checks=[
                VerificationCheck(
                    name="holdout_station",
                    result=CheckResult.PASS,
                    evidence="OK",
                ),
            ],
            abstain_required=False,
        )

        fsm = _make_assess_fsm()
        state: PipelineState = {
            "anomaly_assessment": _SAMPLE_ASSESSMENT,
            "scenario_assessment": scenario.model_dump(),
            "verification_result": verification.model_dump(),
        }

        result = run_pipeline_sync(state, fsm=fsm, report_agent=ReportAgent())

        fa = result["final_assessment"]
        assert fa["status"] == "PROVISIONAL"
        assert fa["status"] != "APPROVED_INTERNAL"

    def test_no_human_decision_keeps_provisional(self) -> None:
        """human_review_node without human_decision returns {} (no status change)."""
        state: PipelineState = {
            "final_assessment": _make_provisional_fa(),
            # No human_decision in state
        }

        result = human_review_node(state)

        assert result == {}

    def test_reject_and_defer_keep_provisional(self) -> None:
        """REJECT and DEFER do not produce APPROVED_INTERNAL."""
        from hazard_assessment.agents.assessment_formatter import format_human_decision
        from hazard_assessment.schemas.final_assessment import (
            AssessmentStatus,
            FinalAssessment,
        )
        from hazard_assessment.schemas.human_decision import (
            HumanDecision,
            ReviewDecision,
        )

        fa = FinalAssessment.model_validate(_make_provisional_fa())

        for decision_type in (ReviewDecision.REJECT, ReviewDecision.DEFER):
            decision = HumanDecision(
                producer="test",
                reviewer_id="operator-1",
                decision=decision_type,
                decision_reason="needs more data",
                decided_at_utc=datetime.now(UTC),
                escalation_packet_id=uuid4(),
            )
            updated = format_human_decision(fa, decision)
            assert updated.status == AssessmentStatus.PROVISIONAL, (
                f"{decision_type.value} must keep PROVISIONAL, got {updated.status}"
            )


class TestAbstainRoutingIsDefenseInDepth:
    """A FAIL or INCOMPLETE outcome must reach ABSTAIN on its own.

    ``PipelineState`` carries plain dicts, so the Pydantic validator that
    derives ``abstain_required`` from the checks does not run at this
    boundary, and the router has to decide from the outcome alone. Existing
    coverage always set the outcome and the flag together, so it asserted
    only that the pair routes correctly.

    These cases pin the behavior, not one particular branch: the explicit
    FAIL/INCOMPLETE test and the fail-closed default at the end of the router
    are deliberately redundant, and removing either one alone leaves these
    assertions passing. What they do catch is the default turning fail-open,
    which is the mutation that would actually let an unverified assessment
    through.
    """

    def test_fail_outcome_routes_to_abstain_without_the_flag(self) -> None:
        state = {"verification_result": {"overall": "FAIL", "abstain_required": False}}
        assert route_after_verify(state) == "abstain"

    def test_incomplete_outcome_routes_to_abstain_without_the_flag(self) -> None:
        state = {
            "verification_result": {"overall": "INCOMPLETE", "abstain_required": False}
        }
        assert route_after_verify(state) == "abstain"

    def test_flag_alone_still_routes_to_abstain(self) -> None:
        state = {"verification_result": {"overall": "PASS", "abstain_required": True}}
        assert route_after_verify(state) == "abstain"

    def test_clean_pass_still_reaches_the_report(self) -> None:
        state = {"verification_result": {"overall": "PASS", "abstain_required": False}}
        assert route_after_verify(state) == "report"
