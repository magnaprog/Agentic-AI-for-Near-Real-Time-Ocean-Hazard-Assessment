"""Tests for the Report Agent.

Verifies manifest, synthesize() for all tiers, confidence mapping,
guardrail enforcement, seismic-only guard, and error cases.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from hazard_assessment.agents.base import AgentCapability
from hazard_assessment.agents.report_agent import ReportAgent
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
)
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

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------

# Shared event_id - scenario and verification must agree
_SHARED_EVENT_ID = uuid4()


def _make_ranked(
    rank: int = 1,
    mw: float = 7.8,
) -> RankedScenario:
    return RankedScenario(
        unit_source_ids=["SRC_A", "SRC_B"],
        weights=[0.6, 0.4],
        waveform_rmse_cm=1.2,
        mw_equivalent=mw,
        rank=rank,
        posterior_weight=0.75,
    )


def _make_scenario(
    constraint_stage: ConstraintStage = ConstraintStage.DART_CONSTRAINED,
    spread: EnsembleSpread = EnsembleSpread.LOW,
    event_id: UUID | None = _SHARED_EVENT_ID,
) -> ScenarioAssessment:
    dart_used = ["D21414"] if constraint_stage != ConstraintStage.SEISMIC_ONLY else []
    if constraint_stage == ConstraintStage.MULTI_STATION:
        dart_used = ["D21414", "D46402"]
    return ScenarioAssessment(
        producer="scenario_agent",
        event_id=event_id,
        type="ScenarioAssessment",
        method="NNLS_UNIT_SOURCE",
        constraint_stage=constraint_stage,
        dart_stations_used=dart_used,
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=1800,
        top_scenarios=[_make_ranked()],
        coastal_proxies=[],
        ensemble_spread=spread,
        bilateral_rupture_evaluated=False,
        limiting_assumptions=["Flat bathymetry assumed"],
    )


def _make_verification(
    outcome: VerificationOutcome = VerificationOutcome.PASS,
    event_id: UUID | None = _SHARED_EVENT_ID,
) -> VerificationResult:
    checks = [
        VerificationCheck(name="holdout_station", result=CheckResult.PASS, evidence="OK"),
    ]
    if outcome == VerificationOutcome.PASS_WITH_CONCERNS:
        checks.append(
            VerificationCheck(
                name="data_coverage", result=CheckResult.CONCERN, evidence="Only 65%"
            )
        )
    abstain_required = outcome == VerificationOutcome.FAIL
    abstain_reason = "Verification failed" if abstain_required else None
    return VerificationResult(
        producer="verification_agent",
        event_id=event_id,
        overall=outcome,
        checks=checks,
        abstain_required=abstain_required,
        abstain_reason=abstain_reason,
    )


# ===================================================================
# TestReportAgentManifest
# ===================================================================


class TestReportAgentManifest:
    """Manifest and process() stub."""

    def test_name(self):
        agent = ReportAgent()
        assert agent.name == "report_agent"

    def test_emit_report_capability(self):
        agent = ReportAgent()
        assert AgentCapability.EMIT_REPORT in agent.manifest.capabilities


# ===================================================================
# TestSynthesizeTier1
# ===================================================================


class TestSynthesizeTier1:
    """Tier 1 Technical Brief via synthesize()."""

    def test_produces_final_assessment(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        assert isinstance(fa, FinalAssessment)

    def test_status_provisional(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        assert fa.status == AssessmentStatus.PROVISIONAL

    def test_summary_passes_guardrails(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        result = scan_text(fa.summary)
        assert result.passed, f"Violations: {result.violations}"

    def test_summary_contains_disclaimer(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.summary

    def test_confidence_pass_low_gives_high(self):
        agent = ReportAgent()
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.uncertainty.confidence_level == ConfidenceLevel.HIGH

    def test_confidence_concerns_moderate_gives_moderate(self):
        """System confidence: 0.40*0.6 + 0.25*0.2 + 0.35*0.5 = 0.465 -> MODERATE."""
        agent = ReportAgent()
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.MODERATE),
            _make_verification(outcome=VerificationOutcome.PASS_WITH_CONCERNS),
        )
        assert fa.uncertainty.confidence_level == ConfidenceLevel.MODERATE

    def test_key_uncertainties_populated(self):
        agent = ReportAgent()
        fa = agent.synthesize(
            _make_scenario(),
            _make_verification(outcome=VerificationOutcome.PASS_WITH_CONCERNS),
        )
        # CONCERN check should appear as key uncertainty
        assert len(fa.uncertainty.key_uncertainties) >= 1
        assert any("data_coverage" in u for u in fa.uncertainty.key_uncertainties)

    def test_decision_trace_populated(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        assert len(fa.decision_trace) == 2
        step_names = [s.step for s in fa.decision_trace]
        assert "template_rendering" in step_names
        assert "guardrail_scan" in step_names

    def test_provenance_bundle_id_auto(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        assert isinstance(fa.provenance_bundle_id, UUID)

    def test_provenance_bundle_id_explicit(self):
        agent = ReportAgent()
        bundle_id = uuid4()
        fa = agent.synthesize(
            _make_scenario(),
            _make_verification(),
            provenance_bundle_id=bundle_id,
        )
        assert fa.provenance_bundle_id == bundle_id

    def test_schema_round_trip(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification())
        data = fa.model_dump()
        fa2 = FinalAssessment.model_validate(data)
        assert fa2.status == fa.status
        assert fa2.report_tier == fa.report_tier
        assert fa2.summary == fa.summary


# ===================================================================
# TestSynthesizeTier2
# ===================================================================


class TestSynthesizeTier2:
    """Tier 2 Situational Awareness Summary."""

    def test_produces_tier_2(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification(), tier=2)
        assert fa.report_tier == 2

    def test_passes_guardrails(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification(), tier=2)
        result = scan_text(fa.summary)
        assert result.passed, f"Violations: {result.violations}"


# ===================================================================
# TestSynthesizeTier3
# ===================================================================


class TestSynthesizeTier3:
    """Tier 3 Post-Event Analysis."""

    def test_produces_tier_3(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification(), tier=3)
        assert fa.report_tier == 3

    def test_passes_guardrails(self):
        agent = ReportAgent()
        fa = agent.synthesize(_make_scenario(), _make_verification(), tier=3)
        result = scan_text(fa.summary)
        assert result.passed, f"Violations: {result.violations}"

    def test_provenance_in_summary(self):
        agent = ReportAgent()
        bundle_id = uuid4()
        fa = agent.synthesize(
            _make_scenario(),
            _make_verification(),
            tier=3,
            provenance_bundle_id=bundle_id,
        )
        assert str(bundle_id) in fa.summary

    def test_auto_provenance_matches_summary_and_metadata(self):
        """When no explicit bundle_id is given, the auto-generated UUID
        must appear in both the summary text and provenance_bundle_id."""
        agent = ReportAgent()
        fa = agent.synthesize(
            _make_scenario(),
            _make_verification(),
            tier=3,
        )
        assert str(fa.provenance_bundle_id) in fa.summary


# ===================================================================
# TestSynthesizeErrors
# ===================================================================


class TestSynthesizeErrors:
    """Error cases for synthesize()."""

    def test_invalid_tier_raises_value_error(self):
        agent = ReportAgent()
        with pytest.raises(ValueError, match="Invalid report tier"):
            agent.synthesize(_make_scenario(), _make_verification(), tier=4)

    def test_tier_zero_raises_value_error(self):
        agent = ReportAgent()
        with pytest.raises(ValueError, match="Invalid report tier"):
            agent.synthesize(_make_scenario(), _make_verification(), tier=0)

    def test_seismic_only_tier_2_raises_value_error(self):
        agent = ReportAgent()
        scenario = _make_scenario(constraint_stage=ConstraintStage.SEISMIC_ONLY)
        verification = _make_verification()
        with pytest.raises(ValueError, match="Seismic-only"):
            agent.synthesize(scenario, verification, tier=2)

    def test_mismatched_event_id_raises_value_error(self):
        agent = ReportAgent()
        scenario = _make_scenario(event_id=uuid4())
        verification = _make_verification(event_id=uuid4())
        with pytest.raises(ValueError, match="does not match"):
            agent.synthesize(scenario, verification)


# ===================================================================
# Coverage gap tests for report_agent.py
# ===================================================================


class TestGuardrailScanErrorConstruction:
    """GuardrailScanError stores scan_result."""

    def test_stores_scan_result(self):
        from hazard_assessment.agents.report_agent import GuardrailScanError

        result = scan_text("WARNING: This is a test")
        err = GuardrailScanError(result)
        assert err.scan_result is result
        assert "Warning" in str(err)

    def test_lists_prohibited_terms(self):
        from hazard_assessment.agents.report_agent import GuardrailScanError

        result = scan_text("TSUNAMI WARNING issued")
        err = GuardrailScanError(result)
        assert len(err.scan_result.violations) >= 1


class TestSynthesizeGuardrailEnforcement:
    """Synthesize with prohibited terms in dynamic data raises."""

    def test_prohibited_term_in_scenario_data_raises(self):
        from unittest.mock import patch

        from hazard_assessment.agents.report_agent import GuardrailScanError

        agent = ReportAgent()
        scenario = _make_scenario()
        verification = _make_verification()

        # Patch render_tier_1 to return text with a prohibited term
        with patch(
            "hazard_assessment.agents.report_agent.render_tier_1",
            return_value="TSUNAMI WARNING: " + NON_AUTHORITATIVE_DISCLAIMER,
        ):
            with pytest.raises(GuardrailScanError):
                agent.synthesize(scenario, verification)


# ===================================================================
# LLM synthesis path tests
# ===================================================================


_FIXED_BUNDLE_ID = UUID("00000000-0000-4000-8000-00000000b1d1")


class _FakeGraph:
    """Mock compiled LangGraph graph for testing LLM synthesis path."""

    def __init__(self, narrative: str | None = "LLM-generated narrative text."):
        self._narrative = narrative

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        return {**state, "narrative": self._narrative}


class _FailingGraph:
    """Mock graph that raises on invoke()."""

    def invoke(self, state: dict, config: dict | None = None) -> dict:
        raise RuntimeError("LLM API unavailable")


class TestLLMSynthesisPath:
    """Tests for the LLM synthesis integration in ReportAgent.synthesize()."""

    def _agent_with_graph(self, graph) -> ReportAgent:
        """Create a ReportAgent with a mocked synthesis graph."""
        agent = ReportAgent()
        agent._synthesis_graph = graph
        return agent

    def test_llm_narrative_stored_as_commentary_not_summary(self):
        """the narrative goes to model_commentary; the emitted
        summary stays byte-identical to the template-only run."""
        template_only = ReportAgent()
        agent = self._agent_with_graph(_FakeGraph("LLM output here."))
        fa_template = template_only.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
            provenance_bundle_id=_FIXED_BUNDLE_ID,
        )
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
            provenance_bundle_id=_FIXED_BUNDLE_ID,
        )
        assert fa.llm_synthesis_used is True
        assert fa.model_commentary is not None
        assert "LLM output here." in fa.model_commentary
        assert "LLM output here." not in fa.summary
        assert fa.summary == fa_template.summary

    def test_disclaimer_appended_to_llm_commentary(self):
        """NON_AUTHORITATIVE_DISCLAIMER is appended when not in LLM output."""
        agent = self._agent_with_graph(_FakeGraph("Clean LLM narrative."))
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.model_commentary is not None
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.model_commentary
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.summary

    def test_disclaimer_not_doubled(self):
        """If LLM already includes the disclaimer, it is not appended again."""
        narrative_with_disclaimer = f"LLM text.\n\n{NON_AUTHORITATIVE_DISCLAIMER}"
        agent = self._agent_with_graph(_FakeGraph(narrative_with_disclaimer))
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.model_commentary is not None
        assert fa.model_commentary.count(NON_AUTHORITATIVE_DISCLAIMER) == 1
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.summary

    def test_llm_failure_leaves_commentary_none(self):
        """Graph.invoke() exception: no commentary, llm_used=False, and the
        deterministic summary is emitted normally."""
        agent = self._agent_with_graph(_FailingGraph())
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.llm_synthesis_used is False
        assert fa.model_commentary is None
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.summary

    def test_llm_returns_none_narrative_leaves_commentary_none(self):
        """Graph returns narrative=None: no commentary recorded."""
        agent = self._agent_with_graph(_FakeGraph(narrative=None))
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.llm_synthesis_used is False
        assert fa.model_commentary is None

    def test_llm_narrative_with_prohibited_term_is_dropped(self):
        """LLM narrative containing prohibited term: commentary dropped,
        deterministic summary unaffected."""
        agent = self._agent_with_graph(
            _FakeGraph("This is a Tsunami WARNING from the system.")
        )
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        assert fa.llm_synthesis_used is False
        assert fa.model_commentary is None
        assert NON_AUTHORITATIVE_DISCLAIMER in fa.summary

    def test_llm_skipped_below_confidence_threshold(self):
        """LLM synthesis is skipped when system confidence < 0.35."""
        agent = self._agent_with_graph(_FakeGraph("Should not appear."))
        # PASS_WITH_CONCERNS + 1 station + HIGH spread + Rayleigh ->
        # 0.40*0.6 + 0.25*0.2 + 0.35*0.0 = 0.29 * 0.7 = 0.203 < 0.35
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.HIGH),
            _make_verification(outcome=VerificationOutcome.PASS_WITH_CONCERNS),
            rayleigh_wave_suspect=True,
        )
        assert fa.llm_synthesis_used is False

    def test_decision_trace_includes_llm_step(self):
        """Decision trace has llm_synthesis step when LLM is used."""
        agent = self._agent_with_graph(_FakeGraph("LLM text."))
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        step_names = [s.step for s in fa.decision_trace]
        assert "llm_synthesis" in step_names

    def test_decision_trace_excludes_llm_step_on_fallback(self):
        """Decision trace omits llm_synthesis when LLM fails."""
        agent = self._agent_with_graph(_FailingGraph())
        fa = agent.synthesize(
            _make_scenario(spread=EnsembleSpread.LOW),
            _make_verification(outcome=VerificationOutcome.PASS),
        )
        step_names = [s.step for s in fa.decision_trace]
        assert "llm_synthesis" not in step_names
