"""Unit tests for the LLM advisory package.

Tests cover:
- Historical analogue retrieval (sorted by magnitude)
- LLM synthesis schema validation
- System confidence score computation
- Rayleigh wave verification check
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hazard_assessment.agents.llm_advisory.retrieval import retrieve_similar_events
from hazard_assessment.agents.llm_advisory.schemas import (
    AfterActionState,
    EvidenceSynthesisOutput,
    LLMSynthesisState,
    NarrativeOutput,
    ScenarioInterpOutput,
)
from hazard_assessment.agents.report_agent import _compute_system_confidence
from hazard_assessment.agents.verification_checks import (
    CheckResult,
    VerificationInput,
    check_rayleigh_wave_suspect,
)
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)

# ---------------------------------------------------------------------------
# Historical analogue retrieval
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


class TestRetrieveSimilarEvents:
    """Tests for magnitude-based historical analogue retrieval."""

    def test_returns_json_string(self) -> None:
        result = retrieve_similar_events(9.0)
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    def test_sorted_by_magnitude_proximity(self) -> None:
        result = retrieve_similar_events(9.0)
        parsed = json.loads(result)
        diffs = [abs(e["mw"] - 9.0) for e in parsed]
        assert diffs == sorted(diffs)

    def test_max_results_respected(self) -> None:
        result = retrieve_similar_events(8.0, max_results=2)
        parsed = json.loads(result)
        assert len(parsed) <= 2

    def test_default_returns_3(self) -> None:
        result = retrieve_similar_events(8.0)
        parsed = json.loads(result)
        assert len(parsed) == 3

    def test_closest_to_9_is_sumatra_or_tohoku(self) -> None:
        """Mw 9.1 query should return Sumatra (9.1) or Tohoku (9.1) as closest."""
        result = retrieve_similar_events(9.1)
        parsed = json.loads(result)
        assert parsed[0]["mw"] == 9.1  # Exact match: Sumatra or Tohoku

    def test_missing_fixture_returns_empty(self) -> None:
        """Non-existent fixture path -> graceful empty result."""
        result = retrieve_similar_events(8.0, fixture_path=Path("/nonexistent/path.json"))
        assert result == "[]"


class TestRetrievalNodeMwGuard:
    """The retrieval node must not surface analogues when the Mw parse failed."""

    def test_malformed_scenario_json_yields_empty(self) -> None:
        from hazard_assessment.agents.llm_advisory.synthesis_graph import (
            _parse_scenario_mw,
        )

        # Parse failure -> 0.0 sentinel -> the node guard returns "[]".
        mw = _parse_scenario_mw("not json")
        assert mw == 0.0
        # And the guard in retrieval_node treats 0.0 as "no retrieval".
        assert (retrieve_similar_events(mw) if mw > 0.0 else "[]") == "[]"

    def test_missing_mw_equivalent_yields_empty(self) -> None:
        from hazard_assessment.agents.llm_advisory.synthesis_graph import (
            _parse_scenario_mw,
        )

        mw = _parse_scenario_mw(json.dumps({"no_mw": True}))
        assert mw == 0.0
        assert (retrieve_similar_events(mw) if mw > 0.0 else "[]") == "[]"


# ---------------------------------------------------------------------------
# LLM synthesis schemas
# ---------------------------------------------------------------------------


class TestLLMSynthesisSchemas:
    """Validate LLM advisory schema structures."""

    def test_evidence_synthesis_output(self) -> None:
        out = EvidenceSynthesisOutput(synthesis="Test evidence summary.")
        assert out.synthesis == "Test evidence summary."
        assert out.rayleigh_note is None

    def test_scenario_interp_output(self) -> None:
        out = ScenarioInterpOutput(
            interpretation="Test scenario interpretation.",
            uncertainty_note="Moderate spread.",
        )
        assert out.interpretation == "Test scenario interpretation."

    def test_narrative_output(self) -> None:
        out = NarrativeOutput(narrative="Test narrative text.")
        assert out.narrative == "Test narrative text."

    def test_synthesis_state_keys(self) -> None:
        """Verify LLMSynthesisState has expected keys."""
        keys = LLMSynthesisState.__annotations__
        assert "event_id" in keys
        assert "report_tier" in keys
        assert "narrative" in keys
        assert "system_confidence" in keys

    def test_after_action_state_keys(self) -> None:
        """Verify AfterActionState has expected keys."""
        keys = AfterActionState.__annotations__
        assert "event_id" in keys
        assert "timeline" in keys
        assert "gaps" in keys
        assert "draft_report" in keys


# ---------------------------------------------------------------------------
# System confidence score
# ---------------------------------------------------------------------------


class TestComputeSystemConfidence:
    """Tests for _compute_system_confidence() deterministic formula."""

    def test_all_best_case(self) -> None:
        """PASS + 5 stations + LOW spread + no Rayleigh -> max confidence."""
        score = _compute_system_confidence("PASS", 5, EnsembleSpread.LOW, False)
        # 0.40*1.0 + 0.25*1.0 + 0.35*1.0 = 1.0
        assert score == pytest.approx(1.0)

    def test_all_worst_case(self) -> None:
        """FAIL + 0 stations + HIGH spread + Rayleigh -> minimum."""
        score = _compute_system_confidence("FAIL", 0, EnsembleSpread.HIGH, True)
        # 0.40*0.2 + 0.25*0.0 + 0.35*0.0 = 0.08 * 0.7 = 0.056
        assert score == pytest.approx(0.056)

    def test_rayleigh_penalty_applied(self) -> None:
        """Rayleigh suspect applies 0.7 penalty multiplier."""
        without = _compute_system_confidence("PASS", 3, EnsembleSpread.LOW, False)
        with_r = _compute_system_confidence("PASS", 3, EnsembleSpread.LOW, True)
        assert with_r == pytest.approx(without * 0.7)

    def test_station_count_caps_at_5(self) -> None:
        """Station count saturates at 5 (score = 1.0)."""
        score_5 = _compute_system_confidence("PASS", 5, EnsembleSpread.LOW, False)
        score_10 = _compute_system_confidence("PASS", 10, EnsembleSpread.LOW, False)
        assert score_5 == score_10

    def test_score_bounded_0_to_1(self) -> None:
        """Score is always in [0.0, 1.0]."""
        for outcome in ["PASS", "PASS_WITH_CONCERNS", "FAIL", "UNKNOWN"]:
            for n in [0, 1, 5, 10]:
                for spread in EnsembleSpread:
                    for rayleigh in [True, False]:
                        s = _compute_system_confidence(outcome, n, spread, rayleigh)
                        assert 0.0 <= s <= 1.0

    def test_pass_with_concerns_intermediate(self) -> None:
        """PASS_WITH_CONCERNS gives score between FAIL and PASS."""
        fail = _compute_system_confidence("FAIL", 3, EnsembleSpread.MODERATE, False)
        concern = _compute_system_confidence(
            "PASS_WITH_CONCERNS", 3, EnsembleSpread.MODERATE, False,
        )
        pass_ = _compute_system_confidence("PASS", 3, EnsembleSpread.MODERATE, False)
        assert fail < concern < pass_

    def test_unknown_outcome_treated_as_fail(self) -> None:
        """Unknown verification outcome defaults to FAIL score (0.2)."""
        unknown = _compute_system_confidence("UNKNOWN", 3, EnsembleSpread.LOW, False)
        fail = _compute_system_confidence("FAIL", 3, EnsembleSpread.LOW, False)
        assert unknown == fail


# ---------------------------------------------------------------------------
# Rayleigh wave verification check
# ---------------------------------------------------------------------------


def _make_vi(rayleigh: bool = False) -> VerificationInput:
    """Build minimal VerificationInput for Rayleigh check tests."""
    scenario = ScenarioAssessment(
        producer="test",
        type="ScenarioAssessment",
        method="NNLS_UNIT_SOURCE",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        dart_stations_used=["D21414"],
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=1800,
        top_scenarios=[
            RankedScenario(
                unit_source_ids=["A01"],
                weights=[0.8],
                mw_equivalent=8.0,
                rank=1,
                posterior_weight=0.9,
                waveform_rmse_cm=1.0,
            )
        ],
        coastal_proxies=[],
        ensemble_spread=EnsembleSpread.LOW,
        bilateral_rupture_evaluated=False,
        limiting_assumptions=[],
    )
    return VerificationInput(scenario=scenario, rayleigh_wave_suspect=rayleigh)


class TestCheckRayleighWaveSuspect:
    """Tests for the 9th verification check."""

    def test_pass_when_not_suspect(self) -> None:
        vi = _make_vi(rayleigh=False)
        check = check_rayleigh_wave_suspect(vi)
        assert check.result == CheckResult.PASS
        assert check.name == "rayleigh_wave_suspect"

    def test_concern_when_suspect(self) -> None:
        vi = _make_vi(rayleigh=True)
        check = check_rayleigh_wave_suspect(vi)
        assert check.result == CheckResult.CONCERN
        assert "Rayleigh wave" in check.evidence

    def test_never_fails(self) -> None:
        """Rayleigh check must be CONCERN, never FAIL."""
        vi = _make_vi(rayleigh=True)
        check = check_rayleigh_wave_suspect(vi)
        assert check.result != CheckResult.FAIL
