"""Tests for the report template engine.

Verifies deterministic confidence mapping, key uncertainty collection,
and all three tier renderers. Every rendered template is scanned for
prohibited NOAA alert terminology.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hazard_assessment.agents.report_templates import (
    collect_key_uncertainties,
    determine_confidence,
    render_tier_1,
    render_tier_2,
    render_tier_3,
)
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.final_assessment import ConfidenceLevel
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
)

# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _make_check(
    name: str = "test_check",
    result: CheckResult = CheckResult.PASS,
    evidence: str = "OK",
) -> VerificationCheck:
    return VerificationCheck(name=name, result=result, evidence=evidence)


def _make_proxy(site_id: str = "SITE_A") -> CoastalProxy:
    return CoastalProxy(
        site_id=site_id,
        arrival_utc=datetime(2024, 1, 15, 3, 30, tzinfo=UTC),
        arrival_uncertainty_min=5.0,
        amplitude_proxy_p10_m=0.02,
        amplitude_proxy_p50_m=0.08,
        amplitude_proxy_p90_m=0.15,
        tidal_correction_applied=True,
    )


def _make_scenario(rank: int = 1, mw: float = 7.8) -> RankedScenario:
    return RankedScenario(
        unit_source_ids=["SRC_A", "SRC_B"],
        weights=[0.6, 0.4],
        waveform_rmse_cm=1.2,
        mw_equivalent=mw,
        rank=rank,
        posterior_weight=0.75,
    )


def _tier_1_kwargs(**overrides):
    """Construct default kwargs for render_tier_1."""
    defaults = dict(
        event_id="EVT-001",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        mw_best=7.8,
        overall_outcome=VerificationOutcome.PASS,
        confidence=ConfidenceLevel.HIGH,
        dart_stations_used=["D21414"],
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=1800,
        top_scenarios=[_make_scenario()],
        coastal_proxies=[_make_proxy()],
        checks=[_make_check()],
        key_uncertainties=["Uncertainty A"],
        limiting_assumptions=["Assumption A"],
        ensemble_spread=EnsembleSpread.LOW,
    )
    defaults.update(overrides)
    return defaults


def _tier_2_kwargs(**overrides):
    """Construct default kwargs for render_tier_2."""
    defaults = dict(
        event_id="EVT-001",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        mw_best=7.8,
        overall_outcome=VerificationOutcome.PASS,
        confidence=ConfidenceLevel.HIGH,
        num_dart_stations=1,
        coastal_proxies=[_make_proxy()],
        key_uncertainties=["Uncertainty A"],
        limiting_assumptions=["Assumption A"],
        ensemble_spread=EnsembleSpread.LOW,
    )
    defaults.update(overrides)
    return defaults


def _tier_3_kwargs(**overrides):
    """Construct default kwargs for render_tier_3."""
    defaults = dict(
        event_id="EVT-001",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        mw_best=7.8,
        overall_outcome=VerificationOutcome.PASS,
        confidence=ConfidenceLevel.HIGH,
        dart_stations_used=["D21414"],
        dart_stations_excluded=["D46402"],
        exclusion_reasons={"D46402": "Insufficient data coverage"},
        inversion_window_sec=1800,
        top_scenarios=[_make_scenario()],
        coastal_proxies=[_make_proxy()],
        checks=[_make_check()],
        key_uncertainties=["Uncertainty A"],
        limiting_assumptions=["Assumption A"],
        ensemble_spread=EnsembleSpread.LOW,
        provenance_bundle_id="prov-abc-123",
    )
    defaults.update(overrides)
    return defaults


# ===================================================================
# TestDetermineConfidence
# ===================================================================


class TestDetermineConfidence:
    """All 6 outcome/spread combinations + FAIL raises ValueError."""

    def test_pass_low_gives_high(self):
        result = determine_confidence(VerificationOutcome.PASS, EnsembleSpread.LOW)
        assert result == ConfidenceLevel.HIGH

    def test_pass_moderate_gives_moderate(self):
        result = determine_confidence(VerificationOutcome.PASS, EnsembleSpread.MODERATE)
        assert result == ConfidenceLevel.MODERATE

    def test_pass_high_gives_moderate(self):
        result = determine_confidence(VerificationOutcome.PASS, EnsembleSpread.HIGH)
        assert result == ConfidenceLevel.MODERATE

    def test_concerns_low_gives_moderate(self):
        result = determine_confidence(
            VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.LOW,
        )
        assert result == ConfidenceLevel.MODERATE

    def test_concerns_moderate_gives_low(self):
        result = determine_confidence(
            VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.MODERATE,
        )
        assert result == ConfidenceLevel.LOW

    def test_concerns_high_gives_low(self):
        result = determine_confidence(
            VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.HIGH,
        )
        assert result == ConfidenceLevel.LOW

    def test_fail_raises_value_error(self):
        with pytest.raises(ValueError, match="FAIL"):
            determine_confidence(VerificationOutcome.FAIL, EnsembleSpread.LOW)


# ===================================================================
# TestCollectKeyUncertainties
# ===================================================================


class TestCollectKeyUncertainties:
    """Priority ordering, source inclusion, and cap at max.

    Limiting assumptions are NOT included - they have their own section
    in every tier template to avoid content duplication.
    """

    def test_concern_checks_first(self):
        checks = [
            _make_check("holdout", CheckResult.PASS, "OK"),
            _make_check("coverage", CheckResult.CONCERN, "Only 60%"),
        ]
        result = collect_key_uncertainties(checks, EnsembleSpread.LOW)
        assert len(result) == 1
        assert "coverage" in result[0]

    def test_high_spread_adds_entry(self):
        result = collect_key_uncertainties([], EnsembleSpread.HIGH)
        assert len(result) == 1
        assert "High ensemble spread" in result[0]

    def test_low_spread_no_entry(self):
        result = collect_key_uncertainties([], EnsembleSpread.LOW)
        assert len(result) == 0

    def test_priority_ordering(self):
        checks = [_make_check("c1", CheckResult.CONCERN, "concern detail")]
        result = collect_key_uncertainties(checks, EnsembleSpread.HIGH)
        assert "c1" in result[0]  # concern first
        assert "spread" in result[1].lower()  # spread second

    def test_capped_at_max(self):
        checks = [
            _make_check(f"check_{i}", CheckResult.CONCERN, f"detail {i}")
            for i in range(12)
        ]
        result = collect_key_uncertainties(checks, EnsembleSpread.HIGH)
        assert len(result) == 10


# ===================================================================
# TestRenderTier1
# ===================================================================


class TestRenderTier1:
    """Tier 1 Technical Brief rendering."""

    def test_contains_disclaimer(self):
        text = render_tier_1(**_tier_1_kwargs())
        assert NON_AUTHORITATIVE_DISCLAIMER in text

    def test_passes_guardrail_scan(self):
        text = render_tier_1(**_tier_1_kwargs())
        result = scan_text(text)
        assert result.passed, f"Guardrail violations: {result.violations}"

    def test_contains_mw(self):
        text = render_tier_1(**_tier_1_kwargs(mw_best=8.1))
        assert "Mw 8.1" in text

    def test_contains_constraint_stage(self):
        text = render_tier_1(**_tier_1_kwargs())
        assert "DART_CONSTRAINED" in text

    def test_empty_coastal_proxies(self):
        text = render_tier_1(**_tier_1_kwargs(coastal_proxies=[]))
        assert "No coastal amplitude proxies" in text

    def test_coastal_proxies_present(self):
        text = render_tier_1(**_tier_1_kwargs())
        assert "SITE_A" in text
        assert "P50=" in text


# ===================================================================
# TestRenderTier2
# ===================================================================


class TestRenderTier2:
    """Tier 2 Situational Awareness Summary - plain language."""

    def test_passes_guardrail_scan(self):
        text = render_tier_2(**_tier_2_kwargs())
        result = scan_text(text)
        assert result.passed, f"Guardrail violations: {result.violations}"

    def test_plain_language_no_rmse(self):
        text = render_tier_2(**_tier_2_kwargs())
        assert "RMSE" not in text
        assert "per-station" not in text.lower()

    def test_contains_coastal_proxies(self):
        text = render_tier_2(**_tier_2_kwargs())
        assert "SITE_A" in text

    def test_contains_situational_awareness_header(self):
        text = render_tier_2(**_tier_2_kwargs())
        assert "SITUATIONAL AWARENESS SUMMARY" in text

    def test_pass_with_concerns_text(self):
        text = render_tier_2(**_tier_2_kwargs(
            overall_outcome=VerificationOutcome.PASS_WITH_CONCERNS,
        ))
        assert "Verification passed with concerns" in text

    def test_pass_outcome_no_concerns_text(self):
        text = render_tier_2(**_tier_2_kwargs())
        assert "Verification passed." in text
        assert "concerns" not in text.lower()


# ===================================================================
# TestRenderTier3
# ===================================================================


class TestRenderTier3:
    """Tier 3 Post-Event Analysis."""

    def test_passes_guardrail_scan(self):
        text = render_tier_3(**_tier_3_kwargs())
        result = scan_text(text)
        assert result.passed, f"Guardrail violations: {result.violations}"

    def test_contains_provenance(self):
        text = render_tier_3(**_tier_3_kwargs())
        assert "prov-abc-123" in text

    def test_contains_excluded_detail(self):
        text = render_tier_3(**_tier_3_kwargs())
        assert "D46402" in text
        assert "Insufficient data coverage" in text

    def test_contains_post_event_header(self):
        text = render_tier_3(**_tier_3_kwargs())
        assert "POST-EVENT ANALYSIS" in text


# ===================================================================
# Coverage gap tests for report_templates.py
# ===================================================================


class TestFormatAssumptionsEmpty:
    """Empty assumptions list renders 'No limiting assumptions recorded.'"""

    def test_tier_1_no_assumptions(self):
        text = render_tier_1(**_tier_1_kwargs(limiting_assumptions=[]))
        assert "No limiting assumptions recorded." in text

    def test_tier_2_no_assumptions(self):
        text = render_tier_2(**_tier_2_kwargs(limiting_assumptions=[]))
        assert "No limiting assumptions recorded." in text


class TestTier2SeismicOnlyBranch:
    """Tier 2 SEISMIC_ONLY constraint renders 'seismic data only'."""

    def test_seismic_only_description(self):
        text = render_tier_2(**_tier_2_kwargs(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            num_dart_stations=0,
        ))
        assert "seismic data only" in text
        assert "no DART constraint" in text

    def test_multi_station_description(self):
        text = render_tier_2(**_tier_2_kwargs(
            constraint_stage=ConstraintStage.MULTI_STATION,
            num_dart_stations=3,
        ))
        assert "constrained by 3 DART stations" in text
