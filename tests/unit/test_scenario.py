"""Unit tests for the ScenarioAssessment schema.

Validates constraint stage consistency, exclusion_reasons matching,
percentile ordering, NNLS weight non-negativity, and disclaimer enforcement.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from hazard_assessment.schemas.scenario import (
    INUNDATION_DISCLAIMER,
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scenario(**overrides) -> RankedScenario:
    defaults = {
        "unit_source_ids": ["A01"],
        "weights": [1.0],
        "waveform_rmse_cm": 0.5,
        "mw_equivalent": 8.0,
        "rank": 1,
        "posterior_weight": 0.8,
    }
    defaults.update(overrides)
    return RankedScenario(**defaults)


def _make_assessment(**overrides) -> ScenarioAssessment:
    defaults = {
        "producer": "scenario_agent",
        "constraint_stage": ConstraintStage.DART_CONSTRAINED,
        "dart_stations_used": ["21413"],
        "dart_stations_excluded": [],
        "exclusion_reasons": {},
        "inversion_window_sec": 3600,
        "top_scenarios": [_make_scenario()],
        "ensemble_spread": EnsembleSpread.LOW,
        "bilateral_rupture_evaluated": True,
    }
    defaults.update(overrides)
    return ScenarioAssessment(**defaults)


# ---------------------------------------------------------------------------
# RankedScenario
# ---------------------------------------------------------------------------


class TestRankedScenario:
    def test_valid_scenario(self) -> None:
        s = _make_scenario()
        assert s.rank == 1

    def test_weights_must_match_sources(self) -> None:
        with pytest.raises(ValidationError, match="same length"):
            _make_scenario(unit_source_ids=["A01", "A02"], weights=[1.0])

    def test_negative_weight_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-negative"):
            _make_scenario(
                unit_source_ids=["A01", "A02"],
                weights=[1.0, -0.5],
            )

    def test_nan_weight_rejected(self) -> None:
        # NaN slips past `w < 0` (nan < 0 is False); it must be rejected.
        with pytest.raises(ValidationError, match="finite"):
            _make_scenario(
                unit_source_ids=["A01", "A02"],
                weights=[1.0, float("nan")],
            )

    def test_empty_sources_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_scenario(unit_source_ids=[], weights=[])

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_scenario(rogue_field="bad")


# ---------------------------------------------------------------------------
# CoastalProxy percentile ordering
# ---------------------------------------------------------------------------


class TestCoastalProxy:
    def test_valid_proxy(self) -> None:
        proxy = CoastalProxy(
            site_id="HILO",
            arrival_utc=datetime(2026, 1, 1, tzinfo=UTC),
            arrival_uncertainty_min=15.0,
            amplitude_proxy_p10_m=0.1,
            amplitude_proxy_p50_m=0.3,
            amplitude_proxy_p90_m=0.8,
            tidal_correction_applied=True,
        )
        assert proxy.site_id == "HILO"

    def test_percentile_order_p10_gt_p50_rejected(self) -> None:
        with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
            CoastalProxy(
                site_id="HILO",
                arrival_utc=datetime(2026, 1, 1, tzinfo=UTC),
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=0.5,
                amplitude_proxy_p50_m=0.3,
                amplitude_proxy_p90_m=0.8,
                tidal_correction_applied=True,
            )

    def test_percentile_order_p50_gt_p90_rejected(self) -> None:
        with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
            CoastalProxy(
                site_id="HILO",
                arrival_utc=datetime(2026, 1, 1, tzinfo=UTC),
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=0.1,
                amplitude_proxy_p50_m=0.9,
                amplitude_proxy_p90_m=0.8,
                tidal_correction_applied=True,
            )

    def test_non_utc_timezone_rejected(self) -> None:
        """arrival_utc must be UTC - non-UTC offsets are rejected."""
        jst = timezone(timedelta(hours=9))  # UTC+09:00
        with pytest.raises(ValidationError, match="arrival_utc must be UTC"):
            CoastalProxy(
                site_id="HILO",
                arrival_utc=datetime(2026, 1, 1, tzinfo=jst),
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=0.1,
                amplitude_proxy_p50_m=0.3,
                amplitude_proxy_p90_m=0.8,
                tidal_correction_applied=True,
            )


# ---------------------------------------------------------------------------
# ScenarioAssessment constraint stage
# ---------------------------------------------------------------------------


class TestConstraintStageConsistency:
    def test_seismic_only_no_dart_stations(self) -> None:
        result = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
        )
        assert result.constraint_stage == ConstraintStage.SEISMIC_ONLY

    def test_seismic_only_with_dart_rejected(self) -> None:
        with pytest.raises(ValidationError, match="SEISMIC_ONLY.*empty"):
            _make_assessment(
                constraint_stage=ConstraintStage.SEISMIC_ONLY,
                dart_stations_used=["21413"],
            )

    def test_dart_constrained_needs_one_station(self) -> None:
        with pytest.raises(ValidationError, match="DART_CONSTRAINED.*>= 1"):
            _make_assessment(
                constraint_stage=ConstraintStage.DART_CONSTRAINED,
                dart_stations_used=[],
            )

    def test_multi_station_needs_two_stations(self) -> None:
        with pytest.raises(ValidationError, match="MULTI_STATION.*>= 2"):
            _make_assessment(
                constraint_stage=ConstraintStage.MULTI_STATION,
                dart_stations_used=["21413"],
            )

    def test_multi_station_with_two_accepted(self) -> None:
        result = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21415"],
        )
        assert len(result.dart_stations_used) == 2


# ---------------------------------------------------------------------------
# Exclusion reasons validation
# ---------------------------------------------------------------------------


class TestExclusionReasons:
    def test_matching_exclusion_reasons_accepted(self) -> None:
        result = _make_assessment(
            dart_stations_excluded=["21415", "21418"],
            exclusion_reasons={
                "21415": "QC flag 4: suspect data",
                "21418": "Station offline",
            },
        )
        assert len(result.exclusion_reasons) == 2

    def test_missing_reason_for_excluded_station_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exclusion_reasons.*match"):
            _make_assessment(
                dart_stations_excluded=["21415", "21418"],
                exclusion_reasons={"21415": "QC flag 4"},
            )

    def test_extra_reason_for_non_excluded_station_rejected(self) -> None:
        with pytest.raises(ValidationError, match="exclusion_reasons.*match"):
            _make_assessment(
                dart_stations_excluded=["21415"],
                exclusion_reasons={
                    "21415": "QC flag 4",
                    "21418": "Should not be here",
                },
            )

    def test_both_empty_accepted(self) -> None:
        result = _make_assessment(
            dart_stations_excluded=[],
            exclusion_reasons={},
        )
        assert len(result.exclusion_reasons) == 0

    def test_empty_string_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            _make_assessment(
                dart_stations_excluded=["21415"],
                exclusion_reasons={"21415": ""},
            )

    def test_whitespace_only_reason_rejected(self) -> None:
        with pytest.raises(ValidationError, match="non-empty"):
            _make_assessment(
                dart_stations_excluded=["21415"],
                exclusion_reasons={"21415": "   "},
            )


# ---------------------------------------------------------------------------
# Disclaimer enforcement
# ---------------------------------------------------------------------------


class TestDisclaimerEnforcement:
    def test_default_disclaimer_accepted(self) -> None:
        result = _make_assessment()
        assert result.inundation_disclaimer == INUNDATION_DISCLAIMER

    def test_custom_disclaimer_rejected(self) -> None:
        with pytest.raises(ValidationError, match="standard text"):
            _make_assessment(inundation_disclaimer="Custom text not permitted")


# ---------------------------------------------------------------------------
# Extra fields
# ---------------------------------------------------------------------------


class TestScenarioAssessmentExtraFields:
    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            _make_assessment(rogue_field="bad")
