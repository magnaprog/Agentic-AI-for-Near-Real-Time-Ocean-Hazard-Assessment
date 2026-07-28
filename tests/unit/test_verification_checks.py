"""Unit tests for the verification checks and outcome policy.

Tests all 9 verification checks, geodetic helpers, outcome policy,
SEISMIC_ONLY mode behavior, and edge cases.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from hazard_assessment.agents.scenario_data import UnitSource
from hazard_assessment.agents.verification_checks import (
    HoldoutData,
    StationPosition,
    VerificationInput,
    _compute_azimuthal_spread,
    _initial_bearing,
    check_data_coverage,
    check_holdout_station,
    check_meteotsunami,
    check_model_fit,
    check_physical_consistency,
    check_posterior_stability,
    check_rayleigh_wave_suspect,
    check_sensitivity,
    check_tidal_state,
    determine_outcome,
    run_all_checks,
)
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckApplicability,
    CheckResult,
    PrerequisiteStatus,
    VerificationCheck,
    VerificationOutcome,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked(**overrides) -> RankedScenario:
    defaults = {
        "unit_source_ids": ["A01"],
        "weights": [1.0],
        "waveform_rmse_cm": 1.0,
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
        "top_scenarios": [_make_ranked()],
        "ensemble_spread": EnsembleSpread.LOW,
        "bilateral_rupture_evaluated": True,
    }
    defaults.update(overrides)
    return ScenarioAssessment(**defaults)


def _make_vi(**overrides) -> VerificationInput:
    """Create a VerificationInput with sensible defaults."""
    if "scenario" not in overrides:
        overrides["scenario"] = _make_assessment()
    return VerificationInput(**overrides)


def _make_source(source_id: str = "src_01", **kwargs) -> UnitSource:
    defaults = dict(
        source_id=source_id,
        latitude=0.0,
        longitude=0.0,
        depth_km=15.0,
        strike_deg=45.0,
        dip_deg=15.0,
        rake_deg=90.0,
        length_km=50.0,
        width_km=25.0,
        rigidity_pa=3.5e10,
        fault_zone_id="zone_A",
        segment_index=0,
    )
    defaults.update(kwargs)
    return UnitSource(**defaults)


def _make_coastal_proxy(
    tidal_correction_applied: bool = True,
    site_id: str = "site_A",
) -> CoastalProxy:
    return CoastalProxy(
        site_id=site_id,
        arrival_utc=datetime(2024, 1, 1, 12, 0, tzinfo=UTC),
        arrival_uncertainty_min=5.0,
        amplitude_proxy_p10_m=0.1,
        amplitude_proxy_p50_m=0.3,
        amplitude_proxy_p90_m=0.6,
        tidal_correction_applied=tidal_correction_applied,
    )


# ---------------------------------------------------------------------------
# Geodetic helpers
# ---------------------------------------------------------------------------


class TestInitialBearing:
    def test_north(self) -> None:
        bearing = _initial_bearing(0.0, 0.0, 10.0, 0.0)
        assert abs(bearing - 0.0) < 0.01

    def test_east(self) -> None:
        bearing = _initial_bearing(0.0, 0.0, 0.0, 10.0)
        assert abs(bearing - 90.0) < 0.1

    def test_south(self) -> None:
        bearing = _initial_bearing(0.0, 0.0, -10.0, 0.0)
        assert abs(bearing - 180.0) < 0.01

    def test_west(self) -> None:
        bearing = _initial_bearing(0.0, 0.0, 0.0, -10.0)
        assert abs(bearing - 270.0) < 0.1


class TestAzimuthalSpread:
    def test_single_bearing_returns_zero(self) -> None:
        assert _compute_azimuthal_spread([45.0]) == 0.0

    def test_empty_returns_zero(self) -> None:
        assert _compute_azimuthal_spread([]) == 0.0

    def test_opposite_bearings(self) -> None:
        spread = _compute_azimuthal_spread([0.0, 180.0])
        assert abs(spread - 180.0) < 0.01

    def test_three_evenly_spaced(self) -> None:
        spread = _compute_azimuthal_spread([0.0, 120.0, 240.0])
        assert abs(spread - 240.0) < 0.01

    def test_narrow_spread(self) -> None:
        spread = _compute_azimuthal_spread([10.0, 20.0])
        assert abs(spread - 10.0) < 0.01

    def test_wrap_around(self) -> None:
        spread = _compute_azimuthal_spread([350.0, 10.0])
        assert abs(spread - 20.0) < 0.01


# ---------------------------------------------------------------------------
# Check 1: Hold-out station validation
# ---------------------------------------------------------------------------


class TestCheckHoldoutStation:
    def test_no_holdout_is_not_evaluated(self) -> None:
        vi = _make_vi(holdout=None)
        result = check_holdout_station(vi)
        # Missing holdout data: prerequisite MISSING, honestly NOT_EVALUATED
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_empty_waveform_is_not_evaluated(self) -> None:
        """An empty holdout waveform degrades to NOT_EVALUATED instead of
        crashing in np.max (fail-closed, not fail-crash)."""
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([]),
            predicted_waveform=np.array([]),
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING
        assert "empty holdout waveform" in result.evidence

    def test_good_holdout_passes(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.01, 0.05, 0.02]),
            predicted_waveform=np.array([0.0, 0.012, 0.048, 0.019]),
            observed_arrival_index=2,
            predicted_arrival_index=2,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.PASS

    def test_arrival_concern(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.01, 0.05]),
            predicted_waveform=np.array([0.0, 0.01, 0.05]),
            observed_arrival_index=0,
            predicted_arrival_index=4,  # 4 * 60s = 4 min > 3 min
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.CONCERN

    def test_arrival_fail(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.01, 0.05]),
            predicted_waveform=np.array([0.0, 0.01, 0.05]),
            observed_arrival_index=0,
            predicted_arrival_index=6,  # 6 * 60s = 6 min > 5 min
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.FAIL

    def test_amplitude_concern(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.0, 0.10]),
            predicted_waveform=np.array([0.0, 0.0, 0.14]),  # 40% error
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.CONCERN

    def test_amplitude_fail(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.0, 0.10]),
            predicted_waveform=np.array([0.0, 0.0, 0.16]),  # 60% error
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.FAIL

    def test_tiny_observation_with_material_prediction_fails(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.0, 1e-12]),
            predicted_waveform=np.array([0.0, 0.0, 0.5]),
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.FAIL
        assert "obs_peak=1.00e-12 below floor" in result.evidence

    def test_both_tiny_peaks_zero_amplitude_error(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.0, 1e-12]),
            predicted_waveform=np.array([0.0, 0.0, 2e-12]),
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.PASS
        assert "both below floor" in result.evidence

    def test_arrival_index_none_skips_arrival_check(self) -> None:
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.01, 0.05]),
            predicted_waveform=np.array([0.0, 0.012, 0.048]),
            observed_arrival_index=None,
            predicted_arrival_index=5,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = check_holdout_station(vi)
        assert result.result == CheckResult.PASS
        assert "arrival_check=skipped" in result.evidence


# ---------------------------------------------------------------------------
# Check 2: Sensitivity analysis
# ---------------------------------------------------------------------------


class TestCheckSensitivity:
    def test_no_inversion_data_is_not_evaluated(self) -> None:
        vi = _make_vi(H=None, d=None, sources=None)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_single_source_is_not_evaluated(self) -> None:
        H = np.array([[1.0], [2.0], [3.0]])
        d = np.array([1.0, 2.0, 3.0])
        sources = [_make_source("s1")]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_single_active_source_passes(self) -> None:
        # Only source 0 has nonzero weight; dropping source 1 (inactive)
        # doesn't change anything.
        H = np.array([
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ])
        d = np.array([1.0, 1.0, 1.0])
        sources = [_make_source("s1"), _make_source("s2")]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.PASS

    def test_orthogonal_sources_concern(self) -> None:
        # Two orthogonal sources with equal weight. Dropping one shifts
        # the other's relative weight from 50% to 100% -> CONCERN.
        H = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ])
        d = np.array([2.0, 3.0, 2.0, 3.0])
        sources = [_make_source("s1"), _make_source("s2")]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.CONCERN

    def test_unstable_solution_fails(self) -> None:
        # Source 1 active in reference, but removing source 0 changes active set
        # Create a problem where sources are nearly collinear
        H = np.array([
            [1.0, 1.01],
            [2.0, 2.02],
            [3.0, 3.03],
            [4.0, 4.04],
        ])
        d = np.array([1.0, 2.0, 3.0, 4.0])
        sources = [_make_source("s1"), _make_source("s2")]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        # With near-collinear columns, dropping one should change active set
        assert result.result in (CheckResult.FAIL, CheckResult.CONCERN)

    def test_no_active_sources_is_not_evaluated(self) -> None:
        # d is zero -> all weights are zero -> no active set to perturb
        H = np.array([[1.0, 2.0], [3.0, 4.0]])
        d = np.array([0.0, 0.0])
        sources = [_make_source("s1"), _make_source("s2")]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_source_count_over_operational_cap_is_not_evaluated(self) -> None:
        H = np.eye(51)
        d = np.ones(51)
        sources = [_make_source(f"s{i}") for i in range(51)]
        vi = _make_vi(H=H, d=d, sources=sources)
        result = check_sensitivity(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING
        assert "n_sources=51 > 50" in result.evidence


# ---------------------------------------------------------------------------
# Check 3: Posterior stability
# ---------------------------------------------------------------------------


class TestCheckPosteriorStability:
    def test_first_assessment_is_not_evaluated(self) -> None:
        vi = _make_vi(previous_leading_source_ids=None)
        result = check_posterior_stability(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_unchanged_passes(self) -> None:
        vi = _make_vi(previous_leading_source_ids=["A01"])
        result = check_posterior_stability(vi)
        assert result.result == CheckResult.PASS

    def test_changed_raises_concern(self) -> None:
        vi = _make_vi(previous_leading_source_ids=["B01"])
        result = check_posterior_stability(vi)
        assert result.result == CheckResult.CONCERN

    def test_set_comparison_ignores_order(self) -> None:
        scenario = _make_assessment(
            top_scenarios=[
                _make_ranked(
                    unit_source_ids=["A01", "A02"],
                    weights=[1.0, 0.5],
                )
            ],
        )
        vi = _make_vi(
            scenario=scenario,
            previous_leading_source_ids=["A02", "A01"],
        )
        result = check_posterior_stability(vi)
        assert result.result == CheckResult.PASS


# ---------------------------------------------------------------------------
# Check 4: Data coverage
# ---------------------------------------------------------------------------


class TestCheckDataCoverage:
    def test_seismic_only_is_matrix_not_applicable(self) -> None:
        """SEISMIC_ONLY is handled by the requirement matrix, not by a
        special case inside the check: the matrix marks it NOT_APPLICABLE,
        so run_all_checks never executes it."""
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
        )
        vi = _make_vi(scenario=scenario)
        checks = {c.name: c for c in run_all_checks(vi)}
        cov = checks["data_coverage"]
        assert cov.result == CheckResult.NOT_APPLICABLE
        assert cov.applicability == CheckApplicability.NOT_APPLICABLE
        assert cov.prerequisite == PrerequisiteStatus.NOT_REQUIRED

    def test_one_station_fails(self) -> None:
        scenario = _make_assessment(
            dart_stations_used=["21413"],
        )
        vi = _make_vi(scenario=scenario)
        result = check_data_coverage(vi)
        assert result.result == CheckResult.FAIL

    def test_three_stations_with_complete_geometry_passes(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414", "21415"],
        )
        vi = _make_vi(
            scenario=scenario,
            station_positions=[
                StationPosition("21413", 10.0, 0.0),
                StationPosition("21414", -10.0, 0.0),
                StationPosition("21415", 0.0, 10.0),
            ],
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.PASS

    def test_two_stations_narrow_spread_concern(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        # Both stations due north of epicenter -> narrow spread
        positions = [
            StationPosition("21413", 10.0, 0.0),
            StationPosition("21414", 20.0, 0.0),
        ]
        vi = _make_vi(
            scenario=scenario,
            station_positions=positions,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.CONCERN

    def test_two_stations_wide_spread_passes(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        # Opposite sides -> wide spread
        positions = [
            StationPosition("21413", 10.0, 0.0),
            StationPosition("21414", -10.0, 0.0),
        ]
        vi = _make_vi(
            scenario=scenario,
            station_positions=positions,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.PASS

    def test_two_stations_no_positions_is_not_evaluated(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        vi = _make_vi(scenario=scenario, station_positions=None)
        result = check_data_coverage(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_two_stations_empty_positions_is_not_evaluated(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        vi = _make_vi(
            scenario=scenario,
            station_positions=[],
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_missing_constraining_station_position_is_not_evaluated(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        vi = _make_vi(
            scenario=scenario,
            station_positions=[StationPosition("21413", 10.0, 0.0)],
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING
        assert "21414" in result.evidence

    def test_extra_station_positions_do_not_change_constraining_geometry(self) -> None:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.MULTI_STATION,
            dart_stations_used=["21413", "21414"],
        )
        vi = _make_vi(
            scenario=scenario,
            station_positions=[
                StationPosition("21413", 10.0, 0.0),
                StationPosition("21414", 20.0, 0.0),
                StationPosition("unconstrained", -10.0, 0.0),
            ],
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_data_coverage(vi)
        assert result.result == CheckResult.CONCERN


# ---------------------------------------------------------------------------
# Check 5: Physical consistency
# ---------------------------------------------------------------------------


class TestCheckPhysicalConsistency:
    def test_no_seismic_mw_is_not_evaluated(self) -> None:
        vi = _make_vi(mw_seismic=None)
        result = check_physical_consistency(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_small_delta_passes(self) -> None:
        vi = _make_vi(mw_seismic=8.1)  # default mw_equivalent=8.0
        result = check_physical_consistency(vi)
        assert result.result == CheckResult.PASS

    def test_moderate_delta_concern(self) -> None:
        vi = _make_vi(mw_seismic=8.4)  # delta=0.4 > 0.3
        result = check_physical_consistency(vi)
        assert result.result == CheckResult.CONCERN

    def test_large_delta_fail(self) -> None:
        vi = _make_vi(mw_seismic=8.7)  # delta=0.7 > 0.6
        result = check_physical_consistency(vi)
        assert result.result == CheckResult.FAIL


# ---------------------------------------------------------------------------
# Check 6: Model fit quality
# ---------------------------------------------------------------------------


class TestCheckModelFit:
    def test_low_rmse_passes(self) -> None:
        vi = _make_vi()  # default rmse=1.0
        result = check_model_fit(vi)
        assert result.result == CheckResult.PASS

    def test_moderate_rmse_concern(self) -> None:
        scenario = _make_assessment(
            top_scenarios=[_make_ranked(waveform_rmse_cm=4.0)],
        )
        vi = _make_vi(scenario=scenario)
        result = check_model_fit(vi)
        assert result.result == CheckResult.CONCERN

    def test_high_rmse_fail(self) -> None:
        scenario = _make_assessment(
            top_scenarios=[_make_ranked(waveform_rmse_cm=6.0)],
        )
        vi = _make_vi(scenario=scenario)
        result = check_model_fit(vi)
        assert result.result == CheckResult.FAIL

    def test_bias_detected_with_systematic_residual(self) -> None:
        # Green's function has a spike shape; observation has a broader
        # shape that can't be fully captured. NNLS fits the peak,
        # leaving positive residuals at all other timepoints for both
        # stations -> systematic bias.
        n_stations = 2
        n_tp = 5
        H = np.zeros((n_stations * n_tp, 1))
        d = np.zeros(n_stations * n_tp)
        for k in range(n_stations):
            # Source predicts a spike at t=0 only
            H[k * n_tp, 0] = 1.0
            # Observation is broad: 1.0 at t=0, 0.1 at t=1..4
            d[k * n_tp] = 1.0
            d[k * n_tp + 1 : (k + 1) * n_tp] = 0.1
        # NNLS fits alpha=1.0. Residual = [0, 0.1, 0.1, 0.1, 0.1] per station.
        # Per-station mean = 0.08 m. Grand mean = 0.08 m.
        # std = 0 (uniform) -> uniform offset branch.
        # 0.08 * 100 = 8.0 cm > 1.0 cm floor -> bias detected.

        scenario = _make_assessment(
            dart_stations_used=["s1", "s2"],
            constraint_stage=ConstraintStage.MULTI_STATION,
            top_scenarios=[_make_ranked(waveform_rmse_cm=2.0)],
        )
        vi = _make_vi(scenario=scenario, H=H, d=d)
        result = check_model_fit(vi)
        assert result.result == CheckResult.FAIL
        assert "bias" in result.evidence

    def test_no_bias_with_single_station(self) -> None:
        # Single station: bias test skipped
        n_tp = 10
        H = np.ones((n_tp, 1))
        d = np.ones(n_tp) * 1.05

        scenario = _make_assessment(
            dart_stations_used=["s1"],
            top_scenarios=[_make_ranked(waveform_rmse_cm=2.0)],
        )
        vi = _make_vi(scenario=scenario, H=H, d=d)
        result = check_model_fit(vi)
        # Only RMSE matters, no bias test
        assert result.result == CheckResult.PASS

    def test_no_bias_without_inversion_data(self) -> None:
        vi = _make_vi()  # No H or d
        result = check_model_fit(vi)
        assert result.result == CheckResult.PASS

    def test_bias_detected_via_z_test(self) -> None:
        # 3 stations with different residual magnitudes but consistent
        # positive bias. Station 0 has small positive residual, station 1
        # moderate, station 2 large - all positive.
        # Spike-shaped Green's function: each station has a spike at t=0.
        # Observation: spike at t=0, plus positive offset at other timepoints
        # that differs per station.
        n_stations = 3
        n_tp = 5
        H = np.zeros((n_stations * n_tp, 1))
        d = np.zeros(n_stations * n_tp)
        offsets = [0.03, 0.05, 0.07]  # per-station positive residual
        for k in range(n_stations):
            H[k * n_tp, 0] = 1.0
            d[k * n_tp] = 1.0
            d[k * n_tp + 1 : (k + 1) * n_tp] = offsets[k]
        # NNLS fits alpha=1.0. Per-station mean residuals:
        #   station 0: mean([0, 0.03, 0.03, 0.03, 0.03]) = 0.024
        #   station 1: mean([0, 0.05, 0.05, 0.05, 0.05]) = 0.040
        #   station 2: mean([0, 0.07, 0.07, 0.07, 0.07]) = 0.056
        # Grand mean = 0.04 m = 4.0 cm > 1.0 cm floor.
        # std = 0.016, z = 0.04 / (0.016/sqrt(3)) ~ 4.33 > 3.0 -> bias.

        scenario = _make_assessment(
            dart_stations_used=["s1", "s2", "s3"],
            constraint_stage=ConstraintStage.MULTI_STATION,
            top_scenarios=[_make_ranked(waveform_rmse_cm=2.0)],
        )
        vi = _make_vi(scenario=scenario, H=H, d=d)
        result = check_model_fit(vi)
        assert result.result == CheckResult.FAIL
        assert "bias" in result.evidence
        assert "z=" in result.evidence  # z-test branch, not uniform branch

    def test_small_bias_below_floor_passes(self) -> None:
        # Two stations, tiny systematic offset below 1cm floor
        n_stations = 2
        n_tp = 10
        H = np.zeros((n_stations * n_tp, 1))
        d = np.zeros(n_stations * n_tp)
        for k in range(n_stations):
            H[k * n_tp : (k + 1) * n_tp, 0] = 1.0
            d[k * n_tp : (k + 1) * n_tp] = 1.005  # offset=0.005m = 0.5cm < 1cm
        scenario = _make_assessment(
            dart_stations_used=["s1", "s2"],
            constraint_stage=ConstraintStage.MULTI_STATION,
            top_scenarios=[_make_ranked(waveform_rmse_cm=1.0)],
        )
        vi = _make_vi(scenario=scenario, H=H, d=d)
        result = check_model_fit(vi)
        assert result.result == CheckResult.PASS


# ---------------------------------------------------------------------------
# Check 7: Tidal state review
# ---------------------------------------------------------------------------


class TestCheckTidalState:
    def test_no_proxies_is_not_evaluated(self) -> None:
        vi = _make_vi()  # No coastal_proxies
        result = check_tidal_state(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING

    def test_all_corrected_passes(self) -> None:
        scenario = _make_assessment(
            coastal_proxies=[_make_coastal_proxy(tidal_correction_applied=True)],
        )
        vi = _make_vi(scenario=scenario)
        result = check_tidal_state(vi)
        assert result.result == CheckResult.PASS

    def test_uncorrected_when_needed_fails(self) -> None:
        scenario = _make_assessment(
            coastal_proxies=[_make_coastal_proxy(tidal_correction_applied=False)],
        )
        vi = _make_vi(scenario=scenario, tidal_correction_needed=True)
        result = check_tidal_state(vi)
        assert result.result == CheckResult.FAIL

    def test_uncorrected_when_not_needed_concern(self) -> None:
        scenario = _make_assessment(
            coastal_proxies=[_make_coastal_proxy(tidal_correction_applied=False)],
        )
        vi = _make_vi(scenario=scenario, tidal_correction_needed=False)
        result = check_tidal_state(vi)
        assert result.result == CheckResult.CONCERN


# ---------------------------------------------------------------------------
# Check 8: Meteotsunami discriminator
# ---------------------------------------------------------------------------


class TestCheckMeteotsunami:
    def test_none_is_not_evaluated(self) -> None:
        vi = _make_vi(meteotsunami_score=None)
        result = check_meteotsunami(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING
        assert "No meteotsunami score available" in result.evidence

    def test_zero_score_passes_without_evaluated_claim(self) -> None:
        """A literal 0.0 is a low evaluated score, nothing more.

        The old 0.0 branch claimed "no correlation detected", which
        dressed up the placeholder as an evaluated-clean result.
        """
        vi = _make_vi(meteotsunami_score=0.0)
        result = check_meteotsunami(vi)
        assert result.result == CheckResult.PASS
        assert result.evidence == "meteotsunami_score=0.00"

    def test_low_score_passes(self) -> None:
        vi = _make_vi(meteotsunami_score=0.1)
        result = check_meteotsunami(vi)
        assert result.result == CheckResult.PASS

    def test_moderate_score_concern(self) -> None:
        vi = _make_vi(meteotsunami_score=0.4)
        result = check_meteotsunami(vi)
        assert result.result == CheckResult.CONCERN

    def test_high_score_fail(self) -> None:
        vi = _make_vi(meteotsunami_score=0.7)
        result = check_meteotsunami(vi)
        assert result.result == CheckResult.FAIL


# ---------------------------------------------------------------------------
# check_rayleigh_wave_suspect (Check 9)
# ---------------------------------------------------------------------------


class TestCheckRayleighWaveSuspect:
    """Tests for the Rayleigh wave false-trigger detection check."""

    def test_not_suspect_returns_pass(self) -> None:
        vi = _make_vi(rayleigh_wave_suspect=False)
        result = check_rayleigh_wave_suspect(vi)
        assert result.result == CheckResult.PASS
        assert result.name == "rayleigh_wave_suspect"
        assert "No Rayleigh wave timing correlation" in result.evidence

    def test_suspect_returns_concern_not_fail(self) -> None:
        """Rayleigh suspect should flag CONCERN, never FAIL.

        Real tsunamis also follow large earthquakes, so this check
        increases scrutiny without auto-dismissing the event.
        """
        vi = _make_vi(rayleigh_wave_suspect=True)
        result = check_rayleigh_wave_suspect(vi)
        assert result.result == CheckResult.CONCERN
        assert result.result != CheckResult.FAIL

    def test_suspect_evidence_mentions_timing(self) -> None:
        vi = _make_vi(rayleigh_wave_suspect=True)
        result = check_rayleigh_wave_suspect(vi)
        assert "Rayleigh wave" in result.evidence
        assert "3.6 km/s" in result.evidence

    def test_default_vi_is_not_evaluated(self) -> None:
        """Default VerificationInput has rayleigh_wave_suspect=None (not
        evaluated), which is NOT_EVALUATED with prerequisite MISSING,
        distinct from an evaluated, clean False."""
        vi = _make_vi()
        assert vi.rayleigh_wave_suspect is None
        result = check_rayleigh_wave_suspect(vi)
        assert result.result == CheckResult.NOT_EVALUATED
        assert result.prerequisite == PrerequisiteStatus.MISSING
        assert "not evaluated" in result.evidence

    def test_suspect_alone_does_not_cause_abstain(self) -> None:
        """Rayleigh CONCERN alone should not trigger ABSTAIN.

        We satisfy the DART_CONSTRAINED required set (coverage,
        physical consistency, model fit) so the only elevated result is
        the Rayleigh CONCERN plus blocked OPTIONAL checks.
        """
        scenario = _make_assessment(
            dart_stations_used=["21418", "21401"],
        )
        vi = _make_vi(
            scenario=scenario,
            rayleigh_wave_suspect=True,
            mw_seismic=8.1,
            station_positions=[
                StationPosition("21418", 10.0, 0.0),
                StationPosition("21401", -10.0, 0.0),
            ],
            epicenter_lat=0.0,
            epicenter_lon=0.0,
        )
        result = check_rayleigh_wave_suspect(vi)
        assert result.result == CheckResult.CONCERN
        # The overall outcome should be PASS_WITH_CONCERNS (not FAIL)
        checks = run_all_checks(vi)
        outcome, abstain_required, _ = determine_outcome(checks)
        assert not abstain_required
        assert outcome == VerificationOutcome.PASS_WITH_CONCERNS


# ---------------------------------------------------------------------------
# SEISMIC_ONLY mode
# ---------------------------------------------------------------------------


class TestSeismicOnlyMode:
    """SEISMIC_ONLY: matrix marks inversion checks NOT_APPLICABLE and
    requires physical consistency; missing evidence is NOT_EVALUATED."""

    def _make_seismic_only_vi(self, **overrides) -> VerificationInput:
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
            top_scenarios=[_make_ranked(waveform_rmse_cm=0.0)],
        )
        return _make_vi(scenario=scenario, **overrides)

    def test_holdout_is_not_evaluated(self) -> None:
        vi = self._make_seismic_only_vi()
        assert check_holdout_station(vi).result == CheckResult.NOT_EVALUATED

    def test_sensitivity_is_not_evaluated(self) -> None:
        vi = self._make_seismic_only_vi()
        assert check_sensitivity(vi).result == CheckResult.NOT_EVALUATED

    def test_posterior_stability_is_not_evaluated(self) -> None:
        vi = self._make_seismic_only_vi()
        assert (
            check_posterior_stability(vi).result == CheckResult.NOT_EVALUATED
        )

    def test_meteotsunami_is_not_evaluated(self) -> None:
        vi = self._make_seismic_only_vi()
        assert check_meteotsunami(vi).result == CheckResult.NOT_EVALUATED

    def test_inversion_checks_are_not_applicable(self) -> None:
        """Regression: the four inversion-dependent checks carry the
        canonical NOT_APPLICABLE row instead of trivial PASSes."""
        vi = self._make_seismic_only_vi()
        checks = {c.name: c for c in run_all_checks(vi)}
        for name in (
            "holdout_station_validation",
            "sensitivity_analysis",
            "data_coverage",
            "model_fit_quality",
        ):
            assert checks[name].result == CheckResult.NOT_APPLICABLE, name
            assert (
                checks[name].applicability
                == CheckApplicability.NOT_APPLICABLE
            ), name

    def test_missing_mw_makes_outcome_incomplete(self) -> None:
        """Physical consistency is REQUIRED at SEISMIC_ONLY. Without a
        seismic magnitude the prior-only scenario is unverifiable:
        INCOMPLETE, abstain required."""
        vi = self._make_seismic_only_vi()
        checks = run_all_checks(vi)
        outcome, abstain_required, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.INCOMPLETE
        assert abstain_required
        assert "physical_consistency" in (reason or "")

    def test_with_mw_outcome_is_pass_with_concerns(self) -> None:
        """With the required Mw comparison satisfied, blocked OPTIONAL
        checks degrade the outcome to PASS_WITH_CONCERNS, not lower."""
        vi = self._make_seismic_only_vi(mw_seismic=8.1)
        checks = run_all_checks(vi)
        outcome, abstain_required, _ = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS_WITH_CONCERNS
        assert not abstain_required


# ---------------------------------------------------------------------------
# run_all_checks
# ---------------------------------------------------------------------------


class TestRunAllChecks:
    def test_returns_9_checks(self) -> None:
        vi = _make_vi()
        checks = run_all_checks(vi)
        assert len(checks) == 9

    def test_check_names_are_unique(self) -> None:
        vi = _make_vi()
        checks = run_all_checks(vi)
        names = [c.name for c in checks]
        assert len(names) == len(set(names))

    def test_applicability_stamped_from_matrix(self) -> None:
        vi = _make_vi()  # DART_CONSTRAINED default
        checks = {c.name: c for c in run_all_checks(vi)}
        assert (
            checks["data_coverage"].applicability
            == CheckApplicability.REQUIRED
        )
        assert (
            checks["physical_consistency"].applicability
            == CheckApplicability.REQUIRED
        )
        assert (
            checks["model_fit_quality"].applicability
            == CheckApplicability.REQUIRED
        )
        assert (
            checks["holdout_station_validation"].applicability
            == CheckApplicability.OPTIONAL
        )

    def test_raising_check_is_captured_as_error(self, monkeypatch) -> None:
        """A check that raises must surface as an ERROR row, not abort
        the whole verification."""
        import hazard_assessment.agents.verification_checks as vc

        def _boom(vi: VerificationInput) -> VerificationCheck:
            raise RuntimeError("synthetic check crash")

        monkeypatch.setitem(vc.CHECK_FUNCTIONS, "model_fit_quality", _boom)
        scenario = _make_assessment(dart_stations_used=["21413", "21414"])
        vi = _make_vi(scenario=scenario, mw_seismic=8.1)
        checks = {c.name: c for c in run_all_checks(vi)}
        err = checks["model_fit_quality"]
        assert err.result == CheckResult.ERROR
        assert err.prerequisite == PrerequisiteStatus.ERROR
        assert "synthetic check crash" in err.evidence
        # model_fit_quality is REQUIRED at DART_CONSTRAINED: the errored
        # required check must drive INCOMPLETE + abstain.
        outcome, abstain_required, reason = determine_outcome(
            list(checks.values())
        )
        assert outcome == VerificationOutcome.INCOMPLETE
        assert abstain_required
        assert "model_fit_quality" in (reason or "")


# ---------------------------------------------------------------------------
# Outcome policy
# ---------------------------------------------------------------------------


class TestDetermineOutcome:
    def test_empty_checks_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            determine_outcome([])

    def test_all_pass(self) -> None:
        checks = [
            VerificationCheck(name=f"c{i}", result=CheckResult.PASS, evidence="ok")
            for i in range(8)
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS
        assert not abstain
        assert reason is None

    def test_single_concern(self) -> None:
        checks = [
            VerificationCheck(name="c1", result=CheckResult.CONCERN, evidence="hmm"),
            VerificationCheck(name="c2", result=CheckResult.PASS, evidence="ok"),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS_WITH_CONCERNS
        assert not abstain
        assert reason is None

    def test_single_fail(self) -> None:
        checks = [
            VerificationCheck(name="c1", result=CheckResult.FAIL, evidence="bad"),
            VerificationCheck(name="c2", result=CheckResult.PASS, evidence="ok"),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.FAIL
        assert abstain
        assert reason is not None
        assert "c1" in reason

    def test_mixed_fail_and_concern(self) -> None:
        checks = [
            VerificationCheck(name="c1", result=CheckResult.FAIL, evidence="bad"),
            VerificationCheck(name="c2", result=CheckResult.CONCERN, evidence="hmm"),
            VerificationCheck(name="c3", result=CheckResult.PASS, evidence="ok"),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.FAIL
        assert abstain

    def test_all_fail(self) -> None:
        checks = [
            VerificationCheck(name=f"c{i}", result=CheckResult.FAIL, evidence=f"bad{i}")
            for i in range(8)
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.FAIL
        assert abstain
        # All check names from this test should appear in reason
        for i in range(8):
            assert f"c{i}" in reason

    def test_single_fail_among_seven_pass(self) -> None:
        checks = [
            VerificationCheck(name=f"c{i}", result=CheckResult.PASS, evidence="ok")
            for i in range(7)
        ]
        checks.append(
            VerificationCheck(name="c_fail", result=CheckResult.FAIL, evidence="bad")
        )
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.FAIL
        assert abstain
        assert "c_fail" in reason

    def _not_eval(
        self, name: str, applicability: CheckApplicability
    ) -> VerificationCheck:
        return VerificationCheck(
            name=name,
            result=CheckResult.NOT_EVALUATED,
            evidence="input data absent",
            applicability=applicability,
            prerequisite=PrerequisiteStatus.MISSING,
        )

    def _error(
        self, name: str, applicability: CheckApplicability
    ) -> VerificationCheck:
        return VerificationCheck(
            name=name,
            result=CheckResult.ERROR,
            evidence="check raised",
            applicability=applicability,
            prerequisite=PrerequisiteStatus.ERROR,
        )

    def test_required_missing_prerequisite_is_incomplete(self) -> None:
        checks = [
            VerificationCheck(name="c0", result=CheckResult.PASS, evidence="ok"),
            self._not_eval("c_req", CheckApplicability.REQUIRED),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.INCOMPLETE
        assert abstain
        assert "c_req" in reason
        assert "MISSING" in reason

    def test_required_errored_prerequisite_is_incomplete(self) -> None:
        checks = [
            VerificationCheck(name="c0", result=CheckResult.PASS, evidence="ok"),
            self._error("c_req", CheckApplicability.REQUIRED),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.INCOMPLETE
        assert abstain
        assert "c_req" in reason

    def test_optional_missing_prerequisite_is_pass_with_concerns(self) -> None:
        checks = [
            VerificationCheck(name="c0", result=CheckResult.PASS, evidence="ok"),
            self._not_eval("c_opt", CheckApplicability.OPTIONAL),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS_WITH_CONCERNS
        assert not abstain

    def test_optional_errored_prerequisite_is_pass_with_concerns(self) -> None:
        checks = [
            VerificationCheck(name="c0", result=CheckResult.PASS, evidence="ok"),
            self._error("c_opt", CheckApplicability.OPTIONAL),
        ]
        outcome, abstain, _ = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS_WITH_CONCERNS
        assert not abstain

    def test_no_applicable_checks_is_incomplete(self) -> None:
        checks = [
            VerificationCheck(
                name=f"c{i}",
                result=CheckResult.NOT_APPLICABLE,
                evidence="matrix",
                applicability=CheckApplicability.NOT_APPLICABLE,
                prerequisite=PrerequisiteStatus.NOT_REQUIRED,
            )
            for i in range(3)
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.INCOMPLETE
        assert abstain
        assert "no applicable checks" in reason

    def test_not_applicable_rows_do_not_degrade_pass(self) -> None:
        checks = [
            VerificationCheck(name="c0", result=CheckResult.PASS, evidence="ok"),
            VerificationCheck(
                name="c_na",
                result=CheckResult.NOT_APPLICABLE,
                evidence="matrix",
                applicability=CheckApplicability.NOT_APPLICABLE,
                prerequisite=PrerequisiteStatus.NOT_REQUIRED,
            ),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS
        assert not abstain
        assert reason is None

    def test_fail_takes_precedence_over_incomplete(self) -> None:
        """A positive FAIL verdict outranks blocked required checks."""
        checks = [
            self._not_eval("c_req", CheckApplicability.REQUIRED),
            VerificationCheck(
                name="c1", result=CheckResult.FAIL, evidence="rmse too high",
            ),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.FAIL
        assert abstain
        assert "c1" in reason

    def test_evidence_text_carries_no_outcome_semantics(self) -> None:
        """Genuine PASS rows stay PASS regardless of their evidence
        wording. Outcome comes from the result field, never from a string
        scan of the evidence text."""
        checks = [
            VerificationCheck(
                name="c0", result=CheckResult.PASS,
                evidence="No inversion data available",
            ),
            VerificationCheck(
                name="c1", result=CheckResult.PASS,
                evidence="First assessment",
            ),
        ]
        outcome, abstain, reason = determine_outcome(checks)
        assert outcome == VerificationOutcome.PASS
        assert not abstain
