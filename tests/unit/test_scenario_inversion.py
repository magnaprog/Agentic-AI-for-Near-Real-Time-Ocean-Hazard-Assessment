"""Unit tests for the scenario inversion service.

Tests NNLS inversion, moment magnitude calculation, scenario ranking,
seismic-only mode, observation/Green's matrix construction,
bootstrap uncertainty estimation, ensemble spread classification,
bootstrap-based scenario ranking, and coastal amplitude proxy generation.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from numpy.typing import NDArray

from hazard_assessment.agents.scenario_data import (
    CoastalForecastFactors,
    GreensFunctionSet,
    InMemoryUnitSourceDatabase,
    UnitSource,
)
from hazard_assessment.agents.scenario_inversion import (
    BootstrapConfig,
    BootstrapResult,
    InversionResult,
    SeismicOnlyConfig,
    build_greens_matrix,
    build_observation_vector,
    classify_ensemble_spread,
    compute_coastal_proxies,
    compute_mw_from_weights,
    compute_seismic_only_weights,
    rank_scenarios,
    rank_scenarios_from_bootstrap,
    run_bootstrap,
    run_seismic_only_estimate,
    solve_nnls,
)
from hazard_assessment.schemas.scenario import EnsembleSpread

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_source(
    source_id: str = "src_01",
    lat: float = 0.0,
    lon: float = 0.0,
    length_km: float = 50.0,
    width_km: float = 25.0,
    rigidity_pa: float = 3.5e10,
    segment_index: int = 0,
    **kwargs,
) -> UnitSource:
    defaults = dict(
        source_id=source_id,
        latitude=lat,
        longitude=lon,
        depth_km=15.0,
        strike_deg=45.0,
        dip_deg=15.0,
        rake_deg=90.0,
        length_km=length_km,
        width_km=width_km,
        rigidity_pa=rigidity_pa,
        fault_zone_id="zone_A",
        segment_index=segment_index,
    )
    defaults.update(kwargs)
    return UnitSource(**defaults)


def _make_greens(
    n_stations: int = 2,
    n_timepoints: int = 60,
    n_sources: int = 3,
) -> GreensFunctionSet:
    rng = np.random.default_rng(42)
    return GreensFunctionSet(
        source_ids=[f"src_{i:02d}" for i in range(n_sources)],
        station_ids=[f"dart_{i:02d}" for i in range(n_stations)],
        time_step_sec=60.0,
        n_timepoints=n_timepoints,
        waveforms=rng.standard_normal((n_stations, n_timepoints, n_sources)),
    )


# ---------------------------------------------------------------------------
# build_observation_vector
# ---------------------------------------------------------------------------


class TestBuildObservationVector:
    def test_shape(self):
        waveforms = {
            "d1": np.ones(60, dtype=np.float64),
            "d2": np.ones(60, dtype=np.float64) * 2,
        }
        d = build_observation_vector(waveforms, ["d1", "d2"])
        assert d.shape == (120,)
        assert d[0] == 1.0
        assert d[60] == 2.0

    def test_length_mismatch_raises(self):
        waveforms = {
            "d1": np.ones(60, dtype=np.float64),
            "d2": np.ones(50, dtype=np.float64),
        }
        with pytest.raises(ValueError, match="mismatch"):
            build_observation_vector(waveforms, ["d1", "d2"])

    def test_truncation(self):
        waveforms = {
            "d1": np.ones(100, dtype=np.float64),
            "d2": np.ones(100, dtype=np.float64) * 2,
        }
        d = build_observation_vector(waveforms, ["d1", "d2"], n_timepoints=60)
        assert d.shape == (120,)

    def test_truncation_too_short_raises(self):
        waveforms = {
            "d1": np.ones(30, dtype=np.float64),
        }
        with pytest.raises(ValueError, match="need at least"):
            build_observation_vector(waveforms, ["d1"], n_timepoints=60)

    def test_empty_station_ids_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            build_observation_vector({}, [])


# ---------------------------------------------------------------------------
# build_greens_matrix
# ---------------------------------------------------------------------------


class TestBuildGreensMatrix:
    def test_shape(self):
        greens = _make_greens(n_stations=2, n_timepoints=60, n_sources=3)
        H = build_greens_matrix(greens, greens.station_ids)
        assert H.shape == (120, 3)

    def test_values_match(self):
        greens = _make_greens(n_stations=1, n_timepoints=10, n_sources=2)
        H = build_greens_matrix(greens, greens.station_ids)
        # First column should match waveforms[0, :, 0]
        np.testing.assert_array_equal(H[:10, 0], greens.waveforms[0, :, 0])


# ---------------------------------------------------------------------------
# compute_mw_from_weights
# ---------------------------------------------------------------------------


class TestComputeMw:
    def test_known_mw(self):
        # M0 = 1e22 N*m -> Mw = (2/3)(log10(1e22) - 9.1) = (2/3)(22 - 9.1) = 8.6
        # Set up: slip=1m, area = 1e22 / (1 * rigidity), rigidity = 1
        src = _make_source(length_km=1.0, width_km=1.0, rigidity_pa=1.0)
        # area_m2 = 1e6, so M0 = weight * 1e6 * 1.0
        # Need M0 = 1e22 -> weight = 1e22 / 1e6 = 1e16
        weights = np.array([1e16], dtype=np.float64)
        mw = compute_mw_from_weights(weights, [src])
        expected = (2.0 / 3.0) * (math.log10(1e22) - 9.1)
        assert abs(mw - expected) < 0.01

    def test_zero_weights(self):
        src = _make_source()
        weights = np.array([0.0], dtype=np.float64)
        assert compute_mw_from_weights(weights, [src]) == 0.0

    def test_near_zero_clamped(self):
        # Very small M0 -> negative Mw -> clamped to 0.0
        src = _make_source(length_km=0.001, width_km=0.001, rigidity_pa=1.0)
        # area_m2 = 0.001 * 0.001 * 1e6 = 1.0, M0 = 1e-10 * 1.0 * 1.0 = 1e-10
        weights = np.array([1e-10], dtype=np.float64)
        mw = compute_mw_from_weights(weights, [src])
        assert mw == 0.0

    def test_nan_rigidity_returns_zero(self):
        """H6: NaN in source rigidity should return 0.0, not propagate NaN."""
        src = _make_source(rigidity_pa=float("nan"))
        weights = np.array([1.0], dtype=np.float64)
        assert compute_mw_from_weights(weights, [src]) == 0.0

    def test_inf_rigidity_returns_zero(self):
        """H6: inf in source rigidity should return 0.0, not propagate inf."""
        src = _make_source(rigidity_pa=float("inf"))
        weights = np.array([1.0], dtype=np.float64)
        assert compute_mw_from_weights(weights, [src]) == 0.0

    def test_nan_weight_returns_zero(self):
        """H6: NaN weight should return 0.0."""
        src = _make_source()
        weights = np.array([float("nan")], dtype=np.float64)
        assert compute_mw_from_weights(weights, [src]) == 0.0


# ---------------------------------------------------------------------------
# solve_nnls
# ---------------------------------------------------------------------------


class TestSolveNnls:
    def test_perfect_fit(self):
        rng = np.random.default_rng(42)
        n_obs = 120
        n_sources = 3
        H = rng.standard_normal((n_obs, n_sources))
        true_weights = np.array([1.0, 2.0, 0.5])
        d = H @ true_weights

        sources = [_make_source(f"s{i}", segment_index=i) for i in range(n_sources)]
        result = solve_nnls(
            H, d, [s.source_id for s in sources],
            ["dart_01", "dart_02"], sources, 3600,
        )

        np.testing.assert_allclose(result.weights, true_weights, atol=1e-8)
        assert result.waveform_rmse_cm < 0.01

    def test_noisy_fit(self):
        rng = np.random.default_rng(42)
        n_obs = 120
        n_sources = 3
        H = rng.standard_normal((n_obs, n_sources))
        true_weights = np.array([1.0, 2.0, 0.5])
        noise = rng.standard_normal(n_obs) * 0.1
        d = H @ true_weights + noise

        sources = [_make_source(f"s{i}", segment_index=i) for i in range(n_sources)]
        result = solve_nnls(
            H, d, [s.source_id for s in sources],
            ["dart_01", "dart_02"], sources, 3600,
        )

        assert result.waveform_rmse_cm > 0
        assert all(w >= 0 for w in result.weights)
        # Verify NNLS recovered weights close to truth despite noise
        np.testing.assert_allclose(result.weights, true_weights, atol=0.5)

    def test_non_negativity(self):
        rng = np.random.default_rng(42)
        H = rng.standard_normal((60, 5))
        d = rng.standard_normal(60)
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(5)]
        result = solve_nnls(
            H, d, [s.source_id for s in sources],
            ["dart_01"], sources, 3600,
        )
        assert all(w >= 0 for w in result.weights)

    def test_deterministic(self):
        rng = np.random.default_rng(42)
        H = rng.standard_normal((60, 3))
        d = rng.standard_normal(60)
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        sids = [s.source_id for s in sources]

        r1 = solve_nnls(H, d, sids, ["d1"], sources, 3600)
        r2 = solve_nnls(H, d, sids, ["d1"], sources, 3600)

        np.testing.assert_array_equal(r1.weights, r2.weights)
        assert r1.residual_norm == r2.residual_norm
        assert r1.mw_equivalent == r2.mw_equivalent


# ---------------------------------------------------------------------------
# rank_scenarios
# ---------------------------------------------------------------------------


class TestRankScenarios:
    def test_single_scenario(self):
        result = InversionResult(
            source_ids=["s1", "s2", "s3"],
            weights=np.array([1.0, 0.0, 0.5]),
            residual_norm=0.1,
            waveform_rmse_cm=0.5,
            mw_equivalent=7.5,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        ranked = rank_scenarios(result)
        assert len(ranked) == 1
        assert ranked[0].rank == 1
        assert ranked[0].posterior_weight == 1.0

    def test_filters_zero_weights(self):
        result = InversionResult(
            source_ids=["s1", "s2", "s3"],
            weights=np.array([1.0, 0.0, 0.5]),
            residual_norm=0.1,
            waveform_rmse_cm=0.5,
            mw_equivalent=7.5,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        ranked = rank_scenarios(result)
        assert "s2" not in ranked[0].unit_source_ids
        assert len(ranked[0].unit_source_ids) == 2

    def test_all_zero_weights(self):
        result = InversionResult(
            source_ids=["s1", "s2", "s3"],
            weights=np.array([0.0, 0.0, 0.0]),
            residual_norm=0.0,
            waveform_rmse_cm=0.0,
            mw_equivalent=0.0,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        ranked = rank_scenarios(result)
        assert len(ranked) == 1
        # All sources included to satisfy min_length=1
        assert len(ranked[0].unit_source_ids) == 3

    def test_all_zero_weights_logs_warning(self, caplog):
        """H1: rank_scenarios emits a warning when NNLS produces all-zero weights."""
        import logging

        result = InversionResult(
            source_ids=["s1", "s2", "s3"],
            weights=np.array([0.0, 0.0, 0.0]),
            residual_norm=0.0,
            waveform_rmse_cm=0.0,
            mw_equivalent=0.0,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        with caplog.at_level(logging.WARNING, logger="hazard_assessment.agents.scenario_inversion"):
            rank_scenarios(result)

        assert any(
            "all-zero weights" in rec.message for rec in caplog.records
        ), "Expected warning about all-zero weights"

    def test_nonzero_weights_no_warning(self, caplog):
        """Nonzero weights should NOT trigger the zero-weight warning."""
        import logging

        result = InversionResult(
            source_ids=["s1", "s2"],
            weights=np.array([1.0, 0.5]),
            residual_norm=0.1,
            waveform_rmse_cm=0.5,
            mw_equivalent=7.5,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        with caplog.at_level(logging.WARNING, logger="hazard_assessment.agents.scenario_inversion"):
            rank_scenarios(result)

        assert not any(
            "all-zero weights" in rec.message for rec in caplog.records
        ), "Should not warn when weights are nonzero"


# ---------------------------------------------------------------------------
# Seismic-only mode
# ---------------------------------------------------------------------------


class TestSeismicOnly:
    def test_weights_decay_with_distance(self):
        sources = [
            _make_source("near", lat=0.0, lon=0.0, segment_index=0),
            _make_source("far", lat=3.0, lon=3.0, segment_index=1),
        ]
        config = SeismicOnlyConfig(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        weights = compute_seismic_only_weights(sources, config)
        assert weights[0] > weights[1]

    def test_no_dart_stations(self):
        db = InMemoryUnitSourceDatabase()
        for i in range(5):
            db.add_source(
                _make_source(f"s{i}", lat=0.0, lon=0.01 * i, segment_index=i)
            )
        config = SeismicOnlyConfig(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        result = run_seismic_only_estimate(db, config)
        assert result.fit_stations == []

    def test_rmse_zero(self):
        db = InMemoryUnitSourceDatabase()
        for i in range(5):
            db.add_source(
                _make_source(f"s{i}", lat=0.0, lon=0.01 * i, segment_index=i)
            )
        config = SeismicOnlyConfig(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        result = run_seismic_only_estimate(db, config)
        assert result.waveform_rmse_cm == 0.0

    def test_mw_scaling(self):
        db = InMemoryUnitSourceDatabase()
        for i in range(5):
            db.add_source(
                _make_source(f"s{i}", lat=0.0, lon=0.01 * i, segment_index=i)
            )

        config_low = SeismicOnlyConfig(
            magnitude=7.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        config_high = SeismicOnlyConfig(
            magnitude=9.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        w_low = compute_seismic_only_weights(
            [_make_source("s0", lat=0.0, lon=0.0)], config_low
        )
        w_high = compute_seismic_only_weights(
            [_make_source("s0", lat=0.0, lon=0.0)], config_high
        )
        assert w_high[0] > w_low[0]

    def test_empty_sources_raises(self):
        db = InMemoryUnitSourceDatabase()
        config = SeismicOnlyConfig(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        with pytest.raises(ValueError, match="No unit sources found"):
            run_seismic_only_estimate(db, config)

    def test_empty_sources_returns_empty_weights(self):
        config = SeismicOnlyConfig(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
        )
        weights = compute_seismic_only_weights([], config)
        assert len(weights) == 0


# ---------------------------------------------------------------------------
# Bootstrap helpers
# ---------------------------------------------------------------------------


def _build_bootstrap_inputs(
    n_stations: int = 3,
    n_timepoints: int = 60,
    n_sources: int = 5,
    seed: int = 42,
):
    """Build consistent H, d, station_ids, and sources for bootstrap tests."""
    rng = np.random.default_rng(seed)
    sources = [_make_source(f"src_{i:02d}", segment_index=i) for i in range(n_sources)]
    station_ids = [f"dart_{i:02d}" for i in range(n_stations)]
    H = rng.standard_normal((n_stations * n_timepoints, n_sources))
    true_weights = rng.uniform(0, 2, size=n_sources)
    d = H @ true_weights + rng.standard_normal(n_stations * n_timepoints) * 0.01
    return H, d, station_ids, sources, n_timepoints


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


class TestBootstrapConfig:
    def test_defaults(self):
        config = BootstrapConfig()
        assert config.n_iterations == 500
        assert config.seed == 42
        assert config.confidence_levels == (0.10, 0.50, 0.90)

    def test_frozen(self):
        config = BootstrapConfig()
        with pytest.raises(AttributeError):
            config.n_iterations = 100  # type: ignore[misc]

    def test_n_iterations_zero_rejected(self):
        with pytest.raises(ValueError, match="n_iterations must be >= 1"):
            BootstrapConfig(n_iterations=0)

    def test_n_iterations_negative_rejected(self):
        with pytest.raises(ValueError, match="n_iterations must be >= 1"):
            BootstrapConfig(n_iterations=-1)


class TestRunBootstrap:
    def test_deterministic(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=20, seed=42)
        r1 = run_bootstrap(H, d, sids, sources, nt, config)
        r2 = run_bootstrap(H, d, sids, sources, nt, config)
        np.testing.assert_array_equal(r1.mw_samples, r2.mw_samples)
        np.testing.assert_array_equal(r1.weight_samples, r2.weight_samples)

    def test_output_shape(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=20, seed=42)
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert result.mw_samples.shape == (20,)
        assert result.weight_samples.shape == (20, len(sources))
        assert result.n_iterations_completed == 20

    def test_non_negative_weights(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=20, seed=42)
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert np.all(result.weight_samples >= 0)

    def test_mw_percentiles_ordered(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=50, seed=42)
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert result.mw_percentiles[0.10] <= result.mw_percentiles[0.50]
        assert result.mw_percentiles[0.50] <= result.mw_percentiles[0.90]

    def test_source_ids_captured(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=5, seed=42)
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert result.source_ids == [s.source_id for s in sources]

    def test_single_station_raises(self):
        H, d, _, sources, nt = _build_bootstrap_inputs(n_stations=1)
        config = BootstrapConfig(n_iterations=5, seed=42)
        with pytest.raises(ValueError, match=">= 2 stations"):
            run_bootstrap(H, d, ["dart_00"], sources, nt, config)

    def test_dimension_mismatch_raises(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(n_iterations=5, seed=42)
        # Pass wrong n_timepoints to trigger dimension check
        with pytest.raises(ValueError, match="row count"):
            run_bootstrap(H, d, sids, sources, nt + 1, config)

    def test_two_stations_minimum(self):
        """Bootstrap with exactly 2 stations (minimum) should succeed."""
        H, d, sids, sources, nt = _build_bootstrap_inputs(n_stations=2)
        config = BootstrapConfig(n_iterations=20, seed=42)
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert result.n_iterations_completed == 20
        assert result.weight_samples.shape == (20, len(sources))
        assert np.all(result.weight_samples >= 0)

    def test_custom_confidence_levels(self):
        H, d, sids, sources, nt = _build_bootstrap_inputs()
        config = BootstrapConfig(
            n_iterations=20, seed=42, confidence_levels=(0.05, 0.50, 0.95)
        )
        result = run_bootstrap(H, d, sids, sources, nt, config)
        assert 0.05 in result.mw_percentiles
        assert 0.95 in result.mw_percentiles
        assert 0.10 not in result.mw_percentiles


# ---------------------------------------------------------------------------
# Ensemble spread classification
# ---------------------------------------------------------------------------


class TestClassifyEnsembleSpread:
    def test_low(self):
        # ratio = 1.5 / 1.0 = 1.5 < 2.0 -> LOW
        assert classify_ensemble_spread(1.0, 1.5) == EnsembleSpread.LOW

    def test_moderate(self):
        # ratio = 3.0 / 1.0 = 3.0 -> MODERATE
        assert classify_ensemble_spread(1.0, 3.0) == EnsembleSpread.MODERATE

    def test_moderate_boundary_low(self):
        # ratio = 2.0 / 1.0 = 2.0 -> MODERATE (2.0 <= ratio <= 5.0)
        assert classify_ensemble_spread(1.0, 2.0) == EnsembleSpread.MODERATE

    def test_moderate_boundary_high(self):
        # ratio = 5.0 / 1.0 = 5.0 -> MODERATE (2.0 <= ratio <= 5.0)
        assert classify_ensemble_spread(1.0, 5.0) == EnsembleSpread.MODERATE

    def test_high_ratio(self):
        # ratio = 6.0 / 1.0 = 6.0 > 5.0 -> HIGH
        assert classify_ensemble_spread(1.0, 6.0) == EnsembleSpread.HIGH

    def test_p10_zero(self):
        assert classify_ensemble_spread(0.0, 1.0) == EnsembleSpread.HIGH

    def test_p10_negative(self):
        assert classify_ensemble_spread(-0.1, 1.0) == EnsembleSpread.HIGH

    def test_both_zero(self):
        assert classify_ensemble_spread(0.0, 0.0) == EnsembleSpread.HIGH


# ---------------------------------------------------------------------------
# Bootstrap-based scenario ranking
# ---------------------------------------------------------------------------


class TestRankScenariosFromBootstrap:
    def _make_bootstrap_result(
        self, weight_samples: NDArray, sources: list[UnitSource]
    ) -> BootstrapResult:
        n_iter = weight_samples.shape[0]
        mw_samples = np.array([
            compute_mw_from_weights(weight_samples[i], sources)
            for i in range(n_iter)
        ])
        return BootstrapResult(
            source_ids=[s.source_id for s in sources],
            mw_samples=mw_samples,
            weight_samples=weight_samples,
            mw_percentiles={0.50: float(np.median(mw_samples))},
            n_iterations_completed=n_iter,
        )

    def test_single_pattern(self):
        """All iterations produce the same activation -> 1 scenario."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # All iterations have same non-zero pattern
        weights = np.array([[1.0, 0.0, 0.5]] * 10)
        bootstrap = self._make_bootstrap_result(weights, sources)
        H = np.random.default_rng(42).standard_normal((60, 3))
        d = np.random.default_rng(99).standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(bootstrap, sources, H, d)
        assert len(ranked) == 1
        assert ranked[0].posterior_weight == 1.0
        assert ranked[0].rank == 1

    def test_multiple_patterns(self):
        """Distinct activation patterns -> multiple scenarios."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # Pattern A: sources 0 and 2 active (6 iterations)
        # Pattern B: source 0 only (4 iterations)
        pattern_a = np.array([[1.0, 0.0, 0.5]] * 6)
        pattern_b = np.array([[1.0, 0.0, 0.0]] * 4)
        weights = np.vstack([pattern_a, pattern_b])
        bootstrap = self._make_bootstrap_result(weights, sources)
        H = np.random.default_rng(42).standard_normal((60, 3))
        d = np.random.default_rng(99).standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(bootstrap, sources, H, d)
        assert len(ranked) == 2
        # Most frequent pattern first
        assert ranked[0].posterior_weight == 0.6
        assert ranked[1].posterior_weight == 0.4
        assert ranked[0].rank == 1
        assert ranked[1].rank == 2

    def test_respects_max_scenarios(self):
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # 3 distinct patterns
        weights = np.vstack([
            np.array([[1.0, 0.0, 0.5]] * 5),
            np.array([[1.0, 0.0, 0.0]] * 3),
            np.array([[0.0, 1.0, 0.5]] * 2),
        ])
        bootstrap = self._make_bootstrap_result(weights, sources)
        H = np.random.default_rng(42).standard_normal((60, 3))
        d = np.random.default_rng(99).standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(
            bootstrap, sources, H, d, max_scenarios=2
        )
        assert len(ranked) == 2

    def test_filters_noise_patterns(self):
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # Dominant pattern (98/100) and noise pattern (2/100)
        weights = np.vstack([
            np.array([[1.0, 0.0, 0.5]] * 98),
            np.array([[0.0, 1.0, 0.0]] * 2),
        ])
        bootstrap = self._make_bootstrap_result(weights, sources)
        H = np.random.default_rng(42).standard_normal((60, 3))
        d = np.random.default_rng(99).standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(
            bootstrap, sources, H, d, min_posterior_weight=0.05
        )
        # Noise pattern (0.02) filtered out
        assert len(ranked) == 1
        assert ranked[0].posterior_weight == 0.98

    def test_computes_per_cluster_rmse(self):
        """RMSE is from H @ mean_weights - d, not from base inversion."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        weights = np.array([[1.0, 0.0, 0.5]] * 10)
        bootstrap = self._make_bootstrap_result(weights, sources)

        rng = np.random.default_rng(42)
        H = rng.standard_normal((60, 3))
        d = rng.standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(bootstrap, sources, H, d)
        # Verify RMSE is computed from the mean weights
        mean_w = weights.mean(axis=0)
        expected_residual = H @ mean_w - d
        expected_rmse = (
            float(np.linalg.norm(expected_residual)) / math.sqrt(60)
        ) * 100.0
        assert abs(ranked[0].waveform_rmse_cm - expected_rmse) < 1e-8

    def test_weights_sum_le_one(self):
        """Posterior weights sum <= 1.0 (filtering can discard some)."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        weights = np.vstack([
            np.array([[1.0, 0.0, 0.5]] * 7),
            np.array([[1.0, 0.0, 0.0]] * 3),
        ])
        bootstrap = self._make_bootstrap_result(weights, sources)
        H = np.random.default_rng(42).standard_normal((60, 3))
        d = np.random.default_rng(99).standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(bootstrap, sources, H, d)
        total = sum(s.posterior_weight for s in ranked)
        assert total <= 1.0 + 1e-10

    def test_fallback_when_all_filtered(self):
        """When all patterns are below min_posterior_weight, fallback to overall mean."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # 5 distinct patterns, each appearing only twice out of 10
        weights = np.vstack([
            np.array([[1.0, 0.0, 0.0]] * 2),
            np.array([[0.0, 1.0, 0.0]] * 2),
            np.array([[0.0, 0.0, 1.0]] * 2),
            np.array([[1.0, 1.0, 0.0]] * 2),
            np.array([[1.0, 0.0, 1.0]] * 2),
        ])
        bootstrap = self._make_bootstrap_result(weights, sources)
        rng = np.random.default_rng(42)
        H = rng.standard_normal((60, 3))
        d = rng.standard_normal(60)

        # min_posterior_weight=0.25 means each pattern (0.2) is below threshold
        ranked = rank_scenarios_from_bootstrap(
            bootstrap, sources, H, d, min_posterior_weight=0.25
        )
        # Should produce exactly 1 fallback scenario from overall mean
        assert len(ranked) == 1
        assert ranked[0].rank == 1
        assert ranked[0].posterior_weight == 1.0

        # Verify fallback used overall-mean weights (not any single pattern)
        mean_all = weights.mean(axis=0)
        expected_mw = compute_mw_from_weights(mean_all, sources)
        assert abs(ranked[0].mw_equivalent - expected_mw) < 1e-10

        # Verify RMSE matches overall-mean weights against full data
        residual = H @ mean_all - d
        expected_rmse = (
            float(np.linalg.norm(residual)) / math.sqrt(len(d))
        ) * 100.0
        assert abs(ranked[0].waveform_rmse_cm - expected_rmse) < 1e-8

    def test_empty_activation_pattern(self):
        """All-zero weights produce an empty-frozenset activation pattern."""
        sources = [_make_source(f"s{i}", segment_index=i) for i in range(3)]
        # All weights zero for every iteration
        weights = np.zeros((10, 3))
        bootstrap = self._make_bootstrap_result(weights, sources)
        rng = np.random.default_rng(42)
        H = rng.standard_normal((60, 3))
        d = rng.standard_normal(60)

        ranked = rank_scenarios_from_bootstrap(bootstrap, sources, H, d)
        assert len(ranked) == 1
        # All-zero pattern -> includes all sources as fallback
        assert len(ranked[0].unit_source_ids) == 3
        assert ranked[0].mw_equivalent == 0.0
        assert ranked[0].posterior_weight == 1.0


# ---------------------------------------------------------------------------
# NNLS zero threshold used in rank_scenarios
# ---------------------------------------------------------------------------


class TestNnlsZeroThreshold:
    def test_rank_scenarios_uses_threshold(self):
        """Values below NNLS_ZERO_THRESHOLD are treated as zero."""
        result = InversionResult(
            source_ids=["s1", "s2", "s3"],
            weights=np.array([1.0, 1e-15, 0.5]),  # s2 has numerical noise
            residual_norm=0.1,
            waveform_rmse_cm=0.5,
            mw_equivalent=7.5,
            fit_stations=["d1"],
            inversion_window_sec=3600,
            elapsed_sec=0.01,
        )
        ranked = rank_scenarios(result)
        # s2 (1e-15) should be treated as zero and filtered out
        assert "s2" not in ranked[0].unit_source_ids
        assert len(ranked[0].unit_source_ids) == 2


# ---------------------------------------------------------------------------
# Coastal amplitude proxies
# ---------------------------------------------------------------------------


FIXED_ORIGIN = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_coastal_factors(
    site_id: str,
    source_ids: list[str],
    peaks: list[float],
    travel_times: list[float],
) -> CoastalForecastFactors:
    return CoastalForecastFactors(
        site_id=site_id,
        unit_source_peak_m=dict(zip(source_ids, peaks)),
        travel_time_sec=dict(zip(source_ids, travel_times)),
    )


class TestComputeCoastalProxies:
    def test_bootstrap_percentiles(self):
        source_ids = ["s0", "s1"]
        # 100 iterations with varying weights
        rng = np.random.default_rng(42)
        weight_samples = rng.uniform(0.5, 2.0, size=(100, 2))
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.1, 0.2], [3600.0, 4000.0]
            ),
        }
        proxies, max_amp = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN
        )
        assert len(proxies) == 1
        p = proxies[0]
        assert p.site_id == "site_A"
        assert p.amplitude_proxy_p10_m <= p.amplitude_proxy_p50_m
        assert p.amplitude_proxy_p50_m <= p.amplitude_proxy_p90_m
        assert p.amplitude_proxy_p10_m >= 0.0

    def test_single_inversion_equal_percentiles(self):
        source_ids = ["s0", "s1"]
        weight_samples = np.array([[1.0, 2.0]])  # single row
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.1, 0.2], [3600.0, 4000.0]
            ),
        }
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN
        )
        p = proxies[0]
        assert p.amplitude_proxy_p10_m == p.amplitude_proxy_p50_m
        assert p.amplitude_proxy_p50_m == p.amplitude_proxy_p90_m
        # Expected: 1.0*0.1 + 2.0*0.2 = 0.5
        assert abs(p.amplitude_proxy_p50_m - 0.5) < 1e-10

    def test_tidal_correction_applied(self):
        source_ids = ["s0"]
        weight_samples = np.array([[1.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [1.0], [3600.0]
            ),
        }
        # +20% tidal correction
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
            tidal_corrections={"site_A": 0.2},
        )
        assert proxies[0].tidal_correction_applied is True
        assert abs(proxies[0].amplitude_proxy_p50_m - 1.2) < 1e-10

    def test_no_tidal_correction(self):
        source_ids = ["s0"]
        weight_samples = np.array([[1.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [1.0], [3600.0]
            ),
        }
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        assert proxies[0].tidal_correction_applied is False
        assert abs(proxies[0].amplitude_proxy_p50_m - 1.0) < 1e-10

    def test_missing_source_contributes_zero(self):
        source_ids = ["s0", "s1"]
        weight_samples = np.array([[1.0, 1.0]])
        # site_A only knows about s0, not s1
        factors = {
            "site_A": CoastalForecastFactors(
                site_id="site_A",
                unit_source_peak_m={"s0": 0.5},  # s1 missing
                travel_time_sec={"s0": 3600.0},
            ),
        }
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        # Only s0 contributes: 1.0 * 0.5 = 0.5
        assert abs(proxies[0].amplitude_proxy_p50_m - 0.5) < 1e-10

    def test_arrival_time_single_inversion(self):
        source_ids = ["s0", "s1"]
        weight_samples = np.array([[1.0, 1.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.1, 0.2], [3600.0, 7200.0]
            ),
        }
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        # Earliest: min(3600, 7200) = 3600 sec
        expected_arrival = FIXED_ORIGIN + timedelta(seconds=3600.0)
        assert proxies[0].arrival_utc == expected_arrival
        assert proxies[0].arrival_uncertainty_min == 0.0

    def test_arrival_time_bootstrap_median(self):
        source_ids = ["s0", "s1"]
        # 3 iterations: different active sources produce different min travel times
        weight_samples = np.array([
            [1.0, 0.0],  # only s0 active -> min_tt = 3600
            [0.0, 1.0],  # only s1 active -> min_tt = 7200
            [1.0, 1.0],  # both active -> min_tt = 3600
        ])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.1, 0.2], [3600.0, 7200.0]
            ),
        }
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        # min_tts = [3600, 7200, 3600], median = 3600
        expected_arrival = FIXED_ORIGIN + timedelta(seconds=3600.0)
        assert proxies[0].arrival_utc == expected_arrival
        # uncertainty = (7200 - 3600) / 60 = 60 minutes
        assert abs(proxies[0].arrival_uncertainty_min - 60.0) < 1e-10

    def test_returns_max_amplitude_array(self):
        source_ids = ["s0"]
        weight_samples = np.array([[1.0], [2.0], [3.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.5], [3600.0]
            ),
            "site_B": _make_coastal_factors(
                "site_B", source_ids, [1.0], [4000.0]
            ),
        }
        _, max_amp = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        assert max_amp.shape == (3,)
        # site_B has higher peak (1.0 vs 0.5), so max_amp = weight * 1.0
        np.testing.assert_allclose(max_amp, [1.0, 2.0, 3.0])

    def test_all_zero_weights(self):
        source_ids = ["s0"]
        weight_samples = np.array([[0.0], [0.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [0.5], [3600.0]
            ),
        }
        proxies, max_amp = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
        )
        assert proxies[0].amplitude_proxy_p50_m == 0.0
        # All zero weights -> arrival = event_origin
        assert proxies[0].arrival_utc == FIXED_ORIGIN
        assert proxies[0].arrival_uncertainty_min == 0.0

    def test_tidal_correction_clamps_negative(self):
        source_ids = ["s0"]
        weight_samples = np.array([[1.0]])
        factors = {
            "site_A": _make_coastal_factors(
                "site_A", source_ids, [1.0], [3600.0]
            ),
        }
        # Extreme negative tidal correction
        proxies, _ = compute_coastal_proxies(
            weight_samples, source_ids, factors, FIXED_ORIGIN,
            tidal_corrections={"site_A": -2.0},
        )
        # 1.0 * (1 + -2.0) = -1.0, clamped to 0.0
        assert proxies[0].amplitude_proxy_p50_m == 0.0

    def test_empty_factors(self):
        proxies, max_amp = compute_coastal_proxies(
            np.array([[1.0, 2.0]]),
            ["s0", "s1"],
            {},
            FIXED_ORIGIN,
        )
        assert proxies == []
        assert max_amp.shape == (1,)


# ---------------------------------------------------------------------------
# CoastalProxy schema validation (negative test)
# ---------------------------------------------------------------------------


class TestCoastalProxyValidation:
    def test_rejects_inverted_percentiles(self):
        """CoastalProxy validator rejects p10 > p50 > p90."""
        from pydantic import ValidationError

        from hazard_assessment.schemas.scenario import CoastalProxy as CoastalProxySchema

        with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
            CoastalProxySchema(
                site_id="test",
                arrival_utc=FIXED_ORIGIN,
                arrival_uncertainty_min=0.0,
                amplitude_proxy_p10_m=1.0,
                amplitude_proxy_p50_m=0.5,  # p50 < p10 -> invalid
                amplitude_proxy_p90_m=0.2,
                tidal_correction_applied=False,
            )
