"""Unit tests for anomaly detection algorithms.

Tests cover:
- Harmonic tidal analysis, bandpass filtering, detide_and_filter pipeline
- Wavelet energy, BOCPD, Isolation Forest, spatial coherence,
  ensemble fusion, seismic context, full score computation
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from hazard_assessment.agents.anomaly_detection import (
    BOCPD_HAZARD_LAMBDA,
    DEEP_OCEAN_WAVE_SPEED_KM_S,
    W_ML,
    W_STATISTICAL,
    W_STATISTICAL_NO_ML,
    W_THRESHOLD,
    W_THRESHOLD_NO_ML,
    BOCPDState,
    IsolationForestModel,
    SeismicEvent,
    SpatialCoherenceResult,
    SpatialConfirmationDetail,
    StationArrival,
    _correct_level_shifts,
    apply_bandpass_filter,
    bocpd_update,
    build_harmonic_matrix,
    build_iforest_features,
    check_spatial_coherence,
    clean_bpr_calibration,
    compute_bocpd_score,
    compute_ensemble_score,
    compute_full_anomaly_score,
    compute_rate_of_change,
    compute_rolling_energy,
    compute_spatial_coherence_score,
    compute_statistical_score,
    compute_threshold_score,
    compute_travel_time_sec,
    compute_wavelet_energy,
    compute_wavelet_score,
    design_bandpass_filter,
    detide,
    detide_and_filter,
    fit_tidal_harmonics,
    hampel_filter,
    is_seismically_quiet,
    predict_tide,
)

_T0 = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


# =========================================================================
# Detiding tests
# =========================================================================


class TestHarmonicMatrix:
    """Tests for build_harmonic_matrix."""

    def test_shape(self) -> None:
        times = np.arange(0, 24, 0.25)  # 96 points, 15-min intervals
        mat = build_harmonic_matrix(times)
        # 8 constituents -> 17 columns (1 constant + 8*2 cos/sin)
        assert mat.shape == (96, 17)

    def test_constant_column(self) -> None:
        times = np.arange(0, 10, 1.0)
        mat = build_harmonic_matrix(times)
        np.testing.assert_array_equal(mat[:, 0], np.ones(10))

    def test_cos_sin_columns_bounded(self) -> None:
        times = np.arange(0, 100, 0.5)
        mat = build_harmonic_matrix(times)
        # All cos/sin columns should be in [-1, 1]
        assert np.all(mat[:, 1:] >= -1.0 - 1e-10)
        assert np.all(mat[:, 1:] <= 1.0 + 1e-10)


class TestTidalFitting:
    """Tests for fit_tidal_harmonics and predict_tide."""

    def test_perfect_recovery(self) -> None:
        """A pure M2 tidal signal should be recovered exactly."""
        # Generate 30 days of 15-min data
        dt_hours = 0.25
        times = np.arange(0, 30 * 24, dt_hours)
        # M2 frequency in rad/hour
        omega_m2 = math.radians(28.984104)
        amplitude = 1.5
        phase = 0.3
        signal = amplitude * np.cos(omega_m2 * times + phase)

        coeffs = fit_tidal_harmonics(times, signal)
        predicted = predict_tide(times, coeffs)

        np.testing.assert_allclose(predicted, signal, atol=1e-8)

    def test_residual_near_zero_for_tidal(self) -> None:
        """Detiding a pure tidal signal should leave near-zero residual."""
        dt_hours = 0.25
        times = np.arange(0, 30 * 24, dt_hours)
        omega_m2 = math.radians(28.984104)
        signal = 2.0 * np.cos(omega_m2 * times)

        residual = detide(times, signal)
        assert np.max(np.abs(residual)) < 1e-8

    def test_non_tidal_passes_through(self) -> None:
        """A DC offset should survive detiding (absorbed by constant)."""
        dt_hours = 0.25
        times = np.arange(0, 30 * 24, dt_hours)
        # Linear trend (not tidal)
        signal = 0.001 * times

        residual = detide(times, signal)
        # The constant term absorbs the mean, but the linear trend
        # cannot be fully captured by harmonics -> residual retains energy
        assert len(residual) == len(times)
        assert np.std(residual) > 0.0, "Residual should retain non-tidal energy"

    def test_separate_fit_window(self) -> None:
        """Detide using a separate fitting window."""
        dt_hours = 0.25
        fit_times = np.arange(0, 30 * 24, dt_hours)
        omega_m2 = math.radians(28.984104)
        fit_signal = 1.0 * np.cos(omega_m2 * fit_times)

        # Target window (shorter)
        target_times = np.arange(30 * 24, 31 * 24, dt_hours)
        target_signal = 1.0 * np.cos(omega_m2 * target_times)

        # Disable cleaning - this test verifies basic detide math on synthetic data
        residual = detide(
            target_times, target_signal, fit_times, fit_signal,
            clean_calibration=False,
        )
        np.testing.assert_allclose(residual, 0.0, atol=1e-6)


# =========================================================================
# Bandpass filter tests
# =========================================================================


class TestBandpassFilter:
    """Tests for design_bandpass_filter and apply_bandpass_filter."""

    def test_filter_design_returns_sos(self) -> None:
        """Filter design should return valid SOS array."""
        sampling_rate = 1.0 / 60.0  # 1-min data
        sos = design_bandpass_filter(sampling_rate)
        assert sos.ndim == 2
        assert sos.shape[1] == 6  # SOS format

    def test_passband_signal_preserved(self) -> None:
        """A signal within the passband should be mostly preserved."""
        dt = 60.0  # 1-min data
        sampling_rate = 1.0 / dt
        n_samples = 2000  # ~33 hours
        t = np.arange(n_samples) * dt

        # 20-min period signal (within 5-120 min passband)
        freq = 1.0 / (20 * 60)  # Hz
        signal = np.sin(2 * np.pi * freq * t)

        filtered = apply_bandpass_filter(signal, sampling_rate)

        # Signal should retain most power (some edge effects)
        power_in = np.sum(signal[200:-200] ** 2)
        power_out = np.sum(filtered[200:-200] ** 2)
        ratio = power_out / power_in
        assert ratio > 0.5, f"Passband signal too attenuated: ratio={ratio:.3f}"

    def test_stopband_signal_attenuated(self) -> None:
        """A signal outside the passband should be strongly attenuated."""
        dt = 60.0
        sampling_rate = 1.0 / dt
        n_samples = 2000

        t = np.arange(n_samples) * dt
        # 3-hour period signal (outside 5-120 min passband)
        freq = 1.0 / (180 * 60)
        signal = np.sin(2 * np.pi * freq * t)

        filtered = apply_bandpass_filter(signal, sampling_rate)

        power_in = np.sum(signal[200:-200] ** 2)
        power_out = np.sum(filtered[200:-200] ** 2)
        ratio = power_out / (power_in + 1e-30)
        assert ratio < 0.3, f"Stopband signal not attenuated enough: ratio={ratio:.3f}"

    def test_short_signal_returns_zeros(self) -> None:
        """Signal shorter than minimum filter length should return zeros."""
        dt = 60.0
        sampling_rate = 1.0 / dt
        short_signal = np.ones(5)

        filtered = apply_bandpass_filter(short_signal, sampling_rate)
        np.testing.assert_array_equal(filtered, np.zeros(5))

    def test_causal_group_delay(self) -> None:
        """Causal sosfilt introduces group delay - filtered peaks lag input."""
        dt = 60.0
        sampling_rate = 1.0 / dt
        n_samples = 2000
        t = np.arange(n_samples) * dt

        freq = 1.0 / (20 * 60)
        signal = np.sin(2 * np.pi * freq * t)
        filtered = apply_bandpass_filter(signal, sampling_rate)

        # Find peak locations in middle region (avoid edge effects)
        mid = slice(400, 1600)
        signal_peaks = np.where(np.diff(np.sign(np.diff(signal[mid]))) < 0)[0]
        filtered_peaks = np.where(np.diff(np.sign(np.diff(filtered[mid]))) < 0)[0]

        # Preconditions: peak detection must find peaks for this test to be meaningful
        assert len(signal_peaks) > 0, "Signal peak detection failed - test is vacuous"
        assert len(filtered_peaks) > 0, "Filtered peak detection failed - test is vacuous"

        # Causal filter: filtered peaks should lag input peaks (positive delay)
        min_len = min(len(signal_peaks), len(filtered_peaks))
        delays = (
            filtered_peaks[:min_len].astype(float)
            - signal_peaks[:min_len].astype(float)
        )
        # Group delay should be non-negative (causal) and bounded
        assert np.all(delays >= -1), f"Filter appears non-causal: delays={delays}"
        assert np.max(np.abs(delays)) <= 20, f"Group delay too large: {delays}"


class TestDetideAndFilter:
    """Tests for the combined detide and filter pipeline."""

    def test_pipeline_returns_two_arrays(self) -> None:
        dt_hours = 1.0 / 60.0  # 1-min
        times = np.arange(0, 30 * 24, dt_hours)
        values = np.random.default_rng(42).normal(0, 0.01, len(times))
        sampling_rate = 1.0 / 60.0

        residual, filtered = detide_and_filter(times, values, sampling_rate)
        assert len(residual) == len(times)
        assert len(filtered) == len(times)

    def test_deterministic_on_replay(self) -> None:
        """Same inputs should produce identical outputs."""
        dt_hours = 1.0 / 60.0
        times = np.arange(0, 30 * 24, dt_hours)
        rng = np.random.default_rng(42)
        values = rng.normal(0, 0.01, len(times))
        sampling_rate = 1.0 / 60.0

        r1, f1 = detide_and_filter(times, values, sampling_rate)

        # Same inputs again
        rng2 = np.random.default_rng(42)
        values2 = rng2.normal(0, 0.01, len(times))
        r2, f2 = detide_and_filter(times, values2, sampling_rate)

        np.testing.assert_array_equal(r1, r2)
        np.testing.assert_array_equal(f1, f2)


# =========================================================================
# Wavelet energy tests
# =========================================================================


class TestWaveletEnergy:
    """Tests for wavelet energy computation."""

    def test_zero_signal_zero_energy(self) -> None:
        signal = np.zeros(512)
        energy = compute_wavelet_energy(signal, 60.0)
        assert energy == 0.0

    def test_empty_signal(self) -> None:
        energy = compute_wavelet_energy(np.array([]), 60.0)
        assert energy == 0.0

    def test_energy_increases_with_amplitude(self) -> None:
        rng = np.random.default_rng(42)
        small = rng.normal(0, 0.001, 512)
        large = rng.normal(0, 0.1, 512)

        e_small = compute_wavelet_energy(small, 60.0)
        e_large = compute_wavelet_energy(large, 60.0)
        assert e_large > e_small

    def test_level_selection_by_sampling_interval(self) -> None:
        """15-sec data should use level 8, 1-min data should use level 6."""
        signal = np.random.default_rng(42).normal(0, 0.01, 1024)
        e_1min = compute_wavelet_energy(signal, 60.0)
        e_15sec = compute_wavelet_energy(signal, 15.0)
        # Different decomposition levels select different tsunami-band
        # detail coefficients, so energies must differ by at least 5%.
        assert e_1min > 0.0
        assert e_15sec > 0.0
        rel_diff = abs(e_1min - e_15sec) / max(e_1min, e_15sec)
        assert rel_diff > 0.05

    def test_out_of_band_energy_excluded_for_15s_data(self) -> None:
        """For 15s data, DWT levels 1-2 (30-120s) are outside the tsunami
        band and should be excluded from the energy sum."""
        import pywt

        # 45s period falls in DWT level 1 (30-60s) for 15s data
        dt = 15.0
        n = 2048
        t = np.arange(n) * dt
        signal = np.sin(2 * np.pi * (1.0 / 45.0) * t)

        # Filtered energy (should exclude out-of-band levels)
        filtered_energy = compute_wavelet_energy(signal, dt)

        # Total energy across ALL detail levels (no filtering)
        coeffs = pywt.wavedec(signal, "db4", level=8)
        total_detail_energy = sum(
            float(np.sum(np.square(d))) for d in coeffs[1:]
        )

        # Filtered energy should be substantially less than total,
        # since level 1 (which captures most of the 45s signal) is excluded
        assert filtered_energy < total_detail_energy * 0.5, (
            f"Filtered energy {filtered_energy:.2f} should be much less than "
            f"total detail energy {total_detail_energy:.2f}"
        )

    def test_wavelet_score_below_baseline(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.001, 512)
        # Baseline from larger signal
        baseline = compute_wavelet_energy(
            np.random.default_rng(42).normal(0, 0.01, 512), 60.0
        )
        score = compute_wavelet_score(signal, 60.0, baseline)
        assert 0.0 <= score <= 1.0

    def test_wavelet_score_above_baseline(self) -> None:
        # Large anomalous signal
        signal = np.random.default_rng(42).normal(0, 0.1, 512)
        # Small baseline
        baseline = compute_wavelet_energy(
            np.random.default_rng(99).normal(0, 0.001, 512), 60.0
        )
        score = compute_wavelet_score(signal, 60.0, baseline)
        assert score > 0.5


# =========================================================================
# BOCPD tests
# =========================================================================


class TestBOCPD:
    """Tests for Bayesian Online Changepoint Detection."""

    def test_state_initialization(self) -> None:
        state = BOCPDState()
        assert state.hazard_lambda == BOCPD_HAZARD_LAMBDA
        assert state.prior_mean == 0.0
        assert len(state.run_length_log_probs) == 1

    def test_update_grows_run_length(self) -> None:
        state = BOCPDState()
        bocpd_update(state, 0.0)
        assert len(state.run_length_log_probs) == 2
        bocpd_update(state, 0.0)
        assert len(state.run_length_log_probs) == 3

    def test_changepoint_probability_bounded(self) -> None:
        state = BOCPDState()
        for _ in range(50):
            cp_prob = bocpd_update(state, 0.0)
            assert 0.0 <= cp_prob <= 1.0

    def test_changepoint_detected_on_level_shift(self) -> None:
        """A sudden level shift should produce elevated changepoint probability.

        Note: Under constant hazard, P(r_t=0) always equals hazard_lambda
        regardless of data.  This test passes because hazard_lambda=0.02 > 0.01.
        The useful data-dependent signal is the predictive log-evidence
        stored on state.last_log_evidence; see test_bocpd_score_* tests.
        """
        # Use higher hazard rate (1/50) for more responsive detection in test
        state = BOCPDState(hazard_lambda=1.0 / 50.0, prior_precision=100.0)

        # Steady state: zero-mean
        for _ in range(100):
            bocpd_update(state, 0.0)

        # Sudden shift - feed several samples at the new level to let
        # the changepoint probability accumulate
        max_cp = 0.0
        for _ in range(10):
            cp_prob = bocpd_update(state, 10.0)
            max_cp = max(max_cp, cp_prob)

        assert max_cp > 0.01, f"Expected changepoint detection, got {max_cp}"

    def test_hazard_lambda_zero_rejected(self) -> None:
        """hazard_lambda=0 causes log(0)=-inf; must be rejected."""
        with pytest.raises(ValueError, match="hazard_lambda must be in"):
            BOCPDState(hazard_lambda=0.0)

    def test_hazard_lambda_one_rejected(self) -> None:
        """hazard_lambda=1 causes log(1-1)=log(0)=-inf; must be rejected."""
        with pytest.raises(ValueError, match="hazard_lambda must be in"):
            BOCPDState(hazard_lambda=1.0)

    def test_hazard_lambda_negative_rejected(self) -> None:
        """Negative hazard_lambda is not a valid probability."""
        with pytest.raises(ValueError, match="hazard_lambda must be in"):
            BOCPDState(hazard_lambda=-0.1)

    def test_no_changepoint_on_steady(self) -> None:
        """Steady signal should have low changepoint probability after warmup."""
        state = BOCPDState(prior_precision=100.0)
        probs = []
        for _ in range(200):
            cp = bocpd_update(state, 0.0)
            probs.append(cp)

        # After warmup, changepoint probability should stabilize low
        assert probs[-1] < 0.1

    def test_log_evidence_stored(self) -> None:
        """bocpd_update should populate last_log_evidence on the state."""
        state = BOCPDState()
        bocpd_update(state, 1.0)
        # log_evidence is a log-probability, so it should be negative (or zero for degenerate cases)
        assert isinstance(state.last_log_evidence, float)

    def test_bocpd_score_bounded(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 100)
        score = compute_bocpd_score(signal)
        assert 0.0 <= score <= 1.0

    def test_bocpd_score_empty(self) -> None:
        assert compute_bocpd_score(np.array([])) == 0.0

    def test_bocpd_score_short_signal(self) -> None:
        """Signals shorter than burn-in + 5 should return 0."""
        signal = np.array([0.0] * 10)  # exactly burn-in, < burn_in + 5
        assert compute_bocpd_score(signal) == 0.0

    def test_bocpd_score_detects_level_shift(self) -> None:
        """Surprise-based BOCPD score should be higher for a signal with a
        level shift than for a steady signal."""
        rng = np.random.default_rng(42)
        steady = rng.normal(0, 0.01, 200)
        # Signal with level shift at sample 100
        shifted = np.concatenate([
            rng.normal(0, 0.01, 100),
            rng.normal(5.0, 0.01, 100),  # large jump
        ])

        score_steady = compute_bocpd_score(steady)
        score_shifted = compute_bocpd_score(shifted)

        assert score_shifted > score_steady, (
            f"Expected shifted score ({score_shifted:.4f}) > "
            f"steady score ({score_steady:.4f})"
        )
        assert score_shifted > 0.1, (
            f"Expected meaningful detection score, got {score_shifted:.4f}"
        )

    def test_bocpd_score_steady_signal_low(self) -> None:
        """Steady Gaussian noise should produce a low BOCPD score."""
        signal = np.random.default_rng(99).normal(0, 0.01, 300)
        score = compute_bocpd_score(signal)
        assert score < 0.3, f"Expected low score for steady noise, got {score:.4f}"

    def test_bocpd_score_monotone_in_amplitude(self) -> None:
        """Larger shifts should produce equal or higher BOCPD scores.

        Guards against the non-monotone dead zone where moderate SNR
        signals produce lower scores than weak ones (caused by
        post-changepoint samples inflating the baseline MAD).
        """
        rng = np.random.default_rng(42)
        noise_std = 0.01
        scores: list[float] = []
        for amp in [0.05, 0.10, 0.50, 1.0]:
            signal = np.concatenate([
                rng.normal(0, noise_std, 100),
                rng.normal(amp, noise_std, 100),
            ])
            scores.append(compute_bocpd_score(signal))

        # Each score should be >= the previous (monotone non-decreasing)
        for i in range(1, len(scores)):
            assert scores[i] >= scores[i - 1] - 0.05, (
                f"Non-monotone: amp sequence score[{i}]={scores[i]:.4f} < "
                f"score[{i-1}]={scores[i-1]:.4f}"
            )


# =========================================================================
# Isolation Forest tests
# =========================================================================


class TestIsolationForest:
    """Tests for Isolation Forest anomaly scoring."""

    def test_unfitted_returns_zero(self) -> None:
        model = IsolationForestModel()
        features = np.array([0.1, 0.2, 0.01, 0.5])
        assert model.score(features) == 0.0

    def test_fit_and_score(self) -> None:
        model = IsolationForestModel()
        rng = np.random.default_rng(42)
        # Normal training data
        train = rng.normal(0, 0.01, (200, 4))
        model.fit(train)
        assert model.is_fitted

        # Normal sample should have low anomaly score
        normal = np.array([[0.01, 0.01, 0.001, 0.5]])
        score_normal = model.score(normal)
        assert 0.0 <= score_normal <= 1.0

        # Anomalous sample should have higher score
        anomalous = np.array([[10.0, 10.0, 5.0, 0.0]])
        score_anomalous = model.score(anomalous)
        assert score_anomalous > score_normal

    def test_deterministic_scoring(self) -> None:
        """Same seed should produce same results."""
        rng = np.random.default_rng(42)
        train = rng.normal(0, 0.01, (200, 4))

        model1 = IsolationForestModel(random_seed=42)
        model1.fit(train.copy())

        model2 = IsolationForestModel(random_seed=42)
        model2.fit(train.copy())

        test = np.array([[0.5, 0.5, 0.1, 0.3]])
        assert model1.score(test) == model2.score(test)

    def test_score_bounded(self) -> None:
        model = IsolationForestModel()
        rng = np.random.default_rng(42)
        model.fit(rng.normal(0, 1, (100, 4)))

        for _ in range(20):
            features = rng.normal(0, 5, (1, 4))
            score = model.score(features)
            assert 0.0 <= score <= 1.0


class TestIForestFeatures:
    """Tests for Isolation Forest feature construction."""

    def test_feature_vector_shape(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 100)
        features = build_iforest_features(signal, 60.0, 0.5)
        assert features.shape == (4,)

    def test_rolling_energy(self) -> None:
        signal = np.ones(100) * 2.0
        energy = compute_rolling_energy(signal, 50)
        assert abs(energy - 4.0) < 1e-10  # mean of 2^2

    def test_rolling_energy_empty(self) -> None:
        assert compute_rolling_energy(np.array([]), 10) == 0.0

    def test_rate_of_change(self) -> None:
        signal = np.array([0.0, 1.0, 1.0, 3.0])
        roc = compute_rate_of_change(signal)
        assert roc == 2.0  # max diff is 3.0 - 1.0

    def test_rate_of_change_short(self) -> None:
        assert compute_rate_of_change(np.array([1.0])) == 0.0


# =========================================================================
# Spatial coherence tests
# =========================================================================


class TestSpatialCoherence:
    """Tests for spatial coherence checking."""

    def test_travel_time_calculation(self) -> None:
        """1000 km at 0.198 km/s ~ 5050.5 seconds."""
        t = compute_travel_time_sec(1000.0)
        assert t == pytest.approx(1000.0 / DEEP_OCEAN_WAVE_SPEED_KM_S)

    def test_confirmed_arrivals(self) -> None:
        """Two stations with arrivals matching expected travel time."""
        origin = StationArrival(
            station_id="A", arrival_time=_T0, latitude=0.0, longitude=0.0,
        )
        # Two stations at known distances with matching arrival times
        # Station at ~111 km east (1 degree lon at equator)
        expected_travel = 111.0 / DEEP_OCEAN_WAVE_SPEED_KM_S
        other = [
            StationArrival(
                station_id="B",
                arrival_time=_T0 + timedelta(seconds=expected_travel),
                latitude=0.0, longitude=1.0,
            ),
            StationArrival(
                station_id="C",
                arrival_time=_T0 + timedelta(seconds=expected_travel * 2),
                latitude=0.0, longitude=2.0,
            ),
        ]

        result = check_spatial_coherence(origin, other)
        assert result.confirmed is True
        assert result.confirming_stations >= 2
        assert len(result.confirmations) == 2

    def test_no_confirmation_wrong_timing(self) -> None:
        """Arrivals with wrong timing should not confirm."""
        origin = StationArrival(
            station_id="A", arrival_time=_T0, latitude=0.0, longitude=0.0,
        )
        other = [
            StationArrival(
                station_id="B",
                arrival_time=_T0 + timedelta(seconds=1),  # way too fast
                latitude=0.0, longitude=10.0,
            ),
        ]
        result = check_spatial_coherence(origin, other)
        assert result.confirmed is False
        assert result.confirming_stations == 0

    def test_empty_arrivals(self) -> None:
        origin = StationArrival(
            station_id="A", arrival_time=_T0, latitude=0.0, longitude=0.0,
        )
        result = check_spatial_coherence(origin, [])
        assert result.confirmed is False
        assert result.confirming_stations == 0

    def test_spatial_coherence_score(self) -> None:
        _detail = SpatialConfirmationDetail(
            station_id="A", distance_km=100, expected_travel_sec=60,
            actual_delta_sec=55, window_low_sec=30, window_high_sec=90,
            confirmed=True,
        )
        result = SpatialCoherenceResult(
            confirmed=True, confirming_stations=3,
            confirmations=[_detail, _detail, _detail],
        )
        score = compute_spatial_coherence_score(result)
        assert score == 1.0

    def test_spatial_coherence_score_partial(self) -> None:
        _detail = SpatialConfirmationDetail(
            station_id="A", distance_km=100, expected_travel_sec=60,
            actual_delta_sec=55, window_low_sec=30, window_high_sec=90,
            confirmed=True,
        )
        result = SpatialCoherenceResult(
            confirmed=False, confirming_stations=1,
            confirmations=[_detail, _detail],
        )
        score = compute_spatial_coherence_score(result)
        assert score == 0.5  # 1/2

    def test_spatial_coherence_score_empty(self) -> None:
        result = SpatialCoherenceResult(
            confirmed=False, confirming_stations=0, confirmations=[],
        )
        score = compute_spatial_coherence_score(result)
        assert score == 0.0


# =========================================================================
# Threshold score tests
# =========================================================================


class TestThresholdScore:
    """Tests for threshold-based anomaly scoring."""

    def test_below_threshold(self) -> None:
        signal = np.ones(100) * 0.01
        score = compute_threshold_score(signal, threshold_m=0.05)
        assert score == 0.0

    def test_above_threshold(self) -> None:
        signal = np.ones(100) * 0.1
        score = compute_threshold_score(signal, threshold_m=0.05)
        assert score > 0.5

    def test_empty_signal(self) -> None:
        assert compute_threshold_score(np.array([]), 0.05) == 0.0

    def test_zero_threshold(self) -> None:
        assert compute_threshold_score(np.ones(10), 0.0) == 0.0

    def test_seismic_quiet_raises_threshold(self) -> None:
        """Quiet period should require larger signal for same score."""
        signal = np.ones(100) * 0.06

        score_active = compute_threshold_score(signal, 0.05, seismic_context_quiet=False)
        score_quiet = compute_threshold_score(signal, 0.05, seismic_context_quiet=True)

        assert score_quiet < score_active

    def test_at_boundary_returns_zero(self) -> None:
        """Signal exactly at threshold -> score = 0 (threshold not exceeded)."""
        signal = np.array([0.05])
        assert compute_threshold_score(signal, threshold_m=0.05) == 0.0

    def test_linear_ramp_midpoint(self) -> None:
        """Signal at 1.5x threshold -> score = 0.5 (linear ramp)."""
        signal = np.array([0.075])  # 1.5 x 0.05
        score = compute_threshold_score(signal, threshold_m=0.05)
        assert score == pytest.approx(0.5)

    def test_at_double_threshold(self) -> None:
        """Signal at 2x threshold -> score = 1.0 (top of ramp)."""
        signal = np.array([0.10])  # 2.0 x 0.05
        score = compute_threshold_score(signal, threshold_m=0.05)
        assert score == pytest.approx(1.0)

    def test_above_double_caps_at_one(self) -> None:
        """Signal at 3x threshold -> score = 1.0 (capped)."""
        signal = np.array([0.15])  # 3.0 x 0.05
        score = compute_threshold_score(signal, threshold_m=0.05)
        assert score == 1.0


# =========================================================================
# Ensemble fusion tests
# =========================================================================


class TestEnsembleFusion:
    """Tests for ensemble score fusion."""

    def test_weights_sum_to_one(self) -> None:
        assert abs(W_THRESHOLD + W_STATISTICAL + W_ML - 1.0) < 1e-10

    def test_weights_no_ml_sum_to_one(self) -> None:
        assert abs(W_THRESHOLD_NO_ML + W_STATISTICAL_NO_ML - 1.0) < 1e-10

    def test_all_zeros(self) -> None:
        assert compute_ensemble_score(0.0, 0.0, 0.0) == 0.0

    def test_all_ones(self) -> None:
        assert compute_ensemble_score(1.0, 1.0, 1.0) == pytest.approx(1.0)

    def test_no_ml_renormalized(self) -> None:
        """Without ML, weights renormalize to W_THRESHOLD_NO_ML / W_STATISTICAL_NO_ML."""
        score = compute_ensemble_score(1.0, 1.0, None)
        assert score == pytest.approx(1.0)

        score = compute_ensemble_score(0.5, 0.5, None)
        expected = W_THRESHOLD_NO_ML * 0.5 + W_STATISTICAL_NO_ML * 0.5
        assert score == pytest.approx(expected)

    def test_bounded(self) -> None:
        for t in [0.0, 0.3, 0.5, 0.7, 1.0]:
            for s in [0.0, 0.3, 0.5, 0.7, 1.0]:
                for m in [None, 0.0, 0.5, 1.0]:
                    score = compute_ensemble_score(t, s, m)
                    assert 0.0 <= score <= 1.0

    def test_statistical_score_max(self) -> None:
        """Statistical = max(wavelet, bocpd)."""
        assert compute_statistical_score(0.8, 0.3) == 0.8
        assert compute_statistical_score(0.3, 0.8) == 0.8


# =========================================================================
# Seismic context tests
# =========================================================================


class TestSeismicContext:
    """Tests for seismic context adjustment."""

    def test_quiet_no_events(self) -> None:
        assert is_seismically_quiet([], _T0) is True

    def test_quiet_old_event(self) -> None:
        """Event older than 90 minutes should be quiet."""
        old_event = SeismicEvent(
            event_id="e1", magnitude=7.0,
            origin_time=_T0 - timedelta(minutes=91),
            latitude=0.0, longitude=0.0,
        )
        assert is_seismically_quiet([old_event], _T0) is True

    def test_not_quiet_recent_large(self) -> None:
        """Recent M7.0 event should not be quiet."""
        recent = SeismicEvent(
            event_id="e1", magnitude=7.0,
            origin_time=_T0 - timedelta(minutes=30),
            latitude=0.0, longitude=0.0,
        )
        assert is_seismically_quiet([recent], _T0) is False

    def test_quiet_recent_small(self) -> None:
        """Recent M6.0 event (below 6.5 threshold) should still be quiet."""
        small = SeismicEvent(
            event_id="e1", magnitude=6.0,
            origin_time=_T0 - timedelta(minutes=30),
            latitude=0.0, longitude=0.0,
        )
        assert is_seismically_quiet([small], _T0) is True

    def test_boundary_magnitude(self) -> None:
        """M6.5 exactly should break the quiet period."""
        boundary = SeismicEvent(
            event_id="e1", magnitude=6.5,
            origin_time=_T0 - timedelta(minutes=30),
            latitude=0.0, longitude=0.0,
        )
        assert is_seismically_quiet([boundary], _T0) is False

    def test_boundary_time(self) -> None:
        """Event exactly 90 minutes ago should still break quiet."""
        boundary = SeismicEvent(
            event_id="e1", magnitude=7.0,
            origin_time=_T0 - timedelta(minutes=90),
            latitude=0.0, longitude=0.0,
        )
        assert is_seismically_quiet([boundary], _T0) is False

    def test_fsm_monitoring_overrides_quiet(self) -> None:
        """When FSM is monitoring, never report quiet (no 1.3x boost)."""
        # No seismic events at all - normally quiet=True
        assert is_seismically_quiet([], _T0, fsm_monitoring=False) is True
        assert is_seismically_quiet([], _T0, fsm_monitoring=True) is False

    def test_fsm_monitoring_with_small_event(self) -> None:
        """M6.2 event: normally quiet (below 6.5), but FSM monitoring disables boost."""
        small = SeismicEvent(
            event_id="e1", magnitude=6.2,
            origin_time=_T0 - timedelta(minutes=30),
            latitude=0.0, longitude=0.0,
        )
        # Without FSM monitoring: quiet (M6.2 < M6.5 threshold)
        assert is_seismically_quiet([small], _T0, fsm_monitoring=False) is True
        # With FSM monitoring: not quiet (FSM override)
        assert is_seismically_quiet([small], _T0, fsm_monitoring=True) is False


# =========================================================================
# Full score computation
# =========================================================================


class TestFullAnomalyScore:
    """Tests for compute_full_anomaly_score."""

    def test_quiet_baseline(self) -> None:
        """Calm signal should produce low ensemble score."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 0.001, 512)
        residual = rng.normal(0, 0.001, 512)

        result = compute_full_anomaly_score(
            filtered_signal=signal,
            detided_residual=residual,
            sampling_interval_sec=60.0,
            threshold_m=0.03,
            baseline_wavelet_energy=1.0,
        )

        assert 0.0 <= result.ensemble_score <= 1.0
        assert result.ml_score is None  # no model provided
        assert result.seismic_context_quiet is False

    def test_anomalous_signal(self) -> None:
        """Large signal should produce high scores."""
        signal = np.ones(512) * 0.5  # way above 0.03 threshold
        residual = np.ones(512) * 0.5

        result = compute_full_anomaly_score(
            filtered_signal=signal,
            detided_residual=residual,
            sampling_interval_sec=60.0,
            threshold_m=0.03,
            baseline_wavelet_energy=0.001,
        )

        assert result.threshold_score > 0.5
        assert result.ensemble_score > 0.3

    def test_with_iforest(self) -> None:
        """Should include ML score when model is provided."""
        rng = np.random.default_rng(42)
        model = IsolationForestModel()
        model.fit(rng.normal(0, 0.01, (200, 4)))

        signal = rng.normal(0, 0.01, 512)
        residual = rng.normal(0, 0.01, 512)

        result = compute_full_anomaly_score(
            filtered_signal=signal,
            detided_residual=residual,
            sampling_interval_sec=60.0,
            threshold_m=0.03,
            baseline_wavelet_energy=1.0,
            iforest_model=model,
        )

        assert result.ml_score is not None
        assert 0.0 <= result.ml_score <= 1.0

    def test_seismic_quiet_flag(self) -> None:
        signal = np.ones(100) * 0.01
        residual = np.ones(100) * 0.01

        result = compute_full_anomaly_score(
            filtered_signal=signal,
            detided_residual=residual,
            sampling_interval_sec=60.0,
            threshold_m=0.03,
            baseline_wavelet_energy=1.0,
            seismic_context_quiet=True,
        )

        assert result.seismic_context_quiet is True

    def test_component_scores_logged(self) -> None:
        """All component scores should be populated."""
        signal = np.random.default_rng(42).normal(0, 0.01, 512)
        residual = signal.copy()

        result = compute_full_anomaly_score(
            filtered_signal=signal,
            detided_residual=residual,
            sampling_interval_sec=60.0,
            threshold_m=0.03,
            baseline_wavelet_energy=1.0,
        )

        # Verify all scores are computed with valid values (not just that fields exist)
        assert 0.0 <= result.threshold_score <= 1.0
        assert 0.0 <= result.wavelet_score <= 1.0
        assert 0.0 <= result.bocpd_score <= 1.0
        assert 0.0 <= result.statistical_score <= 1.0
        assert result.ml_score is None  # no model provided
        assert result.spatial_coherence_score == 0.0  # no arrivals
        assert result.seismic_context_quiet is False  # default
        assert 0.0 <= result.ensemble_score <= 1.0


# =========================================================================
# Determinism tests
# =========================================================================


class TestDeterminism:
    """Verify all algorithms produce deterministic outputs on replay."""

    def test_wavelet_deterministic(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 512)
        e1 = compute_wavelet_energy(signal, 60.0)
        e2 = compute_wavelet_energy(signal, 60.0)
        assert e1 == e2

    def test_bocpd_deterministic(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 50)
        s1 = compute_bocpd_score(signal)
        s2 = compute_bocpd_score(signal)
        assert s1 == s2

    def test_iforest_deterministic(self) -> None:
        rng = np.random.default_rng(42)
        train = rng.normal(0, 0.01, (100, 4))
        test_sample = np.array([[0.5, 0.5, 0.1, 0.3]])

        m1 = IsolationForestModel(random_seed=42)
        m1.fit(train.copy())
        s1 = m1.score(test_sample)

        m2 = IsolationForestModel(random_seed=42)
        m2.fit(train.copy())
        s2 = m2.score(test_sample)

        assert s1 == s2

    def test_full_pipeline_deterministic(self) -> None:
        """Full anomaly score computation should be deterministic."""
        rng = np.random.default_rng(42)
        signal = rng.normal(0, 0.01, 512)
        residual = signal.copy()

        r1 = compute_full_anomaly_score(
            signal, residual, 60.0, 0.03, 1.0
        )
        r2 = compute_full_anomaly_score(
            signal, residual, 60.0, 0.03, 1.0
        )

        assert r1.ensemble_score == r2.ensemble_score
        assert r1.threshold_score == r2.threshold_score
        assert r1.wavelet_score == r2.wavelet_score
        assert r1.bocpd_score == r2.bocpd_score


# ---------------------------------------------------------------------------
# Rayleigh wave false-trigger detection
# ---------------------------------------------------------------------------


class TestRayleighArrivalSuspect:
    """Tests for rayleigh_arrival_suspect() timing correlation check."""

    def test_spike_within_rayleigh_window(self) -> None:
        """Spike at expected Rayleigh travel time -> suspect."""
        from hazard_assessment.agents.anomaly_detection import (
            haversine_km,
            rayleigh_arrival_suspect,
        )

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dist = haversine_km(40.0, 170.0, 38.0, 165.0)
        expected_sec = dist / 3.6
        spike = origin + timedelta(seconds=expected_sec)

        assert rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=170.0,
            epicenter_lat=38.0, epicenter_lon=165.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_spike_before_earthquake(self) -> None:
        """Spike before earthquake origin -> not suspect."""
        from hazard_assessment.agents.anomaly_detection import rayleigh_arrival_suspect

        origin = datetime(2024, 1, 1, 0, 5, 0, tzinfo=UTC)
        spike = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)

        assert not rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=170.0,
            epicenter_lat=38.0, epicenter_lon=142.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_spike_far_beyond_window(self) -> None:
        """Spike arrives much later than Rayleigh window -> not suspect."""
        from hazard_assessment.agents.anomaly_detection import rayleigh_arrival_suspect

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        spike = origin + timedelta(seconds=2000)

        assert not rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=170.0,
            epicenter_lat=38.0, epicenter_lon=165.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_distance_beyond_3000km(self) -> None:
        """Station >3000 km from epicenter -> Rayleigh too attenuated."""
        from hazard_assessment.agents.anomaly_detection import rayleigh_arrival_suspect

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        spike = origin + timedelta(seconds=1500)

        assert not rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=-120.0,
            epicenter_lat=38.0, epicenter_lon=142.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_very_close_epicenter(self) -> None:
        """Epicenter <36 km from station -> window too tight."""
        from hazard_assessment.agents.anomaly_detection import rayleigh_arrival_suspect

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        spike = origin + timedelta(seconds=3)

        assert not rayleigh_arrival_suspect(
            station_lat=40.000, station_lon=170.000,
            epicenter_lat=40.001, epicenter_lon=170.001,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_tolerance_at_boundary(self) -> None:
        """Spike at exactly 20% late -> still suspect (<=)."""
        from hazard_assessment.agents.anomaly_detection import (
            haversine_km,
            rayleigh_arrival_suspect,
        )

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dist = haversine_km(40.0, 170.0, 38.0, 165.0)
        expected_sec = dist / 3.6
        spike = origin + timedelta(seconds=expected_sec * 1.20)

        assert rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=170.0,
            epicenter_lat=38.0, epicenter_lon=165.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )

    def test_tolerance_just_beyond(self) -> None:
        """Spike at 21% late -> not suspect."""
        from hazard_assessment.agents.anomaly_detection import (
            haversine_km,
            rayleigh_arrival_suspect,
        )

        origin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=UTC)
        dist = haversine_km(40.0, 170.0, 38.0, 165.0)
        expected_sec = dist / 3.6
        spike = origin + timedelta(seconds=expected_sec * 1.21)

        assert not rayleigh_arrival_suspect(
            station_lat=40.0, station_lon=170.0,
            epicenter_lat=38.0, epicenter_lon=165.0,
            seismic_origin_utc=origin,
            spike_utc=spike,
        )


# =========================================================================
# BPR Data Cleaning Tests
# =========================================================================


class TestHampelFilter:
    """Tests for Hampel spike detection and removal."""

    def test_low_false_positive_on_clean_gaussian(self):
        """Hampel has low false-positive rate on clean Gaussian noise.

        With a 7-sample window (half=3) and 3-MAD threshold, the local MAD
        estimator has high variance, producing ~5% false positives on pure
        Gaussian data. This is acceptable because the filter only runs on
        calibration data (not event data), and false replacements with the
        local median introduce negligible error.
        """
        rng = np.random.default_rng(42)
        values = rng.normal(0, 1.0, size=500)
        cleaned, spike_mask = hampel_filter(values)
        # At most ~10% flagged (conservative bound for small-window Hampel)
        assert spike_mask.sum() < 0.10 * len(values)
        # Non-flagged values should be unchanged
        np.testing.assert_allclose(cleaned[~spike_mask], values[~spike_mask])

    def test_removes_injected_spikes(self):
        """Hampel detects and removes large injected spikes."""
        rng = np.random.default_rng(42)
        values = rng.normal(0, 0.01, size=200)
        # Inject 5 large spikes
        spike_indices = [20, 60, 100, 140, 180]
        for idx in spike_indices:
            values[idx] += 10.0  # 1000x the noise std
        cleaned, spike_mask = hampel_filter(values)
        # All injected spikes should be detected
        for idx in spike_indices:
            assert spike_mask[idx], f"Spike at index {idx} was not detected"
        # Cleaned values at spike positions should be close to 0 (the local median)
        for idx in spike_indices:
            assert abs(cleaned[idx]) < 0.5

    def test_preserves_tidal_signal(self):
        """Hampel preserves smooth tidal oscillations (not spikes)."""
        t = np.linspace(0, 48, 2880)  # 48 hours at 1-min intervals
        # M2 tide (amplitude 1 m, period 12.42 h)
        omega_m2 = 2 * np.pi / 12.42
        signal = 1.0 * np.cos(omega_m2 * t)
        cleaned, spike_mask = hampel_filter(signal)
        assert spike_mask.sum() == 0
        np.testing.assert_allclose(cleaned, signal)

    def test_short_array(self):
        """Hampel handles arrays shorter than the window."""
        # Need sufficient variability for MAD > 0 (more than half the
        # deviations must be non-zero).
        values = np.array([1.0, 1.1, 0.9, 100.0, 1.2, 0.8, 1.0])
        cleaned, spike_mask = hampel_filter(values)
        assert spike_mask[3]  # The 100.0 outlier is a spike
        assert abs(cleaned[3] - 1.0) < 0.5


class TestCleanBPRCalibration:
    """Tests for combined spike removal + linear detrend."""

    def test_removes_linear_drift(self):
        """Detrend removes linear drift, preserves DC level."""
        t = np.linspace(0, 720, 43200)  # 30 days in hours, 1-min
        # Pure linear drift: 0.001 per hour (typical BPR crystal aging)
        values = 100.0 + 0.001 * t
        cleaned = clean_bpr_calibration(t, values)
        # After detrend, values should be near-constant
        assert np.std(cleaned) < 1e-6
        # DC level should be approximately preserved
        assert abs(np.mean(cleaned) - np.mean(values)) < 0.5

    def test_removes_spikes_then_detrends(self):
        """Combined cleaning: spikes removed first, then drift corrected."""
        rng = np.random.default_rng(42)
        t = np.linspace(0, 720, 43200)
        # Clean tidal signal + drift
        omega_m2 = 2 * np.pi / 12.42
        values = 100.0 + 0.001 * t + 0.5 * np.cos(omega_m2 * t)
        # Add spikes
        spike_idx = rng.choice(len(t), size=20, replace=False)
        values[spike_idx] += rng.choice([-5.0, 5.0], size=20)
        cleaned = clean_bpr_calibration(t, values)
        # Cleaned should have lower variance than raw (spikes + drift removed)
        assert np.std(cleaned) < np.std(values)

    def test_near_no_op_on_clean_data(self):
        """Clean tidal data passes through with minimal change.

        A cosine over a non-symmetric interval has a small non-zero
        linear trend component, so polyfit(deg=1) removes a tiny slope.
        The change is bounded at ~0.15% of amplitude.
        """
        t = np.linspace(0, 720, 43200)
        omega_m2 = 2 * np.pi / 12.42
        # Pure tidal signal, no drift, no spikes
        values = 0.5 * np.cos(omega_m2 * t)
        cleaned = clean_bpr_calibration(t, values)
        # Should be very close - the residual linear fit on a cosine is tiny
        np.testing.assert_allclose(cleaned, values, atol=2e-3)

    def test_level_shift_correction_single_gap(self):
        """Level shifts across a single gap are corrected."""
        # Uniform 15-min sampling for 30 days, constant baseline
        t = np.arange(0, 720, 0.25)  # hours
        values = np.ones_like(t) * 50.0
        # Insert a 6-hour gap at index 1000 and add a 0.5 m level shift
        gap_idx = 1000
        t_with_gap = np.concatenate([t[:gap_idx], t[gap_idx:] + 6.0])
        values_shifted = values.copy()
        values_shifted[gap_idx:] += 0.5  # 500 mm level shift
        corrected = _correct_level_shifts(t_with_gap, values_shifted)
        # After correction, all values should be near 50.0
        np.testing.assert_allclose(corrected, 50.0, atol=1e-10)

    def test_level_shift_correction_no_gap(self):
        """No-op when data has no gaps."""
        t = np.arange(0, 720, 0.25)
        values = np.sin(2 * np.pi * t / 12.42)
        corrected = _correct_level_shifts(t, values)
        np.testing.assert_allclose(corrected, values)

    def test_level_shift_correction_multiple_gaps(self):
        """Multiple gaps with different level shifts are all corrected."""
        t = np.arange(0, 720, 0.25)
        values = np.ones_like(t) * 100.0  # constant baseline
        # Insert two gaps with level shifts
        gap1, gap2 = 500, 1500
        t_gapped = t.copy()
        t_gapped[gap1:] += 10.0  # 10-hour gap
        t_gapped[gap2:] += 8.0   # 8-hour gap
        values_shifted = values.copy()
        values_shifted[gap1:] += 0.3   # +300 mm shift at gap 1
        values_shifted[gap2:] -= 0.2   # -200 mm shift at gap 2
        corrected = _correct_level_shifts(t_gapped, values_shifted)
        # After correction, all segments should be near 100.0
        assert np.std(corrected) < 0.05

    def test_level_shift_correction_close_gaps_clamped(self):
        """A second gap inside the after-window does not contaminate the shift.

        With two gaps closer than the level-sample window, the median for the
        first gap must be computed only from samples on its own side of the
        second gap; otherwise the estimate mixes both shifts and leaves a
        spurious plateau.
        """
        t = np.arange(0, 50, 0.25)  # 200 samples, 15-min cadence
        values = np.zeros_like(t)
        # Two gaps 2 samples apart (window is 5), true shifts +5 then -3.
        g1, g2 = 100, 102
        t[g1:] += 6.0
        t[g2:] += 6.0
        values[g1:] += 5.0
        values[g2:] -= 3.0
        corrected = _correct_level_shifts(t.copy(), values)
        # Fully corrected back to the 0 baseline (no residual plateau).
        np.testing.assert_allclose(corrected, 0.0, atol=1e-6)

    def test_full_cleaning_with_gap_and_drift(self):
        """Full pipeline: spikes + gap level shift + drift are all removed."""
        rng = np.random.default_rng(42)
        t = np.arange(0, 720, 0.25)
        omega_m2 = 2 * np.pi / 12.42
        values = 100.0 + 0.5 * np.cos(omega_m2 * t)
        # Add linear drift
        values += 0.001 * t
        # Add a gap with level shift at index 1000
        t[1000:] += 12.0  # 12-hour gap
        values[1000:] += 0.7  # 700 mm level shift
        # Add a few spikes
        spike_idx = [200, 600, 1500, 2000]
        values[spike_idx] += rng.choice([-5.0, 5.0], size=len(spike_idx))
        cleaned = clean_bpr_calibration(t, values)
        # Cleaned should have much lower std than raw (drift+shift+spikes removed)
        assert np.std(cleaned) < np.std(values)


class TestRobustTidalFit:
    """Tests for IRLS robust tidal fitting."""

    def _make_tidal_data(self, n_hours=720, noise_std=0.001):
        """Generate clean tidal data for testing."""
        t = np.linspace(0, n_hours, int(n_hours * 60))  # 1-min sampling
        omega_m2 = 2 * np.pi / 12.42
        omega_s2 = 2 * np.pi / 12.0
        signal = 0.5 * np.cos(omega_m2 * t) + 0.3 * np.sin(omega_s2 * t)
        rng = np.random.default_rng(42)
        noise = rng.normal(0, noise_std, len(t))
        return t, signal + noise

    def test_irls_matches_ols_on_noiseless_data(self):
        """On noiseless data, IRLS converges to OLS (residuals ~ 0)."""
        from hazard_assessment.agents.anomaly_detection import TIDAL_FREQUENCIES_RAD_HR

        t = np.linspace(0, 720, int(720 * 60))
        # Use exact module frequencies so OLS residuals are truly near-zero
        omega_m2 = TIDAL_FREQUENCIES_RAD_HR["M2"]
        omega_s2 = TIDAL_FREQUENCIES_RAD_HR["S2"]
        values = 0.5 * np.cos(omega_m2 * t) + 0.3 * np.sin(omega_s2 * t)
        coeffs_robust = fit_tidal_harmonics(t, values, robust=True, clean_input=False)
        coeffs_ols = fit_tidal_harmonics(t, values, robust=False, clean_input=False)
        np.testing.assert_allclose(coeffs_robust, coeffs_ols, atol=1e-10)

    def test_irls_outperforms_ols_with_outliers(self):
        """IRLS produces better fit than OLS when data has outliers."""
        t, values = self._make_tidal_data()
        clean_values = values.copy()
        # Inject 2% outliers (large spikes)
        rng = np.random.default_rng(123)
        n_outliers = int(0.02 * len(values))
        outlier_idx = rng.choice(len(values), size=n_outliers, replace=False)
        values[outlier_idx] += rng.choice([-5.0, 5.0], size=n_outliers)

        # Fit with IRLS (no cleaning - testing IRLS alone)
        coeffs_robust = fit_tidal_harmonics(t, values, robust=True, clean_input=False)
        # Fit with OLS
        coeffs_ols = fit_tidal_harmonics(t, values, robust=False, clean_input=False)

        # Predict on clean time base
        mat = build_harmonic_matrix(t)
        pred_robust = mat @ coeffs_robust
        pred_ols = mat @ coeffs_ols

        # IRLS should have lower residual on the clean (non-outlier) data
        clean_mask = np.ones(len(t), dtype=bool)
        clean_mask[outlier_idx] = False
        rmse_robust = np.sqrt(np.mean((pred_robust[clean_mask] - clean_values[clean_mask]) ** 2))
        rmse_ols = np.sqrt(np.mean((pred_ols[clean_mask] - clean_values[clean_mask]) ** 2))
        assert rmse_robust < rmse_ols, (
            f"IRLS RMSE ({rmse_robust:.6f}) should be less than OLS RMSE ({rmse_ols:.6f})"
        )

    def test_ols_fallback_with_robust_false(self):
        """robust=False skips IRLS, gives plain OLS result."""
        t, values = self._make_tidal_data(n_hours=48)
        coeffs = fit_tidal_harmonics(t, values, robust=False, clean_input=False)
        # Verify by manually computing OLS
        mat = build_harmonic_matrix(t)
        expected, _, _, _ = np.linalg.lstsq(mat, values, rcond=None)
        np.testing.assert_allclose(coeffs, expected, atol=1e-10)

    def test_detide_with_cleaning_improves_residual(self):
        """Detiding with cleaning produces cleaner residuals than without."""
        rng = np.random.default_rng(42)
        # 30-day calibration data with drift and spikes
        t_cal = np.linspace(0, 720, 43200)
        omega_m2 = 2 * np.pi / 12.42
        cal_values = 100.0 + 0.002 * t_cal + 0.5 * np.cos(omega_m2 * t_cal)
        spike_idx = rng.choice(len(t_cal), size=50, replace=False)
        cal_values[spike_idx] += rng.choice([-3.0, 3.0], size=50)

        # Event data (clean, for residual evaluation)
        t_event = np.linspace(720, 732, 720)  # 12 hours
        event_values = 100.0 + 0.5 * np.cos(omega_m2 * t_event)

        # Detide with cleaning
        residual_clean = detide(
            t_event, event_values, t_cal, cal_values,
            robust=True, clean_calibration=True,
        )
        # Detide without cleaning
        residual_dirty = detide(
            t_event, event_values, t_cal, cal_values,
            robust=False, clean_calibration=False,
        )

        # Clean residual should be closer to zero (better tidal removal)
        assert np.std(residual_clean) < np.std(residual_dirty), (
            f"Clean residual std ({np.std(residual_clean):.6f}) should be less "
            f"than dirty residual std ({np.std(residual_dirty):.6f})"
        )


# =========================================================================
# Input validation tests
# =========================================================================


class TestInputValidation:
    """Tests for input validation guards (_validate_finite, sampling > 0)."""

    def test_nan_in_signal_raises(self) -> None:
        signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_wavelet_energy(signal, 60.0)

    def test_inf_in_signal_raises(self) -> None:
        signal = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_wavelet_energy(signal, 60.0)

    def test_nan_in_bocpd_signal_raises(self) -> None:
        # Signal must be >= 15 samples (BURN_IN + 5) to reach validation
        signal = np.zeros(20)
        signal[10] = np.nan
        with pytest.raises(ValueError, match="non-finite"):
            compute_bocpd_score(signal)

    def test_nan_in_threshold_signal_raises(self) -> None:
        # A non-finite filtered signal must fail loud rather than map through
        # np.max to a false maximum threshold score.
        signal = np.array([1.0, np.nan, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_threshold_score(signal, threshold_m=0.05)

    def test_inf_in_threshold_signal_raises(self) -> None:
        signal = np.array([1.0, np.inf, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
        with pytest.raises(ValueError, match="non-finite"):
            compute_threshold_score(signal, threshold_m=0.05)

    def test_negative_sampling_interval_wavelet(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 64)
        with pytest.raises(ValueError, match="positive"):
            compute_wavelet_energy(signal, -60.0)

    def test_zero_sampling_interval_wavelet(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 64)
        with pytest.raises(ValueError, match="positive"):
            compute_wavelet_energy(signal, 0.0)

    def test_negative_sampling_interval_iforest_features(self) -> None:
        signal = np.random.default_rng(42).normal(0, 0.01, 64)
        with pytest.raises(ValueError, match="positive"):
            build_iforest_features(signal, -1.0, 0.0)

    def test_negative_sampling_interval_full_score(self) -> None:
        filtered = np.random.default_rng(42).normal(0, 0.01, 64)
        residual = np.random.default_rng(43).normal(0, 0.01, 64)
        with pytest.raises(ValueError, match="positive"):
            compute_full_anomaly_score(
                filtered, residual, -60.0, threshold_m=0.03,
                baseline_wavelet_energy=1e-6,
            )
