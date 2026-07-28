"""Anomaly detection algorithms: detiding, filtering, and score components.

This module implements the deterministic anomaly detection pipeline:
- Tidal removal via harmonic analysis (M2, S2, N2, K1, O1, P1, K2, Q1);
  caller provides the fit window
- Butterworth 4th-order bandpass filter (5-120 min tsunami band)
- Wavelet energy decomposition (db4, PyWavelets)
- Bayesian Online Changepoint Detection (BOCPD)
- Isolation Forest anomaly scoring
- Spatial coherence checking
- Ensemble fusion with seismic context adjustment

All outputs are deterministic on replay (locked random seeds, deterministic math).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import pywt
from numpy.typing import NDArray
from scipy.signal import butter, sosfilt, sosfilt_zi
from scipy.special import gammaln

from hazard_assessment.geo import (
    DEEP_OCEAN_WAVE_SPEED_KM_S as DEEP_OCEAN_WAVE_SPEED_KM_S,
)
from hazard_assessment.geo import (
    compute_travel_time_sec,
    haversine_km,
)

# Single source of the tidal constituents (see tidal.py). Re-exported here
# with redundant aliases so existing `from ...anomaly_detection import
# TIDAL_*` call sites keep working, exactly as the geo constants above are
# re-exported.
from hazard_assessment.tidal import (
    TIDAL_CONSTITUENTS as TIDAL_CONSTITUENTS,
)
from hazard_assessment.tidal import (
    TIDAL_FREQUENCIES_RAD_HR as TIDAL_FREQUENCIES_RAD_HR,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# TIDAL_CONSTITUENTS and TIDAL_FREQUENCIES_RAD_HR are defined once in
# tidal.py and re-exported at the top of this module (see the imports above).

# Bandpass filter specifications - tsunami period band (5-120 min).
#
# Most Pacific tsunamis have dominant periods of 10-60 min, but great
# earthquakes (M9+) can produce significant energy at 90-120 min periods
# (e.g., 2004 Indian Ocean tsunami; Rabinovich, 2009). The 120 min upper
# cutoff captures these long-period components while still rejecting
# storm surge and tidal residuals (periods >> 2 hours).
BANDPASS_LOW_HZ = 1.0 / (120 * 60)  # 120 min period
BANDPASS_HIGH_HZ = 1.0 / (5 * 60)   # 5 min period
BUTTERWORTH_ORDER = 4

# Wavelet parameters
WAVELET_FAMILY = "db4"
WAVELET_LEVEL_1MIN = 6   # for 1-min data (covers up to 2^7*60=7680s ~ 128 min)
WAVELET_LEVEL_15SEC = 8  # for 15-sec data (covers up to 2^9*15=7680s ~ 128 min)

# Tsunami period band boundaries (seconds) for wavelet level selection
TSUNAMI_PERIOD_LOW_SEC = 300.0    # 5 min lower bound
TSUNAMI_PERIOD_HIGH_SEC = 7200.0  # 120 min upper bound

# BOCPD hyperparameters.
#
# Hazard rate lambda = 1/300: at 60-second sampling this means an expected
# changepoint every ~300 samples (~5 hours), roughly matching the typical
# inter-event quiet period for active DART stations. A lower value (e.g.,
# 1/500) would reduce false alarms during quiet periods at the cost of
# slower onset detection; a higher value (e.g., 1/100) would detect faster
# but increase false positives from tidal residuals. Subject to calibration.
#
# IMPORTANT: Under a constant hazard function, the *posterior changepoint
# probability* P(r_t = 0 | x_{1:t}) equals lambda regardless of the data - it is
# uninformative. Scoring therefore uses *predictive log-evidence surprise*
# (state.last_log_evidence) rather than the changepoint probability. See
# compute_bocpd_score() for the surprise-based scoring logic.
#
# Prior precision kappa0 = 1.0 (set in AnomalyAgent): weak prior equivalent
# to 1 pseudo-observation. Should be calibrated per-station from baseline
# data (e.g., kappa0 = 1/sigma^2_baseline) for optimal sensitivity. Tests use
# larger values (kappa0=100) for deterministic level-shift detection.
BOCPD_HAZARD_LAMBDA = 1.0 / 300.0  # expect changepoint every ~300 samples
BOCPD_MAX_RUN_LENGTHS = 600        # truncation limit for efficiency

# Isolation Forest parameters
IFOREST_N_ESTIMATORS = 100
IFOREST_CONTAMINATION = 0.01
IFOREST_RANDOM_SEED = 42  # deterministic
IFOREST_SIGMOID_STEEPNESS = 5.0  # maps decision_function range ~[-0.5,0.5] to [0,1]

# Spatial coherence: deep-ocean wave speed lives in hazard_assessment.geo
# (re-exported here so existing importers keep working).
# +/-20% tolerance on inter-station travel time matching. This accounts for
# bathymetric variation (ocean depth ranges ~2-6 km along typical trans-Pacific
# paths, yielding wave speeds of ~140-240 m/s). The +/-20% window around the
# 4 km mean-depth estimate covers most paths but may reject valid detections
# along paths with extreme depth variation (e.g., crossing mid-ocean ridges).
# Source geometry loosens it further, and in one direction only: the expected
# delay d_ij/c holds only for a source aligned with the station baseline, and
# is an upper bound otherwise, so real delays sit in the lower part of the
# window and pairs offset in azimuth from the source fall out of it entirely.
# A more rigorous approach would use station-pair-specific travel time tables
# from pre-computed ray tracing (e.g., MOST/RIFT model grids).
SPATIAL_CONFIRMATION_WINDOW_FRACTION = 0.20  # +/-20%
SPATIAL_MIN_CONFIRMING_STATIONS = 2

# Ensemble fusion weights.
#
# Rationale for the 0.50 / 0.35 / 0.15 split:
#
# - Threshold (0.50): Highest weight because it is the most directly
#   interpretable detector - a physical amplitude above a known hardware
#   trigger level (DART 3 cm). It has zero latency (instantaneous) and its
#   false-positive rate is bounded by the detiding quality. Against the
#   checked-in retrospective artifacts the threshold detector alone crossed
#   T1 at 7 of 8 Tohoku 2011 stations (46403 reached 0.344) and at 2 of 7
#   Chile 2010 stations, so it carries the near-field signal but does not
#   stand alone at range; the statistical detector supplies the rest.
#
# - Statistical (0.35): Combined wavelet energy ratio and BOCPD changepoint
#   score. These detect distributional shifts that precede or accompany
#   the amplitude threshold crossing, providing early warning and confirming
#   signal persistence. Lower weight because wavelet scoring requires
#   pre-calibrated baseline energy and BOCPD is sensitive to hyperparameters.
#
# - ML / Isolation Forest (0.15): Lowest weight because the model is
#   optional (not always available), requires offline training on labeled
#   data, and adds marginal discriminative power beyond the threshold and
#   statistical detectors in validation testing.
#
# These are initial weights subject to formal calibration via ablation study
# on historical events. The ensemble_weights parameter in
# compute_ensemble_score() allows per-call override for sensitivity analysis.
W_THRESHOLD = 0.50
W_STATISTICAL = 0.35
W_ML = 0.15
# Renormalized weights when ML is unavailable - computed from base weights so they
# stay in sync automatically if W_THRESHOLD or W_STATISTICAL are ever adjusted.
# W_THRESHOLD / (W_THRESHOLD + W_STATISTICAL) = 0.50 / 0.85 ~ 0.5882
# W_STATISTICAL / (W_THRESHOLD + W_STATISTICAL) = 0.35 / 0.85 ~ 0.4118
_W_NO_ML_SUM = W_THRESHOLD + W_STATISTICAL
W_THRESHOLD_NO_ML = W_THRESHOLD / _W_NO_ML_SUM
W_STATISTICAL_NO_ML = W_STATISTICAL / _W_NO_ML_SUM

# Seismic context - when no M>=6.5 earthquake has occurred within the look-back
# window, the threshold detector is made less sensitive (higher threshold) to
# reduce false positives from non-seismic ocean noise.
#
# SEISMIC_QUIET_THRESHOLD_FACTOR (1.3): Raises the amplitude threshold by 30%
# during seismically quiet periods. Derived empirically: Chile 2010 DART
# residual noise (post-detide) peaks at ~0.025 m, while the T1 threshold is
# 0.03 m - a 1.3x multiplier lifts the quiet-period threshold to 0.039 m,
# safely above the noise floor while still detecting >10 cm tsunami signals.
#
# SEISMIC_QUIET_WINDOW_MINUTES (90): Look-back window over earthquake origin
# times. An origin older than this no longer suppresses the quiet-period
# threshold boost. The value is an operational default, not a derived
# quantity: it is long enough to span initial teleseismic magnitude and
# moment determination for a large event (tens of minutes) with margin for
# late or revised origins, and short enough that unrelated older seismicity
# does not hold the threshold down indefinitely. Subject to calibration
# against historical event catalogs.
#
# SEISMIC_QUIET_MAGNITUDE_THRESHOLD (6.5): Minimum magnitude considered
# tsunamigenic. Matches the PTWC threshold for tsunami information statements
# (PTWC Operations Manual, section 4.2).
SEISMIC_QUIET_THRESHOLD_FACTOR = 1.3
SEISMIC_QUIET_MAGNITUDE_THRESHOLD = 6.5
SEISMIC_QUIET_WINDOW_MINUTES = 90

# BPR data cleaning - Hampel spike filter
# Window of 7 (half=3) and a 3.0-MAD cut are chosen defaults, not a cited
# DART QC specification. The window is deliberately short so real tidal and
# tsunami structure is preserved; the cost is a noisy local scale estimate.
# 1.4826*MAD is the asymptotic Gaussian-consistency factor, and over 7 samples
# (4 at the array edges) it underestimates sigma by about 12% (about 27% at
# the edges), so the effective cut sits tighter than 3 sigma. The measured
# false-flag rate on clean Gaussian noise is about 5%, not 0.3%; see
# tests/unit/test_anomaly_detection.py::test_low_false_positive_on_clean_gaussian.
# In the scoring path that is acceptable: clean_bpr_calibration is the only
# caller, it runs on calibration data alone, and it replaces flagged samples
# with the local median. The paper-figure scripts also call hampel_filter on
# event windows, but they pass their own wider half_window rather than these
# constants.
HAMPEL_WINDOW_HALF = 3
HAMPEL_THRESHOLD_MAD = 3.0

# Gap detection and level-shift correction
# DART BPR calibration records commonly have gaps (instrument resets,
# communication drops) accompanied by pressure level shifts (Bourdon
# tube settling, clock resets). A gap is flagged when the sample
# interval exceeds GAP_FACTOR x the median interval.
GAP_FACTOR = 6.0
# Number of samples on each side of a gap used to estimate the level shift.
GAP_LEVEL_SAMPLES = 5

# IRLS robust tidal fitting - Cauchy weight function
# Standard in UTide (Codiga 2011, Leffler & Jay 2009).
# Converges in 1 iteration on clean data (equivalent to OLS).
IRLS_MAX_ITERATIONS = 20
IRLS_CONVERGENCE_TOL = 1e-6


# ---------------------------------------------------------------------------
# BPR data cleaning - spike removal and drift correction
# ---------------------------------------------------------------------------


def hampel_filter(
    values: NDArray[np.float64],
    half_window: int = HAMPEL_WINDOW_HALF,
    threshold: float = HAMPEL_THRESHOLD_MAD,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Sliding-window median + MAD-based spike detection and replacement.

    For each sample, computes the local median and MAD (median absolute
    deviation) within a symmetric window. Samples deviating more than
    ``threshold`` MAD from the local median are replaced by the median.

    Uses the ``1.4826 * MAD`` scale factor for consistency with the
    standard deviation of a Gaussian distribution (Hampel 1974).

    Args:
        values: Input 1-D array.
        half_window: Half-width of the sliding window (full width = 2*half_window + 1).
        threshold: Number of MAD units above which a sample is flagged as a spike.

    Returns:
        Tuple of (cleaned_values, spike_mask) where spike_mask is boolean
        (True = spike detected and replaced).
    """
    n = len(values)
    cleaned = values.copy()
    spike_mask = np.zeros(n, dtype=bool)

    for i in range(n):
        lo = max(0, i - half_window)
        hi = min(n, i + half_window + 1)
        window = values[lo:hi]
        local_median = np.median(window)
        mad = np.median(np.abs(window - local_median))
        # 1.4826 * MAD = consistent estimator of sigma for Gaussian data
        scale = 1.4826 * mad
        if scale < 1e-12:
            # Near-constant window - no meaningful deviation
            continue
        if abs(values[i] - local_median) > threshold * scale:
            cleaned[i] = local_median
            spike_mask[i] = True

    return cleaned, spike_mask


def _correct_level_shifts(
    times_hours: NDArray[np.float64],
    values: NDArray[np.float64],
    gap_factor: float = GAP_FACTOR,
    level_samples: int = GAP_LEVEL_SAMPLES,
) -> NDArray[np.float64]:
    """Detect gaps in BPR data and correct pressure level shifts.

    DART BPR records commonly have gaps (instrument resets, communication
    drops) accompanied by pressure level shifts (Bourdon tube settling,
    clock resets).  A gap is detected when the sampling interval exceeds
    ``gap_factor`` times the median interval.  At each gap, the level
    shift is estimated from the median of ``level_samples`` points on
    each side, and all subsequent data is adjusted to remove the shift.

    Args:
        times_hours: Time values in hours.
        values: Pressure values (already spike-cleaned).
        gap_factor: Multiplier on median dt to detect gaps.
        level_samples: Points on each side of gap for shift estimate.

    Returns:
        Level-shift-corrected values.
    """
    if len(values) < 2 * level_samples + 1:
        return values.copy()

    dt = np.diff(times_hours)
    median_dt = np.median(dt)
    if median_dt <= 0:
        return values.copy()

    gap_indices = np.where(dt > gap_factor * median_dt)[0]
    if len(gap_indices) == 0:
        return values.copy()

    corrected = values.copy()

    for gap_pos, gi in enumerate(gap_indices):
        # Median of `level_samples` points before and after the gap, clamped
        # to the adjacent gaps so a close pair does not contaminate each
        # other's level estimate.
        prev_boundary = gap_indices[gap_pos - 1] + 1 if gap_pos > 0 else 0
        next_boundary = (
            gap_indices[gap_pos + 1] + 1
            if gap_pos + 1 < len(gap_indices)
            else len(values)
        )
        before_lo = max(prev_boundary, gi + 1 - level_samples)
        before_hi = gi + 1
        after_lo = gi + 1
        after_hi = min(next_boundary, gi + 1 + level_samples)

        if before_hi <= before_lo or after_hi <= after_lo:
            continue

        median_before = float(np.median(corrected[before_lo:before_hi]))
        median_after = float(np.median(corrected[after_lo:after_hi]))
        shift = median_after - median_before

        # Adjust all data after this gap
        corrected[after_lo:] -= shift
        logger.info(
            "Level shift of %.1f mm corrected at gap (t=%.1f h, dt=%.1f h)",
            shift * 1000, times_hours[gi], dt[gi],
        )

    return corrected


def clean_bpr_calibration(
    times_hours: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    detrend: bool = True,
) -> NDArray[np.float64]:
    """Clean BPR calibration data: spike removal, level-shift correction, detrend.

    Three-step cleaning applied to calibration (non-event) data only:
    1. Hampel spike removal - replaces outlier samples with local medians.
    2. Level-shift correction - detects gaps (>6x median sampling interval)
       and removes pressure level shifts across them. This preserves the
       full record length for harmonic resolution while eliminating DC
       discontinuities that corrupt the tidal fit.
    3. Linear detrend (optional) - removes instrumental drift (crystal aging)
       via ``np.polyfit(deg=1)``. Skip when drift terms are included in the
       harmonic design matrix (``include_drift=True``).

    This function must NOT be applied to event-window data - pressure
    anomalies in the event window could be real tsunami signals.

    Args:
        times_hours: Time values in hours (for detrending).
        values: Raw BPR pressure values from the calibration window.
        detrend: If True (default), remove linear drift. Set False when
            drift terms are included in the harmonic design matrix.

    Returns:
        Cleaned values with spikes removed, level shifts corrected,
        and optionally linear drift removed.
    """
    # Step 1: Spike removal
    cleaned, spike_mask = hampel_filter(values)
    n_spikes = int(np.sum(spike_mask))
    if n_spikes > 0:
        logger.info("Hampel filter removed %d spike(s) from calibration data", n_spikes)

    # Step 2: Level-shift correction at gaps
    cleaned = _correct_level_shifts(times_hours, cleaned)

    # Step 3: Linear detrend - remove drift, preserve DC level
    if detrend and len(times_hours) >= 2:
        poly = np.polyfit(times_hours, cleaned, deg=1)
        slope = poly[0]
        # Remove only the linear trend (slope * t), keep the intercept
        cleaned = cleaned - slope * times_hours

    return cleaned


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_finite(arr: NDArray[np.float64], name: str) -> None:
    """Raise ValueError if array contains NaN or Inf values."""
    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN/Inf); "
            "upstream data must be cleaned before anomaly detection"
        )


# ---------------------------------------------------------------------------
# Detiding - 30-day rolling harmonic analysis
# ---------------------------------------------------------------------------

def build_harmonic_matrix(
    times_hours: NDArray[np.float64],
    constituents: dict[str, float] | None = None,
    include_drift: bool = False,
    drift_t_mean: float | None = None,
    drift_t_range: float | None = None,
) -> NDArray[np.float64]:
    """Build the design matrix for harmonic tidal analysis.

    For each constituent, adds cos(omega*t) and sin(omega*t) columns plus a
    constant (mean) column.  When *include_drift* is True, appends
    normalized linear and quadratic drift columns to absorb BPR
    instrument drift (Watts & Kontoyiannis, 1990).

    Args:
        times_hours: Time values in hours relative to an epoch.
        constituents: Mapping of constituent name -> angular frequency in
            radians/hour.  Defaults to the 8 standard constituents.
        include_drift: If True, append linear and quadratic drift columns.
        drift_t_mean: Mean time for drift normalization (from fitting).
            If None, computed from *times_hours*.
        drift_t_range: Time range for drift normalization (from fitting).
            If None, computed from *times_hours*.

    Returns:
        Design matrix of shape (n_times, 2*n_constituents + 1) without
        drift, or (n_times, 2*n_constituents + 3) with drift.
    """
    if constituents is None:
        constituents = TIDAL_FREQUENCIES_RAD_HR

    n = len(times_hours)
    n_const = len(constituents)
    mat = np.ones((n, 2 * n_const + 1), dtype=np.float64)

    for i, (_, omega) in enumerate(sorted(constituents.items())):
        phase = omega * times_hours
        mat[:, 2 * i + 1] = np.cos(phase)
        mat[:, 2 * i + 2] = np.sin(phase)

    if include_drift:
        t_mean = drift_t_mean if drift_t_mean is not None else float(times_hours.mean())
        t_range = drift_t_range if drift_t_range is not None else float(
            times_hours.max() - times_hours.min())
        if t_range < 1e-6:
            t_range = 1.0
        t_norm = (times_hours - t_mean) / t_range
        mat = np.column_stack([mat, t_norm, t_norm ** 2])

    return mat


def _cauchy_weights(
    residuals: NDArray[np.float64],
    scale: float,
) -> NDArray[np.float64]:
    """Cauchy (Lorentzian) weight function for IRLS robust regression.

    w(r) = 1 / (1 + (r/scale)^2)

    The Cauchy weight function down-weights outliers more aggressively than
    Huber weights but less than bisquare (Tukey). It is the standard choice
    in UTide (Codiga 2011) for robust tidal harmonic fitting.

    Args:
        residuals: Fit residuals.
        scale: Robust scale estimate (e.g. 1.4826 * MAD).

    Returns:
        Weight array in (0, 1], same shape as residuals.
    """
    if scale < 1e-12:
        return np.ones_like(residuals)
    z = residuals / scale
    return 1.0 / (1.0 + z * z)


def fit_tidal_harmonics(
    times_hours: NDArray[np.float64],
    values: NDArray[np.float64],
    *,
    robust: bool = True,
    clean_input: bool = False,
    include_drift: bool = False,
) -> NDArray[np.float64] | tuple[NDArray[np.float64], float, float]:
    """Fit tidal harmonics and return coefficients.

    When ``clean_input`` is True, applies BPR calibration cleaning (Hampel
    spike removal + linear detrend) before fitting. When ``robust`` is True,
    uses Iteratively Reweighted Least Squares (IRLS) with Cauchy weights to
    down-weight remaining outliers. On clean data, IRLS converges in 1
    iteration to the OLS solution - no behavior change for synthetic tests.

    When ``include_drift`` is True, appends linear+quadratic drift columns
    to the design matrix (Watts & Kontoyiannis, 1990) and returns a tuple
    ``(coeffs, t_mean, t_range)`` so ``predict_tide`` can apply the same
    normalization.

    Args:
        times_hours: Time values in hours relative to an epoch.
        values: Observed water level values.
        robust: If True, use IRLS with Cauchy weights after initial OLS.
        clean_input: If True, apply ``clean_bpr_calibration()`` to the
            input before fitting.
        include_drift: If True, add linear+quadratic drift columns.

    Returns:
        Coefficient vector of shape (2*n_constituents + 1,) without drift,
        or ``(coeffs, t_mean, t_range)`` tuple when ``include_drift=True``.
    """
    # Optionally clean calibration data (spike removal + detrend)
    fit_values = values
    if clean_input:
        fit_values = clean_bpr_calibration(times_hours, values)

    # Compute drift normalization from fitting times
    drift_t_mean = float(times_hours.mean()) if include_drift else None
    drift_t_range = (float(times_hours.max() - times_hours.min())
                     if include_drift else None)

    mat = build_harmonic_matrix(
        times_hours, include_drift=include_drift,
        drift_t_mean=drift_t_mean, drift_t_range=drift_t_range)

    # Initial OLS fit
    coeffs, _, _, _ = np.linalg.lstsq(mat, fit_values, rcond=None)

    if not robust:
        return coeffs  # type: ignore[return-value]

    # IRLS with Cauchy weights
    delta = float("inf")  # guard: defined before loop in case MAX_ITERATIONS=0
    for _ in range(IRLS_MAX_ITERATIONS):
        residuals = fit_values - mat @ coeffs
        mad = float(np.median(np.abs(residuals - np.median(residuals))))
        scale = 1.4826 * mad

        if scale < 1e-12:
            # Residuals are near-zero - OLS fit is already excellent
            break

        weights = _cauchy_weights(residuals, scale)
        # Weighted least squares: W^{1/2} * A * x = W^{1/2} * b
        sqrt_w = np.sqrt(weights)
        w_mat = mat * sqrt_w[:, np.newaxis]
        w_vals = fit_values * sqrt_w

        new_coeffs, _, _, _ = np.linalg.lstsq(w_mat, w_vals, rcond=None)

        # Check convergence
        delta = float(np.max(np.abs(new_coeffs - coeffs)))
        coeffs = new_coeffs
        if delta < IRLS_CONVERGENCE_TOL:
            break
    else:
        logger.warning(
            "IRLS robust fitting did not converge after %d iterations "
            "(delta=%.2e, tol=%.2e)",
            IRLS_MAX_ITERATIONS, delta, IRLS_CONVERGENCE_TOL,
        )

    if include_drift:
        return coeffs, drift_t_mean, drift_t_range  # type: ignore[return-value]
    return coeffs  # type: ignore[return-value]


def predict_tide(
    times_hours: NDArray[np.float64],
    coeffs: NDArray[np.float64],
    *,
    include_drift: bool = False,
    drift_t_mean: float | None = None,
    drift_t_range: float | None = None,
) -> NDArray[np.float64]:
    """Predict tidal signal from harmonic coefficients.

    Args:
        times_hours: Time values in hours relative to an epoch.
        coeffs: Coefficients from fit_tidal_harmonics.
        include_drift: If True, include drift columns using the provided
            normalization parameters from fitting.
        drift_t_mean: Mean time from fitting (required when include_drift=True).
        drift_t_range: Time range from fitting (required when include_drift=True).

    Returns:
        Predicted tidal (+ drift) signal.
    """
    mat = build_harmonic_matrix(
        times_hours, include_drift=include_drift,
        drift_t_mean=drift_t_mean, drift_t_range=drift_t_range)
    return mat @ coeffs


def detide(
    times_hours: NDArray[np.float64],
    values: NDArray[np.float64],
    fit_times_hours: NDArray[np.float64] | None = None,
    fit_values: NDArray[np.float64] | None = None,
    *,
    robust: bool = True,
    clean_calibration: bool = True,
) -> NDArray[np.float64]:
    """Remove tidal signal from water level data.

    If fit_times/fit_values are provided, the harmonic fit is computed on
    those (the 30-day rolling window) and applied to the target times/values.
    Otherwise, the fit is computed on the target data itself.

    When separate calibration data is provided and ``clean_calibration`` is
    True, the calibration values are cleaned (spike removal + detrend) before
    tidal fitting. When self-fitting (no separate calibration), cleaning is
    disabled - event-window spikes may be real tsunami signals.

    Args:
        times_hours: Times for the data to detide (hours from epoch).
        values: Water level values to detide.
        fit_times_hours: Optional separate times for fitting the harmonics.
        fit_values: Optional separate values for fitting the harmonics.
        robust: If True, use IRLS robust fitting.
        clean_calibration: If True, clean calibration data before fitting.

    Returns:
        Detided residual values.
    """
    if (fit_times_hours is None) != (fit_values is None):
        raise ValueError(
            "fit_times_hours and fit_values must both be provided or both be None"
        )
    if fit_times_hours is not None and fit_values is not None:
        # Separate calibration data - clean it if requested
        coeffs = fit_tidal_harmonics(
            fit_times_hours, fit_values,
            robust=robust, clean_input=clean_calibration,
        )
    else:
        # Self-fitting on event data - never clean (spikes may be real)
        coeffs = fit_tidal_harmonics(
            times_hours, values,
            robust=robust, clean_input=False,
        )

    assert isinstance(coeffs, np.ndarray)  # include_drift=False -> ndarray
    predicted = predict_tide(times_hours, coeffs)
    return values - predicted


# ---------------------------------------------------------------------------
# Bandpass filter pipeline
# ---------------------------------------------------------------------------

def design_bandpass_filter(
    sampling_rate_hz: float,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> NDArray[np.float64]:
    """Design a Butterworth bandpass filter in SOS form.

    Args:
        sampling_rate_hz: Sampling rate of the data in Hz.
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Second-order sections (SOS) representation of the filter.
    """
    nyquist = sampling_rate_hz / 2.0
    low_norm = low_hz / nyquist
    high_norm = high_hz / nyquist

    # Clamp normalized frequencies to valid range
    low_norm = max(low_norm, 1e-10)
    if high_norm >= 1.0:
        logger.warning(
            "Bandpass high cutoff (%.4f Hz) exceeds Nyquist (%.4f Hz); "
            "filter degraded (passband clamped to Nyquist). Consider higher sampling rate.",
            high_hz, nyquist,
        )
    high_norm = min(high_norm, 1.0 - 1e-10)
    low_norm = min(low_norm, high_norm - 1e-10)  # guard: low < high always

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")
    return sos  # type: ignore[no-any-return]


def apply_bandpass_filter(
    values: NDArray[np.float64],
    sampling_rate_hz: float,
    low_hz: float = BANDPASS_LOW_HZ,
    high_hz: float = BANDPASS_HIGH_HZ,
    order: int = BUTTERWORTH_ORDER,
) -> NDArray[np.float64]:
    """Apply causal Butterworth bandpass filter.

    Uses scipy.signal.sosfilt for forward-only (causal) filtering with
    steady-state initial conditions from sosfilt_zi.  This ensures that
    the filtered value at sample t depends only on samples 0...t, which is
    required for real-time processing where future data is unavailable.

    Args:
        values: Input signal (detided residual).
        sampling_rate_hz: Sampling rate in Hz.
        low_hz: Low cutoff frequency in Hz.
        high_hz: High cutoff frequency in Hz.
        order: Filter order.

    Returns:
        Bandpass-filtered signal.
    """
    min_samples = 2 * order + 1
    if len(values) <= min_samples:
        logger.warning(
            "Insufficient samples for bandpass filter (%d < %d), returning zeros",
            len(values), min_samples,
        )
        return np.zeros_like(values)

    sos = design_bandpass_filter(sampling_rate_hz, low_hz, high_hz, order)
    # Steady-state initial conditions at the DC level of the input signal.
    # This prevents startup transients that would otherwise produce spurious
    # high-frequency energy at the beginning of the filtered output.
    zi = sosfilt_zi(sos) * values[0]
    filtered, _ = sosfilt(sos, values, zi=zi)
    return filtered  # type: ignore[no-any-return]


def detide_and_filter(
    times_hours: NDArray[np.float64],
    values: NDArray[np.float64],
    sampling_rate_hz: float,
    fit_times_hours: NDArray[np.float64] | None = None,
    fit_values: NDArray[np.float64] | None = None,
    *,
    robust: bool = True,
    clean_calibration: bool = True,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Full pipeline: detide then bandpass filter.

    Precondition: For accurate tidal constituent separation, ``fit_times_hours``
    /``fit_values`` (or ``times_hours``/``values`` when no separate fit window
    is provided) should ideally span >=14.8 days.  This covers one spring-neap
    cycle, which is the minimum needed to separate M2 (12.42 h) from S2
    (12.00 h) and resolve the dominant fortnightly beat frequency.  Shorter
    windows yield poorly constrained harmonic coefficients that leave tidal
    residuals in the detided signal, which can alias into the tsunami band
    and produce inflated anomaly scores.  30 days is the recommended minimum
    window.  Per the Rayleigh criterion, 30 days cleanly resolves 6 of the 8
    standard constituents (M2, S2, N2, K1, O1, Q1); the P1/K2 pair requires
    ~183 days to separate from K1/S2 respectively.  In deep-ocean BPR data,
    P1 and K2 have small amplitudes, so the practical impact on detiding
    quality is minor.

    Args:
        times_hours: Times for the target data (hours from epoch).
        values: Water level observations.
        sampling_rate_hz: Sampling rate in Hz.
        fit_times_hours: Optional 30-day window times for tidal fit.
        fit_values: Optional 30-day window values for tidal fit.
        robust: If True, use IRLS robust tidal fitting.
        clean_calibration: If True, clean calibration data before fitting.

    Returns:
        Tuple of (detided_residual, filtered_signal).

    Raises:
        ValueError: If any input array contains NaN or Inf values.
    """
    if len(values) == 0:
        logger.warning("detide_and_filter called with empty values array")
        empty = np.array([], dtype=np.float64)
        return empty, empty
    _validate_finite(values, "values")
    _validate_finite(times_hours, "times_hours")
    if fit_values is not None:
        _validate_finite(fit_values, "fit_values")
    if fit_times_hours is not None:
        _validate_finite(fit_times_hours, "fit_times_hours")
    residual = detide(
        times_hours, values, fit_times_hours, fit_values,
        robust=robust, clean_calibration=clean_calibration,
    )
    filtered = apply_bandpass_filter(residual, sampling_rate_hz)
    return residual, filtered


# ---------------------------------------------------------------------------
# Wavelet energy decomposition
# ---------------------------------------------------------------------------

def compute_wavelet_energy(
    signal: NDArray[np.float64],
    sampling_interval_sec: float,
) -> float:
    """Compute tsunami-band wavelet energy using db4 decomposition.

    Uses Daubechies-4 wavelet. Decomposition level is chosen based on
    sampling interval: level 6 for 1-min data, level 8 for 15-sec data.

    The tsunami energy is the sum of squared detail coefficients at levels
    corresponding to the 5-120 minute band.

    Args:
        signal: Input signal (detided/filtered residual).
        sampling_interval_sec: Sampling interval in seconds.

    Returns:
        Tsunami-band wavelet energy (sum of squared detail coefficients).
    """
    if sampling_interval_sec <= 0:
        raise ValueError(
            f"sampling_interval_sec must be positive, got {sampling_interval_sec}"
        )
    min_wavelet_samples = pywt.Wavelet(WAVELET_FAMILY).dec_len  # 8 for db4
    if len(signal) < min_wavelet_samples:
        return 0.0
    _validate_finite(signal, "signal")

    # Select decomposition level based on sampling interval
    if sampling_interval_sec <= 15:
        level = WAVELET_LEVEL_15SEC
    else:
        level = WAVELET_LEVEL_1MIN

    # Ensure the signal is long enough for the requested decomposition level
    max_level = pywt.dwt_max_level(len(signal), pywt.Wavelet(WAVELET_FAMILY).dec_len)
    if max_level < level:
        level = max(1, max_level)

    coeffs = pywt.wavedec(signal, WAVELET_FAMILY, level=level)

    # Compute energy from detail coefficients (skip approximation coeffs[0]).
    # pywt.wavedec returns [cA_n, cD_n, cD_{n-1}, ..., cD_2, cD_1]:
    #   coeffs[1] = cD_n (coarsest detail), coeffs[-1] = cD_1 (finest).
    # DWT level j captures periods [2^j * dt, 2^(j+1) * dt].
    # Only include levels whose band overlaps the tsunami period range
    # (TSUNAMI_PERIOD_LOW_SEC to TSUNAMI_PERIOD_HIGH_SEC).
    # For 60s data (levels 1-6): levels 2-6 in-band; level 1 excluded.
    # For 15s data (levels 1-8): levels 4-8 in-band; levels 1-3 excluded.
    n_detail = len(coeffs) - 1  # number of detail levels
    energy = 0.0
    for idx, detail in enumerate(coeffs[1:]):
        # Map array position to actual DWT level (coarsest-first ordering)
        dwt_level = n_detail - idx
        level_low_period = (2**dwt_level) * sampling_interval_sec
        level_high_period = (2 ** (dwt_level + 1)) * sampling_interval_sec
        # Include level if its band overlaps the tsunami period range
        if (
            level_low_period < TSUNAMI_PERIOD_HIGH_SEC
            and level_high_period > TSUNAMI_PERIOD_LOW_SEC
        ):
            energy += float(np.sum(np.square(detail)))

    return energy


def compute_wavelet_score(
    signal: NDArray[np.float64],
    sampling_interval_sec: float,
    baseline_energy: float,
) -> float:
    """Compute a normalized wavelet anomaly score [0, 1].

    Computes the tsunami-band wavelet energy and maps it to [0, 1] using a
    hyperbolic transform: score = 1 - 1/ratio, where ratio = current/baseline.
    Returns 0.0 when ratio <= 1.0 (no excess energy). The score approaches 1.0
    asymptotically as energy grows well above baseline.

    Known limitation, deliberately unfixed. ``compute_wavelet_energy`` returns
    a sum of squared detail coefficients, an extensive quantity that grows with
    record length at constant signal power. This ratio carries no length term,
    and the calibration window is normally much longer than the scored event
    window, so the ratio tracks the length difference more than the signal.
    Chile station 51407 is the clearest case: the event window holds 5.1x the
    calibration window's energy per sample, but the calibration record is 102.7x
    longer, so the raw ratio is 0.050 and the score is exactly 0.0. Every
    filter_degraded station in results/ scores 0.0 here for the same reason.

    Do not "fix" this by dividing through by sample count on its own. Measured
    over the same 414 quiet-period windows as
    results/false_positive_evaluation.json, a power-density ratio moves the
    false-trigger counts from 1/1/0 at T1/T2/T3 to 2/1/1: it doubles the false
    T1 rate (0.29 to 0.57 per station-month) and introduces a T3-level false
    escalation at station 46408, which rises from 0.802 to 0.997. T1, T2 and
    T3 were chosen against the current suppressed scale, so the dimensional
    correction and a threshold recalibration have to land together, with
    results/ and the paper regenerated.

    Args:
        signal: Input signal.
        sampling_interval_sec: Sampling interval in seconds.
        baseline_energy: Baseline energy from non-event data.

    Returns:
        Normalized wavelet score in [0, 1].
    """
    if baseline_energy <= 0.0:
        return 0.0

    current_energy = compute_wavelet_energy(signal, sampling_interval_sec)
    ratio = current_energy / baseline_energy

    # Map ratio to [0, 1] - ratio > 1 indicates anomalous energy
    # Hyperbolic mapping: score = 1 - 1/ratio (approaches 1.0 asymptotically)
    if ratio <= 1.0:
        return 0.0

    score = 1.0 - 1.0 / ratio
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Bayesian Online Changepoint Detection (BOCPD)
# ---------------------------------------------------------------------------

@dataclass
class BOCPDState:
    """State for Bayesian Online Changepoint Detection.

    Uses a constant hazard function and Gaussian Normal-Gamma conjugate prior.

    Normal-Gamma parameterization: N(mu | mu0, 1/(kappa0*tau)), tau ~ Gamma(alpha0, beta0)
    where tau is the precision (1/sigma^2) of the observation noise.

    Note: ``prior_precision`` here refers to kappa0 (the pseudo-observation count /
    belief strength on the prior mean), NOT the Gaussian noise precision 1/sigma^2.
    Despite the name overlap in some Bayesian texts, these are distinct concepts.
    The conjugate update increments kappa by 1 per observation.
    """

    hazard_lambda: float = BOCPD_HAZARD_LAMBDA
    prior_mean: float = 0.0
    prior_precision: float = 1.0  # kappa0: pseudo-observation count (see class docstring)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    # Run-length distribution (log probabilities)
    run_length_log_probs: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0], dtype=np.float64)
    )
    # Sufficient statistics per run length - initialized from prior in __post_init__
    means: NDArray[np.float64] = field(
        default_factory=lambda: np.array([0.0], dtype=np.float64)
    )
    precisions: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0], dtype=np.float64)
    )
    alphas: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0], dtype=np.float64)
    )
    betas: NDArray[np.float64] = field(
        default_factory=lambda: np.array([1.0], dtype=np.float64)
    )
    # Predictive log-evidence from the most recent bocpd_update() call.
    # This is log P(x_t | x_{1:t-1}) - the marginal predictive probability
    # of the latest observation.  Used by compute_bocpd_score() for
    # changepoint scoring via predictive surprise.
    last_log_evidence: float = 0.0

    def __post_init__(self) -> None:
        """Reinitialize sufficient-statistic arrays from the prior parameters.

        The field default_factory lambdas cannot reference instance attributes,
        so they hardcode [0.0] / [1.0]. __post_init__ replaces them with arrays
        consistent with the actual prior values passed at construction time.
        """
        if not (0.0 < self.hazard_lambda < 1.0):
            raise ValueError(
                f"hazard_lambda must be in (0, 1), got {self.hazard_lambda}"
            )
        self.means = np.array([self.prior_mean], dtype=np.float64)
        self.precisions = np.array([self.prior_precision], dtype=np.float64)
        self.alphas = np.array([self.prior_alpha], dtype=np.float64)
        self.betas = np.array([self.prior_beta], dtype=np.float64)


def bocpd_update(state: BOCPDState, observation: float) -> float:
    """Update BOCPD state with a new observation and return changepoint probability.

    Implements the BOCPD algorithm (Adams & MacKay 2007) with:
    - Constant hazard function H(tau) = hazard_lambda: the conditional
      probability of a changepoint at time t given the previous run length
      was tau samples is hazard_lambda for all tau >= 0.  This is a transition
      probability, not the unconditional probability of being at run length 0.
      Run lengths follow a Geometric(hazard_lambda) prior; the expected run
      length between changepoints is 1/hazard_lambda samples.
    - Gaussian observation model with Normal-Gamma conjugate prior

    Args:
        state: Current BOCPD state (modified in place).
        observation: New observation value.

    Returns:
        Changepoint probability P(r_t = 0 | x_{1:t}), in [0, 1].

    .. note:: Under constant hazard H(tau) = lambda for all tau, the posterior
       P(r_t = 0) equals lambda regardless of the data.  Both the changepoint
       and growth paths share the same run-length-specific predictive
       likelihoods, so these cancel in the normalization.  The useful
       data-dependent signal is the predictive log-evidence
       (``state.last_log_evidence``), which measures how "surprising" each
       observation is under the learned model.  See ``compute_bocpd_score``.
    """
    x = observation
    n_rl = len(state.run_length_log_probs)

    # Step 1: Predictive probabilities under each run length
    # Student-t predictive distribution for Normal-Gamma
    pred_means = state.means
    pred_vars = (
        state.betas * (state.precisions + 1.0) / (state.alphas * state.precisions)
    )
    pred_vars = np.maximum(pred_vars, 1e-300)  # numerical safety
    dof = 2.0 * state.alphas

    # Log-probability of observation under Student-t
    z = (x - pred_means) ** 2 / pred_vars
    log_pred = (
        gammaln((dof + 1.0) / 2.0)
        - gammaln(dof / 2.0)
        - 0.5 * np.log(dof * math.pi * pred_vars)
        - (dof + 1.0) / 2.0 * np.log1p(z / dof)
    )

    # Step 2: Growth probabilities (run length grows)
    log_h = math.log(state.hazard_lambda)
    log_1_minus_h = math.log(1.0 - state.hazard_lambda)

    growth_log_probs = state.run_length_log_probs + log_pred + log_1_minus_h

    # Step 3: Changepoint probability (run length resets to 0)
    cp_log_prob = np.logaddexp.reduce(
        state.run_length_log_probs + log_pred + log_h
    )

    # Step 4: Combine and normalize
    new_log_probs = np.empty(n_rl + 1, dtype=np.float64)
    new_log_probs[0] = cp_log_prob
    new_log_probs[1:] = growth_log_probs

    # Normalize in log space
    log_evidence = np.logaddexp.reduce(new_log_probs)
    new_log_probs -= log_evidence

    # Store predictive log-evidence BEFORE truncation, which is the true
    # marginal log-likelihood log P(x_t | x_{1:t-1}).  The post-truncation
    # renormalization constant (computed below) is merely log(1 - mass_dropped)
    # and must NOT overwrite this value.
    pre_truncation_log_evidence = float(log_evidence)

    # Step 5: Update sufficient statistics
    new_means = np.empty(n_rl + 1, dtype=np.float64)
    new_precisions = np.empty(n_rl + 1, dtype=np.float64)
    new_alphas = np.empty(n_rl + 1, dtype=np.float64)
    new_betas = np.empty(n_rl + 1, dtype=np.float64)

    # Reset stats for run length 0 (changepoint)
    new_means[0] = state.prior_mean
    new_precisions[0] = state.prior_precision
    new_alphas[0] = state.prior_alpha
    new_betas[0] = state.prior_beta

    # Update stats for growing run lengths
    old_prec = state.precisions
    new_prec = old_prec + 1.0
    new_means[1:] = (old_prec * state.means + x) / new_prec
    new_precisions[1:] = new_prec
    new_alphas[1:] = state.alphas + 0.5
    new_betas[1:] = (
        state.betas
        + 0.5 * old_prec * (x - state.means) ** 2 / new_prec
    )

    # Truncate very small run lengths for efficiency (keep top entries).
    # Always retain index 0 (the changepoint / run-length-reset entry)
    # so the algorithm can correctly reset sufficient statistics on the
    # next observation even if the current changepoint probability is low.
    if len(new_log_probs) > BOCPD_MAX_RUN_LENGTHS:
        keep = np.argpartition(new_log_probs, -BOCPD_MAX_RUN_LENGTHS)[-BOCPD_MAX_RUN_LENGTHS:]
        # Ensure index 0 (changepoint entry) is always retained.
        # keep[i] holds an array INDEX (into new_log_probs et al.), not a position.
        # np.argmin returns the POSITION in keep[] with the lowest probability;
        # we then overwrite that position's INDEX value with 0.
        if 0 not in keep:
            min_pos = int(np.argmin(new_log_probs[keep]))
            keep[min_pos] = 0
        keep.sort()
        new_log_probs = new_log_probs[keep]
        new_means = new_means[keep]
        new_precisions = new_precisions[keep]
        new_alphas = new_alphas[keep]
        new_betas = new_betas[keep]
        # Renormalize after truncation
        log_evidence = np.logaddexp.reduce(new_log_probs)
        new_log_probs -= log_evidence

    # Store updated state
    state.run_length_log_probs = new_log_probs
    state.means = new_means
    state.precisions = new_precisions
    state.alphas = new_alphas
    state.betas = new_betas

    # Store predictive log-evidence for surprise-based scoring
    state.last_log_evidence = pre_truncation_log_evidence

    # Return changepoint probability (probability mass at run length 0)
    changepoint_prob = float(np.exp(new_log_probs[0]))
    return changepoint_prob


def compute_bocpd_score(
    signal: NDArray[np.float64],
    prior_precision: float = 1.0,
) -> float:
    """Run BOCPD over a signal window and return a changepoint score.

    Uses **predictive surprise** as the detection signal.  Under constant
    hazard, the posterior P(r_t = 0) is uninformative (always equals the
    hazard rate lambda regardless of data).  Instead, we use the predictive
    log-evidence log P(x_t | x_{1:t-1}), which drops sharply when an
    observation is inconsistent with the model learned from recent data -
    i.e., at a changepoint.

    The score is the maximum standardized surprise (z-score relative to a
    robust baseline) mapped to [0, 1] via a saturating function.

    State is initialized fresh from prior parameters on every call - there
    is no memory of previous calls.  This is intentional: deterministic
    replay requires that identical inputs always produce identical scores.

    Args:
        signal: Input signal (detided, pre-bandpass residual). Using the
            broadband residual rather than the bandpass-filtered signal
            allows BOCPD to detect the step-change onset of a tsunami wave
            before it fully enters the tsunami frequency band.
        prior_precision: kappa0 pseudo-observation count that controls prior
            belief strength on the mean.  Calibrated from baseline data;
            higher values make the detector more resistant to transients.

    Returns:
        Changepoint score in [0, 1].  0 = no changepoint detected,
        higher = stronger evidence of a distributional shift.
    """
    _BURN_IN = 10  # skip initial samples while posterior is prior-dominated

    if len(signal) < _BURN_IN + 5:
        return 0.0
    _validate_finite(signal, "signal")

    state = BOCPDState(
        prior_precision=prior_precision,
    )

    log_evidences: list[float] = []
    for val in signal:
        bocpd_update(state, float(val))
        log_evidences.append(state.last_log_evidence)

    # Predictive surprise = negative log-evidence (higher = more surprising)
    surprises = np.array([-le for le in log_evidences[_BURN_IN:]])

    if len(surprises) < 3:
        return 0.0

    # Compute baseline statistics from samples BEFORE the peak surprise.
    # This prevents post-changepoint samples from inflating the MAD,
    # which would suppress detection at moderate SNR (the "dead zone"
    # where max_z stays below the Gaussian-max correction).
    peak_idx = int(np.argmax(surprises))

    if peak_idx >= 5:
        baseline = surprises[:peak_idx]
    else:
        # Peak near start (typical for stationary data where the
        # learning transient creates the highest surprise); use all.
        baseline = surprises

    median_s = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median_s)))

    if mad < 1e-12:
        # Near-constant surprise - fall back to std-based scaling
        std_s = float(np.std(baseline))
        if std_s < 1e-12:
            return 0.0
        # For Gaussian data, MAD ~ 0.6745 * std
        mad = std_s * 0.6745
        if mad < 1e-12:
            return 0.0

    # Maximum standardized surprise relative to pre-event baseline
    max_z = float(np.max((surprises - median_s) / mad))

    if max_z <= 0.0:
        return 0.0

    # Subtract expected Gaussian maximum to avoid false scores on pure noise.
    # For n i.i.d. Gaussian samples, E[max|z|] ~ sqrt(2*ln(n)) / 0.6745
    # (in MAD units).  Only excess above this level indicates a changepoint.
    n = len(surprises)
    expected_gaussian_max = math.sqrt(2.0 * math.log(max(n, 2))) / 0.6745
    excess_z = max_z - expected_gaussian_max

    if excess_z <= 0.0:
        return 0.0

    # Saturating map to [0, 1]:  excess=5 -> 0.5, excess=15 -> 0.75
    score = excess_z / (excess_z + 5.0)
    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Isolation Forest scoring
# ---------------------------------------------------------------------------

@dataclass
class IsolationForestModel:
    """Wrapper around sklearn IsolationForest with deterministic seeding."""

    n_estimators: int = IFOREST_N_ESTIMATORS
    contamination: float = IFOREST_CONTAMINATION
    random_seed: int = IFOREST_RANDOM_SEED
    _model: object = field(default=None, repr=False)
    _is_fitted: bool = field(default=False, repr=False)

    def fit(self, features: NDArray[np.float64]) -> None:
        """Fit the Isolation Forest on training data.

        Args:
            features: Training feature matrix (n_samples, n_features).
                Expected features: [5-min energy, 15-min energy,
                rate-of-change, spatial coherence].
        """
        from sklearn.ensemble import IsolationForest

        self._model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_seed,
        )
        self._model.fit(features)  # type: ignore[attr-defined]
        self._is_fitted = True

    def score(self, features: NDArray[np.float64]) -> float:
        """Score a single sample. Returns anomaly score in [0, 1].

        sklearn returns decision_function values where more negative = more
        anomalous.  We map to [0, 1] where 1 = most anomalous.

        Args:
            features: Feature vector of shape (1, n_features) or (n_features,).

        Returns:
            Anomaly score in [0, 1].
        """
        if not self._is_fitted or self._model is None:
            return 0.0

        if features.ndim == 1:
            features = features.reshape(1, -1)

        # decision_function: negative = anomalous, positive = normal.
        # The raw output magnitude is dataset-dependent (sklearn normalizes
        # internally but the scale varies with contamination and tree depth).
        raw_score = float(self._model.decision_function(features)[0])  # type: ignore[attr-defined]

        # Map to [0, 1] via sigmoid: 1/(1+exp(k*raw)).  With k=5, the
        # inflection is at raw=0 (boundary score), giving 0.5.  Positive
        # (normal) raw scores approach 0; negative (anomalous) approach 1.
        score = 1.0 / (1.0 + math.exp(IFOREST_SIGMOID_STEEPNESS * raw_score))
        return max(0.0, min(1.0, score))

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def compute_rolling_energy(
    signal: NDArray[np.float64],
    window_samples: int,
) -> float:
    """Compute rolling energy (mean squared value) over a window.

    Args:
        signal: Input signal.
        window_samples: Number of samples in the rolling window.

    Returns:
        Energy value (mean of squared values in the window).
    """
    if len(signal) == 0:
        return 0.0
    window = signal[-window_samples:] if len(signal) >= window_samples else signal
    return float(np.mean(np.square(window)))


def compute_rate_of_change(signal: NDArray[np.float64]) -> float:
    """Compute the maximum absolute rate of change in the signal.

    Args:
        signal: Input signal.

    Returns:
        Maximum absolute first difference.
    """
    if len(signal) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(signal))))


def build_iforest_features(
    signal: NDArray[np.float64],
    sampling_interval_sec: float,
    spatial_coherence: float,
) -> NDArray[np.float64]:
    """Build the 4-feature vector for Isolation Forest scoring.

    Features: [5-min energy, 15-min energy, rate-of-change, spatial coherence]

    Args:
        signal: Input signal (detided/filtered).
        sampling_interval_sec: Sampling interval in seconds.
        spatial_coherence: Spatial coherence score [0, 1].

    Returns:
        Feature vector of shape (4,).
    """
    if sampling_interval_sec <= 0:
        raise ValueError(
            f"sampling_interval_sec must be positive, got {sampling_interval_sec}"
        )
    samples_5min = max(1, int(300.0 / sampling_interval_sec))
    samples_15min = max(1, int(900.0 / sampling_interval_sec))

    energy_5min = compute_rolling_energy(signal, samples_5min)
    energy_15min = compute_rolling_energy(signal, samples_15min)
    roc = compute_rate_of_change(signal)

    return np.array([energy_5min, energy_15min, roc, spatial_coherence], dtype=np.float64)


# ---------------------------------------------------------------------------
# Spatial coherence
# ---------------------------------------------------------------------------

@dataclass
class StationArrival:
    """Observed anomaly arrival at a station."""

    station_id: str
    arrival_time: datetime
    latitude: float
    longitude: float


@dataclass(frozen=True)
class SpatialConfirmationDetail:
    """Per-station spatial coherence check detail."""

    station_id: str
    distance_km: float
    expected_travel_sec: float
    actual_delta_sec: float
    window_low_sec: float
    window_high_sec: float
    confirmed: bool


@dataclass
class SpatialCoherenceResult:
    """Result of spatial coherence analysis."""

    confirmed: bool
    confirming_stations: int
    confirmations: list[SpatialConfirmationDetail]


def check_spatial_coherence(
    origin: StationArrival,
    other_arrivals: list[StationArrival],
    window_fraction: float = SPATIAL_CONFIRMATION_WINDOW_FRACTION,
    min_confirming: int = SPATIAL_MIN_CONFIRMING_STATIONS,
) -> SpatialCoherenceResult:
    """Check if anomaly arrivals at multiple stations are spatially coherent.

    Computes inter-station tsunami travel time between the first-detecting
    station (``origin``) and each subsequent station.  For each pair, checks
    whether the observed detection delay falls within +/-20% of the expected
    travel time at deep-ocean tsunami speed (~0.2 km/s).

    This is a station-to-station coherence check, not a source-to-station
    check.  It does not require knowledge of the earthquake epicenter.

    That independence has a geometric cost.  Using
    d_ij/c as the expected delay assumes the source lies on the extension of
    the station baseline behind the origin station.  For any other source
    position the true delay is smaller: by the triangle inequality the
    difference in source distances is at most d_ij, so d_ij/c is the maximum
    possible delay for a common source, not the typical one.  For a distant
    source the delay is about (d_ij/c)cos(theta), where theta is the angle
    between the baseline and the source direction, so the lower bound of the
    window accepts only pairs within roughly 37 degrees of radial alignment
    and the upper half of the window is not reachable at the assumed speed.
    Treat confirmation as sufficient evidence of coherence, not necessary:
    genuinely coherent arrivals at pairs with large azimuthal offset from the
    source will not confirm.

    Requires 2+ confirming stations for spatial_confirmed = True.

    Args:
        origin: The station where the anomaly was first detected.
        other_arrivals: Arrivals at other stations.
        window_fraction: Fractional tolerance for the confirmation window.
        min_confirming: Minimum number of confirming stations.

    Returns:
        SpatialCoherenceResult with confirmation status and details.
    """
    confirmations: list[SpatialConfirmationDetail] = []
    confirming_count = 0

    for arrival in other_arrivals:
        dist_km = haversine_km(
            origin.latitude, origin.longitude,
            arrival.latitude, arrival.longitude,
        )
        expected_travel_sec = compute_travel_time_sec(dist_km)
        actual_delta_sec = (arrival.arrival_time - origin.arrival_time).total_seconds()

        # Floor at 5s to handle colocated/nearby stations where
        # expected_travel_sec ~ 0 would produce a degenerate window. Keep it
        # small: a 60s floor imposes a [48s, 72s] window that rejects any
        # genuinely nearby pair whose expected travel is under 48s.
        effective_travel = max(expected_travel_sec, 5.0)
        window_low = effective_travel * (1.0 - window_fraction)
        window_high = effective_travel * (1.0 + window_fraction)

        confirmed = window_low <= actual_delta_sec <= window_high
        if confirmed:
            confirming_count += 1

        confirmations.append(SpatialConfirmationDetail(
            station_id=arrival.station_id,
            distance_km=round(dist_km, 1),
            expected_travel_sec=round(expected_travel_sec, 1),
            actual_delta_sec=round(actual_delta_sec, 1),
            window_low_sec=round(window_low, 1),
            window_high_sec=round(window_high, 1),
            confirmed=confirmed,
        ))

    return SpatialCoherenceResult(
        confirmed=confirming_count >= min_confirming,
        confirming_stations=confirming_count,
        confirmations=confirmations,
    )


def compute_spatial_coherence_score(result: SpatialCoherenceResult) -> float:
    """Convert spatial coherence result to a score in [0, 1].

    Args:
        result: SpatialCoherenceResult from check_spatial_coherence.

    Returns:
        Score in [0, 1]. 1.0 if spatially confirmed, fraction otherwise.
    """
    if not result.confirmations:
        return 0.0
    return min(1.0, result.confirming_stations / SPATIAL_MIN_CONFIRMING_STATIONS)


# ---------------------------------------------------------------------------
# Threshold-based detection score
# ---------------------------------------------------------------------------

def compute_threshold_score(
    filtered_signal: NDArray[np.float64],
    threshold_m: float,
    seismic_context_quiet: bool = False,
) -> float:
    """Compute threshold-based anomaly score.

    Uses the maximum absolute value in the filtered signal compared to a
    threshold. When in a seismically quiet period, the threshold is
    multiplied by the quiet factor (1.3x), requiring larger signals.

    Args:
        filtered_signal: Bandpass-filtered detided residual.
        threshold_m: Detection threshold in meters.
        seismic_context_quiet: True if no M>=6.5 event in 90 minutes.

    Returns:
        Threshold score in [0, 1].
    """
    if len(filtered_signal) == 0 or threshold_m <= 0.0:
        return 0.0
    # Fail-loud like the wavelet and BOCPD detectors: a non-finite filtered
    # signal would otherwise map through np.max to a false maximum score
    # (min(1.0, nan - 1.0) == 1.0), the most dangerous output in a fail-safe
    # system. Upstream detide_and_filter validates its inputs, so this guards
    # against reordering or standalone calls rather than a currently reachable
    # path.
    _validate_finite(filtered_signal, "filtered_signal")

    effective_threshold = threshold_m
    if seismic_context_quiet:
        effective_threshold *= SEISMIC_QUIET_THRESHOLD_FACTOR

    max_amplitude = float(np.max(np.abs(filtered_signal)))
    ratio = max_amplitude / effective_threshold

    # Map to [0, 1]: 0 at/below threshold, linear ramp above.
    # Below-threshold signals contribute nothing - clean separation
    # prevents false ensemble inflation from sub-threshold amplitudes.
    if ratio <= 1.0:
        return 0.0
    return min(1.0, ratio - 1.0)  # linear ramp: 0 at threshold, 1.0 at 2x threshold


# ---------------------------------------------------------------------------
# Ensemble fusion
# ---------------------------------------------------------------------------

@dataclass
class AnomalyScoreComponents:
    """Component-level anomaly scores for logging and transparency."""

    threshold_score: float
    wavelet_score: float
    bocpd_score: float
    statistical_score: float  # combined wavelet + BOCPD
    ml_score: float | None  # None if Isolation Forest unavailable
    spatial_coherence_score: float
    seismic_context_quiet: bool
    ensemble_score: float
    rayleigh_wave_suspect: bool | None = None
    """True when the spike timing matches a seismic Rayleigh-wave arrival.
    Set by AnomalyAgent.process_station_data via rayleigh_arrival_suspect
    (this module) when station coordinates are available and the detector
    fired. None means those prerequisites were unavailable and the check
    was never evaluated. (verification_checks.check_rayleigh_wave_suspect
    is the separate VerificationInput-based check used by the verification
    agent.)"""
    filter_degraded: bool = False
    """True when the bandpass filter is degraded due to a Nyquist violation.

    Occurs when sampling_interval_sec >= 1/(2*BANDPASS_HIGH_HZ) (~150 s),
    i.e., the Nyquist frequency falls below the bandpass upper cutoff.
    design_bandpass_filter then clamps the upper edge to Nyquist, so at the
    standard DART 15-min rate the passband is about 30 to 120 min instead of
    5 to 120 min. The clamped filter still passes that band at close to full
    gain, so the threshold score is not suppressed by degradation, and the
    ensemble at a degraded station is not necessarily BOCPD-driven
    (results/chile_detection.json station 54401 scores threshold 0.221 with
    BOCPD at zero). Read the scores as unreliable in either direction rather
    than as absent: the clamped output also carries shorter-period energy
    folded in by the 15-min sampling. Note that every degraded station in the
    committed artifacts happens to score wavelet 0.0, for the separate reason
    that compute_wavelet_score compares energies over unequal windows.
    Standard DART T=1 data (15-min intervals) always triggers this condition.
    """
    detide_fit_source: str = ""
    """What the harmonic detide was fit on: "separate calibration series"
    when a dedicated fit window was supplied to process_station_data,
    "event window" otherwise. Empty when provenance was not recorded."""
    detide_fit_span_minutes: float | None = None
    """Time span in minutes of the series the harmonic fit used."""
    detide_fit_samples: int | None = None
    """Sample count of the series the harmonic fit used."""


def compute_statistical_score(
    wavelet_score: float,
    bocpd_score: float,
) -> float:
    """Combine wavelet and BOCPD into a single statistical score.

    Uses max of the two - either detector firing is significant.

    Args:
        wavelet_score: Wavelet energy anomaly score [0, 1].
        bocpd_score: BOCPD changepoint score [0, 1].

    Returns:
        Combined statistical score in [0, 1].
    """
    return max(wavelet_score, bocpd_score)


def compute_ensemble_score(
    threshold_score: float,
    statistical_score: float,
    ml_score: float | None = None,
    *,
    ensemble_weights: tuple[float, float, float] | None = None,
) -> float:
    """Compute weighted ensemble anomaly score.

    Uses W_THRESHOLD, W_STATISTICAL, W_ML when ML is available.
    When ML is unavailable, renormalizes the threshold and statistical
    weights over the two remaining detectors.

    Args:
        threshold_score: Threshold-based score [0, 1].
        statistical_score: Statistical (wavelet/BOCPD) score [0, 1].
        ml_score: Isolation Forest score [0, 1], or None.
        ensemble_weights: Optional (w_threshold, w_statistical, w_ml)
            override.  When *None* the module-level defaults are used.

    Returns:
        Ensemble anomaly score in [0, 1].
    """
    w_thr, w_stat, w_ml_w = ensemble_weights or (
        W_THRESHOLD,
        W_STATISTICAL,
        W_ML,
    )
    if ml_score is not None:
        score = (
            w_thr * threshold_score
            + w_stat * statistical_score
            + w_ml_w * ml_score
        )
    else:
        no_ml_sum = w_thr + w_stat
        if no_ml_sum > 0.0:
            score = (
                (w_thr / no_ml_sum) * threshold_score
                + (w_stat / no_ml_sum) * statistical_score
            )
        else:
            score = 0.0

    return max(0.0, min(1.0, score))


def compute_full_anomaly_score(
    filtered_signal: NDArray[np.float64],
    detided_residual: NDArray[np.float64],
    sampling_interval_sec: float,
    threshold_m: float,
    baseline_wavelet_energy: float,
    bocpd_prior_precision: float = 1.0,
    iforest_model: IsolationForestModel | None = None,
    spatial_coherence_result: SpatialCoherenceResult | None = None,
    seismic_context_quiet: bool = False,
    ensemble_weights: tuple[float, float, float] | None = None,
) -> AnomalyScoreComponents:
    """Compute all anomaly score components and the ensemble score.

    This is the main entry point for scoring. It runs all detectors
    and produces the fused ensemble score.

    Signal routing rationale:
    - ``filtered_signal`` (bandpass, 5-120 min band) is used by the threshold
      and wavelet detectors because tsunamis are defined by oscillations in
      that period range.  Restricting to this band suppresses tidal residual
      noise and swell.
    - ``detided_residual`` (pre-bandpass, full frequency content) is used by
      BOCPD because the algorithm detects abrupt distributional shifts.  A
      tsunami onset manifests as a step change in the broadband residual
      before the full oscillatory waveform develops inside the pass-band.
      Using the unfiltered residual gives BOCPD a head start on early onset
      detection.

    Args:
        filtered_signal: Bandpass-filtered signal from the detide and filter stage.
        detided_residual: Detided residual (pre-bandpass) for BOCPD.
        sampling_interval_sec: Sampling interval in seconds.
        threshold_m: Detection threshold in meters.
        baseline_wavelet_energy: Baseline wavelet energy from non-event data.
        bocpd_prior_precision: BOCPD prior precision from baseline calibration.
        iforest_model: Optional fitted Isolation Forest model.
        spatial_coherence_result: Optional spatial coherence check result.
        seismic_context_quiet: True if no M>=6.5 event in last 90 minutes.
        ensemble_weights: Optional (w_threshold, w_statistical, w_ml)
            override for ablation studies.  *None* uses module defaults.

    Returns:
        AnomalyScoreComponents with all individual and ensemble scores.
    """
    if sampling_interval_sec <= 0:
        raise ValueError(
            f"sampling_interval_sec must be positive, got {sampling_interval_sec}"
        )
    # Check for bandpass filter degradation (Nyquist violation).
    # Nyquist frequency = 1 / (2 * sampling_interval).  The bandpass upper
    # cutoff must be below Nyquist, i.e. sampling_interval < 1/(2*HIGH_HZ).
    # When this condition is violated the filter clamps to Nyquist and
    # threshold_score / wavelet_score become unreliable.
    filter_degraded = sampling_interval_sec >= 1.0 / (2.0 * BANDPASS_HIGH_HZ)
    if filter_degraded:
        logger.warning(
            "Bandpass filter degraded: sampling interval %.0f s "
            "(Nyquist=%.4f Hz) cannot represent upper cutoff %.4f Hz. "
            "Threshold and wavelet scores are computed on the "
            "Nyquist-clamped filter output and are unreliable; interpret "
            "this station's scores conservatively.",
            sampling_interval_sec,
            1.0 / (2.0 * sampling_interval_sec),
            BANDPASS_HIGH_HZ,
        )

    # Threshold score
    t_score = compute_threshold_score(
        filtered_signal, threshold_m, seismic_context_quiet
    )

    # Wavelet score
    w_score = compute_wavelet_score(
        filtered_signal, sampling_interval_sec, baseline_wavelet_energy
    )

    # BOCPD score
    b_score = compute_bocpd_score(detided_residual, bocpd_prior_precision)

    # Combined statistical
    stat_score = compute_statistical_score(w_score, b_score)

    # Spatial coherence
    if spatial_coherence_result is not None:
        spatial_score = compute_spatial_coherence_score(spatial_coherence_result)
    else:
        spatial_score = 0.0

    # ML score (Isolation Forest)
    ml_score: float | None = None
    if iforest_model is not None and iforest_model.is_fitted:
        features = build_iforest_features(
            filtered_signal, sampling_interval_sec, spatial_score
        )
        ml_score = iforest_model.score(features)

    # Ensemble
    ensemble = compute_ensemble_score(
        t_score, stat_score, ml_score, ensemble_weights=ensemble_weights
    )

    return AnomalyScoreComponents(
        threshold_score=t_score,
        wavelet_score=w_score,
        bocpd_score=b_score,
        statistical_score=stat_score,
        ml_score=ml_score,
        spatial_coherence_score=spatial_score,
        seismic_context_quiet=seismic_context_quiet,
        ensemble_score=ensemble,
        filter_degraded=filter_degraded,
    )


# ---------------------------------------------------------------------------
# Seismic context helper
# ---------------------------------------------------------------------------

@dataclass
class SeismicEvent:
    """Minimal seismic event info for context adjustment."""

    event_id: str
    magnitude: float
    origin_time: datetime
    latitude: float
    longitude: float

    def __post_init__(self) -> None:
        if self.origin_time.tzinfo is None:
            raise ValueError("origin_time must be timezone-aware")


def is_seismically_quiet(
    recent_events: list[SeismicEvent],
    reference_time: datetime,
    magnitude_threshold: float = SEISMIC_QUIET_MAGNITUDE_THRESHOLD,
    window_minutes: float = SEISMIC_QUIET_WINDOW_MINUTES,
    *,
    fsm_monitoring: bool = False,
) -> bool:
    """Check if the seismic context is 'quiet' (no large events recently).

    A quiet period means no M>=6.5 event anywhere in the provided event
    list in the last 90 minutes. During quiet periods, the anomaly detection
    threshold is multiplied by 1.3x to reduce false alarms from non-tectonic
    sources. This function does not filter by location or depth - all events
    above the magnitude threshold count regardless of their position.

    When ``fsm_monitoring`` is *True*, the FSM has already entered MONITOR
    or a higher state due to a seismic trigger (which fires at M>=6.0 in
    tsunamigenic zones). In this case the function returns *False* (not
    quiet) unconditionally - the 1.3x threshold boost must not suppress
    detection of the very event the system is monitoring. This closes the
    M6.0-6.4 detection dead zone where the FSM triggers monitoring but
    the quiet-period heuristic simultaneously raises the detection bar.

    Args:
        recent_events: List of recent seismic events.
        reference_time: Current time for the check.
        magnitude_threshold: Minimum magnitude to break quiet (default 6.5).
        window_minutes: Lookback window in minutes (default 90).
        fsm_monitoring: True if the FSM is in MONITOR, INVESTIGATE, ASSESS,
            or ESCALATE state (i.e., actively tracking a seismic event).

    Returns:
        True if seismically quiet (no large events), False otherwise.
    """
    # When the FSM is actively monitoring, never report quiet - the 1.3x
    # threshold boost must not suppress detection during active monitoring.
    if fsm_monitoring:
        return False
    cutoff = reference_time - timedelta(minutes=window_minutes)

    for event in recent_events:
        if event.magnitude >= magnitude_threshold and event.origin_time >= cutoff:
            return False

    return True


# ---------------------------------------------------------------------------
# Rayleigh Wave False-Trigger Detection
# ---------------------------------------------------------------------------

# Mean Rayleigh wave group velocity for oceanic paths (km/s).
# Range 3.5-4.5 km/s spans continental vs. oceanic paths; 3.6 km/s is the
# correct oceanic default (Lay & Wallace, Modern Global Seismology, 1995, Ch. 4).
RAYLEIGH_GROUP_VELOCITY_KM_S = 3.6

# Maximum epicentral distance for plausible Rayleigh wave false trigger.
# Beyond ~3000 km, Rayleigh wave amplitude attenuates below the DART 30 mm
# event-mode threshold for all but the very largest earthquakes.
RAYLEIGH_MAX_DISTANCE_KM = 3000.0


def rayleigh_arrival_suspect(
    station_lat: float,
    station_lon: float,
    epicenter_lat: float,
    epicenter_lon: float,
    seismic_origin_utc: datetime,
    spike_utc: datetime,
    *,
    rayleigh_speed_km_s: float = RAYLEIGH_GROUP_VELOCITY_KM_S,
    tolerance: float = 0.20,
) -> bool:
    """Check whether a DART pressure spike is consistent with Rayleigh wave arrival.

    DART Bottom Pressure Recorders enter event mode when pressure deviation
    exceeds ~30 mm from the predicted value (Newton cubic extrapolation).
    After a large earthquake, seismic Rayleigh surface waves can cause
    transient seafloor pressure perturbations exceeding this threshold,
    producing a false event-mode trigger.

    This is an amplitude/timing mechanism, NOT spectral. Rayleigh waves
    have dominant periods of 10-50 seconds; the tsunami band is 5-120
    minutes (300-7200 seconds). These ranges largely do not overlap,
    though very long-period Rayleigh waves from great earthquakes
    (M9+) can extend to ~200-300 s, approaching the 300 s (5 min)
    low edge of the bandpass. The bandpass filter (5-120 min) attenuates
    the vast majority of Rayleigh spectral content.
    Detection must therefore use seismic context (timing correlation with
    epicenter distance).

    Critical caveat: Rayleigh wave timing correlation is not proof of
    false trigger - real tsunamis also follow large earthquakes at
    similar or later times. This check is always CONCERN, never FAIL.

    Args:
        station_lat: DART station latitude (degrees).
        station_lon: DART station longitude (degrees).
        epicenter_lat: Earthquake epicenter latitude (degrees).
        epicenter_lon: Earthquake epicenter longitude (degrees).
        seismic_origin_utc: Earthquake origin time (UTC).
        spike_utc: Time of DART pressure excursion (UTC).
        rayleigh_speed_km_s: Rayleigh wave group velocity (default 3.6 km/s).
        tolerance: Fractional tolerance on expected travel time (default +/-20%).

    Returns:
        True if the spike timing is consistent with Rayleigh wave arrival
        from the given epicenter.
    """
    # Actual travel time from earthquake to spike
    actual_travel_sec = (spike_utc - seismic_origin_utc).total_seconds()

    # Spike before earthquake origin = clock issue, not Rayleigh
    if actual_travel_sec <= 0:
        return False

    # Great-circle distance
    distance_km = haversine_km(station_lat, station_lon, epicenter_lat, epicenter_lon)

    # Beyond max distance, Rayleigh amplitude too attenuated for 30 mm threshold
    if distance_km > RAYLEIGH_MAX_DISTANCE_KM:
        return False

    # Expected Rayleigh wave travel time
    expected_travel_sec = distance_km / rayleigh_speed_km_s

    # Epicenter very close to station - timing window too tight for reliable check
    if expected_travel_sec < 10.0:
        return False

    # Check if actual travel time falls within +/-tolerance of expected
    relative_error = abs(actual_travel_sec - expected_travel_sec) / expected_travel_sec
    return relative_error <= tolerance
