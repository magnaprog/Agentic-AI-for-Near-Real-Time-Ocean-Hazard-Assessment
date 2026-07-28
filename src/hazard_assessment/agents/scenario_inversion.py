"""Scenario Inversion Service - NNLS waveform inversion, bootstrap, and coastal proxies.


Implements:
- NNLS-based waveform inversion, seismic-only magnitude scaling,
  scenario ranking, and moment magnitude calculation.
- Bootstrap station resampling for uncertainty estimation, ensemble
  spread classification, bootstrap-based scenario ranking.
- Coastal amplitude proxy generation from precomputed forecast factors.

All outputs are deterministic on replay (scipy.optimize.nnls uses the
Lawson-Hanson algorithm - no random initialization; bootstrap uses seeded RNG).
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from hazard_assessment.agents.scenario_data import (
    CoastalForecastFactors,
    GreensFunctionSet,
    UnitSource,
    UnitSourceDatabase,
    haversine_distance_km,
    select_unit_sources,
)
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    EnsembleSpread,
    RankedScenario,
)

logger = logging.getLogger(__name__)

# Threshold for treating NNLS weights as zero.
# scipy.optimize.nnls (Lawson-Hanson) explicitly sets inactive variables
# to zero, so exact zeros are expected. This threshold is a conservative
# safety margin for edge cases in ill-conditioned problems.
NNLS_ZERO_THRESHOLD: float = 1e-10


# ---------------------------------------------------------------------------
# Inversion result
# ---------------------------------------------------------------------------


@dataclass
class InversionResult:
    """Result of a single NNLS inversion.

    Note: elapsed_sec is wall-clock time and is NOT deterministic on replay.
    All other fields are deterministic given the same inputs.
    """

    source_ids: list[str]
    weights: NDArray[np.float64]  # NNLS solution (non-negative)
    residual_norm: float  # ||H*alpha - d||_2
    waveform_rmse_cm: float  # RMSE in cm across all station-timepoints
    mw_equivalent: float  # moment magnitude from summed slip (clamped >= 0.0)
    fit_stations: list[str]  # stations used in the fit
    inversion_window_sec: int  # time window length
    elapsed_sec: float  # wall-clock time for the solve


# ---------------------------------------------------------------------------
# Observation vector and Green's matrix construction
# ---------------------------------------------------------------------------


def build_observation_vector(
    station_waveforms: dict[str, NDArray[np.float64]],
    station_ids: list[str],
    n_timepoints: int | None = None,
) -> NDArray[np.float64]:
    """Stack observed detided waveforms into the d vector.

    d = [station1_t1, station1_t2, ..., station1_tN, station2_t1, ...]
    Shape: (n_stations * n_timepoints,)

    If n_timepoints is provided, truncates each waveform to that length.
    Raises ValueError if any waveform is shorter than n_timepoints,
    or if waveforms have inconsistent lengths when n_timepoints is None.
    """
    if not station_ids:
        raise ValueError("station_ids must be non-empty")

    parts: list[NDArray[np.float64]] = []
    target_len = n_timepoints

    for sid in station_ids:
        waveform = station_waveforms[sid]
        if target_len is not None:
            if len(waveform) < target_len:
                raise ValueError(
                    f"Station {sid} waveform has {len(waveform)} points, "
                    f"need at least {target_len}"
                )
            parts.append(waveform[:target_len])
        else:
            if parts and len(waveform) != len(parts[0]):
                raise ValueError(
                    f"Station waveform length mismatch: {sid} has "
                    f"{len(waveform)} points, expected {len(parts[0])}"
                )
            parts.append(waveform)

    return np.concatenate(parts)


def build_greens_matrix(
    greens: GreensFunctionSet,
    station_ids: list[str],
) -> NDArray[np.float64]:
    """Reshape Green's functions into the H matrix for NNLS.

    H shape: (n_stations * n_timepoints, n_sources)
    Each column is the concatenated waveform response of one unit source
    across all stations.

    Assumes GreensFunctionSet follows the ordering contract:
    waveforms[i, :, j] is the response at station_ids[i] from source_ids[j].
    """
    n_stations = len(station_ids)
    n_timepoints = greens.n_timepoints
    n_sources = len(greens.source_ids)

    # Build station index mapping for the requested ordering
    station_idx = {sid: i for i, sid in enumerate(greens.station_ids)}

    H = np.zeros((n_stations * n_timepoints, n_sources), dtype=np.float64)
    for i, sid in enumerate(station_ids):
        gi = station_idx[sid]
        H[i * n_timepoints : (i + 1) * n_timepoints, :] = greens.waveforms[gi, :, :]

    return H


# ---------------------------------------------------------------------------
# Moment magnitude calculation
# ---------------------------------------------------------------------------


def compute_mw_from_weights(
    weights: NDArray[np.float64],
    sources: list[UnitSource],
) -> float:
    """Compute moment magnitude from NNLS weights.

    Each weight represents slip (m). Seismic moment:
    M0 = sum(slip_i * area_i * rigidity_i)  [N*m]
    Mw = (2/3) * (log10(M0) - 9.1)

    Note on the 9.1 constant: Hanks & Kanamori (1979) used the CGS
    formula Mw = (2/3)*log10(M0_dyne_cm) - 10.7. A literal unit
    conversion to SI gives 9.05 (i.e. (2/3)*log10(M0_SI) - 6.033).
    The IASPEI (2005) standard uses 9.1 instead (i.e.
    (2/3)*log10(M0_SI) - 6.067), and that is the value used here.
    The 0.05 difference sits inside the parentheses, so it maps to
    (2/3)*0.05 = 0.033 in Mw: for the same moment the IASPEI form
    returns a value about 0.033 lower than the converted Hanks and
    Kanamori form. Either way the offset is negligible next to the
    inversion uncertainty of roughly +/-0.3 Mw.

    Returns 0.0 if all weights are zero (no slip).
    Clamps to max(0.0, mw) - very small non-zero M0 can produce
    negative Mw which would fail RankedScenario's ge=0.0 validator.
    """
    m0 = 0.0
    for w, src in zip(weights, sources, strict=True):
        m0 += float(w) * src.area_m2 * src.rigidity_pa

    if not math.isfinite(m0) or m0 <= 0.0:
        logger.warning(
            "NNLS produced zero total moment (all weights zero or non-finite); "
            "returning Mw=0.0 as sentinel - downstream verification should "
            "flag this via RMSE or magnitude-mismatch checks"
        )
        return 0.0

    mw = (2.0 / 3.0) * (math.log10(m0) - 9.1)
    return max(0.0, mw)


# ---------------------------------------------------------------------------
# NNLS solver
# ---------------------------------------------------------------------------


def solve_nnls(
    H: NDArray[np.float64],
    d: NDArray[np.float64],
    source_ids: list[str],
    station_ids: list[str],
    sources: list[UnitSource],
    inversion_window_sec: int,
) -> InversionResult:
    """Run scipy.optimize.nnls and compute fit metrics.

    Steps:
    1. scipy.optimize.nnls(H, d) -> (alpha, rnorm)
       where rnorm = ||H*alpha - d||_2
    2. RMSE_cm = (rnorm / sqrt(n_observations)) * 100
    3. Mw from slip via compute_mw_from_weights()

    scipy.optimize.nnls uses the Lawson-Hanson algorithm - fully
    deterministic (no random initialization), safe for replay.
    """
    t0 = time.monotonic()
    alpha, rnorm = nnls(H, d)
    elapsed = time.monotonic() - t0

    n_observations = len(d)
    rmse_cm = (rnorm / math.sqrt(n_observations)) * 100.0 if n_observations > 0 else 0.0

    mw = compute_mw_from_weights(alpha, sources)

    logger.info(
        "NNLS solve: %d sources, %d stations, rnorm=%.4f, RMSE=%.4f cm, "
        "Mw=%.2f, elapsed=%.3f s",
        len(source_ids),
        len(station_ids),
        rnorm,
        rmse_cm,
        mw,
        elapsed,
    )

    return InversionResult(
        source_ids=list(source_ids),
        weights=alpha,
        residual_norm=rnorm,
        waveform_rmse_cm=rmse_cm,
        mw_equivalent=mw,
        fit_stations=list(station_ids),
        inversion_window_sec=inversion_window_sec,
        elapsed_sec=elapsed,
    )


# ---------------------------------------------------------------------------
# Scenario ranking
# ---------------------------------------------------------------------------


def rank_scenarios(
    inversion: InversionResult,
) -> list[RankedScenario]:
    """Build the ranked scenario list from a single NNLS inversion result.

    Pre-bootstrap, this returns exactly ONE scenario: the full
    NNLS solution with non-zero-weight sources. If ALL weights are zero
    (rank-deficient system), includes all sources to satisfy the
    RankedScenario min_length=1 constraint.

    The bootstrap path produces multiple distinct solutions to rank.

    The single scenario gets posterior_weight = 1.0 (it is the only
    candidate). The bootstrap path replaces this with posterior weights.

    Returns: list containing one RankedScenario at rank=1.
    """
    nonzero_mask = inversion.weights > NNLS_ZERO_THRESHOLD
    if np.any(nonzero_mask):
        ids = [
            sid
            for sid, nz in zip(inversion.source_ids, nonzero_mask, strict=True)
            if nz
        ]
        wts = inversion.weights[nonzero_mask].tolist()
    else:
        # All weights zero - include all sources to satisfy min_length=1
        logger.warning(
            "rank_scenarios: NNLS produced all-zero weights "
            "(rank-deficient system, %d sources)",
            len(inversion.source_ids),
        )
        ids = list(inversion.source_ids)
        wts = inversion.weights.tolist()

    return [
        RankedScenario(
            unit_source_ids=ids,
            weights=wts,
            waveform_rmse_cm=inversion.waveform_rmse_cm,
            mw_equivalent=inversion.mw_equivalent,
            rank=1,
            posterior_weight=1.0,
        )
    ]


# ---------------------------------------------------------------------------
# Seismic-only mode
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeismicOnlyConfig:
    """Configuration for seismic-only preliminary estimate."""

    magnitude: float
    epicenter_lat: float
    epicenter_lon: float
    region: str
    # Empirical magnitude-to-slip scaling: log10(slip_m) = a * Mw + b
    # Defaults are approximate; production deployment should calibrate
    # against regional historical data.
    scaling_a: float = 0.5
    scaling_b: float = -3.0
    max_sources: int = 15
    max_distance_km: float = 500.0


SEISMIC_ONLY_LABEL = "Seismic-only estimate. No DART constraint. High uncertainty."


def compute_seismic_only_weights(
    sources: list[UnitSource],
    config: SeismicOnlyConfig,
) -> NDArray[np.float64]:
    """Scale unit source weights by seismic magnitude only (no waveform fit).

    Uses a simple magnitude-to-slip scaling relation:
        slip_m = 10^(a * Mw + b)

    Rupture length estimated from Wells & Coppersmith (1994):
        log10(SRL_km) = -3.22 + 0.69 * Mw

    Weight decays with distance from epicenter (Gaussian taper):
        weight_i = slip_m * exp(-d_i^2 / (2 * sigma^2))
        where sigma = SRL_km / 3  (heuristic: 3sigma ~ SRL, concentrating
        most slip within one rupture length of the epicenter)

    This is a crude first-pass estimate. The mandatory "high uncertainty"
    label reflects this.
    """
    if not sources:
        return np.array([], dtype=np.float64)

    # Warn when using default (uncalibrated) scaling coefficients
    if math.isclose(config.scaling_a, 0.5) and math.isclose(config.scaling_b, -3.0):
        logger.warning(
            "Using default seismic-only scaling coefficients (a=%.1f, b=%.1f) - "
            "calibrate against regional historical data before production",
            config.scaling_a,
            config.scaling_b,
        )

    slip_m = 10 ** (config.scaling_a * config.magnitude + config.scaling_b)

    # Wells & Coppersmith (1994) surface rupture length
    srl_km = 10 ** (-3.22 + 0.69 * config.magnitude)
    sigma = max(srl_km / 3.0, 1.0)  # floor at 1 km to avoid division issues

    weights = np.zeros(len(sources), dtype=np.float64)
    for i, src in enumerate(sources):
        dist = haversine_distance_km(
            config.epicenter_lat, config.epicenter_lon,
            src.latitude, src.longitude,
        )
        weights[i] = slip_m * math.exp(-(dist**2) / (2 * sigma**2))

    return weights


def run_seismic_only_estimate(
    database: UnitSourceDatabase,
    config: SeismicOnlyConfig,
) -> InversionResult:
    """Produce a seismic-only preliminary scenario estimate.

    No NNLS solve - weights derived from magnitude scaling only.
    waveform_rmse_cm is set to 0.0 (no observations to fit against -
    constraint_stage=SEISMIC_ONLY communicates that no fit was done).
    Note: NaN would fail the RankedScenario ge=0.0 validator.
    """
    sources = select_unit_sources(
        database,
        config.epicenter_lat,
        config.epicenter_lon,
        max_distance_km=config.max_distance_km,
        max_sources=config.max_sources,
    )
    if not sources:
        raise ValueError(
            f"No unit sources found within {config.max_distance_km} km of epicenter"
        )

    weights = compute_seismic_only_weights(sources, config)
    mw = compute_mw_from_weights(weights, sources)

    return InversionResult(
        source_ids=[s.source_id for s in sources],
        weights=weights,
        residual_norm=0.0,
        waveform_rmse_cm=0.0,
        mw_equivalent=mw,
        fit_stations=[],
        inversion_window_sec=0,
        elapsed_sec=0.0,
    )


# ---------------------------------------------------------------------------
# Bootstrap uncertainty estimation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BootstrapConfig:
    """Configuration for bootstrap station resampling.

    Station-level resampling: each bootstrap iteration draws n_stations
    station indices with replacement, re-solves NNLS on the resampled
    rows, and records the resulting source weights and Mw.

    Note: with fewer than ~5 stations, only a handful of unique multisets
    exist, producing a discrete (non-smooth) bootstrap distribution.
    The caller should add a limiting assumption when station count is low.
    """

    n_iterations: int = 500  # N_bootstrap = 500 (configurable)
    seed: int = 42  # deterministic on replay
    confidence_levels: tuple[float, ...] = (0.10, 0.50, 0.90)  # P10/P50/P90

    def __post_init__(self) -> None:
        if self.n_iterations < 1:
            raise ValueError(
                f"n_iterations must be >= 1 (got {self.n_iterations})"
            )


@dataclass
class BootstrapResult:
    """Result of bootstrap station resampling.

    weight_samples and mw_samples are the raw per-iteration outputs.
    mw_percentiles is derived from mw_samples for convenience.
    ensemble_spread is NOT set here - it depends on coastal amplitude
    distribution, computed downstream (see Design Notes: no Mw fallback).
    """

    source_ids: list[str]  # ordered source IDs matching weight_samples columns
    mw_samples: NDArray[np.float64]  # shape: (n_iterations,)
    weight_samples: NDArray[np.float64]  # shape: (n_iterations, n_sources)
    mw_percentiles: dict[float, float]  # {0.10: 7.1, 0.50: 7.5, 0.90: 8.0}
    n_iterations_completed: int


def run_bootstrap(
    H: NDArray[np.float64],
    d: NDArray[np.float64],
    station_ids: list[str],
    sources: list[UnitSource],
    n_timepoints: int,
    config: BootstrapConfig,
) -> BootstrapResult:
    """Run bootstrap station resampling for uncertainty estimation.

    For each iteration: resample station indices with replacement,
    build resampled H and d from row-blocks, re-solve NNLS, record
    weights and Mw.

    Requires >= 2 stations (resampling 1 station is degenerate).
    Uses seeded RNG for deterministic replay.
    """
    n_stations = len(station_ids)
    if n_stations < 2:
        raise ValueError(
            f"Bootstrap requires >= 2 stations (got {n_stations})"
        )
    if H.shape[0] != n_stations * n_timepoints:
        raise ValueError(
            f"H row count {H.shape[0]} != len(station_ids) * n_timepoints "
            f"({n_stations} * {n_timepoints} = {n_stations * n_timepoints})"
        )

    n_sources = H.shape[1]
    source_ids = [s.source_id for s in sources]

    rng = np.random.default_rng(config.seed)
    weight_samples = np.zeros(
        (config.n_iterations, n_sources), dtype=np.float64
    )
    mw_samples = np.zeros(config.n_iterations, dtype=np.float64)

    for i in range(config.n_iterations):
        idx = rng.choice(n_stations, size=n_stations, replace=True)
        # Build resampled H and d by stacking row-blocks
        H_boot = np.vstack(
            [H[k * n_timepoints : (k + 1) * n_timepoints, :] for k in idx]
        )
        d_boot = np.concatenate(
            [d[k * n_timepoints : (k + 1) * n_timepoints] for k in idx]
        )
        alpha, _ = nnls(H_boot, d_boot)
        weight_samples[i] = alpha
        mw_samples[i] = compute_mw_from_weights(alpha, sources)

    # Compute percentiles using config
    pct_values = np.percentile(
        mw_samples, [cl * 100 for cl in config.confidence_levels]
    )
    mw_percentiles = dict(zip(config.confidence_levels, pct_values.tolist()))

    pct_strs = " ".join(
        f"P{int(cl * 100)}={mw_percentiles[cl]:.2f}"
        for cl in sorted(config.confidence_levels)
    )
    logger.info(
        "Bootstrap: %d iterations, %d stations, Mw %s",
        config.n_iterations,
        n_stations,
        pct_strs,
    )

    return BootstrapResult(
        source_ids=source_ids,
        mw_samples=mw_samples,
        weight_samples=weight_samples,
        mw_percentiles=mw_percentiles,
        n_iterations_completed=config.n_iterations,
    )


def classify_ensemble_spread(p10: float, p90: float) -> EnsembleSpread:
    """Classify ensemble spread from P10/P90 of coastal amplitude distribution.

    Called with coastal amplitude P10/P90 only.
    No Mw fallback - thresholds are calibrated for amplitude ratios.

    Amplitude-ratio classification:
    - If P10 <= 0: HIGH (can't compute ratio)
    - ratio = P90 / P10
    - LOW: ratio < 2.0
    - MODERATE: 2.0 <= ratio <= 5.0
    - HIGH: ratio > 5.0
    """
    if p10 <= 0:
        return EnsembleSpread.HIGH
    ratio = p90 / p10
    if ratio < 2.0:
        return EnsembleSpread.LOW
    if ratio <= 5.0:
        return EnsembleSpread.MODERATE
    return EnsembleSpread.HIGH


def rank_scenarios_from_bootstrap(
    bootstrap: BootstrapResult,
    sources: list[UnitSource],
    H: NDArray[np.float64],
    d: NDArray[np.float64],
    max_scenarios: int = 5,
    min_posterior_weight: float = 0.01,
) -> list[RankedScenario]:
    """Build ranked scenarios from bootstrap by grouping activation patterns.

    Groups bootstrap iterations by which sources have weight >
    NNLS_ZERO_THRESHOLD (activation pattern). For well-constrained
    inversions, all iterations produce the same active set, yielding
    1 scenario.

    For each group: computes mean weights, Mw, and per-cluster RMSE
    against the full (non-resampled) H and d.
    """
    n_iterations = bootstrap.n_iterations_completed
    source_ids = bootstrap.source_ids

    # Group iterations by activation pattern
    pattern_groups: dict[frozenset[str], list[int]] = defaultdict(list)
    for i in range(n_iterations):
        active = frozenset(
            sid
            for j, sid in enumerate(source_ids)
            if bootstrap.weight_samples[i, j] > NNLS_ZERO_THRESHOLD
        )
        pattern_groups[active].append(i)

    # Build scenarios from groups
    candidates: list[tuple[float, dict[str, Any]]] = []
    n_obs = len(d)

    for active_set, iteration_indices in pattern_groups.items():
        posterior = len(iteration_indices) / n_iterations
        if posterior < min_posterior_weight:
            continue

        # Mean weights across iterations in this group
        group_weights = bootstrap.weight_samples[iteration_indices]
        mean_weights = group_weights.mean(axis=0)

        # Mw from mean weights
        mw = compute_mw_from_weights(mean_weights, sources)

        # Per-cluster RMSE against full data
        residual = H @ mean_weights - d
        rnorm = float(np.linalg.norm(residual))
        rmse_cm = (rnorm / math.sqrt(n_obs)) * 100.0 if n_obs > 0 else 0.0

        # Filter to active sources for the RankedScenario
        active_ids = []
        active_wts = []
        for j, sid in enumerate(source_ids):
            if mean_weights[j] > NNLS_ZERO_THRESHOLD:
                active_ids.append(sid)
                active_wts.append(float(mean_weights[j]))

        # Edge case: all mean weights zero (shouldn't happen if active_set
        # was non-empty, but handle gracefully)
        if not active_ids:
            active_ids = list(source_ids)
            active_wts = mean_weights.tolist()

        candidates.append((
            posterior,
            dict(
                unit_source_ids=active_ids,
                weights=active_wts,
                waveform_rmse_cm=rmse_cm,
                mw_equivalent=mw,
                posterior_weight=posterior,
            ),
        ))

    # Sort by posterior weight descending; break ties by RMSE ascending
    candidates.sort(key=lambda x: (-x[0], x[1]["waveform_rmse_cm"]))
    ranked: list[RankedScenario] = []
    for rank_idx, (_, scenario_data) in enumerate(candidates[:max_scenarios]):
        ranked.append(
            RankedScenario(**scenario_data, rank=rank_idx + 1)
        )

    # Fallback: if no candidates survived filtering, create one from
    # the overall mean
    if not ranked:
        mean_all = bootstrap.weight_samples.mean(axis=0)
        mw_all = compute_mw_from_weights(mean_all, sources)
        residual = H @ mean_all - d
        rnorm = float(np.linalg.norm(residual))
        rmse_cm = (rnorm / math.sqrt(n_obs)) * 100.0 if n_obs > 0 else 0.0
        active_ids = []
        active_wts = []
        for j, sid in enumerate(source_ids):
            if mean_all[j] > NNLS_ZERO_THRESHOLD:
                active_ids.append(sid)
                active_wts.append(float(mean_all[j]))
        if not active_ids:
            active_ids = list(source_ids)
            active_wts = mean_all.tolist()
        ranked.append(RankedScenario(
            unit_source_ids=active_ids,
            weights=active_wts,
            waveform_rmse_cm=rmse_cm,
            mw_equivalent=mw_all,
            rank=1,
            posterior_weight=1.0,
        ))

    return ranked


# ---------------------------------------------------------------------------
# Coastal amplitude proxy generator
# ---------------------------------------------------------------------------


def compute_coastal_proxies(
    weight_samples: NDArray[np.float64],
    source_ids: list[str],
    factors: dict[str, CoastalForecastFactors],
    event_origin_utc: datetime,
    tidal_corrections: dict[str, float] | None = None,
) -> tuple[list[CoastalProxy], NDArray[np.float64]]:
    """Compute coastal amplitude proxies from inversion weights.

    Works with both bootstrap (n_iterations, n_sources) and single
    inversion (1, n_sources) weight matrices.

    Returns:
        Tuple of (list of CoastalProxy sorted by site_id,
                  max-per-iteration amplitude array of shape (n_iterations,)).
        The max_amplitude array is used by the caller for ensemble spread.
    """
    site_ids = sorted(factors.keys())
    n_iterations = weight_samples.shape[0]
    n_sources = weight_samples.shape[1]
    n_sites = len(site_ids)

    if n_sites == 0:
        return [], np.zeros(n_iterations, dtype=np.float64)

    # Build peak matrix P: (n_sources, n_sites)
    P = np.zeros((n_sources, n_sites), dtype=np.float64)
    for s, site_id in enumerate(site_ids):
        site_factors = factors[site_id]
        for j, src_id in enumerate(source_ids):
            P[j, s] = site_factors.unit_source_peak_m.get(src_id, 0.0)

    # Amplitude matrix: (n_iterations, n_sites)
    amplitude = weight_samples @ P

    # Apply tidal corrections per site
    if tidal_corrections:
        for s, site_id in enumerate(site_ids):
            if site_id in tidal_corrections:
                amplitude[:, s] *= 1.0 + tidal_corrections[site_id]

    # Clamp to non-negative (extreme tidal corrections could go negative)
    amplitude = np.maximum(amplitude, 0.0)

    # Per-iteration max-site amplitude for ensemble spread
    max_amplitude = amplitude.max(axis=1)

    # Build CoastalProxy for each site
    proxies: list[CoastalProxy] = []
    for s, site_id in enumerate(site_ids):
        site_factors = factors[site_id]
        site_amp = amplitude[:, s]

        # Percentiles
        if n_iterations == 1:
            p10 = p50 = p90 = float(site_amp[0])
        else:
            p10 = float(np.percentile(site_amp, 10))
            p50 = float(np.percentile(site_amp, 50))
            p90 = float(np.percentile(site_amp, 90))

        # Per-iteration earliest arrival time at this site
        min_tts: list[float] = []
        for i in range(n_iterations):
            active_tts = [
                site_factors.travel_time_sec[src_id]
                for j, src_id in enumerate(source_ids)
                if weight_samples[i, j] > NNLS_ZERO_THRESHOLD
                and src_id in site_factors.travel_time_sec
            ]
            if active_tts:
                min_tts.append(min(active_tts))

        # Arrival time and uncertainty
        if min_tts:
            if n_iterations > 1:
                arrival_sec = float(np.median(min_tts))
            else:
                arrival_sec = min_tts[0]
            arrival_utc = event_origin_utc + timedelta(seconds=arrival_sec)
            if len(min_tts) > 1:
                arrival_uncertainty_min = (max(min_tts) - min(min_tts)) / 60.0
            else:
                arrival_uncertainty_min = 0.0
        else:
            # All weights zero - conservative: instant arrival
            arrival_utc = event_origin_utc
            arrival_uncertainty_min = 0.0

        tidal_applied = (
            tidal_corrections is not None and site_id in tidal_corrections
        )

        proxies.append(CoastalProxy(
            site_id=site_id,
            arrival_utc=arrival_utc,
            arrival_uncertainty_min=arrival_uncertainty_min,
            amplitude_proxy_p10_m=p10,
            amplitude_proxy_p50_m=p50,
            amplitude_proxy_p90_m=p90,
            tidal_correction_applied=tidal_applied,
        ))

    return proxies, max_amplitude
