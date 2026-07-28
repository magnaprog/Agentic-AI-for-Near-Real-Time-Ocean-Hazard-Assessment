"""Verification checks - pure functions for validating ScenarioAssessment output.


Implements 9 verification checks over ScenarioAssessment output.
Each check returns a VerificationCheck (name, result, evidence).
The outcome policy aggregates checks into a VerificationOutcome.

Applicability semantics: a versioned
requirement matrix keyed by constraint stage decides which checks are
REQUIRED, OPTIONAL, or NOT_APPLICABLE. Check code never infers its own
requiredness. When a check's input data is absent it reports prerequisite
MISSING with result NOT_EVALUATED instead of a trivial PASS; when it
raises, the runner records prerequisite ERROR with result ERROR. A
required check without an available prerequisite makes the overall
outcome INCOMPLETE, which routes to ABSTAIN.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import nnls

from hazard_assessment.agents.scenario_data import UnitSource
from hazard_assessment.agents.scenario_inversion import NNLS_ZERO_THRESHOLD
from hazard_assessment.geo import compute_initial_bearing_deg as _initial_bearing
from hazard_assessment.schemas.scenario import ConstraintStage, ScenarioAssessment
from hazard_assessment.schemas.verification import (
    CheckApplicability,
    CheckResult,
    PrerequisiteStatus,
    VerificationCheck,
    VerificationOutcome,
)

# ---------------------------------------------------------------------------
# Constants - verification check thresholds
# ---------------------------------------------------------------------------

HOLDOUT_ARRIVAL_FAIL_MIN: float = 5.0
HOLDOUT_ARRIVAL_CONCERN_MIN: float = 3.0
HOLDOUT_AMPLITUDE_FAIL_FRAC: float = 0.50
HOLDOUT_AMPLITUDE_CONCERN_FRAC: float = 0.30

SENSITIVITY_WEIGHT_CONCERN_FRAC: float = 0.30

MIN_STATIONS_FAIL: int = 2
AZIMUTHAL_SPREAD_CONCERN_DEG: float = 90.0

MW_DELTA_FAIL: float = 0.6
MW_DELTA_CONCERN: float = 0.3

RMSE_FAIL_CM: float = 5.0
RMSE_CONCERN_CM: float = 3.0
BIAS_ABSOLUTE_FLOOR_CM: float = 1.0
BIAS_SIGMA_THRESHOLD: float = 3.0

METEO_FAIL_SCORE: float = 0.6
METEO_CONCERN_SCORE: float = 0.3

# Guard for div-by-zero in amplitude error calculation.
_AMPLITUDE_FLOOR: float = 1e-10


def _not_evaluated(name: str, evidence: str) -> VerificationCheck:
    """A check whose input data is absent: prerequisite MISSING, honestly
    NOT_EVALUATED instead of a trivial PASS."""
    return VerificationCheck(
        name=name,
        result=CheckResult.NOT_EVALUATED,
        evidence=evidence,
        prerequisite=PrerequisiteStatus.MISSING,
    )

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StationPosition:
    """Geographic position of a DART station."""

    station_id: str
    latitude: float
    longitude: float


@dataclass(frozen=True)
class HoldoutData:
    """Waveform data for a hold-out station validation."""

    station_id: str
    observed_waveform: NDArray[np.float64]
    predicted_waveform: NDArray[np.float64]
    observed_arrival_index: int | None
    predicted_arrival_index: int | None
    time_step_sec: float


@dataclass(frozen=True)
class VerificationInput:
    """All data needed by the 9 verification checks.

    Fields are None where data may be unavailable. A check whose input
    data is absent reports prerequisite MISSING with result
    NOT_EVALUATED; whether that blocks the overall outcome depends on
    the requirement matrix, not on the check itself.
    """

    scenario: ScenarioAssessment

    # Hold-out station
    holdout: HoldoutData | None = None

    # Raw inversion data (sensitivity + model fit)
    H: NDArray[np.float64] | None = None
    d: NDArray[np.float64] | None = None
    sources: list[UnitSource] | None = None

    # Seismic magnitude (physical consistency)
    mw_seismic: float | None = None

    # Posterior stability
    previous_leading_source_ids: list[str] | None = None

    # Station geometry (azimuthal spread)
    station_positions: list[StationPosition] | None = None
    epicenter_lat: float | None = None
    epicenter_lon: float | None = None

    # Tidal state
    tidal_correction_needed: bool = False

    # Meteotsunami (from AnomalyAssessment)
    meteotsunami_score: float | None = None

    # Rayleigh wave false-trigger flag (from AnomalyAssessment).
    # None means the anomaly layer never evaluated the check (its
    # prerequisites were unavailable); False means evaluated and clean.
    rayleigh_wave_suspect: bool | None = None


# ---------------------------------------------------------------------------
# Geodetic helpers
# ---------------------------------------------------------------------------



def _compute_azimuthal_spread(bearings: list[float]) -> float:
    """Azimuthal coverage in degrees: 360 - max angular gap.

    Bearings must be in [0, 360). Returns 0.0 for fewer than 2 bearings.
    """
    if len(bearings) < 2:
        return 0.0

    sorted_b = sorted(bearings)
    max_gap = 0.0
    for i in range(len(sorted_b) - 1):
        gap = sorted_b[i + 1] - sorted_b[i]
        if gap > max_gap:
            max_gap = gap
    # Wrap-around gap
    wrap_gap = (360.0 - sorted_b[-1]) + sorted_b[0]
    if wrap_gap > max_gap:
        max_gap = wrap_gap

    return 360.0 - max_gap


# ---------------------------------------------------------------------------
# Check 1: Hold-out station validation
# ---------------------------------------------------------------------------


def check_holdout_station(vi: VerificationInput) -> VerificationCheck:
    """Validate scenario prediction against a withheld DART station."""
    if vi.holdout is None:
        return _not_evaluated(
            "holdout_station_validation",
            "No hold-out station available; predictions not verified",
        )

    h = vi.holdout
    if len(h.observed_waveform) == 0 or len(h.predicted_waveform) == 0:
        # Degrade like the no-holdout case instead of letting np.max crash
        # on an empty array: an unusable holdout is missing evidence, not
        # a pipeline error.
        return _not_evaluated(
            "holdout_station_validation",
            f"station={h.station_id}; empty holdout waveform; "
            "predictions not verified",
        )

    parts: list[str] = [f"station={h.station_id}"]
    fail = False
    concern = False

    # Arrival time error
    if h.observed_arrival_index is not None and h.predicted_arrival_index is not None:
        arrival_error_min = (
            abs(h.predicted_arrival_index - h.observed_arrival_index)
            * h.time_step_sec
            / 60.0
        )
        parts.append(f"arrival_error={arrival_error_min:.2f} min")
        if arrival_error_min > HOLDOUT_ARRIVAL_FAIL_MIN:
            fail = True
        elif arrival_error_min > HOLDOUT_ARRIVAL_CONCERN_MIN:
            concern = True
    else:
        parts.append("arrival_check=skipped (index unavailable)")

    # Amplitude error. When both peaks are below the numerical floor they
    # are indistinguishable from zero. If only one is below the floor,
    # normalize by the larger peak: treating a nonzero prediction against
    # an effectively zero observation as zero error would fail open.
    obs_peak = float(np.max(np.abs(h.observed_waveform)))
    pred_peak = float(np.max(np.abs(h.predicted_waveform)))
    if obs_peak < _AMPLITUDE_FLOOR and pred_peak < _AMPLITUDE_FLOOR:
        amplitude_error = 0.0
        parts.append(
            "amplitude_error=0.00 "
            f"(obs_peak={obs_peak:.2e}, pred_peak={pred_peak:.2e}; both below floor)"
        )
    elif obs_peak < _AMPLITUDE_FLOOR:
        amplitude_error = abs(pred_peak - obs_peak) / max(pred_peak, _AMPLITUDE_FLOOR)
        parts.append(
            f"amplitude_error={amplitude_error:.2%} "
            f"(obs_peak={obs_peak:.2e} below floor, pred_peak={pred_peak:.2e})"
        )
    else:
        amplitude_error = abs(pred_peak - obs_peak) / obs_peak
        parts.append(f"amplitude_error={amplitude_error:.2%}")

    if amplitude_error > HOLDOUT_AMPLITUDE_FAIL_FRAC:
        fail = True
    elif amplitude_error > HOLDOUT_AMPLITUDE_CONCERN_FRAC:
        concern = True

    if fail:
        result = CheckResult.FAIL
    elif concern:
        result = CheckResult.CONCERN
    else:
        result = CheckResult.PASS

    return VerificationCheck(
        name="holdout_station_validation",
        result=result,
        evidence="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# Check 2: Sensitivity analysis
# ---------------------------------------------------------------------------


def check_sensitivity(vi: VerificationInput) -> VerificationCheck:
    """Leave-one-out stability of the NNLS active source set."""
    if vi.H is None or vi.d is None or vi.sources is None:
        return _not_evaluated(
            "sensitivity_analysis",
            "No inversion data available; leave-one-out not performed",
        )

    n_sources = vi.H.shape[1]
    if n_sources < 2:
        return _not_evaluated(
            "sensitivity_analysis",
            "Cannot perform leave-one-out with single source",
        )

    # Guard: LOO is O(n_sources^2) - skip for large problems.
    # SIFT-scale unit-source databases can have hundreds of columns.
    # LOO stability is most meaningful for small active source sets,
    # which is the relevant case for well-constrained inversions.
    _LOO_MAX_SOURCES = 50
    if n_sources > _LOO_MAX_SOURCES:
        return _not_evaluated(
            "sensitivity_analysis",
            f"LOO sensitivity not run: n_sources={n_sources} > {_LOO_MAX_SOURCES} "
            "(quadratic runtime exceeds the operational check budget); "
            "full sensitivity analysis is required offline",
        )

    # Reference solve
    ref_weights, _ = nnls(vi.H, vi.d)
    ref_active = set(j for j in range(n_sources) if ref_weights[j] > NNLS_ZERO_THRESHOLD)

    # If no sources are active in the reference, there is no active set
    # whose stability could be tested: the prerequisite for a stability
    # verdict is missing, not satisfied.
    if not ref_active:
        return _not_evaluated(
            "sensitivity_analysis",
            "No active sources in reference solution; stability not tested",
        )

    # Relative weights among active sources for concern check
    ref_active_sum = sum(ref_weights[j] for j in ref_active)
    ref_relative: dict[int, float] = {}
    if ref_active_sum > 0:
        ref_relative = {j: ref_weights[j] / ref_active_sum for j in ref_active}

    fail = False
    concern = False
    flipped_sources: list[int] = []
    shifted_sources: list[int] = []

    for j in range(n_sources):
        # Drop column j
        H_reduced = np.delete(vi.H, j, axis=1)
        loo_weights, _ = nnls(H_reduced, vi.d)

        # Map LOO indices back to original indices (skip j)
        loo_active = set()
        idx = 0
        for k in range(n_sources):
            if k == j:
                continue
            if loo_weights[idx] > NNLS_ZERO_THRESHOLD:
                loo_active.add(k)
            idx += 1

        # Active set excluding the dropped source
        ref_active_minus_j = ref_active - {j}
        if loo_active != ref_active_minus_j:
            fail = True
            flipped_sources.append(j)

        # Relative weight shift among remaining active sources
        if not fail and ref_active_minus_j:
            loo_active_sum = sum(
                loo_weights[_loo_index(k, j)] for k in ref_active_minus_j
            )
            if loo_active_sum > 0:
                for k in ref_active_minus_j:
                    loo_idx = _loo_index(k, j)
                    loo_rel = loo_weights[loo_idx] / loo_active_sum
                    ref_rel = ref_relative.get(k, 0.0)
                    if (
                        ref_rel > 0
                        and abs(loo_rel - ref_rel) / ref_rel
                        > SENSITIVITY_WEIGHT_CONCERN_FRAC
                    ):
                        concern = True
                        shifted_sources.append(j)
                        break

    parts: list[str] = [f"n_sources={n_sources}", f"n_active={len(ref_active)}"]
    if flipped_sources:
        parts.append(f"active_set_flipped_on_drop={flipped_sources}")
    if shifted_sources:
        parts.append(f"weight_shifted_on_drop={shifted_sources}")

    if fail:
        result = CheckResult.FAIL
    elif concern:
        result = CheckResult.CONCERN
    else:
        result = CheckResult.PASS

    return VerificationCheck(
        name="sensitivity_analysis",
        result=result,
        evidence="; ".join(parts),
    )


def _loo_index(original_idx: int, dropped_idx: int) -> int:
    """Map an original source index to the LOO array index after dropping one column."""
    if original_idx < dropped_idx:
        return original_idx
    return original_idx - 1


# ---------------------------------------------------------------------------
# Check 3: Posterior stability
# ---------------------------------------------------------------------------


def check_posterior_stability(vi: VerificationInput) -> VerificationCheck:
    """Compare leading scenario against previous assessment."""
    if vi.previous_leading_source_ids is None:
        return _not_evaluated(
            "posterior_stability",
            "First assessment (no previous leading scenario to compare)",
        )

    if not vi.scenario.top_scenarios:
        return _not_evaluated(
            "posterior_stability",
            "No scenarios available for stability comparison",
        )
    current_ids = set(vi.scenario.top_scenarios[0].unit_source_ids)
    previous_ids = set(vi.previous_leading_source_ids)

    if current_ids == previous_ids:
        return VerificationCheck(
            name="posterior_stability",
            result=CheckResult.PASS,
            evidence=f"Leading scenario unchanged: {sorted(current_ids)}",
        )

    return VerificationCheck(
        name="posterior_stability",
        result=CheckResult.CONCERN,
        evidence=(
            f"Leading scenario changed: "
            f"previous={sorted(previous_ids)}, current={sorted(current_ids)}"
        ),
    )


# ---------------------------------------------------------------------------
# Check 4: Data coverage
# ---------------------------------------------------------------------------


def check_data_coverage(vi: VerificationInput) -> VerificationCheck:
    """Assess DART station coverage and azimuthal spread.

    Applicability at SEISMIC_ONLY is a requirement-matrix decision
    (NOT_APPLICABLE: no DART constraint exists in a prior-only
    scenario), not something this check infers.
    """
    n = len(vi.scenario.dart_stations_used)

    if n < MIN_STATIONS_FAIL:
        return VerificationCheck(
            name="data_coverage",
            result=CheckResult.FAIL,
            evidence=f"n_stations={n} < {MIN_STATIONS_FAIL}",
        )

    # Coverage is a geometric claim, so every constraining station and the
    # epicenter need coordinates. A station count alone cannot pass this
    # REQUIRED check.
    if vi.epicenter_lat is None or vi.epicenter_lon is None:
        return _not_evaluated(
            "data_coverage",
            f"n_stations={n}; epicenter coordinates unavailable; "
            "azimuthal coverage not evaluated",
        )
    if not vi.station_positions:
        return _not_evaluated(
            "data_coverage",
            f"n_stations={n}; station coordinates unavailable; "
            "azimuthal coverage not evaluated",
        )

    positions_by_id = {position.station_id: position for position in vi.station_positions}
    missing = sorted(set(vi.scenario.dart_stations_used) - set(positions_by_id))
    if missing:
        return _not_evaluated(
            "data_coverage",
            f"n_stations={n}; missing coordinates for constraining "
            f"station(s) {missing}; azimuthal coverage not evaluated",
        )

    constraining_positions = [
        positions_by_id[station_id] for station_id in vi.scenario.dart_stations_used
    ]
    bearings = [
        _initial_bearing(
            vi.epicenter_lat, vi.epicenter_lon, position.latitude, position.longitude,
        )
        for position in constraining_positions
    ]
    spread = _compute_azimuthal_spread(bearings)
    if spread < AZIMUTHAL_SPREAD_CONCERN_DEG:
        return VerificationCheck(
            name="data_coverage",
            result=CheckResult.CONCERN,
            evidence=(
                f"n_stations={n}, azimuthal_spread={spread:.1f} deg"
                f" < {AZIMUTHAL_SPREAD_CONCERN_DEG} deg"
            ),
        )

    return VerificationCheck(
        name="data_coverage",
        result=CheckResult.PASS,
        evidence=f"n_stations={n}, azimuthal_spread={spread:.1f} deg",
    )


# ---------------------------------------------------------------------------
# Check 5: Physical consistency
# ---------------------------------------------------------------------------


def check_physical_consistency(vi: VerificationInput) -> VerificationCheck:
    """Compare NNLS-derived Mw against seismic Mw."""
    if vi.mw_seismic is None:
        return _not_evaluated(
            "physical_consistency",
            "No seismic magnitude available; Mw comparison not performed",
        )
    if not vi.scenario.top_scenarios:
        return _not_evaluated(
            "physical_consistency",
            "No scenarios available for consistency check",
        )

    mw_nnls = vi.scenario.top_scenarios[0].mw_equivalent
    delta = abs(mw_nnls - vi.mw_seismic)

    evidence = f"Mw_nnls={mw_nnls:.2f}, Mw_seismic={vi.mw_seismic:.2f}, delta={delta:.2f}"

    if delta > MW_DELTA_FAIL:
        return VerificationCheck(
            name="physical_consistency",
            result=CheckResult.FAIL,
            evidence=evidence,
        )
    if delta > MW_DELTA_CONCERN:
        return VerificationCheck(
            name="physical_consistency",
            result=CheckResult.CONCERN,
            evidence=evidence,
        )
    return VerificationCheck(
        name="physical_consistency",
        result=CheckResult.PASS,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 6: Model fit quality
# ---------------------------------------------------------------------------


def check_model_fit(vi: VerificationInput) -> VerificationCheck:
    """Evaluate waveform RMSE and systematic bias."""
    if not vi.scenario.top_scenarios:
        return _not_evaluated(
            "model_fit_quality",
            "No scenarios available for model fit check",
        )
    rmse = vi.scenario.top_scenarios[0].waveform_rmse_cm
    n_stations = len(vi.scenario.dart_stations_used)

    parts: list[str] = [f"rmse={rmse:.2f} cm"]
    rmse_fail = rmse > RMSE_FAIL_CM
    rmse_concern = rmse > RMSE_CONCERN_CM

    # Systematic bias test
    bias_detected = False
    if vi.H is not None and vi.d is not None and n_stations >= 2:
        if len(vi.d) % n_stations != 0:
            parts.append("bias: skipped (d length not divisible by n_stations)")
        else:
            ref_weights, _ = nnls(vi.H, vi.d)
            residual = vi.H @ ref_weights - vi.d
            n_timepoints = len(vi.d) // n_stations

            # Per-station mean residual
            station_means = np.array([
                float(np.mean(residual[k * n_timepoints : (k + 1) * n_timepoints]))
                for k in range(n_stations)
            ])

            grand_mean = float(np.mean(station_means))
            std_means = float(np.std(station_means, ddof=1))

            # z-test: |grand_mean| > 3sigma / sqrt(n) AND absolute floor
            grand_mean_cm = abs(grand_mean) * 100.0
            if std_means > 0:
                z = abs(grand_mean) / (std_means / math.sqrt(n_stations))
                if z > BIAS_SIGMA_THRESHOLD and grand_mean_cm > BIAS_ABSOLUTE_FLOOR_CM:
                    bias_detected = True
                    parts.append(
                        f"bias: z={z:.2f} > {BIAS_SIGMA_THRESHOLD}, "
                        f"|mean|={grand_mean_cm:.2f} cm > {BIAS_ABSOLUTE_FLOOR_CM} cm"
                    )
            elif grand_mean_cm > BIAS_ABSOLUTE_FLOOR_CM:
                # std == 0 with n >= 2 means all station means identical and nonzero
                bias_detected = True
                parts.append(f"bias: uniform offset |mean|={grand_mean_cm:.2f} cm")

    if rmse_fail or bias_detected:
        result = CheckResult.FAIL
    elif rmse_concern:
        result = CheckResult.CONCERN
    else:
        result = CheckResult.PASS

    return VerificationCheck(
        name="model_fit_quality",
        result=result,
        evidence="; ".join(parts),
    )


# ---------------------------------------------------------------------------
# Check 7: Tidal state review
# ---------------------------------------------------------------------------


def check_tidal_state(vi: VerificationInput) -> VerificationCheck:
    """Verify tidal correction status of coastal amplitude proxies."""
    if not vi.scenario.coastal_proxies:
        return _not_evaluated(
            "tidal_state_review",
            "No coastal proxies available to review",
        )

    has_uncorrected = any(
        not p.tidal_correction_applied for p in vi.scenario.coastal_proxies
    )

    if not has_uncorrected:
        return VerificationCheck(
            name="tidal_state_review",
            result=CheckResult.PASS,
            evidence="All coastal proxies have tidal correction applied",
        )

    n_uncorrected = sum(
        1 for p in vi.scenario.coastal_proxies if not p.tidal_correction_applied
    )
    n_total = len(vi.scenario.coastal_proxies)

    if vi.tidal_correction_needed:
        return VerificationCheck(
            name="tidal_state_review",
            result=CheckResult.FAIL,
            evidence=(
                f"Tidal correction missing when needed: "
                f"{n_uncorrected}/{n_total} proxies uncorrected"
            ),
        )

    return VerificationCheck(
        name="tidal_state_review",
        result=CheckResult.CONCERN,
        evidence=(
            f"Uncorrected coastal proxies present (tidal correction not flagged as needed): "
            f"{n_uncorrected}/{n_total} proxies uncorrected"
        ),
    )


# ---------------------------------------------------------------------------
# Check 8: Meteotsunami discriminator
# ---------------------------------------------------------------------------


def check_meteotsunami(vi: VerificationInput) -> VerificationCheck:
    """Flag elevated meteotsunami likelihood."""
    if vi.meteotsunami_score is None:
        return _not_evaluated(
            "meteotsunami_discriminator",
            "No meteotsunami score available from the anomaly layer",
        )

    score = vi.meteotsunami_score
    evidence = f"meteotsunami_score={score:.2f}"

    if score > METEO_FAIL_SCORE:
        return VerificationCheck(
            name="meteotsunami_discriminator",
            result=CheckResult.FAIL,
            evidence=evidence,
        )
    if score > METEO_CONCERN_SCORE:
        return VerificationCheck(
            name="meteotsunami_discriminator",
            result=CheckResult.CONCERN,
            evidence=evidence,
        )
    return VerificationCheck(
        name="meteotsunami_discriminator",
        result=CheckResult.PASS,
        evidence=evidence,
    )


# ---------------------------------------------------------------------------
# Check 9: Rayleigh wave false-trigger suspect (flag)
# ---------------------------------------------------------------------------


def check_rayleigh_wave_suspect(vi: VerificationInput) -> VerificationCheck:
    """Flag Rayleigh wave timing correlation from AnomalyAssessment.

    Returns CONCERN (never FAIL) when the anomaly detection layer has
    flagged that a DART pressure excursion is consistent with Rayleigh
    wave arrival timing from a known earthquake epicenter.

    Rayleigh wave timing correlation is not proof of false trigger -
    real tsunamis also follow large earthquakes and arrive at similar
    or later times. This check increases scrutiny, it does not
    auto-dismiss the event.

    A None flag means the anomaly layer never ran the timing check (its
    prerequisites were unavailable), which is distinct from an evaluated,
    clean False.
    """
    if vi.rayleigh_wave_suspect is None:
        return _not_evaluated(
            "rayleigh_wave_suspect",
            "Rayleigh timing check not evaluated (station coordinates "
            "or a firing detector were unavailable)",
        )

    if not vi.rayleigh_wave_suspect:
        return VerificationCheck(
            name="rayleigh_wave_suspect",
            result=CheckResult.PASS,
            evidence="No Rayleigh wave timing correlation detected",
        )

    return VerificationCheck(
        name="rayleigh_wave_suspect",
        result=CheckResult.CONCERN,
        evidence=(
            "DART event-mode trigger timing is consistent with Rayleigh wave "
            "arrival from the earthquake epicenter (haversine/3.6 km/s "
            "+/- 20%). Possible false trigger; requires additional scrutiny."
        ),
    )


# ---------------------------------------------------------------------------
# Requirement matrix
# ---------------------------------------------------------------------------

CHECK_FUNCTIONS: dict[str, Callable[[VerificationInput], VerificationCheck]] = {
    "holdout_station_validation": check_holdout_station,
    "sensitivity_analysis": check_sensitivity,
    "posterior_stability": check_posterior_stability,
    "data_coverage": check_data_coverage,
    "physical_consistency": check_physical_consistency,
    "model_fit_quality": check_model_fit,
    "tidal_state_review": check_tidal_state,
    "meteotsunami_discriminator": check_meteotsunami,
    "rayleigh_wave_suspect": check_rayleigh_wave_suspect,
}

REQUIREMENT_MATRIX_VERSION = "1"

# Applicability by constraint stage. Rationale:
#
# SEISMIC_ONLY carries no ocean-constrained inversion, so the four
# inversion-dependent checks (holdout prediction, leave-one-out
# sensitivity, DART coverage geometry, waveform fit) are structurally
# NOT_APPLICABLE: there is no inversion whose quality they could
# measure. The one check that can still anchor a prior-only scenario is
# physical consistency (prior scenario Mw against the seismic Mw that
# triggered the event), so it is the stage's REQUIRED set; without a
# seismic magnitude to compare, a prior-only scenario is unverifiable
# and the outcome is honestly INCOMPLETE. The remaining evidence checks
# stay OPTIONAL because their inputs (previous assessment, coastal
# proxies, anomaly-layer scores) may legitimately not exist yet early
# in an event.
#
# DART_CONSTRAINED adds an inversion, so coverage geometry, waveform
# fit, and physical consistency become REQUIRED: an ocean-constrained
# scenario without station-coverage, fit, or magnitude verification
# must not authorize output. Holdout and sensitivity stay OPTIONAL
# because a withheld station or a multi-source active set may not exist
# with few constraining stations.
#
# MULTI_STATION additionally REQUIRES sensitivity: with several
# constraining stations, an inversion whose active source set is not
# stable under leave-one-out must not authorize output. A degenerate
# multi-station inversion with no testable active set therefore
# produces INCOMPLETE, which is the intended fail-closed behavior.
REQUIREMENT_MATRIX: dict[ConstraintStage, dict[str, CheckApplicability]] = {
    ConstraintStage.SEISMIC_ONLY: {
        "holdout_station_validation": CheckApplicability.NOT_APPLICABLE,
        "sensitivity_analysis": CheckApplicability.NOT_APPLICABLE,
        "posterior_stability": CheckApplicability.OPTIONAL,
        "data_coverage": CheckApplicability.NOT_APPLICABLE,
        "physical_consistency": CheckApplicability.REQUIRED,
        "model_fit_quality": CheckApplicability.NOT_APPLICABLE,
        "tidal_state_review": CheckApplicability.OPTIONAL,
        "meteotsunami_discriminator": CheckApplicability.OPTIONAL,
        "rayleigh_wave_suspect": CheckApplicability.OPTIONAL,
    },
    ConstraintStage.DART_CONSTRAINED: {
        "holdout_station_validation": CheckApplicability.OPTIONAL,
        "sensitivity_analysis": CheckApplicability.OPTIONAL,
        "posterior_stability": CheckApplicability.OPTIONAL,
        "data_coverage": CheckApplicability.REQUIRED,
        "physical_consistency": CheckApplicability.REQUIRED,
        "model_fit_quality": CheckApplicability.REQUIRED,
        "tidal_state_review": CheckApplicability.OPTIONAL,
        "meteotsunami_discriminator": CheckApplicability.OPTIONAL,
        "rayleigh_wave_suspect": CheckApplicability.OPTIONAL,
    },
    ConstraintStage.MULTI_STATION: {
        "holdout_station_validation": CheckApplicability.OPTIONAL,
        "sensitivity_analysis": CheckApplicability.REQUIRED,
        "posterior_stability": CheckApplicability.OPTIONAL,
        "data_coverage": CheckApplicability.REQUIRED,
        "physical_consistency": CheckApplicability.REQUIRED,
        "model_fit_quality": CheckApplicability.REQUIRED,
        "tidal_state_review": CheckApplicability.OPTIONAL,
        "meteotsunami_discriminator": CheckApplicability.OPTIONAL,
        "rayleigh_wave_suspect": CheckApplicability.OPTIONAL,
    },
}


def validate_requirement_matrix(
    matrix: dict[ConstraintStage, dict[str, CheckApplicability]] | None = None,
) -> None:
    """Reject structurally invalid requirement matrices.

    Every constraint stage must assign an applicability to every known
    check (no more, no less), and every output-authorizing stage must
    have a nonempty REQUIRED set - a stage whose checks are all
    optional or inapplicable could authorize output with zero
    verification, which the plan forbids.
    """
    m = REQUIREMENT_MATRIX if matrix is None else matrix
    expected = set(CHECK_FUNCTIONS)
    for stage in ConstraintStage:
        if stage not in m:
            raise ValueError(f"requirement matrix missing stage {stage}")
        assigned = set(m[stage])
        if assigned != expected:
            raise ValueError(
                f"requirement matrix stage {stage}: assigned checks "
                f"{sorted(assigned)} != known checks {sorted(expected)}"
            )
        required = [
            name
            for name, app in m[stage].items()
            if app == CheckApplicability.REQUIRED
        ]
        if not required:
            raise ValueError(
                f"requirement matrix stage {stage}: empty REQUIRED set; "
                "an output-authorizing stage needs at least one required "
                "check"
            )


# Fail fast on import if the shipped matrix is malformed.
validate_requirement_matrix()


# ---------------------------------------------------------------------------
# Run all checks (stage-aware)
# ---------------------------------------------------------------------------


def run_all_checks(vi: VerificationInput) -> list[VerificationCheck]:
    """Run the 9 verification checks under the requirement matrix.

    NOT_APPLICABLE checks are not executed; they are recorded with the
    canonical (NOT_APPLICABLE, NOT_REQUIRED, NOT_APPLICABLE) row so the
    result documents why they carry no verdict. Applicable checks that
    raise are captured as (prerequisite ERROR, result ERROR) instead of
    aborting verification: an errored check must surface as INCOMPLETE
    or PASS_WITH_CONCERNS, never vanish.
    """
    stage = vi.scenario.constraint_stage
    stage_matrix = REQUIREMENT_MATRIX[stage]

    checks: list[VerificationCheck] = []
    for name, check_fn in CHECK_FUNCTIONS.items():
        applicability = stage_matrix[name]
        if applicability == CheckApplicability.NOT_APPLICABLE:
            checks.append(
                VerificationCheck(
                    name=name,
                    result=CheckResult.NOT_APPLICABLE,
                    evidence=(
                        f"Not applicable at constraint stage {stage.value} "
                        f"(requirement matrix v{REQUIREMENT_MATRIX_VERSION})"
                    ),
                    applicability=CheckApplicability.NOT_APPLICABLE,
                    prerequisite=PrerequisiteStatus.NOT_REQUIRED,
                )
            )
            continue
        try:
            check = check_fn(vi)
        except Exception as exc:  # noqa: BLE001 - must capture, not abort
            checks.append(
                VerificationCheck(
                    name=name,
                    result=CheckResult.ERROR,
                    evidence=(
                        f"Check raised {type(exc).__name__}: {exc}"
                    ),
                    applicability=applicability,
                    prerequisite=PrerequisiteStatus.ERROR,
                )
            )
            continue
        checks.append(
            check.model_copy(update={"applicability": applicability})
        )
    return checks


# ---------------------------------------------------------------------------
# Outcome policy (aggregation)
# ---------------------------------------------------------------------------


def determine_outcome(
    checks: list[VerificationCheck],
) -> tuple[VerificationOutcome, bool, str | None]:
    """Aggregate check results into overall outcome.

    Returns (outcome, abstain_required, abstain_reason).

    Aggregation:
    - Any individual FAIL -> overall FAIL, abstain required.
    - Any REQUIRED check with prerequisite MISSING or ERROR ->
      INCOMPLETE, abstain required.
    - No applicable checks at all -> INCOMPLETE, abstain required.
    - Any CONCERN, or any OPTIONAL check with prerequisite MISSING or
      ERROR -> PASS_WITH_CONCERNS.
    - Otherwise every applicable check passed on available data -> PASS.
    """
    if not checks:
        raise ValueError("checks must be non-empty")

    applicable = [
        c for c in checks
        if c.applicability != CheckApplicability.NOT_APPLICABLE
    ]
    if not applicable:
        return (
            VerificationOutcome.INCOMPLETE,
            True,
            "Verification incomplete: no applicable checks at this "
            "constraint stage; output cannot be verified",
        )

    failed = [c for c in applicable if c.result == CheckResult.FAIL]
    if failed:
        reason_parts = [f"{c.name}: {c.evidence}" for c in failed]
        return (
            VerificationOutcome.FAIL,
            True,
            "Verification failed: " + "; ".join(reason_parts),
        )

    blocked_required = [
        c for c in applicable
        if c.applicability == CheckApplicability.REQUIRED
        and c.prerequisite
        in (PrerequisiteStatus.MISSING, PrerequisiteStatus.ERROR)
    ]
    if blocked_required:
        reason_parts = [
            f"{c.name} ({c.prerequisite.value}): {c.evidence}"
            for c in blocked_required
        ]
        return (
            VerificationOutcome.INCOMPLETE,
            True,
            "Verification incomplete: required check(s) without "
            "available prerequisites: " + "; ".join(reason_parts),
        )

    concerns = [c for c in applicable if c.result == CheckResult.CONCERN]
    blocked_optional = [
        c for c in applicable
        if c.applicability == CheckApplicability.OPTIONAL
        and c.prerequisite
        in (PrerequisiteStatus.MISSING, PrerequisiteStatus.ERROR)
    ]
    if concerns or blocked_optional:
        return (VerificationOutcome.PASS_WITH_CONCERNS, False, None)

    return (VerificationOutcome.PASS, False, None)
