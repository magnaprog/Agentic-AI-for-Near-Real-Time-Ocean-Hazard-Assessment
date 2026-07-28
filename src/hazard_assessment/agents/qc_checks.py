"""QARTOD-aligned quality check functions for the QC Agent.

Each check function accepts observation data and returns a QARTODFlag.
Thresholds are QARTOD-aligned and defined in QCThresholds below.

Observations must be sorted by source_timestamp before calling
batch functions. The sort_observations() helper enforces deterministic
ordering for out-of-order arrivals.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from hazard_assessment.schemas.qc import QARTODFlag, QARTODFlags

# ---------------------------------------------------------------------------
# QARTOD-aligned thresholds
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QCThresholds:
    """Station-type-specific QC thresholds.

    Values are QARTOD-aligned.  These define the
    PASS / SUSPECT boundary for each test.  FAIL boundaries are derived
    via interim multipliers in the check functions (1.5x for gross range,
    2x for spike and rate-of-change).  For timing gap, the PASS boundary
    is timing_gap_factor x expected_interval; SUSPECT is 2x that (= 4x
    expected).  The QARTOD manuals recommend separate operator-defined
    SUSPECT and FAIL thresholds; proper calibration is deferred.
    """

    gross_range_min: float
    gross_range_max: float
    spike_threshold_m: float
    spike_window_sec: float
    rate_of_change_max: float  # m/s
    flat_line_min_variation: float  # meters
    flat_line_window_sec: float  # seconds
    timing_gap_factor: float  # multiplier on expected interval


DART_THRESHOLDS = QCThresholds(
    gross_range_min=-0.5,
    gross_range_max=0.5,
    spike_threshold_m=0.1,
    spike_window_sec=15.0,
    rate_of_change_max=0.001,
    flat_line_min_variation=0.0001,
    flat_line_window_sec=3600.0,
    timing_gap_factor=2.0,
)

COOPS_THRESHOLDS = QCThresholds(
    gross_range_min=-3.0,
    gross_range_max=3.0,
    spike_threshold_m=0.3,
    spike_window_sec=60.0,
    rate_of_change_max=0.005,
    flat_line_min_variation=0.001,
    flat_line_window_sec=3600.0,
    timing_gap_factor=2.0,
)


# ---------------------------------------------------------------------------
# Observation container used by QC checks
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class QCObservation:
    """Lightweight container for QC-relevant fields from any source type."""

    source_type: Literal["dart", "coops"]
    station_id: str
    source_timestamp: datetime
    value_m: float | None  # height_m for DART, water_level_m for CO-OPS
    measurement_type: int | None  # DART only (1, 2, 3); None for CO-OPS
    event_mode: bool  # DART only; always False for CO-OPS
    expected_interval_sec: float  # per-mode expected interval
    payload_sha256: str


# ---------------------------------------------------------------------------
# Deterministic sort for out-of-order arrivals
# ---------------------------------------------------------------------------

def sort_observations(observations: list[QCObservation]) -> list[QCObservation]:
    """Sort observations by (source_timestamp, station_id, payload_sha256).

    Deterministic tie-breaking ensures identical output regardless of
    arrival order. This is the continuity guarantee.
    """
    return sorted(
        observations,
        key=lambda o: (o.source_timestamp, o.station_id, o.payload_sha256),
    )


# ---------------------------------------------------------------------------
# Individual QARTOD checks
# ---------------------------------------------------------------------------

def _thresholds_for(obs: QCObservation) -> QCThresholds:
    if obs.source_type == "dart":
        return DART_THRESHOLDS
    return COOPS_THRESHOLDS


def check_gross_range(obs: QCObservation) -> QARTODFlag:
    """Gross range test: value within physically plausible bounds.

    DART: [-0.5, 0.5] residual - requires detided value.
          Returns NOT_EVALUATED when value_m is raw height (not residual).
    CO-OPS: [-3.0, 3.0] from prediction - raw water_level_m relative to
            STND datum is a reasonable proxy until tidal prediction is available.
    """
    if obs.value_m is None:
        return QARTODFlag.MISSING
    if not math.isfinite(obs.value_m):
        return QARTODFlag.FAIL
    if obs.source_type == "dart":
        # DART height_m is raw water column height (typically 1000-6000 m).
        # The gross-range thresholds are for detided residuals, which
        # require detiding. Until then, gross range is not evaluable.
        return QARTODFlag.NOT_EVALUATED
    t = _thresholds_for(obs)
    if t.gross_range_min <= obs.value_m <= t.gross_range_max:
        return QARTODFlag.PASS
    # Interim SUSPECT boundary at 1.5x the PASS range, pending threshold calibration.
    expanded_min = t.gross_range_min * 1.5
    expanded_max = t.gross_range_max * 1.5
    if expanded_min <= obs.value_m <= expanded_max:
        return QARTODFlag.SUSPECT
    return QARTODFlag.FAIL


def check_spike(
    obs: QCObservation,
    prev: QCObservation | None,
) -> QARTODFlag:
    """Spike test: absolute jump between consecutive points exceeds threshold.

    Simplified 2-point test (|current - prev|).  The QARTOD standard uses
    a 3-point test (deviation from neighbor midpoint); implementing the
    standard form requires the *next* observation and is deferred to future work.

    Thresholds - DART: 0.1 m; CO-OPS: 0.3 m.
    Guard: if the time gap exceeds 4x spike_window_sec the test returns
    NOT_EVALUATED (gap too large for meaningful comparison).
    """
    if obs.value_m is None:
        return QARTODFlag.MISSING
    if prev is None or prev.value_m is None:
        return QARTODFlag.NOT_EVALUATED
    dt = (obs.source_timestamp - prev.source_timestamp).total_seconds()
    if dt <= 0:
        return QARTODFlag.NOT_EVALUATED
    t = _thresholds_for(obs)
    if dt > t.spike_window_sec * 4:
        # Gap too large for spike detection to be meaningful
        return QARTODFlag.NOT_EVALUATED
    jump = abs(obs.value_m - prev.value_m)
    if jump <= t.spike_threshold_m:
        return QARTODFlag.PASS
    if jump <= t.spike_threshold_m * 2:  # interim 2x multiplier
        return QARTODFlag.SUSPECT
    return QARTODFlag.FAIL


def check_rate_of_change(
    obs: QCObservation,
    prev: QCObservation | None,
) -> QARTODFlag:
    """Rate of change test: rate exceeds quiet-ocean baselines.

    DART: 0.001 m/s; CO-OPS: 0.005 m/s.

    Note: real tsunami signals (0.003+ m/s at DART event-mode) will be
    flagged SUSPECT or FAIL.  This is correct QC behavior - the signal
    IS anomalous.  The downstream anomaly agent processes raw values
    regardless of QC flags; QC metadata aids post-event auditing.
    Event-mode-aware thresholds are deferred to calibration.
    """
    if obs.value_m is None:
        return QARTODFlag.MISSING
    if prev is None or prev.value_m is None:
        return QARTODFlag.NOT_EVALUATED
    dt = (obs.source_timestamp - prev.source_timestamp).total_seconds()
    if dt <= 0:
        return QARTODFlag.NOT_EVALUATED
    rate = abs(obs.value_m - prev.value_m) / dt
    t = _thresholds_for(obs)
    if rate <= t.rate_of_change_max:
        return QARTODFlag.PASS
    if rate <= t.rate_of_change_max * 2:  # interim 2x multiplier
        return QARTODFlag.SUSPECT
    return QARTODFlag.FAIL


def check_flat_line(
    obs: QCObservation,
    history: list[QCObservation],
) -> QARTODFlag:
    """Flat line test: insufficient variation over 1-hour window.

    DART: 0.0001 m over 1 hr; CO-OPS: 0.001 m over 1 hr.
    Returns SUSPECT (never FAIL) when variation is below threshold.
    The QARTOD standard defines two variation thresholds
    (MIN_VAR_WARN -> SUSPECT, MIN_VAR_FAIL -> FAIL) over the same
    time window; the ioos_qc library instead uses a single tolerance
    with two duration thresholds.  This implementation uses a single
    variation threshold with SUSPECT only; adding a FAIL tier is
    deferred to threshold calibration.
    """
    if obs.value_m is None:
        return QARTODFlag.MISSING
    t = _thresholds_for(obs)
    window_start = obs.source_timestamp - timedelta(seconds=t.flat_line_window_sec)

    # Collect values within the window (including current observation)
    values: list[float] = []
    for h in history:
        if (
            h.value_m is not None
            and window_start <= h.source_timestamp <= obs.source_timestamp
        ):
            values.append(h.value_m)
    values.append(obs.value_m)

    if len(values) < 2:
        return QARTODFlag.NOT_EVALUATED

    variation = max(values) - min(values)
    if variation >= t.flat_line_min_variation:
        return QARTODFlag.PASS
    return QARTODFlag.SUSPECT


def check_timing_gap(
    obs: QCObservation,
    prev: QCObservation | None,
) -> QARTODFlag:
    """Timing gap test: gap exceeds 2x expected interval.

    Uses the maximum of the current and previous observation's expected
    interval to avoid false FAILs during DART mode transitions (e.g.,
    standard 15-min -> event 15-sec produces a ~900s gap that should be
    evaluated against the 15-min cadence, not the 15-sec cadence).
    """
    if prev is None:
        return QARTODFlag.NOT_EVALUATED
    dt = (obs.source_timestamp - prev.source_timestamp).total_seconds()
    if dt <= 0:
        return QARTODFlag.NOT_EVALUATED
    t = _thresholds_for(obs)
    # Use the longer of the two intervals so mode transitions don't
    # produce false gap alarms.
    effective_interval = max(obs.expected_interval_sec, prev.expected_interval_sec)
    max_gap = effective_interval * t.timing_gap_factor
    if dt <= max_gap:
        return QARTODFlag.PASS
    # Interim: up to 2x max_gap (= 4x expected) -> SUSPECT; beyond -> FAIL
    if dt <= max_gap * 2:
        return QARTODFlag.SUSPECT
    return QARTODFlag.FAIL


# ---------------------------------------------------------------------------
# Composite: run all checks for a single observation
# ---------------------------------------------------------------------------

def run_all_checks(
    obs: QCObservation,
    prev: QCObservation | None,
    history: list[QCObservation],
) -> QARTODFlags:
    """Run all QARTOD checks for a single observation.

    Args:
        obs: The current observation to check.
        prev: The immediately preceding observation (same station), or None.
        history: Recent history for the same station (for flat-line window).

    Returns:
        QARTODFlags with all per-test results populated.
    """
    timing = check_timing_gap(obs, prev)
    gross_range = check_gross_range(obs)
    spike = check_spike(obs, prev)
    roc = check_rate_of_change(obs, prev)
    flat_line = check_flat_line(obs, history)

    return QARTODFlags(
        timing=timing,
        range=gross_range,
        spike=spike,
        rate_of_change=roc,
        flat_line=flat_line,
        latency=timing,  # latency uses the same timing gap logic
    )


# ---------------------------------------------------------------------------
# Station confidence scoring
# ---------------------------------------------------------------------------

_INDETERMINATE_FLAGS = (
    QARTODFlag.NOT_APPLICABLE,
    QARTODFlag.NOT_EVALUATED,
    QARTODFlag.MISSING,
)


def _evaluated_flags(flags: QARTODFlags) -> list[QARTODFlag]:
    # latency is excluded: it mirrors timing (set in run_all_checks)
    # and including both would double-count the timing check.
    all_flags = [
        flags.timing,
        flags.range,
        flags.rate_of_change,
        flags.spike,
        flags.flat_line,
    ]
    return [f for f in all_flags if f not in _INDETERMINATE_FLAGS]


def count_evaluated_checks(flags: QARTODFlags) -> int:
    """Number of QARTOD checks that produced a definitive result.

    Zero means the record's station_confidence carries no evidence (e.g. the
    first record of a stream, where every check needs history): consumers
    should read confidence together with this count.
    """
    return len(_evaluated_flags(flags))


def compute_station_confidence(flags: QARTODFlags) -> float:
    """Compute station confidence from QARTOD flags.

    confidence = 1.0 - (0.3 * n_suspect + 1.0 * n_fail) / n_evaluated
    Where n_evaluated counts only tests that produced a definitive result
    (excludes NOT_APPLICABLE, NOT_EVALUATED, and MISSING).
    Clamped to [0, 1].  Returns 1.0 by convention when no tests were
    evaluated; count_evaluated_checks distinguishes that no-evidence case.
    """
    evaluated = _evaluated_flags(flags)

    n_evaluated = len(evaluated)
    if n_evaluated == 0:
        return 1.0

    n_suspect = sum(1 for f in evaluated if f == QARTODFlag.SUSPECT)
    n_fail = sum(1 for f in evaluated if f == QARTODFlag.FAIL)

    confidence = 1.0 - (0.3 * n_suspect + 1.0 * n_fail) / n_evaluated
    return max(0.0, min(1.0, confidence))


CONFIDENCE_EXCLUSION_THRESHOLD = 0.5
"""Records with confidence < this value are marked record_usable=False.

Advisory only: the live worker deliberately does not filter records out of
anomaly scoring (genuine tsunami signals trip the same checks); the flag and
confidence are recorded as audit metadata for the duty scientist.
"""

# Maximum history window - matches the largest check window (flat line = 1 hour).
# Add a small buffer to avoid edge-case pruning of boundary observations.
_HISTORY_RETENTION_SEC = 3600.0 + 60.0


def prune_station_history(
    station_history: dict[str, list[QCObservation]],
    now: datetime,
) -> None:
    """Remove history entries older than the flat-line window.

    Prevents unbounded memory growth in long-running agent instances.
    """
    cutoff = now - timedelta(seconds=_HISTORY_RETENTION_SEC)
    for sid in list(station_history):
        hist = station_history[sid]
        station_history[sid] = [h for h in hist if h.source_timestamp >= cutoff]
        if not station_history[sid]:
            del station_history[sid]
