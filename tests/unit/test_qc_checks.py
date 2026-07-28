"""Unit tests for QC QARTOD checks and continuity handling.

Covers all individual QARTOD tests, station confidence scoring,
deterministic sort for out-of-order arrivals, and edge cases
(empty data, all-fail, boundary values, missing values).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hazard_assessment.agents.qc_checks import (
    CONFIDENCE_EXCLUSION_THRESHOLD,
    QCObservation,
    check_flat_line,
    check_gross_range,
    check_rate_of_change,
    check_spike,
    check_timing_gap,
    compute_station_confidence,
    count_evaluated_checks,
    prune_station_history,
    run_all_checks,
    sort_observations,
)
from hazard_assessment.schemas.qc import QARTODFlag, QARTODFlags

_T0 = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)
_HASH = "a" * 64


def _dart_obs(
    *,
    value_m: float | None = 0.0,
    ts: datetime = _T0,
    measurement_type: int = 1,
    event_mode: bool = False,
    station_id: str = "21413",
    payload_sha256: str = _HASH,
) -> QCObservation:
    expected = {1: 900.0, 2: 60.0, 3: 15.0}.get(measurement_type, 900.0)
    return QCObservation(
        source_type="dart",
        station_id=station_id,
        source_timestamp=ts,
        value_m=value_m,
        measurement_type=measurement_type,
        event_mode=event_mode,
        expected_interval_sec=expected,
        payload_sha256=payload_sha256,
    )


def _coops_obs(
    *,
    value_m: float | None = 0.0,
    ts: datetime = _T0,
    station_id: str = "1612340",
    payload_sha256: str = _HASH,
) -> QCObservation:
    return QCObservation(
        source_type="coops",
        station_id=station_id,
        source_timestamp=ts,
        value_m=value_m,
        measurement_type=None,
        event_mode=False,
        expected_interval_sec=60.0,
        payload_sha256=payload_sha256,
    )


# ===================================================================
# Gross range tests
# ===================================================================


class TestGrossRange:
    def test_dart_not_evaluated_raw_height(self) -> None:
        """DART gross range requires detided residual; raw height is NOT_EVALUATED."""
        assert check_gross_range(_dart_obs(value_m=0.3)) == QARTODFlag.NOT_EVALUATED
        assert check_gross_range(_dart_obs(value_m=4541.0)) == QARTODFlag.NOT_EVALUATED

    def test_dart_missing_value(self) -> None:
        assert check_gross_range(_dart_obs(value_m=None)) == QARTODFlag.MISSING

    def test_coops_pass_within_range(self) -> None:
        assert check_gross_range(_coops_obs(value_m=2.5)) == QARTODFlag.PASS

    def test_coops_pass_at_boundary(self) -> None:
        assert check_gross_range(_coops_obs(value_m=3.0)) == QARTODFlag.PASS
        assert check_gross_range(_coops_obs(value_m=-3.0)) == QARTODFlag.PASS

    def test_coops_suspect(self) -> None:
        # 3.5 is outside [-3, 3] but within [-4.5, 4.5]
        assert check_gross_range(_coops_obs(value_m=3.5)) == QARTODFlag.SUSPECT

    def test_coops_fail(self) -> None:
        assert check_gross_range(_coops_obs(value_m=5.0)) == QARTODFlag.FAIL

    def test_coops_zero_value_passes(self) -> None:
        assert check_gross_range(_coops_obs(value_m=0.0)) == QARTODFlag.PASS

    def test_coops_negative_boundary(self) -> None:
        assert check_gross_range(_coops_obs(value_m=-3.0)) == QARTODFlag.PASS
        assert check_gross_range(_coops_obs(value_m=-4.6)) == QARTODFlag.FAIL

    def test_coops_missing_value(self) -> None:
        assert check_gross_range(_coops_obs(value_m=None)) == QARTODFlag.MISSING


# ===================================================================
# Spike tests
# ===================================================================


class TestSpike:
    def test_dart_pass_small_jump(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.05, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.PASS

    def test_dart_pass_at_threshold(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.1, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.PASS

    def test_dart_suspect_above_threshold(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.15, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.SUSPECT

    def test_dart_fail_large_jump(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.25, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.FAIL

    def test_no_previous_not_evaluated(self) -> None:
        obs = _dart_obs(value_m=0.1)
        assert check_spike(obs, None) == QARTODFlag.NOT_EVALUATED

    def test_previous_missing_value_not_evaluated(self) -> None:
        prev = _dart_obs(value_m=None, ts=_T0)
        obs = _dart_obs(value_m=0.1, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.NOT_EVALUATED

    def test_current_missing_value(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=None, ts=_T0 + timedelta(seconds=15))
        assert check_spike(obs, prev) == QARTODFlag.MISSING

    def test_gap_too_large_not_evaluated(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        # DART spike window = 15s, 4x = 60s
        obs = _dart_obs(value_m=0.5, ts=_T0 + timedelta(seconds=61))
        assert check_spike(obs, prev) == QARTODFlag.NOT_EVALUATED

    def test_zero_dt_not_evaluated(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.5, ts=_T0)
        assert check_spike(obs, prev) == QARTODFlag.NOT_EVALUATED

    def test_coops_pass(self) -> None:
        prev = _coops_obs(value_m=0.0, ts=_T0)
        obs = _coops_obs(value_m=0.2, ts=_T0 + timedelta(seconds=60))
        assert check_spike(obs, prev) == QARTODFlag.PASS

    def test_coops_fail(self) -> None:
        prev = _coops_obs(value_m=0.0, ts=_T0)
        obs = _coops_obs(value_m=0.7, ts=_T0 + timedelta(seconds=60))
        assert check_spike(obs, prev) == QARTODFlag.FAIL


# ===================================================================
# Rate of change tests
# ===================================================================


class TestRateOfChange:
    def test_dart_pass(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.01, ts=_T0 + timedelta(seconds=15))
        # rate = 0.01/15 = 0.000667 m/s < 0.001
        assert check_rate_of_change(obs, prev) == QARTODFlag.PASS

    def test_dart_suspect(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.02, ts=_T0 + timedelta(seconds=15))
        # rate = 0.02/15 = 0.00133 m/s > 0.001 but < 0.002
        assert check_rate_of_change(obs, prev) == QARTODFlag.SUSPECT

    def test_dart_fail(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.1, ts=_T0 + timedelta(seconds=15))
        # rate = 0.1/15 = 0.00667 m/s > 0.002
        assert check_rate_of_change(obs, prev) == QARTODFlag.FAIL

    def test_no_previous(self) -> None:
        obs = _dart_obs(value_m=0.1)
        assert check_rate_of_change(obs, None) == QARTODFlag.NOT_EVALUATED

    def test_missing_value(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=None, ts=_T0 + timedelta(seconds=15))
        assert check_rate_of_change(obs, prev) == QARTODFlag.MISSING

    def test_coops_pass(self) -> None:
        prev = _coops_obs(value_m=0.0, ts=_T0)
        obs = _coops_obs(value_m=0.2, ts=_T0 + timedelta(seconds=60))
        # rate = 0.2/60 = 0.00333 m/s < 0.005
        assert check_rate_of_change(obs, prev) == QARTODFlag.PASS

    def test_coops_fail(self) -> None:
        prev = _coops_obs(value_m=0.0, ts=_T0)
        obs = _coops_obs(value_m=1.0, ts=_T0 + timedelta(seconds=60))
        # rate = 1.0/60 = 0.01667 m/s > 0.010
        assert check_rate_of_change(obs, prev) == QARTODFlag.FAIL

    def test_zero_dt(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=100.0, ts=_T0)
        assert check_rate_of_change(obs, prev) == QARTODFlag.NOT_EVALUATED


# ===================================================================
# Flat line tests
# ===================================================================


class TestFlatLine:
    def test_pass_with_variation(self) -> None:
        history = [
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=30)),
            _dart_obs(value_m=0.001, ts=_T0 - timedelta(minutes=15)),
        ]
        obs = _dart_obs(value_m=0.0, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.PASS

    def test_suspect_no_variation(self) -> None:
        history = [
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=30)),
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=15)),
        ]
        obs = _dart_obs(value_m=0.0, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.SUSPECT

    def test_dart_variation_at_threshold(self) -> None:
        # DART threshold = 0.0001m
        history = [
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=30)),
        ]
        obs = _dart_obs(value_m=0.0001, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.PASS

    def test_dart_variation_below_threshold(self) -> None:
        history = [
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=30)),
        ]
        obs = _dart_obs(value_m=0.00005, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.SUSPECT

    def test_coops_threshold(self) -> None:
        # CO-OPS threshold = 0.001m
        history = [
            _coops_obs(value_m=0.0, ts=_T0 - timedelta(minutes=30)),
        ]
        obs = _coops_obs(value_m=0.001, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.PASS

    def test_no_history_not_evaluated(self) -> None:
        obs = _dart_obs(value_m=0.0, ts=_T0)
        assert check_flat_line(obs, []) == QARTODFlag.NOT_EVALUATED

    def test_history_outside_window_not_evaluated(self) -> None:
        # History older than 1 hour is excluded
        history = [
            _dart_obs(value_m=1.0, ts=_T0 - timedelta(hours=2)),
        ]
        obs = _dart_obs(value_m=0.0, ts=_T0)
        assert check_flat_line(obs, history) == QARTODFlag.NOT_EVALUATED

    def test_missing_value(self) -> None:
        obs = _dart_obs(value_m=None, ts=_T0)
        assert check_flat_line(obs, []) == QARTODFlag.MISSING

    def test_history_with_missing_values_ignored(self) -> None:
        history = [
            _dart_obs(value_m=None, ts=_T0 - timedelta(minutes=30)),
            _dart_obs(value_m=0.0, ts=_T0 - timedelta(minutes=15)),
        ]
        obs = _dart_obs(value_m=0.001, ts=_T0)
        # Only one non-None history value + current = 2 values
        assert check_flat_line(obs, history) == QARTODFlag.PASS


# ===================================================================
# Timing gap tests
# ===================================================================


class TestTimingGap:
    def test_dart_standard_pass(self) -> None:
        # DART standard measurement cadence = 900s, 2x = 1800s (30 min)
        prev = _dart_obs(ts=_T0)
        obs = _dart_obs(ts=_T0 + timedelta(minutes=15))
        assert check_timing_gap(obs, prev) == QARTODFlag.PASS

    def test_dart_standard_suspect(self) -> None:
        # Gap > 2x (30 min) but <= 4x (60 min) expected
        prev = _dart_obs(ts=_T0)
        obs = _dart_obs(ts=_T0 + timedelta(minutes=45))
        assert check_timing_gap(obs, prev) == QARTODFlag.SUSPECT

    def test_dart_standard_fail(self) -> None:
        # Gap > 4x expected (> 60 min)
        prev = _dart_obs(ts=_T0)
        obs = _dart_obs(ts=_T0 + timedelta(hours=2))
        assert check_timing_gap(obs, prev) == QARTODFlag.FAIL

    def test_dart_event_mode_pass(self) -> None:
        # Event mode expected = 60s, 2x = 120s
        prev = _dart_obs(ts=_T0, measurement_type=2)
        obs = _dart_obs(ts=_T0 + timedelta(seconds=60), measurement_type=2)
        assert check_timing_gap(obs, prev) == QARTODFlag.PASS

    def test_dart_event_mode_suspect(self) -> None:
        prev = _dart_obs(ts=_T0, measurement_type=2)
        obs = _dart_obs(ts=_T0 + timedelta(seconds=180), measurement_type=2)
        assert check_timing_gap(obs, prev) == QARTODFlag.SUSPECT

    def test_no_previous(self) -> None:
        obs = _dart_obs(ts=_T0)
        assert check_timing_gap(obs, None) == QARTODFlag.NOT_EVALUATED

    def test_coops_pass(self) -> None:
        prev = _coops_obs(ts=_T0)
        obs = _coops_obs(ts=_T0 + timedelta(seconds=60))
        assert check_timing_gap(obs, prev) == QARTODFlag.PASS

    def test_coops_suspect(self) -> None:
        prev = _coops_obs(ts=_T0)
        obs = _coops_obs(ts=_T0 + timedelta(seconds=180))
        assert check_timing_gap(obs, prev) == QARTODFlag.SUSPECT

    def test_coops_fail(self) -> None:
        prev = _coops_obs(ts=_T0)
        obs = _coops_obs(ts=_T0 + timedelta(seconds=500))
        assert check_timing_gap(obs, prev) == QARTODFlag.FAIL


# ===================================================================
# Station confidence scoring
# ===================================================================


class TestStationConfidence:
    """Confidence = 1.0 - (0.3*n_suspect + 1.0*n_fail) / n_tests.

    n_tests counts timing, range, rate_of_change, spike, flat_line
    (5 fields). latency is excluded because it mirrors timing.
    """

    def test_not_evaluated_excluded_from_denominator(self) -> None:
        # One FAIL among two evaluated checks (three NOT_EVALUATED) must be
        # penalized over n=2, not n=5: 1.0 - 1.0/2 = 0.5. Counting
        # NOT_EVALUATED in the denominator would dilute it to 0.8.
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.NOT_EVALUATED,
            rate_of_change=QARTODFlag.NOT_EVALUATED,
            spike=QARTODFlag.FAIL,
            flat_line=QARTODFlag.NOT_EVALUATED,
            latency=QARTODFlag.PASS,
        )
        assert abs(compute_station_confidence(flags) - 0.5) < 1e-9
        assert count_evaluated_checks(flags) == 2

    def test_zero_coverage_is_distinguishable(self) -> None:
        # A record where nothing evaluated reports confidence 1.0 by
        # convention; count_evaluated_checks == 0 is what marks it as
        # carrying no evidence.
        flags = QARTODFlags(
            timing=QARTODFlag.NOT_EVALUATED,
            range=QARTODFlag.NOT_EVALUATED,
            rate_of_change=QARTODFlag.NOT_EVALUATED,
            spike=QARTODFlag.NOT_EVALUATED,
            flat_line=QARTODFlag.MISSING,
            latency=QARTODFlag.NOT_EVALUATED,
        )
        assert compute_station_confidence(flags) == 1.0
        assert count_evaluated_checks(flags) == 0

    def test_all_pass(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        assert compute_station_confidence(flags) == 1.0

    def test_one_suspect(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.SUSPECT,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        # confidence = 1.0 - (0.3 * 1) / 5 = 1.0 - 0.06 = 0.94
        assert abs(compute_station_confidence(flags) - 0.94) < 1e-9

    def test_one_fail(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.FAIL,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        # confidence = 1.0 - (1.0 * 1) / 5 = 0.8
        assert abs(compute_station_confidence(flags) - 0.8) < 1e-9

    def test_all_fail_clamps_to_zero(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.FAIL,
            range=QARTODFlag.FAIL,
            rate_of_change=QARTODFlag.FAIL,
            spike=QARTODFlag.FAIL,
            flat_line=QARTODFlag.FAIL,
            latency=QARTODFlag.FAIL,
        )
        assert compute_station_confidence(flags) == 0.0

    def test_not_applicable_excluded(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.NOT_APPLICABLE,
            latency=QARTODFlag.PASS,
        )
        # 4 applicable tests, all pass -> confidence = 1.0
        assert compute_station_confidence(flags) == 1.0

    def test_all_not_applicable_returns_1(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.NOT_APPLICABLE,
            range=QARTODFlag.NOT_APPLICABLE,
            rate_of_change=QARTODFlag.NOT_APPLICABLE,
            spike=QARTODFlag.NOT_APPLICABLE,
            flat_line=QARTODFlag.NOT_APPLICABLE,
            latency=QARTODFlag.NOT_APPLICABLE,
        )
        assert compute_station_confidence(flags) == 1.0

    def test_below_exclusion_threshold(self) -> None:
        # 3 fails out of 5 tests:
        # confidence = 1.0 - (3 * 1.0) / 5 = 0.4 < 0.5
        flags = QARTODFlags(
            timing=QARTODFlag.FAIL,
            range=QARTODFlag.FAIL,
            rate_of_change=QARTODFlag.FAIL,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.FAIL,  # ignored in confidence
        )
        conf = compute_station_confidence(flags)
        assert conf < CONFIDENCE_EXCLUSION_THRESHOLD

    def test_mixed_suspect_and_fail(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.SUSPECT,
            range=QARTODFlag.FAIL,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.SUSPECT,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        # confidence = 1.0 - (0.3*2 + 1.0*1) / 5 = 1.0 - 1.6/5 = 0.68
        expected = 1.0 - (0.3 * 2 + 1.0) / 5
        assert abs(compute_station_confidence(flags) - expected) < 1e-9

    def test_latency_not_counted(self) -> None:
        """Latency mirrors timing; including both would double-count."""
        flags_timing_fail = QARTODFlags(
            timing=QARTODFlag.FAIL,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            flat_line=QARTODFlag.PASS,
            latency=QARTODFlag.FAIL,  # should NOT affect confidence
        )
        # Only 1 fail (timing) out of 5 tests -> 1.0 - 1.0/5 = 0.8
        assert abs(compute_station_confidence(flags_timing_fail) - 0.8) < 1e-9


# ===================================================================
# Deterministic sort for out-of-order arrivals
# ===================================================================


class TestSortObservations:
    def test_sorts_by_timestamp(self) -> None:
        obs_late = _dart_obs(ts=_T0 + timedelta(minutes=5))
        obs_early = _dart_obs(ts=_T0)
        result = sort_observations([obs_late, obs_early])
        assert result[0].source_timestamp < result[1].source_timestamp

    def test_tie_breaks_by_station_id(self) -> None:
        obs_b = _dart_obs(ts=_T0, station_id="21415")
        obs_a = _dart_obs(ts=_T0, station_id="21413")
        result = sort_observations([obs_b, obs_a])
        assert result[0].station_id == "21413"
        assert result[1].station_id == "21415"

    def test_tie_breaks_by_hash(self) -> None:
        obs_b = _dart_obs(ts=_T0, station_id="21413", payload_sha256="b" * 64)
        obs_a = _dart_obs(ts=_T0, station_id="21413", payload_sha256="a" * 64)
        result = sort_observations([obs_b, obs_a])
        assert result[0].payload_sha256 == "a" * 64

    def test_empty_list(self) -> None:
        assert sort_observations([]) == []

    def test_single_item(self) -> None:
        obs = _dart_obs(ts=_T0)
        result = sort_observations([obs])
        assert len(result) == 1

    def test_determinism_multiple_calls(self) -> None:
        """Same input in different orders produces identical output."""
        obs1 = _dart_obs(ts=_T0, station_id="21413")
        obs2 = _dart_obs(ts=_T0 + timedelta(seconds=15), station_id="21415")
        obs3 = _dart_obs(ts=_T0 - timedelta(seconds=15), station_id="21413")

        result_a = sort_observations([obs1, obs2, obs3])
        result_b = sort_observations([obs3, obs1, obs2])
        result_c = sort_observations([obs2, obs3, obs1])

        assert result_a == result_b == result_c


# ===================================================================
# run_all_checks composite
# ===================================================================


class TestRunAllChecks:
    def test_first_observation_no_prev_dart(self) -> None:
        obs = _dart_obs(value_m=0.1, ts=_T0)
        flags = run_all_checks(obs, prev=None, history=[])
        assert flags.timing == QARTODFlag.NOT_EVALUATED
        assert flags.range == QARTODFlag.NOT_EVALUATED  # DART: no residual
        assert flags.spike == QARTODFlag.NOT_EVALUATED
        assert flags.rate_of_change == QARTODFlag.NOT_EVALUATED

    def test_first_observation_no_prev_coops(self) -> None:
        obs = _coops_obs(value_m=0.5, ts=_T0)
        flags = run_all_checks(obs, prev=None, history=[])
        assert flags.range == QARTODFlag.PASS

    def test_normal_dart_sequence(self) -> None:
        prev = _dart_obs(value_m=0.0, ts=_T0)
        obs = _dart_obs(value_m=0.01, ts=_T0 + timedelta(seconds=15))
        flags = run_all_checks(obs, prev, [prev])
        assert flags.range == QARTODFlag.NOT_EVALUATED  # DART: no residual
        assert flags.spike == QARTODFlag.PASS
        assert flags.rate_of_change == QARTODFlag.PASS

    def test_coops_all_fail_scenario(self) -> None:
        prev = _coops_obs(value_m=0.0, ts=_T0)
        obs = _coops_obs(
            value_m=10.0,  # way out of gross range for CO-OPS
            ts=_T0 + timedelta(seconds=60),
        )
        flags = run_all_checks(obs, prev, [prev])
        assert flags.range == QARTODFlag.FAIL
        assert flags.spike == QARTODFlag.FAIL
        assert flags.rate_of_change == QARTODFlag.FAIL

    def test_missing_value_produces_missing_flags(self) -> None:
        obs = _dart_obs(value_m=None, ts=_T0)
        flags = run_all_checks(obs, prev=None, history=[])
        assert flags.range == QARTODFlag.MISSING
        assert flags.spike == QARTODFlag.MISSING
        assert flags.rate_of_change == QARTODFlag.MISSING
        assert flags.flat_line == QARTODFlag.MISSING

    def test_latency_mirrors_timing(self) -> None:
        prev = _dart_obs(ts=_T0)
        obs = _dart_obs(ts=_T0 + timedelta(hours=6))
        flags = run_all_checks(obs, prev, [prev])
        assert flags.latency == flags.timing


# ===================================================================
# History pruning
# ===================================================================


class TestHistoryPruning:
    def test_prune_removes_old_entries(self) -> None:
        now = _T0
        old = _dart_obs(ts=_T0 - timedelta(hours=3), station_id="21413")
        recent = _dart_obs(ts=_T0 - timedelta(minutes=30), station_id="21413")
        history: dict[str, list[QCObservation]] = {"21413": [old, recent]}
        prune_station_history(history, now)
        assert len(history["21413"]) == 1
        assert history["21413"][0] is recent

    def test_prune_removes_empty_stations(self) -> None:
        now = _T0
        old = _dart_obs(ts=_T0 - timedelta(hours=3), station_id="21413")
        history: dict[str, list[QCObservation]] = {"21413": [old]}
        prune_station_history(history, now)
        assert "21413" not in history
