#!/usr/bin/env python3
"""Profile pipeline component latencies for the latency budget table.

Generates synthetic DART-like data and times each pipeline stage
independently. Results are printed as a table.

Usage:
    python scripts/profile_latency.py [--iterations N]

Requirements:
    pip install -e .   (project must be installed)
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import (
    compute_full_anomaly_score,
    compute_wavelet_energy,
    detide_and_filter,
)
from hazard_assessment.agents.qc_agent import process_observations
from hazard_assessment.agents.qc_checks import QCObservation
from hazard_assessment.orchestrator.states import FSMOrchestrator


def _synthetic_dart_signal(
    n_points: int = 720,
    sampling_sec: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic DART-like signal (tidal + noise).

    Returns (times_hours, values_meters).
    """
    rng = np.random.default_rng(42)
    dt_hours = sampling_sec / 3600.0
    times = np.arange(n_points) * dt_hours
    # Tidal component (M2 + S2)
    tidal = 0.5 * np.sin(2 * np.pi * times / 12.42) + 0.2 * np.sin(
        2 * np.pi * times / 12.0
    )
    noise = rng.normal(0, 0.005, n_points)
    return times, tidal + noise


def _make_qc_observations(
    times_hours: np.ndarray,
    values: np.ndarray,
    sampling_sec: float,
) -> list[QCObservation]:
    """Convert arrays to QCObservation objects for QC profiling."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    return [
        QCObservation(
            source_type="dart",
            station_id="46402",
            source_timestamp=base + timedelta(seconds=float(times_hours[i] * 3600)),
            value_m=float(values[i]),
            measurement_type=1,
            event_mode=True,
            expected_interval_sec=sampling_sec,
            payload_sha256="0" * 64,
        )
        for i in range(len(values))
    ]


def profile_component(
    name: str,
    func: Callable[[], None],
    iterations: int,
) -> dict[str, float]:
    """Time a callable over multiple iterations.

    Returns dict with median, p95, and max in milliseconds.
    """
    times_ms: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000.0
        times_ms.append(elapsed)

    times_ms.sort()
    p95_idx = int(len(times_ms) * 0.95)
    p99_idx = int(len(times_ms) * 0.99)
    return {
        "component": name,
        "median_ms": statistics.median(times_ms),
        "p95_ms": times_ms[min(p95_idx, len(times_ms) - 1)],
        "p99_ms": times_ms[min(p99_idx, len(times_ms) - 1)],
        "max_ms": max(times_ms),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile pipeline latencies")
    parser.add_argument(
        "--iterations", type=int, default=50, help="Number of iterations per component"
    )
    args = parser.parse_args()
    n = args.iterations

    times_hours, values = _synthetic_dart_signal()
    sampling_sec = 15.0
    sampling_hz = 1.0 / sampling_sec

    # Pre-compute detided/filtered for downstream stages
    detided, filtered = detide_and_filter(times_hours, values, sampling_hz)

    results: list[dict[str, float]] = []

    # 1. QC (process_observations on a batch of 10 records)
    qc_obs = _make_qc_observations(times_hours[:10], values[:10], sampling_sec)

    def run_qc() -> None:
        process_observations(qc_obs)

    results.append(profile_component("QC (10 records)", run_qc, n))

    # 2. Detide + bandpass filter
    def run_detide() -> None:
        detide_and_filter(times_hours, values, sampling_hz)

    results.append(profile_component("Detide + bandpass", run_detide, n))

    # 3. Wavelet energy
    def run_wavelet() -> None:
        compute_wavelet_energy(filtered, sampling_sec)

    results.append(profile_component("Wavelet energy", run_wavelet, n))

    # 4. Full anomaly score
    def run_anomaly() -> None:
        compute_full_anomaly_score(
            filtered_signal=filtered,
            detided_residual=detided,
            sampling_interval_sec=sampling_sec,
            threshold_m=0.03,
            baseline_wavelet_energy=1e-6,
            bocpd_prior_precision=1.0,
        )

    results.append(profile_component("Anomaly detection (full)", run_anomaly, n))

    # 5. AnomalyAgent.process_station_data (end-to-end anomaly)
    agent = AnomalyAgent()
    agent.calibrate_baseline("46402", values[:100], sampling_sec)

    def run_agent() -> None:
        agent.process_station_data(
            station_id="46402",
            times_hours=times_hours,
            values=values,
            sampling_interval_sec=sampling_sec,
            processing_time=datetime.now(UTC),
        )

    results.append(profile_component("AnomalyAgent (end-to-end)", run_agent, n))

    # 6. FSM transition (MONITOR -> INVESTIGATE on score >= T1)
    from hazard_assessment.config.settings import ThresholdSettings

    threshold_config = ThresholdSettings().to_threshold_config()

    def run_fsm() -> None:
        fsm = FSMOrchestrator(threshold_config)
        # Move from IDLE -> MONITOR via seismic trigger
        fsm.evaluate_seismic_trigger(
            magnitude=8.0,
            region="pacific",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific"},
        )
        # Now evaluate anomaly score in MONITOR state
        fsm.evaluate_anomaly_score(0.5)

    results.append(profile_component("FSM transition", run_fsm, n))

    # Print results
    print(f"\nLatency profile ({n} iterations per component)\n")
    print(
        f"{'Component':<30} {'Median (ms)':>12} {'p95 (ms)':>12} "
        f"{'p99 (ms)':>12} {'Max (ms)':>12}"
    )
    print("-" * 80)
    for r in results:
        print(
            f"{r['component']:<30} {r['median_ms']:>12.2f} {r['p95_ms']:>12.2f} "
            f"{r['p99_ms']:>12.2f} {r['max_ms']:>12.2f}"
        )

    # Save results as JSON
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "latency_profile.json"
    with open(output_path, "w") as f:
        json.dump({"iterations": n, "components": results}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
