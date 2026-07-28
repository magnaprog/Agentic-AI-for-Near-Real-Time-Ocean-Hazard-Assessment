#!/usr/bin/env python3
"""Evaluate detection sensitivity on synthetic tidal + tsunami signals.

Generates synthetic DART-like signals (M2 + S2 tidal + Gaussian noise) with
injected tsunami-like waveforms at varying amplitudes and periods, then runs
the full anomaly detection pipeline to characterize the minimum detectable
signal at each FSM threshold.

This is NOT a ROC curve (the system has fixed thresholds, not a swept
decision boundary).  Instead it characterizes system sensitivity: at each
(amplitude, period, noise, sampling rate) configuration, does the ensemble
score exceed T1, T2, or T3?

Output:
    results/synthetic_evaluation.json
    - 2D heatmap data: amplitude x period -> max FSM state reached

Usage:
    python scripts/run_synthetic_evaluation.py

Prerequisites:
    pip install -e .
"""

from __future__ import annotations

import json
import logging
import math
from itertools import product
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.agents.anomaly_agent import AnomalyAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FSM thresholds
T1_INVESTIGATE = 0.35
T2_ASSESS = 0.60
T3_ESCALATE = 0.85

# Evaluation grid
AMPLITUDES_M = [0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
PERIODS_MIN = [5, 10, 15, 25, 45, 90, 120]
NOISE_STDS_M = [0.0005, 0.001, 0.005]
SAMPLING_INTERVALS_SEC = [15, 60]  # 15-sec and 1-min (skip 900s - filter_degraded=True)
# Note: Real DART stations use 15-min standard mode; 15-sec data occurs only
# in short event-mode bursts (~4 hours). The 15-sec calibration scenario here
# is synthetic and represents theoretical sensitivity, not operational conditions.

# Signal generation parameters
CALIBRATION_HOURS = 30 * 24  # 30 days for calibration
EVENT_HOURS = 6              # 6 hours of event data
TSUNAMI_ARRIVAL_HOUR = 1.0   # tsunami arrives 1 hour into event window


def generate_tidal_signal(
    n_hours: float,
    dt_hours: float,
    noise_std: float = 0.001,
    seed: int = 42,
    start_hour: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate synthetic M2+S2 tidal signal with Gaussian noise.

    Amplitudes (0.25 m M2, 0.08 m S2) are representative of deep-ocean
    DART bottom-pressure recorders (Ray, 2013).

    Args:
        n_hours: Duration of signal in hours.
        dt_hours: Sampling interval in hours.
        noise_std: Standard deviation of added Gaussian noise (meters).
        seed: Random seed for reproducibility.
        start_hour: Starting time in hours (for phase-continuous signals).

    Returns:
        (times_hours, signal) arrays.
    """
    times = np.arange(start_hour, start_hour + n_hours, dt_hours)
    omega_m2 = math.radians(28.984104)  # M2 in rad/hr
    omega_s2 = math.radians(30.0)       # S2 in rad/hr

    rng = np.random.default_rng(seed)
    signal = (
        0.25 * np.cos(omega_m2 * times)
        + 0.08 * np.cos(omega_s2 * times)
        + rng.normal(0, noise_std, len(times))
    )
    return times, signal


def inject_tsunami(
    times: NDArray[np.float64],
    signal: NDArray[np.float64],
    arrival_hour: float,
    amplitude_m: float,
    period_min: float,
) -> NDArray[np.float64]:
    """Inject a synthetic tsunami waveform after arrival_hour.

    Tsunami waveform: amplitude * exp(-t/120min) * sin(2*pi*t/period)
    where t is minutes since arrival.
    """
    modified = signal.copy()
    mask = times >= arrival_hour
    t_since = (times[mask] - arrival_hour) * 60.0  # minutes
    decay = np.exp(-t_since / 120.0)
    omega = 2.0 * np.pi / period_min
    modified[mask] += amplitude_m * decay * np.sin(omega * t_since)
    return modified


def evaluate_configuration(
    agent: AnomalyAgent,
    amplitude_m: float,
    period_min: float,
    noise_std: float,
    sampling_sec: float,
) -> dict[str, object]:
    """Run detection pipeline on one synthetic configuration.

    Returns detection results including ensemble score and FSM triggers.
    """
    dt_hours = sampling_sec / 3600.0
    station_id = f"synth_{int(sampling_sec)}s"

    # Generate a single phase-continuous signal covering calibration + event.
    # This avoids phase discontinuity between calibration and event windows
    # that would create spurious tidal residual and inflate BOCPD scores.
    cal_times, cal_signal = generate_tidal_signal(
        n_hours=CALIBRATION_HOURS,
        dt_hours=dt_hours,
        noise_std=noise_std,
        seed=42,
        start_hour=0.0,
    )

    event_times, event_signal = generate_tidal_signal(
        n_hours=EVENT_HOURS,
        dt_hours=dt_hours,
        noise_std=noise_std,
        seed=123,
        start_hour=CALIBRATION_HOURS,  # continues from calibration end
    )

    # Calibrate baseline from calibration data
    agent.calibrate_baseline(station_id, cal_signal, sampling_sec)

    # Inject tsunami into event signal
    # event_times starts at CALIBRATION_HOURS, so arrival is relative to that
    event_with_tsunami = inject_tsunami(
        event_times, event_signal,
        arrival_hour=CALIBRATION_HOURS + TSUNAMI_ARRIVAL_HOUR,
        amplitude_m=amplitude_m,
        period_min=period_min,
    )

    # No seismic events - pure signal detection.
    # Note: this means seismic_context_quiet=True, which applies a 1.3x
    # threshold penalty. Results reflect conservative (worst-case) sensitivity.
    agent.update_seismic_events([])

    try:
        scores, _ = agent.process_station_data(
            station_id=station_id,
            times_hours=event_times,
            values=event_with_tsunami,
            sampling_interval_sec=sampling_sec,
            source_type="dart",
            fit_times_hours=cal_times,
            fit_values=cal_signal,
        )

        ensemble = scores.ensemble_score
        return {
            "amplitude_m": amplitude_m,
            "period_min": period_min,
            "noise_std_m": noise_std,
            "sampling_sec": sampling_sec,
            "ensemble_score": round(ensemble, 6),
            "threshold_score": round(scores.threshold_score, 6),
            "wavelet_score": round(scores.wavelet_score, 6),
            "bocpd_score": round(scores.bocpd_score, 6),
            "statistical_score": round(scores.statistical_score, 6),
            "filter_degraded": scores.filter_degraded,
            "exceeds_t1": ensemble >= T1_INVESTIGATE,
            "exceeds_t2": ensemble >= T2_ASSESS,
            "exceeds_t3": ensemble >= T3_ESCALATE,
            "max_fsm_state": (
                "ESCALATE" if ensemble >= T3_ESCALATE
                else "ASSESS" if ensemble >= T2_ASSESS
                else "INVESTIGATE" if ensemble >= T1_INVESTIGATE
                else "MONITOR"
            ),
        }
    except Exception as exc:
        logger.warning(
            "Failed for amp=%.3f per=%.0f noise=%.4f dt=%.0f: %s",
            amplitude_m, period_min, noise_std, sampling_sec, exc,
        )
        return {
            "amplitude_m": amplitude_m,
            "period_min": period_min,
            "noise_std_m": noise_std,
            "sampling_sec": sampling_sec,
            "error": str(exc),
        }


def main() -> None:
    logger.info("Synthetic detection sensitivity evaluation")
    logger.info("Amplitudes: %s", AMPLITUDES_M)
    logger.info("Periods: %s min", PERIODS_MIN)
    logger.info("Noise levels: %s m", NOISE_STDS_M)
    logger.info("Sampling rates: %s sec", SAMPLING_INTERVALS_SEC)

    total = len(AMPLITUDES_M) * len(PERIODS_MIN) * len(NOISE_STDS_M) * len(SAMPLING_INTERVALS_SEC)
    logger.info("Total configurations: %d", total)

    agent = AnomalyAgent()
    results: list[dict[str, object]] = []
    completed = 0

    for noise_std, sampling_sec in product(NOISE_STDS_M, SAMPLING_INTERVALS_SEC):
        logger.info(
            "Evaluating noise=%.4f m, sampling=%d s",
            noise_std, sampling_sec,
        )
        for amplitude, period in product(AMPLITUDES_M, PERIODS_MIN):
            result = evaluate_configuration(
                agent, amplitude, period, noise_std, sampling_sec,
            )
            results.append(result)
            completed += 1

            if completed % 20 == 0:
                logger.info("Progress: %d/%d (%.0f%%)", completed, total, 100 * completed / total)

    # Build heatmap summary (for the default noise/sampling combination)
    # Primary heatmap: noise=0.001, sampling=60s (1-min DART standard mode)
    heatmap_results = [
        r for r in results
        if not r.get("error")
        and r["noise_std_m"] == 0.001
        and r["sampling_sec"] == 60
    ]

    heatmap: dict[str, dict[str, str]] = {}
    for r in heatmap_results:
        amp_key = f"{r['amplitude_m']:.3f}"
        per_key = f"{r['period_min']}"
        if amp_key not in heatmap:
            heatmap[amp_key] = {}
        heatmap[amp_key][per_key] = r["max_fsm_state"]

    # Print summary for primary configuration
    print(f"\n{'='*80}")
    print("Synthetic Detection Sensitivity - noise=0.001 m, sampling=60 s")
    print(f"{'='*80}")
    header = f"{'Amp (m)':<10}" + "".join(f"{p:>10}" for p in PERIODS_MIN)
    print(header)
    print("-" * 80)
    for amp in AMPLITUDES_M:
        amp_key = f"{amp:.3f}"
        row = f"{amp:<10.3f}"
        for period in PERIODS_MIN:
            per_key = f"{period}"
            state = heatmap.get(amp_key, {}).get(per_key, "?")
            # Abbreviate for display
            abbrev = {
                "MONITOR": "MON",
                "INVESTIGATE": "INV",
                "ASSESS": "ASS",
                "ESCALATE": "ESC",
            }.get(state, state[:3])
            row += f"{abbrev:>10}"
        print(row)

    # Print minimum detectable amplitude per period
    print("\nMinimum detectable amplitude (>= T1) per period:")
    for period in PERIODS_MIN:
        per_key = f"{period}"
        min_amp = None
        for amp in AMPLITUDES_M:
            amp_key = f"{amp:.3f}"
            state = heatmap.get(amp_key, {}).get(per_key, "MONITOR")
            if state != "MONITOR":
                min_amp = amp
                break
        if min_amp is not None:
            print(f"  Period {period:>3} min: >= {min_amp:.3f} m")
        else:
            print(f"  Period {period:>3} min: not detected at any tested amplitude")

    # Summary by sampling rate
    print(f"\n{'='*80}")
    print("Detection rates by sampling interval")
    print(f"{'='*80}")
    for dt in SAMPLING_INTERVALS_SEC:
        dt_results = [r for r in results if not r.get("error") and r["sampling_sec"] == dt]
        n = len(dt_results)
        if n == 0:
            continue
        n_t1 = sum(1 for r in dt_results if r.get("exceeds_t1"))
        n_t2 = sum(1 for r in dt_results if r.get("exceeds_t2"))
        n_t3 = sum(1 for r in dt_results if r.get("exceeds_t3"))
        print(
            f"  {dt:>4}s: {n_t1}/{n} exceed T1 ({100*n_t1/n:.0f}%), "
            f"{n_t2}/{n} exceed T2 ({100*n_t2/n:.0f}%), "
            f"{n_t3}/{n} exceed T3 ({100*n_t3/n:.0f}%)"
        )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "description": "Synthetic detection sensitivity evaluation",
        "parameters": {
            "amplitudes_m": AMPLITUDES_M,
            "periods_min": PERIODS_MIN,
            "noise_stds_m": NOISE_STDS_M,
            "sampling_intervals_sec": SAMPLING_INTERVALS_SEC,
            "calibration_hours": CALIBRATION_HOURS,
            "event_hours": EVENT_HOURS,
            "tsunami_arrival_hour": TSUNAMI_ARRIVAL_HOUR,
        },
        "thresholds": {
            "t1_investigate": T1_INVESTIGATE,
            "t2_assess": T2_ASSESS,
            "t3_escalate": T3_ESCALATE,
        },
        "heatmap_noise001_60s": heatmap,
        "all_results": results,
        "total_configurations": total,
        "total_completed": len([r for r in results if "error" not in r]),
    }
    output_path = results_dir / "synthetic_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
