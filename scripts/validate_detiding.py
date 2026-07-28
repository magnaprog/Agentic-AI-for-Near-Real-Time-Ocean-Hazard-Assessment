#!/usr/bin/env python3
"""Validate harmonic detiding quality on real DART calibration data.

Reads 30-day calibration CSVs produced by download_tohoku_dart.py,
fits 8-constituent tidal harmonics, and reports fit quality metrics
per station: M2/S2 amplitudes, residual RMS, and holdout RMS.

Run this before interpreting anomaly scores: if detiding is poor, the
downstream anomaly detectors produce inflated scores even without a tsunami.

Usage:
    python scripts/validate_detiding.py [--data-dir data/tohoku]

Output:
    results/detiding_validation.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from datetime import datetime
from pathlib import Path

import numpy as np

from hazard_assessment.agents.anomaly_detection import (
    TIDAL_FREQUENCIES_RAD_HR,
    fit_tidal_harmonics,
    predict_tide,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_calibration_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float]:
    """Load a calibration CSV and return (times_hours, values, sampling_sec).

    Returns times relative to the first sample, values in meters,
    and median sampling interval in seconds.
    """
    timestamps: list[float] = []
    values: list[float] = []

    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                ts = datetime.fromisoformat(row["timestamp_utc"])
                val = float(row["height_m"])
            except (ValueError, KeyError):
                continue
            # Skip NDBC 9999.000 missing-data sentinels (defense-in-depth;
            # the download script should already filter these).
            if val >= 9999.0:
                continue
            timestamps.append(ts.timestamp())
            values.append(val)

    if len(timestamps) < 100:
        raise ValueError(f"Insufficient calibration data in {path}: {len(timestamps)} rows")

    ts_arr = np.array(timestamps)
    vals_arr = np.array(values)

    # Convert to hours from first sample
    t0 = ts_arr[0]
    times_hours = (ts_arr - t0) / 3600.0

    # Estimate sampling interval
    diffs = np.diff(ts_arr)
    sampling_sec = float(np.median(diffs))

    return times_hours, vals_arr, sampling_sec


def extract_constituent_amplitudes(
    coeffs: np.ndarray,
) -> dict[str, float]:
    """Extract amplitude for each tidal constituent from fit coefficients.

    The design matrix has columns: [1, cos(w1*t), sin(w1*t), cos(w2*t), sin(w2*t), ...]
    Amplitude for constituent k = sqrt(cos_coeff^2 + sin_coeff^2).
    """
    amplitudes: dict[str, float] = {}
    for i, name in enumerate(sorted(TIDAL_FREQUENCIES_RAD_HR.keys())):
        cos_coeff = coeffs[2 * i + 1]
        sin_coeff = coeffs[2 * i + 2]
        amplitudes[name] = float(math.sqrt(cos_coeff**2 + sin_coeff**2))
    return amplitudes


def validate_station(
    station_id: str,
    times_hours: np.ndarray,
    values: np.ndarray,
    sampling_sec: float,
) -> dict[str, object]:
    """Run detiding validation for a single station.

    Returns a dict with fit quality metrics.
    """
    n = len(values)
    span_days = (times_hours[-1] - times_hours[0]) / 24.0

    logger.info(
        "Station %s: %d samples, %.0f s interval, %.1f days span",
        station_id, n, sampling_sec, span_days,
    )

    # Full fit on all data
    coeffs = fit_tidal_harmonics(times_hours, values, clean_input=True)
    predicted = predict_tide(times_hours, coeffs)
    residual = values - predicted

    full_rms = float(np.sqrt(np.mean(residual**2)))
    full_max = float(np.max(np.abs(residual)))

    # Extract constituent amplitudes
    amplitudes = extract_constituent_amplitudes(coeffs)

    # Holdout validation: fit on first 80%, predict last 20%
    split_idx = int(0.8 * n)
    train_times = times_hours[:split_idx]
    train_values = values[:split_idx]
    test_times = times_hours[split_idx:]
    test_values = values[split_idx:]

    holdout_coeffs = fit_tidal_harmonics(train_times, train_values, clean_input=True)
    holdout_predicted = predict_tide(test_times, holdout_coeffs)
    holdout_residual = test_values - holdout_predicted
    holdout_rms = float(np.sqrt(np.mean(holdout_residual**2)))

    return {
        "station_id": station_id,
        "n_samples": n,
        "span_days": round(span_days, 1),
        "sampling_sec": round(sampling_sec, 0),
        "residual_rms_m": round(full_rms, 6),
        "residual_max_m": round(full_max, 6),
        "holdout_rms_m": round(holdout_rms, 6),
        "M2_amplitude_m": round(amplitudes.get("M2", 0.0), 6),
        "S2_amplitude_m": round(amplitudes.get("S2", 0.0), 6),
        "K1_amplitude_m": round(amplitudes.get("K1", 0.0), 6),
        "O1_amplitude_m": round(amplitudes.get("O1", 0.0), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate detiding on DART calibration data")
    parser.add_argument(
        "--data-dir", type=str, default="data/tohoku",
        help="Directory with calibration CSVs from download_tohoku_dart.py",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error("Data directory %s not found. Run download_tohoku_dart.py first.", data_dir)
        return

    cal_files = sorted(data_dir.glob("dart_*_tohoku_2011_calibration.csv"))
    if not cal_files:
        logger.error("No calibration CSV files found in %s", data_dir)
        return

    logger.info("Found %d calibration files in %s", len(cal_files), data_dir)

    results: list[dict[str, object]] = []
    for path in cal_files:
        station_id = path.stem.split("_")[1]
        try:
            times_hours, values, sampling_sec = load_calibration_csv(path)
            result = validate_station(station_id, times_hours, values, sampling_sec)
            results.append(result)
        except Exception as e:
            logger.error("Failed for station %s: %s", station_id, e)

    # Print summary
    print(f"\n{'='*90}")
    print("Detiding Validation Summary")
    print(f"{'='*90}")
    print(
        f"{'Station':<10} {'Span (d)':>10} {'Samples':>8} {'RMS (m)':>10} "
        f"{'Max (m)':>10} {'Hold RMS':>10} {'M2 (m)':>8} {'S2 (m)':>8}"
    )
    print("-" * 90)
    for r in results:
        print(
            f"{r['station_id']:<10} {r['span_days']:>10} {r['n_samples']:>8} "
            f"{r['residual_rms_m']:>10.6f} {r['residual_max_m']:>10.6f} "
            f"{r['holdout_rms_m']:>10.6f} {r['M2_amplitude_m']:>8.4f} "
            f"{r['S2_amplitude_m']:>8.4f}"
        )

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / "detiding_validation.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
