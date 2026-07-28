#!/usr/bin/env python3
"""Evaluate false positive rate on quiet-period DART data.

Downloads 30 days of DART BPR data from a seismically quiet period
(June 2011 - no tsunamis) for the same 8 stations used in the Tohoku
validation, then runs the full anomaly detection pipeline with NO seismic
context to measure false trigger rates.

The key metric is: how many times does the ensemble score exceed T1 (0.35)
per station during 30 days of quiet data?  This gives false investigations
per station-month.

Usage:
    python scripts/run_false_positive_evaluation.py [--data-dir data/quiet] [--download]

Prerequisites:
    pip install -e .
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import time as time_mod
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

import numpy as np
from _seismic_params import DedupStats

from hazard_assessment.agents.anomaly_agent import AnomalyAgent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Same stations as Tohoku validation
STATIONS = [
    ("21418", "E of epicenter, first arrival"),
    ("21401", "ENE of epicenter"),
    ("21413", "SE, NW Pacific path"),
    ("21419", "NE, near Kuril Trench"),
    ("46408", "NE, W Aleutians"),
    ("46402", "NE, S of Dutch Harbor AK"),
    ("46403", "NE, Eastern Aleutians"),
    ("46411", "ENE, off N California"),
]

# Quiet period: June 1-30, 2011 (no significant tsunamis)
QUIET_START = datetime(2011, 6, 1, 0, 0, 0, tzinfo=UTC)
QUIET_END = datetime(2011, 6, 30, 23, 59, 59, tzinfo=UTC)

# FSM thresholds
T1_INVESTIGATE = 0.35
T2_ASSESS = 0.60
T3_ESCALATE = 0.85

# NDBC URL pattern - same as download_tohoku_dart.py
NDBC_DART_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}t2011.txt.gz&dir=data/historical/dart/"
)

# Window size for pipeline processing (hours)
WINDOW_HOURS = 6


def _fetch_station_data(station_id: str, retry: int = 3) -> str | None:
    """Download raw NDBC DART text for a station."""
    url = NDBC_DART_URL.format(station=station_id)
    for attempt in range(retry):
        try:
            logger.info(
                "Downloading station %s (attempt %d/%d)",
                station_id, attempt + 1, retry,
            )
            with urlopen(url, timeout=60) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")
        except (URLError, OSError) as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, station_id, e)
            if attempt < retry - 1:
                time_mod.sleep(2 ** attempt)
    return None


def _parse_quiet_rows(
    raw: str,
    station_id: str,
) -> list[dict[str, str]]:
    """Parse NDBC DART text and filter rows to the quiet period.

    Only includes standard-mode (T=1, 15-min) data to ensure consistent
    sampling intervals for the false positive evaluation.
    """
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        # Only include standard-mode data (T=1, 15-min intervals)
        if parts[6] != "1":
            continue
        try:
            yr, mo, da, hr, mn, sc = (int(parts[i]) for i in range(6))
            ts = datetime(yr, mo, da, hr, mn, sc, tzinfo=UTC)
        except (ValueError, IndexError):
            continue

        if QUIET_START <= ts <= QUIET_END:
            # Skip NDBC 9999.000 missing-data sentinels.
            try:
                if float(parts[7]) >= 9999.0:
                    continue
            except ValueError:
                continue
            rows.append({
                "station_id": station_id,
                "timestamp_utc": ts.isoformat(),
                "height_m": parts[7],
            })
    return rows


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["station_id", "timestamp_utc", "height_m"],
        )
        writer.writeheader()
        writer.writerows(rows)


def download_quiet_data(output_dir: Path) -> None:
    """Download quiet-period data for all stations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for station_id, desc in STATIONS:
        raw = _fetch_station_data(station_id)
        if raw is None:
            logger.error("Failed to download station %s", station_id)
            continue

        rows = _parse_quiet_rows(raw, station_id)
        if rows:
            out_path = output_dir / f"dart_{station_id}_quiet_june_2011.csv"
            _write_csv(rows, out_path)
            logger.info("Saved %d quiet rows for station %s", len(rows), station_id)
        else:
            logger.warning("No quiet-period data found for station %s", station_id)

        time_mod.sleep(1)  # Be polite to NDBC


def load_station_csv(path: Path) -> tuple[np.ndarray, np.ndarray, float, DedupStats]:
    """Load a station CSV; return (times_hours, values, sampling_sec, dedup_stats)."""
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
            if val >= 9999.0:
                continue
            timestamps.append(ts.timestamp())
            values.append(val)

    # Deduplicate under the frozen equal-time policy (first row wins,
    # matching the live station buffer).
    from _seismic_params import deduplicate_timeseries
    timestamps, values, dedup_stats = deduplicate_timeseries(timestamps, values)

    if len(timestamps) < 10:
        raise ValueError(f"Insufficient data in {path}: {len(timestamps)} rows")

    ts_arr = np.array(timestamps)
    vals_arr = np.array(values)

    t0 = ts_arr[0]
    times_hours = (ts_arr - t0) / 3600.0

    diffs = np.diff(ts_arr)
    sampling_sec = float(np.median(diffs))

    return times_hours, vals_arr, sampling_sec, dedup_stats


def evaluate_station(
    agent: AnomalyAgent,
    station_id: str,
    data_path: Path,
) -> dict[str, object]:
    """Run false-positive evaluation on a single station's quiet data.

    Processes the quiet-period data in sliding windows to count how many
    times the ensemble score exceeds each FSM threshold.
    """
    times_hours, values, dt_sec, dedup_stats = load_station_csv(data_path)
    total_hours = times_hours[-1] - times_hours[0]
    total_days = total_hours / 24.0

    logger.info(
        "Station %s: %d samples, %.0f s interval, %.1f days",
        station_id, len(values), dt_sec, total_days,
    )

    # Use first 15 days as calibration (>= 14.8 days needed to separate
    # M2 and S2 per the Rayleigh criterion), then evaluate on the rest.
    cal_hours = 15 * 24.0
    cal_mask = times_hours <= cal_hours
    cal_values = values[cal_mask]

    if len(cal_values) < 100:
        logger.warning("Insufficient calibration data for %s", station_id)
        return {
            "station_id": station_id,
            "error": "insufficient calibration data",
        }

    agent.calibrate_baseline(station_id, cal_values, dt_sec)

    # No seismic events - this is quiet-period evaluation
    agent.update_seismic_events([])

    # Process in non-overlapping windows of WINDOW_HOURS
    eval_start = cal_hours
    eval_mask = times_hours > eval_start
    eval_times = times_hours[eval_mask]
    eval_values = values[eval_mask]

    if len(eval_times) < 100:
        logger.warning("Insufficient evaluation data for %s", station_id)
        return {
            "station_id": station_id,
            "error": "insufficient evaluation data",
        }

    # Use calibration data for tidal fitting.
    # Calibration times are already in the same epoch as eval times
    # (both relative to the first sample), so pass them directly.
    cal_times = times_hours[cal_mask]
    fit_times = cal_times
    fit_values = cal_values

    window_samples = max(1, int(WINDOW_HOURS * 3600 / dt_sec))
    n_windows = 0
    triggers_t1 = 0
    triggers_t2 = 0
    triggers_t3 = 0
    max_score = 0.0
    all_scores: list[float] = []

    for start_idx in range(0, len(eval_times), window_samples):
        end_idx = min(start_idx + window_samples, len(eval_times))
        if end_idx - start_idx < 10:
            break

        window_times = eval_times[start_idx:end_idx]
        window_values = eval_values[start_idx:end_idx]

        try:
            scores, _ = agent.process_station_data(
                station_id=station_id,
                times_hours=window_times,
                values=window_values,
                sampling_interval_sec=dt_sec,
                source_type="dart",
                fit_times_hours=fit_times,
                fit_values=fit_values,
            )
            n_windows += 1
            score = scores.ensemble_score
            all_scores.append(score)

            if score >= T1_INVESTIGATE:
                triggers_t1 += 1
            if score >= T2_ASSESS:
                triggers_t2 += 1
            if score >= T3_ESCALATE:
                triggers_t3 += 1
            if score > max_score:
                max_score = score

        except Exception as exc:
            logger.warning(
                "Window %d failed for %s: %s", n_windows, station_id, exc,
            )

    eval_days = (eval_times[-1] - eval_times[0]) / 24.0
    eval_months = eval_days / 30.0

    return {
        "station_id": station_id,
        "n_samples": len(values),
        "sampling_sec": round(dt_sec, 0),
        "dedup": dedup_stats.as_dict(),
        "total_days": round(total_days, 1),
        "calibration_days": round(cal_hours / 24.0, 1),
        "evaluation_days": round(eval_days, 1),
        "n_windows": n_windows,
        "triggers_t1": triggers_t1,
        "triggers_t2": triggers_t2,
        "triggers_t3": triggers_t3,
        "triggers_t1_per_month": round(triggers_t1 / eval_months, 2) if eval_months > 0 else 0,
        "triggers_t2_per_month": round(triggers_t2 / eval_months, 2) if eval_months > 0 else 0,
        "max_score": round(max_score, 6),
        "mean_score": round(float(np.mean(all_scores)), 6) if all_scores else 0.0,
        "std_score": round(float(np.std(all_scores)), 6) if all_scores else 0.0,
        "p95_score": round(float(np.percentile(all_scores, 95)), 6) if all_scores else 0.0,
        "p99_score": round(float(np.percentile(all_scores, 99)), 6) if all_scores else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate false positive rate on quiet DART data",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/quiet",
        help="Directory for quiet-period CSV files",
    )
    parser.add_argument(
        "--download", action="store_true",
        help="Download quiet-period data from NDBC (run once)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.download:
        download_quiet_data(data_dir)

    # Find quiet-period files
    quiet_files = sorted(data_dir.glob("dart_*_quiet_june_2011.csv"))
    if not quiet_files:
        logger.error(
            "No quiet-period CSV files found in %s. Run with --download first.",
            data_dir,
        )
        return

    logger.info("Found %d quiet-period files in %s", len(quiet_files), data_dir)

    agent = AnomalyAgent()
    results: list[dict[str, object]] = []

    for path in quiet_files:
        station_id = path.stem.split("_")[1]
        try:
            result = evaluate_station(agent, station_id, path)
            results.append(result)
        except Exception as e:
            logger.error("Failed for station %s: %s", station_id, e)

    # Print summary
    print(f"\n{'='*100}")
    print("False Positive Evaluation - June 2011 (Quiet Period)")
    print(f"{'='*100}")
    print(
        f"{'Station':<10} {'Days':>6} {'Windows':>8} "
        f"{'T1 trig':>8} {'T2 trig':>8} {'T3 trig':>8} "
        f"{'T1/mo':>8} {'Max':>8} {'Mean':>8} {'P95':>8}"
    )
    print("-" * 100)
    for r in results:
        if "error" in r:
            print(f"{r['station_id']:<10} ERROR: {r['error']}")
            continue
        print(
            f"{r['station_id']:<10} {r['evaluation_days']:>6} "
            f"{r['n_windows']:>8} {r['triggers_t1']:>8} "
            f"{r['triggers_t2']:>8} {r['triggers_t3']:>8} "
            f"{r['triggers_t1_per_month']:>8.2f} {r['max_score']:>8.4f} "
            f"{r['mean_score']:>8.4f} {r['p95_score']:>8.4f}"
        )

    # Summary
    valid = [r for r in results if "error" not in r]
    total_t1 = sum(r["triggers_t1"] for r in valid)
    total_t2 = sum(r["triggers_t2"] for r in valid)
    total_t3 = sum(r["triggers_t3"] for r in valid)
    total_windows = sum(r["n_windows"] for r in valid)

    print(f"\nTotal windows evaluated: {total_windows}")
    print(f"Total false triggers at T1 ({T1_INVESTIGATE}): {total_t1}")
    print(f"Total false triggers at T2 ({T2_ASSESS}): {total_t2}")
    print(f"Total false triggers at T3 ({T3_ESCALATE}): {total_t3}")

    if total_t1 == 0:
        print("\nRESULT: Zero false triggers at T1 - excellent specificity.")
    else:
        # Report as triggers per station-month (the operationally meaningful rate)
        total_eval_months = sum(
            r["evaluation_days"] / 30.0 for r in valid
        )
        if total_eval_months > 0:
            rate_per_sm = total_t1 / total_eval_months
            print(f"\nRESULT: {rate_per_sm:.2f} false T1 triggers per station-month.")
        else:
            print(f"\nRESULT: {total_t1} false T1 triggers (evaluation period too short for rate).")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "period": "June 2011 (quiet)",
        "start_utc": QUIET_START.isoformat(),
        "end_utc": QUIET_END.isoformat(),
        "window_hours": WINDOW_HOURS,
        "thresholds": {
            "t1_investigate": T1_INVESTIGATE,
            "t2_assess": T2_ASSESS,
            "t3_escalate": T3_ESCALATE,
        },
        "per_station": results,
        "summary": {
            "n_stations": len(valid),
            "total_windows": total_windows,
            "total_triggers_t1": total_t1,
            "total_triggers_t2": total_t2,
            "total_triggers_t3": total_t3,
        },
    }
    output_path = results_dir / "false_positive_evaluation.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
