#!/usr/bin/env python3
"""Retrospective validation: run the anomaly detection pipeline on Tohoku 2011 DART data.

Reads CSV files produced by download_tohoku_dart.py (event + calibration windows),
runs the full anomaly detection pipeline (detide, bandpass, wavelet, BOCPD, ensemble),
and reports whether the system would have triggered at each station.

This script:
  - Fits tidal harmonics from the 30-day calibration window when available,
    falling back to the first 30 minutes of event data otherwise
  - Runs sliding-window detection latency analysis under --sliding-window
  - Reports per-station score decomposition and time to FSM threshold crossing

Usage:
    python scripts/validate_tohoku.py [--data-dir data/tohoku]

Prerequisites:
    1. pip install -e .
    2. python scripts/download_tohoku_dart.py --calibration-days 30
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from _seismic_params import DedupStats, load_seismic_params

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import SeismicEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


_EQ = load_seismic_params("tohoku")
TOHOKU_ORIGIN_UTC = _EQ.origin_utc
TOHOKU_MAGNITUDE = _EQ.magnitude
TOHOKU_LAT = _EQ.latitude
TOHOKU_LON = _EQ.longitude

# FSM thresholds (from settings defaults)
T1_MONITOR_TO_INVESTIGATE = 0.35
T2_INVESTIGATE_TO_ASSESS = 0.60
T3_ASSESS_TO_ESCALATE = 0.85


def load_station_csv(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, float, DedupStats, float]:
    """Load a station CSV; return (times_hours, values, sampling_sec, dedup_stats, t0_epoch).

    Returns times relative to the first sample, values in meters,
    median sampling interval in seconds, duplicate-collapse stats, and
    the epoch timestamp of the first admitted sample.
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
            # Skip NDBC 9999.000 missing-data sentinels.
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

    # Convert to hours from first sample
    t0 = ts_arr[0]
    times_hours = (ts_arr - t0) / 3600.0

    # Estimate sampling interval
    diffs = np.diff(ts_arr)
    sampling_sec = float(np.median(diffs))

    return times_hours, vals_arr, sampling_sec, dedup_stats, float(t0)


def validate_station(
    agent: AnomalyAgent,
    station_id: str,
    event_path: Path,
    cal_path: Path | None,
) -> dict[str, object]:
    """Run anomaly detection on a single station's Tohoku data.

    Uses 30-day calibration data for harmonic fitting when available.
    Returns a dict with detection results.
    """
    event_times, event_values, event_dt, event_dedup, _ = load_station_csv(event_path)

    logger.info(
        "Station %s: %d event samples, %.1f s interval, %.1f hours span",
        station_id, len(event_values), event_dt,
        event_times[-1] - event_times[0],
    )

    # Load calibration data for harmonic fitting
    fit_times: np.ndarray | None = None
    fit_values: np.ndarray | None = None
    if cal_path is not None and cal_path.exists():
        cal_times, cal_values, cal_dt, _, _ = load_station_csv(cal_path)
        # Convert calibration times to the same epoch as event times.
        # Both start at 0 (relative to first sample), so offset calibration
        # to negative times so it immediately precedes the event window.
        cal_span_hours = cal_times[-1] - cal_times[0]
        fit_times = cal_times - cal_span_hours  # cal ends at t=0, event starts at t=0
        fit_values = cal_values
        logger.info(
            "  Calibration: %d samples, %.0f s interval, %.1f days span",
            len(cal_values), cal_dt, cal_span_hours / 24.0,
        )
        # Calibrate wavelet baseline from calibration data
        agent.calibrate_baseline(station_id, cal_values, cal_dt)
    else:
        logger.warning("  No calibration data - using event data for tidal fit (suboptimal)")
        # Fallback: calibrate from first 30 minutes of event data
        pre_n = max(10, int(1800 / event_dt))
        pre_n = min(pre_n, len(event_values) // 2)
        agent.calibrate_baseline(station_id, event_values[:pre_n], event_dt)

    # Provide seismic context (Tohoku earthquake)
    agent.update_seismic_events([
        SeismicEvent(
            event_id="us2011tohoku",
            magnitude=TOHOKU_MAGNITUDE,
            origin_time=TOHOKU_ORIGIN_UTC,
            latitude=TOHOKU_LAT,
            longitude=TOHOKU_LON,
        ),
    ])

    # Run full pipeline
    scores, spatial_result = agent.process_station_data(
        station_id=station_id,
        times_hours=event_times,
        values=event_values,
        sampling_interval_sec=event_dt,
        source_type="dart",
        fit_times_hours=fit_times,
        fit_values=fit_values,
        processing_time=TOHOKU_ORIGIN_UTC,
    )

    # Determine FSM transitions
    would_investigate = scores.ensemble_score >= T1_MONITOR_TO_INVESTIGATE
    would_assess = scores.ensemble_score >= T2_INVESTIGATE_TO_ASSESS
    would_escalate = scores.ensemble_score >= T3_ASSESS_TO_ESCALATE

    return {
        "station_id": station_id,
        "n_samples": len(event_values),
        "sampling_sec": round(event_dt, 1),
        "has_calibration": cal_path is not None and cal_path.exists(),
        "dedup": event_dedup.as_dict(),
        "ensemble_score": round(scores.ensemble_score, 6),
        "threshold_score": round(scores.threshold_score, 6),
        "wavelet_score": round(scores.wavelet_score, 6),
        "bocpd_score": round(scores.bocpd_score, 6),
        "statistical_score": round(scores.statistical_score, 6),
        "ml_score": scores.ml_score,
        "filter_degraded": scores.filter_degraded,
        "would_investigate": would_investigate,
        "would_assess": would_assess,
        "would_escalate": would_escalate,
    }


def sliding_window_analysis(
    agent: AnomalyAgent,
    station_id: str,
    event_path: Path,
    cal_path: Path | None,
    step_minutes: int = 5,
    max_minutes: int = 360,
) -> list[dict[str, object]]:
    """Run sliding-window detection latency analysis.

    Processes growing windows of event data to determine when the
    ensemble score first crosses T1, T2, T3.
    """
    event_times, event_values, event_dt, _, event_t0 = load_station_csv(event_path)

    # Load calibration data
    fit_times: np.ndarray | None = None
    fit_values: np.ndarray | None = None
    if cal_path is not None and cal_path.exists():
        cal_times, cal_values, cal_dt, _, _ = load_station_csv(cal_path)
        cal_span_hours = cal_times[-1] - cal_times[0]
        fit_times = cal_times - cal_span_hours  # cal ends at t=0, event starts at t=0
        fit_values = cal_values
        agent.calibrate_baseline(station_id, cal_values, cal_dt)
    else:
        logger.warning("  No calibration data for sliding window - using event data")
        pre_n = max(10, int(1800 / event_dt))
        pre_n = min(pre_n, len(event_values) // 2)
        agent.calibrate_baseline(station_id, event_values[:pre_n], event_dt)

    # Causal seismic context: the sliding replay reveals no seismic
    # product to the detector. The reviewed USGS solution seeds origin
    # alignment only: windows ending at or after origin run with
    # fsm_monitoring=True, mirroring the live worker where the seismic
    # trigger has already moved the FSM to MONITOR or higher, and windows
    # ending before origin run against an empty seismic context. Post-hoc
    # magnitude, geometry, and depth never reach window scoring.
    agent.update_seismic_events([])

    # Offset of the earthquake origin from the first admitted sample,
    # taken from actual timestamps. The archive nominally starts one hour
    # before origin, but the first admitted sample can be later (coarse
    # pre-event sampling, leading gaps), so a fixed 60-minute subtraction
    # misstates event-relative times.
    pre_event_minutes = (TOHOKU_ORIGIN_UTC.timestamp() - event_t0) / 60.0

    timeline: list[dict[str, object]] = []
    samples_per_step = max(1, int(step_minutes * 60 / event_dt))

    for end_idx in range(samples_per_step, len(event_values), samples_per_step):
        window_times = event_times[:end_idx]
        window_values = event_values[:end_idx]
        window_minutes = (window_times[-1] - window_times[0]) * 60.0

        if window_minutes > max_minutes:
            break

        # The causal cutoff is the newest admitted sample; processing time
        # and the FSM monitoring flag both derive from it.
        cutoff_epoch = event_t0 + window_times[-1] * 3600.0
        cutoff_utc = datetime.fromtimestamp(cutoff_epoch, tz=UTC)
        fsm_monitoring = cutoff_epoch >= TOHOKU_ORIGIN_UTC.timestamp()

        try:
            scores, _ = agent.process_station_data(
                station_id=station_id,
                times_hours=window_times,
                values=window_values,
                sampling_interval_sec=event_dt,
                source_type="dart",
                fit_times_hours=fit_times,
                fit_values=fit_values,
                processing_time=cutoff_utc,
                fsm_monitoring=fsm_monitoring,
            )
            timeline.append({
                "minutes_from_start": round(window_minutes, 1),
                "minutes_from_earthquake": round(window_minutes - pre_event_minutes, 1),
                "ensemble_score": round(scores.ensemble_score, 6),
                "threshold_score": round(scores.threshold_score, 6),
                "wavelet_score": round(scores.wavelet_score, 6),
                "bocpd_score": round(scores.bocpd_score, 6),
                "seismic_context_quiet": scores.seismic_context_quiet,
            })
        except Exception as exc:
            logger.warning(
                "Sliding window at %.1f min failed for %s: %s",
                window_minutes, station_id, exc,
            )

    return timeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate anomaly detection on Tohoku 2011 DART data",
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/tohoku",
        help="Directory with station CSV files from download_tohoku_dart.py",
    )
    parser.add_argument(
        "--sliding-window", action="store_true",
        help="Run sliding-window detection latency analysis",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        logger.error(
            "Data directory %s not found. Run download_tohoku_dart.py first.", data_dir,
        )
        return

    # Find event files (new naming convention)
    event_files = sorted(data_dir.glob("dart_*_tohoku_2011_event.csv"))
    if not event_files:
        # Fallback to old naming convention
        event_files = sorted(data_dir.glob("dart_*_tohoku_2011.csv"))

    if not event_files:
        logger.error("No DART CSV files found in %s", data_dir)
        return

    logger.info("Found %d station files in %s", len(event_files), data_dir)

    agent = AnomalyAgent()
    results: list[dict[str, object]] = []
    sliding_results: dict[str, list[dict[str, object]]] = {}

    for event_path in event_files:
        station_id = event_path.stem.split("_")[1]
        cal_path = data_dir / f"dart_{station_id}_tohoku_2011_calibration.csv"
        if not cal_path.exists():
            cal_path = None

        try:
            result = validate_station(agent, station_id, event_path, cal_path)
            results.append(result)

            if args.sliding_window:
                timeline = sliding_window_analysis(
                    agent, station_id, event_path, cal_path,
                )
                sliding_results[station_id] = timeline
        except Exception as e:
            logger.error("Failed for station %s: %s", station_id, e)

    # Print summary
    print(f"\n{'='*100}")
    print("Tohoku 2011 Retrospective Validation Summary")
    print(f"{'='*100}")
    print(
        f"{'Station':<10} {'Score':>8} {'Thresh':>8} {'Wavelet':>8} "
        f"{'BOCPD':>8} {'Stat':>8} {'INV?':>6} {'ASS?':>6} "
        f"{'ESC?':>6} {'Degraded':>8} {'Cal?':>5}"
    )
    print("-" * 100)
    for r in results:
        inv = "YES" if r["would_investigate"] else "no"
        ass = "YES" if r["would_assess"] else "no"
        esc = "YES" if r["would_escalate"] else "no"
        deg = "YES" if r["filter_degraded"] else "no"
        cal = "YES" if r["has_calibration"] else "no"
        print(
            f"{r['station_id']:<10} {r['ensemble_score']:>8.4f} "
            f"{r['threshold_score']:>8.4f} {r['wavelet_score']:>8.4f} "
            f"{r['bocpd_score']:>8.4f} {r['statistical_score']:>8.4f} "
            f"{inv:>6} {ass:>6} {esc:>6} {deg:>8} {cal:>5}"
        )

    # Summary statistics
    n_investigate = sum(1 for r in results if r["would_investigate"])
    n_assess = sum(1 for r in results if r["would_assess"])
    n_escalate = sum(1 for r in results if r["would_escalate"])
    print(f"\nStations triggering INVESTIGATE (score >= {T1_MONITOR_TO_INVESTIGATE}): "
          f"{n_investigate}/{len(results)}")
    print(f"Stations triggering ASSESS (score >= {T2_INVESTIGATE_TO_ASSESS}): "
          f"{n_assess}/{len(results)}")
    print(f"Stations triggering ESCALATE (score >= {T3_ASSESS_TO_ESCALATE}): "
          f"{n_escalate}/{len(results)}")

    if n_investigate > 0:
        print("\nRESULT: System would have detected the Tohoku tsunami signal.")
    else:
        print("\nRESULT: System did NOT detect - investigation needed.")

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(parents=True, exist_ok=True)
    output = {
        "event": "Tohoku 2011",
        "origin_utc": TOHOKU_ORIGIN_UTC.isoformat(),
        "magnitude": TOHOKU_MAGNITUDE,
        "per_station": results,
        "summary": {
            "n_stations": len(results),
            "n_investigate": n_investigate,
            "n_assess": n_assess,
            "n_escalate": n_escalate,
        },
    }
    # Replay-class labels: the per-station scores
    # use the full archived record; the sliding timelines replay observation
    # prefixes conditioned on a known event.
    output["analysis_mode"] = "FULL_RECORD_RETROSPECTIVE_ANALYSIS"
    if sliding_results:
        output["sliding_window"] = sliding_results
        output["sliding_window_replay_mode"] = "POST_HOC_CONDITIONED_OCEAN_REPLAY"

    output_path = results_dir / "tohoku_detection.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
