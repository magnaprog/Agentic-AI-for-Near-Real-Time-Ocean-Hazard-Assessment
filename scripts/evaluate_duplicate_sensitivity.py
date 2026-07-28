#!/usr/bin/env python3
"""Evaluate detector sensitivity to the equal-timestamp duplicate policy.

DART event archives contain duplicate timestamps whose rows often carry
conflicting height values. The frozen replay policy is "first"
(earliest row in native archive order wins, matching the live station
buffer). This script reruns the full-record anomaly scoring used by the
validate_* scripts under each alternative policy and reports how much the
ensemble score and the reached FSM tier move relative to "first".

The per-station conflict fingerprints (timestamp plus every observed value
at that timestamp) are written into the artifact, so an archive re-sort
that changes which record wins shows up as a diff in regenerated results
instead of passing silently.

Usage:
    PYTHONPATH=src python3 scripts/evaluate_duplicate_sensitivity.py

Output:
    results/duplicate_sensitivity.json
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
from _seismic_params import (
    DEDUP_POLICIES,
    DedupStats,
    deduplicate_timeseries,
    load_seismic_params,
)

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import SeismicEvent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVENTS = ["tohoku", "chile", "illapel", "iquique", "samoa"]

# FSM thresholds (from settings defaults), matching the validate_* scripts.
T1_MONITOR_TO_INVESTIGATE = 0.35
T2_INVESTIGATE_TO_ASSESS = 0.60
T3_ASSESS_TO_ESCALATE = 0.85

# validate_chile.py deliberately keeps the early USGS determination for
# station-distance consistency; mirror it so the "first" column reproduces
# the published validation configuration.
CHILE_EARLY_COORDS = (-35.846, -72.719)


def load_csv_with_policy(
    path: Path, policy: str
) -> tuple[np.ndarray, np.ndarray, float, DedupStats, list[dict[str, object]]]:
    """Replicate the validate_* loader with an explicit dedup policy.

    Also returns the conflict fingerprint: every duplicate timestamp whose
    rows disagree in value, with all observed values in archive order.
    """
    timestamps: list[float] = []
    values: list[float] = []
    raw_iso: list[str] = []

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
            raw_iso.append(row["timestamp_utc"])

    # Conflict fingerprint from adjacent equal timestamps (archives are
    # time-sorted; deduplicate_timeseries enforces that).
    conflicts: list[dict[str, object]] = []
    i = 0
    n = len(timestamps)
    while i < n:
        j = i
        while j + 1 < n and timestamps[j + 1] == timestamps[i]:
            j += 1
        if j > i:
            group = values[i : j + 1]
            if max(group) != min(group):
                conflicts.append(
                    {"timestamp_utc": raw_iso[i], "values": group}
                )
        i = j + 1

    ts_dedup, vals_dedup, stats = deduplicate_timeseries(
        timestamps, values, policy=policy
    )

    if len(ts_dedup) < 10:
        raise ValueError(f"Insufficient data in {path}: {len(ts_dedup)} rows")

    ts_arr = np.array(ts_dedup)
    vals_arr = np.array(vals_dedup)

    t0 = ts_arr[0]
    times_hours = (ts_arr - t0) / 3600.0
    sampling_sec = float(np.median(np.diff(ts_arr)))

    return times_hours, vals_arr, sampling_sec, stats, conflicts


def reached_tier(score: float) -> str:
    """Highest FSM tier the full-record ensemble score reaches."""
    if score >= T3_ASSESS_TO_ESCALATE:
        return "ESCALATE"
    if score >= T2_INVESTIGATE_TO_ASSESS:
        return "ASSESS"
    if score >= T1_MONITOR_TO_INVESTIGATE:
        return "INVESTIGATE"
    return "MONITOR"


def score_station(
    event_name: str,
    station_id: str,
    event_path: Path,
    cal_path: Path | None,
    policy: str,
) -> tuple[dict[str, object], DedupStats, list[dict[str, object]]]:
    """Score one station under one dedup policy, mirroring validate_*."""
    params = load_seismic_params(event_name)
    lat, lon = params.latitude, params.longitude
    if event_name == "chile":
        lat, lon = CHILE_EARLY_COORDS

    event_times, event_values, event_dt, stats, conflicts = load_csv_with_policy(
        event_path, policy
    )

    agent = AnomalyAgent()

    fit_times: np.ndarray | None = None
    fit_values: np.ndarray | None = None
    if cal_path is not None and cal_path.exists():
        cal_times, cal_values, cal_dt, _, _ = load_csv_with_policy(
            cal_path, policy
        )
        cal_span_hours = cal_times[-1] - cal_times[0]
        fit_times = cal_times - cal_span_hours
        fit_values = cal_values
        agent.calibrate_baseline(station_id, cal_values, cal_dt)
    else:
        # Fallback used by the validate scripts when calibration is absent.
        pre_n = max(10, int(1800 / event_dt))
        pre_n = min(pre_n, len(event_values) // 2)
        agent.calibrate_baseline(station_id, event_values[:pre_n], event_dt)

    agent.update_seismic_events([
        SeismicEvent(
            event_id=params.event_id,
            magnitude=params.magnitude,
            origin_time=params.origin_utc,
            latitude=lat,
            longitude=lon,
        ),
    ])

    scores, _ = agent.process_station_data(
        station_id=station_id,
        times_hours=event_times,
        values=event_values,
        sampling_interval_sec=event_dt,
        source_type="dart",
        fit_times_hours=fit_times,
        fit_values=fit_values,
        processing_time=params.origin_utc,
    )

    result: dict[str, object] = {
        "ensemble_score": round(scores.ensemble_score, 6),
        "threshold_score": round(scores.threshold_score, 6),
        "wavelet_score": round(scores.wavelet_score, 6),
        "bocpd_score": round(scores.bocpd_score, 6),
        "statistical_score": round(scores.statistical_score, 6),
        "tier": reached_tier(scores.ensemble_score),
    }
    return result, stats, conflicts


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    results_dir = repo_root / "results"
    results_dir.mkdir(exist_ok=True)

    events_out: dict[str, object] = {}
    global_max_delta = 0.0
    tier_flips = 0
    n_stations = 0

    for event_name in EVENTS:
        data_dir = repo_root / "data" / event_name
        event_files = sorted(data_dir.glob(f"dart_*_{event_name}_*_event.csv"))
        if not event_files:
            logger.warning("No DART event files for %s, skipping", event_name)
            continue

        stations_out: dict[str, object] = {}
        event_max_delta = 0.0
        event_any_flip = False

        for event_path in event_files:
            station_id = event_path.stem.split("_")[1]
            cal_path = Path(str(event_path).replace("_event.csv", "_calibration.csv"))
            if not cal_path.exists():
                cal_path = None

            by_policy: dict[str, dict[str, object]] = {}
            first_stats: DedupStats | None = None
            first_conflicts: list[dict[str, object]] = []
            failed = False

            for policy in DEDUP_POLICIES:
                try:
                    result, stats, conflicts = score_station(
                        event_name, station_id, event_path, cal_path, policy
                    )
                except Exception as exc:
                    logger.error(
                        "Failed %s/%s policy=%s: %s",
                        event_name, station_id, policy, exc,
                    )
                    failed = True
                    break
                by_policy[policy] = result
                if policy == "first":
                    first_stats = stats
                    first_conflicts = conflicts

            if failed or first_stats is None:
                stations_out[station_id] = {"error": "scoring failed"}
                continue

            n_stations += 1
            base = float(by_policy["first"]["ensemble_score"])  # type: ignore[arg-type]
            base_tier = by_policy["first"]["tier"]
            max_delta = 0.0
            flip = False
            for policy, result in by_policy.items():
                delta = abs(float(result["ensemble_score"]) - base)  # type: ignore[arg-type]
                max_delta = max(max_delta, delta)
                if result["tier"] != base_tier:
                    flip = True
            if flip:
                tier_flips += 1
            event_any_flip = event_any_flip or flip
            event_max_delta = max(event_max_delta, max_delta)
            global_max_delta = max(global_max_delta, max_delta)

            stations_out[station_id] = {
                "dedup_first": first_stats.as_dict(),
                "conflict_fingerprint": first_conflicts,
                "by_policy": by_policy,
                "max_abs_ensemble_delta_vs_first": round(max_delta, 6),
                "tier_flip": flip,
            }
            logger.info(
                "%s/%s: conflicts=%d max_ensemble_delta=%.6f tier_flip=%s",
                event_name, station_id,
                first_stats.n_conflict_timestamps, max_delta, flip,
            )

        events_out[event_name] = {
            "stations": stations_out,
            "event_max_abs_ensemble_delta": round(event_max_delta, 6),
            "any_tier_flip": event_any_flip,
        }

    output = {
        "description": (
            "Sensitivity of full-record anomaly scoring to the equal-timestamp "
            "duplicate policy in archived DART replay data. The frozen policy "
            "is 'first' (earliest row in native archive order wins, matching "
            "the live station buffer). Alternative policies exist only for "
            "this evaluation. Conflict fingerprints list every duplicate "
            "timestamp whose rows disagree in value, so archive re-sorts that "
            "change the winning record are visible as diffs here."
        ),
        "frozen_policy": "first",
        "policies": list(DEDUP_POLICIES),
        "thresholds": {
            "t1": T1_MONITOR_TO_INVESTIGATE,
            "t2": T2_INVESTIGATE_TO_ASSESS,
            "t3": T3_ASSESS_TO_ESCALATE,
        },
        "events": events_out,
        "summary": {
            "n_stations": n_stations,
            "global_max_abs_ensemble_delta": round(global_max_delta, 6),
            "n_stations_with_tier_flip": tier_flips,
        },
    }

    output_path = results_dir / "duplicate_sensitivity.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")
    print(
        f"Stations: {n_stations}, "
        f"max |ensemble delta| vs 'first': {global_max_delta:.6f}, "
        f"tier flips: {tier_flips}"
    )


if __name__ == "__main__":
    main()
