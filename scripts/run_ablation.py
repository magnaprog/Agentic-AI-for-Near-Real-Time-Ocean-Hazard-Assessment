#!/usr/bin/env python3
"""Component ablation study using existing Tohoku detection results.

Recomputes ensemble scores from per-station and sliding-window component
scores under four weight configurations to quantify each detector's
contribution.

Configurations:
  1. Threshold-only   (1.0, 0.0, 0.0)
  2. Statistical-only  (0.0, 1.0, 0.0) - max(wavelet, BOCPD)
  3. Threshold+Statistical (no ML renormalized)
  4. Full ensemble     (0.50, 0.35, 0.15) - production default

Usage:
    python scripts/run_ablation.py [--results results/tohoku_detection.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hazard_assessment.agents.anomaly_detection import compute_ensemble_score

# FSM thresholds
T1 = 0.35
T2 = 0.60
T3 = 0.85

CONFIGS: dict[str, tuple[float, float, float]] = {
    "threshold_only": (1.0, 0.0, 0.0),
    "statistical_only": (0.0, 1.0, 0.0),
    "threshold_statistical": (0.588, 0.412, 0.0),
    "full_ensemble": (0.50, 0.35, 0.15),
}


def recompute_ensemble(
    threshold_score: float,
    statistical_score: float,
    ml_score: float | None,
    weights: tuple[float, float, float],
) -> float:
    """Recompute ensemble score with given weights."""
    return compute_ensemble_score(
        threshold_score, statistical_score, ml_score, ensemble_weights=weights
    )


def ablate_full_window(
    per_station: list[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    """Ablation on full event-window per-station scores."""
    results: dict[str, list[dict[str, object]]] = {}
    for config_name, weights in CONFIGS.items():
        config_results = []
        for station in per_station:
            new_score = recompute_ensemble(
                float(station["threshold_score"]),
                float(station["statistical_score"]),
                station["ml_score"] if station["ml_score"] is not None else None,
                weights,
            )
            config_results.append({
                "station_id": station["station_id"],
                "ensemble_score": round(new_score, 6),
                "crosses_T1": new_score >= T1,
                "crosses_T2": new_score >= T2,
                "crosses_T3": new_score >= T3,
            })
        results[config_name] = config_results
    return results


def ablate_sliding_window(
    sliding_window: dict[str, list[dict[str, object]]],
) -> dict[str, dict[str, dict[str, float | None]]]:
    """Find first T1 crossing time for each config and station.

    Returns dict[config_name][station_id] = {
        "first_T1_minutes": float | None,
        "first_T2_minutes": float | None,
        "first_T3_minutes": float | None,
    }
    """
    results: dict[str, dict[str, dict[str, float | None]]] = {}
    for config_name, weights in CONFIGS.items():
        station_latencies: dict[str, dict[str, float | None]] = {}
        for station_id, windows in sliding_window.items():
            first_t1: float | None = None
            first_t2: float | None = None
            first_t3: float | None = None
            for w in windows:
                minutes = float(w["minutes_from_earthquake"])
                if minutes < 0:
                    continue  # Pre-earthquake window
                thr = float(w["threshold_score"])
                # Sliding window has individual components but statistical_score
                # is max(wavelet, bocpd). Recompute it.
                wav = float(w["wavelet_score"])
                boc = float(w["bocpd_score"])
                stat = max(wav, boc)
                score = recompute_ensemble(thr, stat, None, weights)
                if first_t1 is None and score >= T1:
                    first_t1 = minutes
                if first_t2 is None and score >= T2:
                    first_t2 = minutes
                if first_t3 is None and score >= T3:
                    first_t3 = minutes
            station_latencies[station_id] = {
                "first_T1_minutes": first_t1,
                "first_T2_minutes": first_t2,
                "first_T3_minutes": first_t3,
            }
        results[config_name] = station_latencies
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Component ablation study")
    parser.add_argument(
        "--results", type=str, default="results/tohoku_detection.json",
        help="Path to existing detection results JSON",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found. Run validate_tohoku.py first.")
        return

    with open(results_path) as f:
        data = json.load(f)

    per_station = data["per_station"]
    sliding_window = data.get("sliding_window", {})

    # Full-window ablation
    full_results = ablate_full_window(per_station)

    # Sliding-window latency ablation
    latency_results = ablate_sliding_window(sliding_window)

    # Print summary
    print(f"\n{'='*90}")
    print("Component Ablation Study - Tohoku 2011")
    print(f"{'='*90}")

    for config_name, weights in CONFIGS.items():
        stations = full_results[config_name]
        n_t1 = sum(1 for s in stations if s["crosses_T1"])
        n_t2 = sum(1 for s in stations if s["crosses_T2"])
        n_t3 = sum(1 for s in stations if s["crosses_T3"])
        n_total = len(stations)

        print(f"\n--- {config_name} (weights={weights}) ---")
        print(f"  Stations crossing T1={T1}: {n_t1}/{n_total}")
        print(f"  Stations crossing T2={T2}: {n_t2}/{n_total}")
        print(f"  Stations crossing T3={T3}: {n_t3}/{n_total}")

        if config_name in latency_results:
            latencies = latency_results[config_name]
            t1_vals = [
                v["first_T1_minutes"]
                for v in latencies.values()
                if v["first_T1_minutes"] is not None
            ]
            if t1_vals:
                print(f"  First T1 crossing: {min(t1_vals):.1f} min (earliest), "
                      f"{max(t1_vals):.1f} min (latest)")

    # Print per-station detail table
    print(f"\n{'='*90}")
    print("Per-Station Ensemble Scores by Configuration")
    print(f"{'='*90}")
    header = f"{'Station':<10}"
    for config_name in CONFIGS:
        header += f" {config_name:>22}"
    print(header)
    print("-" * 90)

    station_ids = [s["station_id"] for s in per_station]
    for sid in station_ids:
        row = f"{sid:<10}"
        for config_name in CONFIGS:
            station_data = next(
                s for s in full_results[config_name] if s["station_id"] == sid
            )
            score = station_data["ensemble_score"]
            row += f" {score:>22.4f}"
        print(row)

    # Threshold sensitivity sweep
    t1_values = [0.25, 0.30, 0.35, 0.40, 0.45]
    sensitivity: list[dict[str, object]] = []
    print(f"\n{'='*90}")
    print("Threshold Sensitivity Analysis - T1 sweep (production weights)")
    print(f"{'='*90}")
    print(f"{'T1':>6} {'Stations >= T1':>16} {'Earliest T1 (min)':>20} {'Latest T1 (min)':>18}")
    print("-" * 70)
    for t1_val in t1_values:
        n_cross = sum(
            1
            for s in per_station
            if float(s["ensemble_score"]) >= t1_val
        )
        # Sliding window: find earliest/latest T1 crossing
        earliest: float | None = None
        latest: float | None = None
        for _sid, windows in sliding_window.items():
            for w in windows:
                minutes = float(w["minutes_from_earthquake"])
                if minutes < 0:
                    continue
                score = float(w["ensemble_score"])
                if score >= t1_val:
                    if earliest is None or minutes < earliest:
                        earliest = minutes
                    if latest is None or minutes > latest:
                        latest = minutes
                    break  # first crossing per station
        e_str = f"{earliest:.1f}" if earliest is not None else "---"
        l_str = f"{latest:.1f}" if latest is not None else "---"
        print(f"{t1_val:>6.2f} {n_cross:>11}/8     {e_str:>16}   {l_str:>16}")
        sensitivity.append({
            "t1": t1_val,
            "stations_crossing": n_cross,
            "earliest_min": earliest,
            "latest_min": latest,
        })

    # Save results
    output = {
        "event": data["event"],
        "configurations": {
            name: {"weights": list(weights)}
            for name, weights in CONFIGS.items()
        },
        "full_window": full_results,
        "detection_latency": latency_results,
        "threshold_sensitivity": sensitivity,
    }
    output_path = Path("results/ablation_results.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
