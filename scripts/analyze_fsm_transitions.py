#!/usr/bin/env python3
"""Extract FSM state transition timing from sliding-window detection results.

Reads the sliding-window data from tohoku_detection.json and reports
per-station first T1 (INVESTIGATE), T2 (ASSESS), and T3 (ESCALATE)
crossing times relative to the earthquake origin.

Usage:
    python scripts/analyze_fsm_transitions.py [--results results/tohoku_detection.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

T1 = 0.35
T2 = 0.60
T3 = 0.85

# Station distances from Tohoku epicenter (km)
STATION_DISTANCES: dict[str, float] = {
    "21418": 560,
    "21401": 990,
    "21413": 1240,
    "21419": 1300,
    "46408": 3950,
    "46402": 4350,
    "46403": 4830,
    "46411": 7480,
}


def analyze_transitions(
    sliding_window: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Extract first threshold crossing times per station."""
    results = []
    for station_id, windows in sliding_window.items():
        first_t1: float | None = None
        first_t2: float | None = None
        first_t3: float | None = None

        for w in windows:
            minutes = float(w["minutes_from_earthquake"])
            if minutes < 0:
                continue
            score = float(w["ensemble_score"])
            if first_t1 is None and score >= T1:
                first_t1 = minutes
            if first_t2 is None and score >= T2:
                first_t2 = minutes
            if first_t3 is None and score >= T3:
                first_t3 = minutes

        distance = STATION_DISTANCES.get(station_id, 0.0)
        results.append({
            "station_id": station_id,
            "distance_km": distance,
            "first_T1_min": first_t1,
            "first_T2_min": first_t2,
            "first_T3_min": first_t3,
        })

    # Sort by distance
    results.sort(key=lambda x: x["distance_km"])
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="FSM transition analysis")
    parser.add_argument(
        "--results", type=str, default="results/tohoku_detection.json",
    )
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ERROR: {results_path} not found.")
        return

    with open(results_path) as f:
        data = json.load(f)

    sliding_window = data.get("sliding_window", {})
    if not sliding_window:
        print("ERROR: No sliding_window data in results file.")
        return

    transitions = analyze_transitions(sliding_window)

    print(f"\n{'='*80}")
    print("FSM State Transition Timing - Tohoku 2011")
    print(f"{'='*80}")
    print(
        f"{'Station':<10} {'Dist (km)':>10} {'T1 (min)':>10} "
        f"{'T2 (min)':>10} {'T3 (min)':>10}"
    )
    print("-" * 80)

    for t in transitions:
        t1 = f"{t['first_T1_min']:.1f}" if t["first_T1_min"] is not None else "---"
        t2 = f"{t['first_T2_min']:.1f}" if t["first_T2_min"] is not None else "---"
        t3 = f"{t['first_T3_min']:.1f}" if t["first_T3_min"] is not None else "---"
        print(
            f"{t['station_id']:<10} {t['distance_km']:>10.0f} {t1:>10} "
            f"{t2:>10} {t3:>10}"
        )

    # Summary
    t1_times = [t["first_T1_min"] for t in transitions if t["first_T1_min"] is not None]
    t3_times = [t["first_T3_min"] for t in transitions if t["first_T3_min"] is not None]
    print(f"\nStations reaching INVESTIGATE (T1): {len(t1_times)}/{len(transitions)}")
    print(f"Stations reaching ESCALATE (T3): {len(t3_times)}/{len(transitions)}")
    if t1_times:
        print(f"Earliest T1: {min(t1_times):.1f} min, Latest T1: {max(t1_times):.1f} min")
    if t3_times:
        print(f"Earliest T3: {min(t3_times):.1f} min, Latest T3: {max(t3_times):.1f} min")

    # Save
    output = {"event": data["event"], "transitions": transitions}
    output_path = Path("results/fsm_transitions.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
