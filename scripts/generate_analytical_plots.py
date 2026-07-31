#!/usr/bin/env python3
"""Generate 3 new analytical plots for the paper.

1. Detection latency vs distance with tsunami travel-time overlay
2. Noise-level sensitivity triptych (3 heatmaps)
3. Cross-event component score strip chart
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("results")
FIGURES = Path("paper/figures")

# Station metadata: (station_id, distance_km) for each event
# From generate_paper_figures.py station metadata tuples
STATION_DISTANCES: dict[str, dict[str, float]] = {
    # Distances from generate_paper_figures.py *_STATION_META tuples
    "tohoku": {
        "21418": 560, "21401": 990, "21413": 1240, "21419": 1300,
        "46408": 3950, "46402": 4350, "46403": 4830, "46411": 7480,
    },
    "chile": {
        "32412": 2400, "32411": 4920, "46412": 9080,
        "54401": 8740, "46411": 10040, "21413": 15840, "51407": 10740,
    },
    "illapel": {
        "32402": 580, "32401": 1250, "32412": 2110,
        "51426": 9280, "46411": 9740, "46407": 10110, "46403": 12450,
    },
    "iquique": {
        "32401": 292, "32402": 858, "32412": 1652, "32413": 2803,
        "51426": 9900, "46403": 11476,
    },
    "samoa": {
        "51425": 804, "51426": 932, "54401": 1960,
        "51407": 4250, "46411": 7680, "46403": 7720,
    },
}

MAGNITUDES = {
    "tohoku": 9.1, "chile": 8.8, "illapel": 8.3,
    "iquique": 8.2, "samoa": 8.1,
}

T1, T2, T3 = 0.35, 0.60, 0.85


def plot1_latency_vs_distance() -> None:
    """Detection latency vs distance with tsunami/P-wave travel-time overlay."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cmap = plt.cm.plasma_r
    mw_min, mw_max = 8.0, 9.2

    for event, mw in MAGNITUDES.items():
        with open(RESULTS / f"{event}_detection.json") as f:
            data = json.load(f)

        sw = data.get("sliding_window", {})
        dists = STATION_DISTANCES.get(event, {})
        color = cmap((mw - mw_min) / (mw_max - mw_min))

        labeled = False
        for sid, timeline in sw.items():
            dist = dists.get(sid)
            if dist is None:
                continue

            # Find first T1 crossing
            t1_min = None
            for step in timeline:
                if step["ensemble_score"] >= T1:
                    t1_min = step["minutes_from_earthquake"]
                    break

            if t1_min is not None:
                ax.scatter(
                    dist, t1_min,
                    c=[color], s=50, edgecolors="black", linewidths=0.5,
                    zorder=5, label=f"M{mw}" if not labeled else "",
                )
                labeled = True
                # Alternate labels above/below the marker so the dense
                # near-origin cluster (many stations at <1,500 km, <20 min)
                # stays legible.
                n_annotated = getattr(ax, "_n_annotated", 0)
                ax.annotate(
                    sid, (dist, t1_min),
                    fontsize=6, ha="left",
                    va="bottom" if n_annotated % 2 == 0 else "top",
                    xytext=(3, 3) if n_annotated % 2 == 0 else (3, -3),
                    textcoords="offset points",
                    color="gray",
                )
                ax._n_annotated = n_annotated + 1

    # Theoretical tsunami travel time: t = d / 198 m/s
    d_range = np.linspace(100, 16000, 200)
    t_tsunami = d_range / (198 * 60 / 1000)  # km / (m/s * 60s/min / 1000m/km) = min
    t_tsunami = d_range / 0.198 / 60  # d_km / (speed_km_per_min)
    # 198 m/s = 0.198 km/s = 11.88 km/min
    t_tsunami = d_range / 11.88

    ax.plot(d_range, t_tsunami, "b--", linewidth=1.5, alpha=0.7,
            label=r"Tsunami ($\sqrt{gH}$ = 198 m/s)")

    # P-wave travel time: ~8 km/s = 480 km/min
    t_pwave = d_range / 480
    ax.plot(d_range, t_pwave, ":", color="gray", linewidth=1,
            label="P-wave (~8 km/s)")

    # Zero baseline (no FSM threshold lines on this panel: the y axis is
    # minutes to first T1 detection, not score)
    ax.axhline(y=0, color="black", linewidth=0.3)

    ax.set_xlabel("Distance from epicenter (km)", fontsize=10)
    ax.set_ylabel("Minutes from earthquake to first $T_1$ detection", fontsize=10)
    ax.set_title("Detection Latency vs. Distance Across Five Events", fontsize=11)
    ax.set_xlim(0, 16000)
    ax.set_ylim(-10, 500)

    # Legend - deduplicate
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize=8, loc="upper right")

    ax.grid(True, alpha=0.3)

    # Add colorbar for magnitude
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(mw_min, mw_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, aspect=30)
    cbar.set_label("$M_w$", fontsize=10)

    fig.tight_layout()
    out = FIGURES / "fig_latency_vs_distance.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def plot2_noise_triptych() -> None:
    """Noise-level sensitivity triptych: 3 heatmaps at different noise levels."""
    with open(RESULTS / "synthetic_evaluation.json") as f:
        data = json.load(f)

    results = data.get("all_results", data.get("results", []))

    state_order = {"MONITOR": 0, "INVESTIGATE": 1, "ASSESS": 2, "ESCALATE": 3}

    noise_levels = [0.0005, 0.001, 0.005]
    noise_labels = ["Quiet (0.5 mm)", "Reference (1 mm)", "Noisy (5 mm)"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=True)

    for idx, (noise, label) in enumerate(zip(noise_levels, noise_labels)):
        ax = axes[idx]

        # Filter results for this noise level and 60s sampling
        subset = [r for r in results
                  if abs(r["noise_std_m"] - noise) < 1e-6
                  and r["sampling_sec"] == 60]

        if not subset:
            ax.set_title(f"{label}\n(no data)")
            continue

        # Get unique amplitudes and periods
        amps = sorted(set(r["amplitude_m"] for r in subset))
        periods = sorted(set(r["period_min"] for r in subset))

        grid = np.zeros((len(amps), len(periods)))
        for r in subset:
            ai = amps.index(r["amplitude_m"])
            pi = periods.index(r["period_min"])
            state = r.get("max_fsm_state", "MONITOR")
            grid[ai, pi] = state_order.get(state, 0)

        ax.imshow(
            grid, aspect="auto", origin="lower",
            cmap=plt.cm.RdYlGn_r, vmin=0, vmax=3,
            extent=[-0.5, len(periods) - 0.5, -0.5, len(amps) - 0.5],
        )

        ax.set_xticks(range(len(periods)))
        ax.set_xticklabels([str(p) for p in periods], fontsize=7)
        ax.set_xlabel("Tsunami period (min)", fontsize=9)

        if idx == 0:
            ax.set_yticks(range(len(amps)))
            ax.set_yticklabels([f"{a}" for a in amps], fontsize=7)
            ax.set_ylabel("Tsunami amplitude (m)", fontsize=9)

        ax.set_title(label, fontsize=10, fontweight="bold")

        # Annotate cells with state labels
        for r in subset:
            ai = amps.index(r["amplitude_m"])
            pi = periods.index(r["period_min"])
            state = r.get("max_fsm_state", "MON")
            short = {"MONITOR": "MON", "INVESTIGATE": "INV",
                     "ASSESS": "ASS", "ESCALATE": "ESC"}.get(state, "?")
            ax.text(pi, ai, short, ha="center", va="center",
                    fontsize=5.5, fontweight="bold",
                    color="white" if state_order.get(state, 0) >= 2 else "black")

    fig.suptitle("Detection Sensitivity Across Noise Levels (60 s sampling)",
                 fontsize=12, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    out = FIGURES / "fig_noise_sensitivity_triptych.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


def plot3_component_strip() -> None:
    """Cross-event component score strip chart for event-mode stations."""
    fig, ax = plt.subplots(figsize=(8, 5))

    events_order = [
        ("samoa", "Samoa 2009\n($M_w$ 8.1)"),
        ("iquique", "Iquique 2014\n($M_w$ 8.2)"),
        ("illapel", "Illapel 2015\n($M_w$ 8.3)"),
        ("chile", "Chile 2010\n($M_w$ 8.8)"),
        ("tohoku", "Tohoku 2011\n($M_w$ 9.1)"),
    ]

    component_styles = [
        ("threshold_score", "Threshold", "#3b82f6", "o"),
        ("wavelet_score", "Wavelet", "#22c55e", "s"),
        ("bocpd_score", "BOCPD", "#a855f7", "D"),
        ("ensemble_score", "Ensemble", "#ef4444", "^"),
    ]

    y_positions = []
    y_labels = []

    for i, (event, label) in enumerate(events_order):
        with open(RESULTS / f"{event}_detection.json") as f:
            data = json.load(f)

        # Only event-mode stations (not filter_degraded)
        stations = [s for s in data["per_station"] if not s.get("filter_degraded")]

        y_base = i * 1.5
        y_positions.append(y_base)
        y_labels.append(label)

        for j, (key, comp_label, color, marker) in enumerate(component_styles):
            scores = [s[key] for s in stations if s[key] is not None]
            y_jitter = y_base + (j - 1.5) * 0.15

            ax.scatter(
                scores, [y_jitter] * len(scores),
                c=color, marker=marker, s=35, alpha=0.8,
                edgecolors="black", linewidths=0.3,
                label=comp_label if i == 0 else "",
                zorder=5,
            )

        # Add station labels next to ensemble markers (red triangles)
        ens_y = y_base + (3 - 1.5) * 0.15  # ensemble is index 3
        ens_scores = [(s["station_id"], s["ensemble_score"]) for s in stations]
        ens_scores.sort(key=lambda x: x[1])
        n_sta = len(ens_scores)
        for k, (sid, score) in enumerate(ens_scores):
            # For dense high-score clusters, stagger vertically more
            if n_sta > 4 and score > 0.9:
                # Spread labels across a wider vertical range
                rank = sum(1 for _, s2 in ens_scores if s2 > 0.9 and s2 <= score)
                y_off = -10 + rank * 5
                ha, x_off = "right", -4
            else:
                # Normal alternating above/below
                y_off = 4 if k % 2 == 0 else -4
                ha, x_off = "center", 0
            ax.annotate(
                sid, (score, ens_y),
                fontsize=4.5, color="#555555", ha=ha, va="center",
                xytext=(x_off, y_off), textcoords="offset points",
            )

    # Threshold lines
    for thresh, lbl, ls in [(T1, "$T_1$", ":"), (T2, "$T_2$", "--"), (T3, "$T_3$", "-.")]:
        ax.axvline(x=thresh, color="gray", linestyle=ls, linewidth=0.8, alpha=0.6)
        ax.text(thresh + 0.01, len(events_order) * 1.5 - 0.3, lbl,
                fontsize=8, color="gray", va="bottom")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels, fontsize=9)
    ax.set_xlabel("Detection Score", fontsize=10)
    ax.set_xlim(-0.05, 1.1)
    ax.set_title("Component Score Distribution Across Events (Event-Mode Stations Only)",
                 fontsize=11, pad=16)

    ax.legend(fontsize=8, loc="lower right", ncol=2,
              framealpha=0.9, edgecolor="gray")
    ax.grid(True, axis="x", alpha=0.2)

    fig.tight_layout()
    out = FIGURES / "fig_component_score_strip.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    plot1_latency_vs_distance()
    plot2_noise_triptych()
    plot3_component_strip()
    print("All 3 analytical plots generated.")
