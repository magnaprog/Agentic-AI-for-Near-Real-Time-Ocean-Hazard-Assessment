#!/usr/bin/env python3
"""Generate publication-quality figures from evaluation results.

Reads JSON result files from results/ and raw DART CSV data to produce
matplotlib figures as PDFs in paper/figures/.

Figures produced:
  Per-event DART figures (raw waveforms, detided residuals, station map,
  detection bar chart, sliding-window detection timeline) for Tohoku 2011,
  Chile 2010, Illapel 2015, Iquique 2014 and Samoa 2009.
  Shared detector figures: detiding quality, component score decomposition,
  score/waveform overlay, synthetic sensitivity heatmap.
  Synthetic time-series figures: simulation station map and score overlay,
  synthetic waveforms, residuals, score timeline, multistation panel.
  Physics validation: scenario 1 detail and summary.
  Appendix: CO-OPS water-level panels for all five events, the multi-source
  network map, and the dashboard layout schematic.
  The authoritative list is the set of figure functions invoked from main();
  update that function, not this summary, when adding a figure.

Usage:
    python scripts/generate_paper_figures.py

Prerequisites:
    1. Run scripts/run_full_evaluation.sh first to produce results/*.json
    2. pip install -e ".[paper]"   (matplotlib, cartopy, pandas)

Note: the station-map figures use Cartopy Natural Earth features, which
Cartopy downloads into its local cache on first render; a machine with an
empty Cartopy cache needs network access for those figures.
"""

from __future__ import annotations

import csv
import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend - must precede pyplot import

from datetime import UTC

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("paper/figures")
DATA_DIR = Path("data/tohoku")

# FSM thresholds for annotation
T1 = 0.35
T2 = 0.60
T3 = 0.85

# Tohoku earthquake parameters
TOHOKU_LAT = 38.297
TOHOKU_LON = 142.373

# Station metadata: (id, distance_km, lat, lon)
# Coordinates from NDBC station pages (verified against NOAA)
STATION_META: list[tuple[str, int, float, float]] = [
    ("21418", 560, 38.730, 148.800),
    ("21401", 990, 42.617, 152.583),
    ("21413", 1240, 30.515, 152.117),
    ("21419", 1300, 44.401, 155.653),
    ("46408", 3950, 49.677, -169.825),
    ("46402", 4350, 50.913, -164.147),
    ("46403", 4830, 52.647, -156.940),
    ("46411", 7480, 39.337, -127.040),
]


def load_json(filename: str) -> dict | list | None:
    """Load a JSON result file, returning None if missing."""
    path = RESULTS_DIR / filename
    if not path.exists():
        logger.warning("Missing result file: %s", path)
        return None
    with open(path) as f:
        return json.load(f)


def _load_event_csv(station_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load event CSV, returning (minutes_from_eq, height_m) arrays."""
    path = DATA_DIR / f"dart_{station_id}_tohoku_2011_event.csv"
    if not path.exists():
        logger.warning("Missing event CSV: %s", path)
        return None
    minutes: list[float] = []
    heights: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = float(row["seconds_from_origin"])
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            minutes.append(sec / 60.0)
            heights.append(h)
    if not minutes:
        return None
    return np.array(minutes), np.array(heights)



def _iterative_robust_baseline(
    t: np.ndarray, residual: np.ndarray, deg: int = 2,
    n_iter: int = 5, sigma_clip: float = 3.0,
) -> np.ndarray:
    """Fit a polynomial baseline with iterative sigma-clipping.

    At each iteration:
    1. Fit polynomial of degree *deg* to unmasked residual samples.
    2. Compute MAD of fit residuals.
    3. Mask samples deviating > *sigma_clip* x 1.4826 x MAD.
    4. Repeat until convergence or *n_iter* reached.

    Edge clamping: the polynomial is evaluated only within the range
    of unmasked data; beyond the first/last unmasked points, the
    baseline is held constant to prevent Runge-type divergence.

    This rejects tsunami/seismic anomalies while preserving the
    smooth instrument drift, giving a clean baseline overlay.
    """
    mask = np.ones(len(t), dtype=bool)
    poly = np.polyfit(t[mask], residual[mask], deg=deg)

    for _ in range(n_iter):
        fitted = np.polyval(poly, t)
        fit_residual = residual - fitted
        mad = float(np.median(np.abs(fit_residual[mask] -
                                      np.median(fit_residual[mask]))))
        if mad < 1e-12:
            break
        threshold = sigma_clip * 1.4826 * mad
        new_mask = np.abs(fit_residual) <= threshold
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask
        if np.sum(mask) < deg + 1:
            break
        poly = np.polyfit(t[mask], residual[mask], deg=deg)

    result = np.polyval(poly, t)

    # Edge clamping: hold constant beyond the range of clean data
    # to prevent polynomial divergence at boundaries.
    clean_idx = np.where(mask)[0]
    if len(clean_idx) > 0:
        i_first, i_last = clean_idx[0], clean_idx[-1]
        result[:i_first] = result[i_first]
        result[i_last + 1:] = result[i_last]

    # Edge smoothing: the polynomial's higher-order terms can diverge
    # at the extreme edges.  Compute the actual local residual median
    # at each edge and blend the polynomial toward it over the first/last
    # 8% of points.  This anchors the baseline to the observed data at
    # the boundaries while preserving the polynomial fit in the interior.
    n_pts = len(t)
    taper_len = max(3, n_pts // 12)  # ~8% of data
    if taper_len < n_pts // 2:
        # Left edge: anchor to median of first FEW residual points (not
        # the full taper window, which may span a drifting region)
        left_anchor = float(np.median(residual[:min(5, taper_len)]))
        for i in range(taper_len):
            w = 0.5 * (1.0 - np.cos(np.pi * i / taper_len))  # 0->1
            result[i] = left_anchor + w * (result[i] - left_anchor)
        # Right edge: anchor to median of last FEW residual points
        right_anchor = float(np.median(residual[-min(5, taper_len):]))
        for i in range(taper_len):
            idx = n_pts - taper_len + i
            w = 0.5 * (1.0 + np.cos(np.pi * i / taper_len))  # 1->0
            result[idx] = right_anchor + w * (result[idx] - right_anchor)

    return result


def _load_tidal_prediction(
    cal_path: Path, event_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load calibration data, fit tides, return (event_hours, event_heights, predicted).

    Two-stage approach separating tidal fit from baseline fit:

    Stage 1 - Tidal constituents (from 30-day calibration):
        Clean calibration data (spike removal, level-shift correction,
        linear detrend), then fit 8 tidal harmonics via IRLS.  Drift is
        handled twice: the cleaning step removes the linear trend, and
        include_drift=True additionally puts linear and quadratic drift
        columns in the harmonic design matrix, so the drift is absorbed
        by the model rather than left to corrupt the amplitudes/phases.
        Uses only pre-event data (causal).

    Stage 2 - Baseline alignment (retrospective, event window):
        Predict pure tides for the event window and compute the residual
        (observed - predicted), then fit a baseline to that residual and
        add it to the prediction.  Which baseline depends on how many
        event samples there are:
          n >= 200 (event-mode records): a degree-3 polynomial with
            sigma-clipping over the full record, blended at t=0 with a
            rolling median of the pre-event portion.
          15 <= n < 200 (standard-mode records): a wide rolling median
            only, because a polynomial cannot fix the phase error of a
            harmonic prediction sampled every 15 min.
          6 <= n < 15: a degree-2 polynomial with sigma-clipping.
          2 <= n < 6: a constant median offset.
        Every branch reads the event window, so Stage 2 is retrospective.
        Standard practice for retrospective paper figures.

    Returns None if calibration data is insufficient.
    """
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            predict_tide,
        )
    except ImportError:
        return None

    if not cal_path.exists():
        return None

    cal_times: list[float] = []
    cal_values: list[float] = []
    with open(cal_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            cal_times.append(float(row["seconds_from_origin"]) / 3600.0)
            cal_values.append(h)

    if len(cal_times) < 100:
        return None

    cal_t = np.array(cal_times)
    cal_v = np.array(cal_values)

    # Stage 1: Fit tidal harmonics on cleaned calibration WITH drift terms.
    # include_drift=True adds linear+quadratic drift columns to the design
    # matrix, absorbing BPR crystal aging directly into the model (Watts &
    # Kontoyiannis, 1990).  This eliminates the edge divergence that occurs
    # when a drift-free harmonic model is extrapolated into the event window.
    fit_result = fit_tidal_harmonics(
        cal_t, cal_v, clean_input=True, include_drift=True)
    harmonics, drift_t_mean, drift_t_range = fit_result

    ev_hours: list[float] = []
    ev_heights: list[float] = []
    with open(event_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            ev_hours.append(float(row["seconds_from_origin"]) / 3600.0)
            ev_heights.append(h)

    if not ev_hours:
        return None

    ev_t = np.array(ev_hours)
    ev_h = np.array(ev_heights)

    # Predict tidal signal with drift terms (matches the fitting model)
    predicted = predict_tide(
        ev_t, harmonics, include_drift=True,
        drift_t_mean=drift_t_mean, drift_t_range=drift_t_range,
    )

    # Stage 2: Iterative robust baseline alignment.
    # Fit a polynomial to (observed - tide) with iterative
    # sigma-clipping.  Tsunami/seismic spikes are rejected; the smooth
    # instrument drift is preserved.
    residual = ev_h - predicted
    n = len(ev_t)
    if n >= 6:
        if n >= 200:
            # Event-mode: two-region baseline for best accuracy.
            #
            # Pre-event data (t < 0, quiet): rolling median tracks the
            # actual residual drift without polynomial curvature artifacts.
            # Post-event data (t >= 0, tsunami-contaminated): deg=3
            # polynomial with sigma-clipping rejects the anomaly.
            # The two regions are stitched at t=0.
            from scipy.ndimage import median_filter as _medfilt
            pre_mask = ev_t < 0
            n_pre = int(np.sum(pre_mask))

            # Polynomial baseline for the full record (anchored by
            # sigma-clipping to quiet samples)
            deg = 3
            poly_baseline = _iterative_robust_baseline(
                ev_t, residual, deg=deg, n_iter=20, sigma_clip=2.0)

            if n_pre >= 10:
                # Rolling median on pre-event portion
                win = max(5, n_pre // 4)
                if win % 2 == 0:
                    win += 1
                pre_residual = residual[:n_pre]
                pre_med = _medfilt(pre_residual, size=win, mode="nearest")

                # Blend: use rolling median for pre-event, polynomial for
                # post-event, with a 5-point cosine crossfade at the junction
                baseline = poly_baseline.copy()
                fade_len = min(5, n_pre // 2)
                baseline[:n_pre - fade_len] = pre_med[:n_pre - fade_len]
                for k in range(fade_len):
                    idx = n_pre - fade_len + k
                    w = k / fade_len  # 0->1
                    baseline[idx] = (1 - w) * pre_med[idx] + w * poly_baseline[idx]
            else:
                baseline = poly_baseline
        elif n >= 15:
            # Standard-mode (15-min sampling, ~28 points): the harmonic
            # prediction has phase errors that a polynomial cannot fix.
            # Use a wide rolling median instead - it follows the tidal
            # curvature without being pulled by transient anomalies.
            from scipy.ndimage import median_filter
            win = max(5, n // 3)
            if win % 2 == 0:
                win += 1
            baseline = median_filter(residual, size=win, mode="nearest")
        else:
            deg = 2
            baseline = _iterative_robust_baseline(
                ev_t, residual, deg=deg, n_iter=15, sigma_clip=2.5)
        predicted = predicted + baseline
    elif n >= 2:
        # Very few points: constant median offset
        predicted = predicted + float(np.median(residual))

    return ev_t, ev_h, predicted


def fig_tohoku_waveforms() -> None:
    """Raw BPR time series with tidal prediction overlay.

    8-panel vertical stack ordered by epicentral distance.
    Shows observed (black) with IRLS tidal fit (red dashed).
    """
    fig, axes = plt.subplots(len(STATION_META), 1, figsize=(10, 2.2 * len(STATION_META)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, STATION_META):
        result = _load_event_csv(sid)
        if result is None:
            ax.text(0.5, 0.5, f"{sid}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        mins, heights = result
        heights_dm = heights - np.mean(heights)

        _plot_with_gaps(ax, mins, heights_dm)

        # Overlay tidal+drift prediction.  Not causal: the harmonics come
        # from the pre-event calibration window, but _load_tidal_prediction
        # then adds a Stage-2 baseline fitted to the event-window residual
        # itself, which its own docstring marks as retrospective.
        cal_path = DATA_DIR / f"dart_{sid}_tohoku_2011_calibration.csv"
        event_path = DATA_DIR / f"dart_{sid}_tohoku_2011_event.csv"
        tide_data = _load_tidal_prediction(cal_path, event_path)
        if tide_data is not None:
            _ev_t, _ev_h, predicted = tide_data
            pred_dm = predicted - np.mean(heights)
            pred_mins = _ev_t * 60.0
            label = "Predicted tide" if plotted == 0 else None
            ax.plot(pred_mins, pred_dm, "r--", linewidth=0.6, alpha=0.7,
                    label=label)

        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No event CSV data found for waveform figure")
        return

    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Tohoku 2011 (Mw 9.1): Raw Bottom Pressure Records (De-meaned)")
    fig.tight_layout()

    out = FIGURES_DIR / "fig_tohoku_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def _plot_with_gaps(
    ax: plt.Axes,
    mins: np.ndarray,
    residual: np.ndarray,
    *,
    color: str = "k",
    linewidth: float = 0.6,
    gap_threshold_factor: float = 2.0,
) -> None:
    """Plot a residual time series, breaking the line at data gaps and
    hatching the gap regions with diagonal lines.

    A gap is detected when the time step exceeds *gap_threshold_factor*
    times the median sampling interval.
    """
    diffs = np.diff(mins)
    median_dt = float(np.median(diffs))
    gap_mask = diffs > gap_threshold_factor * median_dt

    if not np.any(gap_mask):
        # No gaps - simple plot
        ax.plot(mins, residual, f"{color}-", linewidth=linewidth)
        return

    # Break the line by inserting NaN at gap positions
    plot_mins = mins.copy().astype(float)
    plot_res = residual.copy().astype(float)
    gap_idxs = np.where(gap_mask)[0]

    # Insert NaN after each gap start (iterate in reverse to keep indices valid)
    for gi in reversed(gap_idxs):
        mid = (plot_mins[gi] + plot_mins[gi + 1]) / 2.0
        plot_mins = np.insert(plot_mins, gi + 1, mid)
        plot_res = np.insert(plot_res, gi + 1, np.nan)

    ax.plot(plot_mins, plot_res, f"{color}-", linewidth=linewidth)

    # Hatch gap regions
    for gi in gap_idxs:
        ax.axvspan(
            mins[gi], mins[gi + 1],
            facecolor="none", edgecolor="gray",
            linewidth=0.0, hatch="//", alpha=0.35,
        )



def fig_tohoku_residuals() -> None:
    """Detided residual time series showing tsunami waveform.

    Computes residual = height - predicted_tide using fitted harmonics
    from calibration data, then plots 8-panel stack.
    """
    # Try to import the tidal fitting function
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping residual figure")
        return

    fig, axes = plt.subplots(len(STATION_META), 1, figsize=(10, 2.2 * len(STATION_META)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, STATION_META):
        # Load calibration data for tidal fit
        cal_path = DATA_DIR / f"dart_{sid}_tohoku_2011_calibration.csv"
        if not cal_path.exists():
            ax.text(0.5, 0.5, f"{sid}: no calibration", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        cal_times: list[float] = []
        cal_values: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                sec = float(row["seconds_from_origin"])
                cal_times.append(sec / 3600.0)  # hours
                cal_values.append(h)

        if len(cal_times) < 100:
            ax.text(0.5, 0.5, f"{sid}: insufficient cal data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        # Fit tidal harmonics on calibration data
        cal_t = np.array(cal_times)
        cal_v = np.array(cal_values)
        harmonics = fit_tidal_harmonics(cal_t, cal_v, clean_input=True)

        # Load event data
        result = _load_event_csv(sid)
        if result is None:
            ax.set_ylabel(f"{sid}")
            continue

        mins, heights = result
        # Use seconds_from_origin converted to hours for consistent time basis
        event_path = DATA_DIR / f"dart_{sid}_tohoku_2011_event.csv"
        event_hours_list: list[float] = []
        event_heights_list: list[float] = []
        with open(event_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                sec = float(row["seconds_from_origin"])
                event_hours_list.append(sec / 3600.0)
                event_heights_list.append(h)

        if not event_hours_list:
            ax.set_ylabel(f"{sid}")
            continue

        ev_t = np.array(event_hours_list)
        ev_h = np.array(event_heights_list)

        # Despike: remove single-sample telemetry glitches.
        # 1 m threshold is safe - even the largest tsunami changes
        # <0.3 m between consecutive 15-sec samples.
        if len(ev_h) > 2:
            left = np.concatenate([[ev_h[0]], ev_h[:-1]])
            right = np.concatenate([ev_h[1:], [ev_h[-1]]])
            neighbor_med = np.median(
                np.column_stack([left, ev_h, right]), axis=1)
            spike = np.abs(ev_h - neighbor_med) > 1.0
            if np.any(spike):
                logger.info("Despiked %d glitch(es) from %s event data",
                            int(np.sum(spike)), sid)
                ev_h = np.where(spike, neighbor_med, ev_h)

        predicted = predict_tide(ev_t, harmonics)
        ev_residual = ev_h - predicted

        # Pre-event linear detrend.
        pre_ev_mask = mins < 0
        if np.sum(pre_ev_mask) >= 2:
            bp = np.polyfit(mins[pre_ev_mask], ev_residual[pre_ev_mask], deg=1)
            ev_residual -= np.polyval(bp, mins)

        # Far-field stations (>3000 km): add 120-min centered rolling
        # median to remove residual tidal curvature that the short
        # (~46 min) pre-event window cannot constrain.  Near-field
        # stations skip this to preserve the long-period Mw 9.1 coda.
        if dist_km > 3000:
            import pandas as pd
            dt_min = float(np.median(np.diff(mins)))
            win = max(3, int(120.0 / max(dt_min, 0.1)))
            win = min(win, len(ev_residual) // 2)
            if win % 2 == 0:
                win += 1
            pad = win // 2
            padded = np.pad(ev_residual, pad, mode='reflect')
            baseline = pd.Series(padded).rolling(
                window=win, min_periods=1, center=True,
            ).median().values[pad:-pad]
            ev_residual -= baseline

        hw = min(10, max(3, len(ev_residual) // 6))
        residual, _ = hampel_filter(ev_residual, half_window=hw, threshold=3.0)

        _plot_with_gaps(ax, mins, residual)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No data for residual figure")
        return

    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Tohoku 2011 (Mw 9.1): Detided Residuals")
    fig.tight_layout()

    out = FIGURES_DIR / "fig_tohoku_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def _plot_score_overlay_station(
    axes_pair: tuple,
    sid: str,
    dist_km: float,
    sliding: dict,
    cal_path,
    event_path,
    apply_bandpass_filter,
    fit_tidal_harmonics,
    predict_tide,
    clean_fn=None,
) -> dict:
    """Plot filtered-residual and score panels for one station.

    Returns dict with zoom data: t_onset, ev_m, filtered, has_waveform.

    Parameters
    ----------
    clean_fn : callable, optional
        Function applied to raw_residual before bandpass filtering
        (e.g. Hampel spike removal).  Signature: array -> array.
    """
    ax_resid, ax_score = axes_pair

    has_waveform = False
    filtered = None
    ev_m = None

    if cal_path.exists() and event_path.exists():
        cal_times_h: list[float] = []
        cal_vals: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                cal_times_h.append(float(row["seconds_from_origin"]) / 3600.0)
                cal_vals.append(h)

        ev_times_h: list[float] = []
        ev_vals: list[float] = []
        ev_mins: list[float] = []
        with open(event_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                sec = float(row["seconds_from_origin"])
                ev_times_h.append(sec / 3600.0)
                ev_vals.append(h)
                ev_mins.append(sec / 60.0)

        if len(cal_times_h) >= 100 and len(ev_times_h) >= 30:
            cal_t = np.array(cal_times_h)
            cal_v = np.array(cal_vals)
            ev_t = np.array(ev_times_h)
            ev_h = np.array(ev_vals)
            ev_m = np.array(ev_mins)

            # Keep the longest contiguous fine-cadence segment (dt within
            # 5x the median) to avoid bandpass-filter artifacts on sparse
            # data. Truncating at the FIRST mode switch (the previous
            # behavior) mishandles records that START in coarse standard
            # mode before switching to event mode (e.g. Chile 46412):
            # the switch is at index 0 and everything was discarded.
            # For records that start fine and degrade later, the longest
            # fine run is the leading segment, so behavior is unchanged.
            time_diffs = np.diff(ev_m)
            med_dt = float(np.median(time_diffs))
            coarse = np.where(time_diffs > 5 * med_dt)[0]
            if len(coarse) > 0:
                # Segment boundaries: indices where a coarse gap follows.
                bounds = [0, *(int(i) + 1 for i in coarse), len(ev_m)]
                seg_spans = [
                    (bounds[k], bounds[k + 1])
                    for k in range(len(bounds) - 1)
                ]
                lo, hi = max(seg_spans, key=lambda ab: ab[1] - ab[0])
                ev_t, ev_h, ev_m = ev_t[lo:hi], ev_h[lo:hi], ev_m[lo:hi]

            harmonics = fit_tidal_harmonics(cal_t, cal_v, clean_input=True)
            predicted = predict_tide(ev_t, harmonics)
            raw_residual = ev_h - predicted

            if clean_fn is not None:
                raw_residual = clean_fn(raw_residual)

            dt_s = float(np.median(np.diff(ev_m))) * 60.0
            sampling_rate = 1.0 / dt_s
            filtered = apply_bandpass_filter(raw_residual, sampling_rate)

            ax_resid.plot(ev_m, filtered, "k-", linewidth=0.6)
            has_waveform = True

    if not has_waveform:
        ax_resid.text(0.5, 0.5, f"{sid}: no data", transform=ax_resid.transAxes,
                      ha="center", va="center", fontsize=9)

    station_label = f"{sid} ({dist_km:,} km)"
    ax_resid.axvline(x=0, color="red", linewidth=1, alpha=0.7)
    ax_resid.set_ylabel("Filtered (m)", fontsize=8)
    ax_resid.text(0.02, 0.85, station_label,
                  transform=ax_resid.transAxes, fontsize=8, va="top",
                  bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                        "alpha": 0.8})
    ax_resid.tick_params(labelsize=7)

    # --- Compute score onset time (needed for all panels) ---
    timeline = sliding[sid]
    sw_mins = [t["minutes_from_earthquake"] for t in timeline]
    score_times = [
        t["minutes_from_earthquake"]
        for t in timeline
        if t["ensemble_score"] > 0.01
    ]
    p_wave_min = dist_km / (8.0 * 60)  # P-wave travel time in minutes
    t_onset = score_times[0] if score_times else None

    # --- Add score onset line to filtered panel ---
    if t_onset is not None:
        ax_resid.axvline(
            x=t_onset, color="#2196F3", linewidth=1,
            linestyle="--", alpha=0.8,
        )

    # --- Score panel ---
    ax_score.plot(sw_mins, [t["threshold_score"] for t in timeline],
                  label="Threshold", color="#4CAF50", linewidth=1)
    ax_score.plot(sw_mins, [t["wavelet_score"] for t in timeline],
                  label="Wavelet", color="#FF9800", linewidth=1)
    ax_score.plot(sw_mins, [t["bocpd_score"] for t in timeline],
                  label="BOCPD", color="#9C27B0", linewidth=1)
    ax_score.plot(sw_mins, [t["ensemble_score"] for t in timeline],
                  label="Ensemble", color="#2196F3", linewidth=2)
    ax_score.axvline(x=0, color="red", linewidth=1, alpha=0.7)
    if t_onset is not None:
        ax_score.axvline(
            x=t_onset, color="#2196F3", linewidth=1,
            linestyle="--", alpha=0.8,
        )
    ax_score.axhline(y=T1, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
    ax_score.axhline(y=T2, color="gray", linestyle="-.", alpha=0.5, linewidth=0.7)
    ax_score.axhline(y=T3, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)
    ax_score.text(1.01, T1, "$T_1$", transform=ax_score.get_yaxis_transform(),
                  fontsize=6, va="center", color="gray")
    ax_score.text(1.01, T2, "$T_2$", transform=ax_score.get_yaxis_transform(),
                  fontsize=6, va="center", color="gray")
    ax_score.text(1.01, T3, "$T_3$", transform=ax_score.get_yaxis_transform(),
                  fontsize=6, va="center", color="gray")
    ax_score.set_ylim(0, 1.05)
    ax_score.set_ylabel("Score", fontsize=8)
    ax_score.tick_params(labelsize=7)
    ax_score.legend(loc="lower right", fontsize=7, ncol=4, framealpha=0.9)

    return {
        "t_onset": t_onset,
        "ev_m": ev_m,
        "filtered": filtered,
        "has_waveform": has_waveform,
        "p_wave_min": p_wave_min,
    }


def _setup_onset_zoom(ax_zoom, ax_ref, zoom_info: dict) -> None:
    """Reposition *ax_zoom* to 1/4 width at onset position and plot zoom data.

    Call **after** ``tight_layout`` so that ``ax_ref.get_position()`` and
    ``ax_ref.get_xlim()`` return the final layout values.  *ax_zoom* is
    an existing placeholder axes whose vertical position is already set
    by GridSpec; this function shrinks it to 1/4 width and shifts it
    horizontally so the score-onset dashed line aligns with the
    full-width panels above.
    """
    if not zoom_info["has_waveform"] or zoom_info["filtered"] is None:
        ax_zoom.set_visible(False)
        return
    t_onset = zoom_info["t_onset"]
    ev_m = zoom_info["ev_m"]
    filtered = zoom_info["filtered"]
    p_wave_min = zoom_info["p_wave_min"]

    if t_onset is None and ev_m is None:
        ax_zoom.set_visible(False)
        return

    t_center = t_onset if t_onset is not None else p_wave_min

    # Zoom window: +/-5 min around onset, minimum 8 min wide
    zoom_lo = max(ev_m[0], t_center - 5)
    zoom_hi = min(ev_m[-1], t_center + 5)
    if zoom_hi - zoom_lo < 8:
        mid = (zoom_lo + zoom_hi) / 2
        zoom_lo = mid - 4
        zoom_hi = mid + 4

    # --- Reposition to 1/4 width aligned on onset ---
    ref_pos = ax_ref.get_position()
    zoom_pos = ax_zoom.get_position()
    xlim = ax_ref.get_xlim()
    x_range = xlim[1] - xlim[0]
    if x_range <= 0:
        ax_zoom.set_visible(False)
        return

    onset_frac = (t_center - xlim[0]) / x_range
    zoom_w = ref_pos.width * 0.25

    # Where onset falls within the zoom xlim (fraction of zoom range)
    onset_in_zoom = (t_center - zoom_lo) / (zoom_hi - zoom_lo)

    # Place zoom so onset_in_zoom aligns with onset_frac in full panel
    onset_fig_x = ref_pos.x0 + onset_frac * ref_pos.width
    zoom_left = onset_fig_x - onset_in_zoom * zoom_w
    zoom_left = max(ref_pos.x0,
                    min(zoom_left, ref_pos.x0 + ref_pos.width - zoom_w))

    ax_zoom.set_position([zoom_left, zoom_pos.y0, zoom_w, zoom_pos.height])

    # --- Plot zoom data ---
    mask = (ev_m >= zoom_lo) & (ev_m <= zoom_hi)
    if np.sum(mask) > 3:
        zoomed_mm = filtered[mask] * 1000
        ax_zoom.plot(ev_m[mask], zoomed_mm, "k-", linewidth=0.8)
        ax_zoom.set_xlim(zoom_lo, zoom_hi)
        ymax = max(np.max(np.abs(zoomed_mm)), 0.5) * 1.3
        ax_zoom.set_ylim(-ymax, ymax)

    ax_zoom.axhline(y=0, color="gray", linewidth=0.3, alpha=0.3)
    if t_onset is not None:
        ax_zoom.axvline(
            x=t_onset, color="#2196F3", linewidth=1,
            linestyle="--", alpha=0.8,
        )
        # Annotate with onset time
        ax_zoom.text(
            t_onset, ax_zoom.get_ylim()[1] * 0.85,
            f"  +{t_onset:.1f} min",
            fontsize=5, color="#2196F3", va="top", ha="left",
        )

    ax_zoom.set_ylabel("Filtered\nSignal (mm)", fontsize=6, labelpad=1)
    ax_zoom.tick_params(labelsize=5)


def fig_score_waveform_overlay(tohoku_data: dict) -> None:
    """Score + waveform overlay for representative stations.

    Two full-width panels per station (filtered, scores) plus a
    1/4-width onset zoom panel positioned below the score panel
    at the score-onset time.
    """
    try:
        from hazard_assessment.agents.anomaly_detection import (
            apply_bandpass_filter,
            fit_tidal_harmonics,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping overlay figure")
        return

    sliding = tohoku_data.get("sliding_window", {})
    if not sliding:
        logger.warning("No sliding window data for overlay figure")
        return

    target_stations = []
    for sid in ["21418", "21401"]:
        if sid in sliding:
            target_stations.append(sid)
            break
    for sid in ["46411", "46402"]:
        if sid in sliding:
            target_stations.append(sid)
            break

    if not target_stations:
        logger.warning("No target stations for overlay figure")
        return

    n_rows = 3  # filtered, scores, zoom-placeholder per station
    n_stations = len(target_stations)
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )

    zoom_infos = []
    for i, sid in enumerate(target_stations):
        dist_km = next((d for s, d, _, _ in STATION_META if s == sid), 0)
        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        # Hide placeholder zoom axes (will be repositioned after layout)
        ax_zoom.set_visible(False)

        cal_path = DATA_DIR / f"dart_{sid}_tohoku_2011_calibration.csv"
        event_path = DATA_DIR / f"dart_{sid}_tohoku_2011_event.csv"

        info = _plot_score_overlay_station(
            (ax_filt, ax_score), sid, dist_km, sliding,
            cal_path, event_path,
            apply_bandpass_filter, fit_tidal_harmonics, predict_tide,
        )
        ax_filt.set_xlim(right=300)
        zoom_infos.append((i, info))

    # X-label on last score panel (not zoom placeholder)
    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title("Tohoku 2011 (Mw 9.1): Bandpass-Filtered Residual and Detection Scores")
    fig.tight_layout()

    # Reposition zoom placeholders to 1/4-width onset panels
    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_score_waveform_overlay.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_station_map() -> None:
    """Station location map with coastlines via _station_map_plain (cartopy)."""
    _fig_station_map_plain()


def _great_circle_points(
    lat1: float, lon1: float, lat2: float, lon2: float, n: int = 50
) -> tuple[list[float], list[float]]:
    """Interpolate n+1 points along great-circle path using spherical slerp.

    Returns (lons, lats) lists.
    """
    import math
    lat1r, lon1r = math.radians(lat1), math.radians(lon1)
    lat2r, lon2r = math.radians(lat2), math.radians(lon2)
    d = math.acos(min(1.0, max(-1.0,
        math.sin(lat1r) * math.sin(lat2r) +
        math.cos(lat1r) * math.cos(lat2r) * math.cos(lon2r - lon1r))))
    if d < 1e-10:
        return [lon1, lon2], [lat1, lat2]
    sin_d = math.sin(d)
    if sin_d < 1e-6:  # near-antipodal guard
        return [lon1, lon2], [lat1, lat2]
    lats: list[float] = []
    lons: list[float] = []
    for i in range(n + 1):
        f = i / n
        A = math.sin((1 - f) * d) / sin_d
        B = math.sin(f * d) / sin_d
        x = A * math.cos(lat1r) * math.cos(lon1r) + B * math.cos(lat2r) * math.cos(lon2r)
        y = A * math.cos(lat1r) * math.sin(lon1r) + B * math.cos(lat2r) * math.sin(lon2r)
        z = A * math.sin(lat1r) + B * math.sin(lat2r)
        lats.append(math.degrees(math.atan2(z, math.sqrt(x * x + y * y))))
        lons.append(math.degrees(math.atan2(y, x)))
    return lons, lats


def _station_map_plain(
    station_meta: list[tuple[str, int, float, float]],
    epicenter_lat: float,
    epicenter_lon: float,
    title: str,
    outfile: str,
    label_offsets: dict[str, tuple[float, float, str]] | None = None,
    lat_margin_bottom: float | None = None,
) -> None:
    """Station map with coastlines using cartopy PlateCarree projection.

    Uses a Pacific-centered central_longitude to avoid date-line clipping.
    label_offsets values are (lon_offset_deg, lat_offset_deg, ha).
    """
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    # Choose central longitude as midpoint of all longitudes
    all_lons = [epicenter_lon] + [lon for _, _, _, lon in station_meta]
    # Shift to 0-360 to find a good center for Pacific views
    shifted = [lon % 360 for lon in all_lons]
    central_lon = (min(shifted) + max(shifted)) / 2
    if central_lon > 180:
        central_lon -= 360

    proj = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()  # data always in standard lon/lat

    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={"projection": proj})
    ax.set_facecolor("#E8F4FD")
    # Rasterize heavy bathymetry/land polygons (zorder < 2.5) to reduce PDF size;
    # labels, markers, and arcs (zorder >= 3) remain vector.
    ax.set_rasterization_zorder(2.5)

    # 12-layer bathymetry shading (presentation basin light theme)
    _bathymetric_shades = [
        ('bathymetry_L_0',     '#D6EAF8'),
        ('bathymetry_K_200',   '#C4DFF2'),
        ('bathymetry_J_1000',  '#AED6F1'),
        ('bathymetry_I_2000',  '#95C8E8'),
        ('bathymetry_H_3000',  '#7FB3D5'),
        ('bathymetry_G_4000',  '#6BA3C7'),
        ('bathymetry_F_5000',  '#5B94B8'),
        ('bathymetry_E_6000',  '#4A84A8'),
        ('bathymetry_D_7000',  '#3A7498'),
        ('bathymetry_C_8000',  '#2E6688'),
        ('bathymetry_B_9000',  '#235878'),
        ('bathymetry_A_10000', '#1A4A68'),
    ]
    for layer_name, color in _bathymetric_shades:
        feat = cfeature.NaturalEarthFeature('physical', layer_name, '10m')
        ax.add_feature(feat, facecolor=color, edgecolor='none', alpha=1.0,
                       zorder=0.5)

    ax.add_feature(cfeature.LAND, facecolor="#F5F0E8", edgecolor="#8B8680",
                   linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#6B6560", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":",
                   edgecolor="#9B9590", zorder=2)

    # Compute extent in projection coordinates (handles date-line crossing)
    lat_vals = [epicenter_lat] + [lat for _, _, lat, _ in station_meta]
    lon_margin = 8
    lat_margin = 5
    half_span = (max(shifted) - min(shifted)) / 2 + lon_margin
    bot_margin = lat_margin_bottom if lat_margin_bottom is not None else lat_margin
    ax.set_extent([-half_span, half_span,
                   min(lat_vals) - bot_margin,
                   max(lat_vals) + lat_margin],
                  crs=proj)

    # Epicenter with concentric rings
    for r, alpha in [(5.5, 0.10), (3.5, 0.18), (1.5, 0.30)]:
        ax.plot(epicenter_lon, epicenter_lat, "o", color="red",
                markersize=18 + r * 12, markeredgewidth=1.2,
                markeredgecolor="red", markerfacecolor="none",
                alpha=alpha, zorder=3, transform=data_crs)
    ax.plot(epicenter_lon, epicenter_lat, "r*", markersize=22,
            markeredgecolor="white", markeredgewidth=0.5, zorder=6,
            transform=data_crs)
    ax.plot([], [], "r*", markersize=14, label="Epicenter")

    # Default label offsets (lon_deg, lat_deg, ha)
    if label_offsets is None:
        label_offsets = {}

    station_color = "#0077B6"
    for sid, dist_km, lat, lon in station_meta:
        ax.plot(lon, lat, "o", color=station_color, markersize=10,
                markeredgecolor="white", markeredgewidth=1.0,
                zorder=5, transform=data_crs)
        dlon, dlat, ha = label_offsets.get(sid, (2.0, 1.5, "left"))
        ax.text(lon + dlon, lat + dlat,
                f"{sid}\n({dist_km:,} km)",
                fontsize=7, ha=ha, va="center", fontweight="bold",
                color="#1A1A2E",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                      "alpha": 0.85, "edgecolor": "#B0B0B0"},
                zorder=7, transform=data_crs)
        # Great-circle arc from epicenter to station
        # Use slerp for smooth 50-point interpolation, then segment at
        # 180 deg date-line crossings so PlateCarree renders correctly
        gc_lons, gc_lats = _great_circle_points(
            epicenter_lat, epicenter_lon, lat, lon)
        # Split into segments at date-line crossings, interpolating
        # the exact +/-180 deg boundary point to eliminate gaps
        seg_lons: list[list[float]] = [[gc_lons[0]]]
        seg_lats: list[list[float]] = [[gc_lats[0]]]
        for j in range(1, len(gc_lons)):
            if abs(gc_lons[j] - gc_lons[j - 1]) > 180:
                # Interpolate latitude at the 180 deg crossing
                lon_a, lon_b = gc_lons[j - 1], gc_lons[j]
                lat_a, lat_b = gc_lats[j - 1], gc_lats[j]
                # Unwrap lon_b to be on same side as lon_a
                if lon_b - lon_a > 180:
                    lon_b_unwrap = lon_b - 360
                else:
                    lon_b_unwrap = lon_b + 360
                boundary = 180.0 if lon_a > 0 else -180.0
                frac = (boundary - lon_a) / (lon_b_unwrap - lon_a)
                lat_cross = lat_a + frac * (lat_b - lat_a)
                # Close current segment at boundary
                seg_lons[-1].append(boundary)
                seg_lats[-1].append(lat_cross)
                # Start new segment from opposite boundary
                seg_lons.append([-boundary, gc_lons[j]])
                seg_lats.append([lat_cross, gc_lats[j]])
            else:
                seg_lons[-1].append(gc_lons[j])
                seg_lats[-1].append(gc_lats[j])
        for sl, sa in zip(seg_lons, seg_lats):
            ax.plot(sl, sa, "-", color="#4A5568", linewidth=1.2, alpha=0.45,
                    zorder=3, transform=data_crs)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4,
                      color="#B0B0B0", x_inline=False, y_inline=False)
    gl.xlabel_style = {"color": "#4A4A4A", "fontsize": 9}
    gl.ylabel_style = {"color": "#4A4A4A", "fontsize": 9}
    ax.set_title(title, fontsize=14, fontweight="bold", color="#1A1A2E",
                 pad=12)
    leg = ax.legend(loc="lower left", fontsize=9,
                    facecolor="white", edgecolor="#CCCCCC", framealpha=0.95)
    for text in leg.get_texts():
        text.set_color("#1A1A2E")
    fig.tight_layout()

    out = FIGURES_DIR / outfile
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out)


def _fig_station_map_plain() -> None:
    """Tohoku station map, the only path fig_station_map takes.

    Not a fallback and not coastline-free: _station_map_plain adds
    cfeature.COASTLINE over the bathymetry shading.
    """
    _station_map_plain(
        station_meta=STATION_META,
        epicenter_lat=TOHOKU_LAT,
        epicenter_lon=TOHOKU_LON,
        title="DART Station Locations: Tohoku 2011 Evaluation",
        outfile="fig_station_map.pdf",
        label_offsets={
            "21418": (4.0, -2.0, "left"),
            "21401": (5.0, -2.0, "left"),
            "21413": (3.0, -3.0, "left"),
            "21419": (3.0, 3.0, "left"),
            "46408": (1.0, -3.5, "left"),
            "46402": (4.0, -1.5, "left"),
            "46403": (3.0, 2.0, "left"),
            "46411": (-5.0, 2.0, "right"),
        },
    )


def fig5_tohoku_detection(data: dict) -> None:
    """Fig 5: Tohoku 2011 detection - per-station score bar chart.

    Shows ensemble score and component scores for each station,
    with horizontal lines at T1, T2, T3.
    """
    stations = data.get("per_station", [])
    if not stations:
        logger.warning("No per-station data for Fig 5")
        return

    # Sort by distance to match waveform/residual figure ordering
    stations = sorted(stations, key=lambda s:
        next((d for sid, d, _, _ in STATION_META if sid == s["station_id"]), 0))

    station_ids = [s["station_id"] for s in stations]
    ensemble_scores = [s["ensemble_score"] for s in stations]
    threshold_scores = [s["threshold_score"] for s in stations]
    wavelet_scores = [s["wavelet_score"] for s in stations]
    bocpd_scores = [s["bocpd_score"] for s in stations]

    x = np.arange(len(station_ids))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - 1.5 * width, ensemble_scores, width, label="Ensemble", color="#2196F3")
    ax.bar(x - 0.5 * width, threshold_scores, width, label="Threshold", color="#4CAF50")
    ax.bar(x + 0.5 * width, wavelet_scores, width, label="Wavelet", color="#FF9800")
    ax.bar(x + 1.5 * width, bocpd_scores, width, label="BOCPD", color="#9C27B0")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7, label=f"T1={T1}")
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7, label=f"T2={T2}")
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7, label=f"T3={T3}")

    ax.set_xlabel("DART Station")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Tohoku 2011 (Mw 9.1): Per-Station Anomaly Detection Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "fig5_tohoku_detection.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)

    # If sliding window data exists, create timeline subplot
    sliding = data.get("sliding_window", {})
    if sliding:
        _fig5b_sliding_window(sliding)


def _fig5b_sliding_window(sliding: dict) -> None:
    """Fig 5b: Sliding-window detection timeline."""
    n_stations = len(sliding)
    if n_stations == 0:
        return

    fig, axes = plt.subplots(n_stations, 1, figsize=(10, 2.5 * n_stations), sharex=True)
    if n_stations == 1:
        axes = [axes]

    # Sort by epicentral distance (nearest first)
    _dist_lookup = {sid: d for sid, d, _, _ in STATION_META}
    station_order = sorted(
        sliding.keys(),
        key=lambda sid: _dist_lookup.get(sid, 99999),
    )
    for ax, station_id in zip(axes, station_order):
        timeline = sliding[station_id]
        minutes = [t["minutes_from_earthquake"] for t in timeline]
        ensemble = [t["ensemble_score"] for t in timeline]
        dist = _dist_lookup.get(station_id)
        label = f"{station_id} ({dist} km)" if dist else station_id

        ax.plot(minutes, ensemble, "b-", linewidth=1.5, label="Ensemble")
        ax.axhline(y=T1, color="green", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.axhline(y=T2, color="orange", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.axhline(y=T3, color="red", linestyle="--", alpha=0.6, linewidth=0.8)
        ax.set_ylabel(label, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.tick_params(labelsize=8)

    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Tohoku 2011 (Mw 9.1): Ensemble Score Timeline")
    fig.tight_layout()

    out = FIGURES_DIR / "fig5b_detection_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig6_detiding_quality(data: list) -> None:
    """Fig 6: Detiding validation - residual RMS and M2 amplitude per station."""
    if not data:
        logger.warning("No detiding data for Fig 6")
        return

    station_ids = [s["station_id"] for s in data]
    rms = [s["residual_rms_m"] for s in data]
    holdout_rms = [s["holdout_rms_m"] for s in data]
    m2_amp = [s["M2_amplitude_m"] for s in data]
    s2_amp = [s["S2_amplitude_m"] for s in data]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))

    # (a) Residual RMS
    x = np.arange(len(station_ids))
    ax1.bar(x, rms, color="#2196F3")
    ax1.set_xticks(x)
    ax1.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=8)
    ax1.set_ylabel("Residual RMS (m)")
    ax1.set_title("(a) Full-fit residual RMS")

    # (b) Holdout RMS
    ax2.bar(x, holdout_rms, color="#FF9800")
    ax2.set_xticks(x)
    ax2.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=8)
    ax2.set_ylabel("Holdout RMS (m)")
    ax2.set_title("(b) Holdout validation RMS")

    # (c) M2 amplitude
    ax3.bar(x, m2_amp, color="#4CAF50")
    ax3.set_xticks(x)
    ax3.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=8)
    ax3.set_ylabel("Amplitude (m)")
    ax3.set_title("(c) M2 constituent amplitude")

    # (d) S2 amplitude
    ax4.bar(x, s2_amp, color="#9C27B0")
    ax4.set_xticks(x)
    ax4.set_xticklabels(station_ids, rotation=45, ha="right", fontsize=8)
    ax4.set_ylabel("Amplitude (m)")
    ax4.set_title("(d) S2 constituent amplitude")

    fig.suptitle("Detiding Validation: 30-Day Calibration Data", fontsize=12)
    fig.tight_layout()

    out = FIGURES_DIR / "fig6_detiding_quality.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig7_score_decomposition(data: dict) -> None:
    """Fig 7: Component score decomposition for selected stations.

    Uses sliding-window data if available, otherwise per-station bar chart.
    """
    sliding = data.get("sliding_window", {})
    if not sliding:
        logger.info("No sliding window data for Fig 7 - using static scores")
        _fig7_static(data)
        return

    # Pick near-field and far-field stations if available
    target_stations = []
    for sid in ["21418", "21401"]:  # near-field
        if sid in sliding:
            target_stations.append(sid)
            break
    for sid in ["46411", "46402", "46403"]:  # far-field
        if sid in sliding:
            target_stations.append(sid)
            break

    if not target_stations:
        target_stations = list(sliding.keys())[:2]

    fig, axes = plt.subplots(len(target_stations), 1, figsize=(10, 4 * len(target_stations)))
    if len(target_stations) == 1:
        axes = [axes]

    for ax, station_id in zip(axes, target_stations):
        timeline = sliding[station_id]
        minutes = [t["minutes_from_earthquake"] for t in timeline]
        threshold = [t["threshold_score"] for t in timeline]
        wavelet = [t["wavelet_score"] for t in timeline]
        bocpd = [t["bocpd_score"] for t in timeline]
        ensemble = [t["ensemble_score"] for t in timeline]

        ax.plot(minutes, threshold, label="Threshold", color="#4CAF50", linewidth=1)
        ax.plot(minutes, wavelet, label="Wavelet", color="#FF9800", linewidth=1)
        ax.plot(minutes, bocpd, label="BOCPD", color="#9C27B0", linewidth=1)
        ax.plot(minutes, ensemble, label="Ensemble", color="#2196F3", linewidth=2)

        ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
        ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.5, linewidth=0.7)
        ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)

        ax.set_ylabel("Score")
        ax.set_title(f"Station {station_id}")
        ax.set_ylim(0, 1.05)
        ax.legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Minutes from earthquake origin")
    fig.suptitle("Component Score Decomposition", fontsize=12)
    fig.tight_layout()

    out = FIGURES_DIR / "fig7_score_decomposition.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def _fig7_static(data: dict) -> None:
    """Fallback static bar chart if no sliding window data."""
    stations = data.get("per_station", [])
    if not stations:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    station_ids = [s["station_id"] for s in stations]
    x = np.arange(len(station_ids))
    width = 0.18

    ax.bar(x - 1.5 * width, [s["threshold_score"] for s in stations], width,
           label="Threshold", color="#4CAF50")
    ax.bar(x - 0.5 * width, [s["wavelet_score"] for s in stations], width,
           label="Wavelet", color="#FF9800")
    ax.bar(x + 0.5 * width, [s["bocpd_score"] for s in stations], width,
           label="BOCPD", color="#9C27B0")
    ax.bar(x + 1.5 * width, [s["ensemble_score"] for s in stations], width,
           label="Ensemble", color="#2196F3")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylabel("Score")
    ax.set_title("Component Score Decomposition")
    ax.legend()
    fig.tight_layout()

    out = FIGURES_DIR / "fig7_score_decomposition.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig8_synthetic_heatmap(data: dict) -> None:
    """Fig 8: Synthetic detection sensitivity heatmap.

    2D heatmap of amplitude x period -> max FSM state.
    """
    heatmap = data.get("heatmap_noise001_60s", {})
    if not heatmap:
        logger.warning("No heatmap data for Fig 8")
        return

    params = data.get("parameters", {})
    amplitudes = params.get("amplitudes_m", sorted(float(k) for k in heatmap.keys()))
    periods = params.get("periods_min", [5, 10, 15, 25, 45, 90])

    state_map = {"MONITOR": 0, "INVESTIGATE": 1, "ASSESS": 2, "ESCALATE": 3}

    grid = np.zeros((len(amplitudes), len(periods)))
    for i, amp in enumerate(amplitudes):
        amp_key = f"{amp:.3f}"
        for j, per in enumerate(periods):
            per_key = f"{per}"
            state = heatmap.get(amp_key, {}).get(per_key, "MONITOR")
            grid[i, j] = state_map.get(state, 0)

    fig, ax = plt.subplots(figsize=(8, 6))
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#E8F5E9", "#FFF9C4", "#FFE0B2", "#FFCDD2"])

    im = ax.imshow(
        grid, aspect="auto", cmap=cmap, vmin=-0.5, vmax=3.5,
        origin="lower",
    )

    ax.set_xticks(range(len(periods)))
    ax.set_xticklabels([str(p) for p in periods])
    ax.set_yticks(range(len(amplitudes)))
    ax.set_yticklabels([f"{a:.3f}" for a in amplitudes])
    ax.set_xlabel("Tsunami Period (min)")
    ax.set_ylabel("Tsunami Amplitude (m)")
    ax.set_title("Detection Sensitivity: Noise 0.001 m, Sampling 60 s")

    cbar = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cbar.set_ticklabels(["MONITOR", "INVESTIGATE", "ASSESS", "ESCALATE"])

    # Add text annotations
    for i in range(len(amplitudes)):
        for j in range(len(periods)):
            state_val = int(grid[i, j])
            abbrev = ["MON", "INV", "ASS", "ESC"][state_val]
            color = "black"  # all pastel backgrounds - black text is always readable
            ax.text(j, i, abbrev, ha="center", va="center", fontsize=7, color=color)

    fig.tight_layout()

    out = FIGURES_DIR / "fig8_synthetic_heatmap.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


# -- Chile 2010 figures --------------------------------------------------

CHILE_DATA_DIR = Path("data/chile")

# Chile earthquake parameters
CHILE_LAT = -35.846
CHILE_LON = -72.719

CHILE_STATION_META: list[tuple[str, int, float, float]] = [
    ("32412", 2400, -17.984, -86.374),
    ("32411", 4920, 4.979, -90.793),
    ("54401", 8740, -33.109, -173.155),
    ("46412", 9080, 32.400, -120.582),
    ("46411", 10040, 39.337, -127.040),
    ("51407", 10740, 19.530, -156.601),
    ("21413", 15840, 30.515, 152.117),
]


def _load_chile_event_csv(station_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load Chile event CSV, returning (minutes_from_eq, height_m) arrays."""
    path = CHILE_DATA_DIR / f"dart_{station_id}_chile_2010_event.csv"
    if not path.exists():
        logger.warning("Missing Chile event CSV: %s", path)
        return None
    minutes: list[float] = []
    heights: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            sec = float(row["seconds_from_origin"])
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            minutes.append(sec / 60.0)
            heights.append(h)
    if not minutes:
        return None
    return np.array(minutes), np.array(heights)


def fig_chile_detection(data: dict) -> None:
    """Chile 2010 detection - per-station score bar chart."""
    stations = data.get("per_station", [])
    if not stations:
        logger.warning("No per-station data for Chile detection figure")
        return

    # Sort by distance to match waveform/residual figure ordering
    stations = sorted(stations, key=lambda s:
        next((d for sid, d, _, _ in CHILE_STATION_META if sid == s["station_id"]), 0))

    # Exclude degraded (standard-mode) stations - scores are unreliable
    stations = [s for s in stations if not s.get("filter_degraded")]
    if not stations:
        logger.warning("No event-mode stations for Chile detection figure")
        return

    station_ids = [s["station_id"] for s in stations]
    ensemble_scores = [s["ensemble_score"] for s in stations]
    threshold_scores = [s["threshold_score"] for s in stations]
    wavelet_scores = [s["wavelet_score"] for s in stations]
    bocpd_scores = [s["bocpd_score"] for s in stations]

    n = len(station_ids)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * n), 5))
    ax.bar(x - 1.5 * width, ensemble_scores, width, label="Ensemble", color="#2196F3")
    ax.bar(x - 0.5 * width, threshold_scores, width, label="Threshold", color="#4CAF50")
    ax.bar(x + 0.5 * width, wavelet_scores, width, label="Wavelet", color="#FF9800")
    ax.bar(x + 1.5 * width, bocpd_scores, width, label="BOCPD", color="#9C27B0")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7, label=f"T1={T1}")
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7, label=f"T2={T2}")
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7, label=f"T3={T3}")

    ax.set_xlabel("DART Station")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Chile 2010 (Mw 8.8): Per-Station Anomaly Detection Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_chile_detection.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_chile_detection_timeline(data: dict) -> None:
    """Chile 2010 score overlay for event-mode stations.

    Two full-width panels per station (filtered, scores) plus a
    1/4-width onset zoom panel positioned below the score panel
    at the score-onset time.
    """
    try:
        from hazard_assessment.agents.anomaly_detection import (
            apply_bandpass_filter,
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Chile timeline")
        return

    sliding = data.get("sliding_window", {})
    if not sliding:
        logger.warning("No sliding window data for Chile timeline")
        return

    # Only plot event-mode stations (those with significant scores)
    event_stations = [sid for sid in ["32412", "32411", "46412"]
                      if sid in sliding]
    if not event_stations:
        event_stations = list(sliding.keys())[:3]

    n_rows = 3  # filtered, scores, zoom-placeholder per station
    n_stations = len(event_stations)
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )

    def _hampel_clean(residual):
        hw = min(10, max(3, len(residual) // 6))
        cleaned, _ = hampel_filter(residual, half_window=hw, threshold=3.0)
        return cleaned

    zoom_infos = []
    for i, sid in enumerate(event_stations):
        dist_km = next((d for s, d, _, _ in CHILE_STATION_META if s == sid), 0)
        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        ax_zoom.set_visible(False)

        cal_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_calibration.csv"
        event_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_event.csv"

        info = _plot_score_overlay_station(
            (ax_filt, ax_score), sid, dist_km, sliding,
            cal_path, event_path,
            apply_bandpass_filter, fit_tidal_harmonics, predict_tide,
            clean_fn=_hampel_clean,
        )
        ax_filt.set_xlim(right=300)
        zoom_infos.append((i, info))

    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title("Chile 2010 (Mw 8.8): Bandpass-Filtered Residual and Detection Scores")
    fig.tight_layout()

    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_chile_detection_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_chile_waveforms() -> None:
    """Chile 2010 raw BPR time series with earthquake origin marker."""
    event_stations = [(sid, d) for sid, d, _, _ in CHILE_STATION_META]

    fig, axes = plt.subplots(len(event_stations), 1,
                             figsize=(10, 2.0 * len(event_stations)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km) in zip(axes, event_stations):
        result = _load_chile_event_csv(sid)
        if result is None:
            ax.text(0.5, 0.5, f"{sid}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        mins, heights = result
        heights_dm = heights - np.mean(heights)
        # Clip extreme sensor-glitch spikes (>5 m from median) for readability
        med = np.median(heights_dm)
        glitch_mask = np.abs(heights_dm - med) > 5.0
        if np.any(glitch_mask):
            heights_dm = heights_dm.copy()
            heights_dm[glitch_mask] = np.nan
        _plot_with_gaps(ax, mins, heights_dm)
        # Overlay tidal+drift prediction
        cal_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_calibration.csv"
        ev_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_event.csv"
        tide_data = _load_tidal_prediction(cal_path, ev_path)
        if tide_data is not None:
            _ev_t, _ev_h, predicted = tide_data
            pred_dm = predicted - np.mean(heights)
            label = "Predicted tide" if plotted == 0 else None
            ax.plot(_ev_t * 60.0, pred_dm, "r--", linewidth=0.6, alpha=0.7,
                    label=label)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No Chile event CSV data found")
        return

    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Chile 2010 (Mw 8.8): Raw Bottom Pressure Records (De-meaned)")
    fig.tight_layout()

    out = FIGURES_DIR / "fig_chile_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_chile_station_map() -> None:
    """Chile 2010 station location map."""
    _station_map_plain(
        station_meta=CHILE_STATION_META,
        epicenter_lat=CHILE_LAT,
        epicenter_lon=CHILE_LON,
        title="DART Station Locations: Chile 2010 Evaluation",
        outfile="fig_chile_station_map.pdf",
        lat_margin_bottom=15,
        label_offsets={
            "32412": (4.0, -4.0, "left"),
            "32411": (4.0, 2.0, "left"),
            "54401": (-5.0, -4.0, "right"),
            "46412": (4.0, -4.0, "left"),
            "46411": (4.0, -3.0, "left"),
            "51407": (4.0, 2.0, "left"),
            "21413": (4.0, 2.0, "left"),
        },
    )


def fig_chile_residuals() -> None:
    """Chile 2010 detided residual time series."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Chile residual figure")
        return

    import pandas as pd

    # Only include stations with >=1-min event data (drop 15-min standard-mode
    # stations whose 28-point records produce unreliable detided residuals).
    _CHILE_RESIDUAL_STATIONS = [
        s for s in CHILE_STATION_META if s[0] in ("32412", "32411", "46412")
    ]

    fig, axes = plt.subplots(len(_CHILE_RESIDUAL_STATIONS), 1,
                             figsize=(10, 2.0 * len(_CHILE_RESIDUAL_STATIONS)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, _CHILE_RESIDUAL_STATIONS):
        cal_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_calibration.csv"
        if not cal_path.exists():
            ax.text(0.5, 0.5, f"{sid}: no calibration", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        cal_times: list[float] = []
        cal_values: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                sec = float(row["seconds_from_origin"])
                cal_times.append(sec / 3600.0)
                cal_values.append(h)

        if len(cal_times) < 100:
            ax.text(0.5, 0.5, f"{sid}: insufficient cal data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            ax.set_ylabel(f"{sid}")
            continue

        cal_t = np.array(cal_times)
        cal_v = np.array(cal_values)
        harmonics = fit_tidal_harmonics(cal_t, cal_v, clean_input=True)

        result = _load_chile_event_csv(sid)
        if result is None:
            ax.set_ylabel(f"{sid}")
            continue

        mins, _heights = result
        event_hours_list: list[float] = []
        event_heights_list: list[float] = []
        event_path = CHILE_DATA_DIR / f"dart_{sid}_chile_2010_event.csv"
        with open(event_path) as f:
            reader_ev = csv.DictReader(f)
            for row in reader_ev:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                sec = float(row["seconds_from_origin"])
                event_hours_list.append(sec / 3600.0)
                event_heights_list.append(h)

        if not event_hours_list:
            ax.set_ylabel(f"{sid}")
            continue

        ev_t = np.array(event_hours_list)
        ev_h = np.array(event_heights_list)

        # Despike: remove single-sample telemetry glitches.
        if len(ev_h) > 2:
            left = np.concatenate([[ev_h[0]], ev_h[:-1]])
            right = np.concatenate([ev_h[1:], [ev_h[-1]]])
            neighbor_med = np.median(
                np.column_stack([left, ev_h, right]), axis=1)
            spike = np.abs(ev_h - neighbor_med) > 1.0
            if np.any(spike):
                logger.info("Despiked %d glitch(es) from %s event data",
                            int(np.sum(spike)), sid)
                ev_h = np.where(spike, neighbor_med, ev_h)

        predicted = predict_tide(ev_t, harmonics)
        ev_residual = ev_h - predicted

        # Pre-event linear detrend + centered rolling-median baseline.
        # The centered (non-causal) median has no lag on monotonic drift.
        # The Tohoku residual figure uses a centered median too, so the
        # difference is the window and its scope, not causality: 60 min
        # for every station here, 120 min and only beyond 3,000 km there.
        # This is not a signal-free window.  32412 sits ~2,400 km from the
        # epicenter, so the first arrival lands near t+3.4 h, well inside
        # the -1.0 h to +6.0 h record, and results/chile_detection.json
        # scores it 0.9957.  The 60-min median here therefore removes real
        # tsunami energy at the longer arrival periods, not just drift.
        pre_ev_mask = mins < 0
        if np.sum(pre_ev_mask) >= 2:
            bp = np.polyfit(mins[pre_ev_mask], ev_residual[pre_ev_mask], deg=1)
            ev_residual -= np.polyval(bp, mins)
        dt_min = float(np.median(np.diff(mins)))
        win = max(3, int(60.0 / max(dt_min, 0.1)))
        win = min(win, len(ev_residual) // 2)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(ev_residual, pad, mode='reflect')
        baseline = pd.Series(padded).rolling(
            window=win, min_periods=1, center=True,
        ).median().values[pad:-pad]
        ev_residual -= baseline

        # Hampel filter: clean isolated instrument-glitch spikes
        hw = min(10, max(3, len(ev_residual) // 6))
        residual, _ = hampel_filter(ev_residual, half_window=hw, threshold=3.0)

        _plot_with_gaps(ax, mins, residual)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        logger.warning("No data for Chile residual figure")
        return

    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Chile 2010 (Mw 8.8): Detided Residuals")
    fig.tight_layout()

    out = FIGURES_DIR / "fig_chile_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


# ---------------------------------------------------------------------------
# CO-OPS water-level figures (Appendix: multi-source data availability)
# ---------------------------------------------------------------------------

# Distances from Tohoku epicenter (38.297N, 142.373E) via haversine (km)
COOPS_STATIONS = {
    "1612340": ("Honolulu, HI", 5960),
    "1617760": ("Hilo, HI", 6301),
    "1619910": ("Midway Island", 3875),
    "1890000": ("Wake Island", 3151),
    "1770000": ("Pago Pago, AS", 7617),    # American Samoa
    "9419750": ("Crescent City, CA", 7542),  # Known tsunami amplification
}

# Distances from Chile epicenter (35.846S, 72.719W) via haversine (km)
COOPS_CHILE_DIST = {
    "1612340": 10960,
    "1617760": 10620,
    "1619910": 13034,
    "1890000": 13978,
    "9419750": 10107,  # Crescent City
}


def _load_coops_csv(
    filepath: Path,
) -> tuple[list[str], list[float]] | None:
    """Load CO-OPS CSV returning (timestamps, values)."""
    if not filepath.exists():
        return None
    timestamps: list[str] = []
    values: list[float] = []
    with open(filepath) as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                val = float(row["v"])
            except (ValueError, KeyError):
                continue
            timestamps.append(row["t"])
            values.append(val)
    if not timestamps:
        return None
    return timestamps, values


def _parse_coops_minutes_from_eq(
    timestamps: list[str], eq_time_str: str,
) -> np.ndarray:
    """Convert CO-OPS timestamps to minutes from earthquake origin.

    All timestamps are assumed UTC (CO-OPS API returns UTC).
    """
    from datetime import datetime
    eq_time = datetime.strptime(eq_time_str, "%Y-%m-%d %H:%M:%S").replace(
        tzinfo=UTC
    )
    minutes = []
    for ts in timestamps:
        t = datetime.strptime(ts.strip(), "%Y-%m-%d %H:%M").replace(
            tzinfo=UTC
        )
        delta = (t - eq_time).total_seconds() / 60.0
        minutes.append(delta)
    return np.array(minutes)


def _plot_coops_two_panel(
    event_name: str,
    eq_time: str,
    title: str,
    data_dir: Path,
    station_order: list[str],
    dist_dict: dict[str, int],
    outfile: str,
) -> None:
    """Two-panel CO-OPS figure: raw + predicted tide (top), de-tided residual (bottom).

    For each station, creates two sub-axes:
      - Upper: observed water level (black) with predicted tide overlay (red dashed)
      - Lower: de-tided residual (blue) revealing the tsunami signal
    """

    stations_with_data = []
    for sid in station_order:
        obs_path = data_dir / f"coops_{sid}_{event_name}.csv"
        pred_path = data_dir / f"coops_{sid}_{event_name}_predictions.csv"
        obs = _load_coops_csv(obs_path)
        pred = _load_coops_csv(pred_path)
        if obs:
            stations_with_data.append((sid, obs, pred))

    if not stations_with_data:
        logger.warning("No CO-OPS data for %s figure", event_name)
        return

    n = len(stations_with_data)
    fig, axes = plt.subplots(n, 2, figsize=(12, 2.2 * n), sharex=True,
                             gridspec_kw={"width_ratios": [1, 1], "wspace": 0.25})
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, (sid, (obs_ts, obs_vals), pred_data) in enumerate(stations_with_data):
        ax_raw = axes[i, 0]
        ax_res = axes[i, 1]
        name = COOPS_STATIONS[sid][0]
        dist_km = dist_dict.get(sid, COOPS_STATIONS[sid][1])
        minutes = _parse_coops_minutes_from_eq(obs_ts, eq_time)
        obs_arr = np.array(obs_vals)

        # --- Left panel: raw observed + predicted tide ---
        ax_raw.plot(minutes, obs_arr, "k-", linewidth=0.5, label="Observed")
        ax_raw.axvline(x=0, color="red", linewidth=0.8, alpha=0.7)
        if pred_data:
            pred_ts, pred_vals = pred_data
            pred_min = _parse_coops_minutes_from_eq(pred_ts, eq_time)
            pred_arr = np.array(pred_vals)

            # Align predictions to observations by timestamp matching
            from datetime import datetime
            obs_time_set = {ts.strip(): v for ts, v in zip(obs_ts, obs_vals)}
            pred_time_set = {ts.strip(): v for ts, v in zip(pred_ts, pred_vals)}
            common_ts = sorted(set(obs_time_set) & set(pred_time_set))

            # De-tide with an IRLS harmonic fit on the CO-OPS observations.
            #
            # NOAA predictions leave ~3 cm residual tidal energy due to
            # shallow-water overtides and slight harmonic constant errors.
            # Fitting our own 8-constituent + drift harmonics directly on
            # the observations gives ~5x cleaner residuals
            # (0.65 cm vs 3.3 cm std at Honolulu).
            #
            # This fit is retrospective, not causal: it spans the whole
            # window, tsunami samples included.  The Cauchy IRLS weights
            # down-weight those samples rather than excluding them.  Drift
            # terms (linear + quadratic) capture meteorological baseline
            # shift.
            if common_ts:
                eq_dt = datetime.strptime(eq_time, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=UTC)
                all_minutes = []
                all_obs_list = []
                for ts in common_ts:
                    t = datetime.strptime(ts, "%Y-%m-%d %H:%M").replace(
                        tzinfo=UTC)
                    all_minutes.append((t - eq_dt).total_seconds() / 60.0)
                    all_obs_list.append(obs_time_set[ts])
                all_min_arr = np.array(all_minutes)
                all_obs_arr = np.array(all_obs_list)
                all_hrs = all_min_arr / 60.0

                pre_mask = all_min_arr < 0
                n_pre = int(np.sum(pre_mask))

                tide_pred = None
                try:
                    from hazard_assessment.agents.anomaly_detection import (
                        build_harmonic_matrix,
                    )
                    if len(all_hrs) >= 100:
                        # Fit 8 tidal constituents + drift over the FULL
                        # CO-OPS window (2 days for Tohoku and Chile,
                        # 3 days for the others) using IRLS.  The Cauchy
                        # weights automatically down-weight tsunami samples.
                        # This window is far short of the 14.8 days the
                        # Rayleigh criterion needs to separate M2 (12.42 h)
                        # from S2 (12.00 h), so that pair is only jointly
                        # constrained.  Individual constituent amplitudes
                        # are therefore not meaningful here; the summed fit
                        # still tracks the tide inside the fitted window,
                        # which is all these appendix figures need.
                        # Drift terms absorb slow weather variability.
                        t_center = float(all_hrs.mean())
                        all_hrs_c = all_hrs - t_center
                        H = build_harmonic_matrix(
                            all_hrs_c, include_drift=True)
                        # IRLS with Cauchy weights (15 iterations)
                        w = np.ones(len(all_obs_arr))
                        coeffs = np.linalg.lstsq(H, all_obs_arr, rcond=None)[0]
                        for _ in range(15):
                            resid = all_obs_arr - H @ coeffs
                            scale = 1.4826 * float(np.median(np.abs(resid)))
                            if scale < 1e-8:
                                break
                            u = resid / (scale * 2.385)
                            w = 1.0 / (1.0 + u ** 2)
                            W = np.diag(w)
                            coeffs = np.linalg.lstsq(
                                W @ H, W @ all_obs_arr, rcond=None)[0]
                        tide_pred = H @ coeffs
                except ImportError:
                    pass

                if tide_pred is not None:
                    residual = all_obs_arr - tide_pred
                else:
                    # Fallback: NOAA prediction with pre-event median
                    noaa_res = np.array([obs_time_set[ts] - pred_time_set[ts]
                                         for ts in common_ts])
                    offset = float(np.nanmedian(noaa_res[pre_mask])) if n_pre > 5 \
                        else float(np.nanmedian(noaa_res))
                    residual = noaa_res - offset
                    tide_pred = all_obs_arr - residual
                    all_hrs = all_min_arr / 60.0

                # --- Right panel: de-tided residual ---
                ax_res.plot(all_min_arr, residual, "b-", linewidth=0.5)
                ax_res.axvline(x=0, color="red", linewidth=0.8, alpha=0.7)
                wave_speed_kmh = 198.0 * 3.6
                eta_min = dist_km / wave_speed_kmh * 60.0
                ax_res.axvline(x=eta_min, color="green", linewidth=0.8,
                               linestyle="--", alpha=0.6)

            # --- Left panel: tidal prediction overlay ---
            if common_ts and len(pred_min) > 0 and tide_pred is not None:
                tide_interp = np.interp(
                    np.array(pred_min) / 60.0, all_hrs, tide_pred)
                ax_raw.plot(pred_min, tide_interp, "r--",
                            linewidth=0.6, alpha=0.7, label="Predicted tide")
            else:
                ax_raw.plot(pred_min, pred_arr, "r--",
                            linewidth=0.6, alpha=0.7, label="Predicted tide")
        else:
            ax_res.text(0.5, 0.5, "No predictions", transform=ax_res.transAxes,
                        ha="center", va="center", fontsize=8, color="gray")

        # Labels
        label_text = f"{name}\n({dist_km:,} km)"
        ax_raw.text(0.02, 0.90, label_text, transform=ax_raw.transAxes,
                    fontsize=7, va="top",
                    bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                          "alpha": 0.8})
        ax_raw.set_ylabel("m", fontsize=7)
        ax_res.set_ylabel("m", fontsize=7)
        ax_raw.tick_params(labelsize=6)
        ax_res.tick_params(labelsize=6)

    # Column titles
    axes[0, 0].set_title("Observed + Predicted Tide", fontsize=9)
    axes[0, 1].set_title("De-tided Residual (Tsunami Signal)", fontsize=9)
    axes[-1, 0].set_xlabel("Minutes from earthquake origin", fontsize=8)
    axes[-1, 1].set_xlabel("Minutes from earthquake origin", fontsize=8)

    # Legend on first row only
    axes[0, 0].legend(fontsize=6, loc="upper right")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIGURES_DIR / outfile
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


# Distances from Tohoku epicenter (38.297N, 142.373E) via haversine (km)
COOPS_TOHOKU_DIST = {
    "1612340": 5960,   # Honolulu
    "1617760": 6301,   # Hilo
    "1619910": 3875,   # Midway
    "1890000": 3151,   # Wake
    "9419750": 7542,   # Crescent City
}


def fig_coops_tohoku() -> None:
    """CO-OPS water-level waveforms for Tohoku 2011."""
    _plot_coops_two_panel(
        event_name="tohoku",
        eq_time="2011-03-11 05:46:24",
        title="Tohoku 2011 (Mw 9.1): CO-OPS Water Level (6-min)",
        data_dir=Path("data/tohoku"),
        station_order=["1890000", "1619910", "1612340", "1617760", "9419750"],
        dist_dict=COOPS_TOHOKU_DIST,
        outfile="fig_coops_tohoku.pdf",
    )


def fig_coops_chile() -> None:
    """CO-OPS water-level waveforms for Chile 2010."""
    _plot_coops_two_panel(
        event_name="chile",
        eq_time="2010-02-27 06:34:11",
        title="Chile 2010 (Mw 8.8): CO-OPS Water Level (6-min)",
        data_dir=Path("data/chile"),
        station_order=["9419750", "1617760", "1612340", "1619910", "1890000"],
        dist_dict=COOPS_CHILE_DIST,
        outfile="fig_coops_chile.pdf",
    )


# Distances from Illapel epicenter (31.573S, 71.674W) via haversine (km)
COOPS_ILLAPEL_DIST = {
    "1612340": 10890,  # Honolulu
    "1617760": 10550,  # Hilo
    "1619910": 12990,  # Midway
    "1890000": 14070,  # Wake
}

# Distances from Iquique epicenter (19.610S, 70.769W) via haversine (km)
COOPS_IQUIQUE_DIST = {
    "1612340": 10500,  # Honolulu
    "1617760": 10170,  # Hilo
    "1619910": 12600,  # Midway
    "1890000": 14030,  # Wake
}

# Distances from Samoa epicenter (15.489S, 172.095W) via haversine (km)
COOPS_SAMOA_DIST = {
    "1770000": 202,    # Pago Pago (near-field!)
    "1612340": 4380,   # Honolulu
    "1617760": 4340,   # Hilo
    "1619910": 4890,   # Midway
    "1890000": 4510,   # Wake
}


def fig_coops_illapel() -> None:
    """CO-OPS water-level waveforms for Illapel 2015."""
    _plot_coops_two_panel(
        event_name="illapel",
        eq_time="2015-09-16 22:54:32",
        title="Illapel 2015 (Mw 8.3): CO-OPS Water Level (6-min)",
        data_dir=Path("data/illapel"),
        station_order=["1617760", "1612340", "1619910", "1890000"],
        dist_dict=COOPS_ILLAPEL_DIST,
        outfile="fig_coops_illapel.pdf",
    )


def fig_coops_iquique() -> None:
    """CO-OPS water-level waveforms for Iquique 2014."""
    _plot_coops_two_panel(
        event_name="iquique",
        eq_time="2014-04-01 23:46:47",
        title="Iquique 2014 (Mw 8.2): CO-OPS Water Level (6-min)",
        data_dir=Path("data/iquique"),
        station_order=["1617760", "1612340", "1619910", "1890000"],
        dist_dict=COOPS_IQUIQUE_DIST,
        outfile="fig_coops_iquique.pdf",
    )


def fig_coops_samoa() -> None:
    """CO-OPS water-level waveforms for Samoa 2009."""
    _plot_coops_two_panel(
        event_name="samoa",
        eq_time="2009-09-29 17:48:10",
        title="Samoa 2009 (Mw 8.1): CO-OPS Water Level (6-min)",
        data_dir=Path("data/samoa"),
        station_order=["1770000", "1617760", "1612340", "1890000", "1619910"],
        dist_dict=COOPS_SAMOA_DIST,
        outfile="fig_coops_samoa.pdf",
    )


def fig_multi_source_network_map() -> None:
    """Global map: all five evaluated epicenters plus operational DART and CO-OPS stations."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError:
        logger.warning("Skipping multi-source network map - cartopy not installed")
        return

    # All active CO-OPS water level stations from NOAA API (March 2026).
    # Loaded from JSON; only the configured stations are labeled on the map.
    coops_json = Path(__file__).parent / "coops_stations.json"
    if not coops_json.exists():
        logger.warning("Missing %s - using configured stations only", coops_json)
        all_coops: dict[str, dict] = {}
    else:
        with open(coops_json) as f:
            all_coops = json.load(f)

    # Configured stations to label by name
    coops_labeled = {
        "1612340": "Honolulu",
        "1617760": "Hilo",
        "1619910": "Midway",
        "1890000": "Wake",
        "9419750": "Crescent City",
        "1770000": "Pago Pago",
    }

    # All operational DART stations from NDBC dart_stations.php.
    # Coordinates from NDBC station pages, verified 2026-03-18.
    dart_stations: dict[str, tuple[float, float]] = {
        # NW Pacific
        "21413": (30.533, 152.132),    # Izu-Bonin
        "21414": (48.967, 178.212),    # South of Adak
        "21415": (50.150, 171.933),    # South of Attu, AK
        "21416": (48.122, 163.328),    # SE of Kamchatka
        "21418": (38.730, 148.800),    # NE of Tokyo
        "21419": (44.435, 155.717),    # SE of Kuril Islands
        "21420": (28.891, 135.023),    # Philippine Sea
        # Indian Ocean
        "23401": (8.861, 88.558),      # Bay of Bengal
        "23461": (9.500, 95.600),      # Andaman Sea
        # SE Pacific
        "32401": (-20.469, -73.433),   # Off northern Chile
        "32402": (-26.735, -73.978),   # Off central Chile
        "32411": (4.960, -90.875),     # East Pacific
        "32413": (-7.407, -93.517),    # East Pacific
        # Atlantic / Caribbean
        "41420": (23.370, -67.500),    # Atlantic
        "41421": (23.421, -63.791),    # Atlantic
        "41425": (28.652, -65.652),    # Atlantic
        "42407": (15.293, -68.188),    # Caribbean
        "42409": (25.856, -89.258),    # Gulf of Mexico
        # East Pacific (off Mexico/Central America)
        "43412": (16.037, -106.975),   # Off Mexico
        "43413": (11.005, -100.097),   # Off Central America
        # NW Atlantic
        "44402": (39.287, -70.633),    # NW Atlantic
        "44403": (41.915, -61.590),    # NW Atlantic
        # Aleutian / Gulf of Alaska
        "46402": (50.978, -163.948),   # Aleutian
        "46403": (52.668, -156.970),   # Aleutian
        "46404": (45.848, -128.775),   # Cascadia
        "46407": (42.704, -128.895),   # Oregon
        "46408": (49.648, -169.888),   # Aleutian
        "46409": (55.323, -148.552),   # SE Kodiak, AK
        "46410": (57.632, -143.750),   # Gulf of Alaska
        "46411": (39.337, -127.040),   # Cape Mendocino
        "46413": (48.002, -174.202),   # South of Adak
        "46414": (53.726, -152.483),   # SE Chirikof, AK
        "46415": (52.975, -139.940),   # Gulf of Alaska
        "46416": (49.900, -134.407),   # West of Vancouver, BC
        "46419": (48.807, -129.622),   # WNW of Seattle, WA
        # Hawaii / South Pacific
        "51407": (19.556, -156.536),   # Hawaii
        "51425": (-9.505, -176.262),   # South Pacific
        # Western Pacific / Micronesia
        "52401": (19.240, 155.729),    # Marianas
        "52402": (11.926, 153.906),    # Micronesia
        "52403": (4.050, 145.588),     # Micronesia
        "52404": (20.627, 132.144),    # Philippine Sea
        "52405": (13.034, 132.151),    # Philippine Sea
        "52406": (-5.374, 164.991),    # Solomon Islands
    }

    # Build CO-OPS coordinate lookup: {station_id: (lat, lon)}
    coops_stations: dict[str, tuple[float, float]] = {
        sid: (info["lat"], info["lon"]) for sid, info in all_coops.items()
    }

    # Collect all longitudes for centering
    all_lons = (
        [TOHOKU_LON, CHILE_LON]
        + [lon for _, lon in dart_stations.values()]
        + [lon for _, lon in coops_stations.values()]
    )
    shifted = [lon % 360 for lon in all_lons]
    central_lon = (min(shifted) + max(shifted)) / 2
    if central_lon > 180:
        central_lon -= 360

    proj = ccrs.PlateCarree(central_longitude=central_lon)
    data_crs = ccrs.PlateCarree()

    fig, ax = plt.subplots(figsize=(14, 8), subplot_kw={"projection": proj})
    ax.set_facecolor("#E8F4FD")
    ax.set_rasterization_zorder(2.5)  # rasterize bathymetry/land, keep markers vector

    # 12-layer bathymetry shading (presentation basin light theme)
    _bathymetric_shades = [
        ('bathymetry_L_0',     '#D6EAF8'),
        ('bathymetry_K_200',   '#C4DFF2'),
        ('bathymetry_J_1000',  '#AED6F1'),
        ('bathymetry_I_2000',  '#95C8E8'),
        ('bathymetry_H_3000',  '#7FB3D5'),
        ('bathymetry_G_4000',  '#6BA3C7'),
        ('bathymetry_F_5000',  '#5B94B8'),
        ('bathymetry_E_6000',  '#4A84A8'),
        ('bathymetry_D_7000',  '#3A7498'),
        ('bathymetry_C_8000',  '#2E6688'),
        ('bathymetry_B_9000',  '#235878'),
        ('bathymetry_A_10000', '#1A4A68'),
    ]
    for layer_name, color in _bathymetric_shades:
        feat = cfeature.NaturalEarthFeature('physical', layer_name, '10m')
        ax.add_feature(feat, facecolor=color, edgecolor='none', alpha=1.0,
                       zorder=0.5)

    ax.add_feature(cfeature.LAND, facecolor="#F5F0E8", edgecolor="#8B8680",
                   linewidth=0.5, zorder=1)
    ax.add_feature(cfeature.COASTLINE, linewidth=0.6, edgecolor="#6B6560", zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":",
                   edgecolor="#9B9590", zorder=2)

    # Set extent to fit all stations with margin
    all_lats = (
        [TOHOKU_LAT, CHILE_LAT]
        + [lat for lat, _ in dart_stations.values()]
        + [lat for lat, _ in coops_stations.values()]
    )
    half_span = (max(shifted) - min(shifted)) / 2 + 8
    lat_bot = max(min(all_lats) - 6, -50)
    lat_top = min(max(all_lats) + 6, 72)
    ax.set_extent([-half_span, half_span, lat_bot, lat_top], crs=proj)

    # Epicenters - all five evaluated events
    ax.plot(TOHOKU_LON, TOHOKU_LAT, "r*", markersize=18, zorder=6,
            label="Tohoku 2011 (Mw 9.1)", transform=data_crs)
    ax.plot(CHILE_LON, CHILE_LAT, "m*", markersize=18, zorder=6,
            label="Chile 2010 (Mw 8.8)", transform=data_crs)
    ax.plot(ILLAPEL_LON, ILLAPEL_LAT, "*", color="#FF6F00", markersize=14,
            zorder=6, label="Illapel 2015 (Mw 8.3)", transform=data_crs)
    ax.plot(IQUIQUE_LON, IQUIQUE_LAT, "*", color="#4CAF50", markersize=14,
            zorder=6, label="Iquique 2014 (Mw 8.2)", transform=data_crs)
    ax.plot(SAMOA_LON, SAMOA_LAT, "*", color="#9C27B0", markersize=14,
            zorder=6, label="Samoa 2009 (Mw 8.1)", transform=data_crs)

    # DART stations - per-station label offsets to avoid overlap.
    # At global scale many stations cluster; alternate above/below.
    dart_label_offsets: dict[str, tuple[float, float, str]] = {
        # NW Pacific cluster near Japan
        "21420": (3.0, -2.5, "left"),
        "21413": (3.0, -2.0, "left"),
        "21418": (3.0, -2.0, "left"),
        "21419": (-4.0, 2.0, "right"),
        "21416": (3.0, 2.0, "left"),
        "21415": (3.0, -2.5, "left"),
        "21414": (3.0, 2.0, "left"),
        # Western Pacific / Micronesia
        "52404": (-4.0, -2.0, "right"),
        "52405": (3.0, -2.0, "left"),
        "52401": (3.0, 2.0, "left"),
        "52402": (3.0, -2.0, "left"),
        "52403": (3.0, -2.0, "left"),
        "52406": (3.0, -2.0, "left"),
        # Indian Ocean - 23461 right-side to avoid left edge clipping
        "23401": (3.0, -2.5, "left"),
        "23461": (5.0, 2.0, "left"),
        # Aleutian arc (west to east, stagger above/below with large offsets)
        "46413": (3.0, -4.0, "left"),
        "46408": (3.0, 4.0, "left"),
        "46402": (3.0, -4.0, "left"),
        "46403": (3.0, 4.0, "left"),
        "46414": (3.0, -4.0, "left"),
        "46409": (3.0, 4.0, "left"),
        "46410": (3.0, -4.0, "left"),
        "46415": (3.0, 4.0, "left"),
        # NE Pacific coast (all left of dot, away from coast)
        "46416": (-5.0, 2.0, "right"),
        "46419": (-5.0, -2.5, "right"),
        "46404": (-5.0, 2.0, "right"),
        "46407": (-5.0, -2.0, "right"),
        "46411": (-5.0, -2.0, "right"),
        # East Pacific
        "43412": (3.0, 2.0, "left"),
        "43413": (3.0, -2.0, "left"),
        "32411": (3.0, 2.0, "left"),
        "32413": (3.0, -2.0, "left"),
        # SE Pacific / Chile
        "32401": (3.0, 2.0, "left"),
        "32402": (3.0, -2.0, "left"),
        # Hawaii / South Pacific
        "51407": (-4.0, 2.0, "right"),
        "51425": (3.0, -2.0, "left"),
        # Atlantic / Caribbean / Gulf
        "41420": (3.0, 2.0, "left"),
        "41421": (3.0, -2.5, "left"),
        "41425": (3.0, -2.0, "left"),
        "42407": (3.0, -2.0, "left"),
        "42409": (3.0, -2.5, "left"),
        "44402": (3.0, -2.0, "left"),
        "44403": (3.0, 2.0, "left"),
    }
    dart_plotted = False
    for sid, (lat, lon) in dart_stations.items():
        ax.plot(lon, lat, "o", color="#2196F3", markersize=5, zorder=4,
                markeredgecolor="black", markeredgewidth=0.4,
                label=f"DART BPR ({len(dart_stations)})" if not dart_plotted else None,
                transform=data_crs)
        dlon, dlat, ha = dart_label_offsets.get(sid, (2.5, 1.5, "left"))
        ax.text(lon + dlon, lat + dlat, sid, fontsize=4, ha=ha, va="center",
                color="#2196F3", zorder=7, transform=data_crs)
        dart_plotted = True

    # CO-OPS stations - plot all, label only the configured stations
    coops_label_offsets: dict[str, tuple[float, float, str]] = {
        "1612340": (2.0, 2.0, "left"),   # Honolulu - above-right
        "1617760": (2.0, -2.5, "left"),  # Hilo - below-right
        "1619910": (2.0, -2.5, "left"),  # Midway - below-right
        "1890000": (2.0, -2.5, "left"),  # Wake - below-right
    }
    coops_plotted = False
    for sid, (lat, lon) in coops_stations.items():
        ax.plot(lon, lat, "s", color="#FF9800", markersize=3.5, zorder=5,
                markeredgecolor="black", markeredgewidth=0.3,
                label=f"CO-OPS Tide Gauge ({len(coops_stations)})" if not coops_plotted else None,
                transform=data_crs)
        # Label only the configured stations by name
        if sid in coops_labeled:
            dlon, dlat, ha = coops_label_offsets.get(sid, (2.0, -2.0, "left"))
            ax.text(lon + dlon, lat + dlat, coops_labeled[sid],
                    fontsize=5, ha=ha, va="center",
                    color="#E65100", fontweight="bold", zorder=7,
                    transform=data_crs)
        coops_plotted = True

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, alpha=0.4,
                      color="#B0B0B0", x_inline=False, y_inline=False)
    gl.xlabel_style = {"color": "#4A4A4A", "fontsize": 9}
    gl.ylabel_style = {"color": "#4A4A4A", "fontsize": 9}
    ax.set_title("Multi-Source Observation Network: DART BPR and CO-OPS Tide Gauges",
                 fontsize=14, fontweight="bold", color="#1A1A2E", pad=12)
    leg = ax.legend(loc="lower left", fontsize=7, framealpha=0.95,
                    facecolor="white", edgecolor="#CCCCCC", markerscale=0.5,
                    ncol=2)
    for text in leg.get_texts():
        text.set_color("#1A1A2E")
    fig.tight_layout()

    out = FIGURES_DIR / "fig_multi_source_network.pdf"
    fig.savefig(out, dpi=300, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_physics_validation_scenario1(data: dict) -> None:
    """Physics validation Scenario 1 - per-station ensemble scores.

    Bar chart color-coded by station type (DART blue, CO-OPS orange),
    sorted by epicentral distance, with FSM threshold lines.
    """
    scenarios = data.get("scenarios", [])
    if not scenarios:
        logger.warning("No scenarios in physics_validation.json")
        return

    stations = scenarios[0].get("stations", [])
    if not stations:
        logger.warning("No stations in Scenario 1")
        return

    # Sort by distance
    stations = sorted(stations, key=lambda s: s["distance_km"])

    labels = [f"{s['station_id']}\n({s['distance_km']:.0f} km)" for s in stations]
    scores = [s["ensemble_score"] for s in stations]
    colors = [
        "#2196F3" if s["station_type"] == "dart" else "#FF9800"
        for s in stations
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(stations))
    ax.bar(x, scores, color=colors, edgecolor="white", linewidth=0.5)

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7)
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7)
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7)

    # Legend entries for station types + thresholds (proxy artists)
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2196F3", label="DART"),
        Patch(facecolor="#FF9800", label="CO-OPS"),
        Line2D([0], [0], color="gray", linestyle="--", alpha=0.7, label=f"T1 = {T1}"),
        Line2D([0], [0], color="gray", linestyle="-.", alpha=0.7, label=f"T2 = {T2}"),
        Line2D([0], [0], color="gray", linestyle=":", alpha=0.7, label=f"T3 = {T3}"),
    ]
    ax.legend(handles=legend_elements, loc="center right", fontsize=8)

    ax.set_xlabel("Station (epicentral distance)", fontsize=9)
    ax.set_ylabel("Ensemble Anomaly Score", fontsize=9)
    sc1_name = scenarios[0].get("name", "Scenario 1")
    ax.set_title(
        f"Physics Simulation: {sc1_name} "
        f"({sum(1 for s in stations if s['station_type']=='dart')} DART and "
        f"{sum(1 for s in stations if s['station_type']=='coops')} CO-OPS Stations)",
        fontsize=10,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1.08)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_physics_scenario1.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_physics_validation_summary(data: dict) -> None:
    """Physics validation - 2x2 summary of all four scenarios."""
    scenarios = data.get("scenarios", [])
    if len(scenarios) < 4:
        logger.warning("Expected 4 scenarios, got %d", len(scenarios))
        return

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 9))

    # --- (a) Scenario 1: large tsunami ---
    s1 = sorted(scenarios[0]["stations"], key=lambda s: s["distance_km"])
    sc1_name = scenarios[0].get("name", "Large Tsunami")
    n_detect_1 = sum(1 for s in s1 if s["ensemble_score"] > T1)
    labels1 = [s["station_id"] for s in s1]
    scores1 = [s["ensemble_score"] for s in s1]
    colors1 = ["#2196F3" if s["station_type"] == "dart" else "#FF9800" for s in s1]
    x1 = np.arange(len(s1))
    ax1.bar(x1, scores1, color=colors1, edgecolor="white", linewidth=0.5)
    ax1.axhline(y=T1, color="gray", linestyle="--", alpha=0.6)
    ax1.axhline(y=T2, color="gray", linestyle="-.", alpha=0.6)
    ax1.axhline(y=T3, color="gray", linestyle=":", alpha=0.6)
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels1, fontsize=7, rotation=45, ha="right")
    ax1.set_ylim(0, 1.08)
    ax1.set_ylabel("Ensemble Score", fontsize=8)
    ax1.set_title(f"(a) {sc1_name}: {n_detect_1} of {len(s1)} Detect", fontsize=9)

    # --- (b) Scenario 2: moderate ---
    s2 = sorted(scenarios[1]["stations"], key=lambda s: s["distance_km"])
    sc2_name = scenarios[1].get("name", "Moderate")
    labels2 = [s["station_id"] for s in s2]
    scores2 = [s["ensemble_score"] for s in s2]
    x2 = np.arange(len(s2))
    ax2.bar(x2, scores2, color="#2196F3", edgecolor="white", linewidth=0.5)
    ax2.axhline(y=T1, color="gray", linestyle="--", alpha=0.6, label=f"T1 = {T1}")
    ax2.set_xticks(x2)
    ax2.set_xticklabels(labels2, fontsize=7)
    ax2.set_ylim(0, 1.08)
    ax2.set_ylabel("Ensemble Score", fontsize=8)
    ax2.set_title(f"(b) {sc2_name}: No False Escalation", fontsize=9)
    ax2.text(
        0.5, 0.55, "All scores < T1",
        transform=ax2.transAxes, ha="center", fontsize=11,
        color="#4CAF50", fontweight="bold",
    )

    # --- (c) Scenario 3: Meteotsunami ---
    result3 = scenarios[2].get("result", {})
    score3 = result3.get("ensemble_score", 0.0)
    meteo_amp = scenarios[2].get("meteotsunami", {}).get("amplitude_m", 0.10)
    coops_thr = data.get("physical_constants", {}).get(
        "coops_detection_threshold_m", 0.15)
    ax3.bar([0], [score3], color="#9C27B0", width=0.4, edgecolor="white")
    ax3.axhline(y=T1, color="gray", linestyle="--", alpha=0.6, label=f"T1 = {T1}")
    ax3.set_xticks([0])
    ax3.set_xticklabels(["coops_honolulu"], fontsize=7)
    ax3.set_ylim(0, 1.08)
    ax3.set_ylabel("Ensemble Score", fontsize=8)
    ax3.set_title("(c) Meteotsunami: Correctly Rejected", fontsize=9)
    ax3.text(
        0.5, 0.55,
        f"Score = {score3:.2f}\n"
        f"Amplitude {meteo_amp:.2f} m < {coops_thr:.2f} m CO-OPS threshold\n"
        "(seismic quiet penalty adds margin)",
        transform=ax3.transAxes, ha="center", fontsize=9,
        color="#4CAF50", fontweight="bold",
    )

    # --- (d) Scenario 4: Partial outage ---
    s4 = sorted(scenarios[3]["stations"], key=lambda s: s["distance_km"])
    offline = scenarios[3].get("offline_stations", [])
    # Online stations
    labels4 = [s["station_id"] for s in s4]
    scores4 = [s["ensemble_score"] for s in s4]
    x4_online = np.arange(len(s4))
    ax4.bar(x4_online, scores4, color="#2196F3", edgecolor="white", linewidth=0.5)
    # Offline markers
    x4_offline = np.arange(len(s4), len(s4) + len(offline))
    ax4.bar(x4_offline, [0] * len(offline), color="#BDBDBD", edgecolor="white",
            linewidth=0.5)
    for i, xo in enumerate(x4_offline):
        ax4.text(xo, 0.02, "OFFLINE", ha="center", va="bottom", fontsize=6,
                 color="#757575", rotation=90)
    all_labels4 = labels4 + offline
    ax4.axhline(y=T1, color="gray", linestyle="--", alpha=0.6)
    ax4.axhline(y=T3, color="gray", linestyle=":", alpha=0.6)
    ax4.set_xticks(np.arange(len(all_labels4)))
    ax4.set_xticklabels(all_labels4, fontsize=7, rotation=45, ha="right")
    ax4.set_ylim(0, 1.08)
    ax4.set_ylabel("Ensemble Score", fontsize=8)
    n_online = len(s4)
    n_total = n_online + len(offline)
    n_detect = sum(1 for s in s4 if s["ensemble_score"] >= T1)
    ax4.set_title(
        f"(d) Partial Outage: {n_online} of {n_total} Online, {n_detect} Detect",
        fontsize=9)

    fig.suptitle("Physics Simulation Validation: Four Scenarios", fontsize=12,
                 fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out = FIGURES_DIR / "fig_physics_summary.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_dashboard_layout() -> None:
    """Mission Control dashboard layout schematic.

    Programmatic mockup using matplotlib patches showing the 8-panel
    grid layout matching the React App.tsx structure.
    """
    from matplotlib.patches import FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-0.5, 8.0)
    ax.axis("off")

    bg_color = "#1a1a2e"
    panel_color = "#16213e"
    panel_border = "#0f3460"
    header_color = "#0f3460"
    accent_green = "#4CAF50"
    accent_blue = "#2196F3"
    accent_orange = "#FF9800"
    accent_red = "#e94560"
    text_color = "#e0e0e0"

    fig.patch.set_facecolor(bg_color)

    def add_panel(x, y, w, h, title, subtitle="", title_color=text_color):
        panel = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.05",
            facecolor=panel_color, edgecolor=panel_border, linewidth=1.5,
        )
        ax.add_patch(panel)
        # Title bar
        title_bar = FancyBboxPatch(
            (x + 0.05, y + h - 0.35), w - 0.1, 0.3,
            boxstyle="round,pad=0.02",
            facecolor=header_color, edgecolor="none",
        )
        ax.add_patch(title_bar)
        ax.text(
            x + w / 2, y + h - 0.2, title,
            ha="center", va="center", fontsize=8, fontweight="bold",
            color=title_color, family="monospace",
        )
        if subtitle:
            ax.text(
                x + w / 2, y + h / 2 - 0.1, subtitle,
                ha="center", va="center", fontsize=6.5,
                color="#9e9e9e", family="monospace", style="italic",
            )

    # Header bar (full width)
    header = FancyBboxPatch(
        (0, 7.1), 11, 0.7,
        boxstyle="round,pad=0.05",
        facecolor=header_color, edgecolor=panel_border, linewidth=1.5,
    )
    ax.add_patch(header)
    ax.text(0.3, 7.45, "OCEAN HAZARD MISSION CONTROL", fontsize=10,
            fontweight="bold", color=text_color, family="monospace")
    ax.text(5.5, 7.45, "UTC 2026-03-17 09:15:42", fontsize=8,
            color="#9e9e9e", family="monospace", ha="center")
    # FSM state badge
    fsm_badge = FancyBboxPatch(
        (7.5, 7.25), 1.2, 0.4,
        boxstyle="round,pad=0.05",
        facecolor=accent_red, edgecolor="none",
    )
    ax.add_patch(fsm_badge)
    ax.text(8.1, 7.45, "ESCALATE", fontsize=7, fontweight="bold",
            color="white", family="monospace", ha="center")
    # Score bar
    ax.text(9.0, 7.45, "Score:", fontsize=7, color="#9e9e9e", family="monospace")
    score_bg = FancyBboxPatch(
        (9.6, 7.3), 1.2, 0.3,
        boxstyle="round,pad=0.02",
        facecolor="#333", edgecolor=panel_border, linewidth=0.5,
    )
    ax.add_patch(score_bg)
    score_fill = FancyBboxPatch(
        (9.6, 7.3), 1.2 * 0.997, 0.3,
        boxstyle="round,pad=0.02",
        facecolor=accent_red, edgecolor="none",
    )
    ax.add_patch(score_fill)
    ax.text(10.2, 7.45, "0.997", fontsize=6.5, color="white",
            family="monospace", ha="center")

    # Row 1: FSM Panel | Ocean Map | Review Gate
    add_panel(0, 3.8, 2.0, 3.1, "FSM STATE",
              "IDLE > MON > INV > ASS > ESC\n\nTransition history\nDwell times",
              title_color=accent_blue)
    add_panel(2.2, 3.8, 5.8, 3.1, "OCEAN MAP",
              "Pacific basin (Leaflet)\nDART markers (blue)\n"
              "CO-OPS markers (orange)\nEpicenter + distance circles",
              title_color=accent_blue)
    add_panel(8.2, 3.8, 2.8, 3.1, "HUMAN REVIEW GATE",
              "Evidence summary\nScenario info\nVerification checks\n\nAPPROVE | REJECT | DEFER",
              title_color=accent_red)

    # Row 2: Agent Status | Anomaly Chart | Events
    add_panel(0, 1.0, 2.0, 2.6, "AGENT STATUS",
              "QC Agent\nAnomaly Agent\nScenario Agent\nVerification Agent",
              title_color=accent_green)
    add_panel(2.2, 1.0, 5.8, 2.6, "ANOMALY SCORE TIMELINE",
              "Time-series (Recharts)\nThreshold lines: T1=0.35, "
              "T2=0.60, T3=0.85\nRolling 30-min window",
              title_color=accent_orange)
    add_panel(8.2, 1.0, 2.8, 2.6, "ACTIVE EVENTS",
              "Event list\nPipeline status\nStation coverage",
              title_color=accent_green)

    # Row 3: Audit Log (full width)
    add_panel(0, 0, 11, 0.8, "AUDIT LOG",
              "Append-only audit trail entries (scrolling, max-height 140px)",
              title_color="#9e9e9e")

    # Annotations with arrows
    ax.annotate(
        "WebSocket\nlive stream",
        xy=(5.5, 7.1), xytext=(5.5, 8.0),
        fontsize=7, color=accent_green, family="monospace",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=accent_green, lw=1.2),
    )
    ax.annotate(
        "Human-in-\nthe-loop",
        xy=(9.6, 6.9), xytext=(11.0, 7.8),
        fontsize=7, color=accent_red, family="monospace",
        ha="center", va="bottom",
        arrowprops=dict(arrowstyle="->", color=accent_red, lw=1.2),
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig_dashboard_layout.pdf"
    fig.savefig(out, dpi=300, facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info("Saved %s", out)


# ===================================================================
# Synthetic time-series figures
# ===================================================================

def fig_simulation_station_map(data: dict) -> None:
    """Station map for simulation: epicenter + all DART/CO-OPS stations.

    Shows the Scenario 1 epicenter with all stations from the simulation
    network, matching the style of Figs 3 and 10 (real-event station maps).
    """
    from hazard_assessment.simulation.propagation import (
        PACIFIC_COOPS_STATIONS,
        PACIFIC_DART_STATIONS,
        compute_arrival_time_hours,
    )

    scenarios = data.get("scenarios", [])
    if not scenarios:
        logger.warning("No scenarios in synthetic_timelines.json for station map")
        return

    eq = scenarios[0]["earthquake"]
    epic_lat = eq["latitude"]
    epic_lon = eq["longitude"]

    # Build station metadata: (id, distance_km, lat, lon)
    all_stations = list(PACIFIC_DART_STATIONS) + list(PACIFIC_COOPS_STATIONS)
    station_meta: list[tuple[str, int, float, float]] = []
    for sc in all_stations:
        arr_h = compute_arrival_time_hours(epic_lat, epic_lon, sc.latitude, sc.longitude)
        dist_km = int(round(arr_h * 3600 * 198.0 / 1000))
        station_meta.append((sc.station_id, dist_km, sc.latitude, sc.longitude))

    station_meta.sort(key=lambda s: s[1])

    # Label offsets to avoid overlap
    label_offsets = {
        "21418": (4.0, -3.0, "left"),
        "21413": (4.0, -3.0, "left"),
        "21419": (3.0, 3.0, "left"),
        "46404": (-6.0, 2.0, "right"),
        "46407": (-6.0, -2.5, "right"),
        "46411": (-6.0, -2.0, "right"),
        "1612340": (3.0, 2.0, "left"),
        "1617760": (3.0, -3.0, "left"),
        "1619910": (-5.0, -3.0, "right"),
    }

    sc_name = scenarios[0].get("name", "Simulation")
    _station_map_plain(
        station_meta=station_meta,
        epicenter_lat=epic_lat,
        epicenter_lon=epic_lon,
        title=f"Simulation Station Network: {sc_name} Scenario",
        outfile="fig_simulation_station_map.pdf",
        label_offsets=label_offsets,
        lat_margin_bottom=10,
    )


def fig_simulation_score_overlay(data: dict) -> None:
    """Simulation score + waveform overlay (like Figs 5 and 9).

    Two full-width panels per station (bandpass-filtered residual, score
    decomposition) plus a 1/4-width onset zoom panel - matching the format
    used for Tohoku and Chile real-event figures.

    Reads waveform data and scores from synthetic_timelines.json rather
    than from CSV files.
    """
    scenarios = data.get("scenarios", [])
    if not scenarios:
        logger.warning("No scenarios in synthetic_timelines.json for overlay")
        return

    scenario = scenarios[0]
    stations = sorted(scenario["stations"],
                      key=lambda s: s["waveform_data"]["distance_km"])

    # Only stations that have timeline data
    stations = [s for s in stations if s.get("timeline")]
    if not stations:
        logger.warning("No stations with timeline data for simulation overlay")
        return

    n_stations = len(stations)
    n_rows = 3  # filtered, scores, zoom-placeholder per station
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )
    if n_stations * n_rows == 1:
        axes = [axes]

    zoom_infos: list[tuple[int, dict]] = []
    for i, stn in enumerate(stations):
        wd = stn["waveform_data"]
        tl = stn["timeline"]
        sid = stn["station_id"]
        dist_km = wd["distance_km"]
        arr_min = wd["arrival_min"]

        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        ax_zoom.set_visible(False)

        ev_m = np.array(wd["times_min"])
        filtered = np.array(wd["filtered_signal"])

        # --- Filtered residual panel ---
        ax_filt.plot(ev_m, filtered, "k-", linewidth=0.6)
        ax_filt.axvline(x=0, color="red", linewidth=1, alpha=0.7)
        ax_filt.axvline(x=arr_min, color="#e94560", linewidth=0.8,
                        linestyle="--", alpha=0.8, label="Expected arrival")

        # Find score onset
        score_times = [p["minutes"] for p in tl if p["ensemble"] > 0.01]
        t_onset = score_times[0] if score_times else None

        if t_onset is not None:
            ax_filt.axvline(x=t_onset, color="#2196F3", linewidth=1,
                            linestyle="--", alpha=0.8)

        station_label = f"{sid} ({dist_km:,.0f} km)"
        ax_filt.set_ylabel("Filtered (m)", fontsize=8)
        ax_filt.text(0.02, 0.85, station_label,
                     transform=ax_filt.transAxes, fontsize=8, va="top",
                     bbox={"boxstyle": "round,pad=0.2", "facecolor": "white",
                           "alpha": 0.8})
        ax_filt.tick_params(labelsize=7)

        # --- Score panel ---
        sw_mins = [p["minutes"] for p in tl]
        ax_score.plot(sw_mins, [p["threshold"] for p in tl],
                      label="Threshold", color="#4CAF50", linewidth=1)
        ax_score.plot(sw_mins, [p["wavelet"] for p in tl],
                      label="Wavelet", color="#FF9800", linewidth=1)
        ax_score.plot(sw_mins, [p["bocpd"] for p in tl],
                      label="BOCPD", color="#9C27B0", linewidth=1)
        ax_score.plot(sw_mins, [p["ensemble"] for p in tl],
                      label="Ensemble", color="#2196F3", linewidth=2)
        ax_score.axvline(x=0, color="red", linewidth=1, alpha=0.7)
        ax_score.axvline(x=arr_min, color="#e94560", linewidth=0.8,
                         linestyle="--", alpha=0.8)
        if t_onset is not None:
            ax_score.axvline(x=t_onset, color="#2196F3", linewidth=1,
                             linestyle="--", alpha=0.8)
        ax_score.axhline(y=T1, color="gray", linestyle="--", alpha=0.5, linewidth=0.7)
        ax_score.axhline(y=T2, color="gray", linestyle="-.", alpha=0.5, linewidth=0.7)
        ax_score.axhline(y=T3, color="gray", linestyle=":", alpha=0.5, linewidth=0.7)
        ax_score.text(1.01, T1, "$T_1$", transform=ax_score.get_yaxis_transform(),
                      fontsize=6, va="center", color="gray")
        ax_score.text(1.01, T2, "$T_2$", transform=ax_score.get_yaxis_transform(),
                      fontsize=6, va="center", color="gray")
        ax_score.text(1.01, T3, "$T_3$", transform=ax_score.get_yaxis_transform(),
                      fontsize=6, va="center", color="gray")
        ax_score.set_ylim(0, 1.05)
        ax_score.set_ylabel("Score", fontsize=8)
        ax_score.tick_params(labelsize=7)
        ax_score.legend(loc="lower right", fontsize=7, ncol=4, framealpha=0.9)

        zoom_infos.append((i, {
            "t_onset": t_onset,
            "ev_m": ev_m,
            "filtered": filtered,
            "has_waveform": True,
            "p_wave_min": dist_km / (8.0 * 60),
        }))

    # X-label on last score panel
    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title(
        f"Simulation: {scenario['name']} - Bandpass-Filtered Residual "
        f"and Detection Scores")
    fig.tight_layout()

    # Reposition zoom placeholders
    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_simulation_score_overlay.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_synthetic_waveforms(data: dict) -> None:
    """Raw synthetic DART signals - vertical stack per station."""
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return

    # Plot Scenario 1 (large tsunami)
    scenario = scenarios[0]
    stations = sorted(scenario["stations"], key=lambda s: s["waveform_data"]["distance_km"])
    n = len(stations)

    fig, axes = plt.subplots(n, 1, figsize=(10, 2.2 * n), sharex=True, sharey=True)
    if n == 1:
        axes = [axes]

    for ax, stn in zip(axes, stations):
        wd = stn["waveform_data"]
        t = np.array(wd["times_min"]) / 60.0  # hours
        ax.plot(t, wd["event_signal"], color="#2196F3", linewidth=0.5, label="Observed")
        ax.plot(t, wd["clean_signal"], color="gray", linewidth=0.4,
               alpha=0.6, label="Tidal baseline")

        # Arrival marker
        arr_h = wd["arrival_min"] / 60.0
        ax.axvline(arr_h, color="#e94560", linestyle="--", linewidth=0.8, alpha=0.8)

        dist = wd["distance_km"]
        amp = wd["tsunami_amplitude_m"]
        # Name the quantity. This is the characteristic amplitude carried to
        # the station, not the peak of the trace in the panel, which runs a
        # few times larger. The bare number read as a peak and disagreed with
        # the plotted excursion by a factor of about three.
        label = f"{stn['station_id']} ({dist:.0f} km, characteristic {amp:.4f} m)"
        ax.set_ylabel("BPR (m)", fontsize=7)
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=7, va="top", fontweight="bold")
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel("Hours from event start", fontsize=8)
    axes[0].set_title(
        f"Synthetic DART Signals: {scenario['name']}",
        fontsize=10, fontweight="bold",
    )

    fig.tight_layout()
    out = FIGURES_DIR / "fig_synthetic_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_synthetic_residuals(data: dict) -> None:
    """Detided residuals showing isolated tsunami signal.

    Uses shared Y-axes so amplitude decay with distance is visible,
    plus an overlay panel comparing all stations on one axis.
    """
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return

    scenario = scenarios[0]
    stations = sorted(scenario["stations"], key=lambda s: s["waveform_data"]["distance_km"])
    n = len(stations)

    # n per-station rows + 1 overlay comparison row
    n_rows = n + 1
    height_ratios = [2.0] * n + [2.5]
    fig, axes = plt.subplots(
        n_rows, 1, figsize=(10, 2.0 * n + 3.0), sharex=True,
        gridspec_kw={"height_ratios": height_ratios},
    )

    # Compute shared Y limits from all filtered signals
    all_filtered = []
    for stn in stations:
        all_filtered.extend(stn["waveform_data"]["filtered_signal"])
    y_max = max(abs(v) for v in all_filtered) * 1.15 if all_filtered else 0.1
    shared_ylim = (-y_max, y_max)

    # Per-station residual panels with shared Y-axis
    for ax, stn in zip(axes[:n], stations):
        wd = stn["waveform_data"]
        t = np.array(wd["times_min"]) / 60.0
        ax.plot(t, wd["detided_residual"], color="#4CAF50", linewidth=0.5, label="Detided")
        ax.plot(t, wd["filtered_signal"], color="#FF9800", linewidth=0.5,
               alpha=0.7, label="Bandpass filtered")

        arr_h = wd["arrival_min"] / 60.0
        ax.axvline(arr_h, color="#e94560", linestyle="--", linewidth=0.8, alpha=0.8)
        ax.axhline(0, color="gray", linewidth=0.3)

        dist = wd["distance_km"]
        amp = wd["tsunami_amplitude_m"]
        # Name the quantity. This is the characteristic amplitude carried to
        # the station, not the peak of the trace in the panel, which runs a
        # few times larger. The bare number read as a peak and disagreed with
        # the plotted excursion by a factor of about three.
        label = f"{stn['station_id']} ({dist:.0f} km, characteristic {amp:.4f} m)"
        ax.set_ylabel("Residual (m)", fontsize=7)
        ax.set_ylim(shared_ylim)
        ax.text(0.01, 0.95, label, transform=ax.transAxes,
                fontsize=7, va="top", fontweight="bold")
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, loc="upper right")
    axes[0].set_title(
        f"Detided Residuals: {scenario['name']}",
        fontsize=10, fontweight="bold",
    )

    # Overlay comparison panel - all stations on one axis
    ax_overlay = axes[n]
    colors = plt.cm.viridis(np.linspace(0, 0.85, n))
    for idx, stn in enumerate(stations):
        wd = stn["waveform_data"]
        t = np.array(wd["times_min"]) / 60.0
        dist = wd["distance_km"]
        ax_overlay.plot(t, wd["filtered_signal"], color=colors[idx], linewidth=0.6,
                        label=f"{stn['station_id']} ({dist:.0f} km)")
        ax_overlay.axvline(wd["arrival_min"] / 60.0, color=colors[idx],
                           linestyle="--", linewidth=0.5, alpha=0.6)
    ax_overlay.axhline(0, color="gray", linewidth=0.3)
    ax_overlay.set_ylim(shared_ylim)
    ax_overlay.set_xlabel("Hours from event start", fontsize=8)
    ax_overlay.set_ylabel("Filtered (m)", fontsize=7)
    ax_overlay.legend(fontsize=6, ncol=3, loc="upper right")
    ax_overlay.set_title("All Stations Compared (bandpass-filtered)", fontsize=9, fontweight="bold")
    ax_overlay.tick_params(labelsize=7)

    fig.tight_layout()
    out = FIGURES_DIR / "fig_synthetic_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_synthetic_score_timeline(data: dict) -> None:
    """Side-by-side score evolution: large (left) vs moderate (right)."""
    scenarios = data.get("scenarios", [])
    if len(scenarios) < 2:
        return

    n_rows = max(
        len(scenarios[0]["stations"]),
        len(scenarios[1]["stations"]),
    )
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 2.5 * n_rows), squeeze=False)

    colors = {
        "threshold": "#4CAF50",
        "wavelet": "#FF9800",
        "bocpd": "#9C27B0",
        "ensemble": "#2196F3",
    }

    for col_idx, scenario in enumerate(scenarios[:2]):
        stations = sorted(scenario["stations"], key=lambda s: s["waveform_data"]["distance_km"])
        for row_idx, stn in enumerate(stations):
            ax = axes[row_idx, col_idx]
            tl = stn["timeline"]
            if not tl:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center")
                continue

            minutes = [p["minutes"] for p in tl]
            for key, color in colors.items():
                vals = [p[key] for p in tl]
                lw = 1.5 if key == "ensemble" else 0.8
                display = "BOCPD" if key == "bocpd" else key.capitalize()
                ax.plot(minutes, vals, color=color, linewidth=lw, label=display)

            # Threshold lines
            ax.axhline(T1, color="green", linestyle="--", linewidth=0.5, alpha=0.5)
            ax.axhline(T2, color="orange", linestyle="-.", linewidth=0.5, alpha=0.5)
            ax.axhline(T3, color="red", linestyle=":", linewidth=0.5, alpha=0.5)

            # Arrival marker
            arr = stn["waveform_data"]["arrival_min"]
            ax.axvline(arr, color="#e94560", linestyle="--", linewidth=0.6, alpha=0.6)

            ax.set_ylim(-0.05, 1.05)
            dist = stn["waveform_data"]["distance_km"]
            ax.text(0.01, 0.95, f"{stn['station_id']} ({dist:.0f} km)",
                    transform=ax.transAxes, fontsize=7, va="top", fontweight="bold")
            ax.tick_params(labelsize=7)
            if row_idx == 0:
                ax.set_title(scenario["name"], fontsize=9, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel("Score", fontsize=7)

        # Hide unused rows
        for row_idx in range(len(stations), n_rows):
            axes[row_idx, col_idx].set_visible(False)

    axes[-1, 0].set_xlabel("Minutes from event start", fontsize=8)
    axes[-1, 1].set_xlabel("Minutes from event start", fontsize=8)
    axes[0, 0].legend(fontsize=6, loc="center left", ncol=2)

    fig.suptitle("Detection Score Evolution: Sliding-Window Analysis",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_synthetic_score_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_synthetic_multistation(data: dict) -> None:
    """3-column grid: raw signal | residual | score evolution."""
    scenarios = data.get("scenarios", [])
    if not scenarios:
        return

    scenario = scenarios[0]
    stations = sorted(scenario["stations"], key=lambda s: s["waveform_data"]["distance_km"])
    n = len(stations)

    fig, axes = plt.subplots(n, 3, figsize=(14, 2.5 * n), sharex="col", sharey="col")
    if n == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Raw Signal", "Detided Residual", "Score Evolution"]
    for j, title in enumerate(col_titles):
        axes[0, j].set_title(title, fontsize=9, fontweight="bold")

    for i, stn in enumerate(stations):
        wd = stn["waveform_data"]
        t_h = np.array(wd["times_min"]) / 60.0
        arr_h = wd["arrival_min"] / 60.0
        dist = wd["distance_km"]
        label = f"{stn['station_id']} ({dist:.0f} km)"

        # Col 1: raw signal
        ax1 = axes[i, 0]
        ax1.plot(t_h, wd["event_signal"], color="#2196F3", linewidth=0.4)
        ax1.axvline(arr_h, color="#e94560", linestyle="--", linewidth=0.6)
        ax1.set_ylabel("BPR (m)", fontsize=6)
        ax1.text(0.01, 0.95, label, transform=ax1.transAxes,
                fontsize=6, va="top", fontweight="bold")
        ax1.tick_params(labelsize=6)

        # Col 2: residual
        ax2 = axes[i, 1]
        ax2.plot(t_h, wd["detided_residual"], color="#4CAF50", linewidth=0.4)
        ax2.axvline(arr_h, color="#e94560", linestyle="--", linewidth=0.6)
        ax2.axhline(0, color="gray", linewidth=0.3)
        ax2.set_ylabel("Residual (m)", fontsize=6)
        ax2.tick_params(labelsize=6)

        # Col 3: score evolution
        ax3 = axes[i, 2]
        tl = stn["timeline"]
        if tl:
            mins = [p["minutes"] for p in tl]
            ax3.plot(mins, [p["ensemble"] for p in tl],
                    color="#2196F3", linewidth=1.2, label="Ensemble")
            ax3.plot(mins, [p["threshold"] for p in tl], color="#4CAF50", linewidth=0.6, alpha=0.7)
            ax3.plot(mins, [p["wavelet"] for p in tl], color="#FF9800", linewidth=0.6, alpha=0.7)
            ax3.plot(mins, [p["bocpd"] for p in tl], color="#9C27B0", linewidth=0.6, alpha=0.7)
            ax3.axhline(T1, color="green", linestyle="--", linewidth=0.4, alpha=0.5)
            ax3.axhline(T2, color="orange", linestyle="-.", linewidth=0.4, alpha=0.5)
            ax3.axhline(T3, color="red", linestyle=":", linewidth=0.4, alpha=0.5)
            ax3.axvline(wd["arrival_min"], color="#e94560", linestyle="--", linewidth=0.6)
            ax3.set_ylim(-0.05, 1.05)
        ax3.set_ylabel("Score", fontsize=6)
        ax3.tick_params(labelsize=6)

    for j in range(3):
        axes[-1, j].set_xlabel("Hours" if j < 2 else "Minutes", fontsize=7)

    fig.suptitle(f"Multi-Station Analysis: {scenario['name']}",
                 fontsize=11, fontweight="bold", y=0.99)
    fig.tight_layout()
    out = FIGURES_DIR / "fig_synthetic_multistation.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_illapel_detection(data: dict) -> None:
    """Illapel 2015 detection - per-station score bar chart."""
    stations = data.get("per_station", [])
    if not stations:
        logger.warning("No per-station data for Illapel detection figure")
        return

    # Sort by distance to match waveform/residual figure ordering
    stations = sorted(stations, key=lambda s:
        next((d for sid, d, _, _ in ILLAPEL_STATION_META if sid == s["station_id"]), 0))

    # Exclude degraded (standard-mode) stations - scores are unreliable
    stations = [s for s in stations if not s.get("filter_degraded")]
    if not stations:
        logger.warning("No event-mode stations for Illapel detection figure")
        return

    station_ids = [s["station_id"] for s in stations]
    ensemble_scores = [s["ensemble_score"] for s in stations]
    threshold_scores = [s["threshold_score"] for s in stations]
    wavelet_scores = [s["wavelet_score"] for s in stations]
    bocpd_scores = [s["bocpd_score"] for s in stations]

    n = len(station_ids)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * n), 5))
    ax.bar(x - 1.5 * width, ensemble_scores, width, label="Ensemble", color="#2196F3")
    ax.bar(x - 0.5 * width, threshold_scores, width, label="Threshold", color="#4CAF50")
    ax.bar(x + 0.5 * width, wavelet_scores, width, label="Wavelet", color="#FF9800")
    ax.bar(x + 1.5 * width, bocpd_scores, width, label="BOCPD", color="#9C27B0")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7, label=f"T1={T1}")
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7, label=f"T2={T2}")
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7, label=f"T3={T3}")

    ax.set_xlabel("DART Station")
    ax.set_ylabel("Anomaly Score")
    ax.set_title(
        "Illapel 2015 (Mw 8.3): Per-Station Anomaly Detection Scores"
    )
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_illapel_detection.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_iquique_detection(data: dict) -> None:
    """Iquique 2014 detection - per-station score bar chart."""
    stations = data.get("per_station", [])
    if not stations:
        logger.warning("No per-station data for Iquique detection figure")
        return

    # Sort by distance to match waveform/residual figure ordering
    stations = sorted(stations, key=lambda s:
        next((d for sid, d, _, _ in IQUIQUE_STATION_META if sid == s["station_id"]), 0))

    # Exclude degraded (standard-mode) stations - scores are unreliable
    stations = [s for s in stations if not s.get("filter_degraded")]
    if not stations:
        logger.warning("No event-mode stations for Iquique detection figure")
        return

    station_ids = [s["station_id"] for s in stations]
    ensemble_scores = [s["ensemble_score"] for s in stations]
    threshold_scores = [s["threshold_score"] for s in stations]
    wavelet_scores = [s["wavelet_score"] for s in stations]
    bocpd_scores = [s["bocpd_score"] for s in stations]

    n = len(station_ids)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * n), 5))
    ax.bar(x - 1.5 * width, ensemble_scores, width, label="Ensemble", color="#2196F3")
    ax.bar(x - 0.5 * width, threshold_scores, width, label="Threshold", color="#4CAF50")
    ax.bar(x + 0.5 * width, wavelet_scores, width, label="Wavelet", color="#FF9800")
    ax.bar(x + 1.5 * width, bocpd_scores, width, label="BOCPD", color="#9C27B0")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7, label=f"T1={T1}")
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7, label=f"T2={T2}")
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7, label=f"T3={T3}")

    ax.set_xlabel("DART Station")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Iquique 2014 (Mw 8.2): Per-Station Anomaly Detection Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_iquique_detection.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_samoa_detection(data: dict) -> None:
    """Samoa 2009 detection - per-station score bar chart."""
    stations = data.get("per_station", [])
    if not stations:
        logger.warning("No per-station data for Samoa detection figure")
        return

    # Sort by distance to match waveform/residual figure ordering
    stations = sorted(stations, key=lambda s:
        next((d for sid, d, _, _ in SAMOA_STATION_META if sid == s["station_id"]), 0))

    # Exclude degraded (standard-mode) stations - scores are unreliable
    stations = [s for s in stations if not s.get("filter_degraded")]
    if not stations:
        logger.warning("No event-mode stations for Samoa detection figure")
        return

    station_ids = [s["station_id"] for s in stations]
    ensemble_scores = [s["ensemble_score"] for s in stations]
    threshold_scores = [s["threshold_score"] for s in stations]
    wavelet_scores = [s["wavelet_score"] for s in stations]
    bocpd_scores = [s["bocpd_score"] for s in stations]

    n = len(station_ids)
    x = np.arange(n)
    width = 0.2

    fig, ax = plt.subplots(figsize=(max(6, 2.0 * n), 5))
    ax.bar(x - 1.5 * width, ensemble_scores, width, label="Ensemble", color="#2196F3")
    ax.bar(x - 0.5 * width, threshold_scores, width, label="Threshold", color="#4CAF50")
    ax.bar(x + 0.5 * width, wavelet_scores, width, label="Wavelet", color="#FF9800")
    ax.bar(x + 1.5 * width, bocpd_scores, width, label="BOCPD", color="#9C27B0")

    ax.axhline(y=T1, color="gray", linestyle="--", alpha=0.7, label=f"T1={T1}")
    ax.axhline(y=T2, color="gray", linestyle="-.", alpha=0.7, label=f"T2={T2}")
    ax.axhline(y=T3, color="gray", linestyle=":", alpha=0.7, label=f"T3={T3}")

    ax.set_xlabel("DART Station")
    ax.set_ylabel("Anomaly Score")
    ax.set_title("Samoa 2009 (Mw 8.1): Per-Station Anomaly Detection Scores")
    ax.set_xticks(x)
    ax.set_xticklabels(station_ids, rotation=45, ha="right")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = FIGURES_DIR / "fig_samoa_detection.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


# -- Illapel 2015 figures -----------------------------------------------

ILLAPEL_DATA_DIR = Path("data/illapel")
ILLAPEL_LAT = -31.573
ILLAPEL_LON = -71.674

ILLAPEL_STATION_META: list[tuple[str, int, float, float]] = [
    ("32402",   580, -26.733, -73.980),
    ("32401",  1250, -20.442, -73.422),
    ("32412",  2110, -17.984, -86.374),
    ("51426",  9280, -23.110, -168.385),
    ("46411",  9740, 39.337, -127.040),
    ("46407", 10110, 42.704, -128.895),
    ("46403", 12450, 52.647, -156.940),
]


def _load_illapel_event_csv(station_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = ILLAPEL_DATA_DIR / f"dart_{station_id}_illapel_2015_event.csv"
    if not path.exists():
        return None
    minutes: list[float] = []
    heights: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            minutes.append(float(row["seconds_from_origin"]) / 60.0)
            heights.append(h)
    if not minutes:
        return None
    return np.array(minutes), np.array(heights)


def fig_illapel_waveforms() -> None:
    """Illapel 2015 raw BPR time series."""
    event_stations = [(sid, d) for sid, d, _, _ in ILLAPEL_STATION_META]
    fig, axes = plt.subplots(len(event_stations), 1,
                             figsize=(10, 2.0 * len(event_stations)),
                             sharex=True)
    plotted = 0
    for ax, (sid, dist_km) in zip(axes, event_stations):
        result = _load_illapel_event_csv(sid)
        if result is None:
            ax.text(0.5, 0.5, f"{sid}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue
        mins, heights = result
        heights_dm = heights - np.mean(heights)
        med = np.median(heights_dm)
        glitch_mask = np.abs(heights_dm - med) > 5.0
        if np.any(glitch_mask):
            heights_dm = heights_dm.copy()
            heights_dm[glitch_mask] = np.nan
        _plot_with_gaps(ax, mins, heights_dm)
        cal_path = ILLAPEL_DATA_DIR / f"dart_{sid}_illapel_2015_calibration.csv"
        ev_path = ILLAPEL_DATA_DIR / f"dart_{sid}_illapel_2015_event.csv"
        tide_data = _load_tidal_prediction(cal_path, ev_path)
        if tide_data is not None:
            _ev_t, _ev_h, predicted = tide_data
            pred_dm = predicted - np.mean(heights)
            label = "Predicted tide" if plotted == 0 else None
            ax.plot(_ev_t * 60.0, pred_dm, "r--", linewidth=0.6, alpha=0.7,
                    label=label)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Illapel 2015 (Mw 8.3): Raw Bottom Pressure Records (De-meaned)")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_illapel_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_illapel_residuals() -> None:
    """Illapel 2015 detided residual time series (event-mode stations only)."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Illapel residuals")
        return

    import pandas as pd

    _ILLAPEL_RES_STATIONS = [
        s for s in ILLAPEL_STATION_META if s[0] in ("32402", "32401", "32412")
    ]

    fig, axes = plt.subplots(len(_ILLAPEL_RES_STATIONS), 1,
                             figsize=(10, 2.0 * len(_ILLAPEL_RES_STATIONS)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, _ILLAPEL_RES_STATIONS):
        cal_path = ILLAPEL_DATA_DIR / f"dart_{sid}_illapel_2015_calibration.csv"
        if not cal_path.exists():
            ax.text(0.5, 0.5, f"{sid}: no calibration", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue

        cal_times: list[float] = []
        cal_values: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                cal_times.append(float(row["seconds_from_origin"]) / 3600.0)
                cal_values.append(h)
        if len(cal_times) < 100:
            continue
        cal_t = np.array(cal_times)
        cal_h = np.array(cal_values)

        result = _load_illapel_event_csv(sid)
        if result is None:
            continue
        mins, ev_h = result
        ev_t = mins / 60.0

        # Despike: remove single-sample telemetry glitches
        if len(ev_h) > 2:
            left = np.concatenate([[ev_h[0]], ev_h[:-1]])
            right = np.concatenate([ev_h[1:], [ev_h[-1]]])
            neighbor_med = np.median(
                np.column_stack([left, ev_h, right]), axis=1)
            spike = np.abs(ev_h - neighbor_med) > 1.0
            if np.any(spike):
                logger.info("Despiked %d glitch(es) from %s event data",
                            int(np.sum(spike)), sid)
                ev_h = np.where(spike, neighbor_med, ev_h)

        harmonics = fit_tidal_harmonics(cal_t, cal_h, clean_input=True)
        predicted = predict_tide(ev_t, harmonics)
        ev_residual = ev_h - predicted

        # Pre-event linear detrend + centered rolling-median baseline.
        # Subtracting a 30-min centered median is a high-pass with a
        # ~30-min corner, not a drift-only correction.  Pushing a sinusoid
        # through this exact step gives a gain of 1.0 or more at periods of
        # 30 min and below, ~0.69 at 40 min and ~0.33 at 60 min, so it
        # removes real tsunami energy from the longer-period arrivals.
        # Re-running this pipeline on the archived records cuts the
        # post-origin peak by about half at Illapel 32401 and Iquique 32401
        # and to about a fifth at Samoa 51425.  This step does NOT preserve
        # tsunami peak amplitude.  The published captions do disclose the
        # median step, so the figures are not misdescribed, but do not read
        # these traces as calibrated amplitudes.
        pre_ev_mask = mins < 0
        if np.sum(pre_ev_mask) >= 2:
            bp = np.polyfit(mins[pre_ev_mask], ev_residual[pre_ev_mask], deg=1)
            ev_residual -= np.polyval(bp, mins)
        dt_min = float(np.median(np.diff(mins)))
        win = max(3, int(30.0 / max(dt_min, 0.1)))
        win = min(win, len(ev_residual) // 2)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(ev_residual, pad, mode='reflect')
        baseline = pd.Series(padded).rolling(
            window=win, min_periods=1, center=True,
        ).median().values[pad:-pad]
        ev_residual -= baseline

        hw = min(10, max(3, len(ev_residual) // 6))
        residual, _ = hampel_filter(ev_residual, half_window=hw, threshold=3.0)

        _plot_with_gaps(ax, mins, residual)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Illapel 2015 (Mw 8.3): Detided Residuals")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_illapel_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_illapel_detection_timeline(data: dict) -> None:
    """Illapel 2015 score overlay for event-mode stations."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            apply_bandpass_filter,
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Illapel timeline")
        return

    sliding = data.get("sliding_window", {})
    if not sliding:
        logger.warning("No sliding window data for Illapel timeline")
        return

    event_stations = [sid for sid in ["32402", "32401", "32412"]
                      if sid in sliding]
    if not event_stations:
        event_stations = list(sliding.keys())[:3]

    n_rows = 3
    n_stations = len(event_stations)
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )

    def _hampel_clean(residual):
        hw = min(10, max(3, len(residual) // 6))
        cleaned, _ = hampel_filter(residual, half_window=hw, threshold=3.0)
        return cleaned

    zoom_infos = []
    for i, sid in enumerate(event_stations):
        dist_km = next((d for s, d, _, _ in ILLAPEL_STATION_META if s == sid), 0)
        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        ax_zoom.set_visible(False)

        cal_path = ILLAPEL_DATA_DIR / f"dart_{sid}_illapel_2015_calibration.csv"
        event_path = ILLAPEL_DATA_DIR / f"dart_{sid}_illapel_2015_event.csv"

        info = _plot_score_overlay_station(
            (ax_filt, ax_score), sid, dist_km, sliding,
            cal_path, event_path,
            apply_bandpass_filter, fit_tidal_harmonics, predict_tide,
            clean_fn=_hampel_clean,
        )
        ax_filt.set_xlim(right=300)
        zoom_infos.append((i, info))

    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title(
        "Illapel 2015 (Mw 8.3): Bandpass-Filtered Residual and Detection Scores")
    fig.tight_layout()

    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_illapel_detection_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_illapel_station_map() -> None:
    """Illapel 2015 station location map."""
    _station_map_plain(
        station_meta=ILLAPEL_STATION_META,
        epicenter_lat=ILLAPEL_LAT,
        epicenter_lon=ILLAPEL_LON,
        title="DART Station Locations: Illapel 2015 Evaluation",
        outfile="fig_illapel_station_map.pdf",
        label_offsets={
            "32402": (3.0, -3.0, "left"),
            "32401": (-4.0, 5.5, "right"),   # left of dot, well above 32412
            "32412": (3.0, -3.0, "left"),   # right of dot, below 32401
            "51426": (3.0, -3.0, "left"),
            "46411": (-5.0, -2.0, "right"),  # left of dot: close to 46407
            "46407": (-5.0, 2.0, "right"),   # left of dot: above 46411
            "46403": (-5.0, 2.0, "right"),
        },
    )


# -- Iquique 2014 figures ----------------------------------------------

IQUIQUE_DATA_DIR = Path("data/iquique")
IQUIQUE_LAT = -19.610
IQUIQUE_LON = -70.769

IQUIQUE_STATION_META: list[tuple[str, int, float, float]] = [
    ("32401",   295, -20.469, -73.433),
    ("32402",   858, -26.733, -73.980),
    ("32412",  1652, -17.984, -86.374),
    ("32413",  2803, -7.407, -93.517),
    ("51426",  9900, -23.110, -168.385),
    ("46403", 11476, 52.647, -156.940),
]


def _load_iquique_event_csv(station_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = IQUIQUE_DATA_DIR / f"dart_{station_id}_iquique_2014_event.csv"
    if not path.exists():
        return None
    minutes: list[float] = []
    heights: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            minutes.append(float(row["seconds_from_origin"]) / 60.0)
            heights.append(h)
    if not minutes:
        return None
    return np.array(minutes), np.array(heights)


def fig_iquique_waveforms() -> None:
    """Iquique 2014 raw BPR time series."""
    event_stations = [(sid, d) for sid, d, _, _ in IQUIQUE_STATION_META]
    fig, axes = plt.subplots(len(event_stations), 1,
                             figsize=(10, 2.0 * len(event_stations)),
                             sharex=True)
    plotted = 0
    for ax, (sid, dist_km) in zip(axes, event_stations):
        result = _load_iquique_event_csv(sid)
        if result is None:
            ax.text(0.5, 0.5, f"{sid}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue
        mins, heights = result
        heights_dm = heights - np.mean(heights)
        med = np.median(heights_dm)
        glitch_mask = np.abs(heights_dm - med) > 5.0
        if np.any(glitch_mask):
            heights_dm = heights_dm.copy()
            heights_dm[glitch_mask] = np.nan
        _plot_with_gaps(ax, mins, heights_dm)
        cal_path = IQUIQUE_DATA_DIR / f"dart_{sid}_iquique_2014_calibration.csv"
        ev_path = IQUIQUE_DATA_DIR / f"dart_{sid}_iquique_2014_event.csv"
        tide_data = _load_tidal_prediction(cal_path, ev_path)
        if tide_data is not None:
            _ev_t, _ev_h, predicted = tide_data
            pred_dm = predicted - np.mean(heights)
            label = "Predicted tide" if plotted == 0 else None
            ax.plot(_ev_t * 60.0, pred_dm, "r--", linewidth=0.6, alpha=0.7,
                    label=label)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Iquique 2014 (Mw 8.2): Raw Bottom Pressure Records (De-meaned)")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_iquique_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_iquique_residuals() -> None:
    """Iquique 2014 detided residual time series (event-mode stations only)."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Iquique residuals")
        return

    import pandas as pd

    _IQUIQUE_RES_STATIONS = [
        s for s in IQUIQUE_STATION_META if s[0] in ("32401", "32402", "32412", "32413")
    ]

    fig, axes = plt.subplots(len(_IQUIQUE_RES_STATIONS), 1,
                             figsize=(10, 2.0 * len(_IQUIQUE_RES_STATIONS)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, _IQUIQUE_RES_STATIONS):
        cal_path = IQUIQUE_DATA_DIR / f"dart_{sid}_iquique_2014_calibration.csv"
        if not cal_path.exists():
            continue

        cal_times: list[float] = []
        cal_values: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                cal_times.append(float(row["seconds_from_origin"]) / 3600.0)
                cal_values.append(h)
        if len(cal_times) < 100:
            continue
        cal_t = np.array(cal_times)
        cal_h = np.array(cal_values)

        result = _load_iquique_event_csv(sid)
        if result is None:
            continue
        mins, ev_h = result
        ev_t = mins / 60.0

        # Despike: remove single-sample telemetry glitches
        if len(ev_h) > 2:
            left = np.concatenate([[ev_h[0]], ev_h[:-1]])
            right = np.concatenate([ev_h[1:], [ev_h[-1]]])
            neighbor_med = np.median(
                np.column_stack([left, ev_h, right]), axis=1)
            spike = np.abs(ev_h - neighbor_med) > 1.0
            if np.any(spike):
                logger.info("Despiked %d glitch(es) from %s event data",
                            int(np.sum(spike)), sid)
                ev_h = np.where(spike, neighbor_med, ev_h)

        harmonics = fit_tidal_harmonics(cal_t, cal_h, clean_input=True)
        predicted = predict_tide(ev_t, harmonics)
        ev_residual = ev_h - predicted

        # Pre-event linear detrend + centered rolling-median baseline.
        # Subtracting a 30-min centered median is a high-pass with a
        # ~30-min corner, not a drift-only correction.  Pushing a sinusoid
        # through this exact step gives a gain of 1.0 or more at periods of
        # 30 min and below, ~0.69 at 40 min and ~0.33 at 60 min, so it
        # removes real tsunami energy from the longer-period arrivals.
        # Re-running this pipeline on the archived records cuts the
        # post-origin peak by about half at Illapel 32401 and Iquique 32401
        # and to about a fifth at Samoa 51425.  This step does NOT preserve
        # tsunami peak amplitude.  The published captions do disclose the
        # median step, so the figures are not misdescribed, but do not read
        # these traces as calibrated amplitudes.
        pre_ev_mask = mins < 0
        if np.sum(pre_ev_mask) >= 2:
            bp = np.polyfit(mins[pre_ev_mask], ev_residual[pre_ev_mask], deg=1)
            ev_residual -= np.polyval(bp, mins)
        dt_min = float(np.median(np.diff(mins)))
        win = max(3, int(30.0 / max(dt_min, 0.1)))
        win = min(win, len(ev_residual) // 2)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(ev_residual, pad, mode='reflect')
        baseline = pd.Series(padded).rolling(
            window=win, min_periods=1, center=True,
        ).median().values[pad:-pad]
        ev_residual -= baseline

        hw = min(10, max(3, len(ev_residual) // 6))
        residual, _ = hampel_filter(ev_residual, half_window=hw, threshold=3.0)

        _plot_with_gaps(ax, mins, residual)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Iquique 2014 (Mw 8.2): Detided Residuals")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_iquique_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_iquique_detection_timeline(data: dict) -> None:
    """Iquique 2014 score overlay for event-mode stations."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            apply_bandpass_filter,
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Iquique timeline")
        return

    sliding = data.get("sliding_window", {})
    if not sliding:
        logger.warning("No sliding window data for Iquique timeline")
        return

    event_stations = [sid for sid in ["32413", "32412", "32402", "32401"]
                      if sid in sliding]
    if not event_stations:
        event_stations = list(sliding.keys())[:4]

    n_rows = 3
    n_stations = len(event_stations)
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )

    def _hampel_clean(residual):
        hw = min(10, max(3, len(residual) // 6))
        cleaned, _ = hampel_filter(residual, half_window=hw, threshold=3.0)
        return cleaned

    zoom_infos = []
    for i, sid in enumerate(event_stations):
        dist_km = next((d for s, d, _, _ in IQUIQUE_STATION_META if s == sid), 0)
        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        ax_zoom.set_visible(False)

        cal_path = IQUIQUE_DATA_DIR / f"dart_{sid}_iquique_2014_calibration.csv"
        event_path = IQUIQUE_DATA_DIR / f"dart_{sid}_iquique_2014_event.csv"

        info = _plot_score_overlay_station(
            (ax_filt, ax_score), sid, dist_km, sliding,
            cal_path, event_path,
            apply_bandpass_filter, fit_tidal_harmonics, predict_tide,
            clean_fn=_hampel_clean,
        )
        ax_filt.set_xlim(right=300)
        zoom_infos.append((i, info))

    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title(
        "Iquique 2014 (Mw 8.2): Bandpass-Filtered Residual and Detection Scores")
    fig.tight_layout()

    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_iquique_detection_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_iquique_station_map() -> None:
    """Iquique 2014 station location map."""
    _station_map_plain(
        station_meta=IQUIQUE_STATION_META,
        epicenter_lat=IQUIQUE_LAT,
        epicenter_lon=IQUIQUE_LON,
        title="DART Station Locations: Iquique 2014 Evaluation",
        outfile="fig_iquique_station_map.pdf",
        label_offsets={
            "32401": (-4.0, 5.5, "right"),   # left of dot, well above 32412
            "32402": (3.0, -3.0, "left"),
            "32412": (3.0, -3.0, "left"),   # right of dot, below 32401
            "32413": (3.0, -2.0, "left"),
            "51426": (3.0, 3.0, "left"),    # above dot, clear of legend
            "46403": (-5.0, 2.0, "right"),
        },
    )


# -- Samoa 2009 figures ------------------------------------------------

SAMOA_DATA_DIR = Path("data/samoa")
SAMOA_LAT = -15.489
SAMOA_LON = -172.095

SAMOA_STATION_META: list[tuple[str, int, float, float]] = [
    ("51425",   804, -9.511, -176.258),
    ("51426",   935, -23.110, -168.385),
    ("54401",  1960, -33.109, -173.155),
    ("51407",  4250, 19.530, -156.601),
    ("46411",  7680, 39.337, -127.040),
    ("46403",  7720, 52.647, -156.940),
]


def _load_samoa_event_csv(station_id: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = SAMOA_DATA_DIR / f"dart_{station_id}_samoa_2009_event.csv"
    if not path.exists():
        return None
    minutes: list[float] = []
    heights: list[float] = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            h = float(row["height_m"])
            if h >= 9999.0:
                continue
            minutes.append(float(row["seconds_from_origin"]) / 60.0)
            heights.append(h)
    if not minutes:
        return None
    return np.array(minutes), np.array(heights)


def fig_samoa_waveforms() -> None:
    """Samoa 2009 raw BPR time series."""
    event_stations = [(sid, d) for sid, d, _, _ in SAMOA_STATION_META]
    fig, axes = plt.subplots(len(event_stations), 1,
                             figsize=(10, 2.0 * len(event_stations)),
                             sharex=True)
    plotted = 0
    for ax, (sid, dist_km) in zip(axes, event_stations):
        result = _load_samoa_event_csv(sid)
        if result is None:
            ax.text(0.5, 0.5, f"{sid}: no data", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9)
            continue
        mins, heights = result
        heights_dm = heights - np.mean(heights)
        med = np.median(heights_dm)
        glitch_mask = np.abs(heights_dm - med) > 5.0
        if np.any(glitch_mask):
            heights_dm = heights_dm.copy()
            heights_dm[glitch_mask] = np.nan
        _plot_with_gaps(ax, mins, heights_dm)
        cal_path = SAMOA_DATA_DIR / f"dart_{sid}_samoa_2009_calibration.csv"
        ev_path = SAMOA_DATA_DIR / f"dart_{sid}_samoa_2009_event.csv"
        tide_data = _load_tidal_prediction(cal_path, ev_path)
        if tide_data is not None:
            _ev_t, _ev_h, predicted = tide_data
            pred_dm = predicted - np.mean(heights)
            label = "Predicted tide" if plotted == 0 else None
            ax.plot(_ev_t * 60.0, pred_dm, "r--", linewidth=0.6, alpha=0.7,
                    label=label)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1
    if plotted == 0:
        plt.close(fig)
        return
    axes[0].legend(fontsize=7, loc="upper right")
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Samoa 2009 (Mw 8.1): Raw Bottom Pressure Records (De-meaned)")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_samoa_waveforms.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_samoa_residuals() -> None:
    """Samoa 2009 detided residual time series (event-mode stations only)."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Samoa residuals")
        return

    import pandas as pd

    _SAMOA_RES_STATIONS = [
        s for s in SAMOA_STATION_META if s[0] in ("51425", "51426", "54401")
    ]

    fig, axes = plt.subplots(len(_SAMOA_RES_STATIONS), 1,
                             figsize=(10, 2.0 * len(_SAMOA_RES_STATIONS)),
                             sharex=True)

    plotted = 0
    for ax, (sid, dist_km, _lat, _lon) in zip(axes, _SAMOA_RES_STATIONS):
        cal_path = SAMOA_DATA_DIR / f"dart_{sid}_samoa_2009_calibration.csv"
        if not cal_path.exists():
            continue

        cal_times: list[float] = []
        cal_values: list[float] = []
        with open(cal_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                h = float(row["height_m"])
                if h >= 9999.0:
                    continue
                cal_times.append(float(row["seconds_from_origin"]) / 3600.0)
                cal_values.append(h)
        if len(cal_times) < 100:
            continue
        cal_t = np.array(cal_times)
        cal_h = np.array(cal_values)

        result = _load_samoa_event_csv(sid)
        if result is None:
            continue
        mins, ev_h = result
        ev_t = mins / 60.0

        # Despike: remove single-sample telemetry glitches
        if len(ev_h) > 2:
            left = np.concatenate([[ev_h[0]], ev_h[:-1]])
            right = np.concatenate([ev_h[1:], [ev_h[-1]]])
            neighbor_med = np.median(
                np.column_stack([left, ev_h, right]), axis=1)
            spike = np.abs(ev_h - neighbor_med) > 1.0
            if np.any(spike):
                logger.info("Despiked %d glitch(es) from %s event data",
                            int(np.sum(spike)), sid)
                ev_h = np.where(spike, neighbor_med, ev_h)

        harmonics = fit_tidal_harmonics(cal_t, cal_h, clean_input=True)
        predicted = predict_tide(ev_t, harmonics)
        ev_residual = ev_h - predicted

        # Pre-event linear detrend + centered rolling-median baseline.
        # Subtracting a 30-min centered median is a high-pass with a
        # ~30-min corner, not a drift-only correction.  Pushing a sinusoid
        # through this exact step gives a gain of 1.0 or more at periods of
        # 30 min and below, ~0.69 at 40 min and ~0.33 at 60 min, so it
        # removes real tsunami energy from the longer-period arrivals.
        # Re-running this pipeline on the archived records cuts the
        # post-origin peak by about half at Illapel 32401 and Iquique 32401
        # and to about a fifth at Samoa 51425.  This step does NOT preserve
        # tsunami peak amplitude.  The published captions do disclose the
        # median step, so the figures are not misdescribed, but do not read
        # these traces as calibrated amplitudes.
        # Tail check on this event: the residual at t+350 min is under
        # 0.008 m at all three stations.
        pre_ev_mask = mins < 0
        if np.sum(pre_ev_mask) >= 2:
            bp = np.polyfit(mins[pre_ev_mask], ev_residual[pre_ev_mask], deg=1)
            ev_residual -= np.polyval(bp, mins)
        dt_min = float(np.median(np.diff(mins)))
        win = max(3, int(30.0 / max(dt_min, 0.1)))
        win = min(win, len(ev_residual) // 2)
        if win % 2 == 0:
            win += 1
        pad = win // 2
        padded = np.pad(ev_residual, pad, mode='reflect')
        baseline = pd.Series(padded).rolling(
            window=win, min_periods=1, center=True,
        ).median().values[pad:-pad]
        ev_residual -= baseline

        hw = min(10, max(3, len(ev_residual) // 6))
        residual, _ = hampel_filter(ev_residual, half_window=hw, threshold=3.0)

        _plot_with_gaps(ax, mins, residual)
        ax.axvline(x=0, color="red", linewidth=1.2, linestyle="-", alpha=0.8)
        ax.set_ylabel("m", fontsize=8)
        ax.text(0.02, 0.85, f"{sid} ({dist_km:,} km)", transform=ax.transAxes,
                fontsize=8, va="top",
                bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "alpha": 0.8})
        ax.tick_params(labelsize=7)
        plotted += 1

    if plotted == 0:
        plt.close(fig)
        return
    axes[-1].set_xlabel("Minutes from earthquake origin")
    axes[0].set_title("Samoa 2009 (Mw 8.1): Detided Residuals")
    fig.tight_layout()
    out = FIGURES_DIR / "fig_samoa_residuals.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_samoa_detection_timeline(data: dict) -> None:
    """Samoa 2009 score overlay for event-mode stations."""
    try:
        from hazard_assessment.agents.anomaly_detection import (
            apply_bandpass_filter,
            fit_tidal_harmonics,
            hampel_filter,
            predict_tide,
        )
    except ImportError:
        logger.warning("Cannot import anomaly_detection - skipping Samoa timeline")
        return

    sliding = data.get("sliding_window", {})
    if not sliding:
        logger.warning("No sliding window data for Samoa timeline")
        return

    event_stations = [sid for sid in ["51426", "54401", "51425"]
                      if sid in sliding]
    if not event_stations:
        event_stations = list(sliding.keys())[:3]

    n_rows = 3
    n_stations = len(event_stations)
    fig, axes = plt.subplots(
        n_stations * n_rows, 1,
        figsize=(10, 5 * n_stations),
        gridspec_kw={"height_ratios": [3, 3, 1] * n_stations},
    )

    def _hampel_clean(residual):
        hw = min(10, max(3, len(residual) // 6))
        cleaned, _ = hampel_filter(residual, half_window=hw, threshold=3.0)
        return cleaned

    zoom_infos = []
    for i, sid in enumerate(event_stations):
        dist_km = next((d for s, d, _, _ in SAMOA_STATION_META if s == sid), 0)
        ax_filt = axes[i * n_rows]
        ax_score = axes[i * n_rows + 1]
        ax_zoom = axes[i * n_rows + 2]
        ax_score.sharex(ax_filt)
        ax_zoom.set_visible(False)

        cal_path = SAMOA_DATA_DIR / f"dart_{sid}_samoa_2009_calibration.csv"
        event_path = SAMOA_DATA_DIR / f"dart_{sid}_samoa_2009_event.csv"

        info = _plot_score_overlay_station(
            (ax_filt, ax_score), sid, dist_km, sliding,
            cal_path, event_path,
            apply_bandpass_filter, fit_tidal_harmonics, predict_tide,
            clean_fn=_hampel_clean,
        )
        ax_filt.set_xlim(right=300)
        zoom_infos.append((i, info))

    axes[(n_stations - 1) * n_rows + 1].set_xlabel(
        "Minutes from earthquake origin", fontsize=9)
    axes[0].set_title(
        "Samoa 2009 (Mw 8.1): Bandpass-Filtered Residual and Detection Scores")
    fig.tight_layout()

    for i, info in zoom_infos:
        ax_zoom = axes[i * n_rows + 2]
        ax_zoom.set_visible(True)
        _setup_onset_zoom(ax_zoom, axes[i * n_rows + 1], info)

    out = FIGURES_DIR / "fig_samoa_detection_timeline.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out)


def fig_samoa_station_map() -> None:
    """Samoa 2009 station location map."""
    _station_map_plain(
        station_meta=SAMOA_STATION_META,
        epicenter_lat=SAMOA_LAT,
        epicenter_lon=SAMOA_LON,
        title="DART Station Locations: Samoa 2009 Evaluation",
        outfile="fig_samoa_station_map.pdf",
        label_offsets={
            "51425": (3.0, 2.0, "left"),
            "51426": (-7.0, 0.0, "right"),   # far left of dot: clear of 54401
            "54401": (5.0, -3.0, "left"),    # right-below dot: clear of 51426
            "51407": (3.0, 2.0, "left"),
            "46411": (-5.0, -2.0, "right"),  # left of dot: avoid right-edge clip
            "46403": (-5.0, 2.0, "right"),
        },
        lat_margin_bottom=10,  # extra room for legend below 54401
    )


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # New figures: raw waveforms, residuals, station map
    fig_tohoku_waveforms()
    fig_tohoku_residuals()
    fig_station_map()

    # Fig 5: Tohoku detection
    tohoku = load_json("tohoku_detection.json")
    if tohoku and isinstance(tohoku, dict):
        fig5_tohoku_detection(tohoku)
        fig7_score_decomposition(tohoku)
        fig_score_waveform_overlay(tohoku)
    else:
        logger.warning("Skipping Figs 5/7/overlay - no tohoku_detection.json")

    # Fig 6: Detiding quality
    detiding = load_json("detiding_validation.json")
    if detiding and isinstance(detiding, list):
        fig6_detiding_quality(detiding)
    else:
        logger.warning("Skipping Fig 6 - no detiding_validation.json")

    # Fig 8: Synthetic heatmap
    synthetic = load_json("synthetic_evaluation.json")
    if synthetic and isinstance(synthetic, dict):
        fig8_synthetic_heatmap(synthetic)
    else:
        logger.warning("Skipping Fig 8 - no synthetic_evaluation.json")

    # Physics simulation validation figures
    phys = load_json("physics_validation.json")
    if phys and isinstance(phys, dict):
        fig_physics_validation_scenario1(phys)
        fig_physics_validation_summary(phys)
    else:
        logger.warning("Skipping physics validation figures - no physics_validation.json")

    # Dashboard layout schematic
    fig_dashboard_layout()

    # Chile 2010 figures
    fig_chile_station_map()
    chile = load_json("chile_detection.json")
    if chile and isinstance(chile, dict):
        fig_chile_detection(chile)
        fig_chile_detection_timeline(chile)
        fig_chile_waveforms()
        fig_chile_residuals()
    else:
        logger.warning("Skipping Chile figures - no chile_detection.json")

    # Synthetic time-series figures
    synth = load_json("synthetic_timelines.json")
    if synth and isinstance(synth, dict):
        fig_simulation_station_map(synth)
        fig_simulation_score_overlay(synth)
        fig_synthetic_waveforms(synth)
        fig_synthetic_residuals(synth)
        fig_synthetic_score_timeline(synth)
        fig_synthetic_multistation(synth)
    else:
        logger.warning("Skipping synthetic figures - no synthetic_timelines.json")

    # Illapel 2015 figures
    fig_illapel_station_map()
    fig_illapel_waveforms()
    fig_illapel_residuals()
    illapel = load_json("illapel_detection.json")
    if illapel and isinstance(illapel, dict):
        fig_illapel_detection(illapel)
        fig_illapel_detection_timeline(illapel)
    else:
        logger.warning("Skipping Illapel detection figure - no illapel_detection.json")

    # Iquique 2014 figures
    fig_iquique_station_map()
    fig_iquique_waveforms()
    fig_iquique_residuals()
    iquique = load_json("iquique_detection.json")
    if iquique and isinstance(iquique, dict):
        fig_iquique_detection(iquique)
        fig_iquique_detection_timeline(iquique)
    else:
        logger.warning("Skipping Iquique detection figure - no iquique_detection.json")

    # Samoa 2009 figures
    fig_samoa_station_map()
    fig_samoa_waveforms()
    fig_samoa_residuals()
    samoa = load_json("samoa_detection.json")
    if samoa and isinstance(samoa, dict):
        fig_samoa_detection(samoa)
        fig_samoa_detection_timeline(samoa)
    else:
        logger.warning("Skipping Samoa detection figure - no samoa_detection.json")

    # CO-OPS water-level figures and multi-source network map (Appendix)
    fig_coops_tohoku()
    fig_coops_chile()
    fig_coops_illapel()
    fig_coops_iquique()
    fig_coops_samoa()
    fig_multi_source_network_map()

    logger.info("Figure generation complete")


if __name__ == "__main__":
    main()
