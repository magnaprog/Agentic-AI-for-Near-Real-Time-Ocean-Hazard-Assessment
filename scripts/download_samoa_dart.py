#!/usr/bin/env python3
"""Download DART BPR data for the 2009 Samoa event retrospective validation.

Downloads data from NDBC for DART stations that recorded the Mw 8.1 Samoa
earthquake and tsunami (2009-09-29 17:48:10 UTC).  Produces two CSV files per
station:

  1. **Calibration window** (30 days pre-event, standard-mode 15-min data)
     for harmonic tidal fitting.
  2. **Event window** (1 h before to 6 h after earthquake, event-mode
     15-sec or 1-min data) for anomaly detection.

Key stations (from NCEI 2009 Samoa DART Summary - 24 stations total):
  51425 - ~804 km NNE, near epicenter (nearest)
  51426 - ~932 km SSE, near Tonga
  54401 - ~1962 km S, south of Samoa
  51407 - ~4247 km NE, off Hawaii
  52402 - ~4829 km W, NE of Papua New Guinea
  46403 - ~7718 km NE, off Alaska
  46411 - ~7680 km NE, off N California

Earthquake parameters (USGS NEIC solution, event usp000h1ys):
  Origin:    2009-09-29 17:48:10 UTC
  Epicenter: 15.489 S, 172.095 W
  Magnitude: Mw 8.1
  Depth:     18 km

Data source: NDBC historical DART data (public, no auth required).
Stations confirmed via NCEI 2009 Samoa DART Summary.

Usage:
    python scripts/download_samoa_dart.py [--output-dir data/samoa] [--calibration-days 30]

Output:
    Per station:
      dart_{station}_samoa_2009_event.csv - event-window data
      dart_{station}_samoa_2009_calibration.csv - 30-day calibration data
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Samoa earthquake parameters (USGS NEIC, event usp000h1ys)
SAMOA_ORIGIN_UTC = datetime(2009, 9, 29, 17, 48, 10, tzinfo=UTC)

# DART stations with confirmed Samoa 2009 records.
# Source: NCEI 2009 Samoa DART Summary (ngdc.noaa.gov/hazard/dart/2009samoa_dart.html)
# Selected subset covering near-field to far-field range.
# Format: (station_id, approx_distance_km, description)
SAMOA_STATIONS = [
    ("51425", 804, "NNE, near epicenter - nearest"),
    ("51426", 932, "SSE, near Tonga"),
    ("54401", 1962, "S, south of Samoa"),
    ("51407", 4247, "NE, off Hawaii"),
    ("52402", 4829, "W, NE of Papua New Guinea"),
    ("46403", 7718, "NE, off Alaska"),
    ("46411", 7680, "NE, off N California"),
]

# NDBC historical DART data URL pattern
NDBC_DART_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}t2009.txt.gz&dir=data/historical/dart/"
)

# Event window: 1 hour before to 6 hours after the earthquake
WINDOW_BEFORE_SEC = 3600
WINDOW_AFTER_SEC = 6 * 3600

# Default calibration window: 30 days before the earthquake
DEFAULT_CALIBRATION_DAYS = 30


def _fetch_station_data(station_id: str, retry: int = 3) -> str | None:
    """Download raw NDBC DART text for a station."""
    url = NDBC_DART_URL.format(station=station_id)

    for attempt in range(retry):
        try:
            logger.info(
                "Downloading station %s (attempt %d/%d): %s",
                station_id, attempt + 1, retry, url,
            )
            with urlopen(url, timeout=60) as resp:  # noqa: S310
                return resp.read().decode("utf-8", errors="replace")
        except (URLError, OSError) as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, station_id, e)
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
            else:
                logger.error("Failed to download %s after %d attempts", station_id, retry)
                return None
    return None


def _parse_rows(
    raw: str,
    station_id: str,
    start_utc: datetime,
    end_utc: datetime,
    measurement_types: set[str] | None = None,
) -> list[dict[str, str]]:
    """Parse NDBC DART text and filter rows to a time window."""
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            yr, mo, da, hr, mn, sc = (int(parts[i]) for i in range(6))
            ts = datetime(yr, mo, da, hr, mn, sc, tzinfo=UTC)
        except (ValueError, IndexError):
            continue

        if measurement_types is not None and parts[6] not in measurement_types:
            continue

        if start_utc <= ts <= end_utc:
            try:
                height_val = float(parts[7])
            except ValueError:
                continue
            if height_val >= 9999.0:
                continue

            delta = (ts - SAMOA_ORIGIN_UTC).total_seconds()
            rows.append({
                "station_id": station_id,
                "timestamp_utc": ts.isoformat(),
                "seconds_from_origin": f"{delta:.0f}",
                "height_m": parts[7],
            })
    return rows


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["station_id", "timestamp_utc", "seconds_from_origin", "height_m"],
        )
        writer.writeheader()
        writer.writerows(rows)


def download_station(
    station_id: str,
    output_dir: Path,
    calibration_days: int = DEFAULT_CALIBRATION_DAYS,
    retry: int = 3,
) -> tuple[Path | None, Path | None]:
    """Download DART event + calibration data for a single station."""
    raw = _fetch_station_data(station_id, retry=retry)
    if raw is None:
        return None, None

    # Event window
    event_start = SAMOA_ORIGIN_UTC - timedelta(seconds=WINDOW_BEFORE_SEC)
    event_end = SAMOA_ORIGIN_UTC + timedelta(seconds=WINDOW_AFTER_SEC)
    event_rows = _parse_rows(raw, station_id, event_start, event_end)

    event_path: Path | None = None
    if event_rows:
        event_path = output_dir / f"dart_{station_id}_samoa_2009_event.csv"
        _write_csv(event_rows, event_path)
        logger.info(
            "Saved %d event rows for station %s -> %s",
            len(event_rows), station_id, event_path,
        )
    else:
        logger.warning("No event-window data found for station %s", station_id)

    # Calibration window (30 days before earthquake, standard mode only)
    cal_path: Path | None = None
    if calibration_days > 0:
        cal_start = SAMOA_ORIGIN_UTC - timedelta(days=calibration_days)
        cal_end = event_start
        cal_rows = _parse_rows(
            raw, station_id, cal_start, cal_end,
            measurement_types={"1"},
        )

        if cal_rows:
            cal_path = output_dir / f"dart_{station_id}_samoa_2009_calibration.csv"
            _write_csv(cal_rows, cal_path)
            logger.info(
                "Saved %d calibration rows for station %s -> %s",
                len(cal_rows), station_id, cal_path,
            )
        else:
            logger.warning("No calibration data found for station %s", station_id)

    return event_path, cal_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Samoa 2009 DART data")
    parser.add_argument(
        "--output-dir", type=str, default="data/samoa",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS,
        help="Number of days before event for calibration data (0 to skip)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Samoa earthquake: %s", SAMOA_ORIGIN_UTC.isoformat())
    logger.info("Downloading DART data for %d stations...", len(SAMOA_STATIONS))
    logger.info("Calibration window: %d days pre-event", args.calibration_days)

    event_results: list[bool] = []
    cal_results: list[bool] = []
    for station_id, dist_km, desc in SAMOA_STATIONS:
        event_path, cal_path = download_station(
            station_id, output_dir,
            calibration_days=args.calibration_days,
        )
        event_results.append(event_path is not None)
        cal_results.append(cal_path is not None)
        time.sleep(1)

    print(
        f"\n{'Station':<10} {'Dist (km)':<12} {'Description':<40} "
        f"{'Event':>6} {'Calib':>6}"
    )
    print("-" * 85)
    for (station_id, dist_km, desc), ev_ok, cal_ok in zip(
        SAMOA_STATIONS, event_results, cal_results,
    ):
        ev_str = "YES" if ev_ok else "NO"
        cal_str = "YES" if cal_ok else "NO"
        print(f"{station_id:<10} {dist_km:<12} {desc:<40} {ev_str:>6} {cal_str:>6}")


if __name__ == "__main__":
    main()
