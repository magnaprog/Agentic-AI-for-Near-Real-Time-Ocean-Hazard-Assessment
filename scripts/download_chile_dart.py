#!/usr/bin/env python3
"""Download DART BPR data for the 2010 Chile event retrospective validation.

Downloads data from NDBC for DART stations that recorded the Mw 8.8 Chile
earthquake and tsunami (2010-02-27 06:34:11 UTC).  Produces two CSV files per
station:

  1. **Calibration window** (30 days pre-event, standard-mode 15-min data)
     for harmonic tidal fitting.
  2. **Event window** (1 h before to 6 h after earthquake, event-mode
     15-sec or 1-min data) for anomaly detection.

Key stations (distances from epicenter computed via haversine):
  32412 - ~2400 km NW, off Peru, first tsunami arrival
  32411 - ~4920 km NW, off Galapagos
  54401 - ~8750 km W, SW Pacific
  46412 - ~9080 km N, off San Diego CA
  46411 - ~10040 km NNE, off N California (also in Tohoku evaluation)
  51407 - ~10740 km WNW, Hawaii
  46402 - ~13110 km NNW, S of Dutch Harbor AK (also in Tohoku evaluation)
  21413 - ~15840 km W, off Japan (also in Tohoku evaluation)

Earthquake parameters (USGS NEIC solution):
  Origin:    2010-02-27 06:34:11 UTC
  Epicenter: 35.846 S, 72.719 W
  Magnitude: Mw 8.8
  Depth:     35 km

Data source: NDBC historical DART data (public, no auth required).
Stations confirmed available via NCEI Chile Feb 2010 DART Summary.

Usage:
    python scripts/download_chile_dart.py [--output-dir data/chile] [--calibration-days 30]

Output:
    Per station:
      dart_{station}_chile_2010_event.csv - event-window data
      dart_{station}_chile_2010_calibration.csv - 30-day calibration data
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

# Chile earthquake parameters (USGS NEIC solution)
CHILE_ORIGIN_UTC = datetime(2010, 2, 27, 6, 34, 11, tzinfo=UTC)

# DART stations with confirmed Chile 2010 records.
# Distances computed via haversine from epicenter (35.846 S, 72.719 W).
# Coordinates from NDBC station pages.
# Stations 46411, 46402, 21413 overlap with Tohoku 2011 evaluation.
# Format: (station_id, approx_distance_km, description)
CHILE_STATIONS = [
    ("32412", 2400, "NW, off Peru, first arrival"),
    ("32411", 4920, "NW, off Galapagos"),
    ("54401", 8750, "W, SW Pacific"),
    ("46412", 9080, "N, off San Diego CA"),
    ("46411", 10040, "NNE, off N California"),
    ("51407", 10740, "WNW, Hawaii"),
    ("46402", 13110, "NNW, S of Dutch Harbor AK"),
    ("21413", 15840, "W, off Japan"),
]

# NDBC historical DART data URL pattern
NDBC_DART_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}t2010.txt.gz&dir=data/historical/dart/"
)

# Event window: 1 hour before to 6 hours after the earthquake
WINDOW_BEFORE_SEC = 3600
WINDOW_AFTER_SEC = 6 * 3600

# Default calibration window: 30 days before the earthquake
DEFAULT_CALIBRATION_DAYS = 30


def _fetch_station_data(station_id: str, retry: int = 3) -> str | None:
    """Download raw NDBC DART text for a station.

    Returns the raw text content, or None on failure.
    """
    url = NDBC_DART_URL.format(station=station_id)

    if retry < 1:
        logger.error("retry must be >= 1, got %d", retry)
        return None

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
    return None  # unreachable, but satisfies type checker


def _parse_rows(
    raw: str,
    station_id: str,
    start_utc: datetime,
    end_utc: datetime,
    measurement_types: set[str] | None = None,
) -> list[dict[str, str]]:
    """Parse NDBC DART text and filter rows to a time window.

    Args:
        raw: Raw NDBC text.
        station_id: Station identifier.
        start_utc: Start of time window (inclusive).
        end_utc: End of time window (inclusive).
        measurement_types: If set, only include rows where column 6
            (measurement type) is in this set.  NDBC DART types:
            "1" = 15-min standard, "2" = 1-min, "3" = 15-sec event.

    Returns list of row dicts with keys:
      station_id, timestamp_utc, seconds_from_origin, height_m
    """
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

            delta = (ts - CHILE_ORIGIN_UTC).total_seconds()
            rows.append({
                "station_id": station_id,
                "timestamp_utc": ts.isoformat(),
                "seconds_from_origin": f"{delta:.0f}",
                "height_m": parts[7],
            })
    return rows


def download_station(
    station_id: str,
    output_dir: Path,
    calibration_days: int = DEFAULT_CALIBRATION_DAYS,
    retry: int = 3,
) -> tuple[Path | None, Path | None]:
    """Download DART event + calibration data for a single station.

    Returns (event_path, calibration_path), either may be None on failure.
    """
    raw = _fetch_station_data(station_id, retry=retry)
    if raw is None:
        return None, None

    # Event window
    event_start = CHILE_ORIGIN_UTC - timedelta(seconds=WINDOW_BEFORE_SEC)
    event_end = CHILE_ORIGIN_UTC + timedelta(seconds=WINDOW_AFTER_SEC)
    event_rows = _parse_rows(raw, station_id, event_start, event_end)

    event_path: Path | None = None
    if event_rows:
        event_path = output_dir / f"dart_{station_id}_chile_2010_event.csv"
        _write_csv(event_rows, event_path)
        logger.info(
            "Saved %d event rows for station %s -> %s",
            len(event_rows), station_id, event_path,
        )
    else:
        logger.warning("No event-window data found for station %s", station_id)

    # Calibration window (30 days before earthquake, ends at event start)
    cal_path: Path | None = None
    if calibration_days > 0:
        cal_start = CHILE_ORIGIN_UTC - timedelta(days=calibration_days)
        cal_end = event_start  # ends right before event window
        cal_rows = _parse_rows(
            raw, station_id, cal_start, cal_end,
            measurement_types={"1"},  # standard mode only (15-min)
        )

        if cal_rows:
            cal_path = output_dir / f"dart_{station_id}_chile_2010_calibration.csv"
            _write_csv(cal_rows, cal_path)
            logger.info(
                "Saved %d calibration rows for station %s -> %s",
                len(cal_rows), station_id, cal_path,
            )
        else:
            logger.warning("No calibration data found for station %s", station_id)

    return event_path, cal_path


def _write_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write rows to a CSV file."""
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["station_id", "timestamp_utc", "seconds_from_origin", "height_m"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Chile 2010 DART data")
    parser.add_argument(
        "--output-dir", type=str, default="data/chile",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS,
        help="Number of days before event for calibration data (0 to skip)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Chile earthquake: %s", CHILE_ORIGIN_UTC.isoformat())
    logger.info("Downloading DART data for %d stations...", len(CHILE_STATIONS))
    logger.info("Calibration window: %d days pre-event", args.calibration_days)

    event_results: list[bool] = []
    cal_results: list[bool] = []
    for station_id, dist_km, desc in CHILE_STATIONS:
        event_path, cal_path = download_station(
            station_id, output_dir,
            calibration_days=args.calibration_days,
        )
        event_results.append(event_path is not None)
        cal_results.append(cal_path is not None)
        time.sleep(1)  # Be polite to NDBC servers

    print(
        f"\n{'Station':<10} {'Dist (km)':<12} {'Description':<35} "
        f"{'Event':>6} {'Calib':>6}"
    )
    print("-" * 80)
    for (station_id, dist_km, desc), ev_ok, cal_ok in zip(
        CHILE_STATIONS, event_results, cal_results,
    ):
        ev_str = "YES" if ev_ok else "NO"
        cal_str = "YES" if cal_ok else "NO"
        print(f"{station_id:<10} {dist_km:<12} {desc:<35} {ev_str:>6} {cal_str:>6}")


if __name__ == "__main__":
    main()
