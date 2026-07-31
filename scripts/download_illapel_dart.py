#!/usr/bin/env python3
"""Download DART BPR data for the 2015 Illapel event retrospective validation.

Downloads data from NDBC for DART stations that recorded the Mw 8.3 Illapel
earthquake and tsunami (2015-09-16 22:54:32 UTC).  Produces two CSV files per
station:

  1. **Calibration window** (30 days pre-event, standard-mode 15-min data)
     for harmonic tidal fitting.
  2. **Event window** (1 h before to 6 h after earthquake) for anomaly
     detection.  No measurement-type filter is applied here, so the window
     holds whatever NDBC recorded: event-mode 15-sec (T=3) and 1-min (T=2)
     rows plus the standard-mode 15-min (T=1) rows that keep reporting
     through the event.  That overlap is what produces the duplicate
     timestamps the equal-time dedup policy in scripts/_seismic_params.py
     resolves ("first" row in native archive order wins).

Key stations (distances from epicenter computed via haversine):
  32402 - ~580 km NW, off central Chile, first tsunami arrival (~30 min)
  32401 - ~1250 km NW, off northern Chile
  32412 - ~2110 km NW, off Peru
  51426 - ~9280 km W, near Tonga
  46403 - ~12440 km NNE, off Alaska
  46411 - ~9740 km N, off N California
  46407 - ~10110 km N, off Oregon

Earthquake parameters (USGS NEIC solution, event us20003k7a):
  Origin:    2015-09-16 22:54:32 UTC
  Epicenter: 31.573 S, 71.674 W
  Magnitude: Mw 8.3
  Depth:     22 km

Data source: NDBC historical DART data (public, no auth required).
Stations confirmed available via NCEI September 2015 Chile DART Summary.

Usage:
    python scripts/download_illapel_dart.py [--output-dir data/illapel] [--calibration-days 30]

Output:
    Per station:
      dart_{station}_illapel_2015_event.csv - event-window data
      dart_{station}_illapel_2015_calibration.csv - 30-day calibration data
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

# Illapel earthquake parameters (USGS NEIC, event us20003k7a)
ILLAPEL_ORIGIN_UTC = datetime(2015, 9, 16, 22, 54, 32, tzinfo=UTC)

# DART stations with confirmed Illapel 2015 records.
# Distances computed via haversine from epicenter (31.573 S, 71.674 W).
# Source: NCEI September 2015 Chile DART Summary.
# Format: (station_id, approx_distance_km, description)
ILLAPEL_STATIONS = [
    ("32402", 580, "NW, off central Chile - first arrival"),
    ("32401", 1250, "NW, off northern Chile"),
    ("32412", 2110, "NW, off Peru"),
    ("51426", 9280, "W, near Tonga"),
    ("46403", 12440, "NNE, off Alaska"),
    ("46411", 9740, "N, off N California"),
    ("46407", 10110, "N, off Oregon"),
]

# NDBC historical DART data URL pattern
NDBC_DART_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}t2015.txt.gz&dir=data/historical/dart/"
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

            delta = (ts - ILLAPEL_ORIGIN_UTC).total_seconds()
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
    event_start = ILLAPEL_ORIGIN_UTC - timedelta(seconds=WINDOW_BEFORE_SEC)
    event_end = ILLAPEL_ORIGIN_UTC + timedelta(seconds=WINDOW_AFTER_SEC)
    event_rows = _parse_rows(raw, station_id, event_start, event_end)

    event_path: Path | None = None
    if event_rows:
        event_path = output_dir / f"dart_{station_id}_illapel_2015_event.csv"
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
        cal_start = ILLAPEL_ORIGIN_UTC - timedelta(days=calibration_days)
        cal_end = event_start
        cal_rows = _parse_rows(
            raw, station_id, cal_start, cal_end,
            measurement_types={"1"},
        )

        if cal_rows:
            cal_path = output_dir / f"dart_{station_id}_illapel_2015_calibration.csv"
            _write_csv(cal_rows, cal_path)
            logger.info(
                "Saved %d calibration rows for station %s -> %s",
                len(cal_rows), station_id, cal_path,
            )
        else:
            logger.warning("No calibration data found for station %s", station_id)

    return event_path, cal_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Download Illapel 2015 DART data")
    parser.add_argument(
        "--output-dir", type=str, default="data/illapel",
        help="Output directory for CSV files",
    )
    parser.add_argument(
        "--calibration-days", type=int, default=DEFAULT_CALIBRATION_DAYS,
        help="Number of days before event for calibration data (0 to skip)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Illapel earthquake: %s", ILLAPEL_ORIGIN_UTC.isoformat())
    logger.info("Downloading DART data for %d stations...", len(ILLAPEL_STATIONS))
    logger.info("Calibration window: %d days pre-event", args.calibration_days)

    event_results: list[bool] = []
    cal_results: list[bool] = []
    for station_id, dist_km, desc in ILLAPEL_STATIONS:
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
        ILLAPEL_STATIONS, event_results, cal_results,
    ):
        ev_str = "YES" if ev_ok else "NO"
        cal_str = "YES" if cal_ok else "NO"
        print(f"{station_id:<10} {dist_km:<12} {desc:<40} {ev_str:>6} {cal_str:>6}")


if __name__ == "__main__":
    main()
