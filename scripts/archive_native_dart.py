#!/usr/bin/env python3
"""Archive native NDBC DART rows for the retrospective validation events.

The transformed per-station CSVs under data/<event>/ keep only
(station_id, timestamp_utc, seconds_from_origin, height_m). The native NDBC
archive rows also carry the measurement-type column (1 = 15-min standard,
2 = 1-min, 3 = 15-sec) and the native row order, and include the 9999.000
missing-data sentinels that the transform drops. Neither can be
reconstructed from the transformed CSVs.

For every station that has a transformed event CSV, this script:

  1. fetches the station's native year file from NDBC (GET-only, public);
  2. stores the verbatim line subset spanning the calibration and event
     windows (all measurement types, sentinels included, original order)
     under data/<event>/native/;
  3. records the source URL, retrieval time, SHA-256 of the full fetched
     payload and of the stored subset in data/<event>/native/manifest.json;
     and
  4. re-derives the transformed event and calibration sequences from the
     native rows with the download scripts' windowing rules and reports
     whether they match the CSVs in use.

A retrieval today proves what NDBC serves today, not what it served at the
original download time; the match flags make any archive drift visible.

Usage:
    python3 scripts/archive_native_dart.py [--events tohoku chile ...]
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from _seismic_params import load_seismic_params

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

EVENTS = ["tohoku", "chile", "illapel", "iquique", "samoa"]

NDBC_DART_URL = (
    "https://www.ndbc.noaa.gov/view_text_file.php"
    "?filename={station}t{year}.txt.gz&dir=data/historical/dart/"
)

# Window rules shared by every download_*_dart.py script.
WINDOW_BEFORE_SEC = 3600
WINDOW_AFTER_SEC = 6 * 3600
CALIBRATION_DAYS = 30

# Subset span kept in the native archive: covers the calibration window
# (30 days pre-origin) and the event window (6 hours post-origin) with
# margin on both sides.
SPAN_BEFORE = timedelta(days=31)
SPAN_AFTER = timedelta(days=7)

NDBC_FORMAT_NOTE = (
    "Native NDBC DART column format: YY MM DD hh mm ss T HEIGHT, where T is "
    "the measurement type (1 = 15-min standard mode, 2 = 1-min, 3 = 15-sec) "
    "and HEIGHT is the water-column height in meters with 9999.000 as the "
    "missing-data sentinel. Lines are verbatim from the NDBC year file in "
    "native order; only lines outside the archived time span are omitted."
)


def fetch_native(station_id: str, year: str, retry: int = 3) -> bytes | None:
    """Download the raw NDBC DART year file. Returns raw bytes or None."""
    url = NDBC_DART_URL.format(station=station_id, year=year)
    for attempt in range(retry):
        try:
            logger.info(
                "Fetching %s (attempt %d/%d): %s",
                station_id, attempt + 1, retry, url,
            )
            with urlopen(url, timeout=120) as resp:  # noqa: S310
                return resp.read()
        except (URLError, OSError) as e:
            logger.warning("Attempt %d failed for %s: %s", attempt + 1, station_id, e)
            if attempt < retry - 1:
                time.sleep(2 ** attempt)
    logger.error("Failed to fetch %s after %d attempts", station_id, retry)
    return None


def parse_line(line: str) -> tuple[datetime, str, str] | None:
    """Parse one native data line into (timestamp, type, height string)."""
    parts = line.split()
    if len(parts) < 8:
        return None
    try:
        yr, mo, da, hr, mn, sc = (int(parts[i]) for i in range(6))
        ts = datetime(yr, mo, da, hr, mn, sc, tzinfo=UTC)
    except (ValueError, IndexError):
        return None
    return ts, parts[6], parts[7]


def extract_subset(
    text: str, span_start: datetime, span_end: datetime
) -> tuple[list[str], dict[str, int], int, int]:
    """Keep header lines and data lines inside the span, in native order.

    Returns (kept_lines, measurement_type_counts, data_row_count,
    sentinel_count).
    """
    kept: list[str] = []
    type_counts: dict[str, int] = {}
    n_rows = 0
    n_sentinel = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            kept.append(line)
            continue
        parsed = parse_line(stripped)
        if parsed is None:
            continue
        ts, mtype, height = parsed
        if span_start <= ts <= span_end:
            kept.append(line)
            n_rows += 1
            type_counts[mtype] = type_counts.get(mtype, 0) + 1
            try:
                if float(height) >= 9999.0:
                    n_sentinel += 1
            except ValueError:
                pass
    return kept, type_counts, n_rows, n_sentinel


def derive_transformed(
    text: str,
    start_utc: datetime,
    end_utc: datetime,
    measurement_types: set[str] | None,
) -> list[tuple[str, float]]:
    """Replicate the download scripts' transform on the native text.

    Returns the in-window (timestamp isoformat, height) sequence with
    measurement-type filtering and 9999 sentinels dropped.
    """
    rows: list[tuple[str, float]] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parsed = parse_line(stripped)
        if parsed is None:
            continue
        ts, mtype, height = parsed
        if measurement_types is not None and mtype not in measurement_types:
            continue
        if not (start_utc <= ts <= end_utc):
            continue
        try:
            height_val = float(height)
        except ValueError:
            continue
        if height_val >= 9999.0:
            continue
        rows.append((ts.isoformat(), height_val))
    return rows


def load_csv_rows(path: Path) -> list[tuple[str, float]]:
    """Load (timestamp_utc, height_m) pairs from a transformed CSV."""
    rows: list[tuple[str, float]] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append((row["timestamp_utc"], float(row["height_m"])))
    return rows


def archive_event(event_name: str, repo_root: Path) -> dict[str, object]:
    """Archive native rows for every station with a transformed event CSV."""
    data_dir = repo_root / "data" / event_name
    native_dir = data_dir / "native"
    native_dir.mkdir(parents=True, exist_ok=True)

    params = load_seismic_params(event_name)
    # The download scripts hardcode the origin at whole-second precision
    # (the USGS millisecond time floored); mirror that for the windows.
    origin = params.origin_utc.replace(microsecond=0)

    event_start = origin - timedelta(seconds=WINDOW_BEFORE_SEC)
    event_end = origin + timedelta(seconds=WINDOW_AFTER_SEC)
    cal_start = origin - timedelta(days=CALIBRATION_DAYS)
    cal_end = event_start
    span_start = origin - SPAN_BEFORE
    span_end = origin + SPAN_AFTER

    stations_out: dict[str, object] = {}
    event_files = sorted(data_dir.glob(f"dart_*_{event_name}_*_event.csv"))

    for event_path in event_files:
        tokens = event_path.stem.split("_")
        station_id, year = tokens[1], tokens[3]

        raw = fetch_native(station_id, year)
        retrieved_at = datetime.now(UTC).isoformat()
        if raw is None:
            stations_out[station_id] = {"error": "fetch failed"}
            continue

        full_sha = hashlib.sha256(raw).hexdigest()
        text = raw.decode("utf-8", errors="replace")

        kept, type_counts, n_rows, n_sentinel = extract_subset(
            text, span_start, span_end
        )
        subset_name = f"dart_{station_id}_{year}_native.txt"
        subset_path = native_dir / subset_name
        subset_bytes = ("\n".join(kept) + "\n").encode("utf-8")
        subset_path.write_bytes(subset_bytes)

        # Verify the transformed CSVs re-derive from today's native rows.
        derived_event = derive_transformed(text, event_start, event_end, None)
        csv_event = load_csv_rows(event_path)
        event_match = derived_event == csv_event

        cal_path = Path(str(event_path).replace("_event.csv", "_calibration.csv"))
        cal_match: bool | None = None
        cal_counts: dict[str, int] = {}
        if cal_path.exists():
            derived_cal = derive_transformed(text, cal_start, cal_end, {"1"})
            csv_cal = load_csv_rows(cal_path)
            cal_match = derived_cal == csv_cal
            cal_counts = {
                "native_rows": len(derived_cal), "csv_rows": len(csv_cal),
            }

        stations_out[station_id] = {
            "source_url": NDBC_DART_URL.format(station=station_id, year=year),
            "retrieved_at_utc": retrieved_at,
            "full_file_sha256": full_sha,
            "full_file_bytes": len(raw),
            "native_subset_file": f"native/{subset_name}",
            "subset_sha256": hashlib.sha256(subset_bytes).hexdigest(),
            "subset_span_utc": [span_start.isoformat(), span_end.isoformat()],
            "subset_data_rows": n_rows,
            "measurement_type_counts": type_counts,
            "sentinel_9999_rows": n_sentinel,
            "event_window_match": {
                "matches_transformed_csv": event_match,
                "native_rows": len(derived_event),
                "csv_rows": len(csv_event),
            },
            "calibration_window_match": (
                {"matches_transformed_csv": cal_match, **cal_counts}
                if cal_match is not None else None
            ),
        }
        logger.info(
            "%s/%s: %d native rows, types=%s, event match=%s, cal match=%s",
            event_name, station_id, n_rows, type_counts, event_match, cal_match,
        )
        time.sleep(1)  # Be polite to NDBC servers

    manifest = {
        "event": event_name,
        "origin_utc": origin.isoformat(),
        "format_note": NDBC_FORMAT_NOTE,
        "window_rules": {
            "event_window_utc": [event_start.isoformat(), event_end.isoformat()],
            "calibration_window_utc": [cal_start.isoformat(), cal_end.isoformat()],
            "calibration_measurement_types": ["1"],
            "event_measurement_types": "all",
        },
        "stations": stations_out,
    }
    manifest_path = native_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Wrote %s", manifest_path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Archive native NDBC DART rows with hashes and provenance",
    )
    parser.add_argument(
        "--events", nargs="+", default=EVENTS, choices=EVENTS,
        help="Events to archive (default: all)",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    all_match = True
    for event_name in args.events:
        manifest = archive_event(event_name, repo_root)
        for station_id, entry in manifest["stations"].items():  # type: ignore[union-attr]
            if not isinstance(entry, dict) or "error" in entry:
                all_match = False
                continue
            ev = entry["event_window_match"]["matches_transformed_csv"]
            cal_entry = entry["calibration_window_match"]
            cal = cal_entry["matches_transformed_csv"] if cal_entry else True
            if not (ev and cal):
                all_match = False
                logger.warning(
                    "MISMATCH %s/%s: event=%s calibration=%s",
                    event_name, station_id, ev, cal,
                )
    print(
        "\nAll transformed CSVs re-derive from today's native rows."
        if all_match else
        "\nWARNING: at least one station failed or mismatched; see manifests."
    )


if __name__ == "__main__":
    main()
