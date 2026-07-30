#!/usr/bin/env python3
"""Download historical CO-OPS water-level data for all five evaluation events.

Uses the NOAA CO-OPS Tides & Currents API (datagetter endpoint) to fetch
verified 6-minute water-level observations from Pacific tide gauge stations
during each of the five events.

Usage::

    python scripts/download_coops_data.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import httpx

COOPS_API_URL = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"

# Pacific tide gauge stations (development set).
STATIONS = {
    "1612340": "Honolulu, HI",
    "1617760": "Hilo, HI",
    "1619910": "Sand Island, Midway",
    "1890000": "Wake Island",
    "1770000": "Pago Pago, AS",       # Near-field for Samoa 2009
    "9419750": "Crescent City, CA",    # Known tsunami amplification
}

# Station-event pairs to download (not all stations for all events)
STATION_EVENT_OVERRIDE: dict[str, list[str]] = {
    "1770000": ["samoa"],                   # Pago Pago: Samoa only
    "9419750": ["tohoku", "chile"],         # Crescent City: Tohoku + Chile
}

# Event windows: ~2-day periods spanning each event (CO-OPS API includes full end date).
EVENTS = {
    "tohoku": {
        "begin": "20110311",
        "end": "20110312",
    },
    "chile": {
        "begin": "20100227",
        "end": "20100228",
    },
    "illapel": {
        "begin": "20150916",
        "end": "20150918",
    },
    "iquique": {
        "begin": "20140401",
        "end": "20140403",
    },
    "samoa": {
        "begin": "20090929",
        "end": "20091001",
    },
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_coops_water_level(
    station_id: str,
    begin_date: str,
    end_date: str,
) -> list[dict[str, str]]:
    """Fetch 6-minute verified water-level data from CO-OPS API."""
    params = {
        "begin_date": begin_date,
        "end_date": end_date,
        "station": station_id,
        "product": "water_level",
        "datum": "STND",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "ocean_hazard_assessment",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(COOPS_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        print(f"  API error for {station_id}: {data['error'].get('message', data['error'])}")
        return []

    return data.get("data", [])


def save_csv(records: list[dict[str, str]], filepath: Path) -> int:
    """Write CO-OPS records to CSV. Returns count written."""
    if not records:
        return 0
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["t", "v", "s", "f", "q"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return len(records)


def fetch_coops_predictions(
    station_id: str,
    begin_date: str,
    end_date: str,
) -> list[dict[str, str]]:
    """Fetch 6-minute tidal predictions from CO-OPS API."""
    params = {
        "begin_date": begin_date,
        "end_date": end_date,
        "station": station_id,
        "product": "predictions",
        "datum": "STND",
        "units": "metric",
        "time_zone": "gmt",
        "format": "json",
        "application": "ocean_hazard_assessment",
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(COOPS_API_URL, params=params)
        resp.raise_for_status()
        data = resp.json()

    if "error" in data:
        message = data["error"].get("message", data["error"])
        print(f"  API error (predictions) for {station_id}: {message}")
        return []

    return data.get("predictions", [])


def save_predictions_csv(records: list[dict[str, str]], filepath: Path) -> int:
    """Write CO-OPS prediction records to CSV. Returns count written."""
    if not records:
        return 0
    filepath.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["t", "v"]
    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)
    return len(records)


def main() -> None:
    for event_name, event_cfg in EVENTS.items():
        print(f"\n=== {event_name.upper()} ===")
        event_dir = DATA_DIR / event_name
        for station_id, station_name in STATIONS.items():
            # Skip station-event pairs not in override list
            if station_id in STATION_EVENT_OVERRIDE:
                if event_name not in STATION_EVENT_OVERRIDE[station_id]:
                    continue
            # Fetch observations
            print(f"  Fetching {station_name} ({station_id}) observations...")
            records = fetch_coops_water_level(
                station_id,
                event_cfg["begin"],
                event_cfg["end"],
            )
            if records:
                filepath = event_dir / f"coops_{station_id}_{event_name}.csv"
                n = save_csv(records, filepath)
                print(f"    -> {n} records saved to {filepath.name}")
            else:
                # A gauge can simply have no record for the window. Hilo
                # (1617760) has none for Chile 2010 in either the 6-minute or
                # the 1-minute product.
                print("    -> No data returned")
            time.sleep(1.0)

            # Predictions come from harmonic constants, so the API returns them
            # for any window whether or not the gauge recorded anything. Every
            # consumer plots them against observations, so fetching them for a
            # station with no observations only produces an orphan file.
            if not records:
                continue

            print(f"  Fetching {station_name} ({station_id}) predictions...")
            preds = fetch_coops_predictions(
                station_id,
                event_cfg["begin"],
                event_cfg["end"],
            )
            if preds:
                pred_path = event_dir / f"coops_{station_id}_{event_name}_predictions.csv"
                n = save_predictions_csv(preds, pred_path)
                print(f"    -> {n} predictions saved to {pred_path.name}")
            else:
                print("    -> No predictions returned")
            time.sleep(1.0)

    print("\nDone.")


if __name__ == "__main__":
    main()
