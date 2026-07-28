#!/usr/bin/env python3
"""Download USGS seismic event records for all five evaluation events.

Uses the USGS FDSN event web service to fetch authoritative earthquake
parameters for each event used in the evaluation.

Usage::

    python scripts/download_seismic_data.py
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx

USGS_EVENT_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"

EVENTS = {
    "tohoku": {
        "starttime": "2011-03-11T05:46:00",
        "endtime": "2011-03-11T05:47:00",
        "minmagnitude": "9.0",
        "maxmagnitude": "9.5",
        "minlatitude": "35",
        "maxlatitude": "42",
        "minlongitude": "139",
        "maxlongitude": "146",
    },
    "chile": {
        "starttime": "2010-02-27T06:34:00",
        "endtime": "2010-02-27T06:35:00",
        "minmagnitude": "8.5",
        "maxmagnitude": "9.0",
        "minlatitude": "-38",
        "maxlatitude": "-33",
        "minlongitude": "-76",
        "maxlongitude": "-70",
    },
    "illapel": {
        "starttime": "2015-09-16T22:54:00",
        "endtime": "2015-09-16T22:55:00",
        "minmagnitude": "8.0",
        "maxmagnitude": "8.5",
        "minlatitude": "-33",
        "maxlatitude": "-29",
        "minlongitude": "-74",
        "maxlongitude": "-69",
    },
    "iquique": {
        "starttime": "2014-04-01T23:46:00",
        "endtime": "2014-04-01T23:47:00",
        "minmagnitude": "8.0",
        "maxmagnitude": "8.5",
        "minlatitude": "-21",
        "maxlatitude": "-18",
        "minlongitude": "-72",
        "maxlongitude": "-69",
    },
    "samoa": {
        "starttime": "2009-09-29T17:48:00",
        "endtime": "2009-09-29T17:49:00",
        "minmagnitude": "8.0",
        "maxmagnitude": "8.5",
        "minlatitude": "-17",
        "maxlatitude": "-14",
        "minlongitude": "-174",
        "maxlongitude": "-170",
    },
}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def fetch_seismic_event(params: dict[str, str]) -> dict:
    """Fetch earthquake event from USGS FDSN web service."""
    query_params = {
        "format": "geojson",
        "limit": "1",
        "orderby": "magnitude",
        **params,
    }
    with httpx.Client(timeout=30.0) as client:
        resp = client.get(USGS_EVENT_URL, params=query_params)
        resp.raise_for_status()
        if resp.status_code == 204 or not resp.content:
            return {"features": []}
        return resp.json()


def main() -> None:
    for event_name, event_params in EVENTS.items():
        print(f"\n=== {event_name.upper()} ===")
        event_dir = DATA_DIR / event_name
        event_dir.mkdir(parents=True, exist_ok=True)

        data = fetch_seismic_event(event_params)
        features = data.get("features", [])

        if not features:
            print(f"  No events found for {event_name}")
            continue

        event = features[0]
        props = event["properties"]
        coords = event["geometry"]["coordinates"]

        print(f"  Event: {props.get('title', 'Unknown')}")
        print(f"  Magnitude: {props.get('mag')}")
        print(f"  Location: {coords[1]:.3f}, {coords[0]:.3f}")
        print(f"  Depth: {coords[2]:.1f} km")
        print(f"  Time: {props.get('time')}")
        print(f"  Tsunami flag: {props.get('tsunami')}")

        filepath = event_dir / "seismic_event.json"
        with open(filepath, "w") as f:
            json.dump(event, f, indent=2)
        print(f"  -> Saved to {filepath.name}")

    print("\nDone.")


if __name__ == "__main__":
    main()
