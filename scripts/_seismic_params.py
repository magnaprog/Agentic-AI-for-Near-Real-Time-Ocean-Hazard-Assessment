"""Load earthquake parameters from USGS seismic_event.json files.

Provides a single function that loads canonical USGS parameters for each
evaluation event, replacing hardcoded constants across validation scripts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class SeismicParams:
    """Earthquake parameters from USGS FDSN event web service."""

    origin_utc: datetime
    magnitude: float
    latitude: float
    longitude: float
    depth_km: float
    event_id: str
    title: str


def load_seismic_params(event_name: str) -> SeismicParams:
    """Load earthquake parameters from data/{event_name}/seismic_event.json.

    Falls back to hardcoded values if the JSON file does not exist,
    ensuring scripts work even without running download_seismic_data.py.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data" / event_name
    json_path = data_dir / "seismic_event.json"

    if json_path.exists():
        with open(json_path) as f:
            event = json.load(f)
        props = event["properties"]
        coords = event["geometry"]["coordinates"]
        origin = datetime.fromtimestamp(
            props["time"] / 1000, tz=UTC
        )
        return SeismicParams(
            origin_utc=origin,
            magnitude=props["mag"],
            latitude=coords[1],
            longitude=coords[0],
            depth_km=coords[2],
            event_id=props.get("code", props.get("ids", "unknown")),
            title=props.get("title", event_name),
        )

    # Fallback: hardcoded canonical values (USGS NEIC)
    _FALLBACK: dict[str, dict] = {
        "tohoku": dict(
            origin_utc=datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC),
            magnitude=9.1, latitude=38.297, longitude=142.373,
            depth_km=29.0, event_id="usp000hvnu",
            title="M 9.1 - 2011 Great Tohoku Earthquake, Japan",
        ),
        "chile": dict(
            origin_utc=datetime(2010, 2, 27, 6, 34, 11, tzinfo=UTC),
            magnitude=8.8, latitude=-36.122, longitude=-72.898,
            depth_km=22.9, event_id="usp000h7rf",
            title="M 8.8 - 2010 Maule, Chile Earthquake",
        ),
        "illapel": dict(
            origin_utc=datetime(2015, 9, 16, 22, 54, 32, tzinfo=UTC),
            magnitude=8.3, latitude=-31.573, longitude=-71.674,
            depth_km=22.4, event_id="us20003k7a",
            title="M 8.3 - 48 km W of Illapel, Chile",
        ),
        "iquique": dict(
            origin_utc=datetime(2014, 4, 1, 23, 46, 47, tzinfo=UTC),
            magnitude=8.2, latitude=-19.610, longitude=-70.769,
            depth_km=25.0, event_id="usc000nzvd",
            title="M 8.2 - 93 km NW of Iquique, Chile",
        ),
        "samoa": dict(
            origin_utc=datetime(2009, 9, 29, 17, 48, 10, tzinfo=UTC),
            magnitude=8.1, latitude=-15.489, longitude=-172.095,
            depth_km=18.0, event_id="usp000h1ys",
            title="M 8.1 - 2009 Samoa Earthquake",
        ),
    }

    if event_name not in _FALLBACK:
        raise ValueError(f"Unknown event: {event_name}")

    return SeismicParams(**_FALLBACK[event_name])


# Equal-timestamp record policies for archived replay data.
# "first" is the frozen default: the earliest row in native archive order
# wins, matching the live station buffer, which keeps the first accepted
# record for a timestamp (workers/station_buffer.py). The other policies
# exist for the duplicate sensitivity evaluation only
# (scripts/evaluate_duplicate_sensitivity.py); scientific results use "first".
DEDUP_POLICIES: tuple[str, ...] = ("first", "last", "min", "max", "mean")


@dataclass(frozen=True)
class DedupStats:
    """Duplicate-timestamp accounting for one deduplicated series."""

    n_rows_in: int
    n_rows_out: int
    n_duplicate_rows: int
    n_conflict_timestamps: int
    max_conflict_delta: float
    policy: str

    def as_dict(self) -> dict[str, object]:
        """JSON-friendly form for result artifacts."""
        return {
            "policy": self.policy,
            "n_rows_in": self.n_rows_in,
            "n_rows_out": self.n_rows_out,
            "n_duplicate_rows": self.n_duplicate_rows,
            "n_conflict_timestamps": self.n_conflict_timestamps,
            "max_conflict_delta": round(self.max_conflict_delta, 6),
        }


def deduplicate_timeseries(
    timestamps: list[float],
    values: list[float],
    policy: str = "first",
) -> tuple[list[float], list[float], DedupStats]:
    """Collapse duplicate timestamps under an explicit equal-time policy.

    DART event archives contain duplicate timestamps (5-25 per station in
    the current data), and most of those duplicate rows carry conflicting
    height values, so the policy choice selects data. The frozen default
    is "first": the earliest row in native archive order wins, which
    matches the live station buffer (workers/station_buffer.py), where the
    first accepted record for a timestamp is kept and later equal-time
    arrivals are dropped.

    Input timestamps must be non-decreasing. Refusing unsorted input keeps
    a re-sorted archive from silently changing which record wins, and the
    returned stats (plus a warning log) make value-selecting duplicates
    visible wherever this runs.

    Policies other than "first" exist for the duplicate sensitivity
    evaluation only.
    """
    if policy not in DEDUP_POLICIES:
        raise ValueError(
            f"Unknown dedup policy {policy!r}; expected one of {DEDUP_POLICIES}"
        )
    for i in range(1, len(timestamps)):
        if timestamps[i] < timestamps[i - 1]:
            raise ValueError(
                "deduplicate_timeseries requires non-decreasing timestamps: "
                f"row {i} ({timestamps[i]}) precedes row {i - 1} "
                f"({timestamps[i - 1]}). Archived replay inputs are "
                "time-sorted; refusing to guess which equal-time record "
                "should win on unsorted input."
            )

    ts_out: list[float] = []
    val_out: list[float] = []
    n_conflicts = 0
    max_delta = 0.0
    i = 0
    n = len(timestamps)
    while i < n:
        j = i
        while j + 1 < n and timestamps[j + 1] == timestamps[i]:
            j += 1
        group = values[i : j + 1]
        if j > i:
            spread = max(group) - min(group)
            if spread > 0.0:
                n_conflicts += 1
                max_delta = max(max_delta, spread)
        if policy == "first":
            chosen = group[0]
        elif policy == "last":
            chosen = group[-1]
        elif policy == "min":
            chosen = min(group)
        elif policy == "max":
            chosen = max(group)
        else:  # mean
            chosen = sum(group) / len(group)
        ts_out.append(timestamps[i])
        val_out.append(chosen)
        i = j + 1

    stats = DedupStats(
        n_rows_in=n,
        n_rows_out=len(ts_out),
        n_duplicate_rows=n - len(ts_out),
        n_conflict_timestamps=n_conflicts,
        max_conflict_delta=max_delta,
        policy=policy,
    )
    if stats.n_duplicate_rows > 0:
        import logging
        logging.getLogger(__name__).warning(
            "Collapsed %d duplicate timestamp row(s) (%d with conflicting "
            "values, max spread %.4g) from %d rows under policy %r",
            stats.n_duplicate_rows, n_conflicts, max_delta, n, policy,
        )
    return ts_out, val_out, stats
