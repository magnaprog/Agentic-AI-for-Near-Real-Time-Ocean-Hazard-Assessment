"""Shared replay helpers for the offline evaluation scripts.

Two unrelated things live here because every validation script needs both:

  * ``load_seismic_params`` loads canonical USGS parameters for each
    evaluation event from ``data/{event}/seismic_event.json``, replacing
    hardcoded constants across the validation scripts.
  * ``deduplicate_timeseries``, ``DedupStats`` and ``DEDUP_POLICIES``
    implement the frozen equal-timestamp record policy that archived DART
    replay data needs (IP 7.4).
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
    # Not read anywhere. Kept so the dataclass mirrors the USGS record.
    title: str


# The five evaluation events. Each has a git-tracked
# data/{event}/seismic_event.json, so the loader below has no fallback.
KNOWN_EVENTS: tuple[str, ...] = (
    "tohoku", "chile", "illapel", "iquique", "samoa",
)


def load_seismic_params(event_name: str) -> SeismicParams:
    """Load earthquake parameters from data/{event_name}/seismic_event.json.

    Raises rather than substituting values when the file is missing.  An
    earlier version of this module carried a hardcoded fallback table, which
    was unreachable (all five JSON files are tracked in git) and had drifted
    from the files it shadowed: different event_id scheme (``usp000hvnu``
    against the file's ``20110311054624120_30``), whole-second origins
    against the file's millisecond values, and coarser coordinates and depth
    for illapel and iquique.  Substituting those silently would have emitted
    results/*_detection.json that no longer match the published artifacts
    with nothing in the run to indicate it, so a missing data/ directory is
    now a hard error.
    """
    if event_name not in KNOWN_EVENTS:
        raise ValueError(
            f"Unknown event: {event_name}; expected one of {KNOWN_EVENTS}"
        )

    data_dir = Path(__file__).resolve().parent.parent / "data" / event_name
    json_path = data_dir / "seismic_event.json"

    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing seismic parameters for {event_name}: {json_path} does "
            "not exist. This file is tracked in git; restore it (or re-run "
            "scripts/download_seismic_data.py) rather than running with "
            "substituted values, which would silently change every artifact "
            "derived from this event."
        )

    with open(json_path) as f:
        event = json.load(f)
    props = event["properties"]
    coords = event["geometry"]["coordinates"]
    origin = datetime.fromtimestamp(props["time"] / 1000, tz=UTC)
    return SeismicParams(
        origin_utc=origin,
        magnitude=props["mag"],
        latitude=coords[1],
        longitude=coords[0],
        depth_km=coords[2],
        event_id=props.get("code", props.get("ids", "unknown")),
        title=props.get("title", event_name),
    )


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
