"""Every station distance printed in the paper must be the real haversine.

The per-event station tables print a distance column that no gate covered.
``results/*_detection.json`` does not carry ``distance_km`` at all, so the
printed values came from constants hand-maintained in the download and figure
scripts, and one of them had drifted: Illapel 32401 was printed as 1,230 km
against a true 1,250 km, a 20 km error that the paper, the figure labels and
three separate scripts all agreed on because they all copied the same
constant.

This recomputes each distance from the event's epicenter and the station
coordinates, so a stale constant fails here rather than being reprinted.

Chile is the one event whose distances are computed from an epicenter other
than the one in ``data/chile/seismic_event.json``. ``scripts/validate_chile.py``
deliberately keeps the early USGS determination because the distances quoted
throughout the evaluation were computed from it, and the paper discloses the
split in a footnote, so this test compares Chile against that same early
epicenter.
"""

from __future__ import annotations

import json
import pathlib
import re

from hazard_assessment.data.station_coordinates import station_coordinates
from hazard_assessment.geo import compute_initial_bearing_deg, haversine_km

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper" / "paper.tex"
DATA = REPO_ROOT / "data"

EVENTS = ("tohoku", "chile", "illapel", "iquique", "samoa")

#: The early USGS determination validate_chile.py pins, and the paper footnotes.
_CHILE_EARLY_EPICENTER = (-35.846, -72.719)

#: Some tables print to the nearest 10 km and some to the nearest km, so a
#: correct value can sit up to half a 10 km step away. One kilometre of slack
#: on top of that keeps the boundary case from failing on a rounding tie.
_TOLERANCE_KM = 6.0

_ROW = re.compile(r"^\s*(\d{5})\s*&\s*([0-9{},]+)\s*&\s*([^&]+?)\s*&")

#: 16-point compass, in bearing order.
_COMPASS = (
    "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW",
)

#: A Location cell either leads with a bearing from the epicenter ("NW, off
#: Oregon") or describes position relative to a place ("W of Iquique, Chile").
#: Only the first form is a claim about the epicenter.
_LEADING_BEARING = re.compile(r"^([NSEW]{1,3}),")


def _compass_point(bearing_deg: float) -> str:
    return _COMPASS[int((bearing_deg + 11.25) % 360 // 22.5)]


def _epicenter(event: str) -> tuple[float, float]:
    if event == "chile":
        return _CHILE_EARLY_EPICENTER
    geo = json.loads((DATA / event / "seismic_event.json").read_text())
    lon, lat, _depth = geo["geometry"]["coordinates"]
    return lat, lon


def _table(tex: str, label: str) -> str:
    start = tex.index(f"\\label{{tab:{label}}}")
    return tex[start : tex.index("\\end{tabular}", start)]


def _stations_table(tex: str, event: str) -> str:
    return _table(tex, f"{event}-stations")


def test_printed_station_distances_are_real_haversine_distances() -> None:
    tex = PAPER.read_text()
    mismatches: list[str] = []
    checked = 0
    skipped: list[str] = []

    for event in EVENTS:
        lat, lon = _epicenter(event)
        # Both tables print the distance. The stations table was once corrected
        # while the results table kept the old value, so the paper disagreed
        # with itself; check both and require them to agree.
        rows = [
            line
            for table in (f"{event}-stations", f"{event}-results")
            for line in _table(tex, table).splitlines()
        ]
        for line in rows:
            row = _ROW.match(line)
            if row is None:
                continue
            station_id = row.group(1)
            printed = float(row.group(2).replace("{,}", "").replace(",", ""))
            coords = station_coordinates(station_id)
            if coords is None:
                # Documented gap: some stations have no entry in
                # DART_STATION_COORDS. Recorded rather than silently passed.
                skipped.append(f"{event}/{station_id}")
                continue
            actual = haversine_km(lat, lon, coords[0], coords[1])
            checked += 1
            if abs(actual - printed) > _TOLERANCE_KM:
                mismatches.append(
                    f"{event} {station_id}: paper {printed:,.0f} km, "
                    f"haversine {actual:,.1f} km (delta {actual - printed:+.1f})"
                )

    assert not mismatches, (
        "printed station distances disagree with the haversine distance:\n  "
        + "\n  ".join(mismatches)
    )
    assert checked >= 30, (
        f"only {checked} distances were checked; has a table's format changed? "
        f"(skipped for missing coordinates: {skipped})"
    )


def test_printed_compass_directions_match_the_bearing() -> None:
    """A Location cell leading with a compass point states where the station
    lies from that event's epicenter.

    Five of these were wrong by roughly 90 degrees: stations west of a South
    American epicenter, including one off Alaska, were labelled NNE, and the
    Chile and Illapel tables disagreed with each other about the same station.
    Four more were two points off. One 16-point step of slack is allowed,
    since the labels are coarse by design.
    """
    tex = PAPER.read_text()
    mismatches: list[str] = []
    checked = 0

    for event in EVENTS:
        lat, lon = _epicenter(event)
        for line in _stations_table(tex, event).splitlines():
            row = _ROW.match(line)
            if row is None:
                continue
            lead = _LEADING_BEARING.match(row.group(3).strip())
            if lead is None:
                continue
            coords = station_coordinates(row.group(1))
            if coords is None:
                continue
            printed = lead.group(1)
            if printed not in _COMPASS:
                mismatches.append(f"{event} {row.group(1)}: {printed!r} is not a compass point")
                continue
            bearing = compute_initial_bearing_deg(lat, lon, coords[0], coords[1])
            actual = _compass_point(bearing)
            checked += 1
            apart = min(
                (_COMPASS.index(printed) - _COMPASS.index(actual)) % 16,
                (_COMPASS.index(actual) - _COMPASS.index(printed)) % 16,
            )
            if apart > 1:
                mismatches.append(
                    f"{event} {row.group(1)}: paper {printed}, bearing "
                    f"{bearing:.1f} deg = {actual} ({apart} points apart)"
                )

    assert not mismatches, (
        "printed compass directions disagree with the bearing from the "
        "epicenter:\n  " + "\n  ".join(mismatches)
    )
    assert checked >= 20, f"only {checked} directions were checked; format changed?"


def test_the_two_tables_print_the_same_distance_for_a_station() -> None:
    """A station's distance appears in both its event's tables.

    Correcting one and not the other left the paper disagreeing with itself
    about how far Chile 54401 is from the epicenter.
    """
    tex = PAPER.read_text()
    disagreements: list[str] = []
    for event in EVENTS:
        printed: dict[str, dict[str, str]] = {}
        for table in ("stations", "results"):
            printed[table] = {}
            for line in _table(tex, f"{event}-{table}").splitlines():
                row = _ROW.match(line)
                if row is not None:
                    printed[table][row.group(1)] = row.group(2)
        for station_id in sorted(set(printed["stations"]) & set(printed["results"])):
            left, right = printed["stations"][station_id], printed["results"][station_id]
            if left != right:
                disagreements.append(
                    f"{event} {station_id}: stations table {left}, results table {right}"
                )
    assert not disagreements, "the two tables disagree:\n  " + "\n  ".join(disagreements)
