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
from hazard_assessment.geo import haversine_km

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

_ROW = re.compile(r"^\s*(\d{5})\s*&\s*([0-9{},]+)\s*&")


def _epicenter(event: str) -> tuple[float, float]:
    if event == "chile":
        return _CHILE_EARLY_EPICENTER
    geo = json.loads((DATA / event / "seismic_event.json").read_text())
    lon, lat, _depth = geo["geometry"]["coordinates"]
    return lat, lon


def _stations_table(tex: str, event: str) -> str:
    marker = f"\\label{{tab:{event}-stations}}"
    start = tex.index(marker)
    return tex[start : tex.index("\\end{tabular}", start)]


def test_printed_station_distances_are_real_haversine_distances() -> None:
    tex = PAPER.read_text()
    mismatches: list[str] = []
    checked = 0
    skipped: list[str] = []

    for event in EVENTS:
        lat, lon = _epicenter(event)
        for line in _stations_table(tex, event).splitlines():
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
