"""The paper's printed numbers must match the artifacts they come from.

The drift chain runs code -> results -> paper. ``tests/artifacts/`` guards the
first link by reproducing ``results/`` from the code. Nothing guarded the
second, which is exactly where the stale numbers surfaced: three appendix
station tables and three prose values printed scores the artifacts no longer
contained, and the only reason anyone noticed was a manual reproduction run.

This reads both files and compares. It runs no generator and takes
milliseconds, so unlike the reproduction tests it belongs in the normal suite.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper" / "paper.tex"
RESULTS = REPO_ROOT / "results"

EVENTS = ("tohoku", "chile", "illapel", "iquique", "samoa")

#: Column order of the per-station appendix tables, from their headers:
#: Station | Dist | Mode | Ens. | Thr. | Wav. | BCP | INV | ASS | ESC | Deg
SCORE_COLUMNS = ("ensemble_score", "threshold_score", "wavelet_score", "bocpd_score")

THRESHOLDS = {"T1": 0.35, "T2": 0.60, "T3": 0.85}


def _artifact(event: str) -> dict[str, Any]:
    return json.loads((RESULTS / f"{event}_detection.json").read_text())


def _tables(tex: str) -> list[str]:
    return [m.group(0) for m in re.finditer(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, re.S)]


def _results_tables(tex: str) -> dict[str, str]:
    """Map each event to its per-station results table, keyed by LaTeX label.

    Keyed by label rather than by scanning the caption. Four of the five
    results tables cite the Tohoku table for comparison, so a caption rule
    like "names exactly one event" would skip those four and leave the guard
    checking almost nothing. Keying on the label makes a rename fail the
    count assertion instead.
    """
    found: dict[str, str] = {}
    for block in _tables(tex):
        match = re.search(r"\\label\{tab:([a-z]+)-results\}", block)
        if match and match.group(1) in EVENTS:
            found[match.group(1)] = block
    return found


def test_appendix_station_tables_match_their_artifacts() -> None:
    """Each per-station row must match its own event's artifact.

    Scoped per table environment on purpose: station 32401 appears in both the
    Illapel and Iquique tables with different scores, so a repository-wide
    search for a station ID compares the wrong rows and invents disagreements.
    """
    tex = PAPER.read_text()
    checked = 0
    mismatches: list[str] = []

    tables = _results_tables(tex)
    assert set(tables) == set(EVENTS), (
        f"expected a results table per event, found {sorted(tables)}"
    )

    per_event_rows: dict[str, int] = {}
    for event, block in tables.items():
        stations = {str(s["station_id"]): s for s in _artifact(event)["per_station"]}
        matched_rows = 0
        for line in block.splitlines():
            # Anything between the station ID and the first column separator:
            # the tables are column-aligned to different widths, and some rows
            # carry a footnote marker such as `$\dagger$`. Requiring a single
            # space here matched no row at all in the Chile table, so that
            # event contributed nothing and the surplus from the other four
            # kept the total above the floor below.
            row = re.match(r"\s*(\d{5})[^&]*&", line)
            if not row or row.group(1) not in stations:
                continue
            matched_rows += 1
            station = stations[row.group(1)]
            # The trailing "Deg" column is the filter_degraded flag. It is
            # printed for every station and tells the reader how to interpret
            # the rest of the row, but nothing compared it against the
            # artifact, at any layer.
            cells = [c.strip() for c in line.split("&")]
            if len(cells) >= len(SCORE_COLUMNS) + 4:
                printed_degraded = "checkmark" in cells[-1]
                stored_degraded = bool(station.get("filter_degraded", False))
                if printed_degraded != stored_degraded:
                    mismatches.append(
                        f"{event} station {row.group(1)} filter_degraded: "
                        f"paper {printed_degraded} vs artifact {stored_degraded}"
                    )
                checked += 1
            printed = re.findall(r"& ([01]\.\d{2,3})", line)
            for text, column in zip(printed[: len(SCORE_COLUMNS)], SCORE_COLUMNS):
                expected = station.get(column)
                if expected is None:
                    continue
                # Derive precision from the printed string, not from the float.
                # str(float("0.50")) is "0.5", which silently widened the
                # comparison to one decimal and let a 0.05 drift pass.
                value = float(text)
                decimals = len(text.split(".")[1])
                if abs(value - round(float(expected), decimals)) > 5e-4:
                    mismatches.append(
                        f"{event} station {row.group(1)} {column}: "
                        f"paper {value} vs artifact {expected}"
                    )
                checked += 1
        per_event_rows[event] = matched_rows

    # Per event, not just in total. A global floor is satisfied by the surplus
    # from the other events, so one table dropping out entirely passes it.
    unmatched = {
        event: len(_artifact(event)["per_station"])
        for event, rows in per_event_rows.items()
        if rows != len(_artifact(event)["per_station"])
    }
    assert not unmatched, (
        "the row regex did not match every station row for: "
        f"{unmatched}; has a table's formatting changed?"
    )
    assert checked >= 4 * len(EVENTS), (
        f"only {checked} score values were checked; has the table format changed?"
    )
    assert not mismatches, "paper values disagree with results/:\n  " + "\n  ".join(mismatches)


@pytest.mark.parametrize("event", EVENTS)
def test_cross_event_summary_counts_match(event: str) -> None:
    """The consolidated table states station counts and detection rates.

    These are the paper's headline detection claims, so they are compared
    exactly rather than at a tolerance.
    """
    tex = PAPER.read_text()
    label = tex.find(r"\label{tab:cross-event-summary}")
    assert label > 0, "cross-event summary table not found"
    block = tex[tex.rfind(r"\begin{table", 0, label) : tex.find(r"\end{table", label)]

    payload = _artifact(event)
    stations = payload["per_station"]
    expected = [
        len(stations),
        *[
            sum(1 for s in stations if float(s.get("ensemble_score", 0.0)) >= threshold)
            for threshold in THRESHOLDS.values()
        ],
    ]

    # Rows are named by the event with LaTeX accents, so match on the year.
    year = {"tohoku": "2011", "chile": "2010", "illapel": "2015",
            "iquique": "2014", "samoa": "2009"}[event]
    row = next((line for line in block.splitlines() if f"{year} " in line and "&" in line), None)
    assert row is not None, f"no cross-event row for {event}"

    counts = [int(x) for x in re.findall(r"& (\d+)\\?,?\(?", row)]
    printed = [c for c in counts if c < 100][-4:]
    assert printed == expected, (
        f"{event}: cross-event table says {printed}, artifact gives {expected} "
        "(stations, T1, T2, T3)"
    )


SYNTHETIC = RESULTS / "synthetic_evaluation.json"


def test_synthetic_grid_matches_the_paper_and_is_complete() -> None:
    """The paper describes the synthetic grid in prose; the artifact defines it.

    Two failures this catches. A grid edited in one place and not the other,
    which is the same coupling the station tables have. And a truncated run:
    the artifact records how many configurations it intended and how many
    finished, and the paper's minimum-detectable-signal discussion assumes the
    full grid.
    """
    payload = json.loads(SYNTHETIC.read_text())
    grid = payload["parameters"]

    sentence = re.search(
        r"The evaluation grid spans amplitudes from ([\d.]+) to ([\d.]+)\\,m,\s*"
        r"periods from (\d+) to (\d+)\\,min, noise levels of ([\d.]+)--([\d.]+)\\,m,\s*"
        r"and sampling intervals of (\d+)\\,s and (\d+)\\,s\.",
        PAPER.read_text(),
    )
    assert sentence is not None, "the paper's grid sentence has changed shape"
    amp_lo, amp_hi, per_lo, per_hi, noise_lo, noise_hi, fast, slow = sentence.groups()

    assert float(amp_lo) == min(grid["amplitudes_m"])
    assert float(amp_hi) == max(grid["amplitudes_m"])
    assert int(per_lo) == min(grid["periods_min"])
    assert int(per_hi) == max(grid["periods_min"])
    assert float(noise_lo) == min(grid["noise_stds_m"])
    assert float(noise_hi) == max(grid["noise_stds_m"])
    assert sorted(int(x) for x in (fast, slow)) == sorted(grid["sampling_intervals_sec"])

    expected = 1
    for key in ("amplitudes_m", "periods_min", "noise_stds_m", "sampling_intervals_sec"):
        expected *= len(grid[key])
    assert payload["total_configurations"] == expected, "grid product disagrees with the header"
    assert payload["total_completed"] == expected, (
        f"only {payload['total_completed']} of {expected} configurations completed; "
        "the minimum-detectable-signal discussion assumes the full grid"
    )


def test_guardrail_test_counts_match_the_paper() -> None:
    """The paper prints how much the guardrail suite covers; keep it honest.

    These two numbers drifted once already, from 56/106 to 67/118, because
    nothing tied them to the file they describe. Counting the source rather
    than pytest's collected total is deliberate: the paper says "test
    functions", and parametrization would make the collected count say
    something different from what the sentence claims.
    """
    sentence = re.search(
        r"exercises (\d+)~guardrail test functions with\s*\n?(\d+)~assertions",
        PAPER.read_text(),
    )
    assert sentence is not None, "the paper's guardrail-coverage sentence has changed shape"
    claimed_functions, claimed_assertions = (int(g) for g in sentence.groups())

    tree = ast.parse((REPO_ROOT / "tests" / "unit" / "test_guardrails.py").read_text())
    functions = sum(
        isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.name.startswith("test_")
        for node in ast.walk(tree)
    )
    assertions = sum(isinstance(node, ast.Assert) for node in ast.walk(tree))

    assert claimed_functions == functions, (
        f"paper claims {claimed_functions} guardrail test functions, "
        f"test_guardrails.py defines {functions}"
    )
    assert claimed_assertions == assertions, (
        f"paper claims {claimed_assertions} assertions, "
        f"test_guardrails.py has {assertions}"
    )


def test_script_threshold_literals_match_the_settings_defaults() -> None:
    """Every script that hardcodes T1/T2/T3 must agree with the settings.

    The artifact generators do not read ThresholdSettings; they carry their own
    literals, and the crossing counts in results/*_detection.json come from
    those literals. Because tests/artifacts/ reproduces the artifacts by
    re-running the same scripts, a change to the settings default would leave
    the published counts on the old thresholds with every check still green.
    Nothing else ties the two together, so this does.
    """
    from hazard_assessment.config.settings import ThresholdSettings

    settings = ThresholdSettings()
    expected = {"1": settings.t1, "2": settings.t2, "3": settings.t3}

    pattern = re.compile(r"^T([123])(?:_[A-Z_]+)?\s*=\s*([0-9.]+)\s*$", re.M)
    checked = 0
    for path in sorted((REPO_ROOT / "scripts").glob("*.py")):
        for tier, literal in pattern.findall(path.read_text()):
            checked += 1
            assert float(literal) == expected[tier], (
                f"{path.name} sets T{tier}={literal}, but "
                f"ThresholdSettings default is {expected[tier]}"
            )

    # Guard the guard: if the naming changes and the regex stops matching, this
    # test would pass while checking nothing.
    assert checked >= 15, f"only matched {checked} threshold literals; regex is stale"
