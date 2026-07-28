"""The committed detection artifacts must still be what the detector produces.

``results/*_detection.json`` are inputs to the paper: the appendix station
tables and six figures are generated from them. Without a reproduction gate a
change to the detector leaves those artifacts, and therefore the published
numbers, silently out of step with the code.

These tests run the offline validators and compare. They are slow by the
standards of the unit suite (roughly one to two minutes per event), so they live
in their own directory and their own CI job, gated on the paths that can cause
the drift.

Coverage is deliberately partial, and here is what is not covered. The five
detection artifacts and the duplicate-sensitivity artifact are reproduced here
because they rescore archived station data through the detector, which is the
path most sensitive to such a change. Verified by hand at the time of writing
and found to match, but not automated: ``detiding_validation.json``,
``ablation_results.json``, ``fsm_transitions.json`` (byte-identical) and
``physics_validation.json`` (identical apart from its embedded
``generated_at``), ``synthetic_evaluation.json`` (byte-identical, all 420 grid
configurations), and the three outputs of ``run_synthetic_pipeline.py``:
``synthetic_timelines.json`` apart from its timestamp, with ``agent_traces.json``
and ``paper/appendix_f_generated.tex`` byte for byte. The last two generators
are left out of CI because each runs for roughly an hour, not because they are
unchecked.

Two levels of comparison, deliberately different:

* Claim level, exact. Threshold-crossing counts and the summary block are what
  the paper asserts, so any change there is a change to a published claim and
  must fail loudly.
* Numeric level, tolerant. Per-station scores are compared at 1e-6, which is far
  below the 1e-3 to 6.6e-2 drift the real regression produced and far above the
  floating-point variation a different BLAS might introduce.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS = REPO_ROOT / "results"
THRESHOLDS = {"T1": 0.35, "T2": 0.60, "T3": 0.85}
SCORE_TOLERANCE = 1e-6

EVENTS = ("tohoku", "chile", "illapel", "iquique", "samoa")

#: Figures generated from each detection artifact, so a failure names what else
#: has to be regenerated. Verified against generate_paper_figures.py: each
#: event's artifact is loaded once and passed to exactly these two functions.
#: Finding this by hand is what made the last regeneration slow.
FIGURES_FROM_DETECTION = {
    event: (f"fig_{event}_detection.pdf", f"fig_{event}_detection_timeline.pdf")
    for event in EVENTS
}


def _crossing_counts(payload: dict[str, Any]) -> dict[str, int]:
    stations = payload.get("per_station", [])
    return {
        name: sum(1 for s in stations if float(s.get("ensemble_score", 0.0)) >= threshold)
        for name, threshold in THRESHOLDS.items()
    }


def _run_validator(event: str) -> dict[str, Any]:
    """Reproduce one event and return the artifact the validator wrote.

    The committed file is restored by the fixture, so a failing run cannot
    leave the working tree carrying half-regenerated results.
    """
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, f"scripts/validate_{event}.py", "--sliding-window"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, (
        f"validate_{event}.py failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
    )
    return json.loads((RESULTS / f"{event}_detection.json").read_text())


@pytest.fixture
def restored_results() -> Any:
    """Restore results/ from git after each test, whatever the outcome."""
    yield
    subprocess.run(["git", "checkout", "--", "results"], cwd=REPO_ROOT, check=False)


@pytest.mark.parametrize("event", EVENTS)
def test_detection_artifact_still_reproduces(event: str, restored_results: Any) -> None:
    committed = json.loads((RESULTS / f"{event}_detection.json").read_text())
    regenerated = _run_validator(event)

    assert _crossing_counts(regenerated) == _crossing_counts(committed), (
        f"{event}: threshold-crossing counts changed, which changes a published claim"
    )
    assert regenerated.get("summary") == committed.get("summary"), (
        f"{event}: summary block changed"
    )

    committed_stations = {s["station_id"]: s for s in committed["per_station"]}
    regenerated_stations = {s["station_id"]: s for s in regenerated["per_station"]}
    assert set(regenerated_stations) == set(committed_stations), f"{event}: station set changed"

    drifted: list[str] = []
    for station_id, produced in regenerated_stations.items():
        stored = committed_stations[station_id]
        for key in ("ensemble_score", "threshold_score", "wavelet_score", "bocpd_score"):
            # Assert presence rather than skipping. A `continue` here turned a
            # key disappearing from either side into a silent pass, so a
            # component dropping out of the output looked identical to a
            # component that still matched.
            assert key in stored, f"{event} {station_id}: committed artifact lost {key}"
            assert key in produced, f"{event} {station_id}: detector no longer emits {key}"
            delta = abs(float(stored[key]) - float(produced[key]))
            if delta > SCORE_TOLERANCE:
                drifted.append(
                    f"{station_id}.{key}: committed {stored[key]} vs produced {produced[key]}"
                )
        # filter_degraded is not a score but it is published: the appendix
        # tables print it as the "Deg" column, and it changes how a reader is
        # told to interpret every other number in the row. Compare it exactly.
        for flag in ("filter_degraded",):
            assert flag in stored, f"{event} {station_id}: committed artifact lost {flag}"
            assert flag in produced, f"{event} {station_id}: detector no longer emits {flag}"
            if bool(stored[flag]) != bool(produced[flag]):
                drifted.append(
                    f"{station_id}.{flag}: committed {stored[flag]} vs produced {produced[flag]}"
                )

    figures = ", ".join(FIGURES_FROM_DETECTION[event])
    assert not drifted, (
        f"{event}: committed artifact no longer matches the detector.\n"
        f"Regenerate: results/{event}_detection.json, paper/figures/{{{figures}}}, "
        "and any paper value they feed (tests/unit/test_paper_matches_artifacts.py "
        "checks the tables).\n  " + "\n  ".join(drifted)
    )


DUPLICATE_SENSITIVITY = RESULTS / "duplicate_sensitivity.json"
DUPLICATE_TOLERANCE = 0.01


def _duplicate_claims(payload: dict[str, Any]) -> tuple[int, int, set[tuple[str, str]]]:
    """The three facts the paper states about duplicate-policy sensitivity.

    Namely: how many station records exist, how many stay within 0.01 of the
    frozen policy under every alternative, and which stations change threshold
    tier.
    """
    records = 0
    stable = 0
    tier_changes: set[tuple[str, str]] = set()

    def tier(score: float) -> str:
        reached = "none"
        for name, threshold in THRESHOLDS.items():
            if score >= threshold:
                reached = name
        return reached

    for event, event_payload in payload["events"].items():
        for station_id, station in event_payload["stations"].items():
            records += 1
            policies = station["by_policy"]
            frozen = float(policies["first"]["ensemble_score"])
            alternatives = [
                float(p["ensemble_score"]) for name, p in policies.items() if name != "first"
            ]
            if all(abs(score - frozen) <= DUPLICATE_TOLERANCE for score in alternatives):
                stable += 1
            if any(tier(score) != tier(frozen) for score in alternatives):
                tier_changes.add((event, station_id))
    return records, stable, tier_changes


def test_duplicate_sensitivity_artifact_still_reproduces(restored_results: Any) -> None:
    """Same drift class as the detection artifacts, same paper exposure.

    This file rescores every station record under four alternative equal-time
    policies, and the paper states counts derived from it: 34 station records,
    31 within 0.01 of the frozen policy, exactly one station changing tier.
    """
    committed = json.loads(DUPLICATE_SENSITIVITY.read_text())

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "scripts/evaluate_duplicate_sensitivity.py"],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    assert completed.returncode == 0, (
        f"evaluate_duplicate_sensitivity.py failed:\n{completed.stdout[-2000:]}\n"
        f"{completed.stderr[-2000:]}"
    )
    regenerated = json.loads(DUPLICATE_SENSITIVITY.read_text())

    assert _duplicate_claims(regenerated) == _duplicate_claims(committed), (
        "duplicate-policy claims changed: the paper states 34 station records, "
        "31 within 0.01 of the frozen policy, and one station changing tier"
    )

    drifted: list[str] = []
    for event, event_payload in regenerated["events"].items():
        for station_id, station in event_payload["stations"].items():
            stored = committed["events"][event]["stations"][station_id]["by_policy"]
            for policy, produced in station["by_policy"].items():
                delta = abs(
                    float(produced["ensemble_score"])
                    - float(stored[policy]["ensemble_score"])
                )
                if delta > SCORE_TOLERANCE:
                    drifted.append(f"{event}/{station_id}/{policy}: delta {delta:.6f}")

    assert not drifted, (
        "committed duplicate_sensitivity.json no longer matches the detector:\n  "
        + "\n  ".join(drifted)
    )
