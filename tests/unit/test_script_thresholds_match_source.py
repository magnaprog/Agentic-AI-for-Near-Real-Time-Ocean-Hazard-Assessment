"""Every FSM threshold hardcoded in scripts/ must equal the source of truth.

The evaluation scripts each declare their own ``T1``/``T2``/``T3`` rather than
reading ``ThresholdSettings``. Only ``scripts/profile_latency.py`` imports the
real config. The copies all agree today, but nothing linked them: changing a
threshold in ``config/settings.py`` would leave every published ``would_*``
boolean, every FSM-transition artifact and two figure labels silently
computed against the old value, and the artifact-reproduction gate would not
notice because it replays the same scripts.

Rewriting fifteen scripts to import the config would be a wide change to
working code for no behavioural gain, so this test closes the gap instead: it
parses each script and fails if any hardcoded threshold drifts from
``ThresholdSettings``. Two spellings are in use, ``T1 = 0.35`` and
``T1_MONITOR_TO_INVESTIGATE = 0.35``, and both are covered.
"""

from __future__ import annotations

import ast
import pathlib
import re

from hazard_assessment.config.settings import ThresholdSettings

SCRIPTS = pathlib.Path(__file__).resolve().parents[2] / "scripts"

#: Threshold name prefix to the settings attribute it must equal.
_EXPECTED = {
    "T1": "t1",
    "T2": "t2",
    "T3": "t3",
}


def _threshold_name(name: str) -> str | None:
    """Return the threshold a constant name declares, or None."""
    for prefix in _EXPECTED:
        if name == prefix or name.startswith(f"{prefix}_"):
            return prefix
    return None


def _hardcoded_thresholds(source: str) -> list[tuple[str, str, float]]:
    """Collect module-level threshold constants as (name, threshold, value)."""
    found: list[tuple[str, str, float]] = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            # T1 = 0.35
            if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
                threshold = _threshold_name(target.id)
                if threshold is not None and isinstance(node.value.value, float):
                    found.append((target.id, threshold, node.value.value))
            # T1, T2, T3 = 0.35, 0.60, 0.85
            elif isinstance(target, ast.Tuple) and isinstance(node.value, ast.Tuple):
                for element, value in zip(target.elts, node.value.elts):
                    if not isinstance(element, ast.Name):
                        continue
                    if not isinstance(value, ast.Constant):
                        continue
                    threshold = _threshold_name(element.id)
                    if threshold is not None and isinstance(value.value, float):
                        found.append((element.id, threshold, value.value))
    return found


def test_script_thresholds_match_settings() -> None:
    settings = ThresholdSettings()
    drifted: list[str] = []
    checked = 0

    for script in sorted(SCRIPTS.glob("*.py")):
        for name, threshold, value in _hardcoded_thresholds(script.read_text()):
            expected = getattr(settings, _EXPECTED[threshold])
            checked += 1
            if value != expected:
                drifted.append(
                    f"{script.name}: {name} = {value}, "
                    f"ThresholdSettings.{_EXPECTED[threshold]} = {expected}"
                )

    assert not drifted, (
        "hardcoded FSM thresholds disagree with config/settings.py:\n  "
        + "\n  ".join(drifted)
    )
    # A floor, so that renaming the constants into a spelling this parser does
    # not recognize fails here rather than quietly checking nothing.
    assert checked >= 42, (
        f"only {checked} hardcoded thresholds were found in scripts/; "
        "have the constants been renamed?"
    )


def test_mission_control_thresholds_match_settings() -> None:
    """The console carries its own copy of the thresholds, twice.

    ``constants.ts`` holds a fallback used before the first snapshot arrives,
    and the demo snapshot ships a fixed set for the no-upstream mode. Neither
    is reachable from Python, so both sat outside every check. A console
    drawing its threshold lines at values the FSM no longer uses would be
    wrong in the one place an operator reads them off a chart.
    """
    settings = ThresholdSettings()
    expected = {"t1": settings.t1, "t2": settings.t2, "t3": settings.t3}

    mc = SCRIPTS.parent / "mission-control"
    sources = {
        "frontend/src/constants.ts": (mc / "frontend" / "src" / "constants.ts"),
        "backend/services/demo_snapshot.py": (
            mc / "backend" / "services" / "demo_snapshot.py"
        ),
    }
    drifted: list[str] = []
    for label, path in sources.items():
        text = path.read_text(encoding="utf-8")
        for name, value in expected.items():
            # Matches both `t1: 0.35` (TypeScript) and `"t1": 0.35` (Python).
            found = re.findall(rf'"?{name}"?\s*:\s*([0-9.]+)', text)
            if not found:
                drifted.append(f"{label}: no value found for {name}")
                continue
            for raw in found:
                if float(raw) != value:
                    drifted.append(f"{label}: {name} = {raw}, settings has {value}")

    assert not drifted, (
        "Mission Control thresholds disagree with config/settings.py:\n  "
        + "\n  ".join(drifted)
    )
