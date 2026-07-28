"""Historical analogue retrieval for tsunami event contextualisation.

Retrieves the most similar historical events from a JSON fixture based on
magnitude difference. This is a simple lookup table (10 canonical events),
not RAG. The retrieved events provide contextual framing for the LLM
narrative - e.g., "this event has a similar magnitude to the 2011 Tohoku
earthquake."

If the fixture is missing or cannot be parsed, returns an empty list
(fail-safe: retrieval failure must never block synthesis).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_FIXTURE_PATH = Path(__file__).resolve().parents[4] / "data" / "historical_events.json"

_MAX_RESULTS = 3


def retrieve_similar_events(
    mw: float,
    *,
    fixture_path: Path | None = None,
    max_results: int = _MAX_RESULTS,
) -> str:
    """Return a JSON string of the most similar historical events.

    Similarity is based on magnitude difference (simplest ranking for a
    10-item fixture). Returns the ``max_results`` closest events sorted
    by ascending magnitude distance.

    Args:
        mw: Moment magnitude of the current event (from top scenario).
        fixture_path: Override path to the historical events JSON.
        max_results: Maximum number of events to return.

    Returns:
        A JSON string representing a list of historical event dicts,
        or ``"[]"`` on any error.
    """
    try:
        path = fixture_path or _FIXTURE_PATH
        events = _load_fixture(path)
        if not events:
            return "[]"

        ranked = sorted(events, key=lambda e: abs(e.get("mw", 0.0) - mw))
        top = ranked[:max_results]
        return json.dumps(top, ensure_ascii=False)
    except Exception:
        logger.exception("Historical analogue retrieval failed")
        return "[]"


def _load_fixture(path: Path) -> list[dict[str, Any]]:
    """Load the historical events JSON fixture."""
    if not path.exists():
        logger.warning("Historical events fixture not found: %s", path)
        return []
    with open(path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        logger.warning("Historical events fixture is not a list")
        return []
    return data
