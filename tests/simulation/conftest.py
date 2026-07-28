"""Shared fixtures for simulation tests.

Module-scoped fixtures for expensive event generation (~2-5s each).
"""

from __future__ import annotations

import pytest

from hazard_assessment.simulation.propagation import (
    PACIFIC_COOPS_STATIONS,
    PACIFIC_DART_STATIONS,
)
from hazard_assessment.simulation.scenario import generate_coherent_event
from hazard_assessment.simulation.source import (
    ALEUTIAN_SCENARIO,
    MODERATE_ALEUTIAN,
    MODERATE_PACIFIC,
    TOHOKU_LIKE,
)


@pytest.fixture(scope="module")
def tohoku_event():
    """Generate a Tohoku-like M9.1 event with 6 DART + 2 CO-OPS stations."""
    return generate_coherent_event(
        earthquake=TOHOKU_LIKE,
        stations=PACIFIC_DART_STATIONS[:6] + PACIFIC_COOPS_STATIONS[:2],
        seed=42,
    )


@pytest.fixture(scope="module")
def moderate_event():
    """Generate a moderate M7.2 event with 4 DART stations."""
    return generate_coherent_event(
        earthquake=MODERATE_PACIFIC,
        stations=PACIFIC_DART_STATIONS[:4],
        seed=123,
    )


@pytest.fixture(scope="module")
def aleutian_event():
    """Generate an M8.5 Aleutian event with all 6 DART stations."""
    return generate_coherent_event(
        earthquake=ALEUTIAN_SCENARIO,
        stations=PACIFIC_DART_STATIONS,
        seed=42,
    )


@pytest.fixture(scope="module")
def moderate_aleutian_event():
    """Generate a moderate M7.0 Aleutian event with 4 DART stations."""
    return generate_coherent_event(
        earthquake=MODERATE_ALEUTIAN,
        stations=PACIFIC_DART_STATIONS[:4],
        seed=456,
    )
