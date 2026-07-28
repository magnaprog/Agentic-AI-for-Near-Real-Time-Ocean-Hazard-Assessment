"""Unit tests for the scenario data interface.

Tests UnitSource, UnitSourceDatabase ABC, InMemoryUnitSourceDatabase,
FallbackUnitSourceDatabase, source selection, contiguity, and fault
orientation filtering with circular strike statistics.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pytest

from hazard_assessment.agents.scenario_data import (
    FallbackUnitSourceDatabase,
    InMemoryUnitSourceDatabase,
    UnitSource,
    UnitSourceDatabase,
    _circular_mean_strike,
    _strike_angular_distance,
    check_contiguity,
    filter_by_fault_orientation,
    haversine_distance_km,
    select_unit_sources,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_source(
    source_id: str = "src_01",
    lat: float = 0.0,
    lon: float = 0.0,
    strike: float = 45.0,
    fault_zone: str = "zone_A",
    segment_index: int = 0,
    **kwargs,
) -> UnitSource:
    defaults = dict(
        source_id=source_id,
        latitude=lat,
        longitude=lon,
        depth_km=15.0,
        strike_deg=strike,
        dip_deg=15.0,
        rake_deg=90.0,
        length_km=50.0,
        width_km=25.0,
        rigidity_pa=3.5e10,
        fault_zone_id=fault_zone,
        segment_index=segment_index,
    )
    defaults.update(kwargs)
    return UnitSource(**defaults)


def _make_db_with_sources(sources: list[UnitSource]) -> InMemoryUnitSourceDatabase:
    db = InMemoryUnitSourceDatabase()
    for src in sources:
        db.add_source(src)
    return db


# ---------------------------------------------------------------------------
# UnitSource
# ---------------------------------------------------------------------------


class TestUnitSource:
    def test_creation(self):
        src = _make_source()
        assert src.source_id == "src_01"
        assert src.latitude == 0.0
        assert src.fault_zone_id == "zone_A"

    def test_frozen(self):
        src = _make_source()
        with pytest.raises(AttributeError):
            src.latitude = 5.0  # type: ignore[misc]

    def test_area_m2(self):
        src = _make_source(length_km=50.0, width_km=25.0)
        assert src.area_m2 == 50.0 * 25.0 * 1e6


# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------


class TestHaversine:
    def test_known_distances(self):
        # Equator, 1 degree of longitude ~ 111.32 km
        dist = haversine_distance_km(0.0, 0.0, 0.0, 1.0)
        assert abs(dist - 111.32) < 0.5

    def test_same_point(self):
        assert haversine_distance_km(45.0, -120.0, 45.0, -120.0) == 0.0

    def test_antipodal(self):
        dist = haversine_distance_km(0.0, 0.0, 0.0, 180.0)
        assert abs(dist - math.pi * 6371.0) < 1.0


# ---------------------------------------------------------------------------
# InMemoryUnitSourceDatabase
# ---------------------------------------------------------------------------


class TestInMemoryDatabase:
    def test_crud(self):
        db = InMemoryUnitSourceDatabase()
        src = _make_source(lat=10.0, lon=20.0)
        db.add_source(src)
        results = db.get_sources_near(10.0, 20.0, max_distance_km=50.0)
        assert len(results) == 1
        assert results[0].source_id == "src_01"

    def test_available(self):
        db = InMemoryUnitSourceDatabase()
        assert db.available() is True

    def test_get_sources_near_distance_filter(self):
        db = InMemoryUnitSourceDatabase()
        db.add_source(_make_source("near", lat=0.0, lon=0.0))
        db.add_source(_make_source("far", lat=10.0, lon=10.0))
        results = db.get_sources_near(0.0, 0.0, max_distance_km=100.0)
        assert len(results) == 1
        assert results[0].source_id == "near"

    def test_get_sources_near_max_count(self):
        db = InMemoryUnitSourceDatabase()
        for i in range(10):
            db.add_source(_make_source(f"src_{i:02d}", lat=0.0, lon=0.001 * i))
        results = db.get_sources_near(0.0, 0.0, max_distance_km=500.0, max_sources=3)
        assert len(results) == 3

    def test_empty_database(self):
        db = InMemoryUnitSourceDatabase()
        assert db.get_sources_near(0.0, 0.0) == []

    def test_greens_function_shape(self):
        db = InMemoryUnitSourceDatabase()
        db.add_source(_make_source("s1"))
        waveform = np.ones(60, dtype=np.float64)
        db.set_greens_function("s1", "dart_01", waveform)
        gf = db.get_greens_functions(["s1"], ["dart_01"])
        assert gf.waveforms.shape == (1, 60, 1)
        assert gf.source_ids == ["s1"]
        assert gf.station_ids == ["dart_01"]

    def test_greens_function_missing_raises(self):
        db = InMemoryUnitSourceDatabase()
        db.add_source(_make_source("s1"))
        with pytest.raises(KeyError):
            db.get_greens_functions(["s1"], ["nonexistent"])

    def test_set_greens_inconsistent_length_raises(self):
        db = InMemoryUnitSourceDatabase()
        db.set_greens_function("s1", "d1", np.ones(60, dtype=np.float64))
        with pytest.raises(ValueError, match="does not match"):
            db.set_greens_function("s1", "d2", np.ones(30, dtype=np.float64))


# ---------------------------------------------------------------------------
# FallbackUnitSourceDatabase
# ---------------------------------------------------------------------------


class TestFallbackDatabase:
    def test_primary_succeeds(self):
        primary = InMemoryUnitSourceDatabase()
        primary.add_source(_make_source("p1", lat=0.0, lon=0.0))
        fallback = InMemoryUnitSourceDatabase()
        fallback.add_source(_make_source("f1", lat=0.0, lon=0.0))

        db = FallbackUnitSourceDatabase(primary, fallback)
        results = db.get_sources_near(0.0, 0.0, max_distance_km=100.0)
        assert len(results) == 1
        assert results[0].source_id == "p1"

    def test_primary_fails(self):
        primary = MagicMock(spec=UnitSourceDatabase)
        primary.get_sources_near.side_effect = RuntimeError("DB down")
        fallback = InMemoryUnitSourceDatabase()
        fallback.add_source(_make_source("f1", lat=0.0, lon=0.0))

        db = FallbackUnitSourceDatabase(primary, fallback)
        results = db.get_sources_near(0.0, 0.0, max_distance_km=100.0)
        assert len(results) == 1
        assert results[0].source_id == "f1"

    def test_primary_empty_falls_back(self):
        primary = InMemoryUnitSourceDatabase()  # empty
        fallback = InMemoryUnitSourceDatabase()
        fallback.add_source(_make_source("f1", lat=0.0, lon=0.0))

        db = FallbackUnitSourceDatabase(primary, fallback)
        results = db.get_sources_near(0.0, 0.0, max_distance_km=100.0)
        assert results[0].source_id == "f1"

    def test_greens_functions_fallback(self):
        primary = MagicMock(spec=UnitSourceDatabase)
        primary.get_greens_functions.side_effect = KeyError("not found")
        fallback = InMemoryUnitSourceDatabase()
        fallback.add_source(_make_source("s1"))
        wf = np.ones(60, dtype=np.float64)
        fallback.set_greens_function("s1", "d1", wf)

        db = FallbackUnitSourceDatabase(primary, fallback)
        gf = db.get_greens_functions(["s1"], ["d1"])
        assert gf.waveforms.shape == (1, 60, 1)

    def test_available_fallback(self):
        primary = MagicMock(spec=UnitSourceDatabase)
        primary.available.side_effect = RuntimeError("broken")
        fallback = MagicMock(spec=UnitSourceDatabase)
        fallback.available.return_value = True

        db = FallbackUnitSourceDatabase(primary, fallback)
        assert db.available() is True


# ---------------------------------------------------------------------------
# Circular strike statistics
# ---------------------------------------------------------------------------


class TestCircularStrike:
    def test_circular_mean_simple(self):
        # All strikes at 45 degrees -> mean = 45
        mean = _circular_mean_strike([45.0, 45.0, 45.0])
        assert abs(mean - 45.0) < 0.01

    def test_circular_mean_wrap(self):
        # 350 and 10 degrees (near 0/360 boundary)
        # Normalized: 170 and 10 -> doubled: 340 and 20
        # Mean of 340 and 20 in circular: 0 -> halved: 0
        mean = _circular_mean_strike([350.0, 10.0])
        assert abs(mean) < 1.0

    def test_circular_mean_180_ambiguity(self):
        # 10 and 190 describe the same fault plane
        # Both normalize to 10 -> mean = 10
        mean = _circular_mean_strike([10.0, 190.0])
        assert abs(mean - 10.0) < 0.01

    def test_strike_angular_distance_simple(self):
        assert abs(_strike_angular_distance(10.0, 30.0) - 20.0) < 0.01

    def test_strike_angular_distance_wrap(self):
        # 170 and 10 in [0,180) domain -> distance = min(160, 20) = 20
        assert abs(_strike_angular_distance(170.0, 10.0) - 20.0) < 0.01

    def test_strike_angular_distance_180_ambiguity(self):
        # 10 and 190 -> both normalize to 10 -> distance = 0
        assert abs(_strike_angular_distance(10.0, 190.0)) < 0.01

    def test_empty_strikes(self):
        assert _circular_mean_strike([]) == 0.0


# ---------------------------------------------------------------------------
# Fault orientation filter
# ---------------------------------------------------------------------------


class TestFaultOrientation:
    def test_consistent_sources_kept(self):
        sources = [
            _make_source("s1", strike=40.0),
            _make_source("s2", strike=45.0),
            _make_source("s3", strike=50.0),
        ]
        result = filter_by_fault_orientation(sources, tolerance_deg=30.0)
        assert len(result) == 3

    def test_divergent_source_removed(self):
        sources = [
            _make_source("s1", strike=45.0),
            _make_source("s2", strike=47.0),
            _make_source("s3", strike=120.0),  # divergent
        ]
        result = filter_by_fault_orientation(sources, tolerance_deg=30.0)
        assert len(result) == 2
        ids = {s.source_id for s in result}
        assert "s3" not in ids

    def test_circular_wrap_kept(self):
        # Strikes near 0/360 boundary
        sources = [
            _make_source("s1", strike=350.0),
            _make_source("s2", strike=5.0),
            _make_source("s3", strike=355.0),
        ]
        result = filter_by_fault_orientation(sources, tolerance_deg=30.0)
        assert len(result) == 3

    def test_180_ambiguity(self):
        # 10 and 190 are the same fault plane
        sources = [
            _make_source("s1", strike=10.0),
            _make_source("s2", strike=190.0),
            _make_source("s3", strike=15.0),
        ]
        result = filter_by_fault_orientation(sources, tolerance_deg=30.0)
        assert len(result) == 3

    def test_empty_input(self):
        assert filter_by_fault_orientation([]) == []


# ---------------------------------------------------------------------------
# Contiguity
# ---------------------------------------------------------------------------


class TestContiguity:
    def test_single_group(self):
        sources = [
            _make_source("s1", fault_zone="A", segment_index=0),
            _make_source("s2", fault_zone="A", segment_index=1),
            _make_source("s3", fault_zone="A", segment_index=2),
        ]
        groups = check_contiguity(sources)
        assert len(groups) == 1
        assert len(groups[0]) == 3

    def test_gap_splits(self):
        sources = [
            _make_source("s1", fault_zone="A", segment_index=0),
            _make_source("s2", fault_zone="A", segment_index=1),
            # gap at index 2
            _make_source("s3", fault_zone="A", segment_index=3),
            _make_source("s4", fault_zone="A", segment_index=4),
            _make_source("s5", fault_zone="A", segment_index=5),
        ]
        groups = check_contiguity(sources)
        assert len(groups) == 2
        # Largest first
        assert len(groups[0]) == 3
        assert len(groups[1]) == 2

    def test_multi_fault_zone(self):
        sources = [
            _make_source("a1", fault_zone="A", segment_index=0),
            _make_source("a2", fault_zone="A", segment_index=1),
            _make_source("b1", fault_zone="B", segment_index=0),
        ]
        groups = check_contiguity(sources)
        assert len(groups) == 2

    def test_empty(self):
        assert check_contiguity([]) == []


# ---------------------------------------------------------------------------
# Source selection (integration)
# ---------------------------------------------------------------------------


class TestSelectUnitSources:
    def test_within_500km(self):
        sources = [
            _make_source("s1", lat=0.0, lon=0.0, segment_index=0, fault_zone="A"),
            _make_source("s2", lat=0.0, lon=0.1, segment_index=1, fault_zone="A"),
            _make_source("far", lat=10.0, lon=10.0, segment_index=2, fault_zone="B"),
        ]
        db = _make_db_with_sources(sources)
        result = select_unit_sources(db, 0.0, 0.0, max_distance_km=500.0)
        ids = {s.source_id for s in result}
        assert "far" not in ids
        assert "s1" in ids

    def test_max_count(self):
        sources = [
            _make_source(
                f"s{i}", lat=0.0, lon=0.001 * i,
                segment_index=i, fault_zone="A", strike=45.0,
            )
            for i in range(20)
        ]
        db = _make_db_with_sources(sources)
        result = select_unit_sources(db, 0.0, 0.0, max_sources=5)
        assert len(result) <= 5

    def test_empty_database(self):
        db = InMemoryUnitSourceDatabase()
        result = select_unit_sources(db, 0.0, 0.0)
        assert result == []


# ---------------------------------------------------------------------------
# CoastalForecastFactors
# ---------------------------------------------------------------------------


class TestCoastalForecastFactors:
    def test_frozen(self):
        from hazard_assessment.agents.scenario_data import CoastalForecastFactors

        factors = CoastalForecastFactors(
            site_id="site_A",
            unit_source_peak_m={"s1": 0.1},
            travel_time_sec={"s1": 3600.0},
        )
        with pytest.raises(AttributeError):
            factors.site_id = "other"  # type: ignore[misc]

    def test_in_memory_add_get(self):
        from hazard_assessment.agents.scenario_data import CoastalForecastFactors

        db = InMemoryUnitSourceDatabase()
        factors = CoastalForecastFactors(
            site_id="site_A",
            unit_source_peak_m={"s1": 0.1, "s2": 0.2},
            travel_time_sec={"s1": 3600.0, "s2": 4000.0},
        )
        db.add_coastal_factors(factors)
        result = db.get_coastal_forecast_factors(["s1", "s2"], ["site_A"])
        assert "site_A" in result
        assert result["site_A"].unit_source_peak_m["s1"] == 0.1
        assert result["site_A"].travel_time_sec["s2"] == 4000.0

    def test_in_memory_missing_site_raises(self):
        db = InMemoryUnitSourceDatabase()
        with pytest.raises(KeyError, match="site=nonexistent"):
            db.get_coastal_forecast_factors(["s1"], ["nonexistent"])

    def test_fallback_coastal_factors(self):
        from hazard_assessment.agents.scenario_data import CoastalForecastFactors

        primary = MagicMock(spec=UnitSourceDatabase)
        primary.get_coastal_forecast_factors.side_effect = KeyError("not found")
        fallback = InMemoryUnitSourceDatabase()
        factors = CoastalForecastFactors(
            site_id="site_A",
            unit_source_peak_m={"s1": 0.3},
            travel_time_sec={"s1": 5000.0},
        )
        fallback.add_coastal_factors(factors)

        db = FallbackUnitSourceDatabase(primary, fallback)
        result = db.get_coastal_forecast_factors(["s1"], ["site_A"])
        assert result["site_A"].unit_source_peak_m["s1"] == 0.3
