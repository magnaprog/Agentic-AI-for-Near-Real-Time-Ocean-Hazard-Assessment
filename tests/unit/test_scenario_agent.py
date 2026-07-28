"""Unit tests for the ScenarioAgent.

Tests seismic-only mode, DART-constrained inversion, station exclusions,
error handling, envelope validation, bootstrap wiring, coastal proxy wiring,
and ensemble spread computation.
"""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pytest

from hazard_assessment.agents.scenario_agent import ScenarioAgent
from hazard_assessment.agents.scenario_data import (
    CoastalForecastFactors,
    InMemoryUnitSourceDatabase,
    UnitSource,
)
from hazard_assessment.agents.scenario_inversion import (
    SEISMIC_ONLY_LABEL,
    BootstrapConfig,
)
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    ScenarioAssessment,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXED_TIME = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)


def _make_source(
    source_id: str,
    lat: float = 0.0,
    lon: float = 0.0,
    segment_index: int = 0,
) -> UnitSource:
    return UnitSource(
        source_id=source_id,
        latitude=lat,
        longitude=lon,
        depth_km=15.0,
        strike_deg=45.0,
        dip_deg=15.0,
        rake_deg=90.0,
        length_km=50.0,
        width_km=25.0,
        rigidity_pa=3.5e10,
        fault_zone_id="zone_A",
        segment_index=segment_index,
    )


def _build_test_db(
    n_sources: int = 5,
    n_timepoints: int = 60,
    station_ids: list[str] | None = None,
) -> InMemoryUnitSourceDatabase:
    """Build a test database with sources near the origin and Green's functions."""
    db = InMemoryUnitSourceDatabase()
    if station_ids is None:
        station_ids = ["dart_01", "dart_02"]

    sources = []
    for i in range(n_sources):
        src = _make_source(f"src_{i:02d}", lat=0.0, lon=0.01 * i, segment_index=i)
        db.add_source(src)
        sources.append(src)

    rng = np.random.default_rng(42)
    for src in sources:
        for sid in station_ids:
            waveform = rng.standard_normal(n_timepoints).astype(np.float64)
            db.set_greens_function(src.source_id, sid, waveform)

    return db


# ---------------------------------------------------------------------------
# Seismic-only mode
# ---------------------------------------------------------------------------


class TestSeismicOnly:
    def test_produces_assessment(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert isinstance(result, ScenarioAssessment)
        assert result.constraint_stage == ConstraintStage.SEISMIC_ONLY

    def test_dart_stations_empty(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert result.dart_stations_used == []

    def test_has_mandatory_label(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert SEISMIC_ONLY_LABEL in result.limiting_assumptions

    def test_ensemble_spread_high(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert result.ensemble_spread == EnsembleSpread.HIGH


# ---------------------------------------------------------------------------
# DART-constrained mode
# ---------------------------------------------------------------------------


class TestDartConstrained:
    def test_single_station(self):
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        waveforms = {"dart_01": np.random.default_rng(42).standard_normal(60)}
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert result.constraint_stage == ConstraintStage.DART_CONSTRAINED
        assert len(result.dart_stations_used) == 1

    def test_multi_station(self):
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert result.constraint_stage == ConstraintStage.MULTI_STATION
        assert len(result.dart_stations_used) == 2

    def test_exclusions(self):
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_99": rng.standard_normal(60),  # not in DB
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert "dart_99" in result.dart_stations_excluded
        assert "dart_99" in result.exclusion_reasons

    def test_station_excluded_on_missing_greens(self):
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        # dart_02 is NOT in the DB's Green's functions
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert "dart_02" in result.dart_stations_excluded
        assert "propagation database" in result.exclusion_reasons.get("dart_02", "")

    def test_pre_excluded_station_not_used_in_fit(self):
        """Pre-excluded stations must not appear in dart_stations_used."""
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            dart_stations_excluded=["dart_02"],
            exclusion_reasons={"dart_02": "QC failed"},
            processing_time=FIXED_TIME,
        )
        assert "dart_02" not in result.dart_stations_used
        assert "dart_02" in result.dart_stations_excluded
        assert result.exclusion_reasons["dart_02"] == "QC failed"

    def test_pre_excluded_without_reason_gets_default(self):
        """Pre-excluded stations without explicit reasons get a default."""
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            dart_stations_excluded=["dart_02"],
            # no exclusion_reasons provided
            processing_time=FIXED_TIME,
        )
        assert "dart_02" in result.exclusion_reasons
        assert result.exclusion_reasons["dart_02"]  # non-empty


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    def test_no_database_raises(self):
        agent = ScenarioAgent()
        with pytest.raises(ValueError, match="UnitSourceDatabase"):
            agent.run_seismic_only(
                magnitude=8.0,
                epicenter_lat=0.0,
                epicenter_lon=0.0,
                region="Pacific",
            )

    def test_empty_waveforms_raises(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        with pytest.raises(ValueError, match="No station waveforms"):
            agent.run_dart_constrained(
                station_waveforms={},
                epicenter_lat=0.0,
                epicenter_lon=0.0,
            )

    def test_all_stations_excluded_raises(self):
        """All stations fail Green's function lookup -> ValueError."""
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        # Only provide stations that are NOT in the DB's Green's functions
        waveforms = {
            "dart_99": rng.standard_normal(60),
            "dart_98": rng.standard_normal(60),
        }
        with pytest.raises(ValueError, match="No usable DART stations"):
            agent.run_dart_constrained(
                station_waveforms=waveforms,
                epicenter_lat=0.0,
                epicenter_lon=0.0,
                processing_time=FIXED_TIME,
            )


# ---------------------------------------------------------------------------
# Assessment validation
# ---------------------------------------------------------------------------


class TestAssessmentValidation:
    def test_schema_validates(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        # Round-trip through Pydantic validation
        validated = ScenarioAssessment.model_validate(result.model_dump())
        assert validated.constraint_stage == result.constraint_stage

    def test_decision_trace_populated(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert len(result.decision_trace) >= 2
        steps = [s.step for s in result.decision_trace]
        assert any("Unit source selection" in s for s in steps)
        assert any("seismic" in s.lower() for s in steps)


# ---------------------------------------------------------------------------
# Helpers for bootstrap, coastal proxy, and handoff tests
# ---------------------------------------------------------------------------


def _build_test_db_with_coastal(
    n_sources: int = 5,
    n_timepoints: int = 60,
    station_ids: list[str] | None = None,
    coastal_site_ids: list[str] | None = None,
) -> InMemoryUnitSourceDatabase:
    """Build a test database with sources, Green's functions, and coastal factors."""
    db = _build_test_db(n_sources=n_sources, n_timepoints=n_timepoints,
                        station_ids=station_ids)
    if coastal_site_ids is None:
        coastal_site_ids = ["site_A"]

    source_ids = [f"src_{i:02d}" for i in range(n_sources)]
    rng = np.random.default_rng(99)
    for site_id in coastal_site_ids:
        peaks = {sid: float(rng.uniform(0.01, 0.5)) for sid in source_ids}
        travel_times = {sid: float(rng.uniform(1800, 7200)) for sid in source_ids}
        db.add_coastal_factors(CoastalForecastFactors(
            site_id=site_id,
            unit_source_peak_m=peaks,
            travel_time_sec=travel_times,
        ))
    return db


EVENT_ORIGIN = datetime(2024, 1, 15, 11, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Bootstrap wiring
# ---------------------------------------------------------------------------


class TestDartConstrainedBootstrap:
    def test_with_bootstrap(self):
        db = _build_test_db_with_coastal(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            bootstrap_config=BootstrapConfig(n_iterations=20, seed=42),
            coastal_site_ids=["site_A"],
        )
        assert isinstance(result, ScenarioAssessment)
        # With bootstrap + coastal data, spread is computed from coastal amplitude
        # P90/P10 ratio - verify it is a valid EnsembleSpread (not just defaulted)
        assert isinstance(result.ensemble_spread, EnsembleSpread)
        # Verify the decision trace shows the spread was computed via coastal proxies
        coastal_steps = [
            s for s in result.decision_trace if "Coastal amplitude proxies" in s.step
        ]
        assert len(coastal_steps) == 1
        assert "ensemble_spread=" in coastal_steps[0].evidence

    def test_single_station_skips_bootstrap(self):
        db = _build_test_db_with_coastal(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            bootstrap_config=BootstrapConfig(n_iterations=20, seed=42),
        )
        # Single station -> bootstrap skipped -> spread = HIGH
        assert result.ensemble_spread == EnsembleSpread.HIGH
        # Verify limiting assumption explains why bootstrap was skipped
        assert any(
            "bootstrap" in a.lower() and "skipped" in a.lower()
            for a in result.limiting_assumptions
        )

    def test_limiting_assumptions_low_stations(self):
        """< 5 stations with bootstrap -> station diversity warning."""
        db = _build_test_db_with_coastal(
            station_ids=["dart_01", "dart_02", "dart_03"]
        )
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
            "dart_03": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            bootstrap_config=BootstrapConfig(n_iterations=10, seed=42),
        )
        assert any(
            "station diversity" in a.lower()
            for a in result.limiting_assumptions
        )

    def test_decision_trace_bootstrap_entry(self):
        db = _build_test_db_with_coastal(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            bootstrap_config=BootstrapConfig(n_iterations=10, seed=42),
        )
        steps = [s.step for s in result.decision_trace]
        assert any("Bootstrap uncertainty" in s for s in steps)


# ---------------------------------------------------------------------------
# Coastal proxy wiring
# ---------------------------------------------------------------------------


class TestDartConstrainedCoastal:
    def test_with_coastal_proxies(self):
        db = _build_test_db_with_coastal(
            station_ids=["dart_01", "dart_02"],
            coastal_site_ids=["site_A", "site_B"],
        )
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            bootstrap_config=BootstrapConfig(n_iterations=20, seed=42),
            coastal_site_ids=["site_A", "site_B"],
        )
        assert len(result.coastal_proxies) == 2
        for proxy in result.coastal_proxies:
            assert proxy.amplitude_proxy_p10_m <= proxy.amplitude_proxy_p50_m
            assert proxy.amplitude_proxy_p50_m <= proxy.amplitude_proxy_p90_m

    def test_coastal_no_bootstrap(self):
        """No bootstrap + coastal sites -> P10=P50=P90=point estimate."""
        db = _build_test_db_with_coastal(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            coastal_site_ids=["site_A"],
        )
        assert len(result.coastal_proxies) == 1
        p = result.coastal_proxies[0]
        # Single inversion -> equal percentiles
        assert p.amplitude_proxy_p10_m == p.amplitude_proxy_p50_m
        assert p.amplitude_proxy_p50_m == p.amplitude_proxy_p90_m

    def test_no_coastal_sites(self):
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert result.coastal_proxies == []

    def test_decision_trace_coastal_entry(self):
        db = _build_test_db_with_coastal(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            coastal_site_ids=["site_A"],
        )
        steps = [s.step for s in result.decision_trace]
        assert any("Coastal amplitude proxies" in s for s in steps)

    def test_event_origin_required_for_coastal(self):
        db = _build_test_db_with_coastal(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        with pytest.raises(ValueError, match="event_origin_utc required"):
            agent.run_dart_constrained(
                station_waveforms=waveforms,
                epicenter_lat=0.0,
                epicenter_lon=0.0,
                processing_time=FIXED_TIME,
                coastal_site_ids=["site_A"],
                # event_origin_utc intentionally omitted
            )

    def test_naive_event_origin_rejected(self):
        """Naive (tz-unaware) event_origin_utc is rejected early."""
        db = _build_test_db_with_coastal(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        naive_dt = datetime(2024, 1, 15, 12, 0, 0)  # no tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            agent.run_dart_constrained(
                station_waveforms=waveforms,
                epicenter_lat=0.0,
                epicenter_lon=0.0,
                processing_time=FIXED_TIME,
                coastal_site_ids=["site_A"],
                event_origin_utc=naive_dt,
            )

    def test_coastal_partial_missing_site(self):
        """Requesting existing + missing coastal sites skips the missing one."""
        db = _build_test_db_with_coastal(
            station_ids=["dart_01", "dart_02"],
            coastal_site_ids=["site_A"],
        )
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            coastal_site_ids=["site_A", "site_NONEXISTENT"],
        )
        # Only site_A should appear in proxies
        assert len(result.coastal_proxies) == 1
        assert result.coastal_proxies[0].site_id == "site_A"
        # Missing site recorded in limiting assumptions
        assert any(
            "site_NONEXISTENT" in a
            for a in result.limiting_assumptions
        )

    def test_extra_exclusion_reasons_stripped(self):
        """Extra keys in exclusion_reasons not in dart_stations_excluded are stripped."""
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        # Pass extra key "dart_99" in exclusion_reasons but not in excluded list
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            exclusion_reasons={"dart_99": "stale reason"},
        )
        # Should succeed (not raise ValidationError) - extra key stripped
        assert "dart_99" not in result.exclusion_reasons


# ---------------------------------------------------------------------------
# Ensemble spread from coastal amplitude
# ---------------------------------------------------------------------------


class TestEnsembleSpreadWiring:
    def test_no_coastal_sites_spread_high(self):
        """No coastal sites -> ensemble_spread=HIGH regardless of bootstrap."""
        db = _build_test_db(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            bootstrap_config=BootstrapConfig(n_iterations=20, seed=42),
            # No coastal_site_ids
        )
        assert result.ensemble_spread == EnsembleSpread.HIGH
        # Verify limiting assumption explains why spread defaulted to HIGH
        assert any(
            "ensemble spread" in a.lower() and "no coastal" in a.lower()
            for a in result.limiting_assumptions
        )

    def test_schema_round_trip_with_bootstrap_and_coastal(self):
        """Full round-trip validation with bootstrap + coastal data."""
        db = _build_test_db_with_coastal(station_ids=["dart_01", "dart_02"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {
            "dart_01": rng.standard_normal(60),
            "dart_02": rng.standard_normal(60),
        }
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
            event_origin_utc=EVENT_ORIGIN,
            bootstrap_config=BootstrapConfig(n_iterations=20, seed=42),
            coastal_site_ids=["site_A"],
        )
        validated = ScenarioAssessment.model_validate(result.model_dump())
        assert validated.ensemble_spread == result.ensemble_spread
        assert len(validated.coastal_proxies) == len(result.coastal_proxies)


# ---------------------------------------------------------------------------
# Bilateral rupture + seismic-only unchanged
# ---------------------------------------------------------------------------


class TestBilateralAndSeismicUnchanged:
    def test_bilateral_always_false_dart(self):
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert result.bilateral_rupture_evaluated is False

    def test_bilateral_always_false_seismic(self):
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert result.bilateral_rupture_evaluated is False

    def test_limiting_assumptions_bilateral(self):
        """DART-constrained includes 'Bilateral rupture not evaluated'."""
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        result = agent.run_dart_constrained(
            station_waveforms=waveforms,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            processing_time=FIXED_TIME,
        )
        assert any("bilateral" in a.lower() for a in result.limiting_assumptions)

    def test_seismic_only_unchanged(self):
        """Seismic-only still produces HIGH spread, False bilateral, empty coastal."""
        db = _build_test_db()
        agent = ScenarioAgent(database=db)
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert result.ensemble_spread == EnsembleSpread.HIGH
        assert result.bilateral_rupture_evaluated is False
        assert result.coastal_proxies == []


# ---------------------------------------------------------------------------
# Coverage gap tests for scenario_agent.py
# ---------------------------------------------------------------------------


class TestSetDatabase:
    """set_database() stores value."""

    def test_set_database_after_init(self):
        agent = ScenarioAgent()
        db = _build_test_db()
        agent.set_database(db)
        # Should work without raising ValueError
        result = agent.run_seismic_only(
            magnitude=8.0,
            epicenter_lat=0.0,
            epicenter_lon=0.0,
            region="Pacific",
            processing_time=FIXED_TIME,
        )
        assert result.constraint_stage == ConstraintStage.SEISMIC_ONLY


class TestNoUnitSourcesRaises:
    """No unit sources within range raises ValueError."""

    def test_no_sources_near_epicenter(self):
        # Build DB with sources at (0, 0) but search at (80, 170) - far away
        db = _build_test_db(station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        rng = np.random.default_rng(42)
        waveforms = {"dart_01": rng.standard_normal(60)}
        with pytest.raises(ValueError, match="No unit sources"):
            agent.run_dart_constrained(
                station_waveforms=waveforms,
                epicenter_lat=80.0,
                epicenter_lon=170.0,
                processing_time=FIXED_TIME,
            )


class TestShortWaveformExclusion:
    """Short waveforms excluded with appropriate reason."""

    def test_short_waveform_excluded_and_all_short_raises(self):
        """All stations have short waveforms -> ValueError."""
        db = _build_test_db(n_timepoints=100, station_ids=["dart_01"])
        agent = ScenarioAgent(database=db)
        # Provide waveform shorter than n_timepoints (100)
        short_waveform = np.random.default_rng(42).standard_normal(10)
        with pytest.raises(ValueError, match="waveform length check"):
            agent.run_dart_constrained(
                station_waveforms={"dart_01": short_waveform},
                epicenter_lat=0.0,
                epicenter_lon=0.0,
                processing_time=FIXED_TIME,
            )
