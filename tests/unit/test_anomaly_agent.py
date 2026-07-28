"""Unit tests for AnomalyAgent (integration).

Tests the full AnomalyAgent flow: station data -> AnomalyAssessment envelope
with all required fields populated and schema-validated.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import (
    AnomalyScoreComponents,
    IsolationForestModel,
    SeismicEvent,
    StationArrival,
)
from hazard_assessment.schemas.anomaly import AnomalyAssessment
from hazard_assessment.schemas.envelope import StepResult

_T0 = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)


def _generate_tidal_signal(
    n_hours: int = 30 * 24,
    dt_hours: float = 1.0 / 60.0,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic tidal signal for testing."""
    import math

    times = np.arange(0, n_hours, dt_hours)
    omega_m2 = math.radians(28.984104)
    omega_s2 = math.radians(30.0)

    rng = np.random.default_rng(seed)
    signal = (
        1.0 * np.cos(omega_m2 * times)
        + 0.5 * np.cos(omega_s2 * times)
        + rng.normal(0, 0.001, len(times))
    )
    return times, signal


def _make_components(ensemble: float) -> AnomalyScoreComponents:
    """Minimal score components with a controlled ensemble score."""
    return AnomalyScoreComponents(
        threshold_score=0.0,
        wavelet_score=0.0,
        bocpd_score=0.0,
        statistical_score=0.0,
        ml_score=None,
        spatial_coherence_score=0.0,
        seismic_context_quiet=True,
        ensemble_score=ensemble,
    )


class TestAnomalyAgentManifest:
    """Tests for AnomalyAgent manifest."""

    def test_manifest_name(self) -> None:
        agent = AnomalyAgent()
        assert agent.name == "anomaly_agent"

    def test_manifest_version(self) -> None:
        agent = AnomalyAgent()
        assert agent.manifest.version == "1.0.0"

    def test_manifest_capabilities(self) -> None:
        agent = AnomalyAgent()
        caps = agent.manifest.capabilities
        assert len(caps) == 4


class TestAnomalyAgentProcessing:
    """Tests for AnomalyAgent.process_station_data."""

    def test_process_quiet_signal(self) -> None:
        """Quiet signal should produce low anomaly score."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, spatial = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assert 0.0 <= scores.ensemble_score < 0.60  # quiet signal: below ASSESS threshold
        assert spatial is None  # no other arrivals provided

    def test_process_with_seismic_context(self) -> None:
        """Seismic events should affect threshold scoring."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        # No seismic events -> quiet
        scores_quiet, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )
        assert scores_quiet.seismic_context_quiet is True

        # Add recent large event
        agent.update_seismic_events([
            SeismicEvent(
                event_id="e1", magnitude=7.5,
                origin_time=_T0 - timedelta(minutes=30),
                latitude=0.0, longitude=0.0,
            ),
        ])

        scores_active, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )
        assert scores_active.seismic_context_quiet is False

    def test_process_with_baseline_calibration(self) -> None:
        """Calibrated baseline should affect wavelet scoring."""
        agent = AnomalyAgent()

        # Calibrate baseline from quiet data
        rng = np.random.default_rng(42)
        baseline_signal = rng.normal(0, 0.001, 1024)
        baseline_energy = agent.calibrate_baseline("21413", baseline_signal, 60.0)
        assert baseline_energy > 0

        times, signal = _generate_tidal_signal(n_hours=30 * 24)
        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )
        assert 0.0 <= scores.wavelet_score <= 1.0

    def test_process_with_spatial_coherence(self) -> None:
        """Should compute spatial coherence when other arrivals given."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        other = [
            StationArrival(
                station_id="21415",
                arrival_time=_T0 + timedelta(minutes=30),
                latitude=5.0, longitude=170.0,
            ),
            StationArrival(
                station_id="21416",
                arrival_time=_T0 + timedelta(minutes=60),
                latitude=10.0, longitude=175.0,
            ),
        ]

        scores, spatial = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            other_arrivals=other,
            origin_lat=0.0,
            origin_lon=165.0,
            processing_time=_T0,
        )

        assert spatial is not None
        assert len(spatial.confirmations) == 2
        assert 0.0 <= scores.spatial_coherence_score <= 1.0


class TestAnomalyAgentAssessment:
    """Tests for AnomalyAgent.build_assessment."""

    def test_assessment_schema_valid(self) -> None:
        """Built assessment should pass schema validation."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )

        assert isinstance(assessment, AnomalyAssessment)
        assert assessment.producer == "anomaly_agent"
        assert 0.0 <= assessment.anomaly_score <= 1.0
        assert assessment.score_components.threshold >= 0.0
        assert assessment.score_components.statistical >= 0.0

    def test_assessment_decision_trace(self) -> None:
        """Assessment should include decision trace steps."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )

        assert len(assessment.decision_trace) >= 5
        step_names = [s.step for s in assessment.decision_trace]
        assert "Detide and bandpass filter" in step_names
        assert "Threshold detection" in step_names
        assert "Wavelet energy analysis (db4)" in step_names
        assert "BOCPD changepoint detection" in step_names
        assert "Ensemble fusion" in step_names

    def test_assessment_reasoning_trace(self) -> None:
        """Reasoning trace should contain all component scores."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )

        assert "threshold_score=" in assessment.reasoning_trace
        assert "wavelet_score=" in assessment.reasoning_trace
        assert "bocpd_score=" in assessment.reasoning_trace
        assert "ensemble_score=" in assessment.reasoning_trace

    def test_assessment_no_ml(self) -> None:
        """Without ML model, score_components.ml should be None."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )

        assert assessment.score_components.ml is None
        assert "ML unavailable" in assessment.reasoning_trace

    def test_triggering_vs_scored_stations(self) -> None:
        """scored_stations always lists the scored windows;
        triggering_stations only lists them when the ensemble score reaches
        the configured T1 threshold, inclusive (matches FSM semantics)."""
        agent = AnomalyAgent()

        below = agent.build_assessment(
            station_ids=["21413"],
            scores=_make_components(ensemble=0.34),
            processing_time=_T0,
        )
        assert below.scored_stations == ["21413"]
        assert below.triggering_stations == []

        at_boundary = agent.build_assessment(
            station_ids=["21413"],
            scores=_make_components(ensemble=0.35),
            processing_time=_T0,
        )
        assert at_boundary.scored_stations == ["21413"]
        assert at_boundary.triggering_stations == ["21413"]

        fusion = next(
            s for s in at_boundary.decision_trace if s.step == "Ensemble fusion"
        )
        assert fusion.result == StepResult.PASS
        assert "T1=0.35 (inclusive)" in fusion.evidence

    def test_meteotsunami_not_evaluated(self) -> None:
        """The meteotsunami discriminator is not implemented; the
        envelope states that with None instead of a 0.0 placeholder."""
        agent = AnomalyAgent()
        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=_make_components(ensemble=0.1),
            processing_time=_T0,
        )
        assert assessment.meteotsunami_score is None

    def test_detide_trace_reports_fit_provenance(self) -> None:
        """The detide decision step reports what the harmonic fit actually
        used instead of a fixed 30-day claim."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)
        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )
        assessment = agent.build_assessment(
            station_ids=["21413"], scores=scores, processing_time=_T0,
        )
        detide = next(
            s for s in assessment.decision_trace
            if s.step == "Detide and bandpass filter"
        )
        assert "event window" in detide.evidence
        assert "43200 samples" in detide.evidence
        assert "720.0 h" in detide.evidence
        assert "30-day" not in detide.evidence

        # A dedicated fit series is reported as such.
        scores_fit, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            fit_times_hours=times,
            fit_values=signal,
            processing_time=_T0,
        )
        with_fit = agent.build_assessment(
            station_ids=["21413"], scores=scores_fit, processing_time=_T0,
        )
        detide_fit = next(
            s for s in with_fit.decision_trace
            if s.step == "Detide and bandpass filter"
        )
        assert "separate calibration series" in detide_fit.evidence

        # Components built without provenance metadata state that honestly.
        bare = agent.build_assessment(
            station_ids=["21413"],
            scores=_make_components(ensemble=0.1),
            processing_time=_T0,
        )
        detide_bare = next(
            s for s in bare.decision_trace
            if s.step == "Detide and bandpass filter"
        )
        assert "fit provenance not recorded" in detide_bare.evidence

    def test_assessment_with_ml(self) -> None:
        """With ML model, score_components.ml should be set."""
        agent = AnomalyAgent()
        rng = np.random.default_rng(42)
        model = IsolationForestModel()
        model.fit(rng.normal(0, 0.01, (200, 4)))
        agent.set_iforest_model(model)

        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )

        assert assessment.score_components.ml is not None
        assert 0.0 <= assessment.score_components.ml <= 1.0

    def test_assessment_seismic_context(self) -> None:
        """Assessment should reflect seismic context."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        # Quiet (no events)
        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )
        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            processing_time=_T0,
        )
        assert assessment.seismic_quiet is True

    def test_assessment_offline_stations(self) -> None:
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            stations_offline=["21415", "21416"],
            coverage_note="2 stations offline in central Pacific",
            processing_time=_T0,
        )

        assert assessment.stations_offline == ["21415", "21416"]
        assert "2 stations offline" in assessment.coverage_note


class TestAnomalyAgentInputValidation:
    """Tests for input validation guards."""

    def test_zero_sampling_interval_raises(self) -> None:
        """Zero sampling interval must raise, not ZeroDivisionError."""
        agent = AnomalyAgent()
        with pytest.raises(ValueError, match="sampling_interval_sec must be positive"):
            agent.process_station_data(
                station_id="21413",
                times_hours=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0]),
                sampling_interval_sec=0.0,
            )

    def test_negative_sampling_interval_raises(self) -> None:
        """Negative sampling interval must raise."""
        agent = AnomalyAgent()
        with pytest.raises(ValueError, match="sampling_interval_sec must be positive"):
            agent.process_station_data(
                station_id="21413",
                times_hours=np.array([1.0, 2.0]),
                values=np.array([1.0, 2.0]),
                sampling_interval_sec=-1.0,
            )

    def test_nan_values_raise(self) -> None:
        """NaN in values must raise ValueError, not silently propagate."""
        agent = AnomalyAgent()
        with pytest.raises(ValueError, match="non-finite"):
            agent.process_station_data(
                station_id="21413",
                times_hours=np.array([1.0, 2.0, 3.0]),
                values=np.array([1.0, float("nan"), 3.0]),
                sampling_interval_sec=60.0,
            )


# ---------------------------------------------------------------------------
# Coverage gap tests for anomaly_agent.py
# ---------------------------------------------------------------------------


class TestSetBaselineEnergy:
    """set_baseline_energy() stores value."""

    def test_stores_baseline_energy(self) -> None:
        agent = AnomalyAgent()
        agent.set_baseline_energy("21413", 42.5)
        assert agent._baseline_energies[("dart", "21413")] == 42.5

    def test_overwrites_existing(self) -> None:
        agent = AnomalyAgent()
        agent.set_baseline_energy("21413", 42.5)
        agent.set_baseline_energy("21413", 99.0)
        assert agent._baseline_energies[("dart", "21413")] == 99.0

    def test_baselines_keyed_by_source(self) -> None:
        """Equal station identifiers from different sources must not share
        a baseline."""
        agent = AnomalyAgent()
        agent.set_baseline_energy("21413", 42.5, source_type="dart")
        agent.set_baseline_energy("21413", 7.0, source_type="coops")
        assert agent._baseline_energies[("dart", "21413")] == 42.5
        assert agent._baseline_energies[("coops", "21413")] == 7.0


class TestBuildAssessmentWithSpatialConfirmations:
    """build_assessment() with spatial_result and other_arrivals."""

    def test_spatial_confirmations_populated(self) -> None:
        from hazard_assessment.agents.anomaly_detection import (
            SpatialCoherenceResult,
            SpatialConfirmationDetail,
            StationArrival,
        )

        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)

        scores, _ = agent.process_station_data(
            station_id="21413",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        spatial_result = SpatialCoherenceResult(
            confirmed=True,
            confirming_stations=1,
            confirmations=[
                SpatialConfirmationDetail(
                    station_id="21414",
                    distance_km=500.0,
                    expected_travel_sec=3600.0,
                    actual_delta_sec=3650.0,
                    window_low_sec=3000.0,
                    window_high_sec=4200.0,
                    confirmed=True,
                ),
            ],
        )

        other_arrivals = [
            StationArrival(
                station_id="21414",
                arrival_time=_T0 + timedelta(hours=1),
                latitude=46.0,
                longitude=-130.0,
            ),
        ]

        assessment = agent.build_assessment(
            station_ids=["21413"],
            scores=scores,
            spatial_result=spatial_result,
            other_arrivals=other_arrivals,
            processing_time=_T0,
        )

        assert len(assessment.spatial_confirmations) == 1
        sc = assessment.spatial_confirmations[0]
        assert sc.station_id == "21414"
        assert sc.confirmed is True
        assert sc.delta_min is not None


class TestCheckRayleighSuspect:
    """check_rayleigh_suspect() - Rayleigh wave false-trigger detection."""

    def test_suspect_within_distance_and_timing(self) -> None:
        """Station within 3000 km with matching Rayleigh travel time -> True."""
        agent = AnomalyAgent()
        origin_time = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)
        agent.update_seismic_events([
            SeismicEvent(
                event_id="tohoku",
                magnitude=9.1,
                origin_time=origin_time,
                latitude=38.297,
                longitude=142.373,
            ),
        ])
        # Station 21418 at ~560 km -> Rayleigh travel ~156 sec at 3.6 km/s
        expected_sec = 560.0 / 3.6
        spike_time = origin_time + timedelta(seconds=expected_sec)
        assert agent.check_rayleigh_suspect(38.730, 148.800, spike_time) is True

    def test_not_suspect_beyond_max_distance(self) -> None:
        """Station beyond 3000 km -> False regardless of timing."""
        agent = AnomalyAgent()
        origin_time = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)
        agent.update_seismic_events([
            SeismicEvent(
                event_id="tohoku",
                magnitude=9.1,
                origin_time=origin_time,
                latitude=38.297,
                longitude=142.373,
            ),
        ])
        # Station 46411 at ~7480 km - well beyond 3000 km cutoff
        spike_time = origin_time + timedelta(seconds=7480.0 / 3.6)
        assert agent.check_rayleigh_suspect(39.347, -127.068, spike_time) is False

    def test_not_suspect_no_seismic_events(self) -> None:
        """No seismic events -> always False."""
        agent = AnomalyAgent()
        assert agent.check_rayleigh_suspect(38.730, 148.800, _T0) is False

    def test_not_suspect_timing_mismatch(self) -> None:
        """Spike much later than expected Rayleigh arrival -> False."""
        agent = AnomalyAgent()
        origin_time = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)
        agent.update_seismic_events([
            SeismicEvent(
                event_id="tohoku",
                magnitude=9.1,
                origin_time=origin_time,
                latitude=38.297,
                longitude=142.373,
            ),
        ])
        # Spike 1 hour after origin - way past Rayleigh window for ~560 km
        spike_time = origin_time + timedelta(hours=1)
        assert agent.check_rayleigh_suspect(38.730, 148.800, spike_time) is False

    def test_propagates_to_build_assessment(self) -> None:
        """The rayleigh_wave_suspect value on the scores object propagates
        into the AnomalyAssessment envelope. build_assessment must read the
        computed score rather than accept its own parameter, or the flag is
        silently dropped for every station."""
        agent = AnomalyAgent()
        times, signal = _generate_tidal_signal(n_hours=30 * 24)
        scores, _ = agent.process_station_data(
            station_id="21418",
            times_hours=times,
            values=signal,
            sampling_interval_sec=60.0,
            processing_time=_T0,
        )

        # Quiet tidal signal with no station coordinates: the check's
        # prerequisites are unavailable, so the flag stays None (not
        # evaluated) and the envelope reflects that.
        assert scores.rayleigh_wave_suspect is None
        assessment = agent.build_assessment(
            station_ids=["21418"], scores=scores, processing_time=_T0,
        )
        assert assessment.rayleigh_wave_suspect is None

        # When the detector flags a Rayleigh-wave suspect, the envelope must
        # carry it through.
        scores.rayleigh_wave_suspect = True
        assessment = agent.build_assessment(
            station_ids=["21418"], scores=scores, processing_time=_T0,
        )
        assert assessment.rayleigh_wave_suspect is True
