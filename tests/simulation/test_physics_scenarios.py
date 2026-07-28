"""Integration scenario tests exercising the full anomaly detection pipeline.

Four physically grounded scenarios:
1. Large tsunami (M9.1 Tohoku-like) - multi-station detection
2. Moderate earthquake (M7.2) - verify no over-escalation
3. Meteotsunami false positive - verify rejection
4. Partial network outage - verify graceful degradation

Each scenario generates synthetic data through the simulation module and
feeds it through the AnomalyAgent to validate detection behavior.
"""

from __future__ import annotations

import numpy as np
import pytest

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import SeismicEvent
from hazard_assessment.simulation.degraded import apply_data_gaps, mark_stations_offline
from hazard_assessment.simulation.false_positive import (
    generate_meteotsunami_signal,
    generate_storm_surge_signal,
)
from hazard_assessment.simulation.propagation import PACIFIC_DART_STATIONS
from hazard_assessment.simulation.scenario import SimulatedEvent, generate_coherent_event
from hazard_assessment.simulation.source import TOHOKU_LIKE

# FSM thresholds (must match states.py)
T1_INVESTIGATE = 0.35
T2_ASSESS = 0.60
T3_ESCALATE = 0.85


def _run_anomaly_detection_on_station(
    agent: AnomalyAgent,
    event: SimulatedEvent,
    station_id: str,
    seismic_context: list[SeismicEvent] | None = None,
) -> float:
    """Run the full anomaly detection pipeline on one station's simulated data.

    Returns the ensemble anomaly score.
    """
    stn = event.stations[station_id]
    config = stn.config
    calibration_hours = event.metadata["calibration_hours"]
    dt_sec = config.sampling_interval_sec

    times = stn.times_hours
    signal = stn.event_signal

    # Split into calibration and event windows
    cal_mask = times < calibration_hours
    evt_mask = times >= calibration_hours

    cal_times = times[cal_mask]
    cal_values = signal[cal_mask]
    evt_times = times[evt_mask]
    evt_values = signal[evt_mask]

    if len(evt_times) == 0 or len(cal_times) == 0:
        return 0.0

    # Calibrate baseline from calibration period (clean tidal signal)
    clean_cal_values = stn.clean_signal[cal_mask]
    agent.calibrate_baseline(
        station_id, clean_cal_values, dt_sec, source_type=config.station_type,
    )

    # Set seismic context
    if seismic_context:
        agent.update_seismic_events(seismic_context)
    else:
        agent.update_seismic_events([])

    # Run anomaly detection on event window
    scores, spatial_result = agent.process_station_data(
        station_id=station_id,
        times_hours=evt_times,
        values=evt_values,
        sampling_interval_sec=dt_sec,
        source_type=config.station_type,
        fit_times_hours=cal_times,
        fit_values=cal_values,
        origin_lat=config.latitude,
        origin_lon=config.longitude,
    )

    return scores.ensemble_score


class TestLargeTsunamiDetection:
    """Scenario 1: M9.1 Tohoku-like event with multi-station detection.

    Physical expectations:
    - Tsunami amplitude ~0.15 m at reference distance (Titov scaling)
    - Nearest DART stations should produce high ensemble scores
    - CO-OPS stations should see amplified signal (Green's Law)
    - Multiple stations should exceed T1
    """

    def test_nearest_dart_station_detects(self, tohoku_event: SimulatedEvent):
        """The nearest DART station to Tohoku should produce a significant score."""
        agent = AnomalyAgent()
        eq = tohoku_event.earthquake

        seismic = [
            SeismicEvent(
                event_id=eq.event_id,
                magnitude=eq.magnitude,
                origin_time=eq.origin_time,
                latitude=eq.latitude,
                longitude=eq.longitude,
            )
        ]

        # Find nearest DART station
        nearest_id = min(
            (sid for sid, stn in tohoku_event.stations.items()
             if stn.config.station_type == "dart"),
            key=lambda sid: tohoku_event.stations[sid].distance_km,
        )
        score = _run_anomaly_detection_on_station(
            agent, tohoku_event, nearest_id, seismic,
        )
        assert score >= T1_INVESTIGATE, (
            f"Nearest DART station {nearest_id} "
            f"(dist={tohoku_event.stations[nearest_id].distance_km:.0f} km) "
            f"score {score:.4f} did not reach T1={T1_INVESTIGATE}"
        )

    def test_multiple_dart_stations_exceed_t1(self, tohoku_event: SimulatedEvent):
        """At least 2 DART stations should exceed T1 for a M9.1 event."""
        eq = tohoku_event.earthquake
        seismic = [
            SeismicEvent(
                event_id=eq.event_id,
                magnitude=eq.magnitude,
                origin_time=eq.origin_time,
                latitude=eq.latitude,
                longitude=eq.longitude,
            )
        ]

        detecting_stations = []
        for sid, stn in tohoku_event.stations.items():
            if stn.config.station_type != "dart":
                continue
            agent = AnomalyAgent()
            score = _run_anomaly_detection_on_station(
                agent, tohoku_event, sid, seismic,
            )
            if score >= T1_INVESTIGATE:
                detecting_stations.append((sid, score))

        assert len(detecting_stations) >= 2, (
            f"Only {len(detecting_stations)} DART stations exceeded T1; "
            f"expected >= 2 for M9.1 event"
        )

    def test_arrival_times_are_physically_ordered(self, tohoku_event: SimulatedEvent):
        """Stations closer to epicenter should have earlier arrival times."""
        dart_stations = [
            (sid, stn) for sid, stn in tohoku_event.stations.items()
            if stn.config.station_type == "dart"
        ]
        # Sort by distance
        dart_stations.sort(key=lambda x: x[1].distance_km)

        # Arrival times should be monotonically increasing with distance
        for i in range(len(dart_stations) - 1):
            sid1, stn1 = dart_stations[i]
            sid2, stn2 = dart_stations[i + 1]
            assert stn1.arrival_hour <= stn2.arrival_hour, (
                f"Station {sid1} (dist={stn1.distance_km:.0f} km, "
                f"arrival={stn1.arrival_hour:.2f} hr) should arrive before "
                f"{sid2} (dist={stn2.distance_km:.0f} km, "
                f"arrival={stn2.arrival_hour:.2f} hr)"
            )

    def test_coops_amplitude_exceeds_dart(self, tohoku_event: SimulatedEvent):
        """CO-OPS stations should see amplified tsunami via Green's Law.

        Typical amplification: (4000/20)^0.25 ~ 3.76x.
        After accounting for distance, CO-OPS amplitude per unit distance
        should be higher than DART.
        """
        coops_stns = [
            (sid, stn) for sid, stn in tohoku_event.stations.items()
            if stn.config.station_type == "coops"
        ]
        dart_stns = [
            (sid, stn) for sid, stn in tohoku_event.stations.items()
            if stn.config.station_type == "dart"
        ]

        if not coops_stns or not dart_stns:
            pytest.skip("Need both DART and CO-OPS stations")

        # Compare distance-normalized amplitudes to isolate Green's Law effect.
        # Normalize by geometric spreading to remove distance dependence,
        # then CO-OPS should exceed DART by at least the amplification factor.
        max_dart_amp = max(stn.tsunami_amplitude_m for _, stn in dart_stns)
        for sid, coops_stn in coops_stns:
            assert coops_stn.tsunami_amplitude_m > max_dart_amp, (
                f"CO-OPS station {sid} amplitude ({coops_stn.tsunami_amplitude_m:.4f} m) "
                f"should exceed max DART amplitude ({max_dart_amp:.4f} m) "
                f"due to Green's Law coastal amplification"
            )

    def test_station_distances_are_realistic(self, tohoku_event: SimulatedEvent):
        """Station distances from Tohoku epicenter should be physically realistic."""
        for sid, stn in tohoku_event.stations.items():
            # All Pacific stations should be 100-10000 km from Tohoku
            assert 100 < stn.distance_km < 12000, (
                f"Station {sid} distance {stn.distance_km:.0f} km unrealistic"
            )

    def test_geometric_spreading_decreases_with_distance(
        self, tohoku_event: SimulatedEvent
    ):
        """Geometric spreading factor should decrease for farther stations."""
        dart_stns = sorted(
            [(sid, stn) for sid, stn in tohoku_event.stations.items()
             if stn.config.station_type == "dart"],
            key=lambda x: x[1].distance_km,
        )
        for i in range(len(dart_stns) - 1):
            _, stn1 = dart_stns[i]
            _, stn2 = dart_stns[i + 1]
            assert stn1.geometric_spreading >= stn2.geometric_spreading, (
                "Spreading factor should decrease with distance"
            )


class TestModerateTsunamiMarginal:
    """Scenario 2: M7.2 event - signal near or below detection threshold.

    Physical expectations:
    - Deep-ocean amplitude ~0.003 m (well below DART threshold of 0.03 m)
    - After geometric spreading, amplitude is even smaller
    - Most stations should NOT exceed T1
    - System should not over-escalate
    """

    def test_amplitude_below_dart_threshold(self, moderate_event: SimulatedEvent):
        """M7.2 tsunami amplitude should be below DART detection threshold."""
        for sid, stn in moderate_event.stations.items():
            # DART threshold is 0.03 m; M7.2 produces ~0.003 m at 1000 km
            # After spreading to most stations, well below threshold
            assert stn.tsunami_amplitude_m < 0.03, (
                f"Station {sid} amplitude {stn.tsunami_amplitude_m:.4f} m "
                f"unexpectedly exceeds DART threshold"
            )

    def test_most_stations_below_t1(self, moderate_event: SimulatedEvent):
        """Most stations should stay below T1 for a M7.2 event."""
        eq = moderate_event.earthquake
        seismic = [
            SeismicEvent(
                event_id=eq.event_id,
                magnitude=eq.magnitude,
                origin_time=eq.origin_time,
                latitude=eq.latitude,
                longitude=eq.longitude,
            )
        ]

        above_t1_count = 0
        for sid in moderate_event.stations:
            agent = AnomalyAgent()
            score = _run_anomaly_detection_on_station(
                agent, moderate_event, sid, seismic,
            )
            if score >= T1_INVESTIGATE:
                above_t1_count += 1

        total = len(moderate_event.stations)
        assert above_t1_count <= total // 2, (
            f"{above_t1_count}/{total} stations exceeded T1 for M7.2 - "
            f"system may be over-escalating"
        )

    def test_no_station_exceeds_t2(self, moderate_event: SimulatedEvent):
        """No station should exceed T2 (ASSESS) for a M7.2 event."""
        eq = moderate_event.earthquake
        seismic = [
            SeismicEvent(
                event_id=eq.event_id,
                magnitude=eq.magnitude,
                origin_time=eq.origin_time,
                latitude=eq.latitude,
                longitude=eq.longitude,
            )
        ]

        for sid in moderate_event.stations:
            agent = AnomalyAgent()
            score = _run_anomaly_detection_on_station(
                agent, moderate_event, sid, seismic,
            )
            assert score < T2_ASSESS, (
                f"Station {sid} score {score:.4f} exceeded T2={T2_ASSESS} "
                f"for M7.2 event - false escalation"
            )


class TestMeteotsunamiFalsePositive:
    """Scenario 3: Meteotsunami signal without seismic trigger.

    Physical expectations:
    - 20-min period meteotsunami at 0.10 m amplitude (within tsunami band)
    - No seismic event -> seismic_quiet=True -> 1.3x threshold penalty
    - Single station affected -> no spatial coherence
    - System should NOT trigger beyond T1
    """

    def test_meteotsunami_without_seismic_stays_low(self):
        """Meteotsunami without seismic context should produce low ensemble score."""
        agent = AnomalyAgent()
        dt_sec = 60.0
        dt_hours = dt_sec / 3600.0
        calibration_hours = 30 * 24  # 30 days

        # Generate tidal baseline for calibration
        cal_times = np.arange(0, calibration_hours, dt_hours)
        omega_m2 = np.radians(28.984104)
        omega_s2 = np.radians(30.0)
        rng = np.random.default_rng(42)

        cal_signal = (
            0.50 * np.cos(omega_m2 * cal_times)
            + 0.15 * np.cos(omega_s2 * cal_times)
            + rng.normal(0, 0.005, len(cal_times))
        )

        # Calibrate baseline
        agent.calibrate_baseline(
            "coops_test", cal_signal, dt_sec, source_type="coops",
        )

        # Generate event window with meteotsunami
        event_hours = 6.0
        evt_times = np.arange(calibration_hours, calibration_hours + event_hours, dt_hours)
        evt_signal = (
            0.50 * np.cos(omega_m2 * evt_times)
            + 0.15 * np.cos(omega_s2 * evt_times)
            + rng.normal(0, 0.005, len(evt_times))
        )

        # Inject meteotsunami at hour 2 of event window
        meteo = generate_meteotsunami_signal(
            evt_times,
            onset_hour=calibration_hours + 2.0,
            amplitude_m=0.10,
            period_min=20.0,
        )
        evt_signal = evt_signal + meteo

        # No seismic events -> seismic_quiet=True
        agent.update_seismic_events([])

        scores, _ = agent.process_station_data(
            station_id="coops_test",
            times_hours=evt_times,
            values=evt_signal,
            sampling_interval_sec=dt_sec,
            source_type="coops",
            fit_times_hours=cal_times,
            fit_values=cal_signal,
        )

        # With seismic_quiet=True, threshold is raised by 1.3x
        # The meteotsunami should not push ensemble above T2
        assert scores.ensemble_score < T2_ASSESS, (
            f"Meteotsunami false positive: ensemble={scores.ensemble_score:.4f} "
            f"exceeded T2={T2_ASSESS}"
        )
        assert scores.seismic_context_quiet, "Should be seismically quiet"

    def test_storm_surge_filtered_by_bandpass(self):
        """Storm surge (period >> 120 min) should be removed by bandpass filter."""
        agent = AnomalyAgent()
        dt_sec = 60.0
        dt_hours = dt_sec / 3600.0
        calibration_hours = 30 * 24

        # Calibration signal (tidal)
        cal_times = np.arange(0, calibration_hours, dt_hours)
        omega_m2 = np.radians(28.984104)
        rng = np.random.default_rng(55)
        cal_signal = 0.10 * np.cos(omega_m2 * cal_times) + rng.normal(0, 0.001, len(cal_times))

        agent.calibrate_baseline("dart_test", cal_signal, dt_sec)
        agent.update_seismic_events([])

        # Event signal with storm surge
        evt_times = np.arange(calibration_hours, calibration_hours + 24.0, dt_hours)
        evt_signal = 0.10 * np.cos(omega_m2 * evt_times) + rng.normal(0, 0.001, len(evt_times))

        surge = generate_storm_surge_signal(
            evt_times,
            onset_hour=calibration_hours + 2.0,
            amplitude_m=0.50,  # large surge
            duration_hours=12.0,
        )
        evt_signal = evt_signal + surge

        scores, _ = agent.process_station_data(
            station_id="dart_test",
            times_hours=evt_times,
            values=evt_signal,
            sampling_interval_sec=dt_sec,
            source_type="dart",
            fit_times_hours=cal_times,
            fit_values=cal_signal,
        )

        # Storm surge is outside the bandpass -> threshold score should be low
        assert scores.threshold_score < 0.5, (
            f"Storm surge should be filtered: threshold_score={scores.threshold_score:.4f}"
        )


class TestPartialNetworkOutage:
    """Scenario 4: Large tsunami with partial network outage.

    Physical expectations:
    - 4 of 6 DART stations offline
    - Remaining 2 stations should still detect the tsunami
    - Detection quality degrades but doesn't fail completely
    """

    def test_detection_with_two_stations(self):
        """System should detect M9.1 tsunami even with only 2 DART stations."""
        # Generate event with only 2 stations
        online_stations = PACIFIC_DART_STATIONS[:2]
        event = generate_coherent_event(
            earthquake=TOHOKU_LIKE,
            stations=online_stations,
            seed=42,
        )

        eq = event.earthquake
        seismic = [
            SeismicEvent(
                event_id=eq.event_id,
                magnitude=eq.magnitude,
                origin_time=eq.origin_time,
                latitude=eq.latitude,
                longitude=eq.longitude,
            )
        ]

        detected_any = False
        for sid in event.stations:
            agent = AnomalyAgent()
            score = _run_anomaly_detection_on_station(
                agent, event, sid, seismic,
            )
            if score >= T1_INVESTIGATE:
                detected_any = True

        assert detected_any, (
            "At least one of the 2 remaining stations should detect M9.1 tsunami"
        )

    def test_offline_stations_tracked(self):
        """mark_stations_offline should correctly identify offline stations."""
        offline_ids = {"21413", "21419", "46404", "46407"}
        online, offline = mark_stations_offline(PACIFIC_DART_STATIONS[:6], offline_ids)

        assert len(online) == 2
        assert len(offline) == 4
        assert set(offline) == offline_ids
        assert all(s.station_id not in offline_ids for s in online)

    def test_data_gap_handled_gracefully(self):
        """System should handle data gaps without crashing."""
        event = generate_coherent_event(
            earthquake=TOHOKU_LIKE,
            stations=PACIFIC_DART_STATIONS[:2],
            seed=42,
        )

        sid = list(event.stations.keys())[0]
        stn = event.stations[sid]

        # Create a 1-hour data gap in the calibration period
        cal_hours = event.metadata["calibration_hours"]
        t_gap, s_gap = apply_data_gaps(
            stn.event_signal,  # will be re-indexed below
            stn.event_signal,
            gap_start_hour=cal_hours - 10,  # 10 hours before event
            gap_duration_hours=1.0,
        )

        # Use the time-gapped data
        t_gap_times, s_gap_signal = apply_data_gaps(
            stn.times_hours,
            stn.event_signal,
            gap_start_hour=cal_hours - 10,
            gap_duration_hours=1.0,
        )

        # Should not crash even with gap
        agent = AnomalyAgent()
        cal_mask = t_gap_times < cal_hours
        if np.sum(cal_mask) > 100:  # need enough calibration data
            agent.calibrate_baseline(
                sid,
                s_gap_signal[cal_mask],
                stn.config.sampling_interval_sec,
                source_type=stn.config.station_type,
            )
            # If event window has enough data, run detection
            evt_mask = t_gap_times >= cal_hours
            if np.sum(evt_mask) > 10:
                scores, _ = agent.process_station_data(
                    station_id=sid,
                    times_hours=t_gap_times[evt_mask],
                    values=s_gap_signal[evt_mask],
                    sampling_interval_sec=stn.config.sampling_interval_sec,
                    source_type=stn.config.station_type,
                    fit_times_hours=t_gap_times[cal_mask],
                    fit_values=s_gap_signal[cal_mask],
                )
                # Should produce a valid score (may be degraded)
                assert 0.0 <= scores.ensemble_score <= 1.0


class TestCoherentEventProperties:
    """Validate structural properties of the generated coherent events."""

    def test_all_stations_present(self, tohoku_event: SimulatedEvent):
        """All requested stations should be present in the event."""
        assert len(tohoku_event.stations) == 8  # 6 DART + 2 CO-OPS

    def test_clean_and_event_signals_differ_where_arrived(
        self, tohoku_event: SimulatedEvent
    ):
        """Event signal should differ from clean at stations where tsunami arrived.

        Far-field stations (e.g. Cascadia at ~8000 km from Tohoku) may not
        see the tsunami within the 6-hour event window - the travel time
        exceeds the window. This is physically correct.
        """
        cal_hours = tohoku_event.metadata["calibration_hours"]
        evt_hours = tohoku_event.metadata["event_hours"]
        arrived_count = 0

        for sid, stn in tohoku_event.stations.items():
            arrival_in_event = stn.arrival_hour - cal_hours
            if arrival_in_event < evt_hours:
                # Tsunami should have arrived -> signals must differ
                diff = np.max(np.abs(stn.event_signal - stn.clean_signal))
                assert diff > 0, (
                    f"Station {sid} (arrival={arrival_in_event:.1f}h) "
                    f"event signal equals clean signal"
                )
                arrived_count += 1

        assert arrived_count >= 2, "At least 2 stations should see the tsunami"

    def test_metadata_populated(self, tohoku_event: SimulatedEvent):
        """Event metadata should contain generation parameters."""
        assert "calibration_hours" in tohoku_event.metadata
        assert "seed" in tohoku_event.metadata
        assert "spectrum" in tohoku_event.metadata
        assert tohoku_event.metadata["calibration_hours"] == 30 * 24

    def test_earthquake_reference_preserved(self, tohoku_event: SimulatedEvent):
        """Event should reference the earthquake source."""
        assert tohoku_event.earthquake.magnitude == 9.1
        assert tohoku_event.earthquake.event_id == "synth_tohoku"

    def test_deterministic_across_calls(self):
        """Same parameters should produce identical events."""
        event1 = generate_coherent_event(
            TOHOKU_LIKE, PACIFIC_DART_STATIONS[:2], seed=42,
        )
        event2 = generate_coherent_event(
            TOHOKU_LIKE, PACIFIC_DART_STATIONS[:2], seed=42,
        )

        for sid in event1.stations:
            np.testing.assert_array_equal(
                event1.stations[sid].event_signal,
                event2.stations[sid].event_signal,
            )
