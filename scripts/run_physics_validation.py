#!/usr/bin/env python3
"""Physics-correct simulation validation for NOAA AI Workshop 2026 paper.

Runs 4 scenarios through the full anomaly detection pipeline and produces
a JSON report with citable detection metrics.

Output:
    results/physics_validation.json

Usage:
    python scripts/run_physics_validation.py

Prerequisites:
    pip install -e .
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import SeismicEvent
from hazard_assessment.simulation.degraded import mark_stations_offline
from hazard_assessment.simulation.false_positive import generate_meteotsunami_signal
from hazard_assessment.simulation.propagation import (
    PACIFIC_COOPS_STATIONS,
    PACIFIC_DART_STATIONS,
)
from hazard_assessment.simulation.scenario import SimulatedEvent, generate_coherent_event
from hazard_assessment.simulation.source import (
    ALEUTIAN_SCENARIO,
    MODERATE_ALEUTIAN,
    compute_characteristic_amplitude_m,
    compute_dominant_period_min,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FSM thresholds
T1 = 0.35
T2 = 0.60
T3 = 0.85


def _evaluate_station(
    event: SimulatedEvent,
    station_id: str,
    seismic_context: list[SeismicEvent] | None = None,
) -> dict:
    """Run anomaly detection on one station and return results."""
    stn = event.stations[station_id]
    config = stn.config
    cal_hours = event.metadata["calibration_hours"]
    dt_sec = config.sampling_interval_sec

    times = stn.times_hours
    signal = stn.event_signal
    clean = stn.clean_signal

    cal_mask = times < cal_hours
    evt_mask = times >= cal_hours

    agent = AnomalyAgent()

    # Calibrate from clean baseline
    agent.calibrate_baseline(
        station_id, clean[cal_mask], dt_sec, source_type=config.station_type,
    )

    if seismic_context:
        agent.update_seismic_events(seismic_context)

    # Use a processing_time consistent with the simulated earthquake origin
    # so the seismic-quiet 90-minute lookback window works correctly.
    # Default to origin + 30 min (early detection window).
    eq = event.earthquake
    processing_time = eq.origin_time + timedelta(minutes=30)

    scores, spatial_result = agent.process_station_data(
        station_id=station_id,
        times_hours=times[evt_mask],
        values=signal[evt_mask],
        sampling_interval_sec=dt_sec,
        source_type=config.station_type,
        fit_times_hours=times[cal_mask],
        fit_values=clean[cal_mask],
        origin_lat=config.latitude,
        origin_lon=config.longitude,
        processing_time=processing_time,
    )

    return {
        "station_id": station_id,
        "station_type": config.station_type,
        "distance_km": round(stn.distance_km, 1),
        "arrival_time_hours": round(stn.arrival_hour - cal_hours, 3),
        "tsunami_amplitude_m": round(stn.tsunami_amplitude_m, 6),
        "geometric_spreading": round(stn.geometric_spreading, 4),
        "ensemble_score": round(scores.ensemble_score, 4),
        "threshold_score": round(scores.threshold_score, 4),
        "wavelet_score": round(scores.wavelet_score, 4),
        "bocpd_score": round(scores.bocpd_score, 4),
        "statistical_score": round(scores.statistical_score, 4),
        "seismic_quiet": scores.seismic_context_quiet,
        "exceeds_t1": scores.ensemble_score >= T1,
        "exceeds_t2": scores.ensemble_score >= T2,
        "exceeds_t3": scores.ensemble_score >= T3,
    }


def run_scenario_1_large_tsunami() -> dict:
    """Scenario 1: M8.5 Aleutian with 6 DART + 3 CO-OPS stations."""
    logger.info("=" * 60)
    logger.info("Scenario 1: Large Tsunami (M8.5 Aleutian)")
    logger.info("=" * 60)

    stations = PACIFIC_DART_STATIONS[:6] + PACIFIC_COOPS_STATIONS[:3]
    event = generate_coherent_event(ALEUTIAN_SCENARIO, stations, seed=42)

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

    results = []
    for sid in event.stations:
        logger.info("Processing station %s ...", sid)
        r = _evaluate_station(event, sid, seismic)
        results.append(r)
        logger.info(
            "  %s: dist=%.0f km, arrival=%.1f hr, amp=%.4f m, "
            "ensemble=%.4f %s",
            sid, r["distance_km"], r["arrival_time_hours"],
            r["tsunami_amplitude_m"], r["ensemble_score"],
            "(DETECT)" if r["exceeds_t1"] else "",
        )

    detecting = [r for r in results if r["exceeds_t1"]]
    dart_detecting = [r for r in detecting if r["station_type"] == "dart"]

    return {
        "name": "Large Tsunami (M8.5 Aleutian)",
        "earthquake": {
            "magnitude": eq.magnitude,
            "latitude": eq.latitude,
            "longitude": eq.longitude,
            "depth_km": eq.depth_km,
            "region": eq.region,
        },
        "characteristic_amplitude_m": round(
            compute_characteristic_amplitude_m(eq.magnitude), 6
        ),
        "dominant_period_min": round(compute_dominant_period_min(eq.magnitude), 1),
        "stations": results,
        "summary": {
            "total_stations": len(results),
            "dart_stations": sum(1 for r in results if r["station_type"] == "dart"),
            "coops_stations": sum(1 for r in results if r["station_type"] == "coops"),
            "detecting_t1": len(detecting),
            "dart_detecting_t1": len(dart_detecting),
            "detecting_t2": sum(1 for r in results if r["exceeds_t2"]),
            "detecting_t3": sum(1 for r in results if r["exceeds_t3"]),
        },
    }


def run_scenario_2_moderate_earthquake() -> dict:
    """Scenario 2: M7.0 moderate earthquake - marginal/no tsunami."""
    logger.info("=" * 60)
    logger.info("Scenario 2: Moderate Earthquake (M7.0 Aleutian)")
    logger.info("=" * 60)

    stations = PACIFIC_DART_STATIONS[:4]
    event = generate_coherent_event(MODERATE_ALEUTIAN, stations, seed=123)

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

    results = []
    for sid in event.stations:
        logger.info("Processing station %s ...", sid)
        r = _evaluate_station(event, sid, seismic)
        results.append(r)
        logger.info(
            "  %s: amp=%.6f m, ensemble=%.4f %s",
            sid, r["tsunami_amplitude_m"], r["ensemble_score"],
            "(FALSE ALARM)" if r["exceeds_t2"] else "",
        )

    return {
        "name": "Moderate Earthquake (M7.0 Aleutian)",
        "earthquake": {
            "magnitude": eq.magnitude,
            "latitude": eq.latitude,
            "longitude": eq.longitude,
        },
        "characteristic_amplitude_m": round(
            compute_characteristic_amplitude_m(eq.magnitude), 6
        ),
        "stations": results,
        "summary": {
            "total_stations": len(results),
            "detecting_t1": sum(1 for r in results if r["exceeds_t1"]),
            "detecting_t2": sum(1 for r in results if r["exceeds_t2"]),
            "no_false_escalation": all(not r["exceeds_t2"] for r in results),
        },
    }


def run_scenario_3_meteotsunami() -> dict:
    """Scenario 3: Meteotsunami false positive at single CO-OPS station."""
    logger.info("=" * 60)
    logger.info("Scenario 3: Meteotsunami False Positive")
    logger.info("=" * 60)

    agent = AnomalyAgent()
    dt_sec = 60.0
    dt_hours = dt_sec / 3600.0
    cal_hours = 30 * 24

    # Generate tidal calibration
    cal_times = np.arange(0, cal_hours, dt_hours)
    omega_m2 = np.radians(28.984104)
    omega_s2 = np.radians(30.0)
    rng = np.random.default_rng(42)
    cal_signal = (
        0.50 * np.cos(omega_m2 * cal_times)
        + 0.15 * np.cos(omega_s2 * cal_times)
        + rng.normal(0, 0.005, len(cal_times))
    )

    agent.calibrate_baseline(
        "coops_honolulu", cal_signal, dt_sec, source_type="coops",
    )
    agent.update_seismic_events([])  # no seismic events

    # Event window with meteotsunami
    evt_times = np.arange(cal_hours, cal_hours + 6.0, dt_hours)
    evt_signal = (
        0.50 * np.cos(omega_m2 * evt_times)
        + 0.15 * np.cos(omega_s2 * evt_times)
        + rng.normal(0, 0.005, len(evt_times))
    )

    meteo = generate_meteotsunami_signal(
        evt_times, onset_hour=cal_hours + 2.0,
        amplitude_m=0.10, period_min=20.0,
    )
    evt_signal = evt_signal + meteo

    scores, _ = agent.process_station_data(
        station_id="coops_honolulu",
        times_hours=evt_times,
        values=evt_signal,
        sampling_interval_sec=dt_sec,
        source_type="coops",
        fit_times_hours=cal_times,
        fit_values=cal_signal,
    )

    logger.info(
        "Meteotsunami result: ensemble=%.4f, threshold=%.4f, "
        "wavelet=%.4f, seismic_quiet=%s",
        scores.ensemble_score, scores.threshold_score,
        scores.wavelet_score, scores.seismic_context_quiet,
    )

    return {
        "name": "Meteotsunami False Positive",
        "meteotsunami": {
            "amplitude_m": 0.10,
            "period_min": 20.0,
            "duration_hours": 2.0,
        },
        "result": {
            "ensemble_score": round(scores.ensemble_score, 4),
            "threshold_score": round(scores.threshold_score, 4),
            "wavelet_score": round(scores.wavelet_score, 4),
            "bocpd_score": round(scores.bocpd_score, 4),
            "seismic_quiet": scores.seismic_context_quiet,
            "exceeds_t1": scores.ensemble_score >= T1,
            "exceeds_t2": scores.ensemble_score >= T2,
        },
        "summary": {
            "correctly_rejected": scores.ensemble_score < T2,
            "seismic_quiet_penalty_applied": scores.seismic_context_quiet,
        },
    }


def run_scenario_4_partial_outage() -> dict:
    """Scenario 4: M8.5 tsunami with 4 of 6 DART stations offline."""
    logger.info("=" * 60)
    logger.info("Scenario 4: Partial Network Outage (2 of 6 online)")
    logger.info("=" * 60)

    offline_ids = {"21413", "46404", "46407", "46411"}
    online_stns, offline_list = mark_stations_offline(
        PACIFIC_DART_STATIONS[:6], offline_ids,
    )

    event = generate_coherent_event(ALEUTIAN_SCENARIO, online_stns, seed=42)

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

    results = []
    for sid in event.stations:
        logger.info("Processing station %s (online) ...", sid)
        r = _evaluate_station(event, sid, seismic)
        results.append(r)
        logger.info(
            "  %s: ensemble=%.4f %s",
            sid, r["ensemble_score"],
            "(DETECT)" if r["exceeds_t1"] else "",
        )

    detecting = [r for r in results if r["exceeds_t1"]]

    return {
        "name": "Partial Network Outage",
        "online_stations": [s.station_id for s in online_stns],
        "offline_stations": offline_list,
        "stations": results,
        "summary": {
            "online_count": len(online_stns),
            "offline_count": len(offline_list),
            "detecting_t1": len(detecting),
            "detection_with_reduced_network": len(detecting) > 0,
        },
    }


def main() -> None:
    """Run all scenarios and write results."""
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "physics_validation.json"

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "description": (
            "Physics-correct simulation validation: Agentic AI for "
            "Near-Real-Time Ocean Hazard Assessment (NOAA AI Workshop 2026)"
        ),
        "physical_constants": {
            "deep_ocean_wave_speed_m_s": 198.0,
            "fsm_thresholds": {"T1": T1, "T2": T2, "T3": T3},
            "seismic_quiet_penalty": 1.3,
            "seismic_quiet_magnitude_threshold": 6.5,
            "bandpass_range_min": "5-120",
            "dart_detection_threshold_m": 0.03,
            "coops_detection_threshold_m": 0.15,
            "spatial_coherence_tolerance": 0.20,
        },
        "scenarios": [],
    }

    logger.info("Running physics validation with 4 scenarios...")
    logger.info("")

    # Run all scenarios
    results["scenarios"].append(run_scenario_1_large_tsunami())
    logger.info("")
    results["scenarios"].append(run_scenario_2_moderate_earthquake())
    logger.info("")
    results["scenarios"].append(run_scenario_3_meteotsunami())
    logger.info("")
    results["scenarios"].append(run_scenario_4_partial_outage())
    logger.info("")

    # Summary
    logger.info("=" * 60)
    logger.info("VALIDATION SUMMARY")
    logger.info("=" * 60)
    for scenario in results["scenarios"]:
        name = scenario["name"]
        summary = scenario["summary"]
        logger.info("  %s:", name)
        for k, v in summary.items():
            logger.info("    %s: %s", k, v)

    # Write results
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("")
    logger.info("Results written to %s", output_path)


if __name__ == "__main__":
    main()
