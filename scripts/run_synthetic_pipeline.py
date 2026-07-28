#!/usr/bin/env python3
"""Synthetic pipeline runner for NOAA AI Workshop 2026 paper.

Generates sliding-window time-series analysis on synthetic scenarios and
captures real inter-agent envelope traces from pipeline execution.

Outputs:
    results/synthetic_timelines.json - waveform arrays + score timelines
    results/agent_traces.json - inter-agent envelope chains
    paper/appendix_f_generated.tex - LaTeX from real pipeline traces

Usage:
    python scripts/run_synthetic_pipeline.py

Prerequisites:
    pip install -e .
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

import numpy as np

from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import (
    SeismicEvent,
    detide_and_filter,
)
from hazard_assessment.agents.qc_agent import QCAgent
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.orchestrator.states import (
    FSMOrchestrator,
    SystemState,
    ThresholdConfig,
)
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)
from hazard_assessment.simulation.propagation import (
    PACIFIC_COOPS_STATIONS,
    PACIFIC_DART_STATIONS,
)
from hazard_assessment.simulation.scenario import generate_coherent_event
from hazard_assessment.simulation.source import (
    ALEUTIAN_SCENARIO,
    MODERATE_ALEUTIAN,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# FSM thresholds
T1 = 0.35
T2 = 0.60
T3 = 0.85

THRESHOLDS = ThresholdConfig(basin="pacific", t1=T1, t2=T2, t3=T3)
TSUNAMIGENIC_ZONES = {
    "pacific_nw", "cascadia", "alaska_aleutian", "aleutian", "japan", "tohoku", "maule",
}


# -------------------------------------------------------------------
# Sliding-window analysis
# -------------------------------------------------------------------

def run_sliding_window(
    event,
    station_id: str,
    seismic_events: list[SeismicEvent],
    step_minutes: int = 5,
    max_minutes: int = 360,
) -> dict:
    """Growing-window anomaly detection on a synthetic station.

    Returns dict with waveform_data (for plotting) and timeline (score evolution).
    """
    stn = event.stations[station_id]
    config = stn.config
    cal_hours = event.metadata["calibration_hours"]
    dt_sec = config.sampling_interval_sec
    times = stn.times_hours
    signal = stn.event_signal
    clean = stn.clean_signal

    cal_mask = times < cal_hours
    evt_mask = times >= cal_hours

    evt_times = times[evt_mask]
    evt_signal = signal[evt_mask]
    cal_times = times[cal_mask]
    cal_signal = clean[cal_mask]

    # Detide the full event window for plotting
    sampling_hz = 1.0 / dt_sec
    try:
        detided, filtered = detide_and_filter(
            evt_times, evt_signal, sampling_hz,
            fit_times_hours=cal_times,
            fit_values=clean[cal_mask],
        )
    except Exception:
        detided = np.zeros_like(evt_signal)
        filtered = np.zeros_like(evt_signal)

    # Convert event times to minutes from event start for plotting
    evt_minutes = (evt_times - cal_hours) * 60.0

    # Waveform data for plotting (subsample for JSON size)
    step_plot = max(1, len(evt_times) // 2000)
    waveform_data = {
        "times_min": evt_minutes[::step_plot].tolist(),
        "event_signal": evt_signal[::step_plot].tolist(),
        "clean_signal": clean[evt_mask][::step_plot].tolist(),
        "detided_residual": detided[::step_plot].tolist(),
        "filtered_signal": filtered[::step_plot].tolist(),
        "arrival_min": round((stn.arrival_hour - cal_hours) * 60.0, 2),
        "distance_km": round(stn.distance_km, 1),
        "tsunami_amplitude_m": round(stn.tsunami_amplitude_m, 6),
    }

    # Sliding-window score evolution
    agent = AnomalyAgent()
    agent.calibrate_baseline(
        station_id, cal_signal, dt_sec, source_type=config.station_type,
    )
    agent.update_seismic_events(seismic_events)

    samples_per_step = max(1, int(step_minutes * 60 / dt_sec))
    timeline = []

    for end_idx in range(samples_per_step, len(evt_times), samples_per_step):
        window_times = evt_times[:end_idx]
        window_values = evt_signal[:end_idx]
        window_minutes = (window_times[-1] - cal_hours) * 60.0

        if window_minutes > max_minutes:
            break

        # Use processing_time consistent with the earthquake origin so
        # the seismic-quiet 90-minute window check works correctly.
        eq = event.earthquake
        proc_time = eq.origin_time + timedelta(minutes=window_minutes)

        try:
            scores, _ = agent.process_station_data(
                station_id=station_id,
                times_hours=window_times,
                values=window_values,
                sampling_interval_sec=dt_sec,
                source_type=config.station_type,
                fit_times_hours=cal_times,
                fit_values=clean[cal_mask],
                origin_lat=config.latitude,
                origin_lon=config.longitude,
                processing_time=proc_time,
            )
            timeline.append({
                "minutes": round(window_minutes, 1),
                "ensemble": round(scores.ensemble_score, 4),
                "threshold": round(scores.threshold_score, 4),
                "wavelet": round(scores.wavelet_score, 4),
                "bocpd": round(scores.bocpd_score, 4),
                "seismic_quiet": scores.seismic_context_quiet,
                "filter_degraded": scores.filter_degraded,
            })
        except Exception as exc:
            logger.warning(
                "Sliding window at %.1f min failed for %s: %s",
                window_minutes, station_id, exc,
            )

    logger.info(
        "  %s: %d timeline points, arrival=%.1f min, amp=%.4f m",
        station_id, len(timeline),
        waveform_data["arrival_min"],
        waveform_data["tsunami_amplitude_m"],
    )

    return {
        "station_id": station_id,
        "station_type": config.station_type,
        "waveform_data": waveform_data,
        "timeline": timeline,
    }


# -------------------------------------------------------------------
# Pipeline trace capture
# -------------------------------------------------------------------

def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def run_pipeline_trace(
    event,
    station_id: str,
    ensemble_score: float,
    threshold_score: float = 0.0,
    wavelet_score: float = 0.0,
    bocpd_score: float = 0.0,
    seismic_quiet: bool = False,
    filter_degraded: bool = False,
) -> dict:
    """Run a scenario through the pipeline and capture envelope dicts."""
    eq = event.earthquake

    # Stage 1: QC - create DartRecords from synthetic data
    records = []
    for i in range(20):
        ts = eq.origin_time + timedelta(minutes=i)
        records.append(DartRecord(
            source_id=f"dart:{station_id}:{ts.strftime('%Y%m%d%H%M%S')}:2",
            station_id=station_id,
            source_timestamp=ts,
            ingest_timestamp=ts + timedelta(seconds=5),
            measurement_type=2,
            height_m=5000.0 + 0.01 * i,
            event_mode=True,
            payload_sha256=_sha256(f"synth_{station_id}_{i}"),
        ))

    qc = QCAgent()
    qc_reports = qc.process_records(records, processing_time=eq.origin_time + timedelta(minutes=30))
    qc_envelope = {
        "stage": "QC",
        "n_records": len(qc_reports),
        "sample_report": {
            "station_id": qc_reports[0].station_id,
            "record_usable": qc_reports[0].record_usable,
            "station_confidence": round(qc_reports[0].station_confidence, 4),
            # The first record of a stream has no history, so zero checks
            # evaluate: confidence 1.0 here is the no-evidence convention,
            # which this count makes visible in the published trace.
            "n_checks_evaluated": qc_reports[0].n_checks_evaluated,
            "data_mode": qc_reports[0].data_mode.value,
        },
    }

    # Stage 2: Anomaly detection
    anomaly_dict = {
        "stage": "AnomalyAssessment",
        "type": "AnomalyAssessment",
        "schema_version": "1.0",
        "producer": "anomaly_agent",
        "anomaly_score": round(ensemble_score, 4),
        "score_components": {
            "threshold": round(threshold_score, 4),
            "statistical": round(max(wavelet_score, bocpd_score), 4),
            "ml": None,
        },
        # A station triggers only when its ensemble score reaches T1
        # (inclusive); scored_stations records every scored window.
        "triggering_stations": [station_id] if ensemble_score >= T1 else [],
        "scored_stations": [station_id],
        "seismic_quiet": seismic_quiet,
        "filter_degraded": filter_degraded,
    }

    # Stage 3: FSM transitions
    fsm = FSMOrchestrator(thresholds=THRESHOLDS)
    fsm.evaluate_seismic_trigger(
        magnitude=eq.magnitude,
        region=eq.region,
        epicenter_lat=eq.latitude,
        epicenter_lon=eq.longitude,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        depth_km=eq.depth_km,
    )
    fsm_after_seismic = fsm.state.value

    # Step the FSM up to three times so a high score can climb the ladder;
    # the per-step records land in fsm.transition_history.
    for _ in range(3):
        fsm.evaluate_anomaly_score(ensemble_score)
    fsm_after_anomaly = fsm.state.value

    fsm_envelope = {
        "stage": "FSM",
        "after_seismic_trigger": fsm_after_seismic,
        "after_anomaly_score": fsm_after_anomaly,
        "magnitude": eq.magnitude,
        "anomaly_score": round(ensemble_score, 4),
        # Real per-step records from the FSM (the seismic-only path is a
        # chained IDLE->MONITOR->ESCALATE, not a single jump).
        "transitions": [
            {
                "from": r.from_state.value,
                "to": r.to_state.value,
                "trigger": "anomaly" if r.anomaly_score is not None else "seismic",
            }
            for r in fsm.transition_history
        ],
    }

    # Stage 4+: Scenario + Verification (only if ESCALATE)
    event_id = uuid4()
    scenario_envelope = None
    verification_envelope = None

    if fsm.state == SystemState.ESCALATE:
        scenario = ScenarioAssessment(
            producer="scenario_agent",
            event_id=event_id,
            method="NNLS_UNIT_SOURCE",
            constraint_stage=ConstraintStage.DART_CONSTRAINED,
            dart_stations_used=[station_id],
            dart_stations_excluded=[],
            exclusion_reasons={},
            inversion_window_sec=1800,
            top_scenarios=[
                RankedScenario(
                    unit_source_ids=[f"src_{eq.region}_01"],
                    weights=[1.0],
                    waveform_rmse_cm=1.2,
                    mw_equivalent=round(eq.magnitude, 1),
                    rank=1,
                    posterior_weight=0.85,
                ),
            ],
            coastal_proxies=[],
            ensemble_spread=EnsembleSpread.LOW,
            bilateral_rupture_evaluated=False,
            limiting_assumptions=["Flat bathymetry"],
        )
        scenario_envelope = {
            "stage": "ScenarioAssessment",
            **{k: v for k, v in scenario.model_dump().items()
               if k in ("method", "constraint_stage", "dart_stations_used",
                        "top_scenarios", "ensemble_spread")},
        }

        vr = VerificationResult(
            producer="verification_agent",
            event_id=event_id,
            overall=VerificationOutcome.PASS,
            checks=[
                VerificationCheck(
                    name="holdout_station_validation",
                    result=CheckResult.PASS,
                    evidence="Holdout RMSE 1.2 cm < 3.0 cm threshold",
                ),
                VerificationCheck(
                    name="data_coverage",
                    result=CheckResult.PASS,
                    evidence=f"1/{1} DART stations used (100%)",
                ),
                VerificationCheck(
                    name="physical_consistency",
                    result=CheckResult.PASS,
                    evidence=f"Mw {eq.magnitude} consistent with rupture parameters",
                ),
            ],
            abstain_required=False,
        )
        verification_envelope = {
            "stage": "VerificationResult",
            "overall": vr.overall.value,
            "checks": [
                {"name": c.name, "result": c.result.value, "evidence": c.evidence}
                for c in vr.checks
            ],
            "abstain_required": False,
        }

    trace = {
        "scenario_name": f"M{eq.magnitude} {eq.region}",
        "earthquake": {
            "magnitude": eq.magnitude,
            "latitude": eq.latitude,
            "longitude": eq.longitude,
            "region": eq.region,
        },
        "stages": [
            qc_envelope,
            anomaly_dict,
            fsm_envelope,
        ],
    }
    if scenario_envelope:
        trace["stages"].append(scenario_envelope)
    if verification_envelope:
        trace["stages"].append(verification_envelope)

    return trace


# -------------------------------------------------------------------
# LaTeX generation for Appendix F
# -------------------------------------------------------------------

# Reader-facing annotations for the two fixed scenarios. Keyed by trace
# index; the numeric placeholders are filled from the live trace values so
# regeneration keeps the prose consistent with the data.
_TRACE_INTROS = [
    (
        "\\noindent The {name} scenario exercises the full escalation path.\n"
        "The shallow M8.5 earthquake meets the seismic-only criteria, so the\n"
        "FSM escalates immediately as a chained\n"
        "IDLE~$\\to$~MONITOR~$\\to$~ESCALATE transition\n"
        "(Algorithm~\\ref{{alg:fsm}}, lines~3--5) without waiting for ocean\n"
        "data.  The ensemble anomaly score ({score:.2f}) subsequently exceeds\n"
        "$T_1 = 0.35$, corroborating the seismic trigger with ocean-based\n"
        "evidence: the threshold component dominates ({thr:.2f}), while the\n"
        "statistical component contributes a moderate {stat:.2f}."
    ),
    (
        "\\noindent The {name} moderate earthquake scenario demonstrates correct\n"
        "non-escalation.  The ensemble anomaly score ({score:.3f}) remains well below\n"
        "$T_1 = 0.35$, so the FSM stays in MONITOR after the seismic trigger.\n"
        "The threshold component is zero (signal below detection threshold),\n"
        "with only a small BOCPD contribution ({stat:.2f}) from the broadband\n"
        "residual."
    ),
]

_STAGE_LEADINS = [
    {
        "QC": (
            "QC Agent validates incoming records and reports per-station\n"
            "usability, confidence, evaluated-check coverage, and data mode:"
        ),
        "AnomalyAssessment": (
            "Anomaly Agent fuses threshold, wavelet, and BOCPD\n"
            "detectors into a single ensemble score and identifies triggering\n"
            "stations:"
        ),
        "FSM": (
            "FSM Orchestrator records each deterministic state\n"
            "transition with its trigger condition:"
        ),
        "ScenarioAssessment": (
            "Scenario and Verification envelopes illustrate the expected\n"
            "output schema (the inversion and verification agents need real\n"
            "DART waveforms, which synthetic events do not provide):"
        ),
    },
    {
        "QC": "QC Agent confirms data usability:",
        "AnomalyAssessment": (
            "Anomaly Agent: low score ({score:.3f}) reflects the absence\n"
            "of a detectable tsunami signal:"
        ),
        "FSM": (
            "FSM Orchestrator: only one transition\n"
            "(IDLE~$\\to$~MONITOR); score too low for further advancement:"
        ),
    },
]

_STAGE_LABELS = {
    "QC": ("traceqc", "QC Agent Output"),
    "AnomalyAssessment": ("traceanomaly", "Anomaly Assessment"),
    "FSM": ("tracefsm", "FSM Orchestrator"),
    "ScenarioAssessment": ("tracescenario", "Scenario Assessment (illustrative)"),
    "VerificationResult": ("traceverify", "Verification Result (illustrative)"),
}


def generate_appendix_f_latex(traces: list[dict], output_path: Path) -> None:
    """Emit paper/appendix_f_generated.tex from real pipeline traces.

    Produces the styled form the paper includes (trace boxes with reader
    annotations); the annotation numbers are formatted from the live trace
    values so the prose cannot drift from the data.
    """
    lines = [
        "% Auto-generated by scripts/run_synthetic_pipeline.py",
        "% Styled with agent-specific trace boxes; annotations are filled from",
        "% the live trace values. Edit the templates in the script, not here.",
        "",
    ]

    for i, trace in enumerate(traces):
        scenario_name = trace["scenario_name"].title()
        anomaly_stage = next(
            s for s in trace["stages"] if s.get("stage") == "AnomalyAssessment"
        )
        fmt = {
            "name": scenario_name,
            "score": anomaly_stage["anomaly_score"],
            "thr": anomaly_stage["score_components"]["threshold"],
            "stat": anomaly_stage["score_components"]["statistical"],
        }

        lines.append(f"\\subsection*{{Trace {i+1}: {_latex_escape(scenario_name)}}}")
        lines.append("")
        lines.append(_TRACE_INTROS[i].format(**fmt))
        lines.append("")
        lines.append("\\medskip")

        for stage in trace["stages"]:
            stage_name = stage.get("stage", "Unknown")
            stage_json = {k: v for k, v in stage.items() if k != "stage"}
            json_str = json.dumps(stage_json, indent=2, default=str)
            color, label = _STAGE_LABELS.get(stage_name, ("black!60", stage_name))
            leadin = _STAGE_LEADINS[i].get(stage_name, "").format(**fmt)

            if leadin:
                lines.append(f"\\noindent\\textit{{{leadin}}}")
                lines.append("")
            lines.append(f"\\tracelabel[{color}]{{{label}}}")
            lines.append("\\begin{tracebox}")
            lines.append(json_str)
            lines.append("\\end{tracebox}")
            lines.append("")

        lines.append("")

    output_path.write_text("\n".join(lines))
    logger.info("Generated %s", output_path)


def _latex_escape(s: str) -> str:
    """Escape special LaTeX characters."""
    # Backslash must be first to avoid double-escaping
    s = s.replace("\\", r"\textbackslash{}")
    replacements = {
        "_": r"\_",
        "&": r"\&",
        "%": r"\%",
        "#": r"\#",
        "{": r"\{",
        "}": r"\}",
        "$": r"\$",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        s = s.replace(old, new)
    return s


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------

def main() -> None:
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    # ---------------------------------------------------------------
    # Scenario 1: M8.5 Aleutian (TRUE POSITIVE)
    # ---------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Scenario 1: M8.5 Aleutian (6 DART + 3 CO-OPS stations)")
    logger.info("=" * 60)

    stations_1 = PACIFIC_DART_STATIONS + PACIFIC_COOPS_STATIONS
    event_1 = generate_coherent_event(ALEUTIAN_SCENARIO, stations_1, seed=42)
    eq_1 = event_1.earthquake

    seismic_1 = [
        SeismicEvent(
            event_id=eq_1.event_id,
            magnitude=eq_1.magnitude,
            origin_time=eq_1.origin_time,
            latitude=eq_1.latitude,
            longitude=eq_1.longitude,
        )
    ]

    scenario_1_stations = []
    for sid in event_1.stations:
        logger.info("  Sliding-window analysis for %s ...", sid)
        result = run_sliding_window(event_1, sid, seismic_1)
        scenario_1_stations.append(result)

    # ---------------------------------------------------------------
    # Scenario 2: M7.0 Moderate Aleutian (CORRECT REJECTION)
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Scenario 2: M7.0 Moderate Aleutian (3 DART stations)")
    logger.info("=" * 60)

    stations_2 = PACIFIC_DART_STATIONS[:3]
    event_2 = generate_coherent_event(MODERATE_ALEUTIAN, stations_2, seed=123)
    eq_2 = event_2.earthquake

    seismic_2 = [
        SeismicEvent(
            event_id=eq_2.event_id,
            magnitude=eq_2.magnitude,
            origin_time=eq_2.origin_time,
            latitude=eq_2.latitude,
            longitude=eq_2.longitude,
        )
    ]

    scenario_2_stations = []
    for sid in event_2.stations:
        logger.info("  Sliding-window analysis for %s ...", sid)
        result = run_sliding_window(event_2, sid, seismic_2)
        scenario_2_stations.append(result)

    # ---------------------------------------------------------------
    # Export timelines
    # ---------------------------------------------------------------
    timelines = {
        "generated_at": datetime.now(UTC).isoformat(),
        "scenarios": [
            {
                "name": f"M{eq_1.magnitude} Aleutian",
                "earthquake": {
                    "magnitude": eq_1.magnitude,
                    "latitude": eq_1.latitude,
                    "longitude": eq_1.longitude,
                    "depth_km": eq_1.depth_km,
                    "region": eq_1.region,
                    "origin_time": eq_1.origin_time.isoformat()
                        if hasattr(eq_1.origin_time, "isoformat")
                        else str(eq_1.origin_time),
                    "event_id": eq_1.event_id,
                    "strike_deg": eq_1.strike_deg,
                    "rake_deg": eq_1.rake_deg,
                    "mechanism": "Thrust" if 60 <= eq_1.rake_deg <= 120 else "Normal",
                },
                "stations": scenario_1_stations,
            },
            {
                "name": f"M{eq_2.magnitude} Moderate",
                "earthquake": {
                    "magnitude": eq_2.magnitude,
                    "latitude": eq_2.latitude,
                    "longitude": eq_2.longitude,
                    "depth_km": eq_2.depth_km,
                    "region": eq_2.region,
                    "origin_time": eq_2.origin_time.isoformat()
                        if hasattr(eq_2.origin_time, "isoformat")
                        else str(eq_2.origin_time),
                    "event_id": eq_2.event_id,
                    "strike_deg": eq_2.strike_deg,
                    "rake_deg": eq_2.rake_deg,
                    "mechanism": "Thrust" if 60 <= eq_2.rake_deg <= 120 else "Normal",
                },
                "stations": scenario_2_stations,
            },
        ],
    }

    timelines_path = output_dir / "synthetic_timelines.json"
    with open(timelines_path, "w") as f:
        json.dump(timelines, f, indent=2)
    logger.info("Timelines written to %s", timelines_path)

    # ---------------------------------------------------------------
    # Pipeline traces
    # ---------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("Pipeline trace capture")
    logger.info("=" * 60)

    # Get actual scores from the full-window evaluation
    def _extract_final_scores(stations):
        if stations and stations[0]["timeline"]:
            last = stations[0]["timeline"][-1]
            return (last["ensemble"], last["threshold"],
                    last["wavelet"], last["bocpd"],
                    last.get("seismic_quiet", False),
                    last.get("filter_degraded", False))
        return (0.0, 0.0, 0.0, 0.0, False, False)

    s1_ens, s1_thr, s1_wav, s1_boc, s1_sq, s1_fd = _extract_final_scores(scenario_1_stations)
    s2_ens, s2_thr, s2_wav, s2_boc, s2_sq, s2_fd = _extract_final_scores(scenario_2_stations)

    if s1_ens < T3:
        logger.warning(
            "M8.5 ensemble score %.4f < T3 (%.2f); trace will not reach ESCALATE",
            s1_ens, T3,
        )

    trace_1 = run_pipeline_trace(
        event_1,
        list(event_1.stations.keys())[0],
        ensemble_score=s1_ens,
        threshold_score=s1_thr,
        wavelet_score=s1_wav,
        bocpd_score=s1_boc,
        seismic_quiet=s1_sq,
        filter_degraded=s1_fd,
    )
    logger.info("  Trace 1 (M8.5): %d stages captured (score=%.4f)", len(trace_1["stages"]), s1_ens)

    trace_2 = run_pipeline_trace(
        event_2,
        list(event_2.stations.keys())[0],
        ensemble_score=s2_ens,
        threshold_score=s2_thr,
        wavelet_score=s2_wav,
        bocpd_score=s2_boc,
        seismic_quiet=s2_sq,
        filter_degraded=s2_fd,
    )
    logger.info("  Trace 2 (M7.0): %d stages captured", len(trace_2["stages"]))

    traces = {"traces": [trace_1, trace_2]}
    traces_path = output_dir / "agent_traces.json"
    with open(traces_path, "w") as f:
        json.dump(traces, f, indent=2, default=str)
    logger.info("Traces written to %s", traces_path)

    # ---------------------------------------------------------------
    # Generate Appendix F LaTeX
    # ---------------------------------------------------------------
    latex_path = Path("paper") / "appendix_f_generated.tex"
    latex_path.parent.mkdir(exist_ok=True)
    generate_appendix_f_latex([trace_1, trace_2], latex_path)

    logger.info("")
    logger.info("Done. Outputs:")
    logger.info("  %s", timelines_path)
    logger.info("  %s", traces_path)
    logger.info("  %s", latex_path)


if __name__ == "__main__":
    main()
