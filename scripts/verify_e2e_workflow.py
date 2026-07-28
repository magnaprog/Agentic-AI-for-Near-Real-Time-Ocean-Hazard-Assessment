#!/usr/bin/env python3
"""End-to-end workflow verification for the hazard assessment system.

Exercises every real component - QC, anomaly detection, FSM, scenario
inversion, verification, report generation, pipeline integration,
escalation packet, and human review - with synthetic Tohoku-like data.

Usage:
    .venv/bin/python scripts/verify_e2e_workflow.py
"""

from __future__ import annotations

import hashlib
import math
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID, uuid4

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Imports: Agents
# ---------------------------------------------------------------------------
from hazard_assessment.agents.anomaly_agent import AnomalyAgent
from hazard_assessment.agents.anomaly_detection import SeismicEvent
from hazard_assessment.agents.assessment_formatter import format_human_decision
from hazard_assessment.agents.qc_agent import QCAgent
from hazard_assessment.agents.report_agent import ReportAgent
from hazard_assessment.agents.scenario_agent import ScenarioAgent
from hazard_assessment.agents.scenario_data import InMemoryUnitSourceDatabase, UnitSource
from hazard_assessment.agents.verification_agent import VerificationAgent
from hazard_assessment.agents.verification_checks import (
    StationPosition,
    VerificationInput,
)

# ---------------------------------------------------------------------------
# Imports: Orchestrator & Pipeline
# ---------------------------------------------------------------------------
from hazard_assessment.audit.logger import AuditLogger

# ---------------------------------------------------------------------------
# Imports: Ingest
# ---------------------------------------------------------------------------
from hazard_assessment.ingest.coops import CoopsRecord
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.orchestrator.nodes import run_pipeline_sync
from hazard_assessment.orchestrator.states import (
    FSMOrchestrator,
    SystemState,
    ThresholdConfig,
)

# ---------------------------------------------------------------------------
# Imports: Schemas
# ---------------------------------------------------------------------------
from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER
from hazard_assessment.schemas.anomaly import AnomalyAssessment
from hazard_assessment.schemas.final_assessment import AssessmentStatus, FinalAssessment
from hazard_assessment.schemas.human_decision import HumanDecision, ReviewDecision
from hazard_assessment.schemas.qc import DataMode
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

# ---------------------------------------------------------------------------
# Constants: Tohoku-like earthquake
# ---------------------------------------------------------------------------
EPICENTER_LAT = 38.297
EPICENTER_LON = 142.373
MAGNITUDE = 9.1
REGION = "japan"
ORIGIN_TIME = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)
FIXED_TIME = datetime(2011, 3, 11, 6, 30, 0, tzinfo=UTC)
THRESHOLDS = ThresholdConfig(basin="pacific", t1=0.35, t2=0.60, t3=0.85)
TSUNAMIGENIC_ZONES = {"pacific_nw", "cascadia", "alaska_aleutian", "japan"}
FIXED_ESCALATION_PACKET_ID = UUID("8a729518-4413-4043-9791-135864832890")


# ---------------------------------------------------------------------------
# Runner harness
# ---------------------------------------------------------------------------
@dataclass
class StageResult:
    name: str
    passed: bool
    detail: str
    elapsed_sec: float


def run_stage(name: str, func: Callable[[], None]) -> StageResult:
    start = time.perf_counter()
    try:
        func()
        elapsed = time.perf_counter() - start
        return StageResult(name=name, passed=True, detail="OK", elapsed_sec=elapsed)
    except Exception as e:
        elapsed = time.perf_counter() - start
        tb = traceback.format_exception(e)
        detail = "".join(tb[-2:]).strip()
        return StageResult(name=name, passed=False, detail=detail, elapsed_sec=elapsed)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------
def _sha256(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def generate_tidal_signal(
    n_hours: int = 30 * 24,
    dt_hours: float = 1.0 / 60.0,
    seed: int = 42,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Generate realistic M2+S2 tidal signal with noise."""
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


def inject_tsunami(
    times: NDArray[np.float64],
    signal: NDArray[np.float64],
    arrival_hour: float,
    amplitude_m: float = 0.15,
    period_min: float = 25.0,
) -> NDArray[np.float64]:
    """Inject a synthetic tsunami waveform after arrival_hour."""
    modified = signal.copy()
    mask = times >= arrival_hour
    t_since = (times[mask] - arrival_hour) * 60.0  # minutes
    decay = np.exp(-t_since / 120.0)
    omega = 2.0 * np.pi / period_min
    modified[mask] += amplitude_m * decay * np.sin(omega * t_since)
    return modified


def make_dart_records(
    station_id: str = "21413",
    n_records: int = 20,
    event_mode: bool = True,
) -> list[DartRecord]:
    records = []
    for i in range(n_records):
        ts = ORIGIN_TIME + timedelta(minutes=i)
        records.append(
            DartRecord(
                source_id=f"dart:{station_id}:{ts.strftime('%Y%m%d%H%M%S')}:2",
                station_id=station_id,
                source_timestamp=ts,
                ingest_timestamp=ts + timedelta(seconds=5),
                measurement_type=2,
                height_m=5000.0 + 0.01 * i,
                event_mode=event_mode,
                payload_sha256=_sha256(f"dart_{station_id}_{i}"),
            )
        )
    return records


def make_coops_records(
    station_id: str = "1612340",
    n_records: int = 10,
) -> list[CoopsRecord]:
    records = []
    for i in range(n_records):
        ts = ORIGIN_TIME + timedelta(minutes=i * 6)
        records.append(
            CoopsRecord(
                source_id=f"coops:{station_id}:water_level:{ts.strftime('%Y%m%d%H%M')}",
                station_id=station_id,
                station_name="Honolulu",
                product="water_level",
                source_timestamp=ts,
                ingest_timestamp=ts + timedelta(seconds=10),
                water_level_m=1.5 + 0.05 * math.sin(i * 0.5),
                flags="",
                quality="v",
                payload_sha256=_sha256(f"coops_{station_id}_{i}"),
            )
        )
    return records


def build_test_db(
    n_sources: int = 5,
    n_timepoints: int = 60,
    station_ids: list[str] | None = None,
) -> InMemoryUnitSourceDatabase:
    """Build InMemoryUnitSourceDatabase with sources near Tohoku epicenter."""
    db = InMemoryUnitSourceDatabase()
    if station_ids is None:
        station_ids = ["21413", "21418"]

    rng = np.random.default_rng(42)
    for i in range(n_sources):
        src = UnitSource(
            source_id=f"tohoku_{i:02d}",
            latitude=EPICENTER_LAT + 0.1 * i,
            longitude=EPICENTER_LON + 0.05 * i,
            depth_km=15.0,
            strike_deg=193.0,
            dip_deg=14.0,
            rake_deg=81.0,
            length_km=50.0,
            width_km=25.0,
            rigidity_pa=3.5e10,
            fault_zone_id="japan_trench",
            segment_index=i,
        )
        db.add_source(src)
        for sid in station_ids:
            waveform = rng.standard_normal(n_timepoints).astype(np.float64) * 0.01
            db.set_greens_function(src.source_id, sid, waveform)
    return db


def make_fsm_in_state(
    target: SystemState,
    thresholds: ThresholdConfig = THRESHOLDS,
    audit_writer: AuditLogger | None = None,
) -> FSMOrchestrator:
    """Create an FSM in the specified state with an event context."""
    fsm = FSMOrchestrator(thresholds=thresholds, audit_writer=audit_writer)
    if target == SystemState.IDLE:
        return fsm
    fsm.evaluate_seismic_trigger(
        magnitude=7.0,
        region=REGION,
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
    )
    if target == SystemState.MONITOR:
        return fsm
    fsm.evaluate_anomaly_score(0.40)  # MONITOR -> INVESTIGATE
    if target == SystemState.INVESTIGATE:
        return fsm
    fsm.evaluate_anomaly_score(0.65)  # INVESTIGATE -> ASSESS
    if target == SystemState.ASSESS:
        return fsm
    fsm.evaluate_anomaly_score(0.90)  # ASSESS -> ESCALATE
    return fsm


def make_anomaly_dict(score: float) -> dict[str, Any]:
    return {
        "type": "AnomalyAssessment",
        "schema_version": "1.0",
        "producer": "anomaly_agent",
        "anomaly_score": score,
        "score_components": {"threshold": 0.5, "statistical": 0.3, "ml": None},
        "triggering_stations": ["21413"],
        "scored_stations": ["21413"],
        "spatial_confirmations": [],
        "seismic_quiet": False,
        "meteotsunami_score": None,
        "stations_offline": [],
        "coverage_note": "",
        "reasoning_trace": "e2e verification",
        "current_state": "",
        "state_changed": False,
        "filter_degraded": False,
        "rayleigh_wave_suspect": False,
    }


def make_scenario_dict(event_id: UUID | None = None) -> dict[str, Any]:
    eid = event_id or uuid4()
    scenario = ScenarioAssessment(
        producer="scenario_agent",
        event_id=eid,
        method="NNLS_UNIT_SOURCE",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        dart_stations_used=["21413"],
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=1800,
        top_scenarios=[
            RankedScenario(
                unit_source_ids=["SRC_A"],
                weights=[1.0],
                waveform_rmse_cm=1.0,
                mw_equivalent=7.8,
                rank=1,
                posterior_weight=0.8,
            ),
        ],
        coastal_proxies=[],
        ensemble_spread=EnsembleSpread.LOW,
        bilateral_rupture_evaluated=False,
        limiting_assumptions=["Flat bathymetry assumed"],
    )
    return scenario.model_dump()


def make_verification_dict(
    outcome: VerificationOutcome = VerificationOutcome.PASS,
    event_id: UUID | None = None,
) -> dict[str, Any]:
    eid = event_id or uuid4()
    checks = [
        VerificationCheck(
            name="holdout_station", result=CheckResult.PASS, evidence="OK"
        ),
    ]
    if outcome == VerificationOutcome.PASS_WITH_CONCERNS:
        checks.append(
            VerificationCheck(
                name="data_coverage",
                result=CheckResult.CONCERN,
                evidence="Only 65% coverage",
            )
        )
    elif outcome == VerificationOutcome.FAIL:
        checks.append(
            VerificationCheck(
                name="data_coverage",
                result=CheckResult.FAIL,
                evidence="Insufficient constraining-station coverage",
            )
        )
    abstain_required = outcome == VerificationOutcome.FAIL
    abstain_reason = "Verification failed" if abstain_required else None
    vr = VerificationResult(
        producer="verification_agent",
        event_id=eid,
        overall=outcome,
        checks=checks,
        abstain_required=abstain_required,
        abstain_reason=abstain_reason,
    )
    return vr.model_dump()


def make_human_decision_dict(
    decision: ReviewDecision = ReviewDecision.APPROVE,
    reason: str = "Assessment looks correct.",
) -> dict[str, Any]:
    hd = HumanDecision(
        producer="human_review_node",
        reviewer_id="alice",
        decision=decision,
        decision_reason=reason,
        decided_at_utc=datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC),
        escalation_packet_id=FIXED_ESCALATION_PACKET_ID,
    )
    return hd.model_dump()


# ===================================================================
# Stage 1: QC Agent
# ===================================================================
def verify_qc_agent() -> None:
    dart_records = make_dart_records("21413", n_records=20, event_mode=True)
    coops_records = make_coops_records("1612340", n_records=10)
    all_records: list[DartRecord | CoopsRecord] = dart_records + coops_records  # type: ignore[assignment]

    qc = QCAgent()
    reports = qc.process_records(all_records, processing_time=FIXED_TIME)

    assert len(reports) == 30, f"Expected 30 QC reports, got {len(reports)}"

    for r in reports:
        assert r.station_confidence >= 0.0, f"Negative confidence: {r.station_confidence}"
        assert isinstance(r.record_usable, bool)
        assert r.provenance_hash, "Empty provenance hash"

    dart_reports = [r for r in reports if r.station_id == "21413"]
    assert all(
        r.data_mode == DataMode.EVENT for r in dart_reports
    ), "DART event-mode reports should have EVENT data_mode"

    coops_reports = [r for r in reports if r.station_id == "1612340"]
    assert all(
        r.data_mode == DataMode.STANDARD for r in coops_reports
    ), "CO-OPS reports should have STANDARD data_mode"


# ===================================================================
# Stage 2: Anomaly Detection Agent
# ===================================================================
def verify_anomaly_agent() -> None:
    # Generate 30-day tidal signal + inject tsunami at hour 718
    times, signal = generate_tidal_signal(n_hours=30 * 24, dt_hours=1.0 / 60.0)
    tsunami_arrival_hour = 718.0
    signal_with_tsunami = inject_tsunami(times, signal, tsunami_arrival_hour, 0.15)

    agent = AnomalyAgent()

    # Calibrate baseline from first 24 hours of clean signal
    baseline_n = int(24 * 60)  # 24 hours at 1-min sampling
    agent.calibrate_baseline("21413", signal[:baseline_n], 60.0)

    # Set seismic context
    agent.update_seismic_events([
        SeismicEvent(
            event_id="us2011tohoku",
            magnitude=MAGNITUDE,
            origin_time=ORIGIN_TIME,
            latitude=EPICENTER_LAT,
            longitude=EPICENTER_LON,
        ),
    ])

    # Process last 3 hours (containing tsunami arrival at hour 718)
    start_hour = 716.0
    mask = times >= start_hour
    scores, spatial = agent.process_station_data(
        station_id="21413",
        times_hours=times[mask],
        values=signal_with_tsunami[mask],
        sampling_interval_sec=60.0,
        source_type="dart",
        fit_times_hours=times,
        fit_values=signal,
        processing_time=FIXED_TIME,
    )

    assert scores.ensemble_score >= 0.0, f"ensemble_score < 0: {scores.ensemble_score}"
    assert scores.threshold_score >= 0.0, "threshold_score should be >= 0"
    assert not scores.seismic_context_quiet, "Should NOT be seismic quiet with M9.1 event"

    # Build AnomalyAssessment envelope and validate schema
    assessment = agent.build_assessment(
        station_ids=["21413"],
        scores=scores,
    )
    assert isinstance(assessment, AnomalyAssessment)
    assert assessment.anomaly_score == scores.ensemble_score
    # Round-trip validation
    AnomalyAssessment.model_validate(assessment.model_dump())


# ===================================================================
# Stage 3: FSM Orchestrator
# ===================================================================
def verify_fsm() -> None:
    audit = AuditLogger()
    fsm = FSMOrchestrator(thresholds=THRESHOLDS, audit_writer=audit)
    assert fsm.state == SystemState.IDLE

    # IDLE -> MONITOR
    rec = fsm.evaluate_seismic_trigger(
        magnitude=MAGNITUDE,
        region=REGION,
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
    )
    assert rec is not None, "Seismic trigger should produce transition"
    assert fsm.state == SystemState.MONITOR

    # MONITOR -> INVESTIGATE (score >= T1=0.35)
    rec = fsm.evaluate_anomaly_score(0.40)
    assert rec is not None
    assert fsm.state == SystemState.INVESTIGATE

    # INVESTIGATE -> ASSESS (score >= T2=0.60)
    rec = fsm.evaluate_anomaly_score(0.65)
    assert rec is not None
    assert fsm.state == SystemState.ASSESS

    # ASSESS -> ESCALATE (score >= T3=0.85)
    rec = fsm.evaluate_anomaly_score(0.90)
    assert rec is not None
    assert fsm.state == SystemState.ESCALATE

    # ESCALATE -> IDLE (resolve)
    rec = fsm.resolve_event()
    assert rec is not None
    assert fsm.state == SystemState.IDLE
    assert fsm.event_context is None, "Event context should be cleared after resolve"

    assert len(fsm.transition_history) == 5
    assert len(audit.get_entries(event_type="state_transition")) == 5

    # De-escalation test
    fsm2 = FSMOrchestrator(thresholds=THRESHOLDS)
    fsm2.evaluate_seismic_trigger(
        magnitude=7.0, region=REGION,
        epicenter_lat=EPICENTER_LAT, epicenter_lon=EPICENTER_LON,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
    )
    fsm2.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
    fsm2.evaluate_anomaly_score(0.65)  # -> ASSESS
    assert fsm2.state == SystemState.ASSESS

    rec = fsm2.evaluate_anomaly_score(0.50)  # ASSESS -> INVESTIGATE (< T2)
    assert rec is not None
    assert fsm2.state == SystemState.INVESTIGATE

    rec = fsm2.evaluate_anomaly_score(0.20)  # INVESTIGATE -> MONITOR (< T1)
    assert rec is not None
    assert fsm2.state == SystemState.MONITOR


# ===================================================================
# Stage 4: Scenario Agent (Seismic-Only)
# ===================================================================
def verify_scenario_seismic_only() -> None:
    db = build_test_db(n_sources=5, n_timepoints=60)
    agent = ScenarioAgent(database=db)

    result = agent.run_seismic_only(
        magnitude=MAGNITUDE,
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
        region="Pacific",
        processing_time=FIXED_TIME,
    )

    assert result.constraint_stage == ConstraintStage.SEISMIC_ONLY
    assert len(result.dart_stations_used) == 0
    assert len(result.top_scenarios) >= 1
    assert result.top_scenarios[0].mw_equivalent > 0
    assert result.ensemble_spread == EnsembleSpread.HIGH

    # Schema round-trip
    ScenarioAssessment.model_validate(result.model_dump())


# ===================================================================
# Stage 5: Scenario Agent (DART-Constrained)
# ===================================================================
_dart_scenario_result: ScenarioAssessment | None = None


def verify_scenario_dart_constrained() -> None:
    global _dart_scenario_result

    station_ids = ["21413", "21418"]
    db = build_test_db(n_sources=5, n_timepoints=60, station_ids=station_ids)
    agent = ScenarioAgent(database=db)

    rng = np.random.default_rng(123)
    waveforms = {sid: rng.standard_normal(60).astype(np.float64) * 0.01 for sid in station_ids}

    result = agent.run_dart_constrained(
        station_waveforms=waveforms,
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
        processing_time=FIXED_TIME,
    )

    assert result.constraint_stage in (
        ConstraintStage.DART_CONSTRAINED,
        ConstraintStage.MULTI_STATION,
    )
    assert len(result.top_scenarios) >= 1
    assert result.top_scenarios[0].mw_equivalent >= 0.0
    assert result.top_scenarios[0].waveform_rmse_cm >= 0.0

    # Schema round-trip
    ScenarioAssessment.model_validate(result.model_dump())

    _dart_scenario_result = result


# ===================================================================
# Stage 6a: Verification Agent (PASS path)
# ===================================================================
def verify_verification_pass() -> None:
    assert _dart_scenario_result is not None, "Stage 5 must run first"

    # The requirement matrix makes sensitivity_analysis REQUIRED at
    # MULTI_STATION, so supply a small well-conditioned inversion
    # problem: orthogonal columns with a single active source are
    # leave-one-out stable by construction.
    H = np.eye(3, dtype=np.float64)
    d = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    sources = [
        UnitSource(
            source_id=f"verify_{i:02d}",
            latitude=EPICENTER_LAT + 0.1 * i,
            longitude=EPICENTER_LON + 0.05 * i,
            depth_km=15.0,
            strike_deg=193.0,
            dip_deg=14.0,
            rake_deg=81.0,
            length_km=50.0,
            width_km=25.0,
            rigidity_pa=3.5e10,
            fault_zone_id="japan_trench",
            segment_index=i,
        )
        for i in range(3)
    ]

    agent = VerificationAgent()
    vi = VerificationInput(
        scenario=_dart_scenario_result,
        H=H,
        d=d,
        sources=sources,
        mw_seismic=_dart_scenario_result.top_scenarios[0].mw_equivalent,
        station_positions=[
            StationPosition("21413", EPICENTER_LAT + 5.0, EPICENTER_LON - 10.0),
            StationPosition("21418", EPICENTER_LAT - 3.0, EPICENTER_LON + 15.0),
        ],
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
    )

    result = agent.verify(vi)

    assert result.overall in (
        VerificationOutcome.PASS,
        VerificationOutcome.PASS_WITH_CONCERNS,
    ), f"Expected PASS or PASS_WITH_CONCERNS, got {result.overall}"
    assert not result.abstain_required, "PASS path should not require abstain"
    assert len(result.checks) >= 1, "Should have at least 1 check"

    # Schema round-trip
    VerificationResult.model_validate(result.model_dump())


# ===================================================================
# Stage 6c: Verification Agent (INCOMPLETE / ABSTAIN path)
# ===================================================================
def verify_verification_incomplete() -> None:
    assert _dart_scenario_result is not None, "Stage 5 must run first"

    # No seismic magnitude: physical_consistency is REQUIRED at every
    # constraint stage, so its missing prerequisite must fail closed
    # to INCOMPLETE with abstain, never silently pass.
    agent = VerificationAgent()
    vi = VerificationInput(scenario=_dart_scenario_result)

    result = agent.verify(vi)

    assert result.overall == VerificationOutcome.INCOMPLETE, (
        f"Expected INCOMPLETE with missing required inputs, got {result.overall}"
    )
    assert result.abstain_required, "INCOMPLETE must require abstain"
    assert result.abstain_reason, "INCOMPLETE must have abstain_reason"
    assert "physical_consistency" in result.abstain_reason

    # Schema round-trip
    VerificationResult.model_validate(result.model_dump())


# ===================================================================
# Stage 6b: Verification Agent (FAIL / ABSTAIN path)
# ===================================================================
def verify_verification_fail() -> None:
    assert _dart_scenario_result is not None, "Stage 5 must run first"

    agent = VerificationAgent()
    vi = VerificationInput(
        scenario=_dart_scenario_result,
        # Large Mw mismatch to trigger physical_consistency FAIL
        mw_seismic=_dart_scenario_result.top_scenarios[0].mw_equivalent + 2.0,
        station_positions=[
            StationPosition("21413", EPICENTER_LAT + 5.0, EPICENTER_LON - 10.0),
        ],
        epicenter_lat=EPICENTER_LAT,
        epicenter_lon=EPICENTER_LON,
    )

    result = agent.verify(vi)

    assert result.overall == VerificationOutcome.FAIL, (
        f"Expected FAIL with large Mw mismatch, got {result.overall}"
    )
    assert result.abstain_required, "FAIL should require abstain"
    assert result.abstain_reason, "FAIL should have abstain_reason"

    # Schema round-trip
    VerificationResult.model_validate(result.model_dump())


# ===================================================================
# Stage 7: Report Agent
# ===================================================================
def verify_report_agent() -> None:
    assert _dart_scenario_result is not None, "Stage 5 must run first"

    # Build a PASS verification result
    verification = VerificationResult(
        producer="verification_agent",
        overall=VerificationOutcome.PASS,
        checks=[
            VerificationCheck(
                name="holdout_station", result=CheckResult.PASS, evidence="OK"
            ),
            VerificationCheck(
                name="data_coverage", result=CheckResult.PASS, evidence="Full coverage"
            ),
        ],
        abstain_required=False,
    )

    agent = ReportAgent()
    report = agent.synthesize(
        scenario=_dart_scenario_result,
        verification=verification,
        tier=1,
    )

    assert report.status == AssessmentStatus.PROVISIONAL
    assert report.report_tier == 1
    assert report.summary, "Summary should not be empty"
    assert report.system_confidence is not None
    assert 0.0 <= report.system_confidence <= 1.0
    assert not report.llm_synthesis_used, "LLM should not be used without API key"
    assert report.disclaimer == NON_AUTHORITATIVE_DISCLAIMER

    # Schema round-trip
    FinalAssessment.model_validate(report.model_dump())


# ===================================================================
# Stage 8a: Pipeline (PASS path)
# ===================================================================
def verify_pipeline_pass() -> None:
    audit = AuditLogger()
    fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
    event_id = fsm.event_context.event_id if fsm.event_context else uuid4()
    report_agent = ReportAgent()

    state = {
        "anomaly_assessment": make_anomaly_dict(0.65),
        "scenario_assessment": make_scenario_dict(event_id),
        "verification_result": make_verification_dict(VerificationOutcome.PASS, event_id),
        "human_decision": make_human_decision_dict(ReviewDecision.APPROVE),
    }

    result = run_pipeline_sync(
        state, fsm=fsm, report_agent=report_agent, audit_logger=audit,
    )

    assert result["fsm_state"] == "ASSESS", f"Expected ASSESS, got {result['fsm_state']}"
    fa = result.get("final_assessment")
    assert fa is not None, "final_assessment should be populated"
    assert fa.get("status") == "APPROVED_INTERNAL", (
        f"Expected APPROVED_INTERNAL, got {fa.get('status')}"
    )
    # Schema round-trip
    FinalAssessment.model_validate(fa)


# ===================================================================
# Stage 8b: Pipeline (ABSTAIN path)
# ===================================================================
def verify_pipeline_abstain() -> None:
    audit = AuditLogger()
    fsm = make_fsm_in_state(SystemState.INVESTIGATE, audit_writer=audit)
    event_id = fsm.event_context.event_id if fsm.event_context else uuid4()

    state = {
        "anomaly_assessment": make_anomaly_dict(0.65),
        "scenario_assessment": make_scenario_dict(event_id),
        "verification_result": make_verification_dict(VerificationOutcome.FAIL, event_id),
    }

    result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

    assert result.get("abstain_triggered") is True, "ABSTAIN should be triggered"
    fa = result.get("final_assessment")
    assert fa is not None, "final_assessment should be populated"
    assert fa.get("status") == "ABSTAIN", f"Expected ABSTAIN, got {fa.get('status')}"


# ===================================================================
# Stage 8c: Pipeline (insufficient evidence)
# ===================================================================
def verify_pipeline_insufficient() -> None:
    audit = AuditLogger()
    fsm = make_fsm_in_state(SystemState.MONITOR, audit_writer=audit)

    state = {
        "anomaly_assessment": make_anomaly_dict(0.20),
    }

    result = run_pipeline_sync(state, fsm=fsm, audit_logger=audit)

    assert fsm.state == SystemState.MONITOR, "Should stay in MONITOR with low score"
    fa = result.get("final_assessment")
    assert fa is not None, "final_assessment should be populated"
    assert fa.get("outcome") == "insufficient_evidence"


# ===================================================================
# Stage 9a: Escalation Packet
# ===================================================================
def verify_escalation_packet() -> None:
    from hazard_assessment.app import generate_escalation_packet
    from hazard_assessment.schemas.escalation import EscalationPacket

    audit = AuditLogger()
    fsm = make_fsm_in_state(SystemState.ESCALATE, audit_writer=audit)
    ctx = fsm.event_context
    assert ctx is not None

    # Get the transition that moved to ESCALATE
    transition = None
    for t in reversed(fsm.transition_history):
        if t.to_state == SystemState.ESCALATE:
            transition = t
            break

    packet = generate_escalation_packet(
        ctx=ctx,
        transition=transition,
        audit_logger=audit,
    )

    assert isinstance(packet, EscalationPacket)
    assert len(packet.criticality_reasons) >= 1
    assert packet.packet_hash, "packet_hash should be non-empty"
    assert packet.seismic_magnitude == 7.0  # from make_fsm_in_state

    # Schema round-trip
    EscalationPacket.model_validate(packet.model_dump())


# ===================================================================
# Stage 9b: Human Review Decision Flow
# ===================================================================
def verify_human_review_flow() -> None:
    assert _dart_scenario_result is not None, "Stage 5 must run first"

    # Build a PROVISIONAL FinalAssessment
    verification = VerificationResult(
        producer="verification_agent",
        overall=VerificationOutcome.PASS,
        checks=[
            VerificationCheck(
                name="holdout_station", result=CheckResult.PASS, evidence="OK"
            ),
        ],
        abstain_required=False,
    )

    report_agent = ReportAgent()
    provisional = report_agent.synthesize(
        scenario=_dart_scenario_result,
        verification=verification,
        tier=1,
    )
    assert provisional.status == AssessmentStatus.PROVISIONAL

    # Apply APPROVE decision
    decision = HumanDecision(
        producer="scientist_1",
        reviewer_id="scientist_1",
        decision=ReviewDecision.APPROVE,
        decision_reason="Assessment consistent with observations.",
        decided_at_utc=datetime.now(UTC),
        escalation_packet_id=FIXED_ESCALATION_PACKET_ID,
    )

    approved = format_human_decision(provisional, decision)
    assert approved.status == AssessmentStatus.APPROVED_INTERNAL, (
        f"Expected APPROVED_INTERNAL, got {approved.status}"
    )
    assert len(approved.decision_trace) > len(provisional.decision_trace)


# ===================================================================
# Main
# ===================================================================
def main() -> None:
    stages = [
        ("1.  QC Agent", verify_qc_agent),
        ("2.  Anomaly Detection Agent", verify_anomaly_agent),
        ("3.  FSM Orchestrator", verify_fsm),
        ("4.  Scenario (seismic-only)", verify_scenario_seismic_only),
        ("5.  Scenario (DART-constrained)", verify_scenario_dart_constrained),
        ("6a. Verification (PASS)", verify_verification_pass),
        ("6b. Verification (FAIL/ABSTAIN)", verify_verification_fail),
        ("6c. Verification (INCOMPLETE/ABSTAIN)", verify_verification_incomplete),
        ("7.  Report Agent", verify_report_agent),
        ("8a. Pipeline (PASS path)", verify_pipeline_pass),
        ("8b. Pipeline (ABSTAIN path)", verify_pipeline_abstain),
        ("8c. Pipeline (insufficient)", verify_pipeline_insufficient),
        ("9a. Escalation Packet", verify_escalation_packet),
        ("9b. Human Review Flow", verify_human_review_flow),
    ]

    results: list[StageResult] = []
    for name, func in stages:
        r = run_stage(name, func)
        status = "PASS" if r.passed else "FAIL"
        print(f"  [{status}] {r.name:<40} ({r.elapsed_sec:.3f}s)")
        if not r.passed:
            print(f"         {r.detail}")
        results.append(r)

    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)

    print(f"\n{'=' * 60}")
    print(f"E2E Workflow Verification: {n_passed}/{n_total} passed")
    if n_passed == n_total:
        print("ALL STAGES PASSED")
    else:
        print("SOME STAGES FAILED")
        for r in results:
            if not r.passed:
                print(f"  FAILED: {r.name}")
    print(f"{'=' * 60}")

    sys.exit(0 if n_passed == n_total else 1)


if __name__ == "__main__":
    main()
