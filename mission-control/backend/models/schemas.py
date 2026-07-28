"""Pydantic response models for the Mission Control BFF."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventContextOut(BaseModel):
    """Serialisable snapshot of the FSM EventContext."""

    event_id: str
    seismic_magnitude: float
    seismic_region: str
    epicenter_lat: float
    epicenter_lon: float
    trigger_time_utc: datetime
    latest_anomaly_score: float
    dart_confirmation: bool
    active_dart_stations: list[str]
    stations_in_event_mode: list[str]


class ThresholdOut(BaseModel):
    basin: str
    t1: float
    t2: float
    t3: float


class TransitionOut(BaseModel):
    transition_id: str
    event_id: str | None = None
    timestamp_utc: datetime
    from_state: str
    to_state: str
    trigger_reason: str
    anomaly_score: float | None = None
    seismic_magnitude: float | None = None


class FSMStateOut(BaseModel):
    """Full FSM snapshot returned to the dashboard."""

    fsm_state: str
    has_active_event: bool
    recovery_failed: bool = False
    #: True when fewer than two DART stations carry QC-usable data in the
    #: worker's retained window, which is below the minimum for triangulation.
    #: Forwarded from the core /api/fsm response so the console can tell an
    #: operator that the score on screen rests on degraded coverage.
    sensor_degraded: bool = False
    event_context: EventContextOut | None = None
    thresholds: ThresholdOut
    transition_history: list[TransitionOut] = Field(default_factory=list)


class AuditEntryOut(BaseModel):
    entry_id: str
    timestamp_utc: datetime
    event_id: str | None = None
    event_type: str
    producer: str
    data: dict[str, Any] = Field(default_factory=dict)


class AgentOut(BaseModel):
    name: str
    version: str
    execution_path: str
    description: str


class DetectionLatencyRowOut(BaseModel):
    """Per-station time-to-detection for a retrospective event.

    t1_minutes / t3_minutes are minutes after earthquake origin at which the
    station's ensemble score first crossed T1 / T3. None means the threshold
    was never crossed in the analysis window.
    """

    station_id: str
    distance_km: float
    t1_minutes: float | None = None
    t3_minutes: float | None = None


class AblationRowOut(BaseModel):
    """One ensemble-configuration row of a retrospective ablation summary."""

    configuration: str
    t3_hits: str  # e.g. "7/8" stations crossing T3
    peak_score: float


class ScenarioMetricsOut(BaseModel):
    """Retrospective, demo-only enrichment derived from the validated offline
    results (results/*.json).

    Absent in live operation: the deployed worker does not emit per-station
    detection timing or ensemble ablation summaries, so the live snapshot
    leaves this null and the dashboard hides the corresponding panels.
    """

    first_t1_minutes: float | None = None
    detection_latency: list[DetectionLatencyRowOut] = Field(default_factory=list)
    ensemble_ablation: list[AblationRowOut] = Field(default_factory=list)


class SystemSnapshotOut(BaseModel):
    """One WebSocket snapshot of system state.

    Demo mode does not build this model. It broadcasts the built-in Tohoku dict
    verbatim with ``demo_mode: true`` merged in, which is why the frontend type
    carries an optional ``demo_mode`` that is absent from this model: the field
    exists on the wire only when no core API key is configured. Live snapshots
    never carry it, and absence is what the console reads as "not demo".
    """

    fsm: FSMStateOut
    agents: list[dict[str, Any]] = Field(default_factory=list)
    recent_audit: list[AuditEntryOut] = Field(default_factory=list)
    #: Review decisions only, queried separately because recent_audit can be
    #: saturated by per-window anomaly entries within a fraction of a second.
    recent_reviews: list[AuditEntryOut] = Field(default_factory=list)
    scenario_metrics: ScenarioMetricsOut | None = None


class ReviewDecisionIn(BaseModel):
    """Caller-gated review bound to the durable packet of record."""

    event_id: str
    decision: Literal["APPROVE", "REJECT", "DEFER"]
    decision_reason: str = Field(min_length=1, max_length=5000)
    escalation_packet_row_id: int = Field(ge=1)
    escalation_packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
