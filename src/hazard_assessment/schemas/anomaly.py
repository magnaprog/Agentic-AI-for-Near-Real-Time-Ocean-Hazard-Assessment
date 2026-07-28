"""Anomaly Assessment schema - output of the Anomaly Agent.

The Anomaly Agent computes detided residuals, applies bandpass filtering,
runs multi-scale detection, and checks cross-station spatial coherence.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from hazard_assessment.schemas.envelope import AwareDatetime, BaseEnvelope


class ScoreComponents(BaseModel):
    """Component-level anomaly scores for transparency."""

    threshold: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Detection-threshold score"
            " (prediction-residual for DART, amplitude-threshold for CO-OPS)"
        ),
    )
    statistical: float = Field(ge=0.0, le=1.0, description="Wavelet/BOCPD statistical score")
    ml: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Isolation Forest ML score (None if model unavailable)",
    )

    model_config = {"extra": "forbid"}


class SpatialConfirmation(BaseModel):
    """Cross-station spatial coherence check result."""

    station_id: str = Field(
        min_length=1,
        description="Station that was checked for coherent arrival",
    )
    expected_arrival_utc: AwareDatetime = Field(
        description="Predicted arrival time based on wave speed"
    )
    observed_arrival_utc: AwareDatetime | None = Field(
        default=None,
        description="Actual observed arrival time (None if not yet observed)",
    )
    confirmed: bool = Field(description="Whether the arrival was confirmed within tolerance")
    delta_min: float | None = Field(
        default=None,
        description="Difference between expected and observed arrival in minutes",
    )

    @model_validator(mode="after")
    def _confirmed_requires_observation(self) -> SpatialConfirmation:
        """A confirmed arrival must have an observed time and delta."""
        if self.confirmed and self.observed_arrival_utc is None:
            raise ValueError(
                "confirmed=True requires observed_arrival_utc to be set"
            )
        if self.confirmed and self.delta_min is None:
            raise ValueError(
                "confirmed=True requires delta_min to be set"
            )
        return self

    model_config = {"extra": "forbid"}


class AnomalyAssessment(BaseEnvelope):
    """Anomaly Agent output: ensemble anomaly score and context.

    The orchestrator FSM uses anomaly_score for deterministic state transitions.
    """

    type: str = Field(default="AnomalyAssessment", frozen=True)
    # Design note: current_state and state_changed are set by the Orchestrator
    # during the orchestrate pipeline node, *after* the Anomaly Agent creates
    # this envelope. This post-creation mutation is intentional - it keeps the
    # persisted envelope self-contained (carries the FSM context alongside the
    # score) without requiring the downstream consumer to join against
    # PipelineState. The mutation occurs once, before downstream handoff.
    # PipelineState also carries fsm_state/state_changed for pipeline routing.
    current_state: str = Field(
        default="",
        description=(
            "FSM state after this assessment. Set by the Orchestrator "
            "during the orchestrate node, not by the Anomaly Agent."
        ),
    )
    state_changed: bool = Field(
        default=False,
        description=(
            "Whether a state transition occurred. Set by the Orchestrator "
            "during the orchestrate node, not by the Anomaly Agent."
        ),
    )
    anomaly_score: float = Field(
        ge=0.0,
        le=1.0,
        description="Ensemble anomaly score (weighted combination of components)",
    )
    score_components: ScoreComponents = Field(
        description="Component-level scores for transparency"
    )
    triggering_stations: list[str] = Field(
        default_factory=list,
        description=(
            "Station IDs whose scored data reached the configured T1 "
            "ensemble threshold (inclusive). Empty when the assessment "
            "score is below T1."
        ),
    )
    scored_stations: list[str] = Field(
        default_factory=list,
        description=(
            "Station IDs whose data windows were scored for this assessment, "
            "regardless of whether any threshold was reached."
        ),
    )
    spatial_confirmations: list[SpatialConfirmation] = Field(
        default_factory=list,
        description="Cross-station coherence check results",
    )
    seismic_quiet: bool = Field(
        description=(
            "True when no M>=6.5 event occurred in the last 90 minutes "
            "(seismically quiet period). In this state the detection threshold "
            "is raised by 1.3x, making detection harder to reduce false alarms "
            "from non-tectonic sources. False when a recent large earthquake "
            "is present; thresholds remain at their baseline values."
        ),
    )
    meteotsunami_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Score indicating likelihood of meteorological (non-tectonic) "
            "source. None means the meteotsunami discriminator has not been "
            "evaluated; the current pipeline does not compute it."
        ),
    )
    stations_offline: list[str] = Field(
        default_factory=list,
        description="Station IDs known to be offline",
    )
    filter_degraded: bool = Field(
        default=False,
        description=(
            "True when the bandpass filter is degraded due to a Nyquist violation "
            "(sampling_interval >= 1/(2*BANDPASS_HIGH_HZ) ~ 150 s). The passband "
            "upper edge is clamped to Nyquist, so threshold_score and wavelet_score "
            "are still computed but on a clamped, aliased signal and are unreliable "
            "in either direction. Downstream consumers should treat the anomaly "
            "score with reduced confidence."
        ),
    )
    coverage_note: str = Field(
        default="",
        description="Human-readable note about station coverage",
    )
    reasoning_trace: str = Field(
        default="",
        description="Step-by-step deterministic logic trace",
    )
    rayleigh_wave_suspect: bool | None = Field(
        default=None,
        description=(
            "True when a DART pressure excursion falls within the expected "
            "Rayleigh wave arrival window from a known earthquake epicenter "
            "(haversine distance / 3.6 km/s +/- 20%). Indicates potential "
            "false event-mode trigger requiring additional scrutiny. None "
            "means the check was not evaluated because its prerequisites "
            "(station coordinates and a firing detector) were unavailable."
        ),
    )

    model_config = {"extra": "forbid"}
