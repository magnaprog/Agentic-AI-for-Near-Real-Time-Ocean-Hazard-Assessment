"""Application settings and configuration management.

All configuration is loaded from environment variables via pydantic-settings.
FSM thresholds are versioned configuration - changes require deployment,
not runtime modification (Prohibited Action P8).
"""

from __future__ import annotations

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings

from hazard_assessment.orchestrator.states import ThresholdConfig


class ThresholdSettings(BaseSettings):
    """FSM state transition thresholds.

    These are versioned configuration values. Changing them requires a
    new deployment - runtime modification is prohibited (P8).
    """

    t1: float = Field(default=0.35, description="MONITOR -> INVESTIGATE threshold")
    t2: float = Field(default=0.60, description="INVESTIGATE -> ASSESS threshold")
    t3: float = Field(default=0.85, description="ASSESS -> ESCALATE threshold")
    seismic_min_magnitude: float = Field(
        default=6.0, description="Minimum magnitude for seismic trigger"
    )
    escalation_magnitude: float = Field(
        default=7.5, description="Magnitude threshold for direct escalation"
    )
    monitor_timeout_hours: float = Field(
        default=12.0, description="Hours before MONITOR -> IDLE timeout"
    )
    seismic_escalation_depth_km: float = Field(
        default=100.0,
        description="Max depth (km) for seismic-only MONITOR->ESCALATE",
    )
    basin: str = Field(default="pacific", description="Ocean basin for threshold selection")

    @model_validator(mode="after")
    def _check_threshold_ordering(self) -> ThresholdSettings:
        if not (0.0 < self.t1 < self.t2 < self.t3 <= 1.0):
            raise ValueError(
                f"Thresholds must satisfy 0 < t1 < t2 < t3 <= 1, "
                f"got t1={self.t1}, t2={self.t2}, t3={self.t3}"
            )
        if self.seismic_min_magnitude >= self.escalation_magnitude:
            raise ValueError(
                f"seismic_min_magnitude ({self.seismic_min_magnitude}) must be "
                f"less than escalation_magnitude ({self.escalation_magnitude})"
            )
        return self

    def to_threshold_config(self) -> ThresholdConfig:
        """Convert to the FSM's ThresholdConfig dataclass."""
        return ThresholdConfig(
            basin=self.basin,
            t1=self.t1,
            t2=self.t2,
            t3=self.t3,
            seismic_min_magnitude=self.seismic_min_magnitude,
            escalation_magnitude=self.escalation_magnitude,
            monitor_timeout_hours=self.monitor_timeout_hours,
            seismic_escalation_depth_km=self.seismic_escalation_depth_km,
        )

    model_config = {"env_prefix": "THRESHOLD_"}


class IngestSettings(BaseSettings):
    """Data ingestion configuration.

    DART data access uses HTTP text-file polling (not a REST API).
    Realtime data is available at ndbc.noaa.gov station pages; historical
    data via ndbc.noaa.gov/dart_data.php. CO-OPS uses a REST API at
    api.tidesandcurrents.noaa.gov/api/prod/datagetter. USGS seismic data
    uses the FDSN event web service at earthquake.usgs.gov/fdsnws/event/1/.
    """

    # Standard-mode DART data arrives in 6-hour batches (24 values at
    # 15-min intervals). We poll every 60 seconds NOT to retrieve new
    # standard-mode data, but to detect STANDARD->EVENT mode transitions
    # promptly. When event mode is detected, the system switches to the
    # faster event-mode polling interval below.
    dart_poll_interval_standard_sec: int = Field(
        default=60, description="Polling interval for DART mode-transition detection (seconds)"
    )
    dart_poll_interval_event_sec: int = Field(
        default=15, description="Polling interval for DART event mode (seconds)"
    )
    dart_event_mode_timeout_sec: int = Field(
        default=4 * 60 * 60,
        description=(
            "Seconds to keep DART event mode active without new event-mode rows. "
            "Default 4h aligns with current NDBC guidance and can be overridden "
            "if operations chooses a different cutoff."
        ),
    )
    coops_poll_interval_sec: int = Field(
        default=30,
        description=(
            "Polling interval for CO-OPS data (seconds). Default 30 s "
            "meets the <= 30 s ingest latency SLO."
        ),
    )
    seismic_poll_interval_sec: int = Field(
        default=15,
        description=(
            "Polling interval for USGS earthquake feed (seconds). Default "
            "15 s comfortably meets the <= 30 s ingest latency SLO."
        ),
    )
    retry_max_attempts: int = Field(
        default=3, description="Maximum retry attempts for failed requests"
    )
    retry_backoff_sec: float = Field(
        default=2.0, description="Base backoff interval for retries (seconds)"
    )

    model_config = {"env_prefix": "INGEST_"}


class ConfidenceWeights(BaseSettings):
    """Weights for the deterministic system-confidence formula in ReportAgent.

    ``base = w_verification * outcome + w_stations * station + w_spread * spread``

    These are uncalibrated defaults requiring tuning against historical
    event replay (same caveat as FSM thresholds).
    """

    w_verification: float = Field(
        default=0.40, ge=0.0, le=1.0,
        description="Weight for verification outcome score",
    )
    w_stations: float = Field(
        default=0.25, ge=0.0, le=1.0,
        description="Weight for active-station coverage score",
    )
    w_spread: float = Field(
        default=0.35, ge=0.0, le=1.0,
        description="Weight for ensemble spread score",
    )

    @model_validator(mode="after")
    def _check_sum(self) -> ConfidenceWeights:
        total = self.w_verification + self.w_stations + self.w_spread
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"Confidence weights must sum to 1.0, got {total:.6f}"
            )
        return self

    model_config = {"env_prefix": "CONFIDENCE_"}


class LLMSettings(BaseSettings):
    """LLM service configuration for synthesis and after-action analysis.

    The layer stays off until either an API key or a base URL is set; see
    ``is_enabled``. While it is off the Report Agent uses template-based
    summaries only.
    """

    provider: str = Field(
        default="anthropic",
        description=(
            "Chat-model provider. Validated against the registry in "
            "agents/llm_advisory/providers.py when the client is built, which "
            "is also where the accepted names are listed."
        ),
    )
    model: str = Field(
        default="",
        description="Provider model identifier. Required whenever the layer is enabled.",
    )
    api_key: str = Field(
        default="",
        description="API key. Not needed for an endpoint set through base_url.",
    )
    base_url: str = Field(
        default="",
        description=(
            "Override the provider endpoint. With provider=openai this reaches "
            "any OpenAI-compatible server, including a locally served model."
        ),
    )
    fast_model: str = Field(
        default="",
        description="Cheaper model for evidence/scenario nodes. Falls back to model if empty.",
    )
    quality_model: str = Field(
        default="",
        description="Higher-quality model for after-action analysis. Falls back to model if empty.",
    )
    timeout_sec: int = Field(default=30, description="Per-call timeout in seconds")
    max_retries: int = Field(default=2, ge=1, description="Max retries on transient API errors")

    model_config = {"env_prefix": "LLM_"}

    @property
    def is_enabled(self) -> bool:
        """Whether the operator has asked for a model at all.

        Either an API key for a hosted provider, or a base URL for an endpoint
        they run themselves, where authentication is their concern rather than
        ours. The model identifier is deliberately not part of this test: a key
        set without a model must fail loudly when the client is built, not
        disable the layer in silence.
        """
        return bool(self.api_key or self.base_url)
