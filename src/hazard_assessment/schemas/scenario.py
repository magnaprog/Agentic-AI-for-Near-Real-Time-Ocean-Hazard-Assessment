"""Scenario Assessment schema - output of the Scenario Agent.

The Scenario Agent runs NNLS inversion over a unit-source Green's-function
library against incoming DART event-mode waveforms to produce ranked
candidate scenarios with uncertainty.
"""

from __future__ import annotations

from datetime import timedelta
from enum import StrEnum

from pydantic import BaseModel, Field, field_validator, model_validator

from hazard_assessment.schemas.envelope import AwareDatetime, BaseEnvelope


class ConstraintStage(StrEnum):
    """Stage of observational constraint for the scenario assessment."""

    SEISMIC_ONLY = "SEISMIC_ONLY"
    DART_CONSTRAINED = "DART_CONSTRAINED"
    MULTI_STATION = "MULTI_STATION"


class EnsembleSpread(StrEnum):
    """Classification of uncertainty spread across the scenario ensemble."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class RankedScenario(BaseModel):
    """A single ranked scenario from the NNLS inversion."""

    unit_source_ids: list[str] = Field(
        min_length=1, description="Unit source segment identifiers used"
    )
    weights: list[float] = Field(
        min_length=1, description="Non-negative weights from NNLS solution"
    )
    waveform_rmse_cm: float = Field(
        ge=0.0,
        description="Waveform RMSE across constraining stations in cm",
    )
    mw_equivalent: float = Field(
        ge=0.0, le=10.0, description="Moment magnitude equivalent of this scenario"
    )
    rank: int = Field(ge=1, description="Rank position (1 = best fit)")
    posterior_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Posterior weight from bootstrap ensemble",
    )

    @model_validator(mode="after")
    def _check_weights_match_sources(self) -> RankedScenario:
        if len(self.unit_source_ids) != len(self.weights):
            raise ValueError(
                f"unit_source_ids and weights must have the same length "
                f"(got {len(self.unit_source_ids)} sources, {len(self.weights)} weights)"
            )
        if any(w < 0 for w in self.weights):
            raise ValueError(
                "NNLS weights must be non-negative"
            )
        # NaN slips past the comparison above (nan < 0 is False) and would
        # propagate into mw_equivalent and coastal proxies; reject it. NaN is
        # the only float not equal to itself, so no math import is needed.
        if any(w != w for w in self.weights):
            raise ValueError(
                "NNLS weights must be finite (no NaN)"
            )
        return self

    model_config = {"extra": "forbid"}


class CoastalProxy(BaseModel):
    """Coastal amplitude proxy for a specific site."""

    site_id: str = Field(description="Coastal site identifier")
    arrival_utc: AwareDatetime = Field(description="Estimated tsunami arrival time")
    arrival_uncertainty_min: float = Field(
        ge=0.0,
        description="Arrival time uncertainty in minutes",
    )
    amplitude_proxy_p10_m: float = Field(
        ge=0.0,
        description="10th percentile amplitude proxy in meters",
    )
    amplitude_proxy_p50_m: float = Field(
        ge=0.0,
        description="50th percentile (median) amplitude proxy in meters",
    )
    amplitude_proxy_p90_m: float = Field(
        ge=0.0,
        description="90th percentile amplitude proxy in meters",
    )
    tidal_correction_applied: bool = Field(
        description="Whether tidal correction was applied to the proxy"
    )

    @field_validator("arrival_utc")
    @classmethod
    def _enforce_utc(cls, v: AwareDatetime) -> AwareDatetime:
        """Reject non-UTC timezone-aware datetimes.

        The field name ``arrival_utc`` and downstream template formatting
        (literal 'Z' suffix) both assume UTC. AwareDatetime only rejects
        naive datetimes; this validator additionally rejects non-UTC offsets.
        """
        offset = v.utcoffset()
        if offset is None or offset != timedelta(0):
            if offset is None:
                raise ValueError("arrival_utc must be timezone-aware UTC")
            hours = offset.total_seconds() / 3600
            raise ValueError(
                f"arrival_utc must be UTC (got offset {hours:+.1f}h)"
            )
        return v

    @model_validator(mode="after")
    def _check_percentile_ordering(self) -> CoastalProxy:
        if not (self.amplitude_proxy_p10_m <= self.amplitude_proxy_p50_m
                <= self.amplitude_proxy_p90_m):
            raise ValueError(
                "Amplitude proxies must satisfy p10 <= p50 <= p90 "
                f"(got p10={self.amplitude_proxy_p10_m}, "
                f"p50={self.amplitude_proxy_p50_m}, "
                f"p90={self.amplitude_proxy_p90_m})"
            )
        return self

    model_config = {"extra": "forbid"}


INUNDATION_DISCLAIMER = (
    "These are open-ocean amplitude proxies, not inundation depths or run-up estimates."
)


class ScenarioAssessment(BaseEnvelope):
    """Scenario Agent output: ranked scenarios with uncertainty."""

    type: str = Field(default="ScenarioAssessment", frozen=True)
    method: str = Field(
        default="NNLS_UNIT_SOURCE",
        description="Inversion method identifier",
    )
    constraint_stage: ConstraintStage = Field(
        description="Current stage of observational constraint"
    )
    dart_stations_used: list[str] = Field(
        default_factory=list,
        description="DART stations included in the inversion",
    )
    dart_stations_excluded: list[str] = Field(
        default_factory=list,
        description="DART stations excluded from the inversion",
    )
    exclusion_reasons: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Reason for excluding each station (keyed by station_id). "
            "Values must be non-empty."
        ),
    )
    inversion_window_sec: int = Field(
        ge=0,
        description="Length of the waveform window used for inversion in seconds",
    )
    top_scenarios: list[RankedScenario] = Field(
        min_length=1,
        description="Ranked candidate scenarios from the inversion",
    )
    coastal_proxies: list[CoastalProxy] = Field(
        default_factory=list,
        description="Coastal amplitude proxies for monitored sites",
    )
    ensemble_spread: EnsembleSpread = Field(
        description="Classification of bootstrap uncertainty spread"
    )
    bilateral_rupture_evaluated: bool = Field(
        description="Whether bilateral rupture scenarios were considered"
    )
    inundation_disclaimer: str = Field(
        default=INUNDATION_DISCLAIMER,
        frozen=True,
        description="Mandatory disclaimer about output limitations",
    )
    limiting_assumptions: list[str] = Field(
        default_factory=list,
        description="All assumptions limiting this assessment",
    )

    @model_validator(mode="after")
    def _check_constraint_stage_consistency(self) -> ScenarioAssessment:
        """Enforce that dart_stations_used is consistent with constraint_stage."""
        n = len(self.dart_stations_used)
        if self.constraint_stage == ConstraintStage.SEISMIC_ONLY and n > 0:
            raise ValueError(
                f"SEISMIC_ONLY stage must have empty dart_stations_used (got {n})"
            )
        if self.constraint_stage == ConstraintStage.DART_CONSTRAINED and n < 1:
            raise ValueError(
                f"DART_CONSTRAINED stage requires >= 1 DART station (got {n})"
            )
        if self.constraint_stage == ConstraintStage.MULTI_STATION and n < 2:
            raise ValueError(
                f"MULTI_STATION stage requires >= 2 DART stations (got {n})"
            )
        return self

    @model_validator(mode="after")
    def _check_exclusion_reasons_match_excluded(self) -> ScenarioAssessment:
        """Ensure exclusion_reasons keys exactly match dart_stations_excluded.

        Also validates that no reason string is empty - an exclusion must
        always carry a human-readable justification.
        """
        excluded = set(self.dart_stations_excluded)
        reasons = set(self.exclusion_reasons.keys())
        if reasons != excluded:
            missing = excluded - reasons
            extra = reasons - excluded
            parts = []
            if missing:
                parts.append(f"missing reasons for {missing}")
            if extra:
                parts.append(f"reasons for non-excluded stations {extra}")
            raise ValueError(
                "exclusion_reasons must match dart_stations_excluded: "
                + ", ".join(parts)
            )
        empty = [k for k, v in self.exclusion_reasons.items() if not v.strip()]
        if empty:
            raise ValueError(
                f"exclusion_reasons values must be non-empty: {empty}"
            )
        return self

    @field_validator("inundation_disclaimer")
    @classmethod
    def _enforce_inundation_disclaimer(cls, v: str) -> str:
        if v != INUNDATION_DISCLAIMER:
            raise ValueError(
                "inundation_disclaimer must be the standard text. "
                "Custom disclaimer text is not permitted."
            )
        return v

    model_config = {"extra": "forbid"}
