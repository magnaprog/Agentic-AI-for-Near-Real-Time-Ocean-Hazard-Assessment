"""Final Assessment schema - the terminal output of the assessment pipeline.

Represents the final status of an event assessment after all agents have
processed and (for Tier 2) the human reviewer has made a decision.
"""

from __future__ import annotations

from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field, field_validator

from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER
from hazard_assessment.schemas.envelope import BaseEnvelope


class AssessmentStatus(StrEnum):
    """Final assessment status classification."""

    PROVISIONAL = "PROVISIONAL"
    ABSTAIN = "ABSTAIN"
    APPROVED_INTERNAL = "APPROVED_INTERNAL"


class ConfidenceLevel(StrEnum):
    """Confidence level classification for the assessment."""

    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"


class UncertaintyInfo(BaseModel):
    """Structured uncertainty information for the assessment."""

    confidence_level: ConfidenceLevel = Field(description="Overall confidence classification")
    key_uncertainties: list[str] = Field(
        default_factory=list,
        description="Specific sources of uncertainty in this assessment",
    )

    model_config = {"extra": "forbid"}


class FinalAssessment(BaseEnvelope):
    """Terminal output of the assessment pipeline.

    Every instance links to a provenance bundle for full traceability.
    """

    type: str = Field(default="FinalAssessment", frozen=True)
    status: AssessmentStatus = Field(description="Final assessment status")
    report_tier: int = Field(
        ge=1,
        le=3,
        description="Report tier (1=Technical Brief, 2=Situational Summary, 3=Post-Event)",
    )
    summary: str = Field(
        description=(
            "Deterministic template report text. Identical whether LLM "
            "synthesis is enabled or not: optional model "
            "narrative lives in model_commentary, never here."
        )
    )
    model_commentary: str | None = Field(
        default=None,
        description=(
            "Optional LLM-synthesized narrative, stored separately from "
            "the deterministic summary. Advisory only; carries "
            "the non-authoritative disclaimer and passes the guardrail "
            "scan or is dropped."
        ),
    )
    uncertainty: UncertaintyInfo = Field(description="Uncertainty information")
    # NOTE: The disclaimer text is also embedded in every `summary` string
    # (by template rendering) and appended to `model_commentary` when LLM
    # synthesis runs. This dedicated field
    # exists for two reasons: (1) machine-readable extraction by downstream
    # consumers (audit systems, compliance checks) without parsing free text,
    # and (2) schema-level guarantee via the frozen validator that the disclaimer
    # cannot be omitted or modified. The apparent redundancy is intentional.
    disclaimer: str = Field(
        default=NON_AUTHORITATIVE_DISCLAIMER,
        frozen=True,
        description="Mandatory non-authoritative disclaimer (machine-readable copy for audit)",
    )
    provenance_bundle_id: UUID = Field(
        description="UUID linking to the full lineage record for this assessment"
    )
    system_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "Deterministic system-level confidence score (0.0-1.0) computed "
            "from verification outcome, station count, ensemble spread, and "
            "Rayleigh wave flag. Computed on every ReportAgent.synthesize() "
            "path regardless of whether LLM synthesis is enabled; None only "
            "on paths that never invoke synthesize() (e.g. ABSTAIN)."
        ),
    )
    llm_synthesis_used: bool = Field(
        default=False,
        description=(
            "True when the LangGraph synthesis graph produced "
            "model_commentary. The deterministic summary is unaffected "
            "either way."
        ),
    )
    rayleigh_wave_suspect: bool = Field(
        default=False,
        description=(
            "Propagated from AnomalyAssessment. True when a DART pressure "
            "excursion falls within the expected Rayleigh wave arrival window."
        ),
    )

    @field_validator("disclaimer")
    @classmethod
    def _enforce_disclaimer(cls, v: str) -> str:
        if v != NON_AUTHORITATIVE_DISCLAIMER:
            raise ValueError(
                "disclaimer must be the standard non-authoritative text. "
                "Custom disclaimer text is not permitted."
            )
        return v

    model_config = {"extra": "forbid"}
