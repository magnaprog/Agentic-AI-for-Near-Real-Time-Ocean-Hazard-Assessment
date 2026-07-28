"""Base envelope for inter-agent handoff messages.

Subclasses add domain-specific fields while inheriting versioning,
provenance, and decision traceability. Breaking schema changes require
a version bump.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Any
from uuid import UUID, uuid4

from pydantic import AfterValidator, BaseModel, Field, model_validator


def _check_timezone(v: datetime) -> datetime:
    """Reject naive datetimes - all UTC fields must be timezone-aware."""
    if v.tzinfo is None:
        raise ValueError(
            "datetime must be timezone-aware (naive datetimes are rejected)"
        )
    return v


AwareDatetime = Annotated[datetime, AfterValidator(_check_timezone)]
"""datetime that rejects naive (timezone-less) values at validation time."""

# Default for every BaseEnvelope subclass, so every handoff model carries it.
# Additive changes stay within major version 1; a major bump is reserved for a
# breaking change to an existing field.
SCHEMA_VERSION = "1.0"


class DataSource(StrEnum):
    """Data source identifiers for provenance tracking."""

    DART = "dart"
    COOPS = "coops"
    SEISMIC = "seismic"
    INTERNAL = "internal"


class StepResult(StrEnum):
    """Result classification for decision trace steps."""

    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    INFO = "info"


class InputRef(BaseModel):
    """Reference to an input data record with provenance hash."""

    source: DataSource
    record_id: str = Field(
        pattern=r"^[0-9A-Za-z_.:\-]{1,128}$",
        description="Identifier of the referenced raw record (not prose)",
    )
    sha256: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex hash of the raw payload",
    )

    model_config = {"extra": "forbid"}


class DecisionStep(BaseModel):
    """A single step in the decision trace for auditability."""

    step: str = Field(description="Description of the decision step")
    result: StepResult
    evidence: str = Field(description="Supporting detail or measurement")

    model_config = {"extra": "forbid"}


class BaseEnvelope(BaseModel):
    """Base class for all inter-agent handoff messages."""

    @model_validator(mode="before")
    @classmethod
    def _validate_schema_version(cls, data: Any) -> Any:
        # Rejects incompatible major versions as a final safety check;
        # callers should reject incompatible versions before constructing
        # envelopes. SchemaVersionError raised here gets wrapped by Pydantic
        # as a ValidationError (since it inherits ValueError).
        if isinstance(data, dict) and "schema_version" in data:
            from hazard_assessment.schemas.versioning import check_schema_version

            check_schema_version(data["schema_version"])
        return data

    schema_version: str = Field(default=SCHEMA_VERSION, frozen=True)
    handoff_id: UUID = Field(default_factory=uuid4)
    event_id: UUID | None = Field(
        default=None,
        description="Event ID linking all handoffs for a single seismic event",
    )
    trace_id: UUID | None = Field(
        default=None,
        description="Trace ID correlating all artifacts from a single pipeline execution",
    )
    producer: str = Field(min_length=1, description="Name of the producing agent")
    produced_at_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this handoff was produced",
    )
    input_refs: list[InputRef] = Field(
        default_factory=list,
        description=(
            "References to the input data used to produce this handoff. May be "
            "content-deduplicated by payload hash and capped; a capped "
            "escalation packet sets input_refs_truncated."
        ),
    )
    code_version: str = Field(
        default="",
        description=(
            "Build version identifier of the running code. Producers currently "
            "pass the hazard_assessment package version. Keep it reproducible "
            "from a source archive: this field is folded into both "
            "input_manifest_hash and scientific_content_hash. Those hashes are "
            "not the persist idempotency key, which is "
            "(checkpoint_id, assessment_schema_version); they are the "
            "equivalence check applied after a conflict on that key, so an "
            "environment-derived value turns a benign replay into a conflict."
        ),
    )
    model_version: str = Field(
        default="",
        description="Model or ruleset version identifier",
    )
    decision_trace: list[DecisionStep] = Field(
        default_factory=list,
        description="Step-by-step trace of decisions made during processing",
    )

    model_config = {"extra": "forbid"}
