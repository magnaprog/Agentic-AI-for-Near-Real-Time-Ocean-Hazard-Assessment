"""Human Decision schema - recorded at the Human Review Gate.

The duty scientist reviews the assessment and records a decision with
rationale. This decision is immutable once recorded in the audit log.

Decision must be persisted with reviewer_id, timestamp, decision
hash. A decision CANNOT be submitted without viewing the escalation
packet first (escalation_packet_id is required).
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from uuid import UUID

from pydantic import Field, model_validator

from hazard_assessment.schemas.envelope import AwareDatetime, BaseEnvelope


class ReviewDecision(StrEnum):
    """Human reviewer decision options."""

    APPROVE = "APPROVE"
    REJECT = "REJECT"
    DEFER = "DEFER"


class IdentityAssurance(StrEnum):
    """How strongly the reviewer identity on a decision is established.

    CALLER_ASSERTED: the identity came from a caller-supplied header
    behind a shared service API key. It can be audited but does not
    establish an individual human principal, so it can satisfy neither
    the mandatory human distribution gate nor event disposition
    authority on its own.

    TRUSTED_HUMAN_PRINCIPAL: an authenticated and authorized individual
    human, established by a real authentication mechanism. No current
    code path can produce this value; it exists so records are honest
    about what they prove and so the future authentication integration
    has a stable vocabulary.
    """

    CALLER_ASSERTED = "CALLER_ASSERTED"
    TRUSTED_HUMAN_PRINCIPAL = "TRUSTED_HUMAN_PRINCIPAL"


class AssessmentReviewDecision(BaseEnvelope):
    """Caller-gated assessment review bound to immutable durable evidence.

    This is not an event disposition and never changes FSM state. Current API
    records are always CALLER_ASSERTED, so even APPROVE does not authorize
    distribution or event closure. Trusted authority remains a separate,
    unimplemented contract.
    """

    type: str = Field(default="AssessmentReviewDecision", frozen=True)
    reviewer_id: str = Field(min_length=1)
    identity_assurance: IdentityAssurance = Field(
        default=IdentityAssurance.CALLER_ASSERTED
    )
    decision: ReviewDecision
    decision_reason: str = Field(min_length=1)
    decided_at_utc: AwareDatetime
    escalation_packet_row_id: int = Field(ge=1)
    escalation_packet_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    assessment_row_id: int = Field(ge=1)
    assessment_id: UUID
    assessment_scientific_content_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$"
    )
    decision_hash: str = Field(default="", pattern=r"^$|^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _compute_decision_hash(self) -> AssessmentReviewDecision:
        if not self.decision_hash:
            object.__setattr__(self, "decision_hash", self._compute_hash())
        return self

    def _compute_hash(self) -> str:
        payload = {
            "reviewer_id": self.reviewer_id,
            "identity_assurance": self.identity_assurance.value,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "decided_at_utc": self.decided_at_utc.isoformat(),
            "escalation_packet_row_id": self.escalation_packet_row_id,
            "escalation_packet_hash": self.escalation_packet_hash,
            "assessment_row_id": self.assessment_row_id,
            "assessment_id": str(self.assessment_id),
            "assessment_scientific_content_hash": (
                self.assessment_scientific_content_hash
            ),
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    model_config = {"extra": "forbid"}


class HumanDecision(BaseEnvelope):
    """Human Review Gate output: recorded decision with rationale.

    Inherits BaseEnvelope for schema versioning, handoff_id traceability,
    and audit trail integration - not because it is an agent-to-agent handoff.

    Immutable once persisted to the audit trail. No Tier 2 report can be
    distributed without passing through this gate with an APPROVE decision.

    The escalation_packet_id field links this decision to the evidence
    bundle the reviewer saw. The API layer (not this schema) enforces
    that the ID matches the active escalation packet.

    The identity_assurance field records honestly how the reviewer
    identity was established. Every record the current API produces is
    CALLER_ASSERTED: a caller-gated record, not authenticated human
    attribution.
    """

    type: str = Field(default="HumanDecision", frozen=True)
    reviewer_id: str = Field(min_length=1, description="Identifier of the reviewing scientist")
    identity_assurance: IdentityAssurance = Field(
        default=IdentityAssurance.CALLER_ASSERTED,
        description=(
            "How the reviewer identity was established. The current API "
            "accepts a caller-supplied header behind a shared service "
            "key, so every record it produces is CALLER_ASSERTED."
        ),
    )
    decision: ReviewDecision = Field(description="Review decision")
    decision_reason: str = Field(
        min_length=1,
        description="Rationale for the decision (mandatory; empty strings rejected)",
    )
    decided_at_utc: AwareDatetime = Field(description="UTC timestamp of the decision")
    escalation_packet_id: UUID = Field(
        description=(
            "The EscalationPacket's handoff_id (UUID). Must match the "
            "active escalation packet; enforces that the reviewer viewed "
            "the packet before deciding."
        ),
    )
    decision_hash: str = Field(
        default="",
        description=(
            "SHA-256 hash of (reviewer_id, identity_assurance, decision, "
            "decision_reason, decided_at_utc, escalation_packet_id) for "
            "tamper detection."
        ),
    )

    @model_validator(mode="after")
    def _compute_decision_hash(self) -> HumanDecision:
        """Auto-compute decision hash on construction if not provided."""
        if not self.decision_hash:
            h = self._compute_hash()
            object.__setattr__(self, "decision_hash", h)
        return self

    def _compute_hash(self) -> str:
        """Compute SHA-256 hash of the decision-critical fields only.

        Inherited BaseEnvelope fields (producer, code_version, etc.) are
        intentionally excluded - the decision hash proves what the reviewer
        decided, independent of which code version recorded it.
        """
        payload = {
            "reviewer_id": self.reviewer_id,
            "identity_assurance": self.identity_assurance.value,
            "decision": self.decision.value,
            "decision_reason": self.decision_reason,
            "decided_at_utc": self.decided_at_utc.isoformat(),
            "escalation_packet_id": str(self.escalation_packet_id),
        }
        canonical = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    model_config = {"extra": "forbid"}
