"""Verification Result schema - output of the Verification Agent.

The Verification Agent is the system's honesty mechanism. It prevents
scenario output from being presented with more confidence than the evidence
supports, and enforces the ABSTAIN path when data is insufficient.

Applicability and prerequisite semantics:
each check carries an applicability assigned by the versioned requirement
matrix (never inferred by check code from prose evidence) and a typed
prerequisite status. Only four combinations are valid:

| Applicability          | Prerequisite   | Allowed result            |
|------------------------|----------------|---------------------------|
| NOT_APPLICABLE         | NOT_REQUIRED   | NOT_APPLICABLE            |
| REQUIRED or OPTIONAL   | AVAILABLE      | PASS, CONCERN, or FAIL    |
| REQUIRED or OPTIONAL   | MISSING        | NOT_EVALUATED             |
| REQUIRED or OPTIONAL   | ERROR          | ERROR                     |

The schema validator rejects every other combination. Overall outcomes
are PASS, PASS_WITH_CONCERNS, FAIL, and INCOMPLETE; FAIL and INCOMPLETE
both require ABSTAIN.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator

from hazard_assessment.schemas.envelope import BaseEnvelope


class VerificationOutcome(StrEnum):
    """Overall verification verdict."""

    PASS = "PASS"
    PASS_WITH_CONCERNS = "PASS_WITH_CONCERNS"
    FAIL = "FAIL"
    INCOMPLETE = "INCOMPLETE"


class CheckApplicability(StrEnum):
    """Whether a check applies at the current constraint stage.

    Assigned by the versioned requirement matrix, never inferred by
    individual check code from prose evidence.
    """

    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class PrerequisiteStatus(StrEnum):
    """Typed status of a check's input prerequisites."""

    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    ERROR = "ERROR"
    NOT_REQUIRED = "NOT_REQUIRED"


class CheckResult(StrEnum):
    """Result of an individual verification check."""

    PASS = "PASS"
    CONCERN = "CONCERN"
    FAIL = "FAIL"
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    ERROR = "ERROR"


# The only evaluated results allowed when prerequisites are AVAILABLE.
_EVALUATED_RESULTS = frozenset(
    {CheckResult.PASS, CheckResult.CONCERN, CheckResult.FAIL}
)


class VerificationCheck(BaseModel):
    """A single verification check with its result and evidence.

    Defaults (REQUIRED, AVAILABLE) describe an applicable check that
    ran on available data; the validity-table validator then restricts
    the result to PASS, CONCERN, or FAIL.
    """

    name: str = Field(description="Check identifier (e.g., holdout_station_validation)")
    result: CheckResult = Field(description="Check outcome")
    evidence: str = Field(description="Supporting detail or measurement")
    applicability: CheckApplicability = Field(
        default=CheckApplicability.REQUIRED,
        description="Requirement-matrix applicability at the current stage",
    )
    prerequisite: PrerequisiteStatus = Field(
        default=PrerequisiteStatus.AVAILABLE,
        description="Typed status of the check's input prerequisites",
    )

    @model_validator(mode="after")
    def _enforce_validity_table(self) -> VerificationCheck:
        """Reject every combination outside the validity table."""
        if self.applicability == CheckApplicability.NOT_APPLICABLE:
            if self.prerequisite != PrerequisiteStatus.NOT_REQUIRED:
                raise ValueError(
                    f"check {self.name!r}: NOT_APPLICABLE requires "
                    f"prerequisite NOT_REQUIRED, got {self.prerequisite}"
                )
            if self.result != CheckResult.NOT_APPLICABLE:
                raise ValueError(
                    f"check {self.name!r}: NOT_APPLICABLE requires "
                    f"result NOT_APPLICABLE, got {self.result}"
                )
            return self

        # REQUIRED or OPTIONAL rows.
        if self.prerequisite == PrerequisiteStatus.NOT_REQUIRED:
            raise ValueError(
                f"check {self.name!r}: prerequisite NOT_REQUIRED is only "
                "valid for NOT_APPLICABLE checks"
            )
        if self.prerequisite == PrerequisiteStatus.AVAILABLE:
            if self.result not in _EVALUATED_RESULTS:
                raise ValueError(
                    f"check {self.name!r}: AVAILABLE prerequisite requires "
                    f"result PASS, CONCERN, or FAIL, got {self.result}"
                )
        elif self.prerequisite == PrerequisiteStatus.MISSING:
            if self.result != CheckResult.NOT_EVALUATED:
                raise ValueError(
                    f"check {self.name!r}: MISSING prerequisite requires "
                    f"result NOT_EVALUATED, got {self.result}"
                )
        elif self.prerequisite == PrerequisiteStatus.ERROR:
            if self.result != CheckResult.ERROR:
                raise ValueError(
                    f"check {self.name!r}: ERROR prerequisite requires "
                    f"result ERROR, got {self.result}"
                )
        return self

    model_config = {"extra": "forbid"}


class VerificationResult(BaseEnvelope):
    """Verification Agent output: verification verdict with evidence.

    Contains the overall verdict (PASS / PASS_WITH_CONCERNS / FAIL /
    INCOMPLETE) and detailed results for each verification check. FAIL
    and INCOMPLETE verdicts trigger the ABSTAIN path - the system will
    not produce probabilistic coastal guidance when verification fails
    or cannot be completed.
    """

    type: str = Field(default="VerificationResult", frozen=True)
    overall: VerificationOutcome = Field(description="Overall verification verdict")
    checks: list[VerificationCheck] = Field(
        min_length=1,
        description="Individual check results with evidence",
    )
    abstain_required: bool = Field(
        description="Whether the ABSTAIN path must be triggered"
    )
    abstain_reason: str | None = Field(
        default=None,
        description="Explanation of why ABSTAIN is required (None if not required)",
    )

    @model_validator(mode="after")
    def _enforce_abstain_invariants(self) -> VerificationResult:
        """Derive the aggregate from detailed checks and enforce ABSTAIN.

        A caller-supplied aggregate must not contradict the detailed evidence:
        routing trusts ``overall``, so accepting PASS beside a REQUIRED FAIL or
        missing prerequisite would fail open. The derivation mirrors the
        requirement-matrix aggregation in ``verification_checks.determine_outcome``.
        """
        applicable = [
            check
            for check in self.checks
            if check.applicability is not CheckApplicability.NOT_APPLICABLE
        ]
        if any(check.result is CheckResult.FAIL for check in applicable):
            derived = VerificationOutcome.FAIL
        elif not applicable or any(
            check.applicability is CheckApplicability.REQUIRED
            and check.prerequisite
            in (PrerequisiteStatus.MISSING, PrerequisiteStatus.ERROR)
            for check in applicable
        ):
            derived = VerificationOutcome.INCOMPLETE
        elif any(check.result is CheckResult.CONCERN for check in applicable) or any(
            check.applicability is CheckApplicability.OPTIONAL
            and check.prerequisite
            in (PrerequisiteStatus.MISSING, PrerequisiteStatus.ERROR)
            for check in applicable
        ):
            derived = VerificationOutcome.PASS_WITH_CONCERNS
        else:
            derived = VerificationOutcome.PASS

        if self.overall is not derived:
            raise ValueError(
                f"overall {self.overall.value} contradicts detailed checks; "
                f"derived outcome is {derived.value}"
            )

        derived_abstain = derived in (
            VerificationOutcome.FAIL,
            VerificationOutcome.INCOMPLETE,
        )
        if self.abstain_required is not derived_abstain:
            raise ValueError(
                f"abstain_required={self.abstain_required} contradicts "
                f"derived outcome {derived.value}"
            )

        if self.abstain_required and (
            not self.abstain_reason or not self.abstain_reason.strip()
        ):
            raise ValueError(
                "abstain_reason is required when abstain_required=True "
                "(must be non-empty and not whitespace-only)."
            )
        return self

    model_config = {"extra": "forbid"}
