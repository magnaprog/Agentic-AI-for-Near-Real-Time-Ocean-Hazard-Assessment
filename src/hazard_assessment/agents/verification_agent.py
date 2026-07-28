"""Verification Agent - validates scenario output before report generation.

The system's honesty mechanism. Prevents output from being presented with
more confidence than the evidence supports.

"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.agents.verification_checks import (
    VerificationInput,
    determine_outcome,
    run_all_checks,
)
from hazard_assessment.schemas.envelope import DecisionStep, StepResult
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)

_MANIFEST = AgentManifest(
    name="verification_agent",
    version="0.2.0",
    capabilities=[
        AgentCapability.READ_DATA,
        AgentCapability.WRITE_DATA,
        AgentCapability.WRITE_AUDIT,
        AgentCapability.PRODUCE_KAFKA,
        AgentCapability.CONSUME_KAFKA,
    ],
    description=(
        "Validates scenario output and enforces the ABSTAIN path "
        "when evidence is insufficient"
    ),
)

# Map CheckResult to StepResult for decision trace
_CHECK_TO_STEP: dict[CheckResult, StepResult] = {
    CheckResult.PASS: StepResult.PASS,
    CheckResult.CONCERN: StepResult.WARN,
    CheckResult.FAIL: StepResult.FAIL,
    # A check that could not run on missing prerequisites is a warning
    # in the trace; whether it blocks output is the aggregation's call.
    CheckResult.NOT_EVALUATED: StepResult.WARN,
    # Matrix-inapplicable checks carry no verdict at all.
    CheckResult.NOT_APPLICABLE: StepResult.INFO,
    # An errored check failed to produce its verdict.
    CheckResult.ERROR: StepResult.FAIL,
}

# Map VerificationOutcome to StepResult for outcome policy trace step
_OUTCOME_TO_STEP: dict[VerificationOutcome, StepResult] = {
    VerificationOutcome.PASS: StepResult.PASS,
    VerificationOutcome.PASS_WITH_CONCERNS: StepResult.WARN,
    VerificationOutcome.FAIL: StepResult.FAIL,
    VerificationOutcome.INCOMPLETE: StepResult.FAIL,
}


class VerificationAgent(BaseAgent):
    """Verification Agent.

    Runs holdout-station validation, sensitivity analysis, data coverage
    assessment, and other checks. A FAIL verdict triggers the ABSTAIN
    path - the system will not produce probabilistic guidance when
    verification fails.
    """

    def __init__(self) -> None:
        super().__init__(manifest=_MANIFEST)

    def verify(
        self,
        verification_input: VerificationInput,
    ) -> VerificationResult:
        """Run all verification checks and produce a VerificationResult.

        Args:
            verification_input: All data needed for the 9 checks.

        Returns:
            VerificationResult envelope with check details and outcome.
        """
        checks = run_all_checks(verification_input)
        outcome, abstain_required, abstain_reason = determine_outcome(checks)
        return self.build_result(
            checks=checks,
            outcome=outcome,
            abstain_required=abstain_required,
            abstain_reason=abstain_reason,
            event_id=verification_input.scenario.event_id,
        )

    def build_result(
        self,
        *,
        checks: list[VerificationCheck],
        outcome: VerificationOutcome,
        abstain_required: bool,
        abstain_reason: str | None,
        event_id: UUID | None = None,
        processing_time: datetime | None = None,
    ) -> VerificationResult:
        """Construct a VerificationResult envelope with decision trace.

        Each check maps to one DecisionStep. A final step records
        the outcome policy decision.
        """
        trace: list[DecisionStep] = []
        for check in checks:
            trace.append(
                DecisionStep(
                    step=check.name,
                    result=_CHECK_TO_STEP[check.result],
                    evidence=check.evidence,
                )
            )

        # Outcome policy step
        if outcome == VerificationOutcome.FAIL:
            policy_evidence = f"FAIL: abstain required - {abstain_reason}"
        elif outcome == VerificationOutcome.INCOMPLETE:
            policy_evidence = f"INCOMPLETE: abstain required - {abstain_reason}"
        elif outcome == VerificationOutcome.PASS_WITH_CONCERNS:
            flagged = [
                c.name
                for c in checks
                if c.result
                in (CheckResult.CONCERN, CheckResult.NOT_EVALUATED, CheckResult.ERROR)
            ]
            policy_evidence = f"PASS_WITH_CONCERNS: {flagged}"
        else:
            policy_evidence = "All applicable checks passed"

        trace.append(
            DecisionStep(
                step="outcome_policy",
                result=_OUTCOME_TO_STEP[outcome],
                evidence=policy_evidence,
            )
        )

        return VerificationResult(
            producer=self.manifest.name,
            produced_at_utc=processing_time or datetime.now(UTC),
            event_id=event_id,
            decision_trace=trace,
            overall=outcome,
            checks=checks,
            abstain_required=abstain_required,
            abstain_reason=abstain_reason,
        )
