"""Assessment formatter - status transitions and typed ABSTAIN envelopes.

Pure functions, no I/O. Bridges report generation and human
decision capture by applying status transitions to
FinalAssessment envelopes.

``format_human_decision()`` updates PROVISIONAL -> APPROVED_INTERNAL
on APPROVE, or appends an INFO trace step on REJECT/DEFER.

``format_abstain()`` produces a fully validated FinalAssessment with
status=ABSTAIN when verification fails or is absent.

"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.envelope import DecisionStep, StepResult
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.human_decision import HumanDecision, ReviewDecision

# ---------------------------------------------------------------------------
# ABSTAIN template
# ---------------------------------------------------------------------------

_ABSTAIN_SUMMARY_TEMPLATE = """\
VERIFICATION FAILURE NOTICE
{disclaimer}

Event: {event_id}
FSM state: {fsm_state}
Reason: {reason}

The system has declined to produce probabilistic coastal guidance \
for this event. Verification did not pass or was absent. \
No scenario-based assessment is available.

This is an automated safety response. The duty scientist should \
review the underlying data and verification results."""


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def format_human_decision(
    assessment: FinalAssessment,
    decision: HumanDecision,
) -> FinalAssessment:
    """Apply a human review decision to a PROVISIONAL FinalAssessment.

    Returns a new FinalAssessment with updated status and decision trace.
    The original assessment is not mutated; all identity fields (event_id,
    produced_at_utc, and the provenance/handoff references) are preserved, so
    this is an update to the same assessment, not a new handoff.

    Args:
        assessment: Must have ``status=PROVISIONAL``.
        decision: The reviewer's recorded decision.

    Returns:
        New FinalAssessment - APPROVED_INTERNAL on APPROVE,
        PROVISIONAL on REJECT/DEFER (with INFO trace step).

    Raises:
        ValueError: If assessment is not PROVISIONAL, or if the
            evidence string contains prohibited alert terminology.
    """
    if assessment.status != AssessmentStatus.PROVISIONAL:
        raise ValueError(
            f"Can only format PROVISIONAL assessments, got {assessment.status}"
        )

    # Build evidence string from the decision
    evidence = (
        f"{decision.decision.value} by {decision.reviewer_id}: "
        f"{decision.decision_reason}"
    )

    # Defense-in-depth: dynamic decision_reason could contain "Warning"
    scan_result = scan_text(evidence)
    from hazard_assessment.telemetry.metrics import record_guardrail_scan
    record_guardrail_scan(passed=not scan_result.violations)
    if scan_result.violations:
        terms = [v.term for v in scan_result.violations]
        raise ValueError(
            f"Decision reason contains prohibited alert terminology: {terms}"
        )

    # Determine new status and trace step result
    if decision.decision == ReviewDecision.APPROVE:
        new_status = AssessmentStatus.APPROVED_INTERNAL
        step_result = StepResult.PASS
    else:
        # REJECT or DEFER - status stays PROVISIONAL
        new_status = AssessmentStatus.PROVISIONAL
        step_result = StepResult.INFO

    new_step = DecisionStep(
        step="human_review",
        result=step_result,
        evidence=evidence,
    )

    # model_copy creates a new instance with the specified fields updated.
    # Frozen fields (type, disclaimer) are preserved automatically when
    # not included in the update dict.
    return assessment.model_copy(
        update={
            "status": new_status,
            "decision_trace": assessment.decision_trace + [new_step],
        }
    )


def format_abstain(
    *,
    event_id: str | UUID | None,
    fsm_state: str,
    abstain_reason: str,
    producer: str = "abstain_formatter",
    provenance_bundle_id: UUID | None = None,
) -> FinalAssessment:
    """Create a FinalAssessment with status=ABSTAIN.

    Produces a typed, schema-validated envelope when the verification
    agent fails or is absent. The summary includes the mandatory
    non-authoritative disclaimer (required by ``scan_text()``).

    Args:
        event_id: Seismic event identifier (UUID or UUID-formatted string;
            Pydantic coerces valid UUID strings but rejects non-UUID strings).
        fsm_state: Current FSM state label for the summary.
        abstain_reason: Why verification failed (included in summary
            and decision trace).
        producer: Producing component name (default ``"abstain_formatter"``).
        provenance_bundle_id: Provenance UUID; auto-generated if None.

    Returns:
        FinalAssessment with status=ABSTAIN, report_tier=1, confidence=LOW.

    Raises:
        ValueError: If the rendered summary contains prohibited terms
            (defense-in-depth against dynamic ``abstain_reason``).
    """
    summary = _ABSTAIN_SUMMARY_TEMPLATE.format(
        disclaimer=NON_AUTHORITATIVE_DISCLAIMER,
        event_id=event_id or "",
        fsm_state=fsm_state,
        reason=abstain_reason,
    )

    # Defense-in-depth: abstain_reason is system-generated but could
    # theoretically contain prohibited terms.
    scan_result = scan_text(summary)
    from hazard_assessment.telemetry.metrics import record_guardrail_scan
    record_guardrail_scan(passed=not scan_result.violations)
    if scan_result.violations:
        terms = [v.term for v in scan_result.violations]
        raise ValueError(
            f"Abstain summary contains prohibited alert terminology: {terms}"
        )

    return FinalAssessment(
        producer=producer,
        produced_at_utc=datetime.now(UTC),
        event_id=event_id,  # type: ignore[arg-type]
        status=AssessmentStatus.ABSTAIN,
        report_tier=1,
        summary=summary,
        uncertainty=UncertaintyInfo(
            confidence_level=ConfidenceLevel.LOW,
            key_uncertainties=["Verification did not pass or was absent"],
        ),
        provenance_bundle_id=provenance_bundle_id or uuid4(),
        decision_trace=[
            DecisionStep(
                step="verification_outcome",
                result=StepResult.FAIL,
                evidence=abstain_reason,
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Import-time template safety verification
# ---------------------------------------------------------------------------


def _verify_abstain_template_safety() -> None:
    """Render the ABSTAIN template with safe values and scan for violations.

    Called at import time. If the template contains prohibited NOAA alert
    terminology by construction, this raises ImportError - catching
    mistakes at development time, not runtime.

    Same pattern as ``report_templates._verify_template_safety()``.
    """
    rendered = _ABSTAIN_SUMMARY_TEMPLATE.format(
        disclaimer=NON_AUTHORITATIVE_DISCLAIMER,
        event_id="TEST_EVENT",
        fsm_state="ASSESS",
        reason="Model fit quality below threshold",
    )
    result = scan_text(rendered)
    if result.violations:
        terms = [v.term for v in result.violations]
        raise ImportError(
            f"ABSTAIN template contains prohibited NOAA alert terminology "
            f"by construction: {terms}. Fix the template text."
        )
    if not result.has_disclaimer:
        raise ImportError(
            "ABSTAIN template is missing the mandatory "
            "non-authoritative disclaimer."
        )


_verify_abstain_template_safety()
