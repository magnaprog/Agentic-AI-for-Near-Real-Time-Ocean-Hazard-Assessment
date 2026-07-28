"""Pipeline node functions for the hazard assessment workflow.

Each node function takes a PipelineState dict and returns a partial
PipelineState update. ``run_pipeline_sync()`` merges each partial
update into the shared state after each node completes.

Implements:
- Wire deterministic FSM to real anomaly scores
- Emit AnomalyAssessment handoff schema with FSM state annotation
- Enforce ABSTAIN output on verification fail
- Log verification and ABSTAIN decisions to audit trail
- Structured assessment report generation with guardrails
- FinalAssessment formatter - typed ABSTAIN envelope and
  PROVISIONAL -> APPROVED_INTERNAL transition
"""

from __future__ import annotations

import logging
from typing import Any
from uuid import UUID, uuid4

from hazard_assessment.audit.logger import AuditEntry, AuditLogger
from hazard_assessment.orchestrator.pipeline import PipelineNode, PipelineState
from hazard_assessment.orchestrator.states import (
    FSMOrchestrator,
    TransitionRecord,
)

logger = logging.getLogger(__name__)


def _parse_uuid_field(state: PipelineState, field: str) -> UUID | None:
    """Extract a UUID field from pipeline state, returning None if absent/invalid."""
    raw = state.get(field)
    if not raw:
        return None
    if isinstance(raw, UUID):
        return raw
    if isinstance(raw, str):
        try:
            return UUID(raw)
        except ValueError:
            logger.warning("_parse_uuid_field: invalid UUID for %s: %r", field, raw)
            return None
    logger.warning(
        "_parse_uuid_field: unexpected type %s for %s", type(raw).__name__, field,
    )
    return None


def _parse_event_id(state: PipelineState) -> UUID | None:
    return _parse_uuid_field(state, "event_id")


def _parse_trace_id(state: PipelineState) -> UUID | None:
    return _parse_uuid_field(state, "trace_id")


# NOTE: Ingest, QC, anomaly, and scenario data arrive pre-populated
# in pipeline state by the caller.  These stages run outside the
# pipeline graph (connectors poll independently, agents are invoked
# directly).  No pass-through node functions are needed.


def verify_node(
    state: PipelineState,
    *,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Pass-through node for verification results.

    The verification agent runs outside the pipeline because
    VerificationInput requires large numerical data (Green's function
    matrix, observation vector, holdout waveforms) that does not belong
    in PipelineState. The caller constructs VerificationInput, calls
    VerificationAgent.verify(), and serializes the compact
    VerificationResult into state.

    This node logs the verification outcome to the audit trail.
    """
    verification = state.get("verification_result")
    if verification:
        from hazard_assessment.telemetry.metrics import record_verification_outcome
        record_verification_outcome(str(verification.get("overall", "")))
        if audit_logger is not None:
            audit_logger.append(AuditEntry(
                event_id=_parse_event_id(state),
                trace_id=_parse_trace_id(state),
                event_type="verification_complete",
                producer="verify_node",
                data={
                    "overall": verification.get("overall", ""),
                    "abstain_required": verification.get("abstain_required", False),
                },
            ))
    return {}


def abstain_node(
    state: PipelineState,
    *,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Record ABSTAIN decision and produce typed FinalAssessment.

    Calls ``format_abstain()`` to produce a schema-validated
    FinalAssessment(status=ABSTAIN). Sets ``abstain_triggered=True``
    and ``final_assessment`` in pipeline state.

    Backwards-compat: ``"outcome"`` and ``"pipeline_status"`` keys are
    added to the plain dict post-``model_dump()`` for safety tests that
    check these legacy keys. FinalAssessment has ``extra="forbid"`` so
    these cannot be schema fields.
    """
    from hazard_assessment.agents.assessment_formatter import format_abstain

    verification = state.get("verification_result") or {}
    abstain_reason = (
        verification.get("abstain_reason") or "Verification failed or missing"
    )
    if audit_logger is not None:
        audit_logger.append(AuditEntry(
            event_id=_parse_event_id(state),
            trace_id=_parse_trace_id(state),
            event_type="abstain_triggered",
            producer="abstain_node",
            data={
                "reason": abstain_reason,
                "overall": verification.get("overall", ""),
            },
        ))

    fa = format_abstain(
        event_id=_parse_event_id(state),
        fsm_state=state.get("fsm_state", ""),
        abstain_reason=abstain_reason,
        producer="abstain_node",
    )
    fa_dict = fa.model_dump()
    # WARNING: These keys are NOT schema fields (FinalAssessment has extra="forbid").
    # They exist only for dict-level test assertions. Do NOT round-trip this dict
    # through FinalAssessment.model_validate() - it will raise ValidationError.
    fa_dict["outcome"] = "abstain"
    fa_dict["pipeline_status"] = "abstain"

    from hazard_assessment.telemetry.metrics import record_abstain
    record_abstain()

    return {
        "abstain_triggered": True,
        "abstain_reason": abstain_reason,
        "final_assessment": fa_dict,
    }


def report_node(
    state: PipelineState,
    *,
    report_agent: Any | None = None,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Generate a structured assessment report.

    When ``report_agent`` is provided, deserializes ScenarioAssessment and
    VerificationResult from state, calls ``synthesize()``, and returns the
    FinalAssessment as a dict. When ``report_agent`` is None (caller did
    not configure one), returns ``{"pipeline_status": "incomplete_report"}``.

    All exceptions are caught and logged - the node never emits a malformed
    report. On failure, it returns ``incomplete_report`` (fail-closed).
    """
    if report_agent is None:
        return {"pipeline_status": "incomplete_report"}

    try:
        from hazard_assessment.schemas.scenario import ScenarioAssessment
        from hazard_assessment.schemas.verification import VerificationResult

        scenario_dict = state.get("scenario_assessment")
        verification_dict = state.get("verification_result")

        if scenario_dict is None or verification_dict is None:
            logger.warning(
                "report_node: missing scenario_assessment or verification_result"
            )
            return {"pipeline_status": "incomplete_report"}

        scenario = ScenarioAssessment.model_validate(scenario_dict)
        verification = VerificationResult.model_validate(verification_dict)

        anomaly_dict = state.get("anomaly_assessment") or {}
        # rayleigh_wave_suspect is tri-state in AnomalyAssessment (None =
        # never evaluated). The report boundary is a plain bool; a
        # not-evaluated flag is treated as not-suspect here because the
        # report only adds caveats when the flag is True.
        rayleigh_suspect = bool(anomaly_dict.get("rayleigh_wave_suspect") or False)

        fa = report_agent.synthesize(
            scenario, verification,
            rayleigh_wave_suspect=rayleigh_suspect,
            fsm_state=state.get("fsm_state", ""),
        )

        if audit_logger is not None:
            audit_logger.append(AuditEntry(
                event_id=_parse_event_id(state),
                trace_id=_parse_trace_id(state),
                event_type="report_generated",
                producer="report_node",
                data={
                    "tier": fa.report_tier,
                    "confidence": fa.uncertainty.confidence_level.value,
                    "status": fa.status.value,
                },
            ))

        return {
            "final_assessment": fa.model_dump(),
            "pipeline_status": "report_generated",
        }
    except Exception:
        logger.exception("report_node: failed to generate report")
        if audit_logger is not None:
            audit_logger.append(AuditEntry(
                event_id=_parse_event_id(state),
                trace_id=_parse_trace_id(state),
                event_type="report_generation_failed",
                producer="report_node",
                data={"pipeline_status": "incomplete_report"},
            ))
        return {"pipeline_status": "incomplete_report"}


def human_review_node(
    state: PipelineState,
    *,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Apply human decision to a PROVISIONAL FinalAssessment.

    When both ``human_decision`` and ``final_assessment`` are present
    in state and the assessment is PROVISIONAL, applies
    ``format_human_decision()`` to transition the status:

    - APPROVE -> APPROVED_INTERNAL
    - REJECT/DEFER -> PROVISIONAL (unchanged, with INFO trace step)

    Returns ``{}`` (pass-through) when inputs are missing, assessment
    is not PROVISIONAL, or on any exception (fail-open preserves the
    existing PROVISIONAL assessment, which is not distributable as
    Tier 2 without APPROVE - P6 enforcement).
    """
    from pydantic import ValidationError

    from hazard_assessment.agents.assessment_formatter import format_human_decision
    from hazard_assessment.schemas.final_assessment import (
        AssessmentStatus,
        FinalAssessment,
    )
    from hazard_assessment.schemas.human_decision import HumanDecision

    human_decision_dict = state.get("human_decision")
    final_assessment_dict = state.get("final_assessment")

    if not human_decision_dict or not final_assessment_dict:
        return {}

    # Only format typed FinalAssessment with PROVISIONAL status.
    # model_dump() returns StrEnum members (which inherit str); .value
    # is redundant but explicit about comparing against the string form.
    if final_assessment_dict.get("status") != AssessmentStatus.PROVISIONAL.value:
        return {}

    try:
        fa = FinalAssessment.model_validate(final_assessment_dict)
        decision = HumanDecision.model_validate(human_decision_dict)
        updated = format_human_decision(fa, decision)

        if audit_logger is not None:
            audit_logger.append(AuditEntry(
                event_id=_parse_event_id(state),
                trace_id=_parse_trace_id(state),
                event_type="assessment_formatted",
                producer="human_review_node",
                data={
                    "decision": decision.decision.value,
                    "reviewer_id": decision.reviewer_id,
                    "new_status": updated.status.value,
                },
            ))

        return {"final_assessment": updated.model_dump()}
    except (ValidationError, ValueError) as exc:
        logger.warning("human_review_node: %s", exc)
        return {}
    except Exception:
        logger.exception("human_review_node: unexpected error applying formatter")
        # Fail-open: preserves existing PROVISIONAL assessment (not
        # distributable as Tier 2 without APPROVE - P6 enforcement).
        return {}


def orchestrate_node(
    state: PipelineState,
    *,
    fsm: FSMOrchestrator,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Feed anomaly score into the FSM and update pipeline state.

    This is the core integration point. The orchestrate node:
    1. Extracts the ensemble anomaly score from the anomaly assessment
    2. Feeds it to FSMOrchestrator.evaluate_anomaly_score()
    3. Records the resulting FSM state and whether a transition occurred
    4. Annotates the AnomalyAssessment with current_state and state_changed

    Audit logging for state transitions is handled by the FSM's internal
    audit_writer (see states.py write_transition). The audit_logger parameter
    is accepted for interface consistency but not used by this node.

    The FSM's evaluate_anomaly_score() handles all threshold logic:
    - MONITOR -> INVESTIGATE when score >= T1
    - INVESTIGATE -> ASSESS when score >= T2
    - INVESTIGATE -> MONITOR when score < T1 (de-escalation)
    - ASSESS -> ESCALATE when score >= T3 (or seismic override)
    - ASSESS -> INVESTIGATE when score < T2 (de-escalation)
    """
    assessment = state.get("anomaly_assessment")
    if not assessment:
        logger.warning("orchestrate_node: no anomaly_assessment in state")
        return {
            "fsm_state": fsm.state.value,
            "state_changed": False,
        }

    anomaly_score = assessment.get("anomaly_score")
    if anomaly_score is None:
        logger.error(
            "orchestrate_node: anomaly_score missing from assessment; "
            "refusing to update FSM (fail-safe)"
        )
        return {
            "fsm_state": fsm.state.value,
            "state_changed": False,
        }
    prev_state = fsm.state

    # Feed anomaly score to FSM - this is the deterministic transition.
    # trace_id is passed so the FSM's write_transition() audit entry
    # carries the pipeline trace context (avoids duplicate audit writes).
    transition: TransitionRecord | None = fsm.evaluate_anomaly_score(
        anomaly_score, trace_id=_parse_trace_id(state),
    )

    state_changed = transition is not None
    current_state = fsm.state

    # Annotate the AnomalyAssessment envelope with FSM context.
    # This post-creation mutation is documented in the AnomalyAssessment
    # schema (see anomaly.py design note on current_state/state_changed).
    updated_assessment = dict(assessment)
    updated_assessment["current_state"] = current_state.value
    updated_assessment["state_changed"] = state_changed

    # Propagate event_id from FSM context
    event_id = state.get("event_id", "")
    if not event_id and fsm.event_context is not None:
        event_id = str(fsm.event_context.event_id)

    logger.info(
        "Orchestrate: score=%.4f, %s -> %s (changed=%s)",
        anomaly_score,
        prev_state.value,
        current_state.value,
        state_changed,
    )

    result: PipelineState = {
        "fsm_state": current_state.value,
        "state_changed": state_changed,
        "anomaly_assessment": updated_assessment,
    }
    if event_id:
        result["event_id"] = event_id

    return result


def final_node(state: PipelineState) -> PipelineState:
    """Terminal node that assembles the final pipeline output.

    Handles three cases when no upstream node set ``final_assessment``:

    1. **ABSTAIN** - verification failed or is absent. The system refuses
       to produce probabilistic coastal guidance.
    2. **Report agent not configured** - verification passed but no
       report agent was configured for this pipeline run.
    3. **Insufficient evidence** - FSM is below ASSESS (IDLE, MONITOR,
       INVESTIGATE). The pipeline will re-evaluate on next data arrival.

    For ASSESS/ESCALATE paths where upstream nodes (report_node or
    human_review_node) already set ``final_assessment``, this node
    is a no-op - it returns ``{}``.
    """
    fsm_state = state.get("fsm_state", "")

    if "final_assessment" not in state:
        # Case 1: ABSTAIN - verification did not pass.
        # Defense-in-depth fallback: abstain_node should have set
        # final_assessment via format_abstain(). This branch only
        # fires if abstain_node failed to set it.
        if state.get("abstain_triggered"):
            logger.warning(
                "final_node: ABSTAIN fallback - abstain_node did not set "
                "final_assessment"
            )
            return {
                "final_assessment": {
                    "fsm_state": fsm_state,
                    "pipeline_status": "abstain",
                    "outcome": "abstain",
                    "abstain_reason": state.get("abstain_reason", ""),
                    "detail": (
                        "Verification did not pass. The system will not "
                        "produce probabilistic coastal guidance for this event."
                    ),
                },
            }

        pipeline_status = state.get("pipeline_status", "complete")

        # Case 2: Verified but report agent not configured for this run
        if pipeline_status == "incomplete_report":
            return {
                "final_assessment": {
                    "fsm_state": fsm_state,
                    "pipeline_status": pipeline_status,
                    "outcome": "verified_pending_report",
                    "detail": (
                        "Verification passed but no report agent was "
                        "configured for this pipeline run."
                    ),
                },
            }

        # Case 3: Insufficient evidence (below ASSESS threshold)
        assessment = state.get("anomaly_assessment") or {}
        anomaly_score = assessment.get("anomaly_score", 0.0)

        return {
            "final_assessment": {
                "fsm_state": fsm_state,
                "anomaly_score": anomaly_score,
                "pipeline_status": pipeline_status,
                "outcome": "insufficient_evidence",
                "detail": (
                    f"FSM in {fsm_state} state. "
                    f"Anomaly score {anomaly_score:.4f} below scenario threshold. "
                    "Pipeline will re-evaluate on next data arrival."
                ),
            },
        }

    return {}


def run_pipeline_sync(
    state: PipelineState,
    *,
    fsm: FSMOrchestrator,
    report_agent: Any | None = None,
    audit_logger: AuditLogger | None = None,
) -> PipelineState:
    """Run the pipeline nodes sequentially.

    Executes ingest -> qc -> anomaly -> orchestrate -> route -> final.

    Thread-safety: This function copies ``state`` into a local dict
    and mutates the copy via ``dict.update()``.  The original ``state``
    is not modified.  However, the function reads/writes the shared
    ``FSMOrchestrator`` instance, so it is **not thread-safe**.
    Callers must ensure that concurrent pipeline runs use separate
    ``FSMOrchestrator`` instances, or serialize access externally.

    The caller must pre-populate state with the data they want processed
    (e.g., qc_reports, anomaly_assessment, scenario_assessment,
    verification_result).

    When the FSM reaches ASSESS/ESCALATE, the scenario path is executed:
    scenario -> verify -> route_after_verify -> report or abstain ->
    human_review -> final.

    If verification_result is missing or verification fails, the pipeline
    routes to ABSTAIN (fail-closed). When ``human_decision`` is
    pre-populated in state, ``human_review_node`` applies
    ``format_human_decision()`` to transition PROVISIONAL ->
    APPROVED_INTERNAL on APPROVE. When ``report_agent`` is provided,
    the report node generates a structured FinalAssessment.
    """
    from hazard_assessment.orchestrator.pipeline import (
        route_after_orchestrate,
        route_after_verify,
    )
    from hazard_assessment.telemetry.tracing import pipeline_span

    # Run nodes sequentially, merging results.
    # dict(state) copies the TypedDict into a plain dict; mypy can't track
    # that the keys are preserved through .update() mutations.
    current: PipelineState = dict(state)  # type: ignore[assignment]

    # Generate trace_id if not provided by caller
    if not current.get("trace_id"):
        current["trace_id"] = str(uuid4())

    with pipeline_span(str(current["trace_id"])):
        try:
            # ingest, qc, and anomaly data arrive pre-populated in state
            # by the caller (connectors and agents run outside the pipeline).

            # orchestrate (core FSM integration)
            current.update(
                orchestrate_node(current, fsm=fsm, audit_logger=audit_logger),
            )

            # Route after orchestrate
            route = route_after_orchestrate(current)

            if route == PipelineNode.SCENARIO:
                # Scenario data arrives pre-populated; verify -> route -> report/abstain
                current.update(verify_node(current, audit_logger=audit_logger))

                verify_route = route_after_verify(current)
                if verify_route == PipelineNode.ABSTAIN:
                    current.update(
                        abstain_node(current, audit_logger=audit_logger),
                    )
                elif verify_route == PipelineNode.REPORT:
                    current.update(report_node(
                        current,
                        report_agent=report_agent,
                        audit_logger=audit_logger,
                    ))
                else:
                    # Unexpected route - fail-closed to ABSTAIN
                    logger.warning(
                        "Unexpected verify route %s; falling back to ABSTAIN",
                        verify_route,
                    )
                    current.update(
                        abstain_node(current, audit_logger=audit_logger),
                    )

                current.update(
                    human_review_node(current, audit_logger=audit_logger),
                )

            # final
            current.update(final_node(current))
        except Exception:
            # Fail-closed: guarantee a final_assessment even on unexpected
            # errors.  The system must never exit without producing an
            # ABSTAIN when it cannot complete the pipeline normally.
            logger.exception("run_pipeline_sync: unhandled node error")
            if "final_assessment" not in current:
                current["final_assessment"] = {
                    "fsm_state": current.get("fsm_state", ""),
                    "pipeline_status": "error",
                    "outcome": "abstain",
                    "abstain_reason": "pipeline_error",
                    "detail": (
                        "An unexpected error interrupted the pipeline. "
                        "The system abstained from producing an assessment."
                    ),
                }

    return current
