"""Pipeline definition for the hazard assessment workflow.

Defines the directed graph of agent nodes with deterministic routing.
The FSM orchestrator controls state transitions; this module defines
the graph topology and routing logic. Execution uses
``run_pipeline_sync()`` (sequential node runner in ``nodes.py``).

Pipeline flow:
  ingest -> qc -> anomaly -> orchestrate(FSM)
    -> if ASSESS or ESCALATE: scenario -> verify
        -> route: PASS/PASS_WITH_CONCERNS -> report -> human_review -> final
                 FAIL -> abstain -> human_review -> final
    -> if IDLE, MONITOR, or INVESTIGATE: -> final (insufficient evidence)

The Human Review Gate is a mandatory checkpoint (Prohibited Action P6).
Both the REPORT and ABSTAIN paths pass through human review before
reaching FINAL.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, TypedDict

from hazard_assessment.orchestrator.states import SystemState
from hazard_assessment.schemas.verification import VerificationOutcome

logger = logging.getLogger(__name__)


class PipelineNode(StrEnum):
    """Named nodes in the assessment pipeline graph."""

    INGEST = "ingest"
    QC = "qc"
    ANOMALY = "anomaly"
    ORCHESTRATE = "orchestrate"
    SCENARIO = "scenario"
    VERIFY = "verify"
    REPORT = "report"
    ABSTAIN = "abstain"
    HUMAN_REVIEW = "human_review"
    FINAL = "final"


class PipelineState(TypedDict, total=False):
    """Shared state passed between pipeline nodes.

    Each key holds the typed envelope output from the corresponding agent.
    Using total=False because keys are populated incrementally as the
    pipeline progresses.
    """

    event_id: str
    trace_id: str
    fsm_state: str
    state_changed: bool
    qc_reports: list[dict[str, Any]]
    anomaly_assessment: dict[str, Any]
    scenario_assessment: dict[str, Any]
    verification_result: dict[str, Any]
    human_decision: dict[str, Any]
    final_assessment: dict[str, Any]
    abstain_triggered: bool
    abstain_reason: str
    pipeline_status: str
    """Execution status of the pipeline run.

    Values:
    - ``"complete"``: all nodes ran successfully.
    - ``"incomplete_report"``: verification passed but report generation
      is not yet implemented.  Set by ``report_node()`` when no
      ``report_agent`` is provided.
    - ``"report_generated"``: ``report_node()`` successfully produced a
      structured FinalAssessment via the template engine.
    - ``"abstain"``: verification failed or was absent - the system refused
      to produce probabilistic coastal guidance.  Set by ``final_node()``
      when ``abstain_triggered=True``.
    """


def route_after_orchestrate(state: PipelineState) -> str:
    """Route after FSM orchestration: only ASSESS/ESCALATE proceed to scenario.

    States below ASSESS (IDLE, MONITOR, INVESTIGATE) route to FINAL without
    invoking the scenario agent - there is insufficient evidence to justify
    scenario ranking. The outer event-processing loop will invoke the
    pipeline again when new data arrives.
    """
    fsm_state = state.get("fsm_state", "")

    if fsm_state in (SystemState.ASSESS, SystemState.ESCALATE):
        return PipelineNode.SCENARIO

    # IDLE, MONITOR, INVESTIGATE: insufficient evidence for scenarios.
    # Unknown/missing states also route here - this is conservative
    # because FINAL without scenario analysis produces no assessment.
    if fsm_state not in (SystemState.IDLE, SystemState.MONITOR, SystemState.INVESTIGATE):
        logger.warning(
            "Unknown FSM state in router: %s - routing to FINAL",
            fsm_state,
        )

    return PipelineNode.FINAL


def route_after_verify(state: PipelineState) -> str:
    """Route after verification: fail-closed defense-in-depth for ABSTAIN.

    This is the second layer of defense. The first layer is the Pydantic
    model_validator on VerificationResult that enforces FAIL ->
    abstain_required=True at the schema level. This router enforces
    the same invariant at the pipeline routing level.

    Fail-closed design: if verification_result is missing entirely (e.g.,
    verify node errored, timed out, or produced incomplete state), the
    pipeline routes to ABSTAIN rather than allowing an unverified report.
    """
    verification = state.get("verification_result")

    # Fail-closed: missing verification -> ABSTAIN (never generate
    # a report without a successful verification step)
    if not verification:
        return PipelineNode.ABSTAIN

    overall = verification.get("overall", "")
    abstain_required = verification.get("abstain_required", False)

    # Defense-in-depth: FAIL or INCOMPLETE outcome OR abstain flag
    # triggers ABSTAIN. INCOMPLETE also implies abstain_required=True at
    # the schema level; the explicit check here is a second layer.
    if (
        overall in (VerificationOutcome.FAIL, VerificationOutcome.INCOMPLETE)
        or abstain_required
    ):
        return PipelineNode.ABSTAIN

    # Only PASS or PASS_WITH_CONCERNS with no abstain flag reach REPORT
    if overall in (VerificationOutcome.PASS, VerificationOutcome.PASS_WITH_CONCERNS):
        return PipelineNode.REPORT

    # Unknown/unexpected outcome -> ABSTAIN (fail-closed)
    return PipelineNode.ABSTAIN


def build_pipeline_graph() -> dict[str, Any]:
    """Build the pipeline graph specification.

    Returns a dict describing the graph structure. This declarative
    specification documents the flow and is used by the pipeline
    worker's startup validation.

    The graph is a DAG with two conditional branch points and a
    mandatory Human Review Gate before FINAL:
    1. After orchestrate: ASSESS/ESCALATE -> scenario, else -> final
    2. After verify: PASS/PASS_WITH_CONCERNS -> report, FAIL -> abstain
    3. Both report and abstain -> human_review -> final
    """
    # Use PipelineNode enum members consistently throughout the spec.
    # PipelineNode is a StrEnum, so members serialize to their string
    # values and compare equal to plain strings.
    return {
        "nodes": list(PipelineNode),
        "edges": [
            (PipelineNode.INGEST, PipelineNode.QC),
            (PipelineNode.QC, PipelineNode.ANOMALY),
            (PipelineNode.ANOMALY, PipelineNode.ORCHESTRATE),
            # Conditional: orchestrate -> scenario or final
            (PipelineNode.SCENARIO, PipelineNode.VERIFY),
            # Conditional: verify -> report or abstain
            (PipelineNode.REPORT, PipelineNode.HUMAN_REVIEW),
            (PipelineNode.ABSTAIN, PipelineNode.HUMAN_REVIEW),
            # Human review is mandatory before final output (P6)
            (PipelineNode.HUMAN_REVIEW, PipelineNode.FINAL),
        ],
        "conditional_edges": {
            PipelineNode.ORCHESTRATE: {
                "router": "route_after_orchestrate",
                "branches": {
                    PipelineNode.SCENARIO: "ASSESS or ESCALATE",
                    PipelineNode.FINAL: "IDLE, MONITOR, or INVESTIGATE",
                },
            },
            PipelineNode.VERIFY: {
                "router": "route_after_verify",
                "branches": {
                    PipelineNode.REPORT: "PASS or PASS_WITH_CONCERNS",
                    PipelineNode.ABSTAIN: "FAIL or abstain_required",
                },
            },
        },
        "entry_point": PipelineNode.INGEST,
    }
