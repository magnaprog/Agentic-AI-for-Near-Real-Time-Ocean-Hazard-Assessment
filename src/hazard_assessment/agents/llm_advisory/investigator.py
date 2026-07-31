"""Active-event evidence investigator.

The deterministic pipeline decides. This reads what the pipeline recorded for
an event that is still open and writes findings a duty scientist can act on,
for questions the pipeline does not answer about its own evidence.

Why it cannot influence the assessment, structurally rather than by
instruction:

* It runs only once the worker has persisted an assessment checkpoint, and
  binds its findings to that row. The evidence it reads is the event's audit
  trail rather than the row's contents; the row supplies identity and the
  foreign key. Either way it sits off the detection path, so it adds no
  latency and no network dependency to the FSM loop, and a failure here
  cannot stall ingest or scoring.
* Its tools are the same read-only, event-scoped queries the after-action
  analysis uses. The event identifier is bound by the caller and the model
  cannot widen it.
* Findings land in ``evidence_issue_results``, which migration 009 grants to
  ``investigator_writer`` alone. ``pipeline_worker``, which drives the FSM,
  holds neither insert nor select on it, and neither does the API's own role,
  so a finding can be neither written nor read by the code that escalates.
  Insert denial alone would not establish that; the read denial is what makes
  a finding unusable as input to an escalation.
* Everything the model authors is guardrail-scanned before persistence, which
  means the finding text and the tool-call log, since the model chooses tool
  names and arguments and both are recorded. Whichever reaches for reserved
  alert wording is withheld rather than stored.

The three issues below were chosen because each is answerable from records the
pipeline already writes, and none is already answered by it:

``station_agreement``
    The FSM escalates on the single highest station score. Nothing examines
    whether the other stations corroborate it, so a duty scientist cannot see
    from the state alone whether an escalation rests on one instrument.

``evidence_gaps``
    ``sensor_degraded`` is one flag. It does not say which stations produced no
    scored window, or whether quality control explains their absence.

``timeline_consistency``
    Transitions are recorded with their trigger reasons, but nothing checks
    whether their order and spacing make sense against the seismic origin, for
    instance whether escalation happened before any ocean evidence arrived.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from uuid import UUID

from hazard_assessment.agents.llm_advisory.prompts import INVESTIGATOR_ISSUE_PROMPTS
from hazard_assessment.policy.guardrails import scan_structure, scan_text

if TYPE_CHECKING:
    from hazard_assessment.audit.logger import AuditLogger
    from hazard_assessment.config.settings import LLMSettings

logger = logging.getLogger(__name__)

#: Bumped when a prompt changes, so a re-investigation under new wording gets
#: its own invocation identity instead of colliding with the old finding.
PROMPT_VERSION = 1

#: Investigated in this order. Kept explicit rather than derived from the
#: prompt mapping so the order is reviewable.
ISSUE_NAMES: tuple[str, ...] = (
    "station_agreement",
    "evidence_gaps",
    "timeline_consistency",
)


@dataclass(frozen=True)
class IssueFinding:
    """One issue investigated against one assessment checkpoint."""

    issue_name: str
    invocation_id: str
    finding: str
    tool_calls: list[dict[str, Any]]
    guardrail_violations: list[str]

    def to_result_payload(self) -> dict[str, Any]:
        """The JSONB body stored in ``evidence_issue_results.result``."""
        return {
            "issue_name": self.issue_name,
            "finding": self.finding,
            "tool_calls": self.tool_calls,
            "guardrail_violations": self.guardrail_violations,
            "prompt_version": PROMPT_VERSION,
        }

    @property
    def result_sha256(self) -> str:
        """Digest of the stored payload, for the column of the same name.

        Derived rather than stored so it cannot drift from what it describes.
        Computed over the whole of ``to_result_payload()``: an earlier version
        hashed only the finding text and the tool log, which left
        ``guardrail_violations`` outside the digest, so the one field recording
        that a finding was withheld was the one a hash could not detect a
        change to.
        """
        return hashlib.sha256(
            json.dumps(
                self.to_result_payload(),
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            ).encode("utf-8")
        ).hexdigest()


def compute_invocation_id(
    *,
    assessment_row_id: int,
    issue_name: str,
    model: str,
    prompt_version: int = PROMPT_VERSION,
) -> str:
    """Deterministic identity for one investigation.

    Migration 009 makes these rows unique on this value and append-only, so
    the digest has to cover everything that would make a finding a different
    finding: which assessment, which question, which model, which prompt. Two
    runs of the same investigation then collide by design instead of storing a
    second opinion on the same evidence.
    """
    material = json.dumps(
        {
            "assessment_row_id": assessment_row_id,
            "issue_name": issue_name,
            "model": model,
            "prompt_version": prompt_version,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def investigate_issue(
    settings: LLMSettings,
    audit_logger: AuditLogger,
    *,
    event_id: UUID,
    assessment_row_id: int,
    issue_name: str,
    max_rounds: int = 3,
) -> IssueFinding:
    """Investigate one issue against one persisted assessment checkpoint.

    Raises:
        KeyError: If ``issue_name`` has no prompt.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    from hazard_assessment.agents.llm_advisory.factory import build_chat_model
    from hazard_assessment.agents.llm_advisory.tools import (
        make_event_query_tools,
        resolve_tool_calls,
    )

    system_prompt = INVESTIGATOR_ISSUE_PROMPTS[issue_name]

    call_log: list[dict[str, Any]] = []
    tools = make_event_query_tools(
        audit_logger, pinned_event_id=event_id, call_log=call_log
    )
    llm_with_tools = build_chat_model(settings, purpose="quality", tools=tools)

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(
            content=(
                f"Event {event_id} is still open. Investigate this issue "
                "against the evidence recorded so far and report only what "
                "the records support."
            )
        ),
    ]
    result = llm_with_tools.invoke(messages)
    finding = resolve_tool_calls(
        result,
        llm_with_tools,
        tools,
        max_rounds=max_rounds,
        initial_messages=messages,
        call_log=call_log,
        node=issue_name,
    )

    # Everything operator-facing goes through the same scanner as any other
    # emitted narrative, and the finding text is not the only such string. The
    # model also chooses tool names and tool arguments, and both are persisted
    # and returned: an unknown-tool request echoes the name it asked for, and a
    # recorded call echoes its arguments. Scanning only the finding left a route
    # for reserved wording to reach an operator while the finding itself came
    # back clean.
    violations = [v.term for v in scan_text(finding).violations]
    tool_violations = [v.term for v in scan_structure(call_log)]

    if violations:
        # Dropped, not redacted: a partially rewritten finding would
        # misrepresent what the model said.
        logger.warning(
            "Investigator finding for %s dropped on reserved terms: %s",
            issue_name,
            ", ".join(sorted(set(violations))),
        )
        finding = (
            "(finding withheld: the generated text used reserved alert "
            "terminology and was dropped by the guardrail scanner)"
        )

    if tool_violations:
        # The log is evidence of what the model did, so replacing entries would
        # destroy the record. Withhold the whole log instead and say why, while
        # keeping the counts, which carry no model-authored text.
        logger.warning(
            "Investigator tool log for %s withheld on reserved terms: %s",
            issue_name,
            ", ".join(sorted(set(tool_violations))),
        )
        call_log = [
            {
                "withheld": (
                    "tool-call log used reserved alert terminology and was "
                    "dropped by the guardrail scanner"
                ),
                "n_calls": len(call_log),
            }
        ]
        violations = sorted(set(violations) | set(tool_violations))

    return IssueFinding(
        issue_name=issue_name,
        invocation_id=compute_invocation_id(
            assessment_row_id=assessment_row_id,
            issue_name=issue_name,
            model=settings.quality_model or settings.model,
        ),
        finding=finding,
        tool_calls=call_log,
        guardrail_violations=sorted(set(violations)),
    )


def investigate_assessment(
    settings: LLMSettings,
    audit_logger: AuditLogger,
    *,
    event_id: UUID,
    assessment_row_id: int,
    issue_names: tuple[str, ...] = ISSUE_NAMES,
    max_rounds: int = 3,
) -> list[IssueFinding]:
    """Investigate each issue in turn, keeping going if one fails.

    One issue failing is not a reason to lose the others: the findings are
    independent, and a duty scientist is better served by two of three than by
    nothing. A failure is logged and that issue is simply absent from the
    result, which the caller reports rather than hides.
    """
    findings: list[IssueFinding] = []
    for issue_name in issue_names:
        try:
            findings.append(
                investigate_issue(
                    settings,
                    audit_logger,
                    event_id=event_id,
                    assessment_row_id=assessment_row_id,
                    issue_name=issue_name,
                    max_rounds=max_rounds,
                )
            )
        except Exception:
            logger.exception("Investigation of issue %s failed", issue_name)
    return findings
