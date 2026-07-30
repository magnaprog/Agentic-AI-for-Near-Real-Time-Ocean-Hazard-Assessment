"""Read-only query tools for the after-action analysis graph.

These tools give the LLM the ability to decide which data to retrieve,
making the after-action graph genuinely agentic (tool-use pattern).
All tools are read-only - they cannot modify any system state.

The tools operate on an AuditLogger instance injected at graph
construction time via closure.  The event_id is pinned at construction
time so the LLM cannot query data for a different event.

Contract: every tool invocation is recorded into the caller's
``call_log`` (name, arguments, match counts, truncation), and every
tool result is a JSON object that carries an explicit ``truncated``
flag plus the total match count, so a capped read is never mistaken
for a complete one.
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uuid import UUID

    from hazard_assessment.audit.logger import AuditLogger

from langchain_core.tools import tool

logger = logging.getLogger(__name__)

_MAX_TOOL_ENTRIES = 200

# Agent-output and live-evidence event types the pipeline actually
# writes. qc_complete and anomaly_scored are the live worker's QC and
# anomaly evidence records: the after-action analysis must be able to
# see them, not only FSM transitions.
_FEATURE_EVENT_TYPES = (
    "qc_complete",
    "anomaly_scored",
    "verification_complete",
    "report_generated",
    "assessment_formatted",
    "abstain_triggered",
)


def make_event_query_tools(
    audit_logger: AuditLogger,
    *,
    pinned_event_id: UUID,
    call_log: list[dict[str, Any]] | None = None,
) -> list[Any]:
    """Create read-only query tools bound to a specific event.

    Shared by the after-action analysis and the active-event investigator:
    both need the same event-scoped, bounded, logged reads.

    Args:
        audit_logger: The audit logger to query.
        pinned_event_id: The event ID to constrain all queries to.
            The LLM cannot override this - it is enforced structurally.
        call_log: Optional mutable list; every successful tool
            invocation appends one record (tool, args, n_total_matching,
            n_returned, truncated). The caller owns persistence.

    Returns:
        A list of LangChain tool functions for use with ``llm.bind_tools()``.
    """

    def _package(
        tool_name: str,
        args: dict[str, Any],
        matching: list[Any],
    ) -> str:
        returned = matching[:_MAX_TOOL_ENTRIES]
        truncated = len(matching) > len(returned)
        if call_log is not None:
            call_log.append({
                "tool": tool_name,
                "args": args,
                "n_total_matching": len(matching),
                "n_returned": len(returned),
                "truncated": truncated,
            })
        return json.dumps(
            {
                "entries": [e.model_dump(mode="json") for e in returned],
                "n_total_matching": len(matching),
                "n_returned": len(returned),
                "truncated": truncated,
            },
            default=str,
        )

    def _all_event_entries(event_type: str | None = None) -> list[Any]:
        # Read past the default 200-entry window so the total match
        # count (and therefore the truncation flag) is honest.
        return audit_logger.get_entries(
            event_id=pinned_event_id,
            event_type=event_type,
            limit=audit_logger.count + 1,
        )

    @tool
    def query_audit_trail(event_type: str = "") -> str:
        """Query audit trail entries for the current event.

        Args:
            event_type: Optional filter by event type (e.g., 'state_transition',
                'policy_denial', 'qc_complete'). Empty string returns all types.

        Returns:
            JSON object with 'entries' (max 200), 'n_total_matching',
            'n_returned', and a 'truncated' flag.
        """
        matching = _all_event_entries(event_type if event_type else None)
        return _package(
            "query_audit_trail", {"event_type": event_type}, matching
        )

    @tool
    def query_fsm_transitions() -> str:
        """Query FSM state transition records for the current event.

        Returns:
            JSON object with 'entries' (max 200), 'n_total_matching',
            'n_returned', and a 'truncated' flag.
        """
        matching = _all_event_entries("state_transition")
        return _package("query_fsm_transitions", {}, matching)

    @tool
    def query_processed_features(agent_name: str = "") -> str:
        """Query pipeline evidence records: live QC results, anomaly
        scores, and agent outputs for the current event.

        Args:
            agent_name: Optional filter by producing node/agent name
                (e.g., 'verify_node', 'report_node'). Empty returns all.

        Returns:
            JSON object with 'entries' (max 200), 'n_total_matching',
            'n_returned', and a 'truncated' flag.
        """
        matching = [
            entry
            for entry in _all_event_entries()
            if entry.event_type in _FEATURE_EVENT_TYPES
            and (not agent_name or entry.producer == agent_name)
        ]
        return _package(
            "query_processed_features", {"agent_name": agent_name}, matching
        )

    return [query_audit_trail, query_fsm_transitions, query_processed_features]


def resolve_tool_calls(
    result: Any,
    llm_with_tools: Any,
    tools: list[Any],
    *,
    max_rounds: int = 3,
    initial_messages: list[Any] | None = None,
    call_log: list[dict[str, Any]] | None = None,
    node: str = "",
) -> str:
    """Run the model's tool calls until it answers in text, or the cap is hit.

    This is the agentic loop: the model chooses which tools to call, they are
    executed, and the results go back to it. Shared by the after-action
    analysis and the active-event investigator so the bound and the logging
    have one implementation rather than two.

    Every call is recorded, including ones for tools that do not exist, ones
    that raise, and the case where the model never stops asking. Failing to
    converge returns a sentence saying so instead of a partial answer, because
    a caller cannot tell a truncated analysis from a complete one.

    Args:
        initial_messages: The system and human messages from the first
            invocation, replayed each round so context is not lost.
        call_log: Optional mutable list receiving one record per call. The
            tools log their own successful calls; this loop adds the failures
            and stamps every record with the node that made it.
        node: Label of the calling node, for attribution in the log.
    """
    from langchain_core.messages import ToolMessage

    messages: list[Any] = list(initial_messages or []) + [result]
    tool_map = {t.name: t for t in tools}

    for _ in range(max_rounds):
        if not hasattr(result, "tool_calls") or not result.tool_calls:
            break

        for tc in result.tool_calls:
            tool_fn = tool_map.get(tc["name"])
            if tool_fn is None:
                if call_log is not None:
                    call_log.append({
                        "node": node,
                        "tool": tc["name"],
                        "args": dict(tc.get("args") or {}),
                        "error": "unknown_tool",
                    })
                messages.append(
                    ToolMessage(content=f"Unknown tool: {tc['name']}", tool_call_id=tc["id"])
                )
                continue
            before = len(call_log) if call_log is not None else 0
            try:
                output = tool_fn.invoke(tc["args"])
            except Exception as exc:
                logger.warning("Tool %s failed: %s", tc["name"], exc)
                output = f"Tool error: {type(exc).__name__}"
                if call_log is not None:
                    call_log.append({
                        "node": node,
                        "tool": tc["name"],
                        "args": dict(tc.get("args") or {}),
                        "error": type(exc).__name__,
                    })
            if call_log is not None:
                # Stamp records the tool itself appended with the node.
                for record in call_log[before:]:
                    record.setdefault("node", node)
            messages.append(ToolMessage(content=str(output), tool_call_id=tc["id"]))

        result = llm_with_tools.invoke(messages)
        messages.append(result)

    if hasattr(result, "tool_calls") and result.tool_calls:
        logger.warning("Tool-call loop did not converge within %d rounds", max_rounds)
        if call_log is not None:
            call_log.append({
                "node": node,
                "tool": None,
                "error": "tool_loop_did_not_converge",
                "max_rounds": max_rounds,
            })
        return "(analysis incomplete: tool-call loop did not converge)"

    if hasattr(result, "content"):
        return str(result.content)
    return str(result)
