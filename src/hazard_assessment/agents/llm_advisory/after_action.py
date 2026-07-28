"""3-node LangGraph after-action analysis graph with tool use.

Graph topology: START -> timeline -> gaps -> draft -> END

The timeline_node and gaps_node use LangChain tools to query the audit
trail on demand. The LLM decides which data to retrieve based on its
analysis - this is genuine tool-use agency within bounded constraints
(all tools are read-only).

Triggered post-event via API endpoint. Compiled without a checkpointer
since the graph runs once per event (avoids unbounded memory growth).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any
from uuid import UUID

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from hazard_assessment.agents.llm_advisory.prompts import (
    DRAFT_SYSTEM_PROMPT,
    GAPS_SYSTEM_PROMPT,
    TIMELINE_SYSTEM_PROMPT,
)
from hazard_assessment.agents.llm_advisory.schemas import AfterActionState

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from hazard_assessment.audit.logger import AuditLogger
    from hazard_assessment.config.settings import LLMSettings

logger = logging.getLogger(__name__)


def build_after_action_graph(
    settings: LLMSettings,
    audit_logger: AuditLogger,
    *,
    pinned_event_id: UUID,
    tool_call_log: list[dict[str, Any]] | None = None,
) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build the 3-node after-action analysis graph with tool use.

    Args:
        settings: LLM configuration.
        audit_logger: Audit logger for tool queries.
        pinned_event_id: Event UUID to scope all tool queries to.
        tool_call_log: Optional mutable list; every tool invocation the
            LLM makes (including unknown-tool requests, tool errors, and
            loop non-convergence) appends one record here.
            The caller owns persistence of this log.

    Returns:
        A compiled LangGraph graph.
    """
    from hazard_assessment.agents.llm_advisory.factory import build_chat_model
    from hazard_assessment.agents.llm_advisory.tools import make_after_action_tools

    quality_llm = build_chat_model(settings, purpose="quality")
    tools = make_after_action_tools(
        audit_logger, pinned_event_id=pinned_event_id, call_log=tool_call_log
    )
    llm_with_tools = quality_llm.bind_tools(tools)

    def timeline_node(state: AfterActionState) -> dict[str, Any]:
        """Reconstruct event timeline using tool-based audit queries."""
        try:
            initial_messages = [
                SystemMessage(content=TIMELINE_SYSTEM_PROMPT),
                HumanMessage(content=f"Reconstruct the timeline for event {state['event_id']}."),
            ]
            result = llm_with_tools.invoke(initial_messages)
            timeline_text = _resolve_tool_calls(
                result, llm_with_tools, tools, state,
                initial_messages=initial_messages,
                call_log=tool_call_log,
                node="timeline",
            )
            return {"timeline": timeline_text}
        except Exception:
            logger.exception("Timeline node failed")
            return {"timeline": "(timeline reconstruction failed)"}

    def gaps_node(state: AfterActionState) -> dict[str, Any]:
        """Identify detection gaps using tool queries and timeline context."""
        try:
            context = state.get("timeline", "(no timeline available)")
            initial_messages = [
                SystemMessage(content=GAPS_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Event: {state['event_id']}\n\n"
                        f"Timeline:\n{context}\n\n"
                        f"Identify gaps and near-misses."
                    )
                ),
            ]
            result = llm_with_tools.invoke(initial_messages)
            gaps_text = _resolve_tool_calls(
                result, llm_with_tools, tools, state,
                initial_messages=initial_messages,
                call_log=tool_call_log,
                node="gaps",
            )
            return {"gaps": gaps_text}
        except Exception:
            logger.exception("Gaps node failed")
            return {"gaps": "(gap analysis failed)"}

    def draft_node(state: AfterActionState) -> dict[str, Any]:
        """Draft the after-action report from timeline and gap analysis."""
        try:
            result = quality_llm.invoke([
                SystemMessage(content=DRAFT_SYSTEM_PROMPT),
                HumanMessage(
                    content=(
                        f"Event: {state['event_id']}\n\n"
                        f"Timeline:\n{state.get('timeline', '(unavailable)')}\n\n"
                        f"Gaps:\n{state.get('gaps', '(unavailable)')}\n\n"
                        f"Produce the after-action report."
                    )
                ),
            ])
            return {"draft_report": result.content}
        except Exception:
            logger.exception("Draft node failed")
            return {"draft_report": None}

    graph = StateGraph(AfterActionState)
    graph.add_node("timeline", timeline_node)
    graph.add_node("gaps", gaps_node)
    graph.add_node("draft", draft_node)
    graph.add_edge(START, "timeline")
    graph.add_edge("timeline", "gaps")
    graph.add_edge("gaps", "draft")
    graph.add_edge("draft", END)

    return graph.compile()


def _resolve_tool_calls(
    result: Any,
    llm_with_tools: Any,
    tools: list[Any],
    state: AfterActionState,
    *,
    max_rounds: int = 3,
    initial_messages: list[Any] | None = None,
    call_log: list[dict[str, Any]] | None = None,
    node: str = "",
) -> str:
    """Execute tool calls iteratively until the LLM produces a text response.

    This implements the agentic tool-use loop: the LLM decides which tools
    to call, we execute them, and feed results back until the LLM is satisfied
    or max_rounds is reached.

    Args:
        initial_messages: The system/human messages from the original invocation.
            Included in each LLM re-invocation to preserve context.
        call_log: Optional mutable list receiving one record per tool
            call. Successful calls are recorded inside the
            tools themselves; this loop additionally records unknown
            tools, tool errors, and non-convergence, and stamps each
            record with the graph node that made the call.
        node: Graph node label ("timeline" or "gaps") for attribution.
    """
    from langchain_core.messages import ToolMessage

    messages: list[Any] = list(initial_messages or []) + [result]
    tool_map = {t.name: t for t in tools}

    for _ in range(max_rounds):
        if not hasattr(result, "tool_calls") or not result.tool_calls:
            break

        # Execute each tool call
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

        # Feed tool results back to LLM
        result = llm_with_tools.invoke(messages)
        messages.append(result)

    # Guard: if the loop exhausted max_rounds with tool calls still pending
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

    # Return the final text content
    if hasattr(result, "content"):
        return str(result.content)
    return str(result)
