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
    from hazard_assessment.agents.llm_advisory.tools import (
        make_event_query_tools,
        resolve_tool_calls,
    )

    tools = make_event_query_tools(
        audit_logger, pinned_event_id=pinned_event_id, call_log=tool_call_log
    )
    # Two clients over the same model: the draft node writes prose and must not
    # be offered tools, while the timeline and gaps nodes need them. Tools are
    # bound inside the factory because it returns a retry wrapper that cannot
    # accept them afterwards.
    quality_llm = build_chat_model(settings, purpose="quality")
    llm_with_tools = build_chat_model(settings, purpose="quality", tools=tools)

    def timeline_node(state: AfterActionState) -> dict[str, Any]:
        """Reconstruct event timeline using tool-based audit queries."""
        try:
            initial_messages = [
                SystemMessage(content=TIMELINE_SYSTEM_PROMPT),
                HumanMessage(content=f"Reconstruct the timeline for event {state['event_id']}."),
            ]
            result = llm_with_tools.invoke(initial_messages)
            timeline_text = resolve_tool_calls(
                result, llm_with_tools, tools,
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
            gaps_text = resolve_tool_calls(
                result, llm_with_tools, tools,
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
