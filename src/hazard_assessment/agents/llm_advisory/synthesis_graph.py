"""4-node LangGraph synthesis graph for operator narrative generation.

Graph topology: START -> retrieval -> evidence -> scenario -> narrative -> END

- retrieval_node: pure Python (no LLM). Retrieves similar historical events.
- evidence_node: LLM (fast model). Summarises sensor evidence.
- scenario_node: LLM (fast model). Interprets NNLS scenario results.
- narrative_node: LLM (standard model). Produces full operator narrative.

The graph is compiled once at ReportAgent construction time and reused
for all invocations. No checkpointer is used (linear graph runs to
completion in one pass; omitting avoids unbounded memory growth).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from hazard_assessment.agents.llm_advisory.prompts import (
    EVIDENCE_SYSTEM_PROMPT,
    NARRATIVE_SYSTEM_PROMPT,
    SCENARIO_SYSTEM_PROMPT,
    UNCERTAINTY_CAVEAT,
)
from hazard_assessment.agents.llm_advisory.retrieval import retrieve_similar_events
from hazard_assessment.agents.llm_advisory.schemas import (
    EvidenceSynthesisOutput,
    LLMSynthesisState,
    NarrativeOutput,
    ScenarioInterpOutput,
)

if TYPE_CHECKING:
    from langgraph.graph.state import CompiledStateGraph

    from hazard_assessment.config.settings import LLMSettings

logger = logging.getLogger(__name__)


def build_synthesis_graph(settings: LLMSettings) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the 4-node synthesis graph.

    Args:
        settings: LLM configuration for model construction.

    Returns:
        A compiled LangGraph graph ready for ``graph.invoke(state, config)``.
    """
    from hazard_assessment.agents.llm_advisory.factory import build_chat_model

    fast_llm = build_chat_model(settings, purpose="fast")
    standard_llm = build_chat_model(settings, purpose="standard")

    def retrieval_node(state: LLMSynthesisState) -> dict[str, Any]:
        """Pure Python: retrieve similar historical events by magnitude."""
        try:
            mw = _parse_scenario_mw(state["top_scenario_json"])
            # A non-positive Mw means parsing failed (see _parse_scenario_mw):
            # retrieve nothing rather than rank the lowest-magnitude analogues
            # as if they were comparable to an unknown-magnitude event.
            similar = retrieve_similar_events(mw) if mw > 0.0 else "[]"
        except (ValueError, KeyError, IndexError):
            logger.exception("Retrieval node failed; returning empty list")
            similar = "[]"
        return {"similar_events_json": similar}

    def evidence_node(state: LLMSynthesisState) -> dict[str, Any]:
        """LLM: summarise sensor evidence in plain language."""
        try:
            llm = fast_llm.with_structured_output(EvidenceSynthesisOutput)
            result = llm.invoke([
                SystemMessage(content=EVIDENCE_SYSTEM_PROMPT),
                HumanMessage(content=_format_evidence_input(state)),
            ])
            if not isinstance(result, EvidenceSynthesisOutput):
                logger.warning("Evidence node: structured output returned %s", type(result))
                return {"evidence_synthesis": "(LLM did not produce parseable output)"}
            text = result.synthesis
            if result.rayleigh_note:
                text += f"\nNote: {result.rayleigh_note}"
            return {"evidence_synthesis": text}
        except Exception:
            # Broad catch is intentional: LLM provider errors vary widely
            # (rate limits, timeouts, malformed responses). Log full traceback.
            logger.exception("Evidence node failed - LLM or network error")
            return {"evidence_synthesis": "(evidence synthesis unavailable)"}

    def scenario_node(state: LLMSynthesisState) -> dict[str, Any]:
        """LLM: interpret NNLS scenario inversion results."""
        try:
            llm = fast_llm.with_structured_output(ScenarioInterpOutput)
            result = llm.invoke([
                SystemMessage(content=SCENARIO_SYSTEM_PROMPT),
                HumanMessage(content=_format_scenario_input(state)),
            ])
            if not isinstance(result, ScenarioInterpOutput):
                logger.warning("Scenario node: structured output returned %s", type(result))
                return {"scenario_interpretation": "(LLM did not produce parseable output)"}
            return {
                "scenario_interpretation": (
                    f"{result.interpretation}\n{result.uncertainty_note}"
                ),
            }
        except Exception:
            # Broad catch: LLM provider errors vary widely.
            logger.exception("Scenario node failed - LLM or network error")
            return {"scenario_interpretation": "(scenario interpretation unavailable)"}

    def narrative_node(state: LLMSynthesisState) -> dict[str, Any]:
        """LLM: produce full operator narrative combining all inputs."""
        try:
            llm = standard_llm.with_structured_output(NarrativeOutput)
            result = llm.invoke([
                SystemMessage(content=NARRATIVE_SYSTEM_PROMPT),
                HumanMessage(content=_format_narrative_input(state)),
            ])
            if not isinstance(result, NarrativeOutput):
                logger.warning("Narrative node: structured output returned %s", type(result))
                return {"narrative": None}
            return {"narrative": result.narrative}
        except Exception:
            # Broad catch: LLM provider errors vary widely.
            logger.exception("Narrative node failed - LLM or network error")
            return {"narrative": None}

    graph = StateGraph(LLMSynthesisState)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("evidence", evidence_node)
    graph.add_node("scenario", scenario_node)
    graph.add_node("narrative", narrative_node)
    graph.add_edge(START, "retrieval")
    graph.add_edge("retrieval", "evidence")
    graph.add_edge("evidence", "scenario")
    graph.add_edge("scenario", "narrative")
    graph.add_edge("narrative", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Input formatting helpers
# ---------------------------------------------------------------------------


def _parse_scenario_mw(top_scenario_json: str) -> float:
    """Extract Mw from the top scenario JSON.

    Returns 0.0 on parse failure so that similar-event retrieval
    returns an empty list rather than misleading M7.0 comparisons.
    """
    try:
        data = json.loads(top_scenario_json)
    except (json.JSONDecodeError, TypeError):
        logger.warning("_parse_scenario_mw: could not parse scenario JSON")
        return 0.0
    if isinstance(data, dict) and "mw_equivalent" in data:
        return float(data["mw_equivalent"])
    logger.warning("_parse_scenario_mw: mw_equivalent not found in scenario")
    return 0.0


def _format_evidence_input(state: LLMSynthesisState) -> str:
    """Format sensor evidence for the evidence_node prompt."""
    return (
        f"<sensor_data>\n"
        f"Verification outcome: {state['verification_outcome']}\n"
        f"Active DART stations: {state['station_count']}\n"
        f"Ensemble spread: {state['ensemble_spread']}\n"
        f"Rayleigh wave suspect: {state['rayleigh_wave_suspect']}\n"
        f"System confidence: {state['system_confidence']:.2f}\n"
        f"</sensor_data>"
    )


def _format_scenario_input(state: LLMSynthesisState) -> str:
    """Format scenario data for the scenario_node prompt."""
    return (
        f"<scenario_data>\n"
        f"{state['top_scenario_json']}\n"
        f"Ensemble spread: {state['ensemble_spread']}\n"
        f"</scenario_data>"
    )


def _format_narrative_input(state: LLMSynthesisState) -> str:
    """Combine evidence and scenario for the narrative_node."""
    similar = state.get("similar_events_json", "[]")
    parts = [
        f"Report tier: {state['report_tier']}",
        f"FSM state: {state['fsm_state']}",
        f"System confidence: {state['system_confidence']:.2f}",
        "",
        "Evidence summary:",
        state.get("evidence_synthesis") or "(unavailable)",
        "",
        "Scenario interpretation:",
        state.get("scenario_interpretation") or "(unavailable)",
        "",
        "<historical_context>",
        similar,
        "</historical_context>",
    ]
    if state["system_confidence"] < 0.55:
        parts.append(f"\n{UNCERTAINTY_CAVEAT}")
    return "\n".join(parts)
