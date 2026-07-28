"""State and output schemas for the LLM advisory graphs.

LLMSynthesisState is passed through the 4-node synthesis graph.
Pydantic output models constrain LLM responses to structured fields.

Design constraint: no output schema includes a "confidence" field.
LLM self-reported confidence is unreliable; system confidence is
computed deterministically before graph invocation.
"""

from __future__ import annotations

from typing import TypedDict

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Synthesis graph state - TypedDict for LangGraph StateGraph
# ---------------------------------------------------------------------------


class LLMSynthesisState(TypedDict):
    """Shared state passed through the 4-node synthesis graph.

    Read-only inputs (set once before graph.invoke, never modified by nodes):
        event_id, report_tier, fsm_state, rayleigh_wave_suspect,
        top_scenario_json, verification_outcome, station_count,
        ensemble_spread, system_confidence.

    Progressive outputs (each populated by the corresponding node):
        similar_events_json, evidence_synthesis, scenario_interpretation,
        narrative.
    """

    # Read-only inputs from deterministic pipeline
    event_id: str
    report_tier: int
    fsm_state: str
    rayleigh_wave_suspect: bool
    top_scenario_json: str
    verification_outcome: str  # "PASS" | "PASS_WITH_CONCERNS"
    station_count: int  # len(scenario.dart_stations_used), NOT scenario.station_count
    ensemble_spread: str  # EnsembleSpread enum value: "LOW" | "MODERATE" | "HIGH"
    system_confidence: float  # pre-computed deterministic score (0.0-1.0)

    # Progressive node outputs
    similar_events_json: str  # populated by retrieval_node (pure Python, no LLM)
    evidence_synthesis: str | None
    scenario_interpretation: str | None
    narrative: str | None


# ---------------------------------------------------------------------------
# After-action graph state
# ---------------------------------------------------------------------------


class AfterActionState(TypedDict):
    """Shared state for the 3-node after-action analysis graph.

    The after-action graph uses tool-calling: timeline_node and gaps_node
    invoke read-only query tools to retrieve audit data on demand rather
    than receiving everything upfront. This makes the LLM layer genuinely
    agentic (LLM decides which data to retrieve).
    """

    event_id: str
    timeline: str | None
    gaps: str | None
    draft_report: str | None


# ---------------------------------------------------------------------------
# Structured output schemas for synthesis graph LLM nodes
# ---------------------------------------------------------------------------


class EvidenceSynthesisOutput(BaseModel):
    """Output of the evidence_node: plain-language summary of sensor data."""

    synthesis: str
    rayleigh_note: str | None = None


class ScenarioInterpOutput(BaseModel):
    """Output of the scenario_node: interpretation of best-fit scenario."""

    interpretation: str
    uncertainty_note: str


class NarrativeOutput(BaseModel):
    """Output of the narrative_node: full operator-facing narrative.

    The prompt constrains the LLM to avoid specific numerical values for
    wave heights, arrival times, or probabilities. All quantitative values
    come from the structured FinalAssessment fields, not from prose.
    """

    narrative: str
