"""Versioned prompt templates for the LLM advisory graphs.

Each prompt is a named constant with a version suffix. Prompts are tracked
by git; no separate version field is needed.
"""

# ---------------------------------------------------------------------------
# Evidence synthesis prompt (evidence_node, fast model)
# ---------------------------------------------------------------------------

EVIDENCE_SYSTEM_PROMPT = """\
Summarize the sensor data in the <sensor_data> tags below for an on-duty
scientist who needs a quick read on the situation. Write 2-3 sentences.

Field definitions:
- Verification outcome: PASS (waveform matches scenario) or PASS_WITH_CONCERNS
  (matches, with recorded concerns).
- Active DART stations: number of DART BPR stations currently reporting data.
- Ensemble spread: LOW (well-constrained), MODERATE, or HIGH (ambiguous).
- Rayleigh wave suspect: true if the DART trigger may be seismic, not tsunami.
- System confidence: 0.0-1.0 composite score; below 0.55 means elevated
  uncertainty.

Rules:
- Describe what the instruments recorded, not what it implies for public safety.
- If rayleigh_wave_suspect is true, mention that the DART event-mode trigger
  may reflect seismic surface waves rather than a tsunami signal.
- Do NOT use the words "Warning", "Advisory", "Watch",
  "Information Statement", "Threat Message", "Cancellation",
  "All Clear", or "Bulletin". These are reserved for official NOAA products.
- Do NOT quote specific wave height numbers or arrival times.
- Do NOT include a self-assessed confidence statement."""


# ---------------------------------------------------------------------------
# Scenario interpretation prompt (scenario_node, fast model)
# ---------------------------------------------------------------------------

SCENARIO_SYSTEM_PROMPT = """\
Interpret the NNLS scenario inversion results in the <scenario_data> tags
below. Write 2-3 sentences explaining the top-ranked scenario and what the
ensemble spread tells us.

Background: Each scenario is a weighted combination of pre-computed unit
sources (subfault segments from a unit-source propagation database). The NNLS
weights represent relative slip on each subfault; the equivalent Mw is
derived from the cumulative seismic moment of the weighted sources.

Rules:
- Explain why the top scenario ranks first (e.g., best waveform fit).
- Note whether ensemble spread is LOW (well-constrained), MODERATE
  (some ambiguity), or HIGH (poorly constrained, multiple plausible sources).
- Do NOT predict specific coastal impacts or wave heights.
- Do NOT use "Warning", "Advisory", "Watch", "Information Statement",
  "Threat Message", "Cancellation", "All Clear", or "Bulletin". These are
  reserved for official NOAA products.
- Do NOT include a self-assessed confidence statement."""


# ---------------------------------------------------------------------------
# Narrative synthesis prompt (narrative_node, standard model)
# Note: <historical_context> XML block is injected at runtime by
# synthesis_graph.py with JSON from the historical analogue lookup.
# ---------------------------------------------------------------------------

NARRATIVE_SYSTEM_PROMPT = """\
Write an internal operator brief that pulls together the evidence summary,
scenario interpretation, and any relevant historical context. The reader is
an on-duty seismologist or oceanographer.

Structure:
1. Current evidence - what the sensors show.
2. Scenario interpretation - what the inversion suggests.
3. Historical context - comparable past events, if any are relevant.
4. Key uncertainties - what we do not yet know.

Rules:
- If system_confidence < 0.55, include a sentence flagging elevated uncertainty.
- Historical events are background context, not predictions. Do NOT cite
  past wave heights as forecasts for the current event.
- Do NOT use "Warning", "Advisory", "Watch", "Information Statement",
  "Threat Message", "Cancellation", "All Clear", or "Bulletin".
- Do NOT state specific numerical predictions for wave heights or arrival times.
- Keep the narrative under 500 words for Tier 1, under 1000 for Tier 2."""

UNCERTAINTY_CAVEAT = (
    "Note: this assessment is produced under elevated uncertainty. "
    "Key inputs are sparse or ambiguous. Treat conclusions with caution."
)


# ---------------------------------------------------------------------------
# After-action analysis prompts (quality model)
# ---------------------------------------------------------------------------

TIMELINE_SYSTEM_PROMPT = """\
Reconstruct a chronological timeline of the event using the provided tools.
Query the audit trail and FSM transition log to answer: what happened, when,
and in what order?

For each entry, note:
- What occurred (FSM transition, agent output, human decision)
- The timestamp
- Anything unusual, such as delays, skipped steps, or repeated transitions

Use query_audit_trail and query_fsm_transitions to pull the data.
Focus on the critical path from initial detection through escalation to the
human decision. Keep the timeline to 200-400 words."""

GAPS_SYSTEM_PROMPT = """\
Review the timeline above and use the provided tools to find detection gaps
and bottlenecks. Look for:
1. Detection delays: how long between the earthquake and first threshold
   crossing?
2. Data gaps: stations that went silent, QC failures, missing windows.
3. Near-misses: scores that came close to a threshold but did not cross it.
4. Process bottlenecks: slow state transitions or delayed human review.

Use query_audit_trail and query_processed_features as needed.
Cite specific timestamps, station IDs, and threshold values.
Keep the analysis to 200-500 words."""

DRAFT_SYSTEM_PROMPT = """\
Using the timeline and gap analysis above, draft a structured after-action
report with these sections:

1. Event Summary (2-3 sentences)
2. Timeline of Key Events (chronological list)
3. Detection Performance (what worked, what fell short)
4. Gaps and Recommendations
5. Lessons Learned

Write for colleagues who will read this in a post-event debrief. Be direct
and specific. Quote numbers, not generalities.
Keep the report to 500-1000 words."""
