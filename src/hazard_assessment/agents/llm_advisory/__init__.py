"""Bounded LLM advisory layer for the hazard assessment pipeline.

Provides optional LLM-powered narrative synthesis (4-node LangGraph graph)
and after-action analysis (3-node LangGraph graph with tool use). All LLM
outputs are advisory - they cannot modify FSM state, alter anomaly scores,
or suppress detection signals.

The synthesis graph enhances the Report Agent's template-based output with
natural-language narratives when an LLM API key is configured and system
confidence is above the routing threshold. The after-action graph is
triggered post-event via API endpoint for forensic timeline reconstruction.

Both graphs degrade gracefully: any failure falls back to template output
(synthesis) or returns an error response (after-action).
"""
