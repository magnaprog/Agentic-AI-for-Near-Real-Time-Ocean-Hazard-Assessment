"""Bounded LLM advisory layer for the hazard assessment pipeline.

Three model-backed paths, distinguished by when they run. All of them are
advisory: none can modify FSM state, alter anomaly scores, or suppress a
detection signal.

* Narrative synthesis (``synthesis_graph``): a four-node graph with fixed
  edges that supplements the Report Agent's template output once an
  assessment exists, and only when system confidence clears the routing
  threshold. The template text is emitted either way.
* Active-event investigation (``investigator``): the only path that runs
  while an event is open. For each of three issues the model chooses which
  read-only audit queries to run. Findings are written under a dedicated
  database role that the FSM-driving role cannot use, so a finding cannot
  become input to an escalation.
* After-action analysis (``after_action``): a three-node graph, triggered
  through an API endpoint once an event is no longer active.

The investigator and the after-action graph share the bounded tool loop and
the event-scoped read-only tools in ``tools``, so the round cap and the
per-call logging have one implementation rather than two.

Every path degrades rather than failing the caller: synthesis falls back to
template output, the investigator drops the affected issue and keeps the
others, and after-action returns an error response. Provider selection lives
in ``providers``, which is the one module that names vendors; nothing
else in the package does.
"""
