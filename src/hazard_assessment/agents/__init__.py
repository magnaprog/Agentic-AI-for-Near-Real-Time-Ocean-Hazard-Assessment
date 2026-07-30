"""Agent definitions for the ocean hazard assessment system.

Each agent declares its capabilities via an AgentManifest and exposes
typed entry points (see agents/base.py). The manifest states the bounded
autonomy contract; it is a declaration, not an interceptor. No policy
middleware exists, and nothing consults the declared set at run time. The
bounds are held by per-role database grants, the terminology guardrail
scanner, fail-closed ABSTAIN routing, and the human review gate.
"""

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent

__all__ = [
    "AgentCapability",
    "AgentManifest",
    "BaseAgent",
]
