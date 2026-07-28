"""Agent definitions for the ocean hazard assessment system.

Each agent declares its capabilities via an AgentManifest and exposes
typed entry points (see agents/base.py). The manifest declares the bounded autonomy contract,
intended for enforcement by the policy check middleware.
"""

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent

__all__ = [
    "AgentCapability",
    "AgentManifest",
    "BaseAgent",
]
