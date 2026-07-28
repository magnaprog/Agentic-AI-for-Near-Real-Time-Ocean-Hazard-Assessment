"""Agent base classes for the hazard assessment pipeline.

Each agent declares its capabilities via an AgentManifest. The manifest
is a machine-readable contract evaluated on demand by
``policy/approval.py``; it is not enforced as an interceptor around agent
execution.

Each concrete agent exposes its own typed entry point (e.g.,
``process_records()``, ``process_station_data()``, ``verify()``,
``synthesize()``) rather than a uniform ``process()`` signature,
because each agent's inputs and outputs differ fundamentally.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class AgentCapability(StrEnum):
    """Declared capabilities for policy enforcement.

    The declared policy lives in ``policy/permissions.yaml`` and is
    evaluated on demand by ``policy/approval.py``. It is a declarative
    contract, not an interceptor: nothing in the pipeline execution path
    consults it, so bounded autonomy rests on architectural separation
    rather than on a runtime capability check.
    """

    READ_DATA = "RD"
    WRITE_DATA = "WD"
    WRITE_AUDIT = "WA"
    PRODUCE_KAFKA = "PK"
    CONSUME_KAFKA = "CK"
    INVOKE_LLM = "IL"
    MODIFY_STATE = "MS"
    EMIT_REPORT = "ER"
    APPROVE_OUTPUT = "AO"


class AgentManifest(BaseModel):
    """Machine-readable declaration of an agent's identity and capabilities.

    Declares the bounded autonomy contract: an agent should not perform
    actions outside its declared capabilities. The contract is evaluated
    on demand by ``policy/approval.py`` and exposed through the
    ``/api/policy/check`` endpoint. It is not wired as an interceptor in
    the pipeline execution path.
    """

    name: str = Field(description="Unique agent name (e.g., 'qc_agent')")
    version: str = Field(default="0.1.0", description="Agent version")
    capabilities: list[AgentCapability] = Field(
        description="Declared capabilities for policy enforcement"
    )
    description: str = Field(default="", description="Human-readable purpose")

    model_config = {"extra": "forbid"}


class BaseAgent:
    """Base class for all pipeline agents.

    Provides manifest access and common identity properties.
    Concrete agents expose their own typed entry points rather
    than a uniform ``process()`` method.
    """

    def __init__(self, manifest: AgentManifest) -> None:
        self._manifest = manifest

    @property
    def manifest(self) -> AgentManifest:
        return self._manifest

    @property
    def name(self) -> str:
        return self._manifest.name
