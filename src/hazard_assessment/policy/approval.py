"""On-demand policy check for agent capability and human approval.

Evaluates a proposed agent action against the permission matrix and
against the human-approval requirement for critical outputs, returning a
structured denial when either fails. This is a query interface, not an
interceptor: it is reached only through the /api/policy/check endpoint,
and nothing in the pipeline execution path calls it.

"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import yaml

from hazard_assessment.agents.base import AgentCapability
from hazard_assessment.audit.logger import AuditEntry

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# FSM state names - string literals to avoid a circular import from
# orchestrator.states.  These must stay in sync with SystemState values.
# SystemState is a StrEnum, so string comparison works correctly.
# ---------------------------------------------------------------------------

_HUMAN_APPROVAL_REQUIRED_STATES: frozenset[str] = frozenset({"ESCALATE"})

# Capabilities that require human approval when the FSM is in an
# approval-required state.  Currently only EMIT_REPORT, but the set
# is extensible.
_HUMAN_GATED_CAPABILITIES: frozenset[AgentCapability] = frozenset(
    {AgentCapability.EMIT_REPORT}
)


# ---------------------------------------------------------------------------
# Policy denial codes
# ---------------------------------------------------------------------------


class DenialReason(StrEnum):
    """Codes indicating why a policy check failed."""

    CAPABILITY_NOT_ALLOWED = "CAPABILITY_NOT_ALLOWED"
    HUMAN_APPROVAL_REQUIRED = "HUMAN_APPROVAL_REQUIRED"
    AGENT_NOT_REGISTERED = "AGENT_NOT_REGISTERED"


# ---------------------------------------------------------------------------
# Result data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PolicyDenial:
    """Details of a denied agent action."""

    agent_name: str
    capability: str
    fsm_state: str
    reason: DenialReason
    detail: str


@dataclass
class PolicyCheckResult:
    """Result of a policy check. ``denial`` is set when ``allowed`` is False."""

    allowed: bool
    denial: PolicyDenial | None = None
    checked_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.checked_at_utc.tzinfo is None:
            raise ValueError("checked_at_utc must be timezone-aware")
        if not self.allowed and self.denial is None:
            raise ValueError(
                "denial must be set when allowed is False"
            )
        if self.allowed and self.denial is not None:
            raise ValueError(
                "denial must be None when allowed is True"
            )


# ---------------------------------------------------------------------------
# Permission matrix loader
# ---------------------------------------------------------------------------

_DEFAULT_PERMISSIONS_PATH = Path(__file__).parent / "permissions.yaml"


def load_permission_matrix(
    path: Path | None = None,
) -> dict[str, Any]:
    """Load the YAML permission matrix from *path* (defaults to packaged file)."""
    resolved = path or _DEFAULT_PERMISSIONS_PATH
    with open(resolved) as f:
        return yaml.safe_load(f)  # type: ignore[no-any-return]


def _agent_allowed_capabilities(
    matrix: dict[str, Any], agent_name: str
) -> set[str] | None:
    """Return the set of allowed capability codes for *agent_name*.

    Returns ``None`` if the agent is not listed in the matrix.
    """
    agents = matrix.get("agents", {})
    agent_entry = agents.get(agent_name)
    if agent_entry is None:
        return None
    return set(agent_entry.get("allowed", []))


# ---------------------------------------------------------------------------
# Core policy check
# ---------------------------------------------------------------------------


def check_policy(
    *,
    agent_name: str,
    capability: AgentCapability,
    fsm_state: str,
    human_decision_present: bool = False,
    matrix: dict[str, Any] | None = None,
) -> PolicyCheckResult:
    """Check whether *agent_name* may exercise *capability* in *fsm_state*.

    Validates agent registration, capability allowlist, and
    human-approval requirements. Returns a denial when any check fails.

    .. warning:: Trust boundary for ``human_decision_present``

       This parameter is **caller-asserted** - the function trusts the
       caller's claim that a human decision exists. It does NOT verify
       against the audit trail or any persistent store. Until the
       orchestrator pipeline wires this as a mandatory interceptor
       with audit-backed verification, a misbehaving agent could bypass
       the human gate by passing ``human_decision_present=True``.

       The intended enforcement point is the orchestrator pipeline, where
       the ``human_decision_present`` flag will be derived from the audit
       trail rather than from agent self-reporting.
    """
    # SAFETY NOTE: human_decision_present is caller-asserted. In a
    # production deployment, derive this from the audit trail (verify
    # that an APPROVE HumanDecision exists for the active event) rather
    # than trusting the caller's claim.
    if human_decision_present:
        import warnings
        warnings.warn(
            "human_decision_present is caller-asserted and unverified. "
            "Production deployments must derive this from the audit trail.",
            stacklevel=2,
        )

    if matrix is None:
        matrix = load_permission_matrix()

    # 1) Agent must be registered in the matrix
    allowed_caps = _agent_allowed_capabilities(matrix, agent_name)
    if allowed_caps is None:
        return PolicyCheckResult(
            allowed=False,
            denial=PolicyDenial(
                agent_name=agent_name,
                capability=capability.value,
                fsm_state=fsm_state,
                reason=DenialReason.AGENT_NOT_REGISTERED,
                detail=f"Agent '{agent_name}' is not registered in the permission matrix.",
            ),
        )

    # 2) Capability must be in the agent's allowed set
    if capability.value not in allowed_caps:
        return PolicyCheckResult(
            allowed=False,
            denial=PolicyDenial(
                agent_name=agent_name,
                capability=capability.value,
                fsm_state=fsm_state,
                reason=DenialReason.CAPABILITY_NOT_ALLOWED,
                detail=(
                    f"Agent '{agent_name}' does not have the "
                    f"'{capability.name}' capability."
                ),
            ),
        )

    # 3) Human-gated capabilities in approval-required states need
    #    a human decision to proceed.
    if (
        capability in _HUMAN_GATED_CAPABILITIES
        and fsm_state in _HUMAN_APPROVAL_REQUIRED_STATES
        and not human_decision_present
    ):
        return PolicyCheckResult(
            allowed=False,
            denial=PolicyDenial(
                agent_name=agent_name,
                capability=capability.value,
                fsm_state=fsm_state,
                reason=DenialReason.HUMAN_APPROVAL_REQUIRED,
                detail=(
                    f"Capability '{capability.name}' requires human approval "
                    f"in '{fsm_state}' state. No human decision is present."
                ),
            ),
        )

    return PolicyCheckResult(allowed=True)


# ---------------------------------------------------------------------------
# Audit helper
# ---------------------------------------------------------------------------


class _AuditAppender(Protocol):
    """Minimal interface required by log_denial (satisfied by AuditLogger)."""

    def append(self, entry: AuditEntry) -> None: ...


def log_denial(
    denial: PolicyDenial,
    audit_logger: _AuditAppender,
    *,
    event_id: UUID | None = None,
) -> None:
    """Write a policy denial to the audit trail."""
    entry = AuditEntry(
        event_id=event_id,
        event_type="policy_denial",
        producer="policy_middleware",
        data={
            "agent_name": denial.agent_name,
            "capability": denial.capability,
            "fsm_state": denial.fsm_state,
            "reason": denial.reason.value,
            "detail": denial.detail,
        },
    )
    audit_logger.append(entry)


def log_policy_result(
    result: PolicyCheckResult,
    audit_logger: Any,
    *,
    agent_name: str,
    capability: str,
    fsm_state: str,
    event_id: UUID | None = None,
) -> None:
    """Write ALL policy check results (passes and denials) to the audit trail.

    Used for agentic activity recording - captures the full permission
    decision surface for paper analysis.  Requires an ``AuditLogger``
    (or any object with ``log_permission_check``).
    """
    log_fn = getattr(audit_logger, "log_permission_check", None)
    if log_fn is not None:
        log_fn(
            event_id=event_id,
            agent=agent_name,
            capability=capability,
            allowed=result.allowed,
            reason=result.denial.detail if result.denial else f"allowed in {fsm_state}",
        )
    else:
        logger.warning(
            "audit_logger has no log_permission_check; policy result not recorded"
        )


# ---------------------------------------------------------------------------
# Structured denial response (for API layer)
# ---------------------------------------------------------------------------


def denial_to_response(denial: PolicyDenial) -> dict[str, Any]:
    """Convert a ``PolicyDenial`` to a structured dict for API responses."""
    return {
        "status": "denied",
        "policy_violation": {
            "agent_name": denial.agent_name,
            "capability": denial.capability,
            "fsm_state": denial.fsm_state,
            "reason": denial.reason.value,
            "detail": denial.detail,
        },
    }
