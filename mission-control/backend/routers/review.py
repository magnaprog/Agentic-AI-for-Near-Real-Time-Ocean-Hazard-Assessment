"""Human review and escalation endpoints for the Mission Control BFF.

Escalation packet retrieval.
Caller-gated review bound to durable packet identity and hash.
"""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter, Depends, HTTPException

from backend.errors import UpstreamKeyNotConfiguredError, raise_upstream_error
from backend.models.schemas import ReviewDecisionIn
from backend.security import require_reviewer_id_header
from backend.services.demo_snapshot import DEMO_ESCALATION_PACKET, TOHOKU_SNAPSHOT
from backend.services.hazard_client import hazard_client

router = APIRouter(tags=["review"])


@router.get("/review/pending")
async def get_pending_reviews() -> list[dict[str, Any]]:
    """Return assessments awaiting human review."""
    try:
        return await hazard_client.get_pending_reviews()
    except UpstreamKeyNotConfiguredError:
        # No core API key configured: genuine demo mode.
        ctx = TOHOKU_SNAPSHOT["fsm"]["event_context"]
        return [{**ctx, "fsm_state": TOHOKU_SNAPSHOT["fsm"]["fsm_state"]}]
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)


@router.get("/review/escalation")
async def get_escalation_packet() -> dict[str, Any]:
    """Return the active escalation packet for the current ESCALATE event.

    The durable row ID and canonical packet hash from this response must be
    included in the review request. Demo mode serves the same wrapper shape.
    """
    try:
        return await hazard_client.get_escalation_packet()
    except UpstreamKeyNotConfiguredError:
        # No core API key configured: genuine demo mode.
        return DEMO_ESCALATION_PACKET
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)


@router.post("/review/decide")
async def submit_decision(
    decision: ReviewDecisionIn,
    reviewer_id: str = Depends(require_reviewer_id_header),
) -> dict[str, Any]:
    """Submit a human review decision (APPROVE/REJECT/DEFER).

    Core API verifies packet row identity, canonical hash, event binding,
    and bound assessment identity before recording the review.
    """
    try:
        return await hazard_client.submit_review(
            decision,
            reviewer_id=reviewer_id,
        )
    except UpstreamKeyNotConfiguredError:
        # No core API key configured: decisions cannot be recorded in demo mode.
        raise HTTPException(
            status_code=503,
            detail="Review decisions require the core API (unavailable in demo mode)",
        )
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)
