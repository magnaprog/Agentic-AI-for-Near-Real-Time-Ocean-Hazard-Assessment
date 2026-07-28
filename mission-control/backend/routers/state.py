"""GET /api/mc/state - aggregated FSM snapshot."""

from __future__ import annotations

import httpx
from fastapi import APIRouter

from backend.errors import raise_upstream_error
from backend.models.schemas import FSMStateOut
from backend.services.demo_snapshot import TOHOKU_SNAPSHOT
from backend.services.hazard_client import hazard_client

router = APIRouter(tags=["state"])


@router.get("/state", response_model=FSMStateOut)
async def get_state() -> FSMStateOut:
    """Return the current FSM state, event context, and transition history."""
    try:
        return await hazard_client.get_fsm_state()
    except RuntimeError:
        # No core API key configured: genuine demo mode.
        return FSMStateOut(**TOHOKU_SNAPSHOT["fsm"])
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)
