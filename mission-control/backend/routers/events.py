"""Event endpoints for the Mission Control BFF."""

from __future__ import annotations

from typing import Any

import httpx
from fastapi import APIRouter

from backend.errors import UpstreamKeyNotConfiguredError, raise_upstream_error
from backend.services.demo_snapshot import TOHOKU_SNAPSHOT
from backend.services.hazard_client import hazard_client

router = APIRouter(tags=["events"])


@router.get("/events")
async def get_events() -> list[dict[str, Any]]:
    """Return the current active event, or an empty list."""
    try:
        return await hazard_client.get_events()
    except UpstreamKeyNotConfiguredError:
        # No core API key configured: genuine demo mode.
        ctx = TOHOKU_SNAPSHOT["fsm"]["event_context"]
        return [
            {**ctx, "fsm_state": TOHOKU_SNAPSHOT["fsm"]["fsm_state"], "status": "active"}
        ]
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)
