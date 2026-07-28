"""Audit trail endpoint for the Mission Control BFF."""

from __future__ import annotations

import httpx
from fastapi import APIRouter, Query

from backend.errors import raise_upstream_error
from backend.models.schemas import AuditEntryOut
from backend.services.demo_snapshot import TOHOKU_SNAPSHOT
from backend.services.hazard_client import hazard_client

router = APIRouter(tags=["audit"])


@router.get("/audit", response_model=list[AuditEntryOut])
async def get_audit(
    event_id: str | None = Query(None),
    event_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
) -> list[AuditEntryOut]:
    """Return recent audit trail entries."""
    try:
        return await hazard_client.get_audit_entries(
            event_id=event_id, event_type=event_type, limit=limit
        )
    except RuntimeError:
        # No core API key configured: genuine demo mode.
        entries = [AuditEntryOut(**e) for e in TOHOKU_SNAPSHOT["recent_audit"]]
        return entries[:limit]
    except httpx.HTTPError as exc:
        raise_upstream_error(exc)
