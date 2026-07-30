"""HTTP client for the main hazard assessment system API."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

from backend.config import settings
from backend.models.schemas import (
    AuditEntryOut,
    FSMStateOut,
    ReviewDecisionIn,
    SystemSnapshotOut,
)
from backend.security import REVIEWER_ID_HEADER_NAME, required_hazard_api_key

logger = logging.getLogger(__name__)


class HazardClient:
    """Async HTTP client that proxies requests to the main system at :8000.

    Uses a shared httpx.AsyncClient for connection pooling. Call ``close()``
    on shutdown (handled by the FastAPI lifespan).
    """

    def __init__(self) -> None:
        self._base = settings.hazard_api_url
        self._client: httpx.AsyncClient | None = None

    async def startup(self) -> None:
        """Initialize the shared HTTP client once for connection reuse."""
        if self._client is None:
            self._client = httpx.AsyncClient(base_url=self._base, timeout=5.0)

    async def shutdown(self) -> None:
        """Close the shared HTTP client on application shutdown."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            await self.startup()
        if self._client is None:
            raise RuntimeError("HazardClient.startup() failed to initialize HTTP client")
        return self._client

    def _auth_headers(self) -> dict[str, str]:
        return {"X-Hazard-Api-Key": required_hazard_api_key()}

    async def get_fsm_state(self) -> FSMStateOut:
        client = await self._get_client()
        resp = await client.get("/api/fsm", headers=self._auth_headers())
        resp.raise_for_status()
        return FSMStateOut(**resp.json())

    async def get_events(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        resp = await client.get("/api/fsm", headers=self._auth_headers())
        resp.raise_for_status()
        data = resp.json()
        events: list[dict[str, Any]] = []
        if data.get("event_context"):
            events.append(
                {
                    **data["event_context"],
                    "fsm_state": data["fsm_state"],
                    "status": "active",
                }
            )
        return events

    async def get_pending_reviews(self) -> list[dict[str, Any]]:
        client = await self._get_client()
        resp = await client.get("/api/fsm", headers=self._auth_headers())
        resp.raise_for_status()
        data = resp.json()
        if data.get("fsm_state") != "ESCALATE" or not data.get("event_context"):
            return []

        packet_resp = await client.get(
            "/api/escalation/packet-of-record",
            headers=self._auth_headers(),
        )
        if packet_resp.status_code == 404:
            return []
        packet_resp.raise_for_status()
        packet = packet_resp.json()
        return [
            {
                **data["event_context"],
                "fsm_state": data["fsm_state"],
                "packet_row_id": packet["packet_row_id"],
                "packet_hash": packet["content_sha256"],
            }
        ]

    async def get_escalation_packet(self) -> dict[str, Any]:
        """Fetch the immutable packet-of-record row for the active event."""
        client = await self._get_client()
        resp = await client.get(
            "/api/escalation/packet-of-record",
            headers=self._auth_headers(),
        )
        resp.raise_for_status()
        packet: dict[str, Any] = resp.json()
        return packet

    async def submit_review(
        self,
        decision: ReviewDecisionIn,
        *,
        reviewer_id: str,
    ) -> dict[str, Any]:
        client = await self._get_client()
        # httpx encodes header values as ASCII unless told otherwise, and a
        # UnicodeEncodeError is a ValueError, which the router's RuntimeError
        # and HTTPError handlers do not catch. A reviewer whose name carries an
        # accent would get an opaque 500 on the one route that records human
        # authority. The core accepts any value that is 128 characters or
        # fewer and free of control characters, so encode rather than reject.
        headers = httpx.Headers(encoding="utf-8")
        for name, value in self._auth_headers().items():
            headers[name] = value
        headers[REVIEWER_ID_HEADER_NAME] = reviewer_id
        resp = await client.post(
            "/api/review",
            json=decision.model_dump(),
            headers=headers,
        )
        resp.raise_for_status()
        result: dict[str, Any] = resp.json()
        return result

    async def get_audit_entries(
        self,
        event_id: str | None = None,
        event_type: str | None = None,
        limit: int = 50,
    ) -> list[AuditEntryOut]:
        params: dict[str, str | int] = {"limit": limit}
        if event_id:
            params["event_id"] = event_id
        if event_type:
            params["event_type"] = event_type
        client = await self._get_client()
        resp = await client.get("/api/audit", params=params, headers=self._auth_headers())
        resp.raise_for_status()
        return [AuditEntryOut(**e) for e in resp.json()]

    async def get_snapshot(self) -> dict[str, Any]:
        """Fetch a complete snapshot for WebSocket broadcast.

        Requests are made concurrently via ``asyncio.gather`` with
        ``return_exceptions=True`` so a single endpoint failure does not
        discard the entire snapshot. Partial data (e.g., FSM state without
        component registry) is better than no data for a monitoring dashboard.
        """
        client = await self._get_client()
        headers = self._auth_headers()
        # The review feed is a separate query, not a slice of recent_audit. A
        # live worker writes an anomaly_scored entry per scored window, so the
        # 20 most recent entries can span a fraction of a second: a recorded
        # review falls out of that window almost immediately, and the console
        # would show a decided packet as undecided.
        fsm_result, agents_result, audit_result, reviews_result = await asyncio.gather(
            client.get("/api/fsm", headers=headers),
            client.get("/api/agents", headers=headers),
            client.get("/api/audit", params={"limit": 20}, headers=headers),
            client.get(
                "/api/audit",
                params={"limit": 20, "event_type": "assessment_review_decision"},
                headers=headers,
            ),
            return_exceptions=True,
        )

        # FSM is the critical component - if it fails, re-raise
        if isinstance(fsm_result, BaseException):
            raise fsm_result
        fsm_result.raise_for_status()

        # Non-critical components degrade gracefully, but a degraded section
        # is recorded by name. Silently returning an empty list made a failed
        # query indistinguishable from a genuinely empty one, and the console
        # renders those two very differently: an unavailable review history
        # reads as "NOT RECORDED" and re-arms the decision controls.
        degraded: list[str] = []

        agents_data: list[dict[str, Any]] = []
        if isinstance(agents_result, BaseException):
            logger.warning("agents query failed: %s", agents_result)
            degraded.append("agents")
        else:
            try:
                agents_result.raise_for_status()
                agents_data = agents_result.json()
            except Exception as exc:
                logger.warning("agents query failed: %s", exc)
                degraded.append("agents")

        audit_data: list[AuditEntryOut] = []
        if isinstance(audit_result, BaseException):
            logger.warning("audit query failed: %s", audit_result)
            degraded.append("recent_audit")
        else:
            try:
                audit_result.raise_for_status()
                audit_data = [AuditEntryOut(**entry) for entry in audit_result.json()]
            except Exception as exc:
                logger.warning("audit query failed: %s", exc)
                degraded.append("recent_audit")

        reviews_data: list[AuditEntryOut] = []
        if isinstance(reviews_result, BaseException):
            logger.warning("reviews query failed: %s", reviews_result)
            degraded.append("recent_reviews")
        else:
            try:
                reviews_result.raise_for_status()
                reviews_data = [AuditEntryOut(**entry) for entry in reviews_result.json()]
            except Exception as exc:
                logger.warning("reviews query failed: %s", exc)
                degraded.append("recent_reviews")

        snapshot = SystemSnapshotOut(
            fsm=FSMStateOut(**fsm_result.json()),
            agents=agents_data,
            recent_audit=audit_data,
            recent_reviews=reviews_data,
            degraded_sections=degraded,
        )
        return snapshot.model_dump(mode="json")


hazard_client = HazardClient()
