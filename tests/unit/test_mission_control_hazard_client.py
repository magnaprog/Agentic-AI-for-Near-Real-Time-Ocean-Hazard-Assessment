"""Unit tests for the Mission Control HazardClient escalation packet flow."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

_repo_root = Path(__file__).resolve().parents[2]
_mc_root = _repo_root / "mission-control"
if str(_mc_root) not in sys.path:
    sys.path.insert(0, str(_mc_root))

from backend.config import settings  # noqa: E402
from backend.services.hazard_client import HazardClient  # noqa: E402

_PACKET: dict[str, Any] = {
    "packet_row_id": 3,
    "assessment_row_id": 41,
    "event_id": "ev-1",
    "content_sha256": "a" * 64,
    "packet": {"kind": "escalation_reviewer_packet"},
}


@pytest.fixture(autouse=True)
def _hazard_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The settings singleton is frozen at import time; patch the key directly
    so these tests do not depend on env state or module import order."""
    monkeypatch.setattr(settings, "hazard_api_key", "hazard-key")


def _client_with_transport(
    handler: Any,
) -> HazardClient:
    client = HazardClient()
    client._client = httpx.AsyncClient(
        base_url="http://core", transport=httpx.MockTransport(handler)
    )
    return client


async def test_packet_fetch_uses_durable_packet_of_record() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/escalation/packet-of-record":
            return httpx.Response(200, json=_PACKET)
        return httpx.Response(404, json={"detail": "unexpected path"})

    client = _client_with_transport(handler)
    try:
        packet = await client.get_escalation_packet()
    finally:
        await client.shutdown()
    assert packet == _PACKET
    assert requested_paths == ["/api/escalation/packet-of-record"]


async def test_packet_fetch_forwards_other_404s_without_generating() -> None:
    """A 404 that is not the no-packet-yet case (FSM not in ESCALATE, or a
    stale packet for a different event) must be forwarded unchanged, without
    triggering generation."""
    state = {"generate_calls": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/escalation/generate":
            state["generate_calls"] += 1
            return httpx.Response(200, json={"status": "generated"})
        return httpx.Response(
            404,
            json={
                "detail": (
                    "No active escalation packet: FSM is not in ESCALATE state"
                )
            },
        )

    client = _client_with_transport(handler)
    try:
        with pytest.raises(httpx.HTTPStatusError) as exc_info:
            await client.get_escalation_packet()
    finally:
        await client.shutdown()
    assert exc_info.value.response.status_code == 404
    assert state["generate_calls"] == 0


async def test_pending_review_requires_durable_packet_of_record() -> None:
    requested_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested_paths.append(request.url.path)
        if request.url.path == "/api/fsm":
            return httpx.Response(
                200,
                json={
                    "fsm_state": "ESCALATE",
                    "event_context": {"event_id": "ev-1"},
                },
            )
        if request.url.path == "/api/escalation/packet-of-record":
            return httpx.Response(404, json={"detail": "packet not yet persisted"})
        return httpx.Response(404)

    client = _client_with_transport(handler)
    try:
        pending = await client.get_pending_reviews()
    finally:
        await client.shutdown()

    assert pending == []
    assert requested_paths == ["/api/fsm", "/api/escalation/packet-of-record"]


async def test_pending_review_includes_packet_identity_when_ready() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/fsm":
            return httpx.Response(
                200,
                json={
                    "fsm_state": "ESCALATE",
                    "event_context": {"event_id": "ev-1"},
                },
            )
        if request.url.path == "/api/escalation/packet-of-record":
            return httpx.Response(200, json=_PACKET)
        return httpx.Response(404)

    client = _client_with_transport(handler)
    try:
        pending = await client.get_pending_reviews()
    finally:
        await client.shutdown()

    assert pending == [
        {
            "event_id": "ev-1",
            "fsm_state": "ESCALATE",
            "packet_row_id": 3,
            "packet_hash": "a" * 64,
        }
    ]
