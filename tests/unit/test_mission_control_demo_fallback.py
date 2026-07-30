"""Regression tests for the Mission Control BFF demo-vs-incident fallback.

The safety contract (finding B1): the BFF serves the static Tohoku demo snapshot
only in genuine demo mode - when no ``MISSION_CONTROL_HAZARD_API_KEY`` is set,
which raises ``UpstreamKeyNotConfiguredError`` before any request is made. A
configured but unreachable core is a live-mode incident: the BFF must surface
the failure (5xx) rather than fabricate demo evidence a duty scientist could
act on.

The demo signal used to be a bare ``RuntimeError``, which was not specific
enough to carry that contract. httpx raises ``RuntimeError`` for a client that
has been closed, and ``HazardClient._get_client`` raises one when startup did
not run, so either transport fault reached the same handler and answered with
the demo snapshot at HTTP 200 in a deployment that did have a key configured.
The last test here pins the narrower type.
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

_repo_root = Path(__file__).resolve().parents[2]
_mc_root = _repo_root / "mission-control"
if str(_mc_root) not in sys.path:
    sys.path.insert(0, str(_mc_root))

from backend.errors import raise_upstream_error  # noqa: E402

_MISSION_CONTROL_ENV_KEYS = (
    "MISSION_CONTROL_API_KEY",
    "MISSION_CONTROL_HAZARD_API_KEY",
    "MISSION_CONTROL_HAZARD_API_URL",
)


@pytest.fixture(autouse=True)
def _clean_mission_control_env() -> Any:
    saved = {k: os.environ.get(k) for k in _MISSION_CONTROL_ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _load_mission_control_app(*, hazard_api_key: str | None) -> Any:
    os.environ["MISSION_CONTROL_API_KEY"] = "mc-key"
    os.environ["MISSION_CONTROL_HAZARD_API_URL"] = "http://127.0.0.1:9999"
    if hazard_api_key is None:
        os.environ.pop("MISSION_CONTROL_HAZARD_API_KEY", None)
    else:
        os.environ["MISSION_CONTROL_HAZARD_API_KEY"] = hazard_api_key

    for module_name in tuple(sys.modules):
        if module_name == "backend" or module_name.startswith("backend."):
            sys.modules.pop(module_name, None)

    import backend.main as mc_main

    return mc_main


# --- raise_upstream_error policy (pure, no app) ------------------------------


def _request() -> httpx.Request:
    return httpx.Request("GET", "http://core/api/fsm")


def test_request_error_becomes_503() -> None:
    exc = httpx.ConnectError("connection refused", request=_request())
    with pytest.raises(HTTPException) as info:
        raise_upstream_error(exc)
    assert info.value.status_code == 503


def test_upstream_4xx_is_forwarded_with_detail() -> None:
    req = _request()
    resp = httpx.Response(404, json={"detail": "no such event"}, request=req)
    exc = httpx.HTTPStatusError("not found", request=req, response=resp)
    with pytest.raises(HTTPException) as info:
        raise_upstream_error(exc)
    assert info.value.status_code == 404
    assert info.value.detail == "no such event"


@pytest.mark.parametrize("status", [401, 403])
def test_upstream_auth_failure_becomes_502_not_a_client_401(status: int) -> None:
    """An upstream credential rejection must not read as the operator's.

    A 401 or 403 from the core API means the BFF's own
    MISSION_CONTROL_HAZARD_API_KEY was rejected, which is a server
    misconfiguration. Forwarding it verbatim made the console treat its own
    stored access key as rejected and re-lock, so a duty scientist was told to
    re-enter a key that was already correct and could not get back in.
    Reporting 502 keeps a probe-confirmed 401 meaning exactly one thing: the
    key the operator supplied is bad.
    """
    req = _request()
    resp = httpx.Response(status, json={"detail": "Unauthorized"}, request=req)
    exc = httpx.HTTPStatusError("auth", request=req, response=resp)
    with pytest.raises(HTTPException) as info:
        raise_upstream_error(exc)
    assert info.value.status_code == 502
    assert info.value.detail == "Core hazard API rejected the service credentials"


def test_upstream_5xx_becomes_502() -> None:
    req = _request()
    resp = httpx.Response(500, text="boom", request=req)
    exc = httpx.HTTPStatusError("server error", request=req, response=resp)
    with pytest.raises(HTTPException) as info:
        raise_upstream_error(exc)
    assert info.value.status_code == 502


# --- router behavior end to end ----------------------------------------------


def test_demo_mode_serves_demo_snapshot_without_hazard_key() -> None:
    """No hazard key configured is the documented demo toggle: the state and
    escalation-packet routes serve the static Tohoku demo (200)."""
    mc_main = _load_mission_control_app(hazard_api_key=None)
    headers = {"X-Mission-Control-Api-Key": "mc-key"}

    with TestClient(mc_main.app) as client:
        state = client.get("/api/mc/state", headers=headers)
        packet = client.get("/api/mc/review/escalation", headers=headers)

    assert state.status_code == 200
    assert state.json()["fsm_state"] == "ESCALATE"
    assert packet.status_code == 200
    assert packet.json()["event_id"] == "tohoku-2011-03-11T05:46:24Z"


def test_live_mode_unreachable_core_fails_loud_not_demo() -> None:
    """A hazard key is configured but the core is unreachable: the escalation
    packet route must fail loud (503), never fabricate a demo packet."""
    mc_main = _load_mission_control_app(hazard_api_key="hazard-key")
    headers = {"X-Mission-Control-Api-Key": "mc-key"}

    with TestClient(mc_main.app) as client:
        packet = client.get("/api/mc/review/escalation", headers=headers)

    assert packet.status_code == 503


def test_live_mode_runtime_error_is_not_treated_as_demo_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A transport RuntimeError in a key-configured deployment must not
    produce demo evidence.

    This is the shape of the real fault: httpx raises
    ``RuntimeError("Cannot send a request, as the client has been closed.")``
    once its client is closed. Catching the base class made that answer with
    the Tohoku demo packet at 200, with no demo marker on the response, which
    is precisely the substitution the module contract forbids.
    """
    mc_main = _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client

    async def closed_client(*_args: Any, **_kwargs: Any) -> Any:
        raise RuntimeError("Cannot send a request, as the client has been closed.")

    monkeypatch.setattr(hazard_client, "get_escalation_packet", closed_client)
    headers = {"X-Mission-Control-Api-Key": "mc-key"}

    with TestClient(mc_main.app) as client, pytest.raises(RuntimeError):
        client.get("/api/mc/review/escalation", headers=headers)

    # The point is what must NOT happen: the demo packet must never be the
    # answer to a transport fault. An unhandled error surfaces the fault; a
    # 200 carrying DEMO_ESCALATION_PACKET would hide it.


@pytest.mark.asyncio
async def test_live_websocket_poll_retains_state_and_reports_upstream_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client
    from backend.services.ws_manager import WSManager

    manager = WSManager()
    prior = {"fsm": {"fsm_state": "MONITOR"}}
    manager._last_snapshot = prior
    messages: list[dict[str, Any]] = []

    async def fail_snapshot() -> dict[str, Any]:
        raise httpx.ConnectError("connection refused", request=_request())

    async def capture(message: dict[str, Any]) -> None:
        messages.append(message)

    async def stop_after_iteration(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(hazard_client, "get_snapshot", fail_snapshot)
    monkeypatch.setattr(manager, "broadcast", capture)
    monkeypatch.setattr(asyncio, "sleep", stop_after_iteration)

    with pytest.raises(asyncio.CancelledError):
        await manager.poll_loop()

    assert manager._last_snapshot is prior
    assert manager._upstream_error is True
    assert messages == [
        {"type": "upstream_error", "data": {"snapshot_retained": True}}
    ]


@pytest.mark.asyncio
async def test_websocket_connect_receives_retained_snapshot_and_outage_state() -> None:
    from backend.services.ws_manager import WSManager

    class _Socket:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def send_text(self, payload: str) -> None:
            import json

            self.messages.append(json.loads(payload))

    manager = WSManager()
    manager._last_snapshot = {"fsm": {"fsm_state": "MONITOR"}}
    manager._upstream_error = True
    socket = _Socket()

    await manager.connect(socket)  # type: ignore[arg-type]

    assert [message["type"] for message in socket.messages] == [
        "snapshot",
        "upstream_error",
    ]


@pytest.mark.asyncio
async def test_live_websocket_recovery_is_reported_when_snapshot_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client
    from backend.services.ws_manager import WSManager

    manager = WSManager()
    prior = {"fsm": {"fsm_state": "MONITOR"}}
    manager._last_snapshot = prior
    manager._upstream_error = True
    messages: list[dict[str, Any]] = []

    async def same_snapshot() -> dict[str, Any]:
        return prior

    async def capture(message: dict[str, Any]) -> None:
        messages.append(message)

    async def stop_after_iteration(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(hazard_client, "get_snapshot", same_snapshot)
    monkeypatch.setattr(manager, "broadcast", capture)
    monkeypatch.setattr(asyncio, "sleep", stop_after_iteration)

    with pytest.raises(asyncio.CancelledError):
        await manager.poll_loop()

    assert manager._upstream_error is False
    assert [message["type"] for message in messages] == ["upstream_recovered", "heartbeat"]


@pytest.mark.asyncio
async def test_quiet_ocean_still_heartbeats_so_staleness_is_visible(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unchanged snapshot must still produce a liveness signal.

    Snapshots broadcast only on change, so a console watching a quiet ocean
    receives nothing for hours. Without the heartbeat that is indistinguishable
    from a BFF that lost the core API, and the dashboard cannot say which one
    the operator is looking at.
    """
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client
    from backend.services.ws_manager import WSManager

    manager = WSManager()
    prior = {"fsm": {"fsm_state": "MONITOR"}}
    manager._last_snapshot = prior
    messages: list[dict[str, Any]] = []

    async def same_snapshot() -> dict[str, Any]:
        return prior

    async def capture(message: dict[str, Any]) -> None:
        messages.append(message)

    async def stop_after_iteration(_seconds: float) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(hazard_client, "get_snapshot", same_snapshot)
    monkeypatch.setattr(manager, "broadcast", capture)
    monkeypatch.setattr(asyncio, "sleep", stop_after_iteration)

    with pytest.raises(asyncio.CancelledError):
        await manager.poll_loop()

    assert [message["type"] for message in messages] == ["heartbeat"]
    assert messages[0]["data"]["polled_at_utc"] == manager._last_poll_utc
    assert manager._last_poll_utc is not None


@pytest.mark.asyncio
async def test_heartbeat_is_throttled_below_the_poll_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The poll interval can be set well under a second; heartbeats are not."""
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client
    from backend.services.ws_manager import HEARTBEAT_INTERVAL_SECONDS, WSManager

    manager = WSManager()
    prior = {"fsm": {"fsm_state": "MONITOR"}}
    manager._last_snapshot = prior
    messages: list[dict[str, Any]] = []
    polls = 0

    async def same_snapshot() -> dict[str, Any]:
        return prior

    async def capture(message: dict[str, Any]) -> None:
        messages.append(message)

    async def stop_after_three(_seconds: float) -> None:
        nonlocal polls
        polls += 1
        if polls >= 3:
            raise asyncio.CancelledError

    monkeypatch.setattr(hazard_client, "get_snapshot", same_snapshot)
    monkeypatch.setattr(manager, "broadcast", capture)
    monkeypatch.setattr(asyncio, "sleep", stop_after_three)

    with pytest.raises(asyncio.CancelledError):
        await manager.poll_loop()

    assert polls == 3
    assert len(messages) == 1

    # Past the throttle window the next poll beats again.
    manager._last_heartbeat_at = (
        asyncio.get_running_loop().time() - HEARTBEAT_INTERVAL_SECONDS - 1
    )
    await manager._maybe_heartbeat()
    assert len(messages) == 2


@pytest.mark.asyncio
async def test_connect_withholds_the_heartbeat_while_the_core_api_is_down() -> None:
    """A stale poll time must not reset a new client's staleness clock.

    The client measures staleness from when a heartbeat arrives, so replaying
    the last successful poll time to a socket that connects during an outage
    would paint an unreachable core API as freshly polled.
    """
    from backend.services.ws_manager import WSManager

    class _Socket:
        def __init__(self) -> None:
            self.messages: list[dict[str, Any]] = []

        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def send_text(self, payload: str) -> None:
            import json

            self.messages.append(json.loads(payload))

    manager = WSManager()
    manager._last_snapshot = {"fsm": {"fsm_state": "MONITOR"}}
    manager._last_poll_utc = "2026-01-01T00:00:00+00:00"
    manager._upstream_error = True
    down = _Socket()

    await manager.connect(down)  # type: ignore[arg-type]

    assert [message["type"] for message in down.messages] == ["snapshot", "upstream_error"]

    manager._upstream_error = False
    healthy = _Socket()
    await manager.connect(healthy)  # type: ignore[arg-type]

    assert [message["type"] for message in healthy.messages] == ["snapshot", "heartbeat"]


@pytest.mark.asyncio
async def test_connect_does_not_leak_a_slot_when_the_initial_send_fails() -> None:
    """A client that vanishes during connect must not hold a connection slot.

    connect() sends the current snapshot right after accept(). If the client
    is already gone, send_text raises. The WebSocket route calls connect()
    outside its try/finally, so a socket registered before that send would
    never be discarded. Those phantom slots count against
    MAX_WS_CONNECTIONS and eventually reject real operators with 1013, and
    broadcast() only prunes dead sockets when a snapshot changes, so an idle
    period never clears them.
    """
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.ws_manager import WSManager

    class _VanishingWebSocket:
        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def send_text(self, _message: str) -> None:
            raise RuntimeError("client gone")

    manager = WSManager()
    manager._last_snapshot = {"fsm": {"fsm_state": "MONITOR"}}
    websocket = _VanishingWebSocket()

    with pytest.raises(RuntimeError):
        await manager.connect(websocket)

    assert websocket not in manager._connections
    assert len(manager._connections) == 0


@pytest.mark.asyncio
async def test_connect_registers_a_healthy_client_and_sends_the_snapshot() -> None:
    """The happy path still registers the socket and delivers state."""
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.ws_manager import WSManager

    class _RecordingWebSocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def accept(self, subprotocol: str | None = None) -> None:
            return None

        async def send_text(self, message: str) -> None:
            self.sent.append(message)

    manager = WSManager()
    manager._last_snapshot = {"fsm": {"fsm_state": "MONITOR"}}
    websocket = _RecordingWebSocket()

    await manager.connect(websocket)

    assert websocket in manager._connections
    assert len(websocket.sent) == 1

    await manager.broadcast({"type": "snapshot", "data": {"fsm": "IDLE"}})

    assert len(websocket.sent) == 2


@pytest.mark.asyncio
async def test_snapshot_queries_review_decisions_separately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Recorded reviews cannot be read out of the general audit window.

    The live worker writes an anomaly_scored entry per scored window, so the 20
    most recent entries of every type can span a fraction of a second. Measured
    against a running stack, those 20 entries covered 165 ms. A review recorded
    a second earlier is already gone, and the console's review gate, which
    reads a decision back from the snapshot, would show a decided packet as
    undecided. The BFF therefore asks for review decisions by event type.
    """
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client

    review_entry = {
        "entry_id": "22222222-2222-4222-8222-222222222222",
        "timestamp_utc": "2026-03-05T00:00:00+00:00",
        "event_id": None,
        "event_type": "assessment_review_decision",
        "producer": "duty-scientist-1",
        "data": {"decision": "APPROVE", "escalation_packet_hash": "a" * 64},
    }
    anomaly_entry = {
        "entry_id": "11111111-1111-4111-8111-111111111111",
        "timestamp_utc": "2026-03-05T00:00:01+00:00",
        "event_id": None,
        "event_type": "anomaly_scored",
        "producer": "anomaly_agent",
        "data": {},
    }
    fsm_payload = {
        "fsm_state": "IDLE",
        "has_active_event": False,
        "event_context": None,
        "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
        "transition_history": [],
    }
    requests: list[tuple[str, dict[str, Any] | None]] = []

    class _Response:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._payload

    class _Client:
        async def get(
            self, path: str, params: dict[str, Any] | None = None, headers: Any = None
        ) -> _Response:
            requests.append((path, params))
            if path == "/api/fsm":
                return _Response(fsm_payload)
            if path == "/api/agents":
                return _Response([])
            if params and params.get("event_type") == "assessment_review_decision":
                return _Response([review_entry])
            return _Response([anomaly_entry])

    async def fake_client() -> _Client:
        return _Client()

    monkeypatch.setattr(hazard_client, "_get_client", fake_client)
    snapshot = await hazard_client.get_snapshot()

    assert ("/api/audit", {"limit": 20, "event_type": "assessment_review_decision"}) in requests
    assert [e["event_type"] for e in snapshot["recent_reviews"]] == [
        "assessment_review_decision"
    ]
    # The general window is untouched: the activity strip still shows everything.
    assert [e["event_type"] for e in snapshot["recent_audit"]] == ["anomaly_scored"]


@pytest.mark.asyncio
async def test_snapshot_keeps_serving_when_the_review_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed review query degrades to an empty feed, not a lost snapshot."""
    _load_mission_control_app(hazard_api_key="hazard-key")
    from backend.services.hazard_client import hazard_client

    fsm_payload = {
        "fsm_state": "IDLE",
        "has_active_event": False,
        "event_context": None,
        "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
        "transition_history": [],
    }

    class _Response:
        def __init__(self, payload: Any) -> None:
            self._payload = payload

        def raise_for_status(self) -> None:
            return None

        def json(self) -> Any:
            return self._payload

    class _Client:
        async def get(
            self, path: str, params: dict[str, Any] | None = None, headers: Any = None
        ) -> _Response:
            if path == "/api/fsm":
                return _Response(fsm_payload)
            if params and params.get("event_type"):
                raise httpx.ConnectError("connection refused", request=_request())
            return _Response([])

    async def fake_client() -> _Client:
        return _Client()

    monkeypatch.setattr(hazard_client, "_get_client", fake_client)
    snapshot = await hazard_client.get_snapshot()

    assert snapshot["recent_reviews"] == []
    assert snapshot["fsm"]["fsm_state"] == "IDLE"


@pytest.mark.asyncio
async def test_one_wedged_client_cannot_stall_the_broadcast(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A client that never drains must not freeze every other console.

    gather waits for all sends and poll_loop awaits the broadcast, so an
    unbounded send on a stalled socket stops snapshots, heartbeats and upstream
    polling for everyone until the kernel gives up on that one connection.
    """
    from backend.services import ws_manager as ws_manager_module
    from backend.services.ws_manager import WSManager

    monkeypatch.setattr(ws_manager_module, "SEND_TIMEOUT_SECONDS", 0.05)

    class _WedgedSocket:
        def __init__(self) -> None:
            self.closed = False

        async def send_text(self, _message: str) -> None:
            await asyncio.sleep(3600)

        async def close(self) -> None:
            self.closed = True

    class _HealthySocket:
        def __init__(self) -> None:
            self.sent: list[str] = []

        async def send_text(self, message: str) -> None:
            self.sent.append(message)

    manager = WSManager()
    wedged = _WedgedSocket()
    healthy = _HealthySocket()
    manager._connections = {wedged, healthy}  # type: ignore[assignment]

    loop = asyncio.get_running_loop()
    started = loop.time()
    await manager.broadcast({"type": "heartbeat", "data": {"polled_at_utc": None}})
    elapsed = loop.time() - started

    assert elapsed < 1.0, "broadcast waited on the wedged client"
    assert len(healthy.sent) == 1
    assert wedged not in manager._connections
    assert healthy in manager._connections
    assert wedged.closed is True
