"""Unit tests for Mission Control BFF authentication boundaries."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

_MISSION_CONTROL_ENV_KEYS = (
    "MISSION_CONTROL_API_KEY",
    "MISSION_CONTROL_HAZARD_API_KEY",
    "MISSION_CONTROL_HAZARD_API_URL",
)


@pytest.fixture(autouse=True)
def _clean_mission_control_env() -> Any:
    """Save and restore MISSION_CONTROL_ env vars to prevent cross-test leakage."""
    saved = {k: os.environ.get(k) for k in _MISSION_CONTROL_ENV_KEYS}
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def _load_mission_control_app(
    *,
    mission_control_api_key: str | None,
    hazard_api_key: str | None = "hazard-key",
) -> Any:
    repo_root = Path(__file__).resolve().parents[2]
    mission_control_root = repo_root / "mission-control"
    if str(mission_control_root) not in sys.path:
        sys.path.insert(0, str(mission_control_root))

    if mission_control_api_key is None:
        os.environ.pop("MISSION_CONTROL_API_KEY", None)
    else:
        os.environ["MISSION_CONTROL_API_KEY"] = mission_control_api_key

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


def test_mission_control_startup_fails_without_api_key() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key=None)

    with pytest.raises(RuntimeError, match="MISSION_CONTROL_API_KEY is required"):
        with TestClient(mc_main.app):
            pass


def test_mission_control_starts_demo_mode_without_hazard_api_key() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key", hazard_api_key=None)

    # Without MISSION_CONTROL_HAZARD_API_KEY, the app should start in demo mode (not crash).
    with TestClient(mc_main.app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_mission_control_health_is_public() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_mission_control_http_routes_require_api_key() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        unauthorized = client.get("/api/mc/state")
        wrong_key = client.get(
            "/api/mc/state",
            headers={"X-Mission-Control-Api-Key": "wrong"},
        )
        legacy_header = client.get(
            "/api/mc/state",
            headers={"X-MC-Api-Key": "mc-key"},
        )
        authorized = client.get(
            "/api/mc/state",
            headers={"X-Mission-Control-Api-Key": "mc-key"},
        )

    assert unauthorized.status_code == 401
    assert wrong_key.status_code == 401
    assert legacy_header.status_code == 401
    # A valid key is configured with an unreachable core (hazard key set, dead
    # URL), so the request passes auth and reaches the handler, which then fails
    # upstream with 503. That 503 (not 401) is what proves auth succeeded: a
    # configured-but-unreachable core is a live-mode incident, not demo mode, so
    # the BFF must surface the failure rather than fabricate a demo snapshot.
    assert authorized.status_code == 503


def test_mission_control_websocket_rejects_missing_api_key() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        with pytest.raises(WebSocketDisconnect) as exc:
            with client.websocket_connect("/api/mc/ws/live"):
                pass

    assert exc.value.code == 1008


def test_mission_control_websocket_accepts_valid_query_api_key() -> None:
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        with client.websocket_connect("/api/mc/ws/live?api_key=mc-key"):
            pass


def test_mission_control_rejects_non_ascii_api_key_with_401_not_500() -> None:
    """A non-ASCII key must fail authentication, not crash the handler.

    ``hmac.compare_digest`` raises TypeError on ``str`` arguments holding
    non-ASCII characters instead of returning False, so one high byte in the
    header used to surface as a 500. That also broke the console's re-lock
    path, which only treats a confirmed 401 as "key rejected". The header is
    sent as raw bytes because HTTP clients refuse to encode a non-ASCII
    ``str`` header.
    """
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        response = client.get(
            "/api/mc/state",
            headers={b"X-Mission-Control-Api-Key": b"\xff\xfe"},
        )

    assert response.status_code == 401


def test_mission_control_websocket_rejects_non_ascii_api_key() -> None:
    """A non-ASCII ``api_key`` query parameter must close with 1008.

    The query parameter is percent-decoded as UTF-8, so any non-ASCII
    character reaches the check. It used to raise TypeError before the route
    could send its 1008 close, and the console only re-locks on 1008: any
    other close is read as a transient outage and retried forever, leaving a
    bad key stuck in a silent reconnect loop with no way back to the gate.
    """
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/api/mc/ws/live?api_key=caf%C3%A9"):
                pass

    assert exc_info.value.code == 1008


def test_mission_control_websocket_accepts_key_via_subprotocol() -> None:
    """The browser client authenticates without putting the key in the URL.

    A query parameter is written verbatim into the uvicorn access log on every
    handshake, and the console reconnects with backoff, so the shared key would
    be logged repeatedly. base64url because a subprotocol value must be an HTTP
    token while an access key need not be.
    """
    import base64

    key = "mc-key/with+odd=chars"
    mc_main = _load_mission_control_app(mission_control_api_key=key)
    encoded = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")

    with TestClient(mc_main.app) as client:
        with client.websocket_connect(
            "/api/mc/ws/live", subprotocols=[f"mc-key.{encoded}"]
        ):
            pass


def test_mission_control_websocket_rejects_wrong_key_in_subprotocol() -> None:
    import base64

    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")
    encoded = base64.urlsafe_b64encode(b"not-the-key").decode().rstrip("=")

    with TestClient(mc_main.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/mc/ws/live", subprotocols=[f"mc-key.{encoded}"]
            ):
                pass


def test_mission_control_websocket_rejects_malformed_subprotocol() -> None:
    """A non-base64 payload must fail authentication, not raise."""
    mc_main = _load_mission_control_app(mission_control_api_key="mc-key")

    with TestClient(mc_main.app) as client:
        with pytest.raises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/mc/ws/live", subprotocols=["mc-key.!!!not-base64!!!"]
            ):
                pass


@pytest.mark.parametrize(
    "payload",
    [
        "mc-key.",              # empty
        "mc-key.!!!",           # outside the base64 alphabet entirely
        "mc-key.YWJj ZGU",      # embedded whitespace
        "mc-key.a+b/c=",        # standard alphabet, not URL-safe
        "mc-key.YWJjZ",         # length can never be valid base64
        "mc-key.YQ===",         # over-padded
    ],
)
def test_malformed_key_subprotocol_decodes_to_none(payload: str) -> None:
    """A malformed payload must be rejected, not turned into a comparison.

    `base64.b64decode` without `validate=True` discards characters outside the
    alphabet, so a payload of pure garbage decoded to an empty string, which
    was then compared against the configured key. An empty configured key
    already fails startup so this was not exploitable, but an auth path should
    not depend on that.
    """
    from backend.security import _decode_key_subprotocol

    assert _decode_key_subprotocol(payload) is None


def test_key_subprotocol_round_trips_urlsafe_and_non_ascii_keys() -> None:
    """Keys whose encoding uses - or _ must survive, as must non-ASCII keys."""
    import base64

    from backend.security import _decode_key_subprotocol

    for key in ("kéy-with/odd+chars=", "日本語", "a" * 128, "?~ÿ"):
        encoded = base64.urlsafe_b64encode(key.encode()).decode().rstrip("=")
        assert _decode_key_subprotocol(f"mc-key.{encoded}") == key
