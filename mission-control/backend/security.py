"""Authentication and runtime setting validation for Mission Control BFF."""

from __future__ import annotations

import base64
import binascii
from hmac import compare_digest

from fastapi import Header, HTTPException, WebSocket

from backend.config import settings
from backend.errors import UpstreamKeyNotConfiguredError

MISSION_CONTROL_API_KEY_HEADER_NAME = "X-Mission-Control-Api-Key"
# Subprotocol carrying the access key on a browser WebSocket handshake. The
# browser WebSocket API cannot set request headers, and a query parameter ends
# up in the uvicorn access log on every handshake and every reconnect. A
# subprotocol travels in Sec-WebSocket-Protocol, which is not part of the
# logged request line. The key is base64url-encoded because a subprotocol value
# must be an HTTP token and an access key need not be.
MISSION_CONTROL_KEY_SUBPROTOCOL_PREFIX = "mc-key."
_URLSAFE_TO_STANDARD_B64 = str.maketrans("-_", "+/")
REVIEWER_ID_HEADER_NAME = "X-Reviewer-Id"


def _api_key_matches(provided: str, expected: str) -> bool:
    """Compare API keys in constant time without raising on any input.

    ``hmac.compare_digest`` rejects ``str`` arguments holding non-ASCII
    characters with a TypeError rather than returning False. Both key sources
    can carry them: header values arrive decoded as latin-1, and the WebSocket
    ``api_key`` query parameter is percent-decoded as UTF-8. Either one made
    this check raise, so an invalid key surfaced as a 500 on the HTTP routes
    and as an abnormal WebSocket close instead of the 1008 the client needs to
    re-lock. Comparing the UTF-8 encodings keeps the comparison constant-time
    and makes every invalid key an ordinary mismatch.
    """
    return compare_digest(provided.encode("utf-8"), expected.encode("utf-8"))


def _required_value(
    raw_value: str,
    setting_name: str,
    exc_type: type[RuntimeError] = RuntimeError,
) -> str:
    value = raw_value.strip()
    if value == "":
        raise exc_type(f"{setting_name} is required for Mission Control")
    return value


def required_mission_control_api_key() -> str:
    """Return the configured Mission Control API key or fail fast."""
    return _required_value(settings.api_key, "MISSION_CONTROL_API_KEY")


def required_hazard_api_key() -> str:
    """Return the configured upstream hazard API key or fail fast.

    Raises ``UpstreamKeyNotConfiguredError`` rather than a bare ``RuntimeError`` so
    a router can tell "no upstream is configured, serve the demo snapshot"
    apart from a transport fault that also surfaces as ``RuntimeError``.
    """
    return _required_value(
        settings.hazard_api_key,
        "MISSION_CONTROL_HAZARD_API_KEY",
        UpstreamKeyNotConfiguredError,
    )


def require_reviewer_id_header(
    x_reviewer_id: str | None = Header(default=None, alias=REVIEWER_ID_HEADER_NAME),
) -> str:
    """Extract reviewer identity from the X-Reviewer-Id request header.

    The identity must be supplied per-request so each decision in the audit
    trail is attributed to the actual operator, not a static server config value.
    """
    if x_reviewer_id is None or not x_reviewer_id.strip():
        raise HTTPException(
            status_code=400,
            detail="X-Reviewer-Id header is required for decision provenance",
        )
    return x_reviewer_id.strip()


def require_mission_control_api_key(
    x_mission_control_api_key: str | None = Header(
        default=None,
        alias=MISSION_CONTROL_API_KEY_HEADER_NAME,
    ),
) -> None:
    """Authenticate HTTP requests using the Mission Control API key."""
    expected = required_mission_control_api_key()
    if x_mission_control_api_key is None or not _api_key_matches(
        x_mission_control_api_key,
        expected,
    ):
        raise HTTPException(status_code=401, detail="Unauthorized")


def offered_key_subprotocol(websocket: WebSocket) -> str | None:
    """Return the offered mc-key subprotocol value, if the client sent one.

    A server that accepts a handshake must echo one of the client's offered
    subprotocols, so the caller needs the exact string back.
    """
    offered = websocket.headers.get("sec-websocket-protocol")
    if not offered:
        return None
    for candidate in (part.strip() for part in offered.split(",")):
        if candidate.startswith(MISSION_CONTROL_KEY_SUBPROTOCOL_PREFIX):
            return candidate
    return None


def _decode_key_subprotocol(value: str) -> str | None:
    """Decode an ``mc-key.`` payload, or None if it is not a valid key.

    ``validate=True`` matters here: the default discards characters outside
    the base64 alphabet, so a payload of pure garbage decodes to an empty
    string rather than failing. That empty string would then be compared
    against the configured key. An empty configured key already fails
    startup, so this was not exploitable, but an auth path should reject
    malformed input rather than turn it into a comparison.
    """
    encoded = value[len(MISSION_CONTROL_KEY_SUBPROTOCOL_PREFIX):]
    if not encoded:
        return None
    padded = encoded + "=" * (-len(encoded) % 4)
    # base64.urlsafe_b64decode has no validate parameter, so translate the
    # URL-safe alphabet and validate against the standard one.
    standard = padded.translate(_URLSAFE_TO_STANDARD_B64)
    try:
        decoded = base64.b64decode(standard, validate=True).decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return decoded or None


def websocket_has_valid_api_key(websocket: WebSocket) -> bool:
    """Authenticate a WebSocket handshake.

    Three transports, in decreasing preference: the ``mc-key.`` subprotocol
    (what the browser client uses, and the only one that keeps the key out of
    the access log), the API key header (for non-browser clients), and the
    ``api_key`` query parameter, which is retained for compatibility but logs
    the key on every handshake.
    """
    expected = required_mission_control_api_key()
    offered = offered_key_subprotocol(websocket)
    if offered is not None:
        decoded = _decode_key_subprotocol(offered)
        if decoded is not None and _api_key_matches(decoded, expected):
            return True
    header_api_key = websocket.headers.get(MISSION_CONTROL_API_KEY_HEADER_NAME)
    if header_api_key is not None and _api_key_matches(header_api_key, expected):
        return True
    query_api_key = websocket.query_params.get("api_key")
    if query_api_key is not None and _api_key_matches(query_api_key, expected):
        return True
    return False
