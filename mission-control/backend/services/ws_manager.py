"""WebSocket connection manager with background polling."""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime
from typing import Any

from fastapi import WebSocket

from backend.config import settings

logger = logging.getLogger(__name__)


MAX_WS_CONNECTIONS = 50
"""Global cap on concurrent WebSocket connections to prevent resource exhaustion."""

SEND_TIMEOUT_SECONDS = 5.0
"""Per-client bound on a single WebSocket send.

Without it one wedged client stalls every other operator's console. A browser
whose TCP receive window has filled (a closed laptop lid, a blackholed VPN)
leaves ``send_text`` awaiting the transport drain for as long as the kernel
keeps retrying, ``asyncio.gather`` waits for all sends, and ``poll_loop``
awaits the broadcast: no snapshots, no heartbeats and no upstream polling for
anyone until that one socket resolves. Five seconds is far beyond any healthy
send at this payload size and far short of a TCP timeout.
"""

HEARTBEAT_INTERVAL_SECONDS = 5.0
"""Minimum gap between heartbeat broadcasts.

Snapshots are broadcast only when they change, so a console watching a quiet
ocean receives nothing for hours and cannot tell that from a BFF that lost the
core API. The heartbeat is the periodic signal that separates the two cases. It
is throttled rather than sent on every poll tick because the poll interval can
be set well below a second.
"""


class WSManager:
    """Manages WebSocket connections and broadcasts state snapshots."""

    def __init__(self, max_connections: int = MAX_WS_CONNECTIONS) -> None:
        self._connections: set[WebSocket] = set()
        self._last_snapshot: dict[str, Any] | None = None
        self._upstream_error = False
        self._max_connections = max_connections
        self._last_poll_utc: str | None = None
        self._last_heartbeat_at: float | None = None

    async def connect(self, websocket: WebSocket, subprotocol: str | None = None) -> bool:
        if len(self._connections) >= self._max_connections:
            await websocket.close(code=1013, reason="Too many connections")
            logger.warning(
                "Rejected WebSocket connection: at capacity (%d/%d)",
                len(self._connections),
                self._max_connections,
            )
            return False
        # A server accepting a handshake must echo one of the client's offered
        # subprotocols, or the browser closes the connection immediately.
        await websocket.accept(subprotocol=subprotocol)
        # Send current state immediately on connect (same envelope as poll).
        # These sends happen before the socket joins _connections: a client that
        # vanishes between accept and the first send makes send_text raise, and
        # the WebSocket route calls connect() outside its try/finally, so a
        # socket registered first would never be discarded. Those phantom slots
        # accumulate against MAX_WS_CONNECTIONS and eventually lock real
        # operators out with 1013. broadcast() only prunes dead sockets when a
        # snapshot actually changes, so a quiet period never cleans them up.
        if self._last_snapshot:
            await websocket.send_text(
                json.dumps({"type": "snapshot", "data": self._last_snapshot})
            )
        # Only when the last poll actually succeeded. Replaying a stale
        # timestamp to a client that just connected would reset its staleness
        # clock and paint an unreachable core API as freshly polled.
        if self._last_poll_utc is not None and not self._upstream_error:
            await websocket.send_text(json.dumps(self._heartbeat_message()))
        if self._upstream_error:
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "upstream_error",
                        "data": {"snapshot_retained": self._last_snapshot is not None},
                    }
                )
            )
        self._connections.add(websocket)
        return True

    def _heartbeat_message(self) -> dict[str, Any]:
        """Envelope reporting the last successful upstream poll.

        ``polled_at_utc`` is the BFF's own clock and is carried for display and
        debugging only. Staleness is measured by the client from the arrival of
        this message, which needs no agreement between the two clocks. Demo mode
        is not repeated here: the snapshot already carries ``demo_mode``.
        """
        return {"type": "heartbeat", "data": {"polled_at_utc": self._last_poll_utc}}

    async def _maybe_heartbeat(self) -> None:
        now = asyncio.get_running_loop().time()
        if (
            self._last_heartbeat_at is not None
            and now - self._last_heartbeat_at < HEARTBEAT_INTERVAL_SECONDS
        ):
            return
        self._last_heartbeat_at = now
        await self.broadcast(self._heartbeat_message())

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast(self, data: dict[str, Any]) -> None:
        """Send data to all connected clients concurrently, removing dead ones."""
        if not self._connections:
            return
        message = json.dumps(data)

        async def _send(ws: WebSocket) -> WebSocket | None:
            try:
                await asyncio.wait_for(ws.send_text(message), timeout=SEND_TIMEOUT_SECONDS)
                return None
            except Exception:
                return ws

        results = await asyncio.gather(*[_send(ws) for ws in list(self._connections)])
        for ws in results:
            if ws is not None:
                self.disconnect(ws)
                # A timed-out socket is still open as far as the server is
                # concerned; dropping it from the set alone would leak it.
                try:
                    await asyncio.wait_for(ws.close(), timeout=SEND_TIMEOUT_SECONDS)
                except Exception:
                    logger.debug("Closing an unresponsive WebSocket failed", exc_info=True)

    async def poll_loop(self) -> None:
        """Poll core API and broadcast snapshots without fabricating fallback data.

        Static demo data is used only when no upstream core API key is
        configured. Live-mode failures retain the last snapshot and send an
        explicit availability message.
        """
        from backend.services.hazard_client import hazard_client

        demo_mode = not settings.hazard_api_key.strip()
        while True:
            try:
                snapshot = await hazard_client.get_snapshot()
                was_unavailable = self._upstream_error
                self._upstream_error = False
                self._last_poll_utc = datetime.now(UTC).isoformat()
                if snapshot != self._last_snapshot:
                    self._last_snapshot = snapshot
                    await self.broadcast({"type": "snapshot", "data": snapshot})
                if was_unavailable:
                    await self.broadcast({"type": "upstream_recovered", "data": {}})
                await self._maybe_heartbeat()
            except Exception as exc:
                if demo_mode:
                    from backend.services.demo_snapshot import TOHOKU_SNAPSHOT

                    demo = {**TOHOKU_SNAPSHOT, "demo_mode": True}
                    if demo != self._last_snapshot:
                        self._last_snapshot = demo
                        await self.broadcast({"type": "snapshot", "data": demo})
                    self._last_poll_utc = datetime.now(UTC).isoformat()
                    await self._maybe_heartbeat()
                elif not self._upstream_error:
                    self._upstream_error = True
                    logger.warning("Core API poll failed; retaining last snapshot: %s", exc)
                    await self.broadcast(
                        {
                            "type": "upstream_error",
                            "data": {"snapshot_retained": self._last_snapshot is not None},
                        }
                    )
            await asyncio.sleep(max(settings.poll_interval_seconds, 0.5))


ws_manager = WSManager()
