"""WebSocket endpoint for real-time dashboard updates."""

from __future__ import annotations

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.security import offered_key_subprotocol, websocket_has_valid_api_key
from backend.services.ws_manager import ws_manager

router = APIRouter(tags=["websocket"])


@router.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Accept a WebSocket connection and stream system snapshots."""
    if not websocket_has_valid_api_key(websocket):
        await websocket.close(code=1008, reason="Unauthorized")
        return
    # connect() closes the socket itself when at capacity. Without this
    # check the route would fall through to receive_text() on a closed
    # socket and then disconnect() a connection that was never registered.
    if not await ws_manager.connect(websocket, offered_key_subprotocol(websocket)):
        return
    try:
        while True:
            # Keep connection alive; client can send pings or commands
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(websocket)
