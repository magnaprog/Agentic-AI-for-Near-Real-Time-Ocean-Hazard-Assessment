"""FastAPI application for the Mission Control BFF.

Serves the React SPA as static files in production and provides
API routes that aggregate data from the main hazard assessment system.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.config import settings
from backend.routers import audit, events, review, state, ws
from backend.security import (
    require_mission_control_api_key,
    required_hazard_api_key,
    required_mission_control_api_key,
)
from backend.services.hazard_client import hazard_client
from backend.services.ws_manager import ws_manager

STATIC_DIR = Path(__file__).resolve().parent.parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start the background polling task on startup, cancel on shutdown."""
    required_mission_control_api_key()
    # MISSION_CONTROL_HAZARD_API_KEY is optional. When empty, the WS manager falls
    # back to the built-in Tohoku 2011 demo snapshot (no core API needed).
    try:
        required_hazard_api_key()
        await hazard_client.startup()
    except RuntimeError:
        import logging

        logging.getLogger(__name__).warning(
            "MISSION_CONTROL_HAZARD_API_KEY not set; running in demo mode (Tohoku 2011)"
        )
    task = asyncio.create_task(ws_manager.poll_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    await hazard_client.shutdown()


app = FastAPI(
    title="Mission Control",
    version="0.1.0",
    description="Mission control dashboard: Agentic AI for Near-Real-Time Ocean Hazard Assessment",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["X-Mission-Control-Api-Key", "X-Reviewer-Id", "Content-Type"],
)

@app.get("/health")
def health_check() -> dict[str, str]:
    """Liveness probe; does not depend on upstream api-server."""
    return {"status": "healthy", "service": "mission-control"}


app.include_router(
    state.router,
    prefix="/api/mc",
    dependencies=[Depends(require_mission_control_api_key)],
)
app.include_router(
    events.router,
    prefix="/api/mc",
    dependencies=[Depends(require_mission_control_api_key)],
)
app.include_router(
    review.router,
    prefix="/api/mc",
    dependencies=[Depends(require_mission_control_api_key)],
)
app.include_router(
    audit.router,
    prefix="/api/mc",
    dependencies=[Depends(require_mission_control_api_key)],
)
# WebSocket router has no Depends(require_mission_control_api_key) because FastAPI
# dependency injection does not work with WebSocket endpoints in the same
# way. Auth is handled inside the WS handler via query-param API key.
app.include_router(ws.router, prefix="/api/mc")

if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
