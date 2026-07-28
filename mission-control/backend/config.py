"""Configuration for the Mission Control BFF."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Mission Control BFF settings loaded from environment variables."""

    hazard_api_url: str = "http://localhost:8000"
    hazard_api_key: str = ""
    api_key: str = ""
    poll_interval_seconds: float = 2.0
    cors_origins: list[str] = ["http://localhost:5173"]

    model_config = {"env_prefix": "MISSION_CONTROL_"}


settings = Settings()
