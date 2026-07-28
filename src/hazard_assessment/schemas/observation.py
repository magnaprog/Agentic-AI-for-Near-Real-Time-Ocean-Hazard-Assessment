"""Canonical observation schemas for pipeline-boundary validation.

These Pydantic v2 models validate ingest connector records before they
enter the processing pipeline, enforcing field types, value ranges, and
payload hash presence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hazard_assessment.schemas.envelope import AwareDatetime

DART_STATION_ID_PATTERN = r"^[0-9]{5}$"


def is_dart_station_id(value: object) -> bool:
    """Return True for canonical five-digit ASCII DART station IDs."""
    return isinstance(value, str) and len(value) == 5 and value.isascii() and value.isdigit()


class BaseObservation(BaseModel):
    """Common fields shared by all observation types."""

    source_id: str = Field(min_length=1)
    source_timestamp: AwareDatetime
    ingest_timestamp: AwareDatetime
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    model_config = {"extra": "forbid"}


class DartObservation(BaseObservation):
    """Validated DART buoy observation."""

    station_id: str = Field(pattern=DART_STATION_ID_PATTERN)
    measurement_type: Literal[1, 2, 3]
    # DART HEIGHT = water column height above seafloor BPR (always positive).
    # Deployments typically range ~2,600-6,000 m (NDBC DART spec max 6,000 m).
    # Upper bound 12,000 m is a generous gross-range guard (exceeds
    # Challenger Deep at ~10,994 m) to avoid false rejections from
    # unusual BPR readings; QARTOD calibration will tighten.
    height_m: float = Field(ge=0.0, le=12_000.0)
    event_mode: bool


class CoopsObservation(BaseObservation):
    """Validated CO-OPS water-level observation."""

    station_id: str = Field(min_length=1)
    station_name: str | None = None
    product: str = Field(min_length=1)
    # CO-OPS water levels relative to STND datum.  Typical range 0-10 m;
    # extremes (Sandy peak at The Battery: 5.3 m) stay well under 30 m.
    water_level_m: float | None = Field(default=None, ge=-15.0, le=30.0)
    flags: str
    quality: str


class SeismicObservation(BaseObservation):
    """Validated seismic event observation.

    NOTE: Unlike DART/CO-OPS observations, seismic events identify by
    ``event_id`` rather than ``station_id``. The SQL ``raw_observations``
    table uses a generic ``station_id TEXT NOT NULL`` column - the E2
    persistence layer must map ``event_id`` -> ``station_id`` when
    inserting seismic records.
    """

    event_id: str = Field(min_length=1)
    magnitude: float | None = Field(default=None, ge=-2.0, le=10.0)
    place: str
    event_type: str
    tsunami_flag: int | None = Field(default=None, ge=0, le=1)
    longitude: float | None = Field(default=None, ge=-180.0, le=180.0)
    latitude: float | None = Field(default=None, ge=-90.0, le=90.0)
    depth_km: float | None = Field(default=None, ge=-10.0, le=1000.0)
    updated_timestamp: AwareDatetime | None = None
    is_revision: bool
