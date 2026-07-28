"""USGS seismic ingest connector."""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import httpx

from hazard_assessment.config.settings import IngestSettings
from hazard_assessment.ingest.base import BaseIngestConnector, safe_float, safe_int
from hazard_assessment.ingest.hashing import canonicalize_json, compute_payload_hash

USGS_EVENT_QUERY_URL: Final[str] = "https://earthquake.usgs.gov/fdsnws/event/1/query"
USGS_UPDATED_AFTER_OVERLAP_SEC: Final[int] = 5
USGS_MAX_PAGES_PER_POLL: Final[int] = 5
SEISMIC_EVENT_STATE_RETENTION_DAYS: Final[int] = 30
SEISMIC_MAX_TRACKED_EVENTS: Final[int] = 10_000

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SeismicEventRecord:
    """Structured seismic event record emitted by the connector."""

    source_id: str
    event_id: str
    source_timestamp: datetime
    ingest_timestamp: datetime
    magnitude: float | None
    place: str
    event_type: str
    tsunami_flag: int | None
    longitude: float | None
    latitude: float | None
    depth_km: float | None
    updated_timestamp: datetime | None
    is_revision: bool
    payload_sha256: str = ""


class SeismicIngestConnector(BaseIngestConnector[SeismicEventRecord]):
    """Poll USGS FDSN event feed and emit deduplicated/revised events."""

    def __init__(
        self,
        *,
        settings: IngestSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
        min_magnitude: float = 5.5,
        lookback_minutes: int = 5,
        limit: int = 20,
        max_pages_per_poll: int = USGS_MAX_PAGES_PER_POLL,
        event_state_retention_days: int = SEISMIC_EVENT_STATE_RETENTION_DAYS,
        max_tracked_events: int = SEISMIC_MAX_TRACKED_EVENTS,
        clock: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        ingest_settings = settings or IngestSettings()
        super().__init__(
            name="seismic",
            poll_interval_sec=ingest_settings.seismic_poll_interval_sec,
            expected_interval_sec=ingest_settings.seismic_poll_interval_sec,
            settings=ingest_settings,
            http_client=http_client,
            clock=clock,
            sleep_fn=sleep_fn,
        )
        if limit < 1 or limit > 20_000:
            raise ValueError("Seismic limit must be between 1 and 20000")
        if lookback_minutes <= 0:
            raise ValueError("Seismic lookback_minutes must be greater than 0")
        self._min_magnitude = min_magnitude
        self._lookback_minutes = lookback_minutes
        self._limit = limit
        self._max_pages_per_poll = max(1, max_pages_per_poll)
        self._event_state_retention_days = max(1, event_state_retention_days)
        self._max_tracked_events = max(1, max_tracked_events)
        self._version_by_event_id: dict[str, str] = {}
        self._last_seen_update_by_event_id: dict[str, datetime] = {}
        self._latest_seen_update_timestamp: datetime | None = None

    def has_new_data(self, records: list[SeismicEventRecord]) -> bool:
        """Treat any successful poll as fresh for event-driven feeds."""
        return True

    async def fetch_records(self) -> list[SeismicEventRecord]:
        now = self._clock()
        params = self._build_query_params(now=now)
        features, page_window_truncated = await self._fetch_features(params=params)

        records: list[SeismicEventRecord] = []
        previous_updated_cursor = self._latest_seen_update_timestamp
        latest_seen_update = previous_updated_cursor
        ingest_timestamp = self._clock()
        for feature in features:
            if not isinstance(feature, dict):
                continue

            event_id = feature.get("id")
            properties = feature.get("properties") or {}
            geometry = feature.get("geometry") or {}
            if not isinstance(event_id, str) or not isinstance(properties, dict):
                continue

            source_timestamp = _from_epoch_millis(properties.get("time"))
            if source_timestamp is None:
                continue

            updated_timestamp = _from_epoch_millis(properties.get("updated"))
            source_id_timestamp = updated_timestamp or source_timestamp
            self._last_seen_update_by_event_id[event_id] = source_id_timestamp
            if latest_seen_update is None or source_id_timestamp > latest_seen_update:
                latest_seen_update = source_id_timestamp

            version_token = _build_event_version(properties=properties, geometry=geometry)
            previous_version = self._version_by_event_id.get(event_id)
            if previous_version == version_token:
                continue

            self._version_by_event_id[event_id] = version_token
            longitude, latitude, depth_km = _parse_coordinates(geometry)

            records.append(
                SeismicEventRecord(
                    source_id=(
                        f"seismic:{event_id}:"
                        f"{source_id_timestamp.strftime('%Y%m%d%H%M%S%f')}"
                    ),
                    event_id=event_id,
                    source_timestamp=source_timestamp,
                    ingest_timestamp=ingest_timestamp,
                    magnitude=safe_float(properties.get("mag")),
                    place=str(properties.get("place", "")),
                    event_type=str(properties.get("type", "")),
                    tsunami_flag=safe_int(properties.get("tsunami")),
                    longitude=longitude,
                    latitude=latitude,
                    depth_km=depth_km,
                    updated_timestamp=updated_timestamp,
                    is_revision=previous_version is not None,
                    payload_sha256=compute_payload_hash(canonicalize_json(feature)),
                )
            )

        if previous_updated_cursor is None:
            # Keep the first updatedafter cursor anchored to the bootstrap
            # lookback floor for one cycle. This avoids missing revisions to
            # older-origin events that were updated shortly before startup but
            # were excluded by the initial starttime filter.
            self._latest_seen_update_timestamp = now - timedelta(minutes=self._lookback_minutes)
        elif page_window_truncated:
            # Every fetched page was full and we hit the configured page cap;
            # do not advance cursor or we can skip unseen rows beyond the cap.
            logger.warning(
                "Seismic poll page window truncated (limit=%s, max_pages=%s); "
                "retaining updatedafter cursor to avoid skips",
                self._limit,
                self._max_pages_per_poll,
            )
            self._latest_seen_update_timestamp = previous_updated_cursor
        else:
            self._latest_seen_update_timestamp = latest_seen_update
        self._prune_event_state(now=now)
        records.sort(key=lambda record: record.source_timestamp)
        return records

    def _build_query_params(self, *, now: datetime) -> dict[str, str]:
        poll_end = now
        if (
            self._latest_seen_update_timestamp is not None
            and poll_end < self._latest_seen_update_timestamp
        ):
            # Guard against clock skew/rollback producing invalid query windows
            # (updatedafter > endtime).
            poll_end = self._latest_seen_update_timestamp

        params = {
            "format": "geojson",
            "minmagnitude": f"{self._min_magnitude:.1f}",
            "orderby": "time",
            "limit": str(self._limit),
            # Snapshot poll horizon to reduce pagination churn while fetching
            # multiple pages from a mutable upstream result set.
            "endtime": _format_usgs_timestamp(poll_end),
        }
        if self._latest_seen_update_timestamp is None:
            params["starttime"] = _format_usgs_timestamp(
                now - timedelta(minutes=self._lookback_minutes)
            )
            return params

        params["updatedafter"] = _format_usgs_timestamp(
            self._latest_seen_update_timestamp
            - timedelta(seconds=USGS_UPDATED_AFTER_OVERLAP_SEC)
        )
        return params

    async def _fetch_features(
        self, *, params: dict[str, str]
    ) -> tuple[list[dict[str, object]], bool]:
        all_features: list[dict[str, object]] = []
        page_window_truncated = False
        for page in range(self._max_pages_per_poll):
            page_params = dict(params)
            if page > 0:
                page_params["offset"] = str((page * self._limit) + 1)

            raw = await self.fetch_bounded(USGS_EVENT_QUERY_URL, params=page_params)

            page_features = _extract_usgs_features(json.loads(raw))
            all_features.extend(page_features)
            if len(page_features) < self._limit:
                break
            page_window_truncated = page == (self._max_pages_per_poll - 1)
        return all_features, page_window_truncated

    def _prune_event_state(self, *, now: datetime) -> None:
        cutoff = now - timedelta(days=self._event_state_retention_days)
        stale_ids = [
            event_id
            for event_id, last_seen in self._last_seen_update_by_event_id.items()
            if last_seen < cutoff
        ]
        for event_id in stale_ids:
            self._last_seen_update_by_event_id.pop(event_id, None)
            self._version_by_event_id.pop(event_id, None)

        overflow = len(self._version_by_event_id) - self._max_tracked_events
        if overflow <= 0:
            return

        oldest_default = datetime.min.replace(tzinfo=UTC)
        oldest_ids = sorted(
            self._version_by_event_id,
            key=lambda event_id: self._last_seen_update_by_event_id.get(event_id, oldest_default),
        )[:overflow]
        for event_id in oldest_ids:
            self._last_seen_update_by_event_id.pop(event_id, None)
            self._version_by_event_id.pop(event_id, None)


def _format_usgs_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _extract_usgs_features(payload: object) -> list[dict[str, object]]:
    if not isinstance(payload, dict):
        raise ValueError("USGS response payload must be a JSON object")
    features = payload.get("features")
    if features is None:
        raise ValueError("USGS payload field 'features' is required")
    if not isinstance(features, list):
        raise ValueError("USGS payload field 'features' must be a list")
    return [feature for feature in features if isinstance(feature, dict)]


def _from_epoch_millis(raw_value: object) -> datetime | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if not isinstance(raw_value, (int, float, str)):
        return None
    try:
        milliseconds = int(raw_value)
    except (TypeError, ValueError):
        return None
    try:
        return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def _parse_coordinates(geometry: object) -> tuple[float | None, float | None, float | None]:
    if not isinstance(geometry, dict):
        return (None, None, None)
    coords = geometry.get("coordinates")
    if not isinstance(coords, list) or len(coords) < 3:
        return (None, None, None)
    return (safe_float(coords[0]), safe_float(coords[1]), safe_float(coords[2]))


def _build_event_version(*, properties: dict[str, object], geometry: object) -> str:
    """Build a stable fingerprint for deduping and revision detection."""
    version_payload = {
        "mag": properties.get("mag"),
        "time": properties.get("time"),
        "updated": properties.get("updated"),
        "tsunami": properties.get("tsunami"),
        "place": properties.get("place"),
        "type": properties.get("type"),
        "geometry": geometry,
    }
    return canonicalize_json(version_payload).decode("utf-8")


