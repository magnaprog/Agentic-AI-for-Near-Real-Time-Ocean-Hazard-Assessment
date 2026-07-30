"""NOAA CO-OPS water-level ingest connector."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Final

import httpx

from hazard_assessment.config.settings import IngestSettings
from hazard_assessment.ingest.base import (
    BaseIngestConnector,
    ConnectorHealthStatus,
    StationHealth,
    StationHealthState,
    safe_float,
)
from hazard_assessment.ingest.hashing import canonicalize_json, compute_payload_hash

logger = logging.getLogger(__name__)

COOPS_DATAGETTER_URL: Final[str] = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
COOPS_PRODUCT_ONE_MINUTE: Final[str] = "one_minute_water_level"
COOPS_PRODUCT_SIX_MINUTE: Final[str] = "water_level"
COOPS_PRODUCT_EXPECTED_INTERVAL_SEC: Final[dict[str, int]] = {
    COOPS_PRODUCT_ONE_MINUTE: 60,
    COOPS_PRODUCT_SIX_MINUTE: 6 * 60,
}

# NOAA CO-OPS tsunami-capable tide gauge stations in the
# Pacific basin.  Station IDs are 7-digit NOS identifiers verified against the
# official NOAA Tides & Currents station pages (tidesandcurrents.noaa.gov).
# Organized west-to-east so the tuple is human-readable.
#
# Coverage rationale:
#   - Hawaii: first-arrival stations for trans-Pacific tsunamis from Aleutians/Japan
#   - US Pacific territories: Pago Pago (American Samoa), Guam - critical for
#     South Pacific / Western Pacific sources
#   - Wake Island: mid-Pacific early-warning node
#   - Alaska (Adak, Kodiak, St. Paul): Aleutian arc sources; Adak is the westernmost
#     US station and the earliest US land detector for Aleutian events
#   - US West Coast (WA -> CA): receiving coast; Crescent City is included because
#     its harbor geometry historically produces 3-4x amplification of tsunami waves
#   - Note: this list is curated for coverage, not filtered on product
#     availability. The connector asks each station for one_minute_water_level
#     and falls back to the 6-minute water_level product, keeping the station
#     either way. The fallback matters downstream: 6-minute sampling sits above
#     the anomaly detector's Nyquist limit for its 5-minute high cutoff, so a
#     station is scored with filter_degraded set and its threshold and wavelet
#     scores are flagged unreliable.
COOPS_PACIFIC_STATION_IDS: Final[tuple[str, ...]] = (
    # Hawaii
    "1612340",  # Honolulu, HI
    "1617760",  # Hilo, HI
    "1615680",  # Kahului (Maui), HI
    "1619910",  # Midway Island, HI
    # US Pacific territories and mid-Pacific
    "1890000",  # Wake Island
    "1770000",  # Pago Pago, American Samoa
    "1631428",  # Pago Bay, Guam
    # Alaska - Aleutian arc (east-to-west order)
    "9464212",  # Village Cove, St. Paul Island (Pribilof Islands), AK
    "9457292",  # Kodiak Island, AK
    "9461380",  # Adak Island, AK  (westernmost US tsunami gauge)
    # US Pacific Coast - Washington
    "9444900",  # Port Townsend, WA
    "9443090",  # Neah Bay, WA
    # Oregon
    "9432780",  # Charleston, OR
    # California (north to south)
    "9418767",  # North Spit (Eureka), CA
    "9419750",  # Crescent City, CA  (high amplification harbor)
    "9414290",  # San Francisco, CA
    "9413450",  # Monterey, CA
    "9410230",  # La Jolla (San Diego), CA
)


@dataclass(frozen=True, slots=True)
class CoopsRecord:
    """Structured CO-OPS observation emitted by the connector."""

    source_id: str
    station_id: str
    station_name: str | None
    product: str
    source_timestamp: datetime
    ingest_timestamp: datetime
    water_level_m: float | None
    flags: str
    quality: str
    payload_sha256: str = ""


class ProductUnavailableError(RuntimeError):
    """Raised when a CO-OPS product is unavailable for a station."""


def _set_station_status(
    state: StationHealthState,
    new_status: ConnectorHealthStatus,
    *,
    station_id: str,
    reason: str,
) -> None:
    """Change a station's status, logging the transition.

    A single buoy going quiet is the case this state exists to describe, and
    it is otherwise visible only to a caller that polls the health snapshot.
    Logging on change surfaces it while keeping the output to one line per
    real transition rather than one per poll of every station.
    """
    if new_status == state.status:
        return
    previous, state.status = state.status, new_status
    level = (
        logging.INFO if new_status == ConnectorHealthStatus.ONLINE else logging.WARNING
    )
    logger.log(
        level,
        "Station %s: %s -> %s (%s)",
        station_id,
        previous.value,
        new_status.value,
        reason,
    )


class CoopsIngestConnector(BaseIngestConnector[CoopsRecord]):
    """Poll CO-OPS datagetter and emit new water-level records."""

    def __init__(
        self,
        *,
        station_ids: tuple[str, ...] = COOPS_PACIFIC_STATION_IDS,
        settings: IngestSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
        preferred_product: str = COOPS_PRODUCT_ONE_MINUTE,
        fallback_product: str = COOPS_PRODUCT_SIX_MINUTE,
        lookback_minutes: int = 10,
        application: str = "ocean_hazard_assessment",
        clock: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        valid_products = {COOPS_PRODUCT_ONE_MINUTE, COOPS_PRODUCT_SIX_MINUTE}
        if preferred_product not in valid_products:
            raise ValueError(
                "CO-OPS preferred_product must be one of "
                f"{sorted(valid_products)} (got {preferred_product!r})"
            )
        if fallback_product not in valid_products:
            raise ValueError(
                "CO-OPS fallback_product must be one of "
                f"{sorted(valid_products)} (got {fallback_product!r})"
            )
        if lookback_minutes <= 0:
            raise ValueError("CO-OPS lookback_minutes must be greater than 0")

        ingest_settings = settings or IngestSettings()
        super().__init__(
            name="coops",
            poll_interval_sec=ingest_settings.coops_poll_interval_sec,
            expected_interval_sec=ingest_settings.coops_poll_interval_sec,
            settings=ingest_settings,
            http_client=http_client,
            clock=clock,
            sleep_fn=sleep_fn,
        )
        self._station_ids = station_ids
        self._preferred_product = preferred_product
        self._fallback_product = fallback_product
        self._lookback_minutes = lookback_minutes
        self._application = application
        self._latest_timestamp_by_station_product: dict[tuple[str, str], datetime] = {}
        self._active_product_by_station: dict[str, str] = {}
        self._station_health_by_id: dict[str, StationHealthState] = {
            station_id: StationHealthState() for station_id in station_ids
        }

    @property
    def expected_interval_sec(self) -> int:
        if not self._active_product_by_station:
            return COOPS_PRODUCT_EXPECTED_INTERVAL_SEC.get(self._preferred_product, 60)
        # Use the fastest active product cadence so stale detection does not
        # hide outages on 1-minute stations when another station is on fallback.
        return min(
            COOPS_PRODUCT_EXPECTED_INTERVAL_SEC.get(product, 60)
            for product in self._active_product_by_station.values()
        )

    @property
    def station_health(self) -> dict[str, StationHealth]:
        return {
            station_id: StationHealth(
                status=state.status,
                last_successful_poll_at=state.last_successful_poll_at,
                last_new_data_at=state.last_new_data_at,
                last_error_at=state.last_error_at,
                last_error=state.last_error,
            )
            for station_id, state in self._station_health_by_id.items()
        }

    async def fetch_records(self) -> list[CoopsRecord]:
        records: list[CoopsRecord] = []
        station_errors: list[tuple[str, Exception]] = []
        successful_stations = 0
        station_results = await asyncio.gather(
            *(self._fetch_station_records(station_id) for station_id in self._station_ids),
            return_exceptions=True,
        )
        for station_id, station_result in zip(self._station_ids, station_results, strict=True):
            if isinstance(station_result, asyncio.CancelledError):
                raise station_result
            if isinstance(station_result, Exception):
                station_errors.append((station_id, station_result))
                self._mark_station_offline(station_id, error=station_result)
                continue
            if isinstance(station_result, BaseException):
                wrapped_error = RuntimeError(str(station_result))
                station_errors.append((station_id, wrapped_error))
                self._mark_station_offline(station_id, error=wrapped_error)
                continue
            successful_stations += 1
            self._mark_station_success(station_id, had_new_data=bool(station_result))
            records.extend(station_result)

        if station_errors and successful_stations == 0:
            first_station, first_exc = station_errors[0]
            raise RuntimeError(
                f"CO-OPS polling failed for all stations (first failure at {first_station}: "
                f"{first_exc})"
            ) from first_exc
        return records

    async def _fetch_station_records(self, station_id: str) -> list[CoopsRecord]:
        try:
            records = await self._fetch_product_records(
                station_id=station_id,
                product=self._preferred_product,
            )
            self._active_product_by_station[station_id] = self._preferred_product
            return records
        except (httpx.HTTPError, ProductUnavailableError) as preferred_exc:
            logger.warning(
                "CO-OPS station %s: preferred product %s unavailable (%s), "
                "falling back to %s",
                station_id,
                self._preferred_product,
                preferred_exc,
                self._fallback_product,
            )

        try:
            records = await self._fetch_product_records(
                station_id=station_id,
                product=self._fallback_product,
            )
            self._active_product_by_station[station_id] = self._fallback_product
            return records
        except (httpx.HTTPError, ProductUnavailableError) as fallback_exc:
            raise RuntimeError(
                f"CO-OPS preferred and fallback products failed for station {station_id}: "
                f"{fallback_exc}"
            ) from fallback_exc

    def _station_expected_interval_sec(self, station_id: str) -> int:
        product = self._active_product_by_station.get(station_id, self._preferred_product)
        return COOPS_PRODUCT_EXPECTED_INTERVAL_SEC.get(product, 60)

    def _mark_station_offline(self, station_id: str, *, error: Exception) -> None:
        state = self._station_health_by_id.setdefault(station_id, StationHealthState())
        _set_station_status(
            state,
            ConnectorHealthStatus.OFFLINE,
            station_id=station_id,
            reason=f"poll failed: {error}",
        )
        state.last_error_at = self._clock()
        state.last_error = str(error)

    def _mark_station_success(self, station_id: str, *, had_new_data: bool) -> None:
        state = self._station_health_by_id.setdefault(station_id, StationHealthState())
        now = self._clock()
        state.last_successful_poll_at = now
        state.last_error_at = None
        state.last_error = None

        if had_new_data:
            _set_station_status(
                state, ConnectorHealthStatus.ONLINE,
                station_id=station_id, reason="new data",
            )
            state.last_new_data_at = now
            state.no_new_data_since = None
            return

        if state.no_new_data_since is None:
            state.no_new_data_since = now

        reference = state.last_new_data_at or state.no_new_data_since
        if reference is None:
            _set_station_status(
                state, ConnectorHealthStatus.ONLINE,
                station_id=station_id, reason="no freshness reference yet",
            )
            return

        stale_after = timedelta(seconds=self._station_expected_interval_sec(station_id) * 2)
        if now - reference >= stale_after:
            _set_station_status(
                state, ConnectorHealthStatus.STALE,
                station_id=station_id,
                reason=f"no new data for {int((now - reference).total_seconds())}s",
            )
            return
        _set_station_status(
            state, ConnectorHealthStatus.ONLINE,
            station_id=station_id, reason="new data within cadence",
        )

    async def _fetch_product_records(self, *, station_id: str, product: str) -> list[CoopsRecord]:
        now = self._clock()
        start = now - timedelta(minutes=self._lookback_minutes)
        params = {
            "begin_date": _format_coops_timestamp(start),
            "end_date": _format_coops_timestamp(now),
            "station": station_id,
            "product": product,
            "datum": "STND",
            "time_zone": "gmt",
            "units": "metric",
            "format": "json",
            "application": self._application,
        }
        raw = await self.fetch_bounded(COOPS_DATAGETTER_URL, params=params)

        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("CO-OPS response payload must be a JSON object")
        if "error" in payload:
            raise ProductUnavailableError(str(payload["error"]))

        metadata = payload.get("metadata") or {}
        station_name_raw = metadata.get("name")
        station_name = str(station_name_raw) if station_name_raw is not None else None
        rows = payload.get("data") or []
        if not isinstance(rows, list):
            raise ValueError("CO-OPS payload field 'data' must be a list")

        watermark_key = (station_id, product)
        watermark = self._latest_timestamp_by_station_product.get(watermark_key)

        parsed_records: list[CoopsRecord] = []
        skipped = 0
        ingest_timestamp = self._clock()
        for row in rows:
            if not isinstance(row, dict):
                skipped += 1
                continue
            source_timestamp = _parse_coops_timestamp(row.get("t"))
            if source_timestamp is None:
                skipped += 1
                continue
            # Rows before the watermark are already ingested. Strict less-than
            # keeps the record at the exact watermark timestamp: when the API
            # re-serves it in the next window, deduplication downstream (by
            # source_id) prevents double-counting. Trade-off: re-emitting the
            # boundary record makes had_new_data=True even when no genuinely
            # new data arrived, which can delay stale detection by one poll
            # cycle. This is acceptable - data loss from <= is worse than
            # delayed staleness.
            if watermark is not None and source_timestamp < watermark:
                continue

            parsed_records.append(
                CoopsRecord(
                    source_id=(
                        f"coops:{station_id}:{product}:{source_timestamp.strftime('%Y%m%d%H%M')}"
                    ),
                    station_id=station_id,
                    station_name=station_name,
                    product=product,
                    source_timestamp=source_timestamp,
                    ingest_timestamp=ingest_timestamp,
                    water_level_m=safe_float(row.get("v")),
                    flags=str(row.get("f", "")),
                    quality=str(row.get("q", "")),
                    payload_sha256=compute_payload_hash(canonicalize_json(row)),
                )
            )

        if skipped:
            logger.warning(
                "_fetch_product(%s, %s): skipped %d malformed row(s)",
                station_id,
                product,
                skipped,
            )

        if not parsed_records:
            return []

        parsed_records.sort(key=lambda record: record.source_timestamp)
        self._latest_timestamp_by_station_product[watermark_key] = parsed_records[
            -1
        ].source_timestamp
        return parsed_records


def _format_coops_timestamp(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y%m%d %H:%M")


def _parse_coops_timestamp(raw_value: object) -> datetime | None:
    if not isinstance(raw_value, str):
        return None
    try:
        parsed = datetime.strptime(raw_value, "%Y-%m-%d %H:%M")
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC)


