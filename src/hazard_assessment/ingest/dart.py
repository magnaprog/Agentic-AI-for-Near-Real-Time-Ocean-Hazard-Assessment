"""DART ingest connector."""

from __future__ import annotations

import asyncio
import logging
import math
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
)
from hazard_assessment.ingest.hashing import compute_payload_hash

logger = logging.getLogger(__name__)

DART_REALTIME_URL: Final[str] = "https://www.ndbc.noaa.gov/data/realtime2/{station_id}.dart"
# DART II standard mode: BPR subsamples every 15 min, but the surface buoy
# transmits a 6-hour batch (24 observations) to shore via Iridium.  New rows
# appear in the NDBC realtime2 file in ~6-hour batches, so stale detection
# should allow for this gap (2x cadence = 12 h before marking STALE).
DART_STANDARD_DATA_CADENCE_SEC: Final[int] = 6 * 60 * 60
DART_EVENT_DATA_CADENCE_SEC: Final[int] = 60
# NDBC notes that DART typically returns to standard mode after about 4 hours
# of 1-minute transmissions if no further events are detected.
DART_EVENT_MODE_TIMEOUT_SEC: Final[int] = 4 * 60 * 60

# Pacific DART stations polled by default: 214xx in the northwest Pacific,
# 464xx in the northeast Pacific.
# Validated against realtime2 availability and <=48h recency on 2026-03-04.
DART_PACIFIC_STATION_IDS: Final[tuple[str, ...]] = (
    "21413",
    "21415",
    "21416",
    "21419",
    "21420",
    "46404",
    "46407",
    "46409",
    "46411",
    "46413",
    "46414",
    "46416",
    "46419",
)


@dataclass(frozen=True, slots=True)
class DartRecord:
    """Structured DART observation emitted by the connector."""

    source_id: str
    station_id: str
    source_timestamp: datetime
    ingest_timestamp: datetime
    measurement_type: int
    height_m: float
    event_mode: bool  # Row-level: True only for event-mode rows (measurement_type 2/3)
    payload_sha256: str = ""


@dataclass(frozen=True, slots=True)
class _ParsedDartRow:
    source_timestamp: datetime
    measurement_type: int
    height_m: float
    raw_line: str = ""


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


class DartIngestConnector(BaseIngestConnector[DartRecord]):
    """Poll NOAA NDBC DART text files and emit new observations."""

    def __init__(
        self,
        *,
        station_ids: tuple[str, ...] = DART_PACIFIC_STATION_IDS,
        settings: IngestSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        ingest_settings = settings or IngestSettings()
        super().__init__(
            name="dart",
            poll_interval_sec=ingest_settings.dart_poll_interval_standard_sec,
            expected_interval_sec=ingest_settings.dart_poll_interval_standard_sec,
            settings=ingest_settings,
            http_client=http_client,
            clock=clock,
            sleep_fn=sleep_fn,
        )
        self._station_ids = station_ids
        self._standard_interval_sec = ingest_settings.dart_poll_interval_standard_sec
        self._event_interval_sec = ingest_settings.dart_poll_interval_event_sec
        self._event_mode_timeout_sec = ingest_settings.dart_event_mode_timeout_sec

        self._event_mode_stations: set[str] = set()
        self._event_mode_observed_at_by_station: dict[str, datetime] = {}
        self._latest_timestamp_by_station: dict[str, datetime] = {}
        self._next_poll_at_by_station: dict[str, datetime] = {}
        self._station_health_by_id: dict[str, StationHealthState] = {
            station_id: StationHealthState() for station_id in station_ids
        }

    @property
    def poll_interval_sec(self) -> int:
        if self._event_mode_stations:
            return self._event_interval_sec
        return self._standard_interval_sec

    @property
    def expected_interval_sec(self) -> int:
        if self._event_mode_stations:
            # Event mode polling is 15s for low-latency detection, but sustained
            # event packets are typically 1-minute averages (T=2) after the
            # initial short burst of 15-second samples (T=3).
            return DART_EVENT_DATA_CADENCE_SEC
        return DART_STANDARD_DATA_CADENCE_SEC

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

    async def fetch_records(self) -> list[DartRecord]:
        self._expire_stale_event_modes()
        now = self._clock()
        due_station_ids = [
            station_id
            for station_id in self._station_ids
            if self._is_station_due(station_id=station_id, now=now)
        ]
        if not due_station_ids:
            return []

        records: list[DartRecord] = []
        station_errors: list[tuple[str, Exception]] = []
        successful_stations = 0
        station_results = await asyncio.gather(
            *(self._fetch_station_records(station_id) for station_id in due_station_ids),
            return_exceptions=True,
        )
        for station_id, station_result in zip(due_station_ids, station_results, strict=True):
            if isinstance(station_result, asyncio.CancelledError):
                raise station_result
            if isinstance(station_result, Exception):
                station_errors.append((station_id, station_result))
                self._mark_station_offline(station_id, error=station_result)
                # Do NOT schedule the next poll here: scheduling pushes the
                # station past its due time, so the base-class retry of a full
                # outage would see an empty due-set, return success with zero
                # records, and mask the outage as "no new data".
                continue
            if isinstance(station_result, BaseException):
                wrapped_error = RuntimeError(str(station_result))
                station_errors.append((station_id, wrapped_error))
                self._mark_station_offline(station_id, error=wrapped_error)
                continue
            successful_stations += 1
            latest_source_timestamp = (
                station_result[-1].source_timestamp if station_result else None
            )
            self._mark_station_success(
                station_id,
                had_new_data=bool(station_result),
                latest_source_timestamp=latest_source_timestamp,
            )
            self._schedule_next_poll(station_id=station_id, now=self._clock())
            records.extend(station_result)

        if station_errors and successful_stations == 0:
            first_station, first_exc = station_errors[0]
            raise RuntimeError(
                "DART polling failed for all due stations "
                f"(first failure at {first_station}: {first_exc})"
            ) from first_exc
        return records

    async def _fetch_station_records(self, station_id: str) -> list[DartRecord]:
        payload = await self.fetch_bounded_text(
            DART_REALTIME_URL.format(station_id=station_id)
        )

        parsed = parse_dart_payload(payload, station_id=station_id)
        if not parsed:
            return []

        watermark = self._latest_timestamp_by_station.get(station_id)
        new_rows = [
            row for row in parsed if watermark is None or row.source_timestamp > watermark
        ]
        if not new_rows:
            return []

        latest = max(row.source_timestamp for row in new_rows)
        self._latest_timestamp_by_station[station_id] = latest

        # Event mode is latched from RECENT event rows, and the expiry clock is
        # stamped from the row's own timestamp rather than the poll time.
        # A .dart file carries weeks of history, so the first poll after a
        # restart returns the whole file: stamping the poll time put any
        # station with an old event row into event mode for the full timeout,
        # which polls NDBC every 15s instead of every 60s and reports the
        # station stale against a 60s expected cadence while it transmits
        # normally on its 6-hour batch. Live station 21416 carries 96 type-2
        # and 16 type-3 rows from a past event and did exactly that.
        now = self._clock()
        recent_event_rows = [
            row
            for row in new_rows
            if row.measurement_type in (2, 3)
            and (now - row.source_timestamp).total_seconds() < self._event_mode_timeout_sec
        ]
        if recent_event_rows:
            self._event_mode_stations.add(station_id)
            self._event_mode_observed_at_by_station[station_id] = max(
                row.source_timestamp for row in recent_event_rows
            )
        elif all(row.measurement_type == 1 for row in new_rows):
            self._event_mode_stations.discard(station_id)
            self._event_mode_observed_at_by_station.pop(station_id, None)

        ingest_timestamp = self._clock()
        return [
            DartRecord(
                source_id=(
                    f"dart:{station_id}:{row.source_timestamp.strftime('%Y%m%d%H%M%S')}:"
                    f"{row.measurement_type}"
                ),
                station_id=station_id,
                source_timestamp=row.source_timestamp,
                ingest_timestamp=ingest_timestamp,
                measurement_type=row.measurement_type,
                height_m=row.height_m,
                # Row-level evidence: measurement types 2 (1-min event data)
                # and 3 (15-s burst) ARE event-mode rows; a standard 15-min
                # row (type 1) in the same poll must not be stamped as
                # event-mode just because the station's polling cadence has
                # switched (which _event_mode_stations above tracks
                # separately).
                event_mode=row.measurement_type in (2, 3),
                payload_sha256=compute_payload_hash(row.raw_line.encode("utf-8")),
            )
            for row in new_rows
        ]

    def _expire_stale_event_modes(self) -> None:
        if not self._event_mode_stations:
            return

        now = self._clock()
        for station_id in list(self._event_mode_stations):
            observed_at = self._event_mode_observed_at_by_station.get(station_id)
            if observed_at is None:
                continue
            if (now - observed_at).total_seconds() < self._event_mode_timeout_sec:
                continue
            self._event_mode_stations.discard(station_id)
            self._event_mode_observed_at_by_station.pop(station_id, None)

    def _station_expected_interval_sec(self, station_id: str) -> int:
        if station_id in self._event_mode_stations:
            return DART_EVENT_DATA_CADENCE_SEC
        return DART_STANDARD_DATA_CADENCE_SEC

    def _station_poll_interval_sec(self, station_id: str) -> int:
        if station_id in self._event_mode_stations:
            return self._event_interval_sec
        return self._standard_interval_sec

    def _is_station_due(self, *, station_id: str, now: datetime) -> bool:
        next_poll_at = self._next_poll_at_by_station.get(station_id)
        if next_poll_at is None:
            return True
        return now >= next_poll_at

    def _schedule_next_poll(self, *, station_id: str, now: datetime) -> None:
        interval = self._station_poll_interval_sec(station_id)
        self._next_poll_at_by_station[station_id] = now + timedelta(seconds=interval)

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

    def _mark_station_success(
        self,
        station_id: str,
        *,
        had_new_data: bool,
        latest_source_timestamp: datetime | None = None,
    ) -> None:
        state = self._station_health_by_id.setdefault(station_id, StationHealthState())
        now = self._clock()
        state.last_successful_poll_at = now
        state.last_error_at = None
        state.last_error = None

        if had_new_data:
            if latest_source_timestamp is None:
                _set_station_status(
                    state, ConnectorHealthStatus.ONLINE,
                    station_id=station_id, reason="new data",
                )
                state.last_new_data_at = now
                state.no_new_data_since = None
                return

            stale_after = timedelta(seconds=self._station_expected_interval_sec(station_id) * 2)
            if now - latest_source_timestamp >= stale_after:
                _set_station_status(
                    state, ConnectorHealthStatus.STALE,
                    station_id=station_id,
                    reason="newest sample older than twice the expected cadence",
                )
                # Preserve source-time recency so stale classification is based on
                # observed source freshness, not only local ingest poll time.
                state.last_new_data_at = latest_source_timestamp
                state.no_new_data_since = latest_source_timestamp
                return

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


#: NDBC writes this in a data column when the value is missing. It is a
#: documented convention, not corruption, and appears in every realtime2 file
#: that has a gap. The numeric 9999.0 sentinel below means the same thing.
_MISSING_VALUE_MARKER = "MM"


def parse_dart_payload(payload: str, *, station_id: str) -> list[_ParsedDartRow]:
    """Parse a `.dart` payload into typed rows sorted by source timestamp.

    Rows are dropped for two different reasons and the distinction matters:
    a missing-value marker is routine, while an unparseable row means the feed
    or this parser is wrong. Counting them together produced a warning on
    every poll of any station with a data gap, which is most of them, so a
    genuinely corrupt feed would have been indistinguishable from the steady
    background noise.
    """
    rows: list[_ParsedDartRow] = []
    skipped = 0
    missing = 0
    for raw_line in payload.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue

        parts = line.split()
        if len(parts) < 8:
            skipped += 1
            continue

        if any(part.upper() == _MISSING_VALUE_MARKER for part in parts[:8]):
            missing += 1
            continue

        try:
            year = int(parts[0])
            if year < 100:
                year += 2000
            month = int(parts[1])
            day = int(parts[2])
            hour = int(parts[3])
            minute = int(parts[4])
            second = int(parts[5])
            measurement_type = int(parts[6])
            height_m = float(parts[7])
        except ValueError:
            skipped += 1
            continue
        if measurement_type not in (1, 2, 3):
            skipped += 1
            continue
        if not math.isfinite(height_m) or height_m >= 9999.0:
            # The numeric form of the same missing-value convention.
            missing += 1
            continue

        try:
            source_timestamp = datetime(
                year,
                month,
                day,
                hour,
                minute,
                second,
                tzinfo=UTC,
            )
        except ValueError:
            skipped += 1
            continue
        rows.append(
            _ParsedDartRow(
                source_timestamp=source_timestamp,
                measurement_type=measurement_type,
                height_m=height_m,
                raw_line=line,
            )
        )

    if skipped:
        logger.warning(
            "parse_dart_payload(%s): skipped %d unparseable row(s)",
            station_id,
            skipped,
        )
    if missing:
        logger.debug(
            "parse_dart_payload(%s): skipped %d row(s) marked missing by NDBC",
            station_id,
            missing,
        )
    rows.sort(key=lambda row: row.source_timestamp)
    return rows
