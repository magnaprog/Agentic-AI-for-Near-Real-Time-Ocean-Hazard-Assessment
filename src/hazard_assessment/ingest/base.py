"""Shared ingestion connector primitives.

Connectors implement source-specific fetching and parsing while inheriting
poll-loop control, retry policy, and health-state tracking.
"""

from __future__ import annotations

import asyncio
import inspect
import logging
import math
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Generic, Protocol, TypeVar

import httpx

from hazard_assessment.config.settings import IngestSettings

logger = logging.getLogger(__name__)

# Default ceiling for a single fetched response body, in decoded bytes.
# Comfortably above a real payload from any of the three upstreams at their
# shipped settings: the largest live DART realtime2 file is about 180 kB and a
# CO-OPS or USGS page is smaller still. The one way to approach it is to raise
# SeismicIngestConnector(limit=...) far above its default of 20; at roughly
# 800 bytes per USGS feature a single page would need about 21,000 features to
# reach this ceiling. Pass an explicit max_bytes if you configure it that high.
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


class ResponseTooLargeError(ValueError):
    """A fetched response body exceeded the caller's byte budget.

    Subclasses ValueError so existing connector error handling, which already
    treats ValueError as a malformed-payload signal, is unchanged.
    """


class ConnectorHealthStatus(StrEnum):
    """Operational status for an ingest connector."""

    ONLINE = "ONLINE"
    OFFLINE = "OFFLINE"
    STALE = "STALE"


@dataclass(frozen=True, slots=True)
class StationHealth:
    """Health snapshot for a single source partition (e.g., station)."""

    status: ConnectorHealthStatus
    last_successful_poll_at: datetime | None
    last_new_data_at: datetime | None
    last_error_at: datetime | None
    last_error: str | None


@dataclass(slots=True)
class StationHealthState:
    """Mutable per-station health tracking state used by connectors."""

    status: ConnectorHealthStatus = ConnectorHealthStatus.ONLINE
    last_successful_poll_at: datetime | None = None
    last_new_data_at: datetime | None = None
    no_new_data_since: datetime | None = None
    last_error_at: datetime | None = None
    last_error: str | None = None


class IngestRecord(Protocol):
    """Minimum record contract emitted by all connectors."""

    @property
    def source_id(self) -> str: ...

    @property
    def source_timestamp(self) -> datetime: ...

    @property
    def ingest_timestamp(self) -> datetime: ...

    @property
    def payload_sha256(self) -> str: ...


RecordT = TypeVar("RecordT", bound=IngestRecord)
RecordEmitter = Callable[[RecordT], None | Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ConnectorHealth:
    """Snapshot of connector health and timing state."""

    status: ConnectorHealthStatus
    poll_interval_sec: int
    expected_interval_sec: int
    last_successful_poll_at: datetime | None
    last_new_data_at: datetime | None
    last_error_at: datetime | None
    last_error: str | None


class BaseIngestConnector(ABC, Generic[RecordT]):
    """Base class for periodic async ingest connectors."""

    def __init__(
        self,
        *,
        name: str,
        poll_interval_sec: int,
        expected_interval_sec: int | None = None,
        settings: IngestSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
        clock: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.name = name
        self._settings = settings or IngestSettings()
        self._base_poll_interval_sec = poll_interval_sec
        self._base_expected_interval_sec = expected_interval_sec or poll_interval_sec
        self._clock = clock or (lambda: datetime.now(UTC))
        self._sleep = sleep_fn or asyncio.sleep

        self._client = http_client or httpx.AsyncClient(timeout=httpx.Timeout(30.0))
        self._owns_client = http_client is None

        self._status = ConnectorHealthStatus.ONLINE
        self._last_successful_poll_at: datetime | None = None
        self._last_new_data_at: datetime | None = None
        self._no_new_data_since: datetime | None = None
        self._last_error_at: datetime | None = None
        self._last_error: str | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        return self._client

    async def _stream_bounded(
        self,
        url: str,
        params: dict[str, str] | None,
        max_bytes: int,
    ) -> tuple[bytes, str | None]:
        """Stream one response body, capped, and return it with its charset.

        The body is drained before ``raise_for_status()`` so that an error
        response still leaves the connection reusable: closing a half-read
        stream forces httpcore to drop the TCP connection, which would mean a
        fresh handshake per station per retry exactly when an upstream is
        already failing.
        """
        buf = bytearray()
        async with self.client.stream("GET", url, params=params) as response:
            async for chunk in response.aiter_bytes():
                buf += chunk
                if len(buf) > max_bytes:
                    raise ResponseTooLargeError(
                        f"{self.name}: response body from {url} exceeded "
                        f"{max_bytes} bytes"
                    )
            charset = response.charset_encoding
            if response.is_error:
                # The body is already drained here but httpx will not expose it
                # through the raised error (Response.text needs read()), and
                # CO-OPS and USGS both return machine-readable error payloads.
                # Log a bounded snippet so the reason is not lost.
                logger.warning(
                    "%s: %s returned HTTP %d: %s",
                    self.name, url, response.status_code,
                    bytes(buf[:512]).decode("utf-8", errors="replace"),
                )
            response.raise_for_status()
        return bytes(buf), charset

    async def fetch_bounded(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        max_bytes: int = MAX_RESPONSE_BYTES,
    ) -> bytes:
        """GET a response body, giving up once max_bytes have arrived.

        Connectors buffer a whole payload before parsing it, so without a cap
        an oversized response is fully materialized before any validation can
        reject it. The cap applies to decoded bytes as they accumulate, which
        a Content-Length check cannot do: that header describes the compressed
        body and is supplied by the peer.

        Two limits worth stating plainly. The cap bounds one request, not the
        process: connectors fan out with ``asyncio.gather``, so the worst case
        is roughly ``max_bytes`` times the station count. And it bounds
        accumulated output, not the decompressor, which httpx calls without a
        length limit, so a single compressed chunk can expand well past
        ``max_bytes`` before this loop sees it. The cap shortens a hostile or
        malformed response; it is not a hard memory ceiling.
        """
        body, _ = await self._stream_bounded(url, params, max_bytes)
        return body

    async def fetch_bounded_text(
        self,
        url: str,
        *,
        params: dict[str, str] | None = None,
        max_bytes: int = MAX_RESPONSE_BYTES,
    ) -> str:
        """Same as :meth:`fetch_bounded`, decoded the way ``Response.text`` is.

        The charset comes from the response headers, falling back to UTF-8,
        with undecodable bytes replaced. Hardcoding UTF-8 here would be wrong:
        NDBC serves ``text/plain; charset=ISO-8859-1``, and the two decodings
        disagree on high bytes in a way that changes which rows a parser
        accepts and what payload hash they carry.
        """
        body, charset = await self._stream_bounded(url, params, max_bytes)
        return body.decode(charset or "utf-8", errors="replace")

    @property
    def status(self) -> ConnectorHealthStatus:
        return self._status

    def _set_status(self, new_status: ConnectorHealthStatus, *, reason: str) -> None:
        """Change connector status, logging the transition.

        A connector that fetches successfully but returns nothing new is
        otherwise silent, so a stale feed would reach an operator only if
        something polled the health snapshot. Logging on change surfaces it
        while keeping the output bounded to real transitions rather than one
        line per poll.
        """
        if new_status == self._status:
            return
        previous, self._status = self._status, new_status
        level = (
            logging.INFO
            if new_status == ConnectorHealthStatus.ONLINE
            else logging.WARNING
        )
        logger.log(
            level,
            "Connector %s: %s -> %s (%s)",
            self.name,
            previous.value,
            new_status.value,
            reason,
        )

    @property
    def poll_interval_sec(self) -> int:
        """Current poll interval in seconds.

        Connectors with dynamic modes (for example DART event mode)
        can override this property.
        """
        return self._base_poll_interval_sec

    @property
    def expected_interval_sec(self) -> int:
        """Expected data cadence in seconds for stale detection."""
        return self._base_expected_interval_sec

    @property
    def health(self) -> ConnectorHealth:
        """Return an immutable health snapshot."""
        return ConnectorHealth(
            status=self._status,
            poll_interval_sec=self.poll_interval_sec,
            expected_interval_sec=self.expected_interval_sec,
            last_successful_poll_at=self._last_successful_poll_at,
            last_new_data_at=self._last_new_data_at,
            last_error_at=self._last_error_at,
            last_error=self._last_error,
        )

    @abstractmethod
    async def fetch_records(self) -> list[RecordT]:
        """Fetch, parse, and deduplicate source records."""

    def has_new_data(self, records: list[RecordT]) -> bool:
        """Return whether a successful poll contains freshness signal."""
        return bool(records)

    async def poll_once(self) -> list[RecordT]:
        """Poll once with retry/backoff and health-state updates."""
        try:
            records = await self._fetch_with_retry()
        except Exception as exc:
            self._set_status(ConnectorHealthStatus.OFFLINE, reason="poll failed")
            self._last_error_at = self._clock()
            self._last_error = str(exc)
            raise

        now = self._clock()
        self._last_successful_poll_at = now
        self._last_error = None
        self._last_error_at = None
        # A successful poll clears transient OFFLINE state; stale logic below may
        # still downgrade to STALE when freshness exceeds threshold.
        self._set_status(ConnectorHealthStatus.ONLINE, reason="poll succeeded")

        if self.has_new_data(records):
            self._last_new_data_at = now
            self._no_new_data_since = None
            return records

        if self._no_new_data_since is None:
            self._no_new_data_since = now
        self._update_stale_status(now)
        return records

    async def run(
        self,
        *,
        emit: RecordEmitter[RecordT] | None = None,
        stop_event: asyncio.Event | None = None,
    ) -> None:
        """Run an indefinite poll loop until stopped."""
        while True:
            if stop_event is not None and stop_event.is_set():
                return

            try:
                records = await self.poll_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Connector '%s' poll failed; continuing loop", self.name)
                records = []

            if emit is not None:
                for record in records:
                    try:
                        result = emit(record)
                        if inspect.isawaitable(result):
                            await result
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        logger.exception(
                            "Connector '%s' emit failed; dropping record source_id=%s",
                            self.name,
                            getattr(record, "source_id", "<unknown>"),
                        )

            timeout = float(self.poll_interval_sec)
            if stop_event is None:
                await self._sleep(timeout)
                continue
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=timeout)
            except TimeoutError:
                continue
            return

    async def close(self) -> None:
        """Close internally-managed HTTP resources."""
        if self._owns_client:
            await self._client.aclose()

    async def __aenter__(self) -> BaseIngestConnector[RecordT]:
        return self

    async def __aexit__(
        self,
        _exc_type: type[BaseException] | None,
        _exc: BaseException | None,
        _tb: object | None,
    ) -> None:
        await self.close()

    async def _fetch_with_retry(self) -> list[RecordT]:
        """Retry failed fetches using exponential backoff configured in settings."""
        total_attempts = self._settings.retry_max_attempts + 1
        backoff_base = self._settings.retry_backoff_sec

        last_exc: Exception | None = None
        for attempt in range(total_attempts):
            try:
                return await self.fetch_records()
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # pragma: no cover - specific failures tested via poll_once
                last_exc = exc
                if attempt == total_attempts - 1:
                    break
                delay = backoff_base * (2**attempt)
                await self._sleep(delay)

        if last_exc is None:
            raise RuntimeError("retry loop exited without result or exception")
        raise last_exc

    def _update_stale_status(self, now: datetime) -> None:
        """Mark STALE after 2x expected interval with no new data."""
        reference = self._last_new_data_at or self._no_new_data_since
        if reference is None:
            return

        stale_after = timedelta(seconds=self.expected_interval_sec * 2)
        if now - reference >= stale_after:
            self._set_status(
                ConnectorHealthStatus.STALE,
                reason=f"no new data for {int((now - reference).total_seconds())}s",
            )
            return

        if self._status != ConnectorHealthStatus.OFFLINE:
            self._set_status(ConnectorHealthStatus.ONLINE, reason="new data within cadence")


def safe_float(raw_value: object) -> float | None:
    """Parse a value to float, returning None for non-finite or unparseable inputs."""
    if raw_value is None:
        return None
    if isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, (int, float)):
        parsed = float(raw_value)
        if not math.isfinite(parsed):
            return None
        return parsed
    if isinstance(raw_value, str) and raw_value.strip() == "":
        return None
    if not isinstance(raw_value, str):
        return None
    try:
        parsed = float(raw_value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def safe_int(raw_value: object) -> int | None:
    """Parse a value to int, returning None for non-finite or unparseable inputs."""
    if raw_value is None or isinstance(raw_value, bool):
        return None
    if isinstance(raw_value, int):
        return raw_value
    if isinstance(raw_value, float):
        if not math.isfinite(raw_value):
            return None
        try:
            return int(raw_value)
        except (TypeError, ValueError, OverflowError):
            return None
    if not isinstance(raw_value, str):
        return None
    try:
        return int(raw_value)
    except (TypeError, ValueError):
        return None
