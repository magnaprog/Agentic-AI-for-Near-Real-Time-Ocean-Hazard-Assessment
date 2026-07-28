"""Unit tests for ingest connectors."""

from __future__ import annotations

import asyncio
import gzip
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from hazard_assessment.config.settings import IngestSettings
from hazard_assessment.ingest.base import (
    BaseIngestConnector,
    ConnectorHealthStatus,
    ResponseTooLargeError,
    safe_int,
)
from hazard_assessment.ingest.coops import (
    COOPS_PRODUCT_ONE_MINUTE,
    COOPS_PRODUCT_SIX_MINUTE,
    CoopsIngestConnector,
)
from hazard_assessment.ingest.dart import (
    DART_EVENT_DATA_CADENCE_SEC,
    DART_EVENT_MODE_TIMEOUT_SEC,
    DART_STANDARD_DATA_CADENCE_SEC,
    DartIngestConnector,
    parse_dart_payload,
)
from hazard_assessment.ingest.seismic import (
    SeismicIngestConnector,
    _from_epoch_millis,
)


class ManualClock:
    """Mutable clock for deterministic timing tests."""

    def __init__(self, start: datetime) -> None:
        self.current = start

    def __call__(self) -> datetime:
        return self.current

    def advance(self, *, seconds: int) -> None:
        self.current += timedelta(seconds=seconds)


@dataclass(frozen=True, slots=True)
class _TestRecord:
    source_id: str
    source_timestamp: datetime
    ingest_timestamp: datetime
    payload_sha256: str = ""


class _TestConnector(BaseIngestConnector[_TestRecord]):
    """Minimal connector used to test base retry/health behavior."""

    def __init__(
        self,
        *,
        responses: list[list[_TestRecord] | Exception],
        settings: IngestSettings,
        clock: ManualClock,
        sleep_calls: list[float],
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        super().__init__(
            name="test",
            poll_interval_sec=10,
            expected_interval_sec=10,
            settings=settings,
            clock=clock,
            sleep_fn=sleep_fn or (lambda duration: _capture_sleep(duration, sleep_calls)),
        )
        self._responses = responses

    async def fetch_records(self) -> list[_TestRecord]:
        if not self._responses:
            return []
        result = self._responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


async def _capture_sleep(duration: float, calls: list[float]) -> None:
    calls.append(duration)


@pytest.mark.asyncio
async def test_base_connector_retries_and_marks_offline_after_exhaustion() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    sleep_calls: list[float] = []
    settings = IngestSettings(retry_max_attempts=3, retry_backoff_sec=2.0)
    connector = _TestConnector(
        responses=[
            RuntimeError("boom-1"),
            RuntimeError("boom-2"),
            RuntimeError("boom-3"),
            RuntimeError("boom-4"),
        ],
        settings=settings,
        clock=clock,
        sleep_calls=sleep_calls,
    )

    with pytest.raises(RuntimeError, match="boom-4"):
        await connector.poll_once()

    assert sleep_calls == [2.0, 4.0, 8.0]
    assert connector.health.status == ConnectorHealthStatus.OFFLINE
    await connector.close()


@pytest.mark.asyncio
async def test_base_connector_marks_stale_after_two_expected_intervals() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    sleep_calls: list[float] = []
    settings = IngestSettings()
    connector = _TestConnector(
        responses=[
            [
                _TestRecord(
                    source_id="test:1",
                    source_timestamp=datetime(2026, 3, 4, 7, 59, tzinfo=UTC),
                    ingest_timestamp=clock(),
                )
            ],
            [],
        ],
        settings=settings,
        clock=clock,
        sleep_calls=sleep_calls,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    clock.advance(seconds=21)  # 2x expected interval is 20 seconds.
    second = await connector.poll_once()
    assert second == []
    assert connector.health.status == ConnectorHealthStatus.STALE
    await connector.close()


@pytest.mark.asyncio
async def test_base_connector_marks_stale_without_prior_new_records() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    sleep_calls: list[float] = []
    connector = _TestConnector(
        responses=[[], []],
        settings=IngestSettings(),
        clock=clock,
        sleep_calls=sleep_calls,
    )

    first = await connector.poll_once()
    assert first == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    clock.advance(seconds=21)
    second = await connector.poll_once()
    assert second == []
    assert connector.health.status == ConnectorHealthStatus.STALE
    await connector.close()


@pytest.mark.asyncio
async def test_base_connector_run_continues_after_poll_exception() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    emitted: list[_TestRecord] = []
    stop_event = asyncio.Event()

    def emit(record: _TestRecord) -> None:
        emitted.append(record)
        if len(emitted) >= 1:
            stop_event.set()

    connector = _TestConnector(
        responses=[
            RuntimeError("poll-failed"),
            [
                _TestRecord(
                    source_id="test:2",
                    source_timestamp=datetime(2026, 3, 4, 8, 0, 10, tzinfo=UTC),
                    ingest_timestamp=clock(),
                )
            ],
        ],
        settings=IngestSettings(retry_max_attempts=0),
        clock=clock,
        sleep_calls=[],
    )
    # Fast-forward loop cadence for deterministic test runtime.
    connector._base_poll_interval_sec = 0

    await connector.run(emit=emit, stop_event=stop_event)
    assert len(emitted) == 1
    assert emitted[0].source_id == "test:2"
    assert connector.health.status == ConnectorHealthStatus.ONLINE
    await connector.close()


@pytest.mark.asyncio
async def test_base_connector_run_continues_after_emit_exception() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    stop_event = asyncio.Event()
    attempted_ids: list[str] = []
    delivered_ids: list[str] = []
    fail_first_emit = True

    def emit(record: _TestRecord) -> None:
        nonlocal fail_first_emit
        attempted_ids.append(record.source_id)
        if fail_first_emit:
            fail_first_emit = False
            raise RuntimeError("emit-failed")
        delivered_ids.append(record.source_id)
        stop_event.set()

    connector = _TestConnector(
        responses=[
            [
                _TestRecord(
                    source_id="test:emit-1",
                    source_timestamp=datetime(2026, 3, 4, 8, 0, 10, tzinfo=UTC),
                    ingest_timestamp=clock(),
                )
            ],
            [
                _TestRecord(
                    source_id="test:emit-2",
                    source_timestamp=datetime(2026, 3, 4, 8, 0, 20, tzinfo=UTC),
                    ingest_timestamp=clock(),
                )
            ],
        ],
        settings=IngestSettings(retry_max_attempts=0),
        clock=clock,
        sleep_calls=[],
    )
    connector._base_poll_interval_sec = 0

    await connector.run(emit=emit, stop_event=stop_event)

    assert attempted_ids == ["test:emit-1", "test:emit-2"]
    assert delivered_ids == ["test:emit-2"]
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()


@pytest.mark.asyncio
async def test_base_connector_recovers_online_after_error_with_empty_success_poll() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    connector = _TestConnector(
        responses=[
            RuntimeError("startup-failure"),
            [],
        ],
        settings=IngestSettings(retry_max_attempts=0),
        clock=clock,
        sleep_calls=[],
    )

    with pytest.raises(RuntimeError, match="startup-failure"):
        await connector.poll_once()
    assert connector.health.status == ConnectorHealthStatus.OFFLINE

    records = await connector.poll_once()
    assert records == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()


@pytest.mark.asyncio
async def test_dart_connector_parses_records_and_switches_to_event_mode() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    responses = [
        """#YY  MM DD hh mm ss  T   HEIGHT
2026 03 04 08 00 00  1   4541.234
2026 03 04 08 15 00  1   4541.238
""",
        """#YY  MM DD hh mm ss  T   HEIGHT
2026 03 04 08 00 00  1   4541.234
2026 03 04 08 15 00  1   4541.238
2026 03 04 08 15 15  3   4541.260
""",
    ]
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        assert request.url.path.endswith("/21413.dart")
        payload = responses[min(call_count, len(responses) - 1)]
        call_count += 1
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert [record.measurement_type for record in first] == [1, 1]
    assert connector.poll_interval_sec == 60

    clock.advance(seconds=60)
    second = await connector.poll_once()
    assert len(second) == 1
    assert second[0].measurement_type == 3
    assert second[0].event_mode is True
    assert connector.poll_interval_sec == 15

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_continues_when_one_station_fails() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/21413.dart"):
            return httpx.Response(500, text="upstream failure")
        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 04 08 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413", "21414"),
        http_client=client,
        clock=clock,
    )

    records = await connector.poll_once()
    assert len(records) == 1
    assert records[0].station_id == "21414"
    station_health = connector.station_health
    assert station_health["21413"].status == ConnectorHealthStatus.OFFLINE
    assert station_health["21414"].status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_retries_failed_station_and_fails_loud() -> None:
    """A failed station is retried, not scheduled past its due time.

    On the next poll the still-failing station is fetched again; with no
    successful station the poll raises and the connector reports OFFLINE
    rather than masking the outage as "no new data".
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/21413.dart"):
            return httpx.Response(500, text="upstream failure")
        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 04 08 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413", "21414"),
        http_client=client,
    )

    first = await connector.poll_once()
    assert len(first) == 1

    # Second poll retries 21413 (still failing) -> RuntimeError, OFFLINE.
    with pytest.raises(RuntimeError):
        await connector.poll_once()
    assert connector.health.status == ConnectorHealthStatus.OFFLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_standard_mode_uses_data_cadence_for_stale_detection() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 04 08 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.expected_interval_sec == 6 * 60 * 60

    clock.advance(seconds=180)
    second = await connector.poll_once()
    assert second == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_station_health_marks_stale_for_old_source_timestamp() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 01 00 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    records = await connector.poll_once()
    assert len(records) == 1
    station_health = connector.station_health
    assert station_health["21413"].status == ConnectorHealthStatus.STALE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_event_mode_expires_after_timeout_without_new_rows() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 08 00 00  3   4541.260\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.poll_interval_sec == 15

    clock.advance(seconds=DART_EVENT_MODE_TIMEOUT_SEC + 1)
    second = await connector.poll_once()
    assert second == []
    assert connector.poll_interval_sec == 60
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_event_mode_timeout_is_configurable() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 08 00 00  3   4541.260\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        settings=IngestSettings(dart_event_mode_timeout_sec=30),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.poll_interval_sec == 15

    clock.advance(seconds=31)
    second = await connector.poll_once()
    assert second == []
    assert connector.poll_interval_sec == 60

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_polls_non_event_stations_at_standard_cadence_during_event() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    call_counts = {"21413": 0, "21414": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        station_id = request.url.path.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
        call_counts[station_id] += 1
        if station_id == "21413":
            return httpx.Response(
                200,
                text=(
                    "#YY  MM DD hh mm ss  T   HEIGHT\n"
                    "2026 03 04 08 00 00  3   4541.260\n"
                ),
            )
        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 04 08 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413", "21414"),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 2
    assert connector.poll_interval_sec == 15
    assert call_counts == {"21413": 1, "21414": 1}

    clock.advance(seconds=15)
    second = await connector.poll_once()
    assert second == []
    assert call_counts == {"21413": 2, "21414": 1}

    clock.advance(seconds=15)
    third = await connector.poll_once()
    assert third == []
    assert call_counts == {"21413": 3, "21414": 1}

    clock.advance(seconds=30)
    fourth = await connector.poll_once()
    assert fourth == []
    assert call_counts == {"21413": 4, "21414": 2}

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_marks_offline_when_all_due_stations_fail() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    call_counts = {"21413": 0, "21414": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        station_id = request.url.path.rsplit("/", maxsplit=1)[-1].split(".", maxsplit=1)[0]
        call_counts[station_id] += 1

        if station_id == "21413":
            if call_counts[station_id] == 1:
                return httpx.Response(
                    200,
                    text=(
                        "#YY  MM DD hh mm ss  T   HEIGHT\n"
                        "2026 03 04 08 00 00  3   4541.260\n"
                    ),
                )
            return httpx.Response(500, text="event station failure")

        return httpx.Response(
            200,
            text=(
                "#YY  MM DD hh mm ss  T   HEIGHT\n"
                "2026 03 04 08 00 00  1   4541.234\n"
            ),
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413", "21414"),
        http_client=client,
        clock=clock,
        settings=IngestSettings(retry_max_attempts=0),
    )

    first = await connector.poll_once()
    assert len(first) == 2

    clock.advance(seconds=15)
    with pytest.raises(RuntimeError, match="all due stations"):
        await connector.poll_once()
    assert connector.health.status == ConnectorHealthStatus.OFFLINE
    station_health = connector.station_health
    assert station_health["21413"].status == ConnectorHealthStatus.OFFLINE
    assert station_health["21414"].status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_event_mode_uses_one_minute_stale_cadence() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 08 00 00  3   4541.260\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.expected_interval_sec == 60

    clock.advance(seconds=90)
    second = await connector.poll_once()
    assert second == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    clock.advance(seconds=31)
    third = await connector.poll_once()
    assert third == []
    assert connector.health.status == ConnectorHealthStatus.STALE

    await connector.close()
    await client.aclose()


def test_dart_parser_skips_non_finite_height_values() -> None:
    parsed = parse_dart_payload(
        (
            "#YY  MM DD hh mm ss  T   HEIGHT\n"
            "2026 03 04 08 00 00  1   NaN\n"
            "2026 03 04 08 15 00  1   4541.238\n"
        ),
        station_id="21413",
    )
    assert len(parsed) == 1
    assert parsed[0].height_m == pytest.approx(4541.238)


def test_dart_parser_skips_unknown_measurement_type() -> None:
    parsed = parse_dart_payload(
        (
            "#YY  MM DD hh mm ss  T   HEIGHT\n"
            "2026 03 04 08 00 00  9   4541.200\n"
            "2026 03 04 08 15 00  2   4541.260\n"
        ),
        station_id="21413",
    )
    assert len(parsed) == 1
    assert parsed[0].measurement_type == 2


@pytest.mark.asyncio
async def test_coops_connector_falls_back_to_six_minute_product() -> None:
    seen_products: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        product = request.url.params["product"]
        seen_products.append(product)

        if product == COOPS_PRODUCT_ONE_MINUTE:
            return httpx.Response(200, json={"error": {"message": "No data available"}})

        assert request.url.params["datum"] == "STND"
        assert request.url.params["time_zone"] == "gmt"
        assert request.url.params["units"] == "metric"
        return httpx.Response(
            200,
            json={
                "metadata": {"id": "1612340", "name": "Honolulu"},
                "data": [
                    {"t": "2026-03-04 08:01", "v": "0.584", "f": "0,0,0,0", "q": "p"}
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(station_ids=("1612340",), http_client=client)

    records = await connector.poll_once()
    assert seen_products == [COOPS_PRODUCT_ONE_MINUTE, COOPS_PRODUCT_SIX_MINUTE]
    assert len(records) == 1
    assert records[0].station_id == "1612340"
    assert records[0].product == COOPS_PRODUCT_SIX_MINUTE
    assert records[0].water_level_m == pytest.approx(0.584)

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_maps_non_finite_water_level_to_none() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "metadata": {"id": "1612340", "name": "Honolulu"},
                "data": [{"t": "2026-03-04 08:01", "v": "NaN", "f": "", "q": "p"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(station_ids=("1612340",), http_client=client)

    records = await connector.poll_once()
    assert len(records) == 1
    assert records[0].water_level_m is None

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_skips_non_object_rows() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "metadata": {"id": "1612340", "name": "Honolulu"},
                "data": [
                    "invalid-row",
                    {"t": "2026-03-04 08:01", "v": "0.584", "f": "", "q": "p"},
                    123,
                ],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(station_ids=("1612340",), http_client=client)

    records = await connector.poll_once()
    assert len(records) == 1
    assert records[0].water_level_m == pytest.approx(0.584)

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_does_not_fallback_when_preferred_has_no_new_records() -> None:
    request_products: list[str] = []
    payload = {
        "metadata": {"id": "1612340", "name": "Honolulu"},
        "data": [{"t": "2026-03-04 08:01", "v": "0.584", "f": "0,0,0,0", "q": "p"}],
    }

    def handler(request: httpx.Request) -> httpx.Response:
        request_products.append(request.url.params["product"])
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(station_ids=("1612340",), http_client=client)

    first = await connector.poll_once()
    assert len(first) == 1

    second = await connector.poll_once()
    # With strict-less-than watermark, the record at the watermark timestamp
    # is re-returned (downstream deduplication handles it); the key assertion
    # is that the connector did NOT fall back to a different product.
    assert len(second) == 1
    assert request_products == [COOPS_PRODUCT_ONE_MINUTE, COOPS_PRODUCT_ONE_MINUTE]

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_continues_when_one_station_fails() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        station_id = request.url.params["station"]
        if station_id == "1612340":
            return httpx.Response(500, text="failure")
        return httpx.Response(
            200,
            json={
                "metadata": {"id": station_id, "name": "Hilo"},
                "data": [{"t": "2026-03-04 08:01", "v": "0.200", "f": "", "q": "p"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(
        station_ids=("1612340", "1617760"),
        http_client=client,
    )

    records = await connector.poll_once()
    assert len(records) == 1
    assert records[0].station_id == "1617760"
    station_health = connector.station_health
    assert station_health["1612340"].status == ConnectorHealthStatus.OFFLINE
    assert station_health["1617760"].status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_does_not_fail_if_healthy_station_has_no_new_rows() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        station_id = request.url.params["station"]
        if station_id == "1612340":
            return httpx.Response(500, text="failure")
        return httpx.Response(
            200,
            json={
                "metadata": {"id": station_id, "name": "Hilo"},
                "data": [{"t": "2026-03-04 08:01", "v": "0.200", "f": "", "q": "p"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(
        station_ids=("1612340", "1617760"),
        http_client=client,
    )

    first = await connector.poll_once()
    assert len(first) == 1

    second = await connector.poll_once()
    # Record at watermark is re-returned with strict-less-than deduplication;
    # the key assertion is that health stays ONLINE despite one station failing.
    assert len(second) == 1
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_fallback_uses_six_minute_stale_cadence() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))

    def handler(request: httpx.Request) -> httpx.Response:
        product = request.url.params["product"]
        if product == COOPS_PRODUCT_ONE_MINUTE:
            return httpx.Response(200, json={"error": {"message": "No data available"}})
        return httpx.Response(
            200,
            json={
                "metadata": {"id": "1612340", "name": "Honolulu"},
                "data": [{"t": "2026-03-04 08:00", "v": "0.584", "f": "0,0,0,0", "q": "p"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(
        station_ids=("1612340",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert connector.expected_interval_sec == 6 * 60

    clock.advance(seconds=90)
    second = await connector.poll_once()
    # Record at watermark is re-returned with strict-less-than deduplication;
    # the key assertion is that health stays ONLINE with the six-minute cadence.
    assert len(second) == 1
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_mixed_products_uses_fastest_stale_cadence() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        station_id = request.url.params["station"]
        product = request.url.params["product"]
        if station_id == "1612340" and product == COOPS_PRODUCT_ONE_MINUTE:
            return httpx.Response(200, json={"error": {"message": "No data available"}})
        return httpx.Response(
            200,
            json={
                "metadata": {"id": station_id, "name": "Station"},
                "data": [{"t": "2026-03-04 08:00", "v": "0.100", "f": "", "q": "p"}],
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = CoopsIngestConnector(
        station_ids=("1612340", "1617760"),
        http_client=client,
    )

    records = await connector.poll_once()
    assert len(records) == 2
    assert connector.expected_interval_sec == 60

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_coops_connector_default_stale_cadence_uses_configured_preferred_product() -> None:
    connector = CoopsIngestConnector(
        station_ids=("1612340",),
        preferred_product=COOPS_PRODUCT_SIX_MINUTE,
        fallback_product=COOPS_PRODUCT_ONE_MINUTE,
    )

    assert connector.expected_interval_sec == 6 * 60

    await connector.close()


def test_coops_connector_rejects_non_positive_lookback() -> None:
    with pytest.raises(ValueError, match="lookback_minutes must be greater than 0"):
        CoopsIngestConnector(lookback_minutes=0)


def test_coops_connector_rejects_unknown_products() -> None:
    with pytest.raises(ValueError, match="preferred_product must be one of"):
        CoopsIngestConnector(preferred_product="bad_product")
    with pytest.raises(ValueError, match="fallback_product must be one of"):
        CoopsIngestConnector(fallback_product="bad_product")


@pytest.mark.asyncio
async def test_seismic_connector_treats_successful_empty_poll_as_healthy() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    call_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_params.append(dict(request.url.params))
        return httpx.Response(200, json={"features": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(http_client=client, clock=clock)

    first = await connector.poll_once()
    assert first == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE

    clock.advance(seconds=120)
    second = await connector.poll_once()
    assert second == []
    assert connector.health.status == ConnectorHealthStatus.ONLINE
    assert "starttime" in call_params[0]
    assert "updatedafter" not in call_params[0]
    assert call_params[0]["endtime"] == "2026-03-04T08:00:00Z"
    assert call_params[0]["starttime"] == "2026-03-04T07:55:00Z"
    assert "updatedafter" in call_params[1]
    assert "starttime" not in call_params[1]
    assert call_params[1]["endtime"] == "2026-03-04T08:02:00Z"
    assert call_params[1]["updatedafter"] == "2026-03-04T07:54:55Z"

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_keeps_bootstrap_floor_for_first_updatedafter_cursor() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    call_params: list[dict[str, str]] = []
    first_payload = {
        "features": [
            {
                "id": "us7000boot1",
                "properties": {
                    "mag": 6.0,
                    "place": "bootstrap event",
                    "time": 1772611140000,
                    "updated": 1772611200000,
                    "type": "earthquake",
                    "tsunami": 1,
                },
                "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
            }
        ]
    }
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_params.append(dict(request.url.params))
        call_count += 1
        if call_count == 1:
            return httpx.Response(200, json=first_payload)
        return httpx.Response(200, json={"features": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(http_client=client, clock=clock)

    first = await connector.poll_once()
    assert len(first) == 1

    clock.advance(seconds=15)
    second = await connector.poll_once()
    assert second == []

    assert call_params[0]["starttime"] == "2026-03-04T07:55:00Z"
    assert call_params[1]["updatedafter"] == "2026-03-04T07:54:55Z"

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_rejects_malformed_features_payload() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"features": {"id": "not-a-list"}})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(
        http_client=client,
        settings=IngestSettings(retry_max_attempts=0),
    )

    with pytest.raises(ValueError, match="USGS payload field 'features' must be a list"):
        await connector.poll_once()
    assert connector.health.status == ConnectorHealthStatus.OFFLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_rejects_missing_features_field() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"type": "FeatureCollection"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(
        http_client=client,
        settings=IngestSettings(retry_max_attempts=0),
    )

    with pytest.raises(ValueError, match="USGS payload field 'features' is required"):
        await connector.poll_once()
    assert connector.health.status == ConnectorHealthStatus.OFFLINE

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_paginates_when_page_hits_limit() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    request_offsets: list[str | None] = []
    request_endtimes: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        offset = request.url.params.get("offset")
        request_offsets.append(offset)
        request_endtimes.append(str(request.url.params.get("endtime")))
        if offset is None:
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "id": "us7000page1a",
                            "properties": {
                                "mag": 6.0,
                                "place": "page-1-a",
                                "time": 1772611200000,
                                "updated": 1772611260000,
                                "type": "earthquake",
                                "tsunami": 0,
                            },
                            "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                        },
                        {
                            "id": "us7000page1b",
                            "properties": {
                                "mag": 6.1,
                                "place": "page-1-b",
                                "time": 1772611260000,
                                "updated": 1772611320000,
                                "type": "earthquake",
                                "tsunami": 1,
                            },
                            "geometry": {"coordinates": [-75.0, -11.0, 30.0]},
                        },
                    ]
                },
            )
        assert offset == "3"
        return httpx.Response(
            200,
            json={
                "features": [
                    {
                        "id": "us7000page2a",
                        "properties": {
                            "mag": 5.9,
                            "place": "page-2-a",
                            "time": 1772611320000,
                            "updated": 1772611380000,
                            "type": "earthquake",
                            "tsunami": 0,
                        },
                        "geometry": {"coordinates": [-74.0, -10.0, 35.0]},
                    }
                ]
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(
        http_client=client,
        clock=clock,
        limit=2,
        max_pages_per_poll=3,
    )

    records = await connector.poll_once()
    assert len(records) == 3
    assert request_offsets == [None, "3"]
    assert request_endtimes == ["2026-03-04T08:00:00Z", "2026-03-04T08:00:00Z"]

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_keeps_cursor_when_page_window_is_truncated(
    caplog: pytest.LogCaptureFixture,
) -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    initial_updatedafter: str | None = None
    later_updatedafter: str | None = None
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal initial_updatedafter, later_updatedafter, call_count
        params = dict(request.url.params)
        offset = params.get("offset")
        call_count += 1

        if call_count == 1:
            assert "starttime" in params
            return httpx.Response(200, json={"features": []})

        if call_count == 2:
            initial_updatedafter = params.get("updatedafter")
            assert initial_updatedafter is not None
            assert offset is None
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "id": "us7000trunc1",
                            "properties": {
                                "mag": 6.0,
                                "place": "truncated-page-1",
                                "time": 1772611200000,
                                "updated": 1772611260000,
                                "type": "earthquake",
                                "tsunami": 0,
                            },
                            "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                        }
                    ]
                },
            )

        if call_count == 3:
            assert offset == "2"
            return httpx.Response(
                200,
                json={
                    "features": [
                        {
                            "id": "us7000trunc2",
                            "properties": {
                                "mag": 6.1,
                                "place": "truncated-page-2",
                                "time": 1772611320000,
                                "updated": 1772611380000,
                                "type": "earthquake",
                                "tsunami": 1,
                            },
                            "geometry": {"coordinates": [-75.0, -11.0, 30.0]},
                        }
                    ]
                },
            )

        assert call_count == 4
        later_updatedafter = params.get("updatedafter")
        assert offset is None
        return httpx.Response(200, json={"features": []})

    caplog.set_level("WARNING")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(
        http_client=client,
        clock=clock,
        limit=1,
        max_pages_per_poll=2,
    )

    first = await connector.poll_once()
    assert first == []

    clock.advance(seconds=15)
    second = await connector.poll_once()
    assert len(second) == 2

    clock.advance(seconds=15)
    third = await connector.poll_once()
    assert third == []

    assert initial_updatedafter == "2026-03-04T07:54:55Z"
    assert later_updatedafter == initial_updatedafter
    assert "page window truncated" in caplog.text
    assert "retaining updatedafter cursor" in caplog.text

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_guards_endtime_when_clock_moves_backwards() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    call_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        call_params.append(dict(request.url.params))
        return httpx.Response(200, json={"features": []})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(http_client=client, clock=clock)
    connector._latest_seen_update_timestamp = datetime(2026, 3, 4, 8, 10, tzinfo=UTC)

    records = await connector.poll_once()
    assert records == []
    assert len(call_params) == 1
    assert call_params[0]["updatedafter"] == "2026-03-04T08:09:55Z"
    assert call_params[0]["endtime"] == "2026-03-04T08:10:00Z"

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_maps_non_finite_numbers_to_none() -> None:
    payload = {
        "features": [
            {
                "id": "us7000finite",
                "properties": {
                    "mag": "NaN",
                    "place": "offshore",
                    "time": 1772611200000,
                    "updated": 1772611260000,
                    "type": "earthquake",
                    "tsunami": "NaN",
                },
                "geometry": {"coordinates": ["inf", "-inf", "NaN"]},
            }
        ]
    }

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(http_client=client)

    records = await connector.poll_once()
    assert len(records) == 1
    record = records[0]
    assert record.magnitude is None
    assert record.tsunami_flag is None
    assert record.longitude is None
    assert record.latitude is None
    assert record.depth_km is None

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_measurement_type_2_triggers_event_mode() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 08 00 00  2   4541.260\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    first = await connector.poll_once()
    assert len(first) == 1
    assert first[0].measurement_type == 2
    assert first[0].event_mode is True
    assert connector.poll_interval_sec == 15

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_dart_connector_event_mode_is_row_level_in_mixed_poll() -> None:
    """event_mode is row-level evidence: a standard (type 1) row in the same
    poll as an event row must not be stamped event_mode=True, even though the
    station's polling cadence switches to the event interval."""
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 08 00 00  2   4541.260\n"
        "2026 03 04 08 15 00  1   4541.238\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=client,
        clock=clock,
    )

    records = await connector.poll_once()
    flags = {r.measurement_type: r.event_mode for r in records}
    assert flags == {2: True, 1: False}
    # Cadence is station-level and stays on the event interval.
    assert connector.poll_interval_sec == 15

    await connector.close()
    await client.aclose()


def test_dart_parser_handles_two_digit_year() -> None:
    parsed = parse_dart_payload(
        (
            "#YY  MM DD hh mm ss  T   HEIGHT\n"
            "26 03 04 08 00 00  1   4541.234\n"
        ),
        station_id="21413",
    )
    assert len(parsed) == 1
    assert parsed[0].source_timestamp.year == 2026
    assert parsed[0].height_m == pytest.approx(4541.234)


def test_seismic_safe_int_handles_non_finite_float() -> None:
    assert safe_int(float("nan")) is None
    assert safe_int(float("inf")) is None


def test_seismic_from_epoch_millis_handles_out_of_range() -> None:
    assert _from_epoch_millis("99999999999999999999999") is None


def test_seismic_connector_rejects_invalid_limit_bounds() -> None:
    with pytest.raises(ValueError, match="between 1 and 20000"):
        SeismicIngestConnector(limit=0)
    with pytest.raises(ValueError, match="between 1 and 20000"):
        SeismicIngestConnector(limit=20_001)


def test_seismic_connector_rejects_non_positive_lookback() -> None:
    with pytest.raises(ValueError, match="lookback_minutes must be greater than 0"):
        SeismicIngestConnector(lookback_minutes=0)


@pytest.mark.asyncio
async def test_seismic_connector_prunes_event_state_when_max_tracking_exceeded() -> None:
    payloads = [
        {
            "features": [
                {
                    "id": "us7000prune1",
                    "properties": {
                        "mag": 6.0,
                        "place": "prune-1",
                        "time": 1772611200000,
                        "updated": 1772611260000,
                        "type": "earthquake",
                        "tsunami": 0,
                    },
                    "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                }
            ]
        },
        {
            "features": [
                {
                    "id": "us7000prune2",
                    "properties": {
                        "mag": 6.1,
                        "place": "prune-2",
                        "time": 1772611320000,
                        "updated": 1772611380000,
                        "type": "earthquake",
                        "tsunami": 1,
                    },
                    "geometry": {"coordinates": [-75.0, -11.0, 30.0]},
                }
            ]
        },
        {
            "features": [
                {
                    "id": "us7000prune3",
                    "properties": {
                        "mag": 6.2,
                        "place": "prune-3",
                        "time": 1772611440000,
                        "updated": 1772611500000,
                        "type": "earthquake",
                        "tsunami": 0,
                    },
                    "geometry": {"coordinates": [-74.0, -10.0, 35.0]},
                }
            ]
        },
    ]
    call_count = 0

    def handler(_request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        payload = payloads[min(call_count, len(payloads) - 1)]
        call_count += 1
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(
        http_client=client,
        max_tracked_events=2,
        event_state_retention_days=365,
    )

    await connector.poll_once()
    await connector.poll_once()
    await connector.poll_once()

    assert len(connector._version_by_event_id) == 2
    assert "us7000prune1" not in connector._version_by_event_id
    assert "us7000prune2" in connector._version_by_event_id
    assert "us7000prune3" in connector._version_by_event_id

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_seismic_connector_deduplicates_and_emits_revisions() -> None:
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payloads = [
        {
            "features": [
                {
                    "id": "us7000abc1",
                    "properties": {
                        "mag": 6.1,
                        "place": "near coast",
                        "time": 1772611200000,
                        "updated": 1772611260000,
                        "type": "earthquake",
                        "tsunami": 1,
                    },
                    "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                }
            ]
        },
        {
            "features": [
                {
                    "id": "us7000abc1",
                    "properties": {
                        "mag": 6.1,
                        "place": "near coast",
                        "time": 1772611200000,
                        "updated": 1772611260000,
                        "type": "earthquake",
                        "tsunami": 1,
                    },
                    "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                }
            ]
        },
        {
            "features": [
                {
                    "id": "us7000abc1",
                    "properties": {
                        "mag": 6.4,
                        "place": "near coast",
                        "time": 1772611200000,
                        "updated": 1772611320000,
                        "type": "earthquake",
                        "tsunami": 1,
                    },
                    "geometry": {"coordinates": [-76.1, -12.3, 25.0]},
                }
            ]
        },
    ]
    call_count = 0
    call_params: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_params.append(dict(request.url.params))
        payload = payloads[min(call_count, len(payloads) - 1)]
        call_count += 1
        return httpx.Response(200, json=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = SeismicIngestConnector(http_client=client, clock=clock)

    first = await connector.poll_once()
    assert len(first) == 1
    assert first[0].event_id == "us7000abc1"
    assert first[0].is_revision is False

    second = await connector.poll_once()
    assert second == []

    third = await connector.poll_once()
    assert len(third) == 1
    assert third[0].event_id == "us7000abc1"
    assert third[0].is_revision is True
    assert third[0].magnitude == pytest.approx(6.4)
    assert third[0].source_id != first[0].source_id
    assert "starttime" in call_params[0]
    assert "updatedafter" not in call_params[0]
    assert "updatedafter" in call_params[1]
    assert "starttime" not in call_params[1]
    assert "updatedafter" in call_params[2]
    assert "starttime" not in call_params[2]

    await connector.close()
    await client.aclose()


def test_missing_value_rows_are_not_reported_as_malformed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """NDBC writes MM for a missing value; that is routine, not corruption.

    Live station 46413 carries four MM rows, so the connector logged a warning
    on every poll, once a minute, forever. A genuinely corrupt feed would have
    been lost in that noise.
    """
    payload = "\n".join([
        "#YY  MM DD hh mm ss T   HEIGHT",
        "#yr  mo dy hr mn  s -        m",
        "2026 07 26 12 00 00 1 5604.270",
        "2026 07 26 11 45 00 1       MM",
        "2026 07 26 11 30 00 1 9999.000",
        "2026 07 26 11 15 00 1 5604.418",
    ])

    with caplog.at_level(logging.WARNING, logger="hazard_assessment.ingest.dart"):
        parsed = parse_dart_payload(payload, station_id="46413")

    assert [row.height_m for row in parsed] == [5604.418, 5604.270]
    assert "unparseable" not in caplog.text
    assert "malformed" not in caplog.text


def test_unparseable_rows_still_warn(caplog: pytest.LogCaptureFixture) -> None:
    """A row that is neither a value nor a documented marker is a real fault."""
    payload = "\n".join([
        "#YY  MM DD hh mm ss T   HEIGHT",
        "2026 07 26 12 00 00 1 5604.270",
        "2026 07 26 11 45 00 1 not-a-number",
    ])

    with caplog.at_level(logging.WARNING, logger="hazard_assessment.ingest.dart"):
        parsed = parse_dart_payload(payload, station_id="46413")

    assert len(parsed) == 1
    assert "skipped 1 unparseable row(s)" in caplog.text


def test_connector_status_transitions_are_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A stale feed must produce a signal an operator can see.

    A connector that fetches successfully but returns nothing new is the case
    an operator most needs to hear about, and it is otherwise silent.
    Transitions log; steady state does not, so the signal stays bounded.
    """
    from hazard_assessment.ingest.base import ConnectorHealthStatus

    transport = httpx.MockTransport(lambda request: httpx.Response(200, text=""))
    connector = DartIngestConnector(
        station_ids=("21413",),
        http_client=httpx.AsyncClient(transport=transport),
    )

    with caplog.at_level(logging.INFO, logger="hazard_assessment.ingest.base"):
        connector._set_status(ConnectorHealthStatus.STALE, reason="no new data for 900s")
        connector._set_status(ConnectorHealthStatus.STALE, reason="still nothing")
        connector._set_status(ConnectorHealthStatus.ONLINE, reason="new data within cadence")

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 2, lines
    assert "Connector dart: ONLINE -> STALE (no new data for 900s)" in lines[0]
    assert "Connector dart: STALE -> ONLINE (new data within cadence)" in lines[1]


def test_station_status_transitions_are_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One buoy going quiet is the case per-station health exists to describe."""
    from hazard_assessment.ingest.base import ConnectorHealthStatus, StationHealthState
    from hazard_assessment.ingest.dart import _set_station_status

    state = StationHealthState()

    with caplog.at_level(logging.INFO, logger="hazard_assessment.ingest.dart"):
        _set_station_status(
            state, ConnectorHealthStatus.STALE, station_id="21418", reason="no new data for 43200s"
        )
        _set_station_status(
            state, ConnectorHealthStatus.STALE, station_id="21418", reason="still nothing"
        )

    lines = [record.getMessage() for record in caplog.records]
    assert len(lines) == 1
    assert "Station 21418: ONLINE -> STALE (no new data for 43200s)" in lines[0]
    assert state.status is ConnectorHealthStatus.STALE


@pytest.mark.asyncio
async def test_historical_event_rows_do_not_latch_event_mode() -> None:
    """A .dart file carries weeks of history; old event rows are not evidence.

    The first poll after a restart returns the whole file. Live station 21416
    holds 96 type-2 and 16 type-3 rows from a past event, so stamping the poll
    time put it into event mode for the full timeout: NDBC polled every 15s
    instead of 60s, and the station was reported stale against a 60s cadence
    while transmitting normally on its 6-hour batch.
    """
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 02 01 03 00 00  2   4541.260\n"  # event row from a month ago
        "2026 02 01 03 01 00  3   4541.255\n"
        "2026 03 04 07 45 00  1   4541.238\n"  # current standard transmission
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21416",), http_client=client, clock=clock
    )

    await connector.poll_once()

    assert connector.poll_interval_sec == 60, "historical rows switched the cadence"
    assert connector._station_expected_interval_sec("21416") == DART_STANDARD_DATA_CADENCE_SEC

    await connector.close()
    await client.aclose()


@pytest.mark.asyncio
async def test_recent_event_rows_still_latch_event_mode() -> None:
    """The case the latch exists for keeps working."""
    clock = ManualClock(datetime(2026, 3, 4, 8, 0, tzinfo=UTC))
    payload = (
        "#YY  MM DD hh mm ss  T   HEIGHT\n"
        "2026 03 04 07 58 00  2   4541.260\n"
        "2026 03 04 07 59 00  1   4541.238\n"
    )

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=payload)

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    connector = DartIngestConnector(
        station_ids=("21413",), http_client=client, clock=clock
    )

    await connector.poll_once()

    assert connector.poll_interval_sec == 15
    assert connector._station_expected_interval_sec("21413") == DART_EVENT_DATA_CADENCE_SEC

    # The expiry clock runs from the row's own time, not the poll time: four
    # hours after the event row, event mode is gone.
    clock.advance(seconds=4 * 60 * 60 + 1)
    connector._expire_stale_event_modes()
    assert connector._station_expected_interval_sec("21413") == DART_STANDARD_DATA_CADENCE_SEC

    await connector.close()
    await client.aclose()


class _BoundedFetchConnector(BaseIngestConnector[_TestRecord]):
    """Bare connector used to exercise the shared bounded-fetch helpers."""

    def __init__(self, *, http_client: httpx.AsyncClient) -> None:
        super().__init__(name="bounded", poll_interval_sec=10, http_client=http_client)

    async def fetch_records(self) -> list[_TestRecord]:
        return []

    def has_new_data(self, records: list[_TestRecord]) -> bool:
        return True


def _bounded_connector(
    handler: Callable[[httpx.Request], httpx.Response],
) -> _BoundedFetchConnector:
    return _BoundedFetchConnector(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )


async def test_fetch_bounded_rejects_oversize_body() -> None:
    """The cap is what stops an oversized body reaching a parser."""
    connector = _bounded_connector(
        lambda request: httpx.Response(200, content=b"x" * 4096)
    )
    with pytest.raises(ResponseTooLargeError, match="exceeded 1024 bytes"):
        await connector.fetch_bounded("https://example.test/big", max_bytes=1024)
    await connector.close()


async def test_fetch_bounded_accepts_body_of_exactly_max_bytes() -> None:
    """The bound is inclusive, so a payload at the limit is not rejected."""
    connector = _bounded_connector(
        lambda request: httpx.Response(200, content=b"y" * 1024)
    )
    assert len(await connector.fetch_bounded("https://example.test/exact", max_bytes=1024)) == 1024
    await connector.close()


async def test_fetch_bounded_caps_decoded_bytes_not_transferred_bytes() -> None:
    """A small compressed body that expands past the cap is still rejected.

    A Content-Length check cannot do this: the header describes the compressed
    body, which is well under the cap here.
    """
    cap = 16 * 1024
    payload = gzip.compress(b"\0" * (4 * 1024 * 1024))
    # Compressed body is under the cap; only the decoded size exceeds it, so a
    # Content-Length check would have let this through.
    assert len(payload) < cap
    connector = _bounded_connector(
        lambda request: httpx.Response(
            200, content=payload, headers={"Content-Encoding": "gzip"}
        )
    )
    with pytest.raises(ResponseTooLargeError):
        await connector.fetch_bounded("https://example.test/bomb", max_bytes=cap)
    await connector.close()


async def test_fetch_bounded_raises_for_error_status() -> None:
    """CO-OPS product fallback depends on this staying an httpx.HTTPError."""
    connector = _bounded_connector(lambda request: httpx.Response(503, content=b"down"))
    with pytest.raises(httpx.HTTPStatusError) as excinfo:
        await connector.fetch_bounded("https://example.test/down")
    assert excinfo.value.response.status_code == 503
    assert isinstance(excinfo.value, httpx.HTTPError)
    await connector.close()


async def test_fetch_bounded_text_honours_the_served_charset() -> None:
    """NDBC serves ISO-8859-1; decoding as UTF-8 would corrupt high bytes.

    A mis-decoded byte does not just look wrong: U+00A0 is whitespace to
    str.split() while U+FFFD is not, so the column count of a DART row shifts
    and the row is accepted or dropped differently.
    """
    body = "1 5432.100 café\n".encode("iso-8859-1")
    connector = _bounded_connector(
        lambda request: httpx.Response(
            200, content=body, headers={"Content-Type": "text/plain; charset=ISO-8859-1"}
        )
    )
    assert "café" in await connector.fetch_bounded_text("https://example.test/latin1")
    await connector.close()


async def test_fetch_bounded_text_defaults_to_utf8_without_a_charset() -> None:
    connector = _bounded_connector(
        lambda request: httpx.Response(
            200, content="café".encode(), headers={"Content-Type": "text/plain"}
        )
    )
    assert await connector.fetch_bounded_text("https://example.test/plain") == "café"
    await connector.close()
