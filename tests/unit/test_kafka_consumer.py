"""Tests for the Kafka consumer wrapper (messaging/consumer.py).

Covers the checkpoint-facing transport contract: decoded records must
carry full Kafka coordinates including the
timestamp type, and messages that fail transport-level decoding must be
returned as rejected coordinate markers because the consumer position has
already advanced past them.

A fake confluent-kafka consumer is injected, so these tests run without a
broker or the confluent-kafka package.
"""

from __future__ import annotations

import json
from typing import Any

from hazard_assessment.messaging.consumer import (
    ConsumedBatch,
    KafkaConsumer,
    RejectedMessage,
)


class _FakeMsg:
    """Mimics the confluent_kafka.Message surface poll_batch touches."""

    def __init__(
        self,
        value: bytes | None,
        key: bytes | None = b"dart:21413",
        topic: str = "raw.observations",
        partition: int = 0,
        offset: int = 7,
        timestamp: tuple[int, int] = (1, 1700000000000),
        error: Any = None,
    ) -> None:
        self._value = value
        self._key = key
        self._topic = topic
        self._partition = partition
        self._offset = offset
        self._timestamp = timestamp
        self._error = error

    def value(self) -> bytes | None:
        # confluent_kafka.Message.value() returns None for a tombstone.
        return self._value

    def key(self) -> bytes | None:
        return self._key

    def topic(self) -> str:
        return self._topic

    def partition(self) -> int:
        return self._partition

    def offset(self) -> int:
        return self._offset

    def timestamp(self) -> tuple[int, int]:
        return self._timestamp

    def error(self) -> Any:
        return self._error


class _FakeConsumer:
    """Serves a fixed message sequence, then None."""

    def __init__(self, messages: list[_FakeMsg]) -> None:
        self._messages = list(messages)

    def poll(self, timeout: float = 0.0) -> _FakeMsg | None:
        return self._messages.pop(0) if self._messages else None


def _consumer_with(messages: list[_FakeMsg]) -> KafkaConsumer:
    consumer = KafkaConsumer(topics=["raw.observations"], bootstrap_servers=None)
    assert not consumer.is_connected  # constructed in disabled mode
    consumer._consumer = _FakeConsumer(messages)
    return consumer


def _envelope_bytes(**extra: Any) -> bytes:
    value = {
        "schema_version": "1.0",
        "timestamp": 0.0,
        "source_type": "dart",
        "record": {"station_id": "21413"},
        **extra,
    }
    return json.dumps(value).encode("utf-8")


class TestPollBatchDecoded:
    def test_decoded_record_carries_full_kafka_coordinates(self) -> None:
        consumer = _consumer_with([
            _FakeMsg(_envelope_bytes(message_id="dart:21413:20110311060000:2")),
        ])
        batch = consumer.poll_batch(timeout=0.0)

        assert batch.rejected == []
        assert len(batch.records) == 1
        rec = batch.records[0]
        assert rec["key"] == "dart:21413"
        assert rec["topic"] == "raw.observations"
        assert rec["partition"] == 0
        assert rec["offset"] == 7
        assert rec["timestamp"] == 1700000000000
        assert rec["timestamp_type"] == "CREATE_TIME"
        assert rec["value"]["message_id"] == "dart:21413:20110311060000:2"

    def test_timestamp_type_names(self) -> None:
        consumer = _consumer_with([
            _FakeMsg(_envelope_bytes(), offset=1, timestamp=(0, -1)),
            _FakeMsg(_envelope_bytes(), offset=2, timestamp=(2, 5)),
            _FakeMsg(_envelope_bytes(), offset=3, timestamp=(9, 5)),
        ])
        batch = consumer.poll_batch(timeout=0.0)
        types = [r["timestamp_type"] for r in batch.records]
        assert types == ["NOT_AVAILABLE", "LOG_APPEND_TIME", "9"]

    def test_missing_key_becomes_empty_string(self) -> None:
        consumer = _consumer_with([_FakeMsg(_envelope_bytes(), key=None)])
        batch = consumer.poll_batch(timeout=0.0)
        assert batch.records[0]["key"] == ""

    def test_disabled_consumer_returns_empty_batch(self) -> None:
        consumer = KafkaConsumer(
            topics=["raw.observations"], bootstrap_servers=None
        )
        batch = consumer.poll_batch(timeout=0.0)
        assert batch == ConsumedBatch()


class TestPollBatchRejected:
    def test_undecodable_json_yields_rejected_marker(self) -> None:
        consumer = _consumer_with([
            _FakeMsg(b"{not json", partition=2, offset=41),
            _FakeMsg(_envelope_bytes(), offset=42),
        ])
        batch = consumer.poll_batch(timeout=0.0)

        assert len(batch.records) == 1
        assert batch.rejected == [
            RejectedMessage(topic="raw.observations", partition=2, offset=41)
        ]

    def test_null_value_yields_rejected_marker(self) -> None:
        """A Kafka tombstone must be rejected, not crash the poll loop.

        ``Message.value()`` returns None for a null-valued record, which is
        legal on the wire. Calling ``.decode()`` on it raised AttributeError,
        which poll_batch does not catch, so the exception escaped and killed
        the pipeline worker. Offsets are committed manually only after
        successful processing, so a restart redelivered the same record and
        crashed again, wedging the sole FSM writer. The surrounding valid
        record proves the batch still completes.
        """
        consumer = _consumer_with([
            _FakeMsg(None, partition=3, offset=11),
            _FakeMsg(_envelope_bytes(), offset=12),
        ])
        batch = consumer.poll_batch(timeout=0.0)

        assert len(batch.records) == 1
        assert batch.rejected == [
            RejectedMessage(topic="raw.observations", partition=3, offset=11)
        ]

    def test_invalid_utf8_yields_rejected_marker(self) -> None:
        consumer = _consumer_with([_FakeMsg(b"\xff\xfe\xfd", offset=9)])
        batch = consumer.poll_batch(timeout=0.0)
        assert batch.records == []
        assert batch.rejected == [
            RejectedMessage(topic="raw.observations", partition=0, offset=9)
        ]

    def test_non_object_json_yields_rejected_marker(self) -> None:
        """A decodable JSON array or scalar is not a usable envelope; its
        coordinates must still enter the offset manifest."""
        consumer = _consumer_with([
            _FakeMsg(b"[1, 2, 3]", offset=5),
            _FakeMsg(b"42", offset=6),
        ])
        batch = consumer.poll_batch(timeout=0.0)
        assert batch.records == []
        assert [r.offset for r in batch.rejected] == [5, 6]

    def test_invalid_utf8_key_yields_rejected_marker(self) -> None:
        consumer = _consumer_with([
            _FakeMsg(_envelope_bytes(), key=b"\xff\xfe", offset=3),
        ])
        batch = consumer.poll_batch(timeout=0.0)
        assert batch.records == []
        assert [r.offset for r in batch.rejected] == [3]

    def test_consumer_error_event_is_neither_record_nor_rejection(self) -> None:
        """Error events (broker errors, partition EOF) carry no consumed
        message, so they must not enter the offset manifest."""
        consumer = _consumer_with([
            _FakeMsg(b"", error="broker down"),
            _FakeMsg(_envelope_bytes(), offset=11),
        ])
        batch = consumer.poll_batch(timeout=0.0)
        assert len(batch.records) == 1
        assert batch.records[0]["offset"] == 11
        assert batch.rejected == []
