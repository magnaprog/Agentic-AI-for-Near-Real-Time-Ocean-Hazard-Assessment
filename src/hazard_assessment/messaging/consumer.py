"""Kafka consumer wrapper for pipeline worker.

Thin wrapper around confluent_kafka.Consumer with JSON deserialization,
batch polling, and graceful shutdown.

When confluent-kafka is not installed or Kafka is not configured,
the consumer constructs in disabled mode (``is_connected`` False and
``poll_batch`` returns an empty batch); the pipeline worker then idles
awaiting shutdown instead of processing.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# librdkafka timestamp-type constants (rd_kafka_timestamp_type_t). Mapped
# locally so the names are available even when confluent-kafka is not
# installed. Unknown values fall back to the stringified integer.
_TIMESTAMP_TYPE_NAMES: dict[int, str] = {
    0: "NOT_AVAILABLE",
    1: "CREATE_TIME",
    2: "LOG_APPEND_TIME",
}


@dataclass(frozen=True, slots=True)
class RejectedMessage:
    """Kafka coordinates of a consumed message that failed decoding.

    These markers keep the consumed offset manifest complete: a batch
    whose payloads could not be decoded still advances the consumer
    position, so its coordinates must appear in checkpoint identity.
    """

    topic: str
    partition: int
    offset: int


@dataclass(frozen=True, slots=True)
class ConsumedBatch:
    """One poll_batch result: decoded records plus rejected coordinates."""

    records: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[RejectedMessage] = field(default_factory=list)


class KafkaConsumer:
    """Synchronous Kafka consumer with JSON deserialization.

    Consumer group: ``pipeline-workers`` (configurable).
    Uses manual commit after successful processing (at-least-once delivery).
    """

    def __init__(
        self,
        topics: list[str],
        group_id: str = "pipeline-workers",
        bootstrap_servers: str | None = None,
        auto_offset_reset: str = "earliest",
    ) -> None:
        self._consumer: Any = None
        self._topics = topics
        self._group_id = group_id
        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")

        if not servers:
            logger.info("Kafka consumer disabled (no KAFKA_BOOTSTRAP_SERVERS)")
            return

        try:
            from confluent_kafka import Consumer

            self._consumer = Consumer({
                "bootstrap.servers": servers,
                "group.id": group_id,
                "auto.offset.reset": auto_offset_reset,
                "enable.auto.commit": False,
                "max.poll.interval.ms": 300000,
                "session.timeout.ms": 30000,
            })
            self._consumer.subscribe(topics)
            logger.info(
                "Kafka consumer subscribed to %s (group=%s)", topics, group_id
            )
        except ImportError:
            logger.warning(
                "confluent-kafka not installed; Kafka consumer disabled"
            )
        except Exception:
            logger.exception("Failed to create Kafka consumer")

    @property
    def is_connected(self) -> bool:
        return self._consumer is not None

    @property
    def group_id(self) -> str:
        """Consumer group identity used in checkpoint derivation."""
        return self._group_id

    def poll_batch(
        self,
        timeout: float = 1.0,
        max_records: int = 100,
    ) -> ConsumedBatch:
        """Poll for a batch of messages with JSON deserialization.

        Returns decoded records plus the Kafka coordinates of messages
        that failed deserialization. Decode failures are logged and do
        not raise, but their coordinates are preserved because the
        consumer position has already advanced past them. Consumer-level
        error events (for example broker errors) carry no consumed
        message and are logged only.
        """
        if self._consumer is None:
            return ConsumedBatch()

        records: list[dict[str, Any]] = []
        rejected: list[RejectedMessage] = []
        for _ in range(max_records):
            msg = self._consumer.poll(
                timeout=timeout if not (records or rejected) else 0.0
            )
            if msg is None:
                break
            if msg.error():
                logger.error("Kafka consumer error: %s", msg.error())
                continue

            timestamp_type, timestamp_value = msg.timestamp()
            try:
                raw_value = msg.value()
                if raw_value is None:
                    # A null value is legal on the wire (Kafka tombstone) and
                    # carries no envelope. Calling .decode() on it raised
                    # AttributeError, which this handler does not catch, so the
                    # exception escaped poll_batch and killed the pipeline
                    # worker. The offset was never committed, so a restart
                    # redelivered the same message and crashed again. Route it
                    # into the reject path instead, matching the null-key guard
                    # below and keeping the offset manifest complete.
                    raise json.JSONDecodeError("null message value", "", 0)
                value = json.loads(raw_value.decode("utf-8"))
                if not isinstance(value, dict):
                    # A JSON scalar or array is not a usable envelope
                    # object; record it as a transport-level rejection
                    # rather than crashing downstream dict handling.
                    raise json.JSONDecodeError("not a JSON object", "", 0)
                key = msg.key().decode("utf-8") if msg.key() else ""
                records.append({
                    "key": key,
                    "value": value,
                    "topic": msg.topic(),
                    "partition": msg.partition(),
                    "offset": msg.offset(),
                    "timestamp": timestamp_value,
                    "timestamp_type": _TIMESTAMP_TYPE_NAMES.get(
                        timestamp_type, str(timestamp_type)
                    ),
                })
            except (json.JSONDecodeError, UnicodeDecodeError):
                logger.error(
                    "Failed to deserialize Kafka message at %s/%d/%d",
                    msg.topic(),
                    msg.partition(),
                    msg.offset(),
                )
                rejected.append(
                    RejectedMessage(
                        topic=msg.topic(),
                        partition=msg.partition(),
                        offset=msg.offset(),
                    )
                )

        return ConsumedBatch(records=records, rejected=rejected)

    def commit(self) -> None:
        """Commit current offsets synchronously."""
        if self._consumer is not None:
            self._consumer.commit()

    def close(self) -> None:
        """Close the consumer, leaving the consumer group."""
        if self._consumer is not None:
            self._consumer.close()
            logger.info("Kafka consumer closed")
