"""Kafka producer wrapper for publishing ingest records.

Thin wrapper around confluent_kafka.Producer with JSON serialization,
delivery callbacks, and graceful flush on shutdown.

When confluent-kafka is not installed or Kafka is not configured,
the producer operates in no-op mode (logs records but does not publish).
"""

from __future__ import annotations

import functools
import json
import logging
import os
import time
from datetime import datetime
from typing import Any


def _json_default(obj: Any) -> Any:
    """Handle datetime and other non-JSON-serializable types."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

logger = logging.getLogger(__name__)

# Schema version included in every message for forward compatibility
SCHEMA_VERSION = "1.0"


class KafkaProducer:
    """Synchronous Kafka producer with JSON serialization.

    Falls back to no-op mode when confluent-kafka is not installed
    or bootstrap_servers is empty, so ingest workers can run without
    a Kafka broker during development.
    """

    def __init__(
        self,
        bootstrap_servers: str | None = None,
        client_id: str = "hazard-producer",
    ) -> None:
        self._producer: Any = None
        servers = bootstrap_servers or os.getenv("KAFKA_BOOTSTRAP_SERVERS", "")
        if not servers:
            logger.info("Kafka producer disabled (no KAFKA_BOOTSTRAP_SERVERS)")
            return

        try:
            from confluent_kafka import Producer

            self._producer = Producer({
                "bootstrap.servers": servers,
                "client.id": client_id,
                "acks": "all",
                "retries": 3,
                "retry.backoff.ms": 500,
                "linger.ms": 10,
                "batch.size": 65536,
            })
            logger.info("Kafka producer connected to %s", servers)
        except ImportError:
            logger.warning(
                "confluent-kafka not installed; Kafka producer disabled"
            )
        except Exception:
            logger.exception("Failed to create Kafka producer")

    @property
    def is_connected(self) -> bool:
        return self._producer is not None

    def produce(
        self,
        topic: str,
        key: str,
        value: dict[str, Any],
        headers: dict[str, str] | None = None,
        message_id: str | None = None,
    ) -> bool:
        """Publish a JSON message to a Kafka topic.

        Returns True if the message was queued locally, False if the
        producer is disabled. Queue acceptance is NOT broker
        acknowledgment; delivery outcomes surface asynchronously through
        the delivery callback, correlated to ``message_id`` when the
        caller provides a stable application-level message identity.
        """
        if self._producer is None:
            return False

        try:
            envelope: dict[str, Any] = {
                "schema_version": SCHEMA_VERSION,
                "timestamp": time.time(),
                **value,
            }
            if message_id:
                envelope["message_id"] = message_id
            msg_value = json.dumps(envelope, default=_json_default).encode("utf-8")

            msg_headers = [(k, v.encode("utf-8")) for k, v in (headers or {}).items()]

            self._producer.produce(
                topic=topic,
                key=key.encode("utf-8"),
                value=msg_value,
                headers=msg_headers,
                callback=functools.partial(
                    self._delivery_callback, message_id=message_id or ""
                ),
            )
            # Trigger delivery callbacks without blocking
            self._producer.poll(0)
            return True
        except Exception:
            logger.exception("Failed to produce message to %s", topic)
            # Service delivery callbacks even on failure (e.g. local queue
            # full): without a poll the queue never drains during an outage and
            # every subsequent record also fails.
            try:
                self._producer.poll(0)
            except Exception:
                logger.debug("poll(0) after produce failure also raised")
            return False

    def flush(self, timeout: float = 10.0) -> int:
        """Flush pending messages. Returns number of messages still in queue."""
        if self._producer is None:
            return 0
        remaining = self._producer.flush(timeout)
        if remaining > 0:
            logger.warning(
                "%d messages still in Kafka producer queue after flush", remaining
            )
        return int(remaining)

    def close(self) -> None:
        """Flush and close the producer."""
        if self._producer is not None:
            self.flush()
            logger.info("Kafka producer closed")

    @staticmethod
    def _delivery_callback(err: Any, msg: Any, message_id: str = "") -> None:
        """Log delivery outcome, correlated to the application message ID.

        Broker-confirmed coordinates and timing are logged on success so
        the systems-timing study can join producer acknowledgment to the
        stable application message identity. ``produce()`` returning True
        only means locally queued; this callback is the delivery truth.
        """
        if err is not None:
            logger.error(
                "Kafka delivery failed: topic=%s message_id=%s err=%s",
                msg.topic() if msg else "?",
                message_id,
                err,
            )
            return
        timestamp_type, timestamp_ms = msg.timestamp()
        logger.info(
            "Kafka delivery confirmed: topic=%s partition=%s offset=%s "
            "timestamp_type=%s timestamp_ms=%s message_id=%s",
            msg.topic(),
            msg.partition(),
            msg.offset(),
            timestamp_type,
            timestamp_ms,
            message_id,
        )
