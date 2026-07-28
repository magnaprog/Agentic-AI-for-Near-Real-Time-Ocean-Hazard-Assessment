"""Tests for ingest_runner emit wiring: canonical validation/quarantine and
raw-observation persistence (workers/ingest_runner.py).

Before this wiring, ``_make_emit`` only logged and published to Kafka, so
``validate_and_quarantine`` and ``DatabaseClient.insert_observations`` were
never exercised on the live ingest path.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.workers.ingest_runner import _make_emit


class _StubProducer:
    """Records what would be published, without a real Kafka broker."""

    is_connected = True

    def __init__(self) -> None:
        self.produced: list[dict[str, Any]] = []

    def produce(
        self,
        *,
        topic: str,
        key: str,
        value: dict[str, Any],
        headers: dict[str, str] | None = None,
        message_id: str | None = None,
    ) -> bool:
        self.produced.append({
            "topic": topic,
            "key": key,
            "value": value,
            "message_id": message_id,
        })
        return True


class _StubDb:
    """Records what would be persisted, without a real database."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, dict[str, Any]]] = []
        self.quarantined: list[Any] = []

    def insert_observations(
        self, records: list[dict[str, Any]], source_type: str
    ) -> int:
        self.rows.extend((source_type, r) for r in records)
        return len(records)

    def insert_quarantined_record(self, record: Any) -> bool:
        self.quarantined.append(record)
        return True


def _dart(payload_hash: str) -> DartRecord:
    return DartRecord(
        source_id="ndbc:21413",
        station_id="21413",
        source_timestamp=datetime(2011, 3, 11, 6, 0, tzinfo=UTC),
        ingest_timestamp=datetime(2011, 3, 11, 6, 1, tzinfo=UTC),
        measurement_type=2,
        height_m=5827.5,
        event_mode=True,
        payload_sha256=payload_hash,
    )


class TestEmitValidationAndPersistence:
    def test_valid_record_is_persisted_and_published(self) -> None:
        producer = _StubProducer()
        db = _StubDb()
        emit = _make_emit("dart", producer, db)

        emit(_dart("a" * 64))

        # Published to Kafka with the record nested under "record".
        assert len(producer.produced) == 1
        published = producer.produced[0]
        assert published["key"] == "dart:21413"
        assert published["value"]["record"]["height_m"] == 5827.5
        # Stable application message identity for delivery correlation.
        assert published["message_id"] == "ndbc:21413"

        # Persisted to the raw-observation archive.
        assert len(db.rows) == 1
        source_type, row = db.rows[0]
        assert source_type == "dart"
        assert row["station_id"] == "21413"
        assert row["payload_hash"] == "a" * 64
        assert row["payload"]["height_m"] == 5827.5

    def test_invalid_record_is_quarantined(self) -> None:
        """An empty payload_sha256 fails the canonical schema's 64-hex hash
        pattern, so the record is quarantined and neither persisted nor
        published."""
        producer = _StubProducer()
        db = _StubDb()
        emit = _make_emit("dart", producer, db)

        emit(_dart(""))  # invalid hash -> quarantined

        assert producer.produced == []
        assert db.rows == []
        # Quarantined to the durable sink with a reason code.
        assert len(db.quarantined) == 1
        assert db.quarantined[0].reason_code.value == "schema_validation_failed"

    def test_persistence_is_optional(self) -> None:
        """With no db_client, valid records are still published (DB no-op)."""
        producer = _StubProducer()
        emit = _make_emit("dart", producer, None)

        emit(_dart("b" * 64))

        assert len(producer.produced) == 1
