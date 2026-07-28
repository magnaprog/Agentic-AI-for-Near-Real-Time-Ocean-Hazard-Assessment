"""Run a single ingest connector as a long-lived service.

Usage::

    python -m hazard_assessment.workers.ingest_runner --connector dart
    python -m hazard_assessment.workers.ingest_runner --connector coops
    python -m hazard_assessment.workers.ingest_runner --connector seismic

Each connector polls its upstream data source and emits records via a
callback. Records are published to Kafka ``raw.observations`` topic
(when configured) and optionally written to TimescaleDB.

"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import signal
from datetime import UTC, datetime
from typing import Any

from hazard_assessment.ingest.validation import (
    QuarantinedRecord,
    validate_and_quarantine,
)
from hazard_assessment.messaging.producer import KafkaProducer
from hazard_assessment.telemetry.metrics import record_ingest

logger = logging.getLogger("hazard_assessment.workers.ingest")

CONNECTORS = ("dart", "coops", "seismic")
RAW_OBSERVATIONS_TOPIC = "raw.observations"


def _build_connector(name: str) -> Any:
    """Instantiate the connector for *name* with default settings."""
    if name == "dart":
        from hazard_assessment.ingest.dart import DartIngestConnector

        return DartIngestConnector()
    if name == "coops":
        from hazard_assessment.ingest.coops import CoopsIngestConnector

        return CoopsIngestConnector()
    if name == "seismic":
        from hazard_assessment.ingest.seismic import SeismicIngestConnector

        return SeismicIngestConnector()
    raise ValueError(f"Unknown connector: {name}")


def _make_emit(
    connector_name: str,
    producer: KafkaProducer,
    db_client: Any | None = None,
) -> Any:
    """Build an emit callback that validates, persists, and publishes records.

    Each record is validated against its canonical schema; records that fail
    are quarantined (logged with a reason code) and neither persisted nor
    published.  Valid records are written to the raw-observation archive (when
    a database is configured) and published to Kafka.
    """

    def _emit(record: Any) -> None:
        source_id = getattr(record, "source_id", "?")
        source_ts = getattr(record, "source_timestamp", None)
        payload_hash = getattr(record, "payload_sha256", "?")

        logger.info(
            "record  source_id=%s  ts=%s  hash=%s",
            source_id,
            source_ts,
            str(payload_hash)[:12],
        )

        # Canonical schema validation: quarantine invalid records so
        # they never reach persistence or the processing pipeline.
        validated = validate_and_quarantine(record, now=datetime.now(UTC))
        if isinstance(validated, QuarantinedRecord):
            logger.warning(
                "QUARANTINED %s record source_id=%s: %s - %s",
                validated.source_type,
                validated.source_id,
                validated.reason_code.value,
                validated.reason_detail,
            )
            # Durable quarantine sink (best-effort, DB-optional).
            if db_client is not None:
                try:
                    db_client.insert_quarantined_record(validated)
                except Exception:
                    logger.exception("Failed to persist quarantined record")
            record_ingest("quarantined")
            return

        # Seismic events identify by event_id; the raw_observations table is
        # keyed by station_id, so map event_id -> station_id (documented on
        # schemas/observation.py SeismicObservation).
        station_id = str(
            getattr(record, "station_id", None)
            or getattr(record, "event_id", source_id)
        )
        if hasattr(record, "model_dump"):
            record_dict = record.model_dump(mode="json")
        elif dataclasses.is_dataclass(record) and not isinstance(record, type):
            record_dict = dataclasses.asdict(record)
        else:
            record_dict = {}

        # Raw observation persistence: provenance archive.
        # No-op when no database is configured.
        if db_client is not None:
            try:
                # raw_payload is JSONB and insert_observations json-encodes it,
                # so pass a JSON-safe dict (datetimes rendered as strings).
                payload = json.loads(json.dumps(record_dict, default=str))
                db_client.insert_observations(
                    [{
                        "station_id": station_id,
                        "observed_at": source_ts,
                        "payload": payload,
                        "payload_hash": str(payload_hash),
                    }],
                    source_type=connector_name,
                )
            except Exception:
                logger.exception("Failed to persist raw observation to database")

        # Publish to Kafka if connected.
        if producer.is_connected:
            key = f"{connector_name}:{station_id}"
            try:
                producer.produce(
                    topic=RAW_OBSERVATIONS_TOPIC,
                    key=key,
                    value={"source_type": connector_name, "record": record_dict},
                    headers={"payload_hash": str(payload_hash)},
                    # Stable application message identity for delivery-callback
                    # correlation and transport provenance. Connector
                    # source_ids are deterministic and already source-prefixed.
                    message_id=str(source_id),
                )
            except Exception:
                logger.exception("Failed to publish record to Kafka")

        # 'accepted' = validated + processed (persisted/queued). Kafka delivery
        # is best-effort and not separately confirmed here.
        record_ingest("accepted")

    return _emit


def _make_db_client() -> Any | None:
    """Create an ingest-writer DB client when DB_HOST is configured, else None."""
    db_host = os.getenv("DB_HOST", "").strip()
    if not db_host:
        return None
    try:
        from hazard_assessment.storage.client import ClientConfig, DatabaseClient

        client = DatabaseClient(ClientConfig.from_env(role="ingest_writer"))
        if not client.is_connected:
            return None
    except Exception:
        logger.exception("Failed to connect to database")
        return None
    return client


async def _run(connector_name: str) -> None:
    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, stop.set)

    producer = KafkaProducer(client_id=f"ingest-{connector_name}")
    # Optional raw-observation persistence (no-op without DB_HOST).
    db_client = _make_db_client()
    emit = _make_emit(connector_name, producer, db_client)

    connector = _build_connector(connector_name)
    logger.info("Starting %s ingest connector", connector_name)
    try:
        await connector.run(emit=emit, stop_event=stop)
    finally:
        await connector.close()
        producer.close()
        if db_client is not None:
            db_client.close()
    logger.info("Connector %s shut down cleanly", connector_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an ingest connector")
    parser.add_argument(
        "--connector",
        required=True,
        choices=CONNECTORS,
        help="Which connector to run",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=os.environ.get("APP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )
    metrics_port = os.getenv("METRICS_PORT", "").strip()
    if metrics_port.isdigit():
        from hazard_assessment.telemetry.metrics import start_metrics_exporter
        start_metrics_exporter(int(metrics_port))
    asyncio.run(_run(args.connector))


if __name__ == "__main__":
    main()
