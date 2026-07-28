"""Shared durable reviewer-packet fixtures for API and safety tests."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

from hazard_assessment.audit.logger import AuditEntry, AuditLogger
from hazard_assessment.workers.reviewer_packet import render_reviewer_packet


class DurableReviewDb:
    is_connected = True

    def __init__(self, row: dict[str, Any] | None) -> None:
        self.row = row
        self.closed = False
        self.audit_entries: list[AuditEntry] = []

    def load_fsm_state(self) -> None:
        return None

    def get_escalation_packet_for_event(
        self, event_id: Any
    ) -> dict[str, Any] | None:
        if self.row is None or str(self.row["event_id"]) != str(event_id):
            return None
        return self.row

    def append_audit(self, entry: AuditEntry) -> bool:
        self.audit_entries.append(entry.model_copy(deep=True))
        return True

    def query_audit(self, **kwargs: Any) -> list[dict[str, Any]]:
        entries = self.audit_entries
        for field in ("event_id", "event_type", "trace_id", "producer"):
            value = kwargs.get(field)
            if value is not None:
                entries = [entry for entry in entries if getattr(entry, field) == value]
        entries = list(reversed(entries))
        offset = int(kwargs.get("offset", 0))
        limit = int(kwargs.get("limit", 100))
        return [
            {
                "id": index,
                "event_id": entry.event_id,
                "trace_id": entry.trace_id,
                "recorded_at": entry.timestamp_utc,
                "agent_name": entry.producer,
                "action": entry.event_type,
                "metadata": entry.data,
            }
            for index, entry in enumerate(entries[offset : offset + limit], start=1)
        ]

    def close(self) -> None:
        self.closed = True


def install_durable_review_packet(
    app_module: Any,
    event_id: str,
    *,
    packet_row_id: int = 3,
    assessment_row_id: int = 41,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Install one canonical packet row and return request fields plus row."""
    assessment_id = uuid4()
    scientific_hash = "a" * 64
    assessment = {
        "handoff_id": str(assessment_id),
        "event_id": event_id,
        "checkpoint_id": "b" * 64,
        "produced_at_utc": "2026-07-19T00:00:00+00:00",
        "fsm_state_before": "ASSESS",
        "fsm_state_after": "ESCALATE",
        "pipeline_outcome": "ABSTAIN",
        "input_manifest_hash": "c" * 64,
        "scientific_content_hash": scientific_hash,
        "stations": [],
        "dart_stations_currently_in_event_mode": [],
    }
    packet, content_sha256 = render_reviewer_packet(
        assessment_payload=assessment,
        assessment_row_id=assessment_row_id,
    )
    row = {
        "id": packet_row_id,
        "assessment_row_id": assessment_row_id,
        "event_id": event_id,
        "renderer_version": "1",
        "content_sha256": content_sha256,
        "packet": packet,
    }
    db = DurableReviewDb(row)
    app_module._db_client = db
    app_module._audit = AuditLogger(db_client=db)
    request_fields = {
        "escalation_packet_row_id": packet_row_id,
        "escalation_packet_hash": content_sha256,
    }
    return request_fields, row
