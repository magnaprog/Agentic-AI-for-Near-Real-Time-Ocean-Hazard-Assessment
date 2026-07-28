"""Observation record validation and quarantine.

Validates ingest connector records against canonical Pydantic schemas and
quarantines records that fail validation for later review.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import ValidationError

from hazard_assessment.ingest.coops import CoopsRecord
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.ingest.seismic import SeismicEventRecord
from hazard_assessment.schemas.observation import (
    CoopsObservation,
    DartObservation,
    SeismicObservation,
)


class QuarantineReasonCode(StrEnum):
    """Why a record was quarantined."""

    SCHEMA_VALIDATION_FAILED = "schema_validation_failed"


@dataclass(frozen=True, slots=True)
class QuarantinedRecord:
    """A record that failed validation and was quarantined."""

    source_id: str
    source_type: str
    reason_code: QuarantineReasonCode
    reason_detail: str
    quarantined_at: datetime
    raw_fields: dict[str, Any]


Observation = DartObservation | CoopsObservation | SeismicObservation

_DISPATCH = {
    DartRecord: DartObservation,
    CoopsRecord: CoopsObservation,
    SeismicEventRecord: SeismicObservation,
}


def validate_record(
    record: DartRecord | CoopsRecord | SeismicEventRecord,
) -> Observation:
    """Validate a connector record against its canonical schema.

    Returns the validated Pydantic model on success.
    Raises ``ValidationError`` on failure.
    """
    model_cls: type[DartObservation | CoopsObservation | SeismicObservation] | None = (
        _DISPATCH.get(type(record))  # type: ignore[assignment]
    )
    if model_cls is None:
        raise TypeError(f"Unknown record type: {type(record).__name__}")
    return model_cls.model_validate(asdict(record))


def validate_and_quarantine(
    record: DartRecord | CoopsRecord | SeismicEventRecord,
    *,
    now: datetime,
) -> Observation | QuarantinedRecord:
    """Validate a connector record, returning a quarantine entry on failure."""
    try:
        return validate_record(record)
    except (ValidationError, TypeError) as exc:
        source_type = type(record).__name__
        return QuarantinedRecord(
            source_id=getattr(record, "source_id", "<unknown>"),
            source_type=source_type,
            reason_code=QuarantineReasonCode.SCHEMA_VALIDATION_FAILED,
            reason_detail=str(exc),
            quarantined_at=now,
            raw_fields=asdict(record),
        )
