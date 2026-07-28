"""Canonical identity and hash projections for ocean evidence assessments.

Implements deterministic checkpoint IDs, the input-manifest, scientific-content,
and transport-provenance hash projections, and the canonical encoding
they share. Hashes are computed from these explicit projections, never
from generic model dumps, so adding an operational field cannot silently
change a scientific hash.

Canonical encoding rules, built on
``hazard_assessment.ingest.hashing.canonicalize_json``:

- mappings are key-sorted by the JSON serializer;
- semantically unordered collections are sorted by the projection
  functions before encoding; semantically ordered sequences (retained
  record references, message coordinate lists after canonical sort)
  keep their order;
- timestamps are normalized to UTC with fixed microsecond precision;
- non-finite numbers are rejected; floats use CPython's shortest-repr
  encoding, which is deterministic for equal values; and
- each hash field is excluded from its own projection.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hazard_assessment.ingest.hashing import canonicalize_json
from hazard_assessment.schemas.ocean_evidence import (
    OceanEvidenceAssessment,
)

CANONICAL_ENCODING_VERSION = 1
"""Bumped when the normalization or projection layout changes."""


# ---------------------------------------------------------------------------
# Canonical encoding
# ---------------------------------------------------------------------------


def _normalize(value: Any) -> Any:
    """Recursively normalize a value for canonical JSON encoding."""
    if isinstance(value, BaseModel):
        return _normalize(value.model_dump(mode="python"))
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"Canonical mappings require string keys: {key!r}")
            out[key] = _normalize(item)
        return out
    if isinstance(value, (list, tuple)):
        return [_normalize(item) for item in value]
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("Canonical encoding rejects naive datetimes")
        return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, Enum):
        return _normalize(value.value)
    if isinstance(value, bool) or value is None or isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Canonical encoding rejects non-finite float {value!r}")
        return value
    raise TypeError(f"Unsupported type for canonical encoding: {type(value)!r}")


def canonical_bytes(value: Any) -> bytes:
    """Normalize *value* and serialize it to canonical JSON bytes."""
    return canonicalize_json(_normalize(value))


def canonical_sha256(value: Any) -> str:
    """SHA-256 hex digest of the canonical encoding of *value*."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


# ---------------------------------------------------------------------------
# Checkpoint identity
# ---------------------------------------------------------------------------


def derive_live_checkpoint_id(
    consumer_group: str,
    offset_ranges: Sequence[tuple[str, int, int, int]],
    rejected_markers: Sequence[tuple[str, int, int]] = (),
) -> str:
    """Deterministic checkpoint ID for one live Kafka batch.

    ``offset_ranges`` holds (topic, partition, first_offset, last_offset)
    for every consumed range; ``rejected_markers`` holds (topic,
    partition, offset) for messages that failed transport-level
    decoding, so a batch with rejects has a different identity from the
    same batch without them. Both collections are unordered and are
    sorted here.
    """
    if not consumer_group:
        raise ValueError("consumer_group must be nonempty")
    if not offset_ranges:
        raise ValueError("A live checkpoint requires at least one offset range")
    for topic, partition, first, last in offset_ranges:
        if not topic or partition < 0 or first < 0 or last < first:
            raise ValueError(
                f"Invalid offset range: {(topic, partition, first, last)!r}"
            )
    for topic, partition, offset in rejected_markers:
        if not topic or partition < 0 or offset < 0:
            raise ValueError(
                f"Invalid rejected marker: {(topic, partition, offset)!r}"
            )
    projection = {
        "kind": "live_kafka_checkpoint",
        "encoding_version": CANONICAL_ENCODING_VERSION,
        "consumer_group": consumer_group,
        "offset_ranges": sorted(list(r) for r in offset_ranges),
        "rejected_markers": sorted(list(m) for m in rejected_markers),
    }
    return canonical_sha256(projection)


def derive_replay_checkpoint_id(
    replay_manifest_id: str,
    cutoff_sequence: int,
    source_time_cutoff: datetime,
) -> str:
    """Deterministic checkpoint ID for one observation-time replay step."""
    if not replay_manifest_id:
        raise ValueError("replay_manifest_id must be nonempty")
    if cutoff_sequence < 0:
        raise ValueError("cutoff_sequence must be nonnegative")
    projection = {
        "kind": "replay_checkpoint",
        "encoding_version": CANONICAL_ENCODING_VERSION,
        "replay_manifest_id": replay_manifest_id,
        "cutoff_sequence": cutoff_sequence,
        "source_time_cutoff": source_time_cutoff,
    }
    return canonical_sha256(projection)


# ---------------------------------------------------------------------------
# Transport provenance
# ---------------------------------------------------------------------------


class KafkaMessageCoordinate(BaseModel):
    """Transport identity of one consumed or rejected Kafka message."""

    topic: str = Field(min_length=1)
    partition: int = Field(ge=0)
    offset: int = Field(ge=0)
    timestamp_type: str = Field(
        default="", max_length=32,
        description="Kafka timestamp type name, when available",
    )
    timestamp_ms: int | None = None
    application_message_id: str = Field(
        default="", max_length=128,
        description=(
            "Stable application-level message ID when the envelope "
            "carried one; empty for rejected undecodable messages"
        ),
    )
    transport_rejected: bool = False

    model_config = ConfigDict(extra="forbid", frozen=True)


class TransportProvenance(BaseModel):
    """Transport data describing one assessment creation attempt.

    Ordinary Kafka delivery does not provide a retry-attempt number, so
    none is invented. Later redeliveries append checkpoint-attempt audit
    records; they never mutate the creation attempt recorded here.
    """

    run_id: str = Field(min_length=1, max_length=128)
    consumer_group: str = Field(min_length=1, max_length=256)
    messages: list[KafkaMessageCoordinate] = Field(
        description="Sorted by (topic, partition, offset)"
    )

    @model_validator(mode="after")
    def _sorted_messages(self) -> TransportProvenance:
        keys = [(m.topic, m.partition, m.offset) for m in self.messages]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "messages must be sorted and unique by (topic, partition, offset)"
            )
        return self

    model_config = ConfigDict(extra="forbid", frozen=True)


def transport_provenance_hash(transport: TransportProvenance) -> str:
    """Hash the transport data of the creation attempt."""
    projection = {
        "kind": "transport_provenance",
        "encoding_version": CANONICAL_ENCODING_VERSION,
        "run_id": transport.run_id,
        "consumer_group": transport.consumer_group,
        "messages": transport.messages,
    }
    return canonical_sha256(projection)


# ---------------------------------------------------------------------------
# Input manifest projection
# ---------------------------------------------------------------------------


def input_manifest_projection(assessment: OceanEvidenceAssessment) -> dict[str, Any]:
    """Exact scientific inputs and attempt outcomes for one checkpoint.

    Covers trigger and latest admissible seismic revisions, current
    ocean record attempts and outcomes, retained scoring-window records,
    calibration artifacts, detector and threshold configuration, the
    station manifest when used, and code and ruleset versions. Excludes
    the three hash fields and everything random or wall-clock.
    """
    stations = []
    for entry in assessment.stations:
        stations.append(
            {
                "source": entry.source,
                "station_id": entry.station_id,
                "admission_status": entry.admission_status,
                "n_records_attempted": entry.n_records_attempted,
                "n_records_admitted": entry.n_records_admitted,
                "n_records_rejected": entry.n_records_rejected,
                "retained_record_refs": entry.retained_record_refs,
                "calibration_status": entry.calibration_status,
                "calibration_sha256": entry.calibration_sha256,
            }
        )
    return {
        "kind": "ocean_evidence_input_manifest",
        "encoding_version": CANONICAL_ENCODING_VERSION,
        "assessment_schema_version": assessment.assessment_schema_version,
        "condition_registry_version": assessment.condition_registry_version,
        "code_version": assessment.code_version,
        "ruleset_version": assessment.model_version,
        "seismic_context": assessment.seismic_context,
        "station_scope": assessment.station_scope,
        "station_manifest": assessment.station_manifest,
        "detector_config": assessment.detector_config,
        "stations": stations,
    }


def input_manifest_hash(assessment: OceanEvidenceAssessment) -> str:
    """SHA-256 of the input manifest projection."""
    return canonical_sha256(input_manifest_projection(assessment))


# ---------------------------------------------------------------------------
# Scientific content projection
# ---------------------------------------------------------------------------

_SCIENTIFIC_EXCLUDED_TOP_LEVEL = (
    # Random identity and envelope metadata.
    "handoff_id",
    "event_id",  # remintable internal UUID; external identity stays
    "trace_id",
    "contributing_trace_ids",
    "producer",
    "produced_at_utc",
    "decision_trace",
    "input_refs",
    "model_version",  # model metadata is excluded from scientific content
    "schema_version",  # envelope version; assessment_schema_version stays
    # Transport and database references.
    "checkpoint_id",  # derived from Kafka coordinates or replay cutoffs
    "checkpoint_source",
    "fsm_transition_ref",
    # Hash fields are excluded from their own and each other's projections.
    "input_manifest_hash",
    "scientific_content_hash",
    "transport_provenance_hash",
)

_SCIENTIFIC_EXCLUDED_STATION = ("operational_age_at_production_sec",)

# Infrastructure state inside the provenance block. The rest of that block is
# genuine evidence-completeness accounting (how many references were expected,
# resolved, malformed, capped) and belongs in the scientific hash. These two do
# not. `companion_persistence_failures` is assembled from transient QC and
# lineage insert outcomes, including exception class names, so the same
# checkpoint replayed after a storage hiccup would produce a different
# scientific hash and land as a persist conflict rather than a benign
# duplicate: exactly the failure the code_version field description warns
# about. `database_available` is excluded for the same category reason rather
# than an observed one, since the only production call site hardcodes it True;
# it is deployment state, not evidence.
_SCIENTIFIC_EXCLUDED_PROVENANCE = (
    "database_available",
    "companion_persistence_failures",
)


def scientific_content_projection(
    assessment: OceanEvidenceAssessment,
) -> dict[str, Any]:
    """Normalized scientific facts and statuses for one checkpoint.

    Excludes database IDs, random UUIDs, wall-clock production time,
    Kafka coordinates, and model metadata, plus per-station
    operational ages and the two infrastructure fields inside the
    provenance block, all of which depend on the run rather than on the
    evidence.
    """
    dump = assessment.model_dump(mode="python")
    for field in _SCIENTIFIC_EXCLUDED_TOP_LEVEL:
        dump.pop(field, None)
    for station in dump.get("stations", []):
        for field in _SCIENTIFIC_EXCLUDED_STATION:
            station.pop(field, None)
    provenance = dump.get("provenance")
    if isinstance(provenance, dict):
        for field in _SCIENTIFIC_EXCLUDED_PROVENANCE:
            provenance.pop(field, None)
    dump["kind"] = "ocean_evidence_scientific_content"
    dump["encoding_version"] = CANONICAL_ENCODING_VERSION
    return dump


def scientific_content_hash(assessment: OceanEvidenceAssessment) -> str:
    """SHA-256 of the scientific content projection."""
    return canonical_sha256(scientific_content_projection(assessment))


# ---------------------------------------------------------------------------
# Finalization helper
# ---------------------------------------------------------------------------


def finalize_assessment_hashes(
    assessment: OceanEvidenceAssessment,
    transport: TransportProvenance | None,
) -> OceanEvidenceAssessment:
    """Return a copy of *assessment* with its three hash fields set.

    The draft assessment is fully validated with empty hash fields; the
    hashes are then computed from the explicit projections, which
    exclude the hash fields themselves, so the returned artifact hashes
    identically to the draft. ``transport`` is None for replay
    checkpoints, leaving the transport hash empty.
    """
    updates = {
        "input_manifest_hash": input_manifest_hash(assessment),
        "scientific_content_hash": scientific_content_hash(assessment),
    }
    if transport is not None:
        updates["transport_provenance_hash"] = transport_provenance_hash(transport)
    return assessment.model_copy(update=updates)


__all__ = [
    "CANONICAL_ENCODING_VERSION",
    "KafkaMessageCoordinate",
    "TransportProvenance",
    "canonical_bytes",
    "canonical_sha256",
    "derive_live_checkpoint_id",
    "derive_replay_checkpoint_id",
    "finalize_assessment_hashes",
    "input_manifest_hash",
    "input_manifest_projection",
    "scientific_content_hash",
    "scientific_content_projection",
    "transport_provenance_hash",
]
