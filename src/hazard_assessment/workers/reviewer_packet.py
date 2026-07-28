"""Deterministic reviewer packet for ESCALATE checkpoints.

The packet is a pure function of exactly one persisted assessment row:
the OceanEvidenceAssessment payload as stored in ``processed_features``
plus the database row id that stores it. No FSM snapshot, audit query,
or caller-supplied dictionary participates, so rendering the same row
twice yields byte-identical JSON and the canonical content hash is
reproducible from durable storage alone.

The worker renders and persists this packet immediately after the
checkpoint that enters ESCALATE commits its assessment. Review reads
the immutable ``escalation_packets`` row; it never re-assembles
evidence from changing cross-process state.

Hash caveat: ``content_sha256`` is computed at render time over the
canonical Python JSON serialization below. The packet column is JSONB,
which does not preserve key order or numeric formatting bit-for-bit,
so verification against storage must re-serialize with the same
canonical rules (sorted keys, compact separators) rather than hash the
database's own textual rendering.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from hazard_assessment.policy.guardrails import (
    NON_AUTHORITATIVE_DISCLAIMER,
    scan_text,
)

# Bump when the packet structure below changes. The storage key is
# (assessment_row_id, renderer_version), so a new version creates a new
# immutable row instead of mutating an existing packet.
RENDERER_VERSION = "1"

REVIEWER_PACKET_KIND = "escalation_reviewer_packet"


def canonical_packet_hash(packet: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON serialization of the packet.

    ``allow_nan=False`` matches the assessment-payload canonical rules
    (``ingest.hashing.canonicalize_json``): a non-finite float fails here at
    hash time instead of producing a hash for content the JSONB column would
    then reject on insert.
    """
    canonical = json.dumps(
        packet, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def render_reviewer_packet(
    *, assessment_payload: dict[str, Any], assessment_row_id: int
) -> tuple[dict[str, Any], str]:
    """Render the reviewer-visible packet for one persisted assessment.

    ``assessment_payload`` must be the exact JSON payload stored in the
    assessment's ``processed_features`` row (the worker's
    ``model_dump(mode="json")``), and ``assessment_row_id`` that row's
    primary key. Returns ``(packet, content_sha256)``.

    The summary fields are mechanical projections of the payload; the
    full payload rides along unmodified as the evidence of record.
    """
    best: dict[str, Any] | None = None
    for station in assessment_payload.get("stations", []):
        te = station.get("threshold_evaluation")
        if isinstance(te, dict) and te.get("ensemble_score") is not None:
            score = te["ensemble_score"]
            if best is None or score > best["ensemble_score"]:
                best = {
                    "source": station.get("source"),
                    "station_id": station.get("station_id"),
                    "ensemble_score": score,
                }

    packet = {
        "kind": REVIEWER_PACKET_KIND,
        "renderer_version": RENDERER_VERSION,
        "assessment_row_id": assessment_row_id,
        "checkpoint_id": assessment_payload.get("checkpoint_id"),
        "event_id": assessment_payload.get("event_id"),
        "produced_at_utc": assessment_payload.get("produced_at_utc"),
        "fsm_state_before": assessment_payload.get("fsm_state_before"),
        "fsm_state_after": assessment_payload.get("fsm_state_after"),
        "pipeline_outcome": assessment_payload.get("pipeline_outcome"),
        "input_manifest_hash": assessment_payload.get("input_manifest_hash"),
        "scientific_content_hash": assessment_payload.get(
            "scientific_content_hash"
        ),
        "best_scoring_station": best,
        "dart_stations_currently_in_event_mode": assessment_payload.get(
            "dart_stations_currently_in_event_mode", []
        ),
        "seismic_context": assessment_payload.get("seismic_context"),
        "recommended_action": "Human review required",
        "disclaimer": NON_AUTHORITATIVE_DISCLAIMER,
        "assessment": assessment_payload,
    }
    # The packet of record is what Mission Control shows the duty scientist,
    # so it gets the same reserved-language check as the in-memory escalation
    # packet and the ABSTAIN document. Every prose field here is an internal
    # literal or a controlled enum today, so this should never fire; it is a
    # standing check against a future field carrying third-party text, and it
    # runs before hashing so a rejected packet is never persisted.
    prose = " ".join(
        str(packet[field])
        for field in ("recommended_action", "disclaimer")
        if packet.get(field) is not None
    )
    packet_scan = scan_text(prose)
    if packet_scan.violations:
        # ValueError, matching how app.py rejects the in-memory escalation
        # packet. GuardrailScanError lives in the report agent, and importing
        # it here would point workers at an agent module.
        raise ValueError(
            "Reviewer packet contains reserved terminology: "
            + ", ".join(sorted({v.term for v in packet_scan.violations}))
        )
    return packet, canonical_packet_hash(packet)
