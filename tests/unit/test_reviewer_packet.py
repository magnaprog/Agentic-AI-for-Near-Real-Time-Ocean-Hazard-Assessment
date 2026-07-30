"""Tests for the deterministic reviewer packet renderer."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER
from hazard_assessment.workers.reviewer_packet import (
    RENDERER_VERSION,
    REVIEWER_PACKET_KIND,
    canonical_packet_hash,
    render_reviewer_packet,
)

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def _payload(**overrides: Any) -> dict[str, Any]:
    """Minimal assessment payload shaped like model_dump(mode='json')."""
    base: dict[str, Any] = {
        "checkpoint_id": "ckpt-42",
        "event_id": "6f1c1c1c-0000-4000-8000-000000000001",
        "produced_at_utc": "2026-07-17T00:00:00+00:00",
        "fsm_state_before": "ASSESS",
        "fsm_state_after": "ESCALATE",
        "pipeline_outcome": "ABSTAIN",
        "input_manifest_hash": "a" * 64,
        "scientific_content_hash": "b" * 64,
        "dart_stations_currently_in_event_mode": ["21418"],
        "seismic_context": {"magnitude": 9.0, "region": "japan_trench"},
        "stations": [
            {
                "source": "dart",
                "station_id": "21418",
                "threshold_evaluation": {"ensemble_score": 0.91},
            },
            {
                "source": "coops",
                "station_id": "1611400",
                "threshold_evaluation": {"ensemble_score": 0.4},
            },
            {
                "source": "dart",
                "station_id": "21413",
                "threshold_evaluation": None,
            },
        ],
    }
    base.update(overrides)
    return base


class TestRenderDeterminism:
    def test_two_renders_are_byte_identical(self) -> None:
        p1, h1 = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        p2, h2 = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        assert h1 == h2
        assert json.dumps(p1, sort_keys=True) == json.dumps(p2, sort_keys=True)

    def test_hash_matches_canonical_recompute(self) -> None:
        packet, digest = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        canonical = json.dumps(packet, sort_keys=True, separators=(",", ":"))
        assert digest == hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        assert digest == canonical_packet_hash(packet)
        assert _HEX64.match(digest)

    def test_row_id_change_changes_hash(self) -> None:
        _, h7 = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        _, h8 = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=8
        )
        assert h7 != h8

    def test_payload_change_changes_hash(self) -> None:
        _, h1 = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        _, h2 = render_reviewer_packet(
            assessment_payload=_payload(pipeline_outcome="COMPLETED"),
            assessment_row_id=7,
        )
        assert h1 != h2


class TestPacketContent:
    def test_identity_and_evidence_fields(self) -> None:
        payload = _payload()
        packet, _ = render_reviewer_packet(
            assessment_payload=payload, assessment_row_id=7
        )
        assert packet["kind"] == REVIEWER_PACKET_KIND
        assert packet["renderer_version"] == RENDERER_VERSION
        assert packet["assessment_row_id"] == 7
        assert packet["checkpoint_id"] == "ckpt-42"
        assert packet["event_id"] == payload["event_id"]
        assert packet["fsm_state_before"] == "ASSESS"
        assert packet["fsm_state_after"] == "ESCALATE"
        assert packet["input_manifest_hash"] == "a" * 64
        assert packet["scientific_content_hash"] == "b" * 64
        # The full payload rides along unmodified as evidence of record.
        #
        # Asserting equality alone proves nothing here: the builder stores the
        # caller's dict by reference, so this compares an object with itself
        # and holds even if the payload handed in were gutted upstream. Name
        # the fields a reviewer needs, so a narrowed payload fails here rather
        # than only in whichever unrelated test happens to index one of them.
        assert packet["assessment"] is payload
        for required in (
            "event_id",
            "checkpoint_id",
            "stations",
            "scientific_content_hash",
            "input_manifest_hash",
            "seismic_context",
        ):
            assert required in packet["assessment"], (
                f"the packet of record dropped {required!r}; a duty scientist "
                "reviews this document and cannot see what is missing from it"
            )
        assert packet["assessment"]["stations"], "no station evidence in the packet"
        assert packet["disclaimer"] == NON_AUTHORITATIVE_DISCLAIMER
        assert packet["recommended_action"] == "Human review required"

    def test_best_scoring_station_is_max_over_scored(self) -> None:
        packet, _ = render_reviewer_packet(
            assessment_payload=_payload(), assessment_row_id=7
        )
        assert packet["best_scoring_station"] == {
            "source": "dart",
            "station_id": "21418",
            "ensemble_score": 0.91,
        }

    def test_best_scoring_station_none_when_nothing_scored(self) -> None:
        payload = _payload(
            stations=[
                {
                    "source": "dart",
                    "station_id": "21413",
                    "threshold_evaluation": None,
                }
            ]
        )
        packet, _ = render_reviewer_packet(
            assessment_payload=payload, assessment_row_id=7
        )
        assert packet["best_scoring_station"] is None
