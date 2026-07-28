"""Tests for canonical serialization, checkpoint identity, and hash projections.

Covers deterministic canonical bytes,
live and replay checkpoint identity, transport provenance, and the two
content projections with their exclusion sets. Golden vectors pin the
exact hex digests so any silent change to the canonical encoding or the
projections fails loudly here.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum
from uuid import UUID

import pytest

from hazard_assessment.schemas.ocean_evidence import OceanEvidenceAssessment
from hazard_assessment.schemas.ocean_evidence_hashing import (
    TransportProvenance,
    canonical_bytes,
    derive_live_checkpoint_id,
    derive_replay_checkpoint_id,
    finalize_assessment_hashes,
    input_manifest_hash,
    input_manifest_projection,
    scientific_content_hash,
    scientific_content_projection,
    transport_provenance_hash,
)
from tests.unit._ocean_evidence_fixtures import (
    T0,
    make_assessment,
    make_dart_entry,
    make_transport,
)


class TestCanonicalEncoding:
    def test_map_key_order_is_irrelevant(self) -> None:
        a = {"b": 1, "a": [1, 2], "c": {"y": 2, "x": 1}}
        b = {"c": {"x": 1, "y": 2}, "a": [1, 2], "b": 1}
        assert canonical_bytes(a) == canonical_bytes(b)

    def test_list_order_matters(self) -> None:
        assert canonical_bytes([1, 2]) != canonical_bytes([2, 1])
        assert canonical_bytes((1, 2)) == canonical_bytes([1, 2])

    def test_datetime_normalized_to_utc_microseconds(self) -> None:
        plus_nine = timezone(timedelta(hours=9))
        local = datetime(2011, 3, 11, 14, 46, 24, 500, tzinfo=plus_nine)
        utc = datetime(2011, 3, 11, 5, 46, 24, 500, tzinfo=UTC)
        assert canonical_bytes(local) == canonical_bytes(utc)
        assert b"2011-03-11T05:46:24.000500Z" in canonical_bytes(utc)

    def test_naive_datetime_rejected(self) -> None:
        with pytest.raises(ValueError, match="naive datetimes"):
            canonical_bytes(datetime(2011, 3, 11))

    def test_non_finite_floats_rejected(self) -> None:
        for value in (math.nan, math.inf, -math.inf):
            with pytest.raises(ValueError, match="finite"):
                canonical_bytes({"x": value})

    def test_non_string_keys_rejected(self) -> None:
        with pytest.raises(TypeError, match="string keys"):
            canonical_bytes({1: "x"})

    def test_unsupported_types_rejected(self) -> None:
        with pytest.raises(TypeError, match="Unsupported"):
            canonical_bytes({"x": {1, 2}})
        with pytest.raises(TypeError, match="Unsupported"):
            canonical_bytes(b"raw")

    def test_enum_uuid_normalization(self) -> None:
        class Color(StrEnum):
            RED = "red"

        assert canonical_bytes(Color.RED) == canonical_bytes("red")
        uid = UUID("11111111-1111-4111-8111-111111111111")
        assert canonical_bytes(uid) == canonical_bytes(str(uid))

    def test_bool_and_int_are_distinct(self) -> None:
        assert canonical_bytes(True) != canonical_bytes(1)


class TestCheckpointIdentity:
    def test_live_id_deterministic_and_order_invariant(self) -> None:
        ranges = [("raw.observations", 0, 5, 9), ("raw.observations", 1, 0, 3)]
        first = derive_live_checkpoint_id("g1", ranges)
        again = derive_live_checkpoint_id("g1", list(reversed(ranges)))
        assert first == again
        assert len(first) == 64

    def test_live_id_sensitive_to_every_input(self) -> None:
        base = derive_live_checkpoint_id("g1", [("t", 0, 5, 9)])
        assert derive_live_checkpoint_id("g2", [("t", 0, 5, 9)]) != base
        assert derive_live_checkpoint_id("g1", [("t", 0, 5, 10)]) != base
        assert (
            derive_live_checkpoint_id("g1", [("t", 0, 5, 9)], [("t", 0, 7)]) != base
        )

    def test_rejected_marker_order_invariant(self) -> None:
        markers = [("t", 0, 8), ("t", 0, 7)]
        assert derive_live_checkpoint_id(
            "g1", [("t", 0, 5, 9)], markers
        ) == derive_live_checkpoint_id("g1", [("t", 0, 5, 9)], list(reversed(markers)))

    def test_live_id_input_validation(self) -> None:
        with pytest.raises(ValueError, match="consumer_group"):
            derive_live_checkpoint_id("", [("t", 0, 5, 9)])
        with pytest.raises(ValueError, match="at least one"):
            derive_live_checkpoint_id("g1", [])
        with pytest.raises(ValueError, match="Invalid offset range"):
            derive_live_checkpoint_id("g1", [("t", 0, 9, 5)])
        with pytest.raises(ValueError, match="Invalid offset range"):
            derive_live_checkpoint_id("g1", [("t", -1, 5, 9)])
        with pytest.raises(ValueError, match="Invalid offset range"):
            derive_live_checkpoint_id("g1", [("", 0, 5, 9)])
        with pytest.raises(ValueError, match="Invalid rejected marker"):
            derive_live_checkpoint_id("g1", [("t", 0, 5, 9)], [("t", 0, -1)])

    def test_replay_id_deterministic(self) -> None:
        cutoff = datetime(2011, 3, 11, 11, 46, 24, tzinfo=UTC)
        first = derive_replay_checkpoint_id("manifest-1", 42, cutoff)
        assert first == derive_replay_checkpoint_id("manifest-1", 42, cutoff)
        assert first != derive_replay_checkpoint_id("manifest-1", 43, cutoff)
        assert first != derive_replay_checkpoint_id("manifest-2", 42, cutoff)

    def test_replay_id_input_validation(self) -> None:
        cutoff = datetime(2011, 3, 11, tzinfo=UTC)
        with pytest.raises(ValueError, match="manifest"):
            derive_replay_checkpoint_id("", 42, cutoff)
        with pytest.raises(ValueError, match="nonnegative"):
            derive_replay_checkpoint_id("m", -1, cutoff)
        with pytest.raises(ValueError, match="naive datetimes"):
            derive_replay_checkpoint_id("m", 42, datetime(2011, 3, 11))

    def test_live_and_replay_kinds_do_not_collide(self) -> None:
        # The domain-separation kind field keeps the two derivations in
        # disjoint hash domains even for crafted inputs.
        live = derive_live_checkpoint_id("g", [("t", 0, 0, 0)])
        replay = derive_replay_checkpoint_id("g", 0, T0)
        assert live != replay

    def test_golden_checkpoint_vectors(self) -> None:
        assert derive_live_checkpoint_id(
            "g1", [("raw.observations", 0, 5, 9)], [("raw.observations", 0, 7)]
        ) == ("36e6125be7f67717be1e131f4cda5a6160b52bc0f3a3d6122525fc479da67882")
        assert derive_replay_checkpoint_id(
            "m1", 42, datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)
        ) == ("577b5bac85900489f99457a5419a716e6919582583041948cd0153656ff5020a")


class TestTransportProvenance:
    def test_messages_must_be_sorted_unique(self) -> None:
        msgs = make_transport().messages
        with pytest.raises(Exception, match="sorted and unique"):
            TransportProvenance(
                run_id="r",
                consumer_group="g",
                messages=list(reversed(msgs)),
            )
        with pytest.raises(Exception, match="sorted and unique"):
            TransportProvenance(
                run_id="r", consumer_group="g", messages=[msgs[0], msgs[0]]
            )

    def test_hash_sensitive_to_coordinates(self) -> None:
        base = transport_provenance_hash(make_transport())
        moved = TransportProvenance(
            run_id="run-1",
            consumer_group="hazard-pipeline",
            messages=[
                make_transport().messages[0].model_copy(update={"offset": 99}),
                make_transport().messages[1],
            ],
        )
        assert transport_provenance_hash(moved) != base

    def test_golden_transport_vector(self) -> None:
        assert transport_provenance_hash(make_transport()) == (
            "2187b625cdee24b387f6b12db277e9546ea9870d1a9a6f485fd7cfb4c6d37e71"
        )


class TestProjections:
    def test_operational_fields_do_not_move_content_hashes(self) -> None:
        base = make_assessment()
        moved = make_assessment(
            handoff_id=UUID("99999999-9999-4999-8999-999999999999"),
            trace_id=None,
            event_id=UUID("88888888-8888-4888-8888-888888888888"),
            produced_at_utc=T0 + timedelta(hours=3),
            fsm_transition_ref="audit-999999",
            contributing_trace_ids=[],
            checkpoint_id="0" * 64,
        )
        assert scientific_content_hash(moved) == scientific_content_hash(base)
        assert input_manifest_hash(moved) == input_manifest_hash(base)

    def test_station_operational_age_excluded_from_content(self) -> None:
        base = make_assessment()
        aged = make_assessment(
            stations=[
                make_assessment().stations[0],
                make_dart_entry(operational_age_at_production_sec=9999.0),
            ]
        )
        assert scientific_content_hash(aged) == scientific_content_hash(base)
        assert input_manifest_hash(aged) == input_manifest_hash(base)

    def test_model_version_moves_manifest_but_not_science(self) -> None:
        base = make_assessment()
        bumped = make_assessment(model_version="ruleset-2")
        assert input_manifest_hash(bumped) != input_manifest_hash(base)
        assert scientific_content_hash(bumped) == scientific_content_hash(base)

    def test_code_version_moves_both_projections(self) -> None:
        # Deliberately conservative: a code bump makes replayed
        # content differ so idempotency conflict detection fires.
        base = make_assessment()
        bumped = make_assessment(code_version="feedc0de")
        assert scientific_content_hash(bumped) != scientific_content_hash(base)
        assert input_manifest_hash(bumped) != input_manifest_hash(base)

    def test_scientific_change_moves_science_hash(self) -> None:
        base = make_assessment()
        entry = make_dart_entry()
        te = entry.threshold_evaluation
        assert te is not None
        rescored = make_assessment(
            stations=[
                make_assessment().stations[0],
                make_dart_entry(
                    threshold_evaluation=te.model_copy(
                        update={"ensemble_score": 0.73}
                    )
                ),
            ]
        )
        assert scientific_content_hash(rescored) != scientific_content_hash(base)
        # Detector outputs are results, not inputs.
        assert input_manifest_hash(rescored) == input_manifest_hash(base)

    def test_projection_kind_fields_differ(self) -> None:
        a = make_assessment()
        assert input_manifest_projection(a)["kind"] != (
            scientific_content_projection(a)["kind"]
        )

    def test_golden_projection_vectors(self) -> None:
        a = make_assessment()
        assert input_manifest_hash(a) == (
            "7da7b962e7e933a87b7ef6250195cdc033ca6e6509f8fee2fbfd57ea1f9c9c92"
        )
        assert scientific_content_hash(a) == (
            "679a19788d8e38f122afaa1eeb0eb61f580cd20e3a75eef2abdd43fef784baee"
        )


    def test_infrastructure_provenance_is_outside_the_scientific_hash(self) -> None:
        """A storage hiccup must not change what the evidence hashes to.

        `database_available` describes the deployment and
        `companion_persistence_failures` is assembled from transient QC and
        lineage insert outcomes. Hashing either one means the same checkpoint,
        replayed after a failed companion write, produces a different
        scientific hash and is recorded as a persist conflict instead of a
        benign duplicate.
        """
        base = make_assessment()
        degraded = base.model_copy(
            update={
                "provenance": base.provenance.model_copy(
                    update={
                        "database_available": not base.provenance.database_available,
                        "companion_persistence_failures": [
                            "qc_report:dart:21418: OperationalError"
                        ],
                    }
                )
            }
        )
        assert scientific_content_hash(degraded) == scientific_content_hash(base)

    def test_evidence_completeness_provenance_still_hashes(self) -> None:
        """The rest of the provenance block is evidence, and must stay in."""
        base = make_assessment()
        altered = base.model_copy(
            update={
                "provenance": base.provenance.model_copy(
                    update={
                        "n_unresolved_raw_records": (
                            base.provenance.n_unresolved_raw_records + 1
                        )
                    }
                )
            }
        )
        assert scientific_content_hash(altered) != scientific_content_hash(base)


class TestFinalization:
    def test_finalize_sets_self_consistent_hashes(self) -> None:
        final = finalize_assessment_hashes(make_assessment(), make_transport())
        # Self-exclusion: each hash field is excluded from its own
        # projection, so recomputing on the finalized artifact matches.
        assert final.input_manifest_hash == input_manifest_hash(final)
        assert final.scientific_content_hash == scientific_content_hash(final)
        assert final.transport_provenance_hash == transport_provenance_hash(
            make_transport()
        )

    def test_finalize_without_transport_leaves_field_empty(self) -> None:
        final = finalize_assessment_hashes(make_assessment(), None)
        assert final.transport_provenance_hash == ""
        assert final.input_manifest_hash != ""

    def test_finalized_artifact_revalidates_and_roundtrips(self) -> None:
        final = finalize_assessment_hashes(make_assessment(), make_transport())
        restored = OceanEvidenceAssessment.model_validate_json(
            final.model_dump_json()
        )
        assert restored == final

    def test_hash_fields_do_not_feed_content_hashes(self) -> None:
        plain = make_assessment()
        final = finalize_assessment_hashes(plain, make_transport())
        assert scientific_content_hash(final) == scientific_content_hash(plain)
        assert input_manifest_hash(final) == input_manifest_hash(plain)
