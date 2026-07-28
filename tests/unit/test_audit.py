"""Unit tests for the audit logger.

Validates append-only semantics, filtering, and defensive copy behavior.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

import pytest
from pydantic import ValidationError

from hazard_assessment.audit.logger import (
    AuditEntry,
    AuditLogger,
    _row_to_audit_entry,
)


class TestAuditLogger:
    def test_append_and_count(self) -> None:
        logger = AuditLogger()
        assert logger.count == 0

        entry = AuditEntry(
            event_type="state_transition",
            producer="orchestrator",
            data={"from": "IDLE", "to": "MONITOR"},
        )
        logger.append(entry)
        assert logger.count == 1

    @pytest.mark.parametrize("result", [False, None])
    def test_append_durable_rejects_unconfirmed_write(self, result: object) -> None:
        class _Db:
            def append_audit(self, _entry: AuditEntry) -> object:
                return result

        logger = AuditLogger(db_client=_Db())  # type: ignore[arg-type]

        assert logger.append_durable(AuditEntry(event_type="review", producer="api")) is False
        assert logger.count == 0

    def test_append_durable_records_confirmed_write(self) -> None:
        class _Db:
            def append_audit(self, _entry: AuditEntry) -> bool:
                return True

        logger = AuditLogger(db_client=_Db())  # type: ignore[arg-type]

        assert logger.append_durable(AuditEntry(event_type="review", producer="api")) is True
        assert logger.count == 1

    def test_get_entries_returns_all(self) -> None:
        logger = AuditLogger()
        for i in range(3):
            logger.append(
                AuditEntry(
                    event_type="handoff",
                    producer=f"agent_{i}",
                )
            )
        entries = logger.get_entries()
        assert len(entries) == 3

    def test_filter_by_event_id(self) -> None:
        logger = AuditLogger()
        event_id = uuid4()
        other_id = uuid4()

        logger.append(
            AuditEntry(event_id=event_id, event_type="handoff", producer="a")
        )
        logger.append(
            AuditEntry(event_id=other_id, event_type="handoff", producer="b")
        )
        logger.append(
            AuditEntry(event_id=event_id, event_type="transition", producer="c")
        )

        filtered = logger.get_entries(event_id=event_id)
        assert len(filtered) == 2
        assert all(e.event_id == event_id for e in filtered)

    def test_filter_by_event_type(self) -> None:
        logger = AuditLogger()
        logger.append(AuditEntry(event_type="handoff", producer="a"))
        logger.append(AuditEntry(event_type="state_transition", producer="b"))
        logger.append(AuditEntry(event_type="handoff", producer="c"))

        filtered = logger.get_entries(event_type="handoff")
        assert len(filtered) == 2
        assert all(e.event_type == "handoff" for e in filtered)

    def test_combined_filters(self) -> None:
        logger = AuditLogger()
        event_id = uuid4()

        logger.append(
            AuditEntry(event_id=event_id, event_type="handoff", producer="a")
        )
        logger.append(
            AuditEntry(event_id=event_id, event_type="transition", producer="b")
        )
        logger.append(
            AuditEntry(event_id=uuid4(), event_type="handoff", producer="c")
        )

        filtered = logger.get_entries(event_id=event_id, event_type="handoff")
        assert len(filtered) == 1
        assert filtered[0].producer == "a"


class TestAuditDefensiveCopy:
    """Verify that get_entries returns a defensive copy."""

    def test_mutation_does_not_affect_internal_buffer(self) -> None:
        logger = AuditLogger()
        logger.append(AuditEntry(event_type="test", producer="test"))

        entries = logger.get_entries()
        entries.clear()  # Mutate the returned list

        # Internal buffer should be unaffected
        assert logger.count == 1
        assert len(logger.get_entries()) == 1

    def test_append_after_get_does_not_affect_previous_result(self) -> None:
        logger = AuditLogger()
        logger.append(AuditEntry(event_type="first", producer="a"))

        snapshot = logger.get_entries()
        assert len(snapshot) == 1

        logger.append(AuditEntry(event_type="second", producer="b"))

        # The snapshot should not have been mutated
        assert len(snapshot) == 1
        # But a new call should see both
        assert len(logger.get_entries()) == 2


class TestAuditSnapshot:
    """Verify snapshot() ordering and scoping (it reads through query_entries
    so durable worker entries are included when a DB is configured)."""

    def test_snapshot_preserves_chronological_order(self) -> None:
        logger = AuditLogger()
        logger.append(AuditEntry(event_type="first", producer="a"))
        logger.append(AuditEntry(event_type="second", producer="b"))
        logger.append(AuditEntry(event_type="third", producer="c"))

        snap = logger.snapshot()
        types = [e.event_type for e in snap.get_entries()]
        assert types == ["first", "second", "third"]

    def test_snapshot_scopes_to_event_id(self) -> None:
        from uuid import uuid4

        logger = AuditLogger()
        event_id = uuid4()
        logger.append(AuditEntry(event_id=event_id, event_type="mine", producer="a"))
        logger.append(AuditEntry(event_id=uuid4(), event_type="other", producer="b"))

        snap = logger.snapshot(event_id=event_id)
        assert [e.event_type for e in snap.get_entries()] == ["mine"]


class TestAuditAppendDeepCopy:
    """Verify that append stores a deep copy, preventing post-hoc mutation."""

    def test_caller_mutation_does_not_affect_persisted_entry(self) -> None:
        logger = AuditLogger()
        data = {"key": "original"}
        entry = AuditEntry(event_type="test", producer="test", data=data)
        logger.append(entry)

        # Mutate the original dict after appending
        data["key"] = "mutated"

        # Persisted entry should retain original value
        persisted = logger.get_entries()[0]
        assert persisted.data["key"] == "original"

    def test_entry_object_mutation_does_not_affect_persisted(self) -> None:
        logger = AuditLogger()
        entry = AuditEntry(
            event_type="test",
            producer="test",
            data={"nested": {"value": 1}},
        )
        logger.append(entry)

        # Mutate the entry's data dict after appending
        entry.data["nested"]["value"] = 999

        # Persisted entry should be unaffected
        persisted = logger.get_entries()[0]
        assert persisted.data["nested"]["value"] == 1


class TestGetEntriesDeepCopy:
    """Verify that get_entries returns deep copies, preventing corruption via returned refs."""

    def test_mutation_via_returned_entry_does_not_affect_buffer(self) -> None:
        logger = AuditLogger()
        logger.append(
            AuditEntry(event_type="test", producer="test", data={"key": "original"})
        )

        # Mutate the data dict through the returned entry reference
        entries = logger.get_entries()
        entries[0].data["key"] = "corrupted"

        # Internal buffer must be unaffected
        fresh = logger.get_entries()
        assert fresh[0].data["key"] == "original"


class TestAuditEntryTimezoneValidation:
    """Verify AuditEntry rejects naive datetimes for timestamp_utc."""

    def test_naive_timestamp_rejected(self) -> None:
        naive = datetime(2026, 2, 27, 1, 30, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            AuditEntry(
                event_type="test",
                producer="test",
                timestamp_utc=naive,
            )


class TestAuditEntryImmutability:
    """Verify AuditEntry rejects unknown fields."""

    def test_rejects_extra_fields(self) -> None:
        with pytest.raises(ValidationError):
            AuditEntry(
                event_type="test",
                producer="test",
                rogue_field="bad",
            )


class TestAuditWriterIntegration:
    """Verify that AuditLogger satisfies the AuditWriter protocol
    and integrates correctly with FSMOrchestrator."""

    def test_fsm_transition_appears_in_audit_log(self) -> None:
        from hazard_assessment.orchestrator.states import (
            FSMOrchestrator,
            SystemState,
        )

        logger = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=logger)

        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        assert fsm.state == SystemState.MONITOR
        assert logger.count == 1

        entries = logger.get_entries(event_type="state_transition")
        assert len(entries) == 1
        assert entries[0].data["from_state"] == "IDLE"
        assert entries[0].data["to_state"] == "MONITOR"
        assert entries[0].event_id is not None

    def test_multiple_transitions_all_audited(self) -> None:
        from hazard_assessment.orchestrator.states import FSMOrchestrator

        logger = AuditLogger()
        fsm = FSMOrchestrator(audit_writer=logger)

        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        fsm.evaluate_anomaly_score(0.40)  # MONITOR -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # INVESTIGATE -> ASSESS

        assert logger.count == 3
        entries = logger.get_entries()
        states = [(e.data["from_state"], e.data["to_state"]) for e in entries]
        assert states == [
            ("IDLE", "MONITOR"),
            ("MONITOR", "INVESTIGATE"),
            ("INVESTIGATE", "ASSESS"),
        ]

    def test_audit_failure_aborts_transition(self) -> None:
        """If the audit writer raises, the FSM state must not change."""
        from hazard_assessment.orchestrator.states import (
            FSMOrchestrator,
            SystemState,
        )

        class FailingWriter:
            def write_transition(self, record: object) -> None:
                raise RuntimeError("Audit write failed")

        fsm = FSMOrchestrator(audit_writer=FailingWriter())  # type: ignore[arg-type]

        with pytest.raises(RuntimeError, match="Audit write failed"):
            fsm.evaluate_seismic_trigger(
                magnitude=7.0,
                region="pacific_rim",
                epicenter_lat=38.3,
                epicenter_lon=142.4,
                tsunamigenic_zones={"pacific_rim"},
            )

        # FSM must still be in IDLE - transition was aborted
        assert fsm.state == SystemState.IDLE
        # event_context must also be restored (no phantom active event)
        assert fsm.event_context is None


class TestQueryEntriesWindow:
    """query_entries must return the most-recent ``limit`` entries on the
    in-memory path, matching the DB branch's ORDER BY recorded_at DESC LIMIT
    window so the two deployment configs cannot diverge for the same data."""

    def test_in_memory_returns_newest_limit_not_oldest(self) -> None:
        logger = AuditLogger()
        tid = uuid4()
        for i in range(10):
            logger.append(
                AuditEntry(
                    event_type="handoff",
                    producer=f"p{i}",
                    trace_id=tid,
                    data={"i": i},
                )
            )

        # query_entries (recent contract) returns the newest three,
        # newest-first, mirroring the DB branch's ORDER BY recorded_at DESC.
        recent = logger.query_entries(trace_id=tid, limit=3)
        assert [e.data["i"] for e in recent] == [9, 8, 7]

        # get_entries keeps its own oldest-first front-slice contract, so the
        # two accessors deliberately differ in order.
        oldest = logger.get_entries(trace_id=tid, limit=3)
        assert [e.data["i"] for e in oldest] == [0, 1, 2]

    def test_in_memory_offset_skips_from_newest_end(self) -> None:
        logger = AuditLogger()
        for i in range(5):
            logger.append(
                AuditEntry(event_type="handoff", producer=f"p{i}", data={"i": i})
            )
        # Skip the newest 1, take the next-newest 2, newest-first.
        page = logger.query_entries(limit=2, offset=1)
        assert [e.data["i"] for e in page] == [3, 2]

    def test_in_memory_empty_when_offset_exceeds_count(self) -> None:
        logger = AuditLogger()
        logger.append(AuditEntry(event_type="handoff", producer="p"))
        assert logger.query_entries(limit=5, offset=10) == []


class TestRowToAuditEntry:
    """_row_to_audit_entry must reconstruct DB-sourced entries so they match
    their in-memory equivalents exactly."""

    def test_trusts_metadata_and_injects_no_spurious_keys(self) -> None:
        # A faithful json.dumps(entry.data) round-trip in metadata, plus the
        # shredded columns carrying the event_type fallback for a non-transition
        # entry (decision_basis == action). The reconstruction must NOT inject a
        # decision_basis key the in-memory entry never had.
        row = {
            "id": 42,
            "recorded_at": datetime(2011, 3, 11, 6, 0, tzinfo=UTC),
            "agent_name": "policy",
            "action": "permission_check",
            "event_id": str(uuid4()),
            "trace_id": str(uuid4()),
            "metadata": json.dumps({"allowed": True, "tool": "fetch_url"}),
            "decision_basis": "permission_check",
            "state_before": None,
            "state_after": None,
        }
        entry = _row_to_audit_entry(row)
        assert entry.data == {"allowed": True, "tool": "fetch_url"}
        assert "decision_basis" not in entry.data
        assert entry.event_type == "permission_check"
        assert entry.producer == "policy"

    def test_reconstructs_from_columns_when_metadata_absent(self) -> None:
        row = {
            "id": 7,
            "recorded_at": datetime(2011, 3, 11, 6, 5, tzinfo=UTC),
            "agent_name": "orchestrator",
            "action": "state_transition",
            "metadata": None,
            "state_before": "IDLE",
            "state_after": "MONITOR",
            "decision_basis": "seismic override",
        }
        entry = _row_to_audit_entry(row)
        assert entry.data["from_state"] == "IDLE"
        assert entry.data["to_state"] == "MONITOR"
        assert entry.data["decision_basis"] == "seismic override"

    def test_entry_id_is_deterministic_from_row_id(self) -> None:
        row = {
            "id": 99,
            "recorded_at": datetime(2011, 3, 11, 6, 0, tzinfo=UTC),
            "agent_name": "a",
            "action": "handoff",
            "metadata": json.dumps({"x": 1}),
        }
        assert _row_to_audit_entry(row).entry_id == _row_to_audit_entry(row).entry_id
