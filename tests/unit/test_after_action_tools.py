"""Tests for after-action read-only query tools.

Verifies that the 3 tools (query_audit_trail, query_fsm_transitions,
query_processed_features) correctly query the audit logger, enforce
event_id pinning, expose truncation honestly, record every invocation
into the caller's call log, and surface live QC and anomaly evidence.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from hazard_assessment.agents.llm_advisory.tools import (
    _MAX_TOOL_ENTRIES,
    make_event_query_tools,
)
from hazard_assessment.audit.logger import AuditEntry, AuditLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EVENT_ID = UUID("00000000-0000-0000-0000-000000000001")
OTHER_ID = UUID("00000000-0000-0000-0000-000000000099")


def _make_entry(
    event_id: UUID,
    event_type: str,
    producer: str,
    data: dict | None = None,
) -> AuditEntry:
    return AuditEntry(
        event_id=event_id,
        event_type=event_type,
        producer=producer,
        data=data or {},
    )


def _make_logger_with_entries(
    event_id: UUID = EVENT_ID,
    n_transitions: int = 3,
    n_outputs: int = 2,
    n_other_event: int = 1,
) -> AuditLogger:
    """Create an audit logger with pre-populated entries."""
    logger = AuditLogger()

    for i in range(n_transitions):
        logger.append(_make_entry(
            event_id=event_id,
            event_type="state_transition",
            producer="fsm_orchestrator",
            data={"from": "MONITOR", "to": "INVESTIGATE", "step": i},
        ))

    for i in range(n_outputs):
        logger.append(_make_entry(
            event_id=event_id,
            event_type="verification_complete",
            producer="verify_node",
            data={"score": 0.75 + i * 0.1},
        ))

    # Entries for a different event (should never be returned)
    for i in range(n_other_event):
        logger.append(_make_entry(
            event_id=OTHER_ID,
            event_type="state_transition",
            producer="fsm_orchestrator",
            data={"from": "IDLE", "to": "MONITOR", "step": i},
        ))

    return logger


def _tool(tools: list[Any], name: str) -> Any:
    return next(t for t in tools if t.name == name)


# ---------------------------------------------------------------------------
# Tool creation
# ---------------------------------------------------------------------------


class TestMakeAfterActionTools:
    def test_returns_three_tools(self) -> None:
        logger = AuditLogger()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        assert len(tools) == 3

    def test_tool_names(self) -> None:
        logger = AuditLogger()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        names = {t.name for t in tools}
        assert names == {
            "query_audit_trail",
            "query_fsm_transitions",
            "query_processed_features",
        }


# ---------------------------------------------------------------------------
# query_audit_trail
# ---------------------------------------------------------------------------


class TestQueryAuditTrail:
    def test_returns_envelope_object(self) -> None:
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        result = _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        parsed = json.loads(result)
        assert set(parsed) == {
            "entries", "n_total_matching", "n_returned", "truncated",
        }
        assert parsed["truncated"] is False

    def test_returns_only_pinned_event(self) -> None:
        """Event_id pinning: must never return entries for OTHER_ID."""
        logger = _make_logger_with_entries(n_other_event=5)
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        )
        # Should have 3 transitions + 2 outputs = 5, not 5 + 5
        assert parsed["n_total_matching"] == 5
        assert len(parsed["entries"]) == 5

    def test_filter_by_event_type(self) -> None:
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_audit_trail").invoke(
                {"event_type": "state_transition"}
            )
        )
        assert len(parsed["entries"]) == 3
        for entry in parsed["entries"]:
            assert entry["event_type"] == "state_transition"

    def test_empty_logger_returns_empty_envelope(self) -> None:
        logger = AuditLogger()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        )
        assert parsed["entries"] == []
        assert parsed["n_total_matching"] == 0
        assert parsed["truncated"] is False

    def test_truncation_is_exposed_not_silent(self) -> None:
        """More matches than the cap: the envelope must say so and report
        the true total, never pretend the capped slice is everything."""
        logger = AuditLogger()
        n = _MAX_TOOL_ENTRIES + 25
        for i in range(n):
            logger.append(_make_entry(
                event_id=EVENT_ID,
                event_type="state_transition",
                producer="fsm_orchestrator",
                data={"step": i},
            ))
        call_log: list[dict[str, Any]] = []
        tools = make_event_query_tools(
            logger, pinned_event_id=EVENT_ID, call_log=call_log
        )
        parsed = json.loads(
            _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        )
        assert parsed["truncated"] is True
        assert parsed["n_total_matching"] == n
        assert parsed["n_returned"] == _MAX_TOOL_ENTRIES
        assert len(parsed["entries"]) == _MAX_TOOL_ENTRIES
        assert call_log[-1]["truncated"] is True
        assert call_log[-1]["n_total_matching"] == n


# ---------------------------------------------------------------------------
# query_fsm_transitions
# ---------------------------------------------------------------------------


class TestQueryFSMTransitions:
    def test_returns_only_transitions(self) -> None:
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(_tool(tools, "query_fsm_transitions").invoke({}))
        assert len(parsed["entries"]) == 3
        assert parsed["truncated"] is False


# ---------------------------------------------------------------------------
# query_processed_features
# ---------------------------------------------------------------------------


class TestQueryProcessedFeatures:
    def test_returns_agent_outputs(self) -> None:
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_processed_features").invoke({"agent_name": ""})
        )
        assert len(parsed["entries"]) == 2  # 2 verification_complete entries

    def test_filter_by_agent_name(self) -> None:
        logger = _make_logger_with_entries()
        # Add an entry from a different agent
        logger.append(_make_entry(
            event_id=EVENT_ID,
            event_type="report_generated",
            producer="report_node",
            data={"top_scenario": "aleutian"},
        ))
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_processed_features").invoke(
                {"agent_name": "verify_node"}
            )
        )
        assert len(parsed["entries"]) == 2
        for entry in parsed["entries"]:
            assert entry["producer"] == "verify_node"

    def test_includes_live_qc_and_anomaly_evidence(self) -> None:
        """the live worker's qc_complete and anomaly_scored
        entries are part of the evidence surface, not filtered out."""
        logger = _make_logger_with_entries(n_transitions=1, n_outputs=1)
        logger.append(_make_entry(
            event_id=EVENT_ID,
            event_type="qc_complete",
            producer="pipeline_worker",
            data={"station": "dart:21413", "aggregate_condition": "SUSPECT"},
        ))
        logger.append(_make_entry(
            event_id=EVENT_ID,
            event_type="anomaly_scored",
            producer="pipeline_worker",
            data={"station": "dart:21413", "ensemble_score": 0.42},
        ))
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_processed_features").invoke({"agent_name": ""})
        )
        types = [e["event_type"] for e in parsed["entries"]]
        assert "qc_complete" in types
        assert "anomaly_scored" in types
        assert "verification_complete" in types

    def test_excludes_non_evidence_types(self) -> None:
        """state_transition entries should not appear in processed features."""
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        parsed = json.loads(
            _tool(tools, "query_processed_features").invoke({"agent_name": ""})
        )
        for entry in parsed["entries"]:
            assert entry["event_type"] != "state_transition"


# ---------------------------------------------------------------------------
# Call log (record every tool call)
# ---------------------------------------------------------------------------


class TestCallLog:
    def test_every_invocation_is_recorded_in_order(self) -> None:
        logger = _make_logger_with_entries()
        call_log: list[dict[str, Any]] = []
        tools = make_event_query_tools(
            logger, pinned_event_id=EVENT_ID, call_log=call_log
        )
        _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        _tool(tools, "query_fsm_transitions").invoke({})
        _tool(tools, "query_processed_features").invoke(
            {"agent_name": "verify_node"}
        )
        assert [c["tool"] for c in call_log] == [
            "query_audit_trail",
            "query_fsm_transitions",
            "query_processed_features",
        ]
        assert call_log[0]["args"] == {"event_type": ""}
        assert call_log[2]["args"] == {"agent_name": "verify_node"}
        for record in call_log:
            assert set(record) >= {
                "tool", "args", "n_total_matching", "n_returned", "truncated",
            }

    def test_no_call_log_is_harmless(self) -> None:
        logger = _make_logger_with_entries()
        tools = make_event_query_tools(logger, pinned_event_id=EVENT_ID)
        result = _tool(tools, "query_audit_trail").invoke({"event_type": ""})
        assert json.loads(result)["n_total_matching"] == 5
