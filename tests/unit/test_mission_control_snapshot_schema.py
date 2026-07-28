"""Unit tests for Mission Control snapshot schema validation."""

from __future__ import annotations

import sys
from pathlib import Path


def _load_snapshot_model():
    repo_root = Path(__file__).resolve().parents[2]
    mission_control_root = repo_root / "mission-control"
    if str(mission_control_root) not in sys.path:
        sys.path.insert(0, str(mission_control_root))

    from backend.models.schemas import SystemSnapshotOut

    return SystemSnapshotOut


def test_system_snapshot_schema_accepts_valid_payload() -> None:
    system_snapshot_out = _load_snapshot_model()
    snapshot = system_snapshot_out(
        fsm={
            "fsm_state": "IDLE",
            "has_active_event": False,
            "event_context": None,
            "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
            "transition_history": [],
        },
        agents=[
            {
                "name": "qc_agent",
                "version": "0.1.0",
                "execution_path": "LIVE_WORKER",
                "description": "Quality control",
            }
        ],
        recent_audit=[
            {
                "entry_id": "a6b27034-bf24-4dc8-962b-777f4eba144d",
                "timestamp_utc": "2026-03-05T00:00:00+00:00",
                "event_id": None,
                "event_type": "system",
                "producer": "test",
                "data": {},
            }
        ],
    )

    assert snapshot.agents[0]["description"] == "Quality control"


def test_system_snapshot_schema_accepts_partial_agents() -> None:
    """Agents field is now a list of dicts for graceful degradation."""
    system_snapshot_out = _load_snapshot_model()

    snapshot = system_snapshot_out(
        fsm={
            "fsm_state": "IDLE",
            "has_active_event": False,
            "event_context": None,
            "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
            "transition_history": [],
        },
        agents=[{"name": "qc_agent", "version": "0.1.0", "execution_path": "LIVE_WORKER"}],
        recent_audit=[],
    )
    assert snapshot.agents[0]["name"] == "qc_agent"


def test_system_snapshot_carries_review_decisions_separately() -> None:
    """recent_reviews is its own field, defaulting to empty.

    The console reads a recorded review from this feed rather than from
    recent_audit, which per-window anomaly entries can fill within a fraction
    of a second.
    """
    system_snapshot_out = _load_snapshot_model()
    fsm = {
        "fsm_state": "ESCALATE",
        "has_active_event": True,
        "event_context": None,
        "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
        "transition_history": [],
    }

    assert system_snapshot_out(fsm=fsm, agents=[], recent_audit=[]).recent_reviews == []

    snapshot = system_snapshot_out(
        fsm=fsm,
        agents=[],
        recent_audit=[],
        recent_reviews=[
            {
                "entry_id": "a6b27034-bf24-4dc8-962b-777f4eba144d",
                "timestamp_utc": "2026-03-05T00:00:00+00:00",
                "event_id": None,
                "event_type": "assessment_review_decision",
                "producer": "duty-scientist-1",
                "data": {"decision": "APPROVE", "escalation_packet_hash": "a" * 64},
            }
        ],
    )
    assert snapshot.recent_reviews[0].data["decision"] == "APPROVE"


def test_system_snapshot_carries_sensor_degraded() -> None:
    """The core publishes it on /api/fsm and the BFF must forward it.

    Without the field the console cannot tell an operator that the score on
    screen rests on fewer than two QC-usable DART stations.
    """
    system_snapshot_out = _load_snapshot_model()
    base = {
        "fsm_state": "MONITOR",
        "has_active_event": True,
        "event_context": None,
        "thresholds": {"basin": "pacific", "t1": 0.35, "t2": 0.6, "t3": 0.85},
        "transition_history": [],
    }

    assert system_snapshot_out(fsm=base, agents=[], recent_audit=[]).fsm.sensor_degraded is False
    degraded = system_snapshot_out(
        fsm={**base, "sensor_degraded": True}, agents=[], recent_audit=[]
    )
    assert degraded.fsm.sensor_degraded is True


def _load_demo() -> tuple[dict, dict]:
    import sys
    from pathlib import Path

    mc = Path(__file__).resolve().parents[2] / "mission-control"
    if str(mc) not in sys.path:
        sys.path.insert(0, str(mc))
    from backend.services.demo_snapshot import DEMO_ESCALATION_PACKET, TOHOKU_SNAPSHOT

    return TOHOKU_SNAPSHOT, DEMO_ESCALATION_PACKET


def test_demo_snapshot_satisfies_the_live_snapshot_schema() -> None:
    """Demo mode serves this dict verbatim, so nothing validates it in production.

    It is also the first thing anyone evaluating the repository sees. A field
    renamed in SystemSnapshotOut would break the demo console silently, with no
    test failing, because the demo path bypasses the model entirely.
    """
    system_snapshot_out = _load_snapshot_model()
    snapshot, _ = _load_demo()

    validated = system_snapshot_out(**{k: v for k, v in snapshot.items() if k != "demo_mode"})

    assert validated.fsm.fsm_state == snapshot["fsm"]["fsm_state"]
    assert len(validated.recent_audit) == len(snapshot["recent_audit"])


def test_demo_escalation_packet_carries_what_the_review_gate_reads() -> None:
    """The console binds a review to the packet row and its canonical hash."""
    _, packet = _load_demo()

    for key in ("packet_row_id", "assessment_row_id", "event_id", "renderer_version",
                "content_sha256", "created_at", "packet"):
        assert key in packet, f"demo packet is missing {key}"

    inner = packet["packet"]
    for key in ("checkpoint_id", "fsm_state_before", "fsm_state_after", "pipeline_outcome",
                "recommended_action", "disclaimer", "assessment"):
        assert key in inner, f"demo packet payload is missing {key}"

    assert len(packet["content_sha256"]) == 64
    assert packet["event_id"] == inner["event_id"]
