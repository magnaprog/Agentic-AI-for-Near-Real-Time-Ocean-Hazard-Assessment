"""Pre-baked Tohoku 2011 demo snapshot for offline/demo mode.

Served only when no core API key is configured (MISSION_CONTROL_HAZARD_API_KEY
empty), which raises RuntimeError before any request is made, so the dashboard
can be exercised without a core API. Both the WS poll loop and the proxying
REST routers use that one condition. A configured but unreachable core is a
live-mode incident and never falls back here: it surfaces 5xx or an
upstream_error message instead. See backend/errors.py.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

# Tohoku 2011 event - matches validated pipeline output
TOHOKU_SNAPSHOT: dict[str, Any] = {
    "fsm": {
        "fsm_state": "ESCALATE",
        "has_active_event": True,
        "event_context": {
            "event_id": "tohoku-2011-03-11T05:46:24Z",
            "seismic_magnitude": 9.1,
            "seismic_region": "Near the east coast of Honshu, Japan",
            "epicenter_lat": 38.297,
            "epicenter_lon": 142.373,
            "trigger_time_utc": "2011-03-11T05:46:24.120000+00:00",
            "latest_anomaly_score": 0.996,
            "dart_confirmation": True,
            "active_dart_stations": [
                "21401",
                "21413",
                "21418",
                "21419",
                "46402",
                "46403",
                "46408",
                "46411",
            ],
            "stations_in_event_mode": [
                "21401",
                "21413",
                "21418",
                "21419",
                "46402",
                "46403",
                "46408",
                "46411",
            ],
        },
        "thresholds": {
            "basin": "pacific",
            "t1": 0.35,
            "t2": 0.60,
            "t3": 0.85,
        },
        "transition_history": [
            {
                "transition_id": "t-001",
                "event_id": "tohoku-2011-03-11T05:46:24Z",
                "timestamp_utc": "2011-03-11T05:46:24+00:00",
                "from_state": "IDLE",
                "to_state": "MONITOR",
                "trigger_reason": "Seismic event M9.1 detected near Honshu, Japan",
                "anomaly_score": None,
                "seismic_magnitude": 9.1,
            },
            {
                "transition_id": "t-002",
                "event_id": "tohoku-2011-03-11T05:46:24Z",
                "timestamp_utc": "2011-03-11T05:46:25+00:00",
                "from_state": "MONITOR",
                "to_state": "ESCALATE",
                "trigger_reason": (
                    "Seismic-only escalation: M9.1 at depth 29 km (PTWC criteria: "
                    "M>=7.5, depth<100 km). No DART event-mode activation required."
                ),
                "anomaly_score": None,
                "seismic_magnitude": 9.1,
            },
        ],
    },
    "agents": [
        {
            "name": "Anomaly Agent",
            "version": "2.1.0",
            "execution_path": "OFFLINE_DEMO_RESULT",
            "description": "Ensemble anomaly detection, 8/8 stations scored",
        },
        {
            "name": "Scenario Agent",
            "version": "1.4.0",
            "execution_path": "OFFLINE_DEMO_RESULT",
            "description": "Source inversion complete, Mw 9.1 rupture model",
        },
        {
            "name": "Verification Agent",
            "version": "1.2.0",
            "execution_path": "OFFLINE_DEMO_RESULT",
            "description": "Nine-check verification suite: PASS_WITH_CONCERNS with two concerns",
        },
    ],
    "recent_audit": [
        {
            "entry_id": "a-001",
            "timestamp_utc": "2011-03-11T05:46:24+00:00",
            "event_id": "tohoku-2011-03-11T05:46:24Z",
            "event_type": "state_transition",
            "producer": "orchestrator",
            "data": {
                "from_state": "IDLE",
                "to_state": "MONITOR",
                "trigger_reason": "Seismic event M9.1 near Honshu, Japan",
            },
        },
        {
            "entry_id": "a-002",
            "timestamp_utc": "2011-03-11T05:46:25+00:00",
            "event_id": "tohoku-2011-03-11T05:46:24Z",
            "event_type": "state_transition",
            "producer": "orchestrator",
            "data": {
                "from_state": "MONITOR",
                "to_state": "ESCALATE",
                "trigger_reason": "Seismic-only escalation: M9.1, depth 29 km",
            },
        },
        {
            "entry_id": "a-003",
            "timestamp_utc": "2011-03-11T05:49:24+00:00",
            "event_id": "tohoku-2011-03-11T05:46:24Z",
            "event_type": "anomaly_scored",
            "producer": "anomaly_agent",
            "data": {
                "station_id": "21418",
                "anomaly_score": 0.947,
                "trigger_reason": "Station 21418 crosses T1/T2/T3 at +3.0 min",
            },
        },
        {
            "entry_id": "a-005",
            "timestamp_utc": "2011-03-11T05:49:30+00:00",
            "event_id": "tohoku-2011-03-11T05:46:24Z",
            "event_type": "escalation_packet_generated",
            "producer": "report_node",
            "data": {
                "handoff_id": "esc-tohoku-001",
                "station_count": 8,
                "stations_in_event_mode": 8,
                "threshold_confirming_stations": 7,
            },
        },
    ],
    # Retrospective enrichment for the dashboard, derived from the validated
    # offline results (this is demo-only; the live worker emits none of it):
    #  - detection_latency: results/tohoku_detection.json sliding-window
    #    first-crossing of T1=0.35 / T3=0.85; distance_km is the great-circle
    #    distance from the epicenter (38.30N, 142.37E) to each station in
    #    src/hazard_assessment/data/station_coordinates.py. None = never crossed.
    #  - ensemble_ablation: results/ablation_results.json full_window, T3 hit
    #    count and peak ensemble score per configuration.
    "scenario_metrics": {
        "first_t1_minutes": 3.0,
        "detection_latency": [
            {"station_id": "21418", "distance_km": 561, "t1_minutes": 3.0, "t3_minutes": 3.0},
            {"station_id": "21401", "distance_km": 987, "t1_minutes": 5.0, "t3_minutes": 62.0},
            {"station_id": "21413", "distance_km": 1243, "t1_minutes": 7.0, "t3_minutes": 7.0},
            {"station_id": "21419", "distance_km": 1297, "t1_minutes": 5.8, "t3_minutes": 7.0},
            {"station_id": "46408", "distance_km": 3953, "t1_minutes": 5.2, "t3_minutes": 5.2},
            {"station_id": "46402", "distance_km": 4353, "t1_minutes": 5.5, "t3_minutes": 6.8},
            {"station_id": "46403", "distance_km": 4833, "t1_minutes": None, "t3_minutes": None},
            {"station_id": "46411", "distance_km": 7479, "t1_minutes": 19.0, "t3_minutes": 20.2},
        ],
        "ensemble_ablation": [
            {"configuration": "Threshold only", "t3_hits": "7/8", "peak_score": 1.000},
            {"configuration": "Statistical only", "t3_hits": "6/8", "peak_score": 0.989},
            {"configuration": "Threshold + statistical", "t3_hits": "7/8", "peak_score": 0.996},
            {"configuration": "Full ensemble", "t3_hits": "7/8", "peak_score": 0.996},
        ],
    },
}

# The core serves audit entries newest-first (ORDER BY recorded_at DESC), and
# the console's audit strip takes the first ten on that assumption. The block
# above is written oldest-first because that is the order the event happened
# in and it is far easier to read that way, so reverse it once here rather
# than maintaining it backwards. Without this the demo strip ran backwards
# relative to live and, past ten entries, would have shown the oldest.
TOHOKU_SNAPSHOT["recent_audit"].reverse()


# Demo wrapper mirrors GET /api/escalation/packet-of-record. It contains only
# worker-produced checkpoint evidence; no live scenario or Verification output
# is implied.
_DEMO_SCIENTIFIC_HASH = "a" * 64
_DEMO_EVENT_ID = "tohoku-2011-03-11T05:46:24Z"
DEMO_ESCALATION_PACKET: dict[str, Any] = {
    "packet_row_id": 1,
    "assessment_row_id": 1,
    "event_id": _DEMO_EVENT_ID,
    "renderer_version": "1",
    "content_sha256": "",
    "created_at": "2011-03-11T05:46:25+00:00",
    "packet": {
        "kind": "escalation_reviewer_packet",
        "renderer_version": "1",
        "assessment_row_id": 1,
        "checkpoint_id": "c" * 64,
        "event_id": _DEMO_EVENT_ID,
        "produced_at_utc": "2011-03-11T05:46:25+00:00",
        "fsm_state_before": "ASSESS",
        "fsm_state_after": "ESCALATE",
        "pipeline_outcome": "ABSTAIN",
        "input_manifest_hash": "d" * 64,
        "scientific_content_hash": _DEMO_SCIENTIFIC_HASH,
        "best_scoring_station": {
            "source": "dart",
            "station_id": "21418",
            "ensemble_score": 0.996,
        },
        "dart_stations_currently_in_event_mode": [
            "21401",
            "21413",
            "21418",
            "21419",
            "46402",
            "46403",
            "46408",
            "46411",
        ],
        "recommended_action": "Human review required",
        "disclaimer": (
            "This is a research decision-support assessment, not an official "
            "NOAA/NWS tsunami product."
        ),
        # render_reviewer_packet always emits this key, so the demo carries
        # it too; omitting it let the console be exercised against a packet
        # shape the real renderer never produces.
        "seismic_context": {
            "magnitude": 9.1,
            "region": "Honshu, Japan",
            "depth_km": 29.0,
            "origin_utc": "2011-03-11T05:46:24+00:00",
        },
        "assessment": {
            "handoff_id": "11111111-1111-4111-8111-111111111111",
            "event_id": _DEMO_EVENT_ID,
            "scientific_content_hash": _DEMO_SCIENTIFIC_HASH,
        },
    },
}
DEMO_ESCALATION_PACKET["content_sha256"] = hashlib.sha256(
    json.dumps(
        DEMO_ESCALATION_PACKET["packet"],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()
