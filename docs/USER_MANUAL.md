# Agentic AI for Near-Real-Time Ocean Hazard Assessment: User Manual

**Version:** 0.1.0
**Date:** 2026-03-11
**Status:** Active

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [System Overview](#2-system-overview)
3. [Prerequisites](#3-prerequisites)
4. [Installation](#4-installation)
5. [Configuration](#5-configuration)
6. [Starting the System](#6-starting-the-system)
7. [Mission Control Dashboard](#7-mission-control-dashboard)
8. [API Reference](#8-api-reference)
9. [Understanding the Assessment Pipeline](#9-understanding-the-assessment-pipeline)
10. [Human Review Guide](#10-human-review-guide)
11. [Audit Trail and Lineage](#11-audit-trail-and-lineage)
12. [Data Sources Reference](#12-data-sources-reference)
13. [Configuration Reference](#13-configuration-reference)
14. [Troubleshooting](#14-troubleshooting)
15. [Simulation and System Verification](#15-simulation-and-system-verification)
16. [Glossary](#16-glossary)

---

## 1. Introduction

This manual describes installation, configuration, and operation of **Agentic AI for Near-Real-Time Ocean Hazard Assessment**, a software framework that ingests NOAA DART deep-ocean observations through a validated pipeline, with architectural support for NOAA CO-OPS water-level data and USGS seismic event feeds (implemented and unit-tested but not yet exercised on real-event data). The system produces structured situational awareness guidance for duty scientists.

### What This System Does

- Polls NOAA DART buoys periodically (60 s standard, 15 s event mode); CO-OPS water-level and USGS seismic connectors are implemented but not yet validated on real events.
- Applies QARTOD-aligned quality control to incoming observations.
- Detects potential tsunami signals using an ensemble anomaly detector (harmonic detiding, bandpass filtering, wavelet energy, Bayesian changepoint detection, spatial coherence).
- Advances a deterministic five-state finite-state machine (FSM) in response to anomaly scores and seismic events.
- Provides offline scientific components for NNLS source-scenario inversion, nine Verification checks, and structured report generation.
- Builds one deterministic `OceanEvidenceAssessment` at each active live-worker checkpoint and fail-closes to ABSTAIN at ASSESS or ESCALATE.
- Persists an immutable reviewer packet from the assessment that entered ESCALATE when database storage is configured.
- Records caller-gated APPROVE, REJECT, or DEFER assessment reviews bound to that durable packet.

> **Deployment status:** Scenario inversion, Verification, and report generation
> run only in offline evaluation. The deployed Kafka worker runs ingest, QC
> metadata, anomaly detection, FSM evaluation, assessment persistence, and
> reviewer-packet rendering. Current review identity is caller-asserted, not an
> authenticated human principal. Review does not authorize distribution, change
> assessment status, close the event, or return the FSM to IDLE.

### What This System Does Not Do

- Issue public tsunami warnings, watches, or advisories. These are the exclusive responsibility of NOAA's National Tsunami Warning Center (NTWC) and Pacific Tsunami Warning Center (PTWC).
- Write to or modify any NOAA operational data streams or sensor networks.
- Make autonomous decisions about public safety.
- Distribute any assessment without explicit human approval.

### Intended Users

- Duty scientists and oceanographers who monitor real-time ocean observations.
- Systems engineers deploying the framework for integration testing or science validation.
- Software developers extending or maintaining the codebase.

---

## 2. System Overview

### Architecture

The system is organized into three planes:

```
+---------------------------------------------------------------------+
|  OUTPUT PLANE                                                        |
|  Report Agent (Tier 1/2/3) | Human Review Gate | ABSTAIN Output     |
+---------------------------------------------------------------------+
|  PROCESSING PLANE                                                    |
|  Orchestrator (Deterministic FSM)                                   |
|  IDLE -> MONITOR -> INVESTIGATE -> ASSESS -> ESCALATE                  |
|  QC Agent | Anomaly Agent | Scenario Agent | Verification Agent     |
+---------------------------------------------------------------------+
|  DATA PLANE                                                          |
|  DART Ingest | CO-OPS Ingest | Seismic Ingest | TimescaleDB + Kafka |
+---------------------------------------------------------------------+
Cross-cutting: OpenTelemetry | Prometheus | Immutable Audit Trail
```

### FSM States

| State | Meaning | Entry Condition |
|---|---|---|
| `IDLE` | No active event | Initial state or MONITOR timeout |
| `MONITOR` | Seismic trigger received; watching | Seismic event >= min. magnitude |
| `INVESTIGATE` | Anomaly score elevated | Anomaly score >= T1 (default 0.35) |
| `ASSESS` | Deterministic FSM threshold state; live checkpoints fail-close to ABSTAIN because Scenario and Verification are offline-only | Anomaly score >= T2 (default 0.60) |
| `ESCALATE` | Event is eligible for packet review once the durable packet exists | Anomaly score >= T3 (default 0.85), M >= 7.5 plus DART event-mode activation, or direct large-shallow-earthquake override from MONITOR |

### Pipeline Flow

```
live:    ingest -> qc metadata -> anomaly -> FSM -> OceanEvidenceAssessment
                                              | ASSESS/ESCALATE
                                              v
                                           ABSTAIN
                                              | entering ESCALATE
                                              v
                                  durable packet -> caller-gated review
                                                   (FSM unchanged)

offline: archived inputs -> scenario -> Verification -> report or ABSTAIN
```

---

## 3. Prerequisites

### Required Software

| Component | Minimum Version |
|---|---|
| Python | 3.11 |
| Docker | 24.0 |
| Docker Compose | 2.20 |
| Node.js | 20 LTS (for Mission Control frontend only) |
| Git | 2.40 |

### Hardware Recommendations

| Environment | Minimum | Recommended |
|---|---|---|
| Development | 4 CPU, 8 GB RAM, 20 GB disk | 8 CPU, 16 GB RAM, 50 GB disk |
| Production (single-node) | 8 CPU, 16 GB RAM, 100 GB disk | 16 CPU, 32 GB RAM, 500 GB disk |

### Network Requirements

The following outbound connections are required for live data ingestion:

| Endpoint | Port | Purpose |
|---|---|---|
| `www.ndbc.noaa.gov` | 443 | DART buoy text files |
| `api.tidesandcurrents.noaa.gov` | 443 | CO-OPS water-level API |
| `earthquake.usgs.gov` | 443 | USGS FDSN seismic event feed |

No inbound external connections are required. The dashboard and API are accessed on `localhost` by default.

---

## 4. Installation

### 4.1 Clone the Repository

```bash
git clone https://github.com/magnaprog/Agentic-AI-for-Near-Real-Time-Ocean-Hazard-Assessment.git
cd Agentic-AI-for-Near-Real-Time-Ocean-Hazard-Assessment
```

### 4.2 Create a Python Virtual Environment

```bash
python -m venv .venv
source .venv/bin/activate          # macOS/Linux
# .venv\Scripts\activate.bat       # Windows
```

### 4.3 Install Python Dependencies

```bash
pip install -e ".[dev]"
```

This installs the core package plus development tools (pytest, ruff, mypy). Optional extras:

```bash
pip install -e ".[telemetry]"      # OpenTelemetry OTLP exporter
pip install -e ".[llm]"            # Model-backed after-action and commentary paths
```

### 4.4 Verify the Installation

```bash
pytest tests/ -q
```

All tests should pass. If PostgreSQL is unavailable, inspect the skip reasons: database provisioning, grant, append-only, and round-trip tests skip rather than proving storage behavior. Run `tests/integration/` against PostgreSQL/TimescaleDB before treating storage checks as complete.

---

## 5. Configuration

### 5.1 Environment File

All configuration is provided via environment variables. A template is provided:

```bash
cp deploy/.env.example deploy/.env
```

Open `deploy/.env` and set the required variables (see [section 13 Configuration Reference](#13-configuration-reference) for full details).

**Minimum required variables:**

```bash
DB_ADMIN_PASSWORD=<strong-password>
DB_PASSWORD=<strong-password>
HAZARD_API_KEY=<random-32-char-string>
GRAFANA_PASSWORD=<strong-password>
MISSION_CONTROL_API_KEY=<random-32-char-string>
```

Generate API keys with:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

### 5.2 Anomaly Thresholds

The FSM thresholds determine how the system responds to anomaly scores. Current defaults are uncalibrated design constants used consistently in development evaluation; they are not validated Pacific operating thresholds:

```bash
THRESHOLD_T1=0.35          # MONITOR -> INVESTIGATE
THRESHOLD_T2=0.60          # INVESTIGATE -> ASSESS
THRESHOLD_T3=0.85          # ASSESS -> ESCALATE
THRESHOLD_SEISMIC_MIN_MAGNITUDE=6.0
THRESHOLD_ESCALATION_MAGNITUDE=7.5   # Seismic override threshold
THRESHOLD_MONITOR_TIMEOUT_HOURS=12.0  # Return to IDLE if no escalation
THRESHOLD_BASIN=pacific
```

Changing these values changes FSM transition behavior and invalidates direct comparison with committed evaluation artifacts. Do not treat any alternate values as operationally validated without a separate calibration and held-out evaluation.

### 5.3 LLM Configuration (Optional)

Offline Report Agent commentary and the after-action API can use an LLM. Install the `llm` extra before configuring a key:

```bash
LLM_API_KEY=<provider-api-key>   # LLM provider API key
LLM_MODEL=<provider-model-id>    # Provider model identifier; no default
```

If `LLM_API_KEY` is not set, offline reports use deterministic templates and `/api/after-action` returns 501. The FSM and numerical outputs do not depend on LLM availability.

---

## 6. Starting the System

### 6.1 Start Infrastructure Services

```bash
cd deploy
docker compose up -d
```

This starts: TimescaleDB, Kafka, Prometheus, Grafana, Jaeger, API server, pipeline worker, ingest workers (DART, CO-OPS, seismic), and Mission Control.

> **Note:** Complete live operation requires both Kafka and shared PostgreSQL/TimescaleDB. Without shared storage, API and worker FSM/audit state are isolated by process, so Mission Control may not see worker state. Without Kafka, ingest cannot reach the pipeline worker. These degraded startup modes are for local development, not coherent live operation.

Check that all services are healthy:

```bash
docker compose ps
```

Long-running services should show `healthy` or `running`; the one-shot `init-db` service should show `Exited (0)` after migrations complete.

### 6.2 Verify the API

```bash
curl http://localhost:8000/health
# Expected: {"status":"healthy","version":"0.1.0"}
```

```bash
curl -H "X-Hazard-Api-Key: <your-key>" http://localhost:8000/status
# Expected: {"fsm_state":"IDLE","has_active_event":false,...}
```

### 6.3 Access Dashboards

| Service | URL | Credentials |
|---|---|---|
| Mission Control | http://localhost:8100 | `MISSION_CONTROL_API_KEY` value |
| Grafana | http://localhost:3000 | `admin` / `GRAFANA_PASSWORD` |
| Jaeger (traces) | http://localhost:16686 | None |
| Prometheus | http://localhost:9090 | None |

Grafana starts with a provisioned "Hazard Assessment Overview" dashboard wired to the bundled Prometheus.

### 6.4 Stop the System

```bash
cd deploy
docker compose down        # Stop services, preserve data volumes
docker compose down -v     # Stop services AND delete all data (use with caution)
```

### 6.5 Running Without Docker (Development)

For development, run the core API and Mission Control as separate services. Each command below runs in its own terminal:

```bash
# Terminal 1: Start the core hazard assessment API (port 8000).
# HAZARD_API_KEY is required at startup (the app refuses to start without it).
HAZARD_API_KEY=<your-key> uvicorn hazard_assessment.app:app --reload --port 8000

# Terminal 2: Start Mission Control backend-for-frontend (port 8100).
# This proxies requests to the core API and serves the dashboard API.
# MISSION_CONTROL_API_KEY is required at startup (the BFF refuses to start without it);
# MISSION_CONTROL_HAZARD_API_KEY must match the key from Terminal 1, or the BFF cannot
# enable live core API access. If omitted, BFF deliberately serves demo data.
cd mission-control
MISSION_CONTROL_API_KEY=<dashboard-key> MISSION_CONTROL_HAZARD_API_KEY=<your-key> uvicorn backend.main:app --reload --port 8100

# Terminal 3: Start Mission Control frontend (Vite dev server)
cd mission-control/frontend
npm install && npm run dev
```

These commands run only the API and dashboard. They do not run live ingest or the pipeline worker. The API uses process memory when `DB_HOST` is empty, and that state is lost on restart. Complete live development also requires Kafka, shared PostgreSQL/TimescaleDB, the pipeline worker, and ingest workers. Mission Control uses `MISSION_CONTROL_HAZARD_API_URL` (default `http://localhost:8000`).

---

## 7. Mission Control Dashboard

Mission Control is the primary interface for duty scientists. It provides near-real-time FSM monitoring (2-second polling), escalation packet review, and audit trail access.

### 7.1 FSM State Display

The top panel shows the current FSM state as a color-coded indicator:

| State | Color | Description |
|---|---|---|
| IDLE | Green | No active event |
| MONITOR | Amber | Seismic trigger; watching |
| INVESTIGATE | Orange | Elevated anomaly score |
| ASSESS | Orange | Threshold state; live checkpoint is ABSTAIN because Scenario/Verification are offline-only |
| ESCALATE | Red | Escalated event; review becomes available after durable packet persistence |

### 7.2 Active Event Panel

When the FSM is non-IDLE, the Active Event panel in the right rail displays:
- Seismic magnitude (the region appears in the map's epicenter popup and, during ESCALATE, in the review gate's event summary)
- Epicenter coordinates (lat/lon)
- Trigger time (UTC)
- Latest event-level anomaly score
- DART station IDs observed in event mode for this event

The dashboard does not expose detector component scores or a complete monitored-station inventory. Its map is a static reference inventory, not proof that every displayed station is polled.

### 7.3 Knowing the Console Is Current

The BFF broadcasts a snapshot only when the snapshot changes, so a console
watching a quiet ocean receives nothing for long stretches. That silence looks
exactly like a console whose upstream has died, so the top bar carries a
**Core API poll** readout: the time since the BFF last reached the core API,
counted from the browser's own clock so the two clocks do not need to agree.

| Readout | Meaning |
|---|---|
| `3s ago`, green | The BFF polled the core API successfully that recently. Silence in the other panels means nothing changed. |
| `waiting` | The console just connected and the first heartbeat has not landed. It arrives within about five seconds. |
| `2m 14s ago`, red, labeled `Core API poll failing` | The BFF has not reached the core API since that time. Everything on screen is the last state it received. |
| `no contact`, red | The BFF reported its upstream down and has no successful poll to report on this connection. |
| `demo` | No `MISSION_CONTROL_HAZARD_API_KEY` is set. The console is serving the built-in Tohoku snapshot and no core API is being polled. |

This measures contact between the BFF and the core API. It is not a statement
that the ingest and pipeline workers are processing observations: the core API
exposes no worker heartbeat, so a stopped worker behind a healthy API still
reads as a fresh poll.

A **Usable DART stations** readout appears in the same row when the worker is
tracking fewer than two DART stations with QC-usable data in its retained
window, which is below the minimum for triangulation. It qualifies the score
next to it rather than the FSM state: coverage never gates a transition, and an
event during degraded coverage still escalates. The readout is absent when
coverage is adequate, because the flag distinguishes "below two" from "two or
more" and nothing finer.

Expect it after a pipeline-worker restart. Station windows are process memory
and are not warm-started from the database, and in standard mode a DART buoy
subsamples every 15 minutes but transmits a 6-hour batch to shore, so a freshly
started worker holds no DART data until the next batch lands. That can be
several hours. The readout is accurate throughout: the detector really does
have no DART evidence yet. It is not, on its own,
evidence that any buoy has failed. Event mode is different: transmission drops
to roughly one minute, so coverage recovers within minutes of an event.

### 7.4 Component Registry and Live Anomaly Score

Below the FSM ladder, the left rail lists each pipeline component with its role
and execution path: `LIVE WORKER` for components the running worker executes,
`OFFLINE EVALUATION ONLY` for those that run only in the offline analysis path.
This is a static registry read from the core API, not a health monitor.

Along the lower edge of the map, a live anomaly score chart plots the most
recent 60 samples with dashed reference lines at T1, T2, and T3. A sample is
appended on every score change plus a 30-second keepalive, and the trace is
built in the browser from page load, so it covers 30 minutes only when nothing
but the keepalive fires and less than that whenever scores update. With no active
event it shows a standby message rather than a zero line, because the stored
score resets to zero between events and zero is not a reading.

### 7.5 Escalation Packet View

When an assessment checkpoint enters `ESCALATE`, the pipeline worker renders a reviewer packet from the exact committed assessment row and persists it in the append-only `escalation_packets` table. Packet persistence requires configured database storage. The packet contains:
- Assessment row ID, assessment ID, and checkpoint ID
- FSM state before and after the checkpoint
- Pipeline outcome and best scored station
- DART stations currently transmitting in event mode
- Scientific-content hash and full immutable assessment payload
- Renderer version, non-authoritative disclaimer, and canonical packet SHA-256

Mission Control reads `GET /api/escalation/packet-of-record` and displays the same immutable packet row. It does not generate a replacement packet from process memory, and it does not claim live Scenario or Verification results that the worker did not compute.

### 7.6 Audit Trail Queries

Query the audit trail via the API. The Mission Control dashboard shows the most
recent entries in its bottom strip, collapsing consecutive entries of the same
type from the same producer into one row with a count: the worker writes an
anomaly-scoring entry per scored window, so an uncollapsed strip showed those
and nothing else. The strip is a recent-activity view, not a history. Query the
API for anything older than the last few seconds. Filter by:
- Event ID (UUID)
- Event type (e.g., `state_transition`, `assessment_persisted`, `assessment_review_decision`)
- Trace ID (single pipeline execution)

The API returns each entry's producer-selected `data` metadata. This is not necessarily the producer's full computational output or input snapshot.

### 7.7 Lineage Queries

Query `/api/lineage/<trace_id>` (or `/api/lineage/event/<event_id>`) to see the audit and decision lineage for a pipeline execution: which agent or state decided what, and when. For the raw-input chain, `/api/lineage/provenance/<trace_id>` walks the worker's persisted `qc_report` and `anomaly_score` feature rows back to raw observation records by payload hash: anomaly rows reference every sample retained in the scored rolling window, and QC rows reference the records QC summarized (database required). Assessment-stage outputs (scenario, verification, report) run in the offline path and have no live feature rows (see 11.5).

---

## 8. API Reference

Protected API endpoints require `X-Hazard-Api-Key`. `/health` and `/metrics` are public; `/docs`, `/redoc`, and `/openapi.json` are disabled.

**Base URL:** `http://localhost:8000`

---

### GET /health

Liveness probe. No authentication required.

**Response 200:**
```json
{"status": "healthy", "version": "0.1.0"}
```

---

### GET /status

Current FSM state and active event summary.

**Response 200:**
```json
{
  "fsm_state": "IDLE",
  "has_active_event": false,
  "event_id": "",
  "recovery_failed": false,
  "timestamp_utc": "2026-03-11T00:00:00Z"
}
```

---

### GET /api/fsm

Full FSM snapshot including event context, thresholds, and transition history.

`dart_confirmation` does not mean a tsunami was confirmed. It is a one-way
latch that turns true when at least one accepted, canonical DART event-mode
record scoped to the active event arrives timestamped at or after the seismic
origin, and it stays true until the event resolves. Event mode is a
high-cadence transmission state, an activation rather than an independent
waveform confirmation, so the flag means "a buoy started reporting fast after
this quake", not "a wave was verified". The name predates the current wording
and is kept because the field is persisted in durable FSM rows and read by
existing clients; the assessment envelope carries the same fact under the
precise name `dart_event_mode_observed_since_event_origin`, and Mission
Control labels it "DART event mode: OBSERVED SINCE ORIGIN".

`recovery_failed` is sticky for the life of the process: it is set if any FSM
recovery from the database has failed since startup and a later success does not
clear it, because the API refreshes state from the database on every read and
clearing on success would let a single corrupt row vanish before an operator saw
it. It is an alarm, not an interlock; processing continues from IDLE. Restarting
the service clears it. `/status` carries the same field.

`sensor_degraded` is true when fewer than two DART stations carry QC-usable
data in the pipeline worker's retained window (six hours), which is below the
minimum for triangulation. The worker evaluates it on every poll tick, so a
network that has gone entirely silent reports degraded coverage rather than
holding the last value. Samples QC never evaluated do not count as usable. The
flag is an alarm, not an interlock: it never gates an FSM transition.

**Response 200:**
```json
{
  "fsm_state": "ASSESS",
  "sensor_degraded": false,
  "recovery_failed": false,
  "has_active_event": true,
  "event_context": {
    "event_id": "550e8400-e29b-41d4-a716-446655440000",
    "seismic_magnitude": 7.8,
    "seismic_region": "Aleutian Islands",
    "epicenter_lat": 52.1,
    "epicenter_lon": -174.6,
    "trigger_time_utc": "2026-03-11T10:22:00Z",
    "latest_anomaly_score": 0.72,
    "dart_confirmation": false,
    "active_dart_stations": [],
    "stations_in_event_mode": []
  },
  "thresholds": {
    "basin": "pacific",
    "t1": 0.35,
    "t2": 0.60,
    "t3": 0.85
  },
  "transition_history": [
    {
      "transition_id": "...",
      "timestamp_utc": "2026-03-11T10:22:01Z",
      "from_state": "IDLE",
      "to_state": "MONITOR",
      "trigger_reason": "Seismic event M7.8 in Aleutian Islands",
      "seismic_magnitude": 7.8
    }
  ]
}
```

---

### GET /api/agents

Returns registered component manifests with version, description, and implemented execution path (`LIVE_WORKER` or `OFFLINE_EVALUATION_ONLY`). It does not report runtime status or permission codes.

---

### GET /api/audit

Returns recent audit trail entries. Query parameters:

| Parameter | Type | Description |
|---|---|---|
| `event_id` | UUID (optional) | Filter by event |
| `event_type` | string (optional) | Filter by event type |
| `trace_id` | UUID (optional) | Filter by pipeline trace |
| `limit` | int (optional, 1-200, default 50) | Maximum entries to return |

**Event types include:** `state_transition`, `verification_complete`, `report_generated`, `report_generation_failed`, `abstain_triggered`, `assessment_formatted`, `assessment_persisted`, `assessment_redelivery`, `assessment_gap`, `assessment_review_decision`, `escalation_packet_generated`, `escalation_packet_persisted`, `escalation_packet_conflict`, `qc_complete`, `anomaly_scored`, `input_provenance`, `seismic_provenance`, `provenance_capped`, `guardrail_scan`, `permission_check`, `policy_denial`, `llm_call`, `fsm_recovery_failed`, `after_action_report`

(The live worker writes `anomaly_scored` entries as lineage companions to its `processed_features` rows; the Mission Control demo snapshot also includes a synthetic one mirroring the validated offline trace.)

---

### GET /api/escalation

Returns the legacy process-memory escalation packet when the FSM is in `ESCALATE`. This endpoint is retained for the older generator path; it is not durable review authority. Use `/api/escalation/packet-of-record` for review.

**Response 404** when no active escalation.

---

### GET /api/escalation/packet-of-record

Returns the durable reviewer packet for the active event: the immutable `escalation_packets` row that the pipeline worker rendered and persisted at the checkpoint that entered ESCALATE. The packet is a deterministic projection of exactly one persisted assessment row and survives API restart.

**Response 200:**
```json
{
  "packet_row_id": 3,
  "assessment_row_id": 41,
  "event_id": "550e8400-...",
  "renderer_version": "1",
  "content_sha256": "sha256-hex",
  "created_at": "2026-07-17T01:02:03+00:00",
  "packet": { "kind": "escalation_reviewer_packet", "...": "..." }
}
```

**Response 404** when no database is configured, no event is active, or no durable packet row exists for the active event.

---

### POST /api/escalation/generate

Generates a legacy process-memory packet for compatibility with the older generator path. The FSM must be in `ESCALATE`. This packet is not accepted by `/api/review`; review requires the worker-produced durable packet of record. The endpoint accepts no request body, so callers cannot inject Scenario or Verification content.

**Response 200:**
```json
{
  "status": "generated",
  "packet_id": "...",
  "packet_hash": "sha256-hex"
}
```

**Response 409:** FSM not in ESCALATE state.
**Response 400:** Guardrail violation (prohibited alert terminology detected).

---

### POST /api/after-action

Runs model-backed analysis for a nonactive event using a 3-node LangGraph graph with tool use. Requires the `llm` optional dependency, `LLM_API_KEY`, and durable database-backed audit storage. This is separate from active-event investigation: the current event is rejected and the requested event must have audit history. Because no trusted event-disposition record exists, these checks do not prove that an event was formally closed.

Every tool call is recorded and returned in `tool_calls` (including unknown-tool requests, tool errors, and loop non-convergence), and tool results carry explicit truncation flags. Response 200 is returned only after the `after_action_report` entry is durably committed.

**Request body:**
```json
{
  "event_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Response 200:**
```json
{
  "event_id": "550e8400-...",
  "timeline": "Chronological reconstruction of events...",
  "gaps": "Identified gaps and weaknesses...",
  "draft_report": "Complete after-action report draft...",
  "tool_calls": [
    {
      "node": "timeline",
      "tool": "query_audit_trail",
      "args": { "event_type": "" },
      "n_total_matching": 42,
      "n_returned": 42,
      "truncated": false
    }
  ],
  "report_correlation_id": "report-correlation-uuid"
}
```

**Response 409:** the requested event is still active.
**Response 404:** no audit records exist for the requested event.
**Response 501:** `LLM_API_KEY` is missing.
**Response 503:** durable audit storage is unavailable or report commit fails.
**Response 500:** graph/provider execution failure, including a missing provider package.

---

### POST /api/review

Records a caller-gated assessment review. The FSM must be in `ESCALATE`, configured database storage must contain the active event's durable packet of record, and packet row ID and hash must match that row.

**Request body:**
```json
{
  "event_id": "550e8400-...",
  "decision": "APPROVE",
  "decision_reason": "Reviewed the packet evidence and recorded this assessment review.",
  "escalation_packet_row_id": 42,
  "escalation_packet_hash": "sha256-hex",
  "trace_id": "..."
}
```

**Headers:** `X-Reviewer-Id: <caller-asserted-identifier>` (required by both the core API and Mission Control; this is not authenticated human identity). Bounded at 128 characters and rejected if it contains control characters, because the value is written to the append-only audit trail as the record's producer and echoed into logs. That bounds its shape, not its truthfulness.

For APPROVE, REJECT, and DEFER, response 200 means the immutable `assessment_review_decision` entry was confirmed in durable audit storage. The endpoint leaves the FSM in `ESCALATE`, derives assessment identity from the packet, and includes packet and assessment bindings in the decision hash. No current decision authorizes distribution, changes assessment status, or closes the event. Durable append failure returns 503.

**Identity assurance:** the reviewer identity comes from a caller-supplied header behind the shared service API key, so every decision this endpoint records carries `identity_assurance: "CALLER_ASSERTED"`. This proves a caller-gated record, not authenticated human attribution or event-disposition authority.

**Response 200:**
```json
{
  "status": "recorded",
  "decision": "APPROVE",
  "event_id": "550e8400-...",
  "decision_hash": "sha256-hex",
  "identity_assurance": "CALLER_ASSERTED",
  "distribution_authorized": false,
  "event_disposition_recorded": false,
  "fsm_state": "ESCALATE"
}
```

---

### GET /api/lineage/{trace_id}

Returns the audit entries for a single pipeline execution (identified by trace_id), up to a 1000-row cap.
The response carries `entry_count`, the chronologically sorted `entries`, and a
`truncated` flag (true if the result hit the internal 1000-row cap; effectively
unreachable for a single run).

---

### GET /api/lineage/event/{event_id}

Returns the audit entries grouped by trace_id for an entire event (multiple
pipeline runs), up to a 1000-row cap. The response carries `total_entries`, the
`traces` map, and a `truncated` flag (true if the result hit the cap).

---

### GET /api/lineage/provenance/{trace_id}

Returns the raw-input provenance chain for one pipeline execution: the
`processed_features` rows the worker persisted for the trace (`qc_report` and
`anomaly_score`), each joined to its companion audit entry by handoff_id and
from there to the `raw_observations` rows referenced by the entry's input
hashes. Requires a configured database (503 otherwise); 404 when the trace has
no persisted features. This is the full feature-to-raw-record lineage, as
opposed to the decision lineage above.

---

### GET /api/activity-report/{event_id}

Returns a structured activity report for a given event, summarizing all agentic AI activities: LLM calls, guardrail scans, tool invocations, permission checks, and station coverage.

**Response 200:**
```json
{
  "event_id": "550e8400-...",
  "total_entries": 42,
  "truncated": false,
  "summary": {
    "fsm_transitions": 4,
    "llm_calls": 2,
    "llm_total_latency_ms": 1840,
    "guardrail_scans": 6,
    "guardrail_violations": 0,
    "permission_checks_total": 0,
    "permission_checks_allowed": 0,
    "permission_checks_denied": 0,
    "tool_invocations": 3,
    "station_coverage_reports": 5
  },
  "entries_by_type": {
    "state_transition": [...],
    "llm_call": [...],
    "guardrail_scan": [...],
    "permission_check": [...]
  }
}
```

The permission counters are shown as zero deliberately. Pipeline processing
never queries the permission matrix, so they stay at zero for an ordinary
event. They rise only if someone calls `/api/policy/check` by hand while that
event is the active one, since that endpoint attributes the check to whichever
event the FSM currently holds rather than to one the caller names. See that
endpoint below.

---

### GET /metrics

Prometheus metrics endpoint. Returns metrics in Prometheus text exposition format. No API key required.

Exposed metrics:
- `hazard_fsm_current_state` (Gauge): current FSM state (1 = active for the labeled state)
- `hazard_ingest_records_total` (Counter): ingest records by outcome (accepted/quarantined)
- `hazard_anomaly_scores_total` (Counter): station anomaly scores computed
- `hazard_fsm_transitions_total` (Counter): FSM transitions by destination state
- `hazard_abstain_total` (Counter): deliberate ABSTAIN decisions
- `hazard_verification_outcomes_total` (Counter): verification outcomes (PASS/PASS_WITH_CONCERNS/INCOMPLETE/FAIL)
- `hazard_assessment_gaps_total` (Counter): checkpoints with an active event whose assessment could not be built or persisted
- `hazard_lineage_persist_failures_total` (Counter): lineage rows the worker failed to persist
- `hazard_guardrail_scans_total` (Counter): alert-language guardrail scans by result (pass/violation)
- `hazard_station_scoring_duration_seconds` (Histogram): per-station anomaly-scoring latency

The domain counters increment in the worker process and are exposed by the worker's exporter (`METRICS_PORT`); the API `/metrics` reflects the FSM gauge and any counters incremented in the API process.

---

### POST /api/policy/check

Reports whether the permission matrix declares a named agent to be permitted
to perform a given action.

This endpoint is a query, not a gate. Nothing in the pipeline calls it, so a
denial recorded here does not stop anything; the matrix documents the intended
capability envelope. The bounds themselves are held by other mechanisms:
per-role database grants, the terminology guardrail scanner, the fail-closed
ABSTAIN routing, and the human review gate. Both `agent_name` and
`human_decision_present` are supplied by the caller and are not
independently established.

**Request body:**
```json
{
  "agent_name": "report_agent",
  "capability": "ER",
  "human_decision_present": false
}
```

**Permission codes:**

| Code | Name | Description |
|---|---|---|
| `RD` | READ_DATA | Read from data stores |
| `WD` | WRITE_DATA | Write to internal stores |
| `WA` | WRITE_AUDIT | Write audit trail |
| `PK` | PRODUCE_KAFKA | Publish to Kafka |
| `CK` | CONSUME_KAFKA | Consume from Kafka |
| `IL` | INVOKE_LLM | Call LLM synthesis |
| `MS` | MODIFY_STATE | Modify FSM state |
| `ER` | EMIT_REPORT | Generate reports |
| `AO` | APPROVE_OUTPUT | Approve outputs |

---

## 9. Understanding the Assessment Pipeline

### 9.1 Quality Control (QC) Stage

Every incoming observation passes through five QARTOD-aligned checks. The resulting flags annotate the record; they do not remove it from detection (see "Deployed-worker behavior" below):

| Check | What It Tests | Failure Effect |
|---|---|---|
| Timing gap | Observation interval vs. expected | Record flagged; station confidence reduced |
| Gross range | Value within physical bounds | Record flagged as unusable |
| Spike | Sudden jump relative to local variance | Record flagged |
| Rate of change | Change rate vs. configured limit | Record flagged |
| Flat line | Repeated identical values | Record flagged; sensor malfunction indicator |

A **station confidence score** `c in [0, 1]` is computed per observation as `c = 1 - (0.3 * n_suspect + 1.0 * n_fail) / n_tests`, where `n_tests` counts only applicable checks. Records with `c < 0.5` are marked `record_usable: false`.

**Deployed-worker behavior:** In the deployed Kafka worker, QC runs as audit metadata only and does NOT filter records out of anomaly scoring. This is deliberate: a genuine tsunami signal trips several QC checks (large gross-range deviation, spike, rate-of-change), so excluding low-confidence records live would discard the very signal of interest. The anomaly detector processes the raw values; the QC flags are recorded for the audit trail and operator review. Offline/replay evaluation may apply stricter record selection.

### 9.2 Anomaly Detection

The anomaly detector operates on a sliding window of recent observations from each DART/CO-OPS station. It produces an **ensemble anomaly score** in [0, 1] that combines three weighted groups:

| Group | Components | Description |
|---|---|---|
| Threshold (50%) | Amplitude ratio | Filtered signal peak vs. configured threshold |
| Statistical (35%) | Wavelet energy (Daubechies-4) OR BOCPD changepoint | Maximum of wavelet and BOCPD scores |
| ML (15%) | Isolation Forest on energy and spatial coherence features | Optional; excluded by default (no pre-trained model loaded) |

When the ML component is unavailable (current default), weights renormalize:
- Threshold: 58.8% (0.50 / 0.85)
- Statistical: 41.2% (0.35 / 0.85)

Spatial coherence is computed only when multi-station arrivals are supplied, and it reaches the score solely as an Isolation Forest feature. Because no pre-trained model ships by default, it does not affect the ensemble score in the default configuration.

**Filter degradation warning:** DART standard-mode data (15-min samples) has a Nyquist frequency below the bandpass upper cutoff. When detected, the system logs a `filter_degraded: true` flag. The upper edge is clamped to Nyquist, so the passband becomes roughly 30 to 120 min instead of 5 to 120 min. The ensemble weights do not change, and the bandpass-dependent components are not suppressed: they are computed on the clamped and partly aliased signal, so read them as unreliable in either direction rather than as absent. In the committed Chile artifact, station 54401 is degraded and scores on the threshold component alone, with BOCPD at zero.

### 9.3 FSM Thresholds and Transitions

The FSM advances based on anomaly score comparisons:

```
score < T1 (0.35):    Stay in MONITOR (or return from INVESTIGATE)
T1 <= score < T2:     MONITOR -> INVESTIGATE (or stay in INVESTIGATE)
T2 <= score < T3:     INVESTIGATE -> ASSESS (or stay in ASSESS)
score >= T3 (0.85):   ASSESS -> ESCALATE
```

**Seismic override:** If DART stations report event-mode transmissions alongside a large earthquake (M >= 7.5), the FSM advances regardless of anomaly score. It applies in three states, not just one: `MONITOR` advances to `INVESTIGATE`, `INVESTIGATE` to `ASSESS`, and `ASSESS` to `ESCALATE`. In `INVESTIGATE` and `ASSESS` it is evaluated before the de-escalation test, so a low interim score cannot pull the event back down. Event mode is a high-cadence transmission state (an activation, not an independent waveform confirmation); combined with a large earthquake it is treated as fail-safe grounds to bring the event to human review.

**De-escalation:** Score decreases can reverse transitions:
- INVESTIGATE -> MONITOR (if score drops below T1; suppressed when M >= 7.5 with DART event-mode activation, which advances to ASSESS instead)
- ASSESS -> INVESTIGATE (if score drops below T2)

`ESCALATE -> IDLE` is available only through the low-level `resolve_event()` method. No current API route has trusted event-disposition authority, and caller-asserted assessment review does not invoke this transition.

### 9.4 Offline Scenario Inversion

In the offline evaluation path, a Scenario component runs NNLS inversion for prepared assessment inputs:

1. Retrieves DART waveform observations.
2. Selects candidate unit-source Green's functions from the configured unit-source library based on epicenter proximity.
3. Solves the non-negative least-squares problem: minimize ||Hm - d|| subject to m >= 0, where H is the Green's function matrix and d is the observation vector.
4. Ranks top-K source models by residual norm.
5. Estimates uncertainty via bootstrap resampling (P10/P50/P90 envelopes).

**Offline seismic-only scenario mode:** The Scenario component can produce a magnitude-scaling estimate labeled `constraint_stage: SEISMIC_ONLY` with the caption *"Seismic-only estimate. No DART constraint. High uncertainty."* Current offline validators exercise this mode. The live Kafka worker does not call it. A live large, shallow seismic trigger can move the FSM directly from MONITOR to ESCALATE, but the worker emits a separate seismic-only ABSTAIN checkpoint, not this Scenario estimate.

> **Deployment note:** The Scenario inversion described above (and Verification
> and Report generation in 9.5-9.6) is the validated behavior exercised by the
> offline/replay evaluation that produces the published results. The unit-source
> library is a precomputed set of unit-source Green's functions; current builds ship
> a synthetic test library, and operational use requires the NOAA NCTR Forecast
> Propagation Database. The *deployed*
> Kafka worker does NOT yet run live Scenario inversion, Verification, or Report
> generation; on reaching `ASSESS`/`ESCALATE` it fail-closes to an ABSTAIN
> record for human situational awareness rather than emit an under-validated
> assessment. Confirm the deployed scope against the worker source and tests.

### 9.5 Verification Checks

Nine checks validate the scenario assessment before a report is generated:

| Check | FAIL condition | CONCERN condition |
|---|---|---|
| Hold-out station validation | Amplitude error > 50% or arrival error > 5 min | > 30% or > 3 min |
| Sensitivity analysis (LOO) | Top scenario flips on single-source removal | Weight shift > 30% |
| Posterior stability | --- | Source set changed between assessments |
| Data coverage | Fewer than 2 DART stations | Azimuthal spread < 90 deg |
| Physical consistency | Delta-Mw (NNLS vs seismic) > 0.6 | > 0.3 |
| Model fit | RMSE > 5 cm or bias > 1 cm AND > 3 sigma | RMSE > 3 cm |
| Tidal state | Needed + uncorrected | Uncorrected coastal proxies |
| Meteotsunami screening | Non-tsunami energy > 60% | > 30% |
| Rayleigh wave suspect | --- (never FAIL) | Timing consistent with Rayleigh arrival |

**ABSTAIN rule:** Any `FAIL`-rated check sets `abstain_required: true`. So does an `INCOMPLETE` outcome, which the aggregator returns when a REQUIRED check has no prerequisite data (result `NOT_EVALUATED`, prerequisite `MISSING` or `ERROR`), or when no check applies at the current constraint stage. Insufficient evidence therefore reaches ABSTAIN by the same path as contradicted evidence. The system immediately routes to the ABSTAIN path and will not generate probabilistic coastal guidance. The ABSTAIN output is a structured document explaining the reason for abstention, and it is still routed to the Human Review Gate.

### 9.6 Report Tiers

| Tier | Audience | Content |
|---|---|---|
| 1 - Technical Brief | Domain experts | All numerical details, component scores, verification evidence |
| 2 - Situational Summary | Duty scientists | Condensed probabilistic guidance; uncertainty summary |
| 3 - Post-Event | Post-event review | Narrative reconstruction; for archival and research |

**LLM narrative synthesis (optional).** When the `llm` dependency is installed and `LLM_MODEL` is configured, reports can include a narrative generated by a 4-node LangGraph synthesis graph (Retrieval -> Evidence -> Scenario -> Narrative), gated by the deterministic system-confidence score (skipped below 0.35). The deterministic template is always the emitted `summary`; the LLM narrative is stored separately in the `model_commentary` field and never replaces or alters deterministic report content. All LLM output passes through the guardrail scanner; a violating narrative is dropped and the report ships template-only. The LLM cannot modify the deterministic report text, numerical results, report tier, or assessment status.

All reports carry a mandatory non-authoritative disclaimer that cannot be suppressed or modified.

### 9.7 Assessment and Review Status

Offline `FinalAssessment` artifacts may use schema status values such as `PROVISIONAL`, `ABSTAIN`, or `APPROVED_INTERNAL`, but that schema capability is not a description of current live review authority. Live worker assessments record their deterministic pipeline outcome and remain immutable.

The current `/api/review` route appends a separate `assessment_review_decision` record. APPROVE, REJECT, and DEFER do not mutate the assessment, authorize distribution, close the event, or change the FSM. `APPROVED_INTERNAL` is therefore not produced by the current live API path.

---

## 10. Human Review Guide

This section is written for duty scientists reviewing escalations surfaced by Mission Control.

### 10.1 When an Escalation Requires Your Review

An escalation becomes immediately reviewable in Mission Control when the FSM enters `ESCALATE`. This occurs when:
- The anomaly score exceeds T3 (0.85), OR
- A large earthquake (M >= 7.5) has occurred AND a DART station is reporting event-mode transmissions, OR
- A large shallow earthquake (M >= 7.5, depth < 100 km) occurs in a tsunamigenic zone (seismic-only escalation; no DART event-mode activation required, since the wave reaches the coast before DART data does).

You will have access to:
- The immutable **packet of record** rendered from the assessment that entered ESCALATE
- The **Mission Control dashboard** at http://localhost:8100

### 10.2 Reviewing the Escalation Packet

When the FSM is in ESCALATE, Mission Control loads the durable packet of record. If no database is configured or the worker has not persisted the packet, the review panel shows an error and no decision can be submitted. The packet displays:

1. **Checkpoint identity**: The checkpoint and assessment row that entered ESCALATE
2. **FSM transition**: State before and after that checkpoint
3. **Pipeline outcome**: Including the live worker's fail-closed ABSTAIN result
4. **Best scored station**: A mechanical projection from the assessment, when a station scored successfully
5. **Assessment and scientific hashes**: Immutable evidence bindings
6. **Packet row, renderer version, and packet hash**: Durable packet identity and tamper check

The live packet does not include Scenario or Verification results because the deployed worker does not compute those stages.

### 10.3 Recording an Assessment Review

The **Human Review Gate** occupies the top of the right rail and activates when
the FSM reaches ESCALATE; there is nothing to navigate to. The gate loads the
escalation packet, and a decision requires three steps in order:

1. **Acknowledge the packet.** Click ACKNOWLEDGE PACKET REVIEWED. Until you do,
   the rationale box and all three decision buttons stay disabled.
2. **Enter a decision reason.** A plain-text rationale, required, maximum 5000
   characters.
3. **Choose a decision:** APPROVE, REJECT, or DEFER. The buttons enable only
   once both preceding steps are complete, and they stay pinned below the
   evidence so they remain reachable while you scroll.

Until both steps are done the gate says which one is outstanding, so the
disabled buttons are never unexplained.

Once a decision exists for the packet on screen, the three buttons are replaced
by the record itself: the decision, the reviewer ID that submitted it, and when
it was recorded. The gate reads that record from the audit trail rather than
from browser state, so it survives a reload and is visible to whoever takes
over the shift. It is bound to the packet hash, not just the event, so a
superseding packet for the same event still reads as unreviewed. The
ESCALATE section also stops pulsing at that point, because the work it was
asking for is done. Use **Record a superseding decision** to bring the buttons
back and record another decision against the same packet; the earlier record is
never overwritten, and the newest one is what the gate shows.

The **Reviewer ID** is a caller-asserted identifier sent as `X-Reviewer-Id`.
Enter it on the unlock screen when you connect. The unlock screen requires it
alongside the access key, because the console does not ask again at submission
time and a decision with an empty reviewer ID is rejected before it leaves the
browser. Both values are held in the browser tab's session storage, so a reload
keeps you signed in. To review under a different identifier, open the console in
a new tab or close and reopen this one, which clears that storage and brings the
unlock screen back.

These labels record the caller's assessment review only:

| Decision | Recorded Meaning | System Effect |
|---|---|---|
| `APPROVE` | Caller records that the packet-supported assessment is acceptable for the caller's review purpose. | Immutable review record; FSM stays in ESCALATE; no distribution authority. |
| `REJECT` | Caller records that the assessment is not supported for that review purpose. | Immutable review record; FSM stays in ESCALATE. |
| `DEFER` | Caller records that more evidence is needed. | Immutable review record; FSM stays in ESCALATE. |

Current reviewer identity is not authenticated as an individual human principal. None of these decisions closes the event, changes assessment status, or authorizes internal or public distribution.

**Demo mode does not accept decisions.** When Mission Control runs without
`MISSION_CONTROL_HAZARD_API_KEY`, it serves a built-in Tohoku 2011 snapshot and
shows a banner reading "DEMO MODE - Static Tohoku 2011 snapshot (core API not
configured)". That snapshot sits in ESCALATE and loads a demo packet, so the
review gate looks live, but submitting a decision returns HTTP 503: the core API
that would persist the record is not connected. Configure the key against a
running core API to record a real review.

### 10.4 Writing a Decision Reason

Your reason is recorded in the append-only audit trail. It should explain:
- What you observed in the packet of record
- Why the selected assessment-review label fits that evidence
- Any relevant external context, clearly identified as external to the packet

**Prohibited terms:** Do not use the words "Warning," "Advisory," "Watch," "Information Statement," "Threat Message," "Cancellation," "All Clear," or "Bulletin" in your decision reason. These eight terms are reserved for official NOAA products; their use here will be blocked by the guardrail scanner. A two-word term matches whether its words are separated by spaces, run together, or joined by a hyphen, so "All Clear," "AllClear," and "All-Clear" are all rejected. Substituting a lookalike character does not help either: the scanner folds the Unicode confusables set to ASCII before matching, and separately collapses characters that only share a shape, so "CanceIlation" written with a capital I is rejected as well. Write plainly and the scanner stays out of your way.

### 10.5 After Review

After any review decision:
- The FSM remains in `ESCALATE`.
- Assessment status and distributability do not change.
- The `assessment_review_decision` record binds packet row ID/hash and assessment row ID/assessment ID/scientific hash.
- Decision lineage is available via `/api/lineage/event/{event_id}`. Raw-input lineage for worker QC and anomaly outputs is available via `/api/lineage/provenance/{trace_id}` when database storage is configured.

A separate trusted-human event-disposition path and authenticated approval authority are planned, not current capabilities.

---

## 11. Audit Trail and Lineage

### 11.1 Audit Trail Design

The audit trail is **append-only**. No records can be modified or deleted. Every significant system action is recorded:

| Event Type | Trigger |
|---|---|
| `state_transition` | FSM state changes (IDLE->MONITOR, etc.) |
| `verification_complete` | Verification Agent finishes |
| `report_generated` | Report Agent produces assessment |
| `report_generation_failed` | Report Agent fails; pipeline falls back to ABSTAIN |
| `abstain_triggered` | ABSTAIN path triggered |
| `assessment_review_decision` | Caller-gated review bound to durable packet and assessment hashes |
| `assessment_formatted` | Offline formatter applies a legacy `HumanDecision` |
| `assessment_persisted` | Live checkpoint assessment committed |
| `assessment_redelivery` | Existing checkpoint assessment adopted on Kafka redelivery |
| `assessment_gap` | Required checkpoint assessment could not be persisted |
| `escalation_packet_generated` | Legacy process-memory packet created |
| `escalation_packet_persisted` | Durable packet of record committed |
| `qc_complete` | Worker QC summary for a station batch (audit metadata only) |
| `input_provenance` | Validly ingested observation linked to the active event |
| `seismic_provenance` | Seismic trigger record linked to the FSM event |
| `provenance_capped` | Per-event observation provenance cap reached |
| `guardrail_scan` | Terminology guardrail scan recorded |
| `permission_check` | Permission matrix queried through `/api/policy/check`. Not emitted by pipeline processing |
| `policy_denial` | A permission-matrix query returned a denial. Recorded for audit; no action was blocked |
| `llm_call` | LLM advisory call recorded |
| `anomaly_scored` | Worker anomaly assessment persisted as a lineage feature row |
| `fsm_recovery_failed` | FSM state recovery from the database failed; event context lost |

### 11.2 Provenance Chain

Each audit entry links:
- **event_id** - The seismic event that triggered the pipeline
- **trace_id** - The specific pipeline execution run
- **producer** - The agent or component that created the entry
- **data** - Producer-selected metadata for that event; not necessarily a full output snapshot

This links every decision step (which agent or state decided what, and when) into a queryable chain per event and per pipeline run. Raw-input linkage is carried where it is recorded: every raw record is hashed at ingest, and escalation packets assemble InputRefs from the recorded provenance audit entries. A single end-to-end join from every raw observation to the final artifact is not yet exposed through the API (the SQL `get_provenance()` function provides that join at the database level).

### 11.3 Querying Audit Records

Via API:
```bash
# All entries for an event
curl -H "X-Hazard-Api-Key: <key>" \
  "http://localhost:8000/api/audit?event_id=<uuid>&limit=100"

# All state transitions
curl -H "X-Hazard-Api-Key: <key>" \
  "http://localhost:8000/api/audit?event_type=state_transition"

# Full lineage for a trace
curl -H "X-Hazard-Api-Key: <key>" \
  "http://localhost:8000/api/lineage/<trace_id>"
```

Via Mission Control: the dashboard's bottom strip shows the most recent audit entries; use the API above for filtered queries.

### 11.4 Content Hashing

Every raw observation record is hashed (SHA-256) at ingest. Schemas that carry `InputRef`s (for example the escalation packet's `input_refs`) store these hashes. This allows after-the-fact verification that a referenced observation was not altered between ingest and the assessment.

To verify a DART record manually:
```python
import hashlib
with open("raw_dart_record.txt", "rb") as f:
    print(hashlib.sha256(f.read()).hexdigest())
```
Compare the result against the `sha256` field in the corresponding `InputRef`.

### 11.5 processed_features Growth and Retention

When a database is configured, the live worker writes lineage rows to the
`processed_features` table: one `qc_report` row per station batch and one
`anomaly_score` row per scored station, each batch. With batches every 60 s
in standard mode (15 s in event mode) and one row pair per actively scored
station, the table grows on the order of thousands to tens of thousands of
rows per day under sustained multi-station monitoring, scaling with the
station count and the (faster) event-mode cadence. Rows are never updated or
deleted by the application.

No retention policy is applied automatically, and this release provides no tested deletion or TimescaleDB-retention migration. Do not apply a blanket retention rule to `processed_features`: immutable `ocean_evidence_assessment` rows are assessment evidence and are referenced by `escalation_packets` and review decisions.

Any deployment-specific policy needs a reviewed migration that distinguishes high-volume `qc_report`/`anomaly_score` rows from immutable assessments, preserves referential and audit requirements, and is tested against `get_provenance()`. Retention for feature rows must also be coordinated with `audit_events` and `raw_observations`, because provenance spans all three tables.

---

## 12. Data Sources Reference

### 12.1 DART Buoys

**Network:** NOAA National Data Buoy Center (NDBC)
**Data URL:** `https://www.ndbc.noaa.gov/data/realtime2/{STATION_ID}.dart`
**Format:** Space-delimited text files (Year, Month, Day, Hour, Minute, Second, Type, Height_m)

**Measurement types:**
- Type 1: 15-minute standard mode
- Type 2: 1-minute event mode average
- Type 3: 15-second event mode high resolution

**Configured stations (13):** 21413, 21415, 21416, 21419, 21420, 46404, 46407, 46409, 46411, 46413, 46414, 46416, 46419

**Polling interval:** 60 seconds (standard), 15 seconds (event mode detection)
**Event mode timeout:** 4 hours

**Station health states:** ONLINE, STALE (>12 h without data), OFFLINE (connection error)

### 12.2 CO-OPS Water Level

**Network:** NOAA Center for Operational Oceanographic Products and Services
**API URL:** `https://api.tidesandcurrents.noaa.gov/api/prod/datagetter`

**Parameters used:**
- `product=one_minute_water_level` (primary), fallback: `water_level` (6-minute)
- `datum=STND`, `units=metric`, `time_zone=gmt`, `format=json`

**Configured stations (18):** the Pacific set in
`src/hazard_assessment/ingest/coops.py` (`COOPS_PACIFIC_STATION_IDS`).
Seven Pacific-island gauges:

| Station ID | Name |
|---|---|
| 1612340 | Honolulu, HI |
| 1617760 | Hilo, HI |
| 1615680 | Kahului (Maui), HI |
| 1619910 | Midway Island, HI |
| 1890000 | Wake Island |
| 1770000 | Pago Pago, American Samoa |
| 1631428 | Pago Bay, Guam |

Plus eleven Alaska and West Coast gauges: 9464212 St. Paul Island, 9457292
Kodiak, 9461380 Adak, 9444900 Port Townsend, 9443090 Neah Bay, 9432780
Charleston, 9418767 North Spit (Eureka), 9419750 Crescent City, 9414290 San
Francisco, 9413450 Monterey, and 9410230 La Jolla.

**Polling interval:** 30 seconds
**Lookback window:** 10 minutes per request

### 12.3 USGS Seismic Events

**Source:** USGS Earthquake Hazards Program FDSN event web service
**URL:** `https://earthquake.usgs.gov/fdsnws/event/1/query`

**Filter parameters:**
- `minmagnitude=5.5` (ingest filter; the FSM trigger threshold is separately configured at 6.0 via `THRESHOLD_SEISMIC_MIN_MAGNITUDE`)
- `format=geojson`
- `orderby=time`

**Polling interval:** 15 seconds
**Deduplication:** By USGS event ID (handles catalog revisions)

---

## 13. Configuration Reference

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `HAZARD_API_KEY` | required | Core API authentication key |
| `MISSION_CONTROL_API_KEY` | required | Mission Control HTTP and WebSocket authentication key |
| `MISSION_CONTROL_HAZARD_API_URL` | `http://localhost:8000` | BFF upstream core API URL; Compose sets `http://api-server:8000` |
| `MISSION_CONTROL_HAZARD_API_KEY` | empty | BFF upstream key. Empty selects deliberate demo mode; a configured live upstream failure never selects demo data. |
| `MISSION_CONTROL_POLL_INTERVAL_SECONDS` | `2.0` | Core snapshot polling interval |
| `DB_ADMIN_USER` | `hazard_admin` | Bootstrap/migration role |
| `DB_ADMIN_PASSWORD` | required in Compose | Bootstrap/migration password |
| `DB_HOST` | empty at process gate | Empty disables database use in API/workers; Compose sets `timescaledb` |
| `DB_PORT` / `DB_NAME` | `5432` / `hazard_assessment` | Database port and name. Pinned by Compose to the service; editable only when running directly |
| `DB_CONNECT_TIMEOUT` | `10` | Connection timeout in seconds |
| `DB_PASSWORD` | required in Compose | Fallback password for fixed runtime roles |
| `DB_DEFAULT_ROLE_PASSWORD` | `DB_PASSWORD` | Explicit name for that fallback; set either one |
| `DB_INGEST_WRITER_PASSWORD` | `DB_PASSWORD` | Optional ingest-role password override |
| `DB_ORCHESTRATOR_WRITER_PASSWORD` | `DB_PASSWORD` | Optional API-role password override |
| `DB_PIPELINE_WORKER_PASSWORD` | `DB_PASSWORD` | Optional pipeline-role password override |
| `DB_AGENT_WRITER_PASSWORD`, `DB_AGENT_READER_PASSWORD`, `DB_AUDIT_READER_PASSWORD`, `DB_INVESTIGATOR_WRITER_PASSWORD` | `DB_PASSWORD` | Optional provisioned-role overrides; investigator role has no current service |
| `KAFKA_BOOTSTRAP_SERVERS` | empty | Broker list; empty disables producer/consumer. Compose sets `kafka:9092`. |
| `CALIBRATION_DIR` | empty | Flat directory of pipeline-worker station calibration CSVs; empty loads none. Compose mounts archived event data at `/app/data` but does not select one event's retrospective calibration set for live processing. |
| `METRICS_PORT` | empty | Worker exporter port; empty disables exporter. Compose sets `9100`. |
| `INGEST_DART_POLL_INTERVAL_STANDARD_SEC` / `INGEST_DART_POLL_INTERVAL_EVENT_SEC` | `60` / `15` | DART polling intervals |
| `INGEST_DART_EVENT_MODE_TIMEOUT_SEC` | `14400` | Event-mode expiry in seconds |
| `INGEST_COOPS_POLL_INTERVAL_SEC` / `INGEST_SEISMIC_POLL_INTERVAL_SEC` | `30` / `15` | Source polling intervals |
| `INGEST_RETRY_MAX_ATTEMPTS` / `INGEST_RETRY_BACKOFF_SEC` | `3` / `2.0` | Ingest retry policy |
| `GRAFANA_PASSWORD` | required in Compose | Grafana admin password |
| `OTLP_ENDPOINT` | empty | OpenTelemetry OTLP gRPC endpoint |
| `LLM_API_KEY` | empty | Enables model-backed after-action and optional offline commentary |
| `LLM_MODEL` | empty | Provider model identifier; required whenever `LLM_API_KEY` is set |
| `APP_ENVIRONMENT` / `APP_LOG_LEVEL` | `development` / `INFO` | Environment label (set by Compose; no source code reads it) and worker logging level (pipeline and ingest workers only) |
| `THRESHOLD_T1` / `THRESHOLD_T2` / `THRESHOLD_T3` | `0.35` / `0.60` / `0.85` | FSM anomaly thresholds |
| `THRESHOLD_SEISMIC_MIN_MAGNITUDE` | `6.0` | Minimum magnitude to enter MONITOR |
| `THRESHOLD_ESCALATION_MAGNITUDE` | `7.5` | Large-earthquake and DART override magnitude |
| `THRESHOLD_SEISMIC_ESCALATION_DEPTH_KM` | `100.0` | Maximum known depth for direct seismic-only escalation |
| `THRESHOLD_MONITOR_TIMEOUT_HOURS` / `THRESHOLD_BASIN` | `12.0` / `pacific` | Monitor timeout and basin label |

### Changing Thresholds

Thresholds are read at startup and cannot be changed without restarting the service. That comes from how configuration is loaded, and it is intentional; prohibited action P8 in the permission matrix records the intent but does not implement it. To update thresholds:

1. Stop the system: `docker compose down`
2. Edit `deploy/.env`
3. Restart: `docker compose up -d`

Verify active values with authenticated `GET /api/fsm`; startup does not emit a dedicated old-versus-new threshold audit record.

---

## 14. Troubleshooting

### API Returns 401 Unauthorized

Verify the `X-Hazard-Api-Key` header matches `HAZARD_API_KEY` in your `.env` file. Keys are case-sensitive.

### FSM Stuck in MONITOR After Seismic Event

The FSM stays in MONITOR until:
- The anomaly score crosses T1 (advancing to INVESTIGATE), or
- The monitor timeout expires (default 12 hours), returning to IDLE.

Do not restart only the API to reset a live event: the pipeline worker owns live FSM transitions, and the API reads worker-persisted state. No current API route records a trusted event disposition. In a throwaway no-database development run, stop and restart every process that owns FSM state. With PostgreSQL configured, restart recovers durable state rather than clearing it.

### Docker Compose Services Not Healthy

Check logs for the failing service:
```bash
docker compose logs timescaledb --tail 50
docker compose logs kafka --tail 50
```

Common issues:
- **TimescaleDB health check fails:** Allow 30-60 seconds for first initialization.
- **Kafka not ready:** KRaft mode requires all controllers to be elected before producing. Allow 30 seconds.
- **init-db exits non-zero:** Check `docker compose logs init-db` for SQL errors.

### Migration Checksum Mismatch

```
RuntimeError: Migration checksum mismatch for 001_baseline:
recorded=<old sha256>, current=<new sha256>.
```

Provisioning records a SHA-256 of each migration file when it applies it, and
refuses to continue if the file has changed since. This is deliberate: silently
re-running an edited migration against a database that already has the old one
would leave the schema in a state nobody can reproduce. The guard fires on the
first mismatching file in filename order, which is not necessarily the one whose
change matters.

It costs nothing on a fresh database. It only bites when a migration already
recorded in `schema_migrations` has been edited, which for this repository
means a database provisioned before a release that touched the SQL. Under
Compose it stops `init-db`, and because the application services wait on
`service_completed_successfully`, none of them start.

Two ways out. On a development database with nothing worth keeping, drop the
volume and provision again:

```bash
docker compose -f deploy/docker-compose.yml down -v
docker compose -f deploy/docker-compose.yml up -d
```

On a database whose contents matter, confirm that the edit was cosmetic (a
comment, whitespace) and not a schema change, then record the new checksum so
the file is treated as already applied:

```bash
python - <<'EOF'
import hashlib, pathlib
f = pathlib.Path("src/hazard_assessment/storage/migrations/001_baseline.sql")
print(f.stem, hashlib.sha256(f.read_bytes()).hexdigest())
EOF
# then, as the admin role:
# UPDATE schema_migrations SET checksum = '<printed value>' WHERE version = '001_baseline';
```

Do not do this for an edit that changed DDL. In that case write a new migration
instead, so the change is applied rather than assumed.

### Guardrail Violation Error (HTTP 400)

The text you submitted (decision reason, escalation recommended action) contains one of the eight prohibited terms: "Warning," "Advisory," "Watch," "Information Statement," "Threat Message," "Cancellation," "All Clear," or "Bulletin." Rephrase to avoid these NOAA-reserved terms.

The check is deliberately broad, because it guards a boundary where a false
rejection costs a rephrase and a miss puts reserved wording in front of an
operator. Three consequences worth knowing before you write a decision reason:

- **Regular plurals count.** "Warnings", "Watches", "Advisories" and
  "Bulletins" are rejected, because that is how the reserved product names are
  most often written in running prose.
- **Ordinary English uses of these words are rejected too.** "No warnings from
  the QC pass" and "the duty scientist watches the sea-level feed" both trip
  the check. Say "no QC flags" and "monitors" instead.
- **Rejoining or repunctuating the words does not clear it.** "AllClear",
  "All-Clear", "All_Clear", "All/Clear" and their visually identical Unicode
  punctuation are all rejected the same way "All Clear" is.

### No DART Data After Startup

DART polling begins automatically. Allow 60-90 seconds after startup. If no records appear:
1. Check network connectivity to `www.ndbc.noaa.gov`.
2. Check ingest worker logs: `docker compose logs ingest-dart --tail 50`.
3. Verify station IDs are correct in the configuration.

A successful fetch that carries nothing new is the easiest case to misread,
because it looks identical to a healthy quiet feed. The connectors log every
health transition, so search for those first:

```bash
docker compose logs ingest-dart | grep -E "Connector dart:|Station [0-9]+:"
```

`ONLINE -> STALE` with the elapsed silence means the fetch worked and the feed
has stopped producing; `ONLINE -> OFFLINE` means the fetch itself failed and
the reason is on the same line. Nothing is logged while a feed is behaving, so
an empty result here means no transition has occurred since startup.

In standard mode a DART buoy transmits a 6-hour batch, so a gap of a few hours
between new rows is expected and is not by itself a fault. The pipeline
worker's `sensor_degraded` flag and the console's usable-station readout
(section 7.3) describe the same situation from the detector's side.

### Pipeline Worker Restarting In A Loop

The worker commits Kafka offsets only after a batch has been processed, and the
consume loop has no catch-all around that processing. A batch it cannot process
therefore ends the process: the offsets stay uncommitted, Docker restarts the
container, the same records are redelivered and the same failure repeats. The
container shows a climbing restart count and the logs a repeating traceback.

That is deliberate in one respect, which is that nothing is silently dropped;
the records are still in Kafka. It is unhelpful in another, which is that a
single unprocessable batch stops the pipeline for every event, not just the one
it belongs to. There is no dead-letter path for the consume loop today. Ingest
quarantines individual malformed records, but that happens before Kafka.

To confirm and recover:

```bash
docker inspect deploy-pipeline-worker-1 --format '{{.RestartCount}}'
docker compose logs pipeline-worker --tail 80        # read the repeating traceback
```

The traceback names the failing stage. If the batch is genuinely unprocessable
and the data is not needed, advancing the consumer group past it is a manual
Kafka operation and should be recorded, since it discards evidence. Note that
the Mission Control console cannot show this: the core API keeps serving the
last durable FSM state, and the console's core-API poll readout stays green
because the BFF and the API are both healthy. Worker liveness is not currently
surfaced anywhere; container restart count and worker logs are the signal.

### Anomaly Score Stays at 0.0

This typically means no DART data has been ingested yet, or the station windows do not hold enough accepted samples to score. QC never removes records from anomaly scoring; ingestion does reject unusable samples (missing-data sentinels of 9999.0 and above, non-finite or non-numeric values, duplicate timestamps). Check:

1. Records are arriving: `docker compose logs pipeline-worker --tail 50` and look for "Processing buffer" lines with nonzero record counts.
2. QC metadata for the affected stations:
```bash
curl -H "X-Hazard-Api-Key: <key>" "http://localhost:8000/api/audit?event_type=qc_complete&limit=20"
```
Each entry carries `n_records`, `n_usable`, and `min_station_confidence`. Low confidence does not block scoring; it is context for the operator.

### Tests Fail After Installation

```bash
pytest tests/ -v --tb=short
```

If import errors occur, verify the package was installed in the active virtual environment:
```bash
python -c "import hazard_assessment; print('OK')"
```

If the package is not found, reinstall:
```bash
pip install -e ".[dev]"
```

---

## 15. Simulation and System Verification

### 15.1 Simulation Module

The system includes a simplified analytic simulation module (`src/hazard_assessment/simulation/`) for validation and testing without real DART data. It is not a propagation model. Available scenarios:

| Script | Description |
|---|---|
| `scripts/run_physics_validation.py` | Runs 4 synthetic scenarios (M9.1 Tohoku-like, M7.2 moderate Pacific, meteotsunami false positive, degraded-mode) |
| `scripts/run_synthetic_pipeline.py` | Generates sliding-window score timelines and full inter-agent pipeline traces |
| `scripts/validate_tohoku.py` | Retrospective validation on archived 2011 Tohoku DART data |
| `scripts/validate_chile.py` | Retrospective validation on archived 2010 Chile DART data |
| `scripts/validate_illapel.py` | Retrospective validation on archived 2015 Illapel DART data |
| `scripts/validate_iquique.py` | Retrospective validation on archived 2014 Iquique DART data |
| `scripts/validate_samoa.py` | Retrospective validation on archived 2009 Samoa DART data |

To run the physics validation:
```bash
python scripts/run_physics_validation.py
```

Results are written to `results/` as JSON files.

### 15.2 End-to-End Verification

The E2E verification script tests the complete pipeline from data ingest through FSM escalation:

```bash
python scripts/verify_e2e_workflow.py
```

This exercises QC, anomaly detection, FSM transitions, scenario assessment, verification checks, and report generation in a single run with synthetic data.

### 15.3 Reproducing the paper artifacts

The evaluation results in `results/` and the figures in `paper/figures/` are regenerated by a single script:

```bash
pip install -e ".[dev,paper]"         # paper extra: matplotlib, cartopy, pandas
bash scripts/run_full_evaluation.sh   # all 17 steps
cd paper && tectonic paper.tex        # rebuilds paper.pdf
```

Three scripts sit outside `run_full_evaluation.sh` because they produce inputs rather than results. `scripts/download_coops_data.py` fetches the CO-OPS water-level and tide-prediction CSVs that the appendix de-tiding figures read, `scripts/archive_native_dart.py` keeps a copy of the raw NDBC payloads alongside the parsed CSVs, and `scripts/evaluate_llm_synthesis.py` produces the 20-scenario guardrail sweep the paper cites for the optional narrative layer. None of them is needed to reproduce the detection artifacts.

Steps 1 and 6 download NDBC historical archive data, so a full run needs network access. The remaining steps read the local CSVs under `data/`. The station-map figures use Cartopy Natural Earth features, which Cartopy fetches into its own cache the first time they render.

Most artifacts reproduce byte for byte from the checked-in data. Three do not, by design. `latency_profile.json` records hardware-bound timings. `physics_validation.json` and `synthetic_timelines.json` embed a `generated_at` wall-clock timestamp; ignoring that field, `synthetic_timelines.json` has shown floating-point reassociation differences up to 5.2e-15.

To check the detection artifacts without a full run:

```bash
pytest tests/artifacts/ -q
```

That replays each archived event plus the duplicate-sensitivity evaluation and fails if `results/*_detection.json` or `results/duplicate_sensitivity.json` no longer matches the detector. CI runs it on any change to the detector, the validation scripts, or those artifacts. Expect roughly one to two minutes per event.

Dependencies are declared with lower bounds rather than pinned versions, so an exact long-term replay would need a lock file or a container image that this release does not provide.

---

## 16. Glossary

**ABSTAIN**: A first-class system output used when required evidence is failed, missing, erroneous, or otherwise insufficient for the requested assessment. The live worker also fail-closes to ABSTAIN because it does not execute Scenario or Verification.

**AnomalyAssessment**: The Pydantic schema produced by the Anomaly Agent. Contains the ensemble anomaly score, component scores, filter degradation flag, and FSM state annotation.

**APPROVED_INTERNAL**: A `FinalAssessment` schema value used by the offline formatting path. Current caller-asserted live review does not produce this status or grant distribution authority.

**Audit trail**: The append-only log of all significant system events. No records can be modified or deleted.

**Bounded agency**: The design principle that each agent has an explicitly declared, limited set of permissions. In this system the declaration lives in the permission matrix, while the bounds are held by per-role database grants, the terminology guardrail scanner, fail-closed ABSTAIN routing, and the human review gate. Human escalation is required for all critical decisions.

**BOCPD**: Bayesian Online Changepoint Detection. A probabilistic method for detecting abrupt distributional shifts in time series, used to detect tsunami onset before the full waveform arrives.

**CO-OPS**: NOAA Center for Operational Oceanographic Products and Services. Provides coastal water-level observations.

**DART**: Deep-ocean Assessment and Reporting of Tsunamis. NOAA's network of bottom pressure recorder buoys for detecting tsunamis in the open ocean.

**Decision reason**: Required free text supplied with each caller-gated APPROVE, REJECT, or DEFER assessment review and stored in the append-only audit trail.

**Escalation packet of record**: Immutable reviewer evidence rendered by the worker from the exact persisted assessment that entered ESCALATE. Also called the reviewer packet; the two names refer to the same object. It carries packet and assessment identities, scientific hash, full assessment payload, and a canonical SHA-256.

**EscalationPacket**: Legacy Pydantic packet schema used by the process-memory generator path. It is not review authority; `/api/review` accepts only the durable packet of record.

**Event mode**: DART operational mode in which buoys transmit at high frequency (15-second or 1-minute intervals) in response to a detected pressure anomaly.

**FinalAssessment**: Terminal Pydantic schema used by offline pipeline formatting. Its status can be PROVISIONAL, ABSTAIN, or APPROVED_INTERNAL; this does not imply those transitions exist in the live worker.

**FSM**: Finite-State Machine. The deterministic orchestrator with five states (IDLE, MONITOR, INVESTIGATE, ASSESS, ESCALATE).

**Green's function**: A precomputed transfer function mapping a unit-source earthquake scenario to expected water-level observations at sensor stations. Used in NNLS inversion.

**Human Review Gate**: Mission Control interface for reading the durable packet of record and submitting a caller-gated assessment review. Current identity is caller-asserted; review does not mutate assessment or FSM state.

**InputRef**: A reference envelope containing source, record ID, URI, and SHA-256. Meaning of the hash depends on the producer. Live observation references use accepted payload hashes; offline replay uses canonical hashes derived from archived parsed fields and labels them accordingly.

**Isolation Forest**: A tree-based ensemble anomaly detection algorithm. Used as an optional ML component in the anomaly ensemble.

**Lineage**: The audit and decision chain for a pipeline execution, queryable by trace_id or event_id. Raw-input linkage is carried where recorded; a complete raw-observation-to-assessment join is not yet exposed through the API.

**NNLS**: Non-Negative Least Squares. The inversion algorithm used by the Scenario Agent to fit unit-source models to DART observations, constrained so that all source amplitudes are non-negative.

**PROVISIONAL**: Offline `FinalAssessment` status indicating that trusted approval has not been established. Current live caller-asserted review does not change it.

**QARTOD**: Quality Assurance of Real-Time Oceanographic Data. A set of standardized data quality tests maintained by the U.S. Integrated Ocean Observing System (IOOS).

**Scenario Agent**: The pipeline agent that performs NNLS inversion and uncertainty estimation.

**SIFT**: Short-term Inundation Forecasting for Tsunamis. NOAA's operational tsunami forecast tool based on a precomputed database of unit-source Green's functions.

**Station confidence**: A score in [0, 1] computed per observation as `c = 1 - (0.3 * n_suspect + 1.0 * n_fail) / n_tests`. SUSPECT flags carry 30% penalty; FAIL flags carry 100% penalty. Records below 0.5 are marked record_usable=false; the flag is advisory and anomaly scoring consumes all records regardless (genuine tsunami signals trip the same checks).

**T1, T2, T3**: Anomaly score thresholds that control FSM state transitions (default 0.35, 0.60, 0.85 respectively).

**Trace ID**: A UUID identifying a single end-to-end pipeline execution (one processed batch). The run-scoped audit entries share it: FSM state transitions, the seismic trigger's `seismic_provenance`, a seismic-only `abstain_triggered`, `qc_complete`, `anomaly_scored`, and the pipeline-node entries. The per-event observation-provenance entries (`input_provenance`, `provenance_capped`) are deliberately event-scoped, not trace-scoped: they accumulate across many batches and are queried by `event_id` (via `/api/lineage/event/{event_id}` and the escalation packet), so they appear under `__no_trace__` in a single-trace view.

**Verification Agent**: The pipeline agent that runs nine checks against the scenario assessment and enforces the ABSTAIN path.

**Wavelet score**: An anomaly component score derived from Daubechies-4 wavelet decomposition, measuring energy concentration in the tsunami frequency band (5-120 min periods).
