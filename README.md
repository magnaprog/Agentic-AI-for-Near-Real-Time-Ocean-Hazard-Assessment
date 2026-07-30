# Agentic AI for Near-Real-Time Ocean Hazard Assessment

This is an agentic AI system that monitors NOAA's deep-ocean tsunami buoys and tells a duty scientist when the ocean starts behaving as though a tsunami is passing. The system reads DART bottom-pressure data together with coastal water-level gauges and the USGS earthquake feed, scores how unusually each station is behaving, and proceeds through a finite state machine from monitoring to escalating the signals. Escalations are sent to a human for the decision; the system itself never issues an alert, approved or not, and none of what it produces is an official NOAA product. We invite the community to contribute and improve the project on platform engineering and the detection algorithms.

## Architecture

Data comes in through three small services that poll data from the DART buoys, NOAA's coastal tide gauges, and the USGS earthquake feed. Every reading is checked for the correct data format and plausible values. The system filters malformed signals out of the data streams into a quarantine table rather than dropping them quietly, so a bad batch can still be recovered afterwards. Everything else proceeds onto a message queue, which lets the collectors and the analysis run as separate parts that restart on their own.

The analysis worker reads from the queue and keeps a moving window of recent readings for each station: the readings are first checked against QARTOD, the IOOS quality-control standard for real-time ocean data, and the results are saved as notes. This prevents the filtering process from throwing away a real tsunami signal, which the checks alone cannot distinguish from a broken sensor.

The monitoring window is then scored from 0 to 1 for how anomalous it looks. Two detectors
carry that score in the shipped configuration. About 59 percent of it is an amplitude check
against the 3 cm level that DART itself uses as a trigger. The rest is statistical: one part
measures how much wave energy sits in the tsunami band (waves with periods between around 5
minutes and 2 hours), and the other detects the moment the signal's statistical behavior
shifts. A third detector, an Isolation Forest that would also weigh whether neighboring
stations see the same signal at the same time, is implemented but inactive, because no
trained model ships with this repository. Its weight is redistributed across the other two.

The anomaly score, along with whatever the earthquake feed has recently reported, is what the state machine uses to decide. The system moves between five states: IDLE, MONITOR, INVESTIGATE, ASSESS and ESCALATE, by comparing the score against fixed thresholds. The same inputs always produce the same transitions, so any run can be replayed and checked afterwards. When the worker reaches a step it cannot complete, it records an ABSTAIN, meaning not enough evidence to judge, instead of blindly guessing.

The heavier analysis, on the other hand, runs offline in `scripts/`, working out what seafloor movement would explain the waves recorded by fitting them against a set of precomputed wave patterns. A set of verification checks then looks for reasons to distrust that estimate, and the report is then generated. Currently, the library used in this repository uses synthetic waveforms for development and testing; a physics-based propagation database such as NOAA's NCTR set would be needed for operational use.

The system is composed of two web services: one holds the current state and the audit history; the other, Mission Control, polls it and serves the dashboard where a duty scientist can read the escalation and approve, reject, or defer the escalated signal. Underneath the system, a PostgreSQL database with the TimescaleDB extension stores readings, computed features, audit records, and assessments. The system also provides optional language-model steps, none of which change anything the deterministic pipeline decided. One writes a summary of a finished assessment after the report exists. Another reviews an event once it is over. The third runs while an event is still open: an investigator that picks its own read-only queries over the recorded evidence and answers three questions the pipeline does not, such as whether the other stations corroborate the one that crossed the threshold. Its findings are advice for the duty scientist and cannot re-enter the decision path, which is enforced by the database rather than by instructions in a prompt: findings are written under a role that only the investigator holds, and the role that drives the state machine cannot write them at all.

## Project Structure

```
├── src/hazard_assessment/  # Core assessment engine
│   ├── agents/             # QC, anomaly, scenario, verification, and report agents
│   ├── app.py              # Core FastAPI service, port 8000
│   ├── audit/              # Append-only audit logger
│   ├── config/             # Application settings
│   ├── data/               # Station coordinate tables
│   ├── geo.py              # Great-circle distance and tsunami travel time
│   ├── ingest/             # DART, CO-OPS, and USGS seismic connectors
│   ├── messaging/          # Kafka producer and consumer
│   ├── orchestrator/       # Deterministic state machine and pipeline runner
│   ├── policy/             # Output guardrails (enforcing) and the declared permission matrix
│   ├── schemas/            # Pydantic handoff schemas
│   ├── simulation/         # Simplified analytic tsunami simulation, not a propagation model
│   ├── storage/            # Database migrations and role provisioning
│   ├── telemetry/          # OpenTelemetry tracing and Prometheus metrics
│   ├── tidal.py            # Shared tidal constituent frequencies
│   └── workers/            # Worker process entrypoints
├── mission-control/        # Operator dashboard
│   ├── frontend/           # React and TypeScript app, built with Vite
│   └── backend/            # FastAPI service that proxies the core API
├── paper/                  # NOAA AI Workshop 2026 manuscript
│   ├── paper.tex           # Manuscript source
│   ├── paper.pdf           # Built manuscript
│   ├── references.bib      # Bibliography, 57 entries
│   ├── neurips_2025.sty    # Conference style file, third party
│   ├── drawio/             # Editable sources for the appendix UML diagrams
│   └── figures/            # Generated figures
├── data/                   # Archived records for five events plus a quiet control
├── results/                # Generated evaluation artifacts, 17 files
├── scripts/                # Download, validation, evaluation, figure generation
├── tests/                  # Unit, simulation, safety, artifact, integration tests
├── deploy/                 # Docker Compose stack and environment template
└── docs/                   # Documentation
    └── USER_MANUAL.md      # Operational manual
```

## Prerequisites

- Python 3.11 or newer
- Node.js 20 or newer, with npm, for the Mission Control frontend
- Docker and Docker Compose v2, for the infrastructure services

## Setup and Run

### 1. Core Assessment Engine

```bash
# Clone the repository
git clone https://github.com/magnaprog/Agentic-AI-for-Near-Real-Time-Ocean-Hazard-Assessment.git
cd Agentic-AI-for-Near-Real-Time-Ocean-Hazard-Assessment

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install the package with dev dependencies
pip install -e ".[dev]"

# Run the fast suites (about five minutes)
pytest tests/unit tests/simulation tests/safety

# Everything, including the artifact-reproduction suite, which rescores the
# archived events and takes roughly eight minutes on its own
pytest
```

The integration tests need PostgreSQL on `DB_HOST`/`DB_PORT`. Without one they
skip rather than fail, so a clean checkout passes with no database running.

### 2. Infrastructure Services (Docker)

```bash
cd deploy
cp .env.example .env
# Edit .env: at minimum set DB_ADMIN_PASSWORD, DB_PASSWORD,
# GRAFANA_PASSWORD, HAZARD_API_KEY, and MISSION_CONTROL_API_KEY.
# Optional: set per-role DB password overrides. Compose uses
# DB_ORCHESTRATOR_WRITER_PASSWORD for api-server,
# DB_PIPELINE_WORKER_PASSWORD for pipeline-worker, and
# DB_INGEST_WRITER_PASSWORD for ingest services.

docker compose up -d

# init-db service runs automatically and applies SQL migrations
```

That brings up twelve services. Every published port binds to `127.0.0.1` only.

| Service | Port | Purpose |
|---|---|---|
| `timescaledb` | 5432 | Time-series database, PostgreSQL with the TimescaleDB extension |
| `init-db` | none | One-shot migration service, safe to run repeatedly |
| `kafka` | 9092 | Message queue between ingest and the analysis worker |
| `prometheus` | 9090 | Metrics collection |
| `grafana` | 3000 | Dashboards and alerting |
| `jaeger` | 16686 | Distributed tracing, with OTLP on 4317 and 4318 |
| `api-server` | 8000 | Serves the state and audit trail the worker has persisted |
| `pipeline-worker` | none | Ingest, quality control, anomaly scoring, and the state machine. Abstains at ASSESS and ESCALATE, since scenario inversion, verification, and reporting run in the offline scripts |
| `ingest-dart` | none | DART buoy polling connector |
| `ingest-coops` | none | CO-OPS water-level polling connector |
| `ingest-seismic` | none | USGS FDSN event polling connector |
| `mission-control` | 8100 | Operator dashboard |

Running live needs both Kafka and a shared PostgreSQL database. The pipeline
worker owns the state machine and persists every transition; the API reads that
stored state. Without a shared database the API and the worker each keep their
own state in memory and drift apart, and without Kafka the ingest services
cannot reach the worker at all. Both degraded modes are fine for local
development and not suitable for anything else.

To re-run provisioning manually (safe to run multiple times):

```bash
cd deploy
docker compose run --rm init-db
```

### 3. Mission Control UI (local development)

To run the Mission Control dashboard locally for development:

```bash
# Terminal 1: Backend (FastAPI)
cd mission-control
pip install -r requirements.txt
export MISSION_CONTROL_HAZARD_API_URL=http://localhost:8000
# Must match the core API's HAZARD_API_KEY. If it expands empty, the backend
# still starts but serves the static demo snapshot instead of live data.
export MISSION_CONTROL_HAZARD_API_KEY="${HAZARD_API_KEY}"
export MISSION_CONTROL_API_KEY="<your_mission_control_api_key>"
uvicorn backend.main:app --reload --port 8100

# Terminal 2: Frontend (Vite dev server)
cd mission-control/frontend
npm install
npm run dev
```

The Vite dev server starts at `http://localhost:5173` with hot-reload enabled.
The backend API is available at `http://localhost:8100`.
The dashboard opens on an unlock screen: enter the Mission Control access key
(the same value as `MISSION_CONTROL_API_KEY`) to connect, together with a
reviewer ID, which attributes any review decision recorded in that session. The key authenticates HTTP and
WebSocket calls to the Mission Control backend and is held for the browser tab only.

### 4. Production Build (Mission Control)

```bash
cd mission-control

# Build the frontend
cd frontend && npm run build && cd ..

# Serve with the backend (frontend/dist is served as static files).
# MISSION_CONTROL_API_KEY is required at startup. For live data,
# MISSION_CONTROL_HAZARD_API_KEY must match the core API's HAZARD_API_KEY;
# without it the backend serves the static demo snapshot.
MISSION_CONTROL_API_KEY=<dashboard-key> MISSION_CONTROL_HAZARD_API_KEY=<core-api-key> \
  uvicorn backend.main:app --host 0.0.0.0 --port 8100
```

Or use Docker:

```bash
cd mission-control
docker build -t mission-control .
# MISSION_CONTROL_HAZARD_API_URL must point at the core API as reachable from inside the
# container (localhost inside the container is the container itself). On
# Linux, also pass --add-host=host.docker.internal:host-gateway.
docker run -p 8100:8100 \
  -e MISSION_CONTROL_API_KEY=<dashboard-key> -e MISSION_CONTROL_HAZARD_API_KEY=<core-api-key> \
  -e MISSION_CONTROL_HAZARD_API_URL=http://host.docker.internal:8000 \
  mission-control
```

## Environment Variables

Copy `deploy/.env.example` and configure:

| Variable                | Required | Description                                |
|-------------------------|----------|--------------------------------------------|
| `DB_ADMIN_USER`         | No       | TimescaleDB bootstrap/migration admin user for `init-db` (default: `hazard_admin`) |
| `DB_ADMIN_PASSWORD`     | Yes in Compose | TimescaleDB bootstrap/migration admin password for `init-db` |
| `DB_HOST`               | No       | Database host. API and worker disable database use when empty; Compose sets `timescaledb`. |
| `DB_PORT` / `DB_NAME`   | No       | Database port/name (defaults: `5432`, `hazard_assessment`). Compose pins both to the service, so setting them in `.env` has no effect there |
| `DB_CONNECT_TIMEOUT`    | No       | Database connection timeout in seconds (default: `10`) |
| `DB_PASSWORD`           | Yes in Compose | Fallback password for fixed runtime roles and provisioning |
| `DB_INGEST_WRITER_PASSWORD` | No  | Optional `ingest_writer` override used by ingest entrypoints |
| `DB_ORCHESTRATOR_WRITER_PASSWORD` | No | Optional `orchestrator_writer` override used by api-server |
| `DB_PIPELINE_WORKER_PASSWORD` | No | Optional `pipeline_worker` override used by pipeline-worker |
| `DB_INVESTIGATOR_WRITER_PASSWORD` | No | Password for the `investigator_writer` role, which `/api/investigate` uses to write findings. Required for that endpoint if it differs from `DB_PASSWORD` |
| `DB_AGENT_WRITER_PASSWORD` / `DB_AGENT_READER_PASSWORD` / `DB_AUDIT_READER_PASSWORD` | No | Optional provisioning overrides for offline/read roles |
| `HAZARD_API_KEY`         | Yes (for API access) | Shared internal API key used by core API (`X-Hazard-Api-Key`) and, by default in Docker Compose, Mission Control upstream calls |
| `MISSION_CONTROL_HAZARD_API_KEY`      | No       | Mission Control key for live core API auth. Not listed in `deploy/.env.example`: Compose supplies it from `HAZARD_API_KEY`. If empty, the Mission Control backend deliberately serves demo data. Live upstream failures never trigger demo fallback. |
| `MISSION_CONTROL_API_KEY`             | Yes (for dashboard access) | Mission Control backend API key required on HTTP `/api/mc/*` routes. The WebSocket accepts the same key by `mc-key.` subprotocol (what the browser uses, and the only transport that keeps it out of the access log), by header, or by `api_key` query parameter |
| `INGEST_DART_EVENT_MODE_TIMEOUT_SEC` | No | DART event-mode expiration timeout in seconds (default: `14400`) |
| `CALIBRATION_DIR`        | No       | Pipeline-worker station calibration directory; no calibrations load when empty |
| `KAFKA_BOOTSTRAP_SERVERS`| No       | Kafka broker address. Messaging is disabled when empty; no implicit localhost broker is used. Compose pins it to the `kafka` service. |
| `METRICS_PORT`           | No       | Worker Prometheus exporter port. Empty disables the exporter; Compose sets `9100` |
| `GRAFANA_PASSWORD`      | Yes in Compose | Grafana admin password |
| `LLM_PROVIDER`          | No       | Chat-model provider: `anthropic` (default), `openai`, or `google_genai`. Install the matching extra, e.g. `pip install -e ".[llm-openai]"` |
| `LLM_API_KEY`           | No       | Enables model-backed after-action analysis and optional offline commentary. After-action returns 501 while the layer is off; offline reports remain template-only. |
| `LLM_BASE_URL`          | No       | Endpoint override. With `LLM_PROVIDER=openai` this reaches any OpenAI-compatible server, so a locally served model needs no API key. Enables the layer on its own. |
| `LLM_MODEL`             | No       | Provider model identifier. No default; required whenever the layer is enabled |
| `APP_ENVIRONMENT`       | No       | Environment label set by Compose. No source code reads it; it does not make database or Kafka mandatory |
| `APP_LOG_LEVEL`         | No       | Worker logging level (default: `INFO`). Read by the pipeline and ingest workers only; the core API does not configure logging from it |

## Documentation

[docs/USER_MANUAL.md](docs/USER_MANUAL.md) is the operational reference: installation,
the API, the review flow, and troubleshooting.
Section 15.3 explains how to regenerate the results and figures in this repository. The
paper in `paper/` covers the architecture and the science behind it.

## Citation

If you use this framework or its results, please cite the accompanying paper:

Kevin Lee and Alison J. March. Agentic AI for Near-Real-Time Ocean Hazard Assessment. In the 8th NOAA AI Workshop, 2026.

```bibtex
@inproceedings{lee2026agentic,
  author    = {Lee, Kevin and March, Alison J.},
  title     = {Agentic {AI} for Near-Real-Time Ocean Hazard Assessment},
  booktitle = {8th NOAA AI Workshop},
  year      = {2026},
  note      = {\url{https://github.com/magnaprog/Agentic-AI-for-Near-Real-Time-Ocean-Hazard-Assessment}}
}
```

## License

[PolyForm Noncommercial License 1.0.0](LICENSE). Noncommercial use is free, including
for government, public safety, research, and educational organizations. Commercial use
needs written permission from the authors. The archived observations under `data/` are
United States Government works from NOAA and USGS and are not ours to license. Vendored
third-party files keep their own terms. The Unicode, Inc. confusables table in
`src/hazard_assessment/policy/_confusables.py` carries its copyright, license URL and
the UTS #39 version it came from. `paper/neurips_2025.sty` is the conference style
file, redistributed as conference templates customarily are; it credits its authors in
a header comment but states no license of its own, so we cannot grant terms for it.

This repository ships no dependency code: `node_modules/`, `dist/` and font binaries are
all untracked. A build does ship it, so the notices travel with whoever distributes one
rather than with the source. Two in the frontend bundle carry obligations worth knowing
before you distribute: the `@fontsource` families are OFL-1.1, which asks that its notice
travel with the fonts, and `react-leaflet` is Hippocratic-2.1, a non-OSI license adding
use restrictions of its own, which some legal reviews decline on principle. On the Python
side `psycopg` is LGPL, so keep it an ordinary installed dependency rather than vendoring
or statically bundling it. `mission-control/frontend/package-lock.json` and
`pyproject.toml` are the authoritative lists; the exact terms are whatever those resolve
to at the version you build.

## Contributing

Contributions are welcome. By opening a pull request you agree that your contribution
ships under the same license, and that the authors may also license it commercially.
