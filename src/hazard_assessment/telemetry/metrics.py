"""Prometheus domain metrics for the ocean hazard assessment system.

Exposes system-specific metrics via prometheus_client. The /metrics
endpoint in app.py calls generate_metrics_response() which updates
gauge values and returns the Prometheus text format.
"""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from starlette.responses import Response

_logger = logging.getLogger(__name__)

# Use a custom registry to avoid default process/platform collectors
# that may not be meaningful in a container environment.
REGISTRY = CollectorRegistry()

# Current FSM state (set on each /metrics scrape by generate_metrics_response).
fsm_state_gauge = Gauge(
    "hazard_fsm_current_state",
    "Current FSM state (1=active for the labeled state)",
    ["state"],
    registry=REGISTRY,
)

# Domain counters. These are incremented at their event sites (ingest runner,
# pipeline worker, FSM). NOTE: counters incremented in the worker process are
# exposed by the worker's own exporter (start_metrics_exporter); the API's
# /metrics endpoint only reflects counters incremented in the API process.
ingest_records_total = Counter(
    "hazard_ingest_records",
    "Ingest records processed, by outcome (accepted or quarantined)",
    ["outcome"],
    registry=REGISTRY,
)
anomaly_scores_total = Counter(
    "hazard_anomaly_scores",
    "Station anomaly scores computed by the worker",
    registry=REGISTRY,
)
fsm_transitions_total = Counter(
    "hazard_fsm_transitions",
    "FSM state transitions, by destination state",
    ["to_state"],
    registry=REGISTRY,
)
abstain_total = Counter(
    "hazard_abstain",
    "ABSTAIN decisions emitted by the pipeline",
    registry=REGISTRY,
)
verification_outcomes_total = Counter(
    "hazard_verification_outcomes",
    "Verification outcomes (PASS/PASS_WITH_CONCERNS/INCOMPLETE/FAIL)",
    ["overall"],
    registry=REGISTRY,
)
guardrail_scans_total = Counter(
    "hazard_guardrail_scans",
    "Alert-language guardrail scans on emitted text, by result (pass/violation)",
    ["result"],
    registry=REGISTRY,
)

lineage_persist_failures_total = Counter(
    "hazard_lineage_persist_failures",
    "processed_features lineage rows that failed to persist (best-effort "
    "path: the pipeline continues, but dropped provenance must be "
    "operator-visible, not log-only)",
    registry=REGISTRY,
)

# Latency of one station's anomaly scoring (the dominant per-station compute:
# detiding, bandpass, wavelet, BOCPD, isolation forest, spatial coherence). A
# histogram so operators can watch the near-real-time scoring latency
# distribution and catch performance regressions. Default prometheus_client
# buckets (5 ms to 10 s) span the expected range.
station_scoring_duration_seconds = Histogram(
    "hazard_station_scoring_duration_seconds",
    "Wall-clock duration of one station's anomaly scoring, successful scores "
    "only (process_station_data)",
    registry=REGISTRY,
)

assessment_gaps_total = Counter(
    "hazard_assessment_gaps",
    "Checkpoints with an active event whose OceanEvidenceAssessment could "
    "not be built or persisted (deterministic processing continues, "
    "the gap must be operator-visible, and the model is never invoked for "
    "that checkpoint)",
    registry=REGISTRY,
)


def record_lineage_persist_failure() -> None:
    """Count a failed processed_features lineage insert."""
    try:
        lineage_persist_failures_total.inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_lineage_persist_failure metric failed", exc_info=True)


def record_assessment_gap() -> None:
    """Count a checkpoint that produced no durable assessment."""
    try:
        assessment_gaps_total.inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_assessment_gap metric failed", exc_info=True)


def record_ingest(outcome: str) -> None:
    """Count an ingest record by outcome ('accepted' or 'quarantined').

    'accepted' means the record passed validation and was processed
    (persisted/queued); Kafka delivery success is not separately confirmed.
    """
    try:
        ingest_records_total.labels(outcome=outcome).inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_ingest metric failed", exc_info=True)


def record_anomaly_score() -> None:
    """Count a station anomaly score computation."""
    try:
        anomaly_scores_total.inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_anomaly_score metric failed", exc_info=True)


def record_fsm_transition(to_state: str) -> None:
    """Count an FSM transition into ``to_state``."""
    try:
        fsm_transitions_total.labels(to_state=to_state).inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_fsm_transition metric failed", exc_info=True)


def record_abstain() -> None:
    """Count a deliberate ABSTAIN decision emitted by the system.

    Incremented at the two intentional ABSTAIN sites: the verification-driven
    abstain_node, and the live worker's seismic-only transition (a seismic
    trigger moved the FSM but no station window was scored). Rare pipeline-error
    fallbacks that also emit ABSTAIN are not counted here - they are an error
    mode, not a deliberate ABSTAIN decision.
    """
    try:
        abstain_total.inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_abstain metric failed", exc_info=True)


def record_verification_outcome(overall: str) -> None:
    """Count a verification outcome by its overall result.

    Incremented once per pipeline verification at the verify_node chokepoint, so
    each PASS / PASS_WITH_CONCERNS / FAIL is counted exactly once. A FAIL here
    also drives an ABSTAIN (record_abstain, a separate counter); the two moving
    together for verification-driven abstains is expected, not double-counting.
    Most increments occur in the offline/script path, since the live worker
    fail-closes before reaching verify_node.
    """
    try:
        verification_outcomes_total.labels(overall=overall).inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_verification_outcome metric failed", exc_info=True)


def record_guardrail_scan(passed: bool) -> None:
    """Count an alert-language guardrail scan by result.

    ``passed=False`` (result="violation") means the scan found prohibited
    terminology, which then blocks or redacts the text at the call site. Called
    at the runtime emission scan sites (report generation, escalation packet,
    after-action narrative, and the ABSTAIN / human-decision formatters), not
    the import-time template self-tests.
    """
    try:
        guardrail_scans_total.labels(result="pass" if passed else "violation").inc()
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("record_guardrail_scan metric failed", exc_info=True)


def observe_station_scoring_duration(seconds: float) -> None:
    """Observe the wall-clock duration of one station's anomaly scoring."""
    try:
        station_scoring_duration_seconds.observe(seconds)
    except Exception:  # pragma: no cover - metrics must never disrupt processing
        _logger.debug("observe_station_scoring_duration metric failed", exc_info=True)


def start_metrics_exporter(port: int) -> None:
    """Start a Prometheus HTTP exporter on ``port`` using ``REGISTRY``.

    Used by the worker processes (the API exposes ``/metrics`` directly via
    ``generate_metrics_response``). Best-effort: logs and continues on failure
    so a metrics problem never breaks ingestion or processing.
    """
    try:
        from prometheus_client import start_http_server

        start_http_server(port, registry=REGISTRY)
        _logger.info("Prometheus metrics exporter listening on port %d", port)
    except Exception:
        _logger.exception("Failed to start metrics exporter")

def generate_metrics_response(current_fsm_state: str) -> Any:
    """Update FSM gauge and return Prometheus text-format response."""
    # Import here to avoid circular import; uses the canonical enum.
    from hazard_assessment.orchestrator.states import SystemState

    for state in SystemState:
        fsm_state_gauge.labels(state=state.value).set(
            1.0 if state.value == current_fsm_state else 0.0
        )
    return Response(
        content=generate_latest(REGISTRY),
        media_type=CONTENT_TYPE_LATEST,
    )
