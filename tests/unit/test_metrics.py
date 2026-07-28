"""Tests for Prometheus domain metrics (telemetry/metrics.py)."""

from __future__ import annotations

from hazard_assessment.telemetry import metrics as m


def _val(name: str, labels: dict[str, str] | None = None) -> float:
    return m.REGISTRY.get_sample_value(name, labels) or 0.0


def test_record_helpers_increment_counters() -> None:
    before = _val("hazard_ingest_records_total", {"outcome": "accepted"})
    m.record_ingest("accepted")
    assert _val("hazard_ingest_records_total", {"outcome": "accepted"}) == before + 1

    before_q = _val("hazard_ingest_records_total", {"outcome": "quarantined"})
    m.record_ingest("quarantined")
    assert (
        _val("hazard_ingest_records_total", {"outcome": "quarantined"}) == before_q + 1
    )

    before_a = _val("hazard_anomaly_scores_total")
    m.record_anomaly_score()
    assert _val("hazard_anomaly_scores_total") == before_a + 1

    before_t = _val("hazard_fsm_transitions_total", {"to_state": "MONITOR"})
    m.record_fsm_transition("MONITOR")
    assert _val("hazard_fsm_transitions_total", {"to_state": "MONITOR"}) == before_t + 1

    before_ab = _val("hazard_abstain_total")
    m.record_abstain()
    assert _val("hazard_abstain_total") == before_ab + 1

    before_v = _val("hazard_verification_outcomes_total", {"overall": "PASS"})
    m.record_verification_outcome("PASS")
    assert (
        _val("hazard_verification_outcomes_total", {"overall": "PASS"}) == before_v + 1
    )

    before_gv = _val("hazard_guardrail_scans_total", {"result": "violation"})
    m.record_guardrail_scan(passed=False)
    assert (
        _val("hazard_guardrail_scans_total", {"result": "violation"}) == before_gv + 1
    )
    before_gp = _val("hazard_guardrail_scans_total", {"result": "pass"})
    m.record_guardrail_scan(passed=True)
    assert _val("hazard_guardrail_scans_total", {"result": "pass"}) == before_gp + 1

    before_h = _val("hazard_station_scoring_duration_seconds_count")
    m.observe_station_scoring_duration(0.01)
    assert _val("hazard_station_scoring_duration_seconds_count") == before_h + 1


def test_metrics_response_includes_counters_and_gauge() -> None:
    m.record_ingest("accepted")
    response = m.generate_metrics_response("MONITOR")
    body = response.body.decode()
    assert "hazard_ingest_records_total" in body
    assert "hazard_fsm_current_state" in body
