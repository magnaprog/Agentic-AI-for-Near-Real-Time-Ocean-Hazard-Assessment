#!/usr/bin/env python3
"""Evaluate LLM synthesis quality across diverse scenario inputs.

Runs 20-30 synthesis calls with varied scenario parameters and
reports guardrail pass rate, latency, and output quality metrics.

Usage:
    python scripts/evaluate_llm_synthesis.py [--api-key KEY] [--n-calls 20]

Prerequisites:
    1. pip install -e .
    2. LLM_API_KEY environment variable or --api-key flag

Output:
    Prints a summary table with per-call metrics and aggregate statistics.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import UTC, datetime
from uuid import UUID, uuid4

from hazard_assessment.agents.report_agent import GuardrailScanError, ReportAgent
from hazard_assessment.config.settings import LLMSettings
from hazard_assessment.policy.guardrails import scan_text
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _make_scenario(
    mw: float = 8.5,
    n_stations: int = 3,
    spread: EnsembleSpread = EnsembleSpread.MODERATE,
    stage: ConstraintStage = ConstraintStage.DART_CONSTRAINED,
    event_id: UUID | None = None,
) -> ScenarioAssessment:
    """Build a synthetic ScenarioAssessment for testing."""
    stations = [f"4640{i}" for i in range(1, n_stations + 1)]
    return ScenarioAssessment(
        producer="test_harness",
        produced_at_utc=datetime.now(UTC),
        event_id=event_id or uuid4(),
        constraint_stage=stage,
        dart_stations_used=stations,
        dart_stations_excluded=[],
        exclusion_reasons={},
        inversion_window_sec=300,
        top_scenarios=[
            RankedScenario(
                unit_source_ids=["A01", "A02"],
                weights=[5.0, 3.0],
                waveform_rmse_cm=2.0,
                mw_equivalent=mw,
                rank=1,
                posterior_weight=1.0,
            ),
        ],
        coastal_proxies=[
            CoastalProxy(
                site_id="honolulu_hi",
                arrival_utc=datetime.now(UTC),
                arrival_uncertainty_min=5.0,
                amplitude_proxy_p10_m=0.3,
                amplitude_proxy_p50_m=0.5,
                amplitude_proxy_p90_m=0.8,
                tidal_correction_applied=False,
            ),
        ],
        ensemble_spread=spread,
        bilateral_rupture_evaluated=False,
        limiting_assumptions=["Wells & Coppersmith regression from continental data"],
    )


def _make_verification(
    outcome: VerificationOutcome = VerificationOutcome.PASS,
    event_id: UUID | None = None,
) -> VerificationResult:
    """Build a synthetic VerificationResult."""
    is_fail = outcome == VerificationOutcome.FAIL
    check_result = CheckResult.FAIL if is_fail else CheckResult.PASS
    return VerificationResult(
        producer="test_harness",
        produced_at_utc=datetime.now(UTC),
        event_id=event_id or uuid4(),
        overall=outcome,
        checks=[
            VerificationCheck(
                name="waveform_fit",
                result=check_result,
                evidence="Synthetic check for evaluation",
            ),
        ],
        abstain_required=is_fail,
        abstain_reason="Verification failed (synthetic)" if is_fail else None,
    )


# Diverse test scenarios covering different event characteristics
SCENARIOS = [
    {"mw": 7.0, "n_stations": 2, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS, "desc": "Small event, low spread"},
    {"mw": 8.0, "n_stations": 3, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Moderate event"},
    {"mw": 8.5, "n_stations": 4, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Large event, good coverage"},
    {"mw": 9.0, "n_stations": 5, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS, "desc": "Great earthquake, high spread"},
    {"mw": 7.5, "n_stations": 1, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS_WITH_CONCERNS, "desc": "Poor coverage"},
    {"mw": 8.0, "n_stations": 3, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS_WITH_CONCERNS, "desc": "Verification concerns"},
    {"mw": 7.0, "n_stations": 2, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Small event, 2 stations"},
    {"mw": 8.8, "n_stations": 4, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS, "desc": "Large event, tight ensemble"},
    {"mw": 7.2, "n_stations": 3, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS_WITH_CONCERNS, "desc": "Small, high uncertainty"},
    {"mw": 9.1, "n_stations": 5, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Tohoku-scale event"},
    {"mw": 7.8, "n_stations": 2, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Moderate, 2 stations"},
    {"mw": 8.3, "n_stations": 4, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS, "desc": "Large, low spread, 4 stations"},
    {"mw": 7.5, "n_stations": 3, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS, "desc": "Moderate, high uncertainty"},
    {"mw": 8.6, "n_stations": 5, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS_WITH_CONCERNS, "desc": "Large with concerns"},
    {"mw": 7.1, "n_stations": 1, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS, "desc": "Small, single station"},
    {"mw": 8.9, "n_stations": 4, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS, "desc": "Major event, tight fit"},
    {"mw": 7.6, "n_stations": 3, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Moderate Pacific event"},
    {"mw": 8.2, "n_stations": 2, "spread": EnsembleSpread.HIGH,
     "outcome": VerificationOutcome.PASS_WITH_CONCERNS, "desc": "Large, poor coverage"},
    {"mw": 7.3, "n_stations": 4, "spread": EnsembleSpread.LOW,
     "outcome": VerificationOutcome.PASS, "desc": "Small, good fit"},
    {"mw": 8.7, "n_stations": 5, "spread": EnsembleSpread.MODERATE,
     "outcome": VerificationOutcome.PASS, "desc": "Major with 5 stations"},
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate LLM synthesis quality")
    parser.add_argument("--api-key", type=str, default="", help="LLM provider API key")
    parser.add_argument("--n-calls", type=int, default=20, help="Number of calls (max 20)")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("LLM_API_KEY", "")
    if not api_key:
        logger.error("No API key provided. Set LLM_API_KEY or use --api-key.")
        return

    n_calls = min(args.n_calls, len(SCENARIOS))
    llm_settings = LLMSettings(api_key=api_key)
    agent = ReportAgent(llm_settings=llm_settings)

    if agent._synthesis_graph is None:
        logger.error("LLM synthesis graph failed to build. Check API key and dependencies.")
        return

    logger.info("Running %d synthesis evaluations...", n_calls)

    results: list[dict[str, object]] = []
    for i, params in enumerate(SCENARIOS[:n_calls]):
        shared_event_id = uuid4()
        scenario = _make_scenario(
            mw=params["mw"],
            n_stations=params["n_stations"],
            spread=params["spread"],
            event_id=shared_event_id,
        )
        verification = _make_verification(
            outcome=params["outcome"],
            event_id=shared_event_id,
        )

        start = time.perf_counter()
        try:
            fa = agent.synthesize(scenario, verification, tier=1)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            # Check guardrail on the operator-facing LLM text. The
            # narrative lives in model_commentary; the deterministic
            # summary is emitted unchanged either way.
            evaluated_text = (
                fa.model_commentary
                if fa.llm_synthesis_used and fa.model_commentary
                else fa.summary
            )
            scan = scan_text(evaluated_text)
            has_disclaimer = (
                "NOT an official" in evaluated_text
                or "non-authoritative" in evaluated_text.lower()
            )

            results.append({
                "idx": i + 1,
                "desc": params["desc"],
                "llm_used": fa.llm_synthesis_used,
                "guardrail_pass": scan.passed,
                "has_disclaimer": has_disclaimer,
                "summary_len": len(evaluated_text),
                "latency_ms": elapsed_ms,
                "error": None,
            })
            logger.info(
                "[%d/%d] %s - LLM=%s, guardrail=%s, %.0f ms",
                i + 1, n_calls, params["desc"],
                fa.llm_synthesis_used, scan.passed, elapsed_ms,
            )
        except GuardrailScanError as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append({
                "idx": i + 1,
                "desc": params["desc"],
                "llm_used": None,
                "guardrail_pass": False,
                "has_disclaimer": False,
                "summary_len": 0,
                "latency_ms": elapsed_ms,
                "error": f"GuardrailScanError: {e}",
            })
            logger.warning("[%d/%d] %s - GUARDRAIL FAIL: %s", i + 1, n_calls, params["desc"], e)
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            results.append({
                "idx": i + 1,
                "desc": params["desc"],
                "llm_used": None,
                "guardrail_pass": None,
                "has_disclaimer": False,
                "summary_len": 0,
                "latency_ms": elapsed_ms,
                "error": f"{type(e).__name__}: {str(e)[:100]}",
            })
            logger.error("[%d/%d] %s - ERROR: %s", i + 1, n_calls, params["desc"], e)

        time.sleep(0.5)  # Rate limiting

    # Print summary
    print(f"\n{'='*90}")
    print("LLM Synthesis Evaluation Summary")
    print(f"{'='*90}")
    print(
        f"{'#':>3} {'Description':<35} {'LLM':>5} {'Guard':>6} "
        f"{'Discl':>6} {'Len':>6} {'ms':>8} {'Error'}"
    )
    print("-" * 90)
    for r in results:
        llm = "yes" if r["llm_used"] else ("no" if r["llm_used"] is False else "?")
        guard = "PASS" if r["guardrail_pass"] else ("FAIL" if r["guardrail_pass"] is False else "?")
        discl = "yes" if r["has_disclaimer"] else "no"
        err = str(r["error"])[:20] if r["error"] else ""
        print(
            f"{r['idx']:>3} {r['desc']:<35} {llm:>5} {guard:>6} "
            f"{discl:>6} {r['summary_len']:>6} {r['latency_ms']:>8.0f} {err}"
        )

    # Aggregate stats
    n_total = len(results)
    n_llm = sum(1 for r in results if r["llm_used"])
    n_guard_pass = sum(1 for r in results if r["guardrail_pass"])
    n_disclaimer = sum(1 for r in results if r["has_disclaimer"])
    n_errors = sum(1 for r in results if r["error"])
    latencies = [r["latency_ms"] for r in results if r["error"] is None]

    print(f"\n{'Metric':<35} {'Value':>10}")
    print("-" * 45)
    print(f"{'Total calls':<35} {n_total:>10}")
    print(f"{'LLM synthesis used':<35} {n_llm:>10}")
    print(f"{'Template fallback':<35} {n_total - n_llm - n_errors:>10}")
    print(f"{'Guardrail pass rate':<35} {n_guard_pass}/{n_total}")
    print(f"{'Disclaimer present':<35} {n_disclaimer}/{n_total}")
    print(f"{'Errors':<35} {n_errors:>10}")
    if latencies:
        import statistics

        print(f"{'Median latency (ms)':<35} {statistics.median(latencies):>10.0f}")
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        print(f"{'p95 latency (ms)':<35} {latencies[min(p95_idx, len(latencies)-1)]:>10.0f}")

    success_rate = n_guard_pass / n_total * 100 if n_total > 0 else 0
    print(f"\nOverall success rate: {success_rate:.0f}%")


if __name__ == "__main__":
    main()
