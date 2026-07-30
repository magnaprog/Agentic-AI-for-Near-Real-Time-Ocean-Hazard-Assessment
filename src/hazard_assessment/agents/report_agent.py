"""Report Agent - generates structured assessment reports with guardrails.

All output text passes through the alert-language guardrail scanner
before emission. The mandatory non-authoritative disclaimer is always
included. The deterministic template is always the emitted summary.
When an LLM API key is configured and system confidence is
above the routing threshold, the agent additionally produces an
LLM-synthesized narrative via a 4-node LangGraph synthesis graph,
stored separately as model_commentary and never replacing
deterministic report content.

"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.agents.report_templates import (
    collect_key_uncertainties,
    determine_confidence,
    render_tier_1,
    render_tier_2,
    render_tier_3,
)
from hazard_assessment.policy.guardrails import (
    NON_AUTHORITATIVE_DISCLAIMER,
    ScanResult,
    scan_text,
)
from hazard_assessment.schemas.envelope import DecisionStep, StepResult
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import VerificationOutcome, VerificationResult

if TYPE_CHECKING:
    from hazard_assessment.config.settings import ConfidenceWeights, LLMSettings

logger = logging.getLogger(__name__)

# System confidence routing thresholds (uncalibrated defaults -
# require tuning against historical event replay)
_CONFIDENCE_LLM_SKIP = 0.35
_CONFIDENCE_LEVEL_HIGH = 0.65
_CONFIDENCE_LEVEL_MODERATE = 0.40

# EnsembleSpread (StrEnum) -> numeric score for confidence formula
_SPREAD_SCORE: dict[str, float] = {"LOW": 1.0, "MODERATE": 0.5, "HIGH": 0.0}


def _compute_system_confidence(
    verification_outcome: VerificationOutcome | str,
    n_active_stations: int,
    ensemble_spread: EnsembleSpread,
    rayleigh_wave_suspect: bool,
    weights: ConfidenceWeights | None = None,
) -> float:
    """Compute a deterministic system-level confidence score.

    Uses observable pipeline state - never asks the LLM to self-report
    confidence (LLM self-confidence is poorly calibrated).

    Args:
        weights: Configurable weights for the confidence formula.
            Defaults to ConfidenceWeights() (0.40/0.25/0.35).

    Returns:
        Float in [0.0, 1.0].
    """
    if weights is None:
        from hazard_assessment.config.settings import ConfidenceWeights

        weights = ConfidenceWeights()

    outcome_score = {"PASS": 1.0, "PASS_WITH_CONCERNS": 0.6, "FAIL": 0.2}.get(
        verification_outcome, 0.2
    )
    # Station normalization: min(n/5, 1.0).
    # The divisor of 5 reflects the PTWC operational practice of using
    # ~3-6 DART stations for a well-constrained inversion (fewer for
    # near-field events, more for far-field). At 5 stations the coverage
    # score saturates - additional stations improve redundancy but don't
    # significantly change inversion quality for the unit-source approach.
    # This is an uncalibrated heuristic; tuning against historical replays
    # may yield a different saturation point.
    station_score = min(n_active_stations / 5.0, 1.0)
    spread_score = _SPREAD_SCORE.get(str(ensemble_spread), 0.0)
    # Rayleigh penalty: 0.7x multiplicative discount when Rayleigh wave
    # contamination is suspected. The ~30% reduction reflects the empirical
    # observation that DART pressure excursions within the Rayleigh wave
    # arrival window (typically 12-25 minutes post-earthquake for near-field
    # M7.5+ events) have ~30-40% false-positive rate as tsunami indicators.
    # This is conservative - it reduces confidence but does not suppress
    # the assessment entirely. Uncalibrated; requires tuning.
    rayleigh_penalty = 0.7 if rayleigh_wave_suspect else 1.0

    base = (
        weights.w_verification * outcome_score
        + weights.w_stations * station_score
        + weights.w_spread * spread_score
    )
    return min(max(base * rayleigh_penalty, 0.0), 1.0)


def _build_manifest(llm_enabled: bool) -> AgentManifest:
    """Build agent manifest, including INVOKE_LLM when LLM is active."""
    caps = [
        AgentCapability.READ_DATA,
        AgentCapability.WRITE_DATA,
        AgentCapability.WRITE_AUDIT,
        AgentCapability.PRODUCE_KAFKA,
        AgentCapability.CONSUME_KAFKA,
        AgentCapability.EMIT_REPORT,
    ]
    if llm_enabled:
        caps.append(AgentCapability.INVOKE_LLM)
    return AgentManifest(
        name="report_agent",
        version="0.3.0",
        capabilities=caps,
        description="Generates assessment reports with optional LLM synthesis",
    )


# Module-level manifest for app.py agent registry (LLM-disabled default)
_MANIFEST = _build_manifest(llm_enabled=False)


class GuardrailScanError(Exception):
    """Raised when report text fails the alert-language guardrail scan.

    Stores the full ScanResult for logging and diagnostics.
    """

    def __init__(self, scan_result: ScanResult) -> None:
        self.scan_result = scan_result
        terms = [v.term for v in scan_result.violations]
        super().__init__(
            f"Report text contains prohibited NOAA alert terminology: {terms}"
        )


class ReportAgent(BaseAgent):
    """Report Generation Agent.

    Synthesizes scenario results and verification status into a
    structured assessment report. All output text is scanned for
    prohibited NOAA alert terminology before emission.

    When ``llm_settings`` is provided with a non-empty API key, the
    agent builds a 4-node LangGraph synthesis graph at construction
    time. Its narrative is stored separately as ``model_commentary``;
    the deterministic template summary is emitted unchanged either way.
    """

    def __init__(
        self,
        *,
        llm_settings: LLMSettings | None = None,
        confidence_weights: ConfidenceWeights | None = None,
        audit_logger: Any | None = None,
    ) -> None:
        llm_enabled = bool(llm_settings and llm_settings.is_enabled)
        super().__init__(manifest=_build_manifest(llm_enabled))
        self._llm_settings = llm_settings
        self._confidence_weights = confidence_weights
        self._audit_logger = audit_logger
        self._synthesis_graph = None

        if llm_enabled and llm_settings is not None:
            try:
                from hazard_assessment.agents.llm_advisory.synthesis_graph import (
                    build_synthesis_graph,
                )

                # Compiled graph is reentrant: each invoke() creates its own
                # state/channel context.  No checkpointer is configured, so
                # no shared mutable state exists between calls.
                self._synthesis_graph = build_synthesis_graph(llm_settings)
                logger.info("LLM synthesis graph built successfully")
            except Exception:
                logger.exception("Failed to build synthesis graph; template-only mode")
                self._synthesis_graph = None

    def synthesize(
        self,
        scenario: ScenarioAssessment,
        verification: VerificationResult,
        *,
        tier: int = 1,
        provenance_bundle_id: UUID | None = None,
        rayleigh_wave_suspect: bool = False,
        fsm_state: str = "",
    ) -> FinalAssessment:
        """Generate a structured assessment report from scenario and verification data.

        Args:
            scenario: Scenario Agent output (ScenarioAssessment envelope).
            verification: Verification Agent output (VerificationResult envelope).
            tier: Report tier (1=Technical Brief, 2=Situational Summary, 3=Post-Event).
            provenance_bundle_id: UUID linking to full lineage record. Auto-generated
                if not provided.
            rayleigh_wave_suspect: From AnomalyAssessment - whether Rayleigh wave
                false trigger is suspected.

        Returns:
            FinalAssessment envelope with status=PROVISIONAL.

        Raises:
            ValueError: If tier is invalid, seismic-only with tier >= 2,
                or scenario/verification event_ids disagree.
            GuardrailScanError: If rendered text contains prohibited terminology.
        """
        if tier not in (1, 2, 3):
            raise ValueError(f"Invalid report tier: {tier}. Must be 1, 2, or 3.")

        # Event identity guard
        if (
            scenario.event_id is not None
            and verification.event_id is not None
            and scenario.event_id != verification.event_id
        ):
            raise ValueError(
                f"Scenario event_id ({scenario.event_id}) does not match "
                f"verification event_id ({verification.event_id}). "
                "Cannot generate a report combining data from different events."
            )

        # Seismic-only guard
        if scenario.constraint_stage == ConstraintStage.SEISMIC_ONLY and tier >= 2:
            raise ValueError(
                f"Seismic-only assessments are not distributable as Tier {tier}. "
                "Seismic-only output is a preliminary working document for the "
                "duty scientist only (Tier 1)."
            )

        # Deterministic system confidence (computed BEFORE any LLM call).
        # Drives: LLM routing skip (_CONFIDENCE_LLM_SKIP),
        #         FinalAssessment.confidence_level (HIGH/MODERATE/LOW),
        #         audit record.
        # NOTE: system_confidence and template_confidence are intentionally
        # separate signals. system_confidence uses an uncalibrated multi-factor
        # formula (verification outcome x station count x ensemble spread x
        # Rayleigh penalty). template_confidence uses a simpler lookup table
        # keyed on verification outcome + ensemble spread and drives the
        # rendered report prose. They may diverge - this is by design.
        # The rendered report (visible to the operator) reflects
        # template_confidence; the audit trail records system_confidence.
        # If unification is desired, replace determine_confidence() with a
        # mapping from system_confidence -> ConfidenceLevel.
        system_confidence = _compute_system_confidence(
            verification.overall,
            len(scenario.dart_stations_used),
            scenario.ensemble_spread,
            rayleigh_wave_suspect,
            weights=self._confidence_weights,
        )

        # Template confidence mapping (existing logic, preserved)
        template_confidence = determine_confidence(
            verification.overall, scenario.ensemble_spread
        )


        # Collect key uncertainties
        key_uncertainties = collect_key_uncertainties(
            verification.checks,
            scenario.ensemble_spread,
        )

        bundle_id = provenance_bundle_id or uuid4()

        # --- Template rendering (always computed first) ---
        template_summary = self._render_template(
            scenario, verification, tier, template_confidence, key_uncertainties, bundle_id
        )

        # --- LLM synthesis (optional, fail-closed) ---
        # the deterministic template is ALWAYS the emitted
        # summary. LLM output is stored separately as model_commentary
        # and never replaces or alters deterministic report content.
        summary = template_summary
        model_commentary: str | None = None
        llm_used = False

        # Graph-level timeout: the 4-node synthesis graph makes 3 LLM calls
        # (evidence, scenario, narrative). Each call has its own per-call
        # timeout (LLMSettings.timeout_sec), but there is no overall timeout
        # on the graph.invoke(). We add one here: 3x the per-call timeout
        # to cover all 3 LLM calls plus overhead. This prevents a slow LLM
        # provider from blocking the pipeline indefinitely.
        _graph_timeout_sec = (
            (self._llm_settings.timeout_sec * 3 + 10)
            if self._llm_settings
            else 100
        )

        if (
            self._synthesis_graph is not None
            and system_confidence >= _CONFIDENCE_LLM_SKIP
        ):
            import signal
            import time as _time

            _llm_start = _time.monotonic()
            _llm_success = False
            _has_sigalrm = hasattr(signal, "SIGALRM")
            # Bound BEFORE the try: signal.signal itself raises when called
            # off the main thread, and the finally below must not then hit
            # an UnboundLocalError (which would mask the original error and
            # break the fail-closed commentary omission). _alarm_set tracks whether the
            # alarm was actually armed (old_handler alone cannot: signal()
            # returns None for a non-Python-installed prior handler).
            old_handler = None
            _alarm_set = False

            try:
                if _has_sigalrm:

                    def _timeout_handler(signum: int, frame: Any) -> None:
                        raise TimeoutError(
                            f"LLM synthesis graph exceeded {_graph_timeout_sec}s timeout"
                        )

                    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
                    signal.alarm(_graph_timeout_sec)
                    _alarm_set = True

                state = self._build_synthesis_state(
                    scenario, verification, tier, system_confidence,
                    rayleigh_wave_suspect, fsm_state=fsm_state,
                )
                result = self._synthesis_graph.invoke(state)
                llm_narrative = result.get("narrative")
                if llm_narrative:
                    # Append mandatory disclaimer - the LLM is not
                    # expected to reproduce the exact string verbatim.
                    if NON_AUTHORITATIVE_DISCLAIMER not in llm_narrative:
                        llm_narrative = (
                            f"{llm_narrative}\n\n{NON_AUTHORITATIVE_DISCLAIMER}"
                        )
                    model_commentary = llm_narrative
                    llm_used = True
                    _llm_success = True
            except Exception as exc:
                logger.warning(
                    "LLM synthesis failed (%s: %s); commentary omitted",
                    type(exc).__name__,
                    str(exc)[:200],
                )
                model_commentary = None
            finally:
                # Cancel the alarm and restore the previous handler (Unix
                # only). Skipped entirely when signal.signal raised before
                # arming the alarm (nothing installed, nothing to restore).
                if _has_sigalrm and _alarm_set:
                    signal.alarm(0)
                    signal.signal(
                        signal.SIGALRM,
                        old_handler if old_handler is not None else signal.SIG_DFL,
                    )
                _llm_ms = (_time.monotonic() - _llm_start) * 1000
                if self._audit_logger is not None:
                    model_name = (
                        self._llm_settings.model if self._llm_settings else "unknown"
                    )
                    self._audit_logger.log_llm_call(
                        event_id=scenario.event_id,
                        agent="report_agent",
                        model=model_name,
                        prompt_tokens=None,  # not available from graph
                        response_tokens=None,  # not available from graph
                        latency_ms=_llm_ms,
                        success=_llm_success,
                    )

        # --- Guardrail scans (always applied) ---
        from hazard_assessment.telemetry.metrics import record_guardrail_scan

        # Commentary scan: a violating LLM narrative is dropped entirely.
        # The deterministic summary is unaffected.
        if model_commentary is not None:
            commentary_scan = scan_text(model_commentary)
            commentary_violations = (
                [v.term for v in commentary_scan.violations]
                if commentary_scan.violations
                else []
            )
            if self._audit_logger is not None:
                self._audit_logger.log_guardrail_scan(
                    event_id=scenario.event_id,
                    agent="report_agent",
                    text_length=len(model_commentary),
                    violations=commentary_violations,
                    passed=commentary_scan.passed,
                )
            record_guardrail_scan(passed=not commentary_scan.violations)
            # Reserved language is the drop condition, not ScanResult.passed.
            # `passed` also requires the non-authoritative disclaimer, which
            # belongs on the assembled report rather than on a commentary
            # fragment: the prompt never asks the model for it, so testing
            # `passed` here discarded every commentary, including clean ones,
            # and left llm_used permanently False. The metric on the line
            # above already uses this predicate.
            if commentary_scan.violations:
                logger.warning(
                    "LLM commentary failed guardrail scan; commentary dropped"
                )
                model_commentary = None
                llm_used = False

        # Deterministic summary scan: this text is always emitted, so a
        # violation here blocks the report outright.
        scan_result = scan_text(summary)
        violations = [v.term for v in scan_result.violations] if scan_result.violations else []
        if self._audit_logger is not None:
            self._audit_logger.log_guardrail_scan(
                event_id=scenario.event_id,
                agent="report_agent",
                text_length=len(summary),
                violations=violations,
                passed=scan_result.passed,
            )
        record_guardrail_scan(passed=not scan_result.violations)
        if not scan_result.passed:
            raise GuardrailScanError(scan_result)

        # Map system confidence -> ConfidenceLevel enum
        confidence_level = (
            ConfidenceLevel.HIGH if system_confidence >= _CONFIDENCE_LEVEL_HIGH else
            ConfidenceLevel.MODERATE if system_confidence >= _CONFIDENCE_LEVEL_MODERATE else
            ConfidenceLevel.LOW
        )

        trace = [
            DecisionStep(
                step="template_rendering",
                result=StepResult.PASS,
                evidence=f"Tier {tier} template rendered successfully",
            ),
            DecisionStep(
                step="guardrail_scan",
                result=StepResult.PASS,
                evidence="No prohibited NOAA alert terminology detected",
            ),
        ]
        if llm_used:
            trace.insert(
                1,
                DecisionStep(
                    step="llm_synthesis",
                    result=StepResult.PASS,
                    evidence="LangGraph synthesis graph completed successfully",
                ),
            )

        return FinalAssessment(
            producer=self.manifest.name,
            produced_at_utc=datetime.now(UTC),
            event_id=scenario.event_id or verification.event_id,
            decision_trace=trace,
            status=AssessmentStatus.PROVISIONAL,
            report_tier=tier,
            summary=summary,
            model_commentary=model_commentary,
            uncertainty=UncertaintyInfo(
                confidence_level=confidence_level,
                key_uncertainties=key_uncertainties,
            ),
            provenance_bundle_id=bundle_id,
            system_confidence=system_confidence,
            llm_synthesis_used=llm_used,
            rayleigh_wave_suspect=rayleigh_wave_suspect,
        )

    def _render_template(
        self,
        scenario: ScenarioAssessment,
        verification: VerificationResult,
        tier: int,
        confidence: ConfidenceLevel,
        key_uncertainties: list[str],
        bundle_id: UUID,
    ) -> str:
        """Render the deterministic template for the given tier."""
        if not scenario.top_scenarios:
            raise ValueError(
                "Cannot render report: scenario.top_scenarios is empty. "
                "At least one ranked scenario is required."
            )
        mw_best = scenario.top_scenarios[0].mw_equivalent

        if tier == 1:
            return render_tier_1(
                event_id=str(scenario.event_id or ""),
                constraint_stage=scenario.constraint_stage,
                mw_best=mw_best,
                overall_outcome=verification.overall,
                confidence=confidence,
                dart_stations_used=scenario.dart_stations_used,
                dart_stations_excluded=scenario.dart_stations_excluded,
                exclusion_reasons=scenario.exclusion_reasons,
                inversion_window_sec=scenario.inversion_window_sec,
                top_scenarios=scenario.top_scenarios,
                coastal_proxies=scenario.coastal_proxies,
                checks=verification.checks,
                key_uncertainties=key_uncertainties,
                limiting_assumptions=scenario.limiting_assumptions,
                ensemble_spread=scenario.ensemble_spread,
            )
        elif tier == 2:
            return render_tier_2(
                event_id=str(scenario.event_id or ""),
                constraint_stage=scenario.constraint_stage,
                mw_best=mw_best,
                overall_outcome=verification.overall,
                confidence=confidence,
                num_dart_stations=len(scenario.dart_stations_used),
                coastal_proxies=scenario.coastal_proxies,
                key_uncertainties=key_uncertainties,
                limiting_assumptions=scenario.limiting_assumptions,
                ensemble_spread=scenario.ensemble_spread,
            )
        else:  # tier == 3
            return render_tier_3(
                event_id=str(scenario.event_id or ""),
                constraint_stage=scenario.constraint_stage,
                mw_best=mw_best,
                overall_outcome=verification.overall,
                confidence=confidence,
                dart_stations_used=scenario.dart_stations_used,
                dart_stations_excluded=scenario.dart_stations_excluded,
                exclusion_reasons=scenario.exclusion_reasons,
                inversion_window_sec=scenario.inversion_window_sec,
                top_scenarios=scenario.top_scenarios,
                coastal_proxies=scenario.coastal_proxies,
                checks=verification.checks,
                key_uncertainties=key_uncertainties,
                limiting_assumptions=scenario.limiting_assumptions,
                ensemble_spread=scenario.ensemble_spread,
                provenance_bundle_id=str(bundle_id),
            )

    def _build_synthesis_state(
        self,
        scenario: ScenarioAssessment,
        verification: VerificationResult,
        tier: int,
        system_confidence: float,
        rayleigh_wave_suspect: bool,
        *,
        fsm_state: str = "",
    ) -> dict[str, Any]:
        """Build LLMSynthesisState dict for graph invocation."""
        if not scenario.top_scenarios:
            raise ValueError(
                "Cannot build synthesis state: scenario.top_scenarios is empty."
            )
        top = scenario.top_scenarios[0]
        return {
            "event_id": str(scenario.event_id or ""),
            "report_tier": tier,
            "fsm_state": fsm_state,
            "rayleigh_wave_suspect": rayleigh_wave_suspect,
            "top_scenario_json": json.dumps(top.model_dump(mode="json")),
            "verification_outcome": str(verification.overall),
            "station_count": len(scenario.dart_stations_used),
            "ensemble_spread": str(scenario.ensemble_spread),
            "system_confidence": system_confidence,
            "similar_events_json": "[]",
            "evidence_synthesis": None,
            "scenario_interpretation": None,
            "narrative": None,
        }
