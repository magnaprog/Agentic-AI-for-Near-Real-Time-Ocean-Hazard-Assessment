"""Report template engine - deterministic text rendering for assessment reports.

Pure functions and template strings. No agent logic. Follows the
``scenario_inversion.py`` pattern (supporting module for the agent).

All templates include the mandatory non-authoritative disclaimer. Import-time
safety verification renders every template with safe placeholder values and
runs ``scan_text()`` - if any template contains prohibited terms by
construction, the module fails to import.

"""

from __future__ import annotations

from hazard_assessment.policy.guardrails import NON_AUTHORITATIVE_DISCLAIMER, scan_text
from hazard_assessment.schemas.final_assessment import ConfidenceLevel
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    VerificationCheck,
    VerificationOutcome,
)

# ---------------------------------------------------------------------------
# Confidence mapping
# ---------------------------------------------------------------------------

# Deterministic mapping: (VerificationOutcome, EnsembleSpread) -> ConfidenceLevel
#
# PASS + LOW spread     -> HIGH   (strong verification, tight ensemble)
# PASS + MODERATE/HIGH  -> MODERATE (strong verification, wider ensemble)
# PASS_WITH_CONCERNS + LOW -> MODERATE (concerns but tight ensemble)
# PASS_WITH_CONCERNS + MODERATE/HIGH -> LOW (concerns and wider ensemble)
# FAIL / INCOMPLETE -> never called (defense-in-depth)

_CONFIDENCE_MAP: dict[tuple[VerificationOutcome, EnsembleSpread], ConfidenceLevel] = {
    (VerificationOutcome.PASS, EnsembleSpread.LOW): ConfidenceLevel.HIGH,
    (VerificationOutcome.PASS, EnsembleSpread.MODERATE): ConfidenceLevel.MODERATE,
    (VerificationOutcome.PASS, EnsembleSpread.HIGH): ConfidenceLevel.MODERATE,
    (VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.LOW): ConfidenceLevel.MODERATE,
    (VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.MODERATE): ConfidenceLevel.LOW,
    (VerificationOutcome.PASS_WITH_CONCERNS, EnsembleSpread.HIGH): ConfidenceLevel.LOW,
}

_MAX_KEY_UNCERTAINTIES = 10


def determine_confidence(
    outcome: VerificationOutcome,
    spread: EnsembleSpread,
) -> ConfidenceLevel:
    """Map verification outcome and ensemble spread to confidence level.

    Raises ValueError for FAIL and INCOMPLETE outcomes - report
    generation should never be called when verification has failed or
    could not be completed (defense-in-depth).
    """
    if outcome in (VerificationOutcome.FAIL, VerificationOutcome.INCOMPLETE):
        raise ValueError(
            f"Cannot determine confidence for {outcome.value} verification "
            "outcome. Report generation must not be called when "
            "verification fails or is incomplete."
        )
    try:
        return _CONFIDENCE_MAP[(outcome, spread)]
    except KeyError:
        raise ValueError(
            f"No confidence mapping for ({outcome!r}, {spread!r}). "
            "Update _CONFIDENCE_MAP for new enum values."
        ) from None


# ---------------------------------------------------------------------------
# Key uncertainties collection
# ---------------------------------------------------------------------------


def collect_key_uncertainties(
    checks: list[VerificationCheck],
    spread: EnsembleSpread,
) -> list[str]:
    """Collect key uncertainty sources in priority order.

    Sources (in priority order):
    1. Verification checks with CONCERN result
    2. Ensemble spread note (if HIGH)

    Limiting assumptions are NOT included here - they have their own
    dedicated section in every tier template to avoid content duplication.

    Returns at most ``_MAX_KEY_UNCERTAINTIES`` items.
    """
    uncertainties: list[str] = []

    # Priority 1: verification concerns
    for check in checks:
        if check.result == CheckResult.CONCERN:
            uncertainties.append(f"Verification concern: {check.name}: {check.evidence}")

    # Priority 2: ensemble spread
    if spread == EnsembleSpread.HIGH:
        uncertainties.append(
            "High ensemble spread: scenario uncertainty is elevated"
        )

    return uncertainties[:_MAX_KEY_UNCERTAINTIES]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _format_coastal_section(proxies: list[CoastalProxy]) -> str:
    """Format coastal amplitude proxies into a text section."""
    if not proxies:
        return "No coastal amplitude proxies available for this assessment."
    lines = ["Coastal amplitude proxies (open-ocean, not inundation):"]
    for p in proxies:
        lines.append(
            f"  {p.site_id}: "
            f"P10={p.amplitude_proxy_p10_m:.2f} m, "
            f"P50={p.amplitude_proxy_p50_m:.2f} m, "
            f"P90={p.amplitude_proxy_p90_m:.2f} m "
            f"(arrival ~{p.arrival_utc.strftime('%Y-%m-%dT%H:%MZ')}"
            f" +/-{p.arrival_uncertainty_min:.0f} min)"
        )
    return "\n".join(lines)


def _format_top_scenarios_section(scenarios: list[RankedScenario]) -> str:
    """Format ranked scenarios into a text section (Tier 1 and Tier 3)."""
    lines = ["Ranked scenarios:"]
    for s in scenarios:
        sources = ", ".join(s.unit_source_ids)
        lines.append(
            f"  #{s.rank}: Mw {s.mw_equivalent:.1f}, "
            f"RMSE {s.waveform_rmse_cm:.2f} cm, "
            f"posterior weight {s.posterior_weight:.3f} "
            f"[{sources}]"
        )
    return "\n".join(lines)


def _format_assumptions(assumptions: list[str]) -> str:
    """Format limiting assumptions as a bulleted list."""
    if not assumptions:
        return "No limiting assumptions recorded."
    lines = ["Limiting assumptions:"]
    for a in assumptions:
        lines.append(f"  - {a}")
    return "\n".join(lines)


def _format_uncertainties(uncertainties: list[str]) -> str:
    """Format key uncertainties as a bulleted list."""
    if not uncertainties:
        return "No key uncertainties identified."
    lines = ["Key uncertainties:"]
    for u in uncertainties:
        lines.append(f"  - {u}")
    return "\n".join(lines)


def _format_verification_checks(checks: list[VerificationCheck]) -> str:
    """Format verification check results (Tier 1 and Tier 3)."""
    lines = ["Verification checks:"]
    for c in checks:
        lines.append(f"  [{c.result.value}] {c.name}: {c.evidence}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tier templates
# ---------------------------------------------------------------------------


def render_tier_1(
    *,
    event_id: str,
    constraint_stage: ConstraintStage,
    mw_best: float,
    overall_outcome: VerificationOutcome,
    confidence: ConfidenceLevel,
    dart_stations_used: list[str],
    dart_stations_excluded: list[str],
    exclusion_reasons: dict[str, str],
    inversion_window_sec: int,
    top_scenarios: list[RankedScenario],
    coastal_proxies: list[CoastalProxy],
    checks: list[VerificationCheck],
    key_uncertainties: list[str],
    limiting_assumptions: list[str],
    ensemble_spread: EnsembleSpread,
) -> str:
    """Render a Tier 1 Technical Brief for the duty scientist.

    Contains full detail: ranked scenarios, waveform RMSE, verification
    checks, coastal proxies. This is the primary internal working document.
    """
    stations_line = ", ".join(dart_stations_used) if dart_stations_used else "none"
    excluded_line = ", ".join(dart_stations_excluded) if dart_stations_excluded else "none"

    sections = [
        "TECHNICAL BRIEF",
        NON_AUTHORITATIVE_DISCLAIMER,
        "",
        f"Event: {event_id}",
        f"Constraint stage: {constraint_stage.value}",
        f"Best-fit magnitude: Mw {mw_best:.1f}",
        f"Verification: {overall_outcome.value}",
        f"Confidence: {confidence.value}",
        f"Ensemble spread: {ensemble_spread.value}",
        f"DART stations used: {stations_line}",
        f"DART stations excluded: {excluded_line}",
        f"Inversion window: {inversion_window_sec} s",
        "",
        _format_top_scenarios_section(top_scenarios),
        "",
        _format_coastal_section(coastal_proxies),
        "",
        _format_verification_checks(checks),
        "",
        _format_uncertainties(key_uncertainties),
        "",
        _format_assumptions(limiting_assumptions),
    ]

    if exclusion_reasons:
        lines = ["Station exclusion details:"]
        for sid, reason in exclusion_reasons.items():
            lines.append(f"  {sid}: {reason}")
        sections.append("")
        sections.append("\n".join(lines))

    return "\n".join(sections)


def render_tier_2(
    *,
    event_id: str,
    constraint_stage: ConstraintStage,
    mw_best: float,
    overall_outcome: VerificationOutcome,
    confidence: ConfidenceLevel,
    num_dart_stations: int,
    coastal_proxies: list[CoastalProxy],
    key_uncertainties: list[str],
    limiting_assumptions: list[str],
    ensemble_spread: EnsembleSpread,
) -> str:
    """Render a Tier 2 Situational Awareness Summary.

    Plain language for distribution to authorized recipients within the
    operational community. Omits ranked scenario table,
    per-station RMSE, and detailed verification checks.
    """
    # Plain-language summary of the best-fit scenario
    station_word = "station" if num_dart_stations == 1 else "stations"
    if constraint_stage in (ConstraintStage.MULTI_STATION, ConstraintStage.DART_CONSTRAINED):
        constraint_desc = f"constrained by {num_dart_stations} DART {station_word}"
    else:
        constraint_desc = "based on seismic data only (no DART constraint)"

    # Concerns language
    if overall_outcome == VerificationOutcome.PASS_WITH_CONCERNS:
        verification_note = (
            "Verification passed with concerns: see key uncertainties below."
        )
    else:
        verification_note = "Verification passed."

    sections = [
        "SITUATIONAL AWARENESS SUMMARY",
        NON_AUTHORITATIVE_DISCLAIMER,
        "",
        f"Event: {event_id}",
        "",
        f"Preliminary analysis indicates a magnitude Mw {mw_best:.1f} event, "
        f"{constraint_desc}.",
        f"Confidence: {confidence.value}. Ensemble spread: {ensemble_spread.value}.",
        verification_note,
        "",
        _format_coastal_section(coastal_proxies),
        "",
        _format_uncertainties(key_uncertainties),
        "",
        _format_assumptions(limiting_assumptions),
    ]

    return "\n".join(sections)


def render_tier_3(
    *,
    event_id: str,
    constraint_stage: ConstraintStage,
    mw_best: float,
    overall_outcome: VerificationOutcome,
    confidence: ConfidenceLevel,
    dart_stations_used: list[str],
    dart_stations_excluded: list[str],
    exclusion_reasons: dict[str, str],
    inversion_window_sec: int,
    top_scenarios: list[RankedScenario],
    coastal_proxies: list[CoastalProxy],
    checks: list[VerificationCheck],
    key_uncertainties: list[str],
    limiting_assumptions: list[str],
    ensemble_spread: EnsembleSpread,
    provenance_bundle_id: str,
) -> str:
    """Render a Tier 3 Post-Event Analysis.

    Full detail plus provenance reference and excluded station details.
    Generated 24-72 hours after event resolution.
    """
    stations_line = ", ".join(dart_stations_used) if dart_stations_used else "none"
    excluded_line = ", ".join(dart_stations_excluded) if dart_stations_excluded else "none"

    sections = [
        "POST-EVENT ANALYSIS",
        NON_AUTHORITATIVE_DISCLAIMER,
        "",
        f"Event: {event_id}",
        f"Provenance bundle: {provenance_bundle_id}",
        f"Constraint stage: {constraint_stage.value}",
        f"Best-fit magnitude: Mw {mw_best:.1f}",
        f"Verification: {overall_outcome.value}",
        f"Confidence: {confidence.value}",
        f"Ensemble spread: {ensemble_spread.value}",
        f"DART stations used: {stations_line}",
        f"DART stations excluded: {excluded_line}",
        f"Inversion window: {inversion_window_sec} s",
        "",
        _format_top_scenarios_section(top_scenarios),
        "",
        _format_coastal_section(coastal_proxies),
        "",
        _format_verification_checks(checks),
        "",
        _format_uncertainties(key_uncertainties),
        "",
        _format_assumptions(limiting_assumptions),
    ]

    if exclusion_reasons:
        lines = ["Station exclusion details:"]
        for sid, reason in exclusion_reasons.items():
            lines.append(f"  {sid}: {reason}")
        sections.append("")
        sections.append("\n".join(lines))

    return "\n".join(sections)


# ---------------------------------------------------------------------------
# Import-time safety verification
# ---------------------------------------------------------------------------


def _verify_template_safety() -> None:
    """Render all templates with safe placeholder values and scan for violations.

    Called at import time. If any *static template text* contains prohibited
    NOAA alert terminology, this raises ImportError - catching mistakes at
    development time. Note: this only verifies the template skeleton, not
    dynamic inputs (event IDs, evidence strings, etc.). Runtime guardrail
    scanning in ``ReportAgent.synthesize()`` via ``scan_text()`` provides
    the second layer of defense against prohibited terms in dynamic data.
    """
    from datetime import UTC, datetime

    safe_proxy = CoastalProxy(
        site_id="TEST_SITE",
        arrival_utc=datetime(2024, 1, 1, tzinfo=UTC),
        arrival_uncertainty_min=10.0,
        amplitude_proxy_p10_m=0.01,
        amplitude_proxy_p50_m=0.05,
        amplitude_proxy_p90_m=0.10,
        tidal_correction_applied=False,
    )
    safe_scenario = RankedScenario(
        unit_source_ids=["A01"],
        weights=[1.0],
        waveform_rmse_cm=0.5,
        mw_equivalent=7.5,
        rank=1,
        posterior_weight=0.8,
    )
    safe_check = VerificationCheck(
        name="test_check",
        result=CheckResult.PASS,
        evidence="Test evidence",
    )

    common = dict(
        event_id="TEST_EVENT",
        constraint_stage=ConstraintStage.DART_CONSTRAINED,
        mw_best=7.5,
        overall_outcome=VerificationOutcome.PASS,
        confidence=ConfidenceLevel.HIGH,
        key_uncertainties=["Test uncertainty"],
        limiting_assumptions=["Test assumption"],
        ensemble_spread=EnsembleSpread.LOW,
    )

    templates = {
        "Tier 1": render_tier_1(
            **common,  # type: ignore[arg-type]
            dart_stations_used=["D001"],
            dart_stations_excluded=["D003"],
            exclusion_reasons={"D003": "Noisy signal"},
            inversion_window_sec=1800,
            top_scenarios=[safe_scenario],
            coastal_proxies=[safe_proxy],
            checks=[safe_check],
        ),
        "Tier 2 (PASS)": render_tier_2(
            **common,  # type: ignore[arg-type]
            num_dart_stations=1,
            coastal_proxies=[safe_proxy],
        ),
        # Exercises the PASS_WITH_CONCERNS branch which renders different text
        "Tier 2 (PASS_WITH_CONCERNS)": render_tier_2(
            event_id="TEST_EVENT",
            constraint_stage=ConstraintStage.DART_CONSTRAINED,
            mw_best=7.5,
            overall_outcome=VerificationOutcome.PASS_WITH_CONCERNS,
            confidence=ConfidenceLevel.MODERATE,
            num_dart_stations=1,
            coastal_proxies=[safe_proxy],
            key_uncertainties=["Test uncertainty"],
            limiting_assumptions=["Test assumption"],
            ensemble_spread=EnsembleSpread.LOW,
        ),
        "Tier 3": render_tier_3(
            **common,  # type: ignore[arg-type]
            dart_stations_used=["D001"],
            dart_stations_excluded=["D002"],
            exclusion_reasons={"D002": "Insufficient data"},
            inversion_window_sec=1800,
            top_scenarios=[safe_scenario],
            coastal_proxies=[safe_proxy],
            checks=[safe_check],
            provenance_bundle_id="test-provenance-id",
        ),
    }

    for tier_name, rendered in templates.items():
        result = scan_text(rendered)
        if result.violations:
            terms = [v.term for v in result.violations]
            raise ImportError(
                f"{tier_name} template contains prohibited NOAA alert terminology "
                f"by construction: {terms}. Fix the template text."
            )
        if not result.has_disclaimer:
            raise ImportError(
                f"{tier_name} template is missing the mandatory "
                f"non-authoritative disclaimer."
            )


_verify_template_safety()
