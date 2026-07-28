"""Unit tests for the Verification Agent (integration).

Tests verify(), build_result(), decision trace, schema roundtrip,
and end-to-end verification flows.
"""

from __future__ import annotations

import numpy as np

from hazard_assessment.agents.verification_agent import VerificationAgent
from hazard_assessment.agents.verification_checks import (
    HoldoutData,
    StationPosition,
    VerificationInput,
)
from hazard_assessment.schemas.envelope import StepResult
from hazard_assessment.schemas.scenario import (
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckResult,
    PrerequisiteStatus,
    VerificationOutcome,
    VerificationResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ranked(**overrides) -> RankedScenario:
    defaults = {
        "unit_source_ids": ["A01"],
        "weights": [1.0],
        "waveform_rmse_cm": 1.0,
        "mw_equivalent": 8.0,
        "rank": 1,
        "posterior_weight": 0.8,
    }
    defaults.update(overrides)
    return RankedScenario(**defaults)


def _make_assessment(**overrides) -> ScenarioAssessment:
    defaults = {
        "producer": "scenario_agent",
        "constraint_stage": ConstraintStage.MULTI_STATION,
        "dart_stations_used": ["21413", "21414"],
        "dart_stations_excluded": [],
        "exclusion_reasons": {},
        "inversion_window_sec": 3600,
        "top_scenarios": [_make_ranked()],
        "ensemble_spread": EnsembleSpread.LOW,
        "bilateral_rupture_evaluated": True,
    }
    defaults.update(overrides)
    return ScenarioAssessment(**defaults)


def _make_vi(**overrides) -> VerificationInput:
    if "scenario" not in overrides:
        overrides["scenario"] = _make_assessment()
    scenario = overrides["scenario"]
    if "station_positions" not in overrides and len(scenario.dart_stations_used) >= 2:
        coordinates = [(10.0, 0.0), (-10.0, 0.0), (0.0, 10.0), (0.0, -10.0)]
        overrides["station_positions"] = [
            StationPosition(station_id, *coordinates[index % len(coordinates)])
            for index, station_id in enumerate(scenario.dart_stations_used)
        ]
        overrides.setdefault("epicenter_lat", 0.0)
        overrides.setdefault("epicenter_lon", 0.0)
    return VerificationInput(**overrides)


# ---------------------------------------------------------------------------
# Agent basics
# ---------------------------------------------------------------------------


class TestVerificationAgentManifest:
    def test_manifest_name(self) -> None:
        agent = VerificationAgent()
        assert agent.manifest.name == "verification_agent"


# ---------------------------------------------------------------------------
# verify()
# ---------------------------------------------------------------------------


class TestVerify:
    def test_default_inputs_are_incomplete(self) -> None:
        """Bare MULTI_STATION inputs (no inversion data, no Mw) leave
        REQUIRED checks without prerequisites: honestly INCOMPLETE with
        abstain, not a trivial pass."""
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)

        assert isinstance(result, VerificationResult)
        assert result.overall == VerificationOutcome.INCOMPLETE
        assert result.abstain_required
        assert result.abstain_reason is not None
        assert "sensitivity_analysis" in result.abstain_reason
        assert "physical_consistency" in result.abstain_reason
        assert len(result.checks) == 9
        by_name = {c.name: c for c in result.checks}
        assert (
            by_name["holdout_station_validation"].result
            == CheckResult.NOT_EVALUATED
        )
        assert (
            by_name["sensitivity_analysis"].prerequisite
            == PrerequisiteStatus.MISSING
        )
        # Checks with available inputs still evaluate normally.
        assert by_name["data_coverage"].result == CheckResult.PASS
        assert by_name["model_fit_quality"].result == CheckResult.PASS

    def test_fail_triggers_abstain(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            top_scenarios=[_make_ranked(waveform_rmse_cm=6.0)],
        )
        vi = _make_vi(scenario=scenario)
        result = agent.verify(vi)

        assert result.overall == VerificationOutcome.FAIL
        assert result.abstain_required
        assert result.abstain_reason is not None
        assert "model_fit_quality" in result.abstain_reason

    def test_concern_produces_pass_with_concerns(self) -> None:
        agent = VerificationAgent()
        # DART_CONSTRAINED: sensitivity is OPTIONAL, so the missing
        # inversion data degrades to concerns instead of INCOMPLETE.
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.DART_CONSTRAINED,
        )
        vi = _make_vi(scenario=scenario, mw_seismic=8.4)  # delta=0.4 -> CONCERN
        result = agent.verify(vi)

        assert result.overall == VerificationOutcome.PASS_WITH_CONCERNS
        assert not result.abstain_required

    def test_seismic_only_missing_mw_is_incomplete(self) -> None:
        """SEISMIC_ONLY requires the Mw comparison; without a seismic
        magnitude the prior-only scenario is unverifiable."""
        agent = VerificationAgent()
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
            top_scenarios=[_make_ranked(waveform_rmse_cm=0.0)],
        )
        vi = _make_vi(scenario=scenario)
        result = agent.verify(vi)

        assert result.overall == VerificationOutcome.INCOMPLETE
        assert result.abstain_required
        assert "physical_consistency" in (result.abstain_reason or "")

    def test_seismic_only_with_mw_is_pass_with_concerns(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
            top_scenarios=[_make_ranked(waveform_rmse_cm=0.0)],
        )
        vi = _make_vi(scenario=scenario, mw_seismic=8.1)
        result = agent.verify(vi)

        # Required Mw comparison satisfied; blocked OPTIONAL checks
        # degrade the outcome to PASS_WITH_CONCERNS.
        assert result.overall == VerificationOutcome.PASS_WITH_CONCERNS
        assert not result.abstain_required

    def test_multiple_failures(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            top_scenarios=[_make_ranked(waveform_rmse_cm=6.0)],
        )
        vi = _make_vi(
            scenario=scenario,
            mw_seismic=9.0,  # delta=1.0 -> FAIL
        )
        result = agent.verify(vi)

        assert result.overall == VerificationOutcome.FAIL
        assert result.abstain_required
        # Both checks should appear in reason
        assert "model_fit_quality" in result.abstain_reason
        assert "physical_consistency" in result.abstain_reason

    def test_holdout_fail_triggers_abstain(self) -> None:
        agent = VerificationAgent()
        holdout = HoldoutData(
            station_id="21413",
            observed_waveform=np.array([0.0, 0.01, 0.10]),
            predicted_waveform=np.array([0.0, 0.01, 0.20]),  # 100% amplitude error
            observed_arrival_index=None,
            predicted_arrival_index=None,
            time_step_sec=60.0,
        )
        vi = _make_vi(holdout=holdout)
        result = agent.verify(vi)

        assert result.overall == VerificationOutcome.FAIL
        assert result.abstain_required
        assert "holdout_station_validation" in result.abstain_reason


# ---------------------------------------------------------------------------
# Decision trace
# ---------------------------------------------------------------------------


class TestDecisionTrace:
    def test_trace_has_10_steps(self) -> None:
        """9 checks + 1 outcome policy step."""
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)
        assert len(result.decision_trace) == 10

    def test_trace_last_step_is_outcome_policy(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)
        assert result.decision_trace[-1].step == "outcome_policy"

    def test_pass_check_maps_to_pass_step(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)
        # First check is holdout_station_validation -> CONCERN (no holdout data)
        # which maps to WARN. Find a check that does map to PASS.
        pass_steps = [s for s in result.decision_trace if s.result == StepResult.PASS]
        assert len(pass_steps) > 0, "At least one check should produce PASS"

    def test_concern_maps_to_warn_step(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi(mw_seismic=8.4)  # CONCERN on physical_consistency
        result = agent.verify(vi)

        # Find the physical_consistency step
        phys_step = next(
            s for s in result.decision_trace if s.step == "physical_consistency"
        )
        assert phys_step.result == StepResult.WARN

    def test_fail_maps_to_fail_step(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            top_scenarios=[_make_ranked(waveform_rmse_cm=6.0)],
        )
        vi = _make_vi(scenario=scenario)
        result = agent.verify(vi)

        fit_step = next(
            s for s in result.decision_trace if s.step == "model_fit_quality"
        )
        assert fit_step.result == StepResult.FAIL

    def test_not_evaluated_maps_to_warn_step(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()  # no holdout data
        result = agent.verify(vi)
        holdout_step = next(
            s
            for s in result.decision_trace
            if s.step == "holdout_station_validation"
        )
        assert holdout_step.result == StepResult.WARN

    def test_not_applicable_maps_to_info_step(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            dart_stations_used=[],
        )
        vi = _make_vi(scenario=scenario, mw_seismic=8.1)
        result = agent.verify(vi)
        cov_step = next(
            s for s in result.decision_trace if s.step == "data_coverage"
        )
        assert cov_step.result == StepResult.INFO

    def test_incomplete_outcome_policy_step_is_fail(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()  # default MULTI_STATION -> INCOMPLETE
        result = agent.verify(vi)
        policy = result.decision_trace[-1]
        assert policy.step == "outcome_policy"
        assert policy.result == StepResult.FAIL
        assert "INCOMPLETE" in policy.evidence


# ---------------------------------------------------------------------------
# Schema roundtrip
# ---------------------------------------------------------------------------


class TestSchemaRoundtrip:
    def test_json_roundtrip(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)

        json_str = result.model_dump_json()
        restored = VerificationResult.model_validate_json(json_str)
        assert restored.overall == result.overall
        assert len(restored.checks) == len(result.checks)
        assert restored.abstain_required == result.abstain_required

    def test_dict_roundtrip(self) -> None:
        agent = VerificationAgent()
        scenario = _make_assessment(
            constraint_stage=ConstraintStage.DART_CONSTRAINED,
        )
        vi = _make_vi(scenario=scenario, mw_seismic=8.4)
        result = agent.verify(vi)

        data = result.model_dump()
        restored = VerificationResult.model_validate(data)
        assert restored.overall == VerificationOutcome.PASS_WITH_CONCERNS

    def test_producer_is_verification_agent(self) -> None:
        agent = VerificationAgent()
        vi = _make_vi()
        result = agent.verify(vi)
        assert result.producer == "verification_agent"

    def test_event_id_propagated(self) -> None:
        from uuid import uuid4

        event_id = uuid4()
        agent = VerificationAgent()
        vi = VerificationInput(scenario=_make_assessment(event_id=event_id))
        result = agent.verify(vi)
        assert result.event_id == event_id
