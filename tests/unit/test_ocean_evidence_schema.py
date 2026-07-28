"""Conformance tests for the deterministic ocean evidence schema.

Covers registry exhaustiveness, the
pure derivation functions, per-model validators, and the self-enforcing
top-level assessment. Every scientific status must be either a typed
input fact or re-derived by a validator, so these tests exercise both
the accept and the reject direction of each rule.
"""

from __future__ import annotations

from datetime import timedelta
from uuid import UUID

import pytest
from pydantic import ValidationError

from hazard_assessment.orchestrator.states import SystemState
from hazard_assessment.schemas.envelope import DataSource
from hazard_assessment.schemas.ocean_evidence import (
    ASSESSMENT_CONDITION_REGISTRY,
    LIMITATION_CODE_EFFECTS,
    REGISTRY_AXES,
    AnalysisCapability,
    AnalysisExecution,
    AnalysisQuality,
    AnalysisStatus,
    CalibrationStatus,
    ConfounderApplicability,
    ConfounderCheck,
    ConfounderPrerequisiteStatus,
    ConfounderResultValue,
    CurrentRecordAdmissionStatus,
    DartDataMode,
    DetectorConfig,
    EvaluationEffect,
    EventSeismicContext,
    FsmState,
    LimitationCode,
    MixedProductCondition,
    ObservationBounds,
    OceanEvidenceAssessment,
    PipelineOutcome,
    ProvenanceStatus,
    QCAggregateCondition,
    QCExecutionStatus,
    QCFlagCounts,
    SeismicContextClass,
    SourceRollup,
    SourceRollupStatus,
    StationEvaluationStatus,
    StationLimitation,
    StationManifestCondition,
    StationManifestRef,
    StationScope,
    StationScoringStatus,
    ThresholdEvaluation,
    ThresholdResultValue,
    derive_evaluation_status,
    derive_highest_tier,
    derive_pipeline_outcome,
    derive_qc_aggregate_condition,
    derive_source_rollup,
    derive_source_time_bounds,
    derive_threshold_result_value,
    registry_lookup,
)
from hazard_assessment.schemas.qc import DataMode
from tests.unit._ocean_evidence_fixtures import (
    SHA_A,
    SHA_C,
    T0,
    W_COOPS,
    W_DART,
    assessment_kwargs,
    make_assessment,
    make_coops_entry,
    make_dart_entry,
    make_detector_config,
    make_provenance,
    make_qc,
)


class TestConditionRegistry:
    def test_every_axis_member_is_mapped(self) -> None:
        missing = [
            (axis, member.value)
            for axis, enum_cls in REGISTRY_AXES.items()
            for member in enum_cls
            if (axis, member.value) not in ASSESSMENT_CONDITION_REGISTRY
        ]
        assert missing == []

    def test_no_orphan_registry_keys(self) -> None:
        valid_keys = {
            (axis, member.value)
            for axis, enum_cls in REGISTRY_AXES.items()
            for member in enum_cls
        }
        assert set(ASSESSMENT_CONDITION_REGISTRY) == valid_keys

    def test_every_limitation_code_has_exactly_one_effect(self) -> None:
        # Every declared code is produced by the registry, and no code
        # appears with two different effects.
        assert set(LIMITATION_CODE_EFFECTS) == set(LimitationCode)
        recomputed: dict[LimitationCode, set[EvaluationEffect]] = {}
        for mapping in ASSESSMENT_CONDITION_REGISTRY.values():
            if mapping is not None:
                recomputed.setdefault(mapping[0], set()).add(mapping[1])
        assert all(len(effects) == 1 for effects in recomputed.values())

    def test_lookup_unmapped_condition_raises(self) -> None:
        with pytest.raises(KeyError):
            registry_lookup("CalibrationStatus", "NO_SUCH_VALUE")
        with pytest.raises(KeyError):
            registry_lookup("NoSuchAxis", "NOMINAL")

    def test_qc_never_prevents_evaluation(self) -> None:
        # QC is metadata, not a filter.
        for axis in ("QCExecutionStatus", "QCAggregateCondition"):
            for (a, _), mapping in ASSESSMENT_CONDITION_REGISTRY.items():
                if a == axis and mapping is not None:
                    assert mapping[1] is not EvaluationEffect.PREVENTS_EVALUATION

    def test_only_calibration_prevents_evaluation(self) -> None:
        preventing = {
            key
            for key, mapping in ASSESSMENT_CONDITION_REGISTRY.items()
            if mapping is not None
            and mapping[1] is EvaluationEffect.PREVENTS_EVALUATION
        }
        assert preventing == {
            ("CalibrationStatus", "UNAVAILABLE"),
            ("CalibrationStatus", "ERROR"),
        }


class TestEnumParity:
    def test_fsm_state_mirrors_orchestrator_system_state(self) -> None:
        assert {(m.name, m.value) for m in FsmState} == {
            (m.name, m.value) for m in SystemState
        }

    def test_dart_data_mode_is_not_qc_data_mode(self) -> None:
        # Assessments must express UNKNOWN; the QC enum cannot.
        assert "UNKNOWN" in DartDataMode.__members__
        assert "UNKNOWN" not in DataMode.__members__


class TestDerivations:
    def test_evaluation_status_requires_successful_scoring(self) -> None:
        for status in (
            StationScoringStatus.NO_RETAINED_DATA,
            StationScoringStatus.INSUFFICIENT_RETAINED_DATA,
            StationScoringStatus.SCORING_FAILED,
        ):
            assert (
                derive_evaluation_status(status, [])
                is StationEvaluationStatus.NOT_EVALUATED
            )

    def test_evaluation_status_effect_precedence(self) -> None:
        ok = StationScoringStatus.SCORING_SUCCEEDED
        assert derive_evaluation_status(ok, []) is StationEvaluationStatus.EVALUATED
        assert (
            derive_evaluation_status(ok, [EvaluationEffect.INFORMATIONAL])
            is StationEvaluationStatus.EVALUATED
        )
        assert (
            derive_evaluation_status(
                ok,
                [EvaluationEffect.INFORMATIONAL, EvaluationEffect.LIMITS_INTERPRETATION],
            )
            is StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS
        )
        assert (
            derive_evaluation_status(
                ok,
                [
                    EvaluationEffect.LIMITS_INTERPRETATION,
                    EvaluationEffect.PREVENTS_EVALUATION,
                ],
            )
            is StationEvaluationStatus.NOT_EVALUATED
        )

    def test_threshold_result_inclusive_comparison(self) -> None:
        evaluated = StationEvaluationStatus.EVALUATED
        met = ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET
        no_met = (
            ThresholdResultValue.NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW
        )
        assert derive_threshold_result_value(evaluated, 0.35, 0.35) is met
        assert derive_threshold_result_value(evaluated, 0.349, 0.35) is no_met
        assert (
            derive_threshold_result_value(
                StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS, 0.9, 0.35
            )
            is met
        )

    def test_threshold_result_not_evaluated_paths(self) -> None:
        not_evaluated = ThresholdResultValue.NOT_EVALUATED
        assert (
            derive_threshold_result_value(
                StationEvaluationStatus.NOT_EVALUATED, 0.9, 0.35
            )
            is not_evaluated
        )
        assert (
            derive_threshold_result_value(StationEvaluationStatus.EVALUATED, None, 0.35)
            is not_evaluated
        )

    def test_highest_tier_inclusive_boundaries(self) -> None:
        assert derive_highest_tier(0.85, 0.35, 0.60, 0.85) == "T3"
        assert derive_highest_tier(0.60, 0.35, 0.60, 0.85) == "T2"
        assert derive_highest_tier(0.35, 0.35, 0.60, 0.85) == "T1"
        assert derive_highest_tier(0.34, 0.35, 0.60, 0.85) is None

    def test_qc_aggregate_severity_order(self) -> None:
        def counts(**kwargs: int) -> QCFlagCounts:
            base = {
                "n_pass": 0,
                "n_not_evaluated": 0,
                "n_suspect": 0,
                "n_fail": 0,
                "n_missing": 0,
            }
            base.update(kwargs)
            return QCFlagCounts(**base)

        agg = derive_qc_aggregate_condition
        assert agg(counts()) is QCAggregateCondition.NO_RETAINED_QC
        assert agg(counts(n_pass=5)) is QCAggregateCondition.WORST_FLAG_PASS
        assert (
            agg(counts(n_pass=5, n_not_evaluated=1))
            is QCAggregateCondition.WORST_FLAG_NOT_EVALUATED
        )
        assert (
            agg(counts(n_not_evaluated=1, n_suspect=1))
            is QCAggregateCondition.WORST_FLAG_SUSPECT
        )
        assert (
            agg(counts(n_suspect=1, n_missing=1))
            is QCAggregateCondition.WORST_FLAG_MISSING
        )
        assert (
            agg(counts(n_missing=1, n_fail=1))
            is QCAggregateCondition.WORST_FLAG_FAIL
        )

    def test_pipeline_outcome_classification(self) -> None:
        derive = derive_pipeline_outcome
        # A seismic-only transition without a scored window abstains
        # regardless of any other field.
        assert (
            derive("verified_pending_report", FsmState.ASSESS, True)
            is PipelineOutcome.ABSTAIN
        )
        assert derive("abstain", FsmState.ASSESS) is PipelineOutcome.ABSTAIN
        assert (
            derive("insufficient_evidence", FsmState.MONITOR)
            is PipelineOutcome.MONITORING_CONTINUES
        )
        assert (
            derive("insufficient_evidence", FsmState.INVESTIGATE)
            is PipelineOutcome.MONITORING_CONTINUES
        )
        # Deliberate fail-safe: a resolved event is not a monitoring claim.
        assert (
            derive("insufficient_evidence", FsmState.IDLE)
            is PipelineOutcome.PROCESSING_INCOMPLETE
        )
        assert (
            derive("verified_pending_report", FsmState.ESCALATE)
            is PipelineOutcome.PROCESSING_INCOMPLETE
        )
        assert derive(None, FsmState.MONITOR) is PipelineOutcome.PROCESSING_INCOMPLETE
        assert (
            derive("something_new", FsmState.MONITOR)
            is PipelineOutcome.PROCESSING_INCOMPLETE
        )

    def test_source_time_bounds_derivation(self) -> None:
        assert derive_source_time_bounds([]) is None
        entries = [make_coops_entry(), make_dart_entry()]
        bounds = derive_source_time_bounds(entries)
        assert bounds is not None
        assert bounds.first_observation_utc == W_COOPS.first_observation_utc
        assert bounds.last_observation_utc == W_DART.last_observation_utc


class TestNestedModels:
    def test_observation_bounds_ordering(self) -> None:
        with pytest.raises(ValidationError, match="must be <="):
            ObservationBounds(
                first_observation_utc=T0 + timedelta(hours=1),
                last_observation_utc=T0,
            )

    def test_post_hoc_seismic_context_requires_single_revision(self) -> None:
        from tests.unit._ocean_evidence_fixtures import make_seismic_context

        with pytest.raises(ValidationError, match="POST_HOC_FINAL_PRODUCT"):
            make_seismic_context(
                context_class=SeismicContextClass.POST_HOC_FINAL_PRODUCT
            )
        ctx = make_seismic_context(
            context_class=SeismicContextClass.POST_HOC_FINAL_PRODUCT,
            latest_admissible_revision=make_seismic_context().trigger_revision,
        )
        assert isinstance(ctx, EventSeismicContext)

    def test_detector_config_threshold_ordering(self) -> None:
        with pytest.raises(ValidationError, match="t1 <= t2 <= t3"):
            DetectorConfig(
                detector_version="x",
                t1_investigate=0.6,
                t2_assess=0.4,
                t3_escalate=0.9,
                configuration_sha256=SHA_A,
            )


class TestRetainedWindowQC:
    def test_usable_counts_must_sum(self) -> None:
        with pytest.raises(ValidationError, match="n_usable"):
            make_qc(W_DART, 8, n_usable=5, n_unusable=2)

    def test_nonempty_requires_bounds(self) -> None:
        with pytest.raises(ValidationError, match="observation_bounds"):
            make_qc(None, 8)

    def test_record_hashes_sorted_unique(self) -> None:
        with pytest.raises(ValidationError, match="sorted and unique"):
            make_qc(W_DART, 8, record_sha256s=["b" * 64, "a" * 64])
        with pytest.raises(ValidationError, match="Invalid sha256"):
            make_qc(W_DART, 8, record_sha256s=["zz"])

    def test_flag_counts_cannot_exceed_records(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed"):
            make_qc(
                W_DART,
                2,
                flag_counts=QCFlagCounts(
                    n_pass=3, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
                ),
                n_usable=2,
                n_unusable=0,
            )

    def test_succeeded_requires_flag_for_every_record(self) -> None:
        with pytest.raises(ValidationError, match="SUCCEEDED requires"):
            make_qc(
                W_DART,
                8,
                flag_counts=QCFlagCounts(
                    n_pass=7, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
                ),
            )

    def test_partial_requires_some_but_not_all(self) -> None:
        with pytest.raises(ValidationError, match="PARTIAL requires"):
            make_qc(W_DART, 8, execution_status=QCExecutionStatus.PARTIAL)
        with pytest.raises(ValidationError, match="PARTIAL requires"):
            make_qc(
                W_DART,
                8,
                execution_status=QCExecutionStatus.PARTIAL,
                flag_counts=QCFlagCounts(
                    n_pass=0, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
                ),
                aggregate_condition=QCAggregateCondition.NO_RETAINED_QC,
            )
        qc = make_qc(
            W_DART,
            8,
            execution_status=QCExecutionStatus.PARTIAL,
            flag_counts=QCFlagCounts(
                n_pass=5, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
            ),
        )
        assert qc.aggregate_condition is QCAggregateCondition.WORST_FLAG_PASS

    def test_not_run_requires_zero_flags(self) -> None:
        with pytest.raises(ValidationError, match="NOT_RUN requires"):
            make_qc(W_DART, 8, execution_status=QCExecutionStatus.NOT_RUN)
        qc = make_qc(
            W_DART,
            8,
            execution_status=QCExecutionStatus.NOT_RUN,
            flag_counts=QCFlagCounts(
                n_pass=0, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
            ),
            aggregate_condition=QCAggregateCondition.NO_RETAINED_QC,
        )
        assert qc.aggregate_condition is QCAggregateCondition.NO_RETAINED_QC

    def test_failed_cannot_flag_every_record(self) -> None:
        with pytest.raises(ValidationError, match="FAILED requires"):
            make_qc(W_DART, 8, execution_status=QCExecutionStatus.FAILED)

    def test_failed_with_partial_flags_is_valid(self) -> None:
        qc = make_qc(
            W_DART,
            8,
            execution_status=QCExecutionStatus.FAILED,
            flag_counts=QCFlagCounts(
                n_pass=3, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
            ),
            aggregate_condition=QCAggregateCondition.WORST_FLAG_PASS,
        )
        assert qc.execution_status is QCExecutionStatus.FAILED

    def test_aggregate_condition_must_match_flag_counts(self) -> None:
        with pytest.raises(ValidationError, match="aggregate_condition"):
            make_qc(
                W_DART,
                8,
                aggregate_condition=QCAggregateCondition.WORST_FLAG_FAIL,
            )


class TestThresholdEvaluation:
    def _te(self, **overrides: object) -> ThresholdEvaluation:
        base: dict[str, object] = {
            "source": DataSource.DART,
            "station_id": "46403",
            "result": ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET,
            "ensemble_score": 0.72,
            "t1_investigate": 0.35,
            "t2_assess": 0.60,
            "t3_escalate": 0.85,
            "highest_tier_met": "T2",
            "evaluated_window": W_DART,
        }
        base.update(overrides)
        return ThresholdEvaluation(**base)  # type: ignore[arg-type]

    def test_met_at_exact_threshold(self) -> None:
        te = self._te(ensemble_score=0.35, highest_tier_met="T1")
        assert te.highest_tier_met == "T1"

    def test_met_requires_score_window_and_tier(self) -> None:
        with pytest.raises(ValidationError, match="score and window"):
            self._te(ensemble_score=None)
        with pytest.raises(ValidationError, match="score and window"):
            self._te(evaluated_window=None)
        with pytest.raises(ValidationError, match="highest_tier_met"):
            self._te(highest_tier_met=None)

    def test_met_requires_score_at_or_above_t1(self) -> None:
        with pytest.raises(ValidationError, match="score >= t1"):
            self._te(ensemble_score=0.30)

    def test_met_tier_must_match_score(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            self._te(highest_tier_met="T3")

    def test_no_crossing_requires_score_below_t1(self) -> None:
        no_met = (
            ThresholdResultValue.NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW
        )
        with pytest.raises(ValidationError, match="score < t1"):
            self._te(result=no_met, ensemble_score=0.35, highest_tier_met=None)
        with pytest.raises(ValidationError, match="cannot carry a tier"):
            self._te(result=no_met, ensemble_score=0.10, highest_tier_met="T1")
        with pytest.raises(ValidationError, match="requires a score and window"):
            self._te(result=no_met, ensemble_score=None, highest_tier_met=None)

    def test_not_evaluated_keeps_diagnostic_score_but_no_tier(self) -> None:
        te = self._te(
            result=ThresholdResultValue.NOT_EVALUATED, highest_tier_met=None
        )
        assert te.ensemble_score == 0.72
        with pytest.raises(ValidationError, match="cannot carry highest_tier_met"):
            self._te(result=ThresholdResultValue.NOT_EVALUATED)


class TestConfounderCheck:
    def _check(self, **overrides: object) -> ConfounderCheck:
        base: dict[str, object] = {
            "name": "storm_surge_screen",
            "applicability": ConfounderApplicability.APPLICABLE,
            "prerequisite_status": ConfounderPrerequisiteStatus.AVAILABLE,
            "result": ConfounderResultValue.FLAG_NOT_RAISED,
        }
        base.update(overrides)
        return ConfounderCheck(**base)  # type: ignore[arg-type]

    def test_not_applicable_pairs(self) -> None:
        check = self._check(
            applicability=ConfounderApplicability.NOT_APPLICABLE,
            result=ConfounderResultValue.NOT_APPLICABLE,
        )
        assert check.result is ConfounderResultValue.NOT_APPLICABLE
        with pytest.raises(ValidationError, match="matching result"):
            self._check(applicability=ConfounderApplicability.NOT_APPLICABLE)
        with pytest.raises(ValidationError, match="cannot report NOT_APPLICABLE"):
            self._check(result=ConfounderResultValue.NOT_APPLICABLE)

    def test_missing_prerequisites_force_not_evaluated(self) -> None:
        with pytest.raises(ValidationError, match="NOT_EVALUATED"):
            self._check(prerequisite_status=ConfounderPrerequisiteStatus.MISSING)
        check = self._check(
            prerequisite_status=ConfounderPrerequisiteStatus.ERROR,
            result=ConfounderResultValue.NOT_EVALUATED,
        )
        assert check.result is ConfounderResultValue.NOT_EVALUATED

    def test_available_prerequisites_allow_any_outcome(self) -> None:
        for result in (
            ConfounderResultValue.FLAG_RAISED,
            ConfounderResultValue.FLAG_NOT_RAISED,
            ConfounderResultValue.NOT_EVALUATED,
        ):
            assert self._check(result=result).result is result


class TestStationLimitation:
    def test_effect_must_match_registry(self) -> None:
        with pytest.raises(ValidationError, match="registry effect"):
            StationLimitation(
                code=LimitationCode.CALIBRATION_UNAVAILABLE,
                effect=EvaluationEffect.INFORMATIONAL,
            )
        lim = StationLimitation(
            code=LimitationCode.CALIBRATION_UNAVAILABLE,
            effect=EvaluationEffect.PREVENTS_EVALUATION,
        )
        assert lim.effect is EvaluationEffect.PREVENTS_EVALUATION


class TestStationAssessmentEntry:
    def test_fixture_entries_are_valid(self) -> None:
        dart = make_dart_entry()
        assert dart.evaluation_status is StationEvaluationStatus.EVALUATED
        coops = make_coops_entry()
        assert (
            coops.evaluation_status
            is StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS
        )

    def test_admission_count_consistency(self) -> None:
        with pytest.raises(ValidationError, match="admitted \\+ rejected"):
            make_dart_entry(n_records_admitted=1)
        cases = [
            (CurrentRecordAdmissionStatus.NO_RECORD_ATTEMPT, 2, 2, 0),
            (CurrentRecordAdmissionStatus.ALL_RECORDS_REJECTED, 2, 2, 0),
            (CurrentRecordAdmissionStatus.SOME_RECORDS_ADMITTED, 2, 2, 0),
            (CurrentRecordAdmissionStatus.ALL_RECORDS_ADMITTED, 2, 1, 1),
        ]
        for status, attempted, admitted, rejected in cases:
            with pytest.raises(ValidationError, match="Admission counts"):
                make_dart_entry(
                    admission_status=status,
                    n_records_attempted=attempted,
                    n_records_admitted=admitted,
                    n_records_rejected=rejected,
                )

    def test_no_retained_data_entry(self) -> None:
        kwargs = {
            "admission_status": CurrentRecordAdmissionStatus.NO_RECORD_ATTEMPT,
            "n_records_attempted": 0,
            "n_records_admitted": 0,
            "n_records_rejected": 0,
            "scoring_status": StationScoringStatus.NO_RETAINED_DATA,
            "evaluation_status": StationEvaluationStatus.NOT_EVALUATED,
            "observation_bounds": None,
            "n_retained_samples": 0,
            "median_cadence_sec": None,
            "latest_observation_utc": None,
            "operational_age_at_production_sec": None,
            "current_dart_data_mode": DartDataMode.UNKNOWN,
            "qc_retained_window": None,
            "calibration_status": CalibrationStatus.UNAVAILABLE,
            "calibration_sha256": "",
            "detector_components": None,
            "threshold_evaluation": None,
            "retained_record_refs": [],
            "limitations": [
                StationLimitation(
                    code=LimitationCode.CALIBRATION_UNAVAILABLE,
                    effect=EvaluationEffect.PREVENTS_EVALUATION,
                ),
                StationLimitation(
                    code=LimitationCode.QC_ABSENT_FOR_RETAINED_WINDOW,
                    effect=EvaluationEffect.LIMITS_INTERPRETATION,
                ),
            ],
        }
        entry = make_dart_entry(**kwargs)
        assert entry.evaluation_status is StationEvaluationStatus.NOT_EVALUATED
        for bad in (
            {"n_retained_samples": 1},
            {"observation_bounds": W_DART},
            {"qc_retained_window": make_qc(W_DART, 8)},
            {"retained_record_refs": make_dart_entry().retained_record_refs},
        ):
            with pytest.raises(ValidationError):
                make_dart_entry(**{**kwargs, **bad})

    def test_nonempty_window_requires_qc(self) -> None:
        with pytest.raises(ValidationError, match="only NO_RETAINED_DATA omits"):
            make_dart_entry(qc_retained_window=None)

    def test_scored_statuses_require_nonempty_window(self) -> None:
        with pytest.raises(ValidationError, match="nonempty retained window"):
            make_dart_entry(n_retained_samples=0)

    def test_qc_must_cover_at_least_one_record(self) -> None:
        empty_qc = make_qc(
            W_DART,
            0,
            execution_status=QCExecutionStatus.NOT_RUN,
            flag_counts=QCFlagCounts(
                n_pass=0, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
            ),
            aggregate_condition=QCAggregateCondition.NO_RETAINED_QC,
            n_usable=0,
            n_unusable=0,
        )
        with pytest.raises(ValidationError, match="at least one record"):
            make_dart_entry(qc_retained_window=empty_qc)

    def test_qc_bounds_must_match_retained_window(self) -> None:
        with pytest.raises(ValidationError, match="exact retained eligibility"):
            make_dart_entry(qc_retained_window=make_qc(W_COOPS, 8))

    def test_scoring_failed_requires_reason_and_no_threshold(self) -> None:
        with pytest.raises(ValidationError, match="failure_reason"):
            make_dart_entry(
                scoring_status=StationScoringStatus.SCORING_FAILED,
                evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
                detector_components=None,
                threshold_evaluation=None,
            )
        entry = make_dart_entry(
            scoring_status=StationScoringStatus.SCORING_FAILED,
            evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
            detector_components=None,
            threshold_evaluation=None,
            failure_reason="detector raised ValueError",
        )
        assert entry.scoring_status is StationScoringStatus.SCORING_FAILED

    def test_threshold_requires_successful_scoring(self) -> None:
        with pytest.raises(ValidationError, match="requires SCORING_SUCCEEDED"):
            make_dart_entry(
                scoring_status=StationScoringStatus.INSUFFICIENT_RETAINED_DATA,
                evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
                detector_components=None,
            )
        with pytest.raises(ValidationError, match="requires threshold_evaluation"):
            make_dart_entry(threshold_evaluation=None, detector_components=None)
        with pytest.raises(
            ValidationError, match="detector_components requires SCORING_SUCCEEDED"
        ):
            make_dart_entry(
                scoring_status=StationScoringStatus.SCORING_FAILED,
                evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
                threshold_evaluation=None,
                failure_reason="detector raised ValueError",
            )

    def test_retained_refs_must_be_time_ordered(self) -> None:
        refs = list(reversed(make_dart_entry().retained_record_refs))
        with pytest.raises(ValidationError, match="ordered by time"):
            make_dart_entry(retained_record_refs=refs)

    def test_confounders_sorted_unique(self) -> None:
        checks = [
            ConfounderCheck(
                name=name,
                applicability=ConfounderApplicability.APPLICABLE,
                prerequisite_status=ConfounderPrerequisiteStatus.AVAILABLE,
                result=ConfounderResultValue.FLAG_NOT_RAISED,
            )
            for name in ("zeta", "alpha")
        ]
        with pytest.raises(ValidationError, match="sorted and unique"):
            make_coops_entry(confounder_checks=checks)

    def test_limitations_must_be_exact_registry_set(self) -> None:
        with pytest.raises(ValidationError, match="registry-derived set"):
            make_coops_entry(limitations=[])
        extra = [
            StationLimitation(
                code=LimitationCode.CADENCE_IRREGULAR,
                effect=EvaluationEffect.INFORMATIONAL,
            )
        ]
        with pytest.raises(ValidationError, match="registry-derived set"):
            make_dart_entry(limitations=extra)

    def test_evaluation_status_must_match_derivation(self) -> None:
        with pytest.raises(ValidationError, match="does not match"):
            make_dart_entry(
                evaluation_status=StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS
            )

    def test_threshold_identity_and_window_must_match_entry(self) -> None:
        te = make_dart_entry().threshold_evaluation
        assert te is not None
        with pytest.raises(ValidationError, match="identity must match"):
            make_dart_entry(
                threshold_evaluation=te.model_copy(update={"station_id": "46404"})
            )
        with pytest.raises(ValidationError, match="evaluated retained"):
            make_dart_entry(
                threshold_evaluation=te.model_copy(
                    update={"evaluated_window": W_COOPS}
                )
            )

    def test_prevented_evaluation_forces_not_evaluated_threshold(self) -> None:
        # Calibration UNAVAILABLE prevents evaluation: the diagnostic
        # score survives, the scientific result does not.
        te = make_dart_entry().threshold_evaluation
        assert te is not None
        prevented_te = ThresholdEvaluation(
            source=DataSource.DART,
            station_id="46403",
            result=ThresholdResultValue.NOT_EVALUATED,
            ensemble_score=0.72,
            t1_investigate=0.35,
            t2_assess=0.60,
            t3_escalate=0.85,
            highest_tier_met=None,
            evaluated_window=W_DART,
        )
        limitations = [
            StationLimitation(
                code=LimitationCode.CALIBRATION_UNAVAILABLE,
                effect=EvaluationEffect.PREVENTS_EVALUATION,
            )
        ]
        entry = make_dart_entry(
            calibration_status=CalibrationStatus.UNAVAILABLE,
            calibration_sha256="",
            evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
            threshold_evaluation=prevented_te,
            limitations=limitations,
        )
        assert entry.threshold_evaluation is not None
        assert entry.threshold_evaluation.ensemble_score == 0.72
        with pytest.raises(ValidationError, match="does not match derived"):
            make_dart_entry(
                calibration_status=CalibrationStatus.UNAVAILABLE,
                calibration_sha256="",
                evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
                threshold_evaluation=te,
                limitations=limitations,
            )

    def test_source_specific_field_rules(self) -> None:
        with pytest.raises(ValidationError, match="require current_dart_data_mode"):
            make_dart_entry(current_dart_data_mode=None)
        with pytest.raises(ValidationError, match="cannot carry coops_products"):
            make_dart_entry(coops_products=["water_level"])
        with pytest.raises(ValidationError, match="NOT_APPLICABLE"):
            make_dart_entry(
                mixed_product_condition=MixedProductCondition.SINGLE_PRODUCT
            )
        with pytest.raises(ValidationError, match="cannot carry a DART data mode"):
            make_coops_entry(current_dart_data_mode=DartDataMode.STANDARD)
        with pytest.raises(ValidationError, match="sorted and unique"):
            make_coops_entry(coops_products=["water_level", "air_pressure"])
        with pytest.raises(ValidationError, match="SINGLE_PRODUCT or MIXED"):
            make_coops_entry(
                mixed_product_condition=MixedProductCondition.NOT_APPLICABLE
            )
        with pytest.raises(ValidationError, match="MIXED_PRODUCTS"):
            make_coops_entry(coops_products=["air_pressure", "water_level"])
        with pytest.raises(ValidationError, match="cannot use source"):
            make_dart_entry(source=DataSource.SEISMIC)


class TestSourceRollup:
    def test_derived_rollups(self) -> None:
        entries = [make_coops_entry(), make_dart_entry()]
        scope = StationScope.OBSERVED_RECORDS_ONLY
        dart = derive_source_rollup(DataSource.DART, entries, scope)
        assert dart.status is SourceRollupStatus.SOURCE_CONFIGURED_THRESHOLD_MET
        assert dart.crossed_station_ids == ["46403"]
        assert [lim.code for lim in dart.limitations] == [
            LimitationCode.NETWORK_COVERAGE_NOT_EVALUATED
        ]
        coops = derive_source_rollup(DataSource.COOPS, entries, scope)
        assert (
            coops.status
            is SourceRollupStatus.SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS
        )
        assert coops.crossed_station_ids == []
        empty = derive_source_rollup(DataSource.DART, [], scope)
        assert empty.status is SourceRollupStatus.SOURCE_NOT_EVALUATED

    def test_not_evaluated_stations_are_not_negative_evidence(self) -> None:
        limitations = [
            StationLimitation(
                code=LimitationCode.CALIBRATION_UNAVAILABLE,
                effect=EvaluationEffect.PREVENTS_EVALUATION,
            )
        ]
        prevented = make_dart_entry(
            calibration_status=CalibrationStatus.UNAVAILABLE,
            calibration_sha256="",
            evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
            threshold_evaluation=ThresholdEvaluation(
                source=DataSource.DART,
                station_id="46403",
                result=ThresholdResultValue.NOT_EVALUATED,
                ensemble_score=0.72,
                t1_investigate=0.35,
                t2_assess=0.60,
                t3_escalate=0.85,
                highest_tier_met=None,
                evaluated_window=W_DART,
            ),
            limitations=limitations,
        )
        rollup = derive_source_rollup(
            DataSource.DART, [prevented], StationScope.OBSERVED_RECORDS_ONLY
        )
        assert rollup.status is SourceRollupStatus.SOURCE_NOT_EVALUATED
        assert rollup.evaluated_station_ids == []

    def test_configured_inventory_scope_has_no_coverage_limitation(self) -> None:
        rollup = derive_source_rollup(
            DataSource.DART,
            [make_dart_entry(manifest_condition=StationManifestCondition.IN_MANIFEST)],
            StationScope.CONFIGURED_INVENTORY,
        )
        assert rollup.limitations == []

    def test_rollup_internal_consistency(self) -> None:
        met = SourceRollupStatus.SOURCE_CONFIGURED_THRESHOLD_MET
        no_met = (
            SourceRollupStatus.SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS
        )
        with pytest.raises(ValidationError, match="sorted and unique"):
            SourceRollup(
                source=DataSource.DART,
                status=met,
                n_evaluated_stations=2,
                evaluated_station_ids=["b", "a"],
                crossed_station_ids=["a"],
            )
        with pytest.raises(
            ValidationError, match="crossed_station_ids must be sorted"
        ):
            SourceRollup(
                source=DataSource.DART,
                status=met,
                n_evaluated_stations=2,
                evaluated_station_ids=["a", "b"],
                crossed_station_ids=["b", "a"],
            )
        with pytest.raises(
            ValidationError, match="n_evaluated_stations must match"
        ):
            SourceRollup(
                source=DataSource.DART,
                status=met,
                n_evaluated_stations=3,
                evaluated_station_ids=["a", "b"],
                crossed_station_ids=["a"],
            )
        with pytest.raises(ValidationError, match="must be evaluated"):
            SourceRollup(
                source=DataSource.DART,
                status=met,
                n_evaluated_stations=1,
                evaluated_station_ids=["a"],
                crossed_station_ids=["b"],
            )
        with pytest.raises(ValidationError, match="at least one crossed"):
            SourceRollup(
                source=DataSource.DART,
                status=met,
                n_evaluated_stations=1,
                evaluated_station_ids=["a"],
                crossed_station_ids=[],
            )
        with pytest.raises(ValidationError, match="cannot list crossings"):
            SourceRollup(
                source=DataSource.DART,
                status=no_met,
                n_evaluated_stations=1,
                evaluated_station_ids=["a"],
                crossed_station_ids=["a"],
            )
        with pytest.raises(ValidationError, match="at least one evaluated"):
            SourceRollup(
                source=DataSource.DART,
                status=no_met,
                n_evaluated_stations=0,
                evaluated_station_ids=[],
                crossed_station_ids=[],
            )
        with pytest.raises(ValidationError, match="empty station lists"):
            SourceRollup(
                source=DataSource.DART,
                status=SourceRollupStatus.SOURCE_NOT_EVALUATED,
                n_evaluated_stations=1,
                evaluated_station_ids=["a"],
                crossed_station_ids=[],
            )


class TestAnalysisStatus:
    def test_not_implemented_cannot_succeed(self) -> None:
        with pytest.raises(ValidationError, match="cannot succeed"):
            AnalysisStatus(
                capability=AnalysisCapability.NOT_IMPLEMENTED,
                execution=AnalysisExecution.SUCCEEDED,
                quality=AnalysisQuality.NOMINAL,
            )

    def test_quality_requires_success(self) -> None:
        with pytest.raises(ValidationError, match="successful execution"):
            AnalysisStatus(
                capability=AnalysisCapability.IMPLEMENTED_LIVE,
                execution=AnalysisExecution.FAILED,
                quality=AnalysisQuality.DEGRADED,
            )
        ok = AnalysisStatus(
            capability=AnalysisCapability.IMPLEMENTED_LIVE,
            execution=AnalysisExecution.SUCCEEDED,
            quality=AnalysisQuality.DEGRADED,
        )
        assert ok.quality is AnalysisQuality.DEGRADED


class TestProvenanceSummary:
    def test_count_and_ordering_rules(self) -> None:
        with pytest.raises(ValidationError, match="cannot exceed"):
            make_provenance(n_references_included=5)
        with pytest.raises(ValidationError, match="must be sorted"):
            make_provenance(companion_persistence_failures=["b", "a"])

    def test_resolved_requires_complete_resolution(self) -> None:
        with pytest.raises(ValidationError, match="RESOLVED requires"):
            make_provenance(n_references_included=3)
        with pytest.raises(ValidationError, match="RESOLVED requires"):
            make_provenance(n_unresolved_raw_records=1)
        partial = make_provenance(
            status=ProvenanceStatus.PARTIAL,
            n_references_included=3,
            n_unresolved_raw_records=1,
        )
        assert partial.status is ProvenanceStatus.PARTIAL

    def test_unavailable_requires_zero_included(self) -> None:
        with pytest.raises(ValidationError, match="zero included"):
            make_provenance(status=ProvenanceStatus.UNAVAILABLE)
        ok = make_provenance(
            status=ProvenanceStatus.UNAVAILABLE,
            n_references_included=0,
            database_available=False,
        )
        assert ok.n_references_included == 0


class TestOceanEvidenceAssessment:
    def test_fixture_assessment_is_valid_and_roundtrips(self) -> None:
        a = make_assessment()
        assert a.pipeline_outcome is PipelineOutcome.ABSTAIN
        restored = OceanEvidenceAssessment.model_validate_json(a.model_dump_json())
        assert restored == a

    def test_requires_active_event_uuid(self) -> None:
        with pytest.raises(ValidationError, match="active internal event"):
            make_assessment(event_id=None)

    def test_version_and_type_are_pinned(self) -> None:
        a = make_assessment()
        assert a.type == "OceanEvidenceAssessment"
        assert a.assessment_schema_version == 1
        assert a.condition_registry_version == "1.0.0"
        with pytest.raises(ValidationError):
            make_assessment(assessment_schema_version=2)
        with pytest.raises(ValidationError):
            make_assessment(condition_registry_version="9.9.9")
        with pytest.raises(ValidationError):
            make_assessment(type="SomethingElse")

    def test_stations_sorted_unique_by_source_and_id(self) -> None:
        kwargs = assessment_kwargs()
        kwargs["stations"] = list(reversed(kwargs["stations"]))
        with pytest.raises(ValidationError, match="sorted and unique"):
            OceanEvidenceAssessment(**kwargs)
        dup = assessment_kwargs(stations=[make_dart_entry(), make_dart_entry()])
        with pytest.raises(ValidationError, match="sorted and unique"):
            OceanEvidenceAssessment(**dup)

    def test_scope_manifest_coupling(self) -> None:
        with pytest.raises(ValidationError, match="CONFIGURED_INVENTORY"):
            make_assessment(station_scope=StationScope.CONFIGURED_INVENTORY)
        manifest = StationManifestRef(
            manifest_id="noaa-dart-2011",
            manifest_version="3",
            effective_at_utc=T0 - timedelta(days=30),
            manifest_sha256=SHA_C,
        )
        with pytest.raises(ValidationError, match="forbids one"):
            make_assessment(station_manifest=manifest)
        with pytest.raises(ValidationError, match="IN_MANIFEST or"):
            make_assessment(
                station_scope=StationScope.CONFIGURED_INVENTORY,
                station_manifest=manifest,
            )
        with pytest.raises(ValidationError, match="Without a manifest"):
            make_assessment(
                stations=[
                    make_coops_entry(
                        manifest_condition=StationManifestCondition.IN_MANIFEST
                    ),
                    make_dart_entry(),
                ]
            )

    def test_configured_inventory_assessment(self) -> None:
        stations = [
            make_coops_entry(
                manifest_condition=StationManifestCondition.IN_MANIFEST
            ),
            make_dart_entry(
                manifest_condition=StationManifestCondition.OUT_OF_MANIFEST,
                limitations=[
                    StationLimitation(
                        code=LimitationCode.OUT_OF_MANIFEST_OBSERVATION,
                        effect=EvaluationEffect.INFORMATIONAL,
                    )
                ],
            ),
        ]
        scope = StationScope.CONFIGURED_INVENTORY
        a = make_assessment(
            stations=stations,
            station_scope=scope,
            station_manifest=StationManifestRef(
                manifest_id="noaa-dart-2011",
                manifest_version="3",
                effective_at_utc=T0 - timedelta(days=30),
                manifest_sha256=SHA_C,
            ),
            dart_rollup=derive_source_rollup(DataSource.DART, stations, scope),
            coops_rollup=derive_source_rollup(DataSource.COOPS, stations, scope),
        )
        assert a.dart_rollup.limitations == []

    def test_fsm_state_pair_consistency(self) -> None:
        with pytest.raises(ValidationError, match="fsm_state_changed"):
            make_assessment(fsm_state_changed=False)
        a = make_assessment(
            fsm_state_before=FsmState.ASSESS, fsm_state_changed=False
        )
        assert a.fsm_state_changed is False

    def test_dart_event_mode_lists(self) -> None:
        with pytest.raises(ValidationError, match="derived from"):
            make_assessment(dart_stations_currently_in_event_mode=[])
        with pytest.raises(ValidationError, match="sorted and unique"):
            make_assessment(
                dart_event_mode_stations_since_event_origin=["46403", "46403"]
            )
        with pytest.raises(ValidationError, match="Invalid station id"):
            make_assessment(
                dart_event_mode_stations_since_event_origin=["46403", "bad id"]
            )
        with pytest.raises(ValidationError, match="match its station list"):
            make_assessment(dart_event_mode_observed_since_event_origin=False)
        # The lifetime list may keep stations that have since left event
        # mode or that have no entry at this checkpoint.
        a = make_assessment(
            dart_event_mode_stations_since_event_origin=["21418", "46403"]
        )
        assert a.dart_event_mode_observed_since_event_origin is True

    def test_rollups_must_match_derivation(self) -> None:
        stations = [make_coops_entry(), make_dart_entry()]
        wrong = derive_source_rollup(
            DataSource.COOPS, [], StationScope.OBSERVED_RECORDS_ONLY
        )
        with pytest.raises(ValidationError, match="coops_rollup"):
            make_assessment(stations=stations, coops_rollup=wrong)
        wrong_dart = derive_source_rollup(
            DataSource.DART, [], StationScope.OBSERVED_RECORDS_ONLY
        )
        with pytest.raises(ValidationError, match="dart_rollup"):
            make_assessment(stations=stations, dart_rollup=wrong_dart)

    def test_source_time_bounds_must_match_derivation(self) -> None:
        with pytest.raises(ValidationError, match="source_time_bounds"):
            make_assessment(source_time_bounds=None)
        with pytest.raises(ValidationError, match="source_time_bounds"):
            make_assessment(source_time_bounds=W_DART)

    def test_station_thresholds_must_match_detector_config(self) -> None:
        with pytest.raises(ValidationError, match="detector configuration"):
            make_assessment(detector_config=make_detector_config(t1_investigate=0.30))

    def test_stations_without_threshold_evaluation_are_accepted(self) -> None:
        failed = make_dart_entry(
            station_id="21419",
            scoring_status=StationScoringStatus.SCORING_FAILED,
            evaluation_status=StationEvaluationStatus.NOT_EVALUATED,
            detector_components=None,
            threshold_evaluation=None,
            failure_reason="detector raised ValueError",
        )
        a = make_assessment(
            stations=[make_coops_entry(), failed, make_dart_entry()]
        )
        assert a.dart_rollup.evaluated_station_ids == ["46403"]

    def test_checkpoint_confounders_sorted_unique(self) -> None:
        checks = [
            ConfounderCheck(
                name=name,
                applicability=ConfounderApplicability.APPLICABLE,
                prerequisite_status=ConfounderPrerequisiteStatus.AVAILABLE,
                result=ConfounderResultValue.FLAG_NOT_RAISED,
            )
            for name in ("zeta", "alpha")
        ]
        with pytest.raises(ValidationError, match="sorted and unique by name"):
            make_assessment(confounder_checks=checks)

    def test_trace_ids_sorted_unique(self) -> None:
        ids = [
            UUID("00000000-0000-4000-8000-000000000002"),
            UUID("00000000-0000-4000-8000-000000000001"),
        ]
        with pytest.raises(ValidationError, match="sorted and unique"):
            make_assessment(contributing_trace_ids=ids)

    def test_extra_fields_rejected_and_instances_frozen(self) -> None:
        with pytest.raises(ValidationError):
            make_assessment(surprise_field=1)
        a = make_assessment()
        with pytest.raises(ValidationError):
            a.fsm_state_changed = False  # type: ignore[misc]
