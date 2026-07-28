"""Shared deterministic fixtures for ocean evidence assessment tests.

Every field, UUID, and timestamp is fixed so the hashing tests can pin
golden vectors. Builders take keyword overrides; the assessment builder
recomputes rollups, bounds, and DART event-mode lists from the station
entries unless the caller overrides them explicitly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

from hazard_assessment.schemas.envelope import DataSource
from hazard_assessment.schemas.ocean_evidence import (
    AnalysisCapability,
    AnalysisExecution,
    AnalysisQuality,
    AnalysisStatus,
    AnalysisStatusSet,
    CadenceCondition,
    CalibrationStatus,
    CheckpointSource,
    ConfounderApplicability,
    ConfounderCheck,
    ConfounderPrerequisiteStatus,
    ConfounderResultValue,
    CurrentRecordAdmissionStatus,
    DartDataMode,
    DetectorComponents,
    DetectorConfig,
    EvaluationEffect,
    EventSeismicContext,
    FilteringCondition,
    FsmState,
    LimitationCode,
    MixedProductCondition,
    ObservationBounds,
    OceanEvidenceAssessment,
    PipelineOutcome,
    ProvenanceStatus,
    ProvenanceSummary,
    QCAggregateCondition,
    QCExecutionStatus,
    QCFlagCounts,
    RetainedRecordRef,
    RetainedWindowQC,
    SeismicContextClass,
    SeismicRevisionRef,
    SourceValidationStatus,
    StationAssessmentEntry,
    StationEvaluationStatus,
    StationLimitation,
    StationManifestCondition,
    StationScope,
    StationScoringStatus,
    ThresholdEvaluation,
    ThresholdResultValue,
    derive_source_rollup,
    derive_source_time_bounds,
)
from hazard_assessment.schemas.ocean_evidence_hashing import (
    KafkaMessageCoordinate,
    TransportProvenance,
    derive_live_checkpoint_id,
)

T0 = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
SHA_D = "d" * 64
SHA_E = "e" * 64
SHA_F = "f" * 64

W_DART = ObservationBounds(
    first_observation_utc=T0,
    last_observation_utc=T0 + timedelta(hours=2),
)
W_COOPS = ObservationBounds(
    first_observation_utc=T0 - timedelta(minutes=30),
    last_observation_utc=T0 + timedelta(hours=1),
)


def make_seismic_context(**overrides: Any) -> EventSeismicContext:
    base: dict[str, Any] = {
        "provider": "usgs",
        "external_event_id": "official20110311054624120_30",
        "context_class": SeismicContextClass.LIVE_RECEIPT_ORDERED,
        "trigger_revision": SeismicRevisionRef(
            provider_revision_id="rev-1",
            payload_sha256=SHA_D,
            provider_updated_at_utc=T0 + timedelta(minutes=5),
        ),
        "latest_admissible_revision": SeismicRevisionRef(
            provider_revision_id="rev-2",
            payload_sha256=SHA_E,
            provider_updated_at_utc=T0 + timedelta(minutes=20),
        ),
    }
    base.update(overrides)
    return EventSeismicContext(**base)


def make_detector_config(**overrides: Any) -> DetectorConfig:
    base: dict[str, Any] = {
        "detector_version": "ensemble-1.0",
        "t1_investigate": 0.35,
        "t2_assess": 0.60,
        "t3_escalate": 0.85,
        "configuration_sha256": SHA_F,
    }
    base.update(overrides)
    return DetectorConfig(**base)


def make_qc(
    bounds: ObservationBounds | None, n_records: int, **overrides: Any
) -> RetainedWindowQC:
    base: dict[str, Any] = {
        "execution_status": QCExecutionStatus.SUCCEEDED,
        "aggregate_condition": QCAggregateCondition.WORST_FLAG_PASS,
        "observation_bounds": bounds,
        "n_records": n_records,
        "flag_counts": QCFlagCounts(
            n_pass=n_records, n_not_evaluated=0, n_suspect=0, n_fail=0, n_missing=0
        ),
        "n_usable": n_records,
        "n_unusable": 0,
        "n_unevaluated_checks": 0,
        "confidence_min": 0.9,
        "confidence_mean": 0.95,
        "confidence_max": 1.0,
        "record_sha256s": [SHA_A, SHA_B],
    }
    base.update(overrides)
    return RetainedWindowQC(**base)


def dart_entry_kwargs() -> dict[str, Any]:
    """A fully evaluated DART entry whose ensemble score meets T2."""
    return {
        "source": DataSource.DART,
        "station_id": "46403",
        "admission_status": CurrentRecordAdmissionStatus.ALL_RECORDS_ADMITTED,
        "n_records_attempted": 2,
        "n_records_admitted": 2,
        "n_records_rejected": 0,
        "scoring_status": StationScoringStatus.SCORING_SUCCEEDED,
        "evaluation_status": StationEvaluationStatus.EVALUATED,
        "observation_bounds": W_DART,
        "n_retained_samples": 8,
        "median_cadence_sec": 900.0,
        "latest_observation_utc": T0 + timedelta(hours=2),
        "operational_age_at_production_sec": 120.0,
        "current_dart_data_mode": DartDataMode.EVENT,
        "coops_products": [],
        "qc_retained_window": make_qc(W_DART, 8),
        "calibration_status": CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE,
        "calibration_sha256": SHA_C,
        "cadence_condition": CadenceCondition.NOMINAL,
        "filtering_condition": FilteringCondition.NOMINAL,
        "mixed_product_condition": MixedProductCondition.NOT_APPLICABLE,
        "manifest_condition": StationManifestCondition.NO_MANIFEST,
        "detector_components": DetectorComponents(
            threshold_score=0.7,
            wavelet_score=0.6,
            bocpd_score=0.8,
            statistical_score=0.5,
        ),
        "threshold_evaluation": ThresholdEvaluation(
            source=DataSource.DART,
            station_id="46403",
            result=ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET,
            ensemble_score=0.72,
            t1_investigate=0.35,
            t2_assess=0.60,
            t3_escalate=0.85,
            highest_tier_met="T2",
            evaluated_window=W_DART,
        ),
        "confounder_checks": [],
        "retained_record_refs": [
            RetainedRecordRef(
                source=DataSource.DART,
                station_id="46403",
                observed_at_utc=T0,
                measurement_type=1,
                payload_sha256=SHA_A,
            ),
            RetainedRecordRef(
                source=DataSource.DART,
                station_id="46403",
                observed_at_utc=T0 + timedelta(hours=1),
                measurement_type=2,
                payload_sha256=SHA_B,
            ),
        ],
        "source_validation_status": (
            SourceValidationStatus.VALIDATED_FOR_CONFIGURED_USE
        ),
        "provenance_status": ProvenanceStatus.RESOLVED,
        "limitations": [],
        "failure_reason": "",
    }


def make_dart_entry(**overrides: Any) -> StationAssessmentEntry:
    kwargs = dart_entry_kwargs()
    kwargs.update(overrides)
    return StationAssessmentEntry(**kwargs)


def coops_entry_kwargs() -> dict[str, Any]:
    """A CO-OPS entry evaluated with the validation-limited code."""
    return {
        "source": DataSource.COOPS,
        "station_id": "1617760",
        "admission_status": CurrentRecordAdmissionStatus.SOME_RECORDS_ADMITTED,
        "n_records_attempted": 3,
        "n_records_admitted": 2,
        "n_records_rejected": 1,
        "scoring_status": StationScoringStatus.SCORING_SUCCEEDED,
        "evaluation_status": StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS,
        "observation_bounds": W_COOPS,
        "n_retained_samples": 6,
        "median_cadence_sec": 60.0,
        "latest_observation_utc": T0 + timedelta(hours=1),
        "operational_age_at_production_sec": 60.0,
        "current_dart_data_mode": None,
        "coops_products": ["water_level"],
        "qc_retained_window": make_qc(W_COOPS, 6),
        "calibration_status": CalibrationStatus.NOT_REQUIRED,
        "calibration_sha256": "",
        "cadence_condition": CadenceCondition.NOMINAL,
        "filtering_condition": FilteringCondition.NOMINAL,
        "mixed_product_condition": MixedProductCondition.SINGLE_PRODUCT,
        "manifest_condition": StationManifestCondition.NO_MANIFEST,
        "detector_components": DetectorComponents(
            threshold_score=0.1,
            wavelet_score=0.05,
            bocpd_score=0.12,
            statistical_score=0.08,
        ),
        "threshold_evaluation": ThresholdEvaluation(
            source=DataSource.COOPS,
            station_id="1617760",
            result=(
                ThresholdResultValue.NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW
            ),
            ensemble_score=0.10,
            t1_investigate=0.35,
            t2_assess=0.60,
            t3_escalate=0.85,
            highest_tier_met=None,
            evaluated_window=W_COOPS,
        ),
        "confounder_checks": [
            ConfounderCheck(
                name="storm_surge_screen",
                applicability=ConfounderApplicability.APPLICABLE,
                prerequisite_status=ConfounderPrerequisiteStatus.AVAILABLE,
                result=ConfounderResultValue.FLAG_NOT_RAISED,
            )
        ],
        "retained_record_refs": [
            RetainedRecordRef(
                source=DataSource.COOPS,
                station_id="1617760",
                observed_at_utc=T0 - timedelta(minutes=30),
                product="water_level",
                payload_sha256=SHA_A,
            ),
        ],
        "source_validation_status": SourceValidationStatus.VALIDATION_LIMITED,
        "provenance_status": ProvenanceStatus.RESOLVED,
        "limitations": [
            StationLimitation(
                code=LimitationCode.SOURCE_VALIDATION_LIMITED,
                effect=EvaluationEffect.LIMITS_INTERPRETATION,
            )
        ],
        "failure_reason": "",
    }


def make_coops_entry(**overrides: Any) -> StationAssessmentEntry:
    kwargs = coops_entry_kwargs()
    kwargs.update(overrides)
    return StationAssessmentEntry(**kwargs)


def make_analyses() -> AnalysisStatusSet:
    offline = AnalysisStatus(
        capability=AnalysisCapability.IMPLEMENTED_OFFLINE_ONLY,
        execution=AnalysisExecution.NOT_RUN,
        quality=AnalysisQuality.NOT_ASSESSED,
    )
    absent = AnalysisStatus(
        capability=AnalysisCapability.NOT_IMPLEMENTED,
        execution=AnalysisExecution.NOT_RUN,
        quality=AnalysisQuality.NOT_ASSESSED,
    )
    return AnalysisStatusSet(
        scenario=offline,
        scenario_verification=offline,
        coastal_arrival=offline,
        coastal_amplitude=absent,
        inundation=absent,
        meteorological_source_discrimination=absent,
        spatial_coherence=AnalysisStatus(
            capability=AnalysisCapability.IMPLEMENTED_LIVE,
            execution=AnalysisExecution.NOT_RUN,
            quality=AnalysisQuality.NOT_ASSESSED,
        ),
    )


def make_provenance(**overrides: Any) -> ProvenanceSummary:
    base: dict[str, Any] = {
        "status": ProvenanceStatus.RESOLVED,
        "n_references_expected": 4,
        "n_references_included": 4,
        "n_unresolved_raw_records": 0,
        "n_malformed_or_absent_hashes": 0,
        "references_capped": False,
        "calibration_provenance_available": True,
        "database_available": True,
        "companion_persistence_failures": [],
    }
    base.update(overrides)
    return ProvenanceSummary(**base)


def assessment_kwargs(
    stations: list[StationAssessmentEntry] | None = None,
) -> dict[str, Any]:
    if stations is None:
        stations = [make_coops_entry(), make_dart_entry()]
    scope = StationScope.OBSERVED_RECORDS_ONLY
    current_event_mode = sorted(
        s.station_id
        for s in stations
        if s.source is DataSource.DART
        and s.current_dart_data_mode is DartDataMode.EVENT
    )
    return {
        "handoff_id": UUID("11111111-1111-4111-8111-111111111111"),
        "event_id": UUID("22222222-2222-4222-8222-222222222222"),
        "trace_id": UUID("33333333-3333-4333-8333-333333333333"),
        "producer": "pipeline_worker",
        "produced_at_utc": T0 + timedelta(minutes=10),
        "input_refs": [],
        "code_version": "deadbeef",
        "model_version": "ruleset-1",
        "decision_trace": [],
        "checkpoint_id": derive_live_checkpoint_id(
            "hazard-pipeline", [("raw.observations", 0, 100, 120)]
        ),
        "checkpoint_source": CheckpointSource.LIVE_KAFKA,
        "seismic_context": make_seismic_context(),
        "contributing_trace_ids": [
            UUID("00000000-0000-4000-8000-000000000001"),
            UUID("00000000-0000-4000-8000-000000000002"),
        ],
        "source_time_bounds": derive_source_time_bounds(stations),
        "station_scope": scope,
        "station_manifest": None,
        "detector_config": make_detector_config(),
        "stations": stations,
        "dart_rollup": derive_source_rollup(DataSource.DART, stations, scope),
        "coops_rollup": derive_source_rollup(DataSource.COOPS, stations, scope),
        "fsm_state_before": FsmState.INVESTIGATE,
        "fsm_state_after": FsmState.ASSESS,
        "fsm_state_changed": True,
        "fsm_transition_ref": "audit-000123",
        "dart_stations_currently_in_event_mode": current_event_mode,
        "dart_event_mode_observed_since_event_origin": bool(current_event_mode),
        "dart_event_mode_stations_since_event_origin": current_event_mode,
        "confounder_checks": [],
        "analyses": make_analyses(),
        "pipeline_outcome": PipelineOutcome.ABSTAIN,
        "provenance": make_provenance(),
    }


def make_assessment(**overrides: Any) -> OceanEvidenceAssessment:
    kwargs = assessment_kwargs(stations=overrides.pop("stations", None))
    kwargs.update(overrides)
    return OceanEvidenceAssessment(**kwargs)


def make_transport() -> TransportProvenance:
    return TransportProvenance(
        run_id="run-1",
        consumer_group="hazard-pipeline",
        messages=[
            KafkaMessageCoordinate(
                topic="raw.observations",
                partition=0,
                offset=100,
                timestamp_type="LogAppendTime",
                timestamp_ms=1299822384000,
                application_message_id="msg-100",
            ),
            KafkaMessageCoordinate(
                topic="raw.observations",
                partition=0,
                offset=101,
                transport_rejected=True,
            ),
        ],
    )
