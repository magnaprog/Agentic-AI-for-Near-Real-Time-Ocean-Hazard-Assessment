"""Unit tests for pure OceanEvidenceAssessment construction.

Covers the frozen calibration classification rule, cadence
regularity, detector-config identity, the full builder path with mixed
station attempts, hash order-invariance, and the construction-error
paths that become disclosed assessment gaps.

Limitation parity needs no dedicated assertion here: the
StationAssessmentEntry validator re-derives the expected limitation set
from the recorded conditions and rejects any divergence, so every
successful construction in this module already proves the builder's
list matches the registry.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from hazard_assessment.agents.anomaly_detection import AnomalyScoreComponents
from hazard_assessment.orchestrator.states import EventContext
from hazard_assessment.schemas.ocean_evidence import (
    CadenceCondition,
    CalibrationStatus,
    CheckpointSource,
    ConfounderResultValue,
    CurrentRecordAdmissionStatus,
    DartDataMode,
    PipelineOutcome,
    StationScoringStatus,
)
from hazard_assessment.schemas.ocean_evidence_hashing import (
    KafkaMessageCoordinate,
    TransportProvenance,
    finalize_assessment_hashes,
)
from hazard_assessment.workers.assessment_builder import (
    CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES,
    DETECTOR_VERSION,
    AssessmentConstructionError,
    StationAttemptResult,
    _cadence,
    _decisive_flag_int,
    build_detector_config,
    build_ocean_evidence_assessment,
    classify_calibration_status,
)
from hazard_assessment.workers.station_buffer import (
    RetainedSample,
    RetainedSampleQC,
)

PRODUCED_AT = datetime(2026, 7, 16, 12, 0, 0, tzinfo=UTC)
BASE_EPOCH = PRODUCED_AT.timestamp() - 3600.0


def _sha(seed: int) -> str:
    return f"{seed:064x}"


def _scores(**overrides: object) -> AnomalyScoreComponents:
    defaults: dict[str, object] = {
        "threshold_score": 0.2,
        "wavelet_score": 0.3,
        "bocpd_score": 0.1,
        "statistical_score": 0.2,
        "ml_score": None,
        "spatial_coherence_score": 0.0,
        "seismic_context_quiet": False,
        "ensemble_score": 0.21,
        "rayleigh_wave_suspect": False,
        "filter_degraded": False,
        "detide_fit_source": "separate calibration series",
        "detide_fit_span_minutes": 30.0 * 24.0 * 60.0,
    }
    defaults.update(overrides)
    return AnomalyScoreComponents(**defaults)  # type: ignore[arg-type]


def _samples(n: int, start_seed: int = 1, step_sec: float = 60.0):
    qc = RetainedSampleQC(
        usable=True, flags=(("gross_range", 1),), confidence=1.0,
        n_checks_evaluated=1,
    )
    return tuple(
        RetainedSample(
            epoch_sec=BASE_EPOCH + i * step_sec,
            value=0.01 * i,
            payload_hash=_sha(start_seed + i),
            measurement_type=1,
            qc=qc,
        )
        for i in range(n)
    )


def _bound_event_context() -> EventContext:
    return EventContext(
        seismic_magnitude=7.1,
        seismic_region="Off Kamchatka",
        epicenter_lat=52.0,
        epicenter_lon=160.0,
        trigger_time_utc=PRODUCED_AT,
        seismic_provider="usgs",
        external_event_id="us7000test",
        trigger_revision_id="rev-1",
        trigger_revision_sha256=_sha(9001),
        seismic_context_class="LIVE_RECEIPT_ORDERED",
    )


def _dart_attempt(**overrides: object) -> StationAttemptResult:
    defaults: dict[str, object] = {
        "source": "dart",
        "station_id": "46404",
        "scoring_status": StationScoringStatus.SCORING_SUCCEEDED,
        "calibration_status": CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE,
        "n_records_attempted": 12,
        "n_records_admitted": 12,
        "retained_samples": _samples(12),
        "scores": _scores(),
        "calibration_sha256": _sha(7001),
        "rayleigh_inputs_available": True,
    }
    defaults.update(overrides)
    return StationAttemptResult(**defaults)  # type: ignore[arg-type]


def _coops_no_data_attempt() -> StationAttemptResult:
    return StationAttemptResult(
        source="coops",
        station_id="9410230",
        scoring_status=StationScoringStatus.NO_RETAINED_DATA,
        calibration_status=CalibrationStatus.NOT_REQUIRED,
        n_records_attempted=3,
        n_records_admitted=0,
    )


def _build(attempts, **overrides: object):
    config = build_detector_config(0.35, 0.60, 0.85)
    kwargs: dict[str, object] = {
        "checkpoint_id": _sha(31),
        "checkpoint_source": CheckpointSource.LIVE_KAFKA,
        "event_id": uuid4(),
        "event_context": _bound_event_context(),
        "trace_id": uuid4(),
        "produced_at_utc": PRODUCED_AT,
        "station_attempts": attempts,
        "detector_config": config,
        "fsm_state_before": "MONITOR",
        "fsm_state_after": "MONITOR",
        "fsm_transition_ref": "",
        "dart_event_mode_stations_since_event_origin": [],
        "pipeline_outcome_field": "insufficient_evidence",
        "seismic_only_no_score": False,
        "spatial_analysis_ran": False,
        "database_available": True,
    }
    kwargs.update(overrides)
    return build_ocean_evidence_assessment(**kwargs)  # type: ignore[arg-type]


class TestDetectorConfig:
    def test_version_matches_anomaly_agent_manifest(self):
        from hazard_assessment.agents.anomaly_agent import _MANIFEST

        assert DETECTOR_VERSION == _MANIFEST.version

    def test_hash_covers_thresholds(self):
        a = build_detector_config(0.35, 0.60, 0.85)
        b = build_detector_config(0.35, 0.60, 0.85)
        c = build_detector_config(0.40, 0.60, 0.85)
        assert a.configuration_sha256 == b.configuration_sha256
        assert a.configuration_sha256 != c.configuration_sha256
        assert a.detector_version == DETECTOR_VERSION


class TestClassifyCalibrationStatus:
    def test_coops_never_requires_calibration(self):
        status = classify_calibration_status(
            source="coops", scores=_scores(detide_fit_source="event window"),
            calibration_span_minutes=None,
        )
        assert status is CalibrationStatus.NOT_REQUIRED

    def test_scored_separate_series_adequate_at_min_span(self):
        scores = _scores(
            detide_fit_span_minutes=CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES
        )
        status = classify_calibration_status(
            source="dart", scores=scores, calibration_span_minutes=None,
        )
        assert status is CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE

    def test_scored_separate_series_limited_below_min_span(self):
        scores = _scores(
            detide_fit_span_minutes=CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES - 1.0
        )
        status = classify_calibration_status(
            source="dart", scores=scores, calibration_span_minutes=None,
        )
        assert status is CalibrationStatus.LIMITED_PRE_EVENT_BASELINE

    def test_scored_separate_series_missing_span_is_limited(self):
        scores = _scores(detide_fit_span_minutes=None)
        status = classify_calibration_status(
            source="dart", scores=scores, calibration_span_minutes=None,
        )
        assert status is CalibrationStatus.LIMITED_PRE_EVENT_BASELINE

    def test_scored_event_window_is_fallback(self):
        scores = _scores(detide_fit_source="event window")
        status = classify_calibration_status(
            source="dart", scores=scores,
            calibration_span_minutes=CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES,
        )
        assert status is CalibrationStatus.EVENT_WINDOW_FALLBACK

    def test_scored_unknown_provenance_is_unavailable(self):
        scores = _scores(detide_fit_source="")
        status = classify_calibration_status(
            source="dart", scores=scores,
            calibration_span_minutes=CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES,
        )
        assert status is CalibrationStatus.UNAVAILABLE

    def test_unscored_uses_loaded_series_span(self):
        adequate = classify_calibration_status(
            source="dart", scores=None,
            calibration_span_minutes=CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES,
        )
        limited = classify_calibration_status(
            source="dart", scores=None, calibration_span_minutes=60.0,
        )
        assert adequate is CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE
        assert limited is CalibrationStatus.LIMITED_PRE_EVENT_BASELINE

    def test_unscored_without_series_is_unavailable(self):
        status = classify_calibration_status(
            source="dart", scores=None, calibration_span_minutes=None,
        )
        assert status is CalibrationStatus.UNAVAILABLE


class TestDecisiveFlag:
    def test_not_applicable_is_never_decisive(self):
        """QARTOD 0 (local NOT_APPLICABLE extension) is a recognized flag
        that can never be a record's decisive verdict: a check that does
        not apply is not evidence about the record. Live regression: the
        worker attaches every check unfiltered, and the extension checks
        default to 0, so ranking must not treat 0 as unknown."""
        assert _decisive_flag_int(
            (("gross_range", 1), ("mode_transition", 0))
        ) == 1
        assert _decisive_flag_int(
            (("mode_transition", 0), ("spike", 4))
        ) == 4
        assert _decisive_flag_int((("mode_transition", 0),)) is None

    def test_unknown_flag_integer_still_rejected(self):
        with pytest.raises(
            AssessmentConstructionError, match="Unknown QARTOD"
        ):
            _decisive_flag_int((("gross_range", 7),))


class TestCadence:
    def test_fewer_than_two_samples_is_unknown(self):
        assert _cadence(()) == (None, CadenceCondition.UNKNOWN)
        assert _cadence(_samples(1)) == (None, CadenceCondition.UNKNOWN)

    def test_regular_window_is_nominal(self):
        med, condition = _cadence(_samples(10, step_sec=60.0))
        assert med == 60.0
        assert condition is CadenceCondition.NOMINAL

    def test_large_gap_is_irregular(self):
        samples = list(_samples(10, step_sec=60.0))
        gapped = samples[:-1] + [
            RetainedSample(
                epoch_sec=samples[-2].epoch_sec + 300.0,
                value=0.0,
                payload_hash=_sha(99),
            )
        ]
        med, condition = _cadence(gapped)
        assert med == 60.0
        assert condition is CadenceCondition.IRREGULAR


class TestBuildAssessment:
    def test_mixed_attempts_build_coherent_assessment(self):
        assessment = _build([_dart_attempt(), _coops_no_data_attempt()])

        assert [
            (e.source.value, e.station_id) for e in assessment.stations
        ] == [("coops", "9410230"), ("dart", "46404")]

        dart_entry = assessment.stations[1]
        assert dart_entry.scoring_status is StationScoringStatus.SCORING_SUCCEEDED
        assert dart_entry.current_dart_data_mode is DartDataMode.STANDARD
        assert dart_entry.n_retained_samples == 12
        assert dart_entry.median_cadence_sec == 60.0
        assert dart_entry.threshold_evaluation is not None
        assert dart_entry.threshold_evaluation.ensemble_score == 0.21
        assert [c.result for c in dart_entry.confounder_checks] == [
            ConfounderResultValue.FLAG_NOT_RAISED
        ]

        coops_entry = assessment.stations[0]
        assert (
            coops_entry.admission_status
            is CurrentRecordAdmissionStatus.ALL_RECORDS_REJECTED
        )
        assert coops_entry.observation_bounds is None
        assert coops_entry.detector_components is None

        assert assessment.pipeline_outcome is PipelineOutcome.MONITORING_CONTINUES
        assert len(assessment.input_refs) == 12
        assert assessment.provenance.n_references_included == 12
        assert assessment.input_manifest_hash == ""
        assert assessment.scientific_content_hash == ""

    def test_not_applicable_qc_flags_build_successfully(self):
        """Full builder path with realistic QC tuples: standard checks
        plus defaulted NOT_APPLICABLE extension checks. Before the
        severity-map fix this raised AssessmentConstructionError and
        gapped every live checkpoint whose records carried QC."""
        qc = RetainedSampleQC(
            usable=True,
            flags=(
                ("gross_range", 1),
                ("latency", 0),
                ("mode_transition", 0),
                ("neighbor_consistency", 0),
            ),
            confidence=1.0,
            n_checks_evaluated=1,
        )
        samples = tuple(
            RetainedSample(
                epoch_sec=BASE_EPOCH + i * 60.0,
                value=0.01 * i,
                payload_hash=_sha(500 + i),
                measurement_type=1,
                qc=qc,
            )
            for i in range(12)
        )
        assessment = _build([_dart_attempt(retained_samples=samples)])
        entry = assessment.stations[0]
        window_qc = entry.qc_retained_window
        assert window_qc is not None
        assert window_qc.flag_counts.n_pass == 12
        assert window_qc.execution_status.value == "SUCCEEDED"

    def test_hashes_invariant_to_attempt_order_and_envelope_ids(self):
        attempts = [_dart_attempt(), _coops_no_data_attempt()]
        transport = TransportProvenance(
            run_id="run-1",
            consumer_group="hazard-pipeline",
            messages=[
                KafkaMessageCoordinate(
                    topic="raw.observations", partition=0, offset=41
                ),
                KafkaMessageCoordinate(
                    topic="raw.observations", partition=0, offset=42
                ),
            ],
        )
        event_id = uuid4()
        trace_id = uuid4()
        first = finalize_assessment_hashes(
            _build(attempts, event_id=event_id, trace_id=trace_id), transport
        )
        second = finalize_assessment_hashes(
            _build(
                list(reversed(attempts)), event_id=event_id, trace_id=trace_id
            ),
            transport,
        )
        # handoff_id is minted per envelope, so equal hashes prove both
        # attempt-order canonicalization and envelope-randomness exclusion.
        assert first.handoff_id != second.handoff_id
        assert first.input_manifest_hash == second.input_manifest_hash
        assert first.scientific_content_hash == second.scientific_content_hash
        assert (
            first.transport_provenance_hash == second.transport_provenance_hash
        )
        assert first.transport_provenance_hash != ""

    def test_duplicate_station_attempts_rejected(self):
        with pytest.raises(AssessmentConstructionError, match="Duplicate"):
            _build([_dart_attempt(), _dart_attempt()])

    def test_unbound_seismic_identity_rejected(self):
        ctx = _bound_event_context()
        ctx.seismic_provider = ""
        ctx.external_event_id = ""
        with pytest.raises(
            AssessmentConstructionError, match="seismic identity"
        ):
            _build([_dart_attempt()], event_context=ctx)

    def test_inconsistent_no_data_attempt_rejected(self):
        attempt = _dart_attempt(
            scoring_status=StationScoringStatus.NO_RETAINED_DATA,
            scores=None,
        )
        with pytest.raises(
            AssessmentConstructionError, match="inconsistent"
        ):
            _build([attempt])

    def test_seismic_only_checkpoint_abstains(self):
        assessment = _build(
            [],
            pipeline_outcome_field=None,
            seismic_only_no_score=True,
            fsm_state_before="IDLE",
            fsm_state_after="MONITOR",
            fsm_transition_ref=str(uuid4()),
        )
        assert assessment.stations == []
        assert assessment.pipeline_outcome is PipelineOutcome.ABSTAIN
        assert assessment.fsm_state_changed is True
