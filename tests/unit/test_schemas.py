"""Unit tests for handoff schema definitions.

Validates that all schemas enforce required fields, reject unknown fields,
and produce valid JSON conforming to the documented schema specifications.
"""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from hazard_assessment.schemas.anomaly import (
    AnomalyAssessment,
    ScoreComponents,
    SpatialConfirmation,
)
from hazard_assessment.schemas.envelope import BaseEnvelope, DataSource, InputRef
from hazard_assessment.schemas.escalation import EscalationPacket
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.human_decision import HumanDecision, ReviewDecision
from hazard_assessment.schemas.qc import DataMode, QARTODFlag, QARTODFlags, QCReport
from hazard_assessment.schemas.scenario import (
    INUNDATION_DISCLAIMER,
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

# Fixed values for deterministic test fixtures
_FIXED_DECISION_TIME = datetime(2026, 3, 11, 12, 0, 0, tzinfo=UTC)
_FIXED_ESCALATION_PACKET_ID = UUID("a1b2c3d4-e5f6-7890-abcd-ef1234567890")


class TestBaseEnvelope:
    def test_creates_with_defaults(self) -> None:
        envelope = BaseEnvelope(producer="test_agent")
        assert envelope.schema_version == "1.0"
        assert envelope.producer == "test_agent"
        assert envelope.handoff_id is not None
        assert envelope.produced_at_utc is not None

    def test_rejects_unknown_fields(self) -> None:
        with pytest.raises(ValidationError):
            BaseEnvelope(producer="test", unknown_field="value")

    def test_input_refs(self) -> None:
        ref = InputRef(
            source=DataSource.DART,
            record_id="rec_001",
            sha256="a" * 64,
        )
        envelope = BaseEnvelope(producer="test", input_refs=[ref])
        assert len(envelope.input_refs) == 1
        assert envelope.input_refs[0].source == DataSource.DART


    def test_naive_datetime_rejected_for_produced_at_utc(self) -> None:
        """produced_at_utc must reject naive datetimes via AwareDatetime."""
        naive = datetime(2026, 2, 27, 1, 30, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            BaseEnvelope(producer="test", produced_at_utc=naive)

    def test_aware_datetime_accepted_for_produced_at_utc(self) -> None:
        """produced_at_utc accepts timezone-aware datetimes."""
        aware = datetime(2026, 2, 27, 1, 30, 0, tzinfo=UTC)
        envelope = BaseEnvelope(producer="test", produced_at_utc=aware)
        assert envelope.produced_at_utc == aware


class TestQCReport:
    def test_valid_qc_report(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        report = QCReport(
            producer="qc_agent",
            station_id="DART_21413",
            observed_at_utc="2026-02-27T01:15:00Z",
            measurement_type=3,
            data_mode=DataMode.EVENT,
            record_usable=True,
            qartod_flags=flags,
            station_confidence=0.87,
            provenance_hash="a" * 64,
        )
        assert report.type == "QCReport"
        assert report.station_id == "DART_21413"
        assert report.data_mode == DataMode.EVENT
        assert report.qartod_flags.timing == QARTODFlag.PASS
        # Pydantic should parse the ISO string into a datetime object
        assert isinstance(report.observed_at_utc, datetime)

    def test_confidence_bounds(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        with pytest.raises(ValidationError):
            QCReport(
                producer="qc_agent",
                station_id="DART_21413",
                observed_at_utc="2026-02-27T01:15:00Z",
                measurement_type=3,
                data_mode=DataMode.STANDARD,
                record_usable=True,
                qartod_flags=flags,
                station_confidence=1.5,  # Out of bounds
                provenance_hash="a" * 64,
            )


class TestAnomalyAssessment:
    def test_valid_assessment(self) -> None:
        assessment = AnomalyAssessment(
            producer="anomaly_agent",
            current_state="INVESTIGATE",
            state_changed=True,
            anomaly_score=0.72,
            score_components=ScoreComponents(
                threshold=0.68,
                statistical=0.78,
                ml=0.65,
            ),
            seismic_quiet=True,
            meteotsunami_score=0.12,
        )
        assert assessment.anomaly_score == 0.72
        assert assessment.score_components.ml == 0.65

    def test_not_evaluated_defaults(self) -> None:
        """Fields for checks the pipeline has not run default to
        not-evaluated (None) rather than clean-looking values."""
        assessment = AnomalyAssessment(
            producer="anomaly_agent",
            current_state="MONITOR",
            state_changed=False,
            anomaly_score=0.1,
            score_components=ScoreComponents(threshold=0.1, statistical=0.1),
            seismic_quiet=True,
        )
        assert assessment.meteotsunami_score is None
        assert assessment.rayleigh_wave_suspect is None
        assert assessment.triggering_stations == []
        assert assessment.scored_stations == []

    def test_ml_component_optional(self) -> None:
        components = ScoreComponents(
            threshold=0.5,
            statistical=0.6,
            ml=None,  # ML model unavailable
        )
        assert components.ml is None

    def test_score_bounds(self) -> None:
        with pytest.raises(ValidationError):
            ScoreComponents(threshold=1.5, statistical=0.5)


class TestScenarioAssessment:
    def test_valid_scenario(self) -> None:
        scenario = RankedScenario(
            unit_source_ids=["USRC_001", "USRC_004"],
            weights=[0.65, 0.35],
            waveform_rmse_cm=3.8,
            mw_equivalent=8.1,
            rank=1,
            posterior_weight=0.42,
        )
        assert scenario.rank == 1

    def test_mismatched_sources_and_weights_rejected(self) -> None:
        """unit_source_ids and weights must have the same length."""
        with pytest.raises(ValidationError, match="same length"):
            RankedScenario(
                unit_source_ids=["USRC_001", "USRC_004", "USRC_007"],
                weights=[0.65, 0.35],  # 3 sources but only 2 weights
                waveform_rmse_cm=3.8,
                mw_equivalent=8.1,
                rank=1,
                posterior_weight=0.42,
            )

    def test_negative_weights_rejected(self) -> None:
        """NNLS weights must be non-negative."""
        with pytest.raises(ValidationError, match="non-negative"):
            RankedScenario(
                unit_source_ids=["USRC_001", "USRC_004"],
                weights=[0.65, -0.35],
                waveform_rmse_cm=3.8,
                mw_equivalent=8.1,
                rank=1,
                posterior_weight=0.42,
            )

    def test_valid_assessment(self) -> None:
        assessment = ScenarioAssessment(
            producer="scenario_agent",
            constraint_stage=ConstraintStage.DART_CONSTRAINED,
            dart_stations_used=["DART_21413"],
            inversion_window_sec=480,
            top_scenarios=[
                RankedScenario(
                    unit_source_ids=["USRC_001"],
                    weights=[1.0],
                    waveform_rmse_cm=3.8,
                    mw_equivalent=8.1,
                    rank=1,
                    posterior_weight=0.42,
                )
            ],
            ensemble_spread=EnsembleSpread.MODERATE,
            bilateral_rupture_evaluated=False,
        )
        assert assessment.type == "ScenarioAssessment"
        assert len(assessment.top_scenarios) == 1
        assert assessment.inundation_disclaimer == INUNDATION_DISCLAIMER

    def test_custom_inundation_disclaimer_rejected(self) -> None:
        """Inundation disclaimer must match the standard text exactly."""
        with pytest.raises(ValidationError, match="standard text"):
            ScenarioAssessment(
                producer="scenario_agent",
                constraint_stage=ConstraintStage.DART_CONSTRAINED,
                dart_stations_used=["DART_21413"],
                inversion_window_sec=480,
                top_scenarios=[
                    RankedScenario(
                        unit_source_ids=["USRC_001"],
                        weights=[1.0],
                        waveform_rmse_cm=3.8,
                        mw_equivalent=8.1,
                        rank=1,
                        posterior_weight=0.42,
                    )
                ],
                ensemble_spread=EnsembleSpread.MODERATE,
                bilateral_rupture_evaluated=False,
                inundation_disclaimer="Custom disclaimer text",
            )

    def test_seismic_only_with_dart_stations_rejected(self) -> None:
        """SEISMIC_ONLY stage must not list any DART stations."""
        with pytest.raises(ValidationError, match="SEISMIC_ONLY"):
            ScenarioAssessment(
                producer="scenario_agent",
                constraint_stage=ConstraintStage.SEISMIC_ONLY,
                dart_stations_used=["DART_21413"],
                inversion_window_sec=0,
                top_scenarios=[
                    RankedScenario(
                        unit_source_ids=["USRC_001"],
                        weights=[1.0],
                        waveform_rmse_cm=0.0,
                        mw_equivalent=8.1,
                        rank=1,
                        posterior_weight=1.0,
                    )
                ],
                ensemble_spread=EnsembleSpread.HIGH,
                bilateral_rupture_evaluated=False,
            )

    def test_dart_constrained_without_stations_rejected(self) -> None:
        """DART_CONSTRAINED stage requires at least 1 DART station."""
        with pytest.raises(ValidationError, match="DART_CONSTRAINED"):
            ScenarioAssessment(
                producer="scenario_agent",
                constraint_stage=ConstraintStage.DART_CONSTRAINED,
                dart_stations_used=[],
                inversion_window_sec=480,
                top_scenarios=[
                    RankedScenario(
                        unit_source_ids=["USRC_001"],
                        weights=[1.0],
                        waveform_rmse_cm=3.8,
                        mw_equivalent=8.1,
                        rank=1,
                        posterior_weight=0.42,
                    )
                ],
                ensemble_spread=EnsembleSpread.MODERATE,
                bilateral_rupture_evaluated=False,
            )

    def test_multi_station_with_one_station_rejected(self) -> None:
        """MULTI_STATION stage requires at least 2 DART stations."""
        with pytest.raises(ValidationError, match="MULTI_STATION"):
            ScenarioAssessment(
                producer="scenario_agent",
                constraint_stage=ConstraintStage.MULTI_STATION,
                dart_stations_used=["DART_21413"],
                inversion_window_sec=480,
                top_scenarios=[
                    RankedScenario(
                        unit_source_ids=["USRC_001"],
                        weights=[1.0],
                        waveform_rmse_cm=3.8,
                        mw_equivalent=8.1,
                        rank=1,
                        posterior_weight=0.42,
                    )
                ],
                ensemble_spread=EnsembleSpread.MODERATE,
                bilateral_rupture_evaluated=False,
            )


class TestCoastalProxy:
    """Verify CoastalProxy validates percentile ordering."""

    def test_valid_proxy(self) -> None:
        proxy = CoastalProxy(
            site_id="SITE_001",
            arrival_utc="2026-02-27T02:00:00Z",
            arrival_uncertainty_min=15.0,
            amplitude_proxy_p10_m=0.3,
            amplitude_proxy_p50_m=0.8,
            amplitude_proxy_p90_m=2.1,
            tidal_correction_applied=True,
        )
        assert proxy.amplitude_proxy_p50_m == 0.8

    def test_equal_percentiles_accepted(self) -> None:
        """Equal values (e.g., very tight distribution) are valid."""
        proxy = CoastalProxy(
            site_id="SITE_001",
            arrival_utc="2026-02-27T02:00:00Z",
            arrival_uncertainty_min=15.0,
            amplitude_proxy_p10_m=1.0,
            amplitude_proxy_p50_m=1.0,
            amplitude_proxy_p90_m=1.0,
            tidal_correction_applied=False,
        )
        assert proxy.amplitude_proxy_p10_m == 1.0

    def test_out_of_order_percentiles_rejected(self) -> None:
        """p10 > p50 violates the statistical invariant."""
        with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
            CoastalProxy(
                site_id="SITE_001",
                arrival_utc="2026-02-27T02:00:00Z",
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=2.0,
                amplitude_proxy_p50_m=0.5,
                amplitude_proxy_p90_m=3.0,
                tidal_correction_applied=False,
            )

    def test_p50_exceeds_p90_rejected(self) -> None:
        """p50 > p90 violates the statistical invariant."""
        with pytest.raises(ValidationError, match="p10 <= p50 <= p90"):
            CoastalProxy(
                site_id="SITE_001",
                arrival_utc="2026-02-27T02:00:00Z",
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=0.3,
                amplitude_proxy_p50_m=3.0,
                amplitude_proxy_p90_m=2.0,
                tidal_correction_applied=False,
            )


class TestVerificationResult:
    def test_pass_result(self) -> None:
        result = VerificationResult(
            producer="verification_agent",
            overall=VerificationOutcome.PASS,
            checks=[
                VerificationCheck(
                    name="holdout_station_validation",
                    result=CheckResult.PASS,
                    evidence="Withheld station predicted within tolerance",
                )
            ],
            abstain_required=False,
        )
        assert result.overall == VerificationOutcome.PASS
        assert not result.abstain_required

    def test_fail_requires_abstain(self) -> None:
        result = VerificationResult(
            producer="verification_agent",
            overall=VerificationOutcome.FAIL,
            checks=[
                VerificationCheck(
                    name="data_coverage",
                    result=CheckResult.FAIL,
                    evidence="Only 1 usable DART station",
                )
            ],
            abstain_required=True,
            abstain_reason="Insufficient DART coverage for reliable constraint",
        )
        assert result.overall == VerificationOutcome.FAIL
        assert result.abstain_required

    def test_fail_without_abstain_rejected(self) -> None:
        """FAIL verdict with abstain_required=False violates safety invariant."""
        with pytest.raises(
            ValidationError, match="abstain_required=False.*FAIL"
        ):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.FAIL,
                checks=[
                    VerificationCheck(
                        name="data_coverage",
                        result=CheckResult.FAIL,
                        evidence="Only 1 usable DART station",
                    )
                ],
                abstain_required=False,
            )

    def test_abstain_required_needs_reason(self) -> None:
        """abstain_required=True must have an abstain_reason."""
        with pytest.raises(ValidationError, match="abstain_reason is required"):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.FAIL,
                checks=[
                    VerificationCheck(
                        name="data_coverage",
                        result=CheckResult.FAIL,
                        evidence="Only 1 usable DART station",
                    )
                ],
                abstain_required=True,
                abstain_reason=None,
            )


class TestEscalationPacket:
    def test_valid_packet_with_verification(self) -> None:
        """Post-verification escalation includes verification status."""
        packet = EscalationPacket(
            producer="orchestrator",
            escalation_trigger="anomaly_score >= T3",
            criticality_reasons=["Anomaly score 0.91 exceeds escalation threshold"],
            verification_status=VerificationOutcome.PASS_WITH_CONCERNS,
        )
        assert packet.type == "EscalationPacket"
        assert len(packet.criticality_reasons) == 1
        assert packet.verification_status == VerificationOutcome.PASS_WITH_CONCERNS

    def test_valid_packet_before_verification(self) -> None:
        """Pre-verification escalation has no verification status."""
        packet = EscalationPacket(
            producer="orchestrator",
            escalation_trigger="anomaly_score >= T3",
            criticality_reasons=["Score 0.92 >= T3 (0.85)"],
        )
        assert packet.verification_status is None

    def test_rejects_invalid_active_dart_station_id(self) -> None:
        with pytest.raises(ValidationError):
            EscalationPacket(
                producer="orchestrator",
                escalation_trigger="anomaly_score >= T3",
                criticality_reasons=["Score 0.92 >= T3 (0.85)"],
                active_dart_stations=["Warning"],
            )


class TestHumanDecision:
    def test_approve_decision(self) -> None:
        decision = HumanDecision(
            producer="human_review_gate",
            reviewer_id="scientist_jdoe",
            decision=ReviewDecision.APPROVE,
            decision_reason="Scenarios consistent with seismic data",
            decided_at_utc="2026-02-27T01:30:00Z",
            escalation_packet_id=uuid4(),
        )
        assert decision.decision == ReviewDecision.APPROVE
        # Pydantic should parse the ISO string into a datetime object
        assert isinstance(decision.decided_at_utc, datetime)

    def test_datetime_accepts_iso_string(self) -> None:
        """Verify Pydantic auto-parses ISO 8601 strings."""
        decision = HumanDecision(
            producer="human_review_gate",
            reviewer_id="scientist_jdoe",
            decision=ReviewDecision.REJECT,
            decision_reason="Insufficient evidence",
            decided_at_utc="2026-02-27T01:30:00+00:00",
            escalation_packet_id=uuid4(),
        )
        assert decision.decided_at_utc.year == 2026

    def test_datetime_accepts_datetime_object(self) -> None:
        """Verify native datetime objects are accepted too."""
        now = datetime.now(UTC)
        decision = HumanDecision(
            producer="human_review_gate",
            reviewer_id="scientist_jdoe",
            decision=ReviewDecision.DEFER,
            decision_reason="Need more data",
            decided_at_utc=now,
            escalation_packet_id=uuid4(),
        )
        assert decision.decided_at_utc == now

    def test_naive_datetime_rejected(self) -> None:
        """Naive datetimes (no timezone) must be rejected for safety."""
        naive = datetime(2026, 2, 27, 1, 30, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            HumanDecision(
                producer="human_review_gate",
                reviewer_id="scientist_jdoe",
                decision=ReviewDecision.APPROVE,
                decision_reason="Test",
                decided_at_utc=naive,
                escalation_packet_id=uuid4(),
            )


class TestQCReportMeasurementType:
    """Verify measurement_type is optional for non-DART stations."""

    def test_coops_without_measurement_type(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        report = QCReport(
            producer="qc_agent",
            station_id="COOPS_8443970",
            observed_at_utc="2026-02-27T01:15:00Z",
            data_mode=DataMode.STANDARD,
            record_usable=True,
            qartod_flags=flags,
            station_confidence=0.92,
            provenance_hash="d" * 64,
        )
        assert report.measurement_type is None
        assert report.station_id == "COOPS_8443970"

    def test_dart_with_measurement_type(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        report = QCReport(
            producer="qc_agent",
            station_id="DART_21413",
            observed_at_utc="2026-02-27T01:15:00Z",
            measurement_type=3,
            data_mode=DataMode.EVENT,
            record_usable=True,
            qartod_flags=flags,
            station_confidence=0.87,
            provenance_hash="a" * 64,
        )
        assert report.measurement_type == 3


class TestAnomalyAssessmentDefaults:
    """Verify AnomalyAssessment state fields default properly."""

    def test_state_fields_default(self) -> None:
        assessment = AnomalyAssessment(
            producer="anomaly_agent",
            anomaly_score=0.72,
            score_components=ScoreComponents(
                threshold=0.68,
                statistical=0.78,
            ),
            seismic_quiet=True,
            meteotsunami_score=0.12,
        )
        assert assessment.current_state == ""
        assert assessment.state_changed is False

    def test_state_fields_set_by_orchestrator(self) -> None:
        assessment = AnomalyAssessment(
            producer="anomaly_agent",
            current_state="INVESTIGATE",
            state_changed=True,
            anomaly_score=0.72,
            score_components=ScoreComponents(
                threshold=0.68,
                statistical=0.78,
            ),
            seismic_quiet=True,
            meteotsunami_score=0.12,
        )
        assert assessment.current_state == "INVESTIGATE"
        assert assessment.state_changed is True


class TestFinalAssessment:
    def test_valid_final(self) -> None:
        assessment = FinalAssessment(
            producer="report_agent",
            status=AssessmentStatus.PROVISIONAL,
            report_tier=1,
            summary="Structured assessment text",
            uncertainty=UncertaintyInfo(
                confidence_level=ConfidenceLevel.MODERATE,
                key_uncertainties=["Limited azimuthal coverage"],
            ),
            provenance_bundle_id=str(uuid4()),
        )
        assert assessment.status == AssessmentStatus.PROVISIONAL
        assert assessment.disclaimer.startswith("Non-authoritative")

    def test_abstain_status(self) -> None:
        assessment = FinalAssessment(
            producer="report_agent",
            status=AssessmentStatus.ABSTAIN,
            report_tier=1,
            summary="Insufficient constraint",
            uncertainty=UncertaintyInfo(
                confidence_level=ConfidenceLevel.LOW,
                key_uncertainties=["Single DART station", "Poor azimuthal coverage"],
            ),
            provenance_bundle_id=str(uuid4()),
        )
        assert assessment.status == AssessmentStatus.ABSTAIN

    def test_custom_disclaimer_rejected(self) -> None:
        """Disclaimer must match the standard non-authoritative text exactly."""
        with pytest.raises(ValidationError, match="standard non-authoritative text"):
            FinalAssessment(
                producer="report_agent",
                status=AssessmentStatus.PROVISIONAL,
                report_tier=1,
                summary="Test",
                uncertainty=UncertaintyInfo(
                    confidence_level=ConfidenceLevel.MODERATE,
                ),
                disclaimer="Custom disclaimer text",
                provenance_bundle_id=str(uuid4()),
            )

    def test_empty_disclaimer_rejected(self) -> None:
        """Empty disclaimer must be rejected."""
        with pytest.raises(ValidationError, match="standard non-authoritative text"):
            FinalAssessment(
                producer="report_agent",
                status=AssessmentStatus.PROVISIONAL,
                report_tier=1,
                summary="Test",
                uncertainty=UncertaintyInfo(
                    confidence_level=ConfidenceLevel.MODERATE,
                ),
                disclaimer="",
                provenance_bundle_id=str(uuid4()),
            )


class TestMeasurementTypeConstraint:
    """Verify measurement_type is constrained to DART codes 1, 2, 3."""

    def test_invalid_measurement_type_rejected(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        with pytest.raises(ValidationError):
            QCReport(
                producer="qc_agent",
                station_id="DART_21413",
                observed_at_utc="2026-02-27T01:15:00Z",
                measurement_type=5,
                data_mode=DataMode.EVENT,
                record_usable=True,
                qartod_flags=flags,
                station_confidence=0.87,
                provenance_hash="a" * 64,
            )

    def test_zero_measurement_type_rejected(self) -> None:
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        with pytest.raises(ValidationError):
            QCReport(
                producer="qc_agent",
                station_id="DART_21413",
                observed_at_utc="2026-02-27T01:15:00Z",
                measurement_type=0,
                data_mode=DataMode.EVENT,
                record_usable=True,
                qartod_flags=flags,
                station_confidence=0.87,
                provenance_hash="a" * 64,
            )


class TestScenarioConstraints:
    """Verify tightened constraints on RankedScenario fields."""

    def test_negative_magnitude_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankedScenario(
                unit_source_ids=["USRC_001"],
                weights=[1.0],
                waveform_rmse_cm=3.8,
                mw_equivalent=-1.0,
                rank=1,
                posterior_weight=0.42,
            )

    def test_magnitude_above_10_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankedScenario(
                unit_source_ids=["USRC_001"],
                weights=[1.0],
                waveform_rmse_cm=3.8,
                mw_equivalent=11.0,
                rank=1,
                posterior_weight=0.42,
            )

    def test_empty_unit_sources_rejected(self) -> None:
        with pytest.raises(ValidationError):
            RankedScenario(
                unit_source_ids=[],
                weights=[],
                waveform_rmse_cm=3.8,
                mw_equivalent=8.1,
                rank=1,
                posterior_weight=0.42,
            )


class TestNaiveDatetimeRejectionOnAllUTCFields:
    """Verify that all AwareDatetime fields reject naive datetimes (F06, F07)."""

    def test_qc_report_observed_at_utc_rejects_naive(self) -> None:
        """QCReport.observed_at_utc must reject naive datetimes."""
        naive = datetime(2026, 2, 27, 1, 15, 0)  # No tzinfo
        flags = QARTODFlags(
            timing=QARTODFlag.PASS,
            range=QARTODFlag.PASS,
            rate_of_change=QARTODFlag.PASS,
            spike=QARTODFlag.PASS,
            latency=QARTODFlag.PASS,
        )
        with pytest.raises(ValidationError, match="timezone-aware"):
            QCReport(
                producer="qc_agent",
                station_id="DART_21413",
                observed_at_utc=naive,
                measurement_type=3,
                data_mode=DataMode.EVENT,
                record_usable=True,
                qartod_flags=flags,
                station_confidence=0.87,
                provenance_hash="a" * 64,
            )

    def test_coastal_proxy_arrival_utc_rejects_naive(self) -> None:
        """CoastalProxy.arrival_utc must reject naive datetimes."""
        naive = datetime(2026, 2, 27, 2, 0, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            CoastalProxy(
                site_id="SITE_001",
                arrival_utc=naive,
                arrival_uncertainty_min=15.0,
                amplitude_proxy_p10_m=0.3,
                amplitude_proxy_p50_m=0.8,
                amplitude_proxy_p90_m=2.1,
                tidal_correction_applied=True,
            )

    def test_spatial_confirmation_expected_arrival_utc_rejects_naive(self) -> None:
        """SpatialConfirmation.expected_arrival_utc must reject naive datetimes."""
        naive = datetime(2026, 2, 27, 2, 0, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            SpatialConfirmation(
                station_id="DART_21414",
                expected_arrival_utc=naive,
                confirmed=False,
            )

    def test_spatial_confirmation_observed_arrival_utc_rejects_naive(self) -> None:
        """SpatialConfirmation.observed_arrival_utc must reject naive datetimes."""
        naive = datetime(2026, 2, 27, 2, 15, 0)  # No tzinfo
        with pytest.raises(ValidationError, match="timezone-aware"):
            SpatialConfirmation(
                station_id="DART_21414",
                expected_arrival_utc=datetime(2026, 2, 27, 2, 0, 0, tzinfo=UTC),
                observed_arrival_utc=naive,
                confirmed=True,
                delta_min=15.0,
            )


class TestSpatialConfirmationInvariant:
    """Verify SpatialConfirmation cross-field invariant."""

    def test_confirmed_without_observation_rejected(self) -> None:
        """confirmed=True with observed_arrival_utc=None is contradictory."""
        with pytest.raises(ValidationError, match="confirmed=True requires observed_arrival_utc"):
            SpatialConfirmation(
                station_id="DART_21414",
                expected_arrival_utc=datetime(2026, 2, 27, 2, 0, 0, tzinfo=UTC),
                confirmed=True,
                # observed_arrival_utc=None (default)
            )

    def test_confirmed_without_delta_rejected(self) -> None:
        """confirmed=True with delta_min=None is contradictory."""
        with pytest.raises(ValidationError, match="confirmed=True requires delta_min"):
            SpatialConfirmation(
                station_id="DART_21414",
                expected_arrival_utc=datetime(2026, 2, 27, 2, 0, 0, tzinfo=UTC),
                observed_arrival_utc=datetime(2026, 2, 27, 2, 15, 0, tzinfo=UTC),
                confirmed=True,
                # delta_min=None (default)
            )

    def test_unconfirmed_without_observation_allowed(self) -> None:
        """confirmed=False with observed_arrival_utc=None is valid (not yet seen)."""
        sc = SpatialConfirmation(
            station_id="DART_21414",
            expected_arrival_utc=datetime(2026, 2, 27, 2, 0, 0, tzinfo=UTC),
            confirmed=False,
        )
        assert sc.observed_arrival_utc is None


# ---------------------------------------------------------------------------
# Edge case tests from code review
# ---------------------------------------------------------------------------


class TestWhitespaceOnlyDecisionReason:
    """Document that whitespace-only decision_reason passes min_length=1.

    Code review section 4 issue #2: human_decision.py:49 uses min_length=1
    which accepts " " (single space). This is known behavior - the
    min_length check counts characters, not stripped characters.
    """

    def test_whitespace_only_decision_reason_accepted(self) -> None:
        hd = HumanDecision(
            producer="test",
            reviewer_id="alice",
            decision=ReviewDecision.APPROVE,
            decision_reason=" ",  # whitespace only
            decided_at_utc=_FIXED_DECISION_TIME,
            escalation_packet_id=_FIXED_ESCALATION_PACKET_ID,
        )
        assert hd.decision_reason == " "


class TestPrecomputedHashAccepted:
    """Document that pre-computed hashes are accepted on deserialization.

    Code review section 4 issue #3: By design for Pydantic deserialization
    roundtrip. The model_validator only computes a hash when
    decision_hash is empty. When deserializing from stored data,
    the pre-existing hash is preserved without re-verification.

    Safety note: The API layer (app.py /api/review endpoint) always
    constructs HumanDecision without pre-setting decision_hash, so
    the validator always computes a fresh hash for real decisions.
    This test documents the schema-level behavior, not an exploitable
    bypass.
    """

    def test_bogus_hash_accepted_on_construction(self) -> None:
        """A pre-computed (bogus) decision_hash is accepted at the schema level."""
        hd = HumanDecision(
            producer="test",
            reviewer_id="bob",
            decision=ReviewDecision.REJECT,
            decision_reason="Looks wrong",
            decided_at_utc=_FIXED_DECISION_TIME,
            escalation_packet_id=_FIXED_ESCALATION_PACKET_ID,
            decision_hash="a" * 64,  # bogus SHA-256
        )
        # The validator sees a non-empty hash and keeps it
        assert hd.decision_hash == "a" * 64

    def test_empty_hash_triggers_recomputation(self) -> None:
        """An empty decision_hash triggers fresh computation."""
        hd = HumanDecision(
            producer="test",
            reviewer_id="bob",
            decision=ReviewDecision.APPROVE,
            decision_reason="Looks correct",
            decided_at_utc=_FIXED_DECISION_TIME,
            escalation_packet_id=_FIXED_ESCALATION_PACKET_ID,
        )
        assert len(hd.decision_hash) == 64
        assert hd.decision_hash != "a" * 64  # was computed, not bogus


class TestProvenanceBundleIdType:
    """Verify provenance_bundle_id rejects non-UUID strings."""

    def test_invalid_provenance_bundle_id_rejected(self) -> None:
        with pytest.raises(ValidationError):
            FinalAssessment(
                producer="report_agent",
                status=AssessmentStatus.PROVISIONAL,
                report_tier=1,
                summary="Test",
                uncertainty=UncertaintyInfo(
                    confidence_level=ConfidenceLevel.MODERATE,
                ),
                provenance_bundle_id="NOT_A_UUID",
            )


class TestMinLengthConstraints:
    """Verify min_length constraints reject empty lists (F08, F09, F11)."""

    def test_escalation_packet_empty_criticality_reasons_rejected(self) -> None:
        """EscalationPacket.criticality_reasons requires min_length=1."""
        with pytest.raises(ValidationError):
            EscalationPacket(
                producer="orchestrator",
                criticality_reasons=[],
            )

    def test_verification_result_empty_checks_rejected(self) -> None:
        """VerificationResult.checks requires min_length=1."""
        with pytest.raises(ValidationError):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.PASS,
                checks=[],
                abstain_required=False,
            )

    def test_scenario_assessment_empty_top_scenarios_rejected(self) -> None:
        """ScenarioAssessment.top_scenarios requires min_length=1."""
        with pytest.raises(ValidationError):
            ScenarioAssessment(
                producer="scenario_agent",
                constraint_stage=ConstraintStage.SEISMIC_ONLY,
                inversion_window_sec=0,
                top_scenarios=[],
                ensemble_spread=EnsembleSpread.HIGH,
                bilateral_rupture_evaluated=False,
            )


class TestVerificationPassWithConcerns:
    """Verify PASS_WITH_CONCERNS is a valid non-abstain path (F12)."""

    def test_pass_with_concerns_and_no_abstain_accepted(self) -> None:
        """PASS_WITH_CONCERNS with abstain_required=False is valid."""
        result = VerificationResult(
            producer="verification_agent",
            overall=VerificationOutcome.PASS_WITH_CONCERNS,
            checks=[
                VerificationCheck(
                    name="holdout_station",
                    result=CheckResult.CONCERN,
                    evidence="Within tolerance but marginal",
                )
            ],
            abstain_required=False,
        )
        assert result.overall == VerificationOutcome.PASS_WITH_CONCERNS
        assert not result.abstain_required

    def test_pass_with_concerns_and_abstain_rejected(self) -> None:
        """Aggregate and ABSTAIN flag must describe the same detailed checks."""
        with pytest.raises(
            ValidationError,
            match="abstain_required=True.*PASS_WITH_CONCERNS",
        ):
            VerificationResult(
                producer="verification_agent",
                overall=VerificationOutcome.PASS_WITH_CONCERNS,
                checks=[
                    VerificationCheck(
                        name="holdout_station",
                        result=CheckResult.CONCERN,
                        evidence="Marginal fit, coverage poor",
                    )
                ],
                abstain_required=True,
                abstain_reason="Coverage too poor despite marginal fit",
            )
