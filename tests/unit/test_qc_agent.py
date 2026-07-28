"""Unit tests for QCAgent (QCReport emission and ordering).

Tests the full QCAgent flow: ingest records -> QCReports with all
required fields populated and schema-validated.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hazard_assessment.agents.qc_agent import (
    QCAgent,
    coops_record_to_qc_obs,
    dart_record_to_qc_obs,
)
from hazard_assessment.agents.qc_checks import CONFIDENCE_EXCLUSION_THRESHOLD
from hazard_assessment.ingest.coops import CoopsRecord
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.schemas.qc import DataMode, QARTODFlag, QCReport

_T0 = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)
_HASH = "a" * 64
_HASH_B = "b" * 64


def _dart_record(
    *,
    ts: datetime = _T0,
    height_m: float = 0.0,
    measurement_type: int = 1,
    event_mode: bool = False,
    station_id: str = "21413",
    payload_sha256: str = _HASH,
) -> DartRecord:
    return DartRecord(
        source_id=f"dart:{station_id}",
        station_id=station_id,
        source_timestamp=ts,
        ingest_timestamp=ts,
        measurement_type=measurement_type,
        height_m=height_m,
        event_mode=event_mode,
        payload_sha256=payload_sha256,
    )


def _coops_record(
    *,
    ts: datetime = _T0,
    water_level_m: float | None = 0.5,
    station_id: str = "1612340",
    payload_sha256: str = _HASH,
) -> CoopsRecord:
    return CoopsRecord(
        source_id=f"coops:{station_id}",
        station_id=station_id,
        station_name="Honolulu",
        product="one_minute_water_level",
        source_timestamp=ts,
        ingest_timestamp=ts,
        water_level_m=water_level_m,
        flags="0,0,0,0",
        quality="p",
        payload_sha256=payload_sha256,
    )


# ===================================================================
# Record conversion
# ===================================================================


class TestRecordConversion:
    def test_dart_record_conversion(self) -> None:
        rec = _dart_record(height_m=4541.234, measurement_type=2, event_mode=True)
        obs = dart_record_to_qc_obs(rec)
        assert obs.source_type == "dart"
        assert obs.value_m == 4541.234
        assert obs.measurement_type == 2
        assert obs.event_mode is True
        assert obs.expected_interval_sec == 60.0

    def test_dart_standard_mode_interval(self) -> None:
        rec = _dart_record(measurement_type=1)
        obs = dart_record_to_qc_obs(rec)
        assert obs.expected_interval_sec == 900.0

    def test_dict_conversion_requires_strict_bool_event_mode(self) -> None:
        """Same strict-bool rule as worker ingestion: a malformed event_mode
        ("false", 1) must not read as event mode in QC metadata."""
        from hazard_assessment.agents.qc_agent import qc_observation_from_dict

        base = {
            "source_timestamp": datetime(2026, 3, 4, 8, 0, tzinfo=UTC),
            "height_m": 4541.234,
            "measurement_type": 2,
            "payload_sha256": "a" * 64,
        }
        for malformed in ("false", "true", 1, 0, None):
            obs = qc_observation_from_dict(
                "dart", "21413", {**base, "event_mode": malformed}
            )
            assert obs.event_mode is False
        obs = qc_observation_from_dict("dart", "21413", {**base, "event_mode": True})
        assert obs.event_mode is True

    def test_coops_record_conversion(self) -> None:
        rec = _coops_record(water_level_m=0.584)
        obs = coops_record_to_qc_obs(rec)
        assert obs.source_type == "coops"
        assert obs.value_m == 0.584
        assert obs.measurement_type is None
        assert obs.event_mode is False
        assert obs.expected_interval_sec == 60.0

    def test_coops_none_water_level(self) -> None:
        rec = _coops_record(water_level_m=None)
        obs = coops_record_to_qc_obs(rec)
        assert obs.value_m is None


# ===================================================================
# QCReport emission
# ===================================================================


class TestQCReportEmission:
    def test_single_dart_report(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_dart_record()])
        assert len(reports) == 1
        report = reports[0]
        assert isinstance(report, QCReport)
        assert report.producer == "qc_agent"
        assert report.station_id == "21413"
        assert report.data_mode == DataMode.STANDARD
        assert report.event_mode_note == ""
        assert report.provenance_hash == _HASH
        assert report.schema_version == "1.0"

    def test_event_mode_note(self) -> None:
        agent = QCAgent()
        reports = agent.process_records(
            [_dart_record(event_mode=True, measurement_type=2)]
        )
        assert reports[0].data_mode == DataMode.EVENT
        assert "Event mode" in reports[0].event_mode_note

    def test_coops_report(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_coops_record()])
        report = reports[0]
        assert report.station_id == "1612340"
        assert report.measurement_type is None
        assert report.data_mode == DataMode.STANDARD

    def test_mixed_records(self) -> None:
        agent = QCAgent()
        records = [
            _dart_record(ts=_T0, station_id="21413"),
            _coops_record(ts=_T0, station_id="1612340"),
        ]
        reports = agent.process_records(records)
        assert len(reports) == 2
        station_ids = {r.station_id for r in reports}
        assert station_ids == {"21413", "1612340"}

    def test_empty_records(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([])
        assert reports == []

    def test_input_refs_populated(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_dart_record()])
        report = reports[0]
        assert len(report.input_refs) == 1
        assert report.input_refs[0].source == "dart"
        assert report.input_refs[0].sha256 == _HASH

    def test_coops_input_refs(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_coops_record()])
        assert reports[0].input_refs[0].source == "coops"

    def test_qartod_flags_present(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_dart_record()])
        flags = reports[0].qartod_flags
        # First record has no prev so timing/spike/roc are NOT_EVALUATED
        assert flags.timing == QARTODFlag.NOT_EVALUATED
        # DART gross range is NOT_EVALUATED (raw height, not residual)
        assert flags.range == QARTODFlag.NOT_EVALUATED

    def test_confidence_score_range(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_dart_record()])
        assert 0.0 <= reports[0].station_confidence <= 1.0

    def test_record_usable_above_threshold(self) -> None:
        agent = QCAgent()
        reports = agent.process_records([_dart_record(height_m=0.0)])
        assert reports[0].record_usable is True
        # A stream-head record has no history: every check is indeterminate,
        # so the report must expose zero evaluated checks (confidence 1.0 is
        # the no-evidence convention, not an evidence-backed score).
        assert reports[0].n_checks_evaluated == 0

    def test_unsupported_record_type_raises(self) -> None:
        import pytest

        agent = QCAgent()
        with pytest.raises(TypeError, match="Unsupported record type"):
            agent.process_records(["not_a_record"])  # type: ignore[list-item]


# ===================================================================
# Stateful history across batches
# ===================================================================


class TestStatefulHistory:
    def test_second_batch_uses_history(self) -> None:
        """Agent remembers previous batch for spike/timing/flat-line checks."""
        agent = QCAgent()
        # First batch
        r1 = _dart_record(ts=_T0, height_m=0.0, payload_sha256=_HASH)
        agent.process_records([r1], processing_time=_T0)
        # Second batch
        r2 = _dart_record(
            ts=_T0 + timedelta(seconds=15),
            height_m=0.01,
            payload_sha256=_HASH_B,
        )
        reports = agent.process_records(
            [r2], processing_time=_T0 + timedelta(seconds=15)
        )
        # Should have prev from first batch -> spike and roc evaluated
        flags = reports[0].qartod_flags
        assert flags.spike != QARTODFlag.NOT_EVALUATED
        assert flags.rate_of_change != QARTODFlag.NOT_EVALUATED

    def test_replay_without_processing_time_preserves_history(self) -> None:
        """Replaying historical records must not wipe the flat-line history.

        When processing_time is omitted (the live-worker path), the prune
        cutoff must follow the newest observation, not wall-clock, so the
        retained history for a replayed stream is not deleted as stale.
        """
        agent = QCAgent()
        # First batch: a historical record, no processing_time (replay).
        r1 = _dart_record(ts=_T0, height_m=0.0, payload_sha256=_HASH)
        agent.process_records([r1])
        # The historical record must still be in retained history (history is
        # keyed by the source-prefixed station id).
        assert len(agent._station_history["dart:21413"]) == 1
        # Second batch: next record 15s later, still no processing_time.
        r2 = _dart_record(
            ts=_T0 + timedelta(seconds=15), height_m=0.0, payload_sha256=_HASH_B
        )
        reports = agent.process_records([r2])
        # History carried across batches -> flat-line is evaluated, not
        # silently NOT_EVALUATED from a wiped window.
        assert len(agent._station_history["dart:21413"]) == 2
        assert reports[0].qartod_flags.flat_line != QARTODFlag.NOT_EVALUATED


# ===================================================================
# Out-of-order handling through the agent
# ===================================================================


class TestOutOfOrderHandling:
    def test_out_of_order_arrivals_sorted(self) -> None:
        """Records arriving out of order are sorted before QC checks."""
        agent = QCAgent()
        late = _dart_record(
            ts=_T0 + timedelta(seconds=30),
            height_m=0.01,
            payload_sha256=_HASH_B,
        )
        early = _dart_record(
            ts=_T0,
            height_m=0.0,
            payload_sha256=_HASH,
        )
        # Pass late first, early second
        reports = agent.process_records([late, early])
        assert len(reports) == 2
        # Reports should be in sorted order (early first)
        assert reports[0].observed_at_utc < reports[1].observed_at_utc

    def test_deterministic_output(self) -> None:
        """Same records in different order produce same reports."""
        r1 = _dart_record(ts=_T0, height_m=0.0, payload_sha256=_HASH)
        r2 = _dart_record(
            ts=_T0 + timedelta(seconds=15),
            height_m=0.01,
            payload_sha256=_HASH_B,
        )

        agent_a = QCAgent()
        reports_a = agent_a.process_records([r1, r2])

        agent_b = QCAgent()
        reports_b = agent_b.process_records([r2, r1])

        assert len(reports_a) == len(reports_b)
        for a, b in zip(reports_a, reports_b):
            assert a.station_id == b.station_id
            assert a.observed_at_utc == b.observed_at_utc
            assert a.qartod_flags == b.qartod_flags
            assert a.station_confidence == b.station_confidence


# ===================================================================
# Edge cases: all-fail, boundary values
# ===================================================================


class TestEdgeCases:
    def test_all_fail_record_not_usable(self) -> None:
        """Record with extremely bad data should be excluded.

        Use CO-OPS so gross range is evaluated. Need 3+ FAILs out of
        5 counted tests to get confidence < 0.5.
        Gap of 3 min (>2x 60s expected, <4x spike window of 60s ->
        spike evaluated). Value jump 10m in 180s:
        - range: 10.0 >> [-4.5, 4.5] -> FAIL
        - spike: 10.0 >> 0.3*2=0.6 -> FAIL
        - roc: 10/180 = 0.056 >> 0.005*2=0.01 -> FAIL
        - timing: 180s > 2*60=120s but <= 4*60=240s -> SUSPECT
        - flat_line: variation 10.0 >> 0.001 -> PASS
        Total: 3 FAILs + 1 SUSPECT + 1 PASS / 5 = confidence 0.34.
        """
        agent = QCAgent()
        r1 = _coops_record(ts=_T0, water_level_m=0.0)
        r2 = _coops_record(
            ts=_T0 + timedelta(seconds=180),  # 3 min gap (>2x but <4x expected)
            water_level_m=10.0,  # way out of [-3, 3] range, huge spike + rate
            payload_sha256=_HASH_B,
        )
        reports = agent.process_records([r1, r2])
        bad_report = reports[1]
        assert bad_report.station_confidence < CONFIDENCE_EXCLUSION_THRESHOLD
        assert bad_report.record_usable is False

    def test_missing_value_record(self) -> None:
        """Record with None value gets MISSING flags."""
        agent = QCAgent()
        rec = _coops_record(water_level_m=None)
        reports = agent.process_records([rec])
        flags = reports[0].qartod_flags
        assert flags.range == QARTODFlag.MISSING
        assert flags.flat_line == QARTODFlag.MISSING

    def test_many_records_performance(self) -> None:
        """1000 records process without raising."""
        agent = QCAgent()
        records = [
            _dart_record(
                ts=_T0 + timedelta(seconds=15 * i),
                height_m=0.001 * (i % 10),
                payload_sha256=f"{i:064x}",
            )
            for i in range(1000)
        ]
        reports = agent.process_records(records)
        assert len(reports) == 1000
        assert all(isinstance(r, QCReport) for r in reports)

    def test_schema_validation_passes(self) -> None:
        """QCReport can be serialized and deserialized (schema valid)."""
        agent = QCAgent()
        reports = agent.process_records([_dart_record()])
        report = reports[0]
        data = report.model_dump()
        restored = QCReport.model_validate(data)
        assert restored.station_id == report.station_id
        assert restored.station_confidence == report.station_confidence

    def test_manifest(self) -> None:
        agent = QCAgent()
        assert agent.name == "qc_agent"
        assert agent.manifest.version == "1.0.0"
