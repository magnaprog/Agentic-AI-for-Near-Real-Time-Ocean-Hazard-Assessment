"""Tests for pipeline_runner live-data wiring (workers/pipeline_runner.py)."""

from __future__ import annotations

import dataclasses
import json
import re
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np
import pytest

from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.ingest.seismic import SeismicEventRecord
from hazard_assessment.messaging.producer import SCHEMA_VERSION
from hazard_assessment.orchestrator.states import SystemState
from hazard_assessment.schemas.ocean_evidence import StationScoringStatus
from hazard_assessment.schemas.ocean_evidence_hashing import (
    KafkaMessageCoordinate,
    derive_live_checkpoint_id,
)
from hazard_assessment.storage.client import AssessmentPersistResult
from hazard_assessment.workers.pipeline_runner import (
    FUTURE_TOLERANCE_SEC,
    PACIFIC_TSUNAMIGENIC_ZONES,
    CheckpointSummary,
    CheckpointTransport,
    PipelineWorkerState,
    _build_seismic_identity,
    _count_usable_dart_stations,
    _ingest_observation_records,
    _ingest_seismic_record,
    _process_buffer,
    _run_qc,
    _score_station,
    _score_station_attempt,
    classify_seismic_zone,
)
from hazard_assessment.workers.reviewer_packet import (
    RENDERER_VERSION,
    canonical_packet_hash,
)
from hazard_assessment.workers.station_buffer import DEFAULT_WINDOW_SEC, RetainedSampleQC


class TestClassifySeismicZone:
    """Test geographic zone classification."""

    def test_japan(self) -> None:
        # Tohoku epicenter: 38.3 N, 142.4 E
        assert classify_seismic_zone(38.3, 142.4) == "japan"

    def test_chile_maule(self) -> None:
        # Chile 2010 epicenter: 35.8 S, 72.7 W
        assert classify_seismic_zone(-35.8, -72.7) == "maule"

    def test_cascadia(self) -> None:
        assert classify_seismic_zone(46.0, -125.0) == "cascadia"

    def test_alaska_aleutian(self) -> None:
        assert classify_seismic_zone(55.0, -160.0) == "alaska_aleutian"

    def test_indonesia(self) -> None:
        assert classify_seismic_zone(-5.0, 120.0) == "indonesia"

    def test_tonga(self) -> None:
        assert classify_seismic_zone(-20.0, -175.0) == "tonga_kermadec"

    def test_unknown_region(self) -> None:
        # Mid-Atlantic (not Pacific)
        assert classify_seismic_zone(30.0, -30.0) == "unknown"

    def test_pacific_rim_fallback(self) -> None:
        # Some Pacific location not in specific zones
        assert classify_seismic_zone(10.0, -150.0) == "pacific_rim"

    def test_all_zones_in_tsunamigenic_set(self) -> None:
        """Every zone returned by classify_seismic_zone should be in the set."""
        test_cases = [
            (38.3, 142.4),   # japan
            (50.0, 155.0),   # kuril
            (55.0, 160.0),   # kamchatka
            (55.0, -160.0),  # alaska_aleutian
            (46.0, -125.0),  # cascadia
            (-35.8, -72.7),  # maule
            (-15.0, -75.0),  # peru_chile
            (12.0, -90.0),   # central_america
            (-25.0, -175.0), # tonga_kermadec
            (-8.0, 155.0),   # solomon_islands
            (-5.0, 120.0),   # indonesia
            (12.0, 125.0),   # philippines
            (-40.0, 175.0),  # new_zealand
            (10.0, -150.0),  # pacific_rim (fallback)
        ]
        for lat, lon in test_cases:
            zone = classify_seismic_zone(lat, lon)
            assert zone in PACIFIC_TSUNAMIGENIC_ZONES, (
                f"Zone '{zone}' for ({lat}, {lon}) not in PACIFIC_TSUNAMIGENIC_ZONES"
            )


class TestPipelineWorkerState:
    def test_init_no_calibration(self) -> None:
        worker = PipelineWorkerState()
        assert worker.agent is not None
        assert worker.fsm is not None
        assert len(worker.station_buffers) == 0
        assert len(worker.calibration) == 0
        assert worker.seismic_events == []

    def test_init_with_calibration_dir(self, tmp_path) -> None:
        """Loading from an empty directory should not crash."""
        worker = PipelineWorkerState(calibration_dir=str(tmp_path))
        assert len(worker.calibration) == 0


class TestIngestSeismicRecord:
    def test_valid_seismic_record(self) -> None:
        worker = PipelineWorkerState()
        record = {
            "event_id": "us2010chile",
            "magnitude": 8.8,
            "latitude": -35.846,
            "longitude": -72.719,
            "source_timestamp": "2010-02-27T06:34:11+00:00",
        }

        _ingest_seismic_record(record, worker)

        assert len(worker.seismic_events) == 1
        assert worker.seismic_events[0].magnitude == 8.8
        # FSM should have transitioned to MONITOR (M8.8 in maule zone)
        from hazard_assessment.orchestrator.states import SystemState
        assert worker.fsm.state == SystemState.MONITOR

    def test_missing_fields_skipped(self) -> None:
        worker = PipelineWorkerState()
        record = {"event_id": "test", "magnitude": None}

        _ingest_seismic_record(record, worker)
        assert len(worker.seismic_events) == 0

    def test_invalid_timestamp_skipped(self) -> None:
        worker = PipelineWorkerState()
        record = {
            "event_id": "test",
            "magnitude": 7.0,
            "latitude": 38.0,
            "longitude": 142.0,
            "source_timestamp": "not-a-date",
        }

        _ingest_seismic_record(record, worker)
        assert len(worker.seismic_events) == 0

    def test_old_events_pruned(self) -> None:
        worker = PipelineWorkerState()
        base_time = datetime(2010, 2, 27, 6, 0, 0, tzinfo=UTC)

        # Add an old event (7+ hours ago)
        old_record = {
            "event_id": "old",
            "magnitude": 5.5,
            "latitude": 38.0,
            "longitude": 142.0,
            "source_timestamp": (base_time - timedelta(hours=7)).isoformat(),
        }
        _ingest_seismic_record(old_record, worker)
        assert len(worker.seismic_events) == 1

        # Add a new event - old one should be pruned
        new_record = {
            "event_id": "new",
            "magnitude": 8.0,
            "latitude": 38.0,
            "longitude": 142.0,
            "source_timestamp": base_time.isoformat(),
        }
        _ingest_seismic_record(new_record, worker)
        assert len(worker.seismic_events) == 1
        assert worker.seismic_events[0].event_id == "new"


class TestIngestObservationRecords:
    def test_dart_records(self) -> None:
        worker = PipelineWorkerState()
        records = [
            {
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
            },
            {
                "source_timestamp": "2010-02-27T07:01:00+00:00",
                "height_m": 5827.6,
                "event_mode": True,
            },
        ]

        _ingest_observation_records("dart:21413", records, worker)

        w = worker.station_buffers.get_window("21413", "dart")
        assert w is not None
        assert len(w) == 2
        assert w.source_type == "dart"
        assert w.event_mode is True

    def test_coops_records(self) -> None:
        worker = PipelineWorkerState()
        records = [
            {
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "water_level_m": 0.5,
            },
        ]

        _ingest_observation_records("coops:1612340", records, worker)

        w = worker.station_buffers.get_window("1612340", "coops")
        assert w is not None
        assert w.source_type == "coops"

    def test_missing_timestamp_skipped(self) -> None:
        worker = PipelineWorkerState()
        records = [{"height_m": 5827.0}]  # No source_timestamp

        _ingest_observation_records("dart:21413", records, worker)
        assert worker.station_buffers.get_window("21413", "dart") is None

    def test_missing_value_skipped(self) -> None:
        worker = PipelineWorkerState()
        records = [
            {"source_timestamp": "2010-02-27T07:00:00+00:00"},  # No height_m
        ]

        _ingest_observation_records("dart:21413", records, worker)
        # Window created but no observations added
        w = worker.station_buffers.get_window("21413", "dart")
        assert w is None or len(w) == 0

    def test_equal_ids_from_different_sources_get_separate_windows(self) -> None:
        """A DART buoy and a CO-OPS gauge with the same identifier must not
        share a window."""
        worker = PipelineWorkerState()
        _ingest_observation_records(
            "dart:21413",
            [{
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
            }],
            worker,
        )
        _ingest_observation_records(
            "coops:21413",
            [{
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "water_level_m": 0.5,
            }],
            worker,
        )

        dart_w = worker.station_buffers.get_window("21413", "dart")
        coops_w = worker.station_buffers.get_window("21413", "coops")
        assert dart_w is not None and coops_w is not None
        assert dart_w is not coops_w
        assert dart_w.retained_samples()[0].value == 5827.5
        assert coops_w.retained_samples()[0].value == 0.5

    def test_metadata_rides_with_accepted_samples(self) -> None:
        """Accepted samples carry measurement type or product plus payload
        hash even when no QC map is supplied (qc stays None)."""
        worker = PipelineWorkerState()
        _ingest_observation_records(
            "dart:21413",
            [{
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
                "measurement_type": 2,
                "payload_sha256": "a" * 64,
            }],
            worker,
        )
        _ingest_observation_records(
            "coops:1612340",
            [{
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "water_level_m": 0.5,
                "product": "water_level",
                "payload_sha256": "b" * 64,
            }],
            worker,
        )

        dart_w = worker.station_buffers.get_window("21413", "dart")
        assert dart_w is not None
        dart_sample = dart_w.retained_samples()[0]
        assert dart_sample.measurement_type == 2
        assert dart_sample.product is None
        assert dart_sample.payload_hash == "a" * 64
        assert dart_sample.qc is None

        coops_w = worker.station_buffers.get_window("1612340", "coops")
        assert coops_w is not None
        coops_sample = coops_w.retained_samples()[0]
        assert coops_sample.measurement_type is None
        assert coops_sample.product == "water_level"
        assert coops_sample.payload_hash == "b" * 64
        assert coops_sample.qc is None

    def test_malformed_measurement_type_rides_as_none(self) -> None:
        """A malformed measurement_type (bool or string) must not skip a
        well-formed pressure sample; it rides as None."""
        worker = PipelineWorkerState()
        _ingest_observation_records(
            "dart:21413",
            [
                {
                    "source_timestamp": "2010-02-27T07:00:00+00:00",
                    "height_m": 5827.5,
                    "event_mode": False,
                    "measurement_type": True,
                },
                {
                    "source_timestamp": "2010-02-27T07:01:00+00:00",
                    "height_m": 5827.6,
                    "event_mode": False,
                    "measurement_type": "2",
                },
            ],
            worker,
        )
        w = worker.station_buffers.get_window("21413", "dart")
        assert w is not None
        assert [s.measurement_type for s in w.retained_samples()] == [None, None]


class TestQCJoinBeforeAdmission:
    """QC runs on parseable records before admission and
    the per-record verdicts ride into the buffer keyed by payload hash."""

    def _dart_records(self) -> list[dict[str, Any]]:
        return [
            {
                "source_timestamp": "2010-02-27T07:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
                "measurement_type": 1,
                "payload_sha256": "a" * 64,
            },
            {
                "source_timestamp": "2010-02-27T07:15:00+00:00",
                "height_m": 5827.6,
                "event_mode": False,
                "measurement_type": 1,
                "payload_sha256": "b" * 64,
            },
        ]

    def test_run_qc_returns_per_record_qc_keyed_by_hash(self) -> None:
        worker = PipelineWorkerState()
        qc_by_hash = _run_qc("dart:21413", self._dart_records(), worker)

        assert set(qc_by_hash) == {"a" * 64, "b" * 64}
        for qc in qc_by_hash.values():
            assert isinstance(qc.usable, bool)
            assert 0.0 <= qc.confidence <= 1.0
            assert qc.n_checks_evaluated >= 0
            # All eight QARTOD checks are represented, sorted by name.
            names = [name for name, _ in qc.flags]
            assert names == sorted(names)
            assert len(names) == 8

    def test_malformed_hash_degrades_qc_to_empty_map(self) -> None:
        """A record with a non-canonical payload hash fails QCReport
        construction, so QC for the batch degrades to no metadata (the
        pre-existing best-effort behavior). Ingestion is unaffected."""
        worker = PipelineWorkerState()
        records = self._dart_records()
        records[0]["payload_sha256"] = "not-a-hash"
        qc_by_hash = _run_qc("dart:21413", records, worker)
        assert qc_by_hash == {}

        _ingest_observation_records(
            "dart:21413", records, worker, qc_by_hash=qc_by_hash
        )
        w = worker.station_buffers.get_window("21413", "dart")
        assert w is not None
        assert len(w) == 2
        assert all(s.qc is None for s in w.retained_samples())

    def test_ingest_attaches_qc_from_map(self) -> None:
        worker = PipelineWorkerState()
        records = self._dart_records()
        qc_by_hash = _run_qc("dart:21413", records, worker)
        _ingest_observation_records(
            "dart:21413", records, worker, qc_by_hash=qc_by_hash
        )

        w = worker.station_buffers.get_window("21413", "dart")
        assert w is not None
        samples = w.retained_samples()
        assert len(samples) == 2
        assert samples[0].qc == qc_by_hash["a" * 64]
        assert samples[1].qc == qc_by_hash["b" * 64]


class TestScoreStation:
    def test_insufficient_data_returns_none(self) -> None:
        worker = PipelineWorkerState()
        # Add only a few observations (need >= 10)
        for i in range(5):
            worker.station_buffers.append_dart(
                "21413", 1000.0 + i * 60, 5827.0 + 0.01 * i,
            )

        result = _score_station(("dart", "21413"), worker)
        assert result is None

    def test_unknown_station_returns_none(self) -> None:
        worker = PipelineWorkerState()
        result = _score_station(("dart", "nonexistent"), worker)
        assert result is None

    def test_scoring_with_sufficient_data(self) -> None:
        """With enough data, scoring should return an AnomalyAssessment dict."""
        worker = PipelineWorkerState()

        # Add enough observations (standard DART 15-min intervals, 6+ hours)
        base_epoch = datetime(2010, 2, 27, 0, 0, 0, tzinfo=UTC).timestamp()
        for i in range(30):  # 30 observations at 15-min = 7.5 hours
            t = base_epoch + i * 900  # 900s = 15 min
            height = 5827.0 + 0.3 * np.sin(2 * np.pi * i / 48)
            worker.station_buffers.append_dart("21413", t, height)

        result = _score_station(("dart", "21413"), worker)

        assert result is not None
        assert "anomaly_score" in result
        assert 0.0 <= result["anomaly_score"] <= 1.0
        assert "score_components" in result
        assert "triggering_stations" in result

    def test_rayleigh_prerequisite_requires_seismic_context(self) -> None:
        worker = PipelineWorkerState()
        base_epoch = datetime(2010, 2, 27, tzinfo=UTC).timestamp()
        for i in range(30):
            worker.station_buffers.append_dart(
                "21413",
                base_epoch + i * 900,
                5827.0 + 0.3 * np.sin(2 * np.pi * i / 48),
            )

        attempt, _, _ = _score_station_attempt(("dart", "21413"), worker)

        assert attempt.scores is not None
        assert attempt.rayleigh_inputs_available is False
        assert attempt.scores.rayleigh_wave_suspect is None


class TestProcessBuffer:
    def test_seismic_processed_before_observations(self) -> None:
        """Seismic events should be processed first to set FSM context."""
        worker = PipelineWorkerState()

        # Build a buffer with both seismic and DART records
        buffer: dict[str, list[dict]] = {
            "seismic:us2010chile": [
                {
                    "event_id": "us2010chile",
                    "magnitude": 8.8,
                    "latitude": -35.846,
                    "longitude": -72.719,
                    "source_timestamp": "2010-02-27T06:34:11+00:00",
                },
            ],
            "dart:21413": [
                {
                    "source_timestamp": "2010-02-27T07:00:00+00:00",
                    "height_m": 5827.5,
                    "event_mode": False,
                },
            ],
        }

        _process_buffer(buffer, worker)

        # Seismic event should have been ingested
        assert len(worker.seismic_events) == 1
        # DART observation should be in buffer
        assert worker.station_buffers.get_window("21413", "dart") is not None

    def test_empty_scoring_produces_no_pipeline_run(self) -> None:
        """If no station has enough data for scoring, pipeline doesn't run.

        The station is recorded as an attempt with INSUFFICIENT_RETAINED_DATA,
        but nothing downstream fires: no scored assessment, no pipeline
        outcome, no FSM movement, and no audit entry. This asserted only that
        the call did not raise, so a regression that scored a one-sample
        window and drove the FSM would still have passed.
        """
        worker = PipelineWorkerState()
        state_before = worker.fsm.state
        audit_before = worker.audit_logger.count

        buffer: dict[str, list[dict]] = {
            "dart:21413": [
                {
                    "source_timestamp": "2010-02-27T07:00:00+00:00",
                    "height_m": 5827.5,
                    "event_mode": False,
                },
            ],
        }

        summary = _process_buffer(buffer, worker)

        assert summary is not None
        assert summary.n_scored_assessments == 0
        assert summary.pipeline_outcome_field is None
        assert summary.seismic_transitioned is False
        assert [a.scoring_status for a in summary.station_attempts] == [
            StationScoringStatus.INSUFFICIENT_RETAINED_DATA
        ]
        assert worker.fsm.state == state_before
        assert worker.audit_logger.count == audit_before

    def test_stale_seismic_record_syncs_agent_context(self) -> None:
        """A live-path stale seismic record that prunes worker.seismic_events
        must also sync the anomaly agent's private context copy: a skipped
        append still changes the list via pruning, and a desynced agent cache
        would keep feeding a pruned event to the Rayleigh/quiet checks."""
        worker = PipelineWorkerState()
        # Seed the context via the direct path (no now_epoch: always appends).
        _ingest_seismic_record(
            {
                "event_id": "seed",
                "magnitude": 7.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": (
                    datetime.now(UTC) - timedelta(hours=7)
                ).isoformat(),
            },
            worker,
        )
        assert len(worker.seismic_events) == 1
        assert len(worker.agent._recent_seismic_events) == 1

        # Live-path stale record: prunes the seed, skips its own append, and
        # must leave BOTH copies empty.
        _ingest_seismic_record(
            {
                "event_id": "stale",
                "magnitude": 6.5,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": (
                    datetime.now(UTC) - timedelta(hours=13)
                ).isoformat(),
            },
            worker,
            now_epoch=datetime.now(UTC).timestamp(),
        )
        assert worker.seismic_events == []
        assert worker.agent._recent_seismic_events == []

    def test_stale_seismic_backlog_policy(self) -> None:
        """POLICY PIN (deliberate fail-safe): seismic records have NO age
        gate before the FSM trigger. A 13h-old large shallow backlog quake
        escalates to a reviewable ESCALATE (far-field waves can still be
        propagating 12-24h after origin, so forcing review beats silently
        dropping it); a 13h-old moderate quake enters MONITOR and the
        origin-based timeout returns it to IDLE. On the live path the old
        event is NOT added to the anomaly agent's Rayleigh context (its
        surface waves are long gone)."""
        now_epoch = datetime.now(UTC).timestamp()
        old_origin = (datetime.now(UTC) - timedelta(hours=13)).isoformat()

        # Old major shallow quake: reviewable ESCALATE that survives timeout.
        worker = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:old8": [
                    {
                        "event_id": "old8",
                        "magnitude": 8.0,
                        "latitude": 38.0,
                        "longitude": 142.0,
                        "depth_km": 29.0,
                        "source_timestamp": old_origin,
                    },
                ],
            },
            worker,
            now_epoch=now_epoch,
        )
        assert worker.fsm.state == SystemState.ESCALATE
        worker.fsm.check_monitor_timeout()
        assert worker.fsm.state == SystemState.ESCALATE  # timeout is MONITOR-only
        assert worker.seismic_events == []  # not in the Rayleigh context

        # Old moderate quake: MONITOR, then origin-based timeout to IDLE.
        worker2 = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:old7": [
                    {
                        "event_id": "old7",
                        "magnitude": 7.0,
                        "latitude": 38.0,
                        "longitude": 142.0,
                        "source_timestamp": old_origin,
                    },
                ],
            },
            worker2,
            now_epoch=now_epoch,
        )
        assert worker2.fsm.state == SystemState.MONITOR
        worker2.fsm.check_monitor_timeout()
        assert worker2.fsm.state == SystemState.IDLE

        # A fresh quake still enters the Rayleigh context on the live path.
        worker3 = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:fresh": [
                    {
                        "event_id": "fresh",
                        "magnitude": 7.0,
                        "latitude": 38.0,
                        "longitude": 142.0,
                        "source_timestamp": datetime.now(UTC).isoformat(),
                    },
                ],
            },
            worker3,
            now_epoch=now_epoch,
        )
        assert len(worker3.seismic_events) == 1

    def test_wall_clock_gate_skips_stale_current_batch_observations(self) -> None:
        """Replay/backlog records older than the live rolling window must not
        be appended, scored, or used for event-mode confirmation."""
        worker = PipelineWorkerState()
        now_epoch = datetime.now(UTC).timestamp()
        stale_start = now_epoch - DEFAULT_WINDOW_SEC - 120.0
        records = [
            {
                "source_timestamp": datetime.fromtimestamp(
                    stale_start + i,
                    UTC,
                ).isoformat(),
                "height_m": 5500.0 + i * 0.01,
                "event_mode": True,
                "payload_sha256": "a" * 64,
            }
            for i in range(12)
        ]

        _process_buffer({"dart:21418": records}, worker, now_epoch=now_epoch)

        assert worker.station_buffers.get_window("21418", "dart") is None
        assert worker.station_buffers.stations_in_event_mode() == []
        assert worker.fsm.event_context is None

    def test_seismic_only_transition_emits_abstain(self) -> None:
        """A seismic-only transition with no scored station window must emit a
        fail-closed ABSTAIN artifact (audit entry), not just state_transition
        entries. Covers the M>=7.5 + shallow seismic-only ESCALATE path."""
        worker = PipelineWorkerState()
        buffer: dict[str, list[dict]] = {
            "seismic:us-tohoku": [
                {
                    "event_id": "us-tohoku",
                    "magnitude": 9.1,
                    "latitude": 38.30,
                    "longitude": 142.37,
                    "depth_km": 29.0,
                    "source_timestamp": "2011-03-11T05:46:24+00:00",
                },
            ],
        }

        _process_buffer(buffer, worker)

        # Seismic-only escalation fired (M9.1, shallow, tsunamigenic zone).
        assert worker.fsm.state == SystemState.ESCALATE
        # A fail-closed ABSTAIN artifact was recorded for the transition.
        abstains = worker.audit_logger.get_entries(event_type="abstain_triggered")
        assert len(abstains) == 1
        assert abstains[0].producer == "pipeline_worker"
        assert abstains[0].data["trigger"] == "seismic_only"
        assert abstains[0].data["fsm_state"] == "ESCALATE"

        # The run-scoped seismic artifacts (state_transition, seismic_provenance,
        # seismic-only abstain) share ONE batch trace, so /api/lineage/{trace}
        # shows the whole transition, not just the state change.
        transitions = worker.audit_logger.get_entries(event_type="state_transition")
        assert transitions, "seismic transition should have a state_transition entry"
        batch_trace = transitions[0].trace_id
        assert batch_trace is not None
        assert abstains[0].trace_id == batch_trace
        for entry in worker.audit_logger.get_entries(event_type="seismic_provenance"):
            assert entry.trace_id == batch_trace

    def test_no_abstain_without_seismic_transition(self) -> None:
        """No ABSTAIN artifact when no seismic transition occurred this batch
        (only DART data with no sufficient window; FSM stays IDLE)."""
        worker = PipelineWorkerState()
        buffer: dict[str, list[dict]] = {
            "dart:21413": [
                {
                    "source_timestamp": "2010-02-27T07:00:00+00:00",
                    "height_m": 5827.5,
                    "event_mode": False,
                },
            ],
        }

        _process_buffer(buffer, worker)

        assert worker.fsm.state == SystemState.IDLE
        assert worker.audit_logger.get_entries(event_type="abstain_triggered") == []

    def test_seismic_record_writes_input_provenance(self) -> None:
        """A seismic record carrying a valid payload hash records an
        input_provenance audit entry tagged with the FSM event id, so the
        escalation packet can assemble real input_refs (no fabrication)."""
        worker = PipelineWorkerState()
        sha = "a" * 64
        buffer: dict[str, list[dict]] = {
            "seismic:us-tohoku": [
                {
                    "event_id": "us-tohoku",
                    "magnitude": 9.1,
                    "latitude": 38.30,
                    "longitude": 142.37,
                    "depth_km": 29.0,
                    "source_timestamp": "2011-03-11T05:46:24+00:00",
                    "payload_sha256": sha,
                },
            ],
        }

        _process_buffer(buffer, worker)

        prov = worker.audit_logger.get_entries(event_type="seismic_provenance")
        assert len(prov) == 1
        assert prov[0].data["sha256"] == sha
        assert prov[0].data["source"] == "seismic"
        assert prov[0].data["record_id"] == "us-tohoku"
        assert prov[0].event_id == worker.fsm.event_context.event_id
        # Run-scoped: shares the batch trace with the state_transition it
        # belongs to, so /api/lineage/{trace} shows the trigger's provenance.
        transitions = worker.audit_logger.get_entries(event_type="state_transition")
        assert transitions and transitions[0].trace_id is not None
        assert prov[0].trace_id == transitions[0].trace_id

    def test_seismic_record_skips_provenance_without_valid_hash(self) -> None:
        """No seismic_provenance entry when the record has no valid payload hash
        (a malformed/absent hash must never be recorded)."""
        worker = PipelineWorkerState()
        buffer: dict[str, list[dict]] = {
            "seismic:us-tohoku": [
                {
                    "event_id": "us-tohoku",
                    "magnitude": 9.1,
                    "latitude": 38.30,
                    "longitude": 142.37,
                    "depth_km": 29.0,
                    "source_timestamp": "2011-03-11T05:46:24+00:00",
                    "payload_sha256": "not-a-hash",
                },
            ],
        }

        _process_buffer(buffer, worker)

        assert worker.audit_logger.get_entries(event_type="seismic_provenance") == []

    def test_observation_provenance_recorded_during_event(self) -> None:
        """DART/CO-OPS observations during an active event record
        input_provenance (deduped per event), so an anomaly-driven escalation
        packet carries real input_refs."""
        worker = PipelineWorkerState()
        dart_sha = "c" * 64
        buffer: dict[str, list[dict]] = {
            "seismic:us-tohoku": [
                {
                    "event_id": "us-tohoku",
                    "magnitude": 9.1,
                    "latitude": 38.30,
                    "longitude": 142.37,
                    "depth_km": 29.0,
                    "source_timestamp": "2011-03-11T05:46:24+00:00",
                },
            ],
            "dart:21418": [
                {
                    "source_timestamp": "2011-03-11T06:00:00+00:00",
                    "height_m": 5000.0,
                    "event_mode": True,
                    "payload_sha256": dart_sha,
                },
            ],
        }

        _process_buffer(buffer, worker)

        dart_prov = [
            e
            for e in worker.audit_logger.get_entries(event_type="input_provenance")
            if e.data.get("source") == "dart"
        ]
        assert len(dart_prov) == 1
        assert dart_prov[0].data["sha256"] == dart_sha
        assert dart_prov[0].data["record_id"] == "21418"
        # Deliberately event-scoped, NOT trace-tagged: input_provenance
        # accumulates across batches and is consumed by event_id, so a single
        # batch trace would misattribute it.
        assert dart_prov[0].trace_id is None

        # Re-ingesting the same record in a later batch must not duplicate it.
        _process_buffer({"dart:21418": buffer["dart:21418"]}, worker)
        dart_prov2 = [
            e
            for e in worker.audit_logger.get_entries(event_type="input_provenance")
            if e.data.get("source") == "dart"
        ]
        assert len(dart_prov2) == 1

    def test_observation_provenance_uses_source_id_when_present(self) -> None:
        """record_id identifies the raw RECORD via the connector source_id
        (station + timestamp + measurement type), not just the station."""
        worker = PipelineWorkerState()
        buffer: dict[str, list[dict]] = {
            "seismic:us-tohoku": [
                {
                    "event_id": "us-tohoku",
                    "magnitude": 9.1,
                    "latitude": 38.30,
                    "longitude": 142.37,
                    "depth_km": 29.0,
                    "source_timestamp": "2011-03-11T05:46:24+00:00",
                },
            ],
            "dart:21418": [
                {
                    "source_id": "dart:21418:20110311060000:2",
                    "source_timestamp": "2011-03-11T06:00:00+00:00",
                    "height_m": 5000.0,
                    "event_mode": True,
                    "payload_sha256": "d" * 64,
                },
            ],
        }

        _process_buffer(buffer, worker)

        dart_prov = [
            e
            for e in worker.audit_logger.get_entries(event_type="input_provenance")
            if e.data.get("source") == "dart"
        ]
        assert len(dart_prov) == 1
        assert dart_prov[0].data["record_id"] == "dart:21418:20110311060000:2"
        assert dart_prov[0].data["station_id"] == "21418"

    def test_no_observation_provenance_without_event(self) -> None:
        """Observations with no active event (FSM IDLE) record no provenance."""
        worker = PipelineWorkerState()
        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2011-03-11T06:00:00+00:00",
                    "height_m": 5000.0,
                    "event_mode": False,
                    "payload_sha256": "d" * 64,
                },
            ],
        }

        _process_buffer(buffer, worker)

        assert worker.audit_logger.get_entries(event_type="input_provenance") == []

    def test_observation_provenance_capped_per_event(self) -> None:
        """Provenance recording is bounded per event so a long, high-cadence
        event cannot flood the audit trail."""
        from hazard_assessment.workers.pipeline_runner import (
            _MAX_PROVENANCE_PER_EVENT,
            _record_observation_provenance,
        )

        worker = PipelineWorkerState()
        worker.fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        for i in range(_MAX_PROVENANCE_PER_EVENT + 5):
            _record_observation_provenance(worker, "dart", "21418", f"{i:064x}")

        # get_entries default-limits to 200; pass a higher limit to count all.
        prov = worker.audit_logger.get_entries(
            event_type="input_provenance", limit=_MAX_PROVENANCE_PER_EVENT + 10
        )
        assert len(prov) == _MAX_PROVENANCE_PER_EVENT
        assert len(worker.recorded_provenance_hashes) == _MAX_PROVENANCE_PER_EVENT
        # Hitting the cap records a one-time truncation marker (deduped).
        assert worker.provenance_capped is True
        markers = worker.audit_logger.get_entries(event_type="provenance_capped")
        assert len(markers) == 1
        # Event-scoped like input_provenance: not tagged with any batch trace.
        assert markers[0].trace_id is None

    def test_event_mode_updates_dart_confirmation(self) -> None:
        """DART event mode should set dart_confirmation on FSM."""
        worker = PipelineWorkerState()

        # First, trigger FSM into MONITOR state
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": 5500.0,
                    "event_mode": True,
                },
            ],
        }

        _process_buffer(buffer, worker)

        # FSM event context should have DART confirmation
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True

    def test_qc_lineage_persisted_with_batch_trace(self) -> None:
        """The live batch path persists a qc_report processed_features row
        whose handoff_id, trace_id, and input hashes match the qc_complete
        audit entry, with the hashes taken from the accepted records."""

        class _LineageDb:
            is_connected = True

            def __init__(self) -> None:
                self.features: list[dict[str, Any]] = []

            def load_fsm_state(self) -> None:
                return None

            def upsert_fsm_state(self, **_kwargs: Any) -> None:
                pass

            def append_audit(self, _entry: Any) -> bool:
                return True

            def persist_dart_confirmation(
                self, _event_id: Any, _stations: list[str] | None = None
            ) -> bool:
                return True

            def insert_processed_feature(self, **kwargs: Any) -> bool:
                self.features.append(kwargs)
                return True

        db = _LineageDb()
        worker = PipelineWorkerState(db_client=db)
        now = datetime.now(UTC)
        hashes = [format(i, "x").rjust(64, "0") for i in range(3)]
        records = [
            {
                "source_timestamp": (now - timedelta(minutes=3 - i)).isoformat(),
                "height_m": 5500.0 + i * 0.01,
                "event_mode": False,
                "payload_sha256": hashes[i],
            }
            for i in range(3)
        ]

        _process_buffer(
            {"dart:21418": records}, worker, now_epoch=now.timestamp()
        )

        qc_rows = [f for f in db.features if f["feature_type"] == "qc_report"]
        assert len(qc_rows) == 1
        row = qc_rows[0]
        assert row["station_id"] == "21418"
        assert [ref["sha256"] for ref in row["source_refs"]] == hashes

        entries = worker.audit_logger.get_entries(event_type="qc_complete")
        assert len(entries) == 1
        entry = entries[0]
        assert entry.data["handoff_id"] == str(row["handoff_id"])
        assert entry.data["input_hashes"] == hashes
        assert str(entry.trace_id) == str(row["trace_id"])

    def test_persist_anomaly_features_links_audit_and_row(self) -> None:
        """Each scored assessment yields an anomaly_score feature row and an
        anomaly_scored audit entry sharing the envelope handoff_id, the batch
        trace, and the station's accepted payload hashes."""
        from uuid import uuid4

        from hazard_assessment.workers.pipeline_runner import (
            _persist_anomaly_features,
        )

        class _LineageDb:
            is_connected = True

            def __init__(self) -> None:
                self.features: list[dict[str, Any]] = []

            def load_fsm_state(self) -> None:
                return None

            def upsert_fsm_state(self, **_kwargs: Any) -> None:
                pass

            def append_audit(self, _entry: Any) -> bool:
                return True

            def insert_processed_feature(self, **kwargs: Any) -> bool:
                self.features.append(kwargs)
                return True

        db = _LineageDb()
        worker = PipelineWorkerState(db_client=db)
        trace = uuid4()
        handoff = str(uuid4())
        hashes = ["a" * 64, "b" * 64]
        # Hashes ride with the retained samples: lineage must cover the whole
        # scored window, not just the newest batch.
        worker.station_buffers.append_dart(
            "21418", 1000.0, 5500.0, payload_hash=hashes[0]
        )
        worker.station_buffers.append_dart(
            "21418", 1060.0, 5500.1, payload_hash=hashes[1]
        )
        assessment = {
            "handoff_id": handoff,
            "anomaly_score": 0.42,
            "station_ids": ["21418"],
        }

        _persist_anomaly_features(worker, [(("dart", "21418"), assessment)], trace)

        rows = [f for f in db.features if f["feature_type"] == "anomaly_score"]
        assert len(rows) == 1
        assert rows[0]["handoff_id"] == handoff
        assert str(rows[0]["trace_id"]) == str(trace)
        assert [ref["sha256"] for ref in rows[0]["source_refs"]] == hashes
        assert rows[0]["payload"]["anomaly_score"] == 0.42

        entries = worker.audit_logger.get_entries(event_type="anomaly_scored")
        assert len(entries) == 1
        assert entries[0].data["handoff_id"] == handoff
        assert entries[0].data["input_hashes"] == hashes
        assert str(entries[0].trace_id) == str(trace)

    def test_invalid_dart_station_id_does_not_set_confirmation(self) -> None:
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:Warning": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": 5500.0,
                    "event_mode": True,
                },
            ],
        }

        _process_buffer(buffer, worker)

        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        assert ctx.active_dart_stations == []
        assert ("dart", "Warning") not in worker.station_buffers
        assert worker.audit_logger.get_entries(event_type="qc_complete") == []

    def test_sentinel_height_does_not_set_confirmation(self) -> None:
        """A missing-data sentinel (9999.0) never enters the anomaly window,
        so an event_mode=True record carrying one must not latch
        dart_confirmation."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": 9999.0,
                    "event_mode": True,
                },
            ],
        }

        _process_buffer(buffer, worker)

        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        assert ctx.active_dart_stations == []

    def test_non_numeric_values_do_not_crash_or_latch(self) -> None:
        """A malformed Kafka record with a non-numeric value must be skipped,
        not raise from np.isfinite and kill the consumer loop."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": "bad",
                    "event_mode": True,
                },
                {
                    "source_timestamp": "2010-02-27T06:31:00+00:00",
                    "height_m": True,
                    "event_mode": True,
                },
            ],
            "coops:1612340": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "water_level_m": "bad",
                },
            ],
        }

        _process_buffer(buffer, worker)

        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        window = worker.station_buffers.get_window("21418", "dart")
        assert window is None or len(window) == 0

    def test_resolved_event_dart_state_does_not_leak_into_next_event(self) -> None:
        """A resolved event's station-buffer event-mode flag must not attach
        dart_confirmation to a later, unrelated event: the latch uses only
        event-mode records accepted while the current event is active."""
        worker = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:eventA": [
                    {
                        "event_id": "eventA",
                        "magnitude": 8.0,
                        "latitude": 38.0,
                        "longitude": 142.0,
                        "depth_km": 29.0,
                        "source_timestamp": "2010-02-27T06:00:00+00:00",
                    },
                ],
                "dart:21418": [
                    {
                        "source_timestamp": "2010-02-27T06:30:00+00:00",
                        "height_m": 5500.0,
                        "event_mode": True,
                        "payload_sha256": "a" * 64,
                    },
                ],
            },
            worker,
        )
        ctx_a = worker.fsm.event_context
        assert ctx_a is not None
        assert ctx_a.dart_confirmation is True

        worker.fsm.resolve_event()
        assert worker.fsm.event_context is None
        # The stale per-window flag is still set; it must be inert.
        assert worker.station_buffers.stations_in_event_mode() == ["21418"]

        _process_buffer(
            {
                "seismic:eventB": [
                    {
                        "event_id": "eventB",
                        "magnitude": 6.5,
                        "latitude": 40.0,
                        "longitude": 143.0,
                        "depth_km": 30.0,
                        "source_timestamp": "2010-02-28T06:00:00+00:00",
                    },
                ],
            },
            worker,
        )
        ctx_b = worker.fsm.event_context
        assert ctx_b is not None
        assert ctx_b.dart_confirmation is False
        assert ctx_b.active_dart_stations == []
        assert ctx_b.stations_in_event_mode == []

    def test_stale_monitor_does_not_drop_seismic_in_first_post_timeout_batch(self) -> None:
        """Restart/backlog race: a MONITOR that crossed its timeout during
        downtime must not swallow a new seismic trigger arriving in the first
        non-empty (seismic-only) batch, when no quiet tick ever ran."""
        worker = PipelineWorkerState()
        old_origin = (datetime.now(UTC) - timedelta(hours=13)).isoformat()
        _ingest_seismic_record(
            {
                "event_id": "old",
                "magnitude": 7.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": old_origin,
            },
            worker,
        )
        assert worker.fsm.state == SystemState.MONITOR

        new_origin = datetime.now(UTC).isoformat()
        _process_buffer(
            {
                "seismic:new": [
                    {
                        "event_id": "new",
                        "magnitude": 8.0,
                        "latitude": 40.0,
                        "longitude": 143.0,
                        "source_timestamp": new_origin,
                    },
                ],
            },
            worker,
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.seismic_magnitude == 8.0  # new event adopted, not dropped

    def test_known_limitation_mixed_stale_batch_drops_new_seismic(self) -> None:
        """KNOWN LIMITATION (single-event FSM): a batch carrying BOTH a new
        seismic record and observation records, processed against a stale
        timed-out MONITOR, still drops the new trigger - the timeout stays
        deferred until the batch's observations are scored (they may
        legitimately keep the event alive), and seismic is processed first.
        Pinned here so any behavior change is visible; the real fix is
        multi-event tracking (documented frontier)."""
        worker = PipelineWorkerState()
        old_origin = (datetime.now(UTC) - timedelta(hours=13)).isoformat()
        _ingest_seismic_record(
            {
                "event_id": "old",
                "magnitude": 7.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": old_origin,
            },
            worker,
        )
        assert worker.fsm.state == SystemState.MONITOR

        now = datetime.now(UTC)
        _process_buffer(
            {
                "seismic:new": [
                    {
                        "event_id": "new",
                        "magnitude": 8.0,
                        "latitude": 40.0,
                        "longitude": 143.0,
                        "source_timestamp": now.isoformat(),
                    },
                ],
                "dart:21418": [
                    {
                        "source_timestamp": now.isoformat(),
                        "height_m": 5500.0,
                        "event_mode": False,
                        "payload_sha256": "e" * 64,
                    },
                ],
            },
            worker,
        )
        ctx = worker.fsm.event_context
        # Documented behavior of _process_buffer for a MIXED batch: the new
        # M8.0 is dropped, because the in-buffer timeout check only runs when
        # every key is seismic, and this batch also carries a DART record. If
        # this assertion starts failing, the limitation has been lifted and
        # the _process_buffer comment needs updating with it.
        if ctx is not None:
            assert ctx.seismic_magnitude == 7.0

        # The live worker then runs a post-batch timeout and clears the old
        # event, but the dropped new trigger is not replayed.
        worker.fsm.check_monitor_timeout()
        assert worker.fsm.state == SystemState.IDLE

    def test_timeout_skipped_while_records_are_buffered(self) -> None:
        """Buffered evidence must be scored before the event can time out:
        the quiet-tick timeout check is a no-op while records are buffered,
        and fires once the buffer is empty."""
        from hazard_assessment.workers.pipeline_runner import (
            _check_monitor_timeout_when_quiet,
        )

        worker = PipelineWorkerState()
        origin = (datetime.now(UTC) - timedelta(hours=13)).isoformat()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 7.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": origin,
            },
            worker,
        )
        assert worker.fsm.state.value == "MONITOR"

        pending: dict[str, list[dict]] = {"dart:21418": [{"height_m": 5500.0}]}
        _check_monitor_timeout_when_quiet(worker, pending)
        assert worker.fsm.state.value == "MONITOR"  # evidence still pending

        _check_monitor_timeout_when_quiet(worker, {})
        assert worker.fsm.state.value == "IDLE"  # true silence times out

    def test_delayed_prior_event_row_does_not_latch_new_event(self) -> None:
        """A delayed DART event-mode row timestamped BEFORE the new event's
        seismic origin must not latch dart_confirmation onto the new event,
        even though it arrives in the same batch (the sample is still
        buffered for scoring)."""
        worker = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:eventA": [
                    {
                        "event_id": "eventA",
                        "magnitude": 8.0,
                        "latitude": 38.0,
                        "longitude": 142.0,
                        "depth_km": 29.0,
                        "source_timestamp": "2010-02-27T06:00:00+00:00",
                    },
                ],
                "dart:21418": [
                    {
                        "source_timestamp": "2010-02-27T06:30:00+00:00",
                        "height_m": 5500.0,
                        "event_mode": True,
                        "payload_sha256": "a" * 64,
                    },
                ],
            },
            worker,
        )
        worker.fsm.resolve_event()

        _process_buffer(
            {
                "seismic:eventB": [
                    {
                        "event_id": "eventB",
                        "magnitude": 6.5,
                        "latitude": 40.0,
                        "longitude": 143.0,
                        "depth_km": 30.0,
                        "source_timestamp": "2010-02-28T06:00:00+00:00",
                    },
                ],
                # Delayed row from event A's window, arriving after B's trigger
                "dart:21418": [
                    {
                        "source_timestamp": "2010-02-27T06:45:00+00:00",
                        "height_m": 5500.2,
                        "event_mode": True,
                        "payload_sha256": "b" * 64,
                    },
                ],
            },
            worker,
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        assert ctx.active_dart_stations == []
        window = worker.station_buffers.get_window("21418", "dart")
        assert window is not None
        assert len(window) == 2  # Both samples still buffered for scoring

    def test_event_mode_row_after_origin_latches(self) -> None:
        """An event-mode row timestamped at or after the current event's
        seismic origin latches normally."""
        worker = PipelineWorkerState()
        _process_buffer(
            {
                "seismic:eventB": [
                    {
                        "event_id": "eventB",
                        "magnitude": 8.0,
                        "latitude": 40.0,
                        "longitude": 143.0,
                        "depth_km": 30.0,
                        "source_timestamp": "2010-02-28T06:00:00+00:00",
                    },
                ],
                "dart:21418": [
                    {
                        "source_timestamp": "2010-02-28T06:20:00+00:00",
                        "height_m": 5500.3,
                        "event_mode": True,
                        "payload_sha256": "c" * 64,
                    },
                ],
            },
            worker,
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True
        assert ctx.active_dart_stations == ["21418"]
        # trigger_time_utc carries the seismic ORIGIN time, not wall clock.
        assert ctx.trigger_time_utc.isoformat() == "2010-02-28T06:00:00+00:00"

    def test_non_bool_event_mode_does_not_latch(self) -> None:
        """A malformed event_mode (string or int) is treated as False: the
        valid pressure sample is still ingested for scoring, but no
        confirmation latches from a flag that is not a real JSON boolean."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": 5500.0,
                    "event_mode": "false",
                },
                {
                    "source_timestamp": "2010-02-27T06:31:00+00:00",
                    "height_m": 5500.1,
                    "event_mode": 1,
                },
            ],
        }

        _process_buffer(buffer, worker)

        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        assert worker.station_buffers.stations_in_event_mode() == []
        window = worker.station_buffers.get_window("21418", "dart")
        assert window is not None
        assert len(window) == 2  # Samples ingested for scoring

    def test_event_mode_detected_despite_trailing_standard_record(self) -> None:
        """Batch-aware detection: a station's event-mode record followed by a
        standard-mode record in the SAME batch (out-of-order / trailing) must
        still set dart_confirmation, even though the window's latest event_mode
        ends False."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "test",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )

        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "height_m": 5500.0,
                    "event_mode": True,
                },
                {
                    "source_timestamp": "2010-02-27T06:31:00+00:00",
                    "height_m": 5501.0,
                    "event_mode": False,
                },
            ],
        }

        _process_buffer(buffer, worker)

        # The window's latest event_mode is False, but an event-mode record was
        # seen this batch, so the confirmation latch must still be set.
        assert worker.station_buffers.stations_in_event_mode() == []
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True

    def test_worker_reconciles_to_idle_after_api_resolution(self) -> None:
        """If the DB shows IDLE (the event was resolved via the API) while the
        worker holds a stale in-memory ESCALATE, the next batch resets the worker
        to IDLE so it accepts new seismic triggers instead of dropping them."""

        class _StubDb:
            is_connected = True

            def load_fsm_state(self) -> dict[str, Any]:
                return {
                    "current_state": "IDLE",
                    "sensor_degraded": False,
                    "event_context": None,
                }

            def upsert_fsm_state(self, **_kwargs: Any) -> None:
                pass

            def append_audit(self, _entry: Any) -> None:
                pass

            def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
                return []

        worker = PipelineWorkerState(db_client=_StubDb())
        _ingest_seismic_record(
            {
                "event_id": "a",
                "magnitude": 9.1,
                "latitude": 38.30,
                "longitude": 142.37,
                "depth_km": 29.0,
                "source_timestamp": "2011-03-11T05:46:24+00:00",
            },
            worker,
        )
        assert worker.fsm.state == SystemState.ESCALATE

        # Next batch: reconcile observes the DB resolution and resets the worker.
        _process_buffer({}, worker)
        assert worker.fsm.state == SystemState.IDLE

        # A new seismic event is now accepted (would have been dropped before).
        _process_buffer(
            {
                "seismic:b": [
                    {
                        "event_id": "b",
                        "magnitude": 9.0,
                        "latitude": -35.85,
                        "longitude": -72.72,
                        "depth_km": 20.0,
                        "source_timestamp": "2010-02-27T06:34:11+00:00",
                    }
                ]
            },
            worker,
        )
        assert worker.fsm.state != SystemState.IDLE

    def test_worker_recovers_accumulated_event_mode_stations(self) -> None:
        event_id = uuid4()

        class _StubDb:
            is_connected = True

            def load_fsm_state(self) -> dict[str, Any]:
                return {
                    "current_state": "ESCALATE",
                    "sensor_degraded": False,
                    "event_context": {
                        "event_id": str(event_id),
                        "seismic_magnitude": 8.0,
                        "seismic_region": "test-zone",
                        "epicenter_lat": 0.0,
                        "epicenter_lon": 0.0,
                        "trigger_time_utc": "2026-01-01T00:00:00+00:00",
                        "latest_anomaly_score": 0.9,
                        "dart_confirmation": True,
                        "active_dart_stations": ["21418", "46403"],
                        "stations_in_event_mode": ["21418", "46403"],
                    },
                }

        worker = PipelineWorkerState(db_client=_StubDb())

        assert worker.event_mode_event_id == event_id
        assert worker.event_mode_station_set == {"21418", "46403"}

    def test_reconcile_does_not_clobber_active_event(self) -> None:
        """Reconciliation adopts only a DB-side IDLE (an API resolution). While
        the durable row shows the worker's own non-IDLE event (the single-writer
        steady state: the worker persisted its own transitions), the in-memory
        FSM, its event context, and the dart_confirmation latch must be left
        untouched."""

        class _StubDb:
            """Stateful stub: serves back whatever the worker last persisted."""

            is_connected = True

            def __init__(self) -> None:
                self.row: dict[str, Any] = {
                    "current_state": "IDLE",
                    "sensor_degraded": False,
                    "event_context": None,
                }

            def load_fsm_state(self) -> dict[str, Any]:
                return dict(self.row)

            def upsert_fsm_state(
                self,
                state: str,
                event_context: dict[str, Any] | None,
                sensor_degraded: bool,
            ) -> None:
                self.row = {
                    "current_state": state,
                    "sensor_degraded": sensor_degraded,
                    "event_context": event_context,
                }

            def append_audit(self, _entry: Any) -> None:
                pass

            def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
                return []

            def persist_dart_confirmation(
                self, _event_id: Any, _stations: list[str] | None = None
            ) -> bool:
                return True

        worker = PipelineWorkerState(db_client=_StubDb())
        _ingest_seismic_record(
            {
                "event_id": "a",
                "magnitude": 9.1,
                "latitude": 38.30,
                "longitude": 142.37,
                "depth_km": 29.0,
                "source_timestamp": "2011-03-11T05:46:24+00:00",
            },
            worker,
        )
        assert worker.fsm.state == SystemState.ESCALATE
        worker.fsm.update_dart_confirmation(
            dart_confirmation=True, stations_in_event_mode=["21418"]
        )
        ctx_before = worker.fsm.event_context
        assert ctx_before is not None
        assert ctx_before.dart_confirmation is True

        # Reconcile runs at the top of the batch, sees the non-IDLE durable
        # row, and must change nothing.
        _process_buffer({}, worker)

        assert worker.fsm.state == SystemState.ESCALATE
        ctx_after = worker.fsm.event_context
        assert ctx_after is not None
        assert ctx_after.event_id == ctx_before.event_id
        assert ctx_after.dart_confirmation is True

    def test_invalid_event_mode_record_does_not_latch_confirmation(self) -> None:
        """A malformed DART record with event_mode=True but missing height_m is
        skipped at ingest and must NOT latch dart_confirmation (only validly
        ingested event-mode records count)."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            {
                "event_id": "a",
                "magnitude": 8.0,
                "latitude": 38.0,
                "longitude": 142.0,
                "source_timestamp": "2010-02-27T06:00:00+00:00",
            },
            worker,
        )
        assert worker.fsm.state == SystemState.MONITOR

        # event_mode=True but no height_m -> skipped at ingest.
        buffer: dict[str, list[dict]] = {
            "dart:21418": [
                {
                    "source_timestamp": "2010-02-27T06:30:00+00:00",
                    "event_mode": True,
                }
            ]
        }
        _process_buffer(buffer, worker)

        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False

    def test_full_pipeline_run_with_sufficient_data(self) -> None:
        """End-to-end: sufficient data should produce a pipeline result."""
        worker = PipelineWorkerState()

        # Build enough DART observations for scoring
        base_time = datetime(2010, 2, 27, 0, 0, 0, tzinfo=UTC)
        dart_records = []
        for i in range(30):
            ts = base_time + timedelta(minutes=i * 15)
            height = 5827.0 + 0.3 * np.sin(2 * np.pi * i / 48)
            dart_records.append({
                "source_timestamp": ts.isoformat(),
                "height_m": round(height, 3),
                "event_mode": False,
            })

        buffer: dict[str, list[dict]] = {
            "dart:21413": dart_records,
        }

        # This should complete without error
        _process_buffer(buffer, worker)

        # FSM should still be IDLE (no seismic trigger, low anomaly score)
        assert worker.fsm.state == SystemState.IDLE


def _kafka_value(source_type: str, record: Any) -> dict[str, Any]:
    """Reproduce the JSON value the ingest producer emits and the consumer
    decodes.

    ``messaging/producer.py`` wraps ``{"source_type", "record"}`` with a
    ``schema_version`` + ``timestamp`` and JSON-encodes it (datetimes become
    ISO strings); ``messaging/consumer.py`` JSON-decodes it back into
    ``rec["value"]``.  Round-tripping through json exercises the real
    on-the-wire shape, including the datetime -> string conversion.
    """
    envelope = {
        "schema_version": SCHEMA_VERSION,
        "timestamp": 0.0,
        "source_type": source_type,
        "record": dataclasses.asdict(record),
    }
    wire = json.dumps(
        envelope,
        default=lambda o: o.isoformat() if isinstance(o, datetime) else str(o),
    )
    decoded: dict[str, Any] = json.loads(wire)
    return decoded


class TestKafkaEnvelopeUnwrap:
    """The live Kafka path nests the canonical record under a ``"record"``
    key, so the worker must unwrap the envelope before reading fields. Reading
    top-level fields instead drops every live record silently: seismic for
    "missing fields", observations for lack of a top-level
    ``source_timestamp``. The enveloped cases below cover that path; the flat
    case alongside them pins the fallback, which a flat-only suite would have
    left as the sole shape under test.
    """

    def test_enveloped_seismic_record_is_ingested(self) -> None:
        worker = PipelineWorkerState()
        rec = SeismicEventRecord(
            source_id="usgs",
            event_id="us-tohoku",
            source_timestamp=datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC),
            ingest_timestamp=datetime(2011, 3, 11, 5, 47, tzinfo=UTC),
            magnitude=9.1,
            place="off the Pacific coast of Tohoku",
            event_type="earthquake",
            tsunami_flag=1,
            longitude=142.37,
            latitude=38.30,
            depth_km=29.0,
            updated_timestamp=None,
            is_revision=False,
        )
        value = _kafka_value("seismic", rec)
        # The producer really does nest the fields under "record".
        assert value["record"]["magnitude"] == 9.1
        assert "magnitude" not in value

        _process_buffer({"seismic:us-tohoku": [value]}, worker)

        assert len(worker.seismic_events) == 1
        assert worker.seismic_events[0].magnitude == 9.1
        # A shallow M9.1 in a tsunamigenic zone drives the FSM out of IDLE.
        assert worker.fsm.state != SystemState.IDLE

    def test_enveloped_dart_record_is_ingested(self) -> None:
        worker = PipelineWorkerState()
        rec = DartRecord(
            source_id="ndbc:21413",
            station_id="21413",
            source_timestamp=datetime(2011, 3, 11, 6, 0, tzinfo=UTC),
            ingest_timestamp=datetime(2011, 3, 11, 6, 1, tzinfo=UTC),
            measurement_type=2,
            height_m=5827.5,
            event_mode=True,
        )
        value = _kafka_value("dart", rec)
        assert value["record"]["height_m"] == 5827.5
        assert "height_m" not in value

        _ingest_observation_records("dart:21413", [value], worker)

        window = worker.station_buffers.get_window("21413", "dart")
        assert window is not None
        assert len(window) == 1  # ingested, not dropped
        assert window.event_mode is True

    def test_flat_records_remain_supported(self) -> None:
        """Backward compatibility: flat (un-enveloped) records still ingest,
        so existing tests and any future flat producer are unaffected.
        """
        worker = PipelineWorkerState()
        _ingest_observation_records(
            "dart:21413",
            [{
                "source_timestamp": "2011-03-11T06:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
            }],
            worker,
        )
        window = worker.station_buffers.get_window("21413", "dart")
        assert window is not None
        assert len(window) == 1


class TestWorkerFsmAudit:
    """The pipeline worker shares a DB-backed AuditLogger with its FSM so
    that worker-driven state transitions are recorded. Constructing
    ``FSMOrchestrator(db_client=...)`` without an ``audit_writer``, alongside
    a separate non-DB ``AuditLogger()``, persists the state but writes nothing
    to the audit trail.
    """

    def test_seismic_trigger_is_audited(self) -> None:
        worker = PipelineWorkerState()
        rec = SeismicEventRecord(
            source_id="usgs",
            event_id="us-audit",
            source_timestamp=datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC),
            ingest_timestamp=datetime(2011, 3, 11, 5, 47, tzinfo=UTC),
            magnitude=9.1,
            place="off the Pacific coast of Tohoku",
            event_type="earthquake",
            tsunami_flag=1,
            longitude=142.37,
            latitude=38.30,
            depth_km=29.0,
            updated_timestamp=None,
            is_revision=False,
        )
        _process_buffer({"seismic:us-audit": [_kafka_value("seismic", rec)]}, worker)

        transitions = worker.audit_logger.get_entries(event_type="state_transition")
        assert transitions, "worker FSM transition was not audited"
        assert transitions[0].data["from_state"] == "IDLE"
        assert transitions[-1].data["to_state"] in ("MONITOR", "ESCALATE")


class TestWorkerQc:
    """The pipeline worker runs QARTOD QC on each observation batch and audits
    a summary.  QC is metadata only: it must NOT drop records from the anomaly
    buffer (genuine tsunami signals deliberately trip the QC checks).
    """

    def test_qc_runs_and_does_not_filter(self) -> None:
        worker = PipelineWorkerState()
        base = datetime(2011, 3, 11, 6, 0, tzinfo=UTC)
        records = [
            _kafka_value(
                "dart",
                DartRecord(
                    source_id="ndbc:21413",
                    station_id="21413",
                    source_timestamp=base + timedelta(minutes=i),
                    ingest_timestamp=base + timedelta(minutes=i, seconds=30),
                    measurement_type=2,
                    height_m=5827.5 + 0.01 * i,
                    event_mode=True,
                    payload_sha256="a" * 64,
                ),
            )
            for i in range(5)
        ]

        _process_buffer({"dart:21413": records}, worker)

        qc_entries = worker.audit_logger.get_entries(event_type="qc_complete")
        assert qc_entries, "QC did not run / was not audited"
        assert qc_entries[0].data["n_records"] == 5
        # The first record of a stream has no history, so no check evaluates;
        # the batch summary must surface that zero-coverage count.
        assert qc_entries[0].data["n_zero_coverage"] >= 1

        # QC must not filter: all five observations remain in the buffer.
        window = worker.station_buffers.get_window("21413", "dart")
        assert window is not None
        assert len(window) == 5


def test_station_coordinates_registry() -> None:
    from hazard_assessment.data.station_coordinates import (
        DART_STATION_COORDS,
        station_coordinates,
    )

    assert station_coordinates("21413") == (30.515, 152.117)
    assert station_coordinates("99999") is None
    assert len(DART_STATION_COORDS) >= 30


class TestWorkerStationCoords:
    """The worker supplies station coordinates to process_station_data so the
    Rayleigh-wave false-trigger check can run on live data. Passing none
    leaves that check permanently disabled.
    """

    def test_known_station_coords_passed_to_agent(self) -> None:
        worker = PipelineWorkerState()
        base = datetime(2011, 3, 11, 6, 0, tzinfo=UTC)
        records = [
            _kafka_value(
                "dart",
                DartRecord(
                    source_id="ndbc:21413",
                    station_id="21413",
                    source_timestamp=base + timedelta(minutes=i),
                    ingest_timestamp=base + timedelta(minutes=i, seconds=30),
                    measurement_type=2,
                    height_m=5827.5 + 0.001 * i,
                    event_mode=True,
                    payload_sha256="a" * 64,
                ),
            )
            for i in range(12)
        ]

        captured: dict[str, Any] = {}

        def _spy(**kwargs: Any) -> None:
            captured.update(kwargs)
            raise RuntimeError("stop after capturing arguments")

        worker.agent.process_station_data = _spy  # type: ignore[method-assign]
        _process_buffer({"dart:21413": records}, worker)

        # Station 21413's coordinates (from the registry) were passed through.
        assert captured.get("origin_lat") == 30.515
        assert captured.get("origin_lon") == 152.117
        # processing_time is the latest observation time (data time), not wall-clock.
        assert captured.get("processing_time") == datetime(
            2011, 3, 11, 6, 11, tzinfo=UTC
        )


def _coord(
    partition: int,
    offset: int,
    *,
    topic: str = "raw.observations",
    rejected: bool = False,
) -> KafkaMessageCoordinate:
    return KafkaMessageCoordinate(
        topic=topic,
        partition=partition,
        offset=offset,
        timestamp_type="" if rejected else "CREATE_TIME",
        timestamp_ms=None if rejected else 1700000000000 + offset,
        application_message_id="" if rejected else f"msg-{partition}-{offset}",
        transport_rejected=rejected,
    )


class TestCheckpointTransport:
    """Checkpoint identity derivation from batch Kafka metadata: decoded
    and rejected coordinates both enter the consumed
    offset manifest, and identity is None without transport metadata."""

    def test_offset_ranges_group_by_topic_partition(self) -> None:
        transport = CheckpointTransport(consumer_group="pipeline-workers")
        transport.messages = [
            _coord(0, 5),
            _coord(0, 9),
            _coord(1, 2),
            _coord(0, 7, rejected=True),
        ]
        assert sorted(transport.offset_ranges()) == [
            ("raw.observations", 0, 5, 9),
            ("raw.observations", 1, 2, 2),
        ]
        assert transport.rejected_markers() == [("raw.observations", 0, 7)]

    def test_checkpoint_id_matches_direct_derivation(self) -> None:
        transport = CheckpointTransport(consumer_group="pipeline-workers")
        transport.messages = [_coord(0, 5), _coord(0, 6, rejected=True)]
        assert transport.checkpoint_id() == derive_live_checkpoint_id(
            "pipeline-workers",
            [("raw.observations", 0, 5, 6)],
            [("raw.observations", 0, 6)],
        )

    def test_rejected_marker_changes_identity(self) -> None:
        with_reject = CheckpointTransport(consumer_group="pipeline-workers")
        with_reject.messages = [_coord(0, 5), _coord(0, 6, rejected=True)]
        without_reject = CheckpointTransport(consumer_group="pipeline-workers")
        without_reject.messages = [_coord(0, 5), _coord(0, 6)]
        assert with_reject.checkpoint_id() != without_reject.checkpoint_id()

    def test_empty_transport_has_no_identity(self) -> None:
        transport = CheckpointTransport(consumer_group="pipeline-workers")
        assert transport.checkpoint_id() is None

    def test_process_buffer_records_checkpoint_identity(self) -> None:
        worker = PipelineWorkerState()
        transport = CheckpointTransport(consumer_group="pipeline-workers")
        transport.messages = [_coord(0, 5)]
        buffer = {
            "dart:21413": [{
                "source_timestamp": "2011-03-11T06:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
            }],
        }
        _process_buffer(buffer, worker, transport=transport)
        assert worker.last_checkpoint_id == transport.checkpoint_id()

    def test_process_buffer_without_transport_has_no_identity(self) -> None:
        worker = PipelineWorkerState()
        worker.last_checkpoint_id = "stale"
        buffer = {
            "dart:21413": [{
                "source_timestamp": "2011-03-11T06:00:00+00:00",
                "height_m": 5827.5,
                "event_mode": False,
            }],
        }
        _process_buffer(buffer, worker)
        assert worker.last_checkpoint_id is None


# ---------------------------------------------------------------------------
# Seismic revision identity wiring
# ---------------------------------------------------------------------------

_REV1_ID = "seismic:us2010chile:20100227064000000000"
_REV2_ID = "seismic:us2010chile:20100227065000000000"


def _seismic_record(**over: Any) -> dict[str, Any]:
    """Flat seismic record as the worker sees it after envelope unwrap.

    ``updated_timestamp`` uses the space-separated form produced by
    ``json.dumps(dataclasses.asdict(record), default=str)`` on the ingest
    side (str() of an aware datetime)."""
    rec: dict[str, Any] = {
        "source_id": _REV1_ID,
        "event_id": "us2010chile",
        "magnitude": 8.8,
        "latitude": -35.846,
        "longitude": -72.719,
        "depth_km": 22.9,
        "source_timestamp": "2010-02-27T06:34:11+00:00",
        "updated_timestamp": "2010-02-27 06:40:00+00:00",
        "payload_sha256": "a" * 64,
    }
    rec.update(over)
    return rec


_RECEIPT_EPOCH = datetime(2010, 2, 27, 7, 0, 0, tzinfo=UTC).timestamp()


class TestBuildSeismicIdentity:
    def test_full_record_builds_complete_identity(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(),
            now_epoch=_RECEIPT_EPOCH,
            kafka_positions={_REV1_ID: (0, 7)},
        )
        assert identity is not None
        assert identity.provider == "usgs"
        assert identity.external_event_id == "us2010chile"
        assert identity.revision_id == _REV1_ID
        assert identity.revision_sha256 == "a" * 64
        assert identity.provider_updated_utc == datetime(
            2010, 2, 27, 6, 40, 0, tzinfo=UTC
        )
        assert identity.kafka_partition == 0
        assert identity.kafka_offset == 7
        assert identity.context_class == "LIVE_RECEIPT_ORDERED"

    def test_missing_event_id_yields_no_identity(self) -> None:
        record = _seismic_record()
        del record["event_id"]
        assert _build_seismic_identity(record) is None
        assert _build_seismic_identity(_seismic_record(event_id="")) is None

    def test_malformed_hash_becomes_empty(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(payload_sha256="NOT-A-HASH")
        )
        assert identity is not None
        assert identity.revision_sha256 == ""

    def test_invalid_update_times_become_none(self) -> None:
        for bad in (None, "", "garbage", "2010-02-27 06:40:00", 1267252800):
            identity = _build_seismic_identity(
                _seismic_record(updated_timestamp=bad),
                now_epoch=_RECEIPT_EPOCH,
            )
            assert identity is not None
            assert identity.provider_updated_utc is None, bad

    def test_post_receipt_future_update_time_becomes_none(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(updated_timestamp="2010-02-27 07:00:01+00:00"),
            now_epoch=_RECEIPT_EPOCH,
        )
        assert identity is not None
        assert identity.provider_updated_utc is None

    def test_update_time_equal_to_receipt_is_kept(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(updated_timestamp="2010-02-27 07:00:00+00:00"),
            now_epoch=_RECEIPT_EPOCH,
        )
        assert identity is not None
        assert identity.provider_updated_utc == datetime(
            2010, 2, 27, 7, 0, 0, tzinfo=UTC
        )

    def test_non_utc_offset_normalized_to_utc(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(updated_timestamp="2010-02-27 15:40:00+09:00"),
            now_epoch=_RECEIPT_EPOCH,
        )
        assert identity is not None
        assert identity.provider_updated_utc == datetime(
            2010, 2, 27, 6, 40, 0, tzinfo=UTC
        )
        assert identity.provider_updated_utc.tzinfo == UTC

    def test_unresolved_kafka_position_is_none(self) -> None:
        identity = _build_seismic_identity(
            _seismic_record(), kafka_positions={"other-id": (0, 3)}
        )
        assert identity is not None
        assert identity.kafka_partition is None
        assert identity.kafka_offset is None


class TestSeismicIdentityWorkerWiring:
    def test_trigger_binds_identity_on_event_context(self) -> None:
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            _seismic_record(),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            kafka_positions={_REV1_ID: (0, 7)},
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.seismic_provider == "usgs"
        assert ctx.external_event_id == "us2010chile"
        assert ctx.trigger_revision_id == _REV1_ID
        assert ctx.trigger_revision_sha256 == "a" * 64
        assert ctx.latest_revision_id == _REV1_ID
        assert ctx.latest_revision_kafka_partition == 0
        assert ctx.latest_revision_kafka_offset == 7
        assert ctx.seismic_context_class == "LIVE_RECEIPT_ORDERED"

    def test_matching_revision_without_magnitude_still_advances_latest(
        self,
    ) -> None:
        """A schema-valid revision missing trigger fields (magnitude) is
        still an admissible identity update: the guard that skips trigger
        evaluation must not skip revision matching."""
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            _seismic_record(),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            kafka_positions={_REV1_ID: (0, 7)},
        )
        n_events = len(worker.seismic_events)

        _ingest_seismic_record(
            _seismic_record(
                source_id=_REV2_ID,
                magnitude=None,
                payload_sha256="b" * 64,
                updated_timestamp="2010-02-27 06:50:00+00:00",
            ),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            kafka_positions={_REV2_ID: (0, 9)},
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_id == _REV2_ID
        assert ctx.latest_revision_sha256 == "b" * 64
        assert ctx.latest_revision_kafka_offset == 9
        # Trigger identity untouched; no new seismic event was created.
        assert ctx.trigger_revision_id == _REV1_ID
        assert len(worker.seismic_events) == n_events

    def test_active_event_revision_does_not_warn_concurrent_drop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A full-field revision of the ACTIVE event skips trigger
        evaluation entirely: it is not a concurrent event, so it must not
        log the SEISMIC TRIGGER DROPPED warning (that warning stays
        reserved for genuinely unrelated events)."""
        import logging

        worker = PipelineWorkerState()
        _ingest_seismic_record(
            _seismic_record(), worker, now_epoch=_RECEIPT_EPOCH
        )
        state_before = worker.fsm.state

        with caplog.at_level(
            logging.WARNING,
            logger="hazard_assessment.orchestrator.states",
        ):
            _ingest_seismic_record(
                _seismic_record(
                    source_id=_REV2_ID,
                    magnitude=8.9,
                    payload_sha256="b" * 64,
                    updated_timestamp="2010-02-27 06:50:00+00:00",
                ),
                worker,
                now_epoch=_RECEIPT_EPOCH,
            )

        assert worker.fsm.state == state_before
        assert not any(
            "SEISMIC TRIGGER DROPPED" in r.getMessage() for r in caplog.records
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_id == _REV2_ID

    def test_unrelated_event_still_warns_concurrent_drop(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A genuinely different external event while an event is active is
        a true concurrent-event drop and keeps its warning."""
        import logging

        worker = PipelineWorkerState()
        _ingest_seismic_record(
            _seismic_record(), worker, now_epoch=_RECEIPT_EPOCH
        )
        with caplog.at_level(
            logging.WARNING,
            logger="hazard_assessment.orchestrator.states",
        ):
            _ingest_seismic_record(
                _seismic_record(
                    source_id="seismic:us9999zzzz:20100227065000000000",
                    event_id="us9999zzzz",
                    updated_timestamp="2010-02-27 06:50:00+00:00",
                ),
                worker,
                now_epoch=_RECEIPT_EPOCH,
            )
        assert any(
            "SEISMIC TRIGGER DROPPED" in r.getMessage() for r in caplog.records
        )

    def test_unrelated_event_revision_does_not_replace_identity(self) -> None:
        worker = PipelineWorkerState()
        _ingest_seismic_record(
            _seismic_record(), worker, now_epoch=_RECEIPT_EPOCH
        )
        _ingest_seismic_record(
            _seismic_record(
                source_id="seismic:us9999zzzz:20100227065000000000",
                event_id="us9999zzzz",
                updated_timestamp="2010-02-27 06:50:00+00:00",
            ),
            worker,
            now_epoch=_RECEIPT_EPOCH,
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.external_event_id == "us2010chile"
        assert ctx.latest_revision_id == _REV1_ID

    def test_process_buffer_resolves_kafka_position_from_transport(self) -> None:
        """The batch transport's application message IDs (the connectors'
        source_ids) resolve each seismic revision's receipt position."""
        worker = PipelineWorkerState()
        transport = CheckpointTransport(consumer_group="pipeline-workers")
        transport.messages = [
            KafkaMessageCoordinate(
                topic="raw.observations",
                partition=2,
                offset=41,
                timestamp_type="CREATE_TIME",
                timestamp_ms=1267252800000,
                application_message_id=_REV1_ID,
            ),
        ]
        buffer = {
            "seismic:us2010chile": [{
                "schema_version": SCHEMA_VERSION,
                "timestamp": _RECEIPT_EPOCH,
                "source_type": "seismic",
                "message_id": _REV1_ID,
                "record": _seismic_record(),
            }],
        }
        _process_buffer(
            buffer, worker, now_epoch=_RECEIPT_EPOCH, transport=transport
        )
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.trigger_revision_id == _REV1_ID
        assert ctx.latest_revision_kafka_partition == 2
        assert ctx.latest_revision_kafka_offset == 41


# ---------------------------------------------------------------------------
# Assessment persistence path
# ---------------------------------------------------------------------------

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class _AssessmentDb:
    """Stateful stub DatabaseClient covering every method the worker touches
    on the assessment path. Serves back whatever FSM row the worker last
    persisted (the single-writer steady state) unless a test arms
    ``pending_loads`` to inject a divergent read; each queued item overrides
    one ``load_fsm_state`` call (None means "serve the live row")."""

    is_connected = True

    def __init__(self) -> None:
        self.row: dict[str, Any] = {
            "current_state": "IDLE",
            "sensor_degraded": False,
            "event_context": None,
        }
        self.pending_loads: list[dict[str, Any] | None] = []
        self.lookup_result: dict[str, Any] | None = None
        self.persist_result: AssessmentPersistResult | None = None
        self.persist_calls: list[dict[str, Any]] = []
        self.attempts: list[dict[str, Any]] = []
        self.audit_entries: list[Any] = []
        self.packet_result: AssessmentPersistResult | None = None
        self.packet_calls: list[dict[str, Any]] = []

    def load_fsm_state(self) -> dict[str, Any]:
        if self.pending_loads:
            override = self.pending_loads.pop(0)
            if override is not None:
                return dict(override)
        return dict(self.row)

    def upsert_fsm_state(
        self,
        state: str,
        event_context: dict[str, Any] | None,
        sensor_degraded: bool,
    ) -> None:
        self.row = {
            "current_state": state,
            "sensor_degraded": sensor_degraded,
            "event_context": event_context,
        }

    def append_audit(self, entry: Any) -> bool:
        self.audit_entries.append(entry)
        return True

    def query_audit(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return []

    def persist_dart_confirmation(
        self, _event_id: Any, _stations: list[str] | None = None
    ) -> bool:
        return True

    def persist_seismic_revision(self, _event_id: Any, _revision: Any) -> bool:
        return True

    def insert_processed_feature(self, **_kwargs: Any) -> bool:
        return True

    def get_assessment_by_checkpoint(
        self, _checkpoint_id: str, _schema_version: int
    ) -> dict[str, Any] | None:
        return self.lookup_result

    def persist_assessment(self, **kwargs: Any) -> AssessmentPersistResult:
        self.persist_calls.append(kwargs)
        if self.persist_result is not None:
            return self.persist_result
        return AssessmentPersistResult(
            status="inserted",
            row={
                "id": len(self.persist_calls),
                "event_id": kwargs["event_id"],
                "handoff_id": kwargs["handoff_id"],
                "payload": kwargs["payload"],
            },
        )

    def append_assessment_checkpoint_attempt(self, **kwargs: Any) -> bool:
        self.attempts.append(kwargs)
        return True

    def persist_escalation_packet(self, **kwargs: Any) -> AssessmentPersistResult:
        self.packet_calls.append(kwargs)
        if self.packet_result is not None:
            return self.packet_result
        return AssessmentPersistResult(
            status="inserted", row={"id": len(self.packet_calls)}
        )


def _transport_at(offset: int, *, message_id: str = "") -> CheckpointTransport:
    """One-message transport whose coordinate reuses the _coord defaults;
    message_id overrides the application message ID when the batch must
    resolve a seismic record's own Kafka position."""
    transport = CheckpointTransport(consumer_group="pipeline-workers")
    coord = _coord(0, offset)
    if message_id:
        coord = coord.model_copy(update={"application_message_id": message_id})
    transport.messages = [coord]
    return transport


def _seismic_batch(record: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    return {
        f"seismic:{record.get('event_id', 'x')}": [{
            "schema_version": SCHEMA_VERSION,
            "timestamp": _RECEIPT_EPOCH,
            "source_type": "seismic",
            "message_id": record.get("source_id", ""),
            "record": record,
        }],
    }


def _dart_records(n: int = 30, step_min: int = 5) -> list[dict[str, Any]]:
    base = datetime(2010, 2, 27, 5, 0, 0, tzinfo=UTC)
    records = []
    for i in range(n):
        ts = base + timedelta(minutes=i * step_min)
        height = 5827.0 + 0.3 * np.sin(2 * np.pi * i / 48)
        records.append({
            "source_timestamp": ts.isoformat(),
            "height_m": round(height, 3),
            "event_mode": False,
        })
    return records


def _audit_types(db: _AssessmentDb) -> list[str]:
    return [entry.event_type for entry in db.audit_entries]


def _monitor_trigger_record() -> dict[str, Any]:
    """Fully bound seismic identity below the escalation magnitude, so the
    trigger lands in MONITOR rather than the seismic-override ESCALATE."""
    return _seismic_record(magnitude=7.0)


class TestAssessmentPersistencePath:
    """Assessment persistence plus the crash boundaries that are
    observable in the worker: redelivery short-circuit, build-failure gap,
    persist-failure gap, crash-window rebuild adoption, and durable-identity
    conflict. All use the stub client; real-database grant/immutability
    behavior lives in tests/integration/."""

    def test_seismic_only_batch_persists_abstain_assessment(self) -> None:
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        transport = _transport_at(11, message_id=_REV1_ID)

        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=transport,
        )

        assert worker.fsm.state == SystemState.MONITOR
        assert len(db.persist_calls) == 1
        call = db.persist_calls[0]
        assert call["checkpoint_id"] == transport.checkpoint_id()
        assert _HEX64.match(call["input_manifest_hash"])
        assert _HEX64.match(call["scientific_content_hash"])
        assert _HEX64.match(call["transport_provenance_hash"])

        payload = call["payload"]
        assert payload["checkpoint_id"] == transport.checkpoint_id()
        assert payload["stations"] == []
        assert payload["pipeline_outcome"] == "ABSTAIN"
        assert payload["fsm_state_before"] == "IDLE"
        assert payload["fsm_state_after"] == "MONITOR"
        assert payload["fsm_state_changed"] is True
        assert payload["input_manifest_hash"] == call["input_manifest_hash"]

        assert len(db.attempts) == 1
        attempt = db.attempts[0]
        assert attempt["attempt_kind"] == "original"
        assert attempt["outcome"] == "inserted"
        assert attempt["worker_run_id"] == worker.run_id
        assert "assessment_persisted" in _audit_types(db)
        assert "assessment_gap" not in _audit_types(db)

    def test_scored_batch_persists_station_attempt_entries(self) -> None:
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        db.persist_calls.clear()
        db.attempts.clear()
        db.audit_entries.clear()

        transport = _transport_at(50)
        _process_buffer(
            {"dart:21413": _dart_records()}, worker, transport=transport
        )

        assert len(db.persist_calls) == 1
        call = db.persist_calls[0]
        assert call["checkpoint_id"] == transport.checkpoint_id()
        payload = call["payload"]
        assert [
            (s["source"], s["station_id"]) for s in payload["stations"]
        ] == [("dart", "21413")]
        entry = payload["stations"][0]
        assert entry["n_records_attempted"] == 30
        assert entry["n_records_admitted"] == 30
        assert entry["scoring_status"] == "SCORING_SUCCEEDED"
        assert payload["pipeline_outcome"] == "MONITORING_CONTINUES"
        assert payload["fsm_state_before"] == "MONITOR"
        assert payload["fsm_state_after"] == "MONITOR"
        assert db.attempts[-1]["outcome"] == "inserted"

    def test_redelivered_checkpoint_short_circuits_forward_evaluation(
        self,
    ) -> None:
        """Crash window between assessment persist and offset commit:
        the redelivered batch must record the redelivery and return before
        any buffering, scoring, or FSM evaluation."""
        db = _AssessmentDb()
        db.lookup_result = {
            "id": 7,
            "handoff_id": str(uuid4()),
            "event_id": str(uuid4()),
        }
        worker = PipelineWorkerState(db_client=db)
        transport = _transport_at(60)

        _process_buffer(
            {"dart:21413": _dart_records()}, worker, transport=transport
        )

        assert worker.fsm.state == SystemState.IDLE
        assert list(worker.station_buffers.station_keys()) == []
        assert db.persist_calls == []
        assert len(db.attempts) == 1
        attempt = db.attempts[0]
        assert attempt["attempt_kind"] == "redelivery"
        assert attempt["outcome"] == "existing"
        assert attempt["checkpoint_id"] == transport.checkpoint_id()
        assert _HEX64.match(attempt["transport_provenance_hash"])
        assert "assessment_redelivery" in _audit_types(db)

    def test_unbound_identity_build_failure_records_gap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a checkpoint whose active event lacks a bound external
        seismic identity (legacy trigger without source_id or payload hash)
        continues deterministic science but records a build_failed attempt
        plus a gap metric and audit entry, and never reaches persist."""
        gap_calls: list[int] = []
        monkeypatch.setattr(
            "hazard_assessment.telemetry.metrics.record_assessment_gap",
            lambda: gap_calls.append(1),
        )
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        legacy = {
            "event_id": "us2010chile",
            "magnitude": 7.0,
            "latitude": -35.846,
            "longitude": -72.719,
            "depth_km": 22.9,
            "source_timestamp": "2010-02-27T06:34:11+00:00",
        }

        _process_buffer(
            _seismic_batch(legacy),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(70),
        )

        assert worker.fsm.state == SystemState.MONITOR
        assert db.persist_calls == []
        assert len(db.attempts) == 1
        attempt = db.attempts[0]
        assert attempt["attempt_kind"] == "original"
        assert attempt["outcome"] == "build_failed"
        assert "AssessmentConstructionError" in attempt["detail"]
        assert "assessment_gap" in _audit_types(db)
        assert gap_calls == [1]

    def test_persist_error_records_gap_and_persist_failed_attempt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gap_calls: list[int] = []
        monkeypatch.setattr(
            "hazard_assessment.telemetry.metrics.record_assessment_gap",
            lambda: gap_calls.append(1),
        )
        db = _AssessmentDb()
        db.persist_result = AssessmentPersistResult(status="error", detail="boom")
        worker = PipelineWorkerState(db_client=db)

        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(80, message_id=_REV1_ID),
        )

        assert len(db.persist_calls) == 1
        assert db.attempts[-1]["outcome"] == "persist_failed"
        assert "assessment_gap" in _audit_types(db)
        assert "assessment_persisted" not in _audit_types(db)
        assert gap_calls == [1]

    def test_persist_existing_row_discloses_rebuild_without_gap(self) -> None:
        """Insert-then-crash boundary: the initial lookup missed the
        row but the idempotent insert finds it; the original row stands and
        the duplicate pass is disclosed as a redelivery, not a gap."""
        db = _AssessmentDb()
        existing_handoff = str(uuid4())
        db.persist_result = AssessmentPersistResult(
            status="existing", row={"handoff_id": existing_handoff}
        )
        worker = PipelineWorkerState(db_client=db)

        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(90, message_id=_REV1_ID),
        )

        assert db.attempts[-1]["outcome"] == "existing"
        assert "assessment_redelivery" in _audit_types(db)
        assert "assessment_gap" not in _audit_types(db)
        redelivery = next(
            e for e in db.audit_entries
            if e.event_type == "assessment_redelivery"
        )
        assert redelivery.data["handoff_id"] == existing_handoff

    def test_durable_identity_divergence_records_conflict(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """a durable event closure between the opening
        reconcile and assessment construction must not let the worker persist
        under stale identity."""
        gap_calls: list[int] = []
        monkeypatch.setattr(
            "hazard_assessment.telemetry.metrics.record_assessment_gap",
            lambda: gap_calls.append(1),
        )
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        assert len(db.persist_calls) == 1
        db.attempts.clear()
        db.audit_entries.clear()

        # Next batch: the opening reconcile still sees the live row, but the
        # step-14 re-read observes an API-side resolution (durable IDLE).
        db.pending_loads = [
            None,
            {
                "current_state": "IDLE",
                "sensor_degraded": False,
                "event_context": None,
            },
        ]
        _process_buffer(
            {"dart:21413": _dart_records(n=1)},
            worker,
            transport=_transport_at(120),
        )

        assert len(db.persist_calls) == 1  # unchanged: persist never ran
        assert len(db.attempts) == 1
        attempt = db.attempts[0]
        assert attempt["attempt_kind"] == "original"
        assert attempt["outcome"] == "conflict"
        assert "diverged" in attempt["detail"]
        assert "assessment_gap" in _audit_types(db)
        assert gap_calls == [1]

    def test_no_active_event_persists_no_assessment(self) -> None:
        """ocean-only IDLE processing buffers and scores but
        produces no assessment identity, no attempt record, and no gap."""
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)

        _process_buffer(
            {"dart:21413": _dart_records()}, worker, transport=_transport_at(130)
        )

        assert worker.fsm.state == SystemState.IDLE
        assert db.persist_calls == []
        assert db.attempts == []
        assert "assessment_gap" not in _audit_types(db)

    def test_db_less_worker_with_transport_does_not_crash(self) -> None:
        worker = PipelineWorkerState()
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        transport = _transport_at(140)
        _process_buffer(
            {"dart:21413": _dart_records()}, worker, transport=transport
        )
        assert worker.last_checkpoint_id == transport.checkpoint_id()

    def test_all_rejected_batch_station_appears_as_no_retained_data(
        self,
    ) -> None:
        """Attempt scope: a batch station whose records were
        all rejected before buffering still appears in the assessment as a
        NO_RETAINED_DATA attempt instead of silently vanishing."""
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        db.persist_calls.clear()

        # height_m missing: parseable timestamps, but every record is
        # rejected before buffering.
        rejected = [
            {"source_timestamp": "2010-02-27T06:50:00+00:00", "event_mode": True},
            {"source_timestamp": "2010-02-27T06:55:00+00:00", "event_mode": True},
        ]
        _process_buffer(
            {"dart:21419": rejected, "dart:21413": _dart_records()},
            worker,
            transport=_transport_at(150),
        )

        assert len(db.persist_calls) == 1
        payload = db.persist_calls[0]["payload"]
        by_station = {s["station_id"]: s for s in payload["stations"]}
        assert set(by_station) == {"21413", "21419"}
        rejected_entry = by_station["21419"]
        assert rejected_entry["n_records_attempted"] == 2
        assert rejected_entry["n_records_admitted"] == 0
        assert rejected_entry["scoring_status"] == "NO_RETAINED_DATA"
        assert by_station["21413"]["scoring_status"] == "SCORING_SUCCEEDED"
        # The rejected records were event-mode, but only ACCEPTED records
        # may latch dart_confirmation.
        ctx = worker.fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False


class TestQcAttachedPersistence:
    def test_hashed_records_persist_without_gap(self) -> None:
        """Every live Kafka record carries payload_sha256, so QC results
        join onto retained samples (pipeline_runner qc_by_hash) and the
        builder ranks their full flag tuples, including defaulted
        NOT_APPLICABLE extension checks. The hashless _dart_records
        fixture never exercised that join, which hid a severity-map gap
        that turned every QC-attached checkpoint into an assessment
        gap."""
        db = _AssessmentDb()
        worker = PipelineWorkerState(db_client=db)
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        db.persist_calls.clear()
        db.audit_entries.clear()

        records = _dart_records()
        for i, record in enumerate(records):
            record["payload_sha256"] = f"{i + 1:064x}"
        _process_buffer(
            {"dart:21413": records}, worker, transport=_transport_at(50)
        )

        assert "assessment_gap" not in _audit_types(db)
        assert len(db.persist_calls) == 1
        entry = db.persist_calls[0]["payload"]["stations"][0]
        assert entry["scoring_status"] == "SCORING_SUCCEEDED"
        window_qc = entry["qc_retained_window"]
        assert window_qc is not None
        assert window_qc["execution_status"] == "SUCCEEDED"
        assert window_qc["n_records"] == 30


class TestProcessBufferReturnValue:
    """CheckpointSummary return contract consumed by an offline
    observation-time replay driver. Live callers ignore the return, so
    these tests pin the replay-facing surface: populated summary on
    forward evaluation, None on a redelivered checkpoint."""

    def test_seismic_only_batch_without_transport_returns_summary(
        self,
    ) -> None:
        worker = PipelineWorkerState()
        summary = _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
        )
        assert isinstance(summary, CheckpointSummary)
        assert summary.fsm_state_before == "IDLE"
        assert summary.seismic_transitioned is True
        assert summary.station_attempts == ()
        assert summary.n_scored_assessments == 0
        assert summary.pipeline_outcome_field is None
        assert summary.spatial_analysis_ran is False
        assert summary.companion_failures == ()

    def test_scored_batch_summary_reflects_attempts(self) -> None:
        worker = PipelineWorkerState()
        _process_buffer(
            _seismic_batch(_monitor_trigger_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
        )
        summary = _process_buffer({"dart:21413": _dart_records()}, worker)
        assert isinstance(summary, CheckpointSummary)
        assert summary.fsm_state_before == "MONITOR"
        assert summary.seismic_transitioned is False
        assert [
            (a.source, a.station_id) for a in summary.station_attempts
        ] == [("dart", "21413")]
        assert summary.n_scored_assessments == 1
        assert summary.pipeline_outcome_field == "insufficient_evidence"

    def test_redelivered_checkpoint_returns_none(self) -> None:
        db = _AssessmentDb()
        db.lookup_result = {
            "id": 7,
            "handoff_id": str(uuid4()),
            "event_id": str(uuid4()),
        }
        worker = PipelineWorkerState(db_client=db)
        result = _process_buffer(
            {"dart:21413": _dart_records()},
            worker,
            transport=_transport_at(60),
        )
        assert result is None


class TestReviewerPacketPersistence:
    """the entering-ESCALATE checkpoint persists the durable
    reviewer packet, rendered purely from the persisted assessment row."""

    def _escalate(self, db: _AssessmentDb) -> PipelineWorkerState:
        """Run one seismic-override batch (M8.8) entering ESCALATE."""
        worker = PipelineWorkerState(db_client=db)
        _process_buffer(
            _seismic_batch(_seismic_record()),
            worker,
            now_epoch=_RECEIPT_EPOCH,
            transport=_transport_at(11, message_id=_REV1_ID),
        )
        assert worker.fsm.state == SystemState.ESCALATE
        return worker

    def test_entering_escalate_persists_reviewer_packet(self) -> None:
        db = _AssessmentDb()
        self._escalate(db)

        assert len(db.persist_calls) == 1
        assert len(db.packet_calls) == 1
        call = db.packet_calls[0]
        # Bound to the exact persisted assessment row, not a lookup.
        assert call["assessment_row_id"] == 1
        assert call["renderer_version"] == RENDERER_VERSION
        assert _HEX64.match(call["content_sha256"])
        assert call["content_sha256"] == canonical_packet_hash(call["packet"])

        packet = call["packet"]
        payload = db.persist_calls[0]["payload"]
        assert packet["fsm_state_before"] == "IDLE"
        assert packet["fsm_state_after"] == "ESCALATE"
        assert packet["assessment"] == payload
        assert packet["checkpoint_id"] == payload["checkpoint_id"]
        assert str(call["event_id"]) == payload["event_id"]

        persisted = next(
            e for e in db.audit_entries
            if e.event_type == "escalation_packet_persisted"
        )
        assert persisted.data["content_sha256"] == call["content_sha256"]
        assert persisted.data["assessment_row_id"] == 1

    def test_early_redelivery_repairs_missing_reviewer_packet(self) -> None:
        source_db = _AssessmentDb()
        self._escalate(source_db)
        persisted_payload = source_db.persist_calls[0]["payload"]

        repair_db = _AssessmentDb()
        repair_db.lookup_result = {
            "id": 7,
            "handoff_id": str(uuid4()),
            "event_id": persisted_payload["event_id"],
            "payload": persisted_payload,
        }
        worker = PipelineWorkerState(db_client=repair_db)

        result = _process_buffer(
            {"dart:21413": _dart_records()},
            worker,
            transport=_transport_at(60),
        )

        assert result is None
        assert repair_db.persist_calls == []
        assert len(repair_db.packet_calls) == 1
        call = repair_db.packet_calls[0]
        assert call["assessment_row_id"] == 7
        assert call["packet"]["assessment"] == persisted_payload
        assert "escalation_packet_persisted" in _audit_types(repair_db)
        assert "assessment_redelivery" in _audit_types(repair_db)

    def test_continuing_escalate_checkpoint_persists_no_packet(self) -> None:
        db = _AssessmentDb()
        worker = self._escalate(db)
        db.packet_calls.clear()
        db.audit_entries.clear()

        _process_buffer(
            {"dart:21413": _dart_records()},
            worker,
            transport=_transport_at(50),
        )

        assert worker.fsm.state == SystemState.ESCALATE
        assert len(db.persist_calls) == 2  # assessment still persisted
        assert db.packet_calls == []
        assert "escalation_packet_persisted" not in _audit_types(db)

    def test_existing_assessment_row_uses_committed_payload_for_packet(self) -> None:
        """An adopted row, not the duplicate rebuild, is packet authority."""
        source_db = _AssessmentDb()
        self._escalate(source_db)
        persisted_payload = dict(source_db.persist_calls[0]["payload"])
        persisted_payload["code_version"] = "persisted-row-sentinel"

        db = _AssessmentDb()
        db.persist_result = AssessmentPersistResult(
            status="existing",
            row={
                "id": 7,
                "handoff_id": str(uuid4()),
                "event_id": persisted_payload["event_id"],
                "payload": persisted_payload,
            },
        )
        self._escalate(db)

        assert len(db.packet_calls) == 1
        call = db.packet_calls[0]
        assert call["assessment_row_id"] == 7
        assert call["packet"]["assessment"] == persisted_payload
        assert (
            call["packet"]["assessment"]
            != db.persist_calls[0]["payload"]
        )

    def test_packet_persist_error_never_raises_and_is_not_audited_as_success(
        self,
    ) -> None:
        db = _AssessmentDb()
        db.packet_result = AssessmentPersistResult(status="error", detail="boom")
        self._escalate(db)  # must not raise

        assert len(db.packet_calls) == 1
        assert "escalation_packet_persisted" not in _audit_types(db)
        # The assessment itself persisted normally.
        assert "assessment_persisted" in _audit_types(db)

    def test_packet_conflict_is_audited(self) -> None:
        db = _AssessmentDb()
        db.packet_result = AssessmentPersistResult(
            status="conflict", detail="mismatched fields: content_sha256"
        )
        self._escalate(db)

        conflict = next(
            e for e in db.audit_entries
            if e.event_type == "escalation_packet_conflict"
        )
        assert "content_sha256" in conflict.data["detail"]
        assert "escalation_packet_persisted" not in _audit_types(db)


class TestUsableDartCoverage:
    """Coverage counting behind the sensor_degraded flag.

    The flag was published by /api/fsm and documented in the user manual while
    nothing in the live worker ever evaluated it, so it always read false. It
    now follows the retained windows.
    """

    def _qc(self, usable: bool) -> RetainedSampleQC:
        return RetainedSampleQC(usable=usable, flags=(), confidence=1.0, n_checks_evaluated=3)

    def test_counts_only_dart_stations_with_qc_usable_samples(self) -> None:
        worker = PipelineWorkerState()
        now = datetime.now(UTC).timestamp()
        worker.station_buffers.append_dart(
            "21418", now, 1.0, event_mode=False, qc=self._qc(True)
        )
        worker.station_buffers.append_dart(
            "21419", now, 1.0, event_mode=False, qc=self._qc(True)
        )
        # QC said this record is not usable.
        worker.station_buffers.append_dart(
            "21413", now, 1.0, event_mode=False, qc=self._qc(False)
        )
        # QC never produced a verdict: unevaluated, which is not usable. Same
        # rule the assessment builder applies.
        worker.station_buffers.append_dart("21401", now, 1.0, event_mode=False, qc=None)
        # Coastal gauges do not contribute to the triangulation minimum.
        worker.station_buffers.append_coops("9410170", now, 1.0, qc=self._qc(True))

        assert _count_usable_dart_stations(worker) == 2

    def test_one_usable_station_marks_coverage_degraded(self) -> None:
        worker = PipelineWorkerState()
        now = datetime.now(UTC).timestamp()
        worker.station_buffers.append_dart(
            "21418", now, 1.0, event_mode=False, qc=self._qc(True)
        )

        worker.fsm.evaluate_coverage(_count_usable_dart_stations(worker))
        assert worker.fsm.sensor_degraded is True

        worker.station_buffers.append_dart(
            "21419", now, 1.0, event_mode=False, qc=self._qc(True)
        )
        worker.fsm.evaluate_coverage(_count_usable_dart_stations(worker))
        assert worker.fsm.sensor_degraded is False

    def test_silent_network_counts_as_no_coverage(self) -> None:
        """Windows aged out leave nothing to count, which is the worst case."""
        worker = PipelineWorkerState()
        stale = datetime.now(UTC).timestamp() - (7 * 3600)
        worker.station_buffers.append_dart(
            "21418", stale, 1.0, event_mode=False, qc=self._qc(True)
        )
        worker.station_buffers.trim_all(now_epoch=datetime.now(UTC).timestamp())

        assert _count_usable_dart_stations(worker) == 0


class TestFutureDatedRecords:
    """The admission window is bounded on both sides.

    Observation and seismic timestamps describe events that already happened.
    A record from the future is a wrong clock or a corrupted row: a .dart line
    whose year field reads "99" parses as 2099, and two-digit years are
    accepted by design.
    """

    def _qc(self) -> RetainedSampleQC:
        return RetainedSampleQC(usable=True, flags=(), confidence=1.0, n_checks_evaluated=3)

    def _observation(self, ts: datetime, payload_hash: str) -> dict[str, Any]:
        return {
            "source_timestamp": ts.isoformat(),
            "height_m": 5604.27,
            "measurement_type": 1,
            "event_mode": False,
            "payload_hash": payload_hash,
        }

    def test_future_dated_observation_is_not_admitted(self) -> None:
        """A future-dated observation must not become the window's
        latest_epoch, which would let it survive every trim.

        That epoch is the clock the assessment and the Rayleigh timing read,
        so a single row dated 2099 would pin a station for 73 years.
        """
        worker = PipelineWorkerState()
        now = datetime.now(UTC)
        counts: dict[tuple[str, str], list[int]] = {}
        qc = self._qc()

        _ingest_observation_records(
            "dart:21418",
            [
                self._observation(now, "a" * 64),
                self._observation(now.replace(year=now.year + 73), "b" * 64),
            ],
            worker,
            now_epoch=now.timestamp(),
            qc_by_hash={"a" * 64: qc, "b" * 64: qc},
            admission_counts=counts,
        )

        window = worker.station_buffers.get_window("21418", "dart")
        assert window is not None
        assert counts[("dart", "21418")] == [2, 1], "the future row was admitted"
        assert window.latest_epoch is not None
        assert datetime.fromtimestamp(window.latest_epoch, tz=UTC).year == now.year

    def test_records_inside_the_tolerance_are_still_admitted(self) -> None:
        """Real clock skew must not cost us data."""
        worker = PipelineWorkerState()
        now = datetime.now(UTC)
        qc = self._qc()

        _ingest_observation_records(
            "dart:21418",
            [self._observation(now + timedelta(seconds=FUTURE_TOLERANCE_SEC - 60), "c" * 64)],
            worker,
            now_epoch=now.timestamp(),
            qc_by_hash={"c" * 64: qc},
        )

        window = worker.station_buffers.get_window("21418", "dart")
        assert window is not None and len(window) == 1

    def test_replay_is_unaffected(self) -> None:
        """Replay passes the cutoff as now_epoch and its records precede it."""
        worker = PipelineWorkerState()
        cutoff = datetime(2011, 3, 11, 6, 0, tzinfo=UTC)
        qc = self._qc()

        _ingest_observation_records(
            "dart:21418",
            [self._observation(cutoff - timedelta(minutes=5), "d" * 64)],
            worker,
            now_epoch=cutoff.timestamp(),
            qc_by_hash={"d" * 64: qc},
        )

        window = worker.station_buffers.get_window("21418", "dart")
        assert window is not None and len(window) == 1

    def test_future_dated_seismic_record_is_rejected(self) -> None:
        """It evaded pruning forever and suppressed the DART evidence latch.

        The prune filter keeps events younger than six hours and a future
        origin has a negative age, so the event stayed in the Rayleigh context
        until the worker restarted. The event-mode latch only accepts records
        at or after the origin, so a future trigger suppressed every real
        event-mode row that followed.
        """
        worker = PipelineWorkerState()
        now = datetime.now(UTC)

        _ingest_seismic_record(
            {
                "event_id": "future-1",
                "magnitude": 9.9,
                "latitude": 38.3,
                "longitude": 142.4,
                "depth_km": 29.0,
                "source_timestamp": now.replace(year=now.year + 73).isoformat(),
            },
            worker,
            now_epoch=now.timestamp(),
        )

        assert worker.seismic_events == []
        assert worker.fsm.state == SystemState.IDLE

    def test_a_current_seismic_record_still_triggers(self) -> None:
        """The guard must not cost a real trigger."""
        worker = PipelineWorkerState()
        now = datetime.now(UTC)

        _ingest_seismic_record(
            {
                "event_id": "real-1",
                "magnitude": 9.1,
                "latitude": 38.3,
                "longitude": 142.4,
                "depth_km": 29.0,
                "source_timestamp": (now - timedelta(minutes=2)).isoformat(),
            },
            worker,
            now_epoch=now.timestamp(),
        )

        assert len(worker.seismic_events) == 1
        assert worker.fsm.state != SystemState.IDLE
