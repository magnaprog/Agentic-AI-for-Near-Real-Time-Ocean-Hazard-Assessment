"""Unit tests for the deterministic FSM orchestrator.

Validates that state transitions are deterministic, threshold-based,
and that invalid transitions are rejected.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from hazard_assessment.orchestrator.states import (
    EventContext,
    FSMOrchestrator,
    InvalidTransitionError,
    SeismicIdentity,
    SystemState,
    ThresholdConfig,
)

TSUNAMIGENIC_ZONES = {"pacific_rim", "cascadia", "alaska_aleutian"}


class TestFSMInitialState:
    def test_starts_idle(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.state == SystemState.IDLE

    def test_no_event_context_initially(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.event_context is None


class TestSeismicTrigger:
    def test_triggers_monitor(self) -> None:
        fsm = FSMOrchestrator()
        record = fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is not None
        assert fsm.state == SystemState.MONITOR
        assert record.from_state == SystemState.IDLE
        assert record.to_state == SystemState.MONITOR

    def test_below_magnitude_no_trigger(self) -> None:
        fsm = FSMOrchestrator()
        record = fsm.evaluate_seismic_trigger(
            magnitude=5.5,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is None
        assert fsm.state == SystemState.IDLE

    def test_non_tsunamigenic_zone_no_trigger(self) -> None:
        fsm = FSMOrchestrator()
        record = fsm.evaluate_seismic_trigger(
            magnitude=7.5,
            region="continental_interior",
            epicenter_lat=40.0,
            epicenter_lon=-100.0,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is None
        assert fsm.state == SystemState.IDLE

    def test_ignores_when_not_idle(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        # Already in MONITOR, second trigger should be ignored
        record = fsm.evaluate_seismic_trigger(
            magnitude=8.0,
            region="cascadia",
            epicenter_lat=45.0,
            epicenter_lon=-125.0,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is None
        assert fsm.state == SystemState.MONITOR

    def test_dropped_seismic_trigger_emits_warning_log(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A seismic trigger dropped while FSM is not IDLE must log at WARNING."""
        import logging

        fsm = FSMOrchestrator()
        # Move FSM to MONITOR
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert fsm.state == SystemState.MONITOR

        # Second event should be dropped with a WARNING log
        with caplog.at_level(logging.WARNING, logger="hazard_assessment.orchestrator.states"):
            result = fsm.evaluate_seismic_trigger(
                magnitude=9.0,
                region="cascadia",
                epicenter_lat=45.0,
                epicenter_lon=-125.0,
                tsunamigenic_zones=TSUNAMIGENIC_ZONES,
            )

        assert result is None
        assert fsm.state == SystemState.MONITOR  # state unchanged
        warning_records = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warning_records) >= 1
        msg = warning_records[-1].getMessage()
        assert "SEISMIC TRIGGER DROPPED" in msg
        assert "9.0" in msg
        assert "cascadia" in msg
        assert "MONITOR" in msg


class TestAnomalyScoreTransitions:
    def _make_monitoring_fsm(self) -> FSMOrchestrator:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        return fsm

    def test_monitor_to_investigate(self) -> None:
        fsm = self._make_monitoring_fsm()
        record = fsm.evaluate_anomaly_score(0.40)
        assert record is not None
        assert fsm.state == SystemState.INVESTIGATE
        assert record.from_state == SystemState.MONITOR
        assert record.to_state == SystemState.INVESTIGATE

    def test_monitor_stays_below_t1(self) -> None:
        fsm = self._make_monitoring_fsm()
        record = fsm.evaluate_anomaly_score(0.20)
        assert record is None
        assert fsm.state == SystemState.MONITOR

    def test_investigate_to_assess(self) -> None:
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        record = fsm.evaluate_anomaly_score(0.65)
        assert record is not None
        assert fsm.state == SystemState.ASSESS

    def test_investigate_back_to_monitor(self) -> None:
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        record = fsm.evaluate_anomaly_score(0.20)  # Below T1
        assert record is not None
        assert fsm.state == SystemState.MONITOR

    def test_assess_to_escalate(self) -> None:
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        record = fsm.evaluate_anomaly_score(0.90)
        assert record is not None
        assert fsm.state == SystemState.ESCALATE

    def test_assess_back_to_investigate(self) -> None:
        """ASSESS -> INVESTIGATE when score drops below T2."""
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        assert fsm.state == SystemState.ASSESS

        record = fsm.evaluate_anomaly_score(0.50)  # Below T2 (0.60)
        assert record is not None
        assert fsm.state == SystemState.INVESTIGATE
        assert record.from_state == SystemState.ASSESS
        assert record.to_state == SystemState.INVESTIGATE

    def test_assess_stays_between_t2_and_t3(self) -> None:
        """ASSESS stays if score is between T2 and T3."""
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        assert fsm.state == SystemState.ASSESS

        record = fsm.evaluate_anomaly_score(0.70)  # Between T2 and T3
        assert record is None
        assert fsm.state == SystemState.ASSESS

    def test_assess_deescalation_full_cycle(self) -> None:
        """ASSESS -> INVESTIGATE -> MONITOR de-escalation path."""
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        fsm.evaluate_anomaly_score(0.50)  # -> INVESTIGATE (below T2)
        assert fsm.state == SystemState.INVESTIGATE
        fsm.evaluate_anomaly_score(0.20)  # -> MONITOR (below T1)
        assert fsm.state == SystemState.MONITOR

    def test_full_lifecycle(self) -> None:
        fsm = FSMOrchestrator()

        # IDLE -> MONITOR
        fsm.evaluate_seismic_trigger(
            magnitude=8.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert fsm.state == SystemState.MONITOR

        # MONITOR -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.40)
        assert fsm.state == SystemState.INVESTIGATE

        # INVESTIGATE -> ASSESS
        fsm.evaluate_anomaly_score(0.65)
        assert fsm.state == SystemState.ASSESS

        # ASSESS -> ESCALATE
        fsm.evaluate_anomaly_score(0.90)
        assert fsm.state == SystemState.ESCALATE

        # ESCALATE -> IDLE (human resolution)
        record = fsm.resolve_event()
        assert record is not None
        assert fsm.state == SystemState.IDLE
        assert fsm.event_context is None


class TestBoundaryValues:
    """Verify FSM behavior at exact threshold boundaries (>= edge cases)."""

    def _make_monitoring_fsm(self) -> FSMOrchestrator:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        return fsm

    def test_score_exactly_at_t1_triggers_investigate(self) -> None:
        """Score == T1 (0.35) uses >= so should trigger MONITOR -> INVESTIGATE."""
        fsm = self._make_monitoring_fsm()
        record = fsm.evaluate_anomaly_score(0.35)
        assert record is not None
        assert fsm.state == SystemState.INVESTIGATE

    def test_score_just_below_t1_stays_in_monitor(self) -> None:
        fsm = self._make_monitoring_fsm()
        record = fsm.evaluate_anomaly_score(0.349)
        assert record is None
        assert fsm.state == SystemState.MONITOR

    def test_score_exactly_at_t2_triggers_assess(self) -> None:
        """Score == T2 (0.60) uses >= so should trigger INVESTIGATE -> ASSESS."""
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        record = fsm.evaluate_anomaly_score(0.60)
        assert record is not None
        assert fsm.state == SystemState.ASSESS

    def test_score_exactly_at_t3_triggers_escalate(self) -> None:
        """Score == T3 (0.85) uses >= so should trigger ASSESS -> ESCALATE."""
        fsm = self._make_monitoring_fsm()
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS
        record = fsm.evaluate_anomaly_score(0.85)
        assert record is not None
        assert fsm.state == SystemState.ESCALATE

    def test_seismic_exactly_at_min_magnitude_triggers(self) -> None:
        """Magnitude == 6.0 uses >= so should trigger IDLE -> MONITOR."""
        fsm = FSMOrchestrator()
        record = fsm.evaluate_seismic_trigger(
            magnitude=6.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is not None
        assert fsm.state == SystemState.MONITOR

    def test_seismic_just_below_min_magnitude_no_trigger(self) -> None:
        fsm = FSMOrchestrator()
        record = fsm.evaluate_seismic_trigger(
            magnitude=5.99,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is None
        assert fsm.state == SystemState.IDLE


class TestDARTConfirmationEscalation:
    def test_magnitude_plus_dart_escalates(self) -> None:
        thresholds = ThresholdConfig(
            basin="pacific",
            t1=0.35,
            t2=0.60,
            t3=0.85,
        )
        fsm = FSMOrchestrator(thresholds=thresholds)

        fsm.evaluate_seismic_trigger(
            magnitude=7.8,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.65)  # -> ASSESS

        # Set DART confirmation via the sanctioned FSM method
        assert fsm.event_context is not None
        fsm.update_dart_confirmation(True)

        # Score below T3 but M >= 7.5 + DART confirmation -> ESCALATE
        record = fsm.evaluate_anomaly_score(0.70)
        assert record is not None
        assert fsm.state == SystemState.ESCALATE

    def test_monitor_override_climbs_to_escalate_on_sub_t1_scores(self) -> None:
        """The M>=7.5 + DART event-mode override must be reachable from
        MONITOR: an M7.8 with unknown depth (seismic-only path cannot fire)
        whose DART stations switch to event mode must not sit in MONITOR on
        persistently sub-T1 scores. Each evaluation advances one state
        (MONITOR -> INVESTIGATE -> ASSESS -> ESCALATE), all audited."""
        fsm = FSMOrchestrator()

        fsm.evaluate_seismic_trigger(
            magnitude=7.8,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert fsm.state == SystemState.MONITOR

        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])

        record = fsm.evaluate_anomaly_score(0.20)
        assert record is not None
        assert fsm.state == SystemState.INVESTIGATE
        assert "DART event-mode" in record.trigger_reason

        fsm.evaluate_anomaly_score(0.20)
        assert fsm.state == SystemState.ASSESS

        fsm.evaluate_anomaly_score(0.20)
        assert fsm.state == SystemState.ESCALATE

    def test_investigate_override_fires_for_score_between_t1_and_t2(self) -> None:
        """A score in [T1, T2) must not strand an M>=7.5 + DART event-mode event.

        The override has to be evaluated for any sub-T2 score, mirroring the
        ASSESS branch. Nested inside the ``score < t1`` branch it would leave
        [T1, T2) matching no branch at all, holding the event in INVESTIGATE.
        INVESTIGATE has no timeout, so that stall would persist and block the
        single-event FSM.
        """
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.8,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])

        # Cross T1 normally so the FSM is in INVESTIGATE.
        fsm.evaluate_anomaly_score(0.40)
        assert fsm.state == SystemState.INVESTIGATE

        # 0.50 sits in [T1=0.35, T2=0.60), the interval the override must cover.
        record = fsm.evaluate_anomaly_score(0.50)
        assert record is not None
        assert fsm.state == SystemState.ASSESS
        assert "DART event-mode" in record.trigger_reason

    def test_investigate_holds_between_t1_and_t2_without_override(self) -> None:
        """Without the override, a score in [T1, T2) still holds in INVESTIGATE."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.8,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.evaluate_anomaly_score(0.40)
        assert fsm.state == SystemState.INVESTIGATE

        record = fsm.evaluate_anomaly_score(0.50)
        assert record is None
        assert fsm.state == SystemState.INVESTIGATE

    def test_monitor_no_override_below_escalation_magnitude(self) -> None:
        """M < 7.5 with DART event-mode and a sub-T1 score stays in MONITOR."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])
        record = fsm.evaluate_anomaly_score(0.20)
        assert record is None
        assert fsm.state == SystemState.MONITOR

    def test_monitor_no_override_without_dart_confirmation(self) -> None:
        """M >= 7.5 without DART event-mode and a sub-T1 score stays in
        MONITOR (the override requires both conditions)."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.8,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        record = fsm.evaluate_anomaly_score(0.20)
        assert record is None
        assert fsm.state == SystemState.MONITOR


class TestResolveEvent:
    def test_resolve_only_from_escalate(self) -> None:
        fsm = FSMOrchestrator()
        record = fsm.resolve_event()
        assert record is None  # Can't resolve from IDLE

    def test_resolve_returns_to_idle(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=8.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.evaluate_anomaly_score(0.40)
        fsm.evaluate_anomaly_score(0.65)
        fsm.evaluate_anomaly_score(0.90)
        assert fsm.state == SystemState.ESCALATE

        record = fsm.resolve_event()
        assert record is not None
        assert fsm.state == SystemState.IDLE


class TestDeterminism:
    """Verify that identical inputs produce identical state sequences."""

    def test_replay_produces_same_transitions(self) -> None:
        scores = [0.10, 0.20, 0.40, 0.55, 0.65, 0.80, 0.90]

        # Run 1
        fsm1 = FSMOrchestrator()
        fsm1.evaluate_seismic_trigger(
            magnitude=7.5,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        states1 = [fsm1.state]
        for score in scores:
            fsm1.evaluate_anomaly_score(score)
            states1.append(fsm1.state)

        # Run 2 (identical inputs)
        fsm2 = FSMOrchestrator()
        fsm2.evaluate_seismic_trigger(
            magnitude=7.5,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        states2 = [fsm2.state]
        for score in scores:
            fsm2.evaluate_anomaly_score(score)
            states2.append(fsm2.state)

        assert states1 == states2


class TestInvalidTransitions:
    """Verify that invalid state transitions raise InvalidTransitionError."""

    def test_idle_to_assess_rejected(self) -> None:
        """IDLE -> ASSESS is not a valid transition."""
        fsm = FSMOrchestrator()
        # Manually force an invalid transition attempt via internal method
        with pytest.raises(InvalidTransitionError):
            fsm._transition(
                to_state=SystemState.ASSESS,
                reason="test",
                anomaly_score=None,
                seismic_magnitude=None,
            )

    def test_idle_to_escalate_rejected(self) -> None:
        fsm = FSMOrchestrator()
        with pytest.raises(InvalidTransitionError):
            fsm._transition(
                to_state=SystemState.ESCALATE,
                reason="test",
                anomaly_score=None,
                seismic_magnitude=None,
            )

    def test_monitor_to_assess_rejected(self) -> None:
        """MONITOR -> ASSESS skips INVESTIGATE - not allowed."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        with pytest.raises(InvalidTransitionError):
            fsm._transition(
                to_state=SystemState.ASSESS,
                reason="test",
                anomaly_score=None,
                seismic_magnitude=None,
            )


class TestMonitorTimeout:
    """Verify that MONITOR state times out and returns to IDLE."""

    def _make_monitoring_fsm(
        self, timeout_hours: float = 4.0
    ) -> FSMOrchestrator:
        thresholds = ThresholdConfig(
            basin="pacific",
            t1=0.35,
            t2=0.60,
            t3=0.85,
            monitor_timeout_hours=timeout_hours,
        )
        fsm = FSMOrchestrator(thresholds=thresholds)
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        return fsm

    def test_timeout_returns_to_idle(self) -> None:
        fsm = self._make_monitoring_fsm(timeout_hours=4.0)
        assert fsm.state == SystemState.MONITOR

        # Simulate 5 hours passing
        future = datetime.now(UTC) + timedelta(hours=5)
        record = fsm.check_monitor_timeout(now=future)

        assert record is not None
        assert fsm.state == SystemState.IDLE
        assert fsm.event_context is None
        assert "timeout" in record.trigger_reason.lower()

    def test_no_timeout_before_duration(self) -> None:
        fsm = self._make_monitoring_fsm(timeout_hours=4.0)

        # Only 2 hours - should not timeout
        future = datetime.now(UTC) + timedelta(hours=2)
        record = fsm.check_monitor_timeout(now=future)

        assert record is None
        assert fsm.state == SystemState.MONITOR

    def test_timeout_occurs_when_score_below_t1(self) -> None:
        """MONITOR times out when score is below T1, even if close to it.

        Score 0.39 < T1 0.40: the timeout guard checks score >= T1 and
        finds it insufficient, so the timeout proceeds normally.
        """
        thresholds = ThresholdConfig(
            basin="pacific", t1=0.40, t2=0.60, t3=0.85,
            monitor_timeout_hours=4.0,
        )
        fsm = FSMOrchestrator(thresholds=thresholds)
        fsm.evaluate_seismic_trigger(
            magnitude=7.0, region="pacific_rim",
            epicenter_lat=38.3, epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        assert fsm.state == SystemState.MONITOR

        fsm.evaluate_anomaly_score(0.39)
        assert fsm.state == SystemState.MONITOR

        future = datetime.now(UTC) + timedelta(hours=5)
        record = fsm.check_monitor_timeout(now=future)

        assert record is not None
        assert fsm.state == SystemState.IDLE

    def test_no_timeout_when_score_at_t1(self) -> None:
        """If score exactly equals T1 in MONITOR, evaluate_anomaly_score
        transitions to INVESTIGATE (score >= T1). This proves the FSM
        cannot be in MONITOR with score >= T1, so the timeout guard
        (score >= T1 -> no timeout) can only protect against a race
        where the score was updated externally between checks.
        """
        fsm = self._make_monitoring_fsm(timeout_hours=4.0)
        record = fsm.evaluate_anomaly_score(0.35)  # exactly T1
        # Should transition to INVESTIGATE, proving MONITOR+elevated is impossible
        assert record is not None
        assert fsm.state == SystemState.INVESTIGATE

    def test_timeout_not_applicable_outside_monitor(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.state == SystemState.IDLE

        future = datetime.now(UTC) + timedelta(hours=100)
        record = fsm.check_monitor_timeout(now=future)

        assert record is None  # Only applies in MONITOR state

    def test_new_trigger_after_timeout(self) -> None:
        """After timeout to IDLE, a new seismic trigger should be accepted."""
        fsm = self._make_monitoring_fsm(timeout_hours=4.0)

        # Timeout
        future = datetime.now(UTC) + timedelta(hours=5)
        fsm.check_monitor_timeout(now=future)
        assert fsm.state == SystemState.IDLE

        # New trigger should work
        record = fsm.evaluate_seismic_trigger(
            magnitude=8.0,
            region="pacific_rim",
            epicenter_lat=45.0,
            epicenter_lon=-125.0,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert record is not None
        assert fsm.state == SystemState.MONITOR

    def test_naive_now_rejected(self) -> None:
        """check_monitor_timeout must reject naive datetimes for the now parameter."""
        fsm = self._make_monitoring_fsm(timeout_hours=4.0)
        naive_future = datetime(2030, 1, 1, 0, 0, 0)  # No tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            fsm.check_monitor_timeout(now=naive_future)


class TestTransitionRecordTimezoneValidation:
    """Verify TransitionRecord rejects naive datetimes."""

    def test_naive_timestamp_rejected(self) -> None:
        from hazard_assessment.orchestrator.states import TransitionRecord

        naive = datetime(2026, 2, 27, 1, 30, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            TransitionRecord(timestamp_utc=naive)

    def test_aware_timestamp_accepted(self) -> None:
        from hazard_assessment.orchestrator.states import TransitionRecord

        aware = datetime(2026, 2, 27, 1, 30, 0, tzinfo=UTC)
        record = TransitionRecord(timestamp_utc=aware)
        assert record.timestamp_utc == aware


class TestEventContextTimezoneValidation:
    """Verify EventContext rejects naive datetimes."""

    def test_naive_trigger_time_rejected(self) -> None:
        from hazard_assessment.orchestrator.states import EventContext

        naive = datetime(2026, 2, 27, 1, 30, 0)
        with pytest.raises(ValueError, match="timezone-aware"):
            EventContext(trigger_time_utc=naive)

    def test_aware_trigger_time_accepted(self) -> None:
        from hazard_assessment.orchestrator.states import EventContext

        aware = datetime(2026, 2, 27, 1, 30, 0, tzinfo=UTC)
        ctx = EventContext(trigger_time_utc=aware)
        assert ctx.trigger_time_utc == aware


class TestThresholdSettingsBridge:
    """Verify ThresholdSettings.to_threshold_config() produces a valid ThresholdConfig."""

    def test_to_threshold_config_preserves_values(self) -> None:
        from hazard_assessment.config.settings import ThresholdSettings

        settings = ThresholdSettings(
            t1=0.30, t2=0.55, t3=0.80,
            seismic_min_magnitude=5.5,
            escalation_magnitude=7.0,
            monitor_timeout_hours=3.0,
            basin="atlantic",
        )
        config = settings.to_threshold_config()
        assert isinstance(config, ThresholdConfig)
        assert config.basin == "atlantic"
        assert config.t1 == 0.30
        assert config.t2 == 0.55
        assert config.t3 == 0.80
        assert config.seismic_min_magnitude == 5.5
        assert config.escalation_magnitude == 7.0
        assert config.monitor_timeout_hours == 3.0

    def test_to_threshold_config_default_values(self) -> None:
        from hazard_assessment.config.settings import ThresholdSettings

        settings = ThresholdSettings()
        config = settings.to_threshold_config()
        assert config.basin == "pacific"
        assert config.t1 == 0.35
        assert config.t2 == 0.60
        assert config.t3 == 0.85


class TestThresholdConfigValidation:
    """Verify that ThresholdConfig rejects invalid threshold ordering."""

    def test_valid_thresholds_accepted(self) -> None:
        config = ThresholdConfig(basin="test", t1=0.3, t2=0.6, t3=0.9)
        assert config.t1 == 0.3

    def test_t1_greater_than_t2_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < t1 < t2 < t3 <= 1"):
            ThresholdConfig(basin="test", t1=0.7, t2=0.5, t3=0.9)

    def test_t2_greater_than_t3_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < t1 < t2 < t3 <= 1"):
            ThresholdConfig(basin="test", t1=0.3, t2=0.9, t3=0.7)

    def test_equal_thresholds_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < t1 < t2 < t3 <= 1"):
            ThresholdConfig(basin="test", t1=0.5, t2=0.5, t3=0.8)

    def test_t3_above_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < t1 < t2 < t3 <= 1"):
            ThresholdConfig(basin="test", t1=0.3, t2=0.6, t3=1.5)

    def test_negative_threshold_rejected(self) -> None:
        with pytest.raises(ValueError, match="0 < t1 < t2 < t3 <= 1"):
            ThresholdConfig(basin="test", t1=-0.1, t2=0.5, t3=0.9)


class TestTransitionHistoryCap:
    """Verify that transition history is bounded to MAX_HISTORY."""

    def test_history_capped_at_max(self) -> None:
        fsm = FSMOrchestrator()
        # Force many transitions by cycling through
        # IDLE->MONITOR->INVESTIGATE->MONITOR->IDLE repeats
        for _ in range(FSMOrchestrator.MAX_HISTORY + 50):
            if fsm.state == SystemState.IDLE:
                fsm.evaluate_seismic_trigger(
                    magnitude=7.0,
                    region="pacific_rim",
                    epicenter_lat=38.3,
                    epicenter_lon=142.4,
                    tsunamigenic_zones={"pacific_rim"},
                )
            elif fsm.state == SystemState.MONITOR:
                fsm.evaluate_anomaly_score(0.40)  # -> INVESTIGATE
            elif fsm.state == SystemState.INVESTIGATE:
                fsm.evaluate_anomaly_score(0.20)  # -> MONITOR (below T1)
                # Then force timeout to get back to IDLE
                from datetime import timedelta
                future = datetime.now(UTC) + timedelta(hours=5)
                fsm.check_monitor_timeout(now=future)
        assert len(fsm.transition_history) <= FSMOrchestrator.MAX_HISTORY


class TestEventContextDefensiveCopy:
    """Verify event_context property returns an isolated copy."""

    def test_mutation_does_not_affect_internal_state(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )

        ctx = fsm.event_context
        assert ctx is not None
        ctx.seismic_magnitude = 999.0
        ctx.active_dart_stations.append("INJECTED")

        internal = fsm.event_context
        assert internal is not None
        assert internal.seismic_magnitude == 7.0
        assert "INJECTED" not in internal.active_dart_stations

    def test_returns_none_when_no_event(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.event_context is None


class TestUpdateDartConfirmation:
    """Verify the sanctioned DART confirmation update method."""

    def test_sets_confirmation_on_active_event(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21413"])

        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True
        assert ctx.stations_in_event_mode == ["21413"]

    def test_seismic_trigger_threads_origin_time(self) -> None:
        """An explicit origin_time_utc becomes the context's trigger_time_utc
        (used to scope DART event-mode evidence to the event)."""
        origin = datetime(2010, 2, 27, 6, 0, tzinfo=UTC)
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
            origin_time_utc=origin,
        )
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.trigger_time_utc == origin

    def test_rejects_invalid_dart_station_ids(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["Warning"])

        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is False
        assert ctx.stations_in_event_mode == []
        assert ctx.active_dart_stations == []

    def test_context_recovery_preserves_confirmation_but_sanitizes_station_ids(self) -> None:
        ctx = EventContext(
            dart_confirmation=True,
            active_dart_stations=["Warning"],
            stations_in_event_mode=["Warning"],
        )

        assert ctx.dart_confirmation is True
        assert ctx.stations_in_event_mode == []
        assert ctx.active_dart_stations == []

    def test_noop_when_no_active_event(self) -> None:
        fsm = FSMOrchestrator()
        fsm.update_dart_confirmation(True)  # should not raise
        assert fsm.event_context is None

    def test_updates_in_escalate_state(self) -> None:
        """DART confirmation updates are applied in ESCALATE so the reviewer sees latest info."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        fsm.evaluate_anomaly_score(0.36)  # -> INVESTIGATE
        fsm.evaluate_anomaly_score(0.61)  # -> ASSESS
        fsm.evaluate_anomaly_score(0.86)  # -> ESCALATE

        assert fsm.state == SystemState.ESCALATE
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21413"])

        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True
        assert ctx.stations_in_event_mode == ["21413"]
        assert ctx.active_dart_stations == ["21413"]

    def test_works_in_investigate_state(self) -> None:
        """DART confirmation updates work in INVESTIGATE state."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones={"pacific_rim"},
        )
        fsm.evaluate_anomaly_score(0.36)  # -> INVESTIGATE

        assert fsm.state == SystemState.INVESTIGATE
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21413"])

        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.dart_confirmation is True
        assert ctx.stations_in_event_mode == ["21413"]


# ---------------------------------------------------------------------------
# Edge case tests from code review
# ---------------------------------------------------------------------------


class TestAnomalyScoreEdgeCases:
    """Edge cases for evaluate_anomaly_score (code review section 4 issue #1).

    NaN, inf, negative, and >1.0 scores should not crash the FSM.
    NaN is fail-safe (no transition). inf triggers max escalation.
    """

    def _make_fsm_in_monitor(self) -> FSMOrchestrator:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_nw",
            epicenter_lat=46.0,
            epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
        )
        return fsm

    def test_nan_score_raises_value_error(self) -> None:
        """NaN anomaly score is rejected by range validation."""
        fsm = self._make_fsm_in_monitor()
        with pytest.raises(ValueError, match="anomaly score must be in"):
            fsm.evaluate_anomaly_score(float("nan"))
        assert fsm.state == SystemState.MONITOR

    def test_inf_score_raises_value_error(self) -> None:
        """Inf anomaly score is rejected by range validation."""
        fsm = self._make_fsm_in_monitor()
        with pytest.raises(ValueError, match="anomaly score must be in"):
            fsm.evaluate_anomaly_score(float("inf"))
        assert fsm.state == SystemState.MONITOR

    def test_negative_score_raises_value_error(self) -> None:
        """Negative anomaly score is rejected by range validation."""
        fsm = self._make_fsm_in_monitor()
        with pytest.raises(ValueError, match="anomaly score must be in"):
            fsm.evaluate_anomaly_score(-1.0)
        assert fsm.state == SystemState.MONITOR

    def test_score_above_one_raises_value_error(self) -> None:
        """Score > 1.0 is rejected by range validation."""
        fsm = self._make_fsm_in_monitor()
        with pytest.raises(ValueError, match="anomaly score must be in"):
            fsm.evaluate_anomaly_score(5.0)
        assert fsm.state == SystemState.MONITOR

    def test_boundary_zero_accepted(self) -> None:
        """Score 0.0 is within [0, 1] and should not raise."""
        fsm = self._make_fsm_in_monitor()
        result = fsm.evaluate_anomaly_score(0.0)
        assert result is None  # below all thresholds
        assert fsm.state == SystemState.MONITOR

    def test_boundary_one_accepted(self) -> None:
        """Score 1.0 is within [0, 1] and should trigger escalation."""
        fsm = self._make_fsm_in_monitor()
        result = fsm.evaluate_anomaly_score(1.0)
        assert result is not None
        assert fsm.state == SystemState.INVESTIGATE


# ---------------------------------------------------------------------------
# Anomaly score rollback on audit failure (C1)
# ---------------------------------------------------------------------------


class TestAnomalyScoreRollback:
    """evaluate_anomaly_score must restore latest_anomaly_score on failure."""

    def test_score_rolled_back_on_audit_failure(self) -> None:
        """If audit write fails, latest_anomaly_score must not remain stale."""

        class FailingWriter:
            def write_transition(self, _record: object) -> None:
                raise RuntimeError("Audit write failed")

        fsm = FSMOrchestrator(audit_writer=FailingWriter())  # type: ignore[arg-type]
        # Trigger seismic event - this also exercises write_transition, so
        # the seismic trigger itself will fail. We need a two-phase approach:
        # first trigger without failing writer, then attach failing writer.
        # Instead, use a writer that only fails on the SECOND call.

        class FailOnSecondWrite:
            def __init__(self) -> None:
                self._calls = 0

            def write_transition(self, _record: object) -> None:
                self._calls += 1
                if self._calls > 1:
                    raise RuntimeError("Audit write failed")

        writer = FailOnSecondWrite()
        fsm = FSMOrchestrator(audit_writer=writer)  # type: ignore[arg-type]

        # First write succeeds -> IDLE -> MONITOR
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert fsm.state == SystemState.MONITOR
        ctx = fsm.event_context
        assert ctx is not None
        initial_score = ctx.latest_anomaly_score

        # Second write fails -> MONITOR -> INVESTIGATE aborted
        with pytest.raises(RuntimeError, match="Audit write failed"):
            fsm.evaluate_anomaly_score(0.50)

        # FSM state must remain MONITOR
        assert fsm.state == SystemState.MONITOR

        # Critical: latest_anomaly_score must be rolled back
        ctx_after = fsm.event_context
        assert ctx_after is not None
        assert ctx_after.latest_anomaly_score == initial_score

    def test_score_updated_when_no_transition(self) -> None:
        """No-transition scores update memory and durable reader state."""
        calls: list[dict[str, object]] = []

        class _StubDb:
            is_connected = True

            def upsert_fsm_state(self, **kwargs: object) -> bool:
                calls.append(kwargs)
                return True

        fsm = FSMOrchestrator(db_client=_StubDb())
        fsm.evaluate_seismic_trigger(
            magnitude=7.0,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        assert fsm.state == SystemState.MONITOR
        calls.clear()

        # Score below T1: no transition, but current event context is persisted.
        result = fsm.evaluate_anomaly_score(0.20)
        assert result is None  # no transition
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_anomaly_score == 0.20
        assert len(calls) == 1
        persisted_context = calls[0]["event_context"]
        assert isinstance(persisted_context, dict)
        assert persisted_context["latest_anomaly_score"] == 0.20


# ---------------------------------------------------------------------------
# Sensor degraded flag
# ---------------------------------------------------------------------------


class TestSensorDegraded:
    """Tests for FSMOrchestrator.evaluate_coverage() and sensor_degraded flag."""

    def test_initial_state_not_degraded(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.sensor_degraded is False

    def test_degraded_when_below_minimum(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_coverage(1)
        assert fsm.sensor_degraded is True

    def test_degraded_when_zero_stations(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_coverage(0)
        assert fsm.sensor_degraded is True

    def test_not_degraded_at_minimum(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_coverage(2)
        assert fsm.sensor_degraded is False

    def test_restored_after_degradation(self) -> None:
        fsm = FSMOrchestrator()
        fsm.evaluate_coverage(1)
        assert fsm.sensor_degraded is True
        fsm.evaluate_coverage(3)
        assert fsm.sensor_degraded is False

    def test_does_not_change_fsm_state(self) -> None:
        """Coverage degradation must NOT trigger FSM state transitions."""
        fsm = FSMOrchestrator()
        fsm.evaluate_coverage(0)
        assert fsm.sensor_degraded is True
        assert fsm.state == SystemState.IDLE  # FSM state unchanged

    def test_coverage_change_is_persisted_once_per_transition(self) -> None:
        """The API and dashboard read durable state, so a change must reach it.

        Every other persist site hangs off a score update or a transition. In
        IDLE with no active event neither fires, which is exactly when a dark
        sensor network matters, so the flag would never have left the worker.
        """
        calls: list[dict[str, object]] = []

        class _Db:
            def upsert_fsm_state(self, **kwargs: object) -> None:
                calls.append(kwargs)

            def load_fsm_state(self) -> None:
                return None

        fsm = FSMOrchestrator(db_client=_Db())
        fsm.evaluate_coverage(1)
        assert [c["sensor_degraded"] for c in calls] == [True]

        # Unchanged coverage must not write again on every tick.
        fsm.evaluate_coverage(0)
        assert len(calls) == 1

        fsm.evaluate_coverage(4)
        assert [c["sensor_degraded"] for c in calls] == [True, False]

    def test_persist_failure_does_not_stop_coverage_tracking(self) -> None:
        """Coverage is an alarm flag, not an interlock."""

        class _FailingDb:
            def upsert_fsm_state(self, **kwargs: object) -> None:
                raise RuntimeError("database down")

            def load_fsm_state(self) -> None:
                return None

        fsm = FSMOrchestrator(db_client=_FailingDb())
        fsm.evaluate_coverage(1)
        assert fsm.sensor_degraded is True


# ---------------------------------------------------------------------------
# Post-hoc replay event initialization
# ---------------------------------------------------------------------------


class TestPostHocReplayEvent:
    @staticmethod
    def _identity(context_class: str = "POST_HOC_FINAL_PRODUCT") -> SeismicIdentity:
        return SeismicIdentity(
            provider="usgs",
            external_event_id="test-event",
            revision_id="test-event:final-product",
            revision_sha256="a" * 64,
            provider_updated_utc=None,
            kafka_partition=None,
            kafka_offset=None,
            context_class=context_class,
        )

    def test_uses_identity_and_origin_without_final_parameters(self) -> None:
        fsm = FSMOrchestrator()
        origin = datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC)

        record = fsm.initialize_post_hoc_replay_event(
            origin_time_utc=origin,
            seismic_identity=self._identity(),
        )

        assert fsm.state is SystemState.MONITOR
        assert record.from_state is SystemState.IDLE
        assert record.to_state is SystemState.MONITOR
        assert record.seismic_magnitude is None
        assert "identity and origin alignment only" in record.trigger_reason
        assert fsm.event_context is not None
        assert fsm.event_context.trigger_time_utc == origin
        assert fsm.event_context.seismic_magnitude == 0.0
        assert fsm.event_context.depth_km is None
        assert fsm.event_context.seismic_region == ""
        assert fsm.event_context.epicenter_lat == 0.0
        assert fsm.event_context.epicenter_lon == 0.0
        assert fsm.event_context.external_event_id == "test-event"
        assert fsm.event_context.trigger_revision_sha256 == "a" * 64

    def test_rejects_live_context_class(self) -> None:
        fsm = FSMOrchestrator()
        with pytest.raises(ValueError, match="POST_HOC_FINAL_PRODUCT"):
            fsm.initialize_post_hoc_replay_event(
                origin_time_utc=datetime.now(UTC),
                seismic_identity=self._identity("LIVE_RECEIPT_ORDERED"),
            )
        assert fsm.state is SystemState.IDLE
        assert fsm.event_context is None

    def test_rejects_second_initialization(self) -> None:
        fsm = FSMOrchestrator()
        fsm.initialize_post_hoc_replay_event(
            origin_time_utc=datetime.now(UTC),
            seismic_identity=self._identity(),
        )
        with pytest.raises(InvalidTransitionError, match="only.*IDLE"):
            fsm.initialize_post_hoc_replay_event(
                origin_time_utc=datetime.now(UTC),
                seismic_identity=self._identity(),
            )


# ---------------------------------------------------------------------------
# Seismic-only escalation (large shallow earthquake bypasses DART gate)
# ---------------------------------------------------------------------------


_SEISMIC_THRESHOLDS = ThresholdConfig(
    basin="pacific", t1=0.35, t2=0.60, t3=0.85,
)


class TestSeismicOnlyEscalation:
    """Tests for MONITOR -> ESCALATE on large shallow earthquakes without DART."""

    def test_m75_shallow_triggers_escalation(self) -> None:
        """M7.5 at 30km depth in tsunamigenic zone -> ESCALATE."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        record = fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=30.0,
        )
        assert record is not None
        assert fsm.state == SystemState.ESCALATE

    def test_m75_deep_does_not_escalate(self) -> None:
        """M7.5 at 150km depth -> stays MONITOR (too deep)."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=150.0,
        )
        assert fsm.state == SystemState.MONITOR

    def test_m70_shallow_does_not_escalate(self) -> None:
        """M7.0 at 30km -> stays MONITOR (below escalation_magnitude)."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=7.0, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=30.0,
        )
        assert fsm.state == SystemState.MONITOR

    def test_m75_unknown_depth_does_not_escalate(self) -> None:
        """M7.5 with depth_km=None -> stays MONITOR (fail-safe)."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=None,
        )
        assert fsm.state == SystemState.MONITOR

    def test_seismic_only_escalation_reason_text(self) -> None:
        """Transition reason mentions seismic-only and PTWC criteria."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        record = fsm.evaluate_seismic_trigger(
            magnitude=8.0, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=25.0,
        )
        assert record is not None
        assert "Seismic-only escalation" in record.trigger_reason
        assert "No DART" in record.trigger_reason

    def test_depth_at_threshold_does_not_escalate(self) -> None:
        """M7.5 at exactly 100km -> stays MONITOR (< 100, not <=)."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=7.5, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=100.0,
        )
        assert fsm.state == SystemState.MONITOR

    def test_dart_after_seismic_only_escalation(self) -> None:
        """DART confirmation updates context even after seismic-only ESCALATE."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=8.0, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=20.0,
        )
        assert fsm.state == SystemState.ESCALATE
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])
        assert fsm.event_context is not None
        assert fsm.event_context.dart_confirmation is True

    def test_human_resolve_after_seismic_only(self) -> None:
        """ESCALATE from seismic-only -> resolve -> IDLE."""
        fsm = FSMOrchestrator(thresholds=_SEISMIC_THRESHOLDS)
        fsm.evaluate_seismic_trigger(
            magnitude=8.0, region="pacific_nw",
            epicenter_lat=46.0, epicenter_lon=-130.0,
            tsunamigenic_zones={"pacific_nw"},
            depth_km=15.0,
        )
        assert fsm.state == SystemState.ESCALATE
        fsm.resolve_event()
        assert fsm.state == SystemState.IDLE


def test_dart_confirmation_one_way_latch_and_conditional_persist() -> None:
    """update_dart_confirmation enforces a one-way per-event latch (OR with the
    current value) and persists it CONDITIONALLY via persist_dart_confirmation
    (the DB layer guards on event_id + non-IDLE so a stale worker cannot
    resurrect a resolved event). It never calls the unconditional upsert for the
    latch.
    """
    persist_calls: list[object] = []
    upsert_calls: list[dict[str, object]] = []

    class _StubDb:
        is_connected = True

        def upsert_fsm_state(self, **kwargs: object) -> None:
            upsert_calls.append(kwargs)

        def persist_dart_confirmation(
            self, event_id: object, stations: list[str] | None = None
        ) -> bool:
            persist_calls.append((event_id, stations))
            return True

    fsm = FSMOrchestrator(db_client=_StubDb())
    fsm.evaluate_seismic_trigger(
        magnitude=7.0,
        region="pacific_rim",
        epicenter_lat=38.3,
        epicenter_lon=142.4,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
    )
    upsert_calls.clear()  # ignore the transition's own persist

    fsm.update_dart_confirmation(
        dart_confirmation=True, stations_in_event_mode=["21418"]
    )
    assert fsm.event_context is not None
    assert fsm.event_context.dart_confirmation is True

    # One-way latch: a later False must NOT clear it.
    fsm.update_dart_confirmation(dart_confirmation=False)
    assert fsm.event_context is not None
    assert fsm.event_context.dart_confirmation is True

    # Persisted conditionally (with the event id), not via the unconditional upsert.
    assert persist_calls
    assert persist_calls[0] == (fsm.event_context.event_id, ["21418"])
    assert persist_calls[1] == (fsm.event_context.event_id, ["21418"])
    assert upsert_calls == []


def test_resolve_event_persists_idle_without_stale_context() -> None:
    """Resolving to IDLE must persist current_state=IDLE with a cleared
    event_context, so recovery never reloads an inconsistent IDLE-with-event.
    """
    calls: list[dict[str, object]] = []

    class _StubDb:
        is_connected = True

        def upsert_fsm_state(self, **kwargs: object) -> None:
            calls.append(kwargs)

    fsm = FSMOrchestrator(db_client=_StubDb())
    # Drive to ESCALATE via a large shallow seismic-only trigger.
    fsm.evaluate_seismic_trigger(
        magnitude=8.0,
        region="pacific_rim",
        epicenter_lat=38.3,
        epicenter_lon=142.4,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        depth_km=10.0,
    )
    assert fsm.state == SystemState.ESCALATE
    calls.clear()

    fsm.resolve_event()

    assert fsm.state == SystemState.IDLE
    assert fsm.event_context is None
    # The persisted row reflects IDLE with NO event context.
    assert calls and calls[-1]["state"] == "IDLE"
    assert calls[-1]["event_context"] is None


def test_recover_from_db_drops_context_on_idle_row() -> None:
    """recover_from_db must not reconstruct an event context for an IDLE row,
    defending against a stale IDLE-with-context row from an older version.
    """
    from uuid import uuid4

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, object]:
            return {
                "current_state": "IDLE",
                "sensor_degraded": False,
                "event_context": {
                    "event_id": str(uuid4()),
                    "seismic_magnitude": 9.1,
                    "seismic_region": "japan",
                    "epicenter_lat": 38.3,
                    "epicenter_lon": 142.37,
                    "depth_km": 29.0,
                    "trigger_time_utc": "2011-03-11T05:46:24+00:00",
                    "latest_anomaly_score": 0.0,
                    "dart_confirmation": True,
                    "active_dart_stations": [],
                    "stations_in_event_mode": [],
                },
            }

    fsm = FSMOrchestrator(db_client=_StubDb())
    assert fsm.recover_from_db() is True
    assert fsm.state == SystemState.IDLE
    assert fsm.event_context is None


def test_recover_from_db_falls_back_to_idle_on_malformed_context() -> None:
    """A corrupt persisted context (bad UUID/timestamp) must leave the FSM in
    IDLE with no context and the recovery_failed flag set, not stranded in
    the row's active state with event_context=None.
    """

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, object]:
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": {
                    "event_id": "NOT-A-UUID",
                    "trigger_time_utc": "garbage",
                },
            }

    fsm = FSMOrchestrator(db_client=_StubDb())
    assert fsm.recover_from_db() is False
    assert fsm.state == SystemState.IDLE
    assert fsm.event_context is None
    assert fsm.recovery_failed is True


def test_recover_from_db_fails_on_active_state_without_context() -> None:
    """A non-IDLE durable row with a null or empty event_context must fall
    back to IDLE with recovery_failed set: every non-IDLE persist writes a
    context, so its absence means the row is corrupt, and recovering as
    active-without-context would drop new seismic triggers while the monitor
    timeout no-ops on the missing context.
    """

    for ctx_value, state in ((None, "ESCALATE"), ({}, "MONITOR")):

        class _StubDb:
            is_connected = True

            def load_fsm_state(self) -> dict[str, object]:
                return {
                    "current_state": state,
                    "sensor_degraded": False,
                    "event_context": ctx_value,
                }

        fsm = FSMOrchestrator(db_client=_StubDb())
        assert fsm.recover_from_db() is False
        assert fsm.state == SystemState.IDLE
        assert fsm.event_context is None
        assert fsm.recovery_failed is True


def test_recover_from_db_normalizes_aware_non_utc_timestamp() -> None:
    """A persisted trigger_time_utc with a non-UTC offset is normalized to
    UTC on recovery (the value is serialized and rendered as UTC downstream).
    """
    from uuid import uuid4

    class _StubDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, object]:
            return {
                "current_state": "MONITOR",
                "sensor_degraded": False,
                "event_context": {
                    "event_id": str(uuid4()),
                    "trigger_time_utc": "2010-02-27T15:00:00+09:00",
                },
            }

    fsm = FSMOrchestrator(db_client=_StubDb())
    assert fsm.recover_from_db() is True
    ctx = fsm.event_context
    assert ctx is not None
    assert ctx.trigger_time_utc.tzinfo == UTC
    assert ctx.trigger_time_utc == datetime(2010, 2, 27, 6, 0, tzinfo=UTC)


def test_recovery_failure_writes_audit_entry_once() -> None:
    """The first recovery failure in a process writes an fsm_recovery_failed
    audit entry through the normal best-effort append path. Repeated failures
    do not spam the trail.
    """
    from hazard_assessment.audit.logger import AuditLogger

    class _CorruptDb:
        is_connected = True

        def load_fsm_state(self) -> dict[str, object]:
            return {
                "current_state": "ESCALATE",
                "sensor_degraded": False,
                "event_context": None,
            }

    audit = AuditLogger()
    fsm = FSMOrchestrator(db_client=_CorruptDb(), audit_writer=audit)
    assert fsm.recover_from_db() is False
    entries = audit.get_entries(event_type="fsm_recovery_failed")
    assert len(entries) == 1
    assert entries[0].data["durable_state"] == "ESCALATE"

    # Second failure (API refreshes per request): no duplicate entry.
    assert fsm.recover_from_db() is False
    assert len(audit.get_entries(event_type="fsm_recovery_failed")) == 1


def test_recovery_failed_is_sticky_across_later_success() -> None:
    """recovery_failed is a per-process sticky alarm: a later successful
    recovery (e.g. the worker writing a fresh row) must not clear it before
    an operator has seen it.
    """
    from uuid import uuid4

    class _FlakyDb:
        is_connected = True

        def __init__(self) -> None:
            self.corrupt = True

        def load_fsm_state(self) -> dict[str, object]:
            if self.corrupt:
                return {
                    "current_state": "ESCALATE",
                    "sensor_degraded": False,
                    "event_context": None,
                }
            return {
                "current_state": "MONITOR",
                "sensor_degraded": False,
                "event_context": {
                    "event_id": str(uuid4()),
                    "trigger_time_utc": "2011-03-11T05:46:24+00:00",
                },
            }

    db = _FlakyDb()
    fsm = FSMOrchestrator(db_client=db)
    assert fsm.recover_from_db() is False
    assert fsm.recovery_failed is True

    db.corrupt = False
    assert fsm.recover_from_db() is True
    assert fsm.state == SystemState.MONITOR
    assert fsm.recovery_failed is True  # sticky until process restart


def test_monitor_timeout_measures_from_seismic_origin() -> None:
    """With origin_time_utc threaded into the context (the worker path), the
    monitor timeout measures from the quake's origin: an event whose origin
    is already past the window times out on the first check.
    """
    thresholds = ThresholdConfig(
        basin="pacific", t1=0.35, t2=0.60, t3=0.85, monitor_timeout_hours=12.0,
    )
    fsm = FSMOrchestrator(thresholds=thresholds)
    origin = datetime.now(UTC) - timedelta(hours=13)
    fsm.evaluate_seismic_trigger(
        magnitude=7.0,
        region="pacific_rim",
        epicenter_lat=38.3,
        epicenter_lon=142.4,
        tsunamigenic_zones={"pacific_rim"},
        origin_time_utc=origin,
    )
    assert fsm.state == SystemState.MONITOR

    record = fsm.check_monitor_timeout()
    assert record is not None
    assert fsm.state == SystemState.IDLE
    assert fsm.event_context is None


# ---------------------------------------------------------------------------
# Seismic revision identity
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 7, 16, 1, 0, 0, tzinfo=UTC)


def _identity(
    *,
    provider: str = "usgs",
    external_event_id: str = "us7000abcd",
    revision_id: str = "seismic:us7000abcd:20260716010000000000",
    revision_sha256: str = "a" * 64,
    provider_updated_utc: datetime | None = _T0,
    kafka_partition: int | None = 0,
    kafka_offset: int | None = 100,
    context_class: str = "LIVE_RECEIPT_ORDERED",
) -> SeismicIdentity:
    return SeismicIdentity(
        provider=provider,
        external_event_id=external_event_id,
        revision_id=revision_id,
        revision_sha256=revision_sha256,
        provider_updated_utc=provider_updated_utc,
        kafka_partition=kafka_partition,
        kafka_offset=kafka_offset,
        context_class=context_class,
    )


def _trigger(
    fsm: FSMOrchestrator, identity: SeismicIdentity | None
) -> None:
    record = fsm.evaluate_seismic_trigger(
        magnitude=7.0,
        region="pacific_rim",
        epicenter_lat=38.3,
        epicenter_lon=142.4,
        tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        seismic_identity=identity,
    )
    assert record is not None
    assert fsm.state == SystemState.MONITOR


class TestSeismicIdentityBinding:
    def test_trigger_binds_identity_and_latest_starts_as_trigger(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.seismic_provider == "usgs"
        assert ctx.external_event_id == "us7000abcd"
        assert ctx.trigger_revision_id == "seismic:us7000abcd:20260716010000000000"
        assert ctx.trigger_revision_sha256 == "a" * 64
        assert ctx.latest_revision_id == ctx.trigger_revision_id
        assert ctx.latest_revision_sha256 == ctx.trigger_revision_sha256
        assert ctx.latest_revision_updated_utc == _T0
        assert ctx.latest_revision_kafka_partition == 0
        assert ctx.latest_revision_kafka_offset == 100
        assert ctx.seismic_context_class == "LIVE_RECEIPT_ORDERED"

    def test_trigger_without_identity_leaves_identity_unbound(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, None)
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.seismic_provider == ""
        assert ctx.external_event_id == ""
        assert ctx.trigger_revision_id == ""
        assert ctx.latest_revision_id == ""
        assert ctx.latest_revision_updated_utc is None
        assert ctx.latest_revision_kafka_partition is None
        assert ctx.seismic_context_class == ""

    def test_naive_provider_update_time_rejected(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            _identity(provider_updated_utc=datetime(2026, 7, 16, 1, 0, 0))

    def test_naive_latest_revision_update_time_rejected_on_context(self) -> None:
        with pytest.raises(ValueError, match="timezone-aware"):
            EventContext(
                latest_revision_updated_utc=datetime(2026, 7, 16, 1, 0, 0)
            )


class TestUpdateSeismicRevision:
    def test_noop_without_active_event(self) -> None:
        fsm = FSMOrchestrator()
        assert fsm.update_seismic_revision(_identity()) is False

    def test_unbound_event_rejects_revisions(self) -> None:
        """An event created without external identity (offline caller or
        pre-identity durable row) can never match a revision."""
        fsm = FSMOrchestrator()
        _trigger(fsm, None)
        assert fsm.update_seismic_revision(_identity()) is False

    def test_unrelated_event_id_does_not_replace_identity(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        later = _identity(
            external_event_id="us9999zzzz",
            revision_id="seismic:us9999zzzz:20260716020000000000",
            provider_updated_utc=_T0 + timedelta(hours=1),
            kafka_offset=200,
        )
        assert fsm.update_seismic_revision(later) is False
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.external_event_id == "us7000abcd"
        assert ctx.latest_revision_id == ctx.trigger_revision_id

    def test_provider_mismatch_rejected(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        assert fsm.update_seismic_revision(
            _identity(
                provider="emsc",
                provider_updated_utc=_T0 + timedelta(minutes=10),
            )
        ) is False

    def test_newer_update_time_supersedes_and_trigger_immutable(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        rev = _identity(
            revision_id="seismic:us7000abcd:20260716011000000000",
            revision_sha256="b" * 64,
            provider_updated_utc=_T0 + timedelta(minutes=10),
            kafka_offset=140,
        )
        assert fsm.update_seismic_revision(rev) is True
        ctx = fsm.event_context
        assert ctx is not None
        # Latest advanced.
        assert ctx.latest_revision_id == rev.revision_id
        assert ctx.latest_revision_sha256 == "b" * 64
        assert ctx.latest_revision_updated_utc == _T0 + timedelta(minutes=10)
        assert ctx.latest_revision_kafka_offset == 140
        # Trigger untouched.
        assert ctx.trigger_revision_id == "seismic:us7000abcd:20260716010000000000"
        assert ctx.trigger_revision_sha256 == "a" * 64

    def test_equal_ordering_tuple_redelivery_is_noop(self) -> None:
        """An at-least-once redelivery of the incumbent revision produces an
        exactly equal ordering tuple and must not supersede."""
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        assert fsm.update_seismic_revision(_identity()) is False

    def test_older_update_time_rejected(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        assert fsm.update_seismic_revision(
            _identity(provider_updated_utc=_T0 - timedelta(minutes=5))
        ) is False

    def test_kafka_position_breaks_equal_update_time(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        rev = _identity(
            revision_id="rev-later-receipt",
            revision_sha256="c" * 64,
            kafka_offset=101,
        )
        assert fsm.update_seismic_revision(rev) is True
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_id == "rev-later-receipt"

    def test_unknown_position_loses_to_known_at_equal_update_time(self) -> None:
        """Unknown coordinates compare as (-1, -1), so they cannot displace a
        known receipt position at the same provider update time."""
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        rev = _identity(
            revision_id="rev-no-coords",
            kafka_partition=None,
            kafka_offset=None,
        )
        assert fsm.update_seismic_revision(rev) is False

    def test_payload_hash_is_final_tiebreak(self) -> None:
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        rev = _identity(revision_id="rev-hash-tiebreak", revision_sha256="f" * 64)
        assert fsm.update_seismic_revision(rev) is True
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_sha256 == "f" * 64

    def test_missing_update_time_never_supersedes(self) -> None:
        """A revision whose provider update time was missing, malformed, or
        post-receipt-future (worker nulls it) stays provenance only."""
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity())
        rev = _identity(
            revision_id="rev-invalid-updated",
            provider_updated_utc=None,
            kafka_offset=999,
        )
        assert fsm.update_seismic_revision(rev) is False
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_id == ctx.trigger_revision_id

    def test_valid_revision_supersedes_incumbent_without_update_time(self) -> None:
        """When the trigger itself carried no valid provider update time, any
        valid matching revision supersedes it."""
        fsm = FSMOrchestrator()
        _trigger(fsm, _identity(provider_updated_utc=None))
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_updated_utc is None
        rev = _identity(revision_id="rev-first-valid", kafka_offset=150)
        assert fsm.update_seismic_revision(rev) is True
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.latest_revision_id == "rev-first-valid"
        assert ctx.latest_revision_updated_utc == _T0

    def test_supersede_persists_conditionally(self) -> None:
        """A superseding revision is persisted via persist_seismic_revision
        (conditional on same event + non-IDLE), never via the unconditional
        upsert."""
        persist_calls: list[tuple[object, dict[str, object]]] = []
        upsert_calls: list[dict[str, object]] = []

        class _StubDb:
            is_connected = True

            def upsert_fsm_state(self, **kwargs: object) -> None:
                upsert_calls.append(kwargs)

            def persist_seismic_revision(
                self, event_id: object, revision: dict[str, object]
            ) -> bool:
                persist_calls.append((event_id, revision))
                return True

        fsm = FSMOrchestrator(db_client=_StubDb())
        _trigger(fsm, _identity())
        upsert_calls.clear()  # ignore the transition's own persist

        rev = _identity(
            revision_id="rev-2",
            revision_sha256="b" * 64,
            provider_updated_utc=_T0 + timedelta(minutes=10),
            kafka_offset=140,
        )
        assert fsm.update_seismic_revision(rev) is True

        ctx = fsm.event_context
        assert ctx is not None
        assert len(persist_calls) == 1
        event_id, payload = persist_calls[0]
        assert event_id == ctx.event_id
        assert payload["latest_revision_id"] == "rev-2"
        assert payload["latest_revision_sha256"] == "b" * 64
        assert payload["latest_revision_updated_utc"] == (
            (_T0 + timedelta(minutes=10)).isoformat()
        )
        assert payload["latest_revision_kafka_partition"] == 0
        assert payload["latest_revision_kafka_offset"] == 140
        assert upsert_calls == []

    def test_rejected_revision_does_not_touch_db(self) -> None:
        persist_calls: list[object] = []

        class _StubDb:
            is_connected = True

            def upsert_fsm_state(self, **kwargs: object) -> None:
                pass

            def persist_seismic_revision(
                self, event_id: object, revision: dict[str, object]
            ) -> bool:
                persist_calls.append(event_id)
                return True

        fsm = FSMOrchestrator(db_client=_StubDb())
        _trigger(fsm, _identity())
        assert fsm.update_seismic_revision(
            _identity(provider_updated_utc=_T0 - timedelta(minutes=5))
        ) is False
        assert persist_calls == []


class TestSeismicIdentityPersistence:
    def test_identity_roundtrips_through_persist_and_recover(self) -> None:
        """The identity fields written by _persist_state recover bit-exact on
        a fresh FSM (both identities survive restart)."""
        calls: list[dict[str, object]] = []

        class _CaptureDb:
            is_connected = True

            def upsert_fsm_state(self, **kwargs: object) -> None:
                calls.append(kwargs)

        fsm = FSMOrchestrator(db_client=_CaptureDb())
        _trigger(fsm, _identity())
        rev = _identity(
            revision_id="rev-2",
            revision_sha256="b" * 64,
            provider_updated_utc=_T0 + timedelta(minutes=10),
            kafka_offset=140,
        )
        # No persist_seismic_revision on the stub: the best-effort durable
        # update is absent (e.g. transient DB error), but the next full
        # persist must still carry the advanced revision.
        assert fsm.update_seismic_revision(rev) is True
        fsm._persist_state()
        durable_ctx = calls[-1]["event_context"]
        assert isinstance(durable_ctx, dict)
        assert durable_ctx["trigger_revision_id"] == (
            "seismic:us7000abcd:20260716010000000000"
        )
        assert durable_ctx["latest_revision_id"] == "rev-2"
        assert durable_ctx["latest_revision_updated_utc"] == (
            (_T0 + timedelta(minutes=10)).isoformat()
        )

        class _LoadDb:
            is_connected = True

            def load_fsm_state(self) -> dict[str, object]:
                return {
                    "current_state": "MONITOR",
                    "sensor_degraded": False,
                    "event_context": durable_ctx,
                }

        recovered = FSMOrchestrator(db_client=_LoadDb())
        assert recovered.recover_from_db() is True
        ctx = recovered.event_context
        assert ctx is not None
        assert ctx.seismic_provider == "usgs"
        assert ctx.external_event_id == "us7000abcd"
        assert ctx.trigger_revision_id == (
            "seismic:us7000abcd:20260716010000000000"
        )
        assert ctx.trigger_revision_sha256 == "a" * 64
        assert ctx.latest_revision_id == "rev-2"
        assert ctx.latest_revision_sha256 == "b" * 64
        assert ctx.latest_revision_updated_utc == _T0 + timedelta(minutes=10)
        assert ctx.latest_revision_updated_utc.tzinfo == UTC
        assert ctx.latest_revision_kafka_partition == 0
        assert ctx.latest_revision_kafka_offset == 140
        assert ctx.seismic_context_class == "LIVE_RECEIPT_ORDERED"
        # A matching revision keeps ordering against the recovered incumbent.
        assert recovered.update_seismic_revision(rev) is False

    def test_recover_defaults_identity_for_pre_identity_rows(self) -> None:
        """A durable row written before identity fields existed recovers as
        an event with no bound external identity."""
        from uuid import uuid4

        class _OldRowDb:
            is_connected = True

            def load_fsm_state(self) -> dict[str, object]:
                return {
                    "current_state": "MONITOR",
                    "sensor_degraded": False,
                    "event_context": {
                        "event_id": str(uuid4()),
                        "trigger_time_utc": "2011-03-11T05:46:24+00:00",
                    },
                }

        fsm = FSMOrchestrator(db_client=_OldRowDb())
        assert fsm.recover_from_db() is True
        ctx = fsm.event_context
        assert ctx is not None
        assert ctx.seismic_provider == ""
        assert ctx.external_event_id == ""
        assert ctx.trigger_revision_id == ""
        assert ctx.latest_revision_id == ""
        assert ctx.latest_revision_updated_utc is None
        assert ctx.latest_revision_kafka_partition is None
        assert ctx.latest_revision_kafka_offset is None
        assert ctx.seismic_context_class == ""
        # And revisions cannot attach to it.
        assert fsm.update_seismic_revision(_identity()) is False


class TestThresholdBoundaryValues:
    """Exact-boundary cases for every inclusive comparison in the FSM.

    Mutation testing showed the suite passed with `M >= 7.5` changed to
    `M > 7.5` at all three seismic-override sites, and with both monitor
    timeout comparisons flipped, because every existing case used a value
    clear of the boundary (7.8 or 7.0 against 7.5, 2 h or 5 h against 4 h).
    An off-by-one on the override is the difference between escalating a
    magnitude-7.5 event and holding it.
    """

    @staticmethod
    def _armed_fsm() -> FSMOrchestrator:
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.5,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            depth_km=150.0,  # deep: no seismic-only escalation, so MONITOR
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])
        return fsm

    def test_magnitude_exactly_at_escalation_threshold_arms_the_override(self) -> None:
        """M == 7.5 must satisfy the override; the comparison is inclusive."""
        fsm = self._armed_fsm()
        assert fsm.state == SystemState.MONITOR
        fsm.evaluate_anomaly_score(0.10)
        assert fsm.state == SystemState.INVESTIGATE
        fsm.evaluate_anomaly_score(0.10)
        assert fsm.state == SystemState.ASSESS
        fsm.evaluate_anomaly_score(0.10)
        assert fsm.state == SystemState.ESCALATE

    def test_magnitude_just_below_threshold_does_not_arm_the_override(self) -> None:
        """M == 7.4 must not, or the boundary test above proves nothing."""
        fsm = FSMOrchestrator()
        fsm.evaluate_seismic_trigger(
            magnitude=7.4,
            region="pacific_rim",
            epicenter_lat=38.3,
            epicenter_lon=142.4,
            depth_km=150.0,
            tsunamigenic_zones=TSUNAMIGENIC_ZONES,
        )
        fsm.update_dart_confirmation(True, stations_in_event_mode=["21418"])
        assert fsm.state == SystemState.MONITOR
        assert fsm.evaluate_anomaly_score(0.10) is None
        assert fsm.state == SystemState.MONITOR
