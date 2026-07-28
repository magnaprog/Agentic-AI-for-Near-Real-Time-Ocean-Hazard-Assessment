"""Tests for StationBuffer and StationBufferManager (workers/station_buffer.py)."""

from __future__ import annotations

import numpy as np
import pytest

from hazard_assessment.workers.station_buffer import (
    RetainedSample,
    RetainedSampleQC,
    StationBufferManager,
    StationWindow,
)


class TestStationWindow:
    def test_append_and_length(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        assert len(w) == 0
        w.append(1000.0, 5827.0)
        w.append(1060.0, 5827.1)
        assert len(w) == 2

    def test_append_skips_non_finite(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        w.append(1000.0, float("nan"))
        w.append(1060.0, float("inf"))
        assert len(w) == 0

    def test_append_skips_missing_sentinel(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        w.append(1000.0, 9999.0)
        w.append(1060.0, 9999.001)
        assert len(w) == 0

    def test_append_deduplicates(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        w.append(1000.0, 5827.0)
        w.append(1000.0, 5827.5)  # Duplicate timestamp
        assert len(w) == 1

    def test_append_returns_acceptance(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        assert w.append(1000.0, 5827.0) is True
        assert w.append(1000.0, 5827.5) is False  # Duplicate timestamp
        assert w.append(1060.0, 9999.0) is False  # Missing-data sentinel
        assert w.append(1120.0, float("nan")) is False  # Non-finite
        assert w.append(980.0, 5826.9) is True  # Out-of-order insert

    def test_duplicate_at_capacity_does_not_mutate(self) -> None:
        """A duplicate timestamp arriving at a full window must be rejected
        without evicting or replacing any retained sample."""
        from collections import deque

        w = StationWindow(station_id="21413", source_type="dart")
        w._observations = deque(maxlen=3)
        for t, v in [(1000.0, 1.0), (1060.0, 2.0), (1120.0, 3.0)]:
            w.append(t, v)

        # Duplicate of the OLDEST retained timestamp
        assert w.append(1000.0, 99.0) is False
        assert list(w._observations) == [
            RetainedSample(1000.0, 1.0),
            RetainedSample(1060.0, 2.0),
            RetainedSample(1120.0, 3.0),
        ]

        # Duplicate of a MIDDLE retained timestamp
        assert w.append(1060.0, 99.0) is False
        assert list(w._observations) == [
            RetainedSample(1000.0, 1.0),
            RetainedSample(1060.0, 2.0),
            RetainedSample(1120.0, 3.0),
        ]

    def test_older_than_all_at_capacity_is_rejected(self) -> None:
        """At capacity, a sample older than every retained observation is
        dropped instead of evicting a newer sample to admit a staler one."""
        from collections import deque

        w = StationWindow(station_id="21413", source_type="dart")
        w._observations = deque(maxlen=3)
        for t, v in [(1000.0, 1.0), (1060.0, 2.0), (1120.0, 3.0)]:
            w.append(t, v)

        assert w.append(940.0, 0.5) is False
        assert list(w._observations) == [
            RetainedSample(1000.0, 1.0),
            RetainedSample(1060.0, 2.0),
            RetainedSample(1120.0, 3.0),
        ]

        # Below capacity the same sample is accepted (insert at front).
        w2 = StationWindow(station_id="21413", source_type="dart")
        w2.append(1000.0, 1.0)
        assert w2.append(940.0, 0.5) is True
        assert list(w2._observations) == [
            RetainedSample(940.0, 0.5),
            RetainedSample(1000.0, 1.0),
        ]

    def test_append_out_of_order(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        w.append(1060.0, 5827.1)
        w.append(1000.0, 5827.0)  # Earlier timestamp
        w.append(1120.0, 5827.2)

        # Only 3 observations (< 10 minimum for to_arrays).
        # Verify the internal deque is correctly time-sorted.
        assert len(w) == 3
        obs_list = list(w._observations)
        assert obs_list[0].epoch_sec == 1000.0  # Earliest first
        assert obs_list[1].epoch_sec == 1060.0
        assert obs_list[2].epoch_sec == 1120.0

    def test_to_arrays_insufficient_data(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        for i in range(5):
            w.append(1000.0 + i * 60, 5827.0 + i * 0.01)
        assert w.to_arrays() is None  # Need >= 10

    def test_to_arrays_returns_correct_shape(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        for i in range(20):
            w.append(1000.0 + i * 60, 5827.0 + 0.1 * np.sin(i))
        result = w.to_arrays()

        assert result is not None
        times_hours, values, sampling_sec, t0_epoch = result
        assert len(times_hours) == 20
        assert len(values) == 20
        assert times_hours[0] == pytest.approx(0.0)
        assert sampling_sec == pytest.approx(60.0)
        assert t0_epoch == pytest.approx(1000.0)

    def test_trim_removes_old_data(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart", window_sec=300)
        base = 1000.0
        for i in range(20):
            w.append(base + i * 60, 5827.0)
        assert len(w) == 20

        # Trim to a 300-second window from the latest observation
        now = base + 19 * 60  # latest = base + 1140
        removed = w.trim(now)
        assert removed > 0
        # Only observations within [now - 300, now] should remain
        assert len(w) <= 6  # ~5 minutes worth at 60s intervals

    def test_trim_with_no_data(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        assert w.trim() == 0

    def test_latest_epoch(self) -> None:
        w = StationWindow(station_id="21413", source_type="dart")
        assert w.latest_epoch is None
        w.append(1000.0, 5827.0)
        w.append(1060.0, 5827.1)
        assert w.latest_epoch == 1060.0


class TestStationBufferManager:
    def test_get_or_create(self) -> None:
        mgr = StationBufferManager()
        w = mgr.get_or_create("21413", "dart")
        assert w.station_id == "21413"
        assert w.source_type == "dart"
        # Second call returns same window
        w2 = mgr.get_or_create("21413", "dart")
        assert w is w2

    def test_append_dart(self) -> None:
        mgr = StationBufferManager()
        mgr.append_dart("21413", 1000.0, 5827.0, event_mode=True)
        assert ("dart", "21413") in mgr
        w = mgr.get_window("21413", "dart")
        assert w is not None
        assert len(w) == 1
        assert w.event_mode is True

    def test_append_coops(self) -> None:
        mgr = StationBufferManager()
        mgr.append_coops("1612340", 1000.0, 0.5)
        w = mgr.get_window("1612340", "coops")
        assert w is not None
        assert w.source_type == "coops"

    def test_equal_station_ids_from_different_sources_do_not_collide(self) -> None:
        """A DART buoy and a CO-OPS gauge with the same identifier must
        keep separate windows."""
        mgr = StationBufferManager()
        mgr.append_dart("21413", 1000.0, 5827.0)
        mgr.append_coops("21413", 1000.0, 0.5)

        dart_w = mgr.get_window("21413", "dart")
        coops_w = mgr.get_window("21413", "coops")
        assert dart_w is not None and coops_w is not None
        assert dart_w is not coops_w
        assert len(mgr) == 2
        assert dart_w._observations[0].value == 5827.0
        assert coops_w._observations[0].value == 0.5

    def test_retained_samples_carry_metadata(self) -> None:
        """Accepted samples must carry source metadata, measurement type or
        product, payload hash, and per-record QC."""
        qc = RetainedSampleQC(
            usable=True,
            flags=(("range", 1), ("spike", 1)),
            confidence=0.97,
            n_checks_evaluated=2,
        )
        mgr = StationBufferManager()
        mgr.append_dart(
            "21413", 1000.0, 5827.0,
            payload_hash="a" * 64, measurement_type=2, qc=qc,
        )
        mgr.append_coops(
            "1612340", 1000.0, 0.5,
            payload_hash="b" * 64, product="water_level", qc=qc,
        )

        dart_w = mgr.get_window("21413", "dart")
        assert dart_w is not None
        dart_sample = dart_w.retained_samples()[0]
        assert dart_sample.payload_hash == "a" * 64
        assert dart_sample.measurement_type == 2
        assert dart_sample.product is None
        assert dart_sample.qc == qc

        coops_w = mgr.get_window("1612340", "coops")
        assert coops_w is not None
        coops_sample = coops_w.retained_samples()[0]
        assert coops_sample.payload_hash == "b" * 64
        assert coops_sample.measurement_type is None
        assert coops_sample.product == "water_level"
        assert coops_sample.qc == qc

    def test_stations_in_event_mode(self) -> None:
        mgr = StationBufferManager()
        mgr.append_dart("21413", 1000.0, 5827.0, event_mode=False)
        mgr.append_dart("21418", 1000.0, 5500.0, event_mode=True)
        mgr.append_coops("1612340", 1000.0, 0.5)

        event_stations = mgr.stations_in_event_mode()
        assert event_stations == ["21418"]

    def test_event_mode_clears_on_standard_record(self) -> None:
        """Event mode should clear when standard-mode records resume."""
        mgr = StationBufferManager()
        mgr.append_dart("21418", 1000.0, 5500.0, event_mode=True)
        assert mgr.stations_in_event_mode() == ["21418"]

        # Standard-mode record clears event mode
        mgr.append_dart("21418", 1060.0, 5500.1, event_mode=False)
        assert mgr.stations_in_event_mode() == []

    def test_rejected_sample_does_not_set_event_mode(self) -> None:
        """A missing-data sentinel must not contribute event-mode evidence."""
        mgr = StationBufferManager()
        accepted = mgr.append_dart("21418", 1000.0, 9999.0, event_mode=True)
        assert accepted is False
        assert mgr.get_window("21418", "dart") is None
        assert mgr.stations_in_event_mode() == []

    def test_rejected_sample_does_not_clear_event_mode(self) -> None:
        """A rejected sample must not overwrite the last accepted record's mode."""
        mgr = StationBufferManager()
        mgr.append_dart("21418", 1000.0, 5500.0, event_mode=True)
        mgr.append_dart("21418", 1060.0, 9999.0, event_mode=False)
        assert mgr.stations_in_event_mode() == ["21418"]

    def test_trim_to_empty_clears_event_mode(self) -> None:
        """A fully trimmed window holds no current evidence: the event-mode
        flag must not keep reporting the station as active (it would pin the
        worker on the faster event-mode buffer cadence indefinitely)."""
        mgr = StationBufferManager()
        mgr.append_dart("21418", 1000.0, 5500.0, event_mode=True)
        assert mgr.stations_in_event_mode() == ["21418"]

        # Trim far past the window: removes the only observation and prunes
        # the empty station entry.
        mgr.trim_all(now_epoch=1000.0 + 10 * 24 * 3600)
        assert mgr.get_window("21418", "dart") is None
        assert mgr.stations_in_event_mode() == []

    def test_delayed_older_record_does_not_change_event_mode(self) -> None:
        """The flag tracks the NEWEST observation by observation time: a
        delayed older standard-mode record must not clear a current
        event-mode state, and a delayed older event-mode record must not
        resurrect one."""
        mgr = StationBufferManager()
        mgr.append_dart("21418", 1120.0, 5500.0, event_mode=True)
        accepted = mgr.append_dart("21418", 1000.0, 5499.9, event_mode=False)
        assert accepted is True  # Sample enters the window...
        assert mgr.stations_in_event_mode() == ["21418"]  # ...flag unchanged

        mgr.append_dart("21418", 1180.0, 5500.1, event_mode=False)
        assert mgr.stations_in_event_mode() == []
        accepted = mgr.append_dart("21418", 1060.0, 5500.0, event_mode=True)
        assert accepted is True
        assert mgr.stations_in_event_mode() == []

    def test_station_keys(self) -> None:
        mgr = StationBufferManager()
        mgr.append_dart("21413", 1000.0, 5827.0)
        mgr.append_dart("21418", 1000.0, 5500.0)
        mgr.append_coops("1612340", 1000.0, 0.5)
        assert set(mgr.station_keys()) == {
            ("dart", "21413"), ("dart", "21418"), ("coops", "1612340"),
        }

    def test_trim_all(self) -> None:
        mgr = StationBufferManager(window_sec=300)
        base = 1000.0
        for i in range(20):
            mgr.append_dart("21413", base + i * 60, 5827.0)

        removed = mgr.trim_all(base + 19 * 60)
        assert removed > 0

    def test_get_window_nonexistent(self) -> None:
        mgr = StationBufferManager()
        assert mgr.get_window("nonexistent", "dart") is None

    def test_contains(self) -> None:
        mgr = StationBufferManager()
        assert ("dart", "21413") not in mgr
        mgr.append_dart("21413", 1000.0, 5827.0)
        assert ("dart", "21413") in mgr
        assert ("coops", "21413") not in mgr
