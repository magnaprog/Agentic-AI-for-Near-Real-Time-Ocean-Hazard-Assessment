"""Rolling station observation buffer for the pipeline worker.

Accumulates live DART and CO-OPS observations per station in a rolling
window, providing numpy arrays suitable for the anomaly detection pipeline.

Each station maintains a time-sorted deque of :class:`RetainedSample`
entries. When the pipeline worker processes a Kafka buffer batch, it
appends new observations and then extracts the rolling window as numpy
arrays for ``AnomalyAgent.process_station_data()``.

Windows are keyed by ``(source_type, station_id)`` so equal station
identifiers from different sources cannot collide (implementation plan
4.5). Retained samples carry measurement type or product, payload hash,
and per-record QC so the ocean evidence assessment can be built without
rerunning stateful QC.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# Default rolling window: 6 hours of data for anomaly detection.
# This provides enough context for tidal fitting (when no separate
# calibration window is available) and wavelet/BOCPD analysis.
DEFAULT_WINDOW_SEC = 6 * 3600

# Maximum observations to keep per station (safety bound against
# memory leaks from stuck connectors producing duplicate records).
MAX_OBSERVATIONS = 50_000


@dataclass(frozen=True, slots=True)
class RetainedSampleQC:
    """Per-record QC computed once at record-processing time.

    Attached to the retained sample so the retained-window QC aggregate
    can be built from the exact records presented to scoring, without
    rerunning stateful QC checks.

    Attributes:
        usable: QC record_usable verdict (metadata only; never filters).
        flags: Sorted ``(check_name, qartod_flag_int)`` pairs.
        confidence: Station confidence for this record.
        n_checks_evaluated: Checks that produced a definitive result.
    """

    usable: bool
    flags: tuple[tuple[str, int], ...]
    confidence: float
    n_checks_evaluated: int


@dataclass(frozen=True, slots=True)
class RetainedSample:
    """One retained observation with its assessment-facing metadata.

    ``measurement_type`` is the DART measurement type (1, 2, or 3);
    ``product`` is the CO-OPS product name. Exactly one of them is
    meaningful per source; both default to None for direct callers
    that supply bare samples.
    """

    epoch_sec: float
    value: float
    payload_hash: str | None = None
    measurement_type: int | None = None
    product: str | None = None
    qc: RetainedSampleQC | None = None


@dataclass
class StationWindow:
    """Rolling observation window for a single station.

    Attributes:
        station_id: Station identifier.
        source_type: "dart" or "coops".
        window_sec: Rolling window duration in seconds.
        event_mode: Whether the station is currently in event mode.
    """

    station_id: str
    source_type: str
    window_sec: float = DEFAULT_WINDOW_SEC
    event_mode: bool = False
    _observations: deque[RetainedSample] = field(
        default_factory=lambda: deque(maxlen=MAX_OBSERVATIONS),
    )

    def append(
        self,
        epoch_sec: float,
        value: float,
        payload_hash: str | None = None,
        *,
        measurement_type: int | None = None,
        product: str | None = None,
        qc: RetainedSampleQC | None = None,
    ) -> bool:
        """Append a single observation, maintaining time-sorted order.

        Duplicate timestamps are silently ignored (idempotent).
        Out-of-order observations are inserted in sorted position.
        ``payload_hash`` carries the raw record's SHA-256 alongside the
        sample so lineage rows can reference every observation actually
        scored, not just the newest batch. ``measurement_type``,
        ``product``, and ``qc`` ride with the sample for assessment
        construction.

        Returns True when the observation was accepted into the window,
        False when it was dropped (sentinel, non-finite, or duplicate).
        """
        if not np.isfinite(value) or value >= 9999.0:
            return False  # Skip missing-data sentinels and non-finite values

        sample = RetainedSample(
            epoch_sec=epoch_sec,
            value=value,
            payload_hash=payload_hash,
            measurement_type=measurement_type,
            product=product,
            qc=qc,
        )

        # Fast path: append in order (most common case for live data)
        if not self._observations or epoch_sec > self._observations[-1].epoch_sec:
            self._observations.append(sample)
            return True

        # Check for exact duplicate timestamp
        if epoch_sec == self._observations[-1].epoch_sec:
            return False

        # Out-of-order: insert in sorted position (rare in live data).
        # Find the insert position BEFORE any eviction so a duplicate
        # timestamp is rejected without mutating the window.
        insert_at = 0
        for i in range(len(self._observations) - 1, -1, -1):
            if self._observations[i].epoch_sec == epoch_sec:
                return False  # Duplicate
            if self._observations[i].epoch_sec < epoch_sec:
                insert_at = i + 1
                break
        # NOTE: deque.insert() on a full deque raises IndexError.  We guard
        # by dropping the oldest observation first when at capacity.
        if len(self._observations) >= (self._observations.maxlen or MAX_OBSERVATIONS):
            if insert_at == 0:
                # Older than every retained observation: evicting a newer
                # sample to admit a staler one would corrupt the rolling
                # window, so drop the incoming value instead.
                return False
            self._observations.popleft()
            insert_at -= 1
        self._observations.insert(insert_at, sample)
        return True

    def trim(self, now_epoch: float | None = None) -> int:
        """Remove observations outside the rolling window.

        Args:
            now_epoch: Current time as epoch seconds.  If None, uses
                the latest observation timestamp.

        Returns:
            Number of observations removed.
        """
        if not self._observations:
            return 0

        if now_epoch is None:
            now_epoch = self._observations[-1].epoch_sec

        cutoff = now_epoch - self.window_sec
        removed = 0
        while self._observations and self._observations[0].epoch_sec < cutoff:
            self._observations.popleft()
            removed += 1
        if not self._observations:
            # A fully trimmed window holds no current evidence: a lingering
            # event_mode flag would keep the worker on the faster event-mode
            # buffer cadence indefinitely (the dart_confirmation latch does
            # not read this flag, so only cadence is at stake). A genuinely
            # active station re-sets the flag with its next accepted row.
            self.event_mode = False
        return removed

    def to_arrays(
        self,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], float, float] | None:
        """Extract the rolling window as numpy arrays.

        Returns:
            Tuple of (times_hours, values, sampling_interval_sec, t0_epoch) or
            None if insufficient data (< 10 observations).

            - times_hours: Time in hours from the first observation.
            - values: Water level values in meters.
            - sampling_interval_sec: Median sampling interval.
            - t0_epoch: Unix epoch (seconds) of the first observation.
        """
        if len(self._observations) < 10:
            return None

        epochs = np.array(
            [obs.epoch_sec for obs in self._observations], dtype=np.float64
        )
        values = np.array(
            [obs.value for obs in self._observations], dtype=np.float64
        )

        t0 = epochs[0]
        times_hours = (epochs - t0) / 3600.0

        diffs = np.diff(epochs)
        sampling_sec = float(np.median(diffs))

        return times_hours, values, sampling_sec, float(t0)

    def __len__(self) -> int:
        return len(self._observations)

    @property
    def latest_epoch(self) -> float | None:
        """Epoch timestamp of the most recent observation, or None."""
        if self._observations:
            return self._observations[-1].epoch_sec
        return None

    def sample_hashes(self) -> list[str]:
        """Payload hashes of every retained sample that carries one.

        This is the lineage input set for anomaly scoring: the scorer
        consumes the whole retained window, so provenance must reference
        all retained samples, not just the newest batch.
        """
        return [
            obs.payload_hash for obs in self._observations if obs.payload_hash
        ]

    def retained_samples(self) -> list[RetainedSample]:
        """Snapshot of the retained window in time order.

        Assessment construction reads this to aggregate retained-window
        QC and per-sample metadata over the exact records presented to
        scoring.
        """
        return list(self._observations)


@dataclass
class StationBufferManager:
    """Manages rolling observation windows for all monitored stations.

    Provides a unified interface for the pipeline worker to append
    observations from Kafka records and extract per-station numpy
    arrays for anomaly detection.
    """

    _windows: dict[tuple[str, str], StationWindow] = field(default_factory=dict)
    window_sec: float = DEFAULT_WINDOW_SEC

    def get_or_create(self, station_id: str, source_type: str) -> StationWindow:
        """Get or create the StationWindow keyed by (source_type, station_id)."""
        key = (source_type, station_id)
        if key not in self._windows:
            self._windows[key] = StationWindow(
                station_id=station_id,
                source_type=source_type,
                window_sec=self.window_sec,
            )
        return self._windows[key]

    def append_dart(
        self,
        station_id: str,
        epoch_sec: float,
        height_m: float,
        event_mode: bool = False,
        payload_hash: str | None = None,
        *,
        measurement_type: int | None = None,
        qc: RetainedSampleQC | None = None,
    ) -> bool:
        """Append a DART observation.  Returns True if the sample was accepted.

        The event_mode flag tracks the mode of the NEWEST accepted
        observation by observation time, not arrival order - when
        standard-mode records resume (event_mode=False), the flag
        clears.  This matches NDBC behavior where event mode is
        time-bounded and the station returns to standard polling.
        A rejected sample (missing-data sentinel, non-finite, duplicate)
        does not update the flag: a record that never entered the window
        must not contribute event-mode evidence (see the dart_confirmation
        latch in pipeline_runner).  An accepted but out-of-order OLDER
        sample also leaves the flag alone, so a delayed standard-mode
        record cannot clear a current event-mode state and a delayed
        event-mode record cannot resurrect one.
        """
        window = self.get_or_create(station_id, "dart")
        accepted = window.append(
            epoch_sec, height_m, payload_hash,
            measurement_type=measurement_type, qc=qc,
        )
        if not accepted:
            if len(window) == 0:
                self._windows.pop(("dart", station_id), None)
            return False
        if window.latest_epoch == epoch_sec:
            window.event_mode = event_mode
        return True

    def append_coops(
        self,
        station_id: str,
        epoch_sec: float,
        water_level_m: float,
        payload_hash: str | None = None,
        *,
        product: str | None = None,
        qc: RetainedSampleQC | None = None,
    ) -> bool:
        """Append a CO-OPS observation.  Returns True if the sample was accepted."""
        window = self.get_or_create(station_id, "coops")
        accepted = window.append(
            epoch_sec, water_level_m, payload_hash,
            product=product, qc=qc,
        )
        if not accepted and len(window) == 0:
            self._windows.pop(("coops", station_id), None)
        return accepted

    def trim_all(self, now_epoch: float | None = None) -> int:
        """Trim all station windows and drop empty ones.  Returns removed count."""
        removed = 0
        for key, window in list(self._windows.items()):
            removed += window.trim(now_epoch)
            if len(window) == 0:
                del self._windows[key]
        return removed

    def get_window(
        self, station_id: str, source_type: str
    ) -> StationWindow | None:
        """Get the StationWindow keyed by (source_type, station_id), or None."""
        return self._windows.get((source_type, station_id))

    def stations_in_event_mode(self) -> list[str]:
        """Return DART station IDs currently in event mode.

        Only DART windows carry event mode, so bare station IDs are
        unambiguous here.
        """
        return [
            w.station_id for w in self._windows.values()
            if w.event_mode and w.source_type == "dart"
        ]

    def station_keys(self) -> list[tuple[str, str]]:
        """Return all tracked (source_type, station_id) keys."""
        return list(self._windows.keys())

    def __len__(self) -> int:
        return len(self._windows)

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._windows
