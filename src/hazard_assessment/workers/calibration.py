"""Calibration data manager for the pipeline worker.

Loads pre-downloaded NDBC calibration CSVs (30-day quiet-period data) and
provides per-station tidal fit windows and baseline wavelet energies for
the anomaly detection pipeline.

Calibration data is loaded at pipeline startup from CSV files matching the
glob ``*_calibration.csv`` (e.g., ``dart_21413_chile_2010_calibration.csv``).
Each CSV has columns:
``station_id,timestamp_utc,seconds_from_origin,height_m``.

For live operation the calibration directory should contain recent quiet-period
data for each monitored DART station.  The validation scripts
(``scripts/download_chile_dart.py``) produce CSVs in this format.
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StationCalibration:
    """Calibration data for a single station.

    Attributes:
        station_id: Station identifier (e.g., "21413").
        times_hours: Time values in hours from first sample.
        values: Water level observations in meters.
        sampling_interval_sec: Median sampling interval in seconds.
        baseline_energy: Wavelet baseline energy computed from quiet data.
        span_days: Total time span of calibration data in days.
        source_sha256: SHA-256 of the raw CSV file bytes this calibration
            was parsed from, for assessment input-manifest provenance.
            Empty when no source file is known.
    """

    station_id: str
    times_hours: NDArray[np.float64]
    values: NDArray[np.float64]
    sampling_interval_sec: float
    baseline_energy: float
    span_days: float
    t0_epoch: float = 0.0  # Unix epoch of first calibration sample (seconds)
    source_sha256: str = ""


@dataclass
class CalibrationManager:
    """Manages calibration data for all monitored stations.

    Loads CSV files from a directory and computes tidal fit windows and
    baseline wavelet energies.  The AnomalyAgent uses these at runtime
    for accurate detiding and wavelet anomaly scoring.

    Usage::

        mgr = CalibrationManager()
        mgr.load_directory(Path("data/calibration"))
        # Access per-station data:
        cal = mgr.get(station_id)
        if cal:
            agent.calibrate_baseline(station_id, cal.values, cal.sampling_interval_sec)
    """

    _calibrations: dict[str, StationCalibration] = field(default_factory=dict)

    def load_csv(self, path: Path, station_id: str | None = None) -> StationCalibration | None:
        """Load calibration data from a single CSV file.

        Args:
            path: Path to the CSV file.
            station_id: Override station ID.  If None, inferred from
                the first data row's ``station_id`` column, or from
                the filename pattern ``dart_{station_id}_*_calibration.csv``.

        Returns:
            StationCalibration if successfully loaded, None on failure.
        """
        timestamps: list[float] = []
        values: list[float] = []
        inferred_id: str | None = station_id

        try:
            # Read the file once as bytes so the recorded SHA-256 is the
            # hash of exactly the bytes this calibration was parsed from
            # (no TOCTOU window between hashing and parsing).
            raw_bytes = path.read_bytes()
            source_sha256 = hashlib.sha256(raw_bytes).hexdigest()
            reader = csv.DictReader(io.StringIO(raw_bytes.decode("utf-8")))
            for row in reader:
                try:
                    ts = datetime.fromisoformat(row["timestamp_utc"])
                    # The column is UTC by contract, but fromisoformat
                    # returns a naive datetime when the value carries no
                    # offset, and .timestamp() would then read it as host
                    # local time and shift every calibration sample.
                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=UTC)
                    val = float(row["height_m"])
                except (ValueError, KeyError):
                    continue
                # Skip NDBC 9999.000 missing-data sentinels and
                # non-finite values (nan/inf compare False against
                # >= 9999.0, and a single NaN would later raise in
                # compute_wavelet_energy's finiteness check, crashing
                # worker startup). Matches the sibling boundaries in
                # ingest/dart.py and workers/station_buffer.py.
                if not math.isfinite(val) or val >= 9999.0:
                    continue
                timestamps.append(ts.timestamp())
                values.append(val)
                if inferred_id is None:
                    inferred_id = row.get("station_id")
        except (OSError, csv.Error, UnicodeDecodeError) as exc:
            logger.warning("Failed to read calibration CSV %s: %s", path, exc)
            return None

        if len(timestamps) < 10:
            logger.warning(
                "Insufficient calibration data in %s: %d rows (need >= 10)",
                path, len(timestamps),
            )
            return None

        # Infer station_id from filename if not found in data
        if inferred_id is None:
            name = path.stem
            parts = name.split("_")
            if len(parts) >= 2 and parts[0] == "dart":
                inferred_id = parts[1]
            else:
                inferred_id = name

        ts_arr = np.array(timestamps, dtype=np.float64)
        vals_arr = np.array(values, dtype=np.float64)

        # Sort by timestamp: an out-of-order CSV (e.g. concatenated downloads)
        # would otherwise yield a negative span and a wrong median sampling
        # interval from diffs containing negative steps.
        order = np.argsort(ts_arr, kind="stable")
        ts_arr = ts_arr[order]
        vals_arr = vals_arr[order]

        # Convert to hours from first sample
        t0 = ts_arr[0]
        times_hours = (ts_arr - t0) / 3600.0

        # Estimate sampling interval from median of consecutive diffs
        diffs = np.diff(ts_arr)
        sampling_sec = float(np.median(diffs))

        span_days = (ts_arr[-1] - ts_arr[0]) / 86400.0

        # Compute baseline wavelet energy on raw (non-detided) calibration data.
        # This intentionally includes tidal energy in the baseline, making the
        # wavelet scorer conservative: at runtime, wavelet energy is computed on
        # bandpass-filtered data (tidal energy removed), so the ratio
        # current/baseline < 1 for normal signals.  Only genuine tsunami-scale
        # anomalies produce filtered energy exceeding the raw baseline.
        # Note: calibration data (15-min standard mode) and runtime data (1-min
        # event mode) have different sampling rates, so the wavelet decomposition
        # covers different frequency bands.  Applying detide+bandpass here would
        # produce near-zero baseline (bandpass degrades at 15-min sampling) and
        # cause false high scores.  The raw-data baseline is the safer choice.
        from hazard_assessment.agents.anomaly_detection import compute_wavelet_energy

        energy = compute_wavelet_energy(vals_arr, sampling_sec)
        energy = max(energy, 1e-10)  # Floor to keep wavelet scoring enabled

        cal = StationCalibration(
            station_id=inferred_id,
            times_hours=times_hours,
            values=vals_arr,
            sampling_interval_sec=sampling_sec,
            baseline_energy=energy,
            span_days=span_days,
            t0_epoch=float(t0),
            source_sha256=source_sha256,
        )
        self._calibrations[inferred_id] = cal

        logger.info(
            "Loaded calibration for station %s: %d samples, %.1f s interval, "
            "%.1f days span, baseline_energy=%.2e",
            inferred_id, len(vals_arr), sampling_sec, span_days, energy,
        )
        return cal

    def load_directory(self, data_dir: Path) -> int:
        """Load all calibration CSVs from a directory.

        Searches for files matching ``*_calibration.csv`` in the given
        directory (non-recursive).

        Args:
            data_dir: Directory containing calibration CSV files.

        Returns:
            Number of stations successfully loaded.
        """
        if not data_dir.is_dir():
            logger.warning("Calibration directory does not exist: %s", data_dir)
            return 0

        count = 0
        for path in sorted(data_dir.glob("*_calibration.csv")):
            if self.load_csv(path) is not None:
                count += 1

        logger.info(
            "CalibrationManager: loaded %d station(s) from %s", count, data_dir,
        )
        return count

    def get(self, station_id: str) -> StationCalibration | None:
        """Get calibration data for a station, or None if not loaded."""
        return self._calibrations.get(station_id)

    def station_ids(self) -> list[str]:
        """Return list of calibrated station IDs."""
        return list(self._calibrations.keys())

    def apply_to_agent(self, agent: object) -> int:
        """Apply all loaded calibrations to an AnomalyAgent.

        Calls ``agent.set_baseline_energy()`` for each station with
        the pre-computed baseline wavelet energy.

        Args:
            agent: An AnomalyAgent instance.

        Returns:
            Number of stations calibrated.
        """
        from hazard_assessment.agents.anomaly_agent import AnomalyAgent

        if not isinstance(agent, AnomalyAgent):
            raise TypeError(f"Expected AnomalyAgent, got {type(agent).__name__}")

        count = 0
        for station_id, cal in self._calibrations.items():
            # Calibration CSVs are DART pressure series, so the baseline is
            # registered under the dart-qualified key.
            agent.set_baseline_energy(
                station_id, cal.baseline_energy, source_type="dart"
            )
            count += 1

        logger.info("Applied calibration to agent for %d station(s)", count)
        return count

    def __len__(self) -> int:
        return len(self._calibrations)

    def __contains__(self, station_id: str) -> bool:
        return station_id in self._calibrations
