"""Tests for CalibrationManager (workers/calibration.py)."""

from __future__ import annotations

import csv
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from hazard_assessment.workers.calibration import CalibrationManager


def _write_calibration_csv(
    path: Path,
    station_id: str = "21413",
    n_samples: int = 100,
    interval_sec: float = 900.0,
    base_height: float = 5827.0,
) -> None:
    """Write a synthetic calibration CSV file."""
    origin = datetime(2010, 1, 28, 0, 0, 0, tzinfo=UTC)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["station_id", "timestamp_utc", "seconds_from_origin", "height_m"])
        for i in range(n_samples):
            ts = origin + timedelta(seconds=i * interval_sec)
            sec_from_origin = i * interval_sec
            # Add small tidal-like variation
            height = base_height + 0.5 * np.sin(2 * np.pi * i / 48)
            writer.writerow([station_id, ts.isoformat(), sec_from_origin, round(height, 3)])


class TestCalibrationManager:
    def test_load_single_csv(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        _write_calibration_csv(csv_path, station_id="21413", n_samples=50)

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is not None
        assert cal.station_id == "21413"
        assert len(cal.times_hours) == 50
        assert len(cal.values) == 50
        assert cal.sampling_interval_sec == pytest.approx(900.0, abs=1.0)
        assert cal.baseline_energy > 0
        assert cal.span_days > 0

    def test_load_directory(self, tmp_path: Path) -> None:
        # Write calibration files for 3 stations
        for sid in ["21413", "21418", "46411"]:
            path = tmp_path / f"dart_{sid}_test_calibration.csv"
            _write_calibration_csv(path, station_id=sid)

        mgr = CalibrationManager()
        count = mgr.load_directory(tmp_path)

        assert count == 3
        assert len(mgr) == 3
        assert "21413" in mgr
        assert "21418" in mgr
        assert "46411" in mgr

    def test_load_insufficient_data(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        _write_calibration_csv(csv_path, n_samples=5)  # Too few

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is None
        assert len(mgr) == 0

    def test_load_nonexistent_directory(self) -> None:
        mgr = CalibrationManager()
        count = mgr.load_directory(Path("/nonexistent/path"))
        assert count == 0

    def test_get_returns_none_for_unknown_station(self) -> None:
        mgr = CalibrationManager()
        assert mgr.get("unknown") is None

    def test_station_ids(self, tmp_path: Path) -> None:
        for sid in ["21413", "46411"]:
            path = tmp_path / f"dart_{sid}_test_calibration.csv"
            _write_calibration_csv(path, station_id=sid)

        mgr = CalibrationManager()
        mgr.load_directory(tmp_path)

        ids = mgr.station_ids()
        assert set(ids) == {"21413", "46411"}

    def test_apply_to_agent(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        _write_calibration_csv(csv_path, station_id="21413")

        mgr = CalibrationManager()
        mgr.load_csv(csv_path)

        from hazard_assessment.agents.anomaly_agent import AnomalyAgent

        agent = AnomalyAgent()
        count = mgr.apply_to_agent(agent)

        assert count == 1
        # Verify the agent has baseline energy set under the dart key
        assert agent._baseline_energies.get(("dart", "21413"), 0.0) > 0

    def test_apply_to_agent_wrong_type(self, tmp_path: Path) -> None:
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        _write_calibration_csv(csv_path, station_id="21413")

        mgr = CalibrationManager()
        mgr.load_csv(csv_path)

        with pytest.raises(TypeError, match="Expected AnomalyAgent"):
            mgr.apply_to_agent("not an agent")

    def test_skips_missing_data_sentinels(self, tmp_path: Path) -> None:
        """CSV rows with height_m >= 9999.0 should be skipped."""
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        origin = datetime(2010, 1, 28, 0, 0, 0, tzinfo=UTC)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["station_id", "timestamp_utc", "seconds_from_origin", "height_m"])
            for i in range(20):
                ts = origin + timedelta(seconds=i * 900)
                # Every 5th row is a sentinel
                height = 9999.0 if i % 5 == 0 else 5827.0 + 0.1 * i
                writer.writerow(["21413", ts.isoformat(), i * 900, height])

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is not None
        # 20 rows, 4 sentinels -> 16 valid
        assert len(cal.values) == 16

    def test_skips_non_finite_values(self, tmp_path: Path) -> None:
        """CSV rows with nan/inf height_m must be skipped like sentinels.

        nan and -inf compare False against >= 9999.0, so without an
        explicit finiteness check they would reach compute_wavelet_energy
        and raise (crashing worker startup) instead of being filtered.
        """
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        origin = datetime(2010, 1, 28, 0, 0, 0, tzinfo=UTC)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["station_id", "timestamp_utc", "seconds_from_origin", "height_m"])
            for i in range(20):
                ts = origin + timedelta(seconds=i * 900)
                if i % 7 == 0:
                    height = "nan"
                elif i % 7 == 1:
                    height = "-inf"
                else:
                    height = str(5827.0 + 0.1 * i)
                writer.writerow(["21413", ts.isoformat(), i * 900, height])

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is not None
        # 20 rows, 3 nan (i=0,7,14) + 3 -inf (i=1,8,15) -> 14 valid
        assert len(cal.values) == 14
        assert np.all(np.isfinite(cal.values))

    def test_infer_station_id_from_filename(self, tmp_path: Path) -> None:
        """When station_id column matches, use it; also test filename fallback."""
        csv_path = tmp_path / "dart_99999_test_calibration.csv"
        _write_calibration_csv(csv_path, station_id="99999")

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is not None
        assert cal.station_id == "99999"

    def test_baseline_energy_floor(self, tmp_path: Path) -> None:
        """Baseline energy should be floored at 1e-10."""
        csv_path = tmp_path / "dart_21413_test_calibration.csv"
        # Write nearly constant values (near-zero wavelet energy)
        origin = datetime(2010, 1, 28, 0, 0, 0, tzinfo=UTC)
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["station_id", "timestamp_utc", "seconds_from_origin", "height_m"])
            for i in range(50):
                ts = origin + timedelta(seconds=i * 900)
                writer.writerow(["21413", ts.isoformat(), i * 900, 5827.000])

        mgr = CalibrationManager()
        cal = mgr.load_csv(csv_path)

        assert cal is not None
        assert cal.baseline_energy >= 1e-10
