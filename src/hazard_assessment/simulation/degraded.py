"""Network degradation utilities for robustness testing.

Simulates operational impairments: data gaps, partial network outages,
and short calibration windows.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.simulation.propagation import StationConfig


def apply_data_gaps(
    times_hours: NDArray[np.float64],
    signal: NDArray[np.float64],
    gap_start_hour: float,
    gap_duration_hours: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Remove a contiguous block of data to simulate a telemetry gap.

    Returns new arrays with the gap period removed. The anomaly detection
    pipeline must handle the resulting non-uniform time spacing.

    Args:
        times_hours: Time axis in hours.
        signal: Signal values.
        gap_start_hour: Start of gap in hours.
        gap_duration_hours: Duration of gap in hours.

    Returns:
        (times_with_gap, signal_with_gap) with gap period removed.
    """
    mask = ~(
        (times_hours >= gap_start_hour)
        & (times_hours < gap_start_hour + gap_duration_hours)
    )
    return times_hours[mask], signal[mask]


def mark_stations_offline(
    stations: list[StationConfig],
    offline_ids: set[str],
) -> tuple[list[StationConfig], list[str]]:
    """Split stations into online and offline groups.

    Args:
        stations: Full list of station configurations.
        offline_ids: Set of station IDs to mark as offline.

    Returns:
        (online_stations, offline_station_ids)
    """
    online = [s for s in stations if s.station_id not in offline_ids]
    offline = sorted(s.station_id for s in stations if s.station_id in offline_ids)
    return online, offline
