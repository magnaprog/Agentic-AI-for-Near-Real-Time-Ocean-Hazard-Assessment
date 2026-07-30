"""Geographic utilities and physical constants shared across packages.

Provides haversine distance calculation, tsunami travel-time estimation,
and the canonical deep-ocean wave speed constant.  These live here (rather
than inside the anomaly-detection module) so that the simulation package
can import them without depending on the agents package.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Deep-ocean tsunami phase speed
# ---------------------------------------------------------------------------
# c = sqrt(g * h) = sqrt(9.81 * 4000) ~ 198 m/s.
# Defined once; import this constant everywhere.
DEEP_OCEAN_SPEED_M_S: float = 198.0
DEEP_OCEAN_WAVE_SPEED_KM_S: float = DEEP_OCEAN_SPEED_M_S / 1000.0  # 0.198


# ---------------------------------------------------------------------------
# Average deep-ocean depth (meters) for dispersion calculations
# ---------------------------------------------------------------------------
MEAN_OCEAN_DEPTH_M: float = 4000.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two points in km."""
    r = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_travel_time_sec(distance_km: float) -> float:
    """Expected tsunami travel time in seconds at deep-ocean wave speed."""
    return distance_km / DEEP_OCEAN_WAVE_SPEED_KM_S


def compute_initial_bearing_deg(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Initial bearing (forward azimuth) from point 1 to point 2.

    Standard spherical formula; returns degrees clockwise from north
    in [0, 360).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_lambda = math.radians(lon2 - lon1)

    x = math.sin(d_lambda) * math.cos(phi2)
    y = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(d_lambda)
    return math.degrees(math.atan2(x, y)) % 360.0
