"""Tidal constituent frequencies shared across detection and simulation.

The eight constituents used for harmonic tidal removal, defined once here
and imported by both the anomaly-detection pipeline and the synthetic-data
generator so that simulated tides match what the detector expects.
"""

from __future__ import annotations

import math

# Major tidal constituents - angular frequencies in degrees/hour.
TIDAL_CONSTITUENTS: dict[str, float] = {
    "M2": 28.984104,  # principal lunar semidiurnal
    "S2": 30.000000,
    "N2": 28.439730,
    "K1": 15.041069,
    "O1": 13.943036,
    "P1": 14.958931,
    "K2": 30.082138,
    "Q1": 13.398661,
}

# Same frequencies in radians/hour.
TIDAL_FREQUENCIES_RAD_HR: dict[str, float] = {
    name: math.radians(deg_per_hr)
    for name, deg_per_hr in TIDAL_CONSTITUENTS.items()
}
