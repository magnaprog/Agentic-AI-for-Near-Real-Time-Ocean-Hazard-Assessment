"""Tsunami propagation: arrival times, geometric spreading, directivity, coastal amplification.

Physical models:
    - Arrival time: great-circle distance / deep-ocean phase speed (198 m/s).
    - Geometric spreading: 2D cylindrical spreading, A proportional to 1/sqrt(r).
    - Directivity: azimuthal radiation pattern based on fault strike/rake.
    - Coastal amplification: Green's Law, A_shore/A_deep = (h_deep/h_shore)^(1/4).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from hazard_assessment.geo import (
    DEEP_OCEAN_SPEED_M_S,
    MEAN_OCEAN_DEPTH_M,
    compute_initial_bearing_deg,
    haversine_km,
)

# Reference distance for amplitude normalization.
# Source amplitude specification corresponds to amplitude at this distance.
REFERENCE_DISTANCE_KM = 1000.0


@dataclass(frozen=True)
class StationConfig:
    """Configuration for a simulated monitoring station."""

    station_id: str
    latitude: float
    longitude: float
    depth_m: float  # ocean depth at station location
    station_type: str  # "dart" or "coops"
    sampling_interval_sec: float  # 15, 60, or 360


def compute_arrival_time_hours(
    epicenter_lat: float,
    epicenter_lon: float,
    station_lat: float,
    station_lon: float,
    wave_speed_m_s: float = DEEP_OCEAN_SPEED_M_S,
) -> float:
    """Compute tsunami arrival time at a station.

    Uses great-circle distance and constant deep-ocean wave speed.

    Args:
        epicenter_lat: Earthquake epicenter latitude.
        epicenter_lon: Earthquake epicenter longitude.
        station_lat: Station latitude.
        station_lon: Station longitude.
        wave_speed_m_s: Tsunami phase speed in m/s.

    Returns:
        Arrival time in hours relative to earthquake origin.
    """
    dist_km = haversine_km(epicenter_lat, epicenter_lon, station_lat, station_lon)
    travel_sec = (dist_km * 1000.0) / wave_speed_m_s
    return travel_sec / 3600.0


def compute_geometric_spreading_factor(
    epicenter_lat: float,
    epicenter_lon: float,
    station_lat: float,
    station_lon: float,
    reference_distance_km: float = REFERENCE_DISTANCE_KM,
) -> float:
    """Compute 2D geometric spreading amplitude decay factor.

    In 2D (surface waves on a sphere), energy spreads as 1/r, so
    amplitude decays as 1/sqrt(r). Normalized to a reference distance
    so the source amplitude corresponds to the amplitude there.

        factor = sqrt(reference_distance / actual_distance)

    Clamped to [0.1, 1.5] for physical plausibility, but at the default
    reference distance only the upper clamp can fire. Distance is floored
    at 10 km and cannot exceed a half circumference (20015 km), so the
    returned factor spans [0.2235, 1.5].

    Args:
        epicenter_lat: Earthquake epicenter latitude.
        epicenter_lon: Earthquake epicenter longitude.
        station_lat: Station latitude.
        station_lon: Station longitude.
        reference_distance_km: Reference distance for normalization.

    Returns:
        Amplitude scaling factor.
    """
    dist_km = haversine_km(epicenter_lat, epicenter_lon, station_lat, station_lon)
    dist_km = max(dist_km, 10.0)  # floor to avoid singularity near epicenter
    factor = math.sqrt(reference_distance_km / dist_km)
    return max(0.1, min(factor, 1.5))



def compute_directivity_factor(
    epicenter_lat: float,
    epicenter_lon: float,
    station_lat: float,
    station_lon: float,
    strike_deg: float,
    rake_deg: float,
    f_min: float = 0.4,
) -> float:
    """Compute azimuthal directivity factor for tsunami amplitude.

    For thrust/normal faults, tsunami radiation is strongest perpendicular
    to the fault strike (in the dip direction).  This function applies a
    cos^2 radiation pattern with a minimum floor to account for scattered
    and diffracted energy that fills null directions.

    The maximum radiation direction is strike + 90 deg (perpendicular to
    the fault trace, toward the dip direction).  For pure strike-slip
    events (rake ~0 deg or ~180 deg), the pattern still applies based on the
    strike direction, though real strike-slip tsunami directivity is
    more complex.

    This is a simplified parameterization: real tsunami directivity
    depends on fault geometry, slip distribution, and bathymetric
    refraction (see Okal et al. 2014 for full numerical treatment).
    Observed azimuthal amplitude variation is typically a factor of
    2-3 (not the 5-10x predicted by free-space models), because
    bathymetric refraction, edge waves, and finite source extent
    partially fill radiation nulls.

    Args:
        epicenter_lat: Earthquake epicenter latitude.
        epicenter_lon: Earthquake epicenter longitude.
        station_lat: Station latitude.
        station_lon: Station longitude.
        strike_deg: Fault strike in degrees clockwise from north.
        rake_deg: Slip direction (90 deg = pure thrust, 0 deg = left-lateral).
        f_min: Minimum directivity factor (default 0.4), representing
            energy from scattering, diffraction, and bathymetric refraction.

    Returns:
        Directivity factor in [f_min, 1.0].
    """
    bearing = compute_initial_bearing_deg(
        epicenter_lat, epicenter_lon, station_lat, station_lon,
    )

    # Maximum radiation perpendicular to strike (dip direction)
    theta_max = (strike_deg + 90.0) % 360.0
    delta = math.radians(bearing - theta_max)

    # Directivity strength scales with dip-slip component of the
    # slip vector.  Pure thrust (rake=90 deg) -> full directivity;
    # pure strike-slip (rake=0 deg) -> isotropic (no azimuthal variation).
    dip_slip_fraction = abs(math.sin(math.radians(rake_deg)))
    effective_f_min = 1.0 - dip_slip_fraction * (1.0 - f_min)

    return effective_f_min + (1.0 - effective_f_min) * math.cos(delta) ** 2


def compute_propagation_effects(
    distance_km: float,
    period_min: float,
    ocean_depth_m: float = MEAN_OCEAN_DEPTH_M,
    wave_speed_m_s: float = DEEP_OCEAN_SPEED_M_S,
) -> tuple[float, float]:
    """Compute weak-dispersion effects for a tsunami spectral component.

    Tsunamis are weakly dispersive: shorter-period components travel
    slower than longer-period ones.  Over large distances this causes
    the wave train to stretch (dispersion) and the waveform shape to
    evolve (phase accumulation).

    For a component at frequency *f* propagating distance *d*:

    - **Group delay**  deltat = d/c_group(f) - d/c0
      Shorter-period energy arrives later than the non-dispersive
      leading edge.

    - **Residual phase shift**  deltaphi = omega*d*(1/c0 - 1/c_phase(f))
      Constructive/destructive interference shifts differently at
      each station distance, changing the apparent waveform shape.

    Dispersion relation:  omega^2 = gk*tanh(kh)
        Phase velocity:  c_phase = c0*sqrt(tanh(kh)/(kh))
        Group velocity:  c_group = (c_phase/2)*(1 + 2kh/sinh(2kh))
    where c0 = sqrt(gh) and k ~ 2pif/c0 (leading-order wavenumber).

    Physical significance (h = 4000 m):
        T = 5 min, d = 5000 km  ->  deltat ~ 40 min, deltaphi ~ -16 rad
        T = 10 min, d = 5000 km ->  deltat ~ 10 min, deltaphi ~  -2 rad
        T = 60 min, d = 5000 km ->  deltat ~  0 min, deltaphi ~   0 rad

    Args:
        distance_km: Source-to-station great-circle distance.
        period_min: Wave period in minutes.
        ocean_depth_m: Average ocean depth along the path (default 4000 m).
        wave_speed_m_s: Non-dispersive wave speed sqrt(gh).

    Returns:
        (delay_min, phase_rad): Dispersion delay in minutes and
        residual propagation phase shift in radians.
    """
    d_m = distance_km * 1000.0
    if d_m < 1.0:
        return 0.0, 0.0

    f_hz = 1.0 / (period_min * 60.0)
    omega = 2.0 * math.pi * f_hz
    g = 9.81  # gravitational acceleration

    # Solve the exact dispersion relation omega^2 = gk*tanh(kh) for k
    # using Newton-Raphson iteration.  The leading-order approximation
    # k0 = omega/c0 is accurate for kh < 0.2 (periods > 10 min at 4000 m)
    # but has ~8% error at T = 3 min.  A few Newton iterations give
    # the exact wavenumber across the full tsunami band.
    k = omega / wave_speed_m_s  # initial guess (shallow-water limit)
    for _ in range(5):  # converges in 2-3 iterations
        kh = k * ocean_depth_m
        if kh > 50.0:
            # Prevent overflow in tanh/cosh; deep-water limit
            kh = 50.0
            k = kh / ocean_depth_m
            break
        f_k = omega * omega - g * k * math.tanh(kh)
        # df/dk = -g * [tanh(kh) + kh * sech^2(kh)]
        sech2 = 1.0 - math.tanh(kh) ** 2
        df_dk = -g * (math.tanh(kh) + kh * sech2)
        dk = -f_k / df_dk
        k += dk
        if abs(dk) < k * 1e-10:
            break

    kh = k * ocean_depth_m

    if kh < 1e-4:
        # Extremely long wave - no measurable dispersion
        return 0.0, 0.0

    # Phase velocity (exact linear dispersion relation)
    c_phase = wave_speed_m_s * math.sqrt(math.tanh(kh) / kh)

    # Group velocity: c_g = (c_phase/2) * (1 + 2kh/sinh(2kh))
    c_group = (c_phase / 2.0) * (1.0 + 2.0 * kh / math.sinh(2.0 * kh))

    # Group delay relative to non-dispersive leading edge
    delay_sec = d_m / c_group - d_m / wave_speed_m_s
    delay_min = delay_sec / 60.0

    # Residual propagation phase shift
    phase_rad = omega * d_m * (1.0 / wave_speed_m_s - 1.0 / c_phase)

    return delay_min, phase_rad


def compute_coastal_amplification(
    offshore_depth_m: float = 4000.0,
    nearshore_depth_m: float = 20.0,
) -> float:
    """Compute Green's Law coastal amplification factor.

    Green's Law: A_shore / A_deep = (h_deep / h_shore)^(1/4)

    For h_deep=4000 m, h_shore=20 m: factor = (4000/20)^0.25 ~ 3.76.

    This is a first-order approximation. Real amplification depends on
    shelf geometry, resonance, and harbor effects. Typical observed
    amplification factors are 2-10x.

    Args:
        offshore_depth_m: Deep-ocean water depth in meters.
        nearshore_depth_m: Nearshore water depth in meters.

    Returns:
        Amplitude amplification factor.
    """
    ratio = offshore_depth_m / max(nearshore_depth_m, 1.0)
    return float(ratio**0.25)


# -----------------------------------------------------------------------
# Pre-defined station networks (approximate real positions)
# -----------------------------------------------------------------------

# DART stations from DART_PACIFIC_STATION_IDS in ingest/dart.py
# Positions from NDBC station metadata
PACIFIC_DART_STATIONS = [
    StationConfig("21413", 30.51, 152.12, 5800.0, "dart", 60.0),  # NW Pacific
    StationConfig("21418", 38.73, 148.80, 5600.0, "dart", 60.0),  # E of Tohoku
    StationConfig("21419", 44.46, 155.74, 5200.0, "dart", 60.0),  # Kuril Trench
    StationConfig("46404", 45.86, -128.77, 2800.0, "dart", 60.0),  # Cascadia
    StationConfig("46407", 42.70, -128.90, 3300.0, "dart", 60.0),  # Oregon
    StationConfig("46411", 39.34, -127.09, 4300.0, "dart", 60.0),  # N California
]

# CO-OPS stations from COOPS_PACIFIC_STATION_IDS in ingest/coops.py
PACIFIC_COOPS_STATIONS = [
    StationConfig("1612340", 21.31, -157.87, 20.0, "coops", 60.0),  # Honolulu
    StationConfig("1617760", 19.73, -155.06, 15.0, "coops", 60.0),  # Hilo
    StationConfig("1619910", 28.21, -177.36, 10.0, "coops", 60.0),  # Midway
]
