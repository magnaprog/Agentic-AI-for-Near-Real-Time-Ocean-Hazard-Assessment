"""Earthquake source model for synthetic tsunami simulation.

Provides magnitude-dependent scaling for tsunami amplitude and dominant
period based on established seismological relationships.

Physical references:
    Comer, R. P. (1980). Tsunami height and earthquake magnitude:
    Theoretical basis of an empirical relation. GRL, 7, 445-448.
        -> The 0.75 slope in log10(A_cm) ~ 0.75*Mw - 5.3 corresponds to
          Comer's NON-DISPERSIVE limit where A proportional to M0^(1/2).  In the fully
          dispersive limit, A proportional to M0 giving slope 1.5.  Abe's (1979)
          empirical tsunami magnitude scale yields an intermediate slope
          of ~1.0, consistent with partial dispersion over transoceanic
          paths.  The 0.75 slope therefore represents a LOWER BOUND on
          the true magnitude dependence.  This is a conservative choice
          for detection testing: if the pipeline detects a tsunami at
          the 0.75-slope amplitude, it will also detect the real (larger)
          signal.  The intercept (-5.3) is calibrated to approximate
          observed DART deep-ocean amplitudes at the 1000 km reference
          distance.

    Abe, K. (1979). Size of great earthquakes of 1837-1974 inferred
    from tsunami data. JGR, 84(B4), 1561-1568.
        -> Empirical tsunami magnitude Mt ~ Mw, implying log10(A) proportional to 1.0*Mw.

    Wells, D. L. & Coppersmith, K. J. (1994). New empirical relationships
    among magnitude, rupture length, rupture width, rupture area, and surface
    displacement. BSSA, 84(4), 974-1002.
        -> All-fault-type SRL (Table 2A): log10(L_km) = 0.69*Mw - 3.22
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

# Canonical wave speed constant: c = sqrt(g*h) = sqrt(9.81*4000) ~ 198 m/s.
from hazard_assessment.geo import DEEP_OCEAN_SPEED_M_S as _DEEP_OCEAN_SPEED_M_S


@dataclass(frozen=True)
class SyntheticEarthquake:
    """Parametric earthquake source for coherent multi-station simulation."""

    event_id: str
    magnitude: float  # Mw
    latitude: float  # epicenter degrees
    longitude: float  # epicenter degrees
    depth_km: float  # focal depth
    origin_time: datetime  # UTC, timezone-aware
    region: str  # descriptive label
    strike_deg: float = 196.0  # fault strike (Tohoku default)
    dip_deg: float = 14.0  # fault dip
    rake_deg: float = 85.0  # slip direction


def compute_characteristic_amplitude_m(magnitude: float) -> float:
    """Compute expected deep-ocean DART amplitude from earthquake magnitude.

    Uses Comer (1980) non-dispersive scaling where far-field tsunami
    amplitude A proportional to M0^(1/2), giving:
        log10(A_cm) ~ 0.75 * Mw - 5.3

    The 0.75 slope is a lower bound (see module docstring); the
    empirical Abe (1979) slope is ~1.0.  This conservative choice
    means simulated amplitudes are smaller than real observations,
    providing a harder test for the detection pipeline.

    The result is the characteristic amplitude at the 1000 km
    reference distance, before geometric spreading.

    Examples (from formula, before geometric spreading):
        Mw 9.1 -> ~0.34 m
        Mw 8.8 -> ~0.20 m
        Mw 7.5 -> ~0.02 m
        Mw 7.0 -> ~0.009 m

    Args:
        magnitude: Moment magnitude (Mw).

    Returns:
        Characteristic deep-ocean amplitude in meters (at reference
        distance, before geometric spreading).
    """
    log_a_cm = 0.75 * magnitude - 5.3
    return float((10.0**log_a_cm) / 100.0)  # cm -> m


def compute_dominant_period_min(magnitude: float) -> float:
    """Compute dominant tsunami period from earthquake magnitude.

    Derives rupture length from Wells & Coppersmith (1994) all-fault-type
    surface rupture length (SRL) regression (Table 2A):
        log10(L_km) = 0.69 * Mw - 3.22

    Note: this regression was developed from crustal earthquakes
    predominantly below Mw 8; extrapolation to giant subduction
    megathrust events (Mw > 8.5) systematically overestimates rupture
    length (e.g., predicts ~1150 km for Tohoku 2011 vs. ~450 km observed
    main-slip zone). The 60-minute clamp mitigates downstream impact.

    Then approximates the dominant period as the time for a tsunami to
    traverse the rupture length at deep-ocean speed:
        T_dominant = L / c_deep

    This is a first-order heuristic, not a physical law.  True tsunami
    dominant period depends on the source time function (moment-rate
    history) and propagation dispersion.  Observed Tohoku 2011 DART
    records show dominant periods of ~10-30 min, whereas this formula
    yields ~96 min (clamped to 60).  The clamp and multi-frequency
    spectrum generation (waveform.py) limit the practical impact.

    Examples (from formula):
        Mw 9.1 -> L ~ 1146 km -> T ~ 96 min (clamped to 60)
        Mw 8.0 -> L ~ 200 km  -> T ~ 17 min
        Mw 7.0 -> L ~ 41 km   -> T ~ 3.4 min (clamped to 5)

    Result is clamped to [5.0, 60.0] minutes (observable tsunami band).

    Args:
        magnitude: Moment magnitude (Mw).

    Returns:
        Dominant period in minutes, clamped to [5, 60].
    """
    log_l_km = 0.69 * magnitude - 3.22
    rupture_length_m = (10.0**log_l_km) * 1000.0
    t_dominant_sec = rupture_length_m / _DEEP_OCEAN_SPEED_M_S
    t_dominant_min = t_dominant_sec / 60.0
    return float(max(5.0, min(t_dominant_min, 60.0)))


def compute_seismic_moment(magnitude: float) -> float:
    """Compute seismic moment M0 from moment magnitude.

    Hanks & Kanamori (1979): M0 = 10^(1.5*Mw + 9.1) in N*m,
    using the IASPEI (2005) standard constant.

    Args:
        magnitude: Moment magnitude (Mw).

    Returns:
        Seismic moment in N*m.
    """
    return float(10.0 ** (1.5 * magnitude + 9.1))


# -----------------------------------------------------------------------
# Pre-defined earthquake sources based on real events
# -----------------------------------------------------------------------

TOHOKU_LIKE = SyntheticEarthquake(
    event_id="synth_tohoku",
    magnitude=9.1,
    latitude=38.297,
    longitude=142.373,
    depth_km=29.0,
    origin_time=datetime(2011, 3, 11, 5, 46, 24, tzinfo=UTC),
    region="tohoku",
    # Focal mechanism: converged USGS W-phase CMT (Duputel et al. 2011).
    # Published W-phase solutions range strike 195-202, dip 12-14,
    # rake 85-92.5.  GCMT: strike 203, dip 10, rake 88.
    strike_deg=196.0,
    dip_deg=14.0,
    rake_deg=85.0,
)

CHILE_LIKE = SyntheticEarthquake(
    event_id="synth_chile",
    magnitude=8.8,
    latitude=-35.846,
    longitude=-72.719,
    depth_km=35.0,
    origin_time=datetime(2010, 2, 27, 6, 34, 11, tzinfo=UTC),
    region="maule",
    strike_deg=16.0,
    dip_deg=18.0,
    rake_deg=104.0,
)

MODERATE_PACIFIC = SyntheticEarthquake(
    event_id="synth_moderate",
    magnitude=7.2,
    latitude=38.0,
    longitude=142.0,
    depth_km=25.0,
    origin_time=datetime(2026, 1, 15, 12, 0, 0, tzinfo=UTC),
    region="japan",
)

# Aleutian Islands - central Aleutian arc (Andreanof Islands, near Adak).
# The 1957 Mw 8.6 (Andreanof Islands) and 1965 Mw 8.7 (Rat Islands)
# events ruptured adjacent segments of this arc.  Epicenter is roughly
# equidistant from the NW Pacific and NE Pacific DART clusters,
# producing a distinctly different arrival-time pattern from Tohoku.
ALEUTIAN_SCENARIO = SyntheticEarthquake(
    event_id="synth_aleutian",
    magnitude=8.5,
    latitude=52.5,
    longitude=-173.0,
    depth_km=25.0,
    origin_time=datetime(2026, 1, 15, 3, 0, 0, tzinfo=UTC),
    region="aleutian",
    strike_deg=250.0,  # Aleutian arc trend ~ENE
    dip_deg=15.0,      # shallow-dipping subduction
    rake_deg=90.0,     # pure thrust
)

MODERATE_ALEUTIAN = SyntheticEarthquake(
    event_id="synth_aleutian_moderate",
    magnitude=7.0,
    latitude=52.0,
    longitude=-172.0,
    depth_km=30.0,
    origin_time=datetime(2026, 2, 10, 8, 0, 0, tzinfo=UTC),
    region="aleutian",
    strike_deg=250.0,  # ENE-trending Aleutian arc
    dip_deg=15.0,      # shallow subduction
    rake_deg=90.0,     # pure thrust
)
