"""Simplified tsunami simulation for pipeline verification.

Provides multi-frequency tsunami waveform synthesis, distance-dependent
propagation, coastal amplification, and false-positive signal generation
for end-to-end software verification of the ocean hazard assessment
pipeline.  Not a substitute for physics-based tsunami models (e.g.,
MOST, GeoClaw); see Simplifications below.

Physical references:
- Comer (1980): non-dispersive amplitude slope (0.75*Mw); intercept (-5.3)
  is project-calibrated to approximate observed DART amplitudes at 1000 km
- Wells & Coppersmith (1994): rupture length-magnitude scaling
- Omega-squared source model: f^(-2) spectral rolloff above corner frequency
- Green's Law: coastal shoaling amplification (valid above ~50 m depth;
  applied down to 10 m as a rough approximation)

Simplifications (see source.py and propagation.py docstrings for details):
- Constant ocean depth (h=4000 m); real bathymetry not modeled
- cos^2 directivity parameterization (ad-hoc, not from a published model)
- 1/sqrt(r) flat-Earth spreading (approximate at transoceanic distances)
- Dominant period heuristic (rupture_length/wave_speed) overestimates for Mw>8.5
"""

from hazard_assessment.simulation.degraded import (
    apply_data_gaps,
    mark_stations_offline,
)
from hazard_assessment.simulation.false_positive import (
    generate_meteotsunami_signal,
    generate_storm_surge_signal,
)
from hazard_assessment.simulation.propagation import (
    PACIFIC_COOPS_STATIONS,
    PACIFIC_DART_STATIONS,
    StationConfig,
    compute_arrival_time_hours,
    compute_coastal_amplification,
    compute_geometric_spreading_factor,
    compute_propagation_effects,
)
from hazard_assessment.simulation.scenario import (
    SimulatedEvent,
    SimulatedStation,
    generate_coherent_event,
)
from hazard_assessment.simulation.source import (
    ALEUTIAN_SCENARIO,
    CHILE_LIKE,
    MODERATE_ALEUTIAN,
    MODERATE_PACIFIC,
    TOHOKU_LIKE,
    SyntheticEarthquake,
    compute_characteristic_amplitude_m,
    compute_dominant_period_min,
)
from hazard_assessment.simulation.waveform import (
    TsunamiSpectralComponent,
    generate_tsunami_spectrum,
    synthesize_dart_waveform,
)

__all__ = [
    "ALEUTIAN_SCENARIO",
    "CHILE_LIKE",
    "MODERATE_ALEUTIAN",
    "MODERATE_PACIFIC",
    "PACIFIC_COOPS_STATIONS",
    "PACIFIC_DART_STATIONS",
    "SimulatedEvent",
    "SimulatedStation",
    "StationConfig",
    "SyntheticEarthquake",
    "TOHOKU_LIKE",
    "TsunamiSpectralComponent",
    "apply_data_gaps",
    "compute_arrival_time_hours",
    "compute_characteristic_amplitude_m",
    "compute_coastal_amplification",
    "compute_dominant_period_min",
    "compute_geometric_spreading_factor",
    "compute_propagation_effects",
    "generate_coherent_event",
    "generate_meteotsunami_signal",
    "generate_storm_surge_signal",
    "generate_tsunami_spectrum",
    "mark_stations_offline",
    "synthesize_dart_waveform",
]
