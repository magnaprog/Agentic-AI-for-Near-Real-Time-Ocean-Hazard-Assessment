"""Coherent multi-station tsunami event simulation.

Generates a complete multi-station event with:
- Tidal baseline using the same 8 constituents as the detection pipeline
- Multi-frequency tsunami waveform per station
- Distance-dependent arrival times and amplitude decay
- Green's Law coastal amplification for CO-OPS stations
- Gaussian instrumental noise
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.geo import haversine_km
from hazard_assessment.simulation.propagation import (
    StationConfig,
    compute_arrival_time_hours,
    compute_coastal_amplification,
    compute_directivity_factor,
    compute_geometric_spreading_factor,
    compute_propagation_effects,
)
from hazard_assessment.simulation.source import SyntheticEarthquake
from hazard_assessment.simulation.waveform import (
    generate_tsunami_spectrum,
    synthesize_dart_waveform,
)
from hazard_assessment.tidal import TIDAL_FREQUENCIES_RAD_HR

# Default tidal amplitudes for DART deep-ocean BPR (meters).
# Values are representative of mid-ocean bottom-pressure observations
# (Ray, 2013: "Precise comparisons of bottom-pressure and altimetric
# ocean tides", JGR Oceans).  Typical deep-ocean M2 ranges 0.15-0.40 m;
# we use 0.25 m as a mid-range value.
_DART_TIDAL_AMPLITUDES: dict[str, float] = {
    "M2": 0.25,
    "S2": 0.08,
    "N2": 0.04,
    "K1": 0.05,
    "O1": 0.04,
    "P1": 0.02,
    "K2": 0.01,
    "Q1": 0.008,
}

# Coastal CO-OPS gauge tidal amplitudes (meters).
# Conservative (large) values for a moderate-tide Pacific coastal station.
# Real Honolulu M2 is ~0.19 m; these larger amplitudes make detection
# harder, providing a more demanding test for the pipeline.
_COOPS_TIDAL_AMPLITUDES: dict[str, float] = {
    "M2": 0.50,
    "S2": 0.15,
    "N2": 0.08,
    "K1": 0.20,
    "O1": 0.15,
    "P1": 0.07,
    "K2": 0.04,
    "Q1": 0.03,
}


@dataclass
class SimulatedStation:
    """Complete simulation output for one station."""

    config: StationConfig
    times_hours: NDArray[np.float64]
    clean_signal: NDArray[np.float64]  # tidal + noise (no tsunami)
    event_signal: NDArray[np.float64]  # tidal + noise + tsunami
    arrival_hour: float  # tsunami arrival time (hours from signal start)
    tsunami_amplitude_m: float  # peak amplitude at this station
    geometric_spreading: float  # spreading factor applied
    distance_km: float  # distance from epicenter


@dataclass
class SimulatedEvent:
    """Complete coherent multi-station simulation."""

    earthquake: SyntheticEarthquake
    stations: dict[str, SimulatedStation]
    metadata: dict[str, Any] = field(default_factory=dict)


def _generate_tidal_signal(
    times_hours: NDArray[np.float64],
    station_type: str,
    seed: int,
    noise_std_m: float,
) -> NDArray[np.float64]:
    """Generate a tidal signal using the same 8 constituents as the detection pipeline.

    Uses TIDAL_FREQUENCIES_RAD_HR from tidal.py (the shared single source) to
    ensure the synthetic tides are consistent with what the detiding algorithm
    expects.

    Args:
        times_hours: Time axis in hours.
        station_type: "dart" or "coops" (determines tidal amplitudes).
        seed: Random seed for noise.
        noise_std_m: Gaussian noise standard deviation in meters.

    Returns:
        Tidal signal with noise.
    """
    amplitudes = (
        _DART_TIDAL_AMPLITUDES if station_type == "dart" else _COOPS_TIDAL_AMPLITUDES
    )

    rng = np.random.default_rng(seed)
    signal = np.zeros_like(times_hours)

    # Phase offsets - fixed per constituent for reproducibility
    phase_rng = np.random.default_rng(seed + 1000)

    for name, omega_rad_hr in sorted(TIDAL_FREQUENCIES_RAD_HR.items()):
        amp = amplitudes.get(name, 0.0)
        if amp > 0:
            phase = phase_rng.uniform(0, 2 * np.pi)
            signal += amp * np.cos(omega_rad_hr * times_hours + phase)

    # Add Gaussian instrumental noise
    signal += rng.normal(0, noise_std_m, len(times_hours))

    # Add infragravity-wave colored noise (3-30 min periods)
    ig_rms = 0.002 if station_type == "dart" else 0.005
    signal += _generate_infragravity_noise(times_hours, rms_m=ig_rms, seed=seed + 2000)

    # Add instrument drift for DART BPR stations.
    # Real DART sensors show slow thermal drift (Watts & Kontoyiannis 1990)
    # with typical rates of 0.1-1.0 mm/hour.  Model as a smooth polynomial
    # with station-specific rate from random seed.
    if station_type == "dart":
        drift_rng = np.random.default_rng(seed + 3000)
        # Drift rate: 0.1-0.5 mm/hour (typical range)
        drift_rate = drift_rng.uniform(0.1e-3, 0.5e-3)  # m/hour
        # Slight quadratic acceleration (crystal aging is exponential)
        t_norm = (times_hours - times_hours.mean()) / max(
            1.0, times_hours.max() - times_hours.min())
        drift = drift_rate * times_hours + 0.1 * drift_rate * times_hours * t_norm
        signal += drift

    return signal


def _generate_infragravity_noise(
    times_hours: NDArray[np.float64],
    rms_m: float = 0.002,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate colored infragravity-wave noise in the 3-30 min period band.

    Deep-ocean BPRs record infragravity (IG) waves with periods of 3-30
    minutes and amplitudes of 1-10 mm RMS (Webb et al. 1991; Aucan &
    Ardhuin 2013).  These overlap the tsunami detection band and create
    a structured noise floor that white Gaussian noise alone cannot
    represent.

    Implementation: white noise -> FFT -> 1/f amplitude shaping (f^{-2}
    PSD, consistent with observed red-noise IG spectra) in the IG band ->
    zero outside band -> IFFT -> scale to target RMS.

    Args:
        times_hours: Time axis in hours.
        rms_m: Target RMS amplitude in meters (default 2 mm).
        seed: Random seed for reproducibility.

    Returns:
        Colored noise array with energy concentrated in 3-30 min periods.
    """
    n = len(times_hours)
    if n < 2:
        return np.zeros(n)

    dt_hours = float(times_hours[1] - times_hours[0])
    dt_sec = dt_hours * 3600.0

    rng = np.random.default_rng(seed)
    white = rng.normal(0, 1.0, n)

    # FFT and frequency axis
    spectrum = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n, d=dt_sec)  # Hz

    # IG band: periods 3-30 min -> frequencies 1/1800 to 1/180 Hz
    f_low = 1.0 / 1800.0   # 30-min period
    f_high = 1.0 / 180.0   # 3-min period

    # Apply 1/f amplitude shape (f^-2 PSD) within the IG band, zero outside
    mask = (freqs >= f_low) & (freqs <= f_high)
    shaped = np.zeros_like(spectrum)
    shaped[mask] = spectrum[mask] / freqs[mask]

    # IFFT back to time domain
    ig_noise = np.fft.irfft(shaped, n=n)

    # Scale to target RMS
    current_rms = float(np.sqrt(np.mean(ig_noise**2)))
    if current_rms > 0:
        ig_noise *= rms_m / current_rms

    return ig_noise


def generate_coherent_event(
    earthquake: SyntheticEarthquake,
    stations: list[StationConfig],
    calibration_hours: float = 30 * 24,
    event_hours: float = 6.0,
    noise_std_m: float = 0.001,
    seed: int = 42,
) -> SimulatedEvent:
    """Generate a complete coherent multi-station tsunami event.

    Physical model per station:
    1. Compute arrival time: t = haversine(epicenter, station) / c_deep
    2. Compute amplitude decay: A = A_source * sqrt(r_ref / r)
    3. Apply azimuthal directivity based on fault strike/rake
    4. For CO-OPS: apply Green's Law amplification (h_deep/h_shore)^(1/4)
    5. Generate 8-constituent tidal signal + Gaussian + IG noise
    6. Inject multi-frequency tsunami waveform at computed arrival time

    Args:
        earthquake: Earthquake source parameters.
        stations: List of monitoring station configurations.
        calibration_hours: Duration of calibration period (hours).
        event_hours: Duration of event period (hours).
        noise_std_m: Gaussian noise std for DART stations (meters).
            CO-OPS stations use 5x this value (noisier coastal environment).
        seed: Master random seed for reproducibility.

    Returns:
        SimulatedEvent with per-station signals.
    """
    # Thrust events (rake 60-120 deg) are given N-wave polarity, applied
    # uniformly to every station. Real polarity depends on which side of the
    # source the station sits on: a megathrust uplifts the seaward part of the
    # rupture and subsides the landward part, so the seaward far field, where
    # this simulated DART network sits, normally leads with an elevation and
    # only the landward side leads with a depression. The uniform sign is a
    # simplification of the synthetic generator, not a physical result. No
    # scoring path reads the sign: the amplitude detector takes an absolute
    # value, wavelet energy squares its coefficients, and BOCPD is symmetric.
    # The sign is still pinned by
    # tests/simulation/test_waveform_physics.py::TestNWavePolarity, so changing
    # it is a deliberate change, not a free one. See Tadepalli and Synolakis
    # (1994) for the N-wave form.
    is_thrust = 60.0 <= abs(earthquake.rake_deg) <= 120.0

    # Generate multi-frequency tsunami spectrum from magnitude
    spectrum = generate_tsunami_spectrum(
        earthquake.magnitude, seed=seed, leading_depression=is_thrust,
    )

    total_hours = calibration_hours + event_hours
    station_results: dict[str, SimulatedStation] = {}

    for i, stn in enumerate(stations):
        station_seed = seed + i * 100  # deterministic per-station seed
        dt_hours = stn.sampling_interval_sec / 3600.0
        times = np.arange(0, total_hours, dt_hours)

        # Station-specific noise level
        stn_noise = noise_std_m if stn.station_type == "dart" else noise_std_m * 5.0

        # Generate tidal baseline (clean signal)
        clean = _generate_tidal_signal(times, stn.station_type, station_seed, stn_noise)

        # Compute propagation parameters
        dist_km = haversine_km(
            earthquake.latitude,
            earthquake.longitude,
            stn.latitude,
            stn.longitude,
        )

        arrival_hours_from_origin = compute_arrival_time_hours(
            earthquake.latitude,
            earthquake.longitude,
            stn.latitude,
            stn.longitude,
        )

        # Arrival time relative to signal start:
        # calibration period ends at calibration_hours, origin is at calibration_hours
        arrival_hour = calibration_hours + arrival_hours_from_origin

        spreading = compute_geometric_spreading_factor(
            earthquake.latitude,
            earthquake.longitude,
            stn.latitude,
            stn.longitude,
        )

        # Compute station-specific amplitude
        base_amplitude = max(c.amplitude_m for c in spectrum)
        station_amplitude = base_amplitude * spreading

        # Apply azimuthal directivity
        directivity = compute_directivity_factor(
            earthquake.latitude,
            earthquake.longitude,
            stn.latitude,
            stn.longitude,
            earthquake.strike_deg,
            earthquake.rake_deg,
        )
        station_amplitude *= directivity

        # Apply coastal amplification for CO-OPS stations
        if stn.station_type == "coops":
            coastal_factor = compute_coastal_amplification(
                offshore_depth_m=4000.0,
                nearshore_depth_m=stn.depth_m,
            )
            station_amplitude *= coastal_factor

        # Scale the spectrum for this station
        amplitude_ratio = station_amplitude / base_amplitude if base_amplitude > 0 else 0
        scaled_spectrum = [
            replace(c, amplitude_m=c.amplitude_m * amplitude_ratio)
            for c in spectrum
        ]

        # Compute per-component dispersion effects (weak-dispersion model).
        # Short-period components travel slower than long-period ones,
        # arriving later and with accumulated phase shift.  This creates
        # physically distinct waveforms at each station distance.
        delays_min: list[float] = []
        prop_phases: list[float] = []
        for c in spectrum:
            delay, phase = compute_propagation_effects(dist_km, c.period_min)
            delays_min.append(delay)
            prop_phases.append(phase)

        # Generate tsunami waveform with distance-dependent dispersion
        tsunami = synthesize_dart_waveform(
            times,
            arrival_hour,
            scaled_spectrum,
            component_delays_min=delays_min,
            propagation_phases_rad=prop_phases,
        )
        event = clean + tsunami

        station_results[stn.station_id] = SimulatedStation(
            config=stn,
            times_hours=times,
            clean_signal=clean,
            event_signal=event,
            arrival_hour=arrival_hour,
            tsunami_amplitude_m=station_amplitude,
            geometric_spreading=spreading,
            distance_km=dist_km,
        )

    metadata = {
        "calibration_hours": calibration_hours,
        "event_hours": event_hours,
        "noise_std_m": noise_std_m,
        "seed": seed,
        "n_spectral_components": len(spectrum),
        "spectrum": [
            {
                "period_min": c.period_min,
                "amplitude_m": c.amplitude_m,
                "phase_rad": c.phase_rad,
            }
            for c in spectrum
        ],
    }

    return SimulatedEvent(
        earthquake=earthquake,
        stations=station_results,
        metadata=metadata,
    )
