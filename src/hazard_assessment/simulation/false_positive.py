"""False positive signal generators for specificity testing.

Generates meteotsunami and storm surge signals that should NOT trigger
tsunami detection, testing the system's discrimination capabilities.

Meteotsunamis differ from seismic tsunamis:
- Single dominant period (atmospheric forcing resonance)
- Gaussian envelope (not exponential decay)
- No coherent inter-station propagation pattern
- No associated seismic event
- Duration typically 1-3 hours
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def generate_meteotsunami_signal(
    times_hours: NDArray[np.float64],
    onset_hour: float,
    amplitude_m: float = 0.10,
    period_min: float = 20.0,
    duration_hours: float = 2.0,
    seed: int = 42,
) -> NDArray[np.float64]:
    """Generate a synthetic meteotsunami signal.

    Uses a Gaussian envelope (not exponential decay) to model the
    passage of an atmospheric disturbance. Phase noise simulates
    the irregular nature of meteorologically-forced oscillations.

    Typical meteotsunami parameters:
        Amplitude: 0.05-0.30 m at coastal gauges
        Period: 5-30 min (overlaps tsunami band)
        Duration: 1-3 hours

    Args:
        times_hours: Time axis in hours.
        onset_hour: Start time of meteotsunami in hours.
        amplitude_m: Peak amplitude in meters.
        period_min: Dominant period in minutes.
        duration_hours: Duration of the event in hours.
        seed: Random seed for phase noise.

    Returns:
        Meteotsunami signal (zero before onset).
    """
    signal = np.zeros_like(times_hours)
    mask = times_hours >= onset_hour
    if not np.any(mask):
        return signal

    t_min = (times_hours[mask] - onset_hour) * 60.0  # minutes since onset
    sigma_min = duration_hours * 60.0 / 2.0  # Gaussian half-width

    # Gaussian envelope centered at sigma_min
    envelope = np.exp(-((t_min - sigma_min) / sigma_min) ** 2)

    # Phase noise: small random walk in phase
    rng = np.random.default_rng(seed)
    phase_noise = np.cumsum(rng.normal(0, 0.05, len(t_min)))

    omega = 2.0 * np.pi / period_min
    signal[mask] = amplitude_m * envelope * np.sin(omega * t_min + phase_noise)

    return signal


def generate_storm_surge_signal(
    times_hours: NDArray[np.float64],
    onset_hour: float,
    amplitude_m: float = 0.30,
    rise_hours: float = 4.0,
    duration_hours: float = 12.0,
) -> NDArray[np.float64]:
    """Generate a synthetic storm surge signal.

    Storm surges have fundamentally different characteristics from tsunamis:
    - Very long period (hours, not minutes) - outside tsunami band
    - Smooth rise and fall (no oscillation)
    - Should be completely rejected by the 5-120 min bandpass filter

    This tests the system's specificity: a large storm surge should NOT
    produce high ensemble scores.

    Args:
        times_hours: Time axis in hours.
        onset_hour: Start time of surge in hours.
        amplitude_m: Peak surge amplitude in meters.
        rise_hours: Time to reach peak in hours.
        duration_hours: Total duration of surge in hours.

    Returns:
        Storm surge signal (zero before onset).
    """
    signal = np.zeros_like(times_hours)
    mask = times_hours >= onset_hour
    if not np.any(mask):
        return signal

    t_hrs = times_hours[mask] - onset_hour

    # Clamp rise_hours to at most half of duration to prevent overlap
    # between ramp-up and ramp-down phases, and floor it away from zero so the
    # ramp-up divisor cannot produce NaN (the decay path uses the same floor).
    rise_hours = max(min(rise_hours, duration_hours / 2.0), 0.01)

    # Smooth ramp-up (half cosine) then ramp-down
    peak_mask = t_hrs <= rise_hours
    decay_start = duration_hours - rise_hours
    decay_mask = t_hrs >= decay_start
    plateau_mask = ~peak_mask & ~decay_mask

    surge = np.zeros(len(t_hrs))
    # Ramp up: half cosine
    if np.any(peak_mask):
        surge[peak_mask] = amplitude_m * 0.5 * (
            1.0 - np.cos(np.pi * t_hrs[peak_mask] / rise_hours)
        )
    # Plateau
    surge[plateau_mask] = amplitude_m
    # Ramp down: half cosine. Clamp t_decay to the ramp length - beyond
    # duration_hours the cosine argument would exceed pi and the surge
    # would spuriously rise back toward full amplitude; the surge must
    # stay at zero once the event has ended.
    if np.any(decay_mask):
        decay_duration = duration_hours - decay_start
        t_decay = np.minimum(
            t_hrs[decay_mask] - decay_start, max(decay_duration, 0.01)
        )
        surge[decay_mask] = amplitude_m * 0.5 * (
            1.0 + np.cos(np.pi * t_decay / max(decay_duration, 0.01))
        )

    signal[mask] = surge
    return signal
