"""Multi-frequency tsunami waveform synthesis.

Generates physically motivated tsunami waveforms with multiple spectral
components, replacing the single-frequency sinusoid used in earlier
synthetic evaluations.

Physical basis:
    - Tsunami source spectrum follows approximate f^(-2) rolloff above
      the corner frequency, consistent with omega-squared source models.
    - Dominant period scales with rupture length via Wells & Coppersmith
      (1994); see source.py for the scaling relationship.
    - Waveform envelope: rise-then-decay shape with 15-min rise time
      and frequency-dependent decay timescale.  Observed trans-Pacific
      energy decay times: 15 h (2-6 min periods) to 29 h (60-180 min
      periods), per Rabinovich et al. (2013).  Default decay_time_min
      of 600 min (~10 h) is a conservative mid-range value.
    - N-wave polarity: thrust events are synthesized with a leading
      depression at every station (Tadepalli & Synolakis 1994 for the
      N-wave form).  This is a uniform simplification, not a physical
      result: polarity depends on station azimuth relative to the
      rupture, and the seaward far field normally leads with an
      elevation.  No scoring path reads the sign, but
      test_waveform_physics.py asserts it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.simulation.source import (
    compute_characteristic_amplitude_m,
    compute_dominant_period_min,
)


@dataclass(frozen=True)
class TsunamiSpectralComponent:
    """One frequency component of a synthetic tsunami waveform."""

    period_min: float  # period in minutes
    amplitude_m: float  # amplitude in meters
    phase_rad: float  # phase offset in radians


def generate_tsunami_spectrum(
    magnitude: float,
    n_components: int = 10,
    seed: int = 42,
    leading_depression: bool = False,
) -> list[TsunamiSpectralComponent]:
    """Generate a multi-frequency tsunami spectrum from earthquake magnitude.

    Components are log-spaced from T_dominant/3 to T_dominant*2 with
    amplitudes following the f^(-2) rolloff above the corner frequency.
    The central component receives the full characteristic amplitude.

    Args:
        magnitude: Moment magnitude (Mw).
        n_components: Number of spectral components (default 10).
        seed: Random seed for reproducible phase offsets.
        leading_depression: If True, set the dominant component phase to
            produce a leading trough (N-wave).  Used for thrust fault
            tsunamis where the first arrival is a depression.

    Returns:
        List of TsunamiSpectralComponent with periods, amplitudes, and phases.
    """
    t_dominant = compute_dominant_period_min(magnitude)
    a0 = compute_characteristic_amplitude_m(magnitude)

    rng = np.random.default_rng(seed)

    # Log-spaced periods from T_dominant/3 to T_dominant*2
    t_low = max(t_dominant / 3.0, 3.0)  # floor at 3 min
    t_high = min(t_dominant * 2.0, 120.0)  # cap at 120 min
    periods = np.logspace(np.log10(t_low), np.log10(t_high), n_components)

    f_corner = 1.0 / (t_dominant * 60.0)  # corner frequency in Hz
    center_idx = n_components // 2

    components: list[TsunamiSpectralComponent] = []
    for i, t_min in enumerate(periods):
        f = 1.0 / (t_min * 60.0)  # frequency in Hz

        if i == center_idx:
            # Central component gets full amplitude
            amp = a0
        elif f > f_corner:
            # Above corner frequency: f^(-2) rolloff
            amp = a0 * (f_corner / f) ** 2
        else:
            # Below corner frequency: flat spectrum (omega-squared model).
            #
            # Every sub-corner component takes the full characteristic
            # amplitude, so the synthesized time series sums them and its peak
            # grows with ``n_components``, which is a discretization choice
            # rather than a physical parameter. At Mw 9.1 the peak runs 3.4
            # to 3.8 times the characteristic amplitude at the default 10
            # components and 7 to 11 times at 40, the spread depending on
            # whether ``leading_depression`` is set (thrust sources set it).
            # Compare waveforms only at equal ``n_components``, and do not
            # read the characteristic amplitude as the peak of the result.
            amp = a0

        if leading_depression and (i == center_idx or f <= f_corner):
            # Phase = pi makes sin(omega*t + pi) = -sin(omega*t),
            # producing a leading depression (negative first excursion).
            # Applied to all flat-spectrum components (f <= f_corner) and
            # the central component so the dominant energy starts negative.
            # Higher-frequency rolloff components keep random phases for
            # realistic waveform complexity.
            phase = math.pi
        else:
            phase = rng.uniform(0, 2 * np.pi)

        components.append(
            TsunamiSpectralComponent(
                period_min=float(t_min),
                amplitude_m=float(amp),
                phase_rad=float(phase),
            )
        )

    return components


def synthesize_dart_waveform(
    times_hours: NDArray[np.float64],
    arrival_hour: float,
    spectrum: list[TsunamiSpectralComponent],
    decay_time_min: float = 600.0,
    rise_time_min: float = 15.0,
    component_delays_min: list[float] | None = None,
    propagation_phases_rad: list[float] | None = None,
) -> NDArray[np.float64]:
    """Synthesize a multi-frequency DART tsunami waveform.

    Waveform = sum_i A_i * envelope(t_i) * sin(2*pi*t_i/T_i + phi_i + dphi_i)

    where t_i is time since component *i* arrives (accounting for
    frequency-dependent dispersion delay), dphi_i is a residual
    propagation phase shift, and envelope(t) combines a rise phase
    and a frequency-dependent exponential decay:

        envelope(t) = (1 - exp(-t/rise)) * exp(-t/decay(T))

    The decay timescale increases with period (Rabinovich et al. 2013):
    shorter-period components (2-6 min) decay in ~15 h, while longer-
    period components (60-180 min) persist for ~29 h.  This is modeled
    as: decay(T) = decay_time_min * (1 + T_min/30), giving ~600 min
    for short periods and ~1800 min for 60-min periods.

    The rise time models the gradual build-up of the leading wave
    group over approximately one dominant period.

    When ``component_delays_min`` and ``propagation_phases_rad`` are None
    (the default), all components share the same arrival time and no
    extra phase is added - reproducing the non-dispersive waveform used
    in earlier versions.

    Args:
        times_hours: Time axis in hours.
        arrival_hour: Tsunami arrival time in hours (non-dispersive leading
            edge, i.e. the longest-period component).
        spectrum: List of spectral components.
        decay_time_min: Base exponential decay timescale in minutes (default 600).
        rise_time_min: Leading-edge rise timescale in minutes (default 15).
            Set to 0 to disable the rise phase (pure exponential decay).
        component_delays_min: Per-component group-velocity delays in minutes
            relative to the leading edge.  Shorter-period components arrive
            later due to weak dispersion.  Length must equal ``len(spectrum)``.
        propagation_phases_rad: Per-component residual phase shifts (radians)
            from propagation.  Length must equal ``len(spectrum)``.

    Returns:
        Tsunami waveform array (zero before leading-edge arrival).
    """
    signal = np.zeros_like(times_hours)

    for i, comp in enumerate(spectrum):
        delay_min = (
            component_delays_min[i] if component_delays_min is not None else 0.0
        )
        extra_phase = (
            propagation_phases_rad[i]
            if propagation_phases_rad is not None
            else 0.0
        )

        # Each component has its own effective arrival time
        effective_arrival = arrival_hour + delay_min / 60.0
        mask = times_hours >= effective_arrival
        if not np.any(mask):
            continue

        t_min = (times_hours[mask] - effective_arrival) * 60.0
        omega = 2.0 * np.pi / comp.period_min

        # Frequency-dependent decay: longer-period components persist
        # longer (Rabinovich et al. 2013).  Scale linearly with period.
        comp_decay = decay_time_min * (1.0 + comp.period_min / 30.0)

        # Envelope: rise-then-decay
        decay = np.exp(-t_min / comp_decay)
        if rise_time_min > 0:
            rise = 1.0 - np.exp(-t_min / rise_time_min)
            envelope = rise * decay
        else:
            envelope = decay

        signal[mask] += comp.amplitude_m * envelope * np.sin(
            omega * t_min + comp.phase_rad + extra_phase
        )

    return signal
