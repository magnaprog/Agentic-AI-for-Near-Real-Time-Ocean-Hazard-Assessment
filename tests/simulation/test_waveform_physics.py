"""Physics unit tests for the simulation module.

Validates that the synthetic signal generators produce physically
correct output consistent with established seismological relationships
and the detection pipeline's constants.
"""

from __future__ import annotations

import numpy as np

from hazard_assessment.simulation.degraded import apply_data_gaps, mark_stations_offline
from hazard_assessment.simulation.false_positive import (
    generate_meteotsunami_signal,
    generate_storm_surge_signal,
)
from hazard_assessment.simulation.propagation import (
    compute_arrival_time_hours,
    compute_coastal_amplification,
    compute_directivity_factor,
    compute_geometric_spreading_factor,
    compute_initial_bearing_deg,
    compute_propagation_effects,
)
from hazard_assessment.simulation.source import (
    CHILE_LIKE,
    MODERATE_PACIFIC,
    TOHOKU_LIKE,
    compute_characteristic_amplitude_m,
    compute_dominant_period_min,
    compute_seismic_moment,
)
from hazard_assessment.simulation.waveform import (
    generate_tsunami_spectrum,
    synthesize_dart_waveform,
)


class TestAmplitudeScaling:
    """Verify Comer (1980) amplitude-magnitude scaling."""

    def test_m91_amplitude_matches_observed(self):
        """Mw 9.1 should produce ~0.10-0.50 m deep-ocean amplitude.

        Comer scaling: log10(A_cm) = 0.75*9.1 - 5.3 = 1.525 -> 33.5 cm.
        Real Tohoku near-field DART records showed 0.15-0.30 m; the regression
        gives a slightly higher value as it represents the characteristic
        amplitude at the reference distance.
        """
        amp = compute_characteristic_amplitude_m(9.1)
        assert 0.05 < amp < 0.50, f"M9.1 amplitude {amp:.4f} m outside expected range"

    def test_m88_amplitude_order_of_magnitude(self):
        """Mw 8.8 should produce ~0.05-0.15 m."""
        amp = compute_characteristic_amplitude_m(8.8)
        assert 0.03 < amp < 0.20

    def test_m72_amplitude_small(self):
        """Mw 7.2 should produce < 0.02 m (near noise floor)."""
        amp = compute_characteristic_amplitude_m(7.2)
        assert amp < 0.02

    def test_amplitude_increases_with_magnitude(self):
        """Larger earthquakes produce larger amplitudes."""
        a7 = compute_characteristic_amplitude_m(7.0)
        a8 = compute_characteristic_amplitude_m(8.0)
        a9 = compute_characteristic_amplitude_m(9.0)
        assert a7 < a8 < a9

    def test_m9_much_larger_than_m7(self):
        """M9 should be roughly 100x larger than M7 (10^(0.75*2) ~ 56)."""
        a7 = compute_characteristic_amplitude_m(7.0)
        a9 = compute_characteristic_amplitude_m(9.0)
        ratio = a9 / a7
        assert 30 < ratio < 200  # log-scale, expect ~56


class TestDominantPeriod:
    """Verify Wells & Coppersmith (1994) period-magnitude relationship."""

    def test_m91_period_long(self):
        """M9.1 -> rupture ~1150 km -> T_dominant clamped to 60 min.

        Wells & Coppersmith: log10(L) = 0.69*9.1 - 3.22 -> L ~1147 km.
        T = 1147000/198 = 5792s ~ 97 min, clamped to upper bound of 60 min.
        """
        t = compute_dominant_period_min(9.1)
        assert t == 60.0, f"M9.1 period {t:.1f} min should be clamped to 60"

    def test_period_increases_with_magnitude(self):
        """Larger earthquakes produce longer-period tsunamis."""
        t7 = compute_dominant_period_min(7.0)
        t8 = compute_dominant_period_min(8.0)
        t9 = compute_dominant_period_min(9.0)
        assert t7 < t8 < t9

    def test_period_clamped_to_tsunami_band(self):
        """Period should be clamped to [5, 60] minutes."""
        t_small = compute_dominant_period_min(6.0)
        t_large = compute_dominant_period_min(9.5)
        assert t_small >= 5.0
        assert t_large <= 60.0


class TestSeismicMoment:
    """Verify Hanks & Kanamori (1979) moment-magnitude relationship."""

    def test_m9_moment_order(self):
        """M9 should produce ~10^22 N*m."""
        m0 = compute_seismic_moment(9.0)
        assert 1e21 < m0 < 1e23


class TestTsunamiSpectrum:
    """Verify multi-frequency spectrum generation."""

    def test_default_10_components(self):
        """Default spectrum has 10 components."""
        spec = generate_tsunami_spectrum(9.0)
        assert len(spec) == 10

    def test_custom_component_count(self):
        """Custom component count is respected."""
        spec = generate_tsunami_spectrum(9.0, n_components=3)
        assert len(spec) == 3

    def test_components_in_tsunami_band(self):
        """All components should have reasonable periods (3-120 min)."""
        for mag in [7.0, 8.0, 9.0]:
            spec = generate_tsunami_spectrum(mag)
            for comp in spec:
                assert 2.0 <= comp.period_min <= 130.0, (
                    f"M{mag} component period {comp.period_min:.1f} min out of range"
                )

    def test_amplitudes_positive(self):
        """All amplitudes should be positive."""
        spec = generate_tsunami_spectrum(9.0)
        for comp in spec:
            assert comp.amplitude_m > 0

    def test_central_component_has_max_amplitude(self):
        """The central component should have the largest amplitude."""
        spec = generate_tsunami_spectrum(9.0)
        center = spec[len(spec) // 2]
        assert center.amplitude_m == max(c.amplitude_m for c in spec)

    def test_deterministic_with_seed(self):
        """Same seed produces identical spectra."""
        spec1 = generate_tsunami_spectrum(8.0, seed=42)
        spec2 = generate_tsunami_spectrum(8.0, seed=42)
        for c1, c2 in zip(spec1, spec2):
            assert c1.period_min == c2.period_min
            assert c1.amplitude_m == c2.amplitude_m
            assert c1.phase_rad == c2.phase_rad

    def test_different_seeds_different_phases(self):
        """Different seeds produce different phase offsets."""
        spec1 = generate_tsunami_spectrum(8.0, seed=42)
        spec2 = generate_tsunami_spectrum(8.0, seed=99)
        phases1 = [c.phase_rad for c in spec1]
        phases2 = [c.phase_rad for c in spec2]
        assert phases1 != phases2


class TestWaveformSynthesis:
    """Verify synthesized waveform properties."""

    def test_zero_before_arrival(self):
        """Waveform should be zero before tsunami arrives."""
        times = np.arange(0, 10, 0.01)
        spec = generate_tsunami_spectrum(9.0)
        wave = synthesize_dart_waveform(times, arrival_hour=5.0, spectrum=spec)
        assert np.allclose(wave[times < 5.0], 0.0)

    def test_nonzero_after_arrival(self):
        """Waveform should be nonzero after tsunami arrives."""
        times = np.arange(0, 10, 0.01)
        spec = generate_tsunami_spectrum(9.0)
        wave = synthesize_dart_waveform(times, arrival_hour=5.0, spectrum=spec)
        assert np.max(np.abs(wave[times >= 5.0])) > 0

    def test_decaying_envelope(self):
        """Waveform amplitude should decrease over time.

        With frequency-dependent decay (tau = 600*(1+T/30) min), long-period
        components persist much longer than short-period ones.  The overall
        envelope peaks a few hours after arrival, then decays.  We check
        that the amplitude at 40+ hours is clearly less than the peak.
        """
        times = np.arange(0, 50, 1.0 / 60)  # 1-min sampling for 50 hours
        spec = generate_tsunami_spectrum(9.0)
        wave = synthesize_dart_waveform(times, arrival_hour=5.0, spectrum=spec)

        # Peak amplitude should occur within the first ~10 hours after arrival
        peak_window = np.abs(wave[(times >= 5.0) & (times < 15.0)])
        # Late window: 40-50 hours after event start (35-45 h after arrival)
        late_window = np.abs(wave[(times >= 40.0) & (times < 50.0)])
        # Peak should be at least 1.5x the late amplitude
        assert np.max(peak_window) > np.max(late_window) * 1.5

    def test_empty_when_arrival_beyond_times(self):
        """Waveform is all zeros if arrival is beyond time range."""
        times = np.arange(0, 5, 0.01)
        spec = generate_tsunami_spectrum(9.0)
        wave = synthesize_dart_waveform(times, arrival_hour=10.0, spectrum=spec)
        assert np.allclose(wave, 0.0)


class TestGeometricSpreading:
    """Verify 2D cylindrical spreading: A proportional to 1/sqrtr."""

    def test_spreading_at_reference_distance(self):
        """Factor should be ~1.0 at reference distance (1000 km)."""
        # Station at roughly 1000 km from epicenter
        factor = compute_geometric_spreading_factor(0.0, 0.0, 0.0, 9.0)
        assert 0.8 < factor < 1.3

    def test_spreading_decreases_with_distance(self):
        """Factor should decrease with distance."""
        f_near = compute_geometric_spreading_factor(0.0, 0.0, 0.0, 5.0)  # ~556 km
        f_far = compute_geometric_spreading_factor(0.0, 0.0, 0.0, 45.0)  # ~5000 km
        assert f_near > f_far

    def test_1_over_sqrt_r_scaling(self):
        """Verify 1/sqrtr scaling: 4x distance -> 0.5x amplitude."""
        # Use stations at ~1000 km and ~4000 km
        f1 = compute_geometric_spreading_factor(0.0, 0.0, 0.0, 9.0)  # ~1000 km
        f2 = compute_geometric_spreading_factor(0.0, 0.0, 0.0, 36.0)  # ~4000 km
        ratio = f2 / f1
        assert 0.35 < ratio < 0.65  # should be ~0.5

    def test_clamped_upper_bound(self):
        """Factor should not exceed 1.5 even very close to epicenter."""
        f = compute_geometric_spreading_factor(0.0, 0.0, 0.01, 0.01)
        assert f <= 1.5

    def test_clamped_lower_bound(self):
        """Factor should not go below 0.1 even very far away."""
        f = compute_geometric_spreading_factor(0.0, 0.0, 80.0, 0.0)
        assert f >= 0.1


class TestCoastalAmplification:
    """Verify Green's Law: A_shore/A_deep = (h_deep/h_shore)^(1/4)."""

    def test_standard_amplification(self):
        """4000m deep to 20m shallow -> 3.76x amplification."""
        factor = compute_coastal_amplification(4000.0, 20.0)
        expected = (4000.0 / 20.0) ** 0.25
        assert abs(factor - expected) < 0.01
        assert abs(factor - 3.76) < 0.05

    def test_amplification_increases_with_shallower_water(self):
        """Shallower nearshore -> more amplification."""
        f_20m = compute_coastal_amplification(4000.0, 20.0)
        f_5m = compute_coastal_amplification(4000.0, 5.0)
        assert f_5m > f_20m

    def test_no_amplification_at_same_depth(self):
        """Same depth -> factor ~1.0."""
        f = compute_coastal_amplification(100.0, 100.0)
        assert abs(f - 1.0) < 0.01

    def test_floor_prevents_division_by_zero(self):
        """Zero depth should not cause an error."""
        f = compute_coastal_amplification(4000.0, 0.0)
        assert f > 0 and np.isfinite(f)


class TestArrivalTimes:
    """Verify tsunami arrival time computation."""

    def test_arrival_time_positive(self):
        """Arrival time should always be positive."""
        t = compute_arrival_time_hours(0.0, 0.0, 10.0, 0.0)
        assert t > 0

    def test_farther_station_later_arrival(self):
        """Farther station should have later arrival."""
        t_near = compute_arrival_time_hours(0.0, 0.0, 5.0, 0.0)
        t_far = compute_arrival_time_hours(0.0, 0.0, 30.0, 0.0)
        assert t_far > t_near

    def test_consistent_with_detection_pipeline(self):
        """Arrival times should be consistent with anomaly_detection.py wave speed.

        Both anomaly_detection.py and simulation use 198 m/s (0.198 km/s).
        """
        from hazard_assessment.agents.anomaly_detection import (
            DEEP_OCEAN_WAVE_SPEED_KM_S,
            haversine_km,
        )

        dist_km = haversine_km(0.0, 0.0, 10.0, 0.0)  # ~1111 km
        our_hours = compute_arrival_time_hours(0.0, 0.0, 10.0, 0.0)
        pipeline_sec = dist_km / DEEP_OCEAN_WAVE_SPEED_KM_S
        our_sec = our_hours * 3600

        # Should agree within 5%
        assert abs(our_sec - pipeline_sec) / pipeline_sec < 0.05

    def test_tohoku_to_21418_reasonable(self):
        """Tohoku to station 21418 (~600 km) -> ~50 min arrival."""
        t = compute_arrival_time_hours(
            38.297, 142.373,  # Tohoku epicenter
            38.73, 148.80,  # Station 21418
        )
        t_min = t * 60
        assert 30 < t_min < 120  # ~50 min expected


class TestFalsePositiveSignals:
    """Verify false positive signal generators."""

    def test_meteotsunami_zero_before_onset(self):
        """Meteotsunami signal should be zero before onset."""
        times = np.arange(0, 10, 0.01)
        signal = generate_meteotsunami_signal(times, onset_hour=5.0)
        assert np.allclose(signal[times < 5.0], 0.0)

    def test_meteotsunami_has_energy_after_onset(self):
        """Meteotsunami should have nonzero energy after onset."""
        times = np.arange(0, 10, 0.01)
        signal = generate_meteotsunami_signal(times, onset_hour=5.0)
        assert np.max(np.abs(signal[times >= 5.0])) > 0

    def test_meteotsunami_gaussian_envelope(self):
        """Meteotsunami amplitude should peak before end (Gaussian shape).

        The peak of |envelope * sin(...)| may not align exactly with the
        envelope center due to phase noise, but it should not be at the
        very end of the signal.
        """
        times = np.arange(0, 12, 1.0 / 60)
        signal = generate_meteotsunami_signal(
            times, onset_hour=5.0, duration_hours=2.0
        )
        after = signal[times >= 5.0]
        abs_after = np.abs(after)
        peak_idx = np.argmax(abs_after)
        total = len(abs_after)
        # Peak should be in the first 80% (not at the tail end)
        assert peak_idx < 0.8 * total

    def test_storm_surge_zero_before_onset(self):
        """Storm surge should be zero before onset."""
        times = np.arange(0, 20, 0.01)
        signal = generate_storm_surge_signal(times, onset_hour=5.0)
        assert np.allclose(signal[times < 5.0], 0.0)

    def test_storm_surge_stays_down_after_event_ends(self):
        """After onset + duration the surge must remain at (near) zero;
        the cosine ramp-down must not wrap past pi and rise again."""
        times = np.arange(0, 40, 0.01)
        signal = generate_storm_surge_signal(
            times, onset_hour=2.0, duration_hours=12.0, amplitude_m=0.3
        )
        after_end = signal[times > 2.0 + 12.0]
        assert np.all(np.abs(after_end) < 1e-9)

    def test_storm_surge_slow_period(self):
        """Storm surge should have no energy in tsunami band (5-120 min).

        Check via FFT that spectral energy is concentrated at very low
        frequencies (periods >> 120 min).
        """
        dt_hours = 1.0 / 60  # 1-min sampling
        times = np.arange(0, 30, dt_hours)
        signal = generate_storm_surge_signal(
            times, onset_hour=5.0, duration_hours=12.0
        )

        # FFT
        fft_vals = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(len(signal), d=dt_hours * 3600)  # Hz

        # Tsunami band: 1/5400 to 1/300 Hz
        tsunami_mask = (freqs >= 1.0 / 5400) & (freqs <= 1.0 / 300)
        low_freq_mask = freqs < 1.0 / 5400

        power_tsunami = np.sum(np.abs(fft_vals[tsunami_mask]) ** 2)
        power_low = np.sum(np.abs(fft_vals[low_freq_mask]) ** 2)

        # Most energy should be below tsunami band
        assert power_low > power_tsunami * 5


class TestDegradation:
    """Verify data gap and network outage utilities."""

    def test_data_gap_removes_samples(self):
        """Data gap should reduce the number of samples."""
        times = np.arange(0, 10, 0.01)
        signal = np.ones_like(times)
        t_gap, s_gap = apply_data_gaps(times, signal, gap_start_hour=3.0, gap_duration_hours=2.0)
        assert len(t_gap) < len(times)
        # No samples in gap period
        assert not np.any((t_gap >= 3.0) & (t_gap < 5.0))

    def test_data_gap_preserves_outside(self):
        """Data outside the gap should be preserved."""
        times = np.arange(0, 10, 0.01)
        signal = times * 2  # simple slope
        t_gap, s_gap = apply_data_gaps(times, signal, gap_start_hour=3.0, gap_duration_hours=2.0)
        # Values before gap
        before_mask = t_gap < 3.0
        np.testing.assert_allclose(s_gap[before_mask], t_gap[before_mask] * 2)

    def test_mark_stations_offline(self):
        """Offline stations should be separated correctly."""
        from hazard_assessment.simulation.propagation import PACIFIC_DART_STATIONS

        online, offline = mark_stations_offline(
            PACIFIC_DART_STATIONS[:4],
            {"21413", "21419"},
        )
        assert len(online) == 2
        assert len(offline) == 2
        assert "21413" in offline
        assert "21419" in offline
        online_ids = {s.station_id for s in online}
        assert "21413" not in online_ids
        assert "21419" not in online_ids


class TestPreDefinedEvents:
    """Verify pre-defined earthquake sources have correct parameters."""

    def test_tohoku_like_magnitude(self):
        assert TOHOKU_LIKE.magnitude == 9.1

    def test_tohoku_like_location(self):
        assert abs(TOHOKU_LIKE.latitude - 38.297) < 0.01
        assert abs(TOHOKU_LIKE.longitude - 142.373) < 0.01

    def test_chile_like_magnitude(self):
        assert CHILE_LIKE.magnitude == 8.8

    def test_chile_like_southern_hemisphere(self):
        assert CHILE_LIKE.latitude < 0  # southern hemisphere

    def test_moderate_pacific_magnitude(self):
        assert MODERATE_PACIFIC.magnitude == 7.2


class TestPropagationEffects:
    """Verify weak-dispersion propagation model.

    Dispersion relation: c_phase = c0*sqrt(tanh(kh)/(kh))
    Group velocity:      c_group = (c_phase/2)*(1 + 2kh/sinh(2kh))

    Key physical expectations:
    - Long-period components (T >> h/c0) are non-dispersive (delay ~ 0).
    - Short-period components travel slower -> positive group delay.
    - Effects scale linearly with distance.
    """

    def test_no_dispersion_for_very_long_period(self):
        """T=60 min at 5000 km -> negligible dispersion."""
        delay, phase = compute_propagation_effects(5000, 60.0)
        assert abs(delay) < 1.0  # less than 1 minute delay
        assert abs(phase) < 0.5  # less than 0.5 rad phase shift

    def test_significant_dispersion_for_short_period(self):
        """T=5 min at 5000 km -> tens of minutes delay.

        At h=4000 m, kh ~ 0.42 for T=5 min.  Group velocity drops to
        ~92% of c0, giving ~35 min delay at 5000 km.
        """
        delay, phase = compute_propagation_effects(5000, 5.0)
        assert delay > 10.0  # should be ~35 min
        assert delay < 60.0  # but not unreasonably large

    def test_dispersion_increases_with_distance(self):
        """Delay should scale approximately linearly with distance."""
        delay_near, _ = compute_propagation_effects(500, 10.0)
        delay_far, _ = compute_propagation_effects(5000, 10.0)
        # Linearity: 10x distance -> ~10x delay
        assert delay_far > delay_near * 5

    def test_shorter_period_disperses_more(self):
        """Shorter periods have larger group-velocity delay."""
        delay_long, _ = compute_propagation_effects(3000, 30.0)
        delay_short, _ = compute_propagation_effects(3000, 5.0)
        assert delay_short > delay_long

    def test_phase_shift_is_negative(self):
        """Phase shift should be negative (c_phase < c0 -> phase lag)."""
        _, phase = compute_propagation_effects(3000, 5.0)
        assert phase < 0

    def test_zero_distance_zero_effects(self):
        """Zero distance -> no propagation effects."""
        delay, phase = compute_propagation_effects(0.0, 10.0)
        assert delay == 0.0
        assert phase == 0.0

    def test_moderate_period_moderate_dispersion(self):
        """T=10 min at 5000 km -> ~9 min delay (intermediate case)."""
        delay, _ = compute_propagation_effects(5000, 10.0)
        assert 3.0 < delay < 20.0


class TestDispersiveWaveform:
    """Verify that dispersion creates inter-station waveform variation."""

    def test_near_and_far_waveforms_differ(self):
        """Waveforms at different distances should have different shapes.

        Near-field: small delays -> components arrive together -> peaked.
        Far-field: large delays -> wave train stretches -> different shape.

        Uses Mw 7.5 (T_dominant ~ 7 min) which produces shorter-period
        spectral components where dispersion is strongest.  For Mw 9.0
        (T_dominant = 60 min), the shortest component is ~20 min and
        dispersion is weaker, though still measurably non-zero.
        """
        times = np.arange(0, 15, 1.0 / 60)  # 1-min sampling, 15 hours
        spec = generate_tsunami_spectrum(7.5)

        # Near-field (500 km): negligible dispersion
        delays_near = [compute_propagation_effects(500, c.period_min)[0] for c in spec]
        phases_near = [compute_propagation_effects(500, c.period_min)[1] for c in spec]

        # Far-field (7000 km): significant dispersion
        delays_far = [compute_propagation_effects(7000, c.period_min)[0] for c in spec]
        phases_far = [compute_propagation_effects(7000, c.period_min)[1] for c in spec]

        wave_near = synthesize_dart_waveform(
            times, 2.0, spec,
            component_delays_min=delays_near,
            propagation_phases_rad=phases_near,
        )
        wave_far = synthesize_dart_waveform(
            times, 2.0, spec,
            component_delays_min=delays_far,
            propagation_phases_rad=phases_far,
        )

        # Both should be nonzero after arrival
        assert np.max(np.abs(wave_near[times >= 2.0])) > 0
        assert np.max(np.abs(wave_far[times >= 2.0])) > 0

        # Waveforms should differ (correlation well below 1)
        after = times >= 2.0
        corr = np.corrcoef(wave_near[after], wave_far[after])[0, 1]
        assert corr < 0.95, f"Near/far waveforms too similar: r={corr:.4f}"

    def test_backward_compatible_without_delays(self):
        """Without delays, new signature reproduces old behavior."""
        times = np.arange(0, 10, 0.01)
        spec = generate_tsunami_spectrum(9.0)
        wave_default = synthesize_dart_waveform(times, 5.0, spec)
        wave_explicit_none = synthesize_dart_waveform(
            times, 5.0, spec,
            component_delays_min=None,
            propagation_phases_rad=None,
        )
        np.testing.assert_array_equal(wave_default, wave_explicit_none)

    def test_zero_delays_match_no_delays(self):
        """Explicit zero delays should match the no-delay case."""
        times = np.arange(0, 10, 0.01)
        spec = generate_tsunami_spectrum(9.0)
        wave_none = synthesize_dart_waveform(times, 5.0, spec)
        wave_zeros = synthesize_dart_waveform(
            times, 5.0, spec,
            component_delays_min=[0.0] * len(spec),
            propagation_phases_rad=[0.0] * len(spec),
        )
        np.testing.assert_allclose(wave_none, wave_zeros, atol=1e-15)

    def test_far_field_wave_train_is_longer(self):
        """Dispersion should stretch the wave train at far-field stations.

        With large delays, the short-period energy arrives later,
        extending the duration of significant wave action.
        """
        times = np.arange(0, 15, 1.0 / 60)
        spec = generate_tsunami_spectrum(9.0)

        # No dispersion: all components arrive together
        wave_nodispersion = synthesize_dart_waveform(times, 2.0, spec)

        # Strong dispersion: short-period components delayed
        delays = [compute_propagation_effects(6000, c.period_min)[0] for c in spec]
        phases = [compute_propagation_effects(6000, c.period_min)[1] for c in spec]
        wave_dispersed = synthesize_dart_waveform(
            times, 2.0, spec,
            component_delays_min=delays,
            propagation_phases_rad=phases,
        )

        # Find the last time with significant amplitude (>10% of peak)
        after = times >= 2.0
        threshold_nd = 0.10 * np.max(np.abs(wave_nodispersion[after]))
        threshold_d = 0.10 * np.max(np.abs(wave_dispersed[after]))

        last_active_nd = times[after][np.abs(wave_nodispersion[after]) > threshold_nd][-1]
        last_active_d = times[after][np.abs(wave_dispersed[after]) > threshold_d][-1]

        # Dispersed wave train should extend later
        assert last_active_d > last_active_nd, (
            f"Dispersed wave should last longer: "
            f"no-dispersion ends at {last_active_nd:.1f}h, "
            f"dispersed ends at {last_active_d:.1f}h"
        )


class TestDirectivity:
    """Verify azimuthal directivity radiation pattern."""

    def test_perpendicular_to_strike_is_maximum(self):
        """Station perpendicular to strike gets factor ~1.0."""
        # Strike = 0 deg (north), so perpendicular = 90 deg (east)
        # Station due east of epicenter
        f = compute_directivity_factor(0.0, 0.0, 0.0, 10.0, strike_deg=0.0, rake_deg=90.0)
        assert f > 0.9

    def test_parallel_to_strike_is_minimum(self):
        """Station parallel to strike gets factor ~f_min."""
        # Strike = 0 deg (north), so parallel = 0 deg (north)
        # Station due north of epicenter
        f = compute_directivity_factor(0.0, 0.0, 10.0, 0.0, strike_deg=0.0, rake_deg=90.0)
        assert f < 0.5

    def test_factor_bounded(self):
        """Factor should always be in [f_min, 1.0]."""
        for bearing_lon in range(-180, 181, 30):
            f = compute_directivity_factor(
                0.0, 0.0, 0.0, float(bearing_lon),
                strike_deg=196.0, rake_deg=85.0,
            )
            assert 0.39 <= f <= 1.01, f"Factor {f} out of bounds at lon={bearing_lon}"

    def test_strike_slip_is_near_isotropic(self):
        """Pure strike-slip (rake=0) should produce near-isotropic radiation."""
        f_east = compute_directivity_factor(0.0, 0.0, 0.0, 10.0, strike_deg=0.0, rake_deg=0.0)
        f_north = compute_directivity_factor(0.0, 0.0, 10.0, 0.0, strike_deg=0.0, rake_deg=0.0)
        # Both should be ~1.0 (isotropic)
        assert abs(f_east - 1.0) < 0.01
        assert abs(f_north - 1.0) < 0.01

    def test_oblique_slip_intermediate(self):
        """Oblique slip (rake=45) should have weaker directivity than pure thrust."""
        f_thrust = compute_directivity_factor(0.0, 0.0, 10.0, 0.0, strike_deg=0.0, rake_deg=90.0)
        f_oblique = compute_directivity_factor(0.0, 0.0, 10.0, 0.0, strike_deg=0.0, rake_deg=45.0)
        # Oblique directivity at the null should be weaker (closer to 1.0)
        assert f_oblique > f_thrust


class TestBearing:
    """Verify initial bearing computation."""

    def test_due_north(self):
        """Bearing from (0,0) to (10,0) should be ~0 deg (north)."""
        b = compute_initial_bearing_deg(0.0, 0.0, 10.0, 0.0)
        assert abs(b) < 1.0 or abs(b - 360.0) < 1.0

    def test_due_east(self):
        """Bearing from (0,0) to (0,10) should be ~90 deg (east)."""
        b = compute_initial_bearing_deg(0.0, 0.0, 0.0, 10.0)
        assert abs(b - 90.0) < 1.0

    def test_due_south(self):
        """Bearing from (0,0) to (-10,0) should be ~180 deg (south)."""
        b = compute_initial_bearing_deg(0.0, 0.0, -10.0, 0.0)
        assert abs(b - 180.0) < 1.0

    def test_due_west(self):
        """Bearing from (0,0) to (0,-10) should be ~270 deg (west)."""
        b = compute_initial_bearing_deg(0.0, 0.0, 0.0, -10.0)
        assert abs(b - 270.0) < 1.0


class TestNWavePolarity:
    """Verify N-wave leading depression for thrust faults."""

    def test_thrust_produces_leading_depression(self):
        """With leading_depression=True, first significant excursion is negative."""
        times = np.arange(0, 10, 1.0 / 120)  # 30-sec sampling
        spec = generate_tsunami_spectrum(9.0, leading_depression=True)
        wave = synthesize_dart_waveform(times, 2.0, spec)

        # Find first significant excursion after arrival (after rise time)
        after_rise = (times >= 2.3) & (times < 3.0)
        segment = wave[after_rise]
        # The dominant component has phase=pi -> sin(wt+pi) = -sin(wt)
        # So the first peak of the dominant component should be negative
        peak_idx = np.argmax(np.abs(segment))
        assert segment[peak_idx] < 0, (
            f"Expected leading depression, got positive first peak: {segment[peak_idx]:.6f}"
        )

    def test_default_no_forced_polarity(self):
        """Without leading_depression, central phase is random (seed-dependent)."""
        spec = generate_tsunami_spectrum(9.0, seed=42, leading_depression=False)
        center = spec[len(spec) // 2]
        # Phase should not be exactly pi
        assert abs(center.phase_rad - np.pi) > 0.01


class TestEnvelopeRiseTime:
    """Verify rise-then-decay envelope model."""

    def test_near_zero_at_arrival(self):
        """Waveform amplitude should be near zero at exact arrival time."""
        times = np.arange(0, 10, 1.0 / 60)
        spec = generate_tsunami_spectrum(9.0, n_components=1)
        wave = synthesize_dart_waveform(times, 5.0, spec, rise_time_min=15.0)

        # At t=0 after arrival, rise factor = 1 - exp(0) = 0
        arrival_idx = np.searchsorted(times, 5.0)
        assert abs(wave[arrival_idx]) < 1e-10

    def test_peak_after_arrival(self):
        """Peak amplitude should occur after arrival, not at arrival."""
        times = np.arange(0, 12, 1.0 / 60)
        spec = generate_tsunami_spectrum(9.0, n_components=1)
        wave = synthesize_dart_waveform(times, 5.0, spec, rise_time_min=15.0)

        after = times >= 5.0
        abs_wave = np.abs(wave[after])
        peak_idx = np.argmax(abs_wave)
        peak_time_min = peak_idx  # 1-min sampling -> index = minutes
        # Peak should be at least a few minutes after arrival
        assert peak_time_min > 3, f"Peak at {peak_time_min} min, expected > 3 min"

    def test_eventual_decay(self):
        """Signal should still decay to near-zero eventually."""
        times = np.arange(0, 30, 1.0 / 60)
        spec = generate_tsunami_spectrum(9.0, n_components=1)
        wave = synthesize_dart_waveform(
            times, 5.0, spec, rise_time_min=15.0, decay_time_min=180.0,
        )

        # Compare amplitude at 1 hour vs 20 hours after arrival
        early = np.max(np.abs(wave[(times >= 5.5) & (times < 6.5)]))
        late = np.max(np.abs(wave[(times >= 25.0) & (times < 30.0)]))
        assert late < early * 0.5, "Signal should decay by at least 50% over 20 hours"

    def test_zero_rise_time_is_pure_decay(self):
        """rise_time_min=0 should reproduce pure exponential decay."""
        times = np.arange(0, 10, 1.0 / 60)
        spec = generate_tsunami_spectrum(9.0, n_components=1, seed=99)
        wave = synthesize_dart_waveform(times, 5.0, spec, rise_time_min=0.0)

        # At arrival, amplitude should be non-zero (no rise phase)
        arrival_idx = np.searchsorted(times, 5.0)
        # First sample after arrival should have some amplitude
        assert abs(wave[arrival_idx + 1]) > 0
