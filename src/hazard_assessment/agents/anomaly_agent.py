"""Anomaly Agent - computes ensemble anomaly scores from QC-passed records.


Implements:
- Detiding (harmonic analysis over the supplied calibration window)
  + Butterworth bandpass filter
- Anomaly score components (wavelet, BOCPD, Isolation Forest,
  spatial coherence, ensemble fusion, seismic context adjustment)

All outputs are deterministic on replay (locked random seeds).
Component-level scores are individually logged in the reasoning_trace.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.agents.anomaly_detection import (
    BOCPD_HAZARD_LAMBDA,
    W_ML,
    W_STATISTICAL,
    W_STATISTICAL_NO_ML,
    W_THRESHOLD,
    W_THRESHOLD_NO_ML,
    AnomalyScoreComponents,
    IsolationForestModel,
    SeismicEvent,
    SpatialCoherenceResult,
    StationArrival,
    check_spatial_coherence,
    compute_full_anomaly_score,
    compute_wavelet_energy,
    detide_and_filter,
    is_seismically_quiet,
    rayleigh_arrival_suspect,
)
from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.config.settings import ThresholdSettings
from hazard_assessment.schemas.anomaly import (
    AnomalyAssessment,
    ScoreComponents,
    SpatialConfirmation,
)
from hazard_assessment.schemas.envelope import (
    DecisionStep,
    InputRef,
    StepResult,
)

logger = logging.getLogger(__name__)

_MANIFEST = AgentManifest(
    name="anomaly_agent",
    version="1.0.0",
    capabilities=[
        AgentCapability.READ_DATA,
        AgentCapability.WRITE_DATA,
        AgentCapability.WRITE_AUDIT,
        AgentCapability.PRODUCE_KAFKA,
    ],
    description="Computes ensemble anomaly scores from admitted raw observations with QC metadata",
)

# Default detection threshold in meters for DART residuals.
#
# DART 3 cm: matches the onboard DART tsunami detection algorithm trigger
# threshold (Mofjeld, PMEL; Meinig et al. 2005). The BPR triggers event-mode
# transmission when observed sea level deviates from the cubic extrapolation
# by more than ~3 cm. Our post-detiding threshold mirrors this to align
# sensitivity with the hardware trigger. The BPR can detect signals as small
# as 1 cm in 6000 m of water, so 3 cm provides margin above instrument noise.
#
# CO-OPS 15 cm: coastal tide gauges have higher ambient noise from harbor
# seiching, boat wake, and wind-driven setup. The 15 cm threshold is set at
# approximately 3x the typical tidal residual standard deviation (~5 cm) at
# well-exposed CO-OPS stations after harmonic detiding, consistent with the
# QARTOD gross/spike test approach (U.S. IOOS, 2020).
# 0.03 m follows the DART Tsunami Detection Algorithm, but two qualifications
# matter and neither is expressed by this constant. NOAA scopes the value
# regionally: "a reasonable threshold for the North Pacific is 3 cm (or 30 mm)".
# And it is not a constant on the instrument, since the DART II command set lets
# a warning center set it anywhere in the 30 to 90 mm range. Four of the five
# evaluated events are South Pacific or South American, so applying the North
# Pacific figure to them is an extrapolation this system does not calibrate.
DEFAULT_DART_THRESHOLD_M = 0.03

# A screening level for coastal gauges. Unlike the DART figure above it has no
# published source, and this repository neither derives nor calibrates it.
# For scale, NWSI 10-701 Section 3.4 puts the operational levels higher: a
# tsunami Warning needs a forecast or observed height above 1 m, an Advisory a
# forecast of 0.3 to 1.0 m, and products are canceled once height falls below
# 0.3 m and is diminishing. Screening below the product levels is the intent,
# but the particular number is a default rather than a result. It is not inert:
# the synthetic meteotsunami in the physics validation is rejected because its
# 0.10 m injection sits under this line, so the value chosen decides that
# outcome.
DEFAULT_COOPS_THRESHOLD_M = 0.15


class AnomalyAgent(BaseAgent):
    """Anomaly Detection Agent.

    Computes detided residuals, applies bandpass filtering, runs
    amplitude-threshold detection on the filtered residual (with separate
    thresholds for DART and CO-OPS), wavelet/BOCPD statistical detection,
    and optional Isolation Forest ML scoring. Produces an AnomalyAssessment
    envelope.

    State:
        - Per-station baseline wavelet energy (calibrated from
          operator-supplied non-event data)
        - Optional fitted Isolation Forest model
        - Recent seismic events for context adjustment
    """

    def __init__(self) -> None:
        super().__init__(manifest=_MANIFEST)
        # Keyed by (source_type, station_id) so equal station identifiers
        # from different sources cannot share a baseline.
        self._baseline_energies: dict[tuple[str, str], float] = {}
        self._iforest_model: IsolationForestModel | None = None
        self._recent_seismic_events: list[SeismicEvent] = []
        self._bocpd_prior_precision: float = 1.0

    def set_baseline_energy(
        self, station_id: str, energy: float, source_type: str = "dart"
    ) -> None:
        """Set the baseline wavelet energy for a station from non-event data."""
        self._baseline_energies[(source_type, station_id)] = energy

    def calibrate_baseline(
        self,
        station_id: str,
        signal: NDArray[np.float64],
        sampling_interval_sec: float,
        source_type: str = "dart",
    ) -> float:
        """Calibrate baseline wavelet energy from non-event data.

        Args:
            station_id: Station identifier.
            signal: Non-event signal for baseline computation.
            sampling_interval_sec: Sampling interval in seconds.
            source_type: Source the baseline belongs to ("dart" or
                "coops"); must match the ``source_type`` later passed to
                ``process_station_data`` or the lookup misses.

        Returns:
            Computed baseline energy.
        """
        energy = compute_wavelet_energy(signal, sampling_interval_sec)
        # Floor at 1e-10 to keep wavelet scoring enabled for calibrated
        # stations even when baseline energy is near zero.  (The uncalibrated
        # default of 0.0 disables wavelet scoring entirely - see process_station_data.)
        energy = max(energy, 1e-10)
        self._baseline_energies[(source_type, station_id)] = energy
        return energy

    def set_iforest_model(self, model: IsolationForestModel) -> None:
        """Set a pre-fitted Isolation Forest model."""
        self._iforest_model = model

    def update_seismic_events(self, events: list[SeismicEvent]) -> None:
        """Update the recent seismic event list."""
        self._recent_seismic_events = list(events)

    @property
    def has_seismic_context(self) -> bool:
        """Whether Rayleigh timing has at least one admissible seismic event."""
        return bool(self._recent_seismic_events)

    def check_rayleigh_suspect(
        self,
        station_lat: float,
        station_lon: float,
        spike_utc: datetime,
    ) -> bool:
        """Check if any recent seismic event could cause a Rayleigh wave false trigger.

        Iterates over ``_recent_seismic_events`` and returns True if the
        spike timing at this station is consistent with Rayleigh wave
        arrival from any known epicenter (within +/-20% of expected travel
        time and <=3000 km epicentral distance).

        Args:
            station_lat: DART station latitude (degrees).
            station_lon: DART station longitude (degrees).
            spike_utc: Time of the DART pressure excursion (UTC).

        Returns:
            True if the spike timing matches Rayleigh wave arrival from
            any recent seismic event.
        """
        for event in self._recent_seismic_events:
            if rayleigh_arrival_suspect(
                station_lat,
                station_lon,
                event.latitude,
                event.longitude,
                event.origin_time,
                spike_utc,
            ):
                return True
        return False

    def process_station_data(
        self,
        station_id: str,
        times_hours: NDArray[np.float64],
        values: NDArray[np.float64],
        sampling_interval_sec: float,
        source_type: str = "dart",
        fit_times_hours: NDArray[np.float64] | None = None,
        fit_values: NDArray[np.float64] | None = None,
        other_arrivals: list[StationArrival] | None = None,
        origin_lat: float = 0.0,
        origin_lon: float = 0.0,
        payload_sha256: str = "",
        *,
        processing_time: datetime | None = None,
        fsm_monitoring: bool = False,
    ) -> tuple[AnomalyScoreComponents, SpatialCoherenceResult | None]:
        """Run the full anomaly detection pipeline on a station's data.

        Detide and bandpass filter.
        Compute all anomaly score components.

        Args:
            station_id: Station identifier.
            times_hours: Observation times in hours from epoch.
            values: Water level values.
            sampling_interval_sec: Sampling interval in seconds.
            source_type: "dart" or "coops".
            fit_times_hours: Optional calibration window times for the tidal
                fit (the offline validation datasets use 30-day windows).
            fit_values: Optional calibration window values for the tidal fit.
            other_arrivals: Arrivals at other stations for spatial coherence.
            origin_lat: Latitude of this station.
            origin_lon: Longitude of this station.
            payload_sha256: SHA-256 hash of the source payload. Accepted here
                for API symmetry with the QC pipeline; the caller should wrap
                it in an InputRef and pass it to build_assessment() via
                input_refs for provenance tracking.
            processing_time: Fixed time for deterministic output.
            fsm_monitoring: True when the FSM is in MONITOR or higher
                state. Disables the seismic-quiet 1.3x threshold boost
                to avoid suppressing detection of the monitored event.

        Returns:
            Tuple of (AnomalyScoreComponents, SpatialCoherenceResult or None).
        """
        # payload_sha256 is accepted for API parity with the QC pipeline but is
        # not consumed internally. The caller should pass it via input_refs to
        # build_assessment() for provenance tracking.
        del payload_sha256

        now = processing_time or datetime.now(UTC)
        if sampling_interval_sec <= 0:
            raise ValueError(
                f"sampling_interval_sec must be positive, got {sampling_interval_sec}"
            )
        sampling_rate_hz = 1.0 / sampling_interval_sec

        # Detide and filter
        detided_residual, filtered_signal = detide_and_filter(
            times_hours, values, sampling_rate_hz,
            fit_times_hours, fit_values,
        )

        # Determine threshold based on source type
        threshold_m = (
            DEFAULT_DART_THRESHOLD_M if source_type == "dart"
            else DEFAULT_COOPS_THRESHOLD_M
        )

        # Seismic context check - when the FSM is actively monitoring a
        # seismic event (MONITOR+), force not-quiet so the 1.3x threshold
        # boost doesn't suppress detection of the event being tracked.
        seismic_quiet = is_seismically_quiet(
            self._recent_seismic_events, now, fsm_monitoring=fsm_monitoring,
        )

        # Spatial coherence check.
        #
        # Known limitation: arrival_time is processing time (`now`), not the
        # last data sample timestamp (times_hours[-1] plus the epoch offset).
        # No production caller reaches this branch, because pipeline_runner
        # does not pass other_arrivals. Deriving arrival from the sample
        # timestamp is a prerequisite for enabling multi-station coherence,
        # since processing-time arrival introduces a systematic error that
        # grows with pipeline latency.
        spatial_result: SpatialCoherenceResult | None = None
        if other_arrivals:
            origin_arrival = StationArrival(
                station_id=station_id,
                arrival_time=now,
                latitude=origin_lat,
                longitude=origin_lon,
            )
            spatial_result = check_spatial_coherence(origin_arrival, other_arrivals)

        # Get baseline energy for this station under its source-qualified
        # key. Default 0.0 disables wavelet scoring until calibrated -
        # compute_wavelet_score guards baseline_energy <= 0 -> return 0.0.
        baseline_energy = self._baseline_energies.get(
            (source_type, station_id), 0.0
        )

        # Compute all scores
        scores = compute_full_anomaly_score(
            filtered_signal=filtered_signal,
            detided_residual=detided_residual,
            sampling_interval_sec=sampling_interval_sec,
            threshold_m=threshold_m,
            baseline_wavelet_energy=baseline_energy,
            bocpd_prior_precision=self._bocpd_prior_precision,
            iforest_model=self._iforest_model,
            spatial_coherence_result=spatial_result,
            seismic_context_quiet=seismic_quiet,
        )

        # Record what the harmonic detide was actually fit on so the decision
        # trace reports real calibration provenance rather than a fixed claim.
        fit_t = fit_times_hours if fit_times_hours is not None else times_hours
        scores.detide_fit_source = (
            "separate calibration series" if fit_times_hours is not None
            else "event window"
        )
        scores.detide_fit_samples = int(fit_t.size)
        scores.detide_fit_span_minutes = (
            float((np.max(fit_t) - np.min(fit_t)) * 60.0) if fit_t.size > 1 else 0.0
        )

        # Check for Rayleigh wave false-trigger only when both station
        # coordinates and admissible seismic-event geometry are available.
        # Empty seismic context is not negative evidence: preserve None so the
        # handoff records a missing prerequisite instead of FLAG_NOT_RAISED.
        if (
            (origin_lat != 0.0 or origin_lon != 0.0)
            and self.has_seismic_context
            and scores.ensemble_score > 0
        ):
            scores.rayleigh_wave_suspect = self.check_rayleigh_suspect(
                origin_lat, origin_lon, now
            )

        logger.info(
            "Anomaly scores for station %s: threshold=%.4f, wavelet=%.4f, "
            "bocpd=%.4f, statistical=%.4f, ml=%s, spatial=%.4f, "
            "ensemble=%.4f, seismic_quiet=%s, rayleigh_suspect=%s",
            station_id,
            scores.threshold_score,
            scores.wavelet_score,
            scores.bocpd_score,
            scores.statistical_score,
            f"{scores.ml_score:.4f}" if scores.ml_score is not None else "N/A",
            scores.spatial_coherence_score,
            scores.ensemble_score,
            scores.seismic_context_quiet,
            scores.rayleigh_wave_suspect,
        )

        return scores, spatial_result

    def build_assessment(
        self,
        station_ids: list[str],
        scores: AnomalyScoreComponents,
        spatial_result: SpatialCoherenceResult | None = None,
        other_arrivals: list[StationArrival] | None = None,
        input_refs: list[InputRef] | None = None,
        stations_offline: list[str] | None = None,
        coverage_note: str = "",
        *,
        processing_time: datetime | None = None,
    ) -> AnomalyAssessment:
        """Build an AnomalyAssessment envelope from computed scores.

        Args:
            station_ids: Station IDs whose windows were scored for this
                assessment. They populate scored_stations always, and
                triggering_stations only when the ensemble score reaches
                the configured T1 threshold (inclusive).
            scores: Computed anomaly score components.
            spatial_result: Spatial coherence result, if available.
            other_arrivals: Other station arrivals for spatial confirmations.
            input_refs: Provenance references to input data.
            stations_offline: Known offline station IDs.
            coverage_note: Note about station coverage.
            processing_time: Fixed time for deterministic output.

        Returns:
            AnomalyAssessment envelope.
        """
        now = processing_time or datetime.now(UTC)

        # Configured T1 threshold; comparisons are inclusive to match the FSM.
        t1 = ThresholdSettings().t1

        # Build spatial confirmations for the schema
        spatial_confirmations: list[SpatialConfirmation] = []
        if spatial_result and other_arrivals:
            for conf, arrival in zip(
                spatial_result.confirmations, other_arrivals, strict=True
            ):
                delta_min: float | None = None
                if conf.confirmed:
                    delta_min = (conf.actual_delta_sec - conf.expected_travel_sec) / 60.0
                expected_arrival = now + timedelta(seconds=conf.expected_travel_sec)

                spatial_confirmations.append(SpatialConfirmation(
                    station_id=conf.station_id,
                    expected_arrival_utc=expected_arrival,
                    observed_arrival_utc=arrival.arrival_time,
                    confirmed=conf.confirmed,
                    delta_min=delta_min,
                ))

        # Build reasoning trace
        trace_parts = [
            f"threshold_score={scores.threshold_score:.4f}",
            f"wavelet_score={scores.wavelet_score:.4f}",
            f"bocpd_score={scores.bocpd_score:.4f}",
            f"statistical_score={scores.statistical_score:.4f}",
            f"ml_score={'N/A' if scores.ml_score is None else f'{scores.ml_score:.4f}'}",
            f"spatial_coherence={scores.spatial_coherence_score:.4f}",
            f"seismic_quiet={scores.seismic_context_quiet}",
            f"ensemble_score={scores.ensemble_score:.4f}",
        ]
        if scores.ml_score is None:
            trace_parts.append(
                f"ML unavailable: weights renormalized to "
                f"{W_THRESHOLD_NO_ML:.2f}/{W_STATISTICAL_NO_ML:.2f}"
            )

        # Detide evidence reflects the actual fit series when recorded.
        if scores.detide_fit_samples is not None:
            span_h = (scores.detide_fit_span_minutes or 0.0) / 60.0
            detide_evidence = (
                f"harmonic detide fit on {scores.detide_fit_source} "
                f"({span_h:.1f} h, {scores.detide_fit_samples} samples) + "
                "Butterworth 4th-order bandpass 5-120 min"
            )
        else:
            detide_evidence = (
                "harmonic detide (fit provenance not recorded) + "
                "Butterworth 4th-order bandpass 5-120 min"
            )

        # Decision trace steps
        decision_trace = [
            DecisionStep(
                step="Detide and bandpass filter",
                result=StepResult.INFO,
                evidence=detide_evidence,
            ),
            DecisionStep(
                step="Threshold detection",
                result=StepResult.PASS if scores.threshold_score > 0.5
                else StepResult.INFO,
                evidence=f"score={scores.threshold_score:.4f}, "
                         f"seismic_quiet={scores.seismic_context_quiet}",
            ),
            DecisionStep(
                step="Wavelet energy analysis (db4)",
                result=StepResult.PASS if scores.wavelet_score > 0.5
                else StepResult.INFO,
                evidence=f"score={scores.wavelet_score:.4f}",
            ),
            DecisionStep(
                step="BOCPD changepoint detection",
                result=StepResult.PASS if scores.bocpd_score > 0.5
                else StepResult.INFO,
                evidence=f"score={scores.bocpd_score:.4f}, "
                         f"hazard_lambda={BOCPD_HAZARD_LAMBDA:.6f}",
            ),
            DecisionStep(
                step="Ensemble fusion",
                result=StepResult.PASS if scores.ensemble_score >= t1
                else StepResult.INFO,
                evidence=(
                    f"ensemble={scores.ensemble_score:.4f}, "
                    f"weights={W_THRESHOLD:.2f}/{W_STATISTICAL:.2f}/{W_ML:.2f}, "
                    f"T1={t1:.2f} (inclusive)"
                    if scores.ml_score is not None else
                    f"ensemble={scores.ensemble_score:.4f}, "
                    f"weights={W_THRESHOLD_NO_ML:.2f}/{W_STATISTICAL_NO_ML:.2f} "
                    f"(no ML), T1={t1:.2f} (inclusive)"
                ),
            ),
        ]

        return AnomalyAssessment(
            producer="anomaly_agent",
            produced_at_utc=now,
            input_refs=input_refs or [],
            anomaly_score=scores.ensemble_score,
            score_components=ScoreComponents(
                threshold=scores.threshold_score,
                statistical=scores.statistical_score,
                ml=scores.ml_score,
            ),
            triggering_stations=(
                list(station_ids) if scores.ensemble_score >= t1 else []
            ),
            scored_stations=list(station_ids),
            spatial_confirmations=spatial_confirmations,
            seismic_quiet=scores.seismic_context_quiet,
            # The meteotsunami discriminator is not implemented; None
            # states that explicitly instead of a 0.0 placeholder.
            meteotsunami_score=None,
            stations_offline=stations_offline or [],
            filter_degraded=scores.filter_degraded,
            coverage_note=coverage_note,
            reasoning_trace="; ".join(trace_parts),
            decision_trace=decision_trace,
            rayleigh_wave_suspect=scores.rayleigh_wave_suspect,
        )
