"""Scenario Agent - runs NNLS inversion over unit-source Green's functions for scenario ranking.


Implements:
- Unit-source data interface integration
- NNLS inversion + seismic-only mode
- Bootstrap uncertainty estimation + ensemble spread
- Coastal amplitude proxy generation
- Wire everything into ScenarioAssessment handoff

Supports two invocation modes:
- DART-constrained: Full NNLS inversion with observed waveforms
- Seismic-only: Magnitude-scaled weights, no waveform fit
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from hazard_assessment.agents.base import AgentCapability, AgentManifest, BaseAgent
from hazard_assessment.agents.scenario_data import (
    CoastalForecastFactors,
    UnitSourceDatabase,
    select_unit_sources,
)
from hazard_assessment.agents.scenario_inversion import (
    SEISMIC_ONLY_LABEL,
    BootstrapConfig,
    BootstrapResult,
    InversionResult,
    SeismicOnlyConfig,
    build_greens_matrix,
    build_observation_vector,
    classify_ensemble_spread,
    compute_coastal_proxies,
    rank_scenarios,
    rank_scenarios_from_bootstrap,
    run_bootstrap,
    run_seismic_only_estimate,
    solve_nnls,
)
from hazard_assessment.schemas.envelope import (
    DecisionStep,
    InputRef,
    StepResult,
)
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)

logger = logging.getLogger(__name__)

_MANIFEST = AgentManifest(
    name="scenario_agent",
    version="1.0.0",
    capabilities=[
        AgentCapability.READ_DATA,
        AgentCapability.WRITE_DATA,
        AgentCapability.WRITE_AUDIT,
        AgentCapability.PRODUCE_KAFKA,
    ],
    description="Runs NNLS inversion over unit-source Green's functions to rank tsunami scenarios",
)


class ScenarioAgent(BaseAgent):
    """Scenario Assessment Agent.

    Performs NNLS inversion using unit-source data to
    produce ranked candidate scenarios. Supports two modes:
    - DART-constrained: Full NNLS inversion with observed waveforms
    - Seismic-only: Magnitude-scaled weights, no waveform fit
    """

    def __init__(self, database: UnitSourceDatabase | None = None) -> None:
        super().__init__(manifest=_MANIFEST)
        self._database: UnitSourceDatabase | None = database

    def set_database(self, database: UnitSourceDatabase) -> None:
        """Set or replace the unit-source database."""
        self._database = database

    def _require_database(self) -> UnitSourceDatabase:
        """Return the database or raise ValueError."""
        if self._database is None:
            raise ValueError(
                "ScenarioAgent requires a UnitSourceDatabase. "
                "Call set_database() or pass database to __init__()."
            )
        return self._database

    def run_seismic_only(
        self,
        magnitude: float,
        epicenter_lat: float,
        epicenter_lon: float,
        region: str,
        *,
        processing_time: datetime | None = None,
    ) -> ScenarioAssessment:
        """Produce a seismic-only preliminary estimate.

        Called by the orchestrator during IDLE->MONITOR transition,
        OUTSIDE the main pipeline.
        """
        database = self._require_database()

        config = SeismicOnlyConfig(
            magnitude=magnitude,
            epicenter_lat=epicenter_lat,
            epicenter_lon=epicenter_lon,
            region=region,
        )

        inversion = run_seismic_only_estimate(database, config)
        ranked = rank_scenarios(inversion)

        return self.build_assessment(
            inversion=inversion,
            constraint_stage=ConstraintStage.SEISMIC_ONLY,
            ranked_scenarios=ranked,
            dart_stations_used=[],
            dart_stations_excluded=[],
            exclusion_reasons={},
            limiting_assumptions=[SEISMIC_ONLY_LABEL],
            processing_time=processing_time,
        )

    def run_dart_constrained(
        self,
        station_waveforms: dict[str, NDArray[np.float64]],
        epicenter_lat: float,
        epicenter_lon: float,
        dart_stations_excluded: list[str] | None = None,
        exclusion_reasons: dict[str, str] | None = None,
        *,
        processing_time: datetime | None = None,
        event_origin_utc: datetime | None = None,
        bootstrap_config: BootstrapConfig | None = None,
        coastal_site_ids: list[str] | None = None,
        tidal_corrections: dict[str, float] | None = None,
    ) -> ScenarioAssessment:
        """Run DART-constrained NNLS inversion.

        Station IDs are derived from station_waveforms.keys() (no
        redundant station_ids parameter to go out of sync).

        Steps:
        1. Validate inputs.
        2. Sort station IDs alphabetically (deterministic ordering).
        3. Select unit sources via select_unit_sources().
        4. Fetch Green's functions; exclude missing stations.
        5. Truncate observed waveforms; exclude short ones.
        6. Build H matrix and d vector, solve NNLS.
        7. Bootstrap station resampling (if >= 2 stations).
        8. Coastal amplitude proxies (if coastal_site_ids provided).
        9. Ensemble spread classification.
        10. Scenario ranking (bootstrap or single inversion).
        11. Determine constraint_stage.
        12. Build and return ScenarioAssessment envelope.
        """
        database = self._require_database()

        if not station_waveforms:
            raise ValueError("No station waveforms provided")

        if coastal_site_ids and event_origin_utc is None:
            raise ValueError(
                "event_origin_utc required when coastal_site_ids is provided"
            )
        if coastal_site_ids and event_origin_utc is not None:
            if event_origin_utc.tzinfo is None:
                raise ValueError(
                    "event_origin_utc must be timezone-aware "
                    "(naive datetimes are rejected)"
                )

        excluded = list(dart_stations_excluded or [])
        # Only carry over reasons for stations actually in the excluded list.
        # Caller may pass stale keys; extra keys would fail the
        # ScenarioAssessment validator that requires reasons == excluded.
        pre_excluded_set = set(excluded)
        reasons = {
            k: v for k, v in (exclusion_reasons or {}).items()
            if k in pre_excluded_set
        }
        limiting_assumptions: list[str] = []

        # Ensure all pre-excluded stations have reasons (required by
        # ScenarioAssessment validator).
        for sid in excluded:
            reasons.setdefault(sid, "Excluded by caller")

        # Step 2: Deterministic station ordering - exclude pre-flagged
        # stations so they are never used in the inversion.
        station_ids = sorted(
            sid for sid in station_waveforms.keys() if sid not in pre_excluded_set
        )

        # Step 3: Select unit sources
        sources = select_unit_sources(
            database, epicenter_lat, epicenter_lon
        )
        if not sources:
            raise ValueError(
                "No unit sources found within range of epicenter"
            )
        source_ids = [s.source_id for s in sources]

        # Step 4: Probe each station for Green's function availability.
        usable_stations: list[str] = []
        for sid in station_ids:
            try:
                database.get_greens_functions(source_ids, [sid])
                usable_stations.append(sid)
            except KeyError:
                excluded.append(sid)
                reasons[sid] = "Station not in propagation database"
                logger.warning("Station %s excluded: not in propagation database", sid)

        if not usable_stations:
            raise ValueError(
                "No usable DART stations after exclusions"
            )

        # Fetch Green's functions for all usable stations at once
        greens = database.get_greens_functions(source_ids, usable_stations)

        # Step 5: Check waveform lengths, exclude short ones
        final_stations: list[str] = []
        for sid in usable_stations:
            if len(station_waveforms[sid]) < greens.n_timepoints:
                excluded.append(sid)
                reasons[sid] = "Insufficient data for inversion window"
                logger.warning(
                    "Station %s excluded: waveform has %d points, need %d",
                    sid,
                    len(station_waveforms[sid]),
                    greens.n_timepoints,
                )
            else:
                final_stations.append(sid)

        if not final_stations:
            raise ValueError(
                "No usable DART stations after waveform length check"
            )

        # Step 6: Build matrices and solve.
        H = build_greens_matrix(greens, final_stations)
        d = build_observation_vector(
            station_waveforms, final_stations, n_timepoints=greens.n_timepoints
        )
        inversion_window_sec = int(greens.n_timepoints * greens.time_step_sec)

        inversion = solve_nnls(
            H, d, source_ids, final_stations, sources, inversion_window_sec
        )

        # Step 7: Bootstrap
        bootstrap_result: BootstrapResult | None = None
        if len(final_stations) >= 2 and bootstrap_config is not None:
            n_timepoints = greens.n_timepoints
            bootstrap_result = run_bootstrap(
                H, d, final_stations, sources, n_timepoints, bootstrap_config
            )
            if len(final_stations) < 5:
                limiting_assumptions.append(
                    "Bootstrap uncertainty may underestimate true uncertainty "
                    "(limited station diversity)"
                )
        elif bootstrap_config is not None and len(final_stations) < 2:
            logger.warning(
                "Bootstrap requested but skipped: requires >= 2 stations, got %d",
                len(final_stations),
            )
            limiting_assumptions.append(
                "Bootstrap requested but skipped "
                f"(requires >= 2 stations, got {len(final_stations)})"
            )

        # Step 8: Coastal proxies
        coastal_proxy_list: list[CoastalProxy] = []
        max_amplitudes: NDArray[np.float64] | None = None
        if coastal_site_ids:
            # Fetch factors, skipping missing sites
            usable_factors: dict[str, CoastalForecastFactors] = {}
            for site_id in coastal_site_ids:
                try:
                    site_factors = database.get_coastal_forecast_factors(
                        source_ids, [site_id]
                    )
                    usable_factors.update(site_factors)
                except KeyError:
                    logger.warning(
                        "Coastal site %s not in database, skipping", site_id
                    )

            missing_sites = [
                s for s in coastal_site_ids if s not in usable_factors
            ]
            if missing_sites:
                limiting_assumptions.append(
                    f"Coastal sites not in database (skipped): "
                    f"{', '.join(missing_sites)}"
                )

            if usable_factors:
                weight_matrix = (
                    bootstrap_result.weight_samples
                    if bootstrap_result is not None
                    else inversion.weights[np.newaxis, :]
                )
                assert event_origin_utc is not None  # guaranteed by line 178 guard
                coastal_proxy_list, max_amplitudes = compute_coastal_proxies(
                    weight_matrix,
                    source_ids,
                    usable_factors,
                    event_origin_utc,
                    tidal_corrections=tidal_corrections,
                )

        # Step 9: Ensemble spread
        if (
            bootstrap_result is not None
            and max_amplitudes is not None
            and len(max_amplitudes) > 1
        ):
            spread = classify_ensemble_spread(
                float(np.percentile(max_amplitudes, 10)),
                float(np.percentile(max_amplitudes, 90)),
            )
        else:
            spread = EnsembleSpread.HIGH
            if bootstrap_result is not None and not coastal_site_ids:
                limiting_assumptions.append(
                    "Ensemble spread defaulted to HIGH "
                    "(no coastal sites for amplitude-based classification)"
                )

        # Step 10: Ranking
        if bootstrap_result is not None:
            ranked = rank_scenarios_from_bootstrap(
                bootstrap_result, sources, H, d
            )
        else:
            ranked = rank_scenarios(inversion)

        # Step 11: Determine constraint stage
        if len(final_stations) >= 2:
            constraint_stage = ConstraintStage.MULTI_STATION
        else:
            constraint_stage = ConstraintStage.DART_CONSTRAINED

        # Bilateral rupture not evaluated (deferred to a future version)
        limiting_assumptions.append("Bilateral rupture not evaluated")

        # Step 12: Build assessment
        return self.build_assessment(
            inversion=inversion,
            constraint_stage=constraint_stage,
            ranked_scenarios=ranked,
            dart_stations_used=final_stations,
            dart_stations_excluded=excluded,
            exclusion_reasons=reasons,
            limiting_assumptions=limiting_assumptions,
            processing_time=processing_time,
            ensemble_spread=spread,
            coastal_proxies=coastal_proxy_list,
            bootstrap_result=bootstrap_result,
        )

    def build_assessment(
        self,
        inversion: InversionResult,
        constraint_stage: ConstraintStage,
        ranked_scenarios: list[RankedScenario],
        dart_stations_used: list[str],
        dart_stations_excluded: list[str],
        exclusion_reasons: dict[str, str],
        limiting_assumptions: list[str],
        input_refs: list[InputRef] | None = None,
        *,
        processing_time: datetime | None = None,
        ensemble_spread: EnsembleSpread = EnsembleSpread.HIGH,
        coastal_proxies: list[CoastalProxy] | None = None,
        bootstrap_result: BootstrapResult | None = None,
    ) -> ScenarioAssessment:
        """Build ScenarioAssessment envelope from inversion results."""
        now = processing_time or datetime.now(UTC)

        # Decision trace
        decision_trace = [
            DecisionStep(
                step="Unit source selection",
                result=StepResult.INFO,
                evidence=f"{len(inversion.source_ids)} sources selected",
            ),
        ]

        if constraint_stage == ConstraintStage.SEISMIC_ONLY:
            decision_trace.append(
                DecisionStep(
                    step="Seismic-only magnitude scaling",
                    result=StepResult.WARN,
                    evidence=f"Mw={inversion.mw_equivalent:.2f}, "
                    f"no DART constraint",
                )
            )
        else:
            decision_trace.append(
                DecisionStep(
                    step="NNLS waveform inversion",
                    result=StepResult.PASS
                    if inversion.waveform_rmse_cm < 5.0
                    else StepResult.WARN,
                    evidence=f"RMSE={inversion.waveform_rmse_cm:.4f} cm, "
                    f"Mw={inversion.mw_equivalent:.2f}, "
                    f"stations={len(dart_stations_used)}",
                )
            )

        if dart_stations_excluded:
            decision_trace.append(
                DecisionStep(
                    step="Station exclusions",
                    result=StepResult.WARN,
                    evidence=f"{len(dart_stations_excluded)} stations excluded: "
                    + ", ".join(
                        f"{sid} ({exclusion_reasons.get(sid, 'unknown')})"
                        for sid in dart_stations_excluded
                    ),
                )
            )

        if bootstrap_result is not None:
            p = bootstrap_result.mw_percentiles
            pct_parts = " ".join(
                f"Mw P{int(k * 100)}={v:.2f}" for k, v in sorted(p.items())
            )
            decision_trace.append(
                DecisionStep(
                    step="Bootstrap uncertainty",
                    result=StepResult.INFO,
                    evidence=(
                        f"{bootstrap_result.n_iterations_completed} iterations, "
                        f"{pct_parts}"
                    ),
                )
            )

        resolved_proxies = coastal_proxies or []
        if resolved_proxies:
            decision_trace.append(
                DecisionStep(
                    step="Coastal amplitude proxies",
                    result=StepResult.INFO,
                    evidence=(
                        f"{len(resolved_proxies)} sites, "
                        f"conservative upper-bound (sum-of-peaks), "
                        f"ensemble_spread={ensemble_spread.value}"
                    ),
                )
            )

        return ScenarioAssessment(
            producer="scenario_agent",
            produced_at_utc=now,
            input_refs=input_refs or [],
            constraint_stage=constraint_stage,
            dart_stations_used=dart_stations_used,
            dart_stations_excluded=dart_stations_excluded,
            exclusion_reasons=exclusion_reasons,
            inversion_window_sec=inversion.inversion_window_sec,
            top_scenarios=ranked_scenarios,
            coastal_proxies=resolved_proxies,
            ensemble_spread=ensemble_spread,
            bilateral_rupture_evaluated=False,
            limiting_assumptions=limiting_assumptions,
            decision_trace=decision_trace,
        )
