"""Scenario Data Interface - unit-source propagation database abstraction.


Implements:
- Abstract data interface for unit-source propagation databases
  (primary + fallback), source selection with fault orientation filtering
  and contiguity enforcement.

All pure functions are deterministic on replay.
"""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UnitSource:
    """A single unit source in the propagation database."""

    source_id: str
    latitude: float
    longitude: float
    depth_km: float
    strike_deg: float  # fault strike in degrees
    dip_deg: float  # fault dip in degrees
    rake_deg: float  # fault rake in degrees
    length_km: float  # along-strike segment length
    width_km: float  # down-dip width
    rigidity_pa: float  # shear modulus in Pa; NCTR uses 4.5e10, 3.5e10 is typical
                        # for shallow (<35 km) subduction megathrust; production
                        # should use zone-specific values
    fault_zone_id: str  # groups sources into named fault zones
    segment_index: int  # sequential index within the fault zone (for contiguity)

    @property
    def area_m2(self) -> float:
        """Rupture area in m^2 (derived from length_km x width_km)."""
        return self.length_km * self.width_km * 1e6


@dataclass(frozen=True)
class CoastalForecastFactors:
    """Precomputed per-unit-source amplitude and travel time at a coastal site.

    Sites should be at offshore forecast points (depth > ~500m) where
    linear superposition holds. Below ~500m, nonlinear shoaling effects
    become significant (cf. Acta Geotechnica 2008, critical zone 400-500m).
    The INUNDATION_DISCLAIMER in ScenarioAssessment communicates this.

    unit_source_peak_m: peak open-ocean amplitude (meters) at this site from
    1m of slip on each unit source. Derived as max|G(t)| from the full
    Green's function waveform. The sum Sigma(slip_j x peak_j) is a conservative
    upper bound on the composite peak - source peaks generally don't coincide
    in time, so sum-of-peaks >= peak-of-sum (triangle inequality of norms,
    applied to the supremum norm with non-negative NNLS weights).

    travel_time_sec: tsunami travel time from each unit source to this site,
    precomputed via ray-tracing or Huygens' principle on bathymetric grids.
    """

    site_id: str
    unit_source_peak_m: dict[str, float]  # source_id -> peak amplitude in meters
    travel_time_sec: dict[str, float]  # source_id -> travel time in seconds


@dataclass
class GreensFunctionSet:
    """Precomputed Green's functions for a set of unit sources at specific DART stations.

    Ordering contract: waveforms[i, :, j] is the response at station_ids[i]
    from unit source source_ids[j].  Callers (build_greens_matrix) depend on
    this - no internal reordering is performed.
    """

    source_ids: list[str]  # ordered source IDs
    station_ids: list[str]  # ordered station IDs
    time_step_sec: float  # sampling interval of the Green's functions
    n_timepoints: int  # number of time samples per source-station pair
    waveforms: NDArray[np.float64]  # shape: (n_stations, n_timepoints, n_sources)


# ---------------------------------------------------------------------------
# Abstract Base Class
# ---------------------------------------------------------------------------


class UnitSourceDatabase(ABC):
    """Abstract interface for accessing unit-source propagation data.

    Supports primary and fallback datasets. Callers instantiate the
    concrete implementation (file-backed, database-backed, etc.) and
    pass it to the ScenarioAgent.
    """

    @abstractmethod
    def get_sources_near(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 500.0,
        max_sources: int = 15,
    ) -> list[UnitSource]:
        """Return unit sources near the given coordinates.

        Returns at most max_sources results, sorted by distance
        (nearest first).  Returns empty list if none found.
        """
        ...

    @abstractmethod
    def get_greens_functions(
        self,
        source_ids: list[str],
        station_ids: list[str],
    ) -> GreensFunctionSet:
        """Return Green's functions for the given sources and stations.

        The returned GreensFunctionSet.waveforms must follow the ordering
        of source_ids (columns) and station_ids (rows) as provided.

        Raises KeyError if any source_id or station_id is not in the database.
        """
        ...

    @abstractmethod
    def get_coastal_forecast_factors(
        self,
        source_ids: list[str],
        site_ids: list[str],
    ) -> dict[str, CoastalForecastFactors]:
        """Return coastal forecast factors for each site.

        Returns: dict keyed by site_id -> CoastalForecastFactors.
        Raises KeyError if any site_id is not in the database.
        """
        ...

    @abstractmethod
    def available(self) -> bool:
        """Check if this database is accessible."""
        ...


# ---------------------------------------------------------------------------
# Concrete: InMemoryUnitSourceDatabase
# ---------------------------------------------------------------------------


class InMemoryUnitSourceDatabase(UnitSourceDatabase):
    """In-memory unit-source database for **testing and development only**.

    WARNING: This implementation stores Green's functions provided by the
    caller (typically random synthetic waveforms in tests).  It does NOT
    contain real physics-based propagation solutions.  Operational deployment
    requires actual Green's functions from the NOAA NCTR Forecast Propagation
    Database or an equivalent ocean model.
    """

    def __init__(self) -> None:
        self._sources: dict[str, UnitSource] = {}
        self._greens: dict[tuple[str, str], NDArray[np.float64]] = {}
        self._coastal_factors: dict[str, CoastalForecastFactors] = {}
        self._time_step_sec: float = 60.0
        self._n_timepoints: int = 60

    def add_source(self, source: UnitSource) -> None:
        """Add a unit source to the database."""
        self._sources[source.source_id] = source

    def set_greens_function(
        self,
        source_id: str,
        station_id: str,
        waveform: NDArray[np.float64],
    ) -> None:
        """Set the Green's function waveform for a source-station pair.

        All waveforms must have the same length.  The first call sets the
        expected length; subsequent calls with a different length raise
        ValueError.
        """
        if self._greens and len(waveform) != self._n_timepoints:
            raise ValueError(
                f"Waveform length {len(waveform)} does not match existing "
                f"n_timepoints={self._n_timepoints}"
            )
        self._greens[(source_id, station_id)] = waveform.copy()
        self._n_timepoints = len(waveform)

    def get_sources_near(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 500.0,
        max_sources: int = 15,
    ) -> list[UnitSource]:
        """Return sources within max_distance_km, sorted by distance."""
        candidates: list[tuple[float, UnitSource]] = []
        for src in self._sources.values():
            dist = haversine_distance_km(latitude, longitude, src.latitude, src.longitude)
            if dist <= max_distance_km:
                candidates.append((dist, src))
        candidates.sort(key=lambda x: x[0])
        return [src for _, src in candidates[:max_sources]]

    def get_greens_functions(
        self,
        source_ids: list[str],
        station_ids: list[str],
    ) -> GreensFunctionSet:
        """Return Green's functions following the requested ordering.

        Raises KeyError if any source_id or station_id is not in the database.
        """
        n_stations = len(station_ids)
        n_sources = len(source_ids)
        waveforms = np.zeros(
            (n_stations, self._n_timepoints, n_sources), dtype=np.float64
        )
        for i, sid in enumerate(station_ids):
            for j, src_id in enumerate(source_ids):
                key = (src_id, sid)
                if key not in self._greens:
                    raise KeyError(
                        f"No Green's function for source={src_id}, station={sid}"
                    )
                waveforms[i, :, j] = self._greens[key]
        return GreensFunctionSet(
            source_ids=list(source_ids),
            station_ids=list(station_ids),
            time_step_sec=self._time_step_sec,
            n_timepoints=self._n_timepoints,
            waveforms=waveforms,
        )

    def add_coastal_factors(self, factors: CoastalForecastFactors) -> None:
        """Add coastal forecast factors for a site."""
        self._coastal_factors[factors.site_id] = factors

    def get_coastal_forecast_factors(
        self,
        source_ids: list[str],
        site_ids: list[str],
    ) -> dict[str, CoastalForecastFactors]:
        """Return coastal forecast factors for each site."""
        result: dict[str, CoastalForecastFactors] = {}
        for sid in site_ids:
            if sid not in self._coastal_factors:
                raise KeyError(f"No coastal forecast factors for site={sid}")
            result[sid] = self._coastal_factors[sid]
        return result

    def available(self) -> bool:
        """In-memory database is always available."""
        return True


# ---------------------------------------------------------------------------
# FallbackUnitSourceDatabase (priority chain)
# ---------------------------------------------------------------------------


class FallbackUnitSourceDatabase(UnitSourceDatabase):
    """Tries primary database first; falls back to secondary on failure.

    Satisfies the acceptance criteria: "Interface supports both primary
    and fallback datasets."

    ALL methods (get_sources_near, get_greens_functions, available)
    delegate with the same fallback logic:
    - Try primary first
    - If primary raises an exception -> try fallback
    - For get_sources_near: also falls back if primary returns empty
      list (primary may not cover this region)
    - Non-empty/successful result from primary -> use without trying fallback

    Note: per-method fallback means sources could come from fallback
    while a subsequent get_greens_functions call tries primary first.
    Primary will KeyError on fallback source IDs, triggering fallback
    for greens too. This is correct but has one wasted attempt.
    """

    def __init__(
        self, primary: UnitSourceDatabase, fallback: UnitSourceDatabase
    ) -> None:
        self._primary = primary
        self._fallback = fallback

    def get_sources_near(
        self,
        latitude: float,
        longitude: float,
        max_distance_km: float = 500.0,
        max_sources: int = 15,
    ) -> list[UnitSource]:
        try:
            result = self._primary.get_sources_near(
                latitude, longitude, max_distance_km, max_sources
            )
            if result:
                return result
            logger.info("Primary database returned empty; trying fallback")
        except Exception:
            logger.warning("Primary database failed; trying fallback", exc_info=True)
        return self._fallback.get_sources_near(
            latitude, longitude, max_distance_km, max_sources
        )

    def get_greens_functions(
        self,
        source_ids: list[str],
        station_ids: list[str],
    ) -> GreensFunctionSet:
        try:
            return self._primary.get_greens_functions(source_ids, station_ids)
        except Exception:
            logger.warning(
                "Primary database get_greens_functions failed; trying fallback",
                exc_info=True,
            )
        return self._fallback.get_greens_functions(source_ids, station_ids)

    def get_coastal_forecast_factors(
        self,
        source_ids: list[str],
        site_ids: list[str],
    ) -> dict[str, CoastalForecastFactors]:
        try:
            return self._primary.get_coastal_forecast_factors(source_ids, site_ids)
        except Exception:
            logger.warning(
                "Primary database get_coastal_forecast_factors failed; trying fallback",
                exc_info=True,
            )
        return self._fallback.get_coastal_forecast_factors(source_ids, site_ids)

    def available(self) -> bool:
        try:
            if self._primary.available():
                return True
        except Exception:
            logger.warning(
                "Primary database available() failed; trying fallback",
                exc_info=True,
            )
        return self._fallback.available()


# ---------------------------------------------------------------------------
# Distance / geometry helpers
# ---------------------------------------------------------------------------


def haversine_distance_km(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> float:
    """Great-circle distance between two points in km.

    Delegates to ``hazard_assessment.geo.haversine_km`` - single implementation.
    """
    from hazard_assessment.geo import haversine_km

    return haversine_km(lat1, lon1, lat2, lon2)


# ---------------------------------------------------------------------------
# Circular strike statistics
# ---------------------------------------------------------------------------


def _circular_mean_strike(strikes_deg: Sequence[float]) -> float:
    """Circular mean of fault strikes handling 180-degree ambiguity.

    Fault strike is axial data (0 = 180 degrees). Uses the doubled-angle
    trick (standard for axial data, cf. Mardia & Jupp 2000):
    1. Normalize all strikes to [0, 180) via modulo 180
    2. Double to [0, 360)
    3. Compute circular mean via atan2(sum_sin, sum_cos)
    4. Halve result back to [0, 180)
    """
    if not strikes_deg:
        return 0.0
    doubled = [math.radians((s % 180) * 2) for s in strikes_deg]
    sum_sin = sum(math.sin(d) for d in doubled)
    sum_cos = sum(math.cos(d) for d in doubled)
    mean_doubled = math.atan2(sum_sin, sum_cos)
    # atan2 returns [-pi, pi]; normalize to [0, 2pi)
    if mean_doubled < 0:
        mean_doubled += 2 * math.pi
    # Halve back to [0, 180)
    return math.degrees(mean_doubled / 2)


def _strike_angular_distance(a_deg: float, b_deg: float) -> float:
    """Angular distance between two strikes (handles 180-degree ambiguity).

    Both strikes are normalized to [0, 180), then the angular distance
    is min(|a-b|, 180-|a-b|).
    """
    a_norm = a_deg % 180
    b_norm = b_deg % 180
    diff = abs(a_norm - b_norm)
    return min(diff, 180 - diff)


# ---------------------------------------------------------------------------
# Source selection logic (pure functions)
# ---------------------------------------------------------------------------


def filter_by_fault_orientation(
    sources: list[UnitSource],
    tolerance_deg: float = 30.0,
) -> list[UnitSource]:
    """Keep sources whose strike is consistent with the group orientation.

    Fault strike is circular (0 = 360 degrees) AND ambiguous by
    180 degrees (strike 10 and 190 describe the same fault plane). This
    function handles both:

    1. Normalize all strikes to [0, 180) via modulo 180
    2. Compute circular mean using doubled-angle trick
    3. Keep sources where angular distance to circular mean <= tolerance
    """
    if not sources:
        return []
    strikes = [s.strike_deg for s in sources]
    mean_strike = _circular_mean_strike(strikes)
    return [
        s
        for s in sources
        if _strike_angular_distance(s.strike_deg, mean_strike) <= tolerance_deg
    ]


def check_contiguity(sources: list[UnitSource]) -> list[list[UnitSource]]:
    """Group sources into contiguous segments within the same fault zone.

    Sources are contiguous if they share the same fault_zone_id and have
    consecutive segment_index values (no gaps).

    Returns: list of contiguous groups, sorted by size (largest first).
    """
    if not sources:
        return []

    # Group by fault zone
    by_zone: dict[str, list[UnitSource]] = defaultdict(list)
    for src in sources:
        by_zone[src.fault_zone_id].append(src)

    all_groups: list[list[UnitSource]] = []
    for zone_sources in by_zone.values():
        # Sort by segment index within each zone
        sorted_sources = sorted(zone_sources, key=lambda s: s.segment_index)
        # Split into contiguous runs
        current_group: list[UnitSource] = [sorted_sources[0]]
        for i in range(1, len(sorted_sources)):
            if sorted_sources[i].segment_index == sorted_sources[i - 1].segment_index + 1:
                current_group.append(sorted_sources[i])
            else:
                all_groups.append(current_group)
                current_group = [sorted_sources[i]]
        all_groups.append(current_group)

    # Sort by size, largest first
    all_groups.sort(key=len, reverse=True)
    return all_groups


def select_unit_sources(
    database: UnitSourceDatabase,
    epicenter_lat: float,
    epicenter_lon: float,
    max_distance_km: float = 500.0,
    min_sources: int = 10,
    max_sources: int = 15,
) -> list[UnitSource]:
    """Select nearest unit sources, filter for fault consistency, enforce contiguity.

    Steps:
    1. Query database for nearest sources within max_distance_km
    2. Filter by fault orientation consistency (circular statistics)
    3. Enforce contiguity: keep only the largest contiguous segment group
    4. Trim to max_sources by keeping sources closest to epicenter
    5. Log warning if result has < min_sources

    Returns: sorted list of UnitSources forming a contiguous fault segment.
             Returns empty list if no sources found within range.
    """
    # Step 1: Query nearby sources (request more than max to allow filtering)
    raw = database.get_sources_near(
        epicenter_lat, epicenter_lon, max_distance_km, max_sources=max_sources * 3
    )
    if not raw:
        return []

    # Step 2: Filter by fault orientation
    oriented = filter_by_fault_orientation(raw)
    if not oriented:
        logger.warning(
            "All %d sources filtered out by fault orientation; "
            "using unfiltered set",
            len(raw),
        )
        oriented = raw

    # Step 3: Enforce contiguity - keep largest contiguous group
    groups = check_contiguity(oriented)
    if not groups:
        return []
    selected = groups[0]

    # Step 4: Trim to max_sources (keep closest to epicenter).
    # Note: distance-based trimming may not preserve strict contiguity
    # for curved faults, but NNLS inversion is unaffected - contiguity
    # is a geological constraint, not a mathematical one.
    if len(selected) > max_sources:
        selected.sort(
            key=lambda s: haversine_distance_km(
                epicenter_lat, epicenter_lon, s.latitude, s.longitude
            )
        )
        selected = selected[:max_sources]

    # Step 5: Warn if fewer than min_sources
    if len(selected) < min_sources:
        logger.warning(
            "Only %d unit sources selected (minimum recommended: %d)",
            len(selected),
            min_sources,
        )

    # Return sorted by segment index for consistent ordering
    selected.sort(key=lambda s: (s.fault_zone_id, s.segment_index))
    return selected
