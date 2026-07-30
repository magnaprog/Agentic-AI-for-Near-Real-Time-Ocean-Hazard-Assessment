"""Pure construction of OceanEvidenceAssessment from worker-owned facts.

The pipeline worker collects one :class:`StationAttemptResult` per
source-qualified station during a checkpoint, then hands the full set,
plus event and FSM context, to
:func:`build_ocean_evidence_assessment`. Construction is pure: no I/O,
no clock reads (production time is an argument), and every scientific
status is computed here from typed facts through the frozen condition
registry, mirroring the schema validators exactly.

Construction failures raise :class:`AssessmentConstructionError` (or a
pydantic ``ValidationError``). The caller treats any such failure as a
disclosed assessment gap: deterministic
science already ran, the failure is audited, and no model dispatch can
occur for the checkpoint.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import pairwise
from statistics import fmean, median
from typing import TYPE_CHECKING, Literal, TypeGuard
from uuid import UUID

from hazard_assessment.agents.anomaly_detection import (
    BANDPASS_HIGH_HZ,
    BANDPASS_LOW_HZ,
    W_ML,
    W_STATISTICAL,
    W_STATISTICAL_NO_ML,
    W_THRESHOLD,
    W_THRESHOLD_NO_ML,
)
from hazard_assessment.agents.qc_checks import N_RUNNABLE_CHECKS
from hazard_assessment.schemas.envelope import DataSource, InputRef
from hazard_assessment.schemas.ocean_evidence import (
    AnalysisCapability,
    AnalysisExecution,
    AnalysisQuality,
    AnalysisStatus,
    AnalysisStatusSet,
    CadenceCondition,
    CalibrationStatus,
    CheckpointSource,
    ConfounderApplicability,
    ConfounderCheck,
    ConfounderPrerequisiteStatus,
    ConfounderResultValue,
    CurrentRecordAdmissionStatus,
    DartDataMode,
    DetectorComponents,
    DetectorConfig,
    EventSeismicContext,
    FilteringCondition,
    FsmState,
    MixedProductCondition,
    ObservationBounds,
    OceanEvidenceAssessment,
    ProvenanceStatus,
    ProvenanceSummary,
    QCAggregateCondition,
    QCExecutionStatus,
    QCFlagCounts,
    RetainedRecordRef,
    RetainedWindowQC,
    SeismicContextClass,
    SeismicRevisionRef,
    SourceValidationStatus,
    StationAssessmentEntry,
    StationLimitation,
    StationManifestCondition,
    StationScope,
    StationScoringStatus,
    ThresholdEvaluation,
    ThresholdResultValue,
    derive_evaluation_status,
    derive_highest_tier,
    derive_pipeline_outcome,
    derive_qc_aggregate_condition,
    derive_source_rollup,
    derive_source_time_bounds,
    derive_threshold_result_value,
    registry_lookup,
)
from hazard_assessment.schemas.ocean_evidence_hashing import canonical_sha256
from hazard_assessment.workers.station_buffer import RetainedSample

if TYPE_CHECKING:
    from hazard_assessment.agents.anomaly_detection import AnomalyScoreComponents
    from hazard_assessment.orchestrator.states import EventContext

# Detector identity recorded in assessments. Matches the AnomalyAgent
# metadata version (agents/anomaly_agent.py); a conformance test asserts
# parity so the two cannot drift silently.
DETECTOR_VERSION = "1.0.0"

# A retained-window cadence is IRREGULAR when the largest inter-sample
# gap exceeds this multiple of the median cadence. Multiplicative, so
# 15-second DART event-mode and 6-minute CO-OPS windows use the same
# relative criterion.
CADENCE_IRREGULAR_FACTOR = 3.0

# Cap on envelope input_refs, aligned with the worker's per-event
# provenance cap (_MAX_PROVENANCE_PER_EVENT) and the escalation packet
# read cap. Beyond this the provenance summary discloses references_capped.
MAX_ENVELOPE_INPUT_REFS = 1000

# Producer identity for the assessment envelope.
ASSESSMENT_PRODUCER = "pipeline_worker"

# Confounder check name for the Rayleigh-wave arrival heuristic
# (agents/anomaly_detection.py::rayleigh_arrival_suspect).
RAYLEIGH_CONFOUNDER_NAME = "rayleigh_wave"

# Frozen v1 adequacy rule for a DART pre-event detide baseline:
# method-specific and versioned, with no universal row count. A separate
# calibration series spanning at least 72 hours resolves the principal
# diurnal and semidiurnal constituents the harmonic fit uses; deployed
# calibration CSVs are 30-day quiet-period series (workers/calibration.py)
# and clear this comfortably, while the 6-hour rolling event window never
# can. Changing this value changes assessment content and requires the
# usual results/figures/paper review.
CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES = 3.0 * 24.0 * 60.0

_SHA256_HEX = set("0123456789abcdef")

# QARTOD flag integer -> severity rank (higher is worse), frozen to the
# same order derive_qc_aggregate_condition uses: FAIL > MISSING >
# SUSPECT > NOT_EVALUATED > PASS (schemas/qc.py QartodFlag values).
# NOT_APPLICABLE (0, the local extension marking checks that do not
# apply to a station type) ranks below the initial worst rank, so it is
# recognized but can never become a record's decisive flag. This
# matches the schema, whose QCFlagCounts has no not-applicable bucket:
# a check that does not apply is not evidence about the record.
_QARTOD_SEVERITY: dict[int, int] = {4: 5, 9: 4, 3: 3, 2: 2, 1: 1, 0: 0}


class AssessmentConstructionError(ValueError):
    """A checkpoint's facts cannot form a valid assessment.

    Raised for unbound seismic identity, inconsistent attempt sets, or
    malformed per-record facts. The worker records the checkpoint as an
    assessment gap and continues.
    """


@dataclass(frozen=True)
class StationAttemptResult:
    """Deterministic facts for one source-qualified station attempt.

    One instance exists per (source, station_id) that either appeared in
    the current buffer batch (even if every record was rejected) or holds
    a nonempty retained window at scoring time. ``retained_samples`` is
    the exact window snapshot presented to the minimum-data check and
    scorer, so QC aggregation and record
    references cover precisely the scored evidence.
    """

    source: str  # "dart" or "coops"
    station_id: str
    scoring_status: StationScoringStatus
    calibration_status: CalibrationStatus
    n_records_attempted: int = 0
    n_records_admitted: int = 0
    retained_samples: tuple[RetainedSample, ...] = ()
    scores: AnomalyScoreComponents | None = None
    failure_reason: str = ""
    # DART window event-mode flag at scoring time (newest accepted
    # observation). Ignored for CO-OPS.
    dart_window_event_mode: bool = False
    calibration_sha256: str = ""
    # Whether the Rayleigh-wave check had its inputs (station coordinates
    # and seismic context) available at scoring time.
    rayleigh_inputs_available: bool = False


def _epoch_to_utc(epoch_sec: float) -> datetime:
    return datetime.fromtimestamp(epoch_sec, tz=UTC)


def _is_sha256(value: str | None) -> TypeGuard[str]:
    return (
        value is not None
        and len(value) == 64
        and all(c in _SHA256_HEX for c in value)
    )


def classify_calibration_status(
    *,
    source: str,
    scores: AnomalyScoreComponents | None,
    calibration_span_minutes: float | None,
) -> CalibrationStatus:
    """Classify one station's calibration state (frozen rule v1).

    CO-OPS scoring has no separate-baseline requirement in this system
    (the worker never loads CO-OPS calibration; the scorer detides on the
    event window by design), so CO-OPS is always NOT_REQUIRED.

    For DART, a successful score carries the scorer's own detide
    provenance (anomaly_agent.py sets ``detide_fit_source`` on every
    successful run), which is authoritative for what the method actually
    used: a separate calibration series is ADEQUATE_PRE_EVENT_BASELINE at
    or above :data:`CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES` and
    LIMITED_PRE_EVENT_BASELINE below it; an event-window fit is
    EVENT_WINDOW_FALLBACK. An empty ``detide_fit_source`` is unreachable
    for a successful score and is classified UNAVAILABLE so unknown
    provenance fails toward PREVENTS_EVALUATION rather than toward a
    baseline claim.

    Without a score, ``calibration_span_minutes`` describes the loaded
    calibration series (None when none is loaded): the same span rule
    applies, and no loaded series is UNAVAILABLE. ERROR is reserved for a
    tracked calibration load or apply failure, which the current
    CalibrationManager does not report (it logs and skips), so this
    function never returns it.
    """
    if source != DataSource.DART.value:
        return CalibrationStatus.NOT_REQUIRED
    if scores is not None:
        if scores.detide_fit_source == "separate calibration series":
            span = scores.detide_fit_span_minutes or 0.0
            if span >= CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES:
                return CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE
            return CalibrationStatus.LIMITED_PRE_EVENT_BASELINE
        if scores.detide_fit_source == "event window":
            return CalibrationStatus.EVENT_WINDOW_FALLBACK
        return CalibrationStatus.UNAVAILABLE
    if calibration_span_minutes is not None:
        if calibration_span_minutes >= CALIBRATION_ADEQUATE_MIN_SPAN_MINUTES:
            return CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE
        return CalibrationStatus.LIMITED_PRE_EVENT_BASELINE
    return CalibrationStatus.UNAVAILABLE


def _decisive_flag_int(qc_flags: tuple[tuple[str, int], ...]) -> int | None:
    """Worst QARTOD flag over one record's checks, or None when unflagged."""
    worst: int | None = None
    worst_rank = 0
    for check_name, flag_int in qc_flags:
        rank = _QARTOD_SEVERITY.get(flag_int)
        if rank is None:
            raise AssessmentConstructionError(
                f"Unknown QARTOD flag integer {flag_int} on check "
                f"{check_name!r}; cannot rank record severity"
            )
        if rank > worst_rank:
            worst_rank = rank
            worst = flag_int
    return worst


def _build_retained_window_qc(
    samples: Sequence[RetainedSample],
    bounds: ObservationBounds,
) -> RetainedWindowQC:
    """Aggregate per-record QC over the exact retained window.

    Records without attached QC count as unusable and carry no decisive
    flag: absence of an auditable QC verdict is never treated as an
    implicit pass.
    """
    counts = {1: 0, 2: 0, 3: 0, 4: 0, 9: 0}
    n_flagged = 0
    n_usable = 0
    n_unevaluated_checks = 0
    confidences: list[float] = []
    hashes: set[str] = set()

    for sample in samples:
        if _is_sha256(sample.payload_hash):
            hashes.add(sample.payload_hash)
        qc = sample.qc
        if qc is None:
            continue
        if qc.usable:
            n_usable += 1
        confidences.append(qc.confidence)
        # Count only the checks the system runs. qc.flags carries every
        # QARTODFlags field, including latency (excluded from the count
        # because it mirrors timing) and the two reserved fields no
        # producer sets, so len(qc.flags) would add three phantom
        # unevaluated checks to every record.
        n_unevaluated_checks += N_RUNNABLE_CHECKS - qc.n_checks_evaluated
        decisive = _decisive_flag_int(qc.flags)
        if decisive is not None:
            counts[decisive] += 1
            n_flagged += 1

    n_records = len(samples)
    if n_flagged == n_records:
        execution = QCExecutionStatus.SUCCEEDED
    elif n_flagged > 0:
        execution = QCExecutionStatus.PARTIAL
    else:
        execution = QCExecutionStatus.NOT_RUN

    flag_counts = QCFlagCounts(
        n_pass=counts[1],
        n_not_evaluated=counts[2],
        n_suspect=counts[3],
        n_fail=counts[4],
        n_missing=counts[9],
    )
    return RetainedWindowQC(
        execution_status=execution,
        aggregate_condition=derive_qc_aggregate_condition(flag_counts),
        observation_bounds=bounds,
        n_records=n_records,
        flag_counts=flag_counts,
        n_usable=n_usable,
        n_unusable=n_records - n_usable,
        n_unevaluated_checks=n_unevaluated_checks,
        confidence_min=min(confidences) if confidences else None,
        confidence_mean=fmean(confidences) if confidences else None,
        confidence_max=max(confidences) if confidences else None,
        record_sha256s=sorted(hashes),
    )


def _cadence(
    samples: Sequence[RetainedSample],
) -> tuple[float | None, CadenceCondition]:
    """Median cadence and regularity over the retained window.

    Timestamps are unique and sorted by StationWindow construction, so
    every diff is strictly positive.
    """
    if len(samples) < 2:
        return None, CadenceCondition.UNKNOWN
    diffs = [
        b.epoch_sec - a.epoch_sec for a, b in pairwise(samples)
    ]
    med = float(median(diffs))
    if med <= 0.0:
        raise AssessmentConstructionError(
            "Non-positive median cadence: retained window is not "
            "strictly time-ordered"
        )
    if max(diffs) > CADENCE_IRREGULAR_FACTOR * med:
        return med, CadenceCondition.IRREGULAR
    return med, CadenceCondition.NOMINAL


def _admission_status(
    attempted: int, admitted: int
) -> CurrentRecordAdmissionStatus:
    if attempted == 0:
        return CurrentRecordAdmissionStatus.NO_RECORD_ATTEMPT
    if admitted == 0:
        return CurrentRecordAdmissionStatus.ALL_RECORDS_REJECTED
    if admitted == attempted:
        return CurrentRecordAdmissionStatus.ALL_RECORDS_ADMITTED
    return CurrentRecordAdmissionStatus.SOME_RECORDS_ADMITTED


def _station_provenance_status(
    samples: Sequence[RetainedSample],
) -> ProvenanceStatus:
    if not samples:
        return ProvenanceStatus.RESOLVED  # vacuous: nothing to reference
    n_hashed = sum(1 for s in samples if _is_sha256(s.payload_hash))
    if n_hashed == len(samples):
        return ProvenanceStatus.RESOLVED
    if n_hashed == 0:
        return ProvenanceStatus.UNAVAILABLE
    return ProvenanceStatus.PARTIAL


def _rayleigh_check(attempt: StationAttemptResult) -> ConfounderCheck:
    """Rayleigh-wave confounder for a successfully scored station."""
    scores = attempt.scores
    if scores is None:  # caller gates on SCORING_SUCCEEDED
        raise AssessmentConstructionError(
            "Rayleigh confounder requires score components"
        )
    suspect = scores.rayleigh_wave_suspect
    if suspect is not None:
        return ConfounderCheck(
            name=RAYLEIGH_CONFOUNDER_NAME,
            applicability=ConfounderApplicability.APPLICABLE,
            prerequisite_status=ConfounderPrerequisiteStatus.AVAILABLE,
            result=(
                ConfounderResultValue.FLAG_RAISED
                if suspect
                else ConfounderResultValue.FLAG_NOT_RAISED
            ),
        )
    prerequisite = (
        ConfounderPrerequisiteStatus.AVAILABLE
        if attempt.rayleigh_inputs_available
        else ConfounderPrerequisiteStatus.MISSING
    )
    return ConfounderCheck(
        name=RAYLEIGH_CONFOUNDER_NAME,
        applicability=ConfounderApplicability.APPLICABLE,
        prerequisite_status=prerequisite,
        result=ConfounderResultValue.NOT_EVALUATED,
    )


def _registry_limitations(
    conditions: Iterable[tuple[str, str]],
) -> list[StationLimitation]:
    """Registry-derived limitation list, deduplicated and sorted by code.

    Mirrors StationAssessmentEntry._expected_limitation_codes exactly;
    the model validator re-derives and rejects any divergence.
    """
    seen: dict[str, StationLimitation] = {}
    for axis, value in conditions:
        mapping = registry_lookup(axis, value)
        if mapping is not None:
            code, effect = mapping
            seen[code.value] = StationLimitation(code=code, effect=effect)
    return [seen[code] for code in sorted(seen)]


def _build_station_entry(
    attempt: StationAttemptResult,
    detector_config: DetectorConfig,
    produced_at_utc: datetime,
) -> StationAssessmentEntry:
    if attempt.source == "dart":
        source = DataSource.DART
    elif attempt.source == "coops":
        source = DataSource.COOPS
    else:
        raise AssessmentConstructionError(
            f"Station attempts accept only dart/coops sources, got "
            f"{attempt.source!r}"
        )

    samples = attempt.retained_samples
    n_rejected = attempt.n_records_attempted - attempt.n_records_admitted
    if n_rejected < 0:
        raise AssessmentConstructionError(
            f"Admitted count {attempt.n_records_admitted} exceeds attempted "
            f"count {attempt.n_records_attempted} for {attempt.source}:"
            f"{attempt.station_id}"
        )

    no_data = attempt.scoring_status is StationScoringStatus.NO_RETAINED_DATA
    if no_data != (len(samples) == 0):
        raise AssessmentConstructionError(
            f"Scoring status {attempt.scoring_status} inconsistent with "
            f"{len(samples)} retained sample(s) for {attempt.source}:"
            f"{attempt.station_id}"
        )

    bounds: ObservationBounds | None = None
    qc_window: RetainedWindowQC | None = None
    latest_obs: datetime | None = None
    operational_age: float | None = None
    median_cadence, cadence_condition = _cadence(samples)
    if samples:
        bounds = ObservationBounds(
            first_observation_utc=_epoch_to_utc(samples[0].epoch_sec),
            last_observation_utc=_epoch_to_utc(samples[-1].epoch_sec),
        )
        qc_window = _build_retained_window_qc(samples, bounds)
        latest_obs = bounds.last_observation_utc
        # Diagnostic only (excluded from the scientific content hash);
        # clamp so minor clock skew between data time and production
        # time cannot fail construction.
        operational_age = max(
            0.0, (produced_at_utc - latest_obs).total_seconds()
        )

    scored = attempt.scoring_status is StationScoringStatus.SCORING_SUCCEEDED
    if scored and attempt.scores is None:
        raise AssessmentConstructionError(
            f"SCORING_SUCCEEDED without score components for "
            f"{attempt.source}:{attempt.station_id}"
        )

    # Per-source descriptive fields.
    dart_mode: DartDataMode | None = None
    coops_products: list[str] = []
    if source is DataSource.DART:
        if samples:
            dart_mode = (
                DartDataMode.EVENT
                if attempt.dart_window_event_mode
                else DartDataMode.STANDARD
            )
        else:
            dart_mode = DartDataMode.UNKNOWN
        mixed_condition = MixedProductCondition.NOT_APPLICABLE
    else:
        coops_products = sorted(
            {s.product for s in samples if s.product is not None}
        )
        mixed_condition = (
            MixedProductCondition.MIXED_PRODUCTS
            if len(coops_products) > 1
            else MixedProductCondition.SINGLE_PRODUCT
        )

    if scored:
        assert attempt.scores is not None
        filtering_condition = (
            FilteringCondition.DEGRADED_NYQUIST
            if attempt.scores.filter_degraded
            else FilteringCondition.NOMINAL
        )
    else:
        filtering_condition = FilteringCondition.UNKNOWN

    source_validation = (
        SourceValidationStatus.VALIDATED_FOR_CONFIGURED_USE
        if source is DataSource.DART
        else SourceValidationStatus.VALIDATION_LIMITED
    )
    provenance_status = _station_provenance_status(samples)
    manifest_condition = StationManifestCondition.NO_MANIFEST

    confounder_checks = [_rayleigh_check(attempt)] if scored else []

    # Retained record references: composite identities for every sample
    # that carries a canonical payload hash. Unhashed samples are
    # disclosed through provenance_status instead of being referenced.
    refs: list[RetainedRecordRef] = []
    for s in samples:
        sha = s.payload_hash
        if not _is_sha256(sha):
            continue
        measurement_type: Literal[1, 2, 3] | None = None
        if source is DataSource.DART and s.measurement_type in (1, 2, 3):
            # Narrowed by the membership test; ingest normalizes other
            # values to None before retention.
            measurement_type = s.measurement_type  # type: ignore[assignment]
        refs.append(
            RetainedRecordRef(
                source=source,
                station_id=attempt.station_id,
                observed_at_utc=_epoch_to_utc(s.epoch_sec),
                measurement_type=measurement_type,
                product=s.product if source is DataSource.COOPS else None,
                payload_sha256=sha,
            )
        )

    # Registry-derived limitations, mirroring the entry validator's axis
    # walk so the constructed list is exactly the expected set.
    conditions: list[tuple[str, str]] = [
        ("CalibrationStatus", attempt.calibration_status.value),
        ("CadenceCondition", cadence_condition.value),
        ("FilteringCondition", filtering_condition.value),
        ("SourceValidationStatus", source_validation.value),
        ("ProvenanceStatus", provenance_status.value),
        ("StationManifestCondition", manifest_condition.value),
        ("MixedProductCondition", mixed_condition.value),
    ]
    if qc_window is None:
        conditions.append(
            ("QCAggregateCondition", QCAggregateCondition.NO_RETAINED_QC.value)
        )
    else:
        conditions.append(
            ("QCExecutionStatus", qc_window.execution_status.value)
        )
        conditions.append(
            ("QCAggregateCondition", qc_window.aggregate_condition.value)
        )
    for check in confounder_checks:
        conditions.append(("ConfounderResultValue", check.result.value))
    limitations = _registry_limitations(conditions)

    evaluation_status = derive_evaluation_status(
        attempt.scoring_status, (lim.effect for lim in limitations)
    )

    detector_components: DetectorComponents | None = None
    threshold_evaluation: ThresholdEvaluation | None = None
    if scored:
        assert attempt.scores is not None and bounds is not None
        scores = attempt.scores
        detector_components = DetectorComponents(
            threshold_score=scores.threshold_score,
            wavelet_score=scores.wavelet_score,
            bocpd_score=scores.bocpd_score,
            statistical_score=scores.statistical_score,
            ml_score=scores.ml_score,
        )
        result = derive_threshold_result_value(
            evaluation_status,
            scores.ensemble_score,
            detector_config.t1_investigate,
        )
        tier = (
            derive_highest_tier(
                scores.ensemble_score,
                detector_config.t1_investigate,
                detector_config.t2_assess,
                detector_config.t3_escalate,
            )
            if result is ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET
            else None
        )
        threshold_evaluation = ThresholdEvaluation(
            source=source,
            station_id=attempt.station_id,
            result=result,
            ensemble_score=scores.ensemble_score,
            t1_investigate=detector_config.t1_investigate,
            t2_assess=detector_config.t2_assess,
            t3_escalate=detector_config.t3_escalate,
            highest_tier_met=tier,
            evaluated_window=bounds,
        )

    return StationAssessmentEntry(
        source=source,
        station_id=attempt.station_id,
        admission_status=_admission_status(
            attempt.n_records_attempted, attempt.n_records_admitted
        ),
        n_records_attempted=attempt.n_records_attempted,
        n_records_admitted=attempt.n_records_admitted,
        n_records_rejected=n_rejected,
        scoring_status=attempt.scoring_status,
        evaluation_status=evaluation_status,
        observation_bounds=bounds,
        n_retained_samples=len(samples),
        median_cadence_sec=median_cadence,
        latest_observation_utc=latest_obs,
        operational_age_at_production_sec=operational_age,
        current_dart_data_mode=dart_mode,
        coops_products=coops_products,
        qc_retained_window=qc_window,
        calibration_status=attempt.calibration_status,
        calibration_sha256=attempt.calibration_sha256,
        cadence_condition=cadence_condition,
        filtering_condition=filtering_condition,
        mixed_product_condition=mixed_condition,
        manifest_condition=manifest_condition,
        detector_components=detector_components,
        threshold_evaluation=threshold_evaluation,
        confounder_checks=confounder_checks,
        retained_record_refs=refs,
        source_validation_status=source_validation,
        provenance_status=provenance_status,
        limitations=limitations,
        failure_reason=attempt.failure_reason,
    )


def build_detector_config(t1: float, t2: float, t3: float) -> DetectorConfig:
    """Detector configuration in force, with a canonical content hash.

    The hash covers everything that changes detector behavior: version,
    thresholds, ensemble weights (with and without ML), and the bandpass
    corner frequencies.
    """
    payload = {
        "detector_version": DETECTOR_VERSION,
        "t1_investigate": t1,
        "t2_assess": t2,
        "t3_escalate": t3,
        "w_threshold": W_THRESHOLD,
        "w_statistical": W_STATISTICAL,
        "w_ml": W_ML,
        "w_threshold_no_ml": W_THRESHOLD_NO_ML,
        "w_statistical_no_ml": W_STATISTICAL_NO_ML,
        "bandpass_low_hz": BANDPASS_LOW_HZ,
        "bandpass_high_hz": BANDPASS_HIGH_HZ,
    }
    return DetectorConfig(
        detector_version=DETECTOR_VERSION,
        t1_investigate=t1,
        t2_assess=t2,
        t3_escalate=t3,
        configuration_sha256=canonical_sha256(payload),
    )


def _build_seismic_context(ctx: EventContext) -> EventSeismicContext:
    """Bind the FSM event context's external seismic identity.

    Events without a fully bound identity (offline scripts, pre-identity
    durable rows, records that carried no canonical payload hash) cannot
    produce an assessment; the resulting construction error becomes a
    disclosed gap.
    """
    if not ctx.seismic_provider or not ctx.external_event_id:
        raise AssessmentConstructionError(
            "Active event carries no bound external seismic identity"
        )
    if not ctx.trigger_revision_id or not _is_sha256(
        ctx.trigger_revision_sha256
    ):
        raise AssessmentConstructionError(
            "Active event's trigger revision lacks an ID or canonical "
            "payload hash"
        )
    try:
        context_class = SeismicContextClass(ctx.seismic_context_class)
    except ValueError as exc:
        raise AssessmentConstructionError(
            f"Unknown seismic context class {ctx.seismic_context_class!r}"
        ) from exc

    latest_is_trigger = (
        not ctx.latest_revision_id
        or ctx.latest_revision_id == ctx.trigger_revision_id
    )
    trigger_ref = SeismicRevisionRef(
        provider_revision_id=ctx.trigger_revision_id,
        payload_sha256=ctx.trigger_revision_sha256,
        # EventContext keeps only the latest revision's provider update
        # time; it equals the trigger's own update time exactly while
        # the trigger is still the incumbent revision.
        provider_updated_at_utc=(
            ctx.latest_revision_updated_utc if latest_is_trigger else None
        ),
    )
    if latest_is_trigger:
        latest_ref = trigger_ref
    else:
        if not _is_sha256(ctx.latest_revision_sha256):
            raise AssessmentConstructionError(
                "Latest admissible revision lacks a canonical payload hash"
            )
        latest_ref = SeismicRevisionRef(
            provider_revision_id=ctx.latest_revision_id,
            payload_sha256=ctx.latest_revision_sha256,
            provider_updated_at_utc=ctx.latest_revision_updated_utc,
        )
    return EventSeismicContext(
        provider=ctx.seismic_provider,
        external_event_id=ctx.external_event_id,
        context_class=context_class,
        trigger_revision=trigger_ref,
        latest_admissible_revision=latest_ref,
    )


def _build_analyses(spatial_analysis_ran: bool) -> AnalysisStatusSet:
    """Version 1 live-path analysis statuses.

    Scenario inversion, verification, and coastal estimates exist only
    on the offline artifact path; inundation and meteorological source
    discrimination are not implemented anywhere. Spatial coherence is
    the one live analysis, and it reports SUCCEEDED only when a spatial
    result was produced at this checkpoint.
    """
    offline_only = AnalysisStatus(
        capability=AnalysisCapability.IMPLEMENTED_OFFLINE_ONLY,
        execution=AnalysisExecution.NOT_RUN,
        quality=AnalysisQuality.NOT_ASSESSED,
    )
    not_implemented = AnalysisStatus(
        capability=AnalysisCapability.NOT_IMPLEMENTED,
        execution=AnalysisExecution.NOT_RUN,
        quality=AnalysisQuality.NOT_ASSESSED,
    )
    return AnalysisStatusSet(
        scenario=offline_only,
        scenario_verification=offline_only,
        coastal_arrival=offline_only,
        coastal_amplitude=offline_only,
        inundation=not_implemented,
        meteorological_source_discrimination=not_implemented,
        spatial_coherence=AnalysisStatus(
            capability=AnalysisCapability.IMPLEMENTED_LIVE,
            execution=(
                AnalysisExecution.SUCCEEDED
                if spatial_analysis_ran
                else AnalysisExecution.NOT_RUN
            ),
            quality=AnalysisQuality.NOT_ASSESSED,
        ),
    )


def _build_input_refs(
    attempts: Sequence[StationAttemptResult],
) -> tuple[list[InputRef], int, int, bool]:
    """Envelope input references over all retained samples.

    Returns (refs, n_distinct_hashes_expected, n_unhashed_samples,
    capped). Duplicate hashes across stations keep the first station in
    sorted attempt order, so the reference set is deterministic.
    """
    by_hash: dict[str, InputRef] = {}
    n_unhashed = 0
    for attempt in attempts:
        for sample in attempt.retained_samples:
            sha = sample.payload_hash
            if not _is_sha256(sha):
                n_unhashed += 1
                continue
            if sha not in by_hash:
                by_hash[sha] = InputRef(
                    source=DataSource(attempt.source),
                    record_id=f"{attempt.source}:{attempt.station_id}",
                    sha256=sha,
                )
    expected = len(by_hash)
    refs = [by_hash[sha] for sha in sorted(by_hash)]
    capped = len(refs) > MAX_ENVELOPE_INPUT_REFS
    if capped:
        refs = refs[:MAX_ENVELOPE_INPUT_REFS]
    return refs, expected, n_unhashed, capped


def _build_provenance_summary(
    *,
    n_expected: int,
    n_included: int,
    n_unhashed: int,
    capped: bool,
    entries: Sequence[StationAssessmentEntry],
    database_available: bool,
    companion_persistence_failures: Sequence[str],
) -> ProvenanceSummary:
    """Checkpoint-level provenance accounting.

    Ingest normalizes malformed hashes to absent before samples reach
    the retained window, so the unresolved-raw-records and
    malformed-or-absent-hashes counters observe the same population of
    unhashed retained samples.
    """
    if n_included == n_expected and n_unhashed == 0 and not capped:
        status = ProvenanceStatus.RESOLVED
    elif n_included == 0 and (n_expected > 0 or n_unhashed > 0):
        status = ProvenanceStatus.UNAVAILABLE
    else:
        status = ProvenanceStatus.PARTIAL
    calibration_available = all(
        entry.calibration_sha256 != ""
        for entry in entries
        if entry.calibration_status
        in (
            CalibrationStatus.ADEQUATE_PRE_EVENT_BASELINE,
            CalibrationStatus.LIMITED_PRE_EVENT_BASELINE,
        )
    )
    return ProvenanceSummary(
        status=status,
        n_references_expected=n_expected,
        n_references_included=n_included,
        n_unresolved_raw_records=n_unhashed,
        n_malformed_or_absent_hashes=n_unhashed,
        references_capped=capped,
        calibration_provenance_available=calibration_available,
        database_available=database_available,
        companion_persistence_failures=sorted(
            companion_persistence_failures
        ),
    )


def build_ocean_evidence_assessment(
    *,
    checkpoint_id: str,
    checkpoint_source: CheckpointSource,
    event_id: UUID,
    event_context: EventContext,
    trace_id: UUID,
    produced_at_utc: datetime,
    station_attempts: Sequence[StationAttemptResult],
    detector_config: DetectorConfig,
    fsm_state_before: str,
    fsm_state_after: str,
    fsm_transition_ref: str,
    dart_event_mode_stations_since_event_origin: Sequence[str],
    pipeline_outcome_field: str | None,
    seismic_only_no_score: bool,
    spatial_analysis_ran: bool,
    database_available: bool,
    companion_persistence_failures: Sequence[str] = (),
    code_version: str = "",
) -> OceanEvidenceAssessment:
    """Assemble one OceanEvidenceAssessment from worker-owned facts.

    Pure and deterministic given its arguments. The returned assessment
    has empty hash fields; the caller applies
    ``ocean_evidence_hashing.finalize_assessment_hashes`` before
    persistence.

    Raises:
        AssessmentConstructionError: unbound seismic identity or
            internally inconsistent attempt facts.
        pydantic.ValidationError: constructed facts violate a schema
            invariant. Both are assessment-gap conditions.
    """
    keys = [(a.source, a.station_id) for a in station_attempts]
    if len(keys) != len(set(keys)):
        raise AssessmentConstructionError(
            "Duplicate (source, station_id) in station attempts"
        )
    ordered = sorted(station_attempts, key=lambda a: (a.source, a.station_id))

    entries = [
        _build_station_entry(attempt, detector_config, produced_at_utc)
        for attempt in ordered
    ]
    refs, n_expected, n_unhashed, capped = _build_input_refs(ordered)

    state_before = FsmState(fsm_state_before)
    state_after = FsmState(fsm_state_after)
    lifetime_stations = sorted(
        set(dart_event_mode_stations_since_event_origin)
    )

    return OceanEvidenceAssessment(
        checkpoint_id=checkpoint_id,
        checkpoint_source=checkpoint_source,
        event_id=event_id,
        trace_id=trace_id,
        producer=ASSESSMENT_PRODUCER,
        produced_at_utc=produced_at_utc,
        input_refs=refs,
        code_version=code_version,
        seismic_context=_build_seismic_context(event_context),
        contributing_trace_ids=[trace_id],
        source_time_bounds=derive_source_time_bounds(entries),
        station_scope=StationScope.OBSERVED_RECORDS_ONLY,
        station_manifest=None,
        detector_config=detector_config,
        stations=entries,
        dart_rollup=derive_source_rollup(
            DataSource.DART, entries, StationScope.OBSERVED_RECORDS_ONLY
        ),
        coops_rollup=derive_source_rollup(
            DataSource.COOPS, entries, StationScope.OBSERVED_RECORDS_ONLY
        ),
        fsm_state_before=state_before,
        fsm_state_after=state_after,
        fsm_state_changed=state_before is not state_after,
        fsm_transition_ref=fsm_transition_ref,
        dart_stations_currently_in_event_mode=sorted(
            e.station_id
            for e in entries
            if e.source is DataSource.DART
            and e.current_dart_data_mode is DartDataMode.EVENT
        ),
        dart_event_mode_observed_since_event_origin=bool(lifetime_stations),
        dart_event_mode_stations_since_event_origin=lifetime_stations,
        confounder_checks=[],
        analyses=_build_analyses(spatial_analysis_ran),
        pipeline_outcome=derive_pipeline_outcome(
            pipeline_outcome_field, state_after, seismic_only_no_score
        ),
        provenance=_build_provenance_summary(
            n_expected=n_expected,
            n_included=len(refs),
            n_unhashed=n_unhashed,
            capped=capped,
            entries=entries,
            database_available=database_available,
            companion_persistence_failures=companion_persistence_failures,
        ),
    )
