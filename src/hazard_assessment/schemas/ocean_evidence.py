"""Deterministic ocean evidence assessment schema.

``OceanEvidenceAssessment`` is the immutable canonical account of one
eligible active-event processing checkpoint. It answers only:

    Which configured detector thresholds were met in buffer-admitted
    ocean observations evaluated in retained station windows at this
    processing checkpoint, and what limitations constrain
    interpretation?

It does not declare that a disturbance is current, persistent, caused
by the active earthquake, or a tsunami. A threshold result is not a
tsunami detection, and a no-crossing result is not evidence that no
tsunami exists.

Determinism contract: every scientific status in this schema is either
a typed input fact or derived by a pure function in this module from
typed facts through the versioned ``AssessmentConditionRegistry``.
Model validators re-run the derivations, so two implementations cannot
construct the same facts with different scientific statuses.

The three content hashes (input manifest, scientific content, transport
provenance) are computed by explicit projection functions in
``hazard_assessment.schemas.ocean_evidence_hashing``, never by generic
model dumps. Each hash field is excluded from its own projection.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum
from typing import Final, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator

from hazard_assessment.schemas.envelope import (
    AwareDatetime,
    BaseEnvelope,
    DataSource,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"
OPTIONAL_SHA256_PATTERN = r"^$|^[0-9a-f]{64}$"
STATION_ID_PATTERN = r"^[0-9A-Za-z_.:\-]{1,64}$"

_STATION_ID_RE = re.compile(STATION_ID_PATTERN)

ASSESSMENT_SCHEMA_VERSION: Final = 1
"""Major version of this assessment artifact.

Distinct from the envelope ``schema_version`` and from the integer
``version`` column already present on ``processed_features``. A change
to outcome meanings, status semantics, or hash projections requires a
bump here.
"""

CONDITION_REGISTRY_VERSION: Final = "1.0.0"
"""Version of the frozen condition-to-limitation registry below."""


# ---------------------------------------------------------------------------
# Canonical enums
# ---------------------------------------------------------------------------


class FsmState(StrEnum):
    """FSM states as recorded in assessments.

    Mirrors ``orchestrator.states.SystemState``; a conformance test
    asserts parity. The schema package does not import the orchestrator
    to keep the dependency direction one way.
    """

    IDLE = "IDLE"
    MONITOR = "MONITOR"
    INVESTIGATE = "INVESTIGATE"
    ASSESS = "ASSESS"
    ESCALATE = "ESCALATE"


class StationScope(StrEnum):
    """Declared station universe for one assessment."""

    OBSERVED_RECORDS_ONLY = "OBSERVED_RECORDS_ONLY"
    CONFIGURED_INVENTORY = "CONFIGURED_INVENTORY"


class CurrentRecordAdmissionStatus(StrEnum):
    """Outcome of record attempts in this checkpoint only."""

    NO_RECORD_ATTEMPT = "NO_RECORD_ATTEMPT"
    ALL_RECORDS_REJECTED = "ALL_RECORDS_REJECTED"
    SOME_RECORDS_ADMITTED = "SOME_RECORDS_ADMITTED"
    ALL_RECORDS_ADMITTED = "ALL_RECORDS_ADMITTED"


class StationScoringStatus(StrEnum):
    """State of the retained window after current attempts."""

    NO_RETAINED_DATA = "NO_RETAINED_DATA"
    INSUFFICIENT_RETAINED_DATA = "INSUFFICIENT_RETAINED_DATA"
    SCORING_FAILED = "SCORING_FAILED"
    SCORING_SUCCEEDED = "SCORING_SUCCEEDED"


class StationEvaluationStatus(StrEnum):
    """Scientific interpretability of one station result."""

    NOT_EVALUATED = "NOT_EVALUATED"
    EVALUATED_WITH_LIMITATIONS = "EVALUATED_WITH_LIMITATIONS"
    EVALUATED = "EVALUATED"


class EvaluationEffect(StrEnum):
    """Frozen effect of one limitation code."""

    INFORMATIONAL = "INFORMATIONAL"
    LIMITS_INTERPRETATION = "LIMITS_INTERPRETATION"
    PREVENTS_EVALUATION = "PREVENTS_EVALUATION"


class CalibrationStatus(StrEnum):
    """Calibration state for one station."""

    NOT_REQUIRED = "NOT_REQUIRED"
    ADEQUATE_PRE_EVENT_BASELINE = "ADEQUATE_PRE_EVENT_BASELINE"
    LIMITED_PRE_EVENT_BASELINE = "LIMITED_PRE_EVENT_BASELINE"
    EVENT_WINDOW_FALLBACK = "EVENT_WINDOW_FALLBACK"
    UNAVAILABLE = "UNAVAILABLE"
    ERROR = "ERROR"


class ThresholdResultValue(StrEnum):
    """Canonical threshold results. Comparison is inclusive."""

    CONFIGURED_ENSEMBLE_THRESHOLD_MET = "CONFIGURED_ENSEMBLE_THRESHOLD_MET"
    NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW = (
        "NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW"
    )
    NOT_EVALUATED = "NOT_EVALUATED"


class SourceRollupStatus(StrEnum):
    """Per-source threshold rollup."""

    SOURCE_CONFIGURED_THRESHOLD_MET = "SOURCE_CONFIGURED_THRESHOLD_MET"
    SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS = (
        "SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS"
    )
    SOURCE_NOT_EVALUATED = "SOURCE_NOT_EVALUATED"


class SourceValidationStatus(StrEnum):
    """Whether a source's detector use is validated.

    Orthogonal to threshold status: ``VALIDATION_LIMITED`` adds a
    limiting code but does not change whether a threshold crossed.
    """

    VALIDATED_FOR_CONFIGURED_USE = "VALIDATED_FOR_CONFIGURED_USE"
    VALIDATION_LIMITED = "VALIDATION_LIMITED"
    VALIDATION_NOT_EVALUATED = "VALIDATION_NOT_EVALUATED"


class DartDataMode(StrEnum):
    """DART transmission mode from the newest retained accepted record.

    Distinct from ``schemas.qc.DataMode``: assessments must express
    ``UNKNOWN`` when no retained record carries mode information.
    """

    STANDARD = "STANDARD"
    EVENT = "EVENT"
    UNKNOWN = "UNKNOWN"


class ConfounderApplicability(StrEnum):
    """Whether a named confounder check applies here."""

    APPLICABLE = "APPLICABLE"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ConfounderPrerequisiteStatus(StrEnum):
    """Availability of a confounder check's inputs."""

    AVAILABLE = "AVAILABLE"
    MISSING = "MISSING"
    ERROR = "ERROR"


class ConfounderResultValue(StrEnum):
    """Result of a named confounder check.

    A raised flag means only that the configured heuristic fired.
    """

    FLAG_RAISED = "FLAG_RAISED"
    FLAG_NOT_RAISED = "FLAG_NOT_RAISED"
    NOT_EVALUATED = "NOT_EVALUATED"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class AnalysisCapability(StrEnum):
    """Whether a downstream analysis exists on the live path."""

    IMPLEMENTED_LIVE = "IMPLEMENTED_LIVE"
    IMPLEMENTED_OFFLINE_ONLY = "IMPLEMENTED_OFFLINE_ONLY"
    NOT_IMPLEMENTED = "NOT_IMPLEMENTED"


class AnalysisExecution(StrEnum):
    """Whether a downstream analysis ran at this checkpoint."""

    NOT_RUN = "NOT_RUN"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class AnalysisQuality(StrEnum):
    """Quality of a downstream analysis result."""

    NOT_ASSESSED = "NOT_ASSESSED"
    NOMINAL = "NOMINAL"
    DEGRADED = "DEGRADED"


class PipelineOutcome(StrEnum):
    """Version 1 checkpoint outcome.

    ``PROCESSING_INCOMPLETE`` cannot dispatch the active-event model.
    """

    MONITORING_CONTINUES = "MONITORING_CONTINUES"
    ABSTAIN = "ABSTAIN"
    PROCESSING_INCOMPLETE = "PROCESSING_INCOMPLETE"


class ProvenanceStatus(StrEnum):
    """Completeness of provenance resolution."""

    RESOLVED = "RESOLVED"
    PARTIAL = "PARTIAL"
    UNAVAILABLE = "UNAVAILABLE"


class SeismicContextClass(StrEnum):
    """Provenance class of the seismic context."""

    LIVE_RECEIPT_ORDERED = "LIVE_RECEIPT_ORDERED"
    HISTORICAL_REVISION_REPLAY = "HISTORICAL_REVISION_REPLAY"
    POST_HOC_FINAL_PRODUCT = "POST_HOC_FINAL_PRODUCT"


class CheckpointSource(StrEnum):
    """How this checkpoint's identity was derived."""

    LIVE_KAFKA = "LIVE_KAFKA"
    REPLAY = "REPLAY"


class QCExecutionStatus(StrEnum):
    """Whether QC metadata exists for the retained window."""

    SUCCEEDED = "SUCCEEDED"
    PARTIAL = "PARTIAL"
    FAILED = "FAILED"
    NOT_RUN = "NOT_RUN"


class QCAggregateCondition(StrEnum):
    """Decisive worst QARTOD flag over the retained window.

    QC is metadata and never filters pressure observations from
    scoring; adverse flags constrain interpretation instead.
    """

    NO_RETAINED_QC = "NO_RETAINED_QC"
    WORST_FLAG_PASS = "WORST_FLAG_PASS"
    WORST_FLAG_NOT_EVALUATED = "WORST_FLAG_NOT_EVALUATED"
    WORST_FLAG_SUSPECT = "WORST_FLAG_SUSPECT"
    WORST_FLAG_FAIL = "WORST_FLAG_FAIL"
    WORST_FLAG_MISSING = "WORST_FLAG_MISSING"


class CadenceCondition(StrEnum):
    """Sampling-cadence condition of the retained window."""

    NOMINAL = "NOMINAL"
    IRREGULAR = "IRREGULAR"
    UNKNOWN = "UNKNOWN"


class FilteringCondition(StrEnum):
    """Bandpass filtering state during scoring."""

    NOMINAL = "NOMINAL"
    DEGRADED_NYQUIST = "DEGRADED_NYQUIST"
    UNKNOWN = "UNKNOWN"


class MixedProductCondition(StrEnum):
    """CO-OPS product homogeneity of the retained window."""

    SINGLE_PRODUCT = "SINGLE_PRODUCT"
    MIXED_PRODUCTS = "MIXED_PRODUCTS"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class StationManifestCondition(StrEnum):
    """Station membership relative to a configured manifest."""

    NO_MANIFEST = "NO_MANIFEST"
    IN_MANIFEST = "IN_MANIFEST"
    OUT_OF_MANIFEST = "OUT_OF_MANIFEST"


class LimitationCode(StrEnum):
    """Named limitation codes produced by the condition registry."""

    CALIBRATION_LIMITED_PRE_EVENT_BASELINE = "CALIBRATION_LIMITED_PRE_EVENT_BASELINE"
    CALIBRATION_EVENT_WINDOW_FALLBACK = "CALIBRATION_EVENT_WINDOW_FALLBACK"
    CALIBRATION_UNAVAILABLE = "CALIBRATION_UNAVAILABLE"
    CALIBRATION_ERROR = "CALIBRATION_ERROR"
    QC_EXECUTION_PARTIAL = "QC_EXECUTION_PARTIAL"
    QC_EXECUTION_FAILED = "QC_EXECUTION_FAILED"
    QC_NOT_RUN = "QC_NOT_RUN"
    QC_ABSENT_FOR_RETAINED_WINDOW = "QC_ABSENT_FOR_RETAINED_WINDOW"
    QC_WORST_FLAG_NOT_EVALUATED = "QC_WORST_FLAG_NOT_EVALUATED"
    QC_WORST_FLAG_SUSPECT = "QC_WORST_FLAG_SUSPECT"
    QC_WORST_FLAG_FAIL = "QC_WORST_FLAG_FAIL"
    QC_WORST_FLAG_MISSING = "QC_WORST_FLAG_MISSING"
    CADENCE_IRREGULAR = "CADENCE_IRREGULAR"
    CADENCE_UNKNOWN = "CADENCE_UNKNOWN"
    FILTER_DEGRADED_NYQUIST = "FILTER_DEGRADED_NYQUIST"
    FILTERING_STATE_UNKNOWN = "FILTERING_STATE_UNKNOWN"
    CONFOUNDER_FLAG_RAISED = "CONFOUNDER_FLAG_RAISED"
    CONFOUNDER_NOT_EVALUATED = "CONFOUNDER_NOT_EVALUATED"
    SOURCE_VALIDATION_LIMITED = "SOURCE_VALIDATION_LIMITED"
    SOURCE_VALIDATION_NOT_EVALUATED = "SOURCE_VALIDATION_NOT_EVALUATED"
    PROVENANCE_PARTIAL = "PROVENANCE_PARTIAL"
    PROVENANCE_UNAVAILABLE = "PROVENANCE_UNAVAILABLE"
    NETWORK_COVERAGE_NOT_EVALUATED = "NETWORK_COVERAGE_NOT_EVALUATED"
    OUT_OF_MANIFEST_OBSERVATION = "OUT_OF_MANIFEST_OBSERVATION"
    MIXED_PRODUCT_WINDOW = "MIXED_PRODUCT_WINDOW"


# ---------------------------------------------------------------------------
# Frozen condition registry
# ---------------------------------------------------------------------------

_Mapping = tuple[LimitationCode, EvaluationEffect] | None

ASSESSMENT_CONDITION_REGISTRY: dict[tuple[str, str], _Mapping] = {
    # Calibration: limited or fallback calibration limits
    # interpretation; unavailable or errored calibration prevents
    # scientific evaluation.
    ("CalibrationStatus", "NOT_REQUIRED"): None,
    ("CalibrationStatus", "ADEQUATE_PRE_EVENT_BASELINE"): None,
    ("CalibrationStatus", "LIMITED_PRE_EVENT_BASELINE"): (
        LimitationCode.CALIBRATION_LIMITED_PRE_EVENT_BASELINE,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("CalibrationStatus", "EVENT_WINDOW_FALLBACK"): (
        LimitationCode.CALIBRATION_EVENT_WINDOW_FALLBACK,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("CalibrationStatus", "UNAVAILABLE"): (
        LimitationCode.CALIBRATION_UNAVAILABLE,
        EvaluationEffect.PREVENTS_EVALUATION,
    ),
    ("CalibrationStatus", "ERROR"): (
        LimitationCode.CALIBRATION_ERROR,
        EvaluationEffect.PREVENTS_EVALUATION,
    ),
    # QC execution: QC never prevents evaluation because it is
    # metadata, not a filter, and genuine tsunami excursions can trip
    # QC checks. Missing or failed QC still limits interpretation.
    ("QCExecutionStatus", "SUCCEEDED"): None,
    ("QCExecutionStatus", "PARTIAL"): (
        LimitationCode.QC_EXECUTION_PARTIAL,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("QCExecutionStatus", "FAILED"): (
        LimitationCode.QC_EXECUTION_FAILED,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("QCExecutionStatus", "NOT_RUN"): (
        LimitationCode.QC_NOT_RUN,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    # Decisive worst QARTOD flag over the retained window.
    ("QCAggregateCondition", "NO_RETAINED_QC"): (
        LimitationCode.QC_ABSENT_FOR_RETAINED_WINDOW,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("QCAggregateCondition", "WORST_FLAG_PASS"): None,
    ("QCAggregateCondition", "WORST_FLAG_NOT_EVALUATED"): (
        LimitationCode.QC_WORST_FLAG_NOT_EVALUATED,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("QCAggregateCondition", "WORST_FLAG_SUSPECT"): (
        LimitationCode.QC_WORST_FLAG_SUSPECT,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("QCAggregateCondition", "WORST_FLAG_FAIL"): (
        LimitationCode.QC_WORST_FLAG_FAIL,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("QCAggregateCondition", "WORST_FLAG_MISSING"): (
        LimitationCode.QC_WORST_FLAG_MISSING,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    # Cadence: irregular or unknown cadence is disclosed but does not
    # by itself change interpretability; filtering degradation is the
    # scientifically consequential downstream effect and has its own
    # axis.
    ("CadenceCondition", "NOMINAL"): None,
    ("CadenceCondition", "IRREGULAR"): (
        LimitationCode.CADENCE_IRREGULAR,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("CadenceCondition", "UNKNOWN"): (
        LimitationCode.CADENCE_UNKNOWN,
        EvaluationEffect.INFORMATIONAL,
    ),
    # Filtering: a Nyquist-degraded bandpass leaves the ensemble
    # BOCPD-dominated, which limits interpretation of the score.
    ("FilteringCondition", "NOMINAL"): None,
    ("FilteringCondition", "DEGRADED_NYQUIST"): (
        LimitationCode.FILTER_DEGRADED_NYQUIST,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("FilteringCondition", "UNKNOWN"): (
        LimitationCode.FILTERING_STATE_UNKNOWN,
        EvaluationEffect.INFORMATIONAL,
    ),
    # Confounders: a raised flag limits interpretation; an
    # unevaluated applicable check is disclosed.
    ("ConfounderResultValue", "FLAG_RAISED"): (
        LimitationCode.CONFOUNDER_FLAG_RAISED,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("ConfounderResultValue", "FLAG_NOT_RAISED"): None,
    ("ConfounderResultValue", "NOT_EVALUATED"): (
        LimitationCode.CONFOUNDER_NOT_EVALUATED,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("ConfounderResultValue", "NOT_APPLICABLE"): None,
    # Source validation: unvalidated configured detector
    # behavior limits interpretation without changing threshold facts.
    ("SourceValidationStatus", "VALIDATED_FOR_CONFIGURED_USE"): None,
    ("SourceValidationStatus", "VALIDATION_LIMITED"): (
        LimitationCode.SOURCE_VALIDATION_LIMITED,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("SourceValidationStatus", "VALIDATION_NOT_EVALUATED"): (
        LimitationCode.SOURCE_VALIDATION_NOT_EVALUATED,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    # Provenance: partial resolution is disclosed; a fully
    # unresolvable provenance chain limits interpretation.
    ("ProvenanceStatus", "RESOLVED"): None,
    ("ProvenanceStatus", "PARTIAL"): (
        LimitationCode.PROVENANCE_PARTIAL,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("ProvenanceStatus", "UNAVAILABLE"): (
        LimitationCode.PROVENANCE_UNAVAILABLE,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    # Station scope: observed-record scope prohibits
    # network-coverage claims. The code attaches to rollups and does
    # not change per-station interpretability.
    ("StationScope", "OBSERVED_RECORDS_ONLY"): (
        LimitationCode.NETWORK_COVERAGE_NOT_EVALUATED,
        EvaluationEffect.INFORMATIONAL,
    ),
    ("StationScope", "CONFIGURED_INVENTORY"): None,
    # Manifest membership: out-of-manifest observations stay
    # admissible and are identified as such.
    ("StationManifestCondition", "NO_MANIFEST"): None,
    ("StationManifestCondition", "IN_MANIFEST"): None,
    ("StationManifestCondition", "OUT_OF_MANIFEST"): (
        LimitationCode.OUT_OF_MANIFEST_OBSERVATION,
        EvaluationEffect.INFORMATIONAL,
    ),
    # Mixed CO-OPS products: a mixed-product window receives
    # an explicit limiting code.
    ("MixedProductCondition", "SINGLE_PRODUCT"): None,
    ("MixedProductCondition", "MIXED_PRODUCTS"): (
        LimitationCode.MIXED_PRODUCT_WINDOW,
        EvaluationEffect.LIMITS_INTERPRETATION,
    ),
    ("MixedProductCondition", "NOT_APPLICABLE"): None,
}

REGISTRY_AXES: dict[str, type[StrEnum]] = {
    "CalibrationStatus": CalibrationStatus,
    "QCExecutionStatus": QCExecutionStatus,
    "QCAggregateCondition": QCAggregateCondition,
    "CadenceCondition": CadenceCondition,
    "FilteringCondition": FilteringCondition,
    "ConfounderResultValue": ConfounderResultValue,
    "SourceValidationStatus": SourceValidationStatus,
    "ProvenanceStatus": ProvenanceStatus,
    "StationScope": StationScope,
    "StationManifestCondition": StationManifestCondition,
    "MixedProductCondition": MixedProductCondition,
}
"""Every condition axis the registry must cover exhaustively."""


def _canonical_effects() -> dict[LimitationCode, EvaluationEffect]:
    effects: dict[LimitationCode, EvaluationEffect] = {}
    for mapping in ASSESSMENT_CONDITION_REGISTRY.values():
        if mapping is None:
            continue
        code, effect = mapping
        existing = effects.get(code)
        if existing is not None and existing is not effect:
            raise RuntimeError(
                f"Limitation code {code} mapped to conflicting effects"
            )
        effects[code] = effect
    return effects


LIMITATION_CODE_EFFECTS: dict[LimitationCode, EvaluationEffect] = _canonical_effects()
"""Each limitation code has exactly one frozen evaluation effect."""


def registry_lookup(axis: str, value: str) -> _Mapping:
    """Return the frozen mapping for one condition, failing loudly.

    A missing mapping is a registry bug, not a soft default. Schema and
    registry tests must fail when a new enum value lacks a mapping.
    """
    key = (axis, value)
    if key not in ASSESSMENT_CONDITION_REGISTRY:
        raise KeyError(f"Condition {key} has no registry mapping")
    return ASSESSMENT_CONDITION_REGISTRY[key]


# ---------------------------------------------------------------------------
# Nested models
# ---------------------------------------------------------------------------

_FROZEN = ConfigDict(extra="forbid", frozen=True)


class ObservationBounds(BaseModel):
    """Inclusive observation-time bounds of a set of records."""

    first_observation_utc: AwareDatetime
    last_observation_utc: AwareDatetime

    @model_validator(mode="after")
    def _ordered(self) -> ObservationBounds:
        if self.first_observation_utc > self.last_observation_utc:
            raise ValueError("first_observation_utc must be <= last_observation_utc")
        return self

    model_config = _FROZEN


class SeismicRevisionRef(BaseModel):
    """One revision of an external seismic product."""

    provider_revision_id: str = Field(min_length=1, max_length=128)
    payload_sha256: str = Field(pattern=SHA256_PATTERN)
    provider_updated_at_utc: AwareDatetime | None = Field(
        default=None,
        description=(
            "Provider update time when valid. A missing, malformed, or "
            "post-receipt-future value is retained as provenance but "
            "cannot supersede the latest valid revision."
        ),
    )

    model_config = _FROZEN


class EventSeismicContext(BaseModel):
    """Seismic identity bound to this assessment."""

    provider: str = Field(min_length=1, max_length=64)
    external_event_id: str = Field(min_length=1, max_length=128)
    context_class: SeismicContextClass
    trigger_revision: SeismicRevisionRef = Field(
        description="Immutable revision that created the active event"
    )
    latest_admissible_revision: SeismicRevisionRef = Field(
        description=(
            "Latest admissible revision of the same external event at "
            "this checkpoint. Equals the trigger for POST_HOC_FINAL_PRODUCT."
        ),
    )

    @model_validator(mode="after")
    def _post_hoc_single_product(self) -> EventSeismicContext:
        if (
            self.context_class is SeismicContextClass.POST_HOC_FINAL_PRODUCT
            and self.trigger_revision != self.latest_admissible_revision
        ):
            raise ValueError(
                "POST_HOC_FINAL_PRODUCT requires trigger and latest "
                "revisions to be the same final product"
            )
        return self

    model_config = _FROZEN


class DetectorConfig(BaseModel):
    """Detector and threshold configuration in force."""

    detector_version: str = Field(min_length=1, max_length=64)
    t1_investigate: float = Field(ge=0.0, le=1.0)
    t2_assess: float = Field(ge=0.0, le=1.0)
    t3_escalate: float = Field(ge=0.0, le=1.0)
    configuration_sha256: str = Field(
        pattern=SHA256_PATTERN,
        description="Hash of the full serialized detector configuration",
    )

    @model_validator(mode="after")
    def _ordered_thresholds(self) -> DetectorConfig:
        if not (self.t1_investigate <= self.t2_assess <= self.t3_escalate):
            raise ValueError("Thresholds must satisfy t1 <= t2 <= t3")
        return self

    model_config = _FROZEN


class StationManifestRef(BaseModel):
    """Versioned, effective-dated reporting manifest identity."""

    manifest_id: str = Field(min_length=1, max_length=128)
    manifest_version: str = Field(min_length=1, max_length=64)
    effective_at_utc: AwareDatetime
    manifest_sha256: str = Field(pattern=SHA256_PATTERN)

    model_config = _FROZEN


class QCFlagCounts(BaseModel):
    """Counts of records by decisive QARTOD flag result."""

    n_pass: int = Field(ge=0)
    n_not_evaluated: int = Field(ge=0)
    n_suspect: int = Field(ge=0)
    n_fail: int = Field(ge=0)
    n_missing: int = Field(ge=0)

    model_config = _FROZEN


class RetainedWindowQC(BaseModel):
    """QC aggregate over the exact retained eligibility window.

    Computed once per record at processing time and aggregated over the
    records presented to the minimum-data and scoring step. Never mixed
    with incoming-batch QC, which is a separate operational feature
    labeled INCOMING_BATCH.
    """

    scope: Literal["RETAINED_ELIGIBILITY_WINDOW"] = "RETAINED_ELIGIBILITY_WINDOW"
    execution_status: QCExecutionStatus
    aggregate_condition: QCAggregateCondition
    observation_bounds: ObservationBounds | None = None
    n_records: int = Field(ge=0)
    flag_counts: QCFlagCounts
    n_usable: int = Field(ge=0)
    n_unusable: int = Field(ge=0)
    n_unevaluated_checks: int = Field(
        ge=0,
        description=(
            "Summed over the aggregated records, the number of QARTOD "
            "checks the system runs that returned no definitive result "
            "for a record. Reserved flag fields with no producer are not "
            "counted, so a record whose every runnable check decided "
            "contributes zero."
        ),
    )
    confidence_min: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_mean: float | None = Field(default=None, ge=0.0, le=1.0)
    confidence_max: float | None = Field(default=None, ge=0.0, le=1.0)
    record_sha256s: list[str] = Field(
        default_factory=list,
        description="Sorted unique payload hashes of the aggregated records",
    )

    @model_validator(mode="after")
    def _consistent(self) -> RetainedWindowQC:
        if self.n_usable + self.n_unusable != self.n_records:
            raise ValueError("n_usable + n_unusable must equal n_records")
        if self.n_records > 0 and self.observation_bounds is None:
            raise ValueError("Nonempty aggregates require observation_bounds")
        if self.record_sha256s != sorted(set(self.record_sha256s)):
            raise ValueError("record_sha256s must be sorted and unique")
        for h in self.record_sha256s:
            if len(h) != 64 or any(c not in "0123456789abcdef" for c in h):
                raise ValueError(f"Invalid sha256 in record_sha256s: {h!r}")
        fc = self.flag_counts
        total_flagged = (
            fc.n_pass + fc.n_not_evaluated + fc.n_suspect + fc.n_fail + fc.n_missing
        )
        if total_flagged > self.n_records:
            raise ValueError("Decisive flag counts cannot exceed n_records")
        if self.execution_status is QCExecutionStatus.SUCCEEDED:
            if total_flagged != self.n_records:
                raise ValueError(
                    "SUCCEEDED requires a decisive flag for every record"
                )
        elif self.execution_status is QCExecutionStatus.PARTIAL:
            if not 0 < total_flagged < self.n_records:
                raise ValueError(
                    "PARTIAL requires decisive flags for some but not all records"
                )
        elif self.execution_status is QCExecutionStatus.NOT_RUN:
            if total_flagged != 0:
                raise ValueError("NOT_RUN requires zero decisive flag counts")
        elif self.n_records > 0 and total_flagged >= self.n_records:
            raise ValueError(
                "FAILED requires decisive flags for fewer records than presented"
            )
        derived = derive_qc_aggregate_condition(fc)
        if self.aggregate_condition is not derived:
            raise ValueError(
                f"aggregate_condition {self.aggregate_condition} does not match "
                f"the flag-count-derived condition {derived}"
            )
        return self

    model_config = _FROZEN


class DetectorComponents(BaseModel):
    """Separate raw detector component scores."""

    threshold_score: float = Field(ge=0.0, le=1.0)
    wavelet_score: float = Field(ge=0.0, le=1.0)
    bocpd_score: float = Field(ge=0.0, le=1.0)
    statistical_score: float = Field(ge=0.0, le=1.0)
    ml_score: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = _FROZEN


class ThresholdEvaluation(BaseModel):
    """Threshold comparison result for one station window.

    Comparison is inclusive (score >= threshold), matching the FSM. A
    successful numerical score whose evaluation status is NOT_EVALUATED
    keeps its number for engineering diagnostics but has result
    NOT_EVALUATED and is not scientific evidence.
    """

    source: DataSource
    station_id: str = Field(pattern=STATION_ID_PATTERN)
    result: ThresholdResultValue
    ensemble_score: float | None = Field(default=None, ge=0.0, le=1.0)
    t1_investigate: float = Field(ge=0.0, le=1.0)
    t2_assess: float = Field(ge=0.0, le=1.0)
    t3_escalate: float = Field(ge=0.0, le=1.0)
    highest_tier_met: Literal["T1", "T2", "T3"] | None = None
    evaluated_window: ObservationBounds | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ThresholdEvaluation:
        met = ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET
        no_met = (
            ThresholdResultValue.NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW
        )
        if self.result is met:
            if self.ensemble_score is None or self.evaluated_window is None:
                raise ValueError("A met result requires a score and window")
            if self.ensemble_score < self.t1_investigate:
                raise ValueError("A met result requires score >= t1 (inclusive)")
            if self.highest_tier_met is None:
                raise ValueError("A met result requires highest_tier_met")
            expected = derive_highest_tier(
                self.ensemble_score,
                self.t1_investigate,
                self.t2_assess,
                self.t3_escalate,
            )
            if self.highest_tier_met != expected:
                raise ValueError(
                    f"highest_tier_met {self.highest_tier_met} does not match "
                    f"score-derived tier {expected}"
                )
        elif self.result is no_met:
            if self.ensemble_score is None or self.evaluated_window is None:
                raise ValueError("A no-crossing result requires a score and window")
            if self.ensemble_score >= self.t1_investigate:
                raise ValueError("A no-crossing result requires score < t1")
            if self.highest_tier_met is not None:
                raise ValueError("A no-crossing result cannot carry a tier")
        else:  # NOT_EVALUATED keeps a diagnostic score but no tier claim.
            if self.highest_tier_met is not None:
                raise ValueError("NOT_EVALUATED cannot carry highest_tier_met")
        return self

    model_config = _FROZEN


class ConfounderCheck(BaseModel):
    """One named confounder check."""

    name: str = Field(min_length=1, max_length=64)
    applicability: ConfounderApplicability
    prerequisite_status: ConfounderPrerequisiteStatus
    result: ConfounderResultValue

    @model_validator(mode="after")
    def _consistent(self) -> ConfounderCheck:
        if self.applicability is ConfounderApplicability.NOT_APPLICABLE:
            if self.result is not ConfounderResultValue.NOT_APPLICABLE:
                raise ValueError(
                    "NOT_APPLICABLE applicability requires the matching result"
                )
            return self
        # Applicable checks.
        if self.result is ConfounderResultValue.NOT_APPLICABLE:
            raise ValueError("An applicable check cannot report NOT_APPLICABLE")
        if self.prerequisite_status is not ConfounderPrerequisiteStatus.AVAILABLE:
            if self.result is not ConfounderResultValue.NOT_EVALUATED:
                raise ValueError(
                    "Missing or errored prerequisites require NOT_EVALUATED"
                )
        elif self.result is ConfounderResultValue.NOT_EVALUATED:
            # Allowed: prerequisites available but the check did not run.
            pass
        return self

    model_config = _FROZEN


class StationLimitation(BaseModel):
    """One named limitation with its frozen evaluation effect."""

    code: LimitationCode
    effect: EvaluationEffect
    detail: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _canonical_effect(self) -> StationLimitation:
        canonical = LIMITATION_CODE_EFFECTS[self.code]
        if self.effect is not canonical:
            raise ValueError(
                f"Limitation {self.code} must carry its registry effect "
                f"{canonical}, not {self.effect}"
            )
        return self

    model_config = _FROZEN


class RetainedRecordRef(BaseModel):
    """Composite identity of one retained record.

    Hash-only joins are insufficient when equal content can occur at
    different stations, so references carry source, station, time,
    measurement type or product, and payload hash.
    """

    source: DataSource
    station_id: str = Field(pattern=STATION_ID_PATTERN)
    observed_at_utc: AwareDatetime
    measurement_type: Literal[1, 2, 3] | None = Field(
        default=None, description="DART measurement type; None for non-DART"
    )
    product: str | None = Field(
        default=None, max_length=64, description="CO-OPS product; None for DART"
    )
    payload_sha256: str = Field(pattern=SHA256_PATTERN)

    model_config = _FROZEN


class StationAssessmentEntry(BaseModel):
    """Deterministic account of one source-qualified station.

    The three status axes are orthogonal and never overwrite one
    another: a rejected current record can coexist with a successfully
    scored older retained window.
    """

    source: DataSource
    station_id: str = Field(pattern=STATION_ID_PATTERN)

    admission_status: CurrentRecordAdmissionStatus
    n_records_attempted: int = Field(ge=0)
    n_records_admitted: int = Field(ge=0)
    n_records_rejected: int = Field(ge=0)

    scoring_status: StationScoringStatus
    evaluation_status: StationEvaluationStatus

    observation_bounds: ObservationBounds | None = Field(
        default=None, description="Bounds of the retained window, when nonempty"
    )
    n_retained_samples: int = Field(ge=0)
    median_cadence_sec: float | None = Field(default=None, gt=0.0)
    latest_observation_utc: AwareDatetime | None = None
    operational_age_at_production_sec: float | None = Field(
        default=None,
        ge=0.0,
        description=(
            "Age of the latest observation at production time. Excluded "
            "from the scientific content hash."
        ),
    )

    current_dart_data_mode: DartDataMode | None = Field(
        default=None,
        description=(
            "DART transmission mode from the newest retained accepted "
            "observation. Event mode means elevated scrutiny, not "
            "tsunami confirmation. None for non-DART stations."
        ),
    )
    coops_products: list[str] = Field(
        default_factory=list,
        description="Sorted CO-OPS products present in the retained window",
    )

    qc_retained_window: RetainedWindowQC | None = Field(
        default=None,
        description="Absent only when the station has no retained data",
    )
    calibration_status: CalibrationStatus
    calibration_sha256: str = Field(
        default="", pattern=OPTIONAL_SHA256_PATTERN,
        description="Hash of the calibration artifact, when one exists",
    )

    cadence_condition: CadenceCondition
    filtering_condition: FilteringCondition
    mixed_product_condition: MixedProductCondition
    manifest_condition: StationManifestCondition

    detector_components: DetectorComponents | None = None
    threshold_evaluation: ThresholdEvaluation | None = None
    confounder_checks: list[ConfounderCheck] = Field(default_factory=list)
    retained_record_refs: list[RetainedRecordRef] = Field(
        default_factory=list,
        description="Ordered by observation time; exact retained identities",
    )

    source_validation_status: SourceValidationStatus
    provenance_status: ProvenanceStatus
    limitations: list[StationLimitation] = Field(
        default_factory=list,
        description="Exactly the registry-derived codes, sorted by code",
    )
    failure_reason: str = Field(default="", max_length=512)

    @model_validator(mode="after")
    def _consistent(self) -> StationAssessmentEntry:
        self._check_source_fields()
        self._check_admission()
        self._check_retained_window()
        self._check_confounders_sorted()
        self._check_limitations()
        self._check_evaluation_status()
        self._check_threshold()
        return self

    def _check_source_fields(self) -> None:
        if self.source is DataSource.DART:
            if self.current_dart_data_mode is None:
                raise ValueError("DART entries require current_dart_data_mode")
            if self.coops_products:
                raise ValueError("DART entries cannot carry coops_products")
            if self.mixed_product_condition is not MixedProductCondition.NOT_APPLICABLE:
                raise ValueError(
                    "DART entries use mixed_product_condition NOT_APPLICABLE"
                )
        elif self.source is DataSource.COOPS:
            if self.current_dart_data_mode is not None:
                raise ValueError("CO-OPS entries cannot carry a DART data mode")
            if self.coops_products != sorted(set(self.coops_products)):
                raise ValueError("coops_products must be sorted and unique")
            if self.mixed_product_condition is MixedProductCondition.NOT_APPLICABLE:
                raise ValueError(
                    "CO-OPS entries must state SINGLE_PRODUCT or MIXED_PRODUCTS"
                )
            if (
                len(self.coops_products) > 1
                and self.mixed_product_condition
                is not MixedProductCondition.MIXED_PRODUCTS
            ):
                raise ValueError(
                    "Multiple retained products require MIXED_PRODUCTS"
                )
        else:
            raise ValueError(f"Station entries cannot use source {self.source}")

    def _check_admission(self) -> None:
        if self.n_records_admitted + self.n_records_rejected != self.n_records_attempted:
            raise ValueError("admitted + rejected must equal attempted")
        s = self.admission_status
        if s is CurrentRecordAdmissionStatus.NO_RECORD_ATTEMPT:
            ok = self.n_records_attempted == 0
        elif s is CurrentRecordAdmissionStatus.ALL_RECORDS_REJECTED:
            ok = self.n_records_attempted > 0 and self.n_records_admitted == 0
        elif s is CurrentRecordAdmissionStatus.SOME_RECORDS_ADMITTED:
            ok = self.n_records_admitted > 0 and self.n_records_rejected > 0
        else:
            ok = self.n_records_attempted > 0 and self.n_records_rejected == 0
        if not ok:
            raise ValueError(f"Admission counts inconsistent with status {s}")

    def _check_retained_window(self) -> None:
        no_data = self.scoring_status is StationScoringStatus.NO_RETAINED_DATA
        if no_data:
            if self.n_retained_samples != 0 or self.observation_bounds is not None:
                raise ValueError("NO_RETAINED_DATA requires an empty window")
            if self.qc_retained_window is not None:
                raise ValueError(
                    "Retained-window QC is absent when no retained data exists"
                )
            if self.retained_record_refs:
                raise ValueError("NO_RETAINED_DATA cannot carry record refs")
        else:
            if self.n_retained_samples == 0 or self.observation_bounds is None:
                raise ValueError(
                    f"{self.scoring_status} requires a nonempty retained window"
                )
            if self.qc_retained_window is None:
                raise ValueError(
                    "Retained-window QC is defined for insufficient-data and "
                    "scoring-failure cases; only NO_RETAINED_DATA omits it"
                )
            if self.qc_retained_window.n_records == 0:
                raise ValueError(
                    "Retained-window QC must aggregate at least one record"
                )
            if self.qc_retained_window.observation_bounds != self.observation_bounds:
                raise ValueError(
                    "Retained-window QC bounds must equal the retained window "
                    "bounds (the exact retained eligibility window)"
                )
        if self.scoring_status is StationScoringStatus.SCORING_FAILED:
            if not self.failure_reason:
                raise ValueError("SCORING_FAILED requires failure_reason")
        times = [r.observed_at_utc for r in self.retained_record_refs]
        if times != sorted(times):
            raise ValueError("retained_record_refs must be ordered by time")

    def _check_confounders_sorted(self) -> None:
        names = [c.name for c in self.confounder_checks]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("confounder_checks must be sorted and unique by name")

    def _expected_limitation_codes(self) -> list[tuple[LimitationCode, EvaluationEffect]]:
        conditions: list[tuple[str, str]] = [
            ("CalibrationStatus", self.calibration_status.value),
            ("CadenceCondition", self.cadence_condition.value),
            ("FilteringCondition", self.filtering_condition.value),
            ("SourceValidationStatus", self.source_validation_status.value),
            ("ProvenanceStatus", self.provenance_status.value),
            ("StationManifestCondition", self.manifest_condition.value),
            ("MixedProductCondition", self.mixed_product_condition.value),
        ]
        if self.qc_retained_window is None:
            conditions.append(
                ("QCAggregateCondition", QCAggregateCondition.NO_RETAINED_QC.value)
            )
        else:
            conditions.append(
                (
                    "QCExecutionStatus",
                    self.qc_retained_window.execution_status.value,
                )
            )
            conditions.append(
                (
                    "QCAggregateCondition",
                    self.qc_retained_window.aggregate_condition.value,
                )
            )
        for check in self.confounder_checks:
            conditions.append(("ConfounderResultValue", check.result.value))
        seen: dict[LimitationCode, EvaluationEffect] = {}
        for axis, value in conditions:
            mapping = registry_lookup(axis, value)
            if mapping is not None:
                seen[mapping[0]] = mapping[1]
        return sorted(seen.items(), key=lambda kv: kv[0].value)

    def _check_limitations(self) -> None:
        expected = self._expected_limitation_codes()
        actual = [(lim.code, lim.effect) for lim in self.limitations]
        if actual != expected:
            raise ValueError(
                "limitations must be exactly the registry-derived set: "
                f"expected {[c.value for c, _ in expected]}, "
                f"got {[c.value for c, _ in actual]}"
            )

    def _check_evaluation_status(self) -> None:
        derived = derive_evaluation_status(
            self.scoring_status, (lim.effect for lim in self.limitations)
        )
        if self.evaluation_status is not derived:
            raise ValueError(
                f"evaluation_status {self.evaluation_status} does not match "
                f"derived status {derived}"
            )

    def _check_threshold(self) -> None:
        succeeded = self.scoring_status is StationScoringStatus.SCORING_SUCCEEDED
        if not succeeded:
            if self.threshold_evaluation is not None:
                raise ValueError(
                    "threshold_evaluation requires SCORING_SUCCEEDED"
                )
            if self.detector_components is not None:
                raise ValueError(
                    "detector_components requires SCORING_SUCCEEDED"
                )
            return
        te = self.threshold_evaluation
        if te is None or self.detector_components is None:
            raise ValueError(
                "SCORING_SUCCEEDED requires threshold_evaluation and "
                "detector_components"
            )
        if te.source is not self.source or te.station_id != self.station_id:
            raise ValueError("threshold_evaluation identity must match the entry")
        if (
            te.evaluated_window is not None
            and te.evaluated_window != self.observation_bounds
        ):
            raise ValueError(
                "threshold_evaluation must carry the evaluated retained "
                "window, matching the entry's observation bounds"
            )
        expected = derive_threshold_result_value(
            self.evaluation_status, te.ensemble_score, te.t1_investigate
        )
        if te.result is not expected:
            raise ValueError(
                f"threshold result {te.result} does not match derived "
                f"result {expected}"
            )

    model_config = _FROZEN


class SourceRollup(BaseModel):
    """Per-source threshold rollup over evaluated stations.

    A no-crossing rollup is a statement about evaluated windows only.
    Under OBSERVED_RECORDS_ONLY scope it carries the network-coverage
    limitation and must not be rendered as source-network silence or
    absence of a tsunami signal.
    """

    source: DataSource
    status: SourceRollupStatus
    n_evaluated_stations: int = Field(ge=0)
    evaluated_station_ids: list[str] = Field(default_factory=list)
    crossed_station_ids: list[str] = Field(default_factory=list)
    limitations: list[StationLimitation] = Field(default_factory=list)

    @model_validator(mode="after")
    def _consistent(self) -> SourceRollup:
        if self.evaluated_station_ids != sorted(set(self.evaluated_station_ids)):
            raise ValueError("evaluated_station_ids must be sorted and unique")
        if self.crossed_station_ids != sorted(set(self.crossed_station_ids)):
            raise ValueError("crossed_station_ids must be sorted and unique")
        if len(self.evaluated_station_ids) != self.n_evaluated_stations:
            raise ValueError("n_evaluated_stations must match the identity list")
        if not set(self.crossed_station_ids) <= set(self.evaluated_station_ids):
            raise ValueError("crossed stations must be evaluated stations")
        met = SourceRollupStatus.SOURCE_CONFIGURED_THRESHOLD_MET
        no_met = (
            SourceRollupStatus.SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS
        )
        if self.status is met and not self.crossed_station_ids:
            raise ValueError("A met rollup requires at least one crossed station")
        if self.status is no_met:
            if not self.evaluated_station_ids:
                raise ValueError(
                    "A no-crossing rollup requires at least one evaluated station"
                )
            if self.crossed_station_ids:
                raise ValueError("A no-crossing rollup cannot list crossings")
        if self.status is SourceRollupStatus.SOURCE_NOT_EVALUATED and (
            self.evaluated_station_ids or self.crossed_station_ids
        ):
            raise ValueError("SOURCE_NOT_EVALUATED requires empty station lists")
        return self

    model_config = _FROZEN


class AnalysisStatus(BaseModel):
    """Capability, execution, and quality of one analysis."""

    capability: AnalysisCapability
    execution: AnalysisExecution
    quality: AnalysisQuality

    @model_validator(mode="after")
    def _consistent(self) -> AnalysisStatus:
        if (
            self.capability is AnalysisCapability.NOT_IMPLEMENTED
            and self.execution is AnalysisExecution.SUCCEEDED
        ):
            raise ValueError("NOT_IMPLEMENTED analyses cannot succeed")
        if (
            self.quality is not AnalysisQuality.NOT_ASSESSED
            and self.execution is not AnalysisExecution.SUCCEEDED
        ):
            raise ValueError("Quality requires successful execution")
        return self

    model_config = _FROZEN


class AnalysisStatusSet(BaseModel):
    """Explicit status for every named downstream analysis.

    Missing objects and placeholder zeros are forbidden; each analysis
    states its capability, execution, and quality explicitly.
    """

    scenario: AnalysisStatus
    scenario_verification: AnalysisStatus
    coastal_arrival: AnalysisStatus
    coastal_amplitude: AnalysisStatus
    inundation: AnalysisStatus
    meteorological_source_discrimination: AnalysisStatus
    spatial_coherence: AnalysisStatus

    model_config = _FROZEN


class ProvenanceSummary(BaseModel):
    """Checkpoint-level provenance accounting."""

    status: ProvenanceStatus
    n_references_expected: int = Field(ge=0)
    n_references_included: int = Field(ge=0)
    n_unresolved_raw_records: int = Field(ge=0)
    n_malformed_or_absent_hashes: int = Field(ge=0)
    references_capped: bool = False
    calibration_provenance_available: bool
    database_available: bool
    companion_persistence_failures: list[str] = Field(
        default_factory=list,
        description="Sorted names of companion features whose writes failed",
    )

    @model_validator(mode="after")
    def _consistent(self) -> ProvenanceSummary:
        if self.n_references_included > self.n_references_expected:
            raise ValueError("included references cannot exceed expected")
        failures = self.companion_persistence_failures
        if failures != sorted(failures):
            raise ValueError("companion_persistence_failures must be sorted")
        if self.status is ProvenanceStatus.RESOLVED and (
            self.n_references_included != self.n_references_expected
            or self.n_unresolved_raw_records != 0
            or self.n_malformed_or_absent_hashes != 0
        ):
            raise ValueError(
                "RESOLVED requires every expected reference included with no "
                "unresolved raw records or malformed hashes"
            )
        if (
            self.status is ProvenanceStatus.UNAVAILABLE
            and self.n_references_included != 0
        ):
            raise ValueError("UNAVAILABLE requires zero included references")
        return self

    model_config = _FROZEN


# ---------------------------------------------------------------------------
# Deterministic derivations (single source of truth for validators)
# ---------------------------------------------------------------------------


def derive_qc_aggregate_condition(
    flag_counts: QCFlagCounts,
) -> QCAggregateCondition:
    """Decisive worst QARTOD flag over the retained window.

    The severity order is frozen as FAIL > MISSING > SUSPECT >
    NOT_EVALUATED > PASS, consistent with the registry effects: FAIL and
    MISSING limit interpretation, SUSPECT and NOT_EVALUATED are
    informational, PASS adds nothing. With no decisively flagged records
    there is no worst flag and the aggregate is NO_RETAINED_QC.
    """
    if flag_counts.n_fail > 0:
        return QCAggregateCondition.WORST_FLAG_FAIL
    if flag_counts.n_missing > 0:
        return QCAggregateCondition.WORST_FLAG_MISSING
    if flag_counts.n_suspect > 0:
        return QCAggregateCondition.WORST_FLAG_SUSPECT
    if flag_counts.n_not_evaluated > 0:
        return QCAggregateCondition.WORST_FLAG_NOT_EVALUATED
    if flag_counts.n_pass > 0:
        return QCAggregateCondition.WORST_FLAG_PASS
    return QCAggregateCondition.NO_RETAINED_QC


def derive_source_time_bounds(
    entries: Iterable[StationAssessmentEntry],
) -> ObservationBounds | None:
    """Observation-time bounds over all retained station windows."""
    bounds = [
        e.observation_bounds for e in entries if e.observation_bounds is not None
    ]
    if not bounds:
        return None
    return ObservationBounds(
        first_observation_utc=min(b.first_observation_utc for b in bounds),
        last_observation_utc=max(b.last_observation_utc for b in bounds),
    )


def derive_evaluation_status(
    scoring_status: StationScoringStatus,
    effects: Iterable[EvaluationEffect],
) -> StationEvaluationStatus:
    """Derive scientific interpretability.

    A non-successful score is always NOT_EVALUATED. For a successful
    score, any preventing limitation yields NOT_EVALUATED; otherwise
    any limiting code yields EVALUATED_WITH_LIMITATIONS; otherwise
    EVALUATED.
    """
    if scoring_status is not StationScoringStatus.SCORING_SUCCEEDED:
        return StationEvaluationStatus.NOT_EVALUATED
    effect_set = set(effects)
    if EvaluationEffect.PREVENTS_EVALUATION in effect_set:
        return StationEvaluationStatus.NOT_EVALUATED
    if EvaluationEffect.LIMITS_INTERPRETATION in effect_set:
        return StationEvaluationStatus.EVALUATED_WITH_LIMITATIONS
    return StationEvaluationStatus.EVALUATED


def derive_threshold_result_value(
    evaluation_status: StationEvaluationStatus,
    ensemble_score: float | None,
    t1: float,
) -> ThresholdResultValue:
    """Derive the threshold result (inclusive comparison).

    Never derived from triggering-station lists. A NOT_EVALUATED
    evaluation keeps any diagnostic score but is not scientific evidence.
    """
    if (
        evaluation_status is StationEvaluationStatus.NOT_EVALUATED
        or ensemble_score is None
    ):
        return ThresholdResultValue.NOT_EVALUATED
    if ensemble_score >= t1:
        return ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET
    return (
        ThresholdResultValue.NO_CONFIGURED_ENSEMBLE_THRESHOLD_MET_IN_EVALUATED_WINDOW
    )


def derive_highest_tier(
    score: float, t1: float, t2: float, t3: float
) -> Literal["T1", "T2", "T3"] | None:
    """Highest configured tier the score meets, inclusive."""
    if score >= t3:
        return "T3"
    if score >= t2:
        return "T2"
    if score >= t1:
        return "T1"
    return None


def derive_source_rollup(
    source: DataSource,
    entries: Iterable[StationAssessmentEntry],
    station_scope: StationScope,
) -> SourceRollup:
    """Derive one source rollup from station entries."""
    evaluated: list[str] = []
    crossed: list[str] = []
    for entry in entries:
        if entry.source is not source:
            continue
        if entry.evaluation_status is StationEvaluationStatus.NOT_EVALUATED:
            continue
        evaluated.append(entry.station_id)
        te = entry.threshold_evaluation
        if (
            te is not None
            and te.result is ThresholdResultValue.CONFIGURED_ENSEMBLE_THRESHOLD_MET
        ):
            crossed.append(entry.station_id)
    evaluated = sorted(set(evaluated))
    crossed = sorted(set(crossed))
    if crossed:
        status = SourceRollupStatus.SOURCE_CONFIGURED_THRESHOLD_MET
    elif evaluated:
        status = (
            SourceRollupStatus.SOURCE_NO_CONFIGURED_THRESHOLD_MET_IN_EVALUATED_WINDOWS
        )
    else:
        status = SourceRollupStatus.SOURCE_NOT_EVALUATED
    limitations: list[StationLimitation] = []
    scope_mapping = registry_lookup("StationScope", station_scope.value)
    if scope_mapping is not None:
        code, effect = scope_mapping
        limitations.append(StationLimitation(code=code, effect=effect))
    return SourceRollup(
        source=source,
        status=status,
        n_evaluated_stations=len(evaluated),
        evaluated_station_ids=evaluated,
        crossed_station_ids=crossed,
        limitations=limitations,
    )


def derive_pipeline_outcome(
    outcome_field: str | None,
    fsm_state_after: FsmState,
    seismic_only_no_score: bool = False,
) -> PipelineOutcome:
    """Exhaustively classify the checkpoint outcome.

    Reads only the typed outcome field of the deterministic pipeline
    result (current values: ``abstain``, ``insufficient_evidence``,
    ``verified_pending_report``) plus the post-evaluation FSM state.
    Every unrecognized or unexpected combination is
    PROCESSING_INCOMPLETE, which cannot dispatch the model. An
    ``insufficient_evidence`` result with post-evaluation state IDLE is
    classified PROCESSING_INCOMPLETE as a deliberate fail-safe, not a
    monitoring claim for a resolved event.
    """
    if seismic_only_no_score:
        return PipelineOutcome.ABSTAIN
    if outcome_field == "abstain":
        return PipelineOutcome.ABSTAIN
    if outcome_field == "insufficient_evidence" and fsm_state_after in (
        FsmState.MONITOR,
        FsmState.INVESTIGATE,
    ):
        return PipelineOutcome.MONITORING_CONTINUES
    return PipelineOutcome.PROCESSING_INCOMPLETE


# ---------------------------------------------------------------------------
# Top-level assessment
# ---------------------------------------------------------------------------


class OceanEvidenceAssessment(BaseEnvelope):
    """Immutable deterministic account of one eligible checkpoint.

    Emitted only when the FSM has an active event after seismic
    processing. Ocean-only IDLE batches emit no assessment and mint no
    event UUID. The assessment is one internally coherent row assembled
    from worker-owned in-memory results; companion QC, anomaly, FSM,
    and audit persistence remains separately disclosed and is not one
    atomic snapshot.
    """

    type: Literal["OceanEvidenceAssessment"] = "OceanEvidenceAssessment"
    assessment_schema_version: Literal[1] = ASSESSMENT_SCHEMA_VERSION
    condition_registry_version: Literal["1.0.0"] = CONDITION_REGISTRY_VERSION
    """Pinned: an artifact validates only under the registry that
    produced it, because the limitation validators re-derive against the
    current registry."""

    checkpoint_id: str = Field(
        pattern=SHA256_PATTERN,
        description=(
            "Deterministic checkpoint identity, distinct from "
            "random trace and handoff IDs"
        ),
    )
    checkpoint_source: CheckpointSource

    seismic_context: EventSeismicContext
    contributing_trace_ids: list[UUID] = Field(
        default_factory=list,
        description="Trace IDs contributing to this checkpoint, sorted",
    )
    source_time_bounds: ObservationBounds | None = Field(
        default=None,
        description="Observation-time bounds over all retained windows",
    )

    station_scope: StationScope
    station_manifest: StationManifestRef | None = None
    detector_config: DetectorConfig
    stations: list[StationAssessmentEntry] = Field(default_factory=list)

    dart_rollup: SourceRollup
    coops_rollup: SourceRollup

    fsm_state_before: FsmState
    fsm_state_after: FsmState
    fsm_state_changed: bool
    fsm_transition_ref: str = Field(
        default="", max_length=128,
        description="Audit reference of the transition, when one occurred",
    )

    dart_stations_currently_in_event_mode: list[str] = Field(
        default_factory=list,
        description=(
            "Sorted canonical IDs of DART stations whose newest retained "
            "accepted observation is event mode. Elevated scrutiny, not "
            "tsunami confirmation."
        ),
    )
    dart_event_mode_observed_since_event_origin: bool = Field(
        description=(
            "True when any event-mode record was accepted while this "
            "event was active with an observation timestamp at or after "
            "seismic origin. This scoping is temporal and event-activity "
            "based, not causal source attribution: a post-origin "
            "event-mode record from an unrelated disturbance still "
            "counts while the event is active, deliberately fail-safe "
            "until multi-event tracking exists. Not tsunami confirmation."
        ),
    )
    dart_event_mode_stations_since_event_origin: list[str] = Field(
        default_factory=list,
        description=(
            "Sorted stations contributing to the event-lifetime flag "
            "above, under the same temporal (not causal) scoping. Resets "
            "with event context."
        ),
    )

    confounder_checks: list[ConfounderCheck] = Field(
        default_factory=list,
        description="Checkpoint-level named confounder checks, sorted",
    )
    analyses: AnalysisStatusSet
    pipeline_outcome: PipelineOutcome
    provenance: ProvenanceSummary

    input_manifest_hash: str = Field(
        default="", pattern=OPTIONAL_SHA256_PATTERN,
        description="Hash of the exact scientific inputs",
    )
    scientific_content_hash: str = Field(
        default="", pattern=OPTIONAL_SHA256_PATTERN,
        description="Hash of normalized scientific facts",
    )
    transport_provenance_hash: str = Field(
        default="", pattern=OPTIONAL_SHA256_PATTERN,
        description="Hash of creation-attempt transport data",
    )

    @model_validator(mode="after")
    def _consistent(self) -> OceanEvidenceAssessment:
        if self.event_id is None:
            raise ValueError("Assessments require an active internal event UUID")
        self._check_stations_sorted()
        self._check_scope()
        self._check_fsm()
        self._check_dart_mode_fields()
        self._check_rollups()
        self._check_bounds_and_config()
        self._check_traces_and_confounders()
        return self

    def _check_stations_sorted(self) -> None:
        keys = [(s.source.value, s.station_id) for s in self.stations]
        if keys != sorted(keys) or len(keys) != len(set(keys)):
            raise ValueError(
                "stations must be sorted and unique by (source, station_id)"
            )

    def _check_scope(self) -> None:
        has_manifest = self.station_manifest is not None
        if (self.station_scope is StationScope.CONFIGURED_INVENTORY) != has_manifest:
            raise ValueError(
                "CONFIGURED_INVENTORY requires a manifest and "
                "OBSERVED_RECORDS_ONLY forbids one"
            )
        if not has_manifest:
            for s in self.stations:
                if s.manifest_condition is not StationManifestCondition.NO_MANIFEST:
                    raise ValueError(
                        "Without a manifest every entry uses NO_MANIFEST"
                    )
        else:
            for s in self.stations:
                if s.manifest_condition is StationManifestCondition.NO_MANIFEST:
                    raise ValueError(
                        "With a manifest every entry states IN_MANIFEST or "
                        "OUT_OF_MANIFEST"
                    )

    def _check_fsm(self) -> None:
        if self.fsm_state_changed != (self.fsm_state_before != self.fsm_state_after):
            raise ValueError("fsm_state_changed must match the state pair")

    def _check_dart_mode_fields(self) -> None:
        expected_current = sorted(
            s.station_id
            for s in self.stations
            if s.source is DataSource.DART
            and s.current_dart_data_mode is DartDataMode.EVENT
        )
        if self.dart_stations_currently_in_event_mode != expected_current:
            raise ValueError(
                "dart_stations_currently_in_event_mode must be derived from "
                "station entries"
            )
        lifetime = self.dart_event_mode_stations_since_event_origin
        if lifetime != sorted(set(lifetime)):
            raise ValueError("Event-lifetime station list must be sorted and unique")
        for sid in lifetime:
            if not _STATION_ID_RE.fullmatch(sid):
                raise ValueError(
                    f"Invalid station id in event-lifetime list: {sid!r}"
                )
        if self.dart_event_mode_observed_since_event_origin != bool(lifetime):
            raise ValueError(
                "Event-lifetime flag must match its station list"
            )

    def _check_rollups(self) -> None:
        expected_dart = derive_source_rollup(
            DataSource.DART, self.stations, self.station_scope
        )
        expected_coops = derive_source_rollup(
            DataSource.COOPS, self.stations, self.station_scope
        )
        if self.dart_rollup != expected_dart:
            raise ValueError("dart_rollup does not match the derived rollup")
        if self.coops_rollup != expected_coops:
            raise ValueError("coops_rollup does not match the derived rollup")

    def _check_bounds_and_config(self) -> None:
        expected_bounds = derive_source_time_bounds(self.stations)
        if self.source_time_bounds != expected_bounds:
            raise ValueError(
                "source_time_bounds must derive from the retained station "
                "windows"
            )
        dc = self.detector_config
        for s in self.stations:
            te = s.threshold_evaluation
            if te is None:
                continue
            if (
                te.t1_investigate != dc.t1_investigate
                or te.t2_assess != dc.t2_assess
                or te.t3_escalate != dc.t3_escalate
            ):
                raise ValueError(
                    "Station threshold evaluations must carry the checkpoint "
                    "detector configuration thresholds"
                )

    def _check_traces_and_confounders(self) -> None:
        traces = [str(t) for t in self.contributing_trace_ids]
        if traces != sorted(traces) or len(traces) != len(set(traces)):
            raise ValueError("contributing_trace_ids must be sorted and unique")
        names = [c.name for c in self.confounder_checks]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError(
                "confounder_checks must be sorted and unique by name"
            )

    model_config = {"extra": "forbid", "frozen": True}
