"""Structured handoff schemas for inter-agent communication.

All inter-agent communication uses a common base envelope with typed payloads.
Schema versioning is mandatory. Breaking changes require a version bump.

Schema rollout order:
1. BaseEnvelope
2. QCReport
3. AnomalyAssessment
4. ScenarioAssessment
5. VerificationResult
6. EscalationPacket
7. HumanDecision
8. FinalAssessment
"""

from hazard_assessment.schemas.anomaly import (
    AnomalyAssessment,
    ScoreComponents,
    SpatialConfirmation,
)
from hazard_assessment.schemas.envelope import (
    AwareDatetime,
    BaseEnvelope,
    DataSource,
    DecisionStep,
    InputRef,
    StepResult,
)
from hazard_assessment.schemas.escalation import EscalationPacket
from hazard_assessment.schemas.final_assessment import (
    AssessmentStatus,
    ConfidenceLevel,
    FinalAssessment,
    UncertaintyInfo,
)
from hazard_assessment.schemas.human_decision import (
    AssessmentReviewDecision,
    HumanDecision,
    ReviewDecision,
)
from hazard_assessment.schemas.observation import (
    BaseObservation,
    CoopsObservation,
    DartObservation,
    SeismicObservation,
)

# NOTE: hazard_assessment.schemas.ocean_evidence_hashing is deliberately
# not re-exported here. It imports hazard_assessment.ingest.hashing, whose
# package init reaches config.settings and orchestrator.states, which in
# turn import this package: re-exporting it here would close an import
# cycle. Import that module directly.
from hazard_assessment.schemas.ocean_evidence import (
    ASSESSMENT_SCHEMA_VERSION,
    CONDITION_REGISTRY_VERSION,
    OceanEvidenceAssessment,
    PipelineOutcome,
    StationAssessmentEntry,
)
from hazard_assessment.schemas.qc import DataMode, QARTODFlag, QARTODFlags, QCReport
from hazard_assessment.schemas.scenario import (
    CoastalProxy,
    ConstraintStage,
    EnsembleSpread,
    RankedScenario,
    ScenarioAssessment,
)
from hazard_assessment.schemas.verification import (
    CheckApplicability,
    CheckResult,
    PrerequisiteStatus,
    VerificationCheck,
    VerificationOutcome,
    VerificationResult,
)
from hazard_assessment.schemas.versioning import (
    SchemaVersionError,
    check_schema_version,
)

__all__ = [
    "ASSESSMENT_SCHEMA_VERSION",
    "AnomalyAssessment",
    "AssessmentReviewDecision",
    "AssessmentStatus",
    "AwareDatetime",
    "BaseEnvelope",
    "BaseObservation",
    "CONDITION_REGISTRY_VERSION",
    "CheckApplicability",
    "CheckResult",
    "CoastalProxy",
    "ConfidenceLevel",
    "ConstraintStage",
    "CoopsObservation",
    "DartObservation",
    "DataMode",
    "DataSource",
    "DecisionStep",
    "EnsembleSpread",
    "EscalationPacket",
    "FinalAssessment",
    "HumanDecision",
    "InputRef",
    "OceanEvidenceAssessment",
    "PipelineOutcome",
    "PrerequisiteStatus",
    "QARTODFlag",
    "QARTODFlags",
    "QCReport",
    "RankedScenario",
    "ReviewDecision",
    "ScenarioAssessment",
    "SchemaVersionError",
    "ScoreComponents",
    "SeismicObservation",
    "SpatialConfirmation",
    "StationAssessmentEntry",
    "StepResult",
    "UncertaintyInfo",
    "VerificationCheck",
    "VerificationOutcome",
    "VerificationResult",
    "check_schema_version",
]
