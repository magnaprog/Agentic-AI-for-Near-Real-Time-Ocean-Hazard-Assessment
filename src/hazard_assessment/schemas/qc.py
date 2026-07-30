"""QC Report schema - output of the QC Agent.

The QC Agent applies QARTOD-aligned tests to each incoming observation record
and produces per-record flags and station confidence scores.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, Field

from hazard_assessment.schemas.envelope import AwareDatetime, BaseEnvelope


class QARTODFlag(IntEnum):
    """IOOS QARTOD flag convention (version 1.2).

    Standard flags per the IOOS manual are 1 (Pass), 2 (Not Evaluated),
    3 (Suspect), 4 (Fail), and 9 (Missing).  Flag 0 (NOT_APPLICABLE) is
    a local extension used by this system to mark tests that do not apply
    to a given station type (e.g., neighbor_consistency on CO-OPS stations).

    Note: the ioos_qc Python library uses synonymous names for the same
    numeric values - GOOD (1), UNKNOWN (2), SUSPECT (3), FAIL (4),
    MISSING (9). The numeric values are what matter for interoperability;
    this enum follows the QARTOD manual's naming convention.

    Reference: https://cdn.ioos.noaa.gov/media/2020/07/QARTOD-Data-Flags-Manual_version1.2final.pdf
    """

    NOT_APPLICABLE = 0  # Local extension (not in IOOS standard)
    PASS = 1
    NOT_EVALUATED = 2
    SUSPECT = 3
    FAIL = 4
    MISSING = 9


class DataMode(StrEnum):
    """Station operating mode.

    DART stations switch between STANDARD and EVENT modes.
    Non-DART stations (e.g., CO-OPS) are always STANDARD.
    """

    STANDARD = "STANDARD"
    EVENT = "EVENT"


class QARTODFlags(BaseModel):
    """Per-test QARTOD flag results for a single observation.

    Standard QARTOD water-level tests (Timing/Gap, Gross Range,
    Spike, Rate of Change, Flat Line) are defined in the QARTOD
    Water Level Manual, and ``run_all_checks`` evaluates those five.
    The remaining fields (neighbor_consistency, mode_transition,
    latency) are local extensions: latency currently just copies the
    timing result, and the other two are reserved slots that no check
    computes.
    """

    # --- Standard QARTOD water-level tests ---
    timing: QARTODFlag = Field(description="Timing integrity check")
    range: QARTODFlag = Field(description="Physical plausibility range check")
    rate_of_change: QARTODFlag = Field(description="Rate of change check")
    spike: QARTODFlag = Field(description="Spike test")
    flat_line: QARTODFlag = Field(
        default=QARTODFlag.NOT_APPLICABLE,
        description="Flat-line detection (DART: 0.0001m/1hr, CO-OPS: 0.001m/1hr)",
    )

    # --- System-specific extensions (not QARTOD core tests) ---
    neighbor_consistency: QARTODFlag = Field(
        default=QARTODFlag.NOT_APPLICABLE,
        description=(
            "[Reserved] Cross-station travel-time consistency (DART only). "
            "No check computes this; it stays NOT_APPLICABLE."
        ),
    )
    mode_transition: QARTODFlag = Field(
        default=QARTODFlag.NOT_APPLICABLE,
        description=(
            "[Reserved] DART event mode transition flag. No check computes "
            "this; it stays NOT_APPLICABLE."
        ),
    )
    latency: QARTODFlag = Field(
        default=QARTODFlag.NOT_APPLICABLE,
        description=(
            "[Extension] Mirrors timing gap; "
            "placeholder for future wall-clock latency check"
        ),
    )

    model_config = {"extra": "forbid"}


EVENT_MODE_NOTE = (
    "Event mode detected. This does not confirm a tsunami. Elevated scrutiny required."
)


class QCReport(BaseEnvelope):
    """QC Agent output: per-record quality assessment."""

    type: str = Field(default="QCReport", frozen=True)
    station_id: str = Field(min_length=1, description="Station identifier (e.g., 21413, 1612340)")
    observed_at_utc: AwareDatetime = Field(description="UTC observation timestamp")
    measurement_type: Literal[1, 2, 3] | None = Field(
        default=None,
        description="DART measurement type code: 1=15-min, 2=1-min, 3=15-sec (None for non-DART)",
    )
    data_mode: DataMode = Field(description="Current operating mode")
    event_mode_note: str = Field(
        default="",
        description="Annotation when event mode is detected",
    )
    record_usable: bool = Field(
        description="Whether this record passes minimum quality for downstream use"
    )
    qartod_flags: QARTODFlags = Field(description="Per-test QARTOD flag results")
    detided_ssh_m: float | None = Field(
        default=None,
        description=(
            "[Reserved] Detided sea-surface height residual in meters. The "
            "QC agent does not detide, so no code path sets this; detiding "
            "happens later, in the anomaly scorer."
        ),
    )
    station_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Station confidence score (0.0 to 1.0)",
    )
    n_checks_evaluated: int = Field(
        default=0,
        ge=0,
        le=5,
        description=(
            "Number of QARTOD checks that produced a definitive result for "
            "this record; 0 means station_confidence carries no evidence "
            "(e.g. the first record of a stream)"
        ),
    )
    provenance_hash: str = Field(
        pattern=r"^[0-9a-f]{64}$",
        description="SHA-256 hex hash of the raw payload",
    )

    model_config = {"extra": "forbid"}
