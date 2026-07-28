"""Escalation Packet schema - evidence bundle for human review.

Generated when the FSM transitions to ESCALATE state. The packet is
assembled from whatever pipeline data is available at generation time.

Full escalation packet with scenario summary, verification results,
anomaly timeline, recommended action, and provenance references.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from hazard_assessment.schemas.envelope import AwareDatetime, BaseEnvelope
from hazard_assessment.schemas.observation import is_dart_station_id
from hazard_assessment.schemas.verification import VerificationOutcome


class ScenarioSummary(BaseModel):
    """Typed scenario summary for the escalation packet.

    Replaces the previous ``dict[str, Any]`` with explicit fields so that
    consumers (UI, audit) get schema validation instead of opaque dicts.
    """

    top_scenario_mw: float = Field(description="Mw of the highest-ranked scenario")
    constraint_stage: str = Field(description="Constraint stage (e.g., 'DART_CONSTRAINED')")
    ensemble_spread: str = Field(description="Ensemble spread classification (LOW/MODERATE/HIGH)")
    n_ranked_scenarios: int = Field(default=0, description="Number of ranked scenarios in ensemble")
    coastal_proxy_highlights: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Abbreviated coastal proxy data (site_id, p50_m, arrival_utc) "
            "for the top sites by amplitude."
        ),
    )

    model_config = {"extra": "forbid"}


class EscalationPacket(BaseEnvelope):
    """Evidence bundle sent to the reviewer when the FSM enters ESCALATE.

    Contains the full context for human decision-making: scenario summary,
    verification results, anomaly timeline, and recommended action with
    provenance references.

    The FSM enters ESCALATE via two triggers (both from ASSESS state):
    1. Anomaly score >= T3 threshold
    2. Seismic magnitude >= escalation threshold with DART confirmation

    The packet may be generated before or after the verification pipeline
    completes. When generated early, verification_status and
    scenario_summary are None.
    """

    type: str = Field(default="EscalationPacket", frozen=True)

    # --- Escalation context ---
    escalation_trigger: str = Field(
        description="What triggered escalation (e.g., 'anomaly_score >= T3', 'seismic override')",
    )
    escalation_time_utc: AwareDatetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when escalation was triggered",
    )
    criticality_reasons: list[str] = Field(
        min_length=1,
        description="Reasons why escalation was triggered",
    )

    # --- Scenario summary ---
    scenario_summary: ScenarioSummary | None = Field(
        default=None,
        description=(
            "Typed scenario summary (top scenario Mw, constraint stage, "
            "ensemble spread, coastal proxy highlights). None if escalation "
            "occurred before scenario inversion."
        ),
    )

    # --- Verification results ---
    verification_status: VerificationOutcome | None = Field(
        default=None,
        description=(
            "Verification status at time of escalation, "
            "or None if escalation occurred before verification"
        ),
    )
    verification_summary: list[dict[str, str]] | None = Field(
        default=None,
        description=(
            "Summary of individual verification check results "
            "(name, result, evidence). None if verification not yet run."
        ),
    )

    # --- Anomaly timeline ---
    anomaly_timeline: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Chronological anomaly score history for this event "
            "(timestamp_utc, anomaly_score, from_state, to_state). "
            "May be empty if no history available."
        ),
    )

    # --- Event context snapshot ---
    seismic_magnitude: float | None = Field(
        default=None,
        description="Seismic magnitude from event context",
    )
    seismic_region: str | None = Field(
        default=None,
        description="Seismic region from event context",
    )
    epicenter_lat: float | None = Field(
        default=None,
        description="Epicenter latitude",
    )
    epicenter_lon: float | None = Field(
        default=None,
        description="Epicenter longitude",
    )
    latest_anomaly_score: float | None = Field(
        default=None,
        description="Latest anomaly score at escalation time",
    )
    dart_confirmation: bool = Field(
        default=False,
        description=(
            "Whether one or more DART stations entered event mode during the "
            "event. Event mode is a necessary but NOT sufficient indicator of a "
            "tsunami (it can also be triggered by seismic Rayleigh waves or "
            "calibration); the anomaly pipeline performs the actual signal "
            "discrimination."
        ),
    )
    active_dart_stations: list[str] = Field(
        default_factory=list,
        description="DART stations active at escalation time",
    )
    input_refs_truncated: bool = Field(
        default=False,
        description=(
            "True if the raw-input provenance (input_refs) was capped and does "
            "not list every contributing raw record."
        ),
    )

    # --- Recommended action ---
    recommended_action: str = Field(
        default="Human review required",
        description="Recommended next action for the reviewer",
    )

    # --- Packet integrity ---
    packet_hash: str = Field(
        default="",
        description=(
            "SHA-256 hash of the packet content for integrity verification. "
            "Auto-computed on construction by model validator."
        ),
    )

    @field_validator("active_dart_stations")
    @classmethod
    def _validate_active_dart_stations(cls, value: list[str]) -> list[str]:
        invalid = [station_id for station_id in value if not is_dart_station_id(station_id)]
        if invalid:
            raise ValueError("active_dart_stations must contain five-digit DART station IDs")
        return value

    @model_validator(mode="after")
    def _compute_hash_if_empty(self) -> EscalationPacket:
        """Auto-compute packet hash on construction if not provided."""
        if not self.packet_hash:
            object.__setattr__(self, "packet_hash", self._compute_hash())
        return self

    def _compute_hash(self) -> str:
        """Compute SHA-256 of all fields except the hash itself.

        This hash includes the per-instance envelope identity (``handoff_id``,
        ``produced_at_utc``), so it is unique to this object instance and not
        reproducible across re-renders of the same evidence. It is used only
        for audit logging (``app.py``). The tamper-evident content hash of
        record is ``workers/reviewer_packet.canonical_packet_hash`` (stored as
        ``content_sha256``), which is deterministic over durable packet
        content and is what the review endpoint compares.
        """
        data = self.model_dump(mode="json", exclude={"packet_hash"})
        canonical = json.dumps(data, sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()

    model_config = {"extra": "forbid"}
