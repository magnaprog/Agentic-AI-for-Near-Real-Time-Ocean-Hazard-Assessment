/** Shared UI constants and selectors for the Mission Control dashboard. */

import type { AuditEntry, EventContext, FSMState, SystemState } from "./types";

/** Default anomaly detection thresholds used when live thresholds are unavailable. */
export const DEFAULT_THRESHOLDS = {
  t1: 0.35,
  t2: 0.6,
  t3: 0.85,
} as const;

/** DART stations with event-mode evidence in the active event context.
 * This is transmission-state evidence, not an anomaly-threshold crossing or a
 * complete inventory of stations monitored by the live connector. */
export function getEventModeStationIds(
  ctx: EventContext | null | undefined,
): Set<string> {
  if (!ctx) return new Set();
  return new Set(ctx.stations_in_event_mode);
}

/** The event type the core writes for a recorded review (POST /api/review). */
export const REVIEW_DECISION_EVENT = "assessment_review_decision";

/** A review already recorded against a specific escalation packet, read from
 *  the audit trail in the snapshot rather than from component state, so it
 *  survives a refresh and is visible to whoever takes over the shift.
 *
 *  Matched on the packet hash, not just the event: a superseding packet for
 *  the same event has not been reviewed just because an earlier one was. */
export function recordedReviewFor(
  auditEntries: AuditEntry[] | undefined,
  eventId: string | null | undefined,
  packetHash: string | null | undefined,
): AuditEntry | null {
  if (!auditEntries || !eventId || !packetHash) return null;
  const matches = auditEntries.filter(
    (e) =>
      e.event_type === REVIEW_DECISION_EVENT &&
      e.event_id === eventId &&
      e.data?.escalation_packet_hash === packetHash,
  );
  if (matches.length === 0) return null;
  // Newest wins: a later decision supersedes an earlier one on the same packet.
  return matches.reduce((a, b) =>
    (b.timestamp_utc ?? "") > (a.timestamp_utc ?? "") ? b : a,
  );
}

/** True when the FSM is in ESCALATE with an active event context a duty
 * scientist must review. Shared by the review gate and the dashboard's
 * escalation-urgency styling so the two predicates cannot diverge.
 *
 * `reviewed` suppresses the urgency treatment once a decision exists for the
 * packet on screen. Without it the section pulsed emergency red forever after
 * the operator had already decided, which reads as "still needs you". */
export function isReviewRequired(
  fsm: FSMState | null,
  reviewed = false,
): boolean {
  return (
    !reviewed &&
    fsm?.fsm_state === "ESCALATE" &&
    fsm?.has_active_event === true &&
    fsm?.event_context != null
  );
}

/** Severity color for an FSM state: a solid CSS variable plus faint rgba tints
 * for backgrounds/borders (inline rgba is used because CSS vars cannot carry an
 * alpha channel here). Single source so the header state readout and the event
 * card cannot show different colors for the same state. Tints track the
 * --state-* hexes in global.css: idle #3aa87f, monitor #e0a45c, warning
 * #e07a52, emergency #e05c5c. */
export interface StateColor {
  solid: string;
  tintBg: string;
  tintBorder: string;
  /** Text colour for a chip whose background is `tintBorder`. `solid` is the
   *  full-strength hue and is readable on the dark bar, but on its own 30%
   *  tint every state fell under WCAG AA 4.5:1 (idle 3.75, monitor 4.44,
   *  warning 3.71, emergency 3.31). These are the same hues lightened until
   *  they clear it. Re-run the contrast check before changing either field. */
  tagText: string;
}

export function stateColor(state: SystemState): StateColor {
  switch (state) {
    case "IDLE":
      return {
        solid: "var(--state-idle)",
          tagText: "#6bbe9f",
        tintBg: "rgba(58, 168, 127, 0.08)",
        tintBorder: "rgba(58, 168, 127, 0.3)",
      };
    case "MONITOR":
      return {
        solid: "var(--state-monitor)",
          tagText: "#e5b274",
        tintBg: "rgba(224, 164, 92, 0.08)",
        tintBorder: "rgba(224, 164, 92, 0.3)",
      };
    case "INVESTIGATE":
    case "ASSESS":
      return {
        solid: "var(--state-warning)",
          tagText: "#e89b7d",
        tintBg: "rgba(224, 122, 82, 0.08)",
        tintBorder: "rgba(224, 122, 82, 0.3)",
      };
    case "ESCALATE":
      return {
        solid: "var(--state-emergency)",
          tagText: "#e88585",
        tintBg: "rgba(224, 92, 92, 0.08)",
        tintBorder: "rgba(224, 92, 92, 0.3)",
      };
  }
}
