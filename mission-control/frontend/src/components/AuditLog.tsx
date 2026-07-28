import { memo } from "react";
import { REVIEW_DECISION_EVENT } from "../constants";
import type { AuditEntry } from "../types";

interface Props {
  entries: AuditEntry[];
}

function eventColor(eventType: string): string {
  switch (eventType) {
    case "state_transition": return "var(--ink)";
    case "policy_denial": return "var(--state-emergency)";
    case "escalation_packet_generated": return "var(--state-emergency)";
    case "fsm_recovery_failed": return "var(--state-emergency)";
    case "abstain_triggered": return "var(--state-monitor)";
    case "guardrail_scan": return "var(--state-idle)";
    case REVIEW_DECISION_EVENT: return "var(--state-idle)";
    default: return "var(--ink-2)";
  }
}

function shortEventDescription(e: AuditEntry): string {
  if (e.data?.from_state && e.data?.to_state) {
    const score = e.data?.anomaly_score;
    const scoreStr = score != null ? ` score=${Number(score).toFixed(3)}` : "";
    return `FSM->${e.data.to_state}${scoreStr}`;
  }
  if (e.event_type === "guardrail_scan") {
    return `guardrail ${e.data?.passed ? "PASS" : "FAIL"}`;
  }
  if (e.event_type === "permission_check") {
    return `perm ${e.data?.allowed ? "ALLOWED" : "DENIED"}`;
  }
  if (e.event_type === "escalation_packet_generated") {
    return "ESCALATION PACKET";
  }
  // assessment_review_decision is what POST /api/review actually writes. This
  // matched "human_decision", which is a schema module name and never an event
  // type, so a recorded review never appeared in the activity strip.
  if (e.event_type === REVIEW_DECISION_EVENT) {
    return `REVIEW: ${e.data?.decision ?? "recorded"}`;
  }
  if (e.event_type === "abstain_triggered") {
    return e.data?.trigger ? `ABSTAIN (${String(e.data.trigger)})` : "ABSTAIN";
  }
  if (e.event_type === "fsm_recovery_failed") {
    return "RECOVERY FAILED";
  }
  if (e.data?.trigger_reason) {
    return String(e.data.trigger_reason).slice(0, 40);
  }
  return e.event_type;
}

/** Human-readable producer names for provenance. */
function producerLabel(producer: string): string {
  switch (producer) {
    case "anomaly_agent": return "anomaly";
    case "orchestrator": return "fsm";
    case "policy_engine": return "policy";
    case "verification_agent": return "verify";
    default: return producer.replace(/_agent$/, "").replace(/_/g, " ");
  }
}

/** Render the HH:MM:SS portion of a timestamp as UTC. */
function utcTime(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(11, 19);
  return d.toISOString().slice(11, 19);
}

function AuditLog({ entries }: Props) {
  // Dedup by entry_id: the rolling audit window can re-deliver an entry across
  // WebSocket snapshots, and the id is the durable identity of the record.
  const seen = new Set<string>();
  const unique = entries.filter((e) => {
    if (seen.has(e.entry_id)) return false;
    seen.add(e.entry_id);
    return true;
  });

  if (unique.length === 0) {
    return (
      <div className="audit-strip">
        <h2 className="audit-strip__title">Activity</h2>
        <div style={{ color: "var(--ink-2)", fontSize: 10, fontStyle: "italic" }}>
          Awaiting system activity...
        </div>
      </div>
    );
  }

  // Collapse consecutive entries that render identically. The worker writes an
  // anomaly_scored entry per scored window, so the strip filled with ten
  // identical rows covering a fraction of a second and any transition or
  // packet event was pushed off the end. The count keeps the run visible
  // rather than hiding what was collapsed.
  //
  // The key is the rendered row, not the event type: IDLE to MONITOR and
  // MONITOR to ESCALATE are both state_transition from the same producer, and
  // keying on the type alone merged them and hid the escalation.
  const runs: { entry: AuditEntry; label: string; count: number }[] = [];
  for (const e of unique) {
    const label = shortEventDescription(e);
    const last = runs[runs.length - 1];
    if (last && last.label === label && last.entry.producer === e.producer) {
      last.count += 1;
      continue;
    }
    runs.push({ entry: e, label, count: 1 });
  }
  const recent = runs.slice(0, 10);

  return (
    <div className="audit-strip">
      <h2 className="audit-strip__title">Activity</h2>
      <div className="audit-strip__items">
        {recent.map(({ entry: e, label, count }) => (
          <div key={e.entry_id} className="audit-item">
            <span className="audit-item__time">{utcTime(e.timestamp_utc)}</span>
            <span style={{ color: "var(--ink-dim)" }}>{producerLabel(e.producer)}</span>
            <span style={{ color: eventColor(e.event_type) }}>{label}</span>
            {count > 1 && <span className="audit-item__count">x{count}</span>}
          </div>
        ))}
      </div>
    </div>
  );
}

export default memo(AuditLog);
