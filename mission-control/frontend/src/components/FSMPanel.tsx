import { memo } from "react";
import type { CSSProperties } from "react";
import { stateColor } from "../constants";
import type { SystemState, Transition } from "../types";

interface Props {
  currentState: SystemState;
  transitions: Transition[];
}

const STATES: SystemState[] = ["IDLE", "MONITOR", "INVESTIGATE", "ASSESS", "ESCALATE"];

type RowStatus = "observed" | "active" | "unobserved";

function rowStatus(
  state: SystemState,
  current: SystemState,
  transitions: Transition[]
): RowStatus {
  if (state === current) return "active";
  if (transitions.some((t) => t.from_state === state || t.to_state === state)) {
    return "observed";
  }
  return "unobserved";
}

const TRIGGER_MAX_CHARS = 20;

/** Extract a short trigger description from trigger_reason. */
function shortTrigger(reason: string): string {
  if (reason.includes("Anomaly score")) {
    const match = reason.match(/([\d.]+)/);
    return match ? `score=${match[1]}` : "anomaly";
  }
  if (reason.includes("Seismic trigger")) {
    // The core writes "Seismic trigger: M9.1 at <region>", so this is the
    // observed magnitude. It was rendered as "M>=9.1", which reads as the
    // threshold that was met rather than the value that met it.
    const match = reason.match(/M([\d.]+)/);
    return match ? `M${match[1]}` : "seismic";
  }
  if (reason.includes("timeout")) return "timeout";
  if (reason.includes("Seismic-only")) return "seismic-only";
  if (reason.includes("No DART")) return "seismic-only";
  if (reason.includes("DART")) return "DART event-mode";
  // Anything unmatched is shown as written, cut on a word boundary and marked
  // as cut. A bare slice ended rows mid-word with nothing to say it had been
  // shortened, so a truncated reason read as the whole reason.
  if (reason.length <= TRIGGER_MAX_CHARS) return reason;
  const cut = reason.slice(0, TRIGGER_MAX_CHARS);
  const lastSpace = cut.lastIndexOf(" ");
  return `${(lastSpace > 8 ? cut.slice(0, lastSpace) : cut).trimEnd()}...`;
}

/** Compute dwell time between consecutive transitions. */
function dwellTime(transitions: Transition[], index: number): string | null {
  if (index >= transitions.length - 1) return null;
  const current = new Date(transitions[index].timestamp_utc).getTime();
  const next = new Date(transitions[index + 1].timestamp_utc).getTime();
  const diffSec = Math.round((next - current) / 1000);
  if (diffSec < 60) return `${diffSec}s`;
  if (diffSec < 3600) return `${Math.round(diffSec / 60)}m`;
  return `${Math.round(diffSec / 3600)}h`;
}

function FSMPanel({ currentState, transitions }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
      {STATES.map((s) => {
        const status = rowStatus(s, currentState, transitions);
        // The active row takes the color of the state it is showing, not a
        // fixed emergency red. A console sits at IDLE for nearly every shift,
        // and a red row every night would both contradict the green IDLE in
        // the topbar and stop meaning anything.
        const accent = status === "active" ? stateColor(s) : null;
        const rowVars = accent
          ? ({
              "--row-accent": accent.solid,
              "--row-tint": accent.tintBg,
              "--row-line": accent.tintBorder,
              "--row-tag": accent.tagText,
            } as CSSProperties)
          : undefined;
        // One suffix for both the row and its tag. Spelling them separately
        // lets them drift, and a suffix with no matching stylesheet rule
        // leaves the NOT SEEN chip unfilled and in the brightest ink on the
        // panel.
        const variant = status === "active" ? "active" : status === "observed" ? "observed" : "unseen";
        return (
          <div
            key={s}
            style={rowVars}
            className={`fsm-state fsm-state--${variant}`}
          >
            <span className="fsm-state__pip" aria-hidden />
            <span className="fsm-state__name">{s}</span>
            <span className={`fsm-tag fsm-tag--${variant}`}>
              {status === "active" ? "ACTIVE" : status === "observed" ? "OBSERVED" : "NOT SEEN"}
            </span>
          </div>
        );
      })}

      <div className="subhead">Transitions</div>

      {transitions.length === 0 && (
        <div style={{ color: "var(--ink-2)", fontSize: 11, fontStyle: "italic" }}>
          No transitions yet
        </div>
      )}

      {transitions
        .slice(-6)
        .reverse()
        .map((t, idx, arr) => {
          const dwell = dwellTime([...arr].reverse(), arr.length - 1 - idx);
          return (
            // The right column used to hold the dwell time when there was one
            // and the trigger reason otherwise, unlabeled, so the newest row
            // read "M9.1" and the rows below it read "5s" with nothing saying
            // the column had changed meaning. The trigger now always sits in
            // the same place, and the dwell is on its own labeled line.
            <div
              key={t.transition_id}
              className="mono"
              style={{
                fontSize: 11,
                color: "var(--ink-2)",
                padding: "4px 0",
                borderBottom: "1px solid rgba(255, 255, 255, 0.04)",
              }}
            >
              <div className="row--between">
                <span>
                  {t.from_state}
                  {"->"}
                  {t.to_state}
                </span>
                <span style={{ color: "var(--accent)" }}>
                  {shortTrigger(t.trigger_reason ?? "")}
                </span>
              </div>
              {dwell != null && (
                <div className="tiny dim-2">held {dwell}</div>
              )}
            </div>
          );
        })}
    </div>
  );
}

export default memo(FSMPanel);
