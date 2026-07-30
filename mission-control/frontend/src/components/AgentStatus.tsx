import { memo } from "react";
import type { Agent, DetectionLatencyRow } from "../types";

interface Props {
  agents: Agent[];
  detectionLatency?: DetectionLatencyRow[];
  /** True when the BFF reported that the component-registry query failed
   *  upstream on this poll, so an empty list means the query failed rather
   *  than that the core registered nothing. */
  registryUnavailable?: boolean;
}

/** Threshold for the latency bands, in minutes. */
const LATENCY_WARN_MINUTES = 12;

function shortLabel(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes("qc")) return "QC";
  if (lower.includes("anomaly")) return "AD";
  if (lower.includes("scenario")) return "SI";
  if (lower.includes("verification") || lower.includes("verif")) return "VC";
  if (lower.includes("report")) return "RP";
  return name.slice(0, 2).toUpperCase();
}

/** Minutes formatter: "2.5m", or "n/a" when the threshold was never crossed. */
function fmtMinutes(m: number | null): string {
  return m == null ? "n/a" : `${m}m`;
}

/** Color class by detection speed (null = never detected). */
function latencyClass(m: number | null): string {
  if (m == null) return "val-bad";
  if (m <= LATENCY_WARN_MINUTES) return "val-good";
  return "val-warn";
}

/** Non-color cue for the same three bands the colors encode. Anyone who cannot
 *  separate the hues, or is reading a monochrome capture of the console, had
 *  no way to tell a fast detection from a slow one: both are just a number.
 *  "n/a" already stands on its own for a threshold that was never crossed. */
function latencyMark(m: number | null): string {
  if (m == null) return "";
  return m <= LATENCY_WARN_MINUTES ? "" : "!";
}

function DetectionLatency({ rows }: { rows: DetectionLatencyRow[] }) {
  return (
    <div>
      <div className="subhead">Detection latency (retrospective)</div>
      <table className="metrics-table">
        <thead>
          <tr><th>Station</th><th>Dist</th><th>T1</th><th>T3</th></tr>
        </thead>
        <tbody>
          {rows.map((r) => (
            <tr key={r.station_id}>
              <td>{r.station_id}</td>
              <td className="val-muted">{Math.round(r.distance_km)}km</td>
              <td className={latencyClass(r.t1_minutes)}>
                {fmtMinutes(r.t1_minutes)}{latencyMark(r.t1_minutes)}
              </td>
              <td className={latencyClass(r.t3_minutes)}>
                {fmtMinutes(r.t3_minutes)}{latencyMark(r.t3_minutes)}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      <div className="tiny dim mt-4">
        ! marks a crossing later than {LATENCY_WARN_MINUTES} minutes. n/a means
        the threshold was never crossed.
      </div>
    </div>
  );
}

function AgentStatus({ agents, detectionLatency, registryUnavailable }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {agents.length === 0 ? (
        <div className="standby">
          {/* "Awaiting" for a failed query would report an outage as patience. */}
          <div className="standby__text">
            {registryUnavailable
              ? "Component registry unavailable: the last query to the core API failed."
              : "Awaiting component registry."}
          </div>
        </div>
      ) : (
        agents.map((a) => (
          <div key={a.name} className="registry-row">
            <div className="registry-row__badge">{shortLabel(a.name)}</div>
            {/* The execution path sits under the text it qualifies. A column
                of its own costs up to 96px of a 235px rail and squeezes the
                description to about eleven characters a line. */}
            <div className="registry-row__body">
              <div className="registry-row__name">{a.name}</div>
              <div className="registry-row__desc">{a.description}</div>
              <div className="registry-row__path">{a.execution_path.replace(/_/g, " ")}</div>
            </div>
          </div>
        ))
      )}
      {agents.length > 0 && detectionLatency && detectionLatency.length > 0 && (
        <DetectionLatency rows={detectionLatency} />
      )}
    </div>
  );
}

export default memo(AgentStatus);
