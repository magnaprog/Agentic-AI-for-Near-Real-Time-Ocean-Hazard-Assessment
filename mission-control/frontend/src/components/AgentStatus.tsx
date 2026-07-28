import { memo } from "react";
import type { Agent, DetectionLatencyRow } from "../types";

interface Props {
  agents: Agent[];
  detectionLatency?: DetectionLatencyRow[];
}

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
  if (m <= 12) return "val-good";
  return "val-warn";
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
              <td className={latencyClass(r.t1_minutes)}>{fmtMinutes(r.t1_minutes)}</td>
              <td className={latencyClass(r.t3_minutes)}>{fmtMinutes(r.t3_minutes)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function AgentStatus({ agents, detectionLatency }: Props) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
      {agents.length === 0 ? (
        <div className="standby">
          <div className="standby__text">Awaiting component registry.</div>
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
