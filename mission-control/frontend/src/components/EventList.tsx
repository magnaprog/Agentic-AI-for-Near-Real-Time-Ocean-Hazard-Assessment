import { memo } from "react";
import type { FSMState, AblationRow } from "../types";
import { getEventModeStationIds, stateColor } from "../constants";

interface Props {
  fsm: FSMState | null;
  ensembleAblation?: AblationRow[];
}

/** Render an ISO timestamp as UTC. Parses via Date so an aware non-UTC offset
 * is converted instead of string-sliced and mislabeled "Z". */
function utcDisplay(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toISOString().slice(0, 19).replace("T", " ") + "Z";
}

function EventList({ fsm, ensembleAblation }: Props) {
  const ctx = fsm?.event_context;

  if (ctx == null) {
    return (
      <div className="standby">
        <div className="standby__mark">NO ACTIVE EVENT</div>
        <div className="standby__text">An event card appears here when the FSM is triggered.</div>
      </div>
    );
  }

  const eventModeStations = getEventModeStationIds(ctx);
  // Color the card by FSM state, not always emergency red: an active event can
  // still be in MONITOR (seismic trigger below the score thresholds).
  const accent = stateColor(fsm?.fsm_state ?? "MONITOR");

  return (
    <div className="col gap-12">
      <div className="box" style={{ background: accent.tintBg, borderColor: accent.tintBorder }}>
        <div className="row--between mb-8">
          <span style={{ fontSize: 12, fontWeight: 700 }}>
            Active seismic event (M{ctx.seismic_magnitude})
          </span>
          <span className="fsm-tag" style={{ background: accent.tintBorder, color: accent.tagText }}>
            {fsm?.fsm_state}
          </span>
        </div>

        <div className="kv">
          <span className="kv__key">Origin time</span>
          <span className="kv__value">{utcDisplay(ctx.trigger_time_utc)}</span>
        </div>
        <div className="kv">
          <span className="kv__key">Epicenter</span>
          <span className="kv__value">
            {ctx.epicenter_lat.toFixed(2)}°{ctx.epicenter_lat >= 0 ? "N" : "S"}{" "}
            {ctx.epicenter_lon.toFixed(2)}°{ctx.epicenter_lon >= 0 ? "E" : "W"}
          </span>
        </div>
        <div className="kv">
          <span className="kv__key">Latest score</span>
          <span className="kv__value" style={{ color: accent.solid }}>
            {ctx.latest_anomaly_score.toFixed(3)}
          </span>
        </div>
        <div className="kv">
          <span className="kv__key">DART event-mode stations</span>
          <span className="kv__value" style={{ color: "var(--state-monitor)" }}>
            {eventModeStations.size}
          </span>
        </div>

        <div className="mt-10">
          <div className="tiny dim mb-5" style={{ fontSize: 9, letterSpacing: "0.1em", textTransform: "uppercase", fontWeight: 700 }}>
            Event-mode DART evidence
          </div>
          <div className="row wrap gap-4">
            {[...eventModeStations].map((sid) => (
              <span key={sid} className="chip">
                {sid}
              </span>
            ))}
            {eventModeStations.size === 0 && (
              <span className="tiny dim-2">None observed</span>
            )}
          </div>
        </div>
      </div>

      {ensembleAblation && ensembleAblation.length > 0 && (
        <div>
          <div className="subhead">Ensemble ablation</div>
          <table className="metrics-table">
            <thead>
              <tr><th>Configuration</th><th>T3 hit</th><th>Score</th></tr>
            </thead>
            <tbody>
              {ensembleAblation.map((r) => (
                <tr
                  key={r.configuration}
                  className={r.configuration === "Full ensemble" ? "row-highlight" : undefined}
                >
                  <td>{r.configuration}</td>
                  <td>{r.t3_hits}</td>
                  <td>{r.peak_score.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export default memo(EventList);
