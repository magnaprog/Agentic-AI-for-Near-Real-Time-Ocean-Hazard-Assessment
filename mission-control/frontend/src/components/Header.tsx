import { useEffect, useState, memo } from "react";
import type { Thresholds, SystemState } from "../types";
import { DEFAULT_THRESHOLDS, stateColor } from "../constants";

interface Props {
  connected: boolean;
  /** True once a socket has opened on this session. Without it an opening
   *  handshake and a dropped link look identical, and the strip reported
   *  "link down" on every page load. */
  everConnected?: boolean;
  fsmState: SystemState;
  hasActiveEvent: boolean;
  anomalyScore: number;
  thresholds: Thresholds | null;
  eventModeStationCount?: number;
  eventMagnitude?: number | null;
  triggerTimeUtc?: string | null;
  firstT1Minutes?: number | null;
  /** Local receipt time of the last message proving the BFF reached the core
   *  API, or null if it never has on this connection. */
  lastContactMs?: number | null;
  upstreamError?: boolean;
  demoMode?: boolean;
  /** Fewer than two DART stations carrying QC-usable data. Shown only when
   *  true: the flag distinguishes degraded coverage from adequate coverage,
   *  and rendering a reassuring readout the rest of the time would claim more
   *  than the count supports. */
  sensorDegraded?: boolean;
}

/** Heartbeats arrive about every 5s while the BFF is polling, so a gap of
 *  15s is roughly three missed beats: enough to mean something is wrong, short
 *  enough to notice during an event. */
const CONTACT_STALE_MS = 15_000;

function scoreColor(score: number, thresholds: Thresholds | null): string {
  const t3 = thresholds?.t3 ?? DEFAULT_THRESHOLDS.t3;
  const t2 = thresholds?.t2 ?? DEFAULT_THRESHOLDS.t2;
  const t1 = thresholds?.t1 ?? DEFAULT_THRESHOLDS.t1;
  if (score >= t3) return "var(--state-emergency)";
  if (score >= t2) return "var(--state-warning)";
  if (score >= t1) return "var(--state-monitor)";
  return "var(--state-idle)";
}

/** The same four bands scoreColor encodes, in text. Which band a score sits in
 *  was carried by hue alone, so it did not survive a color vision deficiency
 *  or a monochrome screenshot. T1/T2/T3 are the console's own labels: they are
 *  on the chart's reference lines directly below this strip. */
function scoreBand(score: number, thresholds: Thresholds | null): string {
  const t3 = thresholds?.t3 ?? DEFAULT_THRESHOLDS.t3;
  const t2 = thresholds?.t2 ?? DEFAULT_THRESHOLDS.t2;
  const t1 = thresholds?.t1 ?? DEFAULT_THRESHOLDS.t1;
  if (score >= t3) return "T3+";
  if (score >= t2) return "T2+";
  if (score >= t1) return "T1+";
  return "<T1";
}

/** Format elapsed time as T+Xh Ym Zs. */
function formatElapsed(triggerUtc: string): string {
  const triggerMs = new Date(triggerUtc).getTime();
  if (isNaN(triggerMs)) return "";
  const diffSec = Math.max(0, Math.floor((Date.now() - triggerMs) / 1000));
  const h = Math.floor(diffSec / 3600);
  const m = Math.floor((diffSec % 3600) / 60);
  const s = diffSec % 60;
  return `T+${h}h ${String(m).padStart(2, "0")}m ${String(s).padStart(2, "0")}s`;
}

function formatAge(ms: number): string {
  const sec = Math.max(0, Math.floor(ms / 1000));
  if (sec < 60) return `${sec}s ago`;
  const min = Math.floor(sec / 60);
  if (min < 60) return `${min}m ${sec % 60}s ago`;
  return `${Math.floor(min / 60)}h ${min % 60}m ago`;
}

/** How long ago the BFF last reached the core API.
 *
 *  Snapshots are broadcast only when they change, so without this a console
 *  watching a quiet ocean looks exactly like one whose upstream died hours
 *  ago. This reports BFF-to-core-API contact, which is what the BFF can
 *  actually observe. It is not a statement that the ingest and pipeline
 *  workers are processing observations: the core API exposes no worker
 *  heartbeat, so the label says core API and nothing broader.
 */
function LinkFreshness({
  lastContactMs,
  upstreamError,
  demoMode,
}: {
  lastContactMs?: number | null;
  upstreamError?: boolean;
  demoMode?: boolean;
}) {
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  if (demoMode) {
    // The label names what is being reported and the value gives the reading.
    // This pair was the other way round, which read as a KPI called "No core
    // API" whose value was "demo". The demo banner already says the core API
    // is not configured, so this slot does not repeat it.
    return <Kpi value="demo" label="Data source" color="var(--state-warning)" />;
  }
  if (lastContactMs == null) {
    // Distinguish "the BFF told us its upstream is down" from "we connected a
    // moment ago and the first beat has not landed yet". Both have no contact
    // time; only one of them is a fault.
    return upstreamError === true ? (
      <Kpi value="no contact" label="Core API poll" color="var(--state-emergency)" />
    ) : (
      <Kpi value="waiting" label="Core API poll" color="var(--ink-dim)" />
    );
  }

  const age = now - lastContactMs;
  const stale = upstreamError === true || age > CONTACT_STALE_MS;
  return (
    <Kpi
      value={formatAge(age)}
      label={upstreamError === true ? "Core API poll failing" : "Core API poll"}
      color={stale ? "var(--state-emergency)" : "var(--state-idle)"}
    />
  );
}

/** Isolated clock: re-renders every second without affecting the parent. */
function UTCClock({ triggerTimeUtc }: { triggerTimeUtc?: string | null }) {
  const [utc, setUtc] = useState(new Date().toISOString());
  const [elapsed, setElapsed] = useState("");

  useEffect(() => {
    const update = () => {
      setUtc(new Date().toISOString());
      setElapsed(triggerTimeUtc ? formatElapsed(triggerTimeUtc) : "");
    };
    update();
    const id = setInterval(update, 1000);
    return () => clearInterval(id);
  }, [triggerTimeUtc]);

  return (
    <div className="kpi">
      <div className="kpi__value" style={{ fontSize: 15 }}>
        {utc.replace("T", " ").slice(0, 19)}
      </div>
      <div className="kpi__label">{elapsed ? `Elapsed ${elapsed}` : "UTC"}</div>
    </div>
  );
}

function Kpi({
  value,
  label,
  color,
}: {
  value: string;
  label: string;
  color?: string;
}) {
  return (
    <div className="kpi">
      <div className="kpi__value" style={color ? { color } : undefined}>
        {value}
      </div>
      <div className="kpi__label">{label}</div>
    </div>
  );
}

function Header({
  connected,
  everConnected,
  fsmState,
  hasActiveEvent,
  anomalyScore,
  thresholds,
  eventModeStationCount,
  eventMagnitude,
  triggerTimeUtc,
  firstT1Minutes,
  lastContactMs,
  upstreamError,
  demoMode,
  sensorDegraded,
}: Props) {
  return (
    <header className="topbar region-topbar">
      <div className="topbar__brand">
        {/* The app title is the document's h1. The four section titles are
            h2, so without this heading navigation cannot reach the name of
            the thing being operated. The class carries all the styling. */}
        <h1 className="topbar__title">Ocean Hazard Mission Control</h1>
        {/* The basin comes off the wire. It was hardcoded to "pacific" while
            fsm.thresholds.basin arrived on every snapshot, so a console
            configured for another basin would have said pacific anyway. */}
        <div className="topbar__sub">
          {thresholds?.basin ? `${thresholds.basin} basin` : "basin not reported"}
          {connected ? "" : everConnected ? "  -  link down" : "  -  connecting"}
        </div>
      </div>

      <div className="topbar__rule" aria-hidden />

      <div className="disclaimer" role="note">
        Non-authoritative situational awareness. Not an official NOAA tsunami
        message.
      </div>

      <div className="topbar__kpis">
        <Kpi
          value={hasActiveEvent && eventMagnitude ? `Mw ${eventMagnitude}` : "none"}
          label="Active event"
          color={hasActiveEvent && eventMagnitude ? "var(--state-monitor)" : "var(--ink-dim)"}
        />
        <Kpi
          value={hasActiveEvent ? String(eventModeStationCount ?? 0) : "0"}
          label="Event-mode DART"
          color={
            hasActiveEvent && (eventModeStationCount ?? 0) > 0
              ? "var(--state-monitor)"
              : "var(--ink-dim)"
          }
        />
        {firstT1Minutes != null && (
          <Kpi value={`${firstT1Minutes} min`} label="First T1 (retrospective)" color="var(--accent)" />
        )}
        {/* The flag means fewer than two stations carry QC-usable data, so the
            count is 0 or 1. The BFF sends only the boolean, and "0 or 1" is
            the most this readout can honestly say. It replaces "under 2",
            which was a sentence fragment in a row of numbers. */}
        {sensorDegraded === true && (
          <Kpi value="0 or 1" label="Usable DART stations" color="var(--state-warning)" />
        )}
        <Kpi
          value={
            hasActiveEvent
              ? `${anomalyScore.toFixed(3)} ${scoreBand(anomalyScore, thresholds)}`
              : "n/a"
          }
          label="Latest score"
          color={hasActiveEvent ? scoreColor(anomalyScore, thresholds) : "var(--ink-dim)"}
        />
        <LinkFreshness
          lastContactMs={lastContactMs}
          upstreamError={upstreamError}
          demoMode={demoMode}
        />
        <UTCClock triggerTimeUtc={triggerTimeUtc} />
        <div className="stateread" role="status" aria-label={`FSM state: ${fsmState}`}>
          <span className="stateread__label">FSM</span>
          <span className="stateread__value" style={{ color: stateColor(fsmState).solid }}>
            {fsmState}
          </span>
        </div>
      </div>
    </header>
  );
}

export default memo(Header);
