import { useEffect, useMemo, useState } from "react";
import { AuthProvider } from "./auth/AuthProvider";
import { useAuth } from "./auth/useAuth";
import { AUTH_REQUIRED_EVENT } from "./hooks/useApi";
import { useWebSocket } from "./hooks/useWebSocket";
import { getEventModeStationIds, isReviewRequired } from "./constants";
import Header from "./components/Header";
import FSMPanel from "./components/FSMPanel";
import OceanMap from "./components/OceanMap";
import ReviewGate from "./components/ReviewGate";
import AgentStatus from "./components/AgentStatus";
import AnomalyChart from "./components/AnomalyChart";
import EventList from "./components/EventList";
import AuditLog from "./components/AuditLog";
import ErrorBoundary from "./components/ErrorBoundary";
import AuthGate from "./components/AuthGate";

function Console() {
  const { apiKey, reviewerId, clearApiKey } = useAuth();
  const [relocked, setRelocked] = useState(false);
  // Reported by the review gate: true once the packet on screen has a decision.
  const [packetReviewed, setPacketReviewed] = useState(false);

  const wsBaseUrl = useMemo(
    () =>
      (window.location.protocol === "https:" ? "wss://" : "ws://") +
      window.location.host +
      "/api/mc/ws/live",
    []
  );

  // A rejected key (HTTP 401 or WS handshake 403) re-locks the console.
  useEffect(() => {
    const onAuthRequired = () => {
      setRelocked(true);
      clearApiKey();
    };
    window.addEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
    return () => window.removeEventListener(AUTH_REQUIRED_EVENT, onAuthRequired);
  }, [clearApiKey]);

  // Both are required to record a decision, so both gate the console. Keying
  // on the access key alone would let a tab that stored a blank reviewer ID
  // reopen straight into the console, where every decision fails on submit
  // with no way to supply the missing value.
  const locked = apiKey === "" || reviewerId === "";
  const { snapshot, connected, upstreamError, lastContactMs } = useWebSocket(
    wsBaseUrl,
    locked ? "" : apiKey,
  );

  const demoMode = snapshot?.demo_mode === true;
  const fsm = snapshot?.fsm ?? null;
  const agents = snapshot?.agents ?? [];
  const audit = snapshot?.recent_audit ?? [];
  const ctx = fsm?.event_context;
  // Retrospective enrichment; null in live operation (panels self-hide).
  const scenarioMetrics = snapshot?.scenario_metrics ?? null;
  const hasActiveEvent = fsm?.has_active_event === true && ctx != null;

  // While the console is locked the socket is intentionally not open, so a
  // connectivity alert here would blame the link for a missing key and would
  // fire alongside the gate's own contradictory message.
  const banner = !locked && !connected ? (
    <div className="banner banner--alert" role="alert">
      CONNECTION LOST - RECONNECTING
    </div>
  ) : demoMode ? (
    <div className="banner banner--demo" role="status">
      DEMO MODE - Static Tohoku 2011 snapshot (core API not configured)
    </div>
  ) : upstreamError ? (
    <div className="banner banner--alert" role="alert">
      CORE API UNAVAILABLE - {snapshot ? "DISPLAYING LAST RECEIVED STATE" : "NO LIVE STATE RECEIVED"}
    </div>
  ) : (fsm?.recovery_failed || audit.some((e) => e.event_type === "fsm_recovery_failed")) ? (
    <div className="banner banner--alert" role="alert">
      FSM STATE RECOVERY FAILED - Verify manually that no active event is in progress
    </div>
  ) : null;

  return (
    <div className={banner ? "console console--banner" : "console"}>
      {banner && <div className="region-banner">{banner}</div>}

      {/* The review gate is the task, but it sits after the map in DOM order.
          Even with the map markers out of the tab order, a keyboard user
          should reach the decision controls in one hop. */}
      <a className="skip-link" href="#review-gate">Skip to review gate</a>

      <Header
        connected={connected}
        fsmState={fsm?.fsm_state ?? "IDLE"}
        hasActiveEvent={hasActiveEvent}
        anomalyScore={ctx?.latest_anomaly_score ?? 0}
        thresholds={fsm?.thresholds ?? null}
        eventModeStationCount={getEventModeStationIds(ctx).size}
        eventMagnitude={ctx?.seismic_magnitude ?? null}
        triggerTimeUtc={ctx?.trigger_time_utc ?? null}
        firstT1Minutes={scenarioMetrics?.first_t1_minutes ?? null}
        lastContactMs={lastContactMs}
        upstreamError={upstreamError}
        demoMode={demoMode}
        sensorDegraded={fsm?.sensor_degraded}
      />

      <aside className="region-rail" aria-label="System status">
        <div className="rail-section sect">
          <h2 className="sect__head">FSM Orchestrator</h2>
          <div className="sect__body">
            <ErrorBoundary fallbackLabel="FSM Panel">
              <FSMPanel currentState={fsm?.fsm_state ?? "IDLE"} transitions={fsm?.transition_history ?? []} />
            </ErrorBoundary>
          </div>
        </div>
        <div className="rail-section sect">
          <h2 className="sect__head">Component Registry</h2>
          <div className="sect__body">
            <ErrorBoundary fallbackLabel="Component Registry">
              <AgentStatus agents={agents} detectionLatency={scenarioMetrics?.detection_latency} />
            </ErrorBoundary>
          </div>
        </div>
      </aside>

      <section className="region-map" aria-label="Station map and live anomaly score">
        <div className="sect__body--flush" style={{ height: "100%" }}>
          <ErrorBoundary fallbackLabel="Ocean Map">
            <OceanMap eventContext={ctx ?? null} />
          </ErrorBoundary>
          <div className="chart-overlay">
            <div className="chart-overlay__title">
              Live anomaly score
              {/* The trace is built client-side from mount, so an operator
                  opening the console mid-event sees an empty chart filling
                  forward. The label says so rather than implying full
                  history. */}
              <span className="chart-overlay__window"> since page load</span>
            </div>
            <div style={{ flex: 1, minHeight: 0 }}>
              <ErrorBoundary fallbackLabel="Anomaly Chart">
                <AnomalyChart
                  currentScore={ctx?.latest_anomaly_score ?? 0}
                  thresholds={fsm?.thresholds ?? null}
                  hasActiveEvent={hasActiveEvent}
                />
              </ErrorBoundary>
            </div>
          </div>
        </div>
      </section>

      <main className="region-side" id="review-gate">
        <div
          className={`rail-section sect${
            isReviewRequired(fsm, packetReviewed)
              ? " sect--escalate"
              : isReviewRequired(fsm)
                ? " sect--escalate-reviewed"
                : ""
          }`}
        >
          <h2 className="sect__head">Human Review Gate</h2>
          <div className="sect__body">
            <ErrorBoundary fallbackLabel="Review Gate">
              <ReviewGate
                fsm={fsm}
                reviewHistory={snapshot?.recent_reviews ?? snapshot?.recent_audit}
                onReviewedChange={setPacketReviewed}
              />
            </ErrorBoundary>
          </div>
        </div>
        <div className="rail-section sect">
          <h2 className="sect__head">Active Event</h2>
          <div className="sect__body">
            <ErrorBoundary fallbackLabel="Event List">
              <EventList fsm={fsm} ensembleAblation={scenarioMetrics?.ensemble_ablation} />
            </ErrorBoundary>
          </div>
        </div>
      </main>

      {/* aria-label, not aria-labelledby: AuditLog's heading is behind an
          ErrorBoundary, and a dangling reference would leave the region
          unnamed and so not exposed as a landmark at all. */}
      <section className="region-audit" aria-label="Activity">
        <ErrorBoundary fallbackLabel="Audit Log">
          <AuditLog entries={audit} />
        </ErrorBoundary>
      </section>

      {locked && <AuthGate expired={relocked} />}
    </div>
  );
}

export default function App() {
  return (
    <AuthProvider>
      <Console />
    </AuthProvider>
  );
}
