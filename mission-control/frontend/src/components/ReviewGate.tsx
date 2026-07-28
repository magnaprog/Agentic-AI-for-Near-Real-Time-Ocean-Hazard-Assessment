import { useEffect, useState, useCallback, useRef, memo } from "react";
import type { FSMState } from "../types";
import { submitReview, fetchEscalationPacket, useCredentials } from "../hooks/useApi";
import type { EscalationPacket } from "../hooks/useApi";
import { getEventModeStationIds, recordedReviewFor } from "../constants";
import type { AuditEntry } from "../types";

interface Props {
  fsm: FSMState | null;
  /** Recorded review decisions from the snapshot. Read from here rather than
   *  held only in component state, so a decision survives a refresh and is
   *  visible to whoever takes over the shift. */
  reviewHistory?: AuditEntry[];
  /** Reports whether the packet on screen already has a decision, so the
   *  dashboard can stop showing escalation urgency for work already done. */
  onReviewedChange?: (reviewed: boolean) => void;
}

/** Render an ISO timestamp as UTC, converting rather than trimming.
 *
 *  The previous version sliced the first 19 characters and appended "Z". The
 *  core writes datetime.now(UTC).isoformat(), so that happened to be right, but
 *  any record carrying a non-UTC offset would have been relabelled as UTC
 *  rather than converted, which is the wrong thing to do to a review record's
 *  time. An unparseable value falls back to the trimmed original with no
 *  timezone claimed. */
function utcStamp(iso: string): string {
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso.slice(0, 19).replace("T", " ");
  return `${d.toISOString().slice(0, 19).replace("T", " ")}Z`;
}

function ReviewGate({ fsm, reviewHistory, onReviewedChange }: Props) {
  const { getApiKey, getReviewerId } = useCredentials();
  const [rationale, setRationale] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [lastResult, setLastResult] = useState<string | null>(null);
  const [reviewRecorded, setReviewRecorded] = useState(false);
  const [packet, setPacket] = useState<EscalationPacket | null>(null);
  const [packetLoading, setPacketLoading] = useState(false);
  const [packetError, setPacketError] = useState<string | null>(null);
  const [superseding, setSuperseding] = useState(false);
  // Synchronous duplicate guard. `submitting` disables the buttons, but the
  // disabled attribute only lands on the next render, and clicks dispatched in
  // the same tick all pass the state check first: three clicks measured three
  // POSTs, which is three review records in an append-only trail for one
  // decision. Key repeat on a focused button is the realistic way to hit it.
  const submittingRef = useRef(false);
  const [packetViewed, setPacketViewed] = useState(false);

  const ctx = fsm?.event_context;
  const needsReview = fsm?.fsm_state === "ESCALATE" && fsm?.has_active_event && ctx != null;
  // Stable primitive for the effect dependency: ctx is a new object reference
  // on every WebSocket update, but event_id compares by value.
  const eventId = ctx?.event_id ?? null;

  const loadPacket = useCallback(async () => {
    setPacketLoading(true);
    setPacketError(null);
    try {
      const p = await fetchEscalationPacket(getApiKey());
      setPacket(p);
    } catch (err) {
      setPacketError(err instanceof Error ? err.message : "Failed to load escalation packet");
    } finally {
      setPacketLoading(false);
    }
  }, [getApiKey]);

  // Reset review state when the event changes. Only keyed on eventId (not
  // needsReview) so the packet is not nulled mid-fetch on each WebSocket tick.
  useEffect(() => {
    setLastResult(null);
    setReviewRecorded(false);
    setRationale("");
    setPacket(null);
    setPacketViewed(false);
    setPacketError(null);
  }, [eventId]);

  // Load the escalation packet when review is needed. The !packetError guard
  // stops a failed load from re-firing in a tight loop; recovery is via RETRY
  // or a new event, both of which clear packetError.
  useEffect(() => {
    if (needsReview && !packet && !packetLoading && !packetError) {
      loadPacket();
    }
  }, [needsReview, eventId, loadPacket, packet, packetLoading, packetError]);

  // A decision already recorded against THIS packet hash. Matching the hash,
  // not just the event, means a superseding packet for the same event still
  // reads as unreviewed.
  const recordedReview = recordedReviewFor(reviewHistory, ctx?.event_id, packet?.content_sha256);
  const reviewed = recordedReview != null || reviewRecorded;

  useEffect(() => {
    onReviewedChange?.(reviewed);
  }, [reviewed, onReviewedChange]);

  const handleDecision = async (decision: "APPROVE" | "REJECT" | "DEFER") => {
    if (submittingRef.current) return;
    if (!ctx || !rationale.trim() || !packet) return;
    if (ctx.event_id !== packet.event_id) {
      // Was a bare return: the highest-stakes click in the console did
      // nothing and said nothing. Name the problem and the recovery.
      setLastResult(
        "Error: this packet belongs to a different event than the one now active. Reload the packet before deciding.",
      );
      return;
    }
    submittingRef.current = true;
    setSubmitting(true);
    setLastResult(null);
    try {
      await submitReview(
        {
          event_id: ctx.event_id,
          decision,
          decision_reason: rationale.trim(),
          escalation_packet_row_id: packet.packet_row_id,
          escalation_packet_hash: packet.content_sha256,
        },
        getApiKey(),
        getReviewerId()
      );
      setReviewRecorded(true);
      setSuperseding(false);
      setLastResult(`Review recorded: ${decision}. Event remains in ESCALATE.`);
      setRationale("");
    } catch (err) {
      setLastResult(`Error: ${err instanceof Error ? err.message : "Unknown"}`);
    } finally {
      submittingRef.current = false;
      setSubmitting(false);
    }
  };

  if (!needsReview) {
    const isError = lastResult?.startsWith("Error:") ?? false;
    return (
      <div className="standby">
        <div className="standby__mark">STANDBY</div>
        <div className="standby__text">
          {fsm == null
            ? "Waiting for system data."
            : fsm.fsm_state === "IDLE"
              ? "No active events."
              : `Monitoring in progress (${fsm.fsm_state}).`}
        </div>
        {lastResult && (
          <div className={`result-line ${isError ? "result-line--err" : "result-line--ok"}`}>
            {lastResult}
          </div>
        )}
      </div>
    );
  }

  const canDecide = packetViewed && packet != null && rationale.trim().length > 0;
  const eventModeCount = getEventModeStationIds(ctx).size;

  // No scroll container on the root: the enclosing .sect__body already scrolls,
  // and a nested scrollport would swallow the sticky decision row.
  return (
    <div className="col gap-12 small" style={{ animation: "rise-in 0.3s ease-out", minHeight: "100%" }}>
      <div className="box">
        <div className="box__label">EVENT SUMMARY</div>
        {([
          ["Event", ctx.event_id.slice(0, 8)],
          ["Magnitude", `M${ctx.seismic_magnitude}, ${ctx.seismic_region}`],
          ["Epicenter", `${ctx.epicenter_lat.toFixed(2)}°${ctx.epicenter_lat >= 0 ? "N" : "S"} ${ctx.epicenter_lon.toFixed(2)}°${ctx.epicenter_lon >= 0 ? "E" : "W"}`],
          ["Latest score", ctx.latest_anomaly_score.toFixed(3), "var(--state-emergency)"],
          ["DART event-mode stations", `${eventModeCount}`],
          ["DART event mode", ctx.dart_confirmation ? "OBSERVED SINCE ORIGIN" : "NOT OBSERVED"],
          ["Review record", recordedReview ? "RECORDED IN AUDIT" : reviewRecorded ? "RECORDED THIS SESSION" : "NOT RECORDED"],
        ] as [string, string, string?][]).map(([label, value, color]) => (
          <div key={label} className="kv">
            <span className="kv__key">{label}</span>
            <span className="kv__value" style={color ? { color } : undefined}>{value}</span>
          </div>
        ))}
      </div>

      {packet && (
        <div className="box">
          <div className="box__label">PACKET OF RECORD</div>
          {([
            ["Checkpoint", packet.packet.checkpoint_id.slice(0, 12)],
            ["FSM transition", `${packet.packet.fsm_state_before} to ${packet.packet.fsm_state_after}`],
            ["Pipeline outcome", packet.packet.pipeline_outcome],
            ["Assessment", packet.packet.assessment.handoff_id.slice(0, 12)],
            ["Scientific hash", packet.packet.assessment.scientific_content_hash.slice(0, 12)],
            ["Packet row", `${packet.packet_row_id} (renderer ${packet.renderer_version})`],
            ["Packet hash", packet.content_sha256.slice(0, 12)],
          ] as [string, string][]).map(([label, value]) => (
            <div key={label} className="kv">
              <span className="kv__key">{label}</span>
              <span className="kv__value">{value}</span>
            </div>
          ))}

          {packet.packet.best_scoring_station && (
            <div className="mt-8 tiny dim">
              <strong>Best scored station:</strong>{" "}
              {packet.packet.best_scoring_station.source}:{packet.packet.best_scoring_station.station_id}{" "}
              ({packet.packet.best_scoring_station.ensemble_score.toFixed(3)})
            </div>
          )}
          <div className="mt-8 tiny dim">
            <div><strong>Action:</strong> {packet.packet.recommended_action}</div>
            <div className="mt-4">{packet.packet.disclaimer}</div>
          </div>
          <div className="note">
            This review is caller-asserted. It does not authorize distribution or close the event.
          </div>

          {!packetViewed && (
            <button
              onClick={() => setPacketViewed(true)}
              className="btn btn--reject btn--full mt-10"
            >
              ACKNOWLEDGE PACKET REVIEWED
            </button>
          )}
          {packetViewed && (
            <div className="result-line result-line--ok">PACKET ACKNOWLEDGED</div>
          )}
        </div>
      )}

      {/* Packet load failure (e.g. core API unreachable -> BFF 503), shown even
          when no packet loaded so the reviewer is never looking at a blank box. */}
      {packetError && (
        <div className="center" style={{ color: "var(--state-emergency)", fontSize: 11 }}>
          {packetError}
          <button
            onClick={loadPacket}
            className="btn btn--reject"
            style={{ marginLeft: 8, padding: "3px 10px", fontSize: 10 }}
          >
            Retry
          </button>
        </div>
      )}

      {packetLoading && (
        <div className="center dim small">Loading escalation packet...</div>
      )}

      <textarea
        className="review-rationale"
        aria-label="Review rationale"
        placeholder={packetViewed ? "Enter rationale..." : "View escalation evidence first..."}
        value={rationale}
        onChange={(e) => setRationale(e.target.value)}
        disabled={!packetViewed}
        rows={3}
        maxLength={5000}
        style={{ width: "100%", resize: "none", opacity: packetViewed ? 1 : 0.5 }}
      />

      {reviewed && !superseding ? (
        <div className="review-recorded" role="status">
          <div className="review-recorded__head">Decision recorded</div>
          <div className="review-recorded__grid">
            <span>Decision</span>
            <strong>{recordedReview?.data?.decision ?? "recorded this session"}</strong>
            <span>Reviewer</span>
            <strong>{recordedReview?.producer ?? "this session"}</strong>
            <span>Recorded</span>
            <strong>
              {recordedReview
                ? utcStamp(
                    String(recordedReview.data?.decided_at_utc ?? recordedReview.timestamp_utc ?? ""),
                  )
                : "just now"}
            </strong>
          </div>
          <div className="review-recorded__note">
            The event stays in ESCALATE. This record does not authorize
            distribution or close the event.
          </div>
          <button className="btn" onClick={() => setSuperseding(true)}>
            Record a superseding decision
          </button>
        </div>
      ) : (
      <div className="row gap-8 review-actions">
        <button
          className="btn btn--approve grow"
          onClick={() => handleDecision("APPROVE")}
          disabled={submitting || !canDecide}
        >
          APPROVE
        </button>
        <button
          className="btn btn--reject grow"
          onClick={() => handleDecision("REJECT")}
          disabled={submitting || !canDecide}
        >
          REJECT
        </button>
        <button
          className="btn btn--defer grow"
          onClick={() => handleDecision("DEFER")}
          disabled={submitting || !canDecide}
        >
          DEFER
        </button>
      </div>
      )}

      {/* Also while superseding: the buttons are back and disabled, and an
          unexplained disabled button is the defect this hint exists to fix. */}
      {(!reviewed || superseding) && !canDecide && (
        <div className="review-hint">
          {!packetViewed
            ? "Acknowledge the packet above, then write a reason to enable the decision."
            : "Write a reason to enable the decision."}
        </div>
      )}

      {lastResult && (
        <div
          role="status"
          className={`result-line ${lastResult.startsWith("Error:") ? "result-line--err" : "result-line--ok"}`}
        >
          {lastResult}
        </div>
      )}
    </div>
  );
}

export default memo(ReviewGate);
