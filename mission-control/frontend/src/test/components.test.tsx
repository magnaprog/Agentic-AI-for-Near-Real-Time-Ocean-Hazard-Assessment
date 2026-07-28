/**
 * UI-semantics tests for the safety-relevant dashboard behaviors:
 * the always-visible disclaimer, the acknowledge-gated review flow,
 * the demo-only retrospective panels hiding without data, and the
 * absence of reserved alert terminology in rendered output.
 */
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import Header from "../components/Header";
import FSMPanel from "../components/FSMPanel";
import AgentStatus from "../components/AgentStatus";
import EventList from "../components/EventList";
import ReviewGate from "../components/ReviewGate";
import AuditLog from "../components/AuditLog";
import type { EscalationPacket } from "../hooks/useApi";
import type { AuditEntry, EventContext, FSMState } from "../types";
import { recordedReviewFor } from "../constants";

vi.mock("../hooks/useApi", () => ({
  fetchEscalationPacket: vi.fn(),
  submitReview: vi.fn(),
  useCredentials: () => ({
    getApiKey: () => "test-key",
    getReviewerId: () => "test-reviewer",
  }),
}));

import { fetchEscalationPacket, submitReview } from "../hooks/useApi";

// All eight terms enforced by policy/guardrails.py PROHIBITED_TERMS. Keep
// this list complete: these assertions are titled as covering reserved
// terminology, so any term left out is silently exempt and a panel rendering
// it would pass. Multi-word terms join on \s+ because textContent
// concatenates across elements, so the spacing between words is not fixed.
const NWS_TERMS =
  /\b(warning|advisory|watch|bulletin|cancellation|all\s+clear|information\s+statement|threat\s+message)\b/i;

function tohokuContext(): EventContext {
  return {
    event_id: "tohoku-2011-03-11T05:46:24Z",
    seismic_magnitude: 9.1,
    seismic_region: "Near the east coast of Honshu, Japan",
    epicenter_lat: 38.297,
    epicenter_lon: 142.373,
    trigger_time_utc: "2011-03-11T05:46:24+00:00",
    latest_anomaly_score: 0.997,
    dart_confirmation: true,
    active_dart_stations: ["21401", "21413", "46408"],
    stations_in_event_mode: ["21401", "21413", "46403"],
  };
}

function escalateFsm(): FSMState {
  return {
    fsm_state: "ESCALATE",
    has_active_event: true,
    event_context: tohokuContext(),
    thresholds: { basin: "pacific", t1: 0.35, t2: 0.6, t3: 0.85 },
    transition_history: [],
  };
}

function packet(): EscalationPacket {
  return {
    packet_row_id: 3,
    assessment_row_id: 41,
    event_id: "tohoku-2011-03-11T05:46:24Z",
    renderer_version: "1",
    content_sha256: "a".repeat(64),
    created_at: "2011-03-11T06:11:00+00:00",
    packet: {
      assessment_row_id: 41,
      checkpoint_id: "b".repeat(64),
      event_id: "tohoku-2011-03-11T05:46:24Z",
      produced_at_utc: "2011-03-11T06:11:00+00:00",
      fsm_state_before: "ASSESS",
      fsm_state_after: "ESCALATE",
      pipeline_outcome: "ABSTAIN",
      scientific_content_hash: "c".repeat(64),
      best_scoring_station: {
        source: "dart",
        station_id: "21418",
        ensemble_score: 0.996,
      },
      dart_stations_currently_in_event_mode: ["21401", "21413"],
      recommended_action: "Human review required",
      disclaimer: "Research decision-support assessment only.",
      assessment: {
        handoff_id: "11111111-1111-4111-8111-111111111111",
        event_id: "tohoku-2011-03-11T05:46:24Z",
        scientific_content_hash: "c".repeat(64),
      },
    },
  };
}

describe("Header", () => {
  it("always renders the non-authoritative disclaimer", () => {
    render(
      <Header connected={false} fsmState="IDLE" hasActiveEvent={false} anomalyScore={0} thresholds={null} />
    );
    expect(
      screen.getByText(/Non-authoritative situational awareness/i)
    ).toBeInTheDocument();
  });

  it("reports no active event honestly instead of a fabricated zero score", () => {
    // states.py resets latest_anomaly_score to 0 between events, so a 0 arriving
    // with no active event is not a reading. The header must say so rather than
    // render 0.000, which would look like a real measurement.
    render(
      <Header connected={true} fsmState="IDLE" hasActiveEvent={false} anomalyScore={0} thresholds={null} />
    );
    expect(screen.getByText("n/a")).toBeInTheDocument();
    expect(screen.getByText("none")).toBeInTheDocument();
    expect(screen.queryByText("0.000")).not.toBeInTheDocument();
  });

  it("hides the First T1 KPI without data and shows it with data", () => {
    const { rerender } = render(
      <Header
        connected={true}
        fsmState="ESCALATE"
        hasActiveEvent={true}
        anomalyScore={0.997}
        thresholds={null}
        firstT1Minutes={null}
      />
    );
    expect(screen.queryByText(/First T1/i)).not.toBeInTheDocument();
    rerender(
      <Header
        connected={true}
        fsmState="ESCALATE"
        hasActiveEvent={true}
        anomalyScore={0.997}
        thresholds={null}
        firstT1Minutes={2.5}
      />
    );
    expect(screen.getByText(/First T1 \(retrospective\)/i)).toBeInTheDocument();
    expect(screen.getByText("2.5 min")).toBeInTheDocument();
  });

  it("separates a quiet ocean from a dead upstream", () => {
    // Snapshots broadcast only on change (ws_manager.poll_loop), so silence is
    // ambiguous by itself. These four readouts are the disambiguation.
    const { rerender } = render(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null} />
    );
    expect(screen.getByText("waiting")).toBeInTheDocument();

    rerender(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null}
        lastContactMs={null} upstreamError={true} />
    );
    expect(screen.getByText("no contact")).toBeInTheDocument();

    rerender(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null}
        lastContactMs={Date.now() - 3000} />
    );
    const fresh = screen.getByText(/^\ds ago$/);
    expect(fresh).toHaveStyle({ color: "var(--state-idle)" });

    rerender(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null}
        lastContactMs={Date.now() - 240_000} />
    );
    const stale = screen.getByText(/^[34]m \d+s ago$/);
    expect(stale).toHaveStyle({ color: "var(--state-emergency)" });
  });

  it("shows degraded DART coverage only when the flag is set", () => {
    // Rendering a reassuring readout when the flag is false would claim more
    // than a count of "two or more" supports, so the KPI appears only when
    // coverage is below the triangulation minimum.
    const { rerender } = render(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null} />
    );
    expect(screen.queryByText(/Usable DART stations/i)).not.toBeInTheDocument();

    rerender(
      <Header connected={true} fsmState="MONITOR" hasActiveEvent={false} anomalyScore={0} thresholds={null}
        sensorDegraded={true} />
    );
    expect(screen.getByText("Usable DART stations")).toBeInTheDocument();
    expect(screen.getByText("under 2")).toBeInTheDocument();
  });

  it("says demo instead of claiming core API contact in demo mode", () => {
    render(
      <Header connected={true} fsmState="ESCALATE" hasActiveEvent={false} anomalyScore={0} thresholds={null}
        lastContactMs={Date.now()} demoMode={true} />
    );
    expect(screen.getByText("demo")).toBeInTheDocument();
    expect(screen.getByText("No core API")).toBeInTheDocument();
    expect(screen.queryByText(/ago$/)).not.toBeInTheDocument();
  });
});

describe("FSMPanel", () => {
  it("renders the five-state ladder with ASCII transition arrows", () => {
    render(
      <FSMPanel
        currentState="INVESTIGATE"
        transitions={[
          {
            transition_id: "t-1",
            event_id: null,
            timestamp_utc: "2011-03-11T05:46:24+00:00",
            from_state: "IDLE",
            to_state: "MONITOR",
            trigger_reason: "Seismic event M9.1",
            anomaly_score: null,
            seismic_magnitude: 9.1,
          },
        ]}
      />
    );
    for (const s of ["IDLE", "MONITOR", "INVESTIGATE", "ASSESS", "ESCALATE"]) {
      expect(screen.getByText(s)).toBeInTheDocument();
    }
    expect(screen.getByText(/IDLE->MONITOR/)).toBeInTheDocument();
  });

  it("does not mark skipped states as observed", () => {
    render(
      <FSMPanel
        currentState="ESCALATE"
        transitions={[
          {
            transition_id: "t-monitor",
            event_id: "event",
            timestamp_utc: "2011-03-11T05:46:23+00:00",
            from_state: "IDLE",
            to_state: "MONITOR",
            trigger_reason: "Seismic trigger",
            anomaly_score: null,
            seismic_magnitude: 9.1,
          },
          {
            transition_id: "t-direct",
            event_id: "event",
            timestamp_utc: "2011-03-11T05:46:24+00:00",
            from_state: "MONITOR",
            to_state: "ESCALATE",
            trigger_reason: "Seismic-only escalation",
            anomaly_score: null,
            seismic_magnitude: 9.1,
          },
        ]}
      />
    );
    expect(screen.getAllByText("NOT SEEN")).toHaveLength(2);
    expect(screen.getAllByText("OBSERVED")).toHaveLength(2);
  });
});

describe("AgentStatus", () => {
  const agents = [
    { name: "Anomaly Agent", version: "1.0", execution_path: "LIVE_WORKER", description: "ok" },
  ];

  it("hides the detection latency table without data", () => {
    render(<AgentStatus agents={agents} />);
    expect(screen.queryByText(/Detection Latency/i)).not.toBeInTheDocument();
  });

  it("renders the detection latency table with data, using n/a for no detection", () => {
    render(
      <AgentStatus
        agents={agents}
        detectionLatency={[
          { station_id: "21418", distance_km: 561, t1_minutes: 2.5, t3_minutes: 3.5 },
          { station_id: "46403", distance_km: 4833, t1_minutes: null, t3_minutes: null },
        ]}
      />
    );
    expect(screen.getByText(/Detection Latency/i)).toBeInTheDocument();
    expect(screen.getByText("2.5m")).toBeInTheDocument();
    expect(screen.getAllByText("n/a")).toHaveLength(2);
  });
});

describe("EventList", () => {
  it("hides ablation data and labels event-mode evidence without inventory coverage", () => {
    render(<EventList fsm={escalateFsm()} />);
    expect(screen.queryByText(/Ensemble Ablation/i)).not.toBeInTheDocument();
    expect(screen.getByText(/DART event-mode stations/i)).toBeInTheDocument();
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.queryByText("3 / 4")).not.toBeInTheDocument();
    expect(screen.queryByText("46408")).not.toBeInTheDocument();
  });

  it("renders the ablation table when data is present", () => {
    render(
      <EventList
        fsm={escalateFsm()}
        ensembleAblation={[
          { configuration: "Full ensemble", t3_hits: "7/8", peak_score: 0.997 },
        ]}
      />
    );
    expect(screen.getByText(/Ensemble Ablation/i)).toBeInTheDocument();
    expect(screen.getByText("Full ensemble")).toBeInTheDocument();
    expect(screen.getByText("7/8")).toBeInTheDocument();
  });
});

describe("ReviewGate", () => {
  it("shows standby outside ESCALATE", () => {
    render(<ReviewGate fsm={null} />);
    expect(screen.getByText(/STANDBY/i)).toBeInTheDocument();
  });

  it("keeps decisions disabled until evidence is acknowledged and a rationale entered", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    render(<ReviewGate fsm={escalateFsm()} />);

    // Packet loads, then the acknowledge step appears.
    const ack = await screen.findByText(/ACKNOWLEDGE PACKET REVIEWED/i);
    const approve = screen.getByRole("button", { name: /APPROVE/ });
    const reject = screen.getByRole("button", { name: /REJECT/ });
    const defer = screen.getByRole("button", { name: /DEFER/ });
    expect(approve).toBeDisabled();
    expect(reject).toBeDisabled();
    expect(defer).toBeDisabled();

    fireEvent.click(ack);
    // Still disabled: a written rationale is required.
    expect(approve).toBeDisabled();

    fireEvent.change(screen.getByLabelText(/Review rationale/i), {
      target: { value: "Reviewed evidence; approving." },
    });
    await waitFor(() => expect(approve).toBeEnabled());
    expect(reject).toBeEnabled();
    expect(defer).toBeEnabled();
  });

  it("keeps the decision row pinned so the controls stay reachable", async () => {
    // The review gate scrolls its evidence; the decision row is sticky so an
    // operator never has to scroll to find APPROVE/REJECT/DEFER during an
    // ESCALATE. Dropping this class silently hides the controls below the fold.
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    render(<ReviewGate fsm={escalateFsm()} />);
    const approve = await screen.findByRole("button", { name: /APPROVE/ });
    const row = approve.closest(".review-actions");
    expect(row).not.toBeNull();
    // A nested scroll container would bind the sticky row to an inner
    // scrollport that never scrolls, making the pinning inert.
    expect(row?.closest(".scroll")).toBeNull();
  });

  it("submits durable packet row identity and canonical hash", async () => {
    const durablePacket = packet();
    vi.mocked(fetchEscalationPacket).mockResolvedValue(durablePacket);
    vi.mocked(submitReview).mockResolvedValue({});
    render(<ReviewGate fsm={escalateFsm()} />);

    fireEvent.click(await screen.findByText(/ACKNOWLEDGE PACKET REVIEWED/i));
    fireEvent.change(screen.getByLabelText(/Review rationale/i), {
      target: { value: "Reviewed immutable packet." },
    });
    fireEvent.click(screen.getByRole("button", { name: /APPROVE/ }));

    await waitFor(() => {
      expect(submitReview).toHaveBeenCalledWith(
        {
          event_id: durablePacket.event_id,
          decision: "APPROVE",
          decision_reason: "Reviewed immutable packet.",
          escalation_packet_row_id: durablePacket.packet_row_id,
          escalation_packet_hash: durablePacket.content_sha256,
        },
        "test-key",
        "test-reviewer"
      );
    });
    expect(await screen.findByText("RECORDED THIS SESSION")).toBeInTheDocument();
  });

  it("records one decision when the button is clicked repeatedly in one tick", async () => {
    // The disabled attribute only lands on the next render, so clicks arriving
    // in the same tick all pass the submitting check. Measured in a browser:
    // three clicks, three POSTs, three review records for one decision.
    vi.mocked(submitReview).mockClear();
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    vi.mocked(submitReview).mockResolvedValue({ status: "recorded" } as never);
    render(<ReviewGate fsm={escalateFsm()} />);
    await waitFor(() => expect(screen.getByText(/ACKNOWLEDGE PACKET REVIEWED/)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/ACKNOWLEDGE PACKET REVIEWED/));
    fireEvent.change(screen.getByLabelText("Review rationale"), {
      target: { value: "Consistent with the packet of record." },
    });

    const approve = screen.getByRole("button", { name: "APPROVE" });
    approve.click();
    approve.click();
    approve.click();

    await waitFor(() => expect(submitReview).toHaveBeenCalledTimes(1));
  });

  it("renders no reserved alert terminology", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    const { container } = render(<ReviewGate fsm={escalateFsm()} />);
    await screen.findByText(/ACKNOWLEDGE PACKET REVIEWED/i);
    expect(container.textContent ?? "").not.toMatch(NWS_TERMS);
  });

  it("shows a visible error and retry when the packet fails to load", async () => {
    // A configured-but-unreachable core makes the BFF answer 503; the reviewer
    // must see the failure (and be unable to acknowledge) rather than a blank box.
    vi.mocked(fetchEscalationPacket).mockRejectedValue(
      new Error("Core hazard API is unreachable")
    );
    render(<ReviewGate fsm={escalateFsm()} />);
    expect(
      await screen.findByText(/Core hazard API is unreachable/i)
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /RETRY/i })).toBeInTheDocument();
    expect(
      screen.queryByText(/ACKNOWLEDGE PACKET REVIEWED/i)
    ).not.toBeInTheDocument();
    // A failed load must NOT auto-refetch in a sustained loop: once the error
    // is shown, no further fetches fire until the operator hits RETRY.
    const callsWhenErrorShown = vi.mocked(fetchEscalationPacket).mock.calls.length;
    await new Promise((resolve) => setTimeout(resolve, 150));
    expect(vi.mocked(fetchEscalationPacket).mock.calls.length).toBe(
      callsWhenErrorShown
    );
  });

  it("renders immutable assessment identity for the reviewer", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    render(<ReviewGate fsm={escalateFsm()} />);
    expect(await screen.findByText("11111111-111")).toBeInTheDocument();
    expect(screen.getByText("PACKET OF RECORD")).toBeInTheDocument();
  });

  it("labels the DART row by event-mode state, not as confirmation", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    const withEventMode = escalateFsm();
    const { unmount } = render(<ReviewGate fsm={withEventMode} />);
    expect(await screen.findByText("OBSERVED SINCE ORIGIN")).toBeInTheDocument();
    unmount();

    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    const noEventMode = escalateFsm();
    noEventMode.event_context = {
      ...tohokuContext(),
      dart_confirmation: false,
    };
    render(<ReviewGate fsm={noEventMode} />);
    expect(await screen.findByText("NOT OBSERVED")).toBeInTheDocument();
  });
});

function reviewAuditEntry(overrides: Partial<AuditEntry> = {}): AuditEntry {
  return {
    entry_id: "22222222-2222-4222-8222-222222222222",
    timestamp_utc: "2011-03-11T06:20:00+00:00",
    event_id: "tohoku-2011-03-11T05:46:24Z",
    event_type: "assessment_review_decision",
    producer: "duty-scientist-1",
    data: {
      decision: "APPROVE",
      escalation_packet_hash: "a".repeat(64),
      decided_at_utc: "2011-03-11T06:20:00+00:00",
    },
    ...overrides,
  };
}

describe("AuditLog activity strip", () => {
  it("collapses a run of identical entries into one row with a count", () => {
    const scored = (n: number): AuditEntry => ({
      entry_id: `scored-${n}`,
      timestamp_utc: `2011-03-11T06:1${n}:00+00:00`,
      event_id: "tohoku-2011-03-11T05:46:24Z",
      event_type: "anomaly_scored",
      producer: "anomaly_agent",
      data: {},
    });
    render(
      <AuditLog
        entries={[
          scored(1),
          scored(2),
          scored(3),
          {
            entry_id: "transition-1",
            timestamp_utc: "2011-03-11T06:09:00+00:00",
            event_id: "tohoku-2011-03-11T05:46:24Z",
            event_type: "state_transition",
            producer: "orchestrator",
            data: { from_state: "ASSESS", to_state: "ESCALATE" },
          },
        ]}
      />
    );
    // Without collapsing, the three scoring rows push the transition out of a
    // ten-row strip during live operation.
    expect(screen.getByText("x3")).toBeInTheDocument();
    expect(screen.getAllByText(/anomaly_scored/)).toHaveLength(1);
    expect(screen.getByText(/FSM->ESCALATE/)).toBeInTheDocument();
  });

  it("does not merge distinct transitions that share an event type", () => {
    // IDLE to MONITOR and MONITOR to ESCALATE are both state_transition from
    // the orchestrator. Keying the collapse on the event type hid the
    // escalation behind a count.
    const transition = (id: string, to: string): AuditEntry => ({
      entry_id: id,
      timestamp_utc: "2011-03-11T05:46:24+00:00",
      event_id: "tohoku-2011-03-11T05:46:24Z",
      event_type: "state_transition",
      producer: "fsm-orchestrator",
      data: { from_state: "IDLE", to_state: to },
    });
    render(<AuditLog entries={[transition("t1", "MONITOR"), transition("t2", "ESCALATE")]} />);
    expect(screen.getByText(/FSM->MONITOR/)).toBeInTheDocument();
    expect(screen.getByText(/FSM->ESCALATE/)).toBeInTheDocument();
    expect(screen.queryByText("x2")).not.toBeInTheDocument();
  });

  it("labels a recorded review with the event type the core writes", () => {
    render(
      <AuditLog
        entries={[
          {
            entry_id: "review-1",
            timestamp_utc: "2011-03-11T06:20:00+00:00",
            event_id: "tohoku-2011-03-11T05:46:24Z",
            event_type: "assessment_review_decision",
            producer: "duty-scientist-1",
            data: { decision: "APPROVE" },
          },
        ]}
      />
    );
    expect(screen.getByText("REVIEW: APPROVE")).toBeInTheDocument();
  });
});

describe("recordedReviewFor", () => {
  it("binds a recorded decision to the packet hash, not just the event", () => {
    const entries = [reviewAuditEntry()];
    expect(recordedReviewFor(entries, "tohoku-2011-03-11T05:46:24Z", "a".repeat(64))).not.toBeNull();
    // A superseding packet for the same event has not been reviewed just
    // because an earlier packet was.
    expect(recordedReviewFor(entries, "tohoku-2011-03-11T05:46:24Z", "d".repeat(64))).toBeNull();
    expect(recordedReviewFor(entries, "other-event", "a".repeat(64))).toBeNull();
    expect(recordedReviewFor(undefined, "tohoku-2011-03-11T05:46:24Z", "a".repeat(64))).toBeNull();
  });

  it("returns the newest decision when a packet was decided more than once", () => {
    const older = reviewAuditEntry({
      timestamp_utc: "2011-03-11T06:20:00+00:00",
      data: { decision: "DEFER", escalation_packet_hash: "a".repeat(64) },
    });
    const newer = reviewAuditEntry({
      entry_id: "33333333-3333-4333-8333-333333333333",
      timestamp_utc: "2011-03-11T07:05:00+00:00",
      data: { decision: "APPROVE", escalation_packet_hash: "a".repeat(64) },
    });
    expect(
      recordedReviewFor([older, newer], "tohoku-2011-03-11T05:46:24Z", "a".repeat(64))?.data?.decision
    ).toBe("APPROVE");
    expect(
      recordedReviewFor([newer, older], "tohoku-2011-03-11T05:46:24Z", "a".repeat(64))?.data?.decision
    ).toBe("APPROVE");
  });
});

describe("ReviewGate recorded decision", () => {
  it("shows the durable decision instead of live buttons, and allows superseding it", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    render(<ReviewGate fsm={escalateFsm()} reviewHistory={[reviewAuditEntry()]} />);

    await waitFor(() => expect(screen.getByText("Decision recorded")).toBeInTheDocument());
    expect(screen.getByText("APPROVE")).toBeInTheDocument();
    expect(screen.getByText("duty-scientist-1")).toBeInTheDocument();
    expect(screen.getByText("2011-03-11 06:20:00Z")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "REJECT" })).not.toBeInTheDocument();
    expect(screen.getByText("RECORDED IN AUDIT")).toBeInTheDocument();

    // Pinned for the same reason the decision row is: on handover the record
    // must be visible without scrolling past the evidence above it.
    expect(screen.getByText("Decision recorded").closest(".review-recorded")).toBeTruthy();

    fireEvent.click(screen.getByRole("button", { name: /superseding decision/i }));
    expect(screen.getByRole("button", { name: "REJECT" })).toBeInTheDocument();
    // The restored buttons start disabled; say why rather than leaving them dim.
    expect(screen.getByText(/write a reason to enable the decision/i)).toBeInTheDocument();
  });

  it("converts a non-UTC decision time instead of relabelling it", async () => {
    // The core writes UTC, but relabelling an offset as Z would misstate when a
    // review was recorded, which is exactly what an audit record must not do.
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    render(
      <ReviewGate
        fsm={escalateFsm()}
        reviewHistory={[
          reviewAuditEntry({
            data: {
              decision: "APPROVE",
              escalation_packet_hash: "a".repeat(64),
              decided_at_utc: "2011-03-11T15:20:00+09:00",
            },
          }),
        ]}
      />
    );
    await waitFor(() => expect(screen.getByText("Decision recorded")).toBeInTheDocument());
    expect(screen.getByText("2011-03-11 06:20:00Z")).toBeInTheDocument();
  });

  it("reports the reviewed packet upward so escalation urgency can stand down", async () => {
    vi.mocked(fetchEscalationPacket).mockResolvedValue(packet());
    const onReviewedChange = vi.fn();
    render(
      <ReviewGate fsm={escalateFsm()} reviewHistory={[reviewAuditEntry()]} onReviewedChange={onReviewedChange} />
    );
    await waitFor(() => expect(onReviewedChange).toHaveBeenCalledWith(true));
  });

  it("names the mismatch instead of silently ignoring a decision on a stale packet", async () => {
    // Earlier suites in this file submit real reviews against the same module
    // mock, so the call count carries over without this.
    vi.mocked(submitReview).mockClear();
    vi.mocked(fetchEscalationPacket).mockResolvedValue({
      ...packet(),
      event_id: "some-other-event",
    });
    render(<ReviewGate fsm={escalateFsm()} />);
    await waitFor(() => expect(screen.getByText(/ACKNOWLEDGE PACKET REVIEWED/)).toBeInTheDocument());
    fireEvent.click(screen.getByText(/ACKNOWLEDGE PACKET REVIEWED/));
    fireEvent.change(screen.getByLabelText("Review rationale"), {
      target: { value: "Recommend review" },
    });
    fireEvent.click(screen.getByRole("button", { name: "APPROVE" }));
    await waitFor(() =>
      expect(screen.getByText(/belongs to a different event/i)).toBeInTheDocument()
    );
    expect(submitReview).not.toHaveBeenCalled();
  });
});

describe("reserved terminology", () => {
  it("is absent from the other panels' rendered output", () => {
    const { container: h } = render(
      <Header connected={true} fsmState="ESCALATE" hasActiveEvent={true} anomalyScore={0.997} thresholds={null} />
    );
    const { container: e } = render(<EventList fsm={escalateFsm()} />);
    const { container: a } = render(<AgentStatus agents={[]} />);
    for (const c of [h, e, a]) {
      expect(c.textContent ?? "").not.toMatch(NWS_TERMS);
    }
  });
});

describe("OceanMap legend", () => {
  it("draws legend swatches from the same palette as the markers", async () => {
    // The legend is hand-built markup next to the DivIcon HTML. Both read from
    // MARKER_COLORS so a marker recolor cannot silently leave the legend behind.
    const { MARKER_COLORS } = await import("../components/mapMarkers");
    expect(MARKER_COLORS.dart).toMatch(/^#[0-9a-f]{6}$/i);
    expect(
      new Set(Object.values(MARKER_COLORS)).size,
    ).toBe(Object.keys(MARKER_COLORS).length);
  });
});
