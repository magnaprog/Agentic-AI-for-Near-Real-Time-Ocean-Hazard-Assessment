/** Minimal fetch wrapper for REST calls to the BFF, plus the session-credential
 * accessors shared by the REST and WebSocket paths. */

import { useCallback } from "react";
import { useAuth } from "../auth/useAuth";

const BASE = "/api/mc";
const MISSION_CONTROL_API_KEY_HEADER_NAME = "X-Mission-Control-Api-Key";
const REVIEWER_ID_HEADER_NAME = "X-Reviewer-Id";

/** Fired on window when a stored access key is rejected (HTTP 401, a WS 1008
 * close, or a probe-confirmed 401 after a WS closed before opening), so the
 * console re-locks and asks for a fresh key. */
export const AUTH_REQUIRED_EVENT = "mc:auth-required";

export function signalAuthRequired(): void {
  if (typeof window === "undefined") return;
  window.dispatchEvent(new Event(AUTH_REQUIRED_EVENT));
}

/** How long the auth probe waits before treating the BFF as unreachable. */
const PROBE_TIMEOUT_MS = 4000;

/** Ask the BFF whether the stored key is actually rejected.
 *
 * A WebSocket that closes before opening is ambiguous: the BFF refuses a bad
 * key at the HTTP handshake (403, surfaced as close code 1006), but a BFF that
 * is restarting or unreachable produces the same close. This probe
 * disambiguates with one cheap authenticated GET:
 *
 * - resolves `true`: the server answered 401, so the key is rejected.
 * - resolves `false`: the server answered anything else, so the key stands and
 *   the close was transient.
 * - resolves `null`: the server could not be reached (network error, timeout),
 *   so nothing is known about the key.
 *
 * Deliberately a raw fetch rather than `apiFetch`: the caller decides whether
 * to re-lock, and a probe against a down server must not throw.
 */
export async function probeAuthRejected(apiKey: string): Promise<boolean | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), PROBE_TIMEOUT_MS);
  try {
    const resp = await fetch(`${BASE}/state`, {
      headers: { [MISSION_CONTROL_API_KEY_HEADER_NAME]: apiKey },
      signal: controller.signal,
    });
    return resp.status === 401;
  } catch {
    return null;
  } finally {
    clearTimeout(timer);
  }
}

/** Hook form of the credential accessors. Reads live state from context so a
 * fresh key is used immediately after the operator unlocks. */
export function useCredentials() {
  const { apiKey, reviewerId } = useAuth();
  const getApiKey = useCallback(() => apiKey, [apiKey]);
  const getReviewerId = useCallback(() => reviewerId, [reviewerId]);
  return { getApiKey, getReviewerId };
}

export async function apiFetch<T>(
  path: string,
  apiKey: string,
  options?: RequestInit
): Promise<T> {
  if (apiKey === "") {
    throw new Error("Mission Control access key is required");
  }

  const headers = new Headers(options?.headers ?? {});
  if (!headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  headers.set(MISSION_CONTROL_API_KEY_HEADER_NAME, apiKey);

  const resp = await fetch(`${BASE}${path}`, {
    ...options,
    headers,
  });
  if (!resp.ok) {
    if (resp.status === 401) {
      signalAuthRequired();
    }
    let detail = resp.statusText;
    try {
      const body = await resp.json();
      if (body.detail) detail = body.detail;
    } catch {
      /* no JSON body - fall back to statusText */
    }
    throw new Error(`API ${resp.status}: ${detail}`);
  }
  return resp.json() as Promise<T>;
}

interface ReviewerPacketContent {
  assessment_row_id: number;
  checkpoint_id: string;
  event_id: string;
  produced_at_utc: string;
  fsm_state_before: string;
  fsm_state_after: string;
  pipeline_outcome: string;
  scientific_content_hash: string;
  best_scoring_station: {
    source: string;
    station_id: string;
    ensemble_score: number;
  } | null;
  dart_stations_currently_in_event_mode: string[];
  recommended_action: string;
  disclaimer: string;
  assessment: {
    handoff_id: string;
    event_id: string;
    scientific_content_hash: string;
  } & Record<string, unknown>;
}

export interface EscalationPacket {
  packet_row_id: number;
  assessment_row_id: number;
  event_id: string;
  renderer_version: string;
  content_sha256: string;
  created_at: string;
  packet: ReviewerPacketContent;
}

export function fetchEscalationPacket(apiKey: string): Promise<EscalationPacket> {
  return apiFetch<EscalationPacket>("/review/escalation", apiKey);
}

export function submitReview(
  body: {
    event_id: string;
    decision: "APPROVE" | "REJECT" | "DEFER";
    decision_reason: string;
    escalation_packet_row_id: number;
    escalation_packet_hash: string;
  },
  apiKey: string,
  reviewerId: string
) {
  if (reviewerId === "") {
    return Promise.reject(new Error("Reviewer ID is required for decision provenance"));
  }
  return apiFetch("/review/decide", apiKey, {
    method: "POST",
    body: JSON.stringify(body),
    headers: { [REVIEWER_ID_HEADER_NAME]: reviewerId },
  });
}
