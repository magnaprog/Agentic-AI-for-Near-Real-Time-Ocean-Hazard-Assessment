/** TypeScript interfaces matching the backend Pydantic schemas. */

export type SystemState =
  | "IDLE"
  | "MONITOR"
  | "INVESTIGATE"
  | "ASSESS"
  | "ESCALATE";

export interface EventContext {
  event_id: string;
  seismic_magnitude: number;
  seismic_region: string;
  epicenter_lat: number;
  epicenter_lon: number;
  trigger_time_utc: string;
  latest_anomaly_score: number;
  dart_confirmation: boolean;
  active_dart_stations: string[];
  stations_in_event_mode: string[];
}

export interface Thresholds {
  basin: string;
  t1: number;
  t2: number;
  t3: number;
}

export interface Transition {
  transition_id: string;
  event_id: string | null;
  timestamp_utc: string;
  from_state: SystemState;
  to_state: SystemState;
  trigger_reason: string;
  anomaly_score: number | null;
  seismic_magnitude: number | null;
}

export interface FSMState {
  fsm_state: SystemState;
  has_active_event: boolean;
  recovery_failed?: boolean;
  /** True when fewer than two DART stations carry QC-usable data in the
   *  worker's retained window: below the minimum for triangulation. It never
   *  gates FSM transitions, so it qualifies the evidence rather than the
   *  state. */
  sensor_degraded?: boolean;
  event_context: EventContext | null;
  thresholds: Thresholds;
  transition_history: Transition[];
}

export interface Agent {
  name: string;
  version: string;
  execution_path: string;
  description: string;
}

/** Known fields on audit log data payloads; index signature preserves extensibility. */
export interface AuditEntryData {
  from_state?: string;
  to_state?: string;
  trigger_reason?: string;
  decision?: string;
  decision_reason?: string;
  /** Written by the core on assessment_review_decision. The packet hash is
   *  what lets the console tell a review of THIS packet from a review of an
   *  earlier one for the same event. */
  escalation_packet_hash?: string;
  decided_at_utc?: string;
  [key: string]: unknown;
}

export interface AuditEntry {
  entry_id: string;
  timestamp_utc: string;
  event_id: string | null;
  event_type: string;
  producer: string;
  data: AuditEntryData;
}

/** One row of the retrospective per-station detection-latency table.
 *  t1_minutes / t3_minutes are null when the threshold was never crossed. */
export interface DetectionLatencyRow {
  station_id: string;
  distance_km: number;
  t1_minutes: number | null;
  t3_minutes: number | null;
}

/** One ensemble-configuration row of the retrospective ablation summary. */
export interface AblationRow {
  configuration: string;
  t3_hits: string;
  peak_score: number;
}

/** Retrospective, demo-only enrichment. Absent (undefined) in live operation. */
export interface ScenarioMetrics {
  first_t1_minutes: number | null;
  detection_latency: DetectionLatencyRow[];
  ensemble_ablation: AblationRow[];
}

export interface SystemSnapshot {
  fsm: FSMState;
  agents: Agent[];
  recent_audit: AuditEntry[];
  /** Review decisions only. Queried separately by the BFF because recent_audit
   *  can be saturated by per-window anomaly entries within a fraction of a
   *  second. Absent from the built-in demo snapshot. */
  recent_reviews?: AuditEntry[];
  /** Present only in demo mode, where the BFF broadcasts its built-in
   *  snapshot verbatim with this merged in. Live snapshots are built from
   *  SystemSnapshotOut, which has no such field, so absence means live. */
  demo_mode?: boolean;
  scenario_metrics?: ScenarioMetrics | null;
  /** Names of the snapshot sections whose upstream query failed on this poll:
   *  "agents", "recent_audit", "recent_reviews". Those three degrade to an
   *  empty list instead of failing the whole snapshot, and an empty list on
   *  its own cannot be told apart from a genuinely empty one. Absent from the
   *  built-in demo snapshot, which never queries the core. */
  degraded_sections?: string[];
}

export type WSMessage =
  | { type: "snapshot"; data: SystemSnapshot }
  | { type: "upstream_error"; data: { snapshot_retained: boolean } }
  | { type: "upstream_recovered"; data: Record<string, never> }
  | { type: "heartbeat"; data: { polled_at_utc: string | null } };

export interface StationInfo {
  id: string;
  name: string;
  lat: number;
  lon: number;
}

export interface StationRegistry {
  dart: StationInfo[];
  coops: StationInfo[];
}
