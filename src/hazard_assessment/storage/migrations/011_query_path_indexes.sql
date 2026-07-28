-- 011: indexes for two verified query paths that had none.
--
-- audit_events: the unfiltered audit listing (GET /api/audit with no
-- parameters) runs
--
--     SELECT * FROM audit_events ORDER BY recorded_at DESC, id DESC
--     LIMIT %s OFFSET %s
--
-- The existing indexes lead with event_id, agent_name, handoff_id, or
-- trace_id, so none serves that ordering: the planner does a sequential scan
-- of the append-only table plus a top-N sort on every unfiltered dashboard
-- refresh. Measured on a seeded 20k-row copy, LIMIT 100 goes from a
-- seq-scan+sort plan (cost ~1192) to an index scan (cost ~3.4) with this
-- index. The id tiebreaker matches the query's deterministic-order contract
-- from migration/code history (equal recorded_at rows).
--
-- escalation_packets: get_escalation_packet_for_event filters
-- WHERE event_id = %s. The table only has its primary key and the
-- (assessment_row_id, renderer_version) unique index, so the packet-of-record
-- lookup on the review path scans the table. Volume is small by design (one
-- packet per assessment row per renderer version), so this is cheap
-- future-proofing for the hot review path rather than a measured win.

CREATE INDEX IF NOT EXISTS idx_audit_recorded_at
    ON audit_events (recorded_at DESC, id DESC);

CREATE INDEX IF NOT EXISTS idx_escalation_packets_event_id
    ON escalation_packets (event_id);
