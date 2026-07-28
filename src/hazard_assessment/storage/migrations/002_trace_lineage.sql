-- 002_trace_lineage.sql
-- Trace propagation and lineage API
--
-- Adds trace_id to audit_events and processed_features, creates a
-- GIN index for input_hashes containment queries, and replaces the
-- baseline provenance_chain view with a full 3-table join that
-- includes raw_observations.

-- ============================================================
-- 1. ADD trace_id COLUMNS
-- ============================================================

ALTER TABLE audit_events
    ADD COLUMN IF NOT EXISTS trace_id UUID;

ALTER TABLE processed_features
    ADD COLUMN IF NOT EXISTS trace_id UUID;

-- Index: filter audit entries by trace_id (single pipeline run)
CREATE INDEX IF NOT EXISTS idx_audit_trace_id
    ON audit_events (trace_id)
    WHERE trace_id IS NOT NULL;

-- Index: filter processed features by trace_id
CREATE INDEX IF NOT EXISTS idx_proc_feat_trace_id
    ON processed_features (trace_id)
    WHERE trace_id IS NOT NULL;

-- ============================================================
-- 2. GIN INDEX for input_hashes (future containment queries)
--
-- Supports queries like:
--   WHERE input_hashes @> ARRAY['<hash>']   (find audit events that used a given input)
-- The default GIN array_ops class does NOT accelerate = ANY(); the
-- provenance_chain view's join relies on idx_raw_obs_hash (B-tree on
-- raw_observations.payload_hash) from 001_baseline instead.
-- ============================================================

CREATE INDEX IF NOT EXISTS idx_audit_input_hashes_gin
    ON audit_events USING GIN (input_hashes);

-- ============================================================
-- 3. REPLACE provenance_chain VIEW
--
-- Full 3-table join: processed_features -> audit_events -> raw_observations.
-- Adds trace_id to output columns and uses LATERAL UNNEST to join
-- raw_observations via individual hashes from input_hashes, with
-- DISTINCT ON to prevent row multiplication when multiple raw
-- observations match a single processed feature.
-- ============================================================

-- NOTE: this redefinition inserts trace_id ahead of handoff_id and adds raw_*
-- columns, which changes the column order/names from the 001 view. Postgres
-- forbids CREATE OR REPLACE VIEW from renaming/reordering existing columns
-- ("cannot change name of view column"), so we must DROP and recreate. The
-- view has no dependents other than the grants re-applied below.
DROP VIEW IF EXISTS provenance_chain;
CREATE VIEW provenance_chain AS
SELECT DISTINCT ON (pf.id, ro.id)
    pf.id            AS output_id,
    pf.feature_type,
    pf.event_id,
    pf.trace_id,
    pf.handoff_id,
    pf.produced_at,
    pf.producer_agent,
    pf.source_refs,
    pf.code_version,
    ae.action        AS audit_action,
    ae.trace_id      AS audit_trace_id,
    ae.state_before,
    ae.state_after,
    ae.input_hashes,
    ae.decision_basis,
    ae.llm_invoked,
    ae.reasoning_trace,
    ro.id            AS raw_observation_id,
    ro.station_id    AS raw_station_id,
    ro.source_type   AS raw_source_type,
    ro.observed_at   AS raw_observed_at,
    ro.payload_hash  AS raw_payload_hash
FROM processed_features pf
LEFT JOIN audit_events ae
    ON ae.handoff_id = pf.handoff_id
LEFT JOIN LATERAL unnest(ae.input_hashes) AS h(hash) ON true
LEFT JOIN raw_observations ro
    ON ro.payload_hash = h.hash
ORDER BY pf.id, ro.id, ae.recorded_at DESC;

-- Maintain same permissions as 001_baseline
GRANT SELECT ON provenance_chain TO audit_reader;
REVOKE ALL ON provenance_chain FROM PUBLIC;
REVOKE SELECT ON provenance_chain FROM agent_reader;
