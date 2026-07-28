-- 003_fsm_and_fixes.sql
-- FSM state persistence, schema fixes, and provenance function.
--
-- 1. FSM current-state table (single-row, shared by api-server & pipeline-worker)
-- 2. Schema fixes: chunk interval, nullable decision_basis, dedup index, grants
-- 3. orchestrator_writer role for FSM mutations
-- 4. Parameterized provenance function replacing LATERAL UNNEST view

-- ============================================================
-- 1. FSM STATE TABLE
-- ============================================================

CREATE TABLE IF NOT EXISTS fsm_current_state (
    id              INTEGER      PRIMARY KEY DEFAULT 1 CHECK (id = 1),
    current_state   TEXT         NOT NULL DEFAULT 'IDLE',
    event_context   JSONB,
    sensor_degraded BOOLEAN      NOT NULL DEFAULT FALSE,
    updated_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Seed the single row if not present
INSERT INTO fsm_current_state (id) VALUES (1) ON CONFLICT DO NOTHING;

-- ============================================================
-- 2. SCHEMA FIXES
-- ============================================================

-- 2a. Explicit chunk interval for raw_observations (default 7 days is too coarse
--     for high-frequency 15-second DART event-mode data)
SELECT set_chunk_time_interval('raw_observations', INTERVAL '1 day');

-- 2b. decision_basis must be nullable - non-transition audit entries (e.g. data
--     ingestion, permission checks) have no decision to record
ALTER TABLE audit_events ALTER COLUMN decision_basis DROP NOT NULL;

-- 2c. agent_writer needs SELECT on raw_observations for lineage lookups
GRANT SELECT ON raw_observations TO agent_writer;

-- 2d. Dedup constraint: prevent duplicate raw observations from concurrent
--     ingest workers. Uses partial index on the three identifying columns.
CREATE UNIQUE INDEX IF NOT EXISTS idx_raw_obs_dedup
    ON raw_observations (station_id, observed_at, payload_hash);

-- ============================================================
-- 3. ORCHESTRATOR_WRITER ROLE
-- ============================================================

-- Role for api-server and pipeline-worker FSM mutations.
-- Password set via current_setting() from provision.py.
-- No PASSWORD clause: the role is created with a NULL password, matching
-- 001_baseline.sql, so no password-authenticated connection can succeed
-- until one is set. A literal placeholder here would instead ship a
-- published, network-usable credential on a LOGIN role for every operator
-- who did not set the session variable. Note that a NULL password does not
-- block a pg_hba "trust" line, so container-local access still depends on
-- the host-based authentication config.
DO $$ BEGIN
    CREATE ROLE orchestrator_writer WITH
        NOINHERIT LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- Set the password from a session variable (set by provision.py before the
-- migration runs). If the variable is not set the role keeps the password it
-- already had, which for a role created just above is NULL, so it cannot log
-- in until an operator supplies one. That case is raised as a WARNING;
-- provision.py installs a notice handler so the warning reaches the log.
DO $$
DECLARE
    pw TEXT;
BEGIN
    pw := current_setting('hazard.orchestrator_writer_password', true);
    IF pw IS NULL OR pw = '' THEN
        RAISE WARNING 'hazard.orchestrator_writer_password not set - orchestrator_writer keeps its existing password (NULL if just created, so it cannot log in)';
    ELSE
        EXECUTE format(
            'ALTER ROLE orchestrator_writer WITH PASSWORD %L',
            pw
        );
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE ON fsm_current_state TO orchestrator_writer;
GRANT INSERT ON audit_events TO orchestrator_writer;
GRANT SELECT ON audit_events TO orchestrator_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO orchestrator_writer;

-- ============================================================
-- 4. PROVENANCE FUNCTION
--
-- Replaces the LATERAL UNNEST provenance_chain view from 002 with a
-- parameterized function. The view's LATERAL join prevents TimescaleDB
-- chunk exclusion, causing full-table scans. A function with explicit
-- WHERE trace_id = $1 allows the planner to push the predicate down.
-- ============================================================

CREATE OR REPLACE FUNCTION get_provenance(p_trace_id UUID)
RETURNS TABLE (
    output_id          BIGINT,
    feature_type       TEXT,
    event_id           UUID,
    trace_id           UUID,
    handoff_id         UUID,
    produced_at        TIMESTAMPTZ,
    producer_agent     TEXT,
    source_refs        JSONB,
    code_version       TEXT,
    audit_action       TEXT,
    audit_trace_id     UUID,
    state_before       TEXT,
    state_after        TEXT,
    input_hashes       TEXT[],
    decision_basis     TEXT,
    llm_invoked        BOOLEAN,
    reasoning_trace    TEXT,
    raw_observation_id BIGINT,
    raw_station_id     TEXT,
    raw_source_type    TEXT,
    raw_observed_at    TIMESTAMPTZ,
    raw_payload_hash   TEXT
) LANGUAGE sql STABLE AS $$
    SELECT DISTINCT ON (pf.id, ro.id)
        pf.id,
        pf.feature_type,
        pf.event_id,
        pf.trace_id,
        pf.handoff_id,
        pf.produced_at,
        pf.producer_agent,
        pf.source_refs,
        pf.code_version,
        ae.action,
        ae.trace_id,
        ae.state_before,
        ae.state_after,
        ae.input_hashes,
        ae.decision_basis,
        ae.llm_invoked,
        ae.reasoning_trace,
        ro.id,
        ro.station_id,
        ro.source_type,
        ro.observed_at,
        ro.payload_hash
    FROM processed_features pf
    LEFT JOIN audit_events ae
        ON ae.handoff_id = pf.handoff_id
       AND ae.trace_id = p_trace_id
    LEFT JOIN LATERAL unnest(ae.input_hashes) AS h(hash) ON true
    LEFT JOIN raw_observations ro
        ON ro.payload_hash = h.hash
    WHERE pf.trace_id = p_trace_id
    ORDER BY pf.id, ro.id, ae.recorded_at DESC;
$$;

-- Grant execute to roles that need lineage queries
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO agent_reader;
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO audit_reader;
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO orchestrator_writer;
