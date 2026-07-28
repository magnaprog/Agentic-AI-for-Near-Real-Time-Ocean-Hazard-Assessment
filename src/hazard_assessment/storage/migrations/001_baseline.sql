-- Baseline schema migration for the Ocean Hazard Assessment System
--
-- This migration creates:
-- 1. raw_observations hypertable for ingested sensor data
-- 2. processed_features table for agent outputs
-- 3. audit_events append-only table for full traceability
-- 4. Database roles with least-privilege access control

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- ROLES (least-privilege access control)
--
-- IMPORTANT: Tables are owned by the migration role (typically a
-- superuser or dedicated schema_admin). Application services connect
-- directly as one of the four least-privilege roles below. Table owners
-- retain full privileges, so the migration user must NOT be the same as
-- any application service user.
-- ============================================================

-- PostgreSQL does not support CREATE ROLE IF NOT EXISTS, so we use
-- catalog checks in a DO block for idempotent role creation. Roles are
-- created with a NULL password; provision.py supplies the real one through
-- the hazard.<role>_password session variables. A role left without one is
-- raised as a WARNING, which provision.py's notice handler forwards to the
-- log, so a partially provisioned cluster is not reported as a clean run.
DO $$
DECLARE
    ingest_writer_password TEXT := current_setting('hazard.ingest_writer_password', true);
    agent_writer_password  TEXT := current_setting('hazard.agent_writer_password', true);
    agent_reader_password  TEXT := current_setting('hazard.agent_reader_password', true);
    audit_reader_password  TEXT := current_setting('hazard.audit_reader_password', true);
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'ingest_writer') THEN
        CREATE ROLE ingest_writer LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    ELSE
        ALTER ROLE ingest_writer WITH LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
    IF ingest_writer_password IS NOT NULL AND ingest_writer_password <> '' THEN
        EXECUTE format('ALTER ROLE ingest_writer WITH PASSWORD %L', ingest_writer_password);
    ELSE
        RAISE WARNING 'hazard.ingest_writer_password not set - ingest_writer keeps its existing password (NULL if just created, so it cannot log in)';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_writer') THEN
        CREATE ROLE agent_writer LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    ELSE
        ALTER ROLE agent_writer WITH LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
    IF agent_writer_password IS NOT NULL AND agent_writer_password <> '' THEN
        EXECUTE format('ALTER ROLE agent_writer WITH PASSWORD %L', agent_writer_password);
    ELSE
        RAISE WARNING 'hazard.agent_writer_password not set - agent_writer keeps its existing password (NULL if just created, so it cannot log in)';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'agent_reader') THEN
        CREATE ROLE agent_reader LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    ELSE
        ALTER ROLE agent_reader WITH LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
    IF agent_reader_password IS NOT NULL AND agent_reader_password <> '' THEN
        EXECUTE format('ALTER ROLE agent_reader WITH PASSWORD %L', agent_reader_password);
    ELSE
        RAISE WARNING 'hazard.agent_reader_password not set - agent_reader keeps its existing password (NULL if just created, so it cannot log in)';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'audit_reader') THEN
        CREATE ROLE audit_reader LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    ELSE
        ALTER ROLE audit_reader WITH LOGIN NOINHERIT NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
    END IF;
    IF audit_reader_password IS NOT NULL AND audit_reader_password <> '' THEN
        EXECUTE format('ALTER ROLE audit_reader WITH PASSWORD %L', audit_reader_password);
    ELSE
        RAISE WARNING 'hazard.audit_reader_password not set - audit_reader keeps its existing password (NULL if just created, so it cannot log in)';
    END IF;
END
$$;

-- ============================================================
-- RAW OBSERVATIONS (TimescaleDB hypertable)
-- ============================================================

CREATE TABLE IF NOT EXISTS raw_observations (
    id              BIGSERIAL,
    station_id      TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,  -- 'dart', 'coops', 'seismic'
    observed_at     TIMESTAMPTZ NOT NULL,
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    data_mode       TEXT,                  -- 'STANDARD', 'EVENT' (DART only)
    measurement_type INTEGER,              -- 1=15-min, 2=1-min, 3=15-sec
    raw_value       DOUBLE PRECISION,
    raw_payload     JSONB,
    payload_hash    TEXT        NOT NULL,  -- SHA-256 of raw payload
    source_url      TEXT,
    fetch_metadata  JSONB,                 -- endpoint, params, timestamps

    PRIMARY KEY (id, observed_at)
);

-- Convert to TimescaleDB hypertable partitioned by observation time
SELECT create_hypertable(
    'raw_observations',
    by_range('observed_at'),
    if_not_exists => TRUE
);

CREATE INDEX IF NOT EXISTS idx_raw_obs_station
    ON raw_observations (station_id, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_raw_obs_source
    ON raw_observations (source_type, observed_at DESC);

CREATE INDEX IF NOT EXISTS idx_raw_obs_hash
    ON raw_observations (payload_hash);

-- ============================================================
-- PROCESSED FEATURES (agent outputs)
-- ============================================================

CREATE TABLE IF NOT EXISTS processed_features (
    id              BIGSERIAL   PRIMARY KEY,
    feature_type    TEXT        NOT NULL,  -- 'qc_report', 'anomaly_score', 'scenario', etc.
    event_id        UUID,
    station_id      TEXT,
    produced_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    producer_agent  TEXT        NOT NULL,
    version         INTEGER     NOT NULL DEFAULT 1,
    source_refs     JSONB       NOT NULL DEFAULT '[]',  -- references to raw_observations
    handoff_id      UUID        NOT NULL,
    payload         JSONB       NOT NULL,                -- typed agent output
    code_version    TEXT,
    model_version   TEXT
);

CREATE INDEX IF NOT EXISTS idx_proc_feat_event
    ON processed_features (event_id, produced_at DESC);

CREATE INDEX IF NOT EXISTS idx_proc_feat_type
    ON processed_features (feature_type, produced_at DESC);

CREATE INDEX IF NOT EXISTS idx_proc_feat_handoff
    ON processed_features (handoff_id);

-- ============================================================
-- AUDIT EVENTS (append-only, immutable)
--
-- MAPPING NOTE (E2 implementation): The in-memory AuditEntry
-- (audit/logger.py) uses a flat schema: event_type, producer,
-- data (dict). The E2 persistence layer must map between the two:
--   AuditEntry.producer    -> audit_events.agent_name
--   AuditEntry.event_type  -> audit_events.action
--   AuditEntry.data["from_state"] -> audit_events.state_before
--   AuditEntry.data["to_state"]   -> audit_events.state_after
--   AuditEntry.data["trigger_reason"] -> audit_events.decision_basis
-- Fields without a direct Python source (llm_invoked, input_hashes,
-- reasoning_trace) should be populated by the producing agent or
-- default to NULL / FALSE. The decision_basis NOT NULL constraint
-- requires the persistence layer to always supply a value.
-- ============================================================

CREATE TABLE IF NOT EXISTS audit_events (
    id              BIGSERIAL   PRIMARY KEY,
    event_id        UUID,
    recorded_at     TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    agent_name      TEXT        NOT NULL,
    action          TEXT        NOT NULL,
    state_before    TEXT,
    state_after     TEXT,
    input_hashes    TEXT[],               -- SHA-256 hashes of all inputs
    handoff_id      UUID,
    llm_invoked     BOOLEAN     NOT NULL DEFAULT FALSE,
    model_version   TEXT,
    reasoning_trace TEXT,
    decision_basis  TEXT        NOT NULL,
    metadata        JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_event_id
    ON audit_events (event_id, recorded_at);

CREATE INDEX IF NOT EXISTS idx_audit_agent
    ON audit_events (agent_name, recorded_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_handoff
    ON audit_events (handoff_id);

-- Defense-in-depth: block any UPDATE/DELETE attempt at the table level.
-- These triggers are intentionally owner-enforced and remain active even if
-- grants are accidentally broadened in the future.
CREATE OR REPLACE FUNCTION deny_audit_events_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'audit_events is append-only; % is not permitted',
        TG_OP
        USING ERRCODE = '42501';
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS audit_events_block_update ON audit_events;
CREATE TRIGGER audit_events_block_update
BEFORE UPDATE ON audit_events
FOR EACH ROW
EXECUTE FUNCTION deny_audit_events_mutation();

DROP TRIGGER IF EXISTS audit_events_block_delete ON audit_events;
CREATE TRIGGER audit_events_block_delete
BEFORE DELETE ON audit_events
FOR EACH ROW
EXECUTE FUNCTION deny_audit_events_mutation();

-- ============================================================
-- PERMISSIONS
-- ============================================================

-- Ingest writers: INSERT + SELECT on raw_observations, INSERT on audit_events.
-- SELECT is required for payload_hash deduplication (checking if a record
-- already exists before inserting). All agents (including ingest) have
-- WRITE_AUDIT permission per permissions.yaml.
GRANT INSERT, SELECT ON raw_observations TO ingest_writer;
GRANT USAGE ON SEQUENCE raw_observations_id_seq TO ingest_writer;
GRANT INSERT ON audit_events TO ingest_writer;
GRANT USAGE ON SEQUENCE audit_events_id_seq TO ingest_writer;

-- Agent writers: INSERT on processed_features and audit_events
GRANT INSERT ON processed_features TO agent_writer;
GRANT USAGE ON SEQUENCE processed_features_id_seq TO agent_writer;
GRANT INSERT ON audit_events TO agent_writer;
GRANT USAGE ON SEQUENCE audit_events_id_seq TO agent_writer;

-- Agent readers: SELECT on raw and processed
GRANT SELECT ON raw_observations TO agent_reader;
GRANT SELECT ON processed_features TO agent_reader;

-- Audit readers: SELECT on audit and processed
GRANT SELECT ON audit_events TO audit_reader;
GRANT SELECT ON processed_features TO audit_reader;
GRANT SELECT ON raw_observations TO audit_reader;

-- Restrict default table access from PUBLIC.
REVOKE ALL ON raw_observations FROM PUBLIC;
REVOKE ALL ON processed_features FROM PUBLIC;
REVOKE ALL ON audit_events FROM PUBLIC;

-- CRITICAL: No UPDATE or DELETE on audit_events/raw_observations for
-- any non-owner role.
-- Enforced by only granting INSERT/SELECT - never UPDATE/DELETE.
-- The migration role (table owner) retains full privileges by PostgreSQL
-- design; this is acceptable because migrations are run by ops, not by
-- application code.
--
-- Verify with:
--   SELECT grantee, privilege_type FROM information_schema.role_table_grants
--   WHERE table_name = 'audit_events' AND privilege_type IN ('UPDATE', 'DELETE');
-- Expected result: only the migration/owner role, never application roles.

-- Explicitly revoke UPDATE/DELETE from all application roles on critical
-- append-only tables.
REVOKE UPDATE, DELETE ON audit_events FROM ingest_writer;
REVOKE UPDATE, DELETE ON audit_events FROM agent_writer;
REVOKE UPDATE, DELETE ON audit_events FROM agent_reader;
REVOKE UPDATE, DELETE ON audit_events FROM audit_reader;
REVOKE UPDATE, DELETE ON raw_observations FROM ingest_writer;
REVOKE UPDATE, DELETE ON raw_observations FROM agent_writer;
REVOKE UPDATE, DELETE ON raw_observations FROM agent_reader;
REVOKE UPDATE, DELETE ON raw_observations FROM audit_reader;

-- ============================================================
-- PROVENANCE VIEW (output -> lineage chain)
-- Baseline lineage view; 002_trace_lineage.sql replaces it with the full join.
-- ============================================================

CREATE OR REPLACE VIEW provenance_chain AS
SELECT
    pf.id AS output_id,
    pf.feature_type,
    pf.event_id,
    pf.handoff_id,
    pf.produced_at,
    pf.producer_agent,
    pf.source_refs,
    pf.code_version,
    ae.action AS audit_action,
    ae.state_before,
    ae.state_after,
    ae.input_hashes,
    ae.decision_basis,
    ae.llm_invoked,
    ae.reasoning_trace
FROM processed_features pf
LEFT JOIN audit_events ae ON ae.handoff_id = pf.handoff_id;

GRANT SELECT ON provenance_chain TO audit_reader;
REVOKE ALL ON provenance_chain FROM PUBLIC;
REVOKE SELECT ON provenance_chain FROM agent_reader;
