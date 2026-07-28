-- 009: Ocean evidence assessment persistence (implementation plan 6.3).
--
-- Assessment rows live in processed_features with
-- feature_type = 'ocean_evidence_assessment'. This migration adds the
-- checkpoint identity and hash columns, conditional immutability for
-- assessment rows only (qc_report and anomaly_score rows keep current
-- mutable-by-privilege behavior), the partial unique checkpoint index,
-- unique assessment handoff identity, the append-only
-- assessment_checkpoint_attempt audit table, the immutable
-- evidence_issue_results and escalation_packets tables, and the
-- dedicated pipeline_worker role. The generic processed_features INSERT
-- grant that 008 gave orchestrator_writer is revoked only after the
-- pipeline_worker role and its grants exist.

-- ============================================================
-- 1. ASSESSMENT IDENTITY COLUMNS ON processed_features
-- ============================================================

-- The existing integer "version" column is the envelope payload version,
-- not the assessment schema version, so a dedicated column is added.
ALTER TABLE processed_features
    ADD COLUMN IF NOT EXISTS checkpoint_id TEXT,
    ADD COLUMN IF NOT EXISTS assessment_schema_version INTEGER,
    ADD COLUMN IF NOT EXISTS input_manifest_hash TEXT,
    ADD COLUMN IF NOT EXISTS scientific_content_hash TEXT,
    ADD COLUMN IF NOT EXISTS transport_provenance_hash TEXT;

-- Assessment rows must carry a complete identity. The transport hash is
-- nullable because it describes the creation attempt only and replay
-- checkpoints may not have one. Non-assessment rows are unconstrained.
DO $$ BEGIN
    ALTER TABLE processed_features
        ADD CONSTRAINT processed_features_assessment_identity_chk CHECK (
            feature_type <> 'ocean_evidence_assessment'
            OR (
                checkpoint_id ~ '^[0-9a-f]{64}$'
                AND assessment_schema_version >= 1
                AND input_manifest_hash ~ '^[0-9a-f]{64}$'
                AND scientific_content_hash ~ '^[0-9a-f]{64}$'
                AND (
                    transport_provenance_hash IS NULL
                    OR transport_provenance_hash ~ '^[0-9a-f]{64}$'
                )
                AND event_id IS NOT NULL
            )
        );
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

-- One assessment per logical checkpoint key. The predicate covers every
-- ocean-evidence-assessment row because the CHECK above forbids NULL
-- checkpoint identity on such rows.
CREATE UNIQUE INDEX IF NOT EXISTS processed_features_assessment_checkpoint_uq
    ON processed_features (checkpoint_id, assessment_schema_version)
    WHERE feature_type = 'ocean_evidence_assessment';

-- Unique assessment handoff identity (get-by-ID retrieval).
CREATE UNIQUE INDEX IF NOT EXISTS processed_features_assessment_handoff_uq
    ON processed_features (handoff_id)
    WHERE feature_type = 'ocean_evidence_assessment';

-- Latest-by-event retrieval.
CREATE INDEX IF NOT EXISTS processed_features_assessment_event_idx
    ON processed_features (event_id, produced_at DESC)
    WHERE feature_type = 'ocean_evidence_assessment';

-- ============================================================
-- 2. CONDITIONAL IMMUTABILITY AND RETENTION PROTECTION
-- ============================================================

-- Shared append-only guard for assessment rows and the new tables.
CREATE OR REPLACE FUNCTION deny_ocean_evidence_mutation()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION
        'table % is append-only for ocean evidence records: % blocked',
        TG_TABLE_NAME, TG_OP;
END;
$$ LANGUAGE plpgsql;

-- UPDATE is blocked when the old OR new row is an assessment, so a row
-- can neither be edited, re-typed away from, nor forged into an
-- assessment. DELETE is blocked for assessment rows, which is also the
-- retention protection for review-bound and evaluation-manifest
-- assessments: no retention job may remove them.
DROP TRIGGER IF EXISTS processed_features_block_assessment_update
    ON processed_features;
CREATE TRIGGER processed_features_block_assessment_update
    BEFORE UPDATE ON processed_features
    FOR EACH ROW
    WHEN (OLD.feature_type = 'ocean_evidence_assessment'
          OR NEW.feature_type = 'ocean_evidence_assessment')
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

DROP TRIGGER IF EXISTS processed_features_block_assessment_delete
    ON processed_features;
CREATE TRIGGER processed_features_block_assessment_delete
    BEFORE DELETE ON processed_features
    FOR EACH ROW
    WHEN (OLD.feature_type = 'ocean_evidence_assessment')
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

-- ============================================================
-- 3. ASSESSMENT CHECKPOINT ATTEMPT AUDIT (APPEND ONLY)
-- ============================================================

-- Every original and redelivery processing attempt for a checkpoint
-- appends one row here. The assessment row's transport hash describes
-- its creation attempt; later attempts land here instead of mutating it.
CREATE TABLE IF NOT EXISTS assessment_checkpoint_attempt (
    id                          BIGSERIAL   PRIMARY KEY,
    checkpoint_id               TEXT        NOT NULL
        CHECK (checkpoint_id ~ '^[0-9a-f]{64}$'),
    assessment_schema_version   INTEGER     NOT NULL CHECK (assessment_schema_version >= 1),
    attempt_kind                TEXT        NOT NULL
        CHECK (attempt_kind IN ('original', 'redelivery')),
    -- inserted / existing / conflict / build_failed / persist_failed /
    -- audit_repaired: what this attempt observed or did.
    outcome                     TEXT        NOT NULL,
    event_id                    UUID,
    trace_id                    UUID,
    worker_run_id               UUID,
    transport_provenance_hash   TEXT
        CHECK (transport_provenance_hash IS NULL
               OR transport_provenance_hash ~ '^[0-9a-f]{64}$'),
    detail                      TEXT        NOT NULL DEFAULT '',
    occurred_at                 TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS assessment_checkpoint_attempt_checkpoint_idx
    ON assessment_checkpoint_attempt (checkpoint_id, assessment_schema_version);

DROP TRIGGER IF EXISTS assessment_checkpoint_attempt_block_update
    ON assessment_checkpoint_attempt;
CREATE TRIGGER assessment_checkpoint_attempt_block_update
    BEFORE UPDATE ON assessment_checkpoint_attempt
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

DROP TRIGGER IF EXISTS assessment_checkpoint_attempt_block_delete
    ON assessment_checkpoint_attempt;
CREATE TRIGGER assessment_checkpoint_attempt_block_delete
    BEFORE DELETE ON assessment_checkpoint_attempt
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

-- ============================================================
-- 4. EVIDENCE ISSUE RESULTS (IMMUTABLE, SEPARATE WRITERS)
-- ============================================================

-- EvidenceIssueResult rows have different writers (the read-only
-- investigator), repeat-run identity, and retention than worker-produced
-- features, so they get their own small table. The invocation ID is
-- deterministic for a given (assessment, issue, ruleset) invocation and
-- unique, so a repeated run cannot duplicate a result; a repeated run
-- with a different result hash fails the unique constraint and is
-- surfaced as a conflict instead of silently coexisting.
CREATE TABLE IF NOT EXISTS evidence_issue_results (
    id                  BIGSERIAL   PRIMARY KEY,
    invocation_id       TEXT        NOT NULL UNIQUE
        CHECK (invocation_id ~ '^[0-9a-f]{64}$'),
    assessment_row_id   BIGINT      NOT NULL REFERENCES processed_features(id),
    issue_name          TEXT        NOT NULL,
    result              JSONB       NOT NULL,
    result_sha256       TEXT        NOT NULL
        CHECK (result_sha256 ~ '^[0-9a-f]{64}$'),
    produced_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS evidence_issue_results_assessment_idx
    ON evidence_issue_results (assessment_row_id);

DROP TRIGGER IF EXISTS evidence_issue_results_block_update
    ON evidence_issue_results;
CREATE TRIGGER evidence_issue_results_block_update
    BEFORE UPDATE ON evidence_issue_results
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

DROP TRIGGER IF EXISTS evidence_issue_results_block_delete
    ON evidence_issue_results;
CREATE TRIGGER evidence_issue_results_block_delete
    BEFORE DELETE ON evidence_issue_results
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

-- ============================================================
-- 5. ESCALATION PACKETS (IMMUTABLE)
-- ============================================================

-- The exact reviewer-visible packet, tied to the assessment from the
-- checkpoint that entered ESCALATE. Review records reference this
-- durable row rather than process memory. One packet per assessment and
-- renderer version.
CREATE TABLE IF NOT EXISTS escalation_packets (
    id                  BIGSERIAL   PRIMARY KEY,
    assessment_row_id   BIGINT      NOT NULL REFERENCES processed_features(id),
    event_id            UUID        NOT NULL,
    renderer_version    TEXT        NOT NULL,
    packet              JSONB       NOT NULL,
    content_sha256      TEXT        NOT NULL
        CHECK (content_sha256 ~ '^[0-9a-f]{64}$'),
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (assessment_row_id, renderer_version)
);

DROP TRIGGER IF EXISTS escalation_packets_block_update
    ON escalation_packets;
CREATE TRIGGER escalation_packets_block_update
    BEFORE UPDATE ON escalation_packets
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

DROP TRIGGER IF EXISTS escalation_packets_block_delete
    ON escalation_packets;
CREATE TRIGGER escalation_packets_block_delete
    BEFORE DELETE ON escalation_packets
    FOR EACH ROW
    EXECUTE FUNCTION deny_ocean_evidence_mutation();

-- ============================================================
-- 6. PIPELINE_WORKER AND INVESTIGATOR_WRITER ROLES
-- ============================================================

-- Dedicated pipeline-worker role: the existing worker write privileges
-- (FSM state, audit events, lineage feature rows, sequences) plus
-- assessment insertion and attempt auditing. Created BEFORE the generic
-- processed_features INSERT grant is revoked from orchestrator_writer.
-- No PASSWORD clause: created with a NULL password, as in 001_baseline.sql,
-- so no password-authenticated connection succeeds until provision.py
-- supplies one below. Does not override a pg_hba "trust" line.
DO $$ BEGIN
    CREATE ROLE pipeline_worker WITH
        NOINHERIT LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
DECLARE
    pw TEXT;
BEGIN
    pw := current_setting('hazard.pipeline_worker_password', true);
    IF pw IS NULL OR pw = '' THEN
        RAISE WARNING 'hazard.pipeline_worker_password not set - pipeline_worker keeps its existing password (NULL if just created, so it cannot log in)';
    ELSE
        EXECUTE format('ALTER ROLE pipeline_worker WITH PASSWORD %L', pw);
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE ON fsm_current_state TO pipeline_worker;
GRANT INSERT, SELECT ON audit_events TO pipeline_worker;
-- SELECT is required for the idempotent insert-or-return-existing read
-- back of the just-claimed or pre-existing assessment row.
GRANT INSERT, SELECT ON processed_features TO pipeline_worker;
GRANT INSERT, SELECT ON assessment_checkpoint_attempt TO pipeline_worker;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO pipeline_worker;

-- Writer role for the read-only investigator's persisted results. The
-- pipeline worker deliberately does not receive this grant.
-- No PASSWORD clause: created with a NULL password, as in 001_baseline.sql,
-- so no password-authenticated connection succeeds until provision.py
-- supplies one below. Does not override a pg_hba "trust" line.
DO $$ BEGIN
    CREATE ROLE investigator_writer WITH
        NOINHERIT LOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION;
EXCEPTION
    WHEN duplicate_object THEN NULL;
END $$;

DO $$
DECLARE
    pw TEXT;
BEGIN
    pw := current_setting('hazard.investigator_writer_password', true);
    IF pw IS NULL OR pw = '' THEN
        RAISE WARNING 'hazard.investigator_writer_password not set - investigator_writer keeps its existing password (NULL if just created, so it cannot log in)';
    ELSE
        EXECUTE format('ALTER ROLE investigator_writer WITH PASSWORD %L', pw);
    END IF;
END $$;

GRANT INSERT, SELECT ON evidence_issue_results TO investigator_writer;
-- Reading the referenced assessment row is part of writing a result.
GRANT SELECT ON processed_features TO investigator_writer;
GRANT USAGE ON ALL SEQUENCES IN SCHEMA public TO investigator_writer;

-- Reviewer packet writes belong to the pipeline worker alone:
-- it renders and persists the packet for the checkpoint that enters
-- ESCALATE. SELECT covers the idempotent existing-row read back. The
-- API-facing role only reads packets (packet-of-record endpoint).
GRANT INSERT, SELECT ON escalation_packets TO pipeline_worker;
GRANT SELECT ON escalation_packets TO orchestrator_writer;

-- ============================================================
-- 7. REVOKE THE GENERIC WORKER GRANT FROM orchestrator_writer
-- ============================================================

-- 008 granted processed_features INSERT to orchestrator_writer as a
-- stopgap for the single-worker topology. The worker now connects as
-- pipeline_worker, so the API-facing role loses lineage insert.
REVOKE INSERT ON processed_features FROM orchestrator_writer;
