-- 004_quarantine.sql
-- Durable quarantine sink for ingest records that fail canonical schema
-- validation. Append-only, mirroring the audit-trail posture:
-- ingest_writer may INSERT/SELECT but not UPDATE/DELETE. Volume is low
-- (only validation failures), so this is a plain table, not a hypertable.

CREATE TABLE IF NOT EXISTS quarantined_records (
    id              BIGSERIAL   PRIMARY KEY,
    source_id       TEXT        NOT NULL,
    source_type     TEXT        NOT NULL,
    reason_code     TEXT        NOT NULL,
    reason_detail   TEXT        NOT NULL,
    raw_fields      JSONB,
    quarantined_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_quarantine_quarantined_at
    ON quarantined_records (quarantined_at DESC);
CREATE INDEX IF NOT EXISTS idx_quarantine_source
    ON quarantined_records (source_type, source_id);

-- Least-privilege grants (roles created in 001_baseline.sql).
GRANT INSERT, SELECT ON quarantined_records TO ingest_writer;
GRANT USAGE ON SEQUENCE quarantined_records_id_seq TO ingest_writer;
GRANT SELECT ON quarantined_records TO audit_reader;

-- Append-only: quarantine records are immutable once written.
REVOKE UPDATE, DELETE ON quarantined_records FROM ingest_writer;

-- Defense-in-depth: block any UPDATE/DELETE at the table level (mirrors the
-- audit_events triggers in 001), so the table stays append-only even if grants
-- are accidentally broadened later.
CREATE OR REPLACE FUNCTION deny_quarantined_records_mutation()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION
        'quarantined_records is append-only; % is not permitted',
        TG_OP
        USING ERRCODE = '42501';
    RETURN NULL;
END;
$$;

DROP TRIGGER IF EXISTS quarantined_records_block_update ON quarantined_records;
CREATE TRIGGER quarantined_records_block_update
BEFORE UPDATE ON quarantined_records
FOR EACH ROW
EXECUTE FUNCTION deny_quarantined_records_mutation();

DROP TRIGGER IF EXISTS quarantined_records_block_delete ON quarantined_records;
CREATE TRIGGER quarantined_records_block_delete
BEFORE DELETE ON quarantined_records
FOR EACH ROW
EXECUTE FUNCTION deny_quarantined_records_mutation();
