-- 010: make the ocean-evidence assessment identity CHECK NULL-safe.
--
-- Migration 009 added processed_features_assessment_identity_chk as:
--
--     feature_type <> 'ocean_evidence_assessment'
--     OR (checkpoint_id ~ '^[0-9a-f]{64}$' AND ... AND event_id IS NOT NULL)
--
-- A CHECK constraint passes when its expression is true OR NULL. On an
-- assessment row the left side is false, and a NULL checkpoint_id makes
-- `checkpoint_id ~ '...'` evaluate to NULL rather than false, so the whole
-- expression is NULL and the row is accepted. The same holds for
-- assessment_schema_version, input_manifest_hash and scientific_content_hash.
-- Only `event_id IS NOT NULL` was NULL-safe.
--
-- That defeats the invariant 009 relies on. Its comment claims the CHECK
-- "forbids NULL checkpoint identity on such rows", which is the stated
-- justification for the partial unique index
-- processed_features_assessment_checkpoint_uq. Because NULLs are distinct in a
-- unique index, an assessment row with a NULL checkpoint_id would also escape
-- that index and the ON CONFLICT (checkpoint_id, assessment_schema_version)
-- idempotency in persist_assessment. The append-only triggers from 009 would
-- then make the row permanently undeletable.
--
-- 009 is checksummed by provision.py, so the constraint is replaced here
-- rather than edited in place. No live caller inserts assessment rows through
-- the generic insert_processed_feature path today, so this closes a latent
-- hole rather than repairing existing data.

DO $$
BEGIN
    ALTER TABLE processed_features
        DROP CONSTRAINT IF EXISTS processed_features_assessment_identity_chk;

    ALTER TABLE processed_features
        ADD CONSTRAINT processed_features_assessment_identity_chk CHECK (
            feature_type <> 'ocean_evidence_assessment'
            OR (
                checkpoint_id IS NOT NULL
                AND checkpoint_id ~ '^[0-9a-f]{64}$'
                AND assessment_schema_version IS NOT NULL
                AND assessment_schema_version >= 1
                AND input_manifest_hash IS NOT NULL
                AND input_manifest_hash ~ '^[0-9a-f]{64}$'
                AND scientific_content_hash IS NOT NULL
                AND scientific_content_hash ~ '^[0-9a-f]{64}$'
                AND (
                    transport_provenance_hash IS NULL
                    OR transport_provenance_hash ~ '^[0-9a-f]{64}$'
                )
                AND event_id IS NOT NULL
            )
        );
END $$;
