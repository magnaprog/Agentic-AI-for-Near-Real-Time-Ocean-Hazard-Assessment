-- 012_revoke_unused_feature_writes.sql
-- Remove agent_writer's INSERT on processed_features.
--
-- 001_baseline granted it when agent outputs were expected to be written by a
-- shared role. No application code connects as agent_writer, and the only
-- writer of processed_features is the pipeline worker: 009 already revoked
-- orchestrator_writer's INSERT, and the API role reaches lineage through the
-- SECURITY DEFINER get_provenance() function rather than the table.
--
-- Leaving the grant in place is not merely untidy. processed_features holds the
-- ocean-evidence assessments, whose identity is (checkpoint_id,
-- assessment_schema_version) and whose rows are append-only by trigger. Any
-- holder of this grant can therefore claim a checkpoint key before the worker
-- does, which makes the genuine assessment fail to persist as a conflict and
-- leaves a row that not even the table owner can delete or correct. The role
-- shares DB_PASSWORD with the ingest containers by default, so the grant
-- widened the blast radius of an ingest credential to permanent poisoning of
-- the evidence record.
--
-- The sequence grant goes with it: without INSERT there is nothing to draw a
-- key for. agent_writer never held SELECT on this table, so nothing it can
-- currently read is affected. Its lineage reads are SELECT on raw_observations
-- (003) and its INSERT on audit_events, which has no unique key to squat on,
-- are both left alone.

REVOKE INSERT ON processed_features FROM agent_writer;
REVOKE USAGE ON SEQUENCE processed_features_id_seq FROM agent_writer;
