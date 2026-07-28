-- 005_provenance_security.sql
-- E8: get_provenance() (migration 003) reads processed_features, audit_events,
-- and raw_observations, and is granted EXECUTE to agent_reader, audit_reader,
-- and orchestrator_writer. But under the default SECURITY INVOKER the function
-- runs with the caller's privileges, and the grants are not uniform:
--   - agent_reader has SELECT on raw_observations + processed_features, but NOT
--     audit_events;
--   - orchestrator_writer has SELECT on audit_events, but NOT raw_observations
--     or processed_features;
--   - only audit_reader has SELECT on all three.
-- So agent_reader and orchestrator_writer would get a permission error executing
-- the function. Rather than broaden direct table access (which would violate
-- least privilege - agent_reader is not meant to read the audit trail directly,
-- and orchestrator_writer is not meant to read raw observations directly), run
-- the function as its definer so the controlled lineage read is exposed without
-- exposing the underlying tables. A hardened search_path prevents object
-- resolution from the caller's schemas (search_path injection); the function
-- body uses unqualified names in the public schema, so public must remain on
-- the path.
--
-- Validated against PostgreSQL 16: with SECURITY DEFINER, a role granted EXECUTE
-- but lacking SELECT on the underlying tables (e.g. agent_reader on audit_events)
-- can run get_provenance(); flipping to SECURITY INVOKER reproduces "permission
-- denied for table audit_events", confirming the fix is both correct and
-- necessary. ALTER FUNCTION is idempotent, so re-running this migration is safe.

ALTER FUNCTION get_provenance(UUID) SECURITY DEFINER;
ALTER FUNCTION get_provenance(UUID) SET search_path = pg_catalog, public;
