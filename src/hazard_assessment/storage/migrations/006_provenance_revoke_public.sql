-- 006_provenance_revoke_public.sql
-- 005 made get_provenance() SECURITY DEFINER. PostgreSQL grants EXECUTE on
-- functions to PUBLIC by default, so without an explicit revoke ANY role -
-- including roles never intended to read lineage - could run the privileged,
-- definer-rights function and read audit_events / raw_observations / processed_
-- features through it. Revoke EXECUTE from PUBLIC and re-assert the intended
-- grants. Idempotent.
--
-- Note on search_path hardening (005): the function uses unqualified names in
-- the public schema, but it runs with SET search_path = pg_catalog, public, and
-- PostgreSQL 16 does not grant CREATE on the public schema to PUBLIC by default,
-- so no untrusted role can shadow the referenced tables. Schema-qualifying the
-- body would be redundant given that lockdown.
--
-- Validated against PostgreSQL 16: after this migration, has_function_privilege
-- ('public', 'get_provenance(uuid)', 'EXECUTE') is false, an ungranted role gets
-- "permission denied for function get_provenance", and the three granted roles
-- still execute it.

REVOKE ALL ON FUNCTION get_provenance(UUID) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO agent_reader;
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO audit_reader;
GRANT EXECUTE ON FUNCTION get_provenance(UUID) TO orchestrator_writer;
