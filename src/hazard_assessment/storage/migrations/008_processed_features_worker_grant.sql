-- The pipeline worker is the processed_features lineage producer (it
-- persists qc_report and anomaly_score rows per batch), but it connects as
-- orchestrator_writer, and 001 granted processed_features INSERT only to
-- agent_writer. Without this grant the producer's best-effort inserts fail
-- with permission denied in any role-correct deployment, silently leaving
-- get_provenance() with no feature rows. Sequence usage is already covered
-- by 003's GRANT USAGE ON ALL SEQUENCES to orchestrator_writer.
--
-- A dedicated pipeline-worker role (FSM + audit + feature write, nothing
-- else) would be the cleaner long-term split; this is the minimal correct
-- grant for the current single-worker topology.

GRANT INSERT ON processed_features TO orchestrator_writer;
