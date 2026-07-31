"""Contract tests for the baseline storage migration SQL.

Everything here reads ``001_baseline.sql`` alone, so it describes the schema
and grants *as that migration authored them*, not the state a provisioned
database ends up in. Two of the baseline's grants have since been undone by
later migrations, and a test named for the architecture role matrix while
reading only the baseline was asserting a role matrix that no longer exists.
The names below say "baseline" for that reason, and
``test_later_migrations_supersede_two_baseline_grants`` records the reversals
so this file cannot drift back into claiming the baseline is the end state.

The authority for effective grants is ``tests/integration/`` against real
PostgreSQL. These are text-level checks on what the SQL says.
"""

from __future__ import annotations

import re
from pathlib import Path

MIGRATIONS_DIR = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "hazard_assessment"
    / "storage"
    / "migrations"
)
MIGRATION_PATH = MIGRATIONS_DIR / "001_baseline.sql"
MIGRATION_SQL = MIGRATION_PATH.read_text(encoding="utf-8")


def test_migration_is_idempotent_for_core_objects() -> None:
    assert "CREATE EXTENSION IF NOT EXISTS timescaledb;" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS raw_observations" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS processed_features" in MIGRATION_SQL
    assert "CREATE TABLE IF NOT EXISTS audit_events" in MIGRATION_SQL
    assert "if_not_exists => TRUE" in MIGRATION_SQL
    assert "CREATE OR REPLACE VIEW provenance_chain AS" in MIGRATION_SQL


def test_migration_defines_required_roles_for_section_8_2() -> None:
    for role in ("ingest_writer", "agent_writer", "agent_reader", "audit_reader"):
        assert f"rolname = '{role}'" in MIGRATION_SQL
    assert (
        "ALTER ROLE ingest_writer WITH LOGIN NOINHERIT NOSUPERUSER "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;"
        in MIGRATION_SQL
    )
    assert (
        "ALTER ROLE agent_writer WITH LOGIN NOINHERIT NOSUPERUSER "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;"
        in MIGRATION_SQL
    )
    assert (
        "ALTER ROLE agent_reader WITH LOGIN NOINHERIT NOSUPERUSER "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;"
        in MIGRATION_SQL
    )
    assert (
        "ALTER ROLE audit_reader WITH LOGIN NOINHERIT NOSUPERUSER "
        "NOCREATEDB NOCREATEROLE NOREPLICATION;"
        in MIGRATION_SQL
    )
    assert "rolname = 'hazard_app'" not in MIGRATION_SQL


def test_baseline_migration_grants_match_role_matrix_as_authored() -> None:
    """Grants exactly as 001 writes them. Two are later revoked; see
    ``test_later_migrations_supersede_two_baseline_grants``."""
    assert "GRANT INSERT, SELECT ON raw_observations TO ingest_writer;" in MIGRATION_SQL
    assert "GRANT INSERT ON audit_events TO ingest_writer;" in MIGRATION_SQL

    # Superseded by 012, which revokes this. Kept because the assertion is
    # about what the baseline authored, and 012 can only revoke a grant the
    # baseline actually made.
    assert "GRANT INSERT ON processed_features TO agent_writer;" in MIGRATION_SQL
    assert "GRANT INSERT ON audit_events TO agent_writer;" in MIGRATION_SQL

    assert "GRANT SELECT ON raw_observations TO agent_reader;" in MIGRATION_SQL
    assert "GRANT SELECT ON processed_features TO agent_reader;" in MIGRATION_SQL

    assert "GRANT SELECT ON audit_events TO audit_reader;" in MIGRATION_SQL
    assert "GRANT SELECT ON raw_observations TO audit_reader;" in MIGRATION_SQL
    assert "GRANT SELECT ON processed_features TO audit_reader;" in MIGRATION_SQL
    # Superseded by 013, which drops the view and its dependent grants.
    assert "GRANT SELECT ON provenance_chain TO audit_reader;" in MIGRATION_SQL
    assert "GRANT SELECT ON provenance_chain TO agent_reader;" not in MIGRATION_SQL
    assert "GRANT ingest_writer TO hazard_app;" not in MIGRATION_SQL


def test_migration_adds_append_only_audit_triggers() -> None:
    assert "CREATE OR REPLACE FUNCTION deny_audit_events_mutation()" in MIGRATION_SQL

    # Anchored inside the function body. An unanchored "RAISE EXCEPTION" in
    # MIGRATION_SQL was satisfied by any raise anywhere in the 200-line file,
    # so replacing this function's body with RETURN NEW, which would make
    # audit_events silently mutable, still passed.
    body = re.search(
        r"CREATE OR REPLACE FUNCTION deny_audit_events_mutation\(\).*?\$\$;",
        MIGRATION_SQL,
        re.S,
    )
    assert body is not None, "the append-only trigger function has been renamed"
    assert "RAISE EXCEPTION" in body.group(0)
    assert "audit_events is append-only" in body.group(0)
    assert "RETURN NEW" not in body.group(0)

    # The triggers that call it. Inserting a test above this one had left
    # these four assertions stranded in that test, so the function named for
    # the triggers no longer checked that any trigger existed.
    assert "CREATE TRIGGER audit_events_block_update" in MIGRATION_SQL
    assert "BEFORE UPDATE ON audit_events" in MIGRATION_SQL

    assert "CREATE TRIGGER audit_events_block_delete" in MIGRATION_SQL
    assert "BEFORE DELETE ON audit_events" in MIGRATION_SQL


def test_later_migrations_supersede_two_baseline_grants() -> None:
    """The baseline is not the effective role matrix.

    Both reversals are load-bearing: 012 removes a write nothing uses, and 013
    drops the view 003's function replaced, whose slower shape stayed the
    discoverable one while the grant survived.
    """
    revoke_sql = (MIGRATIONS_DIR / "012_revoke_unused_feature_writes.sql").read_text(
        encoding="utf-8"
    )
    assert "REVOKE INSERT ON processed_features FROM agent_writer;" in revoke_sql

    drop_sql = (MIGRATIONS_DIR / "013_drop_superseded_provenance_view.sql").read_text(
        encoding="utf-8"
    )
    assert "DROP VIEW IF EXISTS provenance_chain;" in drop_sql


_PROVENANCE_SECURITY_SQL = (
    MIGRATION_PATH.parent / "005_provenance_security.sql"
).read_text(encoding="utf-8")


def test_get_provenance_runs_as_definer_with_hardened_search_path() -> None:
    # get_provenance reads three tables but is granted EXECUTE to roles that
    # lack uniform SELECT on them; SECURITY DEFINER exposes the controlled read
    # without broadening direct table access, and a fixed search_path prevents
    # search_path injection.
    assert (
        "ALTER FUNCTION get_provenance(UUID) SECURITY DEFINER;"
        in _PROVENANCE_SECURITY_SQL
    )
    assert "SET search_path = pg_catalog, public;" in _PROVENANCE_SECURITY_SQL


_PROVENANCE_PG_TEMP_SQL = (
    MIGRATION_PATH.parent / "007_provenance_pg_temp.sql"
).read_text(encoding="utf-8")


def test_get_provenance_search_path_pins_pg_temp_last() -> None:
    # An unlisted pg_temp is searched FIRST for relation names, so a caller
    # with TEMP privilege could shadow the tables this SECURITY DEFINER
    # function reads with a temp view (whose expressions would then run with
    # definer rights). 007 pins pg_temp last so the real tables always win.
    assert (
        "ALTER FUNCTION get_provenance(UUID) "
        "SET search_path = pg_catalog, public, pg_temp;"
        in _PROVENANCE_PG_TEMP_SQL
    )
