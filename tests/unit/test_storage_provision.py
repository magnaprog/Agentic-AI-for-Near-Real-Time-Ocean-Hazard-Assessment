"""Unit tests for storage provisioning utilities."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import pytest

from hazard_assessment.storage import provision
from hazard_assessment.storage.provision import (
    DatabaseConfig,
    Migration,
    apply_migrations,
    configure_limited_roles,
    discover_migrations,
)


class _FakeResult:
    def __init__(
        self,
        row: tuple[Any, ...] | None = None,
    ) -> None:
        self._row = row

    def fetchone(self) -> tuple[Any, ...] | None:
        return self._row


class _FakeConnection:
    def __init__(
        self,
        existing_checksums: dict[str, str] | None = None,
    ) -> None:
        self.existing_checksums = dict(existing_checksums or {})
        self.executed: list[tuple[str, tuple[Any, ...] | None]] = []
        self.commit_calls = 0
        self.rollback_calls = 0
        self.locked = False

    def execute(self, query: Any, params: tuple[Any, ...] | None = None) -> _FakeResult:
        # configure_limited_roles passes psycopg.sql.Composed objects; render
        # them the way a real connection would so assertions see final SQL.
        if not isinstance(query, str):
            query = query.as_string(None)
        normalized = " ".join(query.split())
        self.executed.append((normalized, params))

        if normalized.startswith("SELECT pg_advisory_lock"):
            self.locked = True
            return _FakeResult((True,))

        if normalized.startswith("SELECT pg_advisory_unlock"):
            self.locked = False
            return _FakeResult((True,))

        if normalized.startswith("SELECT checksum FROM schema_migrations"):
            assert params is not None
            version = str(params[0])
            checksum = self.existing_checksums.get(version)
            return _FakeResult((checksum,) if checksum is not None else None)

        if normalized.startswith("INSERT INTO schema_migrations"):
            assert params is not None
            version = str(params[0])
            checksum = str(params[1])
            self.existing_checksums[version] = checksum
            return _FakeResult(None)

        return _FakeResult(None)

    def commit(self) -> None:
        self.commit_calls += 1

    def rollback(self) -> None:
        self.rollback_calls += 1


def _make_migration(version: str, sql_text: str = "SELECT 1;") -> Migration:
    checksum = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
    return Migration(
        version=version,
        path=Path(f"/tmp/{version}.sql"),
        checksum=checksum,
        sql_text=sql_text,
    )


def test_discover_migrations_sorts_by_filename(tmp_path: Path) -> None:
    (tmp_path / "010_second.sql").write_text("SELECT 2;", encoding="utf-8")
    (tmp_path / "001_first.sql").write_text("SELECT 1;", encoding="utf-8")

    migrations = discover_migrations(tmp_path)
    assert [migration.version for migration in migrations] == ["001_first", "010_second"]


def test_discover_migrations_raises_when_empty(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No SQL migrations found"):
        discover_migrations(tmp_path)


def test_database_config_defaults_to_admin_role(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (
        "DB_HOST",
        "DB_PORT",
        "DB_NAME",
        "DB_ADMIN_USER",
        "DB_ADMIN_PASSWORD",
        "DB_USER",
        "DB_PASSWORD",
        "DB_DEFAULT_ROLE_PASSWORD",
        "DB_INGEST_WRITER_PASSWORD",
        "DB_AGENT_WRITER_PASSWORD",
        "DB_AGENT_READER_PASSWORD",
        "DB_AUDIT_READER_PASSWORD",
        "DB_ORCHESTRATOR_WRITER_PASSWORD",
        "DB_PIPELINE_WORKER_PASSWORD",
        "DB_INVESTIGATOR_WRITER_PASSWORD",
        "DB_CONNECT_TIMEOUT",
    ):
        monkeypatch.delenv(var, raising=False)

    config = DatabaseConfig.from_env()

    assert config.admin_user == "hazard_admin"
    assert config.admin_password == ""
    assert config.default_role_password == ""
    assert config.role_passwords() == {
        "ingest_writer": "",
        "agent_writer": "",
        "agent_reader": "",
        "audit_reader": "",
        "orchestrator_writer": "",
        "pipeline_worker": "",
        "investigator_writer": "",
    }


def test_database_config_uses_default_and_explicit_role_passwords(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DB_ADMIN_USER", "admin_user")
    monkeypatch.setenv("DB_ADMIN_PASSWORD", "admin_secret")
    monkeypatch.setenv("DB_DEFAULT_ROLE_PASSWORD", "default_secret")
    monkeypatch.setenv("DB_AGENT_READER_PASSWORD", "agent_reader_secret")

    config = DatabaseConfig.from_env()

    assert config.admin_user == "admin_user"
    assert config.admin_password == "admin_secret"
    assert config.role_passwords() == {
        "ingest_writer": "default_secret",
        "agent_writer": "default_secret",
        "agent_reader": "agent_reader_secret",
        "audit_reader": "default_secret",
        "orchestrator_writer": "default_secret",
        "pipeline_worker": "default_secret",
        "investigator_writer": "default_secret",
    }


def test_database_config_falls_back_to_db_password(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DB_DEFAULT_ROLE_PASSWORD", raising=False)
    monkeypatch.setenv("DB_PASSWORD", "legacy_default_password")

    config = DatabaseConfig.from_env()

    assert config.default_role_password == "legacy_default_password"
    assert config.role_passwords() == {
        "ingest_writer": "legacy_default_password",
        "agent_writer": "legacy_default_password",
        "agent_reader": "legacy_default_password",
        "audit_reader": "legacy_default_password",
        "orchestrator_writer": "legacy_default_password",
        "pipeline_worker": "legacy_default_password",
        "investigator_writer": "legacy_default_password",
    }


def test_database_config_does_not_use_db_user_as_admin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DB_ADMIN_USER", raising=False)
    monkeypatch.setenv("DB_USER", "agent_reader")

    config = DatabaseConfig.from_env()

    assert config.admin_user == "hazard_admin"


def test_apply_migrations_records_applied_versions() -> None:
    conn = _FakeConnection()
    migrations = [
        _make_migration("001_baseline", "CREATE TABLE test_1(id int);"),
        _make_migration("002_more", "CREATE TABLE test_2(id int);"),
    ]

    result = apply_migrations(conn, migrations)

    assert result.applied_versions == ["001_baseline", "002_more"]
    assert result.skipped_versions == []
    assert conn.commit_calls == 2
    assert not conn.locked


def test_apply_migrations_skips_existing_checksum_match() -> None:
    migration = _make_migration("001_baseline")
    conn = _FakeConnection(existing_checksums={migration.version: migration.checksum})

    result = apply_migrations(conn, [migration])

    assert result.applied_versions == []
    assert result.skipped_versions == ["001_baseline"]
    assert conn.commit_calls == 0
    assert conn.rollback_calls == 0
    assert not conn.locked


def test_apply_migrations_fails_on_checksum_mismatch() -> None:
    migration = _make_migration("001_baseline", "SELECT 1;")
    conn = _FakeConnection(existing_checksums={migration.version: "bad-checksum"})

    with pytest.raises(RuntimeError, match="checksum mismatch"):
        apply_migrations(conn, [migration])

    assert conn.rollback_calls == 1
    assert not conn.locked


def test_configure_limited_roles_enforces_login_noinherit_constraints() -> None:
    conn = _FakeConnection()
    configure_limited_roles(
        conn,
        role_passwords={
            "ingest_writer": "",
            "agent_writer": "",
            "agent_reader": "",
            "audit_reader": "",
        },
    )

    assert conn.commit_calls == 1
    for role_name in ("ingest_writer", "agent_writer", "agent_reader", "audit_reader"):
        assert any(
            query.startswith(
                f'ALTER ROLE "{role_name}" WITH LOGIN NOINHERIT NOSUPERUSER '
                "NOCREATEDB NOCREATEROLE NOREPLICATION"
            )
            for query, _ in conn.executed
        )
    assert all(
        "WITH PASSWORD %s" not in query
        for query, _ in conn.executed
    )


def test_configure_limited_roles_sets_passwords_when_provided() -> None:
    conn = _FakeConnection()
    configure_limited_roles(
        conn,
        role_passwords={
            "ingest_writer": "pw-ingest",
            "agent_writer": "",
            "agent_reader": "pw-reader",
            "audit_reader": "",
        },
    )

    assert conn.commit_calls == 1
    # Passwords render as safely quoted SQL literals, matching the migrations'
    # format(..., PASSWORD %L) idiom, never as bound parameters.
    assert any(
        query == 'ALTER ROLE "ingest_writer" WITH PASSWORD \'pw-ingest\''
        for query, _ in conn.executed
    )
    assert any(
        query == 'ALTER ROLE "agent_reader" WITH PASSWORD \'pw-reader\''
        for query, _ in conn.executed
    )
    assert not any(
        '"agent_writer" WITH PASSWORD' in query for query, _ in conn.executed
    )
    assert not any(
        '"audit_reader" WITH PASSWORD' in query for query, _ in conn.executed
    )
    # Regression guard: a bound parameter here produces "ALTER ROLE ... WITH
    # PASSWORD $1", which PostgreSQL rejects with a syntax error. No password
    # statement may carry bound params.
    for query, params in conn.executed:
        if "WITH PASSWORD" in query:
            assert "%s" not in query and "$1" not in query
            assert params is None


class _FakeDiagnostic:
    """Stands in for psycopg's Diagnostic in notice-handler tests."""

    def __init__(self, severity_nonlocalized: str | None, message_primary: str) -> None:
        self.severity_nonlocalized = severity_nonlocalized
        self.severity = severity_nonlocalized
        self.message_primary = message_primary


def test_server_notices_log_at_their_own_severity(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A benign NOTICE must not be logged at WARNING.

    Applying the schema emits roughly a dozen "already exists, skipping"
    notices. Promoting those to WARNING buries the role-password warnings,
    which are the ones an operator has to act on.
    """
    with caplog.at_level(logging.DEBUG, logger=provision.__name__):
        provision.log_server_notice(
            _FakeDiagnostic("NOTICE", 'extension "timescaledb" already exists, skipping')
        )
        provision.log_server_notice(
            _FakeDiagnostic("WARNING", "hazard.agent_writer_password not set")
        )

    levels = {record.levelno for record in caplog.records}
    assert logging.WARNING in levels
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "password not set" in warnings[0].getMessage()
    assert any(r.levelno == logging.INFO for r in caplog.records)


def test_server_notice_without_severity_defaults_to_warning() -> None:
    """An unrecognised or absent severity must not be silently downgraded."""
    handler_input = _FakeDiagnostic(None, "unclassified server message")
    logger = logging.getLogger(provision.__name__)
    records: list[logging.LogRecord] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    capture = _Capture()
    logger.addHandler(capture)
    previous_level = logger.level
    logger.setLevel(logging.DEBUG)
    try:
        provision.log_server_notice(handler_input)
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)

    assert [r.levelno for r in records] == [logging.WARNING]
