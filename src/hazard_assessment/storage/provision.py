"""Database schema provisioning for TimescaleDB.

Applies SQL migrations in deterministic filename order and records applied
versions in a schema_migrations table.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg
from psycopg import sql

MIGRATION_FILENAME_GLOB = "[0-9][0-9][0-9]_*.sql"
# Advisory lock key for serializing concurrent migration runs.
# Any fixed integer works; every process that provisions must use this one.
MIGRATION_LOCK_KEY = 1742910653
DEFAULT_RETRY_DELAY_SEC = 2.0
DEFAULT_MAX_ATTEMPTS = 30
LIMITED_ROLE_NAMES = (
    "ingest_writer",
    "agent_writer",
    "agent_reader",
    "audit_reader",
    "orchestrator_writer",
    "pipeline_worker",
    "investigator_writer",
)


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    name: str
    admin_user: str
    admin_password: str
    default_role_password: str
    ingest_writer_password: str
    agent_writer_password: str
    agent_reader_password: str
    audit_reader_password: str
    orchestrator_writer_password: str
    pipeline_worker_password: str
    investigator_writer_password: str
    connect_timeout_sec: int

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        default_role_password = os.getenv(
            "DB_DEFAULT_ROLE_PASSWORD",
            os.getenv("DB_PASSWORD", ""),
        )
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "hazard_assessment"),
            admin_user=os.getenv("DB_ADMIN_USER", "hazard_admin"),
            admin_password=os.getenv("DB_ADMIN_PASSWORD", ""),
            default_role_password=default_role_password,
            ingest_writer_password=os.getenv("DB_INGEST_WRITER_PASSWORD") or default_role_password,
            agent_writer_password=os.getenv("DB_AGENT_WRITER_PASSWORD") or default_role_password,
            agent_reader_password=os.getenv("DB_AGENT_READER_PASSWORD") or default_role_password,
            audit_reader_password=os.getenv("DB_AUDIT_READER_PASSWORD") or default_role_password,
            orchestrator_writer_password=(
                os.getenv("DB_ORCHESTRATOR_WRITER_PASSWORD") or default_role_password
            ),
            pipeline_worker_password=(
                os.getenv("DB_PIPELINE_WORKER_PASSWORD") or default_role_password
            ),
            investigator_writer_password=(
                os.getenv("DB_INVESTIGATOR_WRITER_PASSWORD") or default_role_password
            ),
            connect_timeout_sec=int(os.getenv("DB_CONNECT_TIMEOUT", "10")),
        )

    def role_passwords(self) -> dict[str, str]:
        return {
            "ingest_writer": self.ingest_writer_password,
            "agent_writer": self.agent_writer_password,
            "agent_reader": self.agent_reader_password,
            "audit_reader": self.audit_reader_password,
            "orchestrator_writer": self.orchestrator_writer_password,
            "pipeline_worker": self.pipeline_worker_password,
            "investigator_writer": self.investigator_writer_password,
        }


@dataclass(frozen=True)
class Migration:
    version: str
    path: Path
    checksum: str
    sql_text: str


@dataclass(frozen=True)
class ProvisionResult:
    applied_versions: list[str]
    skipped_versions: list[str]


def discover_migrations(migrations_dir: Path) -> list[Migration]:
    files = sorted(migrations_dir.glob(MIGRATION_FILENAME_GLOB))
    migrations: list[Migration] = []
    for path in files:
        sql_text = path.read_text(encoding="utf-8")
        checksum = hashlib.sha256(sql_text.encode("utf-8")).hexdigest()
        migrations.append(
            Migration(
                version=path.stem,
                path=path,
                checksum=checksum,
                sql_text=sql_text,
            )
        )
    if not migrations:
        raise FileNotFoundError(
            f"No SQL migrations found in {migrations_dir} matching {MIGRATION_FILENAME_GLOB}"
        )
    return migrations


def ensure_migration_registry(conn: psycopg.Connection[Any]) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            checksum TEXT NOT NULL,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        )
        """
    )


def apply_migrations(
    conn: psycopg.Connection[Any],
    migrations: Sequence[Migration],
) -> ProvisionResult:
    ensure_migration_registry(conn)
    conn.execute("SELECT pg_advisory_lock(%s)", (MIGRATION_LOCK_KEY,))

    applied_versions: list[str] = []
    skipped_versions: list[str] = []

    try:
        for migration in migrations:
            existing_row = conn.execute(
                "SELECT checksum FROM schema_migrations WHERE version = %s",
                (migration.version,),
            ).fetchone()

            if existing_row is not None:
                existing_checksum = str(existing_row[0])
                if existing_checksum != migration.checksum:
                    raise RuntimeError(
                        "Migration checksum mismatch for "
                        f"{migration.version}: recorded={existing_checksum}, "
                        f"current={migration.checksum}."
                    )
                skipped_versions.append(migration.version)
                continue

            conn.execute(migration.sql_text)
            conn.execute(
                """
                INSERT INTO schema_migrations (version, checksum)
                VALUES (%s, %s)
                """,
                (migration.version, migration.checksum),
            )
            conn.commit()
            applied_versions.append(migration.version)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.execute("SELECT pg_advisory_unlock(%s)", (MIGRATION_LOCK_KEY,))

    return ProvisionResult(
        applied_versions=applied_versions,
        skipped_versions=skipped_versions,
    )


def log_server_notice(diag: Any) -> None:
    """Forward a PostgreSQL notice to the logger at its own severity.

    Applying the schema emits a dozen benign NOTICE lines ("extension
    already exists, skipping"). Logging those at WARNING alongside the
    role-password warnings would bury the ones an operator has to act on,
    so the server's severity decides the level. ``severity_nonlocalized``
    is used in preference to ``severity`` because the latter is translated
    under a non-English ``lc_messages``.
    """
    severity = getattr(diag, "severity_nonlocalized", None) or diag.severity or ""
    level = {
        "DEBUG": logging.DEBUG,
        "LOG": logging.DEBUG,
        "INFO": logging.INFO,
        "NOTICE": logging.INFO,
        "WARNING": logging.WARNING,
    }.get(severity.upper(), logging.WARNING)
    logging.getLogger(__name__).log(level, "%s: %s", severity, diag.message_primary)


def connect_with_retry(
    config: DatabaseConfig,
    max_attempts: int,
    retry_delay_sec: float,
) -> psycopg.Connection[Any]:
    attempt = 1
    while True:
        try:
            return psycopg.connect(
                host=config.host,
                port=config.port,
                dbname=config.name,
                user=config.admin_user,
                password=config.admin_password,
                connect_timeout=config.connect_timeout_sec,
            )
        except psycopg.OperationalError:
            if attempt >= max_attempts:
                raise
            logging.getLogger(__name__).info(
                "Database not ready yet "
                "(attempt %d/%d); retrying in %.1fs...",
                attempt, max_attempts, retry_delay_sec,
            )
            attempt += 1
            time.sleep(retry_delay_sec)


def configure_limited_roles(
    conn: psycopg.Connection[Any],
    role_passwords: dict[str, str],
) -> None:
    """Ensure application roles stay constrained across reruns."""
    for role_name in LIMITED_ROLE_NAMES:
        role_ident = sql.Identifier(role_name)
        conn.execute(
            sql.SQL(
                "ALTER ROLE {role} WITH LOGIN NOINHERIT "
                "NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION"
            ).format(role=role_ident)
        )
        role_password = role_passwords.get(role_name, "")
        if role_password != "":
            # ALTER ROLE ... PASSWORD does not accept a bound parameter (the
            # server rejects the "$1" placeholder), so render a safely quoted
            # literal. This mirrors the migrations' format(..., PASSWORD %L)
            # idiom in 001/003/009.
            conn.execute(
                sql.SQL("ALTER ROLE {role} WITH PASSWORD {password}").format(
                    role=role_ident,
                    password=sql.Literal(role_password),
                )
            )
    conn.commit()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Apply SQL migrations to TimescaleDB.")
    parser.add_argument(
        "--migrations-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "migrations",
        help="Directory containing SQL migration files.",
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=DEFAULT_MAX_ATTEMPTS,
        help="Maximum database connection attempts while waiting for readiness.",
    )
    parser.add_argument(
        "--retry-delay-sec",
        type=float,
        default=DEFAULT_RETRY_DELAY_SEC,
        help="Delay between DB connection retries.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    # Without this every message below is discarded: they are all logger.info
    # and the root logger defaults to WARNING. init-db then exits 0 having
    # printed nothing at all, including the "Database not ready yet" retries
    # during a wait that can run to two minutes. Same setup as the two workers.
    logging.basicConfig(
        level=os.environ.get("APP_LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s  %(message)s",
    )
    args = parse_args(argv)
    config = DatabaseConfig.from_env()
    migrations = discover_migrations(args.migrations_dir)

    with connect_with_retry(
        config=config,
        max_attempts=args.max_attempts,
        retry_delay_sec=args.retry_delay_sec,
    ) as conn:
        # The migrations RAISE WARNING when a role password was not supplied.
        # psycopg drops server notices unless a handler is registered, so
        # without this the operator sees a clean success for a role that was
        # left without a usable password.
        conn.add_notice_handler(log_server_notice)
        role_passwords = config.role_passwords()
        for role_name, role_password in role_passwords.items():
            conn.execute(
                "SELECT set_config(%s, %s, false)",
                (f"hazard.{role_name}_password", role_password),
            )
        result = apply_migrations(conn, migrations)
        configure_limited_roles(conn, role_passwords=role_passwords)

    _log = logging.getLogger(__name__)
    _log.info(
        "Schema provisioning complete: applied=%d skipped=%d",
        len(result.applied_versions), len(result.skipped_versions),
    )
    if result.applied_versions:
        _log.info("Applied migrations: %s", ", ".join(result.applied_versions))
    if result.skipped_versions:
        _log.info("Skipped migrations: %s", ", ".join(result.skipped_versions))

    # A role left without a password cannot authenticate, and the failure
    # otherwise surfaces much later as an opaque connection error in a worker.
    # Name them here rather than letting a clean "provisioning complete" stand
    # for a cluster that is only partly usable.
    unset = sorted(name for name, password in role_passwords.items() if password == "")
    if unset:
        _log.warning(
            "No password supplied for %d role(s): %s. They cannot authenticate "
            "until one is set.",
            len(unset), ", ".join(unset),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
