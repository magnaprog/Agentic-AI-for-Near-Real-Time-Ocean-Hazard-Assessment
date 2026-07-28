"""Tests for schema version negotiation (S-T2)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from hazard_assessment.schemas.envelope import SCHEMA_VERSION, BaseEnvelope
from hazard_assessment.schemas.versioning import (
    SchemaVersionError,
    check_schema_version,
    parse_version,
)

# --- parse_version ---


def test_parse_version_valid() -> None:
    assert parse_version("1.0") == (1, 0)
    assert parse_version("3.2") == (3, 2)
    assert parse_version("0.0") == (0, 0)


def test_parse_version_rejects_non_string() -> None:
    # A non-string must not reach ``.split``: AttributeError is not wrapped
    # by pydantic, so it would escape an envelope validator as an unhandled
    # exception rather than a field error.
    for value in (1.0, None, 12, ["1", "0"]):
        with pytest.raises(ValueError, match="major.minor"):
            parse_version(value)  # type: ignore[arg-type]


def test_parse_version_rejects_single_component() -> None:
    with pytest.raises(ValueError, match="major.minor"):
        parse_version("1")


def test_parse_version_rejects_three_components() -> None:
    with pytest.raises(ValueError, match="major.minor"):
        parse_version("1.0.0")


def test_parse_version_rejects_non_integer() -> None:
    with pytest.raises(ValueError, match="integers"):
        parse_version("a.b")


def test_parse_version_rejects_negative() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        parse_version("-1.0")


def test_parse_version_rejects_empty_string() -> None:
    with pytest.raises(ValueError, match="major.minor"):
        parse_version("")


# --- check_schema_version ---


def test_check_schema_version_same_version_passes() -> None:
    check_schema_version("1.0", "1.0")


def test_check_schema_version_minor_diff_passes() -> None:
    check_schema_version("1.3", "1.0")
    check_schema_version("1.0", "1.5")


def test_check_schema_version_major_diff_raises() -> None:
    with pytest.raises(SchemaVersionError) as exc_info:
        check_schema_version("2.0", "1.0")
    assert exc_info.value.incoming == "2.0"
    assert exc_info.value.current == "1.0"
    assert exc_info.value.migration_available is False


def test_check_schema_version_uses_current_default() -> None:
    check_schema_version(SCHEMA_VERSION)


# --- MigrationRegistry ---


def test_envelope_accepts_current_version() -> None:
    env = BaseEnvelope.model_validate(
        {"schema_version": SCHEMA_VERSION, "producer": "test"}
    )
    assert env.schema_version == SCHEMA_VERSION


def test_envelope_accepts_minor_version_difference() -> None:
    env = BaseEnvelope.model_validate(
        {"schema_version": "1.5", "producer": "test"}
    )
    assert env.schema_version == "1.5"


def test_envelope_rejects_incompatible_major_version() -> None:
    with pytest.raises(ValidationError) as exc_info:
        BaseEnvelope.model_validate(
            {"schema_version": "2.0", "producer": "test"}
        )
    errors = exc_info.value.errors()
    assert any(
        "schema version" in str(e).lower() or "incompatible" in str(e).lower()
        for e in errors
    )


def test_envelope_accepts_default_version_when_omitted() -> None:
    env = BaseEnvelope.model_validate({"producer": "test"})
    assert env.schema_version == SCHEMA_VERSION


def test_envelope_subclass_inherits_version_validation() -> None:
    """Subclasses of BaseEnvelope should also validate schema_version."""
    from hazard_assessment.schemas.qc import QCReport

    with pytest.raises(ValidationError):
        QCReport.model_validate(
            {"schema_version": "9.0", "producer": "qc_agent"}
        )
