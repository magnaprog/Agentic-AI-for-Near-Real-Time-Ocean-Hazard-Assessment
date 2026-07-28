"""Schema version negotiation (S-T2).

Validates incoming schema versions against the current version. Same major
version passes (minor differences are backward-compatible additions); a
different major version is a hard error. A migration registry existed here
through v1.0 but was removed: no breaking change ever shipped, so it was
plumbing with zero registered migrations. Reintroduce a migration path
together with the first actual major-version bump.
"""

from __future__ import annotations

from typing import Final

from hazard_assessment.schemas.envelope import SCHEMA_VERSION

# Semantic: major.minor where major = breaking, minor = backward-compatible.
_VERSION_PARTS: Final[int] = 2


def parse_version(version: str) -> tuple[int, int]:
    """Parse a ``"major.minor"`` version string.

    Raises ``ValueError`` on malformed input.
    """
    # A non-string reaches ``.split`` as an AttributeError, which pydantic does
    # not wrap into a ValidationError, so it would escape an envelope validator
    # as an unhandled exception instead of a field error.
    if not isinstance(version, str):
        raise ValueError(
            f"Schema version must be a 'major.minor' string (got {type(version).__name__})"
        )
    parts = version.split(".")
    if len(parts) != _VERSION_PARTS:
        raise ValueError(
            f"Schema version must be 'major.minor' (got {version!r})"
        )
    try:
        major = int(parts[0])
        minor = int(parts[1])
    except ValueError as exc:
        raise ValueError(
            f"Schema version components must be integers (got {version!r})"
        ) from exc
    if major < 0 or minor < 0:
        raise ValueError(
            f"Schema version components must be non-negative (got {version!r})"
        )
    return (major, minor)


class SchemaVersionError(ValueError):
    """Raised when an incoming schema version is incompatible."""

    def __init__(
        self,
        *,
        incoming: str,
        current: str,
        migration_available: bool,
    ) -> None:
        self.incoming = incoming
        self.current = current
        self.migration_available = migration_available
        super().__init__(
            f"Incompatible schema version: incoming={incoming}, "
            f"current={current}, migration_available={migration_available}"
        )


def check_schema_version(incoming: str, current: str = SCHEMA_VERSION) -> None:
    """Validate that *incoming* is compatible with *current*.

    - Same major version: pass (minor differences are backward-compatible).
    - Different major: raise ``SchemaVersionError``.
    """
    incoming_major, _ = parse_version(incoming)
    current_major, _ = parse_version(current)
    if incoming_major == current_major:
        return
    raise SchemaVersionError(
        incoming=incoming,
        current=current,
        migration_available=False,
    )
