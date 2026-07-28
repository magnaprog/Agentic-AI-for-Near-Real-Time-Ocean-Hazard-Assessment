"""Payload hashing and immutable raw archive primitives.

Provides deterministic hashing for provenance tracking and abstract
storage for immutable raw payloads keyed by their SHA-256 digest.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path

_SHA256_HEX_RE = re.compile(r"[0-9a-f]{64}\Z")


def compute_payload_hash(raw_bytes: bytes) -> str:
    """Return the SHA-256 hex digest of *raw_bytes*."""
    return hashlib.sha256(raw_bytes).hexdigest()


def canonicalize_json(data: object) -> bytes:
    """Serialize *data* to deterministic canonical JSON bytes.

    Key ordering and compact separators ensure the same logical value
    always produces the same byte sequence for consistent hashing.
    """
    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


class RawPayloadStore(ABC):
    """Abstract store for immutable raw payloads keyed by SHA-256 hex."""

    @abstractmethod
    def store(self, sha256_hex: str, raw_bytes: bytes) -> None:
        """Persist *raw_bytes* under *sha256_hex*. Idempotent."""

    @abstractmethod
    def retrieve(self, sha256_hex: str) -> bytes | None:
        """Return stored bytes or ``None`` if not found."""

    @abstractmethod
    def contains(self, sha256_hex: str) -> bool:
        """Return whether *sha256_hex* exists in the store."""


class InMemoryPayloadStore(RawPayloadStore):
    """In-memory payload store for tests."""

    def __init__(self) -> None:
        self._data: dict[str, bytes] = {}

    def store(self, sha256_hex: str, raw_bytes: bytes) -> None:
        if sha256_hex not in self._data:
            self._data[sha256_hex] = raw_bytes

    def retrieve(self, sha256_hex: str) -> bytes | None:
        return self._data.get(sha256_hex)

    def contains(self, sha256_hex: str) -> bool:
        return sha256_hex in self._data


class FilePayloadStore(RawPayloadStore):
    """File-system payload store with 2-char prefix fan-out."""

    def __init__(self, base_dir: Path) -> None:
        self._base_dir = base_dir

    def _path_for(self, sha256_hex: str) -> Path:
        if not _SHA256_HEX_RE.fullmatch(sha256_hex):
            raise ValueError(f"Invalid SHA-256 hex digest: {sha256_hex!r}")
        return self._base_dir / sha256_hex[:2] / sha256_hex

    def store(self, sha256_hex: str, raw_bytes: bytes) -> None:
        path = self._path_for(sha256_hex)
        if path.exists():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write to a unique temp file in the same directory, then rename
        # atomically. A deterministic .tmp name would let two concurrent
        # same-hash writers truncate each other's temp file.
        fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
        # os.replace() overwrites on POSIX and never raises FileExistsError,
        # so the only paths that leave the temp file behind are real failures
        # (disk full, permission denied). Remove it unconditionally in finally
        # so a failed write cannot litter the archive directory.
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(raw_bytes)
            os.replace(tmp_name, path)  # atomic; overwrites on POSIX
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.unlink(tmp_name)

    def retrieve(self, sha256_hex: str) -> bytes | None:
        path = self._path_for(sha256_hex)
        if not path.exists():
            return None
        return path.read_bytes()

    def contains(self, sha256_hex: str) -> bool:
        return self._path_for(sha256_hex).exists()
