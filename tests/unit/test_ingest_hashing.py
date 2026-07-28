"""Tests for payload hashing and raw payload store."""

from __future__ import annotations

import hashlib
from pathlib import Path

from hazard_assessment.ingest.hashing import (
    FilePayloadStore,
    InMemoryPayloadStore,
    canonicalize_json,
    compute_payload_hash,
)

# --- compute_payload_hash ---


def test_compute_payload_hash_deterministic() -> None:
    data = b"hello world"
    expected = hashlib.sha256(data).hexdigest()
    assert compute_payload_hash(data) == expected
    assert compute_payload_hash(data) == expected  # same input -> same output


def test_compute_payload_hash_different_inputs() -> None:
    assert compute_payload_hash(b"a") != compute_payload_hash(b"b")


def test_compute_payload_hash_empty_bytes() -> None:
    expected = hashlib.sha256(b"").hexdigest()
    assert compute_payload_hash(b"") == expected


# --- canonicalize_json ---


def test_canonicalize_json_sorts_keys() -> None:
    result = canonicalize_json({"z": 1, "a": 2})
    assert result == b'{"a":2,"z":1}'


def test_canonicalize_json_compact_separators() -> None:
    result = canonicalize_json({"key": "value"})
    assert b" " not in result
    assert result == b'{"key":"value"}'


def test_canonicalize_json_nested_objects() -> None:
    data = {"outer": {"b": 2, "a": 1}}
    result = canonicalize_json(data)
    assert result == b'{"outer":{"a":1,"b":2}}'


def test_canonicalize_json_with_none_and_numbers() -> None:
    data = {"a": None, "b": 1.5, "c": True}
    result = canonicalize_json(data)
    assert result == b'{"a":null,"b":1.5,"c":true}'


def test_canonicalize_json_deterministic_for_same_data() -> None:
    data = {"mag": 6.1, "place": "coast", "time": 123456}
    assert canonicalize_json(data) == canonicalize_json(data)


def test_canonicalize_json_rejects_non_finite_floats() -> None:
    import pytest

    with pytest.raises(ValueError, match="not JSON compliant"):
        canonicalize_json({"value": float("inf")})
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonicalize_json({"value": float("nan")})
    with pytest.raises(ValueError, match="not JSON compliant"):
        canonicalize_json({"value": float("-inf")})


# --- InMemoryPayloadStore ---


def test_in_memory_store_roundtrip() -> None:
    store = InMemoryPayloadStore()
    sha = "abc123"
    data = b"payload"
    store.store(sha, data)
    assert store.contains(sha) is True
    assert store.retrieve(sha) == data


def test_in_memory_store_missing_key() -> None:
    store = InMemoryPayloadStore()
    assert store.contains("missing") is False
    assert store.retrieve("missing") is None


def test_in_memory_store_idempotent() -> None:
    store = InMemoryPayloadStore()
    sha = "abc123"
    store.store(sha, b"original")
    store.store(sha, b"different")  # second store is no-op (first-write-wins)
    assert store.retrieve(sha) == b"original"


# --- FilePayloadStore ---


def test_file_store_roundtrip(tmp_path: Path) -> None:
    store = FilePayloadStore(tmp_path)
    sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    data = b"raw payload bytes"
    store.store(sha, data)
    assert store.contains(sha) is True
    assert store.retrieve(sha) == data
    expected_path = tmp_path / "ab" / sha
    assert expected_path.exists()


def test_file_store_missing_key(tmp_path: Path) -> None:
    store = FilePayloadStore(tmp_path)
    sha = "0000000000000000000000000000000000000000000000000000000000000000"
    assert store.contains(sha) is False
    assert store.retrieve(sha) is None


def test_file_store_idempotent(tmp_path: Path) -> None:
    store = FilePayloadStore(tmp_path)
    sha = "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890"
    store.store(sha, b"first")
    store.store(sha, b"first")  # second store is no-op
    assert store.retrieve(sha) == b"first"


# --- Hash integration with canonicalize_json ---


def test_canonical_json_hash_determinism() -> None:
    data = {"z": 1, "a": [3, 2, 1], "m": {"y": True, "x": None}}
    canonical = canonicalize_json(data)
    h1 = compute_payload_hash(canonical)
    h2 = compute_payload_hash(canonicalize_json(data))
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest length
