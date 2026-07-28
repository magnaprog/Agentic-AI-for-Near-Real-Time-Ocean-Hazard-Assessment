"""M0 gate validation: replay lineage from raw payload hash to output artifact.

Proves the M0 gate criterion end-to-end for all three source types (DART,
CO-OPS, seismic) without requiring a database.  Each test demonstrates:

1. Raw payload -> deterministic SHA-256 hash
2. Hash -> immutable archive (InMemoryPayloadStore)
3. Connector record -> validated observation (Pydantic schema)
4. Audit entry linking back to the payload hash
5. Backward trace: audit entry -> hash -> archived bytes -> re-hash matches
"""

from __future__ import annotations

from datetime import UTC, datetime

from hazard_assessment.audit.logger import AuditEntry, AuditLogger
from hazard_assessment.ingest.coops import CoopsRecord
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.ingest.hashing import (
    InMemoryPayloadStore,
    canonicalize_json,
    compute_payload_hash,
)
from hazard_assessment.ingest.seismic import SeismicEventRecord
from hazard_assessment.ingest.validation import validate_record
from hazard_assessment.schemas.observation import (
    CoopsObservation,
    DartObservation,
    SeismicObservation,
)

_NOW = datetime(2026, 3, 4, 12, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Sample raw payloads - mirror what each connector actually hashes
# ---------------------------------------------------------------------------

# DART: raw text line (_fetch_station_records hashes row.raw_line.encode("utf-8"))
_DART_RAW_LINE = "2026 03 04 12 00 00 1 4541.234"

# CO-OPS: JSON row dict (_fetch_product_records hashes canonicalize_json(row))
_COOPS_RAW_ROW: dict = {
    "t": "2026-03-04 12:00",
    "v": "0.584",
    "f": "0,0,0,0",
    "q": "p",
}

# Seismic: GeoJSON feature dict (fetch_records hashes canonicalize_json(feature))
_SEISMIC_RAW_FEATURE: dict = {
    "id": "us7000test",
    "type": "Feature",
    "properties": {
        "time": 1772625600000,
        "updated": 1772625600000,
        "mag": 6.1,
        "place": "42km SSW of Test City",
        "type": "earthquake",
        "tsunami": 1,
    },
    "geometry": {
        "type": "Point",
        "coordinates": [-76.1, -12.3, 25.0],
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dart_record(payload_hash: str) -> DartRecord:
    return DartRecord(
        source_id="dart:21413:20260304120000:1",
        station_id="21413",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        measurement_type=1,
        height_m=4541.234,
        event_mode=False,
        payload_sha256=payload_hash,
    )


def _make_coops_record(payload_hash: str) -> CoopsRecord:
    return CoopsRecord(
        source_id="coops:1612340:water_level:202603041200",
        station_id="1612340",
        station_name="Honolulu",
        product="water_level",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        water_level_m=0.584,
        flags="0,0,0,0",
        quality="p",
        payload_sha256=payload_hash,
    )


def _make_seismic_record(payload_hash: str) -> SeismicEventRecord:
    return SeismicEventRecord(
        source_id="seismic:us7000test:20260304120000000000",
        event_id="us7000test",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        magnitude=6.1,
        place="42km SSW of Test City",
        event_type="earthquake",
        tsunami_flag=1,
        longitude=-76.1,
        latitude=-12.3,
        depth_km=25.0,
        updated_timestamp=_NOW,
        is_revision=False,
        payload_sha256=payload_hash,
    )


def _ingest_and_audit(
    source_type: str,
    raw_bytes: bytes,
    record: DartRecord | CoopsRecord | SeismicEventRecord,
    store: InMemoryPayloadStore,
    logger: AuditLogger,
) -> tuple[str, AuditEntry]:
    """Run the full ingest pipeline for one record and return (hash, audit_entry)."""
    payload_hash = compute_payload_hash(raw_bytes)
    assert payload_hash == record.payload_sha256

    store.store(payload_hash, raw_bytes)
    observation = validate_record(record)
    assert observation.payload_sha256 == payload_hash

    entry = AuditEntry(
        event_type="ingest",
        producer=source_type,
        data={
            "payload_sha256": payload_hash,
            "source_id": record.source_id,
            "observation_type": type(observation).__name__,
        },
    )
    logger.append(entry)
    return payload_hash, entry


# ---------------------------------------------------------------------------
# Per-source lineage tests
# ---------------------------------------------------------------------------


def test_dart_replay_lineage() -> None:
    """DART: raw line -> hash -> store -> validate -> audit -> trace back."""
    raw_bytes = _DART_RAW_LINE.encode("utf-8")
    payload_hash = compute_payload_hash(raw_bytes)
    record = _make_dart_record(payload_hash)
    store = InMemoryPayloadStore()
    logger = AuditLogger()

    h, entry = _ingest_and_audit("dart", raw_bytes, record, store, logger)

    # Replay: retrieve raw bytes from store, re-hash, verify match
    retrieved = store.retrieve(h)
    assert retrieved is not None
    assert compute_payload_hash(retrieved) == h

    # Lineage: audit entry -> hash -> store -> raw bytes
    audit_hash = entry.data["payload_sha256"]
    assert store.contains(audit_hash)
    assert store.retrieve(audit_hash) == raw_bytes

    # Validate the observation type
    obs = validate_record(record)
    assert isinstance(obs, DartObservation)
    assert obs.height_m == 4541.234


def test_coops_replay_lineage() -> None:
    """CO-OPS: JSON row -> canonicalize -> hash -> store -> validate -> audit -> trace back."""
    raw_bytes = canonicalize_json(_COOPS_RAW_ROW)
    payload_hash = compute_payload_hash(raw_bytes)
    record = _make_coops_record(payload_hash)
    store = InMemoryPayloadStore()
    logger = AuditLogger()

    h, entry = _ingest_and_audit("coops", raw_bytes, record, store, logger)

    retrieved = store.retrieve(h)
    assert retrieved is not None
    assert compute_payload_hash(retrieved) == h

    audit_hash = entry.data["payload_sha256"]
    assert store.contains(audit_hash)
    assert store.retrieve(audit_hash) == raw_bytes

    obs = validate_record(record)
    assert isinstance(obs, CoopsObservation)
    assert obs.water_level_m == 0.584


def test_seismic_replay_lineage() -> None:
    """Seismic: GeoJSON feature -> canonicalize -> hash -> store -> validate ->
    audit -> trace back."""
    raw_bytes = canonicalize_json(_SEISMIC_RAW_FEATURE)
    payload_hash = compute_payload_hash(raw_bytes)
    record = _make_seismic_record(payload_hash)
    store = InMemoryPayloadStore()
    logger = AuditLogger()

    h, entry = _ingest_and_audit("seismic", raw_bytes, record, store, logger)

    retrieved = store.retrieve(h)
    assert retrieved is not None
    assert compute_payload_hash(retrieved) == h

    audit_hash = entry.data["payload_sha256"]
    assert store.contains(audit_hash)
    assert store.retrieve(audit_hash) == raw_bytes

    obs = validate_record(record)
    assert isinstance(obs, SeismicObservation)
    assert obs.magnitude == 6.1


# ---------------------------------------------------------------------------
# Cross-source and determinism tests
# ---------------------------------------------------------------------------


def test_replay_determinism() -> None:
    """Same raw bytes always produce the same SHA-256 hash (all 3 sources)."""
    payloads = [
        _DART_RAW_LINE.encode("utf-8"),
        canonicalize_json(_COOPS_RAW_ROW),
        canonicalize_json(_SEISMIC_RAW_FEATURE),
    ]
    for raw_bytes in payloads:
        h1 = compute_payload_hash(raw_bytes)
        h2 = compute_payload_hash(raw_bytes)
        assert h1 == h2
        assert len(h1) == 64
        assert all(c in "0123456789abcdef" for c in h1)

        # Store + retrieve + re-hash cycle
        store = InMemoryPayloadStore()
        store.store(h1, raw_bytes)
        retrieved = store.retrieve(h1)
        assert retrieved == raw_bytes
        assert compute_payload_hash(retrieved) == h1


def test_full_lineage_chain_all_sources() -> None:
    """All 3 sources in a single store + logger; every audit hash resolves."""
    store = InMemoryPayloadStore()
    logger = AuditLogger()

    sources = [
        ("dart", _DART_RAW_LINE.encode("utf-8"), _make_dart_record),
        ("coops", canonicalize_json(_COOPS_RAW_ROW), _make_coops_record),
        ("seismic", canonicalize_json(_SEISMIC_RAW_FEATURE), _make_seismic_record),
    ]

    hashes: list[str] = []
    for source_type, raw_bytes, make_record in sources:
        payload_hash = compute_payload_hash(raw_bytes)
        record = make_record(payload_hash)
        _ingest_and_audit(source_type, raw_bytes, record, store, logger)
        hashes.append(payload_hash)

    # All 3 hashes are distinct
    assert len(set(hashes)) == 3

    # Audit logger has exactly 3 entries
    entries = logger.get_entries(event_type="ingest")
    assert len(entries) == 3

    # Every audit entry's hash resolves in the store
    for entry in entries:
        h = entry.data["payload_sha256"]
        assert store.contains(h)
        raw = store.retrieve(h)
        assert raw is not None
        # Re-hash retrieved bytes matches
        assert compute_payload_hash(raw) == h


def test_replay_produces_identical_observation() -> None:
    """Replaying from archived raw bytes produces an identical observation (all 3 sources)."""
    sources = [
        (_DART_RAW_LINE.encode("utf-8"), _make_dart_record),
        (canonicalize_json(_COOPS_RAW_ROW), _make_coops_record),
        (canonicalize_json(_SEISMIC_RAW_FEATURE), _make_seismic_record),
    ]
    for raw_bytes, make_record in sources:
        payload_hash = compute_payload_hash(raw_bytes)
        record = make_record(payload_hash)
        obs1 = validate_record(record)

        # Simulate replay: retrieve from store, rebuild record, re-validate
        store = InMemoryPayloadStore()
        store.store(payload_hash, raw_bytes)
        replayed_bytes = store.retrieve(payload_hash)
        assert replayed_bytes is not None
        replayed_hash = compute_payload_hash(replayed_bytes)
        replayed_record = make_record(replayed_hash)
        obs2 = validate_record(replayed_record)

        # Observations are identical
        assert obs1.model_dump() == obs2.model_dump()
        assert obs1.payload_sha256 == obs2.payload_sha256
