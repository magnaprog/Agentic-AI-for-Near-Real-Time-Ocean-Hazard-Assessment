"""Tests for canonical observation validation and quarantine."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from hazard_assessment.ingest.coops import CoopsRecord
from hazard_assessment.ingest.dart import DartRecord
from hazard_assessment.ingest.seismic import SeismicEventRecord
from hazard_assessment.ingest.validation import (
    QuarantinedRecord,
    QuarantineReasonCode,
    validate_and_quarantine,
    validate_record,
)
from hazard_assessment.schemas.observation import (
    CoopsObservation,
    DartObservation,
    SeismicObservation,
)

_NOW = datetime(2026, 3, 4, 8, 0, tzinfo=UTC)
_HASH = "a" * 64  # valid 64-char hex


# --- DartObservation ---


def test_dart_observation_valid() -> None:
    obs = DartObservation.model_validate({
        "source_id": "dart:21413:20260304080000:1",
        "source_timestamp": _NOW,
        "ingest_timestamp": _NOW,
        "payload_sha256": _HASH,
        "station_id": "21413",
        "measurement_type": 1,
        "height_m": 4541.234,
        "event_mode": False,
    })
    assert obs.station_id == "21413"
    assert obs.height_m == pytest.approx(4541.234)


def test_dart_observation_rejects_invalid_station_id() -> None:
    with pytest.raises(ValidationError):
        DartObservation.model_validate({
            "source_id": "dart:21413:20260304080000:1",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "station_id": "Warning",
            "measurement_type": 1,
            "height_m": 4541.234,
            "event_mode": False,
        })


def test_dart_observation_rejects_unicode_digit_station_id() -> None:
    with pytest.raises(ValidationError):
        DartObservation.model_validate({
            "source_id": "dart:21413:20260304080000:1",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "station_id": "٢١٤١٣",
            "measurement_type": 1,
            "height_m": 4541.234,
            "event_mode": False,
        })


def test_dart_observation_rejects_invalid_measurement_type() -> None:
    with pytest.raises(ValidationError):
        DartObservation.model_validate({
            "source_id": "dart:21413:20260304080000:9",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "station_id": "21413",
            "measurement_type": 9,
            "height_m": 4541.234,
            "event_mode": False,
        })


def test_dart_observation_rejects_missing_hash() -> None:
    with pytest.raises(ValidationError):
        DartObservation.model_validate({
            "source_id": "dart:21413:20260304080000:1",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": "",
            "station_id": "21413",
            "measurement_type": 1,
            "height_m": 4541.234,
            "event_mode": False,
        })


def test_dart_observation_rejects_extra_fields() -> None:
    with pytest.raises(ValidationError):
        DartObservation.model_validate({
            "source_id": "dart:21413:20260304080000:1",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "station_id": "21413",
            "measurement_type": 1,
            "height_m": 4541.234,
            "event_mode": False,
            "unexpected_field": "bad",
        })


# --- CoopsObservation ---


def test_coops_observation_valid() -> None:
    obs = CoopsObservation.model_validate({
        "source_id": "coops:1612340:water_level:202603040801",
        "source_timestamp": _NOW,
        "ingest_timestamp": _NOW,
        "payload_sha256": _HASH,
        "station_id": "1612340",
        "station_name": "Honolulu",
        "product": "water_level",
        "water_level_m": 0.584,
        "flags": "0,0,0,0",
        "quality": "p",
    })
    assert obs.water_level_m == pytest.approx(0.584)


def test_coops_observation_allows_null_water_level() -> None:
    obs = CoopsObservation.model_validate({
        "source_id": "coops:1612340:water_level:202603040801",
        "source_timestamp": _NOW,
        "ingest_timestamp": _NOW,
        "payload_sha256": _HASH,
        "station_id": "1612340",
        "station_name": None,
        "product": "water_level",
        "water_level_m": None,
        "flags": "",
        "quality": "p",
    })
    assert obs.water_level_m is None


def test_coops_observation_rejects_out_of_range_water_level() -> None:
    with pytest.raises(ValidationError):
        CoopsObservation.model_validate({
            "source_id": "coops:1612340:water_level:202603040801",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "station_id": "1612340",
            "station_name": "Honolulu",
            "product": "water_level",
            "water_level_m": 999.0,
            "flags": "",
            "quality": "p",
        })


# --- SeismicObservation ---


def test_seismic_observation_valid() -> None:
    obs = SeismicObservation.model_validate({
        "source_id": "seismic:us7000abc1:20260304080000000000",
        "source_timestamp": _NOW,
        "ingest_timestamp": _NOW,
        "payload_sha256": _HASH,
        "event_id": "us7000abc1",
        "magnitude": 6.1,
        "place": "near coast",
        "event_type": "earthquake",
        "tsunami_flag": 1,
        "longitude": -76.1,
        "latitude": -12.3,
        "depth_km": 25.0,
        "updated_timestamp": _NOW,
        "is_revision": False,
    })
    assert obs.magnitude == pytest.approx(6.1)


def test_seismic_observation_allows_null_fields() -> None:
    obs = SeismicObservation.model_validate({
        "source_id": "seismic:us7000abc1:20260304080000000000",
        "source_timestamp": _NOW,
        "ingest_timestamp": _NOW,
        "payload_sha256": _HASH,
        "event_id": "us7000abc1",
        "magnitude": None,
        "place": "",
        "event_type": "earthquake",
        "tsunami_flag": None,
        "longitude": None,
        "latitude": None,
        "depth_km": None,
        "updated_timestamp": None,
        "is_revision": False,
    })
    assert obs.magnitude is None
    assert obs.longitude is None


def test_seismic_observation_rejects_out_of_range_magnitude() -> None:
    with pytest.raises(ValidationError):
        SeismicObservation.model_validate({
            "source_id": "seismic:us7000abc1:20260304080000000000",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "event_id": "us7000abc1",
            "magnitude": 15.0,
            "place": "coast",
            "event_type": "earthquake",
            "tsunami_flag": 0,
            "longitude": 0.0,
            "latitude": 0.0,
            "depth_km": 10.0,
            "updated_timestamp": None,
            "is_revision": False,
        })


def test_seismic_observation_rejects_out_of_range_coordinates() -> None:
    with pytest.raises(ValidationError):
        SeismicObservation.model_validate({
            "source_id": "seismic:us7000abc1:20260304080000000000",
            "source_timestamp": _NOW,
            "ingest_timestamp": _NOW,
            "payload_sha256": _HASH,
            "event_id": "us7000abc1",
            "magnitude": 6.0,
            "place": "coast",
            "event_type": "earthquake",
            "tsunami_flag": 0,
            "longitude": 999.0,
            "latitude": 0.0,
            "depth_km": 10.0,
            "updated_timestamp": None,
            "is_revision": False,
        })


# --- validate_record dispatch ---


def _make_dart_record() -> DartRecord:
    return DartRecord(
        source_id="dart:21413:20260304080000:1",
        station_id="21413",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        measurement_type=1,
        height_m=4541.234,
        event_mode=False,
        payload_sha256=_HASH,
    )


def _make_coops_record() -> CoopsRecord:
    return CoopsRecord(
        source_id="coops:1612340:water_level:202603040801",
        station_id="1612340",
        station_name="Honolulu",
        product="water_level",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        water_level_m=0.584,
        flags="0,0,0,0",
        quality="p",
        payload_sha256=_HASH,
    )


def _make_seismic_record() -> SeismicEventRecord:
    return SeismicEventRecord(
        source_id="seismic:us7000abc1:20260304080000000000",
        event_id="us7000abc1",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        magnitude=6.1,
        place="near coast",
        event_type="earthquake",
        tsunami_flag=1,
        longitude=-76.1,
        latitude=-12.3,
        depth_km=25.0,
        updated_timestamp=_NOW,
        is_revision=False,
        payload_sha256=_HASH,
    )


def test_validate_record_dispatches_dart() -> None:
    result = validate_record(_make_dart_record())
    assert isinstance(result, DartObservation)
    assert result.station_id == "21413"


def test_validate_record_dispatches_coops() -> None:
    result = validate_record(_make_coops_record())
    assert isinstance(result, CoopsObservation)
    assert result.station_id == "1612340"


def test_validate_record_dispatches_seismic() -> None:
    result = validate_record(_make_seismic_record())
    assert isinstance(result, SeismicObservation)
    assert result.event_id == "us7000abc1"


def test_validate_record_raises_for_invalid_dart() -> None:
    record = DartRecord(
        source_id="dart:21413:20260304080000:1",
        station_id="21413",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        measurement_type=1,
        height_m=4541.234,
        event_mode=False,
        payload_sha256="",  # invalid: empty hash
    )
    with pytest.raises(ValidationError):
        validate_record(record)


# --- validate_and_quarantine ---


def test_validate_and_quarantine_returns_observation_on_success() -> None:
    result = validate_and_quarantine(_make_dart_record(), now=_NOW)
    assert isinstance(result, DartObservation)


def test_validate_and_quarantine_returns_quarantine_on_failure() -> None:
    record = DartRecord(
        source_id="dart:21413:20260304080000:1",
        station_id="21413",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        measurement_type=1,
        height_m=4541.234,
        event_mode=False,
        payload_sha256="not-a-valid-hash",
    )
    result = validate_and_quarantine(record, now=_NOW)
    assert isinstance(result, QuarantinedRecord)
    assert result.reason_code == QuarantineReasonCode.SCHEMA_VALIDATION_FAILED
    assert result.source_id == "dart:21413:20260304080000:1"
    assert result.source_type == "DartRecord"
    assert result.quarantined_at == _NOW
    assert "payload_sha256" in result.reason_detail


def test_validate_and_quarantine_preserves_raw_fields() -> None:
    record = DartRecord(
        source_id="dart:21413:20260304080000:1",
        station_id="21413",
        source_timestamp=_NOW,
        ingest_timestamp=_NOW,
        measurement_type=1,
        height_m=4541.234,
        event_mode=False,
        payload_sha256="short",
    )
    result = validate_and_quarantine(record, now=_NOW)
    assert isinstance(result, QuarantinedRecord)
    assert result.raw_fields["station_id"] == "21413"
    assert result.raw_fields["height_m"] == pytest.approx(4541.234)
