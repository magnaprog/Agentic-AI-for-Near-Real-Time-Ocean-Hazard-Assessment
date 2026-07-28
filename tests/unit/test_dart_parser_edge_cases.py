"""Edge-case tests for the DART payload parser.

Tests cover empty payloads, all-comment payloads, malformed rows,
all-sentinel records (9999), and mixed valid/invalid data.
"""

from __future__ import annotations

from hazard_assessment.ingest.dart import parse_dart_payload


class TestDartParserEmpty:
    """Parser should handle empty and header-only payloads gracefully."""

    def test_empty_string(self) -> None:
        rows = parse_dart_payload("", station_id="99999")
        assert rows == []

    def test_only_whitespace(self) -> None:
        rows = parse_dart_payload("   \n  \n  ", station_id="99999")
        assert rows == []

    def test_only_comments(self) -> None:
        payload = "# Station 99999\n# YY MM DD hh mm ss T HEIGHT\n"
        rows = parse_dart_payload(payload, station_id="99999")
        assert rows == []

    def test_header_plus_blank_lines(self) -> None:
        payload = "# header line 1\n# header line 2\n\n\n"
        rows = parse_dart_payload(payload, station_id="99999")
        assert rows == []


class TestDartParserMalformed:
    """Parser should skip malformed rows without crashing."""

    def test_short_row_skipped(self) -> None:
        payload = "2024 03 11 05 46 24 1\n"  # only 7 fields, need 8
        rows = parse_dart_payload(payload, station_id="99999")
        assert rows == []

    def test_non_numeric_height(self) -> None:
        payload = "2024 03 11 05 46 24 1 NaN\n"
        rows = parse_dart_payload(payload, station_id="99999")
        assert rows == []

    def test_mixed_valid_and_invalid(self) -> None:
        payload = (
            "# header\n"
            "2024 03 11 05 46 24 1 5000.123\n"  # valid
            "2024 03 11 05 47 24 1\n"  # too short
            "2024 03 11 05 48 24 1 5000.456\n"  # valid
            "bad line\n"  # garbage
        )
        rows = parse_dart_payload(payload, station_id="99999")
        assert len(rows) == 2
        assert rows[0].height_m == 5000.123
        assert rows[1].height_m == 5000.456

    def test_invalid_measurement_type(self) -> None:
        """Measurement type must be 1, 2, or 3; type 0 or 9 should be skipped."""
        payload = (
            "2024 03 11 05 46 24 0 5000.1\n"  # type 0 - skip
            "2024 03 11 05 46 24 1 5000.2\n"  # type 1 - keep
            "2024 03 11 05 46 24 9 5000.3\n"  # type 9 - skip
        )
        rows = parse_dart_payload(payload, station_id="99999")
        assert len(rows) == 1
        assert rows[0].height_m == 5000.2


class TestDartParserSentinels:
    """Parser should handle 9999 sentinel values."""

    def test_all_sentinel_rows_filtered(self) -> None:
        """All-9999 payload: parser filters sentinels, returns empty list."""
        payload = (
            "2024 03 11 05 46 24 1 9999.000\n"
            "2024 03 11 05 47 24 1 9999.000\n"
        )
        rows = parse_dart_payload(payload, station_id="99999")
        assert rows == []

    def test_mixed_sentinel_and_valid(self) -> None:
        """Valid rows retained, sentinel rows filtered."""
        payload = (
            "2024 03 11 05 46 24 1 5000.100\n"
            "2024 03 11 05 47 24 1 9999.000\n"
            "2024 03 11 05 48 24 1 5000.200\n"
        )
        rows = parse_dart_payload(payload, station_id="99999")
        assert len(rows) == 2
        assert rows[0].height_m == 5000.1
        assert rows[1].height_m == 5000.2


class TestDartParserSorting:
    """Parsed rows should be sorted by timestamp."""

    def test_rows_sorted_by_time(self) -> None:
        payload = (
            "2024 03 11 05 48 24 1 5000.3\n"
            "2024 03 11 05 46 24 1 5000.1\n"
            "2024 03 11 05 47 24 1 5000.2\n"
        )
        rows = parse_dart_payload(payload, station_id="99999")
        assert len(rows) == 3
        # Should be sorted chronologically
        heights = [r.height_m for r in rows]
        assert heights == [5000.1, 5000.2, 5000.3]
