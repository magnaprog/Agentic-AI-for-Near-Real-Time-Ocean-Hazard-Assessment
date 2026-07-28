"""Unit tests for the alert-language guardrail scanner.

Validates that prohibited NOAA alert terminology is detected and blocked,
and that the mandatory disclaimer is enforced.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from hazard_assessment.policy.guardrails import (
    NON_AUTHORITATIVE_DISCLAIMER,
    ScanResult,
    scan_text,
)


class TestProhibitedTermDetection:
    def test_detects_warning(self) -> None:
        result = scan_text("This is a tsunami Warning for the Pacific coast.")
        assert not result.passed
        assert len(result.violations) == 1
        assert result.violations[0].term == "Warning"

    def test_detects_advisory(self) -> None:
        result = scan_text("Tsunami Advisory has been issued.")
        assert not result.passed
        assert any(v.term == "Advisory" for v in result.violations)

    def test_detects_watch(self) -> None:
        result = scan_text("A tsunami Watch is in effect.")
        assert not result.passed
        assert any(v.term == "Watch" for v in result.violations)

    def test_detects_information_statement(self) -> None:
        result = scan_text("This is a tsunami Information Statement.")
        assert not result.passed
        assert any(v.term == "Information Statement" for v in result.violations)

    def test_every_prohibited_term_is_actually_blocked(self) -> None:
        """Each declared term must be caught, not merely listed.

        The individual cases above cover a sample. This walks the whole list so
        a term cannot be added to PROHIBITED_TERMS while its pattern silently
        fails to match, which the sampled tests would not notice.
        """
        from hazard_assessment.policy.guardrails import PROHIBITED_TERMS

        for term in PROHIBITED_TERMS:
            result = scan_text(f"This text mentions a {term} for review.")
            assert not result.passed, f"{term!r} was not blocked"
            assert any(
                v.term == term for v in result.violations
            ), f"{term!r} did not produce a violation naming it"

    def test_case_insensitive(self) -> None:
        result = scan_text("TSUNAMI WARNING issued.")
        assert not result.passed
        assert len(result.violations) == 1

    def test_detects_multiple_violations(self) -> None:
        result = scan_text("Warning and Advisory both apply. Watch for updates.")
        assert not result.passed
        assert len(result.violations) == 3


class TestDisclaimerEnforcement:
    def test_missing_disclaimer_fails(self) -> None:
        result = scan_text("Assessment: moderate tsunami risk at coastal sites.")
        assert not result.passed
        assert not result.has_disclaimer

    def test_with_disclaimer_passes(self) -> None:
        text = f"Assessment: moderate risk. {NON_AUTHORITATIVE_DISCLAIMER}"
        result = scan_text(text)
        assert result.passed
        assert result.has_disclaimer

    def test_clean_text_with_disclaimer(self) -> None:
        text = (
            "Elevated anomaly score detected at DART_21413. "
            "Scenario assessment indicates M8.1 equivalent source. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert result.passed
        assert len(result.violations) == 0

    def test_violation_overrides_disclaimer(self) -> None:
        text = f"Tsunami Warning issued. {NON_AUTHORITATIVE_DISCLAIMER}"
        result = scan_text(text)
        assert not result.passed
        assert result.has_disclaimer  # Disclaimer present but term violation
        assert len(result.violations) == 1


class TestAllowlistedProperNouns:
    """Proper-noun references to NOAA organizations should not trigger violations."""

    def test_tsunami_warning_center_allowed(self) -> None:
        text = (
            "Data from the Pacific Tsunami Warning Center. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert result.passed
        assert len(result.violations) == 0

    def test_national_tsunami_warning_center_allowed(self) -> None:
        text = (
            "As reported by the National Tsunami Warning Center. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert result.passed
        assert len(result.violations) == 0

    def test_standalone_warning_still_blocked(self) -> None:
        """'Warning' outside of a proper noun context is still prohibited."""
        text = (
            "Tsunami Warning issued by Pacific Tsunami Warning Center. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert not result.passed
        # The standalone "Warning" after "Tsunami" should be caught,
        # but the one in "Tsunami Warning Center" should not
        assert len(result.violations) == 1

    def test_mixed_allowed_and_prohibited(self) -> None:
        """Allowlisted proper noun + prohibited standalone term."""
        text = (
            "Pacific Tsunami Warning Center issued a Watch. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert not result.passed
        # "Warning" in "Tsunami Warning Center" is allowed,
        # but standalone "Watch" is prohibited
        assert any(v.term == "Watch" for v in result.violations)
        assert not any(v.term == "Warning" for v in result.violations)


class TestUnicodeHomoglyphBypass:
    """Verify that Unicode homoglyphs cannot bypass the guardrail scanner."""

    def test_cyrillic_a_in_warning_detected(self) -> None:
        """Cyrillic 'а' (U+0430) replacing Latin 'a' in 'Warning'."""
        # W\u0430rning - visually identical to "Warning"
        text = "Tsunami W\u0430rning issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_fullwidth_characters_detected(self) -> None:
        """Fullwidth Latin letters (U+FF21-FF5A) replacing ASCII."""
        # \uff37\uff41\uff52\uff4e\uff49\uff4e\uff47 = fullwidth "Warning"
        text = "Tsunami \uff37\uff41\uff52\uff4e\uff49\uff4e\uff47 issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_fullwidth_advisory_detected(self) -> None:
        """Fullwidth 'Advisory' is caught after NFKC normalization."""
        text = "Tsunami \uff21\uff44\uff56\uff49\uff53\uff4f\uff52\uff59 issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Advisory" for v in result.violations)

    def test_dotless_i_in_warning_detected(self) -> None:
        """Latin dotless ı (U+0131) replacing 'i' in 'Warning'."""
        # Warn\u0131ng - visually similar to "Warning"
        text = "Tsunami Warn\u0131ng issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_normal_text_unaffected_by_normalization(self) -> None:
        """Standard ASCII text should work identically after normalization."""
        text = (
            "Elevated anomaly score detected. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert result.passed
        assert len(result.violations) == 0


class TestZeroWidthCharacterBypass:
    """Verify that zero-width characters cannot split terms to evade detection."""

    def test_zero_width_space_in_warning_detected(self) -> None:
        """Zero-width space (U+200B) inserted into 'Warning'."""
        text = "Tsunami Warn\u200Bing issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_zero_width_joiner_in_advisory_detected(self) -> None:
        """Zero-width joiner (U+200D) inserted into 'Advisory'."""
        text = "Tsunami Advi\u200Dsory issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Advisory" for v in result.violations)

    def test_zero_width_non_joiner_in_watch_detected(self) -> None:
        """Zero-width non-joiner (U+200C) inserted into 'Watch'."""
        text = "Tsunami Wa\u200Ctch issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Watch" for v in result.violations)

    def test_bom_in_warning_detected(self) -> None:
        """BOM (U+FEFF) inserted into 'Warning'."""
        text = "Tsunami W\ufeffarning issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_soft_hyphen_in_warning_detected(self) -> None:
        """Soft hyphen (U+00AD) inserted into 'Warning'."""
        text = "Tsunami W\u00adarning issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_function_application_in_advisory_detected(self) -> None:
        """Invisible math operator (U+2061) inserted into 'Advisory'."""
        text = "Tsunami Adv\u2061isory issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Advisory" for v in result.violations)

    def test_variation_selector_in_watch_detected(self) -> None:
        """Variation selector (U+FE0F) inserted into 'Watch'."""
        text = "Tsunami W\ufe0fatch issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Watch" for v in result.violations)

    def test_mongolian_vowel_separator_detected(self) -> None:
        """Mongolian vowel separator (U+180E) inserted into 'Warning'."""
        text = "Tsunami Warn\u180eing issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)


class TestMultiWordTermSplitting:
    """Verify multi-word reserved terms cannot be split by inter-word
    whitespace variants (or a zero-width character collapsed to none)."""

    def test_zero_width_between_all_clear_detected(self) -> None:
        """Zero-width space between the words of 'All Clear'."""
        text = "Issue an All\u200BClear now."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "All Clear" for v in result.violations)

    def test_tab_between_all_clear_detected(self) -> None:
        """A tab in place of the space in 'All Clear'."""
        text = "Issue an All\tClear now."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "All Clear" for v in result.violations)

    def test_double_space_in_information_statement_detected(self) -> None:
        """Two spaces between the words of 'Information Statement'."""
        text = "This Information  Statement stands."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Information Statement" for v in result.violations)

    def test_zero_width_in_threat_message_detected(self) -> None:
        """Zero-width joiner between the words of 'Threat Message'."""
        text = "A Threat\u200DMessage was posted."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Threat Message" for v in result.violations)

    def test_double_spaced_center_name_not_flagged(self) -> None:
        """An oddly spaced allowlisted organization name stays allowed."""
        text = (
            "Per the Pacific Tsunami  Warning  Center. "
            + NON_AUTHORITATIVE_DISCLAIMER
        )
        result = scan_text(text)
        assert result.passed
        assert not result.violations


class TestAdditionalHomoglyphs:
    """Verify newly added confusable map entries detect bypass attempts."""

    def test_greek_nu_as_v_in_advisory_detected(self) -> None:
        """Greek ν (U+03BD) replacing 'v' in 'Advisory'."""
        text = "Tsunami Ad\u03bdisory issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Advisory" for v in result.violations)

    def test_cyrillic_en_as_n_in_warning_detected(self) -> None:
        """Cyrillic н (U+043D) replacing 'n' in 'Warning'."""
        text = "Tsunami War\u043di\u043dg issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_cyrillic_shha_as_h_in_watch_detected(self) -> None:
        """Cyrillic һ (U+04BB) replacing 'h' in 'Watch'."""
        text = "Tsunami Watc\u04bb issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Watch" for v in result.violations)

    def test_greek_iota_as_i_in_warning_detected(self) -> None:
        """Greek ι (U+03B9) replacing 'i' in 'Warning'."""
        text = "Tsunami Warn\u03b9ng issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_cyrillic_em_as_m_in_information_detected(self) -> None:
        """Cyrillic м (U+043C) replacing 'm' in 'Information Statement'."""
        text = "Tsunami Infor\u043cation Statement issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Information Statement" for v in result.violations)


class TestSmallCapitalAndCombiningBypass:
    """Small-capital Latin letters and combining marks must not evade the scanner.

    Neither vector is folded by NFKC or the Cyrillic/Greek confusable map, so
    the scanner strips combining marks and folds "LATIN LETTER SMALL CAPITAL X"
    characters to ASCII before matching.
    """

    def test_small_capital_warning_detected(self) -> None:
        """Small-capital Latin "\u1d21\u1d00\u0280\u0274\u026a\u0274\u0262" reads as WARNING."""
        text = "Tsunami \u1d21\u1d00\u0280\u0274\u026a\u0274\u0262 issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_small_capital_watch_detected(self) -> None:
        """Small-capital Latin "\u1d21\u1d00\u1d1b\u1d04\u029c" reads as WATCH."""
        text = "Tsunami \u1d21\u1d00\u1d1b\u1d04\u029c issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Watch" for v in result.violations)

    def test_combining_diaeresis_in_warning_detected(self) -> None:
        """A combining diaeresis inside 'Warning' (Wa\u0308rning) must not evade."""
        text = "Tsunami Wa\u0308rning issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_precomposed_accent_in_warning_detected(self) -> None:
        """Precomposed 'Wärning' (U+00E4) must decompose and match."""
        text = "Tsunami W\u00e4rning issued."
        result = scan_text(text)
        assert not result.passed
        assert any(v.term == "Warning" for v in result.violations)

    def test_accented_non_term_text_not_flagged(self) -> None:
        """Stripping accents must not create a false positive on ordinary text."""
        prose = "Post-event note for T\u00f4hoku. r\u00e9sum\u00e9 caf\u00e9."
        text = f"{prose} {NON_AUTHORITATIVE_DISCLAIMER}"
        result = scan_text(text)
        assert result.passed
        assert result.violations == []


class TestHyphenatedMultiWordTerms:
    """Compound hyphens join reserved words just as whitespace does.

    The scanner already treated "AllClear" (words run together after the
    zero-width strip) as a violation, so being lenient about the ordinary
    English hyphenation was inconsistent: "All-Clear" is the same reserved
    product name, and it is how a reviewer typing a decision reason or a
    narrative model writing prose is most likely to render it.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "An All-Clear was posted",
            "An All‑Clear was posted",
            "See the Information-Statement",
            "See the Threat-Message",
        ],
    )
    def test_hyphenated_reserved_term_detected(self, text: str) -> None:
        result = scan_text(f"{text} {NON_AUTHORITATIVE_DISCLAIMER}")
        assert not result.passed
        assert result.violations

    @pytest.mark.parametrize(
        "text",
        [
            "that is all — clear skies ahead",
            "that is all – clear skies ahead",
            "that is all - clear skies ahead",
            "Reviewed all - clear signal on 21418",
        ],
    )
    def test_clause_separating_dash_not_flagged(self, text: str) -> None:
        """A spaced dash separates clauses, so it must not join words.

        The hyphen has to be unspaced to build a compound. This repository
        writes plain ASCII, so " - " is its ordinary clause separator and a
        reviewer will use it in a decision reason; treating that as
        "All Clear" would reject legitimate prose.
        """
        result = scan_text(f"{text} {NON_AUTHORITATIVE_DISCLAIMER}")
        assert result.passed
        assert result.violations == []

    def test_hyphenated_center_name_still_allowlisted(self) -> None:
        """Widening the joiner widens the allowlist patterns identically."""
        text = (
            "Reported by the Pacific-Tsunami-Warning-Center. "
            f"{NON_AUTHORITATIVE_DISCLAIMER}"
        )
        result = scan_text(text)
        assert result.passed
        assert result.violations == []


class TestScanResultTimezoneValidation:
    """Verify ScanResult rejects naive datetimes for scanned_at_utc."""

    def test_naive_scanned_at_utc_rejected(self) -> None:
        naive = datetime(2026, 2, 27, 1, 30, 0)  # No tzinfo
        with pytest.raises(ValueError, match="timezone-aware"):
            ScanResult(
                text_scanned="test",
                passed=True,
                scanned_at_utc=naive,
            )

    def test_aware_scanned_at_utc_accepted(self) -> None:
        aware = datetime(2026, 2, 27, 1, 30, 0, tzinfo=UTC)
        result = ScanResult(text_scanned="test", passed=True, scanned_at_utc=aware)
        assert result.scanned_at_utc == aware

    def test_default_scanned_at_utc_is_aware(self) -> None:
        """The default factory should produce a timezone-aware datetime."""
        result = scan_text(f"Safe text. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert result.scanned_at_utc.tzinfo is not None


class TestUnicodeConfusableCoverage:
    """Reserved terms spelled with confusable characters must still be caught.

    The hand-written map covered Cyrillic and Greek. Sweeping every
    single-character entry of the Unicode confusables data (UTS #39) against
    the eight reserved terms found 406 further characters that spelled a term
    the scanner did not see, across Cherokee, Lisu, Arabic, Coptic, Miao,
    Carian and Warang Citi among others. These cases are one representative
    per family; the generated table in policy/_confusables.py carries the data.
    """

    @pytest.mark.parametrize(
        ("spelling", "term"),
        [
            ("\u13b3arning", "Warning"),          # CHEROKEE LETTER LA
            ("\u13aadvisory", "Advisory"),        # CHEROKEE LETTER GO
            ("\ua4eaarning", "Warning"),          # LISU LETTER ZHA
            ("Advis\u0647ry", "Advisory"),        # ARABIC LETTER HEH
            ("War\u2c9aing", "Warning"),          # COPTIC CAPITAL LETTER NI
            ("W\U00016f40rning", "Warning"),      # MIAO LETTER ZA
            ("W\U000102a0rning", "Warning"),      # CARIAN LETTER A
            ("Ad\U000118a0isory", "Advisory"),    # WARANG CITI CAPITAL NGAA
        ],
    )
    def test_confusable_spellings_are_caught(self, spelling: str, term: str) -> None:
        result = scan_text(f"{spelling} in effect. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert [v.term for v in result.violations] == [term]

    @pytest.mark.parametrize(
        ("spelling", "term"),
        [
            ("In\u017formation Statement", "Information Statement"),  # LATIN SMALL LETTER LONG S
            ("Warn\u02dbng", "Warning"),                              # OGONEK
            ("Advis\U0001d7cery", "Advisory"),                        # MATHEMATICAL BOLD DIGIT ZERO
        ],
    )
    def test_characters_normalization_would_destroy_are_caught(
        self, spelling: str, term: str
    ) -> None:
        """NFKC rewrites these before the fold can see them.

        A long s becomes an s rather than the f it resembles, a spacing ogonek
        becomes a space plus a combining mark that is then stripped, and a
        mathematical digit zero becomes an ASCII 0. Folding confusables before
        normalizing as well as after is what catches them.
        """
        result = scan_text(f"{spelling} in effect. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert [v.term for v in result.violations] == [term]

    @pytest.mark.parametrize(
        "spelling",
        [
            "AlI Clear",          # ASCII capital I for lowercase l
            "CanceIlation",
            "Cance\u0399lation",  # GREEK CAPITAL LETTER IOTA
            "Cance\u0406lation",  # CYRILLIC CAPITAL LETTER BYELORUSSIAN-UKRAINIAN I
        ],
    )
    def test_shape_collapsed_spellings_are_caught(self, spelling: str) -> None:
        """Capital I, digit one and the bar all read as lowercase l.

        The confusable fold cannot resolve these: a capital I really is a
        capital I. A second matching pass collapses the shapes in both the text
        and the terms.
        """
        result = scan_text(f"{spelling} in effect. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert result.violations, f"{spelling!r} evaded the scanner"

    def test_shape_collapse_does_not_break_the_allowlist(self) -> None:
        """The collapsed pass must recognise organisation names too."""
        result = scan_text(
            f"Pacific Tsunami Warning Center issued guidance. {NON_AUTHORITATIVE_DISCLAIMER}"
        )
        assert result.passed
        assert result.violations == []

    def test_ordinary_project_text_still_passes(self) -> None:
        """Digits and identifiers are everywhere in this system's output."""
        result = scan_text(
            "Illapel 2015 replay at T1=0.35, station 21418, 100 samples, "
            f"scores 0.996 and 1.000. {NON_AUTHORITATIVE_DISCLAIMER}"
        )
        assert result.passed

    def test_violation_context_is_not_shape_collapsed(self) -> None:
        """A reader must see what was scanned, not the matching form."""
        result = scan_text(f"CanceIlation notice. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert result.violations
        assert "CanceIlation" in result.violations[0].context

    def test_each_violation_is_reported_once(self) -> None:
        """Two passes over the same text must not double-count a match."""
        result = scan_text(f"Advisory in effect. {NON_AUTHORITATIVE_DISCLAIMER}")
        assert len(result.violations) == 1


class TestInflectedAndStylisedReservedTerms:
    """Spellings a real alert-styled output uses, which a bare word boundary
    or a single Unicode normalization ordering lets through."""

    @pytest.mark.parametrize(
        "text",
        [
            "Tsunami Warnings remain in effect for Hawaii.",
            "All coastal Watches have been lifted.",
            "Advisories in effect.",
            "Bulletins posted to the board.",
            "Cancellations issued for the outer islands.",
            "Threat Messages sent to partners.",
            "Information Statements issued.",
        ],
    )
    def test_regular_plurals_are_flagged(self, text: str) -> None:
        """A trailing s is a word character, so \\b alone does not stop it.

        The plural is the form a narrative most naturally produces, and it
        carries exactly the reserved meaning the singular does.
        """
        assert scan_text(text).violations

    @pytest.mark.parametrize(
        "text",
        [
            "ＷＡＲＮＩＮＧ issued.",
            "ＡＤＶＩＳＯＲＹ in effect.",
            "ＢＵＬＬＥＴＩＮ posted.",
            "ＣＡＮＣＥＬＬＡＴＩＯＮ issued.",
        ],
    )
    def test_fullwidth_uppercase_is_flagged(self, text: str) -> None:
        """Fullwidth capital I shape-folds to l, which destroyed the word.

        The lowercase fullwidth form was already caught; the all-caps
        rendering, which is what an alert-styled output would use, was not.
        """
        assert scan_text(text).violations

    @pytest.mark.parametrize(
        "text",
        [
            "Adviſory issued.",
            "Threat Meſſage sent.",
            "Information ſtatement issued.",
        ],
    )
    def test_long_s_standing_in_for_s_is_flagged(self, text: str) -> None:
        """LATIN SMALL LETTER LONG S shape-folds to f, so it survived as an s.

        The reverse case, long s standing in for f, is covered separately and
        must keep working: both orderings are scanned.
        """
        assert scan_text(text).violations

    @pytest.mark.parametrize(
        "text",
        [
            "An All—Clear was posted.",
            "An All–Clear was posted.",
            "All_Clear posted.",
            "Information/Statement issued.",
            "Threat_Message issued.",
            "All.Clear posted.",
        ],
    )
    def test_unspaced_joiners_are_flagged(self, text: str) -> None:
        """Any unspaced joiner builds the same compound product name."""
        assert scan_text(text).violations

    def test_spaced_dash_remains_a_clause_separator(self) -> None:
        """The spaced-dash exemption must survive the wider joiner set."""
        assert not scan_text("that is all - clear skies ahead").violations

    def test_allowlisted_proper_nouns_survive_both_normalizations(self) -> None:
        """Scanning a second normalization must not defeat the allowlist."""
        assert not scan_text("Pacific Tsunami Warning Center reported.").violations
        assert not scan_text("Posted to the Tsunami Bulletin Board.").violations


class TestAllowlistSurvivesBothNormalizations:
    """A proper noun allowlisted in one normalization must stay allowlisted.

    The allowlist is evaluated per haystack, because the two normalizations
    differ in length and an offset from one does not index the other. That
    makes the alternate pass additive only, so an organisation name that the
    primary normalization preserves but the alternate one mangles had its
    allowlist entry missed and the reserved word inside it was reported.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Issued by the Tsunam˛ Warning Center.",       # OGONEK for i
            "Issued by the Tsunamͺ Warning Center.",       # YPOGEGRAMMENI
            "Issued by the Tsunami Warning ϲenter.",       # lunate sigma
            "Issued by the Tsunami Warning Ϲenter.",       # capital lunate
            "Issued by the Tsunam˛ Bulletin Board.",
        ],
    )
    def test_mangled_allowlisted_name_is_not_a_violation(self, text: str) -> None:
        assert not scan_text(text).violations

    def test_a_real_violation_beside_an_allowlisted_name_still_reports(self) -> None:
        """Carrying the suppression must not silence a genuine occurrence."""
        assert scan_text("Pacific Tsunami Warning Center issued a Warning.").violations


class TestJoinerRunsAndPunctuationHomoglyphs:
    @pytest.mark.parametrize(
        "text",
        [
            "All__Clear posted.",
            "All--Clear posted.",
            "All-.Clear posted.",
            "Status All∕Clear now.",   # DIVISION SLASH, renders as /
            "Status All·Clear now.",   # MIDDLE DOT
            "Status All−Clear now.",   # MINUS SIGN, renders as -
        ],
    )
    def test_joiner_runs_and_lookalike_punctuation_are_flagged(self, text: str) -> None:
        """A run of joiners, and punctuation that renders like one, build the
        same compound product name as a single ASCII hyphen."""
        assert scan_text(text).violations

    def test_spaced_dash_is_still_a_clause_separator(self) -> None:
        assert not scan_text("that is all - clear skies ahead").violations


class TestInvisibleNonFormatCharacters:
    @pytest.mark.parametrize(
        "codepoint",
        [0x180B, 0x180D, 0x115F, 0x3164, 0x2800, 0xFFFC, 0x17B4],
    )
    def test_invisible_non_cf_characters_do_not_split_a_term(
        self, codepoint: int
    ) -> None:
        """These are invisible to a reader but are neither category Cf nor
        combining, so neither the format-character strip nor the mark strip
        removed them and they split a reserved term for the matcher."""
        assert scan_text(f"Tsunami Warn{chr(codepoint)}ing issued.").violations
