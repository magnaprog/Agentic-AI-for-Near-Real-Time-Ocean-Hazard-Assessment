"""Alert-language guardrail scanner for output text.

Scans emitted report, ABSTAIN, decision, and escalation-packet text for
prohibited NOAA alert terminology before emission.

Six of the eight terms are product terminology from NWS Instruction
10-701, Tsunami Warning Center Operations, and are reserved for the
Tsunami Warning Centers. Five are defined in Sections 2.1.1 to 2.1.5;
Threat Message is named in Section 2.2, which leaves the definition to
the regional coordination groups. NWS Policy
Directive 10-7, which 10-701 is issued under, sets roles and
responsibilities and does not itself define the products. The other two
terms are not product names; PROHIBITED_TERMS below says why each is
scanned anyway.

"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime

from hazard_assessment.policy._confusables import CONFUSABLE_TO_ASCII

# Common Unicode homoglyphs that visually resemble Latin letters.
# Maps confusable characters to their Latin equivalents. This covers
# the most common Cyrillic and Greek look-alikes used in homoglyph
# bypass attacks. Fullwidth characters are handled by NFKC normalization.
_CONFUSABLE_MAP: dict[str, str] = {
    "\u0410": "A",  # Cyrillic А
    "\u0412": "B",  # Cyrillic В
    "\u0421": "C",  # Cyrillic С
    "\u0415": "E",  # Cyrillic Е
    "\u041d": "H",  # Cyrillic Н
    "\u041a": "K",  # Cyrillic К
    "\u041c": "M",  # Cyrillic М
    "\u041e": "O",  # Cyrillic О
    "\u0420": "P",  # Cyrillic Р
    "\u0422": "T",  # Cyrillic Т
    "\u0425": "X",  # Cyrillic Х
    "\u0430": "a",  # Cyrillic а
    "\u0435": "e",  # Cyrillic е
    "\u043e": "o",  # Cyrillic о
    "\u0440": "p",  # Cyrillic р
    "\u0441": "c",  # Cyrillic с
    "\u0443": "y",  # Cyrillic у
    "\u0445": "x",  # Cyrillic х
    "\u043c": "m",  # Cyrillic м
    "\u043d": "n",  # Cyrillic н
    "\u0456": "i",  # Cyrillic і (Ukrainian)
    "\u0455": "s",  # Cyrillic ѕ (Macedonian Dze)
    "\u0406": "I",  # Cyrillic І (Ukrainian)
    "\u0405": "S",  # Cyrillic Ѕ (Macedonian Dze)
    "\u04bb": "h",  # Cyrillic һ (shha)
    "\u0131": "i",  # Latin dotless ı (Turkish)
    "\u0391": "A",  # Greek Α
    "\u0392": "B",  # Greek Β
    "\u0395": "E",  # Greek Ε
    "\u0397": "H",  # Greek Η
    "\u0399": "I",  # Greek Ι
    "\u039a": "K",  # Greek Κ
    "\u039c": "M",  # Greek Μ
    "\u039d": "N",  # Greek Ν
    "\u039f": "O",  # Greek Ο
    "\u03a1": "P",  # Greek Ρ
    "\u03a4": "T",  # Greek Τ
    "\u03a7": "X",  # Greek Χ
    "\u03b1": "a",  # Greek α
    "\u03b9": "i",  # Greek ι (iota)
    "\u03bd": "v",  # Greek ν (nu)
    "\u03bf": "o",  # Greek ο
}

def _small_capital_latin_map() -> dict[str, str]:
    """Fold Unicode "LATIN LETTER SMALL CAPITAL X" characters to ASCII.

    Small-capital letters (mostly in the IPA Extensions and Phonetic
    Extensions blocks) read as ordinary capitals - for example
    "\u1d21\u1d00\u0280\u0274\u026a\u0274\u0262" reads as WARNING - but have no
    NFKC decomposition, so normalization alone does not fold them. Derive the
    fold from each character's Unicode name rather than a hand-listed table so
    the whole small-capital Latin alphabet is covered, not just today's terms.
    """
    prefix = "LATIN LETTER SMALL CAPITAL "
    folded: dict[str, str] = {}
    for codepoint in (*range(0x0250, 0x02B0), *range(0x1D00, 0x1D80), *range(0xA720, 0xA800)):
        char = chr(codepoint)
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        if name.startswith(prefix):
            letter = name[len(prefix) :]
            if len(letter) == 1 and "A" <= letter <= "Z":
                folded[char] = letter.lower()
    return folded


# The hand-written map above covers the Cyrillic and Greek letters that were
# the known attack vectors. It is not enough on its own: sweeping every
# single-character entry in the Unicode confusables data against the reserved
# terms found 406 further characters, across Cherokee, Lisu, Arabic, Coptic,
# Miao, Carian and Warang Citi among others, that spelled a reserved term the
# scanner did not see. CONFUSABLE_TO_ASCII carries that data, generated from
# the authoritative source into a committed module so the scanner needs no
# network access at runtime.
#
# Still not covered, deliberately: multi-character confusable sequences such as
# "rn" for "m". Folding a sequence changes the length of the text, and the
# position reported with each violation has to keep pointing at the scanned
# string.
_CONFUSABLE_TRANS = str.maketrans(
    {**CONFUSABLE_TO_ASCII, **_CONFUSABLE_MAP, **_small_capital_latin_map()}
)


PROHIBITED_TERMS: list[str] = [
    # Product terminology from NWSI 10-701. Warning, Advisory, Watch and
    # Information Statement are defined in Sections 2.1.1 to 2.1.4 and
    # Cancellation in 2.1.5. Threat Message is only named, in Section 2.2,
    # as a product PTWC issues to its international designated service area.
    "Warning",
    "Advisory",
    "Watch",
    "Information Statement",
    "Threat Message",
    "Cancellation",
    # The last two are not product names, and are scanned anyway.
    #
    # "All Clear" appears in neither NWSI 10-701 nor NWSPD 10-7, but it reads
    # as an authoritative statement that the threat has ended, which is the
    # role 10-701 gives the Cancellation product. A non-authoritative system
    # saying it would be making exactly the call it must not make.
    "All Clear",
    # "Bulletin" names no entry on the current TWC product list, but it is
    # still center vocabulary. NWSPD 10-7 charges the centers with issuing
    # "Tsunami Warning, Advisory, Watch, and Information Bulletins", and
    # NWSI 10-701 numbers its own worked examples "Bulletin 1: Initial Watch"
    # and "Bulletin 2: Upgrade Watch to Warning".
    "Bulletin",
]

# Word-boundary regex patterns for prohibited terms (case-insensitive).
# Matches "Warning" as a standalone word but not within "forewarning".
# For multi-word terms the inter-word gap is matched as _WORD_SEPARATOR so
# split renderings are still caught: after the zero-width strip in scan_text
# a term like "All\u200bClear" collapses to "AllClear", while "All\tClear"
# or "All  Clear" separate the words with non-single whitespace.
#
# Compound-forming hyphens join the same words, so "All-Clear" and
# "Information-Statement" are the reserved product names too. Without them
# the scanner was stricter about no separator at all ("AllClear" matched)
# than about the ordinary English hyphenation, which is how a reviewer or a
# narrative model is most likely to write the phrase.
#
# The hyphen must be unspaced, because a spaced dash separates clauses
# instead of building a compound. This repository writes plain ASCII, so
# " - " is its ordinary clause separator, and matching it would reject
# legitimate prose such as "that is all - clear skies ahead". En and em
# dashes are excluded for the same reason. The alternation therefore allows
# a run of whitespace, or exactly one unspaced hyphen, or nothing at all
# (the zero-width strip in scan_text collapses "All\u200bClear" to
# "AllClear"). The reserved term strings themselves are unchanged.
# Unspaced joiners that build the same compound: hyphens, underscore, slash,
# period and colon. A SPACED dash is excluded and stays a clause separator,
# so "that is all - clear skies ahead" is still ordinary prose.
_INVISIBLE_NON_FORMAT = frozenset(
    {
        0x180B, 0x180C, 0x180D, 0x180F,  # Mongolian free variation selectors
        0x115F, 0x1160, 0x3164, 0xFFA0,  # Hangul fillers
        0x2800,                          # Braille pattern blank
        0xFFFC,                          # object replacement character
        0x17B4, 0x17B5,                  # Khmer invisible vowel signs
    }
)

_JOINERS = (
    "-_/.:~|"
    "\u2010\u2011\u2012\u2013\u2014\u2015\u2017\u2043\u2212"
    "\u2e17\u2e3a\u2e3b"
    "\u2215\u2044\u29f8"
    "\ua789\u02d0\u0589\u05c3\u2e2e\u30fb\u00b7\u0387\u2022\u2e31\u00a6"
)
# One or more unspaced joiners, or a run of whitespace. The "+" matters: a
# single-character class let "All__Clear" and "All--Clear" through, which are
# the same compound. The non-ASCII members are punctuation that renders
# identically to the ASCII joiners but does not NFKC-fold to it, so
# "All\u2215Clear" is pixel-identical to "All/Clear" and has to be treated the
# same. A joiner adjacent to whitespace is deliberately still not matched,
# because a spaced dash separates clauses: "that is all - clear skies ahead"
# must remain ordinary prose.
_WORD_SEPARATOR = r"(?:\s+|[" + re.escape(_JOINERS) + r"]+)?"


def _inflect(word: str) -> str:
    """Match a term's singular and regular plural.

    "Warnings" and "Watches" are the forms an alert-styled narrative actually
    uses, and a bare ``\b`` boundary lets both through because the suffix is a
    word character. Only the final word of a multi-word term is inflected:
    the reserved product is "Information Statements", never
    "Informations Statement".
    """
    escaped = re.escape(word)
    if len(word) > 1 and word[-1].lower() == "y" and word[-2].lower() not in "aeiou":
        return re.escape(word[:-1]) + "(?:" + re.escape(word[-1]) + "|ies)"
    if word.lower().endswith(("s", "x", "z", "ch", "sh")):
        return escaped + "(?:es)?"
    return escaped + "s?"


def _boundary_pattern(term: str) -> re.Pattern[str]:
    parts = term.split()
    bodies = [re.escape(part) for part in parts[:-1]] + [_inflect(parts[-1])]
    body = _WORD_SEPARATOR.join(bodies)
    return re.compile(rf"\b{body}\b", re.IGNORECASE)


_PROHIBITED_PATTERNS: list[re.Pattern[str]] = [
    _boundary_pattern(term) for term in PROHIBITED_TERMS
]

# Allowlisted phrases where prohibited terms appear as part of a proper
# noun or organizational name rather than labeling system output. They use
# the same whitespace-flexible matching so an oddly spaced organization
# name is not mistaken for a bare reserved term.
_ALLOWLISTED_PHRASES: list[str] = [
    "Tsunami Warning Center",
    "Pacific Tsunami Warning Center",
    "National Tsunami Warning Center",
    # "Bulletin" in proper-noun context (e.g., "Tsunami Bulletin Board")
    "Tsunami Bulletin Board",
]
# Characters that share a glyph shape rather than an identity: capital I,
# digit one and the vertical bar all read as lowercase l, and digit zero reads
# as o. The confusable fold cannot resolve these, because a capital I really is
# a capital I and folding it to l would be wrong for every other purpose. They
# are collapsed in a second matching pass instead, over both the text and the
# terms, so "AlI Clear" and "CanceIlation" are caught. Measured against 187
# files of this repository's own prose, code, tests and results, the collapse
# introduces no match the ordinary pass does not already make.
_SHAPE_TRANS = str.maketrans({"I": "l", "1": "l", "|": "l", "0": "o"})

_SHAPE_PATTERNS: list[re.Pattern[str]] = [
    _boundary_pattern(term.translate(_SHAPE_TRANS)) for term in PROHIBITED_TERMS
]

_ALLOWLISTED_PATTERNS: list[re.Pattern[str]] = [
    _boundary_pattern(phrase) for phrase in _ALLOWLISTED_PHRASES
]

NON_AUTHORITATIVE_DISCLAIMER = (
    "Non-authoritative situational awareness. Not an official NOAA tsunami message."
)


@dataclass(frozen=True)
class GuardrailViolation:
    """A detected prohibited term in output text."""

    term: str
    position: int
    context: str  # Surrounding text for logging


@dataclass
class ScanResult:
    """Result of scanning text for prohibited alert terminology."""

    text_scanned: str
    passed: bool
    violations: list[GuardrailViolation] = field(default_factory=list)
    has_disclaimer: bool = False
    scanned_at_utc: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        if self.scanned_at_utc.tzinfo is None:
            raise ValueError("scanned_at_utc must be timezone-aware")


def scan_text(text: str) -> ScanResult:
    """Scan text for prohibited NOAA alert terminology.

    Text fails if it contains any prohibited term or lacks the mandatory
    non-authoritative disclaimer.

    Applies NFKC Unicode normalization before scanning to prevent
    homoglyph bypass attacks (e.g., Cyrillic 'а' or Greek 'α' used
    in place of Latin 'a' to spell "Wаrning" or "Wαrning").
    """
    # Normalize Unicode to catch homoglyph bypass attempts.
    # Step 1: Strip zero-width characters that could split prohibited
    # terms (e.g., "Warn\u200Bing") and evade word-boundary detection.
    # Step 2: NFKC maps compatibility characters to their canonical
    # forms (e.g., fullwidth 'Ｗ' -> 'W').
    # Step 3: Strip combining marks so an accented homoglyph of an ASCII
    # reserved term (e.g., "Wa\u0308rning" or precomposed "Wärning")
    # collapses back to the bare term. The reserved terms and allowlist
    # are pure ASCII, so this cannot introduce a false positive.
    # Step 4: Confusable map replaces Cyrillic/Greek look-alikes and
    # small-capital Latin letters with their Latin equivalents (e.g.,
    # Cyrillic 'а' -> 'a', 'ᴡ' -> 'w').
    # NOTE: coverage is now the Unicode confusables data itself (UTS #39),
    # every single-character entry whose skeleton is a single ASCII letter,
    # regenerated by scripts/generate_confusable_map.py. What remains
    # uncovered is multi-character sequences such as "rn" for "m", which
    # cannot be folded without shifting the offsets reported with each
    # violation.
    stripped = re.sub(
        r"[\u00ad\u034f\u061c\u180e"
        r"\u200b-\u200f\u2028-\u202f\u2060-\u2064\u2066-\u206f"
        r"\ufeff\ufe00-\ufe0f]",
        "",
        text,
    )
    # Property-driven catch-all behind the explicit ranges above. Every
    # Unicode format character (category Cf) and every variation selector is
    # invisible to a reader but splits a reserved term for the matcher, and
    # each new Unicode version adds more of them. This pass is additive: it
    # can only remove characters the explicit ranges missed, never fewer, so
    # it cannot weaken existing coverage. Visible spacing is deliberately left
    # alone, since deleting a real space would join words and could hide a
    # term rather than expose it.
    stripped = "".join(
        ch
        for ch in stripped
        if unicodedata.category(ch) != "Cf"
        and not (0xFE00 <= ord(ch) <= 0xFE0F)
        and not (0xE0100 <= ord(ch) <= 0xE01EF)
        # Invisible but neither Cf nor combining, so neither the category
        # filter above nor the mark strip below removes them: Mongolian free
        # variation selectors (Mn, combining class 0), Hangul fillers (Lo),
        # Braille blank and the object-replacement character (So), and the
        # Khmer invisible vowel signs (Mn, class 0).
        and ord(ch) not in _INVISIBLE_NON_FORMAT
    )
    # Fold confusables BEFORE normalizing as well as after. NFKC rewrites some
    # of these characters into something that no longer resembles the letter
    # they were standing in for, and the fold can never run on them again:
    # a mathematical digit zero becomes an ASCII 0, a long s becomes an s
    # rather than the f it resembles, and a spacing ogonek becomes a space
    # plus a combining mark that the next step strips. Measured against the
    # Unicode confusables data, folding only after normalization left 234
    # spellings of reserved terms unmatched that this pass catches. Both
    # passes are 1:1 on characters, so reported positions still index the
    # scanned text.
    pre_folded = stripped.translate(_CONFUSABLE_TRANS)
    decomposed = unicodedata.normalize("NFD", unicodedata.normalize("NFKC", pre_folded))
    without_marks = "".join(c for c in decomposed if not unicodedata.combining(c))
    normalized = without_marks.translate(_CONFUSABLE_TRANS)

    # Second normalization with the orderings swapped: NFKC first, fold after.
    # The pre-fold above is shape-based and NFKC is identity-based, and where
    # they disagree the pre-fold wins because NFKC can never see the original
    # character again. Sixteen codepoints are affected: fifteen capital-I
    # variants that shape-fold to "l" (including every mathematical-alphabet
    # capital I and FULLWIDTH LATIN CAPITAL LETTER I) and LATIN SMALL LETTER
    # LONG S, which shape-folds to "f". That let a fullwidth all-caps
    # "WARNING" and a long-s "Advisory" through, while the lowercase fullwidth
    # form was caught. Scanning both orderings costs one extra pass and closes
    # the gap without weakening the shape fold, which still catches the
    # spellings NFKC destroys.
    #
    # This closes single-character substitutions, not all of them. Two
    # substitutions in one term, one that only the pre-fold resolves and one
    # that only NFKC resolves, defeat both orderings at once: "A ϲancellatℐon"
    # (GREEK LUNATE SIGMA plus SCRIPT CAPITAL I) is caught if either character
    # is removed and passes with both. Closing that needs folding each
    # character to a set of candidate letters and matching over the product,
    # not N whole-string orderings.
    nfkc_decomposed = unicodedata.normalize(
        "NFD", unicodedata.normalize("NFKC", stripped)
    )
    alternate = "".join(
        c for c in nfkc_decomposed if not unicodedata.combining(c)
    ).translate(_CONFUSABLE_TRANS)

    violations: list[GuardrailViolation] = []

    def _allowlisted(haystack: str) -> set[int]:
        """Character positions covered by an allowlisted proper noun.

        Computed per haystack: the two normalizations are not the same length,
        so an offset from one does not index the other. The shape-collapsed
        form is included as well, or the collapse pass would report a
        violation inside "Pacific Tsunami Warning Center".
        """
        covered: set[int] = set()
        for allow_pattern in _ALLOWLISTED_PATTERNS:
            for allow_match in allow_pattern.finditer(haystack):
                covered.update(range(allow_match.start(), allow_match.end()))
            for allow_match in allow_pattern.finditer(haystack.translate(_SHAPE_TRANS)):
                covered.update(range(allow_match.start(), allow_match.end()))
        return covered

    # A match position indexes the normalization it was found in, which is
    # what `context` is cut from. It does NOT index `text_scanned`: NFKC is not
    # length preserving, so a compatibility ligature shifts every later offset.
    # No consumer reads `position`; they use `term` and the violation count.
    seen: set[tuple[str, int]] = set()
    reported_terms: set[str] = set()
    # Terms the allowlist suppressed in the primary normalization. The
    # allowlist has to be evaluated per haystack, because the two
    # normalizations are different lengths and an offset from one does not
    # index the other. That asymmetry means the alternate pass can only ever
    # ADD a violation, so a proper noun that survives the primary
    # normalization but is mangled by the alternate one ("Tsunam˛ Warning
    # Center", where OGONEK becomes a space) had its allowlist entry missed
    # and the alternate pass reported the reserved word inside an
    # organization name. Carrying the suppression across restores that.
    allowlisted_terms: set[str] = set()
    for patterns, haystack, context_source in (
        (_PROHIBITED_PATTERNS, normalized, normalized),
        (_SHAPE_PATTERNS, normalized.translate(_SHAPE_TRANS), normalized),
        (_PROHIBITED_PATTERNS, alternate, alternate),
        (_SHAPE_PATTERNS, alternate.translate(_SHAPE_TRANS), alternate),
    ):
        alternate_pass = context_source is alternate
        allowlisted_positions = _allowlisted(context_source)
        for pattern, term in zip(patterns, PROHIBITED_TERMS):
            # The alternate normalization exists to catch spellings the primary
            # one misses, so it only contributes terms not already reported.
            # Its offsets index a different string, and re-reporting the same
            # term from both would double-count one occurrence.
            if alternate_pass and (term in reported_terms or term in allowlisted_terms):
                continue
            for match in pattern.finditer(haystack):
                # Skip if this match falls within an allowlisted proper noun
                if any(
                    pos in allowlisted_positions for pos in range(match.start(), match.end())
                ):
                    if not alternate_pass:
                        allowlisted_terms.add(term)
                    continue
                if (term, match.start()) in seen:
                    continue
                seen.add((term, match.start()))
                reported_terms.add(term)
                start = max(0, match.start() - 30)
                end = min(len(context_source), match.end() + 30)
                # Context comes from the normalization this match was found in,
                # so a reader sees the string that was actually scanned rather
                # than the shape-collapsed form.
                context = context_source[start:end]
                violations.append(
                    GuardrailViolation(
                        term=term,
                        position=match.start(),
                        context=context,
                    )
                )

    has_disclaimer = NON_AUTHORITATIVE_DISCLAIMER in normalized

    passed = len(violations) == 0 and has_disclaimer

    return ScanResult(
        text_scanned=text,
        passed=passed,
        violations=violations,
        has_disclaimer=has_disclaimer,
    )


def scan_structure(value: object) -> list[GuardrailViolation]:
    """Scan every string inside a nested structure, keys included.

    Callers used to serialize a tool-call log with ``json.dumps`` and scan the
    result. That silently disabled most of this module. ``json.dumps`` defaults
    to ``ensure_ascii=True``, so a Cyrillic or fullwidth homoglyph became a
    literal ``\\uXXXX`` escape before the scanner saw it, and the confusable
    fold, the small-capital fold and the NFKC/NFD passes above all had nothing
    left to catch. JSON escaping breaks the whitespace-flexible match too: a tab
    inside a reserved phrase serializes to a backslash and a ``t``. The escaped
    form was scanned while the real characters were persisted and returned.

    Scanning each string on its own also avoids the opposite error, where
    joining values with a separator invents a match that spans two of them.

    Dictionary keys are scanned because they are not always ours: a model
    chooses its own tool-argument names, and those names become keys.
    """
    violations: list[GuardrailViolation] = []
    stack: list[object] = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, str):
            violations.extend(scan_text(item).violations)
        elif isinstance(item, dict):
            for key, sub in item.items():
                stack.append(key)
                stack.append(sub)
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif item is not None and not isinstance(item, (bool, int, float)):
            # Anything else reaches the operator through its string form.
            violations.extend(scan_text(str(item)).violations)
    return violations
