#!/usr/bin/env python3
"""Generate the guardrail confusable-folding table from Unicode data.

The reserved-language scanner folds visually confusable characters to ASCII
before matching, so that a term spelled with a lookalike letter is still
caught. The hand-written map in ``policy/guardrails.py`` covered Cyrillic and
Greek only; a sweep against the Unicode confusables data found 406 further
characters, across Cherokee, Lisu, Arabic, Coptic, Miao, Carian, Warang Citi
and others, that spelled a reserved term the scanner did not see.

This script turns the authoritative data into a committed Python module so the
scanner needs no network access and the table can be reviewed in a diff.

Usage::

    curl -o /tmp/confusables.txt \\
        https://www.unicode.org/Public/security/latest/confusables.txt
    python scripts/generate_confusable_map.py /tmp/confusables.txt

Only single-character sources that fold to a single ASCII letter are kept.
Multi-character confusable sequences (the data also maps "rn" to "m", for
example) are deliberately excluded: folding those changes the length of the
text and would break the position reporting the scanner returns with each
violation.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

OUTPUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "hazard_assessment"
    / "policy"
    / "_confusables.py"
)

HEADER = '''"""Confusable-to-ASCII folding table for the reserved-language scanner.

GENERATED FILE. Do not edit by hand. Regenerate with::

    curl -o /tmp/confusables.txt \\\\
        https://www.unicode.org/Public/security/latest/confusables.txt
    python scripts/generate_confusable_map.py /tmp/confusables.txt

Derived from the Unicode Security Mechanisms data (UTS #39), version {version},
dated {date}. Copyright (C) 1991-2025 Unicode, Inc. Distributed under the
Unicode License; see https://www.unicode.org/terms_of_use.html.

Contains every single-character confusable whose skeleton is a single ASCII
letter. Multi-character sources are excluded on purpose: folding a sequence
would shift the character offsets the scanner reports with each violation.
"""

from typing import Final

#: Maps one confusable character to the ASCII letter it can be mistaken for.
CONFUSABLE_TO_ASCII: Final[dict[str, str]] = {{
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2

    source = Path(sys.argv[1])
    text = source.read_text(encoding="utf-8")

    version_match = re.search(r"^# Version: (\S+)", text, re.M)
    date_match = re.search(r"^# Date: (.+?)\s*$", text, re.M)
    version = version_match.group(1) if version_match else "unknown"
    date = date_match.group(1) if date_match else "unknown"

    mapping: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("#") or ";" not in line:
            continue
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        raw_source, raw_target = parts[0], parts[1]
        if " " in raw_source:
            continue
        try:
            char = chr(int(raw_source, 16))
            target = "".join(chr(int(cp, 16)) for cp in raw_target.split())
        except ValueError:
            continue
        if len(target) != 1 or not target.isascii() or not target.isalpha():
            continue
        if ord(char) < 128:
            continue
        mapping[char] = target

    lines = [HEADER.format(version=version, date=date)]
    for char in sorted(mapping):
        name = f"U+{ord(char):04X}"
        lines.append(f'    "\\u{ord(char):04x}": "{mapping[char]}",  # {name}\n'
                     if ord(char) <= 0xFFFF
                     else f'    "\\U{ord(char):08x}": "{mapping[char]}",  # {name}\n')
    lines.append("}\n")

    OUTPUT.write_text("".join(lines), encoding="utf-8")
    print(f"wrote {OUTPUT} with {len(mapping)} entries (Unicode {version}, {date})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
