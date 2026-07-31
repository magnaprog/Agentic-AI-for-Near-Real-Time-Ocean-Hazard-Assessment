"""The prompt listings in the paper must be byte-exact copies of the source.

The appendix reproduces every system prompt inside a ``promptbox``, which is a
verbatim environment. Nothing tied those listings to ``prompts.py``, and they
drifted twice in different directions.

The narrative listing had picked up an ``<historical_context>`` block that is
not in the system prompt at all: ``synthesis_graph.py`` assembles it into the
*human* message, and ``narrative_node`` passes the system prompt through
unmodified. A reader comparing the appendix against the source would have
concluded the prompt carries a runtime placeholder it does not have.

The evidence listing had been rewrapped by hand, which is harmless to a reader
but defeats any exact comparison, so a real insertion could hide inside a
reflow.

Byte-exact is the only version of this check that is worth having: anything
looser is what let the first drift through.
"""

from __future__ import annotations

import pathlib
import re

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
PAPER = REPO_ROOT / "paper" / "paper.tex"
PROMPTS = REPO_ROOT / "src" / "hazard_assessment" / "agents" / "llm_advisory" / "prompts.py"

#: ``\promptlabel[style]{Title \normalfont\textnormal{\textit{(tier)}}}`` then a
#: verbatim ``promptbox``. The tier annotation is captured, not discarded: it
#: states which model runs the prompt, and a listing relabelled from the fast
#: model to the quality model would otherwise misinform a reader while every
#: assertion still passed.
_BOX = re.compile(
    r"\\promptlabel\[[^\]]*\]\{([^}]*?)(?:\s*\\normalfont(.*?))?\}\s*\n"
    r"\\begin\{promptbox\}\n(.*?)\\end\{promptbox\}",
    re.DOTALL,
)

#: Model tier each listing declares. Transcribed from the ``purpose=`` each
#: node builds (``synthesis_graph.py`` fast/fast/standard, ``after_action.py``
#: quality) and checked against the label in the paper. This is a table, not a
#: read of those files: it catches a relabelled listing, not a changed node.
EXPECTED_TIER = {
    "Evidence Synthesis Prompt": "fast",
    "Scenario Interpretation Prompt": "fast",
    "Narrative Synthesis Prompt": "standard",
    "Timeline Reconstruction Prompt": "quality",
    "Detection Gap Analysis Prompt": "quality",
    "Incident Report Draft Prompt": "quality",
}

#: ``NAME = """`` and ``NAME = f"""`` both define a prompt. Matching only the
#: plain form let an f-string prompt be added to source and omitted from the
#: appendix without failing anything, and prompts.py already writes three of
#: its prompts that way.
_CONST = re.compile(
    r'^([A-Z_]+_PROMPT) = f?"""\\?\n(.*?)"""', re.DOTALL | re.MULTILINE
)

#: Which source constant each appendix listing claims to reproduce.
#:
#: Set membership alone is not enough. Checking only that every listing matches
#: *some* constant lets the bodies be swapped between two titles, or one prompt
#: be printed twice while another silently disappears from the appendix, and
#: the listing still passes. Binding the title to the constant is what makes a
#: mislabelled or missing prompt fail.
#: The investigator's three prompts, deliberately omitted from the appendix for
#: length and disclosed as such. Listed by name rather than skipped by pattern
#: so that a fourth investigator prompt, or any new prompt, still fails below.
KNOWN_OMITTED = {
    "STATION_AGREEMENT_PROMPT",
    "EVIDENCE_GAPS_PROMPT",
    "TIMELINE_CONSISTENCY_PROMPT",
}

EXPECTED = {
    "Evidence Synthesis Prompt": "EVIDENCE_SYSTEM_PROMPT",
    "Scenario Interpretation Prompt": "SCENARIO_SYSTEM_PROMPT",
    "Narrative Synthesis Prompt": "NARRATIVE_SYSTEM_PROMPT",
    "Timeline Reconstruction Prompt": "TIMELINE_SYSTEM_PROMPT",
    "Detection Gap Analysis Prompt": "GAPS_SYSTEM_PROMPT",
    "Incident Report Draft Prompt": "DRAFT_SYSTEM_PROMPT",
}


def _paper_listings() -> list[tuple[str, str, str]]:
    return [
        (title.strip(), tier.strip(), body.strip())
        for title, tier, body in _BOX.findall(PAPER.read_text())
    ]


def _source_prompts() -> dict[str, str]:
    return {name: body.strip() for name, body in _CONST.findall(PROMPTS.read_text())}


def test_every_prompt_listing_is_byte_exact_against_source() -> None:
    listings = _paper_listings()
    prompts = _source_prompts()
    assert listings, "no promptbox listings found; has the appendix macro changed?"

    problems: list[str] = []
    seen: dict[str, str] = {}
    for title, tier, body in listings:
        want_tier = EXPECTED_TIER.get(title)
        if want_tier is not None and want_tier not in tier:
            problems.append(
                f"{title!r} is labelled {tier!r}; source runs it on the "
                f"{want_tier} model"
            )
        expected = EXPECTED.get(title)
        if expected is None:
            problems.append(f"{title!r} is not a listing this test knows about")
            continue
        if expected not in prompts:
            problems.append(f"{title!r} claims {expected}, which prompts.py no longer defines")
            continue
        if prompts[expected] != body:
            problems.append(f"{title!r} is not byte-exact against {expected}")
        if expected in seen:
            problems.append(f"{expected} is reproduced under both {seen[expected]!r} and {title!r}")
        seen[expected] = title

    missing = sorted(set(EXPECTED.values()) - set(seen))
    if missing:
        problems.append(f"prompts the appendix no longer reproduces: {missing}")

    # A prompt added to prompts.py that is neither reproduced nor on the
    # known-omitted list would otherwise disappear from the appendix silently,
    # which is the failure this file exists to prevent.
    unlisted = sorted(set(prompts) - set(EXPECTED.values()) - KNOWN_OMITTED)
    if unlisted:
        problems.append(
            f"prompts.py defines {unlisted}, which the appendix does not "
            "reproduce and this test does not account for"
        )

    assert not problems, (
        "the paper's prompt listings disagree with prompts.py:\n  " + "\n  ".join(problems)
    )


def test_the_paper_reproduces_the_report_time_prompts() -> None:
    """The appendix says it gives the two report-time advisory graphs.

    A count floor keeps a listing from silently disappearing, and keeps the
    appendix's own claim about what it contains honest.
    """
    listings = _paper_listings()
    assert len(listings) >= 6, (
        f"only {len(listings)} prompt listings found; the appendix claims to "
        "reproduce the synthesis and after-action graphs"
    )


def test_the_investigator_prompts_are_not_claimed_as_reproduced() -> None:
    """The investigator prompts are deliberately omitted for length.

    They exist in the same module, so a future edit that drops the disclaimer
    would leave the appendix claiming a completeness it does not have.
    """
    prompts_src = PROMPTS.read_text()
    assert "_INVESTIGATOR_COMMON_RULES" in prompts_src, (
        "investigator prompts moved; the appendix disclaimer needs rechecking"
    )
    paper = PAPER.read_text()
    # Anchor the disclaimer to the prompt appendix. A bare substring search
    # would be satisfied by the phrase appearing anywhere in a 90-page file.
    start = paper.index("\\label{sec:appendix-prompts}")
    section = paper[start : paper.index("\\promptlabel", start)]
    assert "are omitted here for length" in section, (
        "the prompt appendix no longer discloses that the investigator "
        "prompts are omitted, but prompts.py still defines them"
    )
