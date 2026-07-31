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

#: ``\promptlabel[style]{Title \normalfont...}`` then a verbatim ``promptbox``.
_BOX = re.compile(
    r"\\promptlabel\[[^\]]*\]\{([^}]*?)(?:\s*\\normalfont.*?)?\}\s*\n"
    r"\\begin\{promptbox\}\n(.*?)\\end\{promptbox\}",
    re.DOTALL,
)

_CONST = re.compile(r'^([A-Z_]+_PROMPT) = """\\?\n(.*?)"""', re.DOTALL | re.MULTILINE)


def _paper_listings() -> list[tuple[str, str]]:
    return [(title.strip(), body.strip()) for title, body in _BOX.findall(PAPER.read_text())]


def _source_prompts() -> dict[str, str]:
    return {name: body.strip() for name, body in _CONST.findall(PROMPTS.read_text())}


def test_every_prompt_listing_is_byte_exact_against_source() -> None:
    listings = _paper_listings()
    prompts = _source_prompts()
    assert listings, "no promptbox listings found; has the appendix macro changed?"

    unmatched: list[str] = []
    for title, body in listings:
        if body not in prompts.values():
            near = max(
                prompts.items(),
                key=lambda kv: len(set(kv[1].split()) & set(body.split())),
            )
            unmatched.append(
                f"{title!r} is not byte-exact against any prompt in prompts.py "
                f"(closest: {near[0]})"
            )

    assert not unmatched, (
        "the paper reproduces prompts that do not match the source verbatim:\n  "
        + "\n  ".join(unmatched)
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
    assert "omitted here for length" in paper, (
        "the appendix no longer discloses that the investigator prompts are "
        "omitted, but prompts.py still defines them"
    )
