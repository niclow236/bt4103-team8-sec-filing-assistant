"""Build the grounded prompt: numbered sources, the rules, and the question.

The prompt is where the citation contract is made. The model is shown the
retrieved passages as sources ``[1]`` to ``[k]``, each under a header naming
the company, the fiscal year and the Item, and is told to answer from those
alone and to cite by number. A source's stored URL is never put in the prompt
-- a passage's own text may quote one, as 168 of the corpus's 28,179 chunks do
-- and the model is never asked to name a document, so what it writes about a
source is its integer, and an integer is the one thing the resolver (#31) can
check. Everything the citation shows, it takes from the passage's own stored
metadata.

The builder is deterministic. The same passages produce the same prompt, in
the same order, whatever order they arrive in: sources are numbered by rank,
with ``chunk_id`` breaking ties, so ``GenerationConfig.prompt_template_id``
plus the passage set identifies the prompt exactly, which is what lets the
harness attribute an answer without storing the rendered text.

The result carries the passages in their numbered order, so that
``prompt.passages[n - 1]`` is what marker ``[n]`` refers to. That is the
invariant ``Answer`` enforces, and the generator (#30) hands this tuple
straight through so the two cannot drift.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from ..retrieval.records import RetrievedPassage
from .constants import (
    PROMPT_TEMPLATE_ID,
    SOURCE_HEADER,
    SOURCE_SECTION,
    SOURCE_SECTION_NO_ITEM,
    SOURCE_SEPARATOR,
    SOURCE_TABLE_TAG,
    SYSTEM_PROMPT,
    USER_PROMPT,
)


@dataclass(frozen=True)
class GroundedPrompt:
    """One rendered prompt, and the sources it was rendered from.

    ``system`` holds the rules and ``user`` the sources and the question,
    split because a chat provider takes them as two messages. A provider that
    takes one string joins them with :meth:`as_text`. ``passages`` is the
    sources in numbered order, so the generator can put them on the ``Answer``
    without re-deriving which passage was ``[3]``.
    """

    template_id: str
    system: str
    user: str
    passages: tuple[RetrievedPassage, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "passages", tuple(self.passages))

    @property
    def n_sources(self) -> int:
        """How many sources the model was shown: the largest marker it may write."""
        return len(self.passages)

    def to_messages(self) -> list[dict[str, str]]:
        """The prompt as a chat transcript: a system turn, then a user turn.

        The shape every chat API takes, so the generator passes it through
        rather than building its own.
        """
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def as_text(self) -> str:
        """The prompt as one string, for a completion-style or local model."""
        return f"{self.system}\n\n{self.user}"


def render_source(marker: int, passage: RetrievedPassage) -> str:
    """One source as the model sees it: the header line, then the passage text.

    The header names what a model needs to choose between sources -- company,
    fiscal year, Item -- and nothing it could copy into an answer. A table
    passage is tagged so the model reads it as a rendered grid rather than as
    prose: its text opens with the table's caption, then the grid under its
    own column headers. Text is stripped of surrounding whitespace so that
    two passages differing only in a trailing newline render the same.
    """
    if passage.item is not None:
        section = SOURCE_SECTION.format(item=passage.item, title=passage.title)
    else:
        section = SOURCE_SECTION_NO_ITEM.format(title=passage.title)
    header = SOURCE_HEADER.format(
        marker=marker,
        company=passage.company,
        ticker=passage.ticker,
        # A passage whose filing year could not be read still has to be shown
        # under a header, and "unknown" is the truth of it.
        fiscal_year=passage.fiscal_year if passage.fiscal_year is not None else "unknown",
        section=section,
    )
    if passage.content_type == "table":
        header += SOURCE_TABLE_TAG
    return f"{header}\n{passage.text.strip()}"


def build_prompt(question: str, passages: Sequence[RetrievedPassage]) -> GroundedPrompt:
    """Render the grounded prompt for one question over its retrieved passages.

    Sources are numbered in rank order, ties broken by ``chunk_id``, so the
    same set of passages gives the same prompt however it was handed over.
    An empty set is refused: with nothing to ground on there is no prompt to
    render, and abstaining on no evidence is the caller's decision (#33), not
    something to ask a model to do. Duplicate passages are refused for the
    reason ``Answer`` refuses them: one source under two numbers makes a
    citation ambiguous.
    """
    if not question or not question.strip():
        raise ValueError("question must not be blank")

    # Sorted first so the emptiness check sees what actually arrived: a
    # generator is truthy however little it yields.
    ordered = sorted(passages, key=lambda p: (p.rank, p.chunk_id))
    if not ordered:
        raise ValueError("cannot build a grounded prompt with no passages")
    ids = [p.chunk_id for p in ordered]
    if len(set(ids)) != len(ids):
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        raise ValueError(f"passages must not repeat a chunk_id: {', '.join(duplicates)}")

    sources = SOURCE_SEPARATOR.join(
        render_source(marker, passage) for marker, passage in enumerate(ordered, start=1)
    )
    return GroundedPrompt(
        template_id=PROMPT_TEMPLATE_ID,
        system=SYSTEM_PROMPT,
        user=USER_PROMPT.format(sources=sources, question=question.strip()),
        passages=tuple(ordered),
    )
