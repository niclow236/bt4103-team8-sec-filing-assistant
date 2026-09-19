"""Resolve generated markers against the exact sources numbered in the prompt.

Pass ``prompt.passages``, never a newly sorted retrieval result. Resolution
checks source identity only; whether a passage supports a claim is #32's job.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from pydantic import ValidationError

from ..retrieval.records import RetrievedPassage
from .constants import ABSTAIN_PHRASE
from .records import (
    Answer,
    Citation,
    Generation,
    GroundedAnswer,
    SentenceCitations,
    render_sentence,
)

# Include zero and negative numbers so they are removed and flagged, too.
_MARKER = re.compile(r"\[([+-]?[0-9]+)\]")


def _resolve_sentence(
    text: str, passages: tuple[RetrievedPassage, ...], *, incomplete: bool
) -> SentenceCitations:
    citations: dict[int, Citation] = {}
    invalid: list[str] = []

    def replace_marker(match: re.Match[str]) -> str:
        number = int(match[1])
        if number < 1:
            invalid.append(match[0])
            return ""
        passage = passages[number - 1] if number <= len(passages) else None
        citations[number] = Citation(
            marker=number,
            chunk_id=None if passage is None else passage.chunk_id,
            resolved=passage is not None,
        )
        return f"[{number}]" if passage is not None else ""

    cleaned = _MARKER.sub(replace_marker, text)
    # Removing " [9]" must not leave doubled spaces or "claim .".
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned).strip()
    return SentenceCitations(
        raw_text=text,
        text=cleaned,
        citations=tuple(citations[n] for n in sorted(citations)),
        invalid_markers=tuple(dict.fromkeys(invalid)),
        incomplete=incomplete,
    )


def resolve_citations(
    question: str,
    generation: Generation,
    passages: Iterable[RetrievedPassage],
) -> Answer:
    """Build the final Answer, retaining invented citations for audit and scoring.

    ``passages`` must be the exact ``prompt.passages`` used for generation;
    source numbers are positions, not retrieval ranks. Invalid markers are
    removed from ``Answer.text`` but remain in its citation checks. A sentence
    with any invalid marker, or with no citation, is flagged even when other
    sentences have valid citations. Consumers should show ``flagged_sentences``
    alongside the cleaned text.

    Structured output supplies sentence boundaries, including decimals and
    abbreviations. If validation failed only because sources were out of range,
    the unconstrained record can recover those boundaries from the full JSON.
    Otherwise all available prose is retained as one flagged fallback block.
    Parse errors and truncation remain visible on the returned Answer.
    """
    shown = tuple(passages)
    structured = generation.answer
    if structured is None:
        try:
            recovered = GroundedAnswer.model_validate_json(generation.raw)
        except ValidationError:
            pass
        else:
            # Never replace the displayed partial output with a different answer.
            if recovered.render() == generation.text:
                structured = recovered

    abstained = (
        structured.abstained
        if structured is not None
        else generation.text.strip() == ABSTAIN_PHRASE
    )
    incomplete = generation.truncated or (
        structured is None and generation.parse_error is not None
    )
    if abstained:
        sentences = ()
    else:
        raw_sentences = (
            [render_sentence(s.text, s.sources) for s in structured.sentences if s.text.strip()]
            if structured is not None
            else [generation.text] if generation.text.strip() else []
        )
        sentences = tuple(
            _resolve_sentence(text, shown, incomplete=incomplete)
            for text in raw_sentences
        )

    citations = {c.marker: c for sentence in sentences for c in sentence.citations}
    return Answer(
        question=question,
        text=ABSTAIN_PHRASE if abstained else " ".join(s.text for s in sentences),
        citations=tuple(citations[n] for n in sorted(citations)),
        passages=shown,
        abstained=abstained,
        config=generation.config,
        latency_ms=generation.latency_ms,
        sentences=sentences,
        parse_error=generation.parse_error,
        truncated=generation.truncated,
    )


def render_citation(citation: Citation, passages: Sequence[RetrievedPassage]) -> str:
    """A plain-text label from stored metadata, or an explicit unresolved label.

    Use the returned Answer's ``passages``. URLs remain available on those
    records for a UI to link; no model-generated bibliographic text is used.
    Missing metadata is labelled unknown rather than inferred from another field.
    """
    passage = passages[citation.marker - 1] if citation.marker <= len(passages) else None
    expected = None if passage is None else passage.chunk_id
    if citation.chunk_id != expected:
        raise ValueError("citation does not match the numbered passages")
    if passage is None:
        return f"[{citation.marker}] Unresolved citation"

    cik = f"{passage.cik:010d}" if passage.cik is not None else "unknown"
    year = passage.fiscal_year if passage.fiscal_year is not None else "unknown"
    return (
        f"[{citation.marker}] {passage.company or 'unknown'} "
        f"({passage.ticker or 'unknown'}, CIK {cik}), "
        f"{passage.form or 'form unknown'}, fiscal year {year}, "
        f"Part {passage.part or 'unknown'} / Item {passage.item or 'unknown'}, "
        f"{passage.title or 'section title unknown'}, "
        f"filed {passage.filing_date or 'unknown'}"
    )


__all__ = ["render_citation", "resolve_citations"]
