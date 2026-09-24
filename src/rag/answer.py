"""Route a question, retrieve evidence, and either abstain, look it up, or generate."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from math import isfinite
from pathlib import Path
from typing import Any

from ..retrieval.base import Retriever, has_candidates
from ..retrieval.constants import FINAL_K
from ..retrieval.facts import FACTS_FILE
from ..retrieval.records import Query
from .citations import resolve_citations
from .constants import ABSTAIN_PHRASE, MAX_OUTPUT_TOKENS
from .generate import config_from_env, generate
from .numeric import answer_from_facts
from .prompt import build_prompt
from .query import parse_question
from .records import Answer, AbstentionReason, GenerationConfig


def answer_question(
    question: str,
    retriever: Retriever,
    config: GenerationConfig | None = None,
    *,
    query: Query | None = None,
    min_score: float | None = None,
    llm: Any | None = None,
    max_tokens: int = MAX_OUTPUT_TOKENS,
    on_token: Callable[[str], None] | None = None,
    facts_file: Path = FACTS_FILE,
    use_facts: bool = True,
) -> Answer:
    """The shared entry point for the app and evaluation harness.

    A supplied Query overrides automatic filters. ``min_score`` is an optional
    additional floor on this retriever's final scores (inclusive, like rank()).
    Existing thresholds inside the retriever still apply. None adds no floor;
    choose a value on the selected method's scale using benchmark calibration.

    A numeric question is offered to the facts store first (#34): the figure is
    looked up and the passage printing it is cited, with no model involved. The
    lookup returns nothing unless it can fully support the answer, and the
    question then takes the retrieval path below exactly as it otherwise would,
    so this is a shortcut and never a second way to fail. ``use_facts=False``
    turns it off, which is what the ablation matrix needs to measure it.

    Empty retrieval never builds a prompt or calls a model. Built-in indexes
    distinguish empty filters from rejected scores using metadata, without
    relaxing the actual search. A custom retriever may implement
    ``has_candidates(query)``; otherwise an empty result has an unknown cause.
    Provider/index failures propagate as errors, not successful abstentions.
    """
    if not question or not question.strip():
        raise ValueError("question must not be blank")
    if min_score is not None and not isfinite(min_score):
        raise ValueError("min_score must be finite or None")
    parsed = parse_question(question)
    query = query if query is not None else parsed.to_query(top_k=FINAL_K)
    if query.top_k < 1:
        raise ValueError("top_k must be positive when answering a question")
    config = config if config is not None else config_from_env()

    # The Query's filters rather than the parse's, so a caller that narrowed the
    # search by hand gets the figure for the company and year it asked about.
    if use_facts and parsed.question_type == "numeric":
        looked_up = answer_from_facts(
            question, query.tickers, query.fiscal_years, retriever, facts_file=facts_file,
        )
        if looked_up is not None:
            if on_token is not None:
                on_token(looked_up.text)
            return looked_up

    found = retriever.search(query)
    passages = [p for p in found if isfinite(p.score) and p.text.strip()
                and (min_score is None or p.score >= min_score)]
    if not passages:
        reason: AbstentionReason = "no_evidence"
        if found:
            if min_score is not None and all(
                isfinite(p.score) and p.score < min_score for p in found
            ):
                reason = "below_threshold"
        else:
            admitted = has_candidates(retriever, query)
            if admitted is True:
                reason = "below_threshold"
            elif admitted is False and query.filters:
                unrestricted = replace(query, tickers=(), fiscal_years=(), items=(),
                                       content_type=None, key_items_only=False)
                if has_candidates(retriever, unrestricted) is True:
                    reason = "filters_excluded_all"
        answer = Answer(question=question, text=ABSTAIN_PHRASE, citations=(),
                        passages=(), abstained=True, config=config, latency_ms=0.0,
                        abstention_reason=reason)
        if on_token is not None:
            on_token(answer.text)
        return answer

    prompt = build_prompt(question, passages)
    generation = generate(prompt, config, llm=llm, max_tokens=max_tokens, on_token=on_token)
    return resolve_citations(question, generation, prompt.passages)
