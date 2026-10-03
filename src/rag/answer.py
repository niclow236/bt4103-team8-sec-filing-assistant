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
from .constants import ABSTAIN_PHRASE, MAX_OUTPUT_TOKENS, UnanswerableBecause
from .decompose import search_decomposed
from .generate import config_from_env, generate
from .numeric import answer_from_facts
from .prompt import build_prompt
from .query import ParsedQuestion, parse_question
from .records import Answer, AbstentionReason, GenerationConfig

# What an unanswerable question is refused with, by why it is unanswerable
# (``constants.UNANSWERABLE_BECAUSE``). "topic" and "year" are not here: those
# are searched, see answer_question.
REFUSALS: dict[UnanswerableBecause, AbstentionReason] = {
    "request": "beyond_the_filings",
    "company": "company_not_in_corpus",
}


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
    use_decomposition: bool = True,
    use_refusal: bool = True,
    parsed: ParsedQuestion | None = None,
) -> Answer:
    """The shared entry point for the app and evaluation harness.

    A question the parser reads as asking for advice or a prediction, or for
    a figure of a company the corpus holds no filings for, is refused before
    anything is searched. Searched, the first kind gets sixteen passages about
    the company it names and the second sixteen about other companies, and a
    model may then answer from them: asked "Is Meta a good investment?" it
    described Meta's spending plans. The refusal is an abstention with its own
    reason.

    Every other question is searched, the rest of what the parser reads as
    unanswerable included, because a filing may answer it: one about a share
    price, next year or a company outside the corpus (a 10-K prints the price
    it paid for its own shares and the obligations due next year, and names
    its competitors), and one naming only a fiscal year outside the corpus
    (a filing prints the two years before its own). So is a question whose
    Query names a company: the caller chose what to search.
    ``use_refusal=False`` searches everything, which is the without half of
    that comparison.

    A supplied Query overrides automatic filters. ``min_score`` is an optional
    additional floor on this retriever's final scores (inclusive, like rank()).
    Existing thresholds inside the retriever still apply. None adds no floor;
    choose a value on the selected method's scale using benchmark calibration.

    A numeric question is offered to the facts store first (#34): the figure is
    looked up and the passage printing it is cited, with no model involved. The
    lookup returns nothing unless it can fully support the answer, and the
    question then takes the retrieval path below exactly as it otherwise would,
    so this is a shortcut and never a second way to fail. It cites a passage
    only on the same terms this function admits one, ``min_score`` included, so
    it cannot answer where the retrieval path would abstain. ``use_facts=False``
    turns it off, which is what the ablation matrix needs to measure it.

    A question naming more than one company or more than one year is searched
    once per filing and the results interleaved (#35), so the passage budget is
    shared between the filings rather than won by whichever one phrases the
    topic most like the question. ``Answer.sub_questions`` records what it was
    split into, and is empty where it was not.
    ``use_decomposition=False`` is the without half of that ablation.

    ``parsed`` is the question already read, for a caller that has one -- the
    harness builds its Query from one -- so that reading it twice is a choice
    rather than the only option. It has to be a reading of this question, as
    ``verify_answer`` and the app's sidebar require of theirs: the reading
    decides the refusal and the route, so one left over from another question
    would refuse this one unsearched, or look up a figure it never asked for.

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
    parsed = parsed if parsed is not None else parse_question(question, facts_file=facts_file)
    # parse_question strips, so a question with space around it matches its own reading.
    if parsed.question != question.strip():
        raise ValueError("parsed question must match the question")
    query = query if query is not None else parsed.to_query(top_k=FINAL_K)
    if query.top_k < 1:
        raise ValueError("top_k must be positive when answering a question")
    config = config if config is not None else config_from_env()

    because = parsed.unanswerable_because if use_refusal else None
    if because == "company" and query.tickers:
        # The caller chose a company to search by hand, as the app's sidebar
        # lets a user do, and its filings may well mention the one named.
        because = None
    refusal = REFUSALS.get(because)
    if refusal is not None:
        answer = Answer(question=question, text=ABSTAIN_PHRASE, citations=(), passages=(),
                        abstained=True, config=config, latency_ms=0.0,
                        abstention_reason=refusal)
        if on_token is not None:
            on_token(answer.text)
        return answer

    # The Query's filters rather than the parse's, so a caller that narrowed the
    # search by hand gets the figure for the company and year it asked about.
    # The facts route searches its own supporting passages and does not accept
    # Item filters. Let normal retrieval enforce an explicit sidebar Item.
    if use_facts and parsed.question_type == "numeric" and not query.items:
        looked_up = answer_from_facts(
            question, query.tickers, query.fiscal_years, retriever,
            facts_file=facts_file, min_score=min_score,
        )
        if looked_up is not None:
            if on_token is not None:
                on_token(looked_up.text)
            return looked_up

    decomposition = search_decomposed(query, retriever) if use_decomposition else None
    found = list(decomposition.passages) if decomposition is not None else retriever.search(query)
    sub_questions = decomposition.labels if decomposition is not None else ()
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
                                       content_type=None)
                if has_candidates(retriever, unrestricted) is True:
                    reason = "filters_excluded_all"
        answer = Answer(question=question, text=ABSTAIN_PHRASE, citations=(),
                        passages=(), abstained=True, config=config, latency_ms=0.0,
                        abstention_reason=reason, sub_questions=sub_questions)
        if on_token is not None:
            on_token(answer.text)
        return answer

    prompt = build_prompt(question, passages)
    generation = generate(prompt, config, llm=llm, max_tokens=max_tokens, on_token=on_token)
    answered = resolve_citations(question, generation, prompt.passages)
    return replace(answered, sub_questions=sub_questions) if sub_questions else answered
