"""RAG stage: the grounded prompt, generation, citations, and the records they fill.

``records.py`` holds the shapes the stage passes out: ``Answer``,
``Citation``, ``GenerationConfig`` and ``Generation``, plus ``GroundedAnswer``,
the Pydantic model a model's output is constrained to. Nothing in it imports
the code that builds them, so the app and the evaluation harness can read an
answer without pulling in a model client.

``query.py`` is the way in: it reads a question into a ``ParsedQuestion`` and
builds the retrieval ``Query`` from it, with the companies and fiscal years
resolved against the corpus's scope and exposed so the app can show them back.

``prompt.py`` renders the grounded prompt: the retrieved passages as numbered
sources, the rules, and the question, as a ``GroundedPrompt`` the generator
hands to the model and whose ``passages`` go onto the ``Answer`` unchanged.

``generate.py`` runs that prompt through the configured model, a local one
served by Ollama or one on Mistral's API, with the answer's JSON schema as the
output format, streaming the answer as prose and returning a ``Generation``
with the parsed answer, the latency and the token counts.

``citations.py`` resolves a completed generation against ``prompt.passages``,
removes invented markers from displayed text, and attaches sentence warnings.
It renders citation labels exclusively from the passages' stored metadata.

``numeric.py`` answers a numeric question from the XBRL facts store instead,
looking the figure up and citing the passage that prints it, with no model in
between. ``answer.py`` offers every numeric question to it first and falls
back to retrieval and generation whenever it cannot fully support an answer.

``decompose.py`` splits a question that names several companies or years into
one search per filing and interleaves the results, so the passage budget is
shared between the filings rather than won by whichever one phrases the topic
most like the question, and records which sub-question found each passage.
"""

from .answer import answer_question
from .citations import render_citation, resolve_citations
from .decompose import Decomposition, SubQuestion, decompose, search_decomposed
from .numeric import Fact, answer_from_facts, find_metric, lookup_fact
from .generate import (
    ProviderBusy,
    ProviderUnavailable,
    chat_model,
    check_provider,
    config_from_env,
    generate,
    stream,
)
from .prompt import GroundedPrompt, build_prompt, render_source
from .query import ParsedQuestion, parse_question
from .records import (
    Answer,
    Citation,
    CitedSentence,
    Generation,
    GenerationConfig,
    GroundedAnswer,
    SentenceCitations,
    VerificationCheck,
    VerificationResult,
)
from .verify import verify_answer, worst_check

__all__ = [
    "Answer",
    "Citation",
    "CitedSentence",
    "Decomposition",
    "Fact",
    "Generation",
    "GenerationConfig",
    "GroundedAnswer",
    "GroundedPrompt",
    "ParsedQuestion",
    "ProviderBusy",
    "ProviderUnavailable",
    "SentenceCitations",
    "VerificationCheck",
    "VerificationResult",
    "SubQuestion",
    "answer_from_facts",
    "answer_question",
    "build_prompt",
    "decompose",
    "search_decomposed",
    "chat_model",
    "check_provider",
    "config_from_env",
    "find_metric",
    "generate",
    "lookup_fact",
    "parse_question",
    "render_citation",
    "render_source",
    "resolve_citations",
    "stream",
    "verify_answer",
    "worst_check",
]
