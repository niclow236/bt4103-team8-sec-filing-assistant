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

``generate.py`` runs that prompt through a local model served by Ollama, using
LangChain's ``ChatOllama`` with the answer's JSON schema as the output format,
streaming the answer as prose and returning a ``Generation`` with the parsed
answer, the latency and the token counts.

``citations.py`` resolves a completed generation against ``prompt.passages``,
removes invented markers from displayed text, and attaches sentence warnings.
It renders citation labels exclusively from the passages' stored metadata.
"""

from .answer import answer_question
from .citations import render_citation, resolve_citations
from .generate import (
    ProviderUnavailable,
    chat_model,
    config_from_env,
    generate,
    stream,
)
from .prompt import GroundedPrompt, build_prompt, render_source
from .query import ParsedQuestion, build_query, parse_question
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
from .verify import record_verification, verify_answer

__all__ = [
    "Answer",
    "Citation",
    "CitedSentence",
    "Generation",
    "GenerationConfig",
    "GroundedAnswer",
    "GroundedPrompt",
    "ParsedQuestion",
    "ProviderUnavailable",
    "SentenceCitations",
    "VerificationCheck",
    "VerificationResult",
    "build_prompt",
    "answer_question",
    "build_query",
    "chat_model",
    "config_from_env",
    "generate",
    "parse_question",
    "record_verification",
    "render_citation",
    "render_source",
    "resolve_citations",
    "stream",
    "verify_answer",
]
