"""RAG stage: the grounded prompt, generation, citations, and the records they fill.

``records.py`` holds the shapes the stage passes out -- ``Answer``,
``Citation`` and ``GenerationConfig`` -- and nothing in it imports the code
that builds them, so the app and the evaluation harness can read an answer
without pulling in a provider client.

``query.py`` is the way in: it reads a question into a ``ParsedQuestion`` and
builds the retrieval ``Query`` from it, with the companies and fiscal years
resolved against the corpus's scope and exposed so the app can show them back.

``prompt.py`` renders the grounded prompt: the retrieved passages as numbered
sources, the rules, and the question, as a ``GroundedPrompt`` the generator
hands to a provider and whose ``passages`` go onto the ``Answer`` unchanged.
"""

from .prompt import GroundedPrompt, build_prompt, render_source
from .query import ParsedQuestion, build_query, parse_question
from .records import Answer, Citation, GenerationConfig

__all__ = [
    "Answer",
    "Citation",
    "GenerationConfig",
    "GroundedPrompt",
    "ParsedQuestion",
    "build_prompt",
    "build_query",
    "parse_question",
    "render_source",
]
