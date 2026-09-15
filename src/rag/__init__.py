"""RAG stage: the grounded prompt, generation, citations, and the records they fill.

``records.py`` holds the shapes the stage passes out -- ``Answer``,
``Citation`` and ``GenerationConfig`` -- and nothing in it imports the code
that builds them, so the app and the evaluation harness can read an answer
without pulling in a provider client.

``query.py`` is the way in: it reads a question into a ``ParsedQuestion`` and
builds the retrieval ``Query`` from it, with the companies and fiscal years
resolved against the corpus's scope and exposed so the app can show them back.
"""

from .query import ParsedQuestion, build_query, parse_question
from .records import Answer, Citation, GenerationConfig

__all__ = [
    "Answer",
    "Citation",
    "GenerationConfig",
    "ParsedQuestion",
    "build_query",
    "parse_question",
]
