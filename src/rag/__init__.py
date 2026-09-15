"""RAG stage: the grounded prompt, generation, citations, and the records they fill.

``records.py`` holds the shapes the stage passes out -- ``Answer``,
``Citation`` and ``GenerationConfig`` -- and nothing in it imports the code
that builds them, so the app and the evaluation harness can read an answer
without pulling in a provider client.
"""

from .records import Answer, Citation, GenerationConfig

__all__ = ["Answer", "Citation", "GenerationConfig"]
