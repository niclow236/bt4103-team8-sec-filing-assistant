"""Small answer records shared by the Streamlit tests."""

from dataclasses import replace

from src.rag.citations import resolve_citations
from src.rag.constants import PROMPT_TEMPLATE_ID
from src.rag.records import (
    Answer,
    CitedSentence,
    Generation,
    GenerationConfig,
    GroundedAnswer,
    VerificationCheck,
    VerificationResult,
)
from src.retrieval.records import RetrievedPassage


def sample_answer(question: str) -> Answer:
    """Return an answer covering supported, unresolved, mismatched and uncited claims."""
    passage = RetrievedPassage(
        chunk_id="test-passage", text="Example passage: Revenue was $5 billion.\n"
        "This is synthetic evidence used only by the test suite.",
        score=1, rank=1, retriever="test", ticker="AAPL", company="Apple (test)",
        fiscal_year=2024, item="7", title="Management's Discussion and Analysis",
        url="https://www.sec.gov/edgar/search/", cik=320193, form="10-K",
        part="II", filing_date="2024-11-01",
    )
    structured = GroundedAnswer(answerable=True, sentences=(
        CitedSentence(text="Example revenue was $5 billion.", sources=(1,)),
        CitedSentence(text="This claim refers to an unavailable source.", sources=(9,)),
        CitedSentence(text="Example revenue was $8 billion.", sources=(1,)),
        CitedSentence(text="This claim has no citation.", sources=()),
    ))
    generation = Generation(
        text=structured.render(), answer=structured, raw=structured.model_dump_json(),
        config=GenerationConfig("ollama", "test", PROMPT_TEMPLATE_ID), latency_ms=0,
        input_tokens=None, output_tokens=None, stop_reason="stop",
    )
    answer = resolve_citations(question, generation, (passage,))
    return replace(answer, verification=VerificationResult("numeric", (
        VerificationCheck("passage", "supported", 0, "Example revenue", "Matches test passage."),
        VerificationCheck("passage", "mismatch", 2, "Example revenue",
                          "The test passage says $5 billion, not $8 billion."),
    )))
