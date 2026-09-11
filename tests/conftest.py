"""Shared fixtures: a small synthetic corpus, and an encoder that needs no model.

The tests run on a fresh clone, so nothing here reads ``data/``. The corpus is
written through the pipeline's own records and ``write_chunks``, so it has
exactly the shape ``iter_chunks`` reads. The encoder is deterministic and
instant: a passage's vector is seeded from its text, which is enough to tell
whether a vector was re-encoded without paying for bge.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.pipeline.chunk import write_chunks
from src.pipeline.records import ChunkedFiling, ChunkRecord
from src.retrieval import embed

# Three companies, two fiscal years each.
COMPANIES = {"AAA": "Alpha Corp", "BBB": "Beta Inc.", "CCC": "Gamma Holdings"}
YEARS = (2023, 2024)


def _passages(ticker: str, year: int, accession: str) -> list[ChunkRecord]:
    """Prose from three Items and a table from Item 8, all distinct text."""
    chunks = []
    items = [("1", "Business", "part_i_item_1"), ("1A", "Risk Factors", "part_i_item_1a"),
             ("7", "Management's Discussion", "part_ii_item_7")]
    for item, title, section in items:
        for index in range(4):
            text = (f"{ticker} {year} {title} paragraph {index}. The company discusses "
                    f"topic {index} of Item {item} in fiscal {year} at some length.")
            chunks.append(ChunkRecord(
                chunk_id=f"{accession}_{section}_{index}", section_id=section, part=None,
                item=item, title=title, heading=f"Heading {index}" if index % 2 else None,
                text=text, n_chars=len(text), chunk_index=index,
                is_key_section=item in {"1", "1A", "7"},
            ))
    for index in range(3):
        text = (f"Revenue by segment (part {index + 1} of 3)\n\n| Segment | {year} |\n"
                f"| --- | --- |\n| {ticker} segment {index} | {1000 + index:,} |")
        chunks.append(ChunkRecord(
            chunk_id=f"{accession}_part_ii_item_8_t000_{index:02d}",
            section_id="part_ii_item_8", part=None, item="8", title="Financial Statements",
            heading="Revenue by segment", text=text, n_chars=len(text), chunk_index=index,
            is_key_section=True, content_type="table", table_index=0,
            table_caption="Revenue by segment",
        ))
    return chunks


def write_corpus(processed_dir: Path) -> Path:
    for number, (ticker, company) in enumerate(COMPANIES.items(), start=1):
        for year in YEARS:
            accession = f"000000000{number}-{year % 100}-000001"
            write_chunks(
                ChunkedFiling(
                    ticker=ticker, cik=number, company=company, form="10-K",
                    filing_date=f"{year + 1}-02-01", accession_no=accession,
                    url=f"https://example.test/{accession}",
                    source_path=f"data/interim/{ticker}/{accession}.json",
                    chunks=_passages(ticker, year, accession),
                    period_of_report=f"{year}-12-31",
                    chunk_budget=1800, chunk_overlap=300,
                ),
                processed_dir / ticker / f"10-K_{year + 1}-02-01_{accession}.json",
            )
    return processed_dir


@pytest.fixture
def corpus(tmp_path) -> Path:
    """A processed corpus of 6 filings and 90 passages, in a temporary directory."""
    return write_corpus(tmp_path / "processed")


def edit_filing(path: Path, change) -> None:
    """Load one processed file, apply ``change`` to its dict, write it back."""
    data = json.loads(path.read_text(encoding="utf-8"))
    change(data)
    path.write_text(json.dumps(data), encoding="utf-8")


def vector_for(text: str) -> np.ndarray:
    """The fake encoder's vector for a text: unit length, seeded from the text."""
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).standard_normal(embed.EMBED_DIMENSIONS).astype(np.float32)
    return vector / np.linalg.norm(vector)


class FakeTokenizer:
    """Ten tokens a passage, or 600 where the text asks to be truncated."""

    def __call__(self, texts):
        return {"input_ids": [[0] * (600 if "LONGLONG" in text else 10) for text in texts]}


class FakeModel:
    """Stands in for SentenceTransformer: the same two members build() uses."""

    tokenizer = FakeTokenizer()

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def encode(self, texts, batch_size, normalize_embeddings, show_progress_bar):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated kill")
        return np.stack([vector_for(text) for text in texts])


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the encoder for the test; the model it returns can be told to fail."""
    model = FakeModel()
    monkeypatch.setattr(embed, "_load_model", lambda threads=None: (model, 1))
    return model


@pytest.fixture(autouse=True)
def _release_chroma():
    """Drop Chroma's cached clients after each test, so temp dirs can be removed."""
    yield
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass
