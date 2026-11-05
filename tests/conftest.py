"""Shared fixtures: a small synthetic corpus, an encoder that needs no model,
and settings that do not come from the project's .env.

The tests run on a fresh clone, so nothing here reads ``data/`` or ``.env``. The
corpus is written through the pipeline's own records and ``write_chunks``, so it
has exactly the shape ``iter_chunks`` reads. The encoder is deterministic and
instant: a passage's vector is seeded from its text, which is enough to tell
whether a vector was re-encoded without paying for bge.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src import config
from src.pipeline.chunk import write_chunks
from src.pipeline.records import ChunkedFiling, ChunkRecord
from src.rag.constants import (
    LLM_BASE_URL_ENV,
    LLM_MODEL_ENV,
    LLM_NUM_GPU_ENV,
    LLM_PROVIDER_ENV,
    MISTRAL_API_KEY_ENV,
    MISTRAL_BASE_URL_ENV,
    SOCLAAS_API_KEY_ENV,
    SOCLAAS_BASE_URL_ENV,
)
from src.rag.generate import _mistral_client, _ollama_client, _soclaas_client
from src.retrieval import embed
from src.retrieval.records import RetrievedPassage

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
    """Stands in for SentenceTransformer: the same two members build() uses.

    ``encode`` defaults everything but the texts, the way the real one does, so
    a caller that encodes a single query without naming a batch size -- the
    dense retriever -- reaches the same signature the builder does.
    """

    tokenizer = FakeTokenizer()

    def __init__(self, fail_after: int | None = None):
        self.calls = 0
        self.fail_after = fail_after

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise KeyboardInterrupt("simulated kill")
        return np.stack([vector_for(text) for text in texts])


@pytest.fixture
def fake_model(monkeypatch):
    """Replace the encoder for the test; the model it returns can be told to fail."""
    model = FakeModel()
    monkeypatch.setattr(embed, "_load_model", lambda threads=None, **_: (model, 1))
    return model


# The settings that choose and reach the model that answers, read from the
# environment and the project's .env.
LLM_SETTINGS = (LLM_PROVIDER_ENV, LLM_MODEL_ENV, LLM_BASE_URL_ENV, LLM_NUM_GPU_ENV,
                MISTRAL_API_KEY_ENV, MISTRAL_BASE_URL_ENV, SOCLAAS_API_KEY_ENV,
                SOCLAAS_BASE_URL_ENV)


@pytest.fixture(autouse=True)
def _no_local_settings(monkeypatch):
    """Every test starts from the defaults, whatever a teammate's .env or shell sets.

    The project's own .env is never loaded, so a typo in it fails no test, and
    the LLM settings are removed from the environment. A test about reading
    .env writes its own and passes its path, which is still read, and a test
    that sets a variable sets it itself. load_dotenv writes into the real
    os.environ, which monkeypatch does not see, so the settings a test loads
    are removed again afterwards. The chat clients a test builds are dropped
    too, so no later test is handed one back.
    """
    load_dotenv = config.load_dotenv
    monkeypatch.setattr(config, "load_dotenv", lambda path, *args, **kwargs:
                        Path(path) != config.ENV_FILE and load_dotenv(path, *args, **kwargs))
    for name in LLM_SETTINGS:
        monkeypatch.delenv(name, raising=False)
    _mistral_client.cache_clear()
    _ollama_client.cache_clear()
    _soclaas_client.cache_clear()
    yield
    for name in LLM_SETTINGS:
        os.environ.pop(name, None)
    _mistral_client.cache_clear()
    _ollama_client.cache_clear()
    _soclaas_client.cache_clear()


def _tells_case_apart(folder):
    (folder / "probe").mkdir()
    return not (folder / "PROBE").exists()


def case_insensitive(folder, tells_case_apart=_tells_case_apart):
    """``folder``, once it is found not to tell MATRIX from matrix.

    Only such a folder gives one folder two spellings that ``realpath`` leaves
    apart; a symlink does not, since ``realpath`` resolves it, so a test of a
    guard that must treat both spellings as one folder needs it: the results
    report's case-variant ``--out`` guard, and the check that git found the
    project's own repository (#50). A Mac's default
    filesystem is case-insensitive and a Linux one is not. On a laptop with no
    such folder the test skips. In CI it fails instead: there a skip would report
    ``1 skipped`` and leave the guard unchecked (#201). ``CI`` tells the two
    apart, as for the tokenizer test in test_verify.py.
    """
    if tells_case_apart(folder):
        if os.environ.get("CI"):
            raise AssertionError(f"{folder} tells MATRIX from matrix in CI, so a "
                                 "case-variant path guard went unchecked")
        pytest.skip(f"{folder} tells MATRIX from matrix")
    return folder


@pytest.fixture
def case_insensitive_tmp(tmp_path):
    """A fresh case-insensitive folder: under CASE_INSENSITIVE_DIR when it is set,
    as the CI workflow sets it to a casefolded mount, otherwise tmp_path."""
    shared = os.environ.get("CASE_INSENSITIVE_DIR")
    if not shared:
        yield case_insensitive(tmp_path)
        return
    folder = Path(tempfile.mkdtemp(dir=shared))
    try:
        yield case_insensitive(folder)
    finally:
        shutil.rmtree(folder)


class Checkout:
    """A git repository of one committed file, for a run to read its source from."""

    def __init__(self, root: Path):
        self.root = root
        self.git("init", "-q")
        (root / "code.py").write_text("measured = True\n")
        self.git("add", "code.py")
        self.git("commit", "-q", "-m", "measured code")

    def git(self, *args: str) -> str:
        return subprocess.run(["git", *args], cwd=self.root, check=True, capture_output=True,
                              text=True).stdout.strip()

    @property
    def head(self) -> str:
        return self.git("rev-parse", "HEAD")

    def tag(self, name: str) -> None:
        self.git("tag", "-a", name, "-m", name)

    def commit(self) -> None:
        """Commit a change on top, so the checkout is no longer at its tag."""
        (self.root / "code.py").write_text("measured = 'later'\n")
        self.git("commit", "-q", "-am", "later code")

    def edit(self) -> None:
        """Leave a committed file changed, as a run started mid-edit would."""
        (self.root / "code.py").write_text("measured = False\n")


@pytest.fixture
def checkout(tmp_path_factory, monkeypatch) -> Checkout:
    """Point every git call at a fresh repository instead of this clone (#50).

    A run reads its release tag from git, and the clone the tests run in has
    whatever tags its owner made, so git is sent here through GIT_DIR and
    GIT_WORK_TREE. The owner's own git settings are left out, so commit
    signing or hooks set there do not run on these commits.
    """
    root = tmp_path_factory.mktemp("checkout")
    for name, value in {
        "GIT_DIR": str(root / ".git"), "GIT_WORK_TREE": str(root),
        "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_AUTHOR_NAME": "test", "GIT_AUTHOR_EMAIL": "test@example.com",
        "GIT_COMMITTER_NAME": "test", "GIT_COMMITTER_EMAIL": "test@example.com",
    }.items():
        monkeypatch.setenv(name, value)
    return Checkout(root)


@pytest.fixture(autouse=True)
def _release_chroma():
    """Drop Chroma's cached clients after each test, so temp dirs can be removed."""
    yield
    try:
        import chromadb
        chromadb.api.client.SharedSystemClient.clear_system_cache()
    except Exception:
        pass


# Shared deterministic doubles for paired reranking tests.
def rerank_passage(cid, score=1, content_type="prose", ticker="AAA", year=2024):
    return RetrievedPassage(cid, "Evidence " + cid, score, 1, "hybrid", ticker,
                            "Alpha Corp", year, "8", "Statements", "https://example.test",
                            content_type, ("bm25", "dense"))


class RerankModel:
    device = "test"

    def __init__(self, scores, lengths=None):
        self.scores = scores
        self.lengths = lengths
        self.calls = []

    def tokenizer(self, queries, texts, **kwargs):
        self.calls.append(("tokens", queries, texts, kwargs))
        return {"length": self.lengths or [25] * len(texts)}

    def predict(self, pairs, **kwargs):
        self.calls.append(("predict", pairs, kwargs))
        return self.scores


class RerankInner:
    name = "hybrid"

    def __init__(self, passages):
        self.passages = passages
        self.calls = []

    def search(self, query, k=None):
        self.calls.append((query, k))
        return self.passages[:k]

    def has_candidates(self, query):
        return bool(self.passages)
