"""The data records the retrieval stage passes to whoever asked for a passage.

Kept apart from the retrievers that build them, the same way
``src/pipeline/records.py`` is kept apart from the stages, so that BM25, dense
retrieval, hybrid fusion, reranking, the RAG engine and the evaluation harness
can all agree on one shape without importing each other. A retriever imports
this module; nothing here imports a retriever.

Three records, and each answers a question the stage would otherwise answer
four times over:

``RetrievedPassage`` is what every retriever returns. It carries the identity a
citation needs -- company, fiscal year, Item, URL -- alongside the text, so the
answer can be traced to a filing without a second lookup, and it carries
``retriever``, so a result set can be attributed to the method that produced it
when the ablation matrix compares them.

``IndexManifest`` is what a built index says about itself, so a retriever can
refuse an index that no longer matches the corpus on disk rather than quietly
returning results from an older one.

``Query`` is one request: the question text and the filters that must hold
before scoring starts. The corpus is deliberately near-duplicate -- one
industry, fifteen peers, five years -- so a filter is not a convenience, it is
what stops Apple's FY2023 risk factors answering a question about FY2024.

Every record is frozen and holds no mutable default, so a passage handed to
three components cannot be edited by one of them behind the others' backs.

Two functions sit here as well, and for the same reason the records do: they
compute fields of ``IndexManifest``, so every index that writes one writes it
the same way. A manifest field defined once in each retriever is a field with no
definition at all, which is what ``corpus_fingerprint`` was before it moved
here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..config import PROJECT_ROOT


def corpus_fingerprint(
    rows: Iterable[Mapping[str, Any]],
    text_of: Callable[[Mapping[str, Any]], str] | None = None,
) -> str:
    """A digest identifying the exact set of passages an index was built from.

    Here rather than in a retriever because it populates
    :attr:`IndexManifest.corpus_fingerprint`, and a field with two definitions
    has none: written once in ``bm25.py`` and once in ``embed.py``, the two
    hashed the same corpus to different strings, so the manifests of a
    bm25/dense pair could never agree no matter how many times either was
    rebuilt.

    Each passage's ``chunk_id`` is hashed together with its text, those digests
    are sorted, and the sorted list is hashed. Sorting is what makes it a
    fingerprint of the corpus rather than of one walk over it: the same passages
    in a different order, which is all a renamed directory or a changed glob
    amounts to, give the same answer.

    ``text_of`` is what an index counts as the passage's content, defaulting to
    the stored text. A retriever that feeds the encoder something other than the
    stored text passes its own -- the dense index prepends a context header
    built from metadata, so for that index a changed company name really does
    move a vector, and the digest has to see it. That makes the digests of two
    different index types incomparable by construction, which is why
    :attr:`IndexManifest.corpus_fingerprint` is only ever compared against
    another manifest of the same ``index_type``.

    It changes when a passage's text changes, when passages are added or
    removed, and when the chunker cuts the same filing differently, since that
    renumbers ``chunk_id``.
    """
    content = text_of if text_of is not None else lambda row: row["text"]
    digests = sorted(
        hashlib.sha256(
            f"{row['chunk_id']}\x00{content(row)}".encode("utf-8")
        ).hexdigest()
        for row in rows
    )
    total = hashlib.sha256()
    for digest in digests:
        total.update(digest.encode("ascii"))
    return total.hexdigest()


def manifest_path(index_path: Path) -> str:
    """The index location as a manifest records it: project-relative posix.

    An absolute path is machine-local, so a manifest carrying one says nothing
    useful to the next person to read it out of a shared index. A directory
    outside the project root -- pytest's ``tmp_path``, a scratch build on
    another volume -- has no relative form, so it is recorded absolute rather
    than raising, since failing here would throw away a whole build's manifest
    after the build had already finished.
    """
    try:
        return index_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return index_path.as_posix()


@dataclass(frozen=True)
class RetrievedPassage:
    """One passage returned by one retriever for one query.

    This is a projection of a stored chunk, not the chunk itself: it holds what
    a ranked result and its citation need, and drops what only the chunker
    cares about. Build it with :meth:`from_chunk` rather than by hand, so the
    projection is written once.
    """

    chunk_id: str         # as the chunker assigned it, so results join back to the corpus
    text: str             # the passage as stored, never with a context header prepended
    score: float          # the retriever's own scale; comparable within a method, not across
    rank: int             # position in this result set, 1 being the best
    # Which method produced this: "bm25", "dense", "hybrid", "rerank". Carried on
    # the passage rather than tracked alongside it, because the evaluation
    # harness runs several methods over the same question and has to attribute
    # every passage it is handed without threading extra state through the call.
    retriever: str
    ticker: str
    company: str
    fiscal_year: int | None   # the year the filing REPORTS on, not the year it was filed
    item: str | None          # "1", "1A", "7A", ...; None for a section outside the Items
    title: str                # the official Item title, for the citation
    url: str                  # the filing on EDGAR, so a citation can be opened
    # "prose" or "table". A numeric question can weight or restrict to table
    # passages, which are the ones that keep figures under their row and column
    # labels.
    content_type: str = "prose"
    # For a fused result, the methods that surfaced this passage before fusion,
    # in the order they were fused. A tuple rather than a list so the record
    # stays immutable, and empty for a single-method result, where ``retriever``
    # already says everything there is to say.
    sources: tuple[str, ...] = ()

    @classmethod
    def from_chunk(
        cls,
        chunk: Mapping[str, Any],
        score: float,
        rank: int,
        retriever: str,
        sources: tuple[str, ...] = (),
    ) -> RetrievedPassage:
        """Build a result from one row of ``pipeline.chunk.iter_chunks()``.

        That row is what every retriever indexes and scores, and it already
        carries the filing identity joined onto the passage. Converting it here
        means BM25, dense and hybrid do not each write their own version of the
        same field mapping, and a field added to the corpus reaches all three by
        being added once.
        """
        return cls(
            chunk_id=chunk["chunk_id"],
            text=chunk["text"],
            score=score,
            rank=rank,
            retriever=retriever,
            ticker=chunk["ticker"],
            company=chunk["company"],
            fiscal_year=chunk["fiscal_year"],
            item=chunk["item"],
            title=chunk["title"],
            url=chunk["url"],
            content_type=chunk.get("content_type", "prose"),
            sources=sources,
        )


@dataclass(frozen=True)
class IndexManifest:
    """What one built index is a copy of, written beside it as JSON.

    An index is the corpus in another form, and the corpus moves: the chunk
    stage takes ``--budget`` and ``--overlap``, and companies and fiscal years
    are added to the scope. Re-chunk without re-indexing and the retrievers
    still answer, from passages that no longer exist, which is the failure mode
    the download manifest was written to prevent one stage earlier.

    So each index records what it was built from and what built it, and a
    retriever compares that against the corpus it finds on disk before it will
    load one. :meth:`mismatches` is what turns a refusal into an error that says
    which of the two moved.
    """

    index_type: str           # "bm25" or "dense"
    path: str                 # relative to the project root, posix-style, so this is portable
    # A digest over the passages that went in, from :func:`corpus_fingerprint`.
    # Two indexes OF THE SAME TYPE with the same fingerprint were built from the
    # same corpus; a different one means the chunker has been re-run since.
    # Across types it is not comparable: each index hashes what it actually
    # indexed, and the dense index encodes a context header the sparse one never
    # sees, so a bm25 and a dense manifest of one corpus hold different strings
    # by design. Compare a manifest against a freshly computed digest using the
    # same ``text_of``, never against a manifest of the other type.
    corpus_fingerprint: str
    n_passages: int
    n_filings: int
    built_at: str             # ISO-8601, so a stale index can be dated
    # The embedding model, exactly as it is loaded, since two checkpoints of one
    # family produce vectors that are not comparable. Empty for BM25, which has
    # no model.
    model: str = ""
    dimensions: int | None = None   # vector width, None for BM25
    # The chunker settings the corpus was cut with. The fingerprint already
    # catches a change; these say what changed, and they are what the chunk-size
    # sweep reports its rows against.
    chunk_budget: int | None = None
    chunk_overlap: int | None = None

    def mismatches(
        self,
        corpus_fingerprint: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
    ) -> list[str]:
        """Describe every way this index disagrees with what the caller expects.

        Returns one line per disagreement, naming both sides, and an empty list
        when the index is safe to load. Arguments left as ``None`` are not
        checked, so a retriever with no model to compare passes only what it
        knows.
        """
        found: list[str] = []
        if corpus_fingerprint is not None and corpus_fingerprint != self.corpus_fingerprint:
            found.append(
                f"corpus fingerprint: index built from {self.corpus_fingerprint}, "
                f"corpus on disk is {corpus_fingerprint}"
            )
        if model is not None and model != self.model:
            found.append(f"model: index built with {self.model or 'none'}, asked for {model}")
        if dimensions is not None and dimensions != self.dimensions:
            found.append(f"dimensions: index holds {self.dimensions}, asked for {dimensions}")
        return found


@dataclass(frozen=True)
class Query:
    """One retrieval request: what to look for, and where it is allowed to look.

    The filters are a hard pre-filter rather than a hint. Every retriever cuts
    the corpus down to what they allow and then scores inside it, because this
    corpus is fifteen peers in one industry over five years, and Apple's FY2024
    risk factors and its FY2023 risk factors sit almost on top of each other in
    embedding space. Filtering after scoring returns the right company in the
    wrong year, and reads as confident and correct.

    The fields are tuples rather than lists so an instance cannot be edited
    after it is passed on, and empty means unrestricted rather than nothing.
    They are named for the same things ``iter_chunks`` filters on, so a query is
    read against the corpus without a translation step.
    """

    text: str
    top_k: int = 10
    tickers: tuple[str, ...] = ()
    fiscal_years: tuple[int, ...] = ()
    items: tuple[str, ...] = ()       # "1A", "7", ...; matched case-insensitively
    content_type: str | None = None   # "prose" or "table"; None allows both
    key_items_only: bool = False      # the Items the project leans on: 1, 1A, 7, 7A, 8

    @property
    def filters(self) -> dict[str, Any]:
        """The active filters as a mapping, for a backend that takes one.

        Chroma and the like want a dict of field to allowed values, and the
        keys here are the chunk fields they apply to. Only the filters that were
        actually set appear, so an empty mapping means an unrestricted search
        rather than one that matches nothing.
        """
        active: dict[str, Any] = {}
        if self.tickers:
            active["ticker"] = list(self.tickers)
        if self.fiscal_years:
            active["fiscal_year"] = list(self.fiscal_years)
        if self.items:
            active["item"] = [item.upper() for item in self.items]
        if self.content_type:
            active["content_type"] = self.content_type
        if self.key_items_only:
            active["is_key_section"] = True
        return active
