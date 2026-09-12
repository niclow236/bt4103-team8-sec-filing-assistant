"""Dense vector retrieval over the embedded filing passages.

The semantic half of the comparison BM25 is the baseline for. It answers the
questions keyword search cannot -- "what macroeconomic conditions could hurt
demand" against a filing that says "slow growth or recession, high
unemployment" and never uses the word macroeconomic -- and it does so by
searching the index ``embed.py`` built rather than by scoring the corpus itself.

Three things are worth knowing before reading further.

**It searches, it does not build.** ``embed.build`` owns the index; this module
opens it and queries it. The split matters because building is hours of encoding
and searching is milliseconds, so a retriever that could rebuild would be a
retriever that might, in the middle of an ablation.

**The filter goes into the query, not around it.** Chroma applies a ``where``
clause during the search, so a query pinned to one company compares the question
against that company's vectors only. Filtering after the search would ask for
the top k of the whole corpus and then throw most of them away -- returning
three passages where eight were asked for, and none at all for a filter the
global top k happened to miss. ``embed.where_for`` builds the clause, and
``test_embed`` pins it to ``base.matches``, so this method and BM25 search the
same corpus for the same Query.

**Cosine distance is turned back into similarity.** The collection is created
with ``DISTANCE_METRIC`` = cosine and the vectors are normalised at encode time,
so Chroma returns ``1 - cosine_similarity``: smaller is better, which is the
opposite of what every other retriever here reports and of what ``base.rank``
sorts by. Scores are converted once, on the way out of this module, so a
``RetrievedPassage`` from dense retrieval sorts and thresholds like one from
anywhere else. See :func:`similarity_of`.

What it does not do: load a model it does not need. A search needs the query
encoded, which needs bge, which needs torch -- so the model is loaded on the
first search rather than in ``load``, and a retriever that is constructed and
never queried costs nothing. The stale-index check runs in ``load``, before any
of that, because refusing an index is cheaper than loading a model to refuse it.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..config import PROCESSED_DIR
from .base import rank, resolve_k
from .constants import (
    CHROMA_DIR,
    DENSE,
    EMBED_NORMALIZE,
    MIN_DENSE_SCORE,
    QUERY_PREFIX,
)
from .embed import (
    check_index,
    chunk_from_record,
    manifest_file_for,
    open_collection,
    read_manifest,
    where_for,
)
from .records import IndexManifest, Query, RetrievedPassage


def similarity_of(distance: float) -> float:
    """Chroma's cosine distance as a similarity, which is what a score is here.

    The collection is built with cosine distance over normalised vectors, so
    ``distance = 1 - similarity`` and the best match is the smallest number.
    Every other retriever in this stage reports "higher is better", and
    ``base.rank`` sorts descending, so leaving the distance as the score would
    rank the corpus backwards -- correctly ordered, exactly inverted, and with
    nothing in the output to say so.
    """
    return 1.0 - float(distance)


class DenseRetriever:
    """Query a built dense index, refusing one the corpus has moved past.

    Satisfies ``base.Retriever`` structurally -- a ``name`` and a
    ``search(query, k=None)`` -- so the ablation harness holds this exactly as
    it holds ``BM25Retriever``, and a reranker wraps either without knowing
    which it was given.

    Build one with :meth:`load`. The constructor takes an already-open
    collection so a test can hand in its own, and does no checking of its own:
    everything that makes an index safe to search happens in ``load``.
    """

    name = DENSE

    def __init__(
        self,
        collection,
        manifest: IndexManifest | None = None,
        min_score: float | None = MIN_DENSE_SCORE,
    ) -> None:
        self.collection = collection
        self.manifest = manifest
        self.min_score = min_score
        self._model = None

    @classmethod
    def load(
        cls,
        chroma_dir: Path = CHROMA_DIR,
        processed_dir: Path = PROCESSED_DIR,
        verify: bool = True,
        min_score: float | None = MIN_DENSE_SCORE,
    ) -> DenseRetriever:
        """Open the index in ``chroma_dir``, or refuse it and say why.

        The refusal is the point of the issue this answers. Re-chunk without
        re-embedding and the vectors still answer, from passages whose ids no
        longer exist and whose text has moved on -- fluently, and with nothing
        in the result to show for it. ``embed.check_index`` compares the
        manifest against the constants, the digests the vectors carry against
        the corpus on disk, and the manifest against those digests; every
        disagreement it finds is named in the error, so the reader is told
        whether the model, the width or the corpus moved rather than being told
        only that something did.

        ``verify=False`` skips the corpus read, for a test holding a hand-built
        index. It is not what a run that produces numbers for the report wants,
        which is why it is not the default.
        """
        if verify:
            problems = check_index(chroma_dir, processed_dir)
            if problems:
                detail = "\n".join(f"  - {problem}" for problem in problems)
                raise ValueError(
                    f"Dense index at {chroma_dir} cannot be searched:\n{detail}\n"
                    f"Rebuild it: python -m src.retrieval embed"
                )

        collection = open_collection(chroma_dir, create=False)
        if collection is None:
            raise ValueError(
                f"No dense index in {chroma_dir}. Build one: python -m src.retrieval embed"
            )
        return cls(
            collection,
            manifest=read_manifest(manifest_file_for(chroma_dir)),
            min_score=min_score,
        )

    def search(self, query: Query, k: int | None = None) -> list[RetrievedPassage]:
        """The passages closest to the query, best first, inside its filters.

        The filters reach Chroma as a ``where`` clause, so they are applied
        during the search and the k that comes back is k passages the Query
        allows -- not the global top k with the disallowed ones removed.
        """
        wanted = resolve_k(query, k)
        if wanted <= 0:
            return []
        # Chroma raises on an n_results larger than the collection on some
        # versions and merely warns on others, so it is capped here. count() is
        # the whole collection rather than the filtered subset; asking for more
        # than the filter admits is fine and simply returns fewer.
        held = self.collection.count()
        if not held:
            return []

        found = self.collection.query(
            query_embeddings=[self._encode(query.text)],
            n_results=min(wanted, held),
            where=where_for(query),
            include=["documents", "metadatas", "distances"],
        )
        return rank(
            self._rows(found),
            retriever=self.name,
            k=wanted,
            min_score=self.min_score,
        )

    def _rows(self, found: Any) -> list[tuple[dict[str, Any], float]]:
        """Chroma's column-of-lists answer as the (chunk, score) pairs rank takes.

        Chroma returns one list per query and this module sends one query, so
        every field is unwrapped at index 0. A query that matched nothing comes
        back with empty lists rather than missing keys, which ``zip`` handles
        without a special case.
        """
        return [
            (chunk_from_record(chunk_id, document, metadata), similarity_of(distance))
            for chunk_id, document, metadata, distance in zip(
                _first(found, "ids"),
                _first(found, "documents"),
                _first(found, "metadatas"),
                _first(found, "distances"),
            )
        ]

    def _encode(self, text: str) -> list[float]:
        """The query as a vector, on the same terms the passages were encoded on.

        ``QUERY_PREFIX`` is applied here and nowhere else. bge-v1.5 asks for the
        instruction prefix on the query side only, and a prefix applied to both
        sides makes every passage slightly more similar to every other -- so the
        asymmetry is deliberate and this is the only place it is introduced.
        Normalisation matches ``EMBED_NORMALIZE``, because a cosine index scored
        with an unnormalised query returns a ranking that is subtly wrong rather
        than obviously broken.
        """
        vectors = self._load().encode(
            [f"{QUERY_PREFIX}{text}"],
            normalize_embeddings=EMBED_NORMALIZE,
            show_progress_bar=False,
        )
        return [float(value) for value in vectors[0]]

    def _load(self):
        """The encoder, loaded on first use and kept for the retriever's life.

        Late, so constructing a retriever costs nothing; cached, so an ablation
        that asks a hundred questions loads bge once rather than a hundred
        times.
        """
        if self._model is None:
            from .embed import _load_model

            self._model, _ = _load_model()
        return self._model


def _first(found: Any, field: str) -> Sequence[Any]:
    """The first (and only) query's column, or an empty one if Chroma omitted it."""
    value = found.get(field) or []
    return value[0] if len(value) else []


def load_index(
    chroma_dir: Path = CHROMA_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> DenseRetriever:
    """Convenience wrapper for loading the project dense index."""
    return DenseRetriever.load(chroma_dir=chroma_dir, processed_dir=processed_dir)


__all__ = ["DenseRetriever", "load_index", "similarity_of"]
