"""Cut the corpus at several chunk sizes and measure retrieval at each (#45).

Run it from the project root, once ``data/interim/`` and the facts store exist:

    python -m src.evaluation.chunk_sweep --run-id chunk-sweep-20261007
    python -m src.evaluation.chunk_sweep --run-id bm25-sizes --retrievers bm25
    python -m src.evaluation.chunk_sweep --run-id smoke --tickers AAPL --fiscal-years 2024

``CHUNK_CHAR_BUDGET`` is 1,800 because 1,800 characters of prose fit the 512
tokens the encoder reads. That is a reason, not a measurement. This command is
the measurement: it cuts ``data/interim/`` again at 1,200, 1,800, 2,400 and
4,000 characters, builds a BM25 and a dense index over each cut, and asks every
one the same benchmark questions.

Each size is a build of its own, in ``data/sweep/<budget>/``: the passages,
``bm25.pkl``, ``chroma/`` with its manifest, and the benchmark generated for
those passages. Nothing is shared with ``data/processed/`` or ``data/index/``,
so a sweep cannot leave the app searching a corpus it was not built for, and
the four builds stay on disk for the next run to reuse.

Three things keep the sizes comparable.

The overlap and the table budget follow the prose budget, through
``constants.overlap_for`` and ``constants.table_budget_for``. A sweep that
moved the prose budget alone would report a curve tables never moved along.

The questions are the same at every size. Re-chunking renames every passage,
so a benchmark written against one cut cannot score another. The mechanical
XBRL benchmark is generated again for each build from the same facts store:
its question ids do not depend on the cut, and its supporting passages are
found in the build they are scored against. Only questions every build can
answer are kept, and the same number is drawn from each filing with a fixed
seed, as ``notebooks/retrieval/table_boost_sweep.py`` draws them.

Two cutoffs are reported, because a larger passage is more text. At ``top_k``
every size returns the same number of passages, and a 4,000-character size is
then handed over three times the text a 1,200-character one is. ``context_k``
is the number of passages of a size that fill the prompt the app sends today,
``FINAL_K`` passages of ``CHUNK_CHAR_BUDGET`` characters: 24, 16, 12 and 7 for
the four sizes. The first asks which size ranks best, the second which size
puts the answer in front of the model for the same prompt.

Every question is searched once per retriever, at the deeper of the two
cutoffs, and the ranking sliced. ``check_slicing`` compares that with direct
searches before a build's numbers are taken.

A run writes ``results/<run-id>/``: ``<row>/questions.jsonl`` and
``<row>/summary.json`` for each size and retriever, as
``python -m src.evaluation.run`` writes a C row, and the ``summary.csv`` and
``summary.json`` the C, E and G runners share. Both summaries record the
fingerprint each build's indexes wrote in their manifests, so a row can be
traced to the exact passages it was measured on.
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from time import perf_counter
from typing import Any

from src.config import CHROMA_DIR, INTERIM_DIR, PROJECT_ROOT, SWEEP_DIR
from src.pipeline.chunk import (
    chunk_filing,
    interim_files,
    iter_chunks,
    load_parsed,
    processed_path_for,
    write_chunks,
)
from src.pipeline.constants import CHUNK_CHAR_BUDGET, overlap_for, table_budget_for
from src.retrieval import bm25 as bm25_stage
from src.retrieval import embed as embed_stage
from src.retrieval.constants import (
    BM25,
    CANDIDATE_K,
    DENSE,
    EMBED_DIMENSIONS,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    FINAL_K,
    HYBRID,
    PASSAGE_PREFIX,
)
from src.retrieval.facts import FACTS_FILE
from src.retrieval.records import manifest_path
from src.stack import RESULTS_ROOT, RETRIEVERS, build_retriever
from src.utils import start_run_log

from .benchmark import generate_xbrl_questions
from .metrics import score_question
from .records import BenchmarkQuestion, RunResult
from .results import available_run_dir, write_comparison
from .run import _query, _summary

# The sizes #45 names. 1,800 is what ships, and 4,000 is what the corpus was
# first cut at, before anyone counted its passages in the encoder's tokens.
BUDGETS = (1200, 1800, 2400, 4000)

# How many questions are drawn from each filing, and the seed that draws them.
# The same number from every filing, so that one with many tagged facts does
# not outweigh one with few, and the same questions on every run.
PER_FILING = 20
SEED = 4103

# The text the app's prompt holds today, in characters. Each size is also
# scored on the passages that fill this much, so a size is not credited for
# handing the model more to read.
CONTEXT_CHARS = FINAL_K * CHUNK_CHAR_BUDGET

# How many seeded vectors are read from a donor and written at once.
SEED_PAGE = 500

# The columns a sweep row adds to the ones every C, E and G row has. The six
# from max_tokens to median_latency_ms are named as the E rows name them, so
# one table lines up under the other.
SWEEP_FIELDS = (
    "retriever", "chunk_budget", "chunk_overlap", "table_budget", "n_passages",
    "n_filings", "median_chars", "supporting_per_question", "hit_rate", "context_k",
    "context_hit_rate", "context_chars", "max_tokens", "n_indexed", "n_truncated",
    "truncation_rate", "build_ms", "median_latency_ms", "corpus_fingerprint",
    "dense_fingerprint",
)

METRICS = ("recall", "ndcg", "mrr", "hard_negative_accuracy")


@dataclass(frozen=True)
class SweepBuild:
    """One chunk size: the settings that follow from it, and where its files are."""

    budget: int
    root: Path

    @property
    def id(self) -> str:
        return f"S{self.budget}"

    @property
    def overlap(self) -> int:
        return overlap_for(self.budget)

    @property
    def table_budget(self) -> int:
        return table_budget_for(self.budget)

    @property
    def context_k(self) -> int:
        """How many passages of this size fill the prompt the app sends today."""
        return max(1, round(CONTEXT_CHARS / self.budget))

    @property
    def processed_dir(self) -> Path:
        return self.root / "processed"

    @property
    def bm25_index(self) -> Path:
        return self.root / "bm25.pkl"

    @property
    def chroma_dir(self) -> Path:
        return self.root / "chroma"

    @property
    def benchmark(self) -> Path:
        return self.root / "benchmark.jsonl"


# --- building one size ------------------------------------------------------


def chunk_corpus(
    build: SweepBuild, paths: Sequence[Path], fiscal_years: Sequence[int] | None = None,
) -> int:
    """Cut the filings in scope at this build's budget, and return how many.

    Written from nothing every time. Chunking takes seconds, and a folder left
    by an earlier run may hold filings this run narrowed out, or passages an
    older chunker cut. The indexes beside it are not rebuilt for that: each
    vector carries the digest of the text it was encoded from, so ``embed``
    encodes only the passages that changed.
    """
    if build.processed_dir.exists():
        shutil.rmtree(build.processed_dir)
    wanted = set(fiscal_years) if fiscal_years else None
    filings = 0
    for path in paths:
        parsed = load_parsed(path)
        year = parsed.period_of_report[:4]
        if wanted is not None and (not year.isdigit() or int(year) not in wanted):
            continue
        chunked = chunk_filing(
            parsed, source_path=manifest_path(path),
            budget=build.budget, overlap=build.overlap,
        )
        write_chunks(chunked, processed_path_for(path, build.processed_dir))
        filings += 1
    return filings


def describe_corpus(build: SweepBuild) -> tuple[dict[str, Any], dict[str, str]]:
    """What a build's corpus holds, and whether each passage is prose or a table."""
    sizes: list[int] = []
    kinds: dict[str, str] = {}
    filings: set[str] = set()
    for row in iter_chunks(processed_dir=build.processed_dir):
        sizes.append(row["n_chars"])
        kinds[row["chunk_id"]] = row["content_type"]
        filings.add(row["accession_no"])
    tables = sum(1 for kind in kinds.values() if kind == "table")
    return {
        "n_passages": len(kinds),
        "n_prose": len(kinds) - tables,
        "n_tables": tables,
        "n_filings": len(filings),
        "median_chars": float(median(sizes)) if sizes else None,
    }, kinds


def _donor(chroma_dir: Path):
    """The collection in ``chroma_dir``, if its vectors can stand in for ours.

    Only an index a build finished, with this model, this vector width, no
    passage prefix and the encoder's own token limit. A vector's digest says
    which text it was encoded from, not how much of that text the encoder was
    allowed to read, so an index built under a shorter limit is not a donor.
    A manifest written before the limit was recorded was built under the
    model's own, which is the one a sweep build uses.
    """
    manifest = embed_stage.read_manifest(embed_stage.manifest_file_for(chroma_dir))
    if manifest is None or manifest.max_tokens not in (None, EMBED_MAX_TOKENS):
        return None
    if manifest.mismatches(
        model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS, passage_prefix=PASSAGE_PREFIX,
    ):
        return None
    return embed_stage.open_collection(chroma_dir, create=False)


def seed_vectors(chroma_dir: Path, processed_dir: Path, donors: Sequence[Path]) -> int:
    """Copy from other indexes every vector this one needs and they already hold.

    A vector is a function of the text it was encoded from and of nothing
    else, and each one stores the digest of that text and the passage's id. So
    where a donor holds a vector under the digest a passage of this corpus
    has, encoding the passage again would only produce that vector again.
    Copying it takes milliseconds and encoding it about half a second on a
    laptop CPU. A copy matches its original to the last place a 32-bit float
    holds, since Chroma normalises a vector each time it stores one.

    That is the whole of the 1,800-character build when ``data/index/chroma``
    is current, since that build is the corpus the app already indexed, and
    between a twentieth and a quarter of the others: an Item or a table small
    enough to be one passage at every size.

    Only the vector and its token count are taken from the donor. The text and
    the metadata stored with it come from this corpus, and the digest is
    computed again from the row being written, so a copied vector is held to
    exactly what an encoded one is. Returns how many were copied; ``embed``
    then encodes the rest and writes the manifest.
    """
    donors = [donor for donor in donors if donor.resolve() != chroma_dir.resolve()]
    if not donors:
        return 0

    wanted, _, _ = embed_stage._corpus_digests(processed_dir)
    target = embed_stage.open_collection(chroma_dir)
    held = embed_stage._scan(target)
    needed = {
        chunk_id: digest for chunk_id, digest in wanted.items()
        if chunk_id not in held or held[chunk_id][0] != digest
    }
    sources: dict[str, Any] = {}
    for donor in donors:
        if len(sources) == len(needed):
            break
        collection = _donor(donor)
        if collection is None:
            continue
        for chunk_id, (digest, _, model) in embed_stage._scan(collection).items():
            if (chunk_id not in sources and model == EMBED_MODEL
                    and needed.get(chunk_id) == digest):
                sources[chunk_id] = collection
    if not sources:
        return 0

    # As ``embed.build`` does before it first changes a collection: an index
    # between two states has no manifest, so a retriever refuses it.
    embed_stage.manifest_file_for(chroma_dir).unlink(missing_ok=True)

    copied = 0

    def flush(rows: list[dict]) -> None:
        nonlocal copied
        by_donor: dict[int, tuple[Any, list[dict]]] = {}
        for row in rows:
            collection = sources[row["chunk_id"]]
            by_donor.setdefault(id(collection), (collection, []))[1].append(row)
        for collection, group in by_donor.values():
            found = collection.get(
                ids=[row["chunk_id"] for row in group], include=["embeddings", "metadatas"],
            )
            # Chroma answers in its own order, and only with the ids it holds.
            stored = {
                chunk_id: (vector, metadata or {})
                for chunk_id, vector, metadata in zip(
                    found["ids"], found["embeddings"] if found["ids"] else [],
                    found["metadatas"] or [])
            }
            ids, vectors, documents, metadatas = [], [], [], []
            for row in group:
                vector, metadata = stored.get(row["chunk_id"], (None, {}))
                digest = embed_stage.digest_of(row)
                if (vector is None or metadata.get(embed_stage.DIGEST_FIELD) != digest
                        or embed_stage.TOKENS_FIELD not in metadata):
                    continue
                ids.append(row["chunk_id"])
                vectors.append([float(value) for value in vector])
                documents.append(row["text"])
                metadatas.append({
                    **embed_stage.metadata_for(row),
                    embed_stage.DIGEST_FIELD: digest,
                    embed_stage.TOKENS_FIELD: metadata[embed_stage.TOKENS_FIELD],
                    embed_stage.MODEL_FIELD: EMBED_MODEL,
                })
            if not ids:
                continue
            replacing = [chunk_id for chunk_id in ids if chunk_id in held]
            if replacing:
                target.delete(ids=replacing)
            target.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)
            copied += len(ids)

    batch: list[dict] = []
    for row in iter_chunks(processed_dir=processed_dir):
        if row["chunk_id"] in sources:
            batch.append(row)
            if len(batch) >= SEED_PAGE:
                flush(batch)
                batch = []
    if batch:
        flush(batch)
    return copied


def build_indexes(
    build: SweepBuild, *, dense: bool, donors: Sequence[Path] = (),
    threads: int | None = None,
) -> dict[str, Any]:
    """Bring a build's indexes in line with its corpus, and say what they hold.

    BM25 is fitted again each run, which takes seconds. The dense index is
    brought up to date by ``embed.build``, so a run that was interrupted, or
    that follows one over fewer filings, encodes only what is missing. It is
    left alone where no dense or hybrid row was asked for, and a sweep of BM25
    alone then takes minutes over the whole corpus.
    """
    started = perf_counter()
    sparse = bm25_stage.build_index(
        processed_dir=build.processed_dir, index_path=build.bm25_index)
    built: dict[str, Any] = {
        "corpus_fingerprint": sparse.manifest.corpus_fingerprint,
        "dense_fingerprint": None, "seeded_vectors": None,
    }
    del sparse
    if dense:
        built["seeded_vectors"] = seed_vectors(build.chroma_dir, build.processed_dir, donors)
        if built["seeded_vectors"]:
            print(f"copied:   {built['seeded_vectors']:,} vectors another index already "
                  "held for the same text", flush=True)
        manifest = embed_stage.build(
            chroma_dir=build.chroma_dir, processed_dir=build.processed_dir,
            threads=threads, max_tokens=EMBED_MAX_TOKENS,
        )
        built["dense_fingerprint"] = manifest.corpus_fingerprint
    built["build_ms"] = (perf_counter() - started) * 1000
    return built


# --- the questions every size is asked --------------------------------------


def sample_questions(
    generated: Mapping[int, Sequence[BenchmarkQuestion]],
    per_filing: int = PER_FILING, seed: int = SEED,
) -> list[str]:
    """The ids of the questions every size is measured on.

    Only a question every build generated, so no size is scored on a question
    another could not be asked. Then ``per_filing`` from each filing, drawn
    with a fixed seed, or all of them where ``per_filing`` is 0 or a filing
    holds fewer.
    """
    common = set.intersection(*(
        {question.question_id for question in questions} for questions in generated.values()
    ))
    by_filing: dict[tuple[str, int], list[str]] = defaultdict(list)
    for question in next(iter(generated.values())):
        if question.question_id in common:
            by_filing[(question.ticker or "", question.fiscal_year or 0)].append(
                question.question_id)
    rng = random.Random(seed)
    chosen: list[str] = []
    for filing in sorted(by_filing):
        held = sorted(by_filing[filing])
        if per_filing and per_filing < len(held):
            held = sorted(rng.sample(held, per_filing))
        chosen += held
    return chosen


def supported_by(question: BenchmarkQuestion, kinds: Mapping[str, str]) -> str:
    """Whether a question's supporting passages are tables, prose or both."""
    found = {kinds.get(chunk_id, "prose") for chunk_id in question.supporting_chunk_ids}
    return "tables" if found == {"table"} else "prose" if "table" not in found else "both"


# --- measuring one size -----------------------------------------------------


def check_slicing(retrievers: Mapping[str, Any], queries: Sequence[Any], top_k: int) -> None:
    """A deep search cut to ``top_k`` must be what a search at ``top_k`` returns.

    Each question is searched once, at the deeper cutoff, and the ranking is
    cut to the shallower. That is sound while the order does not depend on
    how many passages were asked for. Hybrid fuses the top ``CANDIDATE_K`` of
    each method whatever it is asked for, and BM25 and dense both rank before
    they cut, so it should hold. This checks that it does, on real searches,
    before a build's rows are written.
    """
    for name, retriever in retrievers.items():
        for query in queries:
            deep = [passage.chunk_id for passage in retriever.search(query)][:top_k]
            direct = [passage.chunk_id for passage in retriever.search(query, k=top_k)]
            if deep != direct:
                raise RuntimeError(
                    f"{name}: a top {query.top_k} cut to {top_k} is not a top {top_k} for "
                    f"{query.text!r}, so one search per question cannot score both cutoffs"
                )


NOT_ENCODED = {"max_tokens": None, "n_indexed": None, "n_truncated": None,
               "truncation_rate": None}


def _truncation(retriever: Any) -> dict[str, Any]:
    """How many of a dense index's vectors the encoder cut short, read off the index."""
    dense = getattr(retriever, "dense", retriever)
    collection = getattr(dense, "collection", None)
    if collection is None:
        return dict(NOT_ENCODED)
    limit = getattr(dense.manifest, "max_tokens", None) or EMBED_MAX_TOKENS
    indexed = collection.count()
    truncated = len(collection.get(
        where={embed_stage.TOKENS_FIELD: {"$gt": limit}}, include=[])["ids"])
    return {"max_tokens": limit, "n_indexed": indexed, "n_truncated": truncated,
            "truncation_rate": truncated / indexed if indexed else None}


def _group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The share of a set of rows that found a supporting passage, at each cutoff."""
    ranked = [row["mrr"] for row in rows if row["mrr"] is not None]
    return {
        "questions": len(rows),
        "hit_rate": mean(row["hit"] for row in rows),
        "context_hit_rate": mean(row["context_hit"] for row in rows),
        "mrr": mean(ranked) if ranked else None,
    }


def measure_build(
    build: SweepBuild,
    questions: Sequence[BenchmarkQuestion],
    retrievers: Sequence[str],
    *,
    run_dir: Path,
    top_k: int,
    described: Mapping[str, Any],
    support: Mapping[str, str],
) -> list[dict[str, Any]]:
    """Ask one build every question through each retriever, and write its rows."""
    parts: dict[Any, Any] = {}
    try:
        built = {
            key: build_retriever(
                key, processed_dir=build.processed_dir, chroma_dir=build.chroma_dir,
                bm25_index=build.bm25_index, parts=parts,
            )
            for key in retrievers
        }
    except ValueError as error:
        # A retriever's refusal names the command that rebuilds the app's own
        # index, which is not what rebuilds this one.
        raise RuntimeError(
            f"{build.id}: an index under {build.root} no longer matches the passages beside "
            "it, so something changed that folder after this run built it. Run the same "
            f"command again, which brings each build up to date.\n{error}"
        ) from error
    depth = max(top_k, build.context_k)
    queries = [_query(question, top_k=depth, metadata_filter=True) for question in questions]
    check_slicing(built, queries[:5], top_k)
    truncation = _truncation(built.get(DENSE) or built.get(HYBRID))

    summaries: list[dict[str, Any]] = []
    for key, retriever in built.items():
        configuration = {
            "id": f"{build.id}-{key}",
            "name": f"{build.budget:,} characters, {key}",
            "retriever": key,
            "chunking": "section-aware",
            "metadata_filter": True,
            "top_k": top_k,
            "chunk_budget": build.budget,
            "chunk_overlap": build.overlap,
            "table_budget": build.table_budget,
            "context_k": build.context_k,
        }
        rows: list[dict[str, Any]] = []
        for number, (question, query) in enumerate(zip(questions, queries), start=1):
            started = perf_counter()
            passages = retriever.search(query)
            latency_ms = (perf_counter() - started) * 1000
            top, context = passages[:top_k], passages[:build.context_k]
            supporting = set(question.supporting_chunk_ids)
            metrics = score_question(
                question,
                RunResult.from_passages(question.question_id, top, retriever=retriever.name),
                k=top_k,
            )
            rows.append({
                "question_id": question.question_id,
                "question_type": question.question_type,
                "difficulty": question.difficulty,
                "source": question.source,
                "retriever": retriever.name,
                "config": configuration,
                # To the deeper cutoff, so either one can be scored again from this file.
                "retrieved_chunk_ids": [passage.chunk_id for passage in passages],
                "retrieved_scores": [float(passage.score) for passage in passages],
                **{name: metrics[name] for name in METRICS},
                "hit": any(passage.chunk_id in supporting for passage in top),
                "context_hit": any(passage.chunk_id in supporting for passage in context),
                "context_chars": sum(len(passage.text) for passage in context),
                "supported_by": support[question.question_id],
                "latency_ms": latency_ms,
            })
            if number % 250 == 0:
                print(f"  {configuration['id']}: {number:,} of {len(questions):,} questions",
                      flush=True)

        overall = _group(rows)
        summary = {
            "config": configuration,
            **_summary(rows),
            "hit_rate": overall["hit_rate"],
            "context_hit_rate": overall["context_hit_rate"],
            "context_chars": mean(row["context_chars"] for row in rows),
            # The benchmark names at most three passages that print a figure.
            # A smaller size cuts a filing into more passages, so more of its
            # questions reach that ceiling, and this mean says by how much.
            "supporting_per_question": mean(
                len(question.supporting_chunk_ids) for question in questions),
            "median_latency_ms": median(row["latency_ms"] for row in rows),
            "by_support": {
                kind: _group([row for row in rows if row["supported_by"] == kind])
                for kind in ("tables", "both", "prose")
                if any(row["supported_by"] == kind for row in rows)
            },
            **described,
            # What the encoder cut short is the dense index's to report. BM25
            # reads a passage whole, whatever its length.
            **(truncation if key != BM25 else NOT_ENCODED),
        }
        output_dir = run_dir / configuration["id"]
        output_dir.mkdir(parents=True, exist_ok=True)
        with (output_dir / "questions.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        (output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        summaries.append(summary)
        print(f"  {configuration['id']}: recall {summary['recall']:.3f}, "
              f"supporting passage in the top {top_k} for {summary['hit_rate']:.1%}, "
              f"in the top {build.context_k} for {summary['context_hit_rate']:.1%}", flush=True)
    return summaries


# --- the run ----------------------------------------------------------------


def _commit() -> str | None:
    """The commit the working tree is at, marked where it has changes not in it."""
    try:
        described = subprocess.run(
            ["git", "describe", "--always", "--dirty"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, check=True, timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return described.stdout.strip() or None


def curve(summaries: Sequence[Mapping[str, Any]], top_k: int) -> str:
    """The run as one table: a column for each chunk size, a row for each measure."""
    by_row = {(summary["config"]["retriever"], summary["config"]["chunk_budget"]): summary
              for summary in summaries}
    budgets = sorted({budget for _, budget in by_row})
    of_size = {budget: [summary for (_, size), summary in by_row.items() if size == budget]
               for budget in budgets}
    first = {budget: of_size[budget][0] for budget in budgets}

    def line(label: str, cells: Sequence[str]) -> str:
        return f"{label:<44}" + "".join(f"{cell:>10}" for cell in cells)

    lines = [
        line("chunk budget, characters", [f"{budget:,}" for budget in budgets]),
        line("passages", [f"{first[budget]['n_passages']:,}" for budget in budgets]),
        line("median passage, characters",
             [f"{first[budget]['median_chars']:,.0f}" for budget in budgets]),
    ]
    # A property of the size's dense index, so it is read from whichever row
    # searched one, and left out of a run that built none.
    encoded = {budget: next((summary for summary in of_size[budget]
                             if summary["truncation_rate"] is not None), None)
               for budget in budgets}
    if any(encoded.values()):
        limit = next(summary["max_tokens"] for summary in encoded.values() if summary)
        lines.append(line(
            f"cut short by the encoder at {limit} tokens",
            [f"{encoded[budget]['truncation_rate']:.1%}" if encoded[budget] else ""
             for budget in budgets],
        ))
    depths = "/".join(str(first[budget]["config"]["context_k"]) for budget in budgets)
    for retriever in dict.fromkeys(retriever for retriever, _ in by_row):
        rows = [by_row.get((retriever, budget)) for budget in budgets]
        lines.append(retriever)
        for label, field, form in (
            (f"  Recall@{top_k}", "recall", "{:.3f}"),
            (f"  nDCG@{top_k}", "ndcg", "{:.3f}"),
            (f"  MRR@{top_k}", "mrr", "{:.3f}"),
            (f"  supporting passage in the top {top_k}", "hit_rate", "{:.1%}"),
            (f"  in the same prompt (top {depths})", "context_hit_rate", "{:.1%}"),
            ("  characters in that prompt, mean", "context_chars", "{:,.0f}"),
            ("  median search, ms", "median_latency_ms", "{:.0f}"),
        ):
            lines.append(line(label, [
                "" if row is None or row[field] is None else form.format(row[field])
                for row in rows
            ]))
    return "\n".join(lines)


def run_chunk_sweep(
    *,
    run_id: str,
    budgets: Sequence[int] = BUDGETS,
    retrievers: Sequence[str] = RETRIEVERS,
    tickers: Sequence[str] | None = None,
    fiscal_years: Sequence[int] | None = None,
    per_filing: int = PER_FILING,
    seed: int = SEED,
    top_k: int = 10,
    interim_dir: Path = INTERIM_DIR,
    facts_file: Path = FACTS_FILE,
    sweep_dir: Path = SWEEP_DIR,
    results_root: Path = RESULTS_ROOT,
    donors: Sequence[Path] = (),
    threads: int | None = None,
) -> dict[str, Any]:
    """Build every size, measure each, and write ``results/<run-id>/``.

    Every build is finished before a question is asked or the run directory
    made, so a run that stops while encoding has used no run id and is
    carried on by running the same command again.
    """
    budgets = tuple(budgets)
    retrievers = tuple(dict.fromkeys(retrievers))
    if not budgets or len(set(budgets)) != len(budgets) or min(budgets) < 1:
        raise ValueError("budgets must be distinct positive numbers of characters")
    if not retrievers or set(retrievers) - set(RETRIEVERS):
        raise ValueError(f"retrievers must be among {', '.join(RETRIEVERS)}")
    if top_k < 1 or per_filing < 0:
        raise ValueError("top_k must be positive and per_filing zero or more")
    builds = [SweepBuild(budget, sweep_dir / str(budget)) for budget in budgets]
    for build in builds:
        if max(top_k, build.context_k) > CANDIDATE_K:
            raise ValueError(
                f"a top {max(top_k, build.context_k)} is asked of the {build.budget}-character "
                f"size, and hybrid fuses the top {CANDIDATE_K} of each method (CANDIDATE_K): "
                "beyond it the order depends on the cutoff, and the two cannot be compared"
            )
    run_dir = available_run_dir(results_root, run_id)
    # Read now and not when the run is written, hours later: it is the code
    # this process loaded that the numbers come from.
    commit = _commit()

    paths = interim_files(interim_dir)
    if tickers:
        wanted = {ticker.upper() for ticker in tickers}
        paths = [path for path in paths if path.parent.name in wanted]
    if not paths:
        raise FileNotFoundError(
            f"No parsed filing under {interim_dir} is in scope. Parse the corpus first: "
            "python -m src.pipeline parse"
        )
    if not facts_file.exists():
        raise FileNotFoundError(
            f"No facts store at {facts_file}, and the questions are generated from it. "
            "Build it with: python -m src.retrieval facts"
        )

    dense = bool({DENSE, HYBRID} & set(retrievers))
    described: dict[int, dict[str, Any]] = {}
    kinds: dict[int, dict[str, str]] = {}
    generated: dict[int, list[BenchmarkQuestion]] = {}
    for build in builds:
        print(f"\n{build.id}: {build.budget:,} characters a passage, {build.overlap} carried "
              f"over, {build.table_budget} a table passage", flush=True)
        filings = chunk_corpus(build, paths, fiscal_years)
        if not filings:
            raise FileNotFoundError(
                f"No parsed filing under {interim_dir} reports on the fiscal years asked for."
            )
        corpus, kinds[build.budget] = describe_corpus(build)
        print(f"chunked:  {corpus['n_passages']:,} passages from {filings} filings, "
              f"{corpus['n_tables']:,} of them tables", flush=True)
        # Every other size's index, and the app's own, may hold a vector this
        # one needs: the passages too short for the budget to have cut them.
        others = [other.chroma_dir for other in builds if other is not build]
        indexes = build_indexes(
            build, dense=dense, donors=[*donors, *others], threads=threads)
        described[build.budget] = {**corpus, **indexes}
        generated[build.budget] = generate_xbrl_questions(
            facts_file, processed_dir=build.processed_dir, output_path=build.benchmark)
        print(f"asked:    {len(generated[build.budget]):,} questions have a supporting "
              "passage in this build", flush=True)

    chosen = sample_questions(generated, per_filing=per_filing, seed=seed)
    if not chosen:
        raise ValueError("no question has a supporting passage in every build")
    # Grouped once, by what supports a question in the build nearest the
    # shipped size, so a group holds the same questions in every column.
    reference = min(budgets, key=lambda budget: abs(budget - CHUNK_CHAR_BUDGET))
    by_id = {budget: {question.question_id: question for question in questions}
             for budget, questions in generated.items()}
    support = {question_id: supported_by(by_id[reference][question_id], kinds[reference])
               for question_id in chosen}
    filings_asked = len({(by_id[reference][question_id].ticker,
                          by_id[reference][question_id].fiscal_year) for question_id in chosen})
    print(f"\nmeasuring {len(chosen):,} questions from {filings_asked} filings at each size, "
          f"through {', '.join(retrievers)}", flush=True)

    summaries: list[dict[str, Any]] = []
    for build in builds:
        summaries += measure_build(
            build, [by_id[build.budget][question_id] for question_id in chosen], retrievers,
            run_dir=run_dir, top_k=top_k, described=described[build.budget], support=support,
        )
        # One size's BM25 fit and encoder are released before the next loads.
        gc.collect()

    return write_comparison(
        run_dir, run_id=run_id, top_k=top_k, summaries=summaries,
        extra_fields=SWEEP_FIELDS, matrix="chunk-size",
        measured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        commit=commit,
        embedding_model=EMBED_MODEL if dense else None,
        context_chars=CONTEXT_CHARS,
        corpus={
            "tickers": sorted({ticker.upper() for ticker in tickers}) if tickers else None,
            "fiscal_years": sorted(set(fiscal_years)) if fiscal_years else None,
        },
        questions={
            "asked": len(chosen), "filings": filings_asked, "per_filing": per_filing,
            "seed": seed, "grouped_by_support_at": reference,
            "generated": {str(budget): len(questions) for budget, questions in generated.items()},
        },
        builds=[
            {"id": build.id, "chunk_budget": build.budget, "chunk_overlap": build.overlap,
             "table_budget": build.table_budget, "context_k": build.context_k,
             **described[build.budget]}
            for build in builds
        ],
    )


# --- the command line -------------------------------------------------------


def _positive(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m src.evaluation.chunk_sweep",
        description="Cut the corpus at several chunk sizes and measure retrieval at each.",
    )
    parser.add_argument("--run-id", required=True,
                        help="Names results/<run-id>/. A run id is never written twice.")
    parser.add_argument(
        "--budgets", nargs="+", type=_positive, default=list(BUDGETS), metavar="CHARS",
        help="Characters per passage at each size "
             f"(default: {' '.join(str(budget) for budget in BUDGETS)}).",
    )
    parser.add_argument(
        "--retrievers", nargs="+", choices=RETRIEVERS, default=list(RETRIEVERS),
        help="What searches each size (default: all three). Without dense and hybrid no "
             "passage is encoded, so bm25 alone takes minutes over the whole corpus.",
    )
    parser.add_argument("--tickers", nargs="+", metavar="TICKER",
                        help="Only these companies. Default: every company in data/interim/.")
    parser.add_argument("--fiscal-years", nargs="+", type=int, metavar="YEAR",
                        help="Only filings reporting on these fiscal years.")
    parser.add_argument(
        "--per-filing", type=int, default=PER_FILING, metavar="N",
        help=f"Questions drawn from each filing (default: {PER_FILING}). 0 asks every one.",
    )
    parser.add_argument("--top-k", type=_positive, default=10,
                        help="The cutoff every size is scored at (default: 10).")
    parser.add_argument("--threads", type=_positive, metavar="N",
                        help="Threads for the encoder, as python -m src.retrieval embed takes.")
    parser.add_argument(
        "--no-reuse", action="store_true",
        help="Encode every passage, instead of first copying the vectors data/index/chroma "
             "already holds for the same text.",
    )
    parser.add_argument("--interim-dir", type=Path, default=INTERIM_DIR)
    parser.add_argument("--facts-file", type=Path, default=FACTS_FILE)
    parser.add_argument("--sweep-dir", type=Path, default=SWEEP_DIR)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.per_filing < 0:
        parser.error("--per-filing must be zero or more")
    try:
        available_run_dir(args.results_root, args.run_id)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))

    log_path = start_run_log("chunk-sweep")
    try:
        manifest = run_chunk_sweep(
            run_id=args.run_id, budgets=args.budgets, retrievers=args.retrievers,
            tickers=args.tickers, fiscal_years=args.fiscal_years,
            per_filing=args.per_filing, top_k=args.top_k,
            interim_dir=args.interim_dir, facts_file=args.facts_file,
            sweep_dir=args.sweep_dir, results_root=args.results_root,
            donors=() if args.no_reuse else (CHROMA_DIR,), threads=args.threads,
        )
        print("\n" + curve(manifest["configurations"], manifest["top_k"]))
        print(f"\nwritten: {args.results_root / args.run_id}")
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"\n{error}") from None
    finally:
        print(f"Run log: {log_path}")


if __name__ == "__main__":
    main()
