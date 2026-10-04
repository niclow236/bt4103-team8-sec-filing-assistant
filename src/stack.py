"""One assembly path: a configuration id in, a ready retriever and generator out.

The demo and the evaluation harness have to be the same system, or the numbers
in ``results/`` describe something other than what is on screen. Before this
module there were three places that built a stack, and they had already drifted:
the ablation runner knew five configurations, the answer harness offered three
retrievers on the command line, and the app offered two and could not express a
score floor, a passage budget, or either of the answer-stage switches at all. A
configuration measured at C4 could not be demonstrated, and what the demo showed
was a fourth thing nobody had measured.

So a configuration is named once, here, and everything else asks for it by id.
:data:`STACKS` is the registry, :func:`build_retriever` is the only place a
retriever is constructed, and :func:`build_stack` returns the retriever, the
generator and the answer-time settings as one object whose
:meth:`Stack.answer` is how the app answers a question.

The ids are the ablation's own: C0 to C4 as ``src/evaluation/run.py`` has always
defined them, moved here so the app can select one. ``Stack.answer`` applies the
#34 and #35 switches, and the refusal of a question no filing can answer, from
the configuration rather than from its caller, so a new ablation row for any
of them is a line in this registry instead of an argument threaded through two
commands.

Nothing here imports Streamlit or argparse: the app and the two commands are
callers of this module, never the other way round. The retriever imports are
inside the functions because loading the dense index pulls in sentence
transformers, and a caller asking for BM25 should not pay for that.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PROCESSED_DIR, PROJECT_ROOT
from .retrieval.constants import FINAL_K
from .retrieval.records import Query
from .retrieval.embedding_config import EmbeddingConfig

# Where the ablation runner writes, and so where the app looks for the
# configurations that have actually been measured.
RESULTS_ROOT = PROJECT_ROOT / "results"

# The retriever keys a configuration may name. "bm25-fixed-size" is the naive
# C0 baseline, which is built by the ablation runner rather than here: it
# re-cuts the corpus into fixed-size windows, so it belongs with the run that
# needs it and not in the path the app shares.
RETRIEVERS = ("bm25", "dense", "hybrid")


@dataclass(frozen=True)
class StackConfig:
    """One named configuration: what to retrieve with, and how to answer.

    The first four fields are the ablation row as the retrieval runner has
    always recorded it. The rest are what a complete stack also needs, and
    they carry today's behaviour as defaults, so naming an existing id changes
    nothing about how it is measured.
    """

    id: str
    name: str
    retriever: str              # a key in RETRIEVERS, or the runner's own baseline
    chunking: str               # "section-aware", or "fixed-size" for C0
    metadata_filter: bool       # whether the question's own filters are applied
    top_k: int = FINAL_K        # passages that reach the generator
    min_score: float | None = None   # an extra floor on this retriever's scale
    use_facts: bool = True           # #34: look a numeric question up first
    use_decomposition: bool = True   # #35: one search per filing
    use_refusal: bool = True         # refuse, unsearched, what no filing answers

    def to_dict(self) -> dict[str, Any]:
        """The configuration as a results file records it."""
        return {
            "id": self.id,
            "name": self.name,
            "retriever": self.retriever,
            "chunking": self.chunking,
            "metadata_filter": self.metadata_filter,
            "top_k": self.top_k,
            "min_score": self.min_score,
            "use_facts": self.use_facts,
            "use_decomposition": self.use_decomposition,
            "use_refusal": self.use_refusal,
        }

    def scoped(self, query: Query) -> Query:
        """``query`` as this row searches for it.

        A row measured without the metadata filter searched the whole corpus
        for the question as asked (``src/evaluation/run.py``), so the filters a
        caller's Query carries, and what the parse added for them, are dropped
        here rather than left for each caller to remember. Recording the flag
        and not applying it made C1 to C3 answer exactly as C4 does, so a row
        selected in the app was not the row ``results/`` had measured.
        """
        return query if self.metadata_filter else Query(query.text, top_k=query.top_k)


STACKS: dict[str, StackConfig] = {
    "C0": StackConfig(
        id="C0", name="naive BM25", retriever="bm25-fixed-size",
        chunking="fixed-size", metadata_filter=False,
    ),
    "C1": StackConfig(
        id="C1", name="section-aware BM25", retriever="bm25",
        chunking="section-aware", metadata_filter=False,
    ),
    "C2": StackConfig(
        id="C2", name="dense retrieval", retriever="dense",
        chunking="section-aware", metadata_filter=False,
    ),
    "C3": StackConfig(
        id="C3", name="hybrid retrieval", retriever="hybrid",
        chunking="section-aware", metadata_filter=False,
    ),
    "C4": StackConfig(
        id="C4", name="hybrid retrieval with metadata filters", retriever="hybrid",
        chunking="section-aware", metadata_filter=True,
    ),
}

# The configuration the app opens on, and the one a caller that names none
# gets. C4 because it is the only row whose retrieval the app can honestly
# show: the sidebar's company and year filters are the metadata filter, and a
# row with them off would ignore what the user selected.
DEFAULT_STACK = "C4"

# The rows a caller can build and answer with. C0's fixed-size baseline is
# built only by the ablation runner (see RETRIEVERS), so a run measures it but
# the app and the answer harness cannot offer it: offering it promised a stack
# build_retriever refuses, and the error blamed a retriever nobody typed.
SELECTABLE = tuple(key for key, config in STACKS.items() if config.retriever in RETRIEVERS)


def stack_config(config_id: str) -> StackConfig:
    """The configuration an id names, or an error listing the ids there are."""
    try:
        return STACKS[config_id]
    except KeyError:
        raise ValueError(
            f"unknown configuration {config_id!r}; "
            f"expected one of {', '.join(STACKS)}"
        ) from None


def build_retriever(
    key: str,
    *,
    processed_dir: Path = PROCESSED_DIR,
    embedding: EmbeddingConfig | None = None,
    chroma_dir: Path | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    embedding_query_prefix: str | None = None,
    embedding_passage_prefix: str | None = None,
    model: Any | None = None,
    parts: dict[Any, Any] | None = None,
) -> Any:
    """The retriever a configuration names, loaded against the local corpus.

    The one place a retriever is constructed. ``parts`` is a dict this fills
    with the BM25 and dense retrievers it loads. A caller that builds several
    configurations over one ``processed_dir`` passes the same dict to every
    call, so each index is read and verified once and the hybrid rows share
    it -- without one, a run that builds the bm25, dense and hybrid rows reads
    every index twice and holds two bge encoders.

    ``model`` is not read. It was the reranker's cross-encoder, and it stays
    in the signature only because #114 adds parameters on the line above it,
    so taking it out here would conflict with that branch. It can go once both
    have landed.
    """
    parts = {} if parts is None else parts

    if embedding is not None and any(value is not None for value in (
        chroma_dir, embedding_model, embedding_dimensions,
        embedding_query_prefix, embedding_passage_prefix,
    )):
        raise ValueError("pass embedding or individual embedding settings, not both")
    if embedding is None:
        if chroma_dir is None and any(value is not None for value in (
            embedding_model, embedding_dimensions, embedding_query_prefix, embedding_passage_prefix,
        )):
            raise ValueError("embedding settings need chroma_dir")
        defaults = EmbeddingConfig()
        embedding = EmbeddingConfig(
            model=embedding_model if embedding_model is not None else defaults.model,
            dimensions=embedding_dimensions if embedding_dimensions is not None else defaults.dimensions,
            index_dir=chroma_dir if chroma_dir is not None else defaults.index_dir,
            query_prefix=embedding_query_prefix if embedding_query_prefix is not None else defaults.query_prefix,
            passage_prefix=embedding_passage_prefix if embedding_passage_prefix is not None else defaults.passage_prefix,
        )
    corpus_key = str(processed_dir.resolve())

    def part(name: str, load: Any) -> Any:
        # Scope the cache to the corpus and every setting that affects retrieval.
        # Validation happens above even when a slot was already populated.
        slot = (name, corpus_key, embedding) if name == "dense" else (name, corpus_key)
        if slot not in parts:
            kwargs = {"processed_dir": processed_dir}
            if name == "dense":
                kwargs["embedding"] = embedding
            parts[slot] = load(**kwargs)
        return parts[slot]

    from .retrieval.bm25 import BM25Retriever

    if key == "bm25":
        return part("bm25", BM25Retriever.load)

    from .retrieval.dense import DenseRetriever

    if key == "dense":
        return part("dense", DenseRetriever.load)

    from .retrieval.hybrid import HybridRetriever

    if key == "hybrid":
        return HybridRetriever(
            part("bm25", BM25Retriever.load),
            part("dense", DenseRetriever.load),
        )

    raise ValueError(
        f"unknown retriever {key!r}; expected one of {', '.join(RETRIEVERS)}"
    )


@dataclass(frozen=True)
class Stack:
    """A built configuration: what retrieves, what generates, and how to ask it.

    ``answer`` is how the app asks. The evaluation command passes the same
    configuration's settings to ``evaluate``, which calls ``answer_question``
    itself, and both take what a row searches from ``StackConfig.scoped``.
    """

    config: StackConfig
    retriever: Any
    generation: Any             # a rag.records.GenerationConfig
    llm: Any = None             # a chat model, or None to let generate build one

    def answer(self, question: str, **overrides: Any):
        """Answer one question with this configuration's settings.

        ``overrides`` are passed to ``answer_question``: the app passes the
        ``query`` and ``parsed`` its sidebar built, and a caller may pass
        ``on_token`` to stream. A caller may also override a setting this
        configuration names; the configuration supplies every setting the
        caller does not.
        """
        from .rag.answer import answer_question

        settings: dict[str, Any] = {
            "config": self.generation,
            "llm": self.llm,
            "min_score": self.config.min_score,
            "use_facts": self.config.use_facts,
            "use_decomposition": self.config.use_decomposition,
            "use_refusal": self.config.use_refusal,
        }
        settings.update(overrides)
        # A row measured without the metadata filter searched the whole corpus,
        # so it answers from one here too, whatever filters its caller built.
        if not self.config.metadata_filter:
            given = settings.get("query")
            settings["query"] = self.config.scoped(
                given if given is not None else Query(question, top_k=self.config.top_k))
        return answer_question(question, self.retriever, **settings)


def build_stack(
    config_id: str = DEFAULT_STACK,
    *,
    processed_dir: Path = PROCESSED_DIR,
    embedding: EmbeddingConfig | None = None,
    chroma_dir: Path | None = None,
    embedding_model: str | None = None,
    embedding_dimensions: int | None = None,
    embedding_query_prefix: str | None = None,
    embedding_passage_prefix: str | None = None,
    provider: str | None = None,
    model: str | None = None,
    retriever_key: str | None = None,
    retriever: Any | None = None,
    llm: Any | None = None,
    parts: dict[Any, Any] | None = None,
    **settings: Any,
) -> Stack:
    """Build the configuration ``config_id`` names.

    ``provider`` and ``model`` choose the generator, defaulting to ``.env`` as
    ``config_from_env`` reads it. ``retriever_key`` replaces the retriever the
    configuration names, for a command line that offers that; it is applied
    before anything is loaded, so overriding a hybrid row with BM25 does not
    read the dense index on the way past. ``retriever`` and ``llm`` are
    already-built parts, for a caller that loaded them once for several
    configurations or a test that wants neither an index nor a model.
    ``parts`` is passed to ``build_retriever``, for a caller that builds more
    than one configuration and wants them to share the indexes.
    ``settings`` override the rest of the configuration -- ``top_k``,
    ``min_score``, ``use_facts``, ``use_decomposition``, ``use_refusal``.
    """
    from dataclasses import replace as _replace

    from .rag.generate import chat_model, config_from_env

    config = stack_config(config_id)
    if retriever_key is not None:
        settings["retriever"] = retriever_key
    if settings:
        allowed = {"retriever", "top_k", "min_score", "use_facts", "use_decomposition",
                   "use_refusal"}
        unknown = set(settings) - allowed
        if unknown:
            raise ValueError(
                f"unknown setting(s) {', '.join(sorted(unknown))}; "
                f"expected any of {', '.join(sorted(allowed))}"
            )
        config = _replace(config, **settings)

    # The generator first, and it is the cheap half: reading .env and building
    # a chat model validates the provider settings, and a mistyped
    # LLM_PROVIDER or a missing MISTRAL_API_KEY should stop a caller before it
    # has read an index off disk, not after.
    generation = config_from_env(provider=provider, model=model)
    built_llm = llm if llm is not None else chat_model(generation)
    built_retriever = (
        retriever if retriever is not None
        else build_retriever(
            config.retriever,
            processed_dir=processed_dir,
            embedding=embedding,
            chroma_dir=chroma_dir,
            embedding_model=embedding_model,
            embedding_dimensions=embedding_dimensions,
            embedding_query_prefix=embedding_query_prefix,
            embedding_passage_prefix=embedding_passage_prefix,
            parts=parts,
        )
    )
    return Stack(
        config=config, retriever=built_retriever, generation=generation, llm=built_llm,
    )


# --- the configurations that have actually been measured --------------------

@dataclass(frozen=True)
class MeasuredRun:
    """One configuration as one run in ``results/`` recorded it."""

    run_id: str
    config_id: str
    name: str
    questions: int
    metrics: Mapping[str, float | None] = field(default_factory=dict)

    @property
    def label(self) -> str:
        """How the app names it: the run it came from, then the row."""
        return f"{self.run_id} · {self.config_id} — {self.name}"


def measured_runs(results_root: Path = RESULTS_ROOT) -> list[MeasuredRun]:
    """Every configuration a run under ``results/`` has measured, newest first.

    This is what lets the app offer a configuration *because* it was measured,
    rather than offering a list of its own that happens to overlap. A run whose
    summary is missing or unreadable is skipped: a half-written results
    directory should not stop the app from starting.

    Only ids this module knows are returned. A results directory written by an
    older revision can name a row that no longer exists, and offering it would
    promise a stack that cannot be built.
    """
    found: list[MeasuredRun] = []
    for summary in _summaries(results_root):
        run_id = str(summary.get("run_id") or "").strip()
        for row in summary.get("configurations") or ():
            config = row.get("config") or {}
            config_id = str(config.get("id") or "").strip()
            if not run_id or config_id not in STACKS:
                continue
            found.append(MeasuredRun(
                run_id=run_id,
                config_id=config_id,
                name=str(config.get("name") or STACKS[config_id].name),
                questions=int(row.get("questions") or 0),
                metrics={
                    metric: row.get(metric)
                    for metric in ("recall", "ndcg", "mrr", "hard_negative_accuracy")
                    if metric in row
                },
            ))
    return found


def _summaries(results_root: Path) -> Iterator[dict[str, Any]]:
    """Each run's ``summary.json`` under ``results/``, newest run first."""
    if not results_root.is_dir():
        return
    runs = sorted(
        (path for path in results_root.iterdir() if path.is_dir()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for run in runs:
        summary = run / "summary.json"
        if not summary.is_file():
            continue
        try:
            loaded = json.loads(summary.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(loaded, dict):
            yield loaded


__all__ = [
    "DEFAULT_STACK",
    "MeasuredRun",
    "RESULTS_ROOT",
    "RETRIEVERS",
    "SELECTABLE",
    "STACKS",
    "Stack",
    "StackConfig",
    "build_retriever",
    "build_stack",
    "measured_runs",
    "stack_config",
]
