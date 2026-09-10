"""Build the dense index: every passage as a vector in ``data/index/chroma/``.

Run it from the project root:

    python -m src.retrieval embed
    python -m src.retrieval embed --tickers AAPL --fiscal-years 2024
    python -m src.retrieval embed --rebuild

A table passage is why this stage is not simply "embed the text". Item 8 is
mostly figures, and a passage of bare numbers embeds nowhere useful: "12,345"
carries no company, no year and no subject, so it lands next to every other
number in the corpus rather than next to the question that wants it. So the
context header from ``constants.CONTEXT_HEADER`` -- company, ticker, fiscal
year, form, Item and title -- is prepended to the text before it is encoded.

It is prepended and then thrown away. What Chroma stores as the document is the
passage exactly as the chunker wrote it, because the stored text is what a
citation quotes, and a citation that quotes our own annotation back at the
reader is worse than none. The header exists in the vector and nowhere else.

That has a consequence for staleness. Because the header reaches the encoder,
this index's contents depend on metadata as well as on text: rename a company
in the ticker map and every one of its vectors changes while its passages do
not. So the fingerprint this stage writes is taken over ``embed_text`` rather
than over the raw passage -- see :func:`records.corpus_fingerprint`.

The index is derived data: delete it and it rebuilds from ``data/processed/``.
What cannot be rebuilt is the knowledge of which corpus it came from, so an
``IndexManifest`` is written beside it, and a retriever compares that against
the corpus on disk before it will load one. A manifest is a claim about the
whole index, so this module writes one only when the index holds the whole
corpus: a run narrowed by ``--tickers`` builds real vectors and leaves the
manifest alone.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DIAGNOSTICS_DIR
from ..pipeline.chunk import iter_chunks, resolve_chunk_settings
from ..pipeline.constants import CHUNK_CHAR_BUDGET, CHUNK_CHAR_OVERLAP
from .constants import (
    CHROMA_DIR,
    CONTEXT_HEADER,
    DENSE,
    DISTANCE_METRIC,
    EMBED_BATCH_SIZE,
    EMBED_DIMENSIONS,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    EMBED_NORMALIZE,
    PASSAGE_PREFIX,
)
from .records import IndexManifest, corpus_fingerprint, manifest_path

# One collection holds the whole corpus. Splitting per company would make a
# cross-company question a fan-out over fifteen collections, and the metadata
# pre-filter already narrows a query to one company when it needs to.
COLLECTION_NAME = "passages"

# Beside the index rather than inside it: Chroma owns the contents of its own
# directory and is free to rewrite them, so a file that has to survive a
# rebuild does not belong in there.
MANIFEST_FILE = CHROMA_DIR.parent / "chroma.manifest.json"

# Where the truncation report is written. Named here rather than inside the
# function that writes it, so a resumed run and a rebuild can both find the
# file the run before them left instead of each writing over it blind.
TRUNCATION_FILE = "embed-truncated.json"

# Everything from an ``iter_chunks`` row worth carrying into the index. The
# first five are ``constants.PREFILTER_FIELDS``, which retrieval filters on;
# the rest are what a citation needs, so an answer can name the filing and link
# to it without reopening data/processed/.
METADATA_FIELDS = (
    "ticker", "fiscal_year", "item", "content_type", "is_key_section",
    "company", "cik", "form", "filing_date", "period_of_report",
    "accession_no", "url", "section_id", "part", "title", "heading",
    "chunk_index", "n_chars", "table_index", "table_caption",
)


def context_header(row: dict) -> str:
    """The header prepended to one passage at embed time and never stored.

    Missing fields become empty rather than the string "None": ``item`` is None
    for a passage cut from a section carrying no Item number, and a header
    reading "Item None" would be embedded into every one of them, which is a
    shared token where there should be nothing.
    """
    fiscal_year = row.get("fiscal_year")
    return CONTEXT_HEADER.format(
        company=row.get("company") or row.get("ticker") or "",
        ticker=row.get("ticker") or "",
        fiscal_year=fiscal_year if fiscal_year is not None else "",
        form=row.get("form") or "",
        item=row.get("item") or row.get("section_id") or "",
        title=row.get("title") or "",
    )


def embed_text(row: dict) -> str:
    """What actually goes to the encoder: prefix, header, then the passage.

    ``PASSAGE_PREFIX`` is empty for bge, which asks for a prefix on the query
    side only. It is applied here anyway so that swapping in a model wanting one
    is a change to constants.py rather than to this file.

    This is also what the dense index fingerprints, which is the only reason it
    matters that this function is pure: hand it the same row twice and the
    stale-index check has to get the same string back both times.
    """
    return f"{PASSAGE_PREFIX}{context_header(row)}{row['text']}"


def metadata_for(row: dict) -> dict[str, Any]:
    """The row reduced to what Chroma will store alongside a vector.

    Chroma takes scalars only, so a None or a list has to go somewhere or go
    away. Empty values are dropped rather than stored as "": a filter on a field
    that is absent should not match a passage that merely has nothing in it.
    ``incorporated_into`` is a list, so it is joined, which keeps it readable in
    a citation without pretending it is filterable.
    """
    metadata: dict[str, Any] = {}
    for field in METADATA_FIELDS:
        value = row.get(field)
        if value is None or value == "":
            continue
        if isinstance(value, (str, int, float, bool)):
            metadata[field] = value

    incorporated = row.get("incorporated_into") or []
    if incorporated:
        metadata["incorporated_into"] = ",".join(str(item) for item in incorporated)
    return metadata


def _use_threads(threads: int | None) -> int | None:
    """Give torch a thread count, defaulting to every logical processor.

    torch defaults to the physical core count, which is the right guess for
    most work and the wrong one here: measured on this corpus, 8 threads on a
    4-core, 8-thread laptop encoded about 20 per cent faster than torch's
    default of 4. Encoding is a long series of small matrix multiplies that
    spends much of its time waiting on memory, so the second thread on a core
    has real work to do rather than fighting the first for the same units.

    Measured, not assumed, and overridable: on a machine where it is not true,
    pass ``--threads`` with the physical core count and it goes back.

    Called only from :func:`_load_model`, and only after its guarded import.
    torch is what sentence-transformers pulls in, so touching it here first
    would turn a missing install into a bare ModuleNotFoundError from a helper
    three levels down -- which is the traceback that guard exists to replace
    with a sentence saying what to run.
    """
    import torch

    resolved = threads or os.cpu_count()
    if resolved:
        torch.set_num_threads(resolved)
    return resolved


def _load_model(threads: int | None = None):
    """Import and load the encoder, late and with a usable failure.

    sentence-transformers pulls in torch, which is seconds of import time and a
    multi-gigabyte install. Importing it at module scope would make ``--help``
    pay for both, and would turn a missing dependency into a traceback from an
    import three levels down rather than a sentence saying what to install.

    Returns the model and the thread count it was given, so the caller can
    report the setting without reaching for torch itself.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:   # pragma: no cover - depends on the install
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from error

    used = _use_threads(threads)
    model = SentenceTransformer(EMBED_MODEL)
    # Renamed in sentence-transformers 6.0; the old name still works but warns.
    # Asked for by name so the module runs on either side of that rename, since
    # requirements.txt pins a version but a teammate's venv may predate it.
    measure = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension"
    )
    width = measure()
    if width != EMBED_DIMENSIONS:
        # Two checkpoints of one family produce vectors that are not comparable,
        # and a dimension mismatch is the one case Chroma will not catch for us:
        # it accepts whatever the first add() gives it and rejects the rest.
        raise RuntimeError(
            f"{EMBED_MODEL} produces {width}-dimensional vectors but "
            f"constants.EMBED_DIMENSIONS says {EMBED_DIMENSIONS}. Change one to "
            f"match the other before building an index."
        )
    return model, used


def _collection_names(client) -> set[str]:
    """The collections a client holds, across chromadb's two return shapes.

    0.x returns collection objects, 1.x returns bare names. Asking rather than
    catching is what lets ``--rebuild`` tell "there was nothing to drop" from
    "the drop failed": the same exception, and opposite outcomes.
    """
    return {
        item if isinstance(item, str) else item.name
        for item in client.list_collections()
    }


def _open_collection(chroma_dir: Path, rebuild: bool):
    """The persistent collection, created if absent, dropped first if asked."""
    try:
        import chromadb
    except ImportError as error:   # pragma: no cover - depends on the install
        raise RuntimeError(
            "chromadb is not installed. Run: pip install -r requirements.txt"
        ) from error

    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir))

    if rebuild:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception as error:
            # Chroma's "no such collection" is a different exception per
            # version, so absence is established by looking rather than by
            # catching a name. A delete that failed for any other reason -- a
            # lock on the sqlite file, a permissions problem -- has to stop the
            # run, because carrying on would turn --rebuild into a resume over
            # exactly the stale vectors it was passed to remove.
            if COLLECTION_NAME in _collection_names(client):
                raise RuntimeError(
                    f"--rebuild could not drop the {COLLECTION_NAME!r} collection "
                    f"in {chroma_dir}, and it is still there. Close anything else "
                    f"holding the index and run again."
                ) from error

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Cosine because the vectors are normalised; leaving Chroma on its L2
        # default would score normalised vectors by a metric that disagrees with
        # the one the model was trained under.
        metadata={"hnsw:space": DISTANCE_METRIC},
    )

    if rebuild and collection.count():
        raise RuntimeError(
            f"--rebuild left {collection.count():,} vectors in {chroma_dir}. "
            f"Delete that directory by hand and run again, rather than "
            f"embedding on top of them."
        )
    return collection


def _existing_ids(collection) -> set[str]:
    """Which passages this index already holds, so a run can resume.

    Asked for once and kept, rather than per batch: an interrupted build is the
    normal reason to re-run, and thirty thousand ids is a set worth holding to
    avoid thirty thousand round trips.

    Nothing here swallows a read failure. "The index is empty" and "the index
    could not be read" are one empty set to the caller and opposite situations:
    the first means embed everything, the second means a corrupt or locked
    store has just cost hours of re-encoding and handed ``add()`` thirty
    thousand ids it already holds. Emptiness is settled by asking for the
    count, which leaves only a genuine read problem to raise.
    """
    if not collection.count():
        return set()
    try:
        return set(collection.get(include=[])["ids"])
    except (TypeError, ValueError):
        # Older chromadb rejects an empty include list. The default projection
        # returns the ids too; it just pays for the documents on the way.
        return set(collection.get()["ids"])


def build(
    tickers: list[str] | None = None,
    fiscal_years: range | list[int] | None = None,
    key_items_only: bool = False,
    rebuild: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
    threads: int | None = None,
    chunk_budget: int = CHUNK_CHAR_BUDGET,
    chunk_overlap: int = CHUNK_CHAR_OVERLAP,
    chroma_dir: Path = CHROMA_DIR,
    manifest_file: Path = MANIFEST_FILE,
) -> IndexManifest | None:
    """Embed the corpus into ``chroma_dir``, and write a manifest if it is whole.

    The filters are named parameters rather than a parsed namespace, so this is
    as usable from a notebook as from the command line, which is the shape
    ``passages.select`` uses one stage earlier.

    A run that is interrupted can simply be run again: whatever is already in
    the collection is skipped. That resume matches on ``chunk_id`` alone, which
    is only safe while the corpus has not moved under it -- a re-chunk that
    changes a passage's text without changing the count leaves most ids
    identical, and the old vector would be kept for the new text. So the corpus
    is fingerprinted before anything is embedded and compared against the
    manifest already on disk; a run that would resume onto a corpus that moved
    stops and says to pass ``rebuild``, rather than doing it quietly.

    ``chunk_budget`` and ``chunk_overlap`` are a fallback rather than the
    answer. The chunk stage records what it cut with, so the settings are read
    off the corpus and these are used only for filings written before that was
    recorded. A corpus cut two different ways is reported and recorded as
    neither, since the manifest has one budget field and labelling every sweep
    row with a number that is wrong for part of the corpus is worse than
    labelling none. ``resolve_chunk_settings`` holds that rule, so this index
    and the BM25 one apply it identically.

    Returns the manifest when one was written, and None when the index does not
    hold the whole corpus -- a narrowed run, or one interrupted partway. A
    manifest is a claim about an entire index, and a filtered walk cannot make
    it: build the full index, then run again with ``--tickers AAPL``, and the
    manifest would describe two thousand Apple passages sitting beside
    twenty-eight thousand vectors, which is worse than no manifest at all.

    The corpus is walked twice rather than held in memory. Reading it into a
    list costs a few hundred megabytes that are still resident when torch loads
    its weights on top, and on a 16 GB laptop already running an editor and a
    browser that was the difference between finishing and being killed by the
    system, twice, at 86% built. Disk is cheap here and memory is not: the
    second walk costs seconds against an encode measured in hours.
    """
    if batch_size < 1:
        raise ValueError(
            f"batch_size must be at least 1, got {batch_size}. A batch that "
            f"never fills would gather the whole corpus in memory and then be "
            f"rejected by the encoder."
        )

    narrowed = bool(tickers or fiscal_years or key_items_only)

    # First walk, and the only one over the whole corpus: the digest, the ids
    # and the filings accumulated together rather than in a loop each. All that
    # is kept is strings, so this is megabytes rather than the corpus.
    corpus_ids: set[str] = set()
    filings: set[str] = set()
    settings: set[tuple[int | None, int | None]] = set()
    n_rows = 0

    def observed():
        nonlocal n_rows
        for row in iter_chunks():
            n_rows += 1
            corpus_ids.add(row["chunk_id"])
            filings.add(row["accession_no"])
            settings.add((row.get("chunk_budget"), row.get("chunk_overlap")))
            yield row

    fingerprint = corpus_fingerprint(observed(), text_of=embed_text)
    if not n_rows:
        raise RuntimeError(
            "No passages found in data/processed/. Run the chunk stage first: "
            "python -m src.pipeline rebuild"
        )
    if len(corpus_ids) != n_rows:
        # Resume, the manifest and every id in the index all key on chunk_id
        # being unique. A duplicate would silently drop a passage from the index.
        raise RuntimeError(
            f"data/processed/ holds {n_rows:,} passages under only "
            f"{len(corpus_ids):,} distinct chunk_ids. Re-chunk before indexing: "
            f"python -m src.pipeline chunk --force"
        )
    n_filings = len(filings)
    print(
        f"corpus: {n_rows:,} passages from {n_filings} filings"
        f"  fingerprint {fingerprint[:12]}"
    )

    # Measured off the corpus where it recorded them, rather than taken from the
    # constants in force now, which is what the manifest's own docstring says
    # these fields are for.
    chunk_budget, chunk_overlap, note = resolve_chunk_settings(
        settings, chunk_budget, chunk_overlap
    )
    if note:
        print(f"  NOTE  {note}")

    # Before a single vector is written, not after: resuming onto a corpus that
    # has moved keeps the stale vector for every id that survived the re-chunk,
    # and nothing downstream can see that it happened.
    stored = read_manifest(manifest_file)
    if stored is not None and not rebuild:
        drift = stored.mismatches(
            corpus_fingerprint=fingerprint,
            model=EMBED_MODEL,
            dimensions=EMBED_DIMENSIONS,
        )
        if drift:
            detail = "\n".join(f"  - {problem}" for problem in drift)
            raise RuntimeError(
                f"The index beside {manifest_file} was built from something else:\n"
                f"{detail}\n"
                f"Resuming would keep the old vector for every passage whose "
                f"chunk_id survived the change, and then write a manifest saying "
                f"the index is current. Rebuild it instead:\n"
                f"  python -m src.retrieval embed --rebuild"
            )

    collection = _open_collection(chroma_dir, rebuild=rebuild)
    already = _existing_ids(collection)
    # Known exactly for a full run, and only by counting up for a narrowed one,
    # since the filtered subset is not knowable without walking it.
    n_pending = None if narrowed else len(corpus_ids - already)
    if already:
        pending = "an unknown number" if n_pending is None else f"{n_pending:,}"
        print(f"resuming: {len(already):,} already embedded, {pending} to go")

    started = time.perf_counter()
    done = 0
    model = None
    used_threads: int | None = None
    embedded: set[str] = set()
    # Passages the encoder had to cut short, collected as it goes. bge reads
    # EMBED_MAX_TOKENS and silently drops the rest, so without this the index
    # looks complete while part of a passage was never embedded -- and a table
    # passage is where it bites, since figures tokenise about twice as densely
    # as prose and the chunker's budget is in characters.
    oversized: list[tuple[str, int, str]] = []

    def flush(batch: list[dict]) -> None:
        """Encode one batch and add it, holding nothing after it returns."""
        nonlocal done, model, used_threads, started
        if model is None:
            # Loaded on the first batch rather than up front, so a run with
            # nothing pending never pays for torch, and a narrowed run does not
            # have to know its own size in advance in order to decide.
            print(f"loading {EMBED_MODEL} ...")
            model, used_threads = _load_model(threads)
            print(f"encoding on {used_threads} threads")
            started = time.perf_counter()

        texts = [embed_text(row) for row in batch]
        # Tokenising a batch we are about to encode anyway costs a fraction of
        # the encode, which is what makes measuring this affordable rather than
        # a separate pass over the corpus.
        for row, ids in zip(batch, model.tokenizer(texts)["input_ids"]):
            if len(ids) > EMBED_MAX_TOKENS:
                oversized.append(
                    (row["chunk_id"], len(ids), row.get("content_type", "prose"))
                )
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=EMBED_NORMALIZE,
            show_progress_bar=False,
        )
        collection.add(
            ids=[row["chunk_id"] for row in batch],
            embeddings=[vector.tolist() for vector in vectors],
            # The passage as the chunker wrote it. The header went into the
            # vector above and stops here.
            documents=[row["text"] for row in batch],
            metadatas=[metadata_for(row) for row in batch],
        )
        embedded.update(row["chunk_id"] for row in batch)
        done += len(batch)
        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        if n_pending:
            remaining = (n_pending - done) / rate if rate else 0.0
            print(
                f"  {done:,}/{n_pending:,} passages"
                f"  {rate:.1f}/s  eta {remaining / 60:.1f} min",
                flush=True,
            )
        else:
            print(
                f"  {done:,} passages  {rate:.1f}/s"
                f"  {elapsed / 60:.1f} min elapsed",
                flush=True,
            )

    # Second walk: the passages this run was asked for, one batch at a time.
    # Only the batch in hand is resident, so peak memory no longer grows with
    # the corpus and the model has room to load beside it.
    batch: list[dict] = []
    for row in iter_chunks(
        fiscal_years=fiscal_years,
        tickers=tickers,
        key_items_only=key_items_only,
    ):
        if row["chunk_id"] in already:
            continue
        batch.append(row)
        if len(batch) >= batch_size:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    if not done:
        print("nothing to embed: every passage asked for is already in the index")

    held = already | embedded
    if oversized:
        # Merged into whatever a previous run left, unless this run started
        # from an empty collection. A resumed build embeds only its remainder,
        # so overwriting would discard the record of every truncation before
        # the interruption -- the only record there is, since a truncated
        # vector carries no mark of its own.
        report_truncation(oversized, n_embedded=len(held), merge=not rebuild)
    elif rebuild:
        clear_truncation_report()

    manifest = IndexManifest(
        index_type=DENSE,
        path=manifest_path(chroma_dir),
        corpus_fingerprint=fingerprint,
        n_passages=n_rows,
        n_filings=n_filings,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=EMBED_MODEL,
        dimensions=EMBED_DIMENSIONS,
        chunk_budget=chunk_budget,
        chunk_overlap=chunk_overlap,
    )
    print(f"index:    {chroma_dir}")

    if held != corpus_ids:
        missing = len(corpus_ids - held)
        extra = len(held - corpus_ids)
        print()
        print(f"  WARNING  no manifest written. The index holds {len(held):,} of "
              f"the corpus's {n_rows:,} passages")
        print(f"           ({missing:,} not embedded, {extra:,} not in the corpus).")
        print("           A manifest describes a whole index and the stale-index "
              "check reads it,")
        print("           so a narrowed or interrupted build must not leave one. "
              "Run again")
        print("           without filters to finish the index and get a manifest.")
        return None

    write_manifest(manifest, manifest_file)
    print(f"manifest: {manifest_file}")
    return manifest


def report_truncation(
    oversized: list[tuple[str, int, str]],
    n_embedded: int,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
    merge: bool = True,
) -> Path:
    """Say which passages the encoder cut short, on screen and to a file.

    Loud on purpose. A truncated passage is not a failed one: it is indexed,
    searchable, and wrong in a way nothing downstream can see, because the text
    Chroma stores is whole while the vector represents only its first
    ``EMBED_MAX_TOKENS``. A retrieval miss caused by this looks like a retrieval
    miss caused by the model, which is the kind of confound an ablation cannot
    untangle after the fact.

    ``merge`` folds this run's findings into the file a previous run left,
    keyed on ``chunk_id``. A build that was interrupted and resumed sees only
    its own remainder, so overwriting would report the last few per cent as
    though it were the whole story, and the truncations from the rest would be
    gone -- permanently, since a truncated vector carries no mark of its own. A
    rebuild passes ``merge=False``, every vector in the index then being this
    run's.

    ``n_embedded`` is how many passages the index holds once this run is done,
    rather than how many this run encoded, so the percentage on screen means
    the corpus-level figure it reads as.

    The file goes to ``data/diagnostics/`` for the same reason the parse stage
    writes the HTML of a table it could not rebuild: it explains a run rather
    than feeding the next stage.
    """
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / TRUNCATION_FILE

    found = {
        chunk_id: {"chunk_id": chunk_id, "tokens": tokens, "content_type": kind}
        for chunk_id, tokens, kind in oversized
    }
    carried = 0
    if merge and path.exists():
        previous = json.loads(path.read_text(encoding="utf-8")).get("passages", [])
        kept = {row["chunk_id"]: row for row in previous}
        carried = len(set(kept) - set(found))
        kept.update(found)
        found = kept

    passages = sorted(found.values(), key=lambda row: -row["tokens"])
    tables = sum(1 for row in passages if row["content_type"] == "table")
    worst = max(row["tokens"] for row in passages)
    share = 100 * len(passages) / n_embedded if n_embedded else 0.0

    print()
    print(f"  WARNING  {len(passages):,} of {n_embedded:,} passages ({share:.1f}%) "
          f"exceeded {EMBED_MAX_TOKENS} tokens and were truncated by the encoder.")
    print(f"           {tables:,} of them are table passages. Worst: {worst:,} tokens.")
    if carried:
        print(f"           {carried:,} carried over from an earlier run of this index.")
    print("           Their stored text is whole; their vector is not.")

    path.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "max_tokens": EMBED_MAX_TOKENS,
                "n_embedded": n_embedded,
                "n_truncated": len(passages),
                "n_truncated_tables": tables,
                "worst_tokens": worst,
                "passages": passages,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"           Listed in {path}")
    return path


def clear_truncation_report(diagnostics_dir: Path = DIAGNOSTICS_DIR) -> None:
    """Drop the truncation report after a rebuild that produced none.

    A rebuild replaces every vector in the index, so a file left by the run
    before it describes passages that are no longer in there. Left in place it
    would read as a live finding about the index that is.
    """
    path = diagnostics_dir / TRUNCATION_FILE
    if path.exists():
        path.unlink()
        print(f"cleared:  {path} (nothing was truncated this time)")


def write_manifest(manifest: IndexManifest, path: Path = MANIFEST_FILE) -> Path:
    """Write the manifest as JSON, creating the directory if it is not there."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(manifest), indent=2) + "\n", encoding="utf-8")
    return path


def read_manifest(path: Path = MANIFEST_FILE) -> IndexManifest | None:
    """The manifest beside an index, or None when there is no index yet.

    None rather than an exception because "no index built" is a normal state a
    retriever has to report clearly, not an error in reading the file.
    """
    if not path.exists():
        return None
    return IndexManifest(**json.loads(path.read_text(encoding="utf-8")))
