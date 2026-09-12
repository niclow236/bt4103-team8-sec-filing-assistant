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

**Every vector records what it was encoded from.** Beside the citation
metadata, each one stores the digest of its ``embed_text`` (``digest``), how
many tokens that text ran to (``n_tokens``), and the model that encoded it
(``embed_model``). That is what makes the index checkable one passage at a time
against the corpus on disk, with or without a manifest:

- A resumed build compares digests, so a passage whose text changed since its
  vector was written is re-encoded rather than skipped because its id survived.
- A truncated vector is marked on the vector itself, so the truncation report
  is read back from the index and cannot lose entries to a resume.
- An index encoded by a different model is refused rather than extended.

Because the header reaches the encoder, the digest covers metadata as well as
text: rename a company in the ticker map and each of its vectors is correctly
seen as stale. See :func:`records.corpus_fingerprint`.

**The manifest describes the index as it actually is.** It is deleted before a
build first changes the collection and written again, from the digests the
collection holds, when the build completes. So a manifest on disk was always
written by a run that finished, after the index last changed, and its
fingerprint matches the corpus only when every passage is present and current.
An interrupted build leaves no manifest; a narrowed or partial one leaves a
manifest that truthfully fails to match. :func:`check_index` is the check a
retriever runs before searching.

Every file this stage writes about an index sits beside that index, named after
it: ``chroma.manifest.json`` and ``chroma.truncated.json`` for
``data/index/chroma/``. A build pointed at a scratch directory therefore cannot
read or overwrite the project's own.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import PROCESSED_DIR
from ..pipeline.chunk import iter_chunks, resolve_chunk_settings
from .constants import (
    CHROMA_DIR,
    CONTEXT_HEADER,
    DENSE,
    DISTANCE_METRIC,
    EMBED_BATCH_SIZE,
    EMBED_SORT_WINDOW,
    EMBED_DIMENSIONS,
    EMBED_MAX_TOKENS,
    EMBED_MODEL,
    EMBED_NORMALIZE,
    PASSAGE_PREFIX,
)
from .records import IndexManifest, Query, fingerprint_of, manifest_path, passage_digest

# One collection holds the whole corpus. Splitting per company would make a
# cross-company question a fan-out over fifteen collections, and the metadata
# pre-filter already narrows a query to one company when it needs to.
COLLECTION_NAME = "passages"

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

# Fields the index writes about each vector, as opposed to fields copied from
# the passage. Named once so the builder, the check and the inverse of
# metadata_for agree on which is which.
DIGEST_FIELD = "digest"
TOKENS_FIELD = "n_tokens"
MODEL_FIELD = "embed_model"
INDEX_FIELDS = (DIGEST_FIELD, TOKENS_FIELD, MODEL_FIELD)

# What metadata_for drops as empty, and what an iter_chunks row holds in its
# place, so a stored record can be turned back into that row exactly.
_DROPPED_DEFAULTS: dict[str, Any] = {
    "part": None, "item": None, "heading": None, "title": "",
    "table_index": None, "table_caption": "", "period_of_report": "",
    "fiscal_year": None,
}

# Records read per request when scanning the collection. Only three short
# strings are kept from each, so the page size bounds the transient cost of
# reading full metadata rather than what is held afterwards.
SCAN_PAGE = 5_000


def manifest_file_for(chroma_dir: Path) -> Path:
    """Where the manifest of the index in ``chroma_dir`` lives: beside it.

    Derived rather than passed separately, so no caller can pair one index with
    another index's manifest. Beside the directory rather than inside it,
    because Chroma owns its directory's contents and may rewrite them.
    """
    return chroma_dir.parent / f"{chroma_dir.name}.manifest.json"


def truncation_file_for(chroma_dir: Path) -> Path:
    """Where the truncation report of the index in ``chroma_dir`` lives."""
    return chroma_dir.parent / f"{chroma_dir.name}.truncated.json"


MANIFEST_FILE = manifest_file_for(CHROMA_DIR)


def context_header(row: Mapping[str, Any]) -> str:
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


def embed_text(row: Mapping[str, Any]) -> str:
    """What actually goes to the encoder: prefix, header, then the passage.

    ``PASSAGE_PREFIX`` is empty for bge, which asks for a prefix on the query
    side only. It is applied here anyway so that swapping in a model wanting one
    is a change to constants.py rather than to this file.

    This is also what the dense index fingerprints, which is the only reason it
    matters that this function is pure: hand it the same row twice and the
    stale-index check has to get the same string back both times.
    """
    return f"{PASSAGE_PREFIX}{context_header(row)}{row['text']}"


def digest_of(row: Mapping[str, Any]) -> str:
    """The digest a current vector for this passage would carry."""
    return passage_digest(row["chunk_id"], embed_text(row))


def metadata_for(row: Mapping[str, Any]) -> dict[str, Any]:
    """The row reduced to what Chroma will store alongside a vector.

    Chroma takes scalars only, so a None or a list has to go somewhere or go
    away. Empty values are dropped rather than stored as "": a filter on a field
    that is absent should not match a passage that merely has nothing in it.
    ``incorporated_into`` is a list, so it is joined, which keeps it readable in
    a citation without pretending it is filterable. :func:`chunk_from_record`
    undoes all of this.
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


def chunk_from_record(
    chunk_id: str, document: str, metadata: Mapping[str, Any]
) -> dict[str, Any]:
    """One stored record turned back into the row ``iter_chunks`` would yield.

    The inverse of :func:`metadata_for`, kept beside it so the two directions
    cannot drift. This is what lets the dense retriever hand
    ``RetrievedPassage.from_chunk`` the same shape BM25 hands it: without it, a
    passage whose Item or title was empty would reach ``from_chunk`` with the
    key missing and raise, and one retriever would cite a passage differently
    from another.

    The index's own fields are left out, since they describe the vector rather
    than the passage.
    """
    row: dict[str, Any] = dict(_DROPPED_DEFAULTS)
    row.update({key: value for key, value in metadata.items() if key not in INDEX_FIELDS})
    joined = row.pop("incorporated_into", "")
    row["incorporated_into"] = joined.split(",") if joined else []
    row["chunk_id"] = chunk_id
    row["text"] = document
    return row


def where_for(query: Query) -> dict[str, Any] | None:
    """The Chroma ``where`` clause that applies a Query's filters in the index.

    The dense counterpart of ``base.matches``, which is how BM25 applies the
    same filters to rows. The two have to select exactly the same passages for
    every Query, or the ablation compares the methods over different corpora
    and reports the difference as a difference in retrieval. It is written here
    rather than in the retriever because what makes it correct is how
    :func:`metadata_for` stores each field: values as the corpus holds them,
    and an empty field absent rather than stored as "", so a filter on it
    excludes the passage exactly as ``matches`` does.

    None when the Query sets no filter, which Chroma reads as unrestricted.
    """
    clauses = [
        {field: {"$in": value}} if isinstance(value, list) else {field: value}
        for field, value in query.filters.items()
    ]
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


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


def open_collection(chroma_dir: Path = CHROMA_DIR, rebuild: bool = False, create: bool = True):
    """The persistent collection in ``chroma_dir``.

    ``create=False`` is for a reader: it returns None rather than creating an
    empty collection, and creates no directory, so checking for an index never
    leaves one behind. ``rebuild`` drops the collection first.
    """
    try:
        import chromadb
    except ImportError as error:   # pragma: no cover - depends on the install
        raise RuntimeError(
            "chromadb is not installed. Run: pip install -r requirements.txt"
        ) from error

    if not create:
        if not chroma_dir.is_dir():
            return None
        client = chromadb.PersistentClient(path=str(chroma_dir))
        if COLLECTION_NAME not in _collection_names(client):
            return None
        return client.get_collection(COLLECTION_NAME)

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


def _scan(collection) -> dict[str, tuple[str | None, str | None, str | None]]:
    """What every vector says about itself: digest, filing, and model.

    Read in pages, keeping three short strings per vector rather than full
    metadata, so this costs a few megabytes and is released before the model
    loads. Nothing here swallows a read failure: "the index is empty" and "the
    index could not be read" would otherwise look identical, and the second
    would cost a full re-encode.
    """
    held: dict[str, tuple[str | None, str | None, str | None]] = {}
    total = collection.count()
    for offset in range(0, total, SCAN_PAGE):
        page = collection.get(limit=SCAN_PAGE, offset=offset, include=["metadatas"])
        for chunk_id, metadata in zip(page["ids"], page["metadatas"]):
            metadata = metadata or {}
            held[chunk_id] = (
                metadata.get(DIGEST_FIELD),
                metadata.get("accession_no"),
                metadata.get(MODEL_FIELD),
            )
    return held


def _corpus_digests(processed_dir: Path) -> tuple[dict[str, str], set, int]:
    """Walk the corpus once: each passage's digest, the chunker settings seen,
    and the row count.

    The count is returned separately from the dict so a duplicated chunk_id,
    which the dict would silently collapse, can be detected by the caller.
    """
    digests: dict[str, str] = {}
    settings: set[tuple[int | None, int | None]] = set()
    n_rows = 0
    for row in iter_chunks(processed_dir=processed_dir):
        n_rows += 1
        digests[row["chunk_id"]] = digest_of(row)
        settings.add((row.get("chunk_budget"), row.get("chunk_overlap")))
    return digests, settings, n_rows


def _differences(
    held: Mapping[str, tuple[str | None, ...]], corpus: Mapping[str, str]
) -> dict[str, int]:
    """How an index departs from the corpus, counted in the three ways it can."""
    missing = sum(1 for chunk_id in corpus if chunk_id not in held)
    stale = sum(
        1 for chunk_id, entry in held.items()
        if chunk_id in corpus and entry[0] != corpus[chunk_id]
    )
    orphaned = sum(1 for chunk_id in held if chunk_id not in corpus)
    return {"missing": missing, "stale": stale, "orphaned": orphaned}


def _describe(differences: Mapping[str, int]) -> str:
    return (
        f"{differences['missing']:,} passages missing, "
        f"{differences['stale']:,} stale, "
        f"{differences['orphaned']:,} no longer in the corpus"
    )


def build(
    tickers: list[str] | None = None,
    fiscal_years: range | list[int] | None = None,
    key_items_only: bool = False,
    rebuild: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
    threads: int | None = None,
    chroma_dir: Path = CHROMA_DIR,
    processed_dir: Path = PROCESSED_DIR,
    sort_window: int = EMBED_SORT_WINDOW,
) -> IndexManifest:
    """Bring the index in ``chroma_dir`` up to date with ``processed_dir``.

    ``batch_size`` is what the encoder runs at once; ``sort_window`` is how many
    passages it is handed per call, which it sorts by length before batching.
    See ``constants.EMBED_SORT_WINDOW``: the window changes speed, never results.

    The filters are named parameters rather than a parsed namespace, so this is
    as usable from a notebook as from the command line, which is the shape
    ``passages.select`` uses one stage earlier.

    What a run does to each passage it is asked for:

    - missing from the index: encoded and added;
    - present with the digest of its current text: left alone;
    - present with a different digest: its vector is deleted and re-encoded.
      Deleted and re-added rather than upserted, because Chroma merges metadata
      on upsert, so a field the new passage no longer has would survive from
      the old one.

    A run with no filters also removes vectors whose passage is no longer in
    the corpus. A narrowed run leaves everything outside its filter as it
    found it, including anything stale there; the manifest it writes records
    that honestly and will not match the corpus until a full run completes.

    So an interrupted build is resumed by running it again, and that is correct
    even if the corpus was re-chunked in between: nothing is trusted because its
    id survived. ``rebuild`` drops the collection first, for when every vector
    should be re-encoded regardless -- a changed model, for one, which this
    refuses to mix into an existing index.

    The corpus is walked twice rather than held in memory. Reading it into a
    list costs a few hundred megabytes that are still resident when torch loads
    its weights on top, and on a 16 GB laptop already running an editor and a
    browser that was the difference between finishing and being killed by the
    system, twice, at 86% built. Disk is cheap here and memory is not: the
    second walk costs seconds against an encode measured in hours.
    """
    if batch_size < 1 or sort_window < 1:
        raise ValueError(
            f"batch_size and sort_window must be at least 1, got {batch_size} and "
            f"{sort_window}. A batch that never fills would gather the whole corpus "
            f"in memory and then be rejected by the encoder."
        )
    window = max(batch_size, sort_window)

    narrowed = bool(tickers or fiscal_years or key_items_only)
    manifest_file = manifest_file_for(chroma_dir)

    # First walk, over the whole corpus whatever the filters say: the digests
    # are what every decision below is made against.
    corpus, settings, n_rows = _corpus_digests(processed_dir)
    if not n_rows:
        raise RuntimeError(
            f"No passages found in {processed_dir}. Run the chunk stage first: "
            f"python -m src.pipeline rebuild"
        )
    if len(corpus) != n_rows:
        # Resume, the manifest and every id in the index all key on chunk_id
        # being unique. A duplicate would silently drop a passage from the index.
        raise RuntimeError(
            f"{processed_dir} holds {n_rows:,} passages under only "
            f"{len(corpus):,} distinct chunk_ids. Re-chunk before indexing: "
            f"python -m src.pipeline chunk --force"
        )
    corpus_fingerprint = fingerprint_of(corpus.values())
    print(f"corpus:   {n_rows:,} passages  fingerprint {corpus_fingerprint[:12]}")

    chunk_budget, chunk_overlap, note = resolve_chunk_settings(settings)
    if note:
        print(f"  NOTE  {note}")

    invalidated = False

    def invalidate() -> None:
        """Remove the manifest before the index first changes.

        From this point until a new manifest is written, the index is between
        states, and a manifest describing the old state would certify it. So an
        interrupted run leaves no manifest, and a retriever refuses the index
        rather than loading half of it.
        """
        nonlocal invalidated
        if not invalidated:
            manifest_file.unlink(missing_ok=True)
            invalidated = True

    if rebuild:
        invalidate()
        collection = open_collection(chroma_dir, rebuild=True)
        held: dict[str, tuple[str | None, str | None, str | None]] = {}
    else:
        collection = open_collection(chroma_dir)
        held = _scan(collection)
        # Refused rather than repaired: every vector would change, possibly its
        # width too, and that is a decision to make on purpose.
        foreign = sum(1 for entry in held.values() if entry[2] != EMBED_MODEL)
        if foreign:
            models = sorted({str(entry[2]) for entry in held.values() if entry[2] != EMBED_MODEL})
            raise RuntimeError(
                f"{foreign:,} of the {len(held):,} vectors in {chroma_dir} were not "
                f"encoded by {EMBED_MODEL} (found: {', '.join(models)}; 'None' means "
                f"built before this stage recorded the model). Mixing encoders in "
                f"one index makes their scores incomparable. Rebuild it:\n"
                f"  python -m src.retrieval embed --rebuild"
            )

    before = _differences(held, corpus)
    if held:
        print(f"index:    {len(held):,} vectors, {_describe(before)}")

    # A full run makes the index equal to the corpus, so anything the corpus no
    # longer holds goes. A narrowed run is not entitled to judge the rest.
    orphaned = [chunk_id for chunk_id in held if chunk_id not in corpus]
    if orphaned and not narrowed:
        invalidate()
        for start in range(0, len(orphaned), SCAN_PAGE):
            collection.delete(ids=orphaned[start:start + SCAN_PAGE])
        for chunk_id in orphaned:
            del held[chunk_id]
        print(f"removed:  {len(orphaned):,} vectors whose passage is no longer in the corpus")

    # Exact for a full run. A narrowed run counts up instead, since which of
    # its passages need work is not known without walking them.
    n_pending = None if narrowed else before["missing"] + before["stale"]

    started = time.perf_counter()
    done = 0
    replaced = 0
    model = None

    def flush(batch: list[dict]) -> None:
        """Encode one batch and write it, holding nothing after it returns."""
        nonlocal done, replaced, model, started
        invalidate()
        if model is None:
            # Loaded on the first batch rather than up front, so a run with
            # nothing to do never pays for torch.
            print(f"loading {EMBED_MODEL} ...")
            model, used_threads = _load_model(threads)
            print(f"encoding on {used_threads} threads")
            started = time.perf_counter()

        texts = [embed_text(row) for row in batch]
        # Tokenising a batch about to be encoded anyway costs a fraction of the
        # encode. The count goes onto the vector: bge reads EMBED_MAX_TOKENS and
        # silently drops the rest, so this is the only record that a vector
        # represents less than its stored text.
        counts = [len(ids) for ids in model.tokenizer(texts)["input_ids"]]
        vectors = model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=EMBED_NORMALIZE,
            show_progress_bar=False,
        )
        ids = [row["chunk_id"] for row in batch]
        # Hashed from the text actually encoded, not from the first walk, so a
        # corpus rewritten between the two walks shows up as a mismatch at the
        # end rather than as a vector certified for text it never saw.
        digests = [passage_digest(chunk_id, text) for chunk_id, text in zip(ids, texts)]
        replacing = [chunk_id for chunk_id in ids if chunk_id in held]
        if replacing:
            collection.delete(ids=replacing)
        collection.add(
            ids=ids,
            embeddings=[vector.tolist() for vector in vectors],
            # The passage as the chunker wrote it. The header went into the
            # vector above and stops here.
            documents=[row["text"] for row in batch],
            metadatas=[
                {**metadata_for(row), DIGEST_FIELD: digest,
                 TOKENS_FIELD: count, MODEL_FIELD: EMBED_MODEL}
                for row, digest, count in zip(batch, digests, counts)
            ],
        )
        for row, digest in zip(batch, digests):
            held[row["chunk_id"]] = (digest, row["accession_no"], EMBED_MODEL)
        done += len(batch)
        replaced += len(replacing)

        elapsed = time.perf_counter() - started
        rate = done / elapsed if elapsed else 0.0
        if n_pending:
            remaining = (n_pending - done) / rate if rate else 0.0
            print(f"  {done:,}/{n_pending:,} passages  {rate:.1f}/s"
                  f"  eta {remaining / 60:.1f} min", flush=True)
        else:
            print(f"  {done:,} passages  {rate:.1f}/s"
                  f"  {elapsed / 60:.1f} min elapsed", flush=True)

    # Second walk: the passages this run was asked for, one window at a time.
    # Only the window in hand is resident, so peak memory does not grow with the
    # corpus and the model has room to load beside it.
    batch: list[dict] = []
    for row in iter_chunks(
        processed_dir=processed_dir,
        fiscal_years=fiscal_years,
        tickers=tickers,
        key_items_only=key_items_only,
    ):
        entry = held.get(row["chunk_id"])
        if entry is not None and entry[0] == corpus.get(row["chunk_id"]):
            continue
        batch.append(row)
        if len(batch) >= window:
            flush(batch)
            batch = []
    if batch:
        flush(batch)

    if done:
        print(f"embedded: {done:,} passages ({replaced:,} of them replacing a stale vector)")
    else:
        print("nothing to embed: every passage asked for is current in the index")

    # The collection should now hold exactly what this run believes it does. If
    # it does not, something else wrote to it, and no manifest can describe it.
    if collection.count() != len(held):
        raise RuntimeError(
            f"{chroma_dir} holds {collection.count():,} vectors but this run "
            f"accounted for {len(held):,}. Something else changed the index while "
            f"it was building; rebuild it: python -m src.retrieval embed --rebuild"
        )

    # From the index's own digests, not the corpus walk: the manifest says what
    # the index holds, and matches the corpus only if the two are the same.
    manifest = IndexManifest(
        index_type=DENSE,
        path=manifest_path(chroma_dir),
        corpus_fingerprint=fingerprint_of(entry[0] or "" for entry in held.values()),
        n_passages=len(held),
        n_filings=len({entry[1] for entry in held.values()}),
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=EMBED_MODEL,
        dimensions=EMBED_DIMENSIONS,
        chunk_budget=chunk_budget,
        chunk_overlap=chunk_overlap,
    )
    write_manifest(manifest, manifest_file)
    write_truncation_report(collection, chroma_dir)

    after = _differences(held, corpus)
    print(f"index:    {chroma_dir}")
    print(f"manifest: {manifest_file}")
    if manifest.corpus_fingerprint == corpus_fingerprint:
        print(f"status:   current, {len(held):,} passages from "
              f"{manifest.n_filings} filings")
    else:
        print(f"status:   does NOT match the corpus ({_describe(after)}).")
        print("          The manifest records the index as it is, so a retriever will")
        print("          refuse it until a run without filters completes.")
    return manifest


def check_index(
    chroma_dir: Path = CHROMA_DIR,
    processed_dir: Path = PROCESSED_DIR,
) -> list[str]:
    """Every reason the index in ``chroma_dir`` should not be searched.

    Empty when it is safe to load. This is the check a dense retriever runs
    before its first query, written here beside the builder that defines what
    a current index is, so the two cannot disagree about it.

    It compares three things, and each catches something the others do not:
    the manifest against the constants (model and width), the digests the
    vectors carry against the corpus on disk (missing, stale and orphaned
    passages, counted exactly), and the manifest against those same digests (a
    manifest that does not describe the index beside it).
    """
    manifest_file = manifest_file_for(chroma_dir)
    manifest = read_manifest(manifest_file)
    if manifest is None:
        return [
            f"no manifest at {manifest_file}: the index was never completed, or the "
            f"last build that changed it did not finish. Run: python -m src.retrieval embed"
        ]

    problems = manifest.mismatches(model=EMBED_MODEL, dimensions=EMBED_DIMENSIONS)

    collection = open_collection(chroma_dir, create=False)
    if collection is None:
        return problems + [f"a manifest exists but there is no collection in {chroma_dir}"]
    held = _scan(collection)

    foreign = sum(1 for entry in held.values() if entry[2] != EMBED_MODEL)
    if foreign:
        problems.append(f"{foreign:,} vectors were not encoded by {EMBED_MODEL}")

    index_fingerprint = fingerprint_of(entry[0] or "" for entry in held.values())
    if index_fingerprint != manifest.corpus_fingerprint:
        problems.append(
            "the manifest does not describe the vectors beside it; rebuild the index"
        )

    corpus, _, _ = _corpus_digests(processed_dir)
    if index_fingerprint != fingerprint_of(corpus.values()):
        problems.append(f"the index does not match the corpus: {_describe(_differences(held, corpus))}")
    return problems


def write_truncation_report(collection, chroma_dir: Path) -> Path | None:
    """Say which vectors the encoder cut short, read back from the index itself.

    Loud on purpose. A truncated passage is not a failed one: it is indexed,
    searchable, and wrong in a way nothing downstream can see, because the text
    Chroma stores is whole while the vector represents only its first
    ``EMBED_MAX_TOKENS``. A retrieval miss caused by this looks like a retrieval
    miss caused by the model, which is the kind of confound an ablation cannot
    untangle after the fact.

    Read from the ``n_tokens`` each vector carries rather than collected during
    the run, so it describes every vector in the index however many runs wrote
    them, and a resumed build cannot report only its own remainder. Removed when
    nothing is truncated, so a stale report never reads as a live finding.
    """
    path = truncation_file_for(chroma_dir)
    found = collection.get(
        where={TOKENS_FIELD: {"$gt": EMBED_MAX_TOKENS}}, include=["metadatas"]
    )
    if not found["ids"]:
        path.unlink(missing_ok=True)
        return None

    passages = sorted(
        (
            {"chunk_id": chunk_id, "tokens": metadata[TOKENS_FIELD],
             "content_type": metadata.get("content_type", "prose")}
            for chunk_id, metadata in zip(found["ids"], found["metadatas"])
        ),
        key=lambda row: (-row["tokens"], row["chunk_id"]),
    )
    n_indexed = collection.count()
    tables = sum(1 for row in passages if row["content_type"] == "table")
    worst = passages[0]["tokens"]
    share = 100 * len(passages) / n_indexed if n_indexed else 0.0

    path.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "max_tokens": EMBED_MAX_TOKENS,
                "n_indexed": n_indexed,
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
    print()
    print(f"  WARNING  {len(passages):,} of {n_indexed:,} vectors ({share:.1f}%) were "
          f"truncated at {EMBED_MAX_TOKENS} tokens by the encoder.")
    print(f"           {tables:,} of them are table passages. Worst: {worst:,} tokens.")
    print("           Their stored text is whole; their vector is not.")
    print(f"           Listed in {path}")
    return path


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
