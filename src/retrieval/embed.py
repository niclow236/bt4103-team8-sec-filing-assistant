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

The index is derived data: delete it and it rebuilds from ``data/processed/``.
What cannot be rebuilt is the knowledge of which corpus it came from, so an
``IndexManifest`` is written beside it, and a retriever compares that against
the corpus on disk before it will load one.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Iterable
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..config import DIAGNOSTICS_DIR, PROJECT_ROOT
from ..pipeline.chunk import iter_chunks
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
from .records import IndexManifest

# One collection holds the whole corpus. Splitting per company would make a
# cross-company question a fan-out over fifteen collections, and the metadata
# pre-filter already narrows a query to one company when it needs to.
COLLECTION_NAME = "passages"

# Beside the index rather than inside it: Chroma owns the contents of its own
# directory and is free to rewrite them, so a file that has to survive a
# rebuild does not belong in there.
MANIFEST_FILE = CHROMA_DIR.parent / "chroma.manifest.json"

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
    """
    return f"{PASSAGE_PREFIX}{context_header(row)}{row['text']}"


def corpus_fingerprint(rows: Iterable[dict]) -> str:
    """A digest identifying the exact set of passages an index was built from.

    This is the definition every stale-index check compares against, so it is
    worth saying what it does and does not catch. It hashes each passage's
    ``chunk_id`` together with its text, sorts those digests, then hashes the
    result. Sorting is what makes it a fingerprint of the corpus rather than of
    one walk over it: the same passages in a different order, which is all a
    renamed directory or a changed glob amounts to, give the same answer.

    It changes when a passage's text changes, when passages are added or
    removed, and when the chunker cuts the same filing differently, since that
    renumbers ``chunk_id``. It does not change when metadata alone changes: a
    corrected company name moves no vector, so an index is not stale for it.
    """
    digests = sorted(
        hashlib.sha256(
            f"{row['chunk_id']}\x00{row['text']}".encode("utf-8")
        ).hexdigest()
        for row in rows
    )
    total = hashlib.sha256()
    for digest in digests:
        total.update(digest.encode("ascii"))
    return total.hexdigest()


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
    """
    import torch

    resolved = threads or os.cpu_count()
    if resolved:
        torch.set_num_threads(resolved)
    return resolved


def _load_model():
    """Import and load the encoder, late and with a usable failure.

    sentence-transformers pulls in torch, which is seconds of import time and a
    multi-gigabyte install. Importing it at module scope would make ``--help``
    pay for both, and would turn a missing dependency into a traceback from an
    import three levels down rather than a sentence saying what to install.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:   # pragma: no cover - depends on the install
        raise RuntimeError(
            "sentence-transformers is not installed. "
            "Run: pip install -r requirements.txt"
        ) from error

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
    return model


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
        except Exception:
            # Nothing to drop on the first run, and Chroma's "no such
            # collection" is a different exception per version.
            pass

    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        # Cosine because the vectors are normalised; leaving Chroma on its L2
        # default would score normalised vectors by a metric that disagrees with
        # the one the model was trained under.
        metadata={"hnsw:space": DISTANCE_METRIC},
    )


def _existing_ids(collection) -> set[str]:
    """Which passages this index already holds, so a run can resume.

    Asked for once and kept, rather than per batch: an interrupted build is the
    normal reason to re-run, and thirty thousand ids is a set worth holding to
    avoid thirty thousand round trips.
    """
    try:
        return set(collection.get(include=[])["ids"])
    except Exception:
        return set()


def build(
    tickers: list[str] | None = None,
    fiscal_years: range | list[int] | None = None,
    key_items_only: bool = False,
    rebuild: bool = False,
    batch_size: int = EMBED_BATCH_SIZE,
    threads: int | None = None,
    chroma_dir: Path = CHROMA_DIR,
    manifest_file: Path = MANIFEST_FILE,
) -> IndexManifest:
    """Embed the corpus into ``chroma_dir`` and write the manifest beside it.

    The filters are named parameters rather than a parsed namespace, so this is
    as usable from a notebook as from the command line, which is the shape
    ``passages.select`` uses one stage earlier.

    A run that is interrupted can simply be run again: whatever is already in
    the collection is skipped. ``rebuild`` drops the collection first, which is
    what to use when the corpus has been re-chunked rather than merely extended.

    The corpus is walked twice rather than held in memory. Reading it into a
    list costs a few hundred megabytes that are still resident when torch loads
    its weights on top, and on a 16 GB laptop already running an editor and a
    browser that was the difference between finishing and being killed by the
    system, twice, at 86% built. Disk is cheap here and memory is not: the
    second walk costs seconds against an encode measured in hours.
    """
    def passages():
        return iter_chunks(
            fiscal_years=fiscal_years,
            tickers=tickers,
            key_items_only=key_items_only,
        )

    # First walk: what the corpus is. Only a digest per passage and the set of
    # filings is kept, so this is megabytes rather than the whole corpus.
    fingerprint = corpus_fingerprint(passages())
    n_rows = 0
    filings: set[str] = set()
    for row in passages():
        n_rows += 1
        filings.add(row["accession_no"])
    if not n_rows:
        raise RuntimeError(
            "No passages found in data/processed/. Run the chunk stage first: "
            "python -m src.pipeline rebuild"
        )
    n_filings = len(filings)
    print(
        f"corpus: {n_rows:,} passages from {n_filings} filings"
        f"  fingerprint {fingerprint[:12]}"
    )

    collection = _open_collection(chroma_dir, rebuild=rebuild)
    already = _existing_ids(collection)
    n_pending = sum(1 for row in passages() if row["chunk_id"] not in already)
    if already:
        print(f"resuming: {len(already):,} already embedded, {n_pending:,} to go")

    if n_pending:
        used = _use_threads(threads)
        print(f"loading {EMBED_MODEL} on {used} threads ...")
        model = _load_model()

        started = time.perf_counter()
        done = 0
        # Passages the encoder had to cut short, collected as it goes. bge reads
        # EMBED_MAX_TOKENS and silently drops the rest, so without this the
        # index looks complete while part of a passage was never embedded --
        # and a table passage is where it bites, since figures tokenise about
        # twice as densely as prose and the chunker's budget is in characters.
        oversized: list[tuple[str, int, str]] = []

        def flush(batch: list[dict]) -> None:
            """Encode one batch and add it, holding nothing after it returns."""
            nonlocal done
            texts = [embed_text(row) for row in batch]
            # Tokenising a batch we are about to encode anyway costs a fraction
            # of the encode, which is what makes measuring this affordable
            # rather than a separate pass over the corpus.
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
            done += len(batch)
            elapsed = time.perf_counter() - started
            rate = done / elapsed if elapsed else 0.0
            remaining = (n_pending - done) / rate if rate else 0.0
            print(
                f"  {done:,}/{n_pending:,} passages"
                f"  {rate:.1f}/s  eta {remaining / 60:.1f} min",
                flush=True,
            )

        # Second walk: the passages themselves, one batch at a time. Only the
        # batch in hand is resident, so peak memory no longer grows with the
        # corpus and the model has room to load beside it.
        batch: list[dict] = []
        for row in passages():
            if row["chunk_id"] in already:
                continue
            batch.append(row)
            if len(batch) == batch_size:
                flush(batch)
                batch = []
        if batch:
            flush(batch)

        if oversized:
            report_truncation(oversized, n_pending)
    else:
        print("nothing to embed: every passage is already in the index")

    manifest = IndexManifest(
        index_type=DENSE,
        path=chroma_dir.relative_to(PROJECT_ROOT).as_posix(),
        corpus_fingerprint=fingerprint,
        n_passages=n_rows,
        n_filings=n_filings,
        built_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        model=EMBED_MODEL,
        dimensions=EMBED_DIMENSIONS,
        chunk_budget=CHUNK_CHAR_BUDGET,
        chunk_overlap=CHUNK_CHAR_OVERLAP,
    )
    write_manifest(manifest, manifest_file)
    print(f"index:    {chroma_dir}")
    print(f"manifest: {manifest_file}")
    return manifest


def report_truncation(
    oversized: list[tuple[str, int, str]],
    n_embedded: int,
    diagnostics_dir: Path = DIAGNOSTICS_DIR,
) -> Path:
    """Say which passages the encoder cut short, on screen and to a file.

    Loud on purpose. A truncated passage is not a failed one: it is indexed,
    searchable, and wrong in a way nothing downstream can see, because the text
    Chroma stores is whole while the vector represents only its first
    ``EMBED_MAX_TOKENS``. A retrieval miss caused by this looks like a retrieval
    miss caused by the model, which is the kind of confound an ablation cannot
    untangle after the fact.

    The file goes to ``data/diagnostics/`` for the same reason the parse stage
    writes the HTML of a table it could not rebuild: it explains a run rather
    than feeding the next stage.
    """
    tables = sum(1 for _, _, kind in oversized if kind == "table")
    worst = max(tokens for _, tokens, _ in oversized)
    share = 100 * len(oversized) / n_embedded if n_embedded else 0.0

    print()
    print(f"  WARNING  {len(oversized):,} of {n_embedded:,} passages ({share:.1f}%) "
          f"exceeded {EMBED_MAX_TOKENS} tokens and were truncated by the encoder.")
    print(f"           {tables:,} of them are table passages. Worst: {worst:,} tokens.")
    print("           Their stored text is whole; their vector is not.")

    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    path = diagnostics_dir / "embed-truncated.json"
    path.write_text(
        json.dumps(
            {
                "model": EMBED_MODEL,
                "max_tokens": EMBED_MAX_TOKENS,
                "n_embedded": n_embedded,
                "n_truncated": len(oversized),
                "n_truncated_tables": tables,
                "worst_tokens": worst,
                "passages": [
                    {"chunk_id": cid, "tokens": tokens, "content_type": kind}
                    for cid, tokens, kind in sorted(
                        oversized, key=lambda row: -row[1]
                    )
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
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
