"""The demo's pinned snapshot (#51): what the demo serves, and how a machine gets it.

A live demo that rebuilds an index or downloads a model when it starts can fail
in the room. So the demo serves one fixed snapshot, named here, and a machine is
made ready for it before the day rather than on it:

- the configuration: C4, the row the report recommends and the app opens on;
- the corpus: the frozen evaluation data v1 release, 28,289 passages from 75
  filings, which every reported result was measured on;
- the indexes: built from that corpus, and identified by the corpus
  fingerprints the final study recorded for them
  (``docs/evaluation/full-matrix-20261021-manifest.json``), so an index built
  from it today is checked against the one that was measured, not merely
  against itself;
- the facts store: the release's own ``facts.parquet``, by its hash;
- the encoder: ``bge-base-en-v1.5`` at the revision every run recorded;
- the answer model: SOCLAAS ``qwen3.6:35b``, with Mistral ``ministral-8b-2512``
  as the fallback, as the team settled in #179.

:func:`check` says whether this machine holds all of it, and sends nothing over
the network. :func:`prepare` makes it hold it: it fetches and verifies the
corpus, rebuilds only what does not match, and caches the encoder. Nothing here
starts the app; ``src/demo/cli.py`` does, after :func:`check` passes.

The pins are literals rather than imports of the constants they describe. If
someone changes ``EMBED_MODEL`` or a provider's default model, the snapshot
does not silently follow: :func:`check` reports the difference, because the
demo would no longer be showing what was measured.
"""

from __future__ import annotations

import hashlib
import os
import pickle
import shutil
import tarfile
import urllib.request
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any

from src.config import BM25_INDEX_FILE, CHROMA_DIR, DATA_DIR, PROCESSED_DIR, load_env

GITHUB_REPOSITORY = "niclow236/bt4103-team8-sec-filing-assistant"


@dataclass(frozen=True)
class DemoPin:
    """Everything the demo serves, named exactly."""

    config_id: str = "C4"

    # The frozen evaluation data v1 release: the corpus and facts store every
    # reported result used. Its archive's hash is the release's own SHA256SUMS.
    corpus_release: str = "data/evaluation-v1"
    corpus_asset: str = "sec-eval-150-12579-v1.tar.gz"
    corpus_sha256: str = "a716519e37ae465500294312330748f9d488400ae7ead26ce4bb54e2bfec6d77"
    passages: int = 28289
    filings: int = 75

    # The indexes the final study searched, as their manifests recorded them.
    # BM25 hashes the passage text, so its fingerprint is also the corpus's.
    bm25_fingerprint: str = "94c37f133597e1c47d885cfa8012f0b952df1f795d1f0348d2a7781eea86840a"
    dense_fingerprint: str = "865f2e8d72f0d58b8e0aac54c6be809291e137e4e4b4d59f21462cbba1c6016d"
    facts_sha256: str = "fead73bfc896a62ab0051cefbe383d2d12028127d517811c043eb2de8786153e"

    encoder: str = "BAAI/bge-base-en-v1.5"
    encoder_revision: str = "a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
    dimensions: int = 768
    max_tokens: int = 512

    provider: str = "soclaas"
    model: str = "qwen3.6:35b"
    fallback_provider: str = "mistral"
    fallback_model: str = "ministral-8b-2512"

    @property
    def corpus_url(self) -> str:
        """Where the release asset is downloaded from. The repository is public,
        so this needs no ``gh`` and no login."""
        return (f"https://github.com/{GITHUB_REPOSITORY}/releases/download/"
                f"{self.corpus_release}/{self.corpus_asset}")


PIN = DemoPin()


@dataclass(frozen=True)
class Layout:
    """Where the snapshot lives on disk. The app reads the defaults; a test
    passes a temporary folder."""

    processed_dir: Path = PROCESSED_DIR
    bm25_index: Path = BM25_INDEX_FILE
    chroma_dir: Path = CHROMA_DIR
    facts_file: Path = DATA_DIR / "index" / "facts.parquet"
    data_dir: Path = DATA_DIR

    @property
    def downloads(self) -> Path:
        """Where a downloaded archive is kept, so a second prepare skips it.
        Inside ``data/``, which git ignores."""
        return self.data_dir / "downloads"


@dataclass(frozen=True)
class Item:
    """One thing the snapshot needs, and whether this machine has it.

    ``required`` False is a warning: the demo can start without it, but
    something it may need on the day is missing (the fallback's key).
    """

    name: str
    ok: bool
    detail: str
    required: bool = True


@dataclass(frozen=True)
class Report:
    """What :func:`check` found, one item per part of the snapshot."""

    items: tuple[Item, ...] = field(default_factory=tuple)

    @property
    def problems(self) -> list[Item]:
        return [item for item in self.items if item.required and not item.ok]

    @property
    def warnings(self) -> list[Item]:
        return [item for item in self.items if not item.required and not item.ok]

    @property
    def ready(self) -> bool:
        return not self.problems

    def lines(self) -> list[str]:
        """The report as a command prints it, one line per item."""
        out = []
        for item in self.items:
            status = "ok  " if item.ok else ("FAIL" if item.required else "warn")
            out.append(f"{status}  {item.name:<10} {item.detail}")
        return out


class PrepareError(RuntimeError):
    """Prepare stopped before changing anything it could not put back."""


# --- reading what is on disk -----------------------------------------------

def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _has_corpus(processed_dir: Path) -> bool:
    return processed_dir.is_dir() and any(processed_dir.rglob("*.json"))


def corpus_identity(processed_dir: Path) -> tuple[str, int, int]:
    """The corpus's fingerprint as the BM25 index computes it, its passage
    count and its filing count, in one read."""
    from src.pipeline.chunk import iter_chunks
    from src.retrieval.records import corpus_fingerprint

    counted = {"passages": 0}
    filings: set[str] = set()

    def rows() -> Iterator[dict]:
        for row in iter_chunks(processed_dir=processed_dir):
            counted["passages"] += 1
            filings.add(str(row.get("accession_no") or row.get("source_path") or ""))
            yield row

    fingerprint = corpus_fingerprint(rows())
    return fingerprint, counted["passages"], len(filings)


def bm25_fingerprint(index_path: Path) -> str | None:
    """The corpus fingerprint the BM25 index's manifest records, or None."""
    if not index_path.is_file():
        return None
    with index_path.open("rb") as stream:
        payload = pickle.load(stream)
    manifest = payload.get("manifest") if isinstance(payload, dict) else None
    return getattr(manifest, "corpus_fingerprint", None)


def dense_manifest(chroma_dir: Path) -> Any | None:
    """The dense index's manifest, or None when there is no built index."""
    from src.retrieval.embed import manifest_file_for, read_manifest

    return read_manifest(manifest_file_for(chroma_dir))


def encoder_cached(pin: DemoPin = PIN) -> str | None:
    """Where the pinned encoder sits in the local Hugging Face cache, or None.

    Asked with ``local_files_only``, so nothing is sent: this is how the demo
    knows, before it starts offline, that it will not need to download.
    """
    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(pin.encoder, revision=pin.encoder_revision,
                                 local_files_only=True)
    except Exception:   # the library's not-cached error has moved between versions
        return None


def demo_environment(pin: DemoPin = PIN, base: Mapping[str, str] | None = None) -> dict[str, str]:
    """The environment the demo runs the app in.

    The answer model is set here and not left to ``.env``: ``LLM_MODEL``
    belongs to whichever provider ``LLM_PROVIDER`` names, so a ``.env`` set up
    for Mistral would otherwise hand a Mistral model name to SOCLAAS. Picking
    Mistral in the sidebar then asks for Mistral's default, which is the pinned
    fallback. Hugging Face is set offline, so a missing encoder fails loudly
    instead of downloading in front of the room.
    """
    from src.rag.constants import LLM_MODEL_ENV, LLM_PROVIDER_ENV

    if base is None:
        load_env()
        base = os.environ
    env = dict(base)
    env.update({
        LLM_PROVIDER_ENV: pin.provider,
        LLM_MODEL_ENV: pin.model,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
    return env


# --- the check -------------------------------------------------------------

def check(
    pin: DemoPin = PIN,
    layout: Layout = Layout(),
    *,
    environ: Mapping[str, str] | None = None,
    find_encoder: Callable[[DemoPin], str | None] = encoder_cached,
    check_vectors: bool = True,
) -> Report:
    """Whether this machine holds the snapshot, with nothing sent anywhere.

    ``environ`` is the environment the app will run in, :func:`demo_environment`
    by default. ``check_vectors`` False skips reading every vector's digest out
    of Chroma, for a test with no real index; the demo's own check reads them.
    """
    items = [*_code_items(pin), *_data_items(pin, layout, check_vectors)]
    items.append(_encoder_item(pin, find_encoder))
    items.extend(_provider_items(pin, environ if environ is not None else demo_environment(pin)))
    return Report(tuple(items))


def _code_items(pin: DemoPin) -> list[Item]:
    """The code still builds what the snapshot names."""
    from src.rag import constants as rag
    from src.retrieval import constants as retrieval
    from src.stack import DEFAULT_STACK, STACKS

    problems = []
    config = STACKS.get(pin.config_id)
    if config is None:
        problems.append(f"src/stack.py has no {pin.config_id}")
    elif (config.retriever, config.metadata_filter) != ("hybrid", True):
        problems.append(f"{pin.config_id} is no longer hybrid retrieval with the metadata filter")
    if DEFAULT_STACK != pin.config_id:
        problems.append(f"the app opens on {DEFAULT_STACK}, not {pin.config_id}")
    for name, have, want in (
        ("EMBED_MODEL", retrieval.EMBED_MODEL, pin.encoder),
        ("EMBED_REVISION", retrieval.EMBED_REVISION, pin.encoder_revision),
        ("EMBED_DIMENSIONS", retrieval.EMBED_DIMENSIONS, pin.dimensions),
        ("EMBED_MAX_TOKENS", retrieval.EMBED_MAX_TOKENS, pin.max_tokens),
        ("DEFAULT_SOCLAAS_MODEL", rag.DEFAULT_SOCLAAS_MODEL, pin.model),
        ("DEFAULT_MISTRAL_MODEL", rag.DEFAULT_MISTRAL_MODEL, pin.fallback_model),
    ):
        if have != want:
            problems.append(f"{name} is {have!r}, the snapshot pins {want!r}")
    if problems:
        return [Item("code", False, "; ".join(problems))]
    return [Item("code", True, f"{pin.config_id}, {pin.encoder}@{pin.encoder_revision[:7]}, "
                               f"{pin.provider} {pin.model}")]


def _data_items(pin: DemoPin, layout: Layout, check_vectors: bool) -> list[Item]:
    items = []
    run_prepare = "run: python -m src.demo prepare"

    corpus_ok = False
    if not _has_corpus(layout.processed_dir):
        items.append(Item("corpus", False, f"no corpus in {layout.processed_dir}; {run_prepare}"))
    else:
        fingerprint, passages, filings = corpus_identity(layout.processed_dir)
        corpus_ok = fingerprint == pin.bm25_fingerprint
        detail = f"{passages:,} passages from {filings} filings, fingerprint {fingerprint[:12]}"
        if not corpus_ok:
            detail += (f"; not the frozen corpus ({pin.passages:,} passages, "
                       f"{pin.bm25_fingerprint[:12]}). {run_prepare} --replace")
        items.append(Item("corpus", corpus_ok, detail))

    if not layout.facts_file.is_file():
        items.append(Item("facts", False, f"no facts store at {layout.facts_file}; {run_prepare}"))
    else:
        digest = sha256_of(layout.facts_file)
        ok = digest == pin.facts_sha256
        detail = f"sha256 {digest[:12]}"
        if not ok:
            detail += f"; the frozen store is {pin.facts_sha256[:12]}. {run_prepare} --replace"
        items.append(Item("facts", ok, detail))

    fingerprint = bm25_fingerprint(layout.bm25_index)
    if fingerprint is None:
        items.append(Item("bm25", False, f"no index at {layout.bm25_index}; {run_prepare}"))
    else:
        ok = fingerprint == pin.bm25_fingerprint
        items.append(Item("bm25", ok, f"fingerprint {fingerprint[:12]}" + (
            "" if ok else f"; the measured index is {pin.bm25_fingerprint[:12]}. {run_prepare}")))

    items.append(_dense_item(pin, layout, corpus_ok and check_vectors, run_prepare))
    return items


def _dense_item(pin: DemoPin, layout: Layout, check_vectors: bool, run_prepare: str) -> Item:
    manifest = dense_manifest(layout.chroma_dir)
    if manifest is None:
        return Item("dense", False, f"no index at {layout.chroma_dir}; {run_prepare}")
    problems = []
    if manifest.corpus_fingerprint != pin.dense_fingerprint:
        problems.append(f"fingerprint {manifest.corpus_fingerprint[:12]}, "
                        f"the measured index is {pin.dense_fingerprint[:12]}")
    if (manifest.model, manifest.dimensions) != (pin.encoder, pin.dimensions):
        problems.append(f"built by {manifest.model} at {manifest.dimensions} dimensions")
    if manifest.max_tokens not in (None, pin.max_tokens):
        problems.append(f"vectors read {manifest.max_tokens} tokens, not {pin.max_tokens}")
    if not problems and check_vectors:
        # The manifest says what was built; this reads what the vectors hold.
        from src.retrieval.embed import check_index

        problems.extend(check_index(
            layout.chroma_dir, layout.processed_dir, model=pin.encoder, dimensions=pin.dimensions))
    if problems:
        return Item("dense", False, "; ".join(problems) + f". {run_prepare}")
    return Item("dense", True, f"fingerprint {manifest.corpus_fingerprint[:12]}, "
                               f"{manifest.n_passages:,} vectors")


def _encoder_item(pin: DemoPin, find_encoder: Callable[[DemoPin], str | None]) -> Item:
    where = find_encoder(pin)
    if where is None:
        return Item("encoder", False, f"{pin.encoder}@{pin.encoder_revision[:7]} is not in the "
                                      f"local Hugging Face cache; run: python -m src.demo prepare")
    return Item("encoder", True, f"{pin.encoder}@{pin.encoder_revision[:7]} cached")


def _provider_items(pin: DemoPin, environ: Mapping[str, str]) -> list[Item]:
    """Whether the answer model and its fallback have what they need.

    Only the settings are checked: no client is built and nothing is sent, so
    a key SOCLAAS would refuse still passes here. Try one question before the
    audience arrives for that.
    """
    from src.rag.generate import ProviderUnavailable, check_provider, config_from_env

    items = []
    for name, provider, model, required in (
        ("answers", pin.provider, pin.model, True),
        ("fallback", pin.fallback_provider, pin.fallback_model, False),
    ):
        try:
            config = config_from_env(provider=provider, environ=environ)
            if config.model != model:
                raise ProviderUnavailable(f"would ask for {config.model}, not {model}")
            check_provider(config, environ=environ)
        except (ProviderUnavailable, ValueError) as error:
            reason = getattr(error, "reason", None) or str(error)
            items.append(Item(name, False, f"{provider} {model}: {reason}", required))
        else:
            items.append(Item(name, True, f"{provider} {model}, settings complete", required))
    return items


# --- prepare ---------------------------------------------------------------

def prepare(
    pin: DemoPin = PIN,
    layout: Layout = Layout(),
    *,
    archive: Path | None = None,
    replace: bool = False,
    download: Callable[[str, Path], None] | None = None,
    fetch_encoder: Callable[[DemoPin], Any] | None = None,
    build_bm25: Callable[[Layout], Any] | None = None,
    build_dense: Callable[[Layout], Any] | None = None,
    log: Callable[[str], None] = print,
) -> None:
    """Make this machine hold the snapshot, doing only what is missing.

    A corpus or facts store already here that is not the frozen one is never
    overwritten unless ``replace`` is set, and then it is moved aside into
    ``data/replaced-<time>/`` with copies of both indexes, not deleted. The
    dense index is brought up to date in place, so after a small difference
    only the changed passages are encoded again; a machine with no index at
    all encodes all 28,289, which takes hours on a laptop CPU.

    The steps are parameters so a test can watch them run without a network
    or a model; the defaults are the real ones.
    """
    download = download or _download
    fetch_encoder = fetch_encoder or _fetch_encoder
    build_bm25 = build_bm25 or _build_bm25
    build_dense = build_dense or _build_dense

    corpus_ok = (_has_corpus(layout.processed_dir)
                 and corpus_identity(layout.processed_dir)[0] == pin.bm25_fingerprint)
    facts_ok = layout.facts_file.is_file() and sha256_of(layout.facts_file) == pin.facts_sha256

    if corpus_ok and facts_ok:
        log("corpus:   the frozen corpus and facts store are already here")
    else:
        corpus_here = _has_corpus(layout.processed_dir)
        differs = [
            what for what, here, ok in (
                (f"the corpus in {layout.processed_dir}", corpus_here, corpus_ok),
                (f"the facts store at {layout.facts_file}", layout.facts_file.is_file(), facts_ok),
            ) if here and not ok
        ]
        if differs and not replace:
            raise PrepareError(
                "This machine already has " + " and ".join(differs) + ", and it is not the "
                "frozen one the demo pins. Nothing was changed. Run again with --replace to "
                "move it aside into data/replaced-<time>/ (nothing is deleted), or prepare "
                "in a fresh clone."
            )
        # The archive is fetched and verified before anything here is moved, so
        # a failed download leaves the machine as it was.
        source = archive if archive is not None else _cached_download(pin, layout, download, log)
        _verify_archive(source, pin)
        if differs:
            _move_aside(layout, corpus_ok, facts_ok, log)
        _extract(source, layout, corpus=not corpus_ok, facts=not facts_ok, log=log)
        fingerprint, passages, filings = corpus_identity(layout.processed_dir)
        if fingerprint != pin.bm25_fingerprint:
            raise PrepareError(f"the extracted corpus has fingerprint {fingerprint[:12]}, "
                               f"not {pin.bm25_fingerprint[:12]}")
        if sha256_of(layout.facts_file) != pin.facts_sha256:
            raise PrepareError("the extracted facts store does not match the pinned hash")
        log(f"corpus:   {passages:,} passages from {filings} filings, the frozen corpus")

    if bm25_fingerprint(layout.bm25_index) == pin.bm25_fingerprint:
        log("bm25:     already the measured index")
    else:
        log("bm25:     building from the frozen corpus (about a minute)")
        build_bm25(layout)

    log(f"encoder:  {pin.encoder}@{pin.encoder_revision[:7]}, fetched into the local cache "
        f"if it is not there yet")
    fetch_encoder(pin)

    manifest = dense_manifest(layout.chroma_dir)
    if manifest is not None and manifest.corpus_fingerprint == pin.dense_fingerprint:
        log("dense:    already the measured index")
    else:
        log("dense:    encoding what differs from the frozen corpus. A machine with no index "
            "encodes every passage, which takes hours on a laptop CPU; a run that is stopped "
            "resumes where it left off")
        build_dense(layout)


def _move_aside(layout: Layout, corpus_ok: bool, facts_ok: bool, log: Callable[[str], None]) -> None:
    """Move the corpus and facts store that differ out of the way, and copy the
    indexes beside them, so the machine can be put back as it was."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = layout.data_dir / f"replaced-{stamp}"
    backup.mkdir(parents=True)
    if not corpus_ok and _has_corpus(layout.processed_dir):
        shutil.move(str(layout.processed_dir), str(backup / "processed"))
    if not facts_ok and layout.facts_file.is_file():
        shutil.move(str(layout.facts_file), str(backup / layout.facts_file.name))
    index_copy = backup / "index"
    index_copy.mkdir()
    if layout.bm25_index.is_file():
        shutil.copy2(layout.bm25_index, index_copy / layout.bm25_index.name)
    if layout.chroma_dir.is_dir():
        shutil.copytree(layout.chroma_dir, index_copy / layout.chroma_dir.name)
    from src.retrieval.embed import manifest_file_for

    manifest = manifest_file_for(layout.chroma_dir)
    if manifest.is_file():
        shutil.copy2(manifest, index_copy / manifest.name)
    log(f"moved:    what was here is in {backup}")


def _cached_download(pin: DemoPin, layout: Layout, download: Callable[[str, Path], None],
                     log: Callable[[str], None]) -> Path:
    target = layout.downloads / pin.corpus_asset
    if target.is_file() and sha256_of(target) == pin.corpus_sha256:
        log(f"archive:  {target} already downloaded")
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    log(f"archive:  downloading {pin.corpus_url} (about 9 MB)")
    partial = target.with_name(target.name + ".part")
    download(pin.corpus_url, partial)
    partial.replace(target)
    return target


def _download(url: str, target: Path) -> None:
    with urllib.request.urlopen(url, timeout=60) as response, target.open("wb") as out:
        shutil.copyfileobj(response, out)


def _verify_archive(path: Path, pin: DemoPin) -> None:
    if not path.is_file():
        raise PrepareError(f"no archive at {path}")
    digest = sha256_of(path)
    if digest != pin.corpus_sha256:
        raise PrepareError(f"{path} has sha256 {digest[:12]}, not the release's "
                           f"{pin.corpus_sha256[:12]}; download it again")


def _wanted(members: Iterable[tarfile.TarInfo], layout: Layout, corpus: bool,
            facts: bool) -> Iterator[tuple[tarfile.TarInfo, Path]]:
    """The archive members the demo needs, each with where it goes.

    The archive holds one top folder; inside it the paths are the project's own.
    Only the processed corpus and the facts store are taken. The benchmark files
    beside them are the evaluation's, and the repository already has its own.
    """
    for member in members:
        parts = PurePosixPath(member.name).parts[1:]
        if not parts or ".." in parts or PurePosixPath(member.name).is_absolute():
            continue
        relative = PurePosixPath(*parts)
        if corpus and relative.parts[:2] == ("data", "processed") and relative.suffix == ".json":
            if not member.isfile():
                raise PrepareError(f"unexpected archive entry {member.name}")
            yield member, layout.processed_dir.joinpath(*relative.parts[2:])
        elif facts and relative == PurePosixPath("data/index/facts.parquet"):
            if not member.isfile():
                raise PrepareError(f"unexpected archive entry {member.name}")
            yield member, layout.facts_file


def _extract(path: Path, layout: Layout, *, corpus: bool, facts: bool,
             log: Callable[[str], None]) -> None:
    count = 0
    with tarfile.open(path, "r:gz") as archive:
        for member, target in _wanted(archive.getmembers(), layout, corpus, facts):
            target.parent.mkdir(parents=True, exist_ok=True)
            source = archive.extractfile(member)
            if source is None:
                raise PrepareError(f"cannot read {member.name} from {path}")
            with source, target.open("wb") as out:
                shutil.copyfileobj(source, out)
            count += 1
    log(f"extract:  {count} files from {path.name}")


def _fetch_encoder(pin: DemoPin) -> str:
    from huggingface_hub import snapshot_download

    return snapshot_download(pin.encoder, revision=pin.encoder_revision)


def _build_bm25(layout: Layout) -> Any:
    from src.retrieval.bm25 import build_index

    return build_index(processed_dir=layout.processed_dir, index_path=layout.bm25_index)


def _build_dense(layout: Layout) -> Any:
    """``python -m src.retrieval embed``, on the snapshot's folders. The model
    and its settings are the constants, which :func:`check` holds to the pin."""
    from src.retrieval.embed import build

    return build(chroma_dir=layout.chroma_dir, processed_dir=layout.processed_dir)


__all__ = [
    "DemoPin", "Item", "Layout", "PIN", "PrepareError", "Report",
    "check", "corpus_identity", "demo_environment", "encoder_cached", "prepare",
]
