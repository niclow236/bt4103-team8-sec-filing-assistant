---
name: verify
description: Build, run and drive this project's pipeline and retrieval CLIs to confirm a change works.
---

# Verifying a change in this repo

Two CLIs are the surface. Neither needs arguments to show its commands:

    .venv/bin/python -m src.pipeline      # download, parse, chunk, passages, verify, rebuild
    .venv/bin/python -m src.retrieval     # embed, bm25, facts, check

## Setup

    .venv/bin/python -m pip install -r requirements.txt

Takes several minutes on a cold venv: `sentence-transformers` pulls torch, and
`chromadb` pulls onnxruntime and the opentelemetry stack. `pandas`/`pyarrow`
arrive transitively via `edgartools`, not from requirements.txt directly.

`facts` and `pipeline verify` need network and `EDGAR_IDENTITY` in `.env`.
`embed`, `bm25` and `check` need neither.

## Driving it cheaply

A full dense build is hours. Narrow it — this slice is ~235 passages, ~20s
after the model is cached:

    .venv/bin/python -m src.retrieval embed --tickers AAPL --fiscal-years 2024

The first run downloads BAAI/bge-base-en-v1.5 (~1 min). `--rebuild` drops the
collection first. A narrowed run always ends `status: does NOT match the
corpus` — that is correct, not a failure.

`bm25` is a full rebuild every time and takes ~3s. `check` reports both indexes
and exits 1 if either should not be searched.

## Gotchas

- **Re-chunk before trusting `verify`.** A corpus predating the current chunker
  fails the `passage sizes` gate (1.60% of prose over 512 tokens) and makes
  `embed`/`bm25` print a NOTE that they assumed the chunker settings rather than
  measured them. `python -m src.pipeline chunk --force` takes ~8s and fixes
  both; `verify` then passes all 8 checks.
- Re-chunking changes every chunk_id, so both indexes go stale. Rebuild bm25.
- A filter that matches nothing (`--tickers NOSUCHTICKER`) reports
  "nothing to embed" and exits 0, exactly like an up-to-date index. Confirm the
  filter matched before reading that as success.
- Stage errors (foreign model in the index, Ctrl-C) exit 1 but arrive as raw
  Python tracebacks; the message itself is at the bottom of the trace.
- `data/` is gitignored, so `git diff` will not tell you whether you changed the
  corpus. Back it up (`cp -R data/processed ...`) before a `chunk --force`.

## Useful probes

Interrupting a build must leave no manifest, and a re-run must resume:

    # SIGINT mid-encode, then:
    .venv/bin/python -m src.retrieval check     # "no manifest ... did not finish"
    .venv/bin/python -m src.retrieval embed --tickers AAPL --fiscal-years 2024

Editing one passage's text in `data/processed/` and re-running `embed` should
re-encode exactly that one ("1 of them replacing a stale vector").
