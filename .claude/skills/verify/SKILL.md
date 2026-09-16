---
name: verify
description: Build, run and drive this project's pipeline and retrieval CLIs, and the RAG stage against a local Ollama server, to confirm a change works.
---

# Verifying a change in this repo

Two CLIs are the surface. Neither needs arguments to show its commands:

    .venv/bin/python -m src.pipeline      # download, parse, chunk, passages, verify, rebuild
    .venv/bin/python -m src.retrieval     # embed, bm25, facts, check

On Windows the interpreter is `.venv/Scripts/python.exe`. The RAG stage has no
CLI; see "The RAG stage" below.

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

## The RAG stage

No CLI: the surface is the `src.rag` exports, driven the way the README's
"From a question to an answer" snippet does, over the real indexes and a real
Ollama server. Before driving it:

    curl -s http://127.0.0.1:11434/api/version   # Ollama is up (the tray app starts it)
    ollama list                                  # llama3.2:3b is pulled
    python -m src.retrieval check                # both indexes "current"

Run the README snippet as a script with `PYTHONPATH=.`, printing each `on_token`
piece with a timestamp, then `generation.raw`, `generation.text` and
`generation.to_dict()`. Check the joined pieces equal `generation.text` and
contain no JSON. A question about a company outside the corpus ("What was
Intel's total revenue in fiscal 2024?") exercises abstention: `raw` is
`{"answerable": false, "sentences": []}`.

Evidence Ollama keeps on its side, in `%LOCALAPPDATA%\Ollama\server.log` on
Windows: `n_ctx_slot = 8192` (NUM_CTX reached it), `offloaded N/29 layers to GPU`
or CPU-only buffers (where the model ran), per-request prompt and eval rates,
and `truncating input prompt` if a prompt ever overflows.

Failure paths return in seconds and need no model time:
`LLM_BASE_URL=http://127.0.0.1:9` (no server), `LLM_MODEL=llama9:404b` (not
pulled), `LLM_NUM_GPU=abc` (refused before any request).

Gotchas:

- One answer takes 3 to 4 minutes on a laptop CPU, nearly all of it before
  the first piece streams. Budget for it, and run long drives in the
  background.
- Set `LLM_NUM_GPU=0` on a laptop with a small GPU. Unset, Ollama reuses
  whatever instance of the model is already loaded, so to observe its default
  placement run `ollama stop llama3.2:3b` first.
- The first request after a model is unloaded (after 5 idle minutes) includes
  loading it.
- A question the parser marks `unanswerable` still goes through retrieval and
  generation until #33 lands, and costs a full answer's time to abstain.
