# AI-Powered SEC Filing Assistant

Retrieval-augmented question answering with verifiable evidence over US SEC corporate filings.

BT4103 Business Analytics Capstone, School of Computing, NUS. Team 8, AY26/27 Semester 1.

## Overview

Publicly listed companies file annual reports (Form 10-K) and quarterly reports (Form 10-Q) with the US Securities and Exchange Commission, the federal regulator whose role is similar to MAS and SGX RegCo in Singapore. These filings cover business operations, financial performance, risk factors, management discussion and analysis, and regulatory matters. They are valuable to analysts but long and hard to search, so answering a focused question often means reading across several sections and filings.

This project builds an assistant that answers natural-language questions about the filings and cites the exact filing and section behind each answer, so a response can be traced back to its source rather than trusted blindly.

## Problem

General-purpose LLMs can summarise filings, but their answers are not always traceable and can include unsupported statements. Keyword search misses answers when the question is worded differently from the filing. A useful assistant needs to understand natural-language questions, retrieve the relevant passages from large filings, generate concise and accurate answers, name the filing and section supporting each answer, and keep hallucinations low.

## Objectives

1. Data pipeline. Collect and preprocess a selected set of 10-K filings from SEC EDGAR, with 10-Q as a later extension.
2. Retrieval system. Implement and compare keyword search, BM25, dense embedding retrieval, and hybrid retrieval.
3. Generative QA with citations. A RAG system that generates answers grounded only in retrieved passages, with explicit source attribution.
4. Answer reliability. Improve responses with reranking, query decomposition, hallucination detection, and answer verification.
5. Evaluation. Curate a domain-specific benchmark and compare configurations using standard quantitative metrics.

## Data

All data comes from public SEC EDGAR filings, so no proprietary or subscription databases are needed.

- Industry: technology sector only. A single industry is a deliberate choice, since tech peers share unusually similar risk-factor and MD&A language, which makes the near-duplicate retrieval problem harder and makes cross-company questions genuinely comparable.
- Companies: 15 US-listed tech firms, listed in `config/companies.txt`: Apple, Microsoft, Broadcom, Alphabet, Meta, Amazon, Oracle, Salesforce, Adobe, Cisco, Texas Instruments, Micron, Intuit, ServiceNow, Palo Alto Networks. The list spans consumer hardware, cloud and enterprise software, semiconductors, networking and cybersecurity, so a cross-company question has something to compare rather than fifteen versions of the same business.
- History: fiscal years 2021 to 2025, so 5 filings per firm.
- Documents: Form 10-K for the core system and all evaluation. Form 10-Q is a future extension, layered in once the 10-K pipeline is validated, and is not part of the benchmark or the comparative results.
- Corpus size: 75 filings in scope, small enough to index on a laptop and large enough for meaningful retrieval evaluation.
- Key sections: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements and notes), and other relevant sections.

The scope is set in fiscal years, not filing years, because that is the axis questions are asked on. The two differ: Adobe, Alphabet, Amazon, Meta, ServiceNow and Texas Instruments close their books in November or December and file the following January or February, so their fiscal 2025 report is a 2026 filing while Apple's is a 2025 filing. Selecting on filing date would give those six a different set of years from the other nine, producing a corpus that looks complete at 5 filings per firm but cannot answer a single question across all fifteen.

The download therefore searches EDGAR over filing years, which is how EDGAR indexes, then narrows the result to the fiscal years in scope by reading each filing's `period_of_report`. `DEFAULT_FISCAL_YEARS` sets the scope and `DEFAULT_FILING_YEARS` sets the search window, which runs one year longer to reach the December filers. Every download prints a fiscal-year coverage table naming any year that is missing a company, so a gap is seen when the corpus is built rather than inferred later from a thin answer.

The corpus is rectangular: fiscal years 2021 to 2025, all 15 companies in each, 75 filings with no gaps and nothing outside the scope. `FilingRecord.fiscal_year` gives the year directly, and `iter_chunks(fiscal_years=...)` filters on it.

Filings stay out of version control (see `.gitignore`), so each person runs the pipeline once to build their own local copy.

## Approach

The system is evaluated as a comparative study of at least two configurations, a baseline against the proposed solution. The main elements are document parsing with section-aware chunking, retrieval modelling that compares BM25 with dense and hybrid search, RAG generation with automated citations, and reliability techniques such as reranking and hallucination detection. Performance is measured on a manually built benchmark of representative questions paired with validated supporting passages.

## Deliverables

- Data and preprocessing pipeline, a reproducible Python pipeline for scraping, parsing, and chunking SEC filings.
- Indexed search repository, a vector or hybrid database storing processed chunks with their metadata.
- AI filing assistant engine, a RAG QA system that returns evidence-grounded answers with citations.
- Benchmarking dataset, manually verified ground-truth Q&A pairs with source passages.
- Comparative experimental results across retrieval methods (sparse, dense, hybrid) and system configurations.
- Interactive user interface, a web app built with Streamlit or Gradio for live demonstration.

## Tech stack

Python is the primary language. The team has no budget for paid APIs, so nothing in the project needs a paid account. The embedding model and the search indexes run on the machine running the code, and so, by default, does the language model, with no account or key. The one hosted option is Mistral's API on its free plan, which answers in seconds rather than minutes and needs a free account and API key of your own. Exact versions are pinned in `requirements.txt`.

| Layer | What it uses |
|---|---|
| Filings | `edgartools` for EDGAR search, download and XBRL figures |
| Keyword search | `rank-bm25` |
| Dense search | `sentence-transformers` running `BAAI/bge-base-en-v1.5`, vectors stored in `chromadb` |
| Answer generation | a local model served by [Ollama](https://ollama.com), `llama3.2:3b` by default, or [Mistral's API](https://docs.mistral.ai), `ministral-8b-2512` by default |
| Talking to the model | LangChain's `ChatOllama` (`langchain-ollama`) and `ChatMistralAI` (`langchain-mistralai`) |
| Holding the answer to a shape | `pydantic`, whose JSON schema either provider decodes against |
| Tests | `pytest`, with `pytest-cov` to measure coverage |

Ollama is not a Python package, so it is installed separately; see [Setting up Ollama](#setting-up-ollama). Mistral needs nothing beyond `requirements.txt` except your own key; see [Setting up Mistral](#setting-up-mistral).

## Repository structure

This is the planned layout. Not every folder exists on day one; they are added as the work reaches each part.

```
bt4103-team8-sec-filing-assistant/
├── README.md
├── GIT_WORKFLOW.md          # branching workflow and Git setup
├── requirements.txt
├── pytest.ini               # test settings: python -m pytest from the project root
├── .coveragerc              # coverage settings for python -m pytest --cov
├── .gitignore
├── .env.example             # template for local settings (copy to .env)
├── config/
│   └── companies.txt        # tickers the pipeline downloads
├── data/
│   ├── sample/              # for a small committed sample (none yet)
│   ├── raw/                 # full filings (git-ignored)
│   ├── interim/             # parsed sections (git-ignored)
│   ├── processed/           # chunks ready for indexing (git-ignored)
│   ├── index/               # built BM25 and Chroma indexes (git-ignored)
│   └── diagnostics/         # why a run did what it did (git-ignored, on demand)
├── src/
│   ├── config.py            # project-wide paths, .env loading, EDGAR identity
│   ├── utils.py             # run logging, shared across packages
│   ├── pipeline/            # EDGAR download, parse, chunk
│   │   ├── __main__.py      #   entry point Python needs; defers to cli.py
│   │   ├── cli.py           #   THE command line: every command and what it runs
│   │   ├── constants.py     #   corpus scope, parser and chunker thresholds
│   │   ├── records.py       #   dataclasses passed between stages
│   │   ├── download.py      #   stage 1
│   │   ├── parse.py         #   stage 2
│   │   ├── chunk.py         #   stage 3
│   │   ├── passages.py      #   read passages back, for spot-checking
│   │   └── verify.py        #   gate: is the corpus fit to index?
│   ├── retrieval/           # BM25, dense, hybrid
│   │   ├── __main__.py      #   entry point Python needs; defers to cli.py
│   │   ├── cli.py           #   the command line: embed, bm25, facts, check, benchmark
│   │   ├── base.py          #   the Retriever contract, the metadata pre-filter, ranking helpers
│   │   ├── constants.py     #   models, k values and fusion constants, in one place
│   │   ├── records.py       #   Query, RetrievedPassage, and what an index says of itself
│   │   ├── embed.py         #   builds the dense index, incrementally
│   │   ├── dense.py         #   searches the dense index
│   │   ├── bm25.py          #   builds and searches the BM25 index
│   │   ├── hybrid.py        #   fuses BM25 and dense results by reciprocal rank
│   │   └── facts.py         #   XBRL figures from EDGAR into a table
│   ├── rag/                 # RAG engine and citations
│   │   ├── query.py         #   reads a question into a Query: tickers, fiscal years, question type
│   │   ├── prompt.py        #   renders the grounded prompt: numbered sources, the rules, the question
│   │   ├── generate.py      #   runs the prompt through Ollama or Mistral, streaming
│   │   ├── numeric.py       #   answers a numeric question from the facts store, citing the table
│   │   ├── decompose.py     #   searches a multi-filing question once per filing, then interleaves
│   │   ├── citations.py     #   resolves [n] markers back to the passages they were shown as
│   │   ├── verify.py        #   checks an answer's figures against the facts store
│   │   ├── answer.py        #   the entry point: route, retrieve, abstain or answer
│   │   ├── constants.py     #   company aliases, cue words, the prompt template, generation settings
│   │   └── records.py       #   GroundedAnswer, Generation, Answer, Citation and GenerationConfig
│   ├── evaluation/          # benchmark, metrics and the two evaluation commands
│   │   ├── __main__.py      #   entry point Python needs; defers to cli.py
│   │   ├── cli.py           #   python -m src.evaluation: answers a benchmark, reports abstentions
│   │   ├── harness.py       #   evaluate(): one configuration end to end through answer_question
│   │   ├── run.py           #   python -m src.evaluation.run: the C0-C4 retrieval ablation
│   │   ├── metrics.py       #   Recall@k, nDCG, reciprocal rank, hard-negative accuracy
│   │   ├── benchmark.py     #   loads benchmark/questions.jsonl, generates the XBRL one
│   │   └── records.py       #   BenchmarkQuestion and RunResult
│   ├── stack.py             # the named configurations, and the one way to build one
│   └── app/                 # the Streamlit app, and the viewer for saved answers
│       ├── main.py          #   entry point: streamlit run src/app/main.py; lists the pages
│       ├── app_pages/       #   one script per page
│       │   ├── ask.py       #     Ask: question, resolved filters, answer, retrieval trace
│       │   └── browse.py    #     Browse: company/year/Item passage explorer
│       ├── state.py         #   what is kept between reruns: corpus, indexes, stacks, answers
│       ├── components.py    #   what pages draw: answers, filters, trace, corpus passages
│       └── answers.py       #   renders evaluation answers as an HTML page to review
├── tests/                   # the pytest suite (see Getting started)
├── logs/                    # terminal output of each run (git-ignored)
├── notebooks/               # exploration and experiments
│   ├── answers/             #   test questions and headline figures through the app's answer path; each writes a git-ignored results/
│   ├── mistral/             #   hosted Mistral models through the real RAG path; writes a git-ignored results/
│   ├── retrieval/           #   retrieval sweeps: FINAL_K, fusion weights, search text, table boost, common words
│   ├── test_data/           #   the team's 48 test questions
│   └── removed-results.json #   each result file that was once committed: its rows, checksum and git object
├── benchmark/               # ground-truth Q&A dataset
│   ├── schema.md            #   the fields a benchmark question must have
│   ├── questions.jsonl      #   hand-written questions (none written yet)
│   └── generated.jsonl      #   mechanical XBRL questions (git-ignored, regenerated)
├── results/                 # ablation runs from python -m src.evaluation.run, one per --run-id (git-ignored)
└── docs/                    # reports, minutes, references
    └── mistral-free-tier-evaluation.md   # the hosted-model test behind the model choice
```

### How the pipeline is put together

One rule decides where code goes: **`cli.py` owns the command line, and nothing
else in `pipeline/` knows argparse exists.**

```
  python -m src.pipeline <command>
            │
            ▼
  __main__.py          the module Python requires to make a package runnable,
            │          and nothing more: it calls cli.main()
            ▼
  cli.py               every command, every option, and the orchestration each
            │          one needs. Reads argv, then calls plain functions.
            ▼
  download.py  parse.py  chunk.py  passages.py  verify.py
                         the work. No argparse, no argv, no main().
            │
            ▼
  constants.py  records.py
                         the values that tune the pipeline, and the dataclasses
                         the stages pass between each other.
```

The stages expose ordinary functions with named arguments, so the same code runs
from a notebook, from the retrieval stage, or from a test without anyone having
to fake an argument namespace. `cli.py` is where a string typed at a shell turns
into those arguments, and it is the only place that conversion happens.

That is stricter than it was. The command line used to be spread across three
places: a table of command descriptions in `__main__.py`, a parser builder per
command in `cli.py`, and a `main()` in each stage that glued them together. The
descriptions existed twice and all six had drifted from the modules they
described, which is what a second source of truth does given time. Now a
command's help text is read from the module docstring of whatever runs it, so
there is nothing to keep in step.

`rebuild` lives in `cli.py` rather than in a module of its own, because it is
nothing but the other commands run in order: composition of the command line, not
a stage of the pipeline.

Each file has one job, and no two have the same one:

| File | Purpose |
|---|---|
| `__main__.py` | the entry point Python requires, 14 lines, defers to `cli.py` |
| `cli.py` | the command line: arguments, dispatch, and `rebuild`'s orchestration |
| `constants.py` | corpus scope and the thresholds that tune parsing and chunking |
| `records.py` | the frozen dataclasses each stage writes and the next one reads |
| `download.py` | EDGAR to `data/raw/` |
| `parse.py` | raw HTML to Items in `data/interim/` |
| `chunk.py` | Items to passages in `data/processed/`, plus `iter_chunks` for readers |
| `passages.py` | read passages back and print them, for spot-checking |
| `verify.py` | the checks that decide whether the corpus is fit to index |

Nothing imports in a circle. `cli` depends on the stages, `verify` and `passages`
depend on the stages whose output they read, and the stages depend only on
`constants` and `records`.

## Getting started

Prerequisites are Python 3.10 or newer and Git, since both pinned dependencies require 3.10. See `GIT_WORKFLOW.md` for Git setup and the branching workflow.

Clone and enter the project:

```bash
git clone <repo-url>
cd bt4103-team8-sec-filing-assistant
```

Create and activate a virtual environment.

Mac:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Windows (PowerShell):

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the tests:

```bash
python -m pytest
```

They need no network, no EDGAR identity and nothing under `data/`: each builds
its own small corpus in a temporary directory, and the dense-index tests replace
the embedding model with a deterministic stand-in, so the suite runs in seconds.
One test counts tokens with the real bge tokenizer and is skipped if that cannot
be downloaded.

To see which code the tests reach, run them with coverage:

```bash
python -m pytest --cov
```

After the test results it prints a table of the files under `src/`, least
covered first, with the lines the tests never ran listed beside each. The
settings are in `.coveragerc`, so everyone measures the same code the same way,
branches included. `--cov-report=html` writes a browsable version to
`htmlcov/` instead, which git ignores. Coverage says which lines ran, not
whether a test checked what they did, so it shows where tests are missing
rather than proving the ones that exist are good. On 27 September 2026 it
measured 73% across `src/`: the RAG stage at 96 to 100%, and the pipeline least
covered, with its command line and passage reader at 0%, the downloader at 11%
and the verifier at 25% (#49).

Set up environment variables:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and set `EDGAR_IDENTITY` to your own name and email, for example
`EDGAR_IDENTITY=Jane Tan jane@example.com`. The SEC requires every automated
request to carry a contact string and blocks traffic without one, so the
download stops immediately with a `MissingIdentityError` if this is blank. The
generation settings further down the file are optional. By default answers come
from a local model and need no key; the `# MISTRAL_API_KEY=` line stays commented
out until you have your own key, if you choose to answer with Mistral's API
instead. Both are covered under [Answering a question](#answering-a-question).

Download filings from EDGAR. Edit `config/companies.txt` first if you want a
different set of companies:

```bash
python -m src.pipeline download --dry-run   # what would this fetch?
python -m src.pipeline download --limit 1   # quick test: newest 10-K each
python -m src.pipeline download             # the full corpus
```

Filings land in `data/raw/<TICKER>/`, and every one is recorded in
`data/raw/manifest.jsonl`. The download is resumable, so re-running it skips
whatever is already on disk.

`--dry-run` answers "what would this fetch?" without fetching it. It still asks
EDGAR which filings exist, one index request per company, but downloads no
document and writes nothing, so a mistyped ticker or a wrong year range shows up
as a preview rather than as a long download you have to unpick from the manifest
afterwards.

Split the downloaded filings into their numbered Items:

```bash
python -m src.pipeline parse                 # every filing in the manifest
python -m src.pipeline parse --tickers AAPL  # just one company
python -m src.pipeline parse --force         # re-parse filings already done
```

This writes one JSON file per filing to `data/interim/<TICKER>/`, holding the
filing's metadata and one record per Item, with the text, table count, and how
confident the parser was about the Item's boundaries. It works from the files
already on disk and never calls EDGAR again, so it is cheap to re-run whenever
the preprocessing changes.

Some filers answer Item 8 with a single sentence pointing at the financial
statements printed under Item 15, so the Item is real but nearly empty. Oracle
does this in every year of the corpus. Those sections are marked `is_stub` and
carry a `resolved_from` pointer to the Item that holds the text, which chunking
follows, so the passage is stored once rather than copied into both Items.

The run ends with anything left over: a key Item that is missing from the
filing, still empty with nothing to fall back on, or flagged by the parser.
This matters when choosing companies, and it is what settles the ticker list.
Intel was dropped because it files a narratively organised 10-K with a
cross-reference index instead of Item headings, so its MD&A cannot be located by
Item boundaries at all, and an absent Item 7 would otherwise surface much later
as an unexplained retrieval failure. IBM was dropped for the same reason when
the list grew to fifteen: it answers Items 7, 7A and 8 with a pointer to an
exhibit filed alongside the 10-K, leaving 212 characters where the MD&A should
be. Applied Materials and Qualcomm fail more quietly, which is worse. Applied
Materials' Item 8 stops after 8,650 characters, so the financial statements are
simply absent; Qualcomm's Item 1 runs to 200,000 characters in one year because
the Item 1A boundary is missed, so its risk factors are indexed under the
citation for Item 1. Neither shows up as an error at download time, only as this
stage's closing summary.

Cut the parsed Items into the passages retrieval will search:

```bash
python -m src.pipeline chunk                  # every filing in data/interim/
python -m src.pipeline chunk --tickers AAPL   # just one company
python -m src.pipeline chunk --budget 1500    # try a different passage size
python -m src.pipeline chunk --force          # re-chunk filings already done
```

This writes one JSON file per filing to `data/processed/<TICKER>/`, holding
passages of roughly 1,800 characters. A passage never spans two Items, since a
citation has to name the Item it came from. Paragraphs are packed whole rather
than sliced at a character count, so passages land on sentence boundaries, and
each one carries the nearest heading above it so a passage taken from the middle
of Item 1A still knows which risk it sits under. Like parsing, this works only
from files already on disk, so it is cheap to re-run as the strategy changes.

`--budget` and `--overlap` are the two knobs the retrieval comparison will
sweep, and `--key-items-only` builds a narrow index from the targeted Items
alone, for comparison against the full one.

Read the passages back, to see what retrieval will actually be searching:

```bash
python -m src.pipeline passages --tickers AAPL --item 7      # one company's MD&A
python -m src.pipeline passages --contains "supply chain"    # find a phrase
python -m src.pipeline passages --content-type table --full  # rebuilt tables in full
python -m src.pipeline passages --item 1A --limit 0 --json   # all of them, for piping
```

This reads `data/processed/` and writes nothing. It exists because a chunking
decision is hard to judge from the summary counts alone: whether a passage
begins mid-sentence, whether a table kept its column labels, whether a heading
was picked up, are all questions you have to read a passage to answer. The same
`--tickers`, `--item`, `--fiscal-years` and `--key-items-only` filters that
narrow an index narrow this too, so what you read is what a given index would
hold.

Every command above is also reachable through one entry point, which is the
quickest way to see what the pipeline can do:

```bash
python -m src.pipeline              # list the commands
python -m src.pipeline parse --help # options for one of them
```

### Building the whole corpus in one command

The three stages above are the right thing when you are working on one of them.
For everything else, including proving the corpus can be rebuilt from nothing,
there is one command that runs all of them and then checks the result:

```bash
python -m src.pipeline rebuild            # download, parse, chunk, then verify
python -m src.pipeline rebuild --clean    # delete data/ first, so it starts from empty
```

It exits non-zero if the corpus does not pass, so a rebuild cannot finish quietly
with a broken corpus. `--clean` removes `data/raw`, `data/interim` and
`data/processed` before starting; `data/sample` and `data/diagnostics` are left
alone. Without it the stages resume, which is faster but does not prove an empty
folder can be filled.

Measured on the fifteen-company corpus, 75 filings, over a home connection, with
`--clean` so nothing was resumed:

| Stage | Time | Notes |
|---|---|---|
| download | 2.1 min | 75 filings, held under the SEC's rate limit by edgartools |
| parse | 8 to 20 min | the expensive stage, and the one that varies: 75 filings of HTML, several megabytes each |
| chunk | 15s | pure text processing over the parsed Items |
| verify | 2.0 min | 15 EDGAR index requests plus 15 XBRL fetches, then nine checks over 28,000 passages |
| **total** | **10 to 25 min** | a resumed run skips the download and re-parses only what changed |

Parse is quoted as a range because it is CPU-bound and single-threaded: the same
75 filings took 7.9 minutes on an idle machine and 20.2 minutes on one that was
also running other work. Plan for the upper figure. Everything else is stable.

Parse dominates either way, and neither it nor chunk touches the network, so
re-running them after a preprocessing change costs no EDGAR traffic.

### Checking the corpus is fit to index

```bash
python -m src.pipeline verify
```

The stages report what they did. This asks whether the result can be trusted,
and exits non-zero when it cannot. It is the last thing to run before handing the
corpus to retrieval, and it takes no options: a gate you can narrow is one that
gets narrowed until it passes.

Nine checks, cheapest first:

| Check | What would fail it |
|---|---|
| coverage | a company missing from one fiscal year, a filing counted twice, or one outside the scope |
| stage parity | a filing that reached parse but not chunk, or a stray file no manifest line accounts for |
| key Items | Items 1, 1A, 7, 7A or 8 absent, or a stub with nothing to resolve to |
| chunk integrity | a duplicate passage id, a passage that cannot build a citation, a table row tracing to no source row |
| no prose lost | a paragraph of 200 characters or more in a chunked Item that reaches no passage, unless the chunker dropped it as a flattened copy of a table it rebuilt |
| passage sizes | any table passage, or more than 0.5% of prose passages, past the 512 tokens bge reads, counted with the model's own tokenizer over the passage and its context header |
| statement titles | a filing whose balance sheet, income statement or cash flow statement carries no title a question could name it by |
| matches EDGAR | a filing disagreeing with EDGAR on CIK, form, filing date or period of report, or one in scope on EDGAR that was never downloaded |
| XBRL figures findable | a figure the filing reported to EDGAR that appears in no indexed passage |

The last one is the check that speaks to what the corpus is for. EDGAR publishes
the figures each filing reported, so they serve as an answer key: a number in
that key which appears in no passage is one no retrieval system built on this
corpus could ever cite, however good the retriever.

Because two of the checks query EDGAR, `verify` needs a network connection and
`EDGAR_IDENTITY` even though it downloads nothing. That is deliberate. The
strongest thing that can be said about a corpus is that it still agrees with its
source, and a filing can be amended after you fetch it.

A check that finds the corpus incomplete skips the per-file checks below it,
since each would report the same missing filing once per filing. Those are listed
as `SKIP` rather than left out, so a run that checked four things cannot be
mistaken for a clean bill of health on nine.

What verify does **not** fail on is imperfection the pipeline already handles: 6
of 5,428 tables cannot be rebuilt into grids -- Cisco's signature blocks and one
audit-matter paragraph laid out as a table -- and their text stays in the prose,
so nothing is lost. Those are reported as counts by the parse stage. A gate that
fired on them would cry wolf on every run.

Run the app, once the retrieval indexes are built (see
[Streamlit app and components](#streamlit-app-and-components)):

```bash
streamlit run src/app/main.py
```

## What the pipeline produces

All three stages are built. Each one writes to its own folder under `data/`,
and nothing downstream writes back into an earlier stage's folder.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Download | `python -m src.pipeline download` | `config/companies.txt` | `data/raw/<TICKER>/*.html`, `data/raw/manifest.jsonl` |
| Parse | `python -m src.pipeline parse` | the manifest and the raw HTML | `data/interim/<TICKER>/*.json` |
| Chunk | `python -m src.pipeline chunk` | `data/interim/<TICKER>/*.json` | `data/processed/<TICKER>/*.json` |

`python -m src.pipeline passages` sits outside that table on purpose: it reads
`data/processed/` and writes nothing, so it is an inspection command rather than
a stage.

A fourth folder, `data/diagnostics/`, holds output written to explain a run
rather than to feed the next stage, and is created only when something asks for
it. `python -m src.pipeline parse --table-debug` writes the HTML of every table
that could not be rebuilt to `data/diagnostics/table_failures/`, with an index
naming the reason for each, which is the only way to tell a merged cell from a
spacer row.

`data/raw/manifest.jsonl` holds one JSON object per filing. It is what makes the
download resumable, and it lets later stages see what is on disk without walking
the folder tree or calling EDGAR again:

```json
{"ticker": "AAPL", "cik": 320193, "company": "Apple Inc.", "form": "10-K",
 "filing_date": "2025-10-31", "accession_no": "0000320193-25-000079",
 "url": "https://www.sec.gov/Archives/edgar/data/320193/...",
 "path": "data/raw/AAPL/10-K_2025-10-31_0000320193-25-000079.html",
 "period_of_report": "2025-09-27"}
```

`filing_date` is when the filing was lodged; `period_of_report` is the fiscal
year it reports on. They differ by up to a year, so anything comparing companies
by fiscal year uses the second.

`path` is relative to the project root and always uses forward slashes, so a
manifest built on Windows still resolves on a teammate's Mac. Join it onto
`PROJECT_ROOT` to open the file.

Each `data/interim/<TICKER>/<filing>.json` carries the same filing metadata plus
a `sections` list, one entry per Item:

| Field | Meaning |
|---|---|
| `section_id` | the parser's name for the section, for example `part_ii_item_7` |
| `part`, `item` | `"II"` and `"7"` |
| `title` | the official Item title, used in citations |
| `text` | the Item's text |
| `n_chars`, `n_tables` | how big the Item is, and how many `<table>` elements it holds |
| `n_data_tables` | how many of those hold a grid of data rather than being a layout wrapper around a bullet point |
| `is_key_section` | whether this is one of the Items the project targets |
| `is_stub` | the Item is empty, or answers with a cross-reference rather than the disclosure |
| `resolved_from` | for a stub, the `section_id` that actually holds the text |
| `tables` | the Item's tables, rebuilt as grids of cells with their row and column labels intact. The grid is stored rather than a rendered table, because layout depends on the passage budget, which is a chunking decision |
| `confidence`, `detection_method`, `validated`, `warnings` | what the parser thought of its own boundary detection, kept so a bad answer can be traced back to a bad split |

Each `data/processed/<TICKER>/<filing>.json` carries the filing metadata again
plus a `chunks` list, one entry per passage:

| Field | Meaning |
|---|---|
| `chunk_id` | `<accession_no>_<section_id>_<index>`, unique across the corpus |
| `section_id`, `part`, `item`, `title` | which Item the passage was cut from, for the citation |
| `heading` | the nearest heading above the passage inside that Item |
| `text`, `n_chars` | the passage and its size |
| `chunk_index` | position within the Item, counting from 0 |
| `is_key_section` | whether the Item is one the project targets, so a narrow index can be built by filtering |
| `incorporated_into` | Items that answer with a cross-reference to this one |
| `content_type` | `"prose"` or `"table"`, so retrieval can weight tables when a question is numeric |
| `table_index`, `table_caption` | which table a table passage came from; `table_index` addresses that Item's `tables` list directly |

The corpus currently chunks to 28,179 passages over 75 filings: 17,564 of prose
and 10,615 of tables. Prose runs to a median of 1,529 characters and 95% of it
carries a heading; tables, cut to fit the embedding window, run to a median of
624 and 32%, since a table sits under a caption more often than under a heading.

Four things about that output are worth knowing before you build on it.

An Item that answers with a cross-reference produces no passages of its own, so
Oracle's Item 8 is empty and its Item 15 passages carry `incorporated_into:
["8"]` instead: the text is stored once and cited from either Item.

Financial tables are indexed as tables, not as prose. The extractor's plain text
runs a balance sheet together into a column of bare numbers with the row and
column labels stripped off, which leaves a figure like 245,122 with nothing to
say it is Microsoft's total revenue for 2024. The parse stage therefore rebuilds
each table as a grid, and those grids are chunked separately and marked
`content_type: "table"`, with the header repeated on every slice of a long one.
All but 6 of the tables that hold data rebuild cleanly, 5,422 of 5,428; where one
cannot, its flattened copy is left in the prose, so no figure is ever lost, it
is just harder to read.

That rate is measured against `n_data_tables`, not `n_tables`. Filers wrap
bullet points in a one-cell `<table>` to indent them, and Item 1A is written
almost entirely that way, so 3,121 of the 8,549 `<table>` elements in the corpus
are page formatting rather than data. Their text is already in the Item, so
skipping them loses nothing, and counting them as failed rebuilds would put the
figure around 63% and read as though a third of the financial statements were
broken. Where a
table arrives with no header row of its own, the first row is promoted to the
header if it reads as labels rather than figures, so that every piece of a split
table still says what its columns are.

Figures are punctuated the way the filing writes them, which takes one piece of
care. The rebuilt table hands back a year as the number 2026, indistinguishable
from 2,026 of anything, so a debt or lease maturity schedule would read "2,026"
where the filing says "2026". That is wrong on its face, and it also stops a
keyword search for the year from matching the row it belongs to. A column is
therefore stripped of its separators only when at least three of its values form
a run of consecutive years, which a column of amounts never does.

An Item that is merely short is not the same as an Item that is empty. Apple
answers Item 2 Properties in 488 characters of real fact, while Item 12 uses 303
characters to point at the proxy statement. Only the second is skipped, and the
test is whether the Item reads as a cross-reference rather than how long it is.

Every passage is sized to be read whole by the embedding model rather than
truncated by it. 1,800 characters is about 450 tokens of ordinary prose, inside
the 512-token limit most sentence-transformer models impose. These rules keep
passages there:

- A paragraph longer than the budget is split at sentence ends, and a sentence
  longer than the budget at its clause boundaries, so a cut never lands inside a
  sentence unless the sentence alone exceeds the budget.
- Text dense with figures is charged more of the budget than its length, because
  it tokenises at up to two and a half times the rate of prose. A paragraph whose
  visible characters are under 15% non-letters costs its length, as before; above
  that its cost rises until, at 35%, it is budgeted like a table.
- Tables get their own budget, derived from the prose one, and are split by rows
  and by columns with the header repeated on every piece. The caption line counts
  against that budget too, since it opens every piece.
- A header spanning every column -- "Fair Value Measurements at Reporting Date
  Using" above Total and Levels 1 to 3 -- is stated once, above the row labels,
  instead of inside every column label. Repeated, it took a median of a third of
  each table passage and cut many tables into one-row pieces.
- Every passage cut from a financial statement names that statement. The title
  comes from the filing's own heading above the table, read from the Item's
  markdown, and opens each part: "CONSOLIDATED BALANCE SHEETS (part 2 of 3)"
  where the passage holding Apple's "Total assets | $364,980" used to open
  "Financial Statements (part 2 of 3)" with an empty caption. Nothing in that
  passage said "balance sheet", so a question naming the statement could not
  match it, and neither a retriever nor the model could tell which statement
  the figures belonged to. A table with no heading above it, which is every
  note and schedule, keeps the caption it had.
- A table passage identical to another in the same Item is kept once. Filers do
  print a table twice, and in one Item the two would be the same vector indexed
  twice. Repeated prose keeps each occurrence's heading and source position,
  since its surrounding evidence can differ.
- The flattened copy of a table the parser rebuilt is dropped from the prose, so
  the same figures are not indexed twice, once unreadable. A block is judged a
  copy when the table's own cells account for it and no figure is left over.
- The blank lines between paragraphs count against the budget as well as the
  paragraphs, since they are in the passage too.
- Cells are rendered without alignment padding. Padding would be the largest
  single item in a wide table and carries no meaning to a model reading it.

The result, counted in bge's own tokens with the context header the encoder also
reads: **none of the 28,179 passages exceeds 512 tokens.** The largest prose
passage is 504 tokens and the largest table passage 385. `verify` checks this on
every run. Before the last of these rules, 287 prose passages were being
truncated, almost all of them flattened tables left in the text.

`CHUNK_CHAR_MINIMUM` sets how far a passage may run past the budget, since a
passage is closed only once it is over that minimum and one more paragraph can
then be added. It is held just above `HEADING_CHAR_LIMIT`, high enough that a
heading is never stranded as a passage of its own and low enough that
`CHUNK_CHAR_MINIMUM + CHUNK_CHAR_BUDGET` stays inside 2,048. Raise
`CHUNK_CHAR_BUDGET` only alongside a model whose context window is known to take
it.

### Keeping a record of a run

Every `download`, `parse` and `chunk` run copies its terminal output to
`logs/<command>-<timestamp>.log` and prints the path it used. The point is
traceability: a passage count quoted in a report or on a slide can be traced back
to the run that produced it, rather than to whatever is in the terminal
scrollback that day.

`logs/` is git-ignored, since these are records of what happened on one machine
rather than shared source.

For anything else worth capturing, `src/utils.py` offers the same machinery:

```python
from src.utils import run_log

with run_log("chunk-sweep") as path:
    ...        # everything printed in here also lands in logs/
```

Colour codes are stripped from the file but left on the terminal, so the log
stays readable in a text editor without the console losing its formatting.

The dataclasses behind all three files live in `src/pipeline/records.py`, so a
later stage can read a manifest line, an interim file, or a chunk by importing
the shape alone, without pulling in the download, parse, or chunk logic.

### Reading the corpus from the retrieval stage

A passage is stored knowing which Item it came from, but not which company or
year: that is held once per filing rather than repeated on all two hundred of
its passages. Indexing needs both on one record, so `iter_chunks` does the join:

```python
from src.pipeline.chunk import iter_chunks

for passage in iter_chunks(fiscal_years=range(2021, 2026)):
    passage["text"]          # what to embed
    passage["fiscal_year"]   # 2024, from period_of_report, not the filing date
    passage["ticker"]        # plus company, cik, form, filing_date, url,
    passage["item"]          # accession_no, part, title, heading, content_type
```

Every field needed to build a citation is on the record, so nothing has to go
back to the manifest at query time. Passing `fiscal_years` is what stops a
question like "compare these companies in FY2024" from quietly answering across
a mix of years; `tickers` and `key_items_only` narrow it the same way, the
latter matching the `--key-items-only` flag on the chunk command. Each record
also carries the `chunk_budget` and `chunk_overlap` its filing was cut with, so
an index can record the settings it was built from rather than assume them.

### Building the retrieval indexes

The retrieval stage has its own command line beside the pipeline's:

```bash
python -m src.retrieval embed    # dense index, data/index/chroma/
python -m src.retrieval bm25     # BM25 index, data/index/bm25.pkl
python -m src.retrieval facts    # XBRL figures, data/index/facts.parquet
python -m src.retrieval check    # does each index still match data/processed/?
```

`embed`, `bm25` and `check` read only `data/processed/`. `facts` queries EDGAR,
so it needs a network connection and `EDGAR_IDENTITY`.

`embed` is incremental. Each vector stores a digest of the exact text it was
encoded from, so a run encodes the passages that are missing, re-encodes those
whose text has changed, and removes vectors for passages the corpus no longer
holds. After a re-chunk, run `embed` again rather than `embed --rebuild`.
`--rebuild` re-encodes every passage and is only needed after changing the
embedding model. The first full build took 6.6 hours on a laptop CPU; the encoder
now sorts passages by length across 256 at a time before batching them, which
measured 1.19x faster with identical vectors. An interrupted run is resumed by
running it again, and a long one is best run in your own terminal, since a run
started from a tool session ends when that session does.

Each index records what it was built from: the dense one in
`data/index/chroma.manifest.json`, BM25 inside `bm25.pkl`. A retriever compares
that against `data/processed/` before searching and refuses an index that no
longer matches. `check` runs the same comparison for both and exits non-zero if
either should not be searched, so it is the command to run after pulling
changes to the pipeline.

`embed` also writes `data/index/chroma.truncated.json`, listing the passages
longer than the 512 tokens bge reads. Their stored text is whole, but their
vector covers only the first 512 tokens.

`facts --refresh` pulls the companies asked for again and replaces only their
rows. Companies not asked for, and any whose request fails, keep what is stored.
In a figure's row, `fiscal_year` is the year of the filing it was published in,
which is not necessarily the year the figure describes; use `current_year()`
or the `is_current_year` column for that.

### Searching the indexes

There is no search command; retrieval is called from code, the same way the RAG
stage and the evaluation harness call it. Every method takes a `Query` and
returns `RetrievedPassage` records, best first:

```python
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever
from src.retrieval.records import Query

hybrid = HybridRetriever(BM25Retriever.load(), DenseRetriever.load())
query = Query("How did revenue change?", top_k=8, tickers=("MSFT",), fiscal_years=(2024,))
for passage in hybrid.search(query):
    passage.rank, passage.score, passage.chunk_id, passage.sources
```

Three methods are built. `BM25Retriever` scores keywords, `DenseRetriever`
searches the bge vectors in Chroma, and `HybridRetriever` asks each of them for
their top 50 (`CANDIDATE_K`) and fuses the two lists by reciprocal rank, so
their scores, which are on different scales, are never compared directly. A
fused passage records in `sources` which methods returned it. All of them
satisfy the `Retriever` protocol in `base.py`, so the evaluation harness can
loop over them.

There was a fourth, a cross-encoder reranker (#21) that re-scored hybrid's top
50 with `ms-marco-MiniLM-L-6-v2`. Nothing on the answer path used it, so it was
measured before being wired in, and it did not put more answers in front of
the model. `notebooks/retrieval/rerank_comparison.py`, as of `ed62352`, writes
the rows:

| | Hybrid | Hybrid, reranked | Reranked, no table boost on its scores |
|---|---|---|---|
| Expected figure in the top 16, hand-written (28) | 28 | 27 | 27 |
| Its rank, median | 1 | 4.5 | 5 |
| Prose: expected terms found, mean (20) | 0.994 | 0.961 | 0.961 |
| Supporting chunk in the top 16 (225 benchmark questions, 3 from each filing) | 180 | 184 | 174 |
| Reciprocal rank, mean | 0.501 | 0.362 | 0.319 |
| Tables among the top 16, mean | 13.0 | 9.4 | 6.4 |

On the benchmark it gained 11 questions and lost 7, which 225 questions cannot
tell from no change, and it moved the passages hybrid had found down the list.
On the hand-written questions it lost Salesforce's goodwill. It also took a
median of 9 seconds a question on the team laptop while two other runs shared
its cores. So the reranker and the wrapper class it was built on were removed
rather than left unused. The script that measured it went with them, since it
cannot run without the class, and is in the history as
`notebooks/retrieval/rerank_comparison.py`.

The filters on a `Query` (`tickers`, `fiscal_years`, `items`, `content_type`)
are applied before scoring, not after, in every method. BM25
scores only the passages the filters admit, and the dense retriever passes them
to Chroma as a `where` clause, so a query pinned to Microsoft's FY2024 filing
gets its top k from that filing rather than from whatever survives a
corpus-wide top k. This matters more here than in most corpora: fifteen peers
across five years write near-identical risk factors, and semantic similarity
alone would happily return the right paragraph from the wrong year.

A fifth filter, restricting a search to the key Items (1, 1A, 7, 7A and 8),
was removed. Nothing set it, and measured on Hybrid it found less: the
supporting chunk was in the top 16 for 284 of 375 benchmark questions (five
from each filing) where the unrestricted search has 298, with Oracle's falling
from 19 of 25 to 3, and the 48 test questions came out the same either way.

A question that asks for a figure is fused with its own weights,
`FIGURE_FUSION_WEIGHTS`, which `rag/query.py` selects by setting
`Query.wants_figures`. They are equal today, and that is a measurement rather
than an untouched default. The concern was that equal weights reward the
passage both retrievers returned, and BM25 rarely returned a financial
statement table: it names neither the company nor the fiscal year, and writes
the year as "December 31, 2025", so a table only the dense retriever ranked
highly lost to prose the two agreed on. Meta's FY2025 total assets sat at
dense rank 4 and hybrid rank 14.

`python notebooks/retrieval/fusion_weight_sweep.py` swept BM25's weight over
the 48 test questions with dense held at 1.0. Quieting BM25 costs figure
questions rather than helping: the expected figure reached the top 8 for 20 of
28 questions at equal weights, 18 at 0.3, and 13 with dense alone, while prose
barely moved. Carrying each statement's title into its passages (#88) is what
changed it: a balance sheet passage now holds the words "CONSOLIDATED BALANCE
SHEETS", so BM25 finds the table it used to miss. The sweep is worth re-running
when the corpus changes.

Those numbers, and the search-text ones further down, were measured at the
`FINAL_K` of 8 that was in force at the time. Both scripts report at whatever
`FINAL_K` says, so a re-run today reports a top 16 instead.

`table_boost` leans a query toward table passages without excluding prose, by
raising a table passage's score before the cut to k. It is off by default and is
set by `rag/query.py` only for questions that ask for a figure, to
`retrieval.constants.TABLE_BOOST`, which is 1.2.

That constant stayed at 1.0, which is also off, until the XBRL benchmark (#24)
could give a value measured rather than guessed. `python
notebooks/retrieval/table_boost_sweep.py` is the measurement: 20 benchmark
questions from each of the 75 filings and the 48 test questions, cut at
`FINAL_K`, with the boost applied only where the parser reads a request for a
figure, as the shipped code applies it. It was run twice. The first run, with
every word of a question scored by the keyword search, chose 1.2: hybrid
found the supporting chunk for 76.6% of the benchmark there against 69.0% with
the boost off. The table below is the second run, after the keyword search
stopped scoring the words most passages contain (further down), which is the
search the boost now sits on:

| Table boost | 1.0 | 1.05 | 1.1 | 1.15 | 1.2 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| Supporting chunk in the top 16, hybrid (1,490 benchmark questions) | 75.4% | 77.9% | 78.3% | 78.6% | 78.9% | 78.8% | 78.7% | 77.7% |
| Reciprocal rank, mean | 0.340 | 0.418 | 0.461 | 0.471 | 0.472 | 0.470 | 0.463 | 0.459 |
| Where tables support the question (1,086) | 77.9% | 82.0% | 83.4% | 84.3% | 84.7% | 85.1% | 86.1% | 87.0% |
| Where tables and prose do (292) | 75.7% | 76.0% | 76.0% | 74.7% | 75.3% | 74.0% | 71.9% | 67.5% |
| Where only prose does (112) | 50.0% | 43.8% | 34.8% | 33.9% | 31.2% | 30.4% | 25.0% | 13.4% |
| Tables among the top 16, mean | 6.4 | 9.4 | 11.5 | 12.4 | 12.9 | 13.2 | 14.0 | 14.8 |
| Supporting chunk in the top 16, BM25 alone | 69.5% | 71.0% | 71.7% | 72.3% | 73.1% | 73.3% | 75.2% | 76.2% |
| Expected figure in the top 16, hand-written (28) | 27 | 28 | 28 | 28 | 28 | 28 | 28 | 28 |
| Its rank, median | 3.5 | 2 | 1 | 1 | 1 | 1 | 1 | 1 |
| Prose: expected terms found, mean (20) | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 | 0.994 |

Hybrid's hit rate and reciprocal rank still both peak at 1.2 and neither rises
past it, while the cost keeps rising: a figure printed only in prose loses its
passage more often at every step. At 1.2 the benchmark gains 74 questions
where a table holds the figure and loses 21 of the 112 where only prose does.
The boost buys less than it did, 3.5 points of hit rate where it bought 7.6,
because the keyword search now finds many of those tables without it.
The steps are small because a cosine similarity is: dense scores sit in a
narrow band, about 0.45 for an off-topic query's best passage and 0.68 to 0.74
for an answerable one's, so a multiplier of 1.2 is enough to carry a loosely
related table past the best prose passage. BM25's scores spread wider, which
is why BM25 alone is still gaining at 2.0; one value serves both, set where
hybrid peaks. Re-run the sweep when the embedding model changes or the corpus
is re-chunked, since the right value follows the scale of the scores it
multiplies.

What the lean does to the answers is in
[How often the answers are right](#how-often-the-answers-are-right): on the
825 headline questions the model gave a wrong figure or none for 8 where it
had for 28.

A question is not scored on the words most passages contain. BM25 weighs a
word by how rare it is, and Okapi's formula goes negative for a word in more
than half the passages. `rank_bm25` does not leave such a word at nothing: it
gives it a floor, a quarter of the average weight. On this corpus the floor is
1.89 and ten words sit on it ("a", "and", "as", "for", "in", "of", "on",
"our", "the", "to"), where "revenue" weighs 1.61 and "total" 1.04. So in "What
was the total value of Goodwill at the end?" the two "the" and the "of"
outweighed the line item, and a statement table, which holds "Goodwill" and
none of those words, ranked below prose that holds them all: Salesforce's
FY2023 balance sheet was 260th of the 397 passages in its filing.
`BM25Retriever` now leaves those words out of a question. It counts them from
the corpus when the index loads rather than keeping a list, so a re-chunked
corpus gets its own, and a question made of nothing else is scored as before.

`python notebooks/retrieval/common_words_comparison.py` scores the same
questions both ways, with the table boost as shipped:

| | BM25, every word | BM25, without the common words | Hybrid, every word | Hybrid, without the common words |
|---|---|---|---|---|
| Supporting chunk in the top 16 (1,490 benchmark questions) | 63.2% | 73.1% | 76.6% | 78.9% |
| Reciprocal rank, mean | 0.320 | 0.403 | 0.455 | 0.472 |
| Where tables support the question (1,086) | 690 | 826 | 889 | 920 |
| Where tables and prose do (292) | 203 | 212 | 217 | 220 |
| Where only prose does (112) | 48 | 51 | 36 | 35 |
| Tables among the top 16, mean | 4.2 | 6.0 | 11.4 | 12.9 |
| Expected figure in the top 16, hand-written (28) | 22 | 27 | 27 | 28 |
| Its rank, median | 5.5 | 3 | 1 | 1 |
| Prose: expected terms found, mean (20) | 0.980 | 0.980 | 0.994 | 0.994 |

BM25 alone gains 148 benchmark questions and Hybrid 33, and Hybrid loses one
of the 112 that only prose supports. The 28th hand-written figure is
Salesforce's goodwill, which neither method had retrieved before. What that
does to the answers is in
[How often the answers are right](#how-often-the-answers-are-right).

The `FINAL_K`, fusion-weight and search-text numbers in this README were
measured with the boost off and every word of the question scored. A re-run of
those scripts reports with the boost on and the common words left out.

`Query.top_k` defaults to 10, which suits Recall@10 and nDCG@10. The RAG stage
asks for `FINAL_K`, 16, since that is what goes into the prompt.

`FINAL_K` was 8 for as long as a laptop's Ollama set the ceiling. A hosted
model with a 256K window removes that, so `python
notebooks/retrieval/final_k_sweep.py` measured the cap rather than keeping it.
It sweeps 8, 12, 16 and 20 over both question sets, searching each question
once at 20 and slicing the ranking, which is exact here because hybrid fuses at
`max(CANDIDATE_K, k)` and the order therefore does not depend on k. The script
checks that against real searches before it trusts it.

| | top 8 | top 12 | top 16 | top 20 |
|---|---|---|---|---|
| Supporting chunk in the prompt, XBRL benchmark (20 from each of the 75 filings) | 56.2% | 63.6% | 68.1% | 71.3% |
| Recall, mean | 0.415 | 0.484 | 0.531 | 0.567 |
| nDCG, mean | 0.268 | 0.291 | 0.305 | 0.315 |
| Reciprocal rank, mean | 0.269 | 0.277 | 0.280 | 0.282 |
| Supporting chunk in the prompt, AAPL and AMZN only (1,884), local build | 63.0% | 72.3% | 76.2% | 78.6% |
| Expected figure in the prompt, hand-written (28) | 20 | 22 | 23 | 24 |
| Prose: expected terms found, mean (20) | 0.978 | 0.984 | 0.994 | 0.994 |
| Prompt tokens, median | 2,884 | 4,005 | 5,220 | 6,475 |
| Prompt tokens, largest seen | 3,550 | 4,984 | 6,306 | 7,703 |

The rows are on the corpus `search_text_comparison.csv` was measured on, except
the AAPL and AMZN row. That row, `final_k_sweep.csv` as measured and the
hosted runs below come from a local build that ranks the expected figure
differently for 14 of the 28 figure questions; on it the figure reached the
prompt for 18, 21, 22 and 23 of the 28.

Two things decide it. The curve flattens: 8 to 12 finds the supporting chunk
for another 7.4% of the benchmark, 12 to 16 another 4.5%, and 16 to 20 another
3.2%, so 16 holds 79% of everything 20 buys. And 20 does not fit locally.
Every prompt measured is inside Ollama's 8,192-token window, but the answer
has to fit beside it: at 20 the largest prompt plus `MAX_OUTPUT_TOKENS` comes
to 8,727, over the window, against 7,330 at 16. So 16 is the largest value both
paths can run, and no separate local cap is needed.

Worth reading the columns against each other. Recall and "in the prompt" climb
while reciprocal rank barely moves, from 0.269 to 0.282. A larger `FINAL_K` is
not ranking better; it is cutting the answer off less often. That is the
failure this was opened for: in the September evaluation the table holding the
expected figure was often found and then dropped, at rank 12 to 23.

A retrieval sweep cannot say whether a longer prompt distracts the model, so
the 48 hand-written questions were also run end to end through both hosted
Ministral models at 8 and at 16, with
`notebooks/mistral/mistral_generation_test.ipynb`:

| | 14B @ 8 | 14B @ 16 | 8B @ 8 | 8B @ 16 |
|---|---|---|---|---|
| Figure in the retrieved passages | 18/28 | 22/28 | 18/28 | 22/28 |
| Figure stated, rounding allowed | 18/28 | 22/28 | 18/28 | 22/28 |
| Figure stated, exact digits | 13/28 | 17/28 | 16/28 | 20/28 |
| Abstained | 7 | 4 | 5 | 2 |
| Expected prose terms, mean | 90% | 92% | 93% | 93% |
| Valid answer JSON | 48/48 | 48/48 | 48/48 | 48/48 |
| Prompt tokens, median | 3,139 | 5,751 | 3,139 | 5,751 |
| End to end, median | 2.0 s | 2.2 s | 2.2 s | 2.2 s |

It does not distract them. Both models stated every figure they were given, at
both cutoffs, which is why the first two rows agree: retrieval was the whole of
the gap rather than part of it. Prose did not regress, nothing failed to parse,
and the extra 2,600 prompt tokens cost about two tenths of a second. Abstentions
fell because the evidence arrived, which is the direction this was meant to
move.

Local answers still pay for the extra passages in time rather than in window.
Reading the prompt dominates a laptop's minutes and is roughly linear in its
length, so an Ollama answer takes about twice as long: re-timed on the team
laptop, two questions took 4.1 and 4.3 minutes at 16 against 2.0 and 2.4 at 8.
[How long an answer takes](#how-long-an-answer-takes) has the details, and
what `llama3.2:3b` makes of the extra passages.

Loading both retrievers takes about 15 seconds, and the first search about 30
more while the embedding model loads. After that a hybrid search takes around
half a second.

## Answering a question

The RAG stage turns a question into an answer that cites the passages it came
from. One of two providers writes the answer, chosen with `LLM_PROVIDER` in
`.env`:

- `ollama`, the default: a local model served by Ollama. It needs no account,
  key or network connection once the model is downloaded, but an answer takes
  minutes on a laptop; see [How long an answer takes](#how-long-an-answer-takes).
- `mistral`: an open-weight Ministral model on Mistral's API, on the free plan.
  An answer takes about two seconds, and needs an internet connection and an
  API key from your own Mistral account.

Both get the same prompt and the same answer schema, and everything before and
after generation is the same code. Each answer records the provider and model
that wrote it, in `answer.config`. Two settings in `.env` choose them, and
neither is required:

| Variable | Default | When to set it |
|---|---|---|
| `LLM_PROVIDER` | `ollama` | `mistral` to answer with Mistral's API, once you have [set it up](#setting-up-mistral) |
| `LLM_MODEL` | `llama3.2:3b` with Ollama, `ministral-8b-2512` with Mistral | to use another of the chosen provider's models, named as that provider names it. It belongs to the provider `LLM_PROVIDER` names: a command that picks the other one with `--provider` uses that provider's default |

### Setting up Ollama

Everyone runs their own copy: install Ollama and pull the model on your own
computer. Install it from <https://ollama.com/download> (on Windows,
`winget install Ollama.Ollama` also works). It runs in the background and
listens on `127.0.0.1:11434`. `127.0.0.1` always means the computer the code
runs on, so that address reaches your own Ollama and never a teammate's, and
Ollama accepts no connections from other computers unless it is configured to.
There is no API key. Then download the model the code uses by default:

```bash
ollama pull llama3.2:3b     # 2.0 GB
ollama list                 # it should be listed
```

To use another model, `ollama pull` it first, then set `LLM_MODEL` to its name.
Two more settings in `.env` change how Ollama runs. Neither is required:

| Variable | Default | When to set it |
|---|---|---|
| `LLM_BASE_URL` | `http://127.0.0.1:11434`, your own computer | only if your own Ollama listens on a different port |
| `LLM_NUM_GPU` | Ollama decides | `0` on a laptop with a small GPU, as explained below |

### Setting up Mistral

Only needed to answer with Mistral. Everyone uses a key from their own account.
A key works like a password, and the free plan's rate limits are per account, so
never share yours or use a teammate's. That includes the demo: whoever presents
uses their own key on their own computer.

1. Sign up at <https://console.mistral.ai> and choose the free plan. It needs no
   credit card; signing up with an email address asks you to verify a phone
   number.
2. Turn off training on your data. On the free plan Mistral may train its
   models on your API calls unless you opt out: in the Admin Panel, open
   Privacy and turn off the toggle under Anonymous improvement data.
3. Open API Keys, choose Create new key, and give it a name such as `bt4103`.
   Set its expiry to a date after the demo, or leave it without one, since
   Mistral refuses an expired key. Create it and copy it straight away: the
   full key is shown only once.
4. Put it in your own `.env`: remove the `#` from the `# MISTRAL_API_KEY=` line
   `.env.example` leaves for it, paste your key after the `=`, and choose the
   provider:

   ```
   LLM_PROVIDER=mistral
   MISTRAL_API_KEY=<your key>
   ```

   `.env` is git-ignored, so the key stays on your computer. Never paste it into
   code, a notebook cell, a commit, an issue or a chat. `LLM_PROVIDER=mistral`
   makes Mistral what answers everywhere. Leave that line out and the commands
   answer with Ollama, and the app opens on Ollama with Mistral one click away
   in its sidebar's Answer model box.
5. Ask a question that needs the model, such as the one in
   [From a question to an answer](#from-a-question-to-an-answer).
   `answer.config.provider` should be `'mistral'`.

If you use an AI coding agent in this repository, keep it out of `.env`: in
Claude Code, add `"permissions": {"deny": ["Read(**/.env)"]}` to your own
`.claude/settings.local.json`. If a key is ever exposed, delete it under API
Keys and make a new one.

A missing key, a key Mistral refuses, a model your plan does not include, the
rate limit or a used-up quota, a server error, a lost connection and an answer
that ends before it is finished each raise `ProviderUnavailable` saying what to
do. So do the ways a network can get in the way: a firewall's page in place of
Mistral's answer, a proxy that refuses, or one that re-signs HTTPS so the
certificate does not verify. A key changed in `.env` is read when the notebook
or command starts again. `answer_question` does not ask again, and nothing
falls back to Ollama by itself: set `LLM_PROVIDER=ollama` to answer locally
again. The failures that asking again may fix raise `ProviderBusy`, a kind of
`ProviderUnavailable`, which the [evaluation harness](#the-benchmark) asks again
before it stops a run.

`ministral-8b-2512` is the default because it did best of the four suitable
chat models the free plan serves. Voxtral Small and Codestral stated figures
the passages did not hold. Of the two Ministral models, 8B stated more figures
to the exact digit at `FINAL_K` 16 (20 of 28, against 17 for 14B), abstained
less (2 against 4), and is allowed six times as many requests a minute. The
hosted runs under [Searching the indexes](#searching-the-indexes) and
[the evaluation](docs/mistral-free-tier-evaluation.md) have the details.

### From a question to an answer

```python
from src.rag import answer_question, config_from_env, parse_question
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.constants import FINAL_K
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever

hybrid = HybridRetriever(BM25Retriever.load(), DenseRetriever.load())

question = "What supply chain risks did Apple describe in its FY2024 10-K?"
parsed = parse_question(question)            # AAPL, FY2024, a factual question
answer = answer_question(
    question, hybrid, config_from_env(), query=parsed.to_query(top_k=FINAL_K),
    on_token=lambda t: print(t, end=""),
)
answer.text                    # resolved prose with [n] markers, or an abstention
answer.abstention_reason        # why no answer was given, or None
```

The shared entry point resolves citations before returning the answer. Verify
the completed answer before showing its final citations and checks:

```python
from src.rag import render_citation, verify_answer

answer = verify_answer(answer, parsed=parsed)
print(answer.text)  # unresolvable markers have been removed
for sentence in answer.flagged_sentences:
    print("Citation warning:", sentence.warning_text)
for check in answer.verification_warnings:
    print("Verification warning:", check.status, check.claim, check.reason)
for citation in answer.citations:
    print(render_citation(citation, answer.passages))
```

A label reads `[1] Alpha Corp (AAA, CIK 0000000123), 10-K, fiscal year 2024,
Part II / Item 7, Management's Discussion, filed 2025-02-01`. Every field comes
from stored passage metadata, with missing values explicitly shown as unknown.
The filing URL stays on `answer.passages[citation.marker - 1].url` for resolved
citations, so the app can link the label to the source.

Invented markers remain in `answer.citations` with `resolved=False` and
`chunk_id=None`, even after removal from the displayed text. A sentence with
an invented marker or no marker is flagged; a valid marker elsewhere does not
clear that warning. `answer.sentences` stores the original and cleaned text,
its citations, and its `flagged` status, all included in `answer.to_dict()`.
Resolution establishes source identity; `verify_answer` then checks the figures
against each sentence's cited evidence and, for questions requesting figures,
the local facts store. It returns a new `Answer` with immutable checks under
`answer.verification`; the model's answer text is preserved.

### Verifying and reviewing an answer

Build the facts store once with `python -m src.retrieval facts`. Verification
reads it locally and never downloads data. A missing or unreadable store is a
visible `unverified` result. Company, fiscal year, annual period, financial
concept and unit must agree; a matching value from a different company,
quarter or concept cannot validate a claim. Figures use decimal arithmetic,
including currency scales, percentages, basis points and accounting negatives.
Rounded claims are compared at the precision they display. Markdown table
checks use the row's metric, year column and declared scale.

The verifier confirms extractive wording but marks paraphrases for review.
Numeric consistency is a proxy for faithfulness, not proof of semantic
entailment. The initial financial-concept aliases cover revenue, net income,
operating income, total assets, total liabilities, cash and cash equivalents,
basic/diluted EPS and operating cash flow. Unknown concepts, ambiguous
multi-company or multi-year sentences, calculations and unsupported number
notation need review; they are not silently marked supported. Comparative
answers can use one company and year per sentence for an unambiguous check.

Four rules keep the checker from flagging an answer it has the evidence for.
A dollar amount for a per-share line item is dollars a share whether or not it
says so, since a statement prints "$4.67" in the row and an answer writes "EPS
of $4.67". One with a scale word after it, as in "it returned $94 billion to
shareholders", is a dollar total and is read as one. The day in a written date
("September 30, 2023") is not a figure,
though its year still scopes the claim. A name inside a longer name of another
line item is part of the longer one, so "diluted net income per share", which
is what Adobe, Salesforce, Alphabet and Intuit call earnings per share, names
earnings per share and not net income, to the router and the checker alike.
Without "basic" or "diluted" in front, "net income per share" does not say
which of the two it is, so it names neither, and it does not name net income:
a claim in those words is checked as earnings per share where the question
asked for that and is `unverified` otherwise, and a table row in those words
("Net income per share - diluted", as ServiceNow, Broadcom and Palo Alto print
it) is a row that names no line item.
And where the facts store confirms a figure for the line item and year, a
cited passage that prints it without tying it to them, in a table row that
names no line item or in a sentence beside more than one year, leaves the
passage check `unverified` with the reason, not `mismatch`. Where the store
cannot confirm the figure such a passage is a mismatch, as before.

What the answer card showed for the answers to the 825 headline questions,
with Hybrid, before those rules and with them:

| What the checker says of an answer | Supported | Unverified | Mismatch |
|---|---|---|---|
| The facts route's 756 answers, before (`headline-6-common-words-hybrid-checked`) | 700 | 0 | 56 |
| The same, with the rules (`headline-9-final-hybrid`) | 756 | 0 | 0 |
| The 6 a model answered right, before | 1 | 0 | 5 |
| The same, with the rules | 3 | 0 | 3 |
| The 29 answered where the store has no single figure to grade, before | 1 | 17 | 11 |
| The same, with the rules | 8 | 16 | 5 |

Every one of the 56 was a right answer the store had just supplied. The
checker is still weakest on line items the facts route does not cover: of the
150 right answers in `line-items-9-final-hybrid` it marks 48 a mismatch, 62
unverified and 40 supported, since no store concept is mapped for those lines
and the cited passage has to tie the figure to the line item itself.

These rows were recorded under an earlier rule for summing an answer up, and a
run made since uses the evaluation harness's (`verify.worst_check`), which
differs in one way that reaches them. One supported figure made a whole answer
supported, where an answer is now counted by its worst figure, so one that
also states a figure the checker could not check is unverified. (The harness's
`unchecked` does not come into it: it is for a question that asks for no
figure, and every question in these two sets asks for one, so an answer
without a figure is unverified under either rule.) The facts route's answers
state one figure each, so their rows
are the same under either rule. For the answers a model wrote, the supported
counts above are an upper bound: a new run can move an answer from supported
to unverified and not the other way.

The evaluation harness checks every answer this way before it records it, so
a saved run carries its checks. Write a run's answers and open them in a
browser:

```bash
python -m src.evaluation benchmark/questions.jsonl --run-id hybrid-baseline --output logs/evaluation.json --answers logs/evaluation-answers.jsonl
python -m src.app.answers logs/evaluation-answers.jsonl --output logs/answers.html
```

Warnings appear above each answer, with expandable check details and source
passages. For answers held in memory,
`write_answer_page([answer], Path("logs/answers.html"))` writes the same page.

Every JSONL row retains the question/run IDs, generation configuration,
answer, sources and individual checks. `numeric_support_rate` is the supported
numeric checks divided by all applicable passage/fact checks; unverified checks
stay in the denominator. Abstentions have no numeric score (`null`). Keep the
groundedness checks separate when evaluating semantic faithfulness.

An abstention has no sentence warnings. Malformed or truncated output keeps
its `parse_error` and `truncated` status. When structured sentence boundaries
cannot be recovered, the available prose is kept as one flagged block.
Streamed prose is provisional; replace it with the resolved answer and show
its warnings when generation finishes. The app does: it passes `on_token`,
writes the prose to the page as it arrives, and clears it once the checked
answer card is ready, or when the provider fails part way.

`parse_question` reads the companies, fiscal years and question type out of the
question, and `parsed.describe()` says what it read. The app shows it under
the answer, in the scope the sidebar ended with ("Question type: numeric ·
Companies: AAPL · Fiscal years: FY2024"), so the user can see when the reading
was wrong. A question read as unanswerable and searched all the same says
"unanswerable, searched in case a filing answers it", since the line sits
under whatever answer a filing gave. Where a question was split into one
search per filing, the same line names the filings (`Answer.sub_questions`).
A company the corpus does not hold, such as Intel, is reported in
`parsed.unresolved` rather than silently ignored. `build_prompt` numbers the
passages as sources, puts the rules above them, and never shows the model a
URL.

Numeric cues include the metric aliases used by the facts router and the XBRL
labels in `data/index/facts.parquet`. Newly supported labels need a request
for a figure: "What was Apple's commercial paper?" is numeric, while
"What is Apple's commercial paper program?" remains factual. The parser reads
only the label column and caches successful reads until the file changes;
failed reads are retried. Built-in cues remain available without a store.
Pass `facts_file=...` to `parse_question` to use another store; both
`answer_question` and `verify_answer` pass their store through automatically.
Use `facts_file=None` to disable label lookup, as verification does when it
only needs sentence-level entities. Possessive total questions match recognised
company names. Comparative, temporal and unanswerable classifications keep
their existing priority over numeric cues.

A line of the statements that the facts route does not answer is a cue as
well, by the name a question gives it: accounts receivable, income tax
expense, stock-based compensation, capital expenditures and the others in
`FIGURE_LINE_ITEM_CUES`. The XBRL label of such a line is not what a question
calls it ("Accounts Receivable, after Allowance for Credit Loss, Current"), so
the store's labels did not cover them, and "What was Microsoft's accounts
receivable in fiscal year 2024?" was read as prose and searched with no lean
toward tables. Like a stored label, each needs a request for the figure: "How
does Microsoft manage accounts receivable risk?" stays factual.

The app reads a question with the store's labels too. It used to pass
`facts_file=None`, so a question naming a line item by its label, such as
gross profit, was searched as prose in the app and as a figure question by
`answer_question` and the evaluation harness. `filter_sidebar` reads a
question the same way when it is not handed a parse. What both changes do to
the answers is in
[How often the answers are right](#how-often-the-answers-are-right).

Accounts payable, inventories and net sales questions can use the facts route
when a matching figure and supporting passage exist. Recognising another
stored label enables numeric retrieval, but a direct facts answer still needs
an unambiguous metric mapping and citation; otherwise it falls back to retrieval.
The supporting search uses the question's metric alias, so "net sales" searches
for that wording in the filing. Inventory purchase obligations, reserves and
write-downs require different concepts and cannot be checked against `InventoryNet`.

The route looks through the top 50 passages of that search for one that prints
the figure (`FACT_PASSAGE_K`), where it used to look through 20. The figure was
in the store and the statement in the filing, but a search for "total revenue"
ranks Amazon's income statement, which says "net sales", below the prose that
uses the word. 50 is what Hybrid already fetches for any smaller request, so
it searches and scores nothing more.

Where that search finds nothing to cite, the route searches once more, for
the line item's other names together. A question need not use the filer's
word: "What was Amazon's total revenue in fiscal year 2024?" is a search for
"total revenue" and Amazon's statement prints "Total net sales", and only
Adobe calls its accounts payable "trade payables". The figure is the same
whatever the line is called, and the passage still has to print it in a row
that names the line item, so the second search changes where the route looks
and not what it accepts. It runs only after the first has found nothing, on a
question that was on its way to a model, so an answer the route already gave
cannot change.

A passage prints the figure when it shows it at a scale it declares or beside
a scale word, or in a table row that names the line item. A filer that reports
in millions to one decimal place is matched in that form as well: Palo Alto
Networks prints total assets of $10,241,600,000 as "10,241.6" and never as a
whole number of millions, so the route could cite none of its statements. A
decimal has to sit in a row that names the line item or carry its scale word,
since a table in millions holds rates written the same way.

A line item answers to the names the statements give it as well as the names
a question uses: "income from operations" (Salesforce, Alphabet, Meta and
ServiceNow), "operating profit" (Texas Instruments), "cash and equivalents"
(Micron), "trade payables" (Adobe), and the cash flow statement's line for
cash from operations, which few filers word alike. Each was read off the rows
that print a stored figure across the 75 filings. The route cites a row only
where it names the line item and the checker reads a row the same way. Before
these were added the route answered operating cash flow for no filing of
Apple, Amazon, Salesforce, Cisco, Alphabet, Meta or Micron, and where it did
cite a passage for operating cash flow or operating income the checker marked
the route's own figure a mismatch against that passage, 27 times in the 623
answers it gave with BM25. "Inventory", singular, is left out though Alphabet
and Broadcom print it: the word is in too much prose about purchase
commitments for a sentence that uses it to be checked against the balance.

How many headline questions the route answers at each step is in
[How often the answers are right](#how-often-the-answers-are-right).

`parsed.to_query()` gives BM25 a different text from dense search. Once the
filters confine the search to Meta's FY2025 filing, "Meta's" and "fiscal year
2025" tell no passage in it apart, but BM25 still scores the prose that repeats
them above the balance sheet, which never does. So BM25 matches
`parsed.search_text`, the question without the companies and years the filters
apply ("What was total assets at the end?"). Dense search still reads the whole
question, since the embedding uses the company name to place it. On the 48 test
questions, this split put the expected figure in the hybrid top 8 for 18 of 28
figure questions, against 12 with the whole question and 16 with the trimmed
text for both retrievers, and prose questions improved too. The comparison is
`python notebooks/retrieval/search_text_comparison.py`.

`generate` returns a `Generation`: `answer`, the parsed answer; `text`, the
same answer as prose; `raw`, exactly what the model emitted; and the
`latency_ms`, `input_tokens`, `output_tokens` and `stop_reason` of the call.
`stream` is the same call as a generator, for writing the answer into a page as
it arrives. Joined, what it yields is `generation.text`.

### What keeps the answer on the sources

`answer_question` checks retrieval before building a prompt or contacting
the model. If there are no usable passages, it returns the fixed sentence
"The filings do not answer this question." with `Answer.abstained=True`, no
citations, and an `abstention_reason` that the browser viewer displays:

- `filters_excluded_all`: the selected metadata filters exclude every indexed
  passage. Candidate checks inspect the actual index before score thresholds.
- `below_threshold`: matching passages exist, but none meet the score floor.
- `no_evidence`: the index is empty, returned evidence is unusable, or a custom
  retriever cannot report why it returned nothing.
- `model_declined`: passages reached the model, but it declined to answer.
- `beyond_the_filings`: the question asks for advice, or for a prediction of
  the assistant's own. Nothing was searched.
- `company_not_in_corpus`: the question asks for a figure of a company the
  corpus holds no filings for. Nothing was searched.

The last two are refusals, decided from the question alone. Searched, "Is
Meta a good investment?" gets sixteen passages about Meta and a model that may
answer from them, and a question about Intel's revenue gets sixteen passages
from other companies. `parse_question` already read both as unanswerable, and
`ParsedQuestion.unanswerable_because` now says why (`request`, `company`,
`topic` or `year`), so `answer_question` can refuse the first two kinds before
it searches. A figure is an outside company's own when the question names the
company as its owner: by a possessive ("Intel's revenue"), after "of" ("the
revenue of Intel"), or as the subject of a verb of having or reporting ("how
many employees did NVIDIA have"). Named any other way the company is a topic.
"Did NVIDIA account for more than 10% of any company's revenue?" asks the
filings in the corpus about their customers, and a wording the rule does not
list, "Intel revenue in FY2024?", is searched and left to the model.
`ParsedQuestion.refused` is the one place that says whether a reading is
refused, so the line under an answer and what was done cannot disagree.

The other two kinds are still searched, because a filing may answer them. A
refusal that is wrong costs the answer, and a search that finds nothing costs
one abstention by the model. `topic` is a question about a share price, about
next year, or about a company outside the corpus that is not asked for a
figure of its own. The words do not tell "What is Apple's current stock
price?" from "What average share price did Apple pay for repurchases in
FY2024?", which Item 5 answers, or "What will Microsoft's revenue be next
year?" from "What were Microsoft's purchase obligations due next year in
FY2024?", which the contractual obligations table answers. The filings the
corpus holds also name the companies it does not, as competitors, suppliers
and customers, so "Which companies named NVIDIA as a competitor in FY2024?"
is a question about them. `year` is a question naming only a fiscal year
outside the corpus: a filing prints the two years before its own, so Apple's
FY2020 revenue is in its FY2021 statements and the app answers it, and it
says what falls due in the years after. A question is also searched when the
caller's `Query` names a company, as the app's sidebar lets a user choose one
by hand.

`python notebooks/answers/refusal_check.py` counts the answerable questions
the refusal would turn away: none of 16,006 (the 48 test questions, every name
of every headline line item, every line-item question for every year, and the
12,579 of the generated benchmark). One benchmark question was refused until
the parser stopped reading "Loss Contingency, Estimate of Possible Loss" as an
instruction to estimate. Those sets hold no question about a repurchase price
or an obligation due next year, which is how an earlier rule that refused on
those words passed the same count, so four such questions are among the
script's probes now. The script prints how each probe is read: seven that
should be refused, and nineteen that should be searched.

With `--provider mistral` it also asks a model each probe with the refusal off
and on. That was run on fifteen probes, under the earlier rule. Of the six
that are still refused, the model declined five on its own and answered "Is
Meta a good investment?" with Meta's spending plans. It also declined the two
that are now searched instead, "What will Microsoft's revenue be next year?"
and "What is Apple's current stock price?", so searching them shows a user
the same abstention. The seven that were always searched came out the same
both ways. The eleven probes added since have not been asked of a model.

Pass `min_score=<calibrated value>` to `answer_question` to add an inclusive
floor on the selected retriever's final scores. Its internal thresholds also
remain in force. `None` adds no floor; the existing defaults remain unset
pending benchmark calibration (#26). BM25, cosine similarity and fused ranks
have different scales and must not share an arbitrary cutoff.
The gate prevents generation on empty evidence; score alone does not prove
that a nonempty set answers the question, so the model can still abstain.

The built-in BM25, dense and hybrid retrievers support candidate checks.
Custom retrievers can add `has_candidates(query)` to report metadata matches;
without it, an empty search still abstains but uses `no_evidence`. Index and
provider errors propagate instead of being counted as abstentions. A zero
`top_k` is rejected as a configuration error. Stream callbacks receive the
fixed abstention sentence immediately when retrieval cannot supply evidence.

The lower-level `build_prompt`, `generate` and `resolve_citations` functions
remain available for experiments with known evidence. Use `answer_question`
for the full retrieval path; `build_prompt` deliberately rejects empty sources.

The model does not write free text. `GroundedAnswer` in `src/rag/records.py` is
a Pydantic model: whether the sources answer the question at all, then the
answer as a list of sentences, each with the numbers of the sources it draws on.
Its JSON schema goes with every request, as Ollama's output format or as
Mistral's strict JSON-schema response format, and either one restricts the
model's decoding to that shape. For a prompt with eight sources the schema only
admits the numbers 1 to 8, so the model cannot cite a source it was not shown:
asked outright to cite source 9 of 3, the model wrote `[2]`. The same Pydantic
model validates the output afterwards.

Three things follow from that.

- An abstention is a field, not a sentence to reproduce. When the model sets
  `answerable` to false, `generation.text` is the fixed sentence "The filings
  do not answer this question.", and `generation.answer.abstained` is true.
- The model writes JSON, but `stream` yields prose. It reads the partial JSON
  as it grows and adds a sentence's `[n]` markers once that sentence is closed.
- An answer cut off at the token limit does not parse. `generation.answer` is
  then None, `parse_error` says why, `truncated` is true, and `text` still holds
  what arrived.

With Ollama, the context window is set on every request, to 8,192 tokens
(`NUM_CTX`). This matters more than it looks. When a prompt is longer than
Ollama's window, Ollama cuts it from the front without telling the caller: the
only sign is a warning in its own server log. Run with a 2,048-token window, it
kept 1,026 of 3,205 prompt tokens, dropping the rules and the first sources,
and the model answered a question about revenue with a paragraph about hiring.

So the window is what caps `FINAL_K`, and #85 measured the fit rather than
assuming it. Counted with llama3.2's own tokenizer over the XBRL benchmark
across all 75 filings and the 48 hand-written questions, a prompt over 16
passages runs to a median of about 5,200 tokens and 6,306 at its largest,
which leaves room for `MAX_OUTPUT_TOKENS` beside it, 7,330 against a window of
8,192. That is the largest seen, not a bound: the 16 largest passages of one
filing can reach 7,561 tokens, where the ceiling no longer fits beside the
prompt for 27 of the 75 filings, but none can pass the window itself, so the
prompt is never cut. Twenty passages do not fit: 7,703 plus the output ceiling
is 8,727, and every filing's 20 largest passages pass the window itself. An
eight-passage prompt, the earlier setting, ran 2,700 to 3,400 tokens. Ollama's
own default depends on the GPU's memory and is 4,096 tokens on a laptop, which
is why `NUM_CTX` is set explicitly on every request rather than left to it.
None of this applies to Mistral, whose models have windows far larger than any
prompt here.

With Ollama, a server that is not running, a model that has not been pulled, a
response that times out, a full queue on a shared server, a proxy that catches
this computer's own address, or a connection Ollama closes part-way, cleanly or
not, raises `ProviderUnavailable` saying what to run or check.
[Setting up Mistral](#setting-up-mistral) lists what Mistral's raise.

### How long an answer takes

With Mistral, about two seconds: over the 48 test questions on the free plan,
the median from question to full answer, retrieval included, was 2.2 s at
`FINAL_K` 16. With Ollama, minutes on a laptop, and the rest of this section is
about that. Measured on a team laptop (Intel i5-1135G7, 16 GB of RAM, an NVIDIA
MX450 with 2 GB), with a browser and an editor open, over real questions from
the corpus:

| Model | Where it ran | Reading the prompt | Writing | First words appear | Whole answer |
|---|---|---|---|---|---|
| `llama3.2:3b` | CPU (`LLM_NUM_GPU=0`) | about 17 tokens/s | about 3 tokens/s | 2.1 to 3.1 min | 2.3 to 3.4 min |

These are ten questions in one sitting, and the same laptop is slower when it
is busier or warmer: run again later the same day, the Apple question from the
snippet above took 4.2 minutes against 2.3 the first time, reading the prompt
at 13.6 tokens/s. Treat the table as typical rather than as a bound.

Nearly all of that is the model reading the prompt, 2,700 to 3,400 tokens,
before it writes anything; the answers themselves ran from 13 to 151 tokens.
`llama3.1:8b` took 10.4 minutes on the same laptop for the first of these
questions, against 3.2 for the 3B model, which is why the smaller model is the
default. Comparing models properly is #46's job.

These times were measured at the old `FINAL_K` of 8. Re-timed at 16 on the
same laptop, with the model reloaded before each answer so nothing was cached,
two questions took 4.1 and 4.3 minutes against 2.0 and 2.4 at 8: about twice
as long, as reading the prompt at a roughly constant rate predicts, and the
local price of the retrieval gain #85 measured. That gain is smaller locally:
of the three questions 16 newly brings the figure for (Q35, Q38 and Q40),
`llama3.2:3b` stated none, where both hosted Ministral models stated all three.
Three things follow.

- Stream the answer (#42), and show the passages first. On this hardware they
  arrive minutes before the first word of the answer.
- On a laptop with a small GPU, set `LLM_NUM_GPU=0`. Ollama put 3 of the 3B
  model's 29 layers on the MX450, and the split ran 2.4 times slower at reading
  the prompt and 4.4 times slower at writing than the CPU alone. On a capable
  GPU or an Apple silicon Mac, leave it unset and let Ollama place the model.
- The first request also loads the model into memory, which took 10 to 50
  seconds here. Ollama unloads a model after five idle minutes, so the next
  request pays for loading it again.

### How often the answers are right

`python notebooks/answers/app_path_accuracy.py` asks the 48 test questions the
way the app does, through `answer_question`, and checks the answers
automatically. The Mistral notebook calls retrieval and the model itself, so it
never takes the facts route or the split of a multi-filing question. This runs
both, so it measures what a user is shown. The checks are the notebook's: a
figure question is right when the answer states every expected figure, rounding
allowed, and a prose answer is scored by the share of the expected items it
mentions. They are proxies, as they are there.

It asks every question three times, because a hosted model does not repeat
itself. Mistral's API gave three different answers to one prompt at temperature
0, with or without a seed, and where the figure was missing from its sources
the same model abstained in some answers and stated a wrong figure in others.
Retrieval and the facts route are the same in every run, so the spread across
runs is the model's.

With `ministral-8b-2512` at `FINAL_K` 16, three runs of the 48 at each state of
the code. Each row is a file `notebooks/answers/app_path_accuracy.py` writes to
its git-ignored `results/` folder, named in brackets, and the three
numbers in a cell are the three runs. The files these rows were read from were
committed and then removed. `notebooks/removed-results.json` lists every such
file with its row count, its SHA-256 and the git object that holds it, so
`git cat-file -p <git_blob>` prints the one a number came from:

| 48 test questions | Figures right, of 28 | Wrong figure or none | Abstained | From the facts store | Figure in the passages, of the 20 a model answered | Prose: expected terms in the answer |
|---|---|---|---|---|---|---|
| main, BM25 (`0-main-bm25`) | 23, 23, 23 | 4, 4, 3 | 1, 1, 2 | 8 | 16 | 0.87, 0.88, 0.87 |
| main, Hybrid (`0-main-hybrid`) | 26, 26, 26 | 1, 1, 1 | 1, 1, 1 | 8 | 18 | 0.92, 0.91, 0.93 |
| Table boost 1.2, BM25 (`1-table-boost-bm25`) | 26, 25, 25 | 0, 1, 1 | 2, 2, 2 | 8 | 18 | 0.85, 0.87, 0.90 |
| Table boost 1.2, Hybrid (`1-table-boost-hybrid`) | 26, 27, 26 | 1, 0, 1 | 1, 1, 1 | 8 | 19 | 0.93, 0.92, 0.95 |
| Facts route changes, BM25 (`4-statement-names-bm25`) | 25, 25, 26 | 1, 1, 0 | 2, 2, 2 | 8 | 18 | 0.88, 0.88, 0.89 |
| Facts route changes, Hybrid (`4-statement-names-hybrid`) | 27, 27, 27 | 0, 0, 0 | 1, 1, 1 | 8 | 19 | 0.93, 0.91, 0.92 |
| Second citation search, BM25 (`5-other-names-bm25`) | 26, 26, 25 | 1, 1, 1 | 1, 1, 2 | 8 | 18 | 0.87, 0.87, 0.87 |
| Second citation search, Hybrid (`5-other-names-hybrid`) | 27, 27, 26 | 0, 0, 1 | 1, 1, 1 | 8 | 19 | 0.90, 0.93, 0.92 |
| Common words left out, BM25 (`6-common-words-bm25`) | 26, 27, 26 | 2, 1, 2 | 0, 0, 0 | 8 | 19 | 0.88, 0.90, 0.88 |
| Common words left out, Hybrid (`6-common-words-hybrid`) | 28, 28, 28 | 0, 0, 0 | 0, 0, 0 | 8 | 20 | 0.92, 0.93, 0.92 |
| Final state, BM25 (`9-final-bm25`) | 26, 26, 27 | 2, 2, 1 | 0, 0, 0 | 8 | 19 | 0.89, 0.88, 0.90 |
| Final state, Hybrid (`9-final-hybrid`) | 27, 28, 28 | 1, 0, 0 | 0, 0, 0 | 8 | 20 | 0.90, 0.93, 0.91 |

The figure columns barely move between runs and the prose column moves by a
point or two, so one figure question is a real difference and 0.02 of prose
terms is not. A model that is not given the figure states a wrong one more
often than it abstains. Of the four figures BM25 never put in front of it on
main, it stated a wrong one in all three runs for two (Oracle's FY2025 net
income, and Salesforce's share of revenue from the Americas), in two runs of
three for Google's marketable securities, and abstained on Salesforce's
goodwill. BM25's fifth miss had the figure among its passages and gave the
neighbouring year's: 46% for Google's FY2022 share of revenue from the United
States, which was 48%.

The table boost puts the figure in front of the model for two more of BM25's
questions and one more of Hybrid's. BM25 now gets three more right: Google's
share of revenue from the United States, Oracle's net income and Salesforce's
share from the Americas. It loses Microsoft's effective tax rate in two runs
of three: the filing gives it twice, 17.6% in the tax note's table and a
rounded 18% in the MD&A's prose, and with both among its passages the model
gave 18%. Hybrid gains Google's marketable securities in its passages, at rank
10, and the model stated it in one run of three and the equity securities'
figure in the other two. Salesforce's goodwill is still not retrieved by
either.

The facts route changes below do not touch these 48: the route answers the
same eight, and the other questions reach the model with the same passages as
before. So the three pairs of rows from the table boost to the second
citation search are one state asked three times, and the differences are the
model's. With Hybrid it stated Google's marketable securities in one run of
three, then in all three, then in two.

Leaving the common words out of the keyword search (see
[Searching the indexes](#searching-the-indexes)) does change what the 48 are
searched with. With Hybrid every figure is right in all three runs.
Salesforce's goodwill is printed in the fifth passage, where no passage that
prints it had been among the sixteen, and Google's marketable securities is
stated every time. In one run of the three the goodwill answer gives an
acquisition's goodwill first and the balance after it, which the check counts
as right because the expected figure is stated. With BM25 alone the figure is
among the passages for 19 of the 20 questions the model answers, from 18, and
Google's marketable securities is now right. Salesforce's goodwill turns from
an abstention into a wrong answer: BM25 now ranks an acquisition's table among
the sixteen and the passage with the balance seventeenth, one past the cut,
and the model reads the goodwill off the table it was given.

The last pair of rows is the branch as it ends, after the changes to how a
question is read, the checker and the refusal. None of them changes what
these 48 are searched with, so the rows are the state above asked again. The
one Hybrid miss is Google's marketable securities in one run of three, the
equity securities' figure given for the total, with the right figure among
the passages. BM25's are Salesforce's goodwill in all three runs and
Microsoft's effective tax rate, 18% for 17.6%, in two.

Those 48 questions cover eight of the fifteen companies, and most are answered
right. `python notebooks/answers/headline_figures.py` asks the plain question
for each of the 75 filings and each line item the facts route supports, such
as "What was Micron's operating income in fiscal year 2023?", 825 in all, and
grades the answers against the XBRL store. It separates the two ways such a
question is answered: from the facts store, exactly and with no model, or by a
model from retrieved passages where the route gave way. Without `--provider` it
asks no model and counts what the route answers, which needs no key and comes
out the same on every run.

| 825 headline questions, one run | From the facts store | Model: right | Model: right, to fewer digits | Model: a wrong figure or none | Model: abstained | Right, of the 762 the store can grade |
|---|---|---|---|---|---|---|
| main, BM25 (`headline-0-main-bm25`) | 623 | 43 | 24 | 44 | 28 | 690 |
| main, Hybrid (`headline-0-main-hybrid`) | 637 | 58 | 22 | 28 | 17 | 717 |
| Table boost 1.2, Hybrid (`headline-1-table-boost-hybrid`) | 651 | 87 | 5 | 8 | 11 | 743 |
| Facts route changes, BM25 (`headline-4-statement-names-bm25`) | 747 | 3 | 0 | 1 | 11 | 750 |
| Facts route changes, Hybrid (`headline-4-statement-names-hybrid`) | 752 | 8 | 0 | 1 | 1 | 760 |
| Second citation search, BM25 (`headline-5-other-names-bm25`) | 752 | 3 | 0 | 1 | 6 | 755 |
| Second citation search, Hybrid (`headline-5-other-names-hybrid`) | 756 | 6 | 0 | 0 | 0 | 762 |
| Common words left out, BM25 (`headline-6-common-words-bm25`) | 752 | 3 | 0 | 2 | 5 | 755 |
| Common words left out, Hybrid (`headline-6-common-words-hybrid`) | 756 | 6 | 0 | 0 | 0 | 762 |
| Final state, Hybrid (`headline-9-final-hybrid`) | 756 | 6 | 0 | 0 | 0 | 762 |

The other 63 have no single figure in the store to grade against: a software
company has no inventories, some filers report no total for liabilities, and
Oracle tags two net incomes that disagree. The store holds the figure for
every one of the 762, so each question left to the model is one the route
found a figure for and then no passage to cite. The retriever decides that
too, which is why the first column differs between rows. Operating cash flow
is the largest group, 47 of the 75 with BM25 on main, and the model then gave
a wrong figure for 25 of them.

With the table boost the model's wrong answers fall from 28 to 8 and it states
the exact figure far more often, 87 against 58, because the statement table is
now among its sources. Thirty questions became right and four stopped being:
Adobe's accounts payable in four of its five years, where the model now
abstains. Adobe's balance sheet calls the line "Trade payables".

The first column is the facts route's, and it can be counted with no model
asked. With Hybrid, as each change to the route went in:

| Facts route, Hybrid, no model asked | Answered from the store, of the 762 it holds a figure for |
|---|---|
| Table boost 1.2 (`headline-1-table-boost-hybrid`) | 651 |
| Looking through 50 passages for the citation, not 20 (`headline-2-passage-depth-hybrid-no-model`) | 660 |
| A figure printed to one decimal of a million (`headline-3-decimal-millions-hybrid-no-model`) | 692 |
| The statements' own names for a line (`headline-4-statement-names-hybrid`) | 752 |
| A second search, for the line item's other names (`headline-5-other-names-hybrid`) | 756 |

The nine the deeper search added were answered by the model before: five right,
one to fewer digits, two wrong and one abstained. The 32 the decimal form added
are all Palo Alto Networks', and the model had 28 of them right and abstained
on four. The 60 the statements' names added are operating cash flow (39),
operating income (12), accounts payable (5) and cash (4), and the model had 46
right, four to fewer digits, five wrong and five abstained. No step lost a
question the route had answered before it.

Before the second search ten of the 762 were left to the model with Hybrid,
and it got eight right. The two it missed were Amazon's revenue for FY2024 and
FY2025, asked as "total revenue": Amazon's statement says "net sales", and a
search for the question's words does not reach it. The second search does,
and the route answers Amazon's revenue for all five years. The six still left
to the model are four inventories and ServiceNow's accounts payable in two
years, and it gets all six right, so every one of the 762 is answered right
with Hybrid. BM25 leaves ten and the model misses seven of them, all
inventories of Broadcom and Alphabet.

Those 825 questions use one name for each line item, the first in
`FINANCIAL_METRICS`. `--every-name` asks each filing once for every name a
line item has there, 2,325 questions, to see whether the route copes with a
question in words the filing does not use:

| Every name of every line item, no model asked | Answered from the store, of the 2,237 it holds a figure for |
|---|---|
| Before the second search, BM25 (`headline-4-statement-names-bm25-every-name-no-model`) | 2,035 |
| Before the second search, Hybrid (`headline-4-statement-names-hybrid-every-name-no-model`) | 2,164 |
| With it, BM25 (`headline-5-other-names-bm25-every-name-no-model`) | 2,225 |
| With it, Hybrid (`headline-5-other-names-hybrid-every-name-no-model`) | 2,229 |
| With the filers' names for earnings per share, Hybrid (`headline-8-checker-hybrid-every-name-no-model`) | 2,379 of 2,387 |

No question the route answered before was lost, and none of those answers
changed. Of the 65 Hybrid gained, 43 were asked as "trade payables" of a filer
that prints "accounts payable", 16 as "revenue", "revenues" or "total revenue"
of Amazon and Cisco, and six as ServiceNow's operating or net loss in years it
reported income. BM25 gained 190, most of them revenue asked in a word the
statement does not print: the dense half of Hybrid often gets from "revenue"
to "net sales" on the first search, and a keyword search cannot.

The last row adds "diluted net income per share" and "basic net income per
share", which is what four of the filers call earnings per share: 150 more
questions, all answered from the store, with every earlier answer unchanged.
The checker supports all 2,379.

Leaving the common words out of the keyword search changes the search the
route cites from, and not what the route answers. Run again after that change,
both sets come out as they were under both retrievers: the same 756 and 752
of the 825, the same 2,229 and 2,225 under every name, and each answer the
same sentence as before.

The facts route answers eleven line items. A filing reports many more, and a
question about one of those is answered by a model from the passages. `python
notebooks/answers/line_item_figures.py` asks the plain question for sixteen of
them, such as "What was Cisco's accounts receivable in fiscal year 2024?", of
every FY2024 filing whose store holds one figure for the line, 177 questions,
and grades the answers against the store as the headline questions are graded.
With Hybrid and `ministral-8b-2512`, one run on each side of the change to how
a question is read (see [From a question to an answer](#from-a-question-to-an-answer)):

| 177 line-item questions, FY2024, one run | Read as asking for a figure | Tables among the 16 passages, mean | Right | Right, to fewer digits | A wrong figure or none | Abstained |
|---|---|---|---|---|---|---|
| Before (`line-items-6-common-words-hybrid`) | 0 | 4.5 | 120 | 5 | 48 | 4 |
| Read as figure questions (`line-items-7-figure-questions-hybrid`) | 162 | 11.7 | 145 | 2 | 24 | 6 |
| Final state (`line-items-9-final-hybrid`) | 162 | 11.7 | 148 | 2 | 21 | 6 |

147 of the 177 are right where 125 were, and the wrong answers halve. 27
questions became right and five stopped being. Two of the five are abstentions
where the model had the figure before (Salesforce's purchases of property and
equipment, Micron's capital expenditures), two are a neighbouring figure
(Meta's depreciation of property and equipment for its depreciation and
amortization, and Texas Instruments' tax rate for its tax expense), and one is
Cisco's accounts receivable given as $6.7 billion with a second figure beside
it. The model does not repeat itself, so a question or two of the difference
in any one line is its own, as in the 48.

Twelve of the sixteen names are the new cues. Gross profit, interest expense,
depreciation and amortization and marketable securities are read as figure
questions through the store's labels, which the app's path now reads with.
The fifteen "share repurchases" questions are the ones still read as prose,
on purpose: a filing reports the cash paid for repurchases and the amount
bought under its programme as two figures, the question does not say which,
and read as a figure question it was answered right less often.

None of the 48 test questions and none of the 2,325 headline questions is read
differently, so the tables above stand as they are.

## Streamlit app and components

The app's Ask page reads the processed filings and indexes on this machine,
searches them, sends retrieved passages to the configured answer model,
verifies the result, and shows citations to those filings. Its Browse page
reads the processed passages directly, without an index or model:

```bash
streamlit run src/app/main.py
```

Run it from the project root. `python -m streamlit run src/app/main.py` does
the same.

For the Ask page, build the local indexes first if they do not exist (`python
-m src.retrieval bm25` and `python -m src.retrieval embed`). The app checks each
index against the current processed corpus before searching. The sidebar's
Configuration box
picks one of the rows in `src/stack.py` and opens on C4, hybrid retrieval with
the metadata filter, so the first Ask also checks the dense index and loads the
embedding model (25 seconds on the team laptop, 0.4 for the next question); C1
is BM25 alone and loads neither. The app opens on hybrid because, with the
sidebar's filters applied, it gets more of the test questions' figures right
than BM25: 26 of 28 against 23 before the table boost was set, 26 or 27 against
25 or 26 with it, and all 28 against 26 or 27 since the keyword search stopped
scoring the commonest words (see
[How often the answers are right](#how-often-the-answers-are-right)). A row
measured without the metadata filter searches every filing, and the sidebar
says so when one is picked.

The sidebar's Answer model box chooses what writes the answer: the local
Ollama model, or Mistral's free API (`ministral-8b-2512`), which answers in
seconds rather than minutes and needs your own `MISTRAL_API_KEY` in `.env`.
It opens on the provider `.env` names (`LLM_PROVIDER`, which unset means
Ollama), and switching it lasts for the browser session and changes nothing
in `.env`. Before this box the provider was `.env`'s alone to choose, so a
`.env` with a key and no `LLM_PROVIDER=mistral` line answered with the local
model, minutes at a time, with nothing on the page to say so.

A figure question the facts store answers needs neither provider, whichever
is picked: the chat model is built when an answer first needs one, not when
the configuration is loaded. So Mistral with no key still answers "What was
Apple's total revenue in FY2024?", and a question that does need the model
fails with what the provider lacks. The box does not wait for that. As
soon as a provider that cannot answer is picked, it says under it what the
provider lacks and what to do about it.

An explicit Item filter is enforced for numeric questions too. When the
question, the filters, the configuration or the answer model change, the app
hides the prior answer until Ask is pressed again. While a model is answering,
its prose is written to the page as it arrives, and the checked answer card
replaces it.

The Ask page (#38) shows, in order:

- the question box and Ask. Pressing Enter in the box reads the question;
  Ask answers it.
- the filters the question resolved to, as soon as it is read and before
  anything is searched: the companies, fiscal years and Items, and the
  question type. They follow the sidebar, so a filter changed by hand shows
  there too, and an empty one reads "every company".
- the outcome, the configuration, what wrote the answer (a model, or the facts
  store) and how long it took.
- the answer card with its inline citations. An abstention is stated in its
  place: "No answer from the filings", the reason, and what to try.
- the retrieval trace, four steps that each open onto their detail: how the
  question was read, what was found (every passage the answer was written
  from, in prompt order, with its rank, score, the retrievers that found it,
  whether it was cited, and a link to the filing), what wrote the answer, and
  what the checks found. It is drawn from the answer already given, so
  opening it searches nothing.

The Browse page (#40) reads `data/processed/` directly and needs neither an
index nor an answer model. Its company, fiscal-year and Item menus are
dependent: each contains only values that exist under the choices before it,
so every selectable combination has passages. It shows 25 passages at a time
and makes every page reachable. Each expander is named for its nearest heading,
table caption or Item title; inside it, the chunk ID is shown beside a link to
the source filing on EDGAR. Prose is marked with an article icon and rendered
as text; table passages are marked with a table icon and rendered in a
spacing-preserving block so the two cannot be mistaken for one another.

### Where things go in `src/app/`

Each file has one job, so that a second page does not grow its own copy of
what the first one does:

| File | What belongs in it |
|---|---|
| `main.py` | The entry point. Makes the project importable, sets the page title, lists the pages in `PAGES`, and keeps Streamlit's file watcher from importing transformers' alias modules. No page content. |
| `app_pages/<page>.py` | One page, as a script: what is asked, and the order the page is drawn in. It loads through `state.py` and draws with `components.py`. |
| `state.py` | Everything kept between reruns: `corpus_passages`, `load_stack` and `measured` (cached for the process), `Remembered` (answers already given), `keep` and `kept` (the answer a page is showing, for the `Request` it answers), `Stopwatch`. Also what a page reads from `.env`: `answer_models`. |
| `components.py` | What a page draws from the data it is handed: corpus selectors and passage panels, `filter_sidebar`, provider/configuration pickers, `resolved_filters`, the answer card and summary, abstention notice, and retrieval trace. A component builds no stack, asks no model and caches nothing. Widget selections are the only state one holds. |
| `answers.py` | The saved-answers viewer, a command of its own. Not part of the Streamlit app. |

To add a page, write `app_pages/<name>.py` and add one `st.Page` to `PAGES` in
`main.py`. The menu appears once there is a second page.

Issue #37's reusable UI is in `src/app/components.py`, with the Ask page's
pieces from #38.

- `resolved_filters(query, parsed)` shows what will be searched, in the same
  words as the sidebar's own "Searching:" line.
- `configuration_picker(runs)` is the sidebar's Configuration box, and returns
  the id picked. `runs` is `dict(state.measured())`.
- `provider_picker(state.answer_models())` is the sidebar's Answer model box,
  and returns the provider picked, to pass to
  `state.load_stack(config_id, provider)`. `answer_models` asks each
  provider's settings what it lacks with `check_provider` and builds no
  client to do it.
- `answer_summary(answer, config, seconds)` is the row above an answer.
- `abstention_notice(answer)` states an abstention. `answer_card` calls it for
  an abstained answer, so a page does not have to. `answer_card_html` is for
  an answer with claims and refuses an abstention.
- `retrieval_trace(answer, parsed, config, key="trace-id")` draws the four
  steps. `trace_rows(answer)` is the passage table's rows, one per passage.
- `answer_card(answer, key="answer-id")` displays a completed `Answer`. Inline
  markers open and focus the corresponding citation expander without another
  model call. Each expander holds the exact stored passage, the full source
  line (company, ticker, CIK, form, fiscal year, Part, Item, title, filing
  date) and the filing link, with missing metadata labelled unknown. Source
  numbering follows prompt order. Use a distinct, stable key for every card on
  the page. `show_question=False` leaves the question heading out, for a page
  whose question box is directly above the card.
- `filter_sidebar(question, parsed=None)` returns the effective `Query` to
  pass to `answer_question(query=...)`. It reflects companies and years from
  the shared parser and explicit Item mentions such as `Items 7 and 8`.
  Users can override or clear any selection; empty means unrestricted.
  Manual edits survive reruns. A changed question replaces all three filter
  axes, and **Use question filters** restores the extracted selections.
  Out-of-scope mentions produce a visible warning. Unrecognised Items remain
  selected rather than silently broadening retrieval.

Call the sidebar once per key on every rerun, outside a form. A controller
with an existing retriever can use the components as follows:

```python
import streamlit as st
from src.app.components import answer_card, filter_sidebar
from src.rag import answer_question, verify_answer

question = st.text_input("Question")
query = filter_sidebar(question)
if st.button("Ask", disabled=not question.strip()):
    answer = answer_question(question.strip(), retriever, query=query)
    st.session_state["last_answer"] = verify_answer(answer)
if "last_answer" in st.session_state:
    answer_card(st.session_state["last_answer"], key="last-answer")
```

An explicit Item filter sends numeric questions through normal retrieval,
because the facts shortcut cannot enforce Item filters. Verification remains
explicit: resolved citations alone are not labelled as factual support.
Missing, invalid, unresolved, incomplete and unverified claims carry visible
warning labels; mismatches have a separate label and colour. Model and filing
text is HTML-escaped, and only HTTP(S) filing links are clickable. The fixed
JavaScript click handler uses Streamlit 1.64's
[`st.html`](https://docs.streamlit.io/develop/api-reference/text/st.html);
untrusted text is never inserted into that script.

Acceptance checks are in `tests/test_app_components.py`: source mapping,
escaping, warning states, real Streamlit widget reruns and the Item-filter
handoff to the RAG engine. The Ask page is checked through the entry point in
`tests/test_app_live.py` and `tests/test_app_state.py`, which run
`src/app/main.py` as Streamlit does, with a stand-in for the built
configuration, so they need no index and no model. Run them with:

```bash
python -m pytest tests/test_app_components.py tests/test_app_live.py tests/test_app_state.py -q
```

## The benchmark

Hand-written questions go in `benchmark/questions.jsonl`, one JSON object per
line, and `benchmark/schema.md` lists the fields each one needs. The file does
not exist yet; it is filled as each member writes their questions.

```python
from src.evaluation import load_questions

questions = load_questions()   # benchmark/questions.jsonl, checked against data/processed/
```

A second, mechanical benchmark is generated from the XBRL facts store rather
than written:

```bash
python -m src.retrieval benchmark      # benchmark/generated.jsonl, from data/index/facts.parquet
```

Every current-year fact whose value can be found in a passage of the filing it
came from becomes one question, with those passages as its supporting chunks:
12,579 of them on today's corpus with a full facts store, all `numeric` and all `mechanical`. They are
narrow and repetitive by construction, and that is the point -- they are far
too many to write by hand, so they say whether a retrieval change holds across
the corpus or only on the questions someone chose. It needs no network, only
the facts store and a chunked corpus, and it is regenerated rather than
committed: re-chunking renames every chunk, and the loader rejects a benchmark
whose chunk ids no longer exist.

The loader refuses a file it cannot trust rather than skipping the bad lines:
a missing or unknown field, a duplicate `question_id`, a `question_type` other
than the five the RAG engine assigns (`factual`, `comparative`, `temporal`,
`numeric`, `unanswerable`), or a chunk id that is not in the corpus on disk.
The last one is the check that matters over time. Re-chunking renames every
chunk, so a benchmark written against an older corpus fails loudly on load
instead of scoring every retriever at zero.

A question does not have to be about one company in one year. `ticker` and
`fiscal_year` are `null` for a comparison or a change across years, and an
`unanswerable` question has no supporting chunks at all, only hard negatives:
the passages that look relevant, which the system should see and still
abstain from.

`RunResult` in `src/evaluation/records.py` is what the harness will record per
question and retriever: the chunk ids returned, their scores and the latency.
Pass a result and its benchmark question to `score_question` for the shared
retrieval metrics used across retrievers:

```python
from src.evaluation import score_question

metrics = score_question(question, result, k=10)
```

The returned row contains Recall@k, nDCG@k, reciprocal rank, hard-negative
accuracy, the cutoff, question type, retriever, and latency. Unanswerable
questions leave the supporting-chunk metrics unset and are evaluated through
their hard-negative accuracy instead.

The answer evaluation harness runs the same `answer_question` path and reports
abstentions divided by all completed questions, both overall and separately
for answerable and unanswerable questions. Each summary includes its total,
abstention count, rate and counts by reason. Empty subsets have a `null` rate,
and errors stop the run instead of inflating the abstention count. The one
exception is a failure that asking again may fix, such as Mistral's rate limit,
a server error, a full Ollama queue or a dropped connection (`ProviderBusy`):
the question is asked again after 10 seconds and then after a minute, or after
as long as the provider asks for, up to five minutes. Only the model is asked
again, with the passages and prompt the first attempt built, so a retry does
not search again. Each row records how many times its question was asked, as
`attempts`. When the provider still cannot
answer, or you press Ctrl-C, the run stops there, but the report and
`--answers` file are still written with every question before it, the report's
`stopped` names the question and the error, and the command exits 1. A mistyped
`LLM_PROVIDER` or a missing `MISTRAL_API_KEY` stops the command before it loads
anything. A high rate on answerable questions indicates lost coverage, not
better answer quality.

```bash
python -m src.evaluation benchmark/questions.jsonl --retriever hybrid --run-id hybrid-baseline --output logs/evaluation.json --answers logs/evaluation-answers.jsonl
python -m src.app.answers logs/evaluation-answers.jsonl --output logs/evaluation-answers.html
```

Add `--min-score <value>` for a calibrated floor, `--top-k` for retrieval depth,
`--provider ollama` or `--provider mistral` to override `LLM_PROVIDER`, or
`--model` to choose one of that provider's models. A `--provider` other than
`LLM_PROVIDER` uses its own default model rather than `LLM_MODEL`. The JSON
report records these settings, individual answers and aggregate rates. The
command needs a populated benchmark and built indexes. Programmatic runs use
`src.evaluation.evaluate(questions, retriever, config, run_id="baseline")`,
which returns the report. A run that stops part-way raises instead:
`RunStopped`, a `ProviderUnavailable`, when the provider cannot answer, and
`RunInterrupted`, a `KeyboardInterrupt`, on Ctrl-C. Either carries the report
of the questions before it on `.report`, as the command writes it.

`--no-facts` sends every question to retrieval and generation instead of looking
a numeric one up in the XBRL facts store first, which is the without half of
that comparison. Each row records the `route` that answered it and the report
counts them, so a run that mixes the two reads as what it is. Use `--no-facts`
for `benchmark/generated.jsonl` in particular: those questions are generated
from the same store the route answers from, so leaving it on measures the store
against itself.

`--config` selects a named configuration, `C1` to `C4` as `src/stack.py`
defines them and `python -m src.evaluation.run` measures them, and sets the
retriever and the metadata filter from it: a row measured without the filter is
answered from a search of the whole corpus. `--retriever` and the switches
below override what the configuration says, and the report records the
configuration it answered with, including any override, under `stack`. The
app's sidebar offers the same configurations and builds them through the same
function, so the system on screen is the system a number in `results/`
describes; a row a run has measured is labelled there with the run that
measured it. `C0`, the fixed-size baseline, is built by the ablation runner
only and cannot be selected.

`--no-refusal` searches and asks a model about every question, instead of
refusing one the parser reads as asking for advice or a prediction, or for a
figure of a company outside the corpus. The report counts the rows it refused
as `refused`.

`--no-decompose` searches each question once instead of once per filing. A
question naming more than one company or more than one year is otherwise split
into one search per filing and the results interleaved, so the `FINAL_K`
passages that reach the generator cover every filing the question asks about
rather than whichever one phrases the topic most like the question. Each row
records the `sub_questions` its evidence came from and the report counts the
rows that were split, so the comparison says how many questions it could apply
to at all.

Every answer is put through `verify_answer` before it is recorded, against the
company and year the question is about, as the app checks an answer before
showing it. That holds under a configuration measured without the metadata
filter too: it searches every filing, and its answer is still checked against
the benchmark's company and year. The report's `checks` counts the answered
rows by the worst of the figures each states. A figure is a `mismatch` where a
check contradicts it, `supported` where a passage or the facts store supports
it, and `unverified` otherwise, and an answer is counted as its worst figure:
`mismatch` first, then `unverified`, then `supported`. So a supported answer
is one whose every figure is supported, and one supported claim does not hide
another that could not be checked. `unchecked` is an answer to a question that
asks for no figure and states none; a question that asks for one and is
answered without it is `unverified`. A run then says how many of its answers
a user would see flagged, and
how many hold a figure nothing confirmed. The facts store is read once for the
run.

### Embedding and generation ablations

Issue #46 compares three embedding configurations and both answer providers.
The registry in `src/evaluation/model_ablation.py` records the encoder, vector
width, query and passage prefixes, token limit and index directory together.
`EmbeddingConfig` carries those settings through stack assembly and dense
retrieval. The default index still uses BGE.

Build the processed corpus, BM25 index and facts store, then generate the
benchmark (`python -m src.retrieval benchmark`) as described above. With that
benchmark on disk, run a small comparison:

```bash
python -m src.evaluation.model_ablation embedding \
  benchmark/generated.jsonl --prepare-indexes --limit 14 --run-id embeddings-20261004
python -m src.evaluation.model_ablation generation \
  benchmark/generated.jsonl --no-facts --limit 14 --run-id providers-20261004
```

`--limit` takes an evenly spaced sample across the full file, including its
ends when at least two questions are selected. It does not guarantee that a
small sample covers every company, year or question type; use a larger sample
or the complete benchmark for reported conclusions. Without `--limit`, all
questions are measured. Fourteen questions is a smoke run, not a final result.

E1 is BGE base, E2 MiniLM and E3 E5 base. Each encoder is measured twice:
filtered hybrid retrieval (E1–E3, the C4 retrieval path) and filtered dense-only
retrieval (E1-dense–E3-dense), which isolates the encoder from BM25 fusion.
Both views use the same question filters and indexes. `--top-k` sets their
retrieval cutoff (10 by default); it applies only to the embedding command.
E5 encodes `query: ` and `passage: ` on the corresponding sides. MiniLM reads
256 tokens, compared with BGE and E5's 512. Each row records the actual limit,
indexed and truncated passage counts, truncation rate and median search latency.
When indexes are prepared in that command, it also records elapsed preparation
time (`build_ms`), including loading and validation; a current index can have
no new passages to encode. A run without preparation leaves that field null.

`--prepare-indexes` is incremental and resumes interrupted builds. E1 reuses
`data/index/chroma`; E2 and E3 use sibling directories `data/index/chroma-E2`
and `data/index/chroma-E3`, with their manifests beside them. Incompatible models
are refused, and prefixes participate in vector digests and manifest checks.
Add `--rebuild-indexes` only when re-encoding is intended, such as after changing
a model or token limit. Index preparation only accepts the default corpus
`data/processed` (including a path that resolves to it). Combining
`--prepare-indexes` with another `--processed-dir` is refused before any files
are read or changed: preparing a smaller corpus would remove passages from the
app's own E1 index. Without preparation, `--processed-dir` reaches benchmark
validation and both matrices' retrievers, which still require the fixed indexes
to match that corpus; it does not select separate indexes for a second corpus.
If E2/E3 indexes were already built at `data/index/chroma/E2` or `E3`, move
their directories and their adjacent `.manifest.json` and `.truncated.json`
files to the new sibling names before running again; moving existing indexes
does not require re-encoding.

G1 is local Ollama with the pinned default `llama3.2:3b`; G2 is hosted Mistral
with the app's pinned `ministral-8b-2512`. They share retrieval indexes and use
C4's passage budget and answer settings, including decomposition and refusal.
The facts shortcut is disabled by default so numeric questions exercise the
providers; `--no-facts` states that explicitly. `--with-facts` includes the
shortcut and records that override. The table reports overall abstention,
provider-routed question count and abstention rate, median answer latency and
how many completed questions needed a retry. Reports retain routes and numeric
verification checks. G1 needs Ollama and its model; G2 needs `MISTRAL_API_KEY`.

Each run writes `report.json` for each configuration and a `summary.csv` and
`summary.json` under `results/<run-id>/`. C, E and G runners share the table
writer and JSON format (`run_id`, `top_k`, `configurations` with configuration
objects). Model-specific columns extend the common C-row columns. Reports
retain retrieved chunk IDs and scores for inspection and rescoring. A missing
index or key fails before any question is measured or run directory is created.
A stopped or interrupted provider run saves completed answers, its stop reason
and both summaries before exiting unsuccessfully. Completed runs print the
table and output path. Use a fresh run ID; existing results are never overwritten.

## Team and course

BT4103 Business Analytics Capstone, Team 8, AY26/27 Semester 1, supervised by A/Prof Oh Hyelim. The main milestones are the requirements presentation in Week 6, the interim presentation in Week 9, and the final presentation in Week 13, with deliverables handed over the following week.

## Contributing

Read `GIT_WORKFLOW.md` before pushing. Work on a branch, open a pull request, and get one review before merging into `main`. Do not commit large filing data or secrets.

## Note

This is an academic project that uses only publicly available data. Nothing here is financial advice.
