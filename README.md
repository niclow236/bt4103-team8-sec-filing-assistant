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

Python is the primary language. The project uses LLM APIs and open-source LLMs, text embeddings and vector databases, RAG frameworks, information-retrieval methods, and an interactive UI framework. Exact libraries are pinned in `requirements.txt` as the project develops.

## Repository structure

This is the planned layout. Not every folder exists on day one; they are added as the work reaches each part.

```
bt4103-team8-sec-filing-assistant/
├── README.md
├── GIT_WORKFLOW.md          # branching workflow and Git setup
├── requirements.txt
├── .gitignore
├── .env.example             # template for API keys (copy to .env)
├── config/
│   └── companies.txt        # tickers the pipeline downloads
├── data/
│   ├── sample/              # small committed sample
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
│   │   ├── base.py          #   the Retriever contract every method satisfies
│   │   ├── constants.py     #   models, k values and fusion constants, in one place
│   │   └── records.py       #   what a retriever returns and what an index says of itself
│   ├── rag/                 # RAG engine and citations
│   ├── evaluation/          # benchmark and metrics
│   └── app/                 # Streamlit or Gradio UI
├── logs/                    # terminal output of each run (git-ignored)
├── notebooks/               # exploration and experiments
├── benchmark/               # ground-truth Q&A dataset
└── docs/                    # reports, minutes, references
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

Set up environment variables:

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

Open `.env` and set `EDGAR_IDENTITY` to your own name and email, for example
`EDGAR_IDENTITY=Jane Tan jane@example.com`. The SEC requires every automated
request to carry a contact string and blocks traffic without one, so the
download stops immediately with a `MissingIdentityError` if this is blank. The
LLM API keys further down the file are not needed until the RAG work starts.

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
| verify | 2.0 min | 15 EDGAR index requests plus 15 XBRL fetches, then eight checks over 28,000 passages |
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

Eight checks, cheapest first:

| Check | What would fail it |
|---|---|
| coverage | a company missing from one fiscal year, a filing counted twice, or one outside the scope |
| stage parity | a filing that reached parse but not chunk, or a stray file no manifest line accounts for |
| key Items | Items 1, 1A, 7, 7A or 8 absent, or a stub with nothing to resolve to |
| chunk integrity | a duplicate passage id, a passage that cannot build a citation, a table row tracing to no source row |
| no prose lost | a paragraph of 200 characters or more in a chunked Item that reaches no passage, unless the chunker dropped it as a flattened copy of a table it rebuilt |
| passage sizes | any table passage, or more than 0.5% of prose passages, past the 512 tokens bge reads, counted with the model's own tokenizer over the passage and its context header |
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
mistaken for a clean bill of health on eight.

What verify does **not** fail on is imperfection the pipeline already handles: 6
of 5,428 tables cannot be rebuilt into grids -- Cisco's signature blocks and one
audit-matter paragraph laid out as a table -- and their text stays in the prose,
so nothing is lost. Those are reported as counts by the parse stage. A gate that
fired on them would cry wolf on every run.

Run the app once it is built:

```bash
streamlit run src/app/app.py
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
- A passage identical to another in the same Item is kept once. Filers do print a
  table twice, and in one Item the two would be the same vector indexed twice.
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

## Team and course

BT4103 Business Analytics Capstone, Team 8, AY26/27 Semester 1, supervised by A/Prof Oh Hyelim. The main milestones are the requirements presentation in Week 6, the interim presentation in Week 9, and the final presentation in Week 13, with deliverables handed over the following week.

## Contributing

Read `GIT_WORKFLOW.md` before pushing. Work on a branch, open a pull request, and get one review before merging into `main`. Do not commit large filing data or secrets.

## Note

This is an academic project that uses only publicly available data. Nothing here is financial advice.
