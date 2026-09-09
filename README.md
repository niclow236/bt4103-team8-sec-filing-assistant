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
- Companies: 10 US-listed tech firms, listed in `config/companies.txt`: Apple, Microsoft, Broadcom, Alphabet, Meta, Amazon, Oracle, Salesforce, Adobe, Cisco.
- History: fiscal years 2021 to 2025, so 5 filings per firm.
- Documents: Form 10-K for the core system and all evaluation. Form 10-Q is a future extension, layered in once the 10-K pipeline is validated, and is not part of the benchmark or the comparative results.
- Corpus size: 50 filings in scope, small enough to index on a laptop and large enough for meaningful retrieval evaluation.
- Key sections: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements and notes), and other relevant sections.

The scope is set in fiscal years, not filing years, because that is the axis questions are asked on. The two differ: Alphabet, Meta, Amazon and Adobe close their books in November or December and file the following January or February, so their fiscal 2025 report is a 2026 filing while Apple's is a 2025 filing. Selecting on filing date would give those four a different set of years from the other six, producing a corpus that looks complete at 5 filings per firm but cannot answer a single question across all ten.

The download therefore searches EDGAR over filing years, which is how EDGAR indexes, then narrows the result to the fiscal years in scope by reading each filing's `period_of_report`. `DEFAULT_FISCAL_YEARS` sets the scope and `DEFAULT_FILING_YEARS` sets the search window, which runs one year longer to reach the December filers. Every download prints a fiscal-year coverage table naming any year that is missing a company, so a gap is seen when the corpus is built rather than inferred later from a thin answer.

The corpus is rectangular: fiscal years 2021 to 2025, all 10 companies in each, 50 filings with no gaps and nothing outside the scope. `FilingRecord.fiscal_year` gives the year directly, and `iter_chunks(fiscal_years=...)` filters on it.

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
│   └── processed/           # chunks ready for indexing (git-ignored)
├── src/
│   ├── config.py            # project-wide paths, .env loading, EDGAR identity
│   ├── utils.py             # run logging, shared across packages
│   ├── pipeline/            # EDGAR download, parse, chunk
│   │   ├── constants.py     #   corpus scope, parser and chunker thresholds
│   │   ├── records.py       #   dataclasses passed between stages
│   │   ├── cli.py           #   argument parsers for the pipeline commands
│   │   ├── download.py
│   │   ├── parse.py
│   │   └── chunk.py
│   ├── retrieval/           # BM25, dense, hybrid
│   ├── rag/                 # RAG engine and citations
│   ├── evaluation/          # benchmark and metrics
│   └── app/                 # Streamlit or Gradio UI
├── notebooks/               # exploration and experiments
├── benchmark/               # ground-truth Q&A dataset
└── docs/                    # reports, minutes, references
```

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
python -m src.pipeline.download --limit 1   # quick test: newest 10-K each
python -m src.pipeline.download            # the full corpus
```

Filings land in `data/raw/<TICKER>/`, and every one is recorded in
`data/raw/manifest.jsonl`. The download is resumable, so re-running it skips
whatever is already on disk.

Split the downloaded filings into their numbered Items:

```bash
python -m src.pipeline.parse                 # every filing in the manifest
python -m src.pipeline.parse --tickers AAPL  # just one company
python -m src.pipeline.parse --force         # re-parse filings already done
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
This matters when choosing companies. Intel was dropped from the ticker list
because it files a narratively organised 10-K with a cross-reference index
instead of Item headings, so its MD&A cannot be located by Item boundaries at
all, and an absent Item 7 would otherwise surface much later as an unexplained
retrieval failure.

Cut the parsed Items into the passages retrieval will search:

```bash
python -m src.pipeline.chunk                  # every filing in data/interim/
python -m src.pipeline.chunk --tickers AAPL   # just one company
python -m src.pipeline.chunk --budget 1500    # try a different passage size
python -m src.pipeline.chunk --force          # re-chunk filings already done
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

Run the app once it is built:

```bash
streamlit run src/app/app.py
```

## What the pipeline produces

All three stages are built. Each one writes to its own folder under `data/`,
and nothing downstream writes back into an earlier stage's folder.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Download | `python -m src.pipeline.download` | `config/companies.txt` | `data/raw/<TICKER>/*.html`, `data/raw/manifest.jsonl` |
| Parse | `python -m src.pipeline.parse` | the manifest and the raw HTML | `data/interim/<TICKER>/*.json` |
| Chunk | `python -m src.pipeline.chunk` | `data/interim/<TICKER>/*.json` | `data/processed/<TICKER>/*.json` |

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

The corpus currently chunks to 16,264 passages over 50 filings: 12,206 of prose
and 4,058 of tables, at a median of 1,472 characters. 86% carry a heading.

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
99% of the tables that hold data rebuild cleanly, 3,648 of 3,686; where one
cannot, its flattened copy is left in the prose, so no figure is ever lost, it
is just harder to read.

That rate is measured against `n_data_tables`, not `n_tables`. Filers wrap
bullet points in a one-cell `<table>` to indent them, and Item 1A is written
almost entirely that way, so 1,597 of the 5,283 `<table>` elements in the corpus
are page formatting rather than data. Their text is already in the Item, so
skipping them loses nothing, and counting them as failed rebuilds would put the
figure around 69% and read as though a third of the financial statements were
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
truncated by it. 1,800 characters is about 450 tokens, inside the 512-token
limit most sentence-transformer models impose. Four things keep passages there:

- A paragraph longer than the budget is split at sentence ends, and a sentence
  longer than the budget at its clause boundaries, so a cut never lands inside a
  sentence unless the sentence alone exceeds the budget.
- A table too long for one passage is split by rows, and one too wide by columns,
  with the header repeated on every piece. Column splitting is what bounds a
  table whose single row is already wider than the budget, which no amount of row
  slicing reaches.
- The blank lines between paragraphs count against the budget as well as the
  paragraphs, since they are in the passage too.
- Cells are rendered without alignment padding. Padding would be the largest
  single item in a wide table and carries no meaning to a model reading it.

The result is that **0.12% of passages exceed 2,048 characters, by at most 42
characters**, and no table passage exceeds it at all. Before this work the figure
was 6.6%, and the widest table passage was 9,907 characters, most of which a
512-token model would have discarded in silence.

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
latter matching the `--key-items-only` flag on the chunk command.

## Team and course

BT4103 Business Analytics Capstone, Team 8, AY26/27 Semester 1, supervised by A/Prof Oh Hyelim. The main milestones are the requirements presentation in Week 6, the interim presentation in Week 9, and the final presentation in Week 13, with deliverables handed over the following week.

## Contributing

Read `GIT_WORKFLOW.md` before pushing. Work on a branch, open a pull request, and get one review before merging into `main`. Do not commit large filing data or secrets.

## Note

This is an academic project that uses only publicly available data. Nothing here is financial advice.
