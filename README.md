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
- History: filing years 2021 to 2025, so 5 filings per firm.
- Documents: Form 10-K for the core system and all evaluation. Form 10-Q is a future extension, layered in once the 10-K pipeline is validated, and is not part of the benchmark or the comparative results.
- Corpus size: 50 filings, small enough to index on a laptop and large enough for meaningful retrieval evaluation.
- Key sections: Item 1 (Business), Item 1A (Risk Factors), Item 7 (MD&A), Item 8 (Financial Statements and notes), and other relevant sections.

The year range filters on filing date rather than fiscal year. Firms that file in January or February (Alphabet, Meta, Amazon, Adobe) therefore contribute fiscal years 2020 to 2024, while the mid-year and late-year filers contribute roughly 2021 to 2025. Year-on-year questions within one company are unaffected, but a cross-company question that pins a specific fiscal year should read `filing_date` from the manifest rather than assume the years line up.

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
│   ├── pipeline/            # EDGAR download, parse, chunk
│   │   ├── constants.py     #   corpus scope and parser thresholds
│   │   ├── records.py       #   dataclasses passed between stages
│   │   ├── cli.py           #   argument parsers for the pipeline commands
│   │   ├── download.py
│   │   └── parse.py
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

Run the app once it is built:

```bash
streamlit run src/app/app.py
```

## What the pipeline produces

Two of the three stages are built. Each one writes to its own folder under
`data/`, and nothing downstream writes back into an earlier stage's folder.

| Stage | Command | Reads | Writes |
|---|---|---|---|
| Download | `python -m src.pipeline.download` | `config/companies.txt` | `data/raw/<TICKER>/*.html`, `data/raw/manifest.jsonl` |
| Parse | `python -m src.pipeline.parse` | the manifest and the raw HTML | `data/interim/<TICKER>/*.json` |
| Chunk | not built yet | the interim JSON | `data/processed/` |

`data/raw/manifest.jsonl` holds one JSON object per filing. It is what makes the
download resumable, and it lets later stages see what is on disk without walking
the folder tree or calling EDGAR again:

```json
{"ticker": "AAPL", "cik": 320193, "company": "Apple Inc.", "form": "10-K",
 "filing_date": "2025-10-31", "accession_no": "0000320193-25-000079",
 "url": "https://www.sec.gov/Archives/edgar/data/320193/...",
 "path": "data/raw/AAPL/10-K_2025-10-31_0000320193-25-000079.html"}
```

`path` is written with the host's own separator, so a manifest built on Windows
carries backslashes. Join it onto `PROJECT_ROOT` rather than splitting on `/`.

Each `data/interim/<TICKER>/<filing>.json` carries the same filing metadata plus
a `sections` list, one entry per Item:

| Field | Meaning |
|---|---|
| `section_id` | the parser's name for the section, for example `part_ii_item_7` |
| `part`, `item` | `"II"` and `"7"` |
| `title` | the official Item title, used in citations |
| `text` | the Item's text |
| `n_chars`, `n_tables` | how big the Item is and how many tables it holds |
| `is_key_section` | whether this is one of the Items the project targets |
| `is_stub` | the Item parsed cleanly but holds almost no text |
| `resolved_from` | for a stub, the `section_id` that actually holds the text |
| `confidence`, `detection_method`, `validated`, `warnings` | what the parser thought of its own boundary detection, kept so a bad answer can be traced back to a bad split |

The dataclasses behind both files live in `src/pipeline/records.py`, so a later
stage can read a manifest line or an interim file by importing the shape alone,
without pulling in the download or parse logic.

## Team and course

BT4103 Business Analytics Capstone, Team 8, AY26/27 Semester 1, supervised by A/Prof Oh Hyelim. The main milestones are the requirements presentation in Week 6, the interim presentation in Week 9, and the final presentation in Week 13, with deliverables handed over the following week.

## Contributing

Read `GIT_WORKFLOW.md` before pushing. Work on a branch, open a pull request, and get one review before merging into `main`. Do not commit large filing data or secrets.

## Note

This is an academic project that uses only publicly available data. Nothing here is financial advice.
