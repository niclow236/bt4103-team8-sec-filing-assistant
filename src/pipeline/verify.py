"""Check that the corpus on disk is one you can build an answer on.

Run it from the project root, after the three stages:

    python -m src.pipeline verify

The stages already report what they did. This asks a different question: is the
result trustworthy enough to index. A corpus can look finished and still be
wrong, and the ways it goes wrong are quiet ones. A company missing from one
fiscal year makes a comparison silently narrower than it reads. A duplicate
chunk id makes a citation ambiguous. A table row that traces to no source table
means a figure was invented somewhere between the filing and the passage.

So every check here is written to fail rather than to warn, and the command
exits non-zero when any of them does. That is the point: it is a gate, meant to
be the last thing run before the corpus is handed to the retrieval stage.

Checks that need EDGAR are part of the gate rather than an option, because the
strongest thing that can be said about a corpus is that it still matches the
source. That means this command needs a network connection and EDGAR_IDENTITY,
even when nothing is being downloaded.

What it does not fail on is imperfection the pipeline already documents and
handles: 6 tables in this corpus cannot be rebuilt into grids -- signature
blocks and a paragraph laid out as a table -- and their text stays in the prose,
so nothing is lost. Those are reported as counts, and a check that fired on
them would cry wolf on every run.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from ..config import (
    INTERIM_DIR,
    PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    configure_edgar,
    read_tickers,
)
from .constants import CHUNK_CHAR_MINIMUM, DEFAULT_FISCAL_YEARS, KEY_ITEMS
from .chunk import iter_chunks, load_parsed, processed_path_for, prose_blocks
from .download import load_manifest
from .parse import interim_path_for
from .records import FilingRecord

logger = logging.getLogger(__name__)

# The share of passages, per content type, allowed past the embedding model's
# window before the corpus is not fit to embed. Measured in tokens, with the
# model's own tokenizer, over exactly what the encoder reads: the context header
# and the passage. A character limit cannot do this job, because the characters
# per token differ by a factor of two between prose and figures; the gate this
# replaces passed a corpus in which 5.38% of passages were being truncated.
#
# Tables allow none. They are cut to a budget derived to fit, so a single one
# over is a regression in the chunker, and reverting that budget fails this.
# Prose allows a little, since a paragraph is never cut mid-sentence and an
# unusually dense one can run over; a real regression -- raising the budget to
# 4,000 characters, say -- puts far more than this past the window.
OVERRUN_TOLERANCE = {"table": 0.0, "prose": 0.005}
# Figures pulled from EDGAR's own XBRL data, per company, as an answer key. Each
# is a concept a question would actually ask about.
XBRL_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "NetIncomeLoss",
    "Assets",
    "Liabilities",
    "StockholdersEquity",
    "ResearchAndDevelopmentExpense",
    "CashAndCashEquivalentsAtCarryingValue",
]

_FIGURE_SQUASH = re.compile(r"\s+")


@dataclass
class Check:
    """One question asked of the corpus, and what the answer turned out to be."""

    name: str
    passed: bool
    detail: str
    failures: list[str] = field(default_factory=list)
    # A check not run, because an earlier one already found the fault it would
    # report. Neither a pass nor a failure, and said so rather than left out: a
    # gate that quietly runs four checks instead of eight reads as a clean bill
    # of health it never gave.
    skipped: bool = False


def _squash(text: str) -> str:
    return _FIGURE_SQUASH.sub("", text)


# --- structural checks ------------------------------------------------------


def check_coverage(records: list[FilingRecord], tickers: list[str]) -> Check:
    """Every company in config, in every fiscal year in scope, exactly once.

    This is the check that matters most for a comparative question. A corpus
    holding four of five years for one company still reads as complete at the
    file count, and only goes wrong later, as an answer that quietly covers
    fewer companies than the question asked about.
    """
    low, high = DEFAULT_FISCAL_YEARS
    years = range(low, high + 1)
    held: dict[tuple[str, int], int] = {}
    for record in records:
        if record.fiscal_year is not None:
            held[(record.ticker, record.fiscal_year)] = held.get((record.ticker, record.fiscal_year), 0) + 1

    missing = [f"{ticker} FY{year}" for ticker in tickers for year in years
               if (ticker, year) not in held]
    duplicated = [f"{ticker} FY{year} x{count}" for (ticker, year), count in sorted(held.items())
                  if count > 1]
    outside = sorted({f"{record.ticker} FY{record.fiscal_year}" for record in records
                      if record.fiscal_year is None or not low <= record.fiscal_year <= high})

    problems = (
        [f"missing: {entry}" for entry in missing]
        + [f"duplicated: {entry}" for entry in duplicated]
        + [f"outside FY{low}-{high}: {entry}" for entry in outside]
    )
    expected = len(tickers) * len(list(years))
    return Check(
        name="coverage",
        passed=not problems,
        detail=f"{len(records)} filings, {len(tickers)} companies x {len(list(years))} "
               f"fiscal years, expected {expected}",
        failures=problems,
    )


def check_stage_parity(records: list[FilingRecord]) -> Check:
    """Every filing reached all three stages, and no stage holds strays.

    Chaining the stages by hand is how a corpus ends up parsed but only
    part-chunked, which nothing downstream would notice.
    """
    problems: list[str] = []
    expected_interim: set[Path] = set()
    expected_processed: set[Path] = set()

    for record in records:
        raw = PROJECT_ROOT / record.path
        interim = interim_path_for(record)
        processed = processed_path_for(interim)
        expected_interim.add(interim)
        expected_processed.add(processed)
        for label, path in (("raw", raw), ("interim", interim), ("processed", processed)):
            if not path.exists():
                problems.append(f"{record.ticker} {record.filing_date}: no {label} file")

    for label, directory, pattern, expected in (
        ("interim", INTERIM_DIR, "*/*.json", expected_interim),
        ("processed", PROCESSED_DIR, "*/*.json", expected_processed),
    ):
        for stray in sorted(set(directory.glob(pattern)) - expected):
            problems.append(f"{label}: {stray.relative_to(PROJECT_ROOT).as_posix()} is not in the manifest")

    raw_files = {path for path in RAW_DIR.glob("*/*") if path.is_file()}
    for stray in sorted(raw_files - {PROJECT_ROOT / record.path for record in records}):
        problems.append(f"raw: {stray.relative_to(PROJECT_ROOT).as_posix()} is not in the manifest")

    return Check(
        name="stage parity",
        passed=not problems,
        detail=f"{len(records)} filings present at raw, interim and processed",
        failures=problems,
    )


def check_key_items(records: list[FilingRecord]) -> Check:
    """Every filing carries the Items the project targets.

    A stub counts only when it resolves to the Item that holds its text, which
    is how Oracle's Item 8 is answered from its Item 15.
    """
    problems: list[str] = []
    for record in records:
        parsed = load_parsed(interim_path_for(record))
        wanted = KEY_ITEMS.get(record.form, set())
        have = {section.item for section in parsed.sections
                if section.item and (not section.is_stub or section.resolved_from)}
        for item in sorted(wanted - have):
            problems.append(f"{record.ticker} {record.filing_date}: Item {item} absent or an unresolved stub")
    return Check(
        name="key Items",
        passed=not problems,
        detail=f"Items {', '.join(sorted(KEY_ITEMS['10-K']))} present in all {len(records)} filings",
        failures=problems,
    )


def check_chunk_integrity(records: list[FilingRecord]) -> Check:
    """Passages are uniquely identified, attributable, and traceable to a source.

    Three separate ways a passage can be unusable, checked together because they
    all need the interim file open alongside the processed one: an id that is not
    unique makes a citation ambiguous, a missing field makes it uncitable, and a
    body that traces to no source means text appeared from somewhere.
    """
    problems: list[str] = []
    seen: dict[str, str] = {}
    passages = rows_traced = 0

    for record in records:
        interim = interim_path_for(record)
        parsed = load_parsed(interim)
        sections = {section.section_id: section for section in parsed.sections}
        chunks = json.loads(processed_path_for(interim).read_text(encoding="utf-8"))["chunks"]

        for chunk in chunks:
            passages += 1
            identifier = chunk["chunk_id"]
            if identifier in seen:
                problems.append(f"duplicate chunk_id {identifier} (also in {seen[identifier]})")
            seen[identifier] = f"{record.ticker} {record.filing_date}"

            for required in ("chunk_id", "section_id", "item", "title", "text"):
                if not chunk.get(required):
                    problems.append(f"{identifier}: empty {required}, so it cannot be cited")
            if chunk["n_chars"] != len(chunk["text"]):
                problems.append(f"{identifier}: n_chars disagrees with the text it holds")

            section = sections.get(chunk["section_id"])
            if section is None:
                problems.append(f"{identifier}: section {chunk['section_id']} is not in the interim file")
                continue

            if chunk.get("content_type") != "table":
                continue
            index = chunk.get("table_index")
            if index is None or index >= len(section.tables):
                problems.append(f"{identifier}: table_index {index} is out of range for its Item")
                continue
            table = section.tables[index]
            source = [[_squash(cell) for cell in row] for row in table.rows]
            lines = [line for line in chunk["text"].split("\n") if line.strip().startswith("|")]
            for line in lines[2:]:
                cells = [_squash(cell) for cell in line.strip().strip("|").split("|")]
                if _row_in(cells, source):
                    rows_traced += 1
                else:
                    problems.append(f"{identifier}: a table row traces to no row of its source table")

    return Check(
        name="chunk integrity",
        passed=not problems,
        detail=f"{passages:,} passages, {len(seen):,} unique ids, {rows_traced:,} table rows traced to source",
        failures=problems,
    )


def _row_in(cells: list[str], source: list[list[str]]) -> bool:
    """Whether these cells appear, in order, in some row of the source table.

    A passage may hold a subset of the columns, because a table too wide for the
    budget is split by column, so the test is order-preserving containment
    rather than equality.
    """
    for row in source:
        position = 0
        for cell in cells:
            while position < len(row) and row[position] != cell:
                position += 1
            if position == len(row):
                break
            position += 1
        else:
            return True
    return False


def check_no_prose_lost(records: list[FilingRecord]) -> Check:
    """Every substantial paragraph of a chunked Item survived into a passage.

    The chunker drops page furniture and rejoins sentences split across a page
    boundary, so a passage is not a verbatim slice of the Item. What must hold is
    that no paragraph disappeared, which is checked with whitespace removed so
    those repairs do not read as losses.
    """
    problems: list[str] = []
    checked = 0
    as_debris = 0
    for record in records:
        interim = interim_path_for(record)
        parsed = load_parsed(interim)
        chunks = json.loads(processed_path_for(interim).read_text(encoding="utf-8"))["chunks"]
        haystack = _squash("".join(chunk["text"] for chunk in chunks))
        chunked = {chunk["section_id"] for chunk in chunks}

        for section in parsed.sections:
            if section.section_id not in chunked:
                continue
            # What the chunker dropped on purpose: flattened copies of tables it
            # rebuilt, judged by the chunker's own rule so the two cannot
            # disagree about what counts as lost.
            _, dropped = prose_blocks(section)
            discarded = _squash("".join(dropped))
            for paragraph in section.text.split("\n\n"):
                squashed = _squash(paragraph)
                if len(squashed) < 200:
                    continue
                checked += 1
                if squashed in haystack:
                    continue
                if squashed in discarded:
                    as_debris += 1
                    continue
                problems.append(
                    f"{record.ticker} {record.filing_date} {section.section_id}: "
                    f"lost {paragraph.strip()[:60]!r}"
                )
    return Check(
        name="no prose lost",
        passed=not problems,
        detail=(
            f"{checked:,} paragraphs of 200 characters or more all accounted for, "
            f"{as_debris:,} of them as flattened copies of a rebuilt table"
        ),
        failures=problems,
    )


def bge_token_counter() -> Callable[[list[str]], list[int]]:
    """Count tokens the way the embedding model will, without loading the model.

    ``tokenizers`` reads the model's own tokenizer file and pulls in no torch, so
    counting the whole corpus takes seconds. Checked against the counts the
    encoder recorded on every vector of the index: identical for all 28,544.
    Truncation and padding are switched off explicitly, since a tokenizer file
    can carry either, and a counter that stops at 512 can never report a
    passage over it.
    """
    from tokenizers import Tokenizer

    from ..retrieval.constants import EMBED_MODEL

    tokenizer = Tokenizer.from_pretrained(EMBED_MODEL)
    tokenizer.no_truncation()
    tokenizer.no_padding()
    return lambda texts: [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]


def check_passage_sizes(
    corpus: list[dict],
    count_tokens: Callable[[list[str]], list[int]] | None = None,
) -> Check:
    """Passages are small enough for the embedding model to read whole.

    A passage past the model's context window is not an error the pipeline
    reports: the model truncates it and the tail is indexed as though it were
    never written. So this counts tokens rather than characters, over what the
    encoder is actually handed -- ``embed_text``, which puts the context header
    in front of the passage. Counting the passage alone misses about a third of
    the overruns, since the header alone can be thirty tokens.

    Reported per content type, because prose and tables are cut to different
    budgets and fail for different reasons, and one share across both would
    let a regression in either hide inside the other. ``count_tokens`` replaces
    the tokenizer, for a test that should not need the model's files.
    """
    if not corpus:
        return Check("passage sizes", False, "no passages found", ["data/processed/ is empty"])

    from ..retrieval.constants import EMBED_MAX_TOKENS
    from ..retrieval.embed import embed_text

    counter = count_tokens or bge_token_counter()
    counts = counter([embed_text(passage) for passage in corpus])

    problems: list[str] = []
    parts: list[str] = []
    for content_type in sorted({passage.get("content_type", "prose") for passage in corpus}):
        mine = [n for passage, n in zip(corpus, counts)
                if passage.get("content_type", "prose") == content_type]
        over = sum(1 for n in mine if n > EMBED_MAX_TOKENS)
        share = over / len(mine)
        tolerance = OVERRUN_TOLERANCE.get(content_type, 0.0)
        parts.append(f"{content_type} {over} of {len(mine):,} over ({share:.2%}), "
                     f"largest {max(mine):,}")
        if share > tolerance:
            problems.append(
                f"{over:,} {content_type} passages ({share:.2%}) exceed the model's "
                f"{EMBED_MAX_TOKENS} tokens, above the {tolerance:.1%} tolerance: "
                "the encoder truncates them without warning"
            )

    runts = sum(1 for passage in corpus if passage["n_chars"] < CHUNK_CHAR_MINIMUM // 2)
    detail = f"{EMBED_MAX_TOKENS} tokens: " + "; ".join(parts) + f"; {runts} under {CHUNK_CHAR_MINIMUM // 2} chars"
    return Check("passage sizes", not problems, detail, problems)


# --- checks against EDGAR ---------------------------------------------------


def check_against_edgar(records: list[FilingRecord], tickers: list[str]) -> Check:
    """Every filing still matches what EDGAR serves, and none in scope is absent.

    The strongest thing that can be said about a corpus is that it still agrees
    with the source. This also catches the opposite mistake: a filing EDGAR has
    inside the scope that we never downloaded, which no local check can see.
    """
    from edgar import Company

    low, high = DEFAULT_FISCAL_YEARS
    problems: list[str] = []
    ours = {record.accession_no: record for record in records}
    checked = 0

    for ticker in tickers:
        try:
            filings = Company(ticker).get_filings(
                form=["10-K"], year=list(range(low, high + 2)),
            )
        except Exception as error:
            problems.append(f"{ticker}: EDGAR lookup failed, {type(error).__name__}: {error}")
            continue
        if filings is None or filings.empty:
            problems.append(f"{ticker}: EDGAR returned no 10-K filings")
            continue

        live = {filing.accession_no: filing for filing in filings}
        for accession, record in ((a, r) for a, r in ours.items() if r.ticker == ticker):
            filing = live.get(accession)
            if filing is None:
                problems.append(f"{ticker} {accession}: in our manifest but not on EDGAR")
                continue
            checked += 1
            for field_name, mine, theirs in (
                ("cik", record.cik, filing.cik),
                ("form", record.form, filing.form),
                ("filing_date", record.filing_date, str(filing.filing_date)),
                ("period_of_report", record.period_of_report,
                 str(getattr(filing, "period_of_report", "") or "")),
            ):
                if str(mine) != str(theirs):
                    problems.append(f"{ticker} {accession}: {field_name} is {mine!r}, EDGAR says {theirs!r}")

        for accession, filing in live.items():
            if accession in ours:
                continue
            period = str(getattr(filing, "period_of_report", "") or "")
            year = int(period[:4]) if period[:4].isdigit() else None
            if year is not None and low <= year <= high:
                problems.append(
                    f"{ticker} {accession}: EDGAR has a FY{year} 10-K that is in scope "
                    "and not downloaded"
                )

    return Check(
        name="matches EDGAR",
        passed=not problems,
        detail=f"{checked} filings agree with EDGAR on cik, form, filing date and period of report",
        failures=problems,
    )


def check_xbrl_figures(
    records: list[FilingRecord], identity: str, corpus: list[dict],
) -> Check:
    """Figures EDGAR reports as structured data are findable in the passages.

    This is the check that speaks to what the corpus is for. EDGAR publishes the
    figures each filing reported, so they can be treated as an answer key: if a
    number in that key cannot be found in any indexed passage, then no retrieval
    system built on this corpus could ever cite it, however good the retriever.
    """
    problems: list[str] = []
    checked = found = in_table = 0

    # Grouped once. Asking iter_chunks for one company at a time re-reads every
    # processed file on each call, so fifteen companies meant fifteen passes over
    # the whole corpus to look at a fifteenth of it each time.
    by_filing: dict[str, list[dict]] = {}
    for passage in corpus:
        by_filing.setdefault(passage["accession_no"], []).append(passage)

    newest: dict[str, FilingRecord] = {}
    for record in records:
        current = newest.get(record.ticker)
        if current is None or record.filing_date > current.filing_date:
            newest[record.ticker] = record

    for ticker, record in sorted(newest.items()):
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{record.cik:010d}.json"
        try:
            response = httpx.get(url, headers={"User-Agent": identity}, timeout=60)
            response.raise_for_status()
            facts = response.json()
        except Exception as error:
            problems.append(f"{ticker}: companyfacts fetch failed, {type(error).__name__}: {error}")
            continue
        time.sleep(0.2)   # stay well inside the SEC's published request limit

        passages = by_filing.get(record.accession_no, [])
        everything = "\n".join(passage["text"] for passage in passages)
        tables = "\n".join(passage["text"] for passage in passages
                           if passage["content_type"] == "table")

        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        for concept in XBRL_CONCEPTS:
            node = us_gaap.get(concept)
            if not node:
                continue
            reported = [entry for unit in node.get("units", {}).values() for entry in unit
                        if entry.get("accn") == record.accession_no and entry.get("form") == "10-K"]
            if not reported:
                continue
            reported.sort(key=lambda entry: (entry.get("end", ""), abs(entry.get("val", 0))), reverse=True)
            value = reported[0]["val"]
            if abs(value) < 1_000_000:
                continue

            checked += 1
            millions = f"{round(abs(value) / 1_000_000):,}"
            billions = f"{abs(value) / 1_000_000_000:,.1f}"
            if millions in everything or billions in everything:
                found += 1
                if millions in tables or billions in tables:
                    in_table += 1
            else:
                problems.append(
                    f"{ticker} FY{record.fiscal_year}: {concept} = {millions} million "
                    "is reported to EDGAR but appears in no passage"
                )

    return Check(
        name="XBRL figures findable",
        passed=not problems,
        detail=f"{found} of {checked} reported figures found in passages, {in_table} of them in a table passage",
        failures=problems,
    )


# --- running the gate -------------------------------------------------------


def run_checks() -> list[Check]:
    """Every check, cheapest first, so an obvious fault is reported quickly."""
    tickers = read_tickers()
    records = load_manifest()
    if not records:
        return [Check("corpus present", False, "the manifest is empty",
                      ["run python -m src.pipeline download first"])]

    checks = [
        check_coverage(records, tickers),
        check_stage_parity(records),
    ]
    # Read once and shared. Every check that wants passages wants all of them,
    # and iter_chunks opens all 75 processed files per call.
    corpus: list[dict] = []
    # The per-file checks read every interim and processed file, which is
    # pointless while the corpus is incomplete: each would report the same
    # missing filing once per filing. They are still listed, as skipped, so the
    # summary cannot be mistaken for a corpus that passed them.
    if all(check.passed for check in checks):
        corpus = list(iter_chunks())
        checks += [
            check_key_items(records),
            check_chunk_integrity(records),
            check_no_prose_lost(records),
            check_passage_sizes(corpus),
        ]
    else:
        blocked = ", ".join(check.name for check in checks if not check.passed)
        checks += [
            Check(name, False, f"not run: {blocked} failed first", skipped=True)
            for name in ("key Items", "chunk integrity", "no prose lost", "passage sizes")
        ]

    identity = configure_edgar()
    logger.info("Checking against EDGAR as: %s", identity)
    checks.append(check_against_edgar(records, tickers))
    if corpus:
        checks.append(check_xbrl_figures(records, identity, corpus))
    else:
        checks.append(Check(
            "XBRL figures findable",
            False,
            "not run: the corpus is incomplete, so there is nothing to look in",
            skipped=True,
        ))
    return checks


def report(checks: list[Check]) -> bool:
    """Print each check and whether the corpus passed. Returns True if it did."""
    width = max(len(check.name) for check in checks)
    print()
    for check in checks:
        mark = "SKIP" if check.skipped else ("PASS" if check.passed else "FAIL")
        print(f"  [{mark}]  {check.name:<{width}}  {check.detail}")
        for failure in check.failures[:10]:
            print(f"           {failure}")
        if len(check.failures) > 10:
            print(f"           ... and {len(check.failures) - 10} more")

    failed = [check for check in checks if not check.passed and not check.skipped]
    skipped = [check for check in checks if check.skipped]
    print()
    if failed or skipped:
        if failed:
            print(f"{len(failed)} of {len(checks)} checks failed: "
                  f"{', '.join(check.name for check in failed)}")
        if skipped:
            print(f"{len(skipped)} were not run: {', '.join(check.name for check in skipped)}")
        print("The corpus is not ready to index.")
    else:
        print(f"All {len(checks)} checks passed. The corpus is ready to index.")
    return not failed and not skipped
