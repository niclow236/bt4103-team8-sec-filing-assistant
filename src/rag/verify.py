"""Deterministic numeric consistency checks after citation resolution.

Only cited evidence can support a sentence. Decimal arithmetic normalizes
scales, percentages and accounting negatives. A rounded claim is compared at
its displayed precision, not with a blanket percentage tolerance. Facts are
matched by company, annual fiscal period, concept and unit, never by value
alone. Unknown concepts, ambiguous scopes and unavailable stores are visible
unverified checks. Derived calculations are not inferred.

Groundedness is deliberately conservative: an extractive sentence can be
confirmed, but a paraphrase needs semantic review. Numeric matches do not
establish entailment. No model, network call or facts download is needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

from ..retrieval.facts import FACTS_FILE, current_year, load_facts
from .constants import FINANCIAL_METRICS, UNIT_ALIASES
from .query import ParsedQuestion, parse_question
from .records import Answer, SentenceCitations, VerificationCheck, VerificationResult

_MARKERS = re.compile(r"\[[+-]?\d+\]")
_DATES = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")
_NUMBER = re.compile(
    r"(?<![\w.])(?P<open>\()?\s*(?P<before_sign>[-−+])?\s*"
    r"(?P<currency>US\$|USD\s*|\$|EUR\s*|€|GBP\s*|£)?"
    r"(?P<sign>[-−+])?\s*(?P<number>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?|\.\d+)"
    r"(?:\s*(?P<scale>trillion|billion|million|thousand|bn|mn)\b)?"
    r"(?:\s*(?P<unit>%|percent\b|basis points\b|bps\b|shares\b|employees\b|users\b|stores\b))?"
    r"(?P<close>\))?", re.I,
)
_SCALES = {"thousand": Decimal("1e3"), "million": Decimal("1e6"),
           "mn": Decimal("1e6"), "billion": Decimal("1e9"),
           "bn": Decimal("1e9"), "trillion": Decimal("1e12")}
_TABLE_SCALE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_FACT_SCOPE_UNSUPPORTED = re.compile(
    r"\b(?:segment|iphone|ipad|aws|azure|google cloud|product revenue|services revenue|"
    r"quarter|quarterly|q[1-4]|combined|sum|average|difference|ratio|margin|"
    r"(?:increased?|decreased?|grew|fell) by)\b", re.I,
)

# The aliases, concepts and units live in constants.py, because #34 routes a
# numeric question on the same table that this module checks the answer
# against: a question answered from a concept its checker did not know would
# be flagged for having been answered at all. Aliases are intentionally
# narrow. Extend using the benchmark; do not let an unrelated concept validate
# a claim just because the value happens to match.
_METRICS = FINANCIAL_METRICS


def _metrics(text: str) -> set[str]:
    return {key for key, (aliases, _, _) in _METRICS.items()
            if any(re.search(r"\b" + re.escape(alias) + r"\b", text, re.I) for alias in aliases)}


def _plain(text: str) -> str:
    return " ".join(_MARKERS.sub("", text).casefold().split()).rstrip(".!?")


@dataclass(frozen=True)
class _Figure:
    text: str
    value: Decimal
    unit: str
    quantum: Decimal | None
    start: int
    end: int


def _figures(text: str, *, scale: Decimal = Decimal(1), unit: str = "", ignore_years: bool = True) -> list[_Figure]:
    # Preserve offsets when masking source numbers and ISO dates.
    masked = _MARKERS.sub(lambda m: " " * len(m[0]), text)
    masked = _DATES.sub(lambda m: " " * len(m[0]), masked)
    figures = []
    for match in _NUMBER.finditer(masked):
        raw = match["number"].replace(",", "")
        prefix = masked[max(0, match.start() - 12):match.start()]
        suffix = masked[match.end():match.end() + 12]
        explicit = any(match[k] for k in ("currency", "scale", "unit", "sign", "before_sign", "open"))
        if not explicit and (
            (ignore_years and match["number"].isdigit() and 1900 <= int(raw) <= 2099)
            or re.search(r"(?:item|part|form|fy)\s*$", prefix, re.I)
            or re.match(r"-(?:K|Q)\b", suffix, re.I)
        ):
            continue
        magnitude = _SCALES.get((match["scale"] or "").lower(), scale)
        currency = (match["currency"] or "").strip().upper()
        kind = {"$": "USD", "US$": "USD", "€": "EUR", "£": "GBP"}.get(currency, currency) or unit
        suffix_unit = (match["unit"] or "").lower()
        if suffix_unit in {"%", "percent", "basis points", "bps"}:
            kind = "ratio"
            magnitude = Decimal("0.0001") if suffix_unit in {"basis points", "bps"} else Decimal("0.01")
        elif suffix_unit:
            kind = suffix_unit
        if re.match(r"\s*(?:per share|/\s*share)", suffix, re.I):
            kind = "USD/shares" if kind in {"", "USD"} else kind + "/shares"
        value = Decimal(raw) * magnitude
        if match["sign"] in {"-", "−"} or match["before_sign"] in {"-", "−"} or (match["open"] and match["close"]):
            value = -value
        # An unscaled integer is exact; decimals/scaled values may be rounded.
        decimals = len(raw.partition(".")[2])
        quantum = magnitude * Decimal(10) ** -decimals if decimals or magnitude != 1 else None
        figures.append(_Figure(match[0].strip(), value, kind, quantum, match.start(), match.end()))
    return figures


def _matches(claim: _Figure, value: Decimal, unit: str) -> bool:
    if claim.unit != unit:
        return False
    if claim.quantum is None:
        return claim.value == value
    rounded = (value / claim.quantum).quantize(Decimal(1), rounding=ROUND_HALF_UP) * claim.quantum
    return claim.value == rounded


def _years(text: str) -> set[int]:
    masked = list(_MARKERS.sub(lambda m: " " * len(m[0]), text))
    for figure in _figures(text):
        masked[figure.start:figure.end] = " " * (figure.end - figure.start)
    return {int(year) for year in _YEAR.findall("".join(masked))}


def _scope(text: str, parsed: ParsedQuestion, passages) -> tuple[set[str], set[int]]:
    local = parse_question(text) if text.strip() else parsed
    if local.unresolved or parsed.unresolved:
        return set(), set()
    tickers = set(local.tickers or parsed.tickers or tuple(p.ticker for p in passages))
    years = _years(text)
    years = years or set(parsed.fiscal_years) or {p.fiscal_year for p in passages if p.fiscal_year}
    if (parsed.tickers and not tickers.issubset(parsed.tickers)) or (
        parsed.fiscal_years and not years.issubset(parsed.fiscal_years)
    ):
        return set(), set()
    return tickers, years


def _passage_values(passage, metric: str | None):
    """Yield values with row context and year headers from Markdown tables."""
    text = passage.text
    table_scale = _TABLE_SCALE.search(text)
    scale = _SCALES[table_scale[1].lower().rstrip("s")] if table_scale else Decimal(1)
    caption = " ".join(line for line in text.splitlines() if "|" not in line)
    currencies = {code for pattern, code in (
        (r"\bUSD\b|\bdollars\b|\$", "USD"), (r"\bEUR\b|\beuros\b|€", "EUR"),
        (r"\bGBP\b|\bpounds\b|£", "GBP"), (r"\bJPY\b|\byen\b", "JPY"),
    ) if re.search(pattern, caption, re.I)}
    header_years: dict[int, int] = {}
    for line in text.splitlines():
        if "|" in line:
            cells = line.strip().strip("|").split("|")
            years = {i: int(c.strip()) for i, c in enumerate(cells) if _YEAR.fullmatch(c.strip())}
            if years and not _metrics(line):
                header_years = years
                continue
            if metric and metric not in _metrics(line):
                continue
            default_unit = _METRICS[metric][2] if metric else ""
            if currencies:
                currency = next(iter(currencies)) if len(currencies) == 1 else "unknown"
                default_unit = currency + ("/shares" if default_unit.endswith("/shares") else "")
            for i, cell in enumerate(cells):
                # A separate currency cell is common in SEC tables.
                if i and cells[i - 1].strip() == "$" and not cell.strip().startswith("$"):
                    cell = "$" + cell.strip()
                cell_scale = Decimal(1) if default_unit.endswith("/shares") else scale
                for figure in _figures(cell, scale=cell_scale, unit=default_unit, ignore_years=False):
                    yield figure, header_years.get(i, passage.fiscal_year)
        else:
            # Do not let revenue in one sentence validate assets in another.
            for clause in re.split(r"(?<=[.!?])\s+|;", line):
                if metric and metric not in _metrics(clause):
                    continue
                local_years = _years(clause)
                year = next(iter(local_years)) if len(local_years) == 1 else passage.fiscal_year
                if len(local_years) > 1:
                    continue  # A multi-year prose comparison needs semantic alignment.
                default_unit = _METRICS[metric][2] if metric else ""
                yield from ((f, year) for f in _figures(clause, unit=default_unit))


def _check(kind, status, index, claim, reason, figure=None, evidence=()):
    return VerificationCheck(
        kind, status, index, claim, reason,
        None if figure is None else figure.text,
        None if figure is None else str(figure.value),
        None if figure is None else figure.unit,
        tuple(evidence),
    )


def _fact_check(frame, error, figure, metric, tickers, years, passages, index, claim, question):
    if error:
        return _check("fact", "unverified", index, claim, error, figure)
    if _FACT_SCOPE_UNSUPPORTED.search(claim + " " + question):
        return _check("fact", "unverified", index, claim,
                      "Segment, quarterly and derived figures need a more specific facts lookup.", figure)
    if metric is None or len(tickers) != 1 or len(years) != 1:
        return _check("fact", "unverified", index, claim,
                      "A unique company, fiscal year and supported financial concept are required.", figure)
    ticker, year = next(iter(tickers)), next(iter(years))
    concepts = _METRICS[metric][1]
    rows = current_year(frame)
    rows = rows[(rows["ticker"].str.upper() == ticker) & (rows["fiscal_year"] == year)
                & rows["concept"].str.split(":").str[-1].isin(concepts)]
    accessions = {m[0] for p in passages
                  if (m := re.match(r"\d{10}-\d{2}-\d{6}", p.chunk_id))}
    if accessions:
        rows = rows[rows["accession"].isin(accessions)]
    candidates = []
    for row in rows.to_dict("records"):
        unit = UNIT_ALIASES.get(str(row["unit"]).lower(), str(row["unit"]))
        if unit != figure.unit:
            continue
        try:
            value = Decimal(str(row["value"]))
        except InvalidOperation:
            continue
        if not value.is_finite():
            continue
        # value is already numeric_value in base units; scale is display metadata.
        reference = " | ".join(str(row.get(k, "")) for k in
                               ("accession", "concept", "unit", "period_start", "period_end", "value"))
        candidates.append((value, reference))
    if not candidates:
        return _check("fact", "unverified", index, claim,
                      "No annual fact with the required company, year, concept and unit.", figure)
    matches = [_matches(figure, value, figure.unit) for value, _ in candidates]
    status = "supported" if all(matches) else "unverified" if any(matches) else "mismatch"
    reason = {"supported": "The annual fact agrees at the displayed precision.",
              "unverified": "Conflicting facts exist for this scope; review the filing.",
              "mismatch": "The annual fact differs from the answer."}[status]
    return _check("fact", status, index, claim, reason, figure, [ref for _, ref in candidates])


def verify_answer(
    answer: Answer,
    *,
    parsed: ParsedQuestion | None = None,
    facts_file: Path = FACTS_FILE,
) -> Answer:
    """Return a new Answer with checks; never change the model's answer text.

    Numeric/comparative questions use the existing local parquet facts store.
    Missing or unreadable stores are recorded, not downloaded or ignored.
    Call after resolve_citations and before displaying or recording an answer.
    Passing the original ParsedQuestion preserves any caller-selected scope.
    """
    parsed = parsed or parse_question(answer.question)
    # parse_question strips, so an untrimmed question would fail its own default.
    if parsed.question != answer.question.strip():
        raise ValueError("parsed question must match Answer.question")
    checks = []
    if answer.parse_error or answer.truncated:
        checks.append(_check("output", "unverified", None, answer.text,
                             "Generation was malformed or truncated; review the available output."))
    if answer.abstained:
        checks.append(_check("output", "not_applicable", None, answer.text,
                             "The assistant abstained; there are no claims to verify."))
        return replace(answer, verification=VerificationResult(parsed.question_type, tuple(checks)))

    numeric = parsed.wants_figures or parsed.question_type == "numeric"
    frame, facts_error = None, None
    if numeric:
        try:
            frame = load_facts(Path(facts_file))
            required = {"ticker", "fiscal_year", "concept", "unit", "value", "accession"}
            if not required.issubset(frame.columns):
                raise ValueError("Facts store is missing required verification columns.")
        except (OSError, ValueError, KeyError, ImportError) as exc:
            facts_error = f"Facts store unavailable: {exc}"

    sentences = answer.sentences or (
        (SentenceCitations(answer.text, answer.text, answer.citations, incomplete=True),)
        if answer.text.strip() else ()
    )
    if not sentences:
        checks.append(_check("output", "unverified", None, answer.text, "The answer contains no claims."))
    number_count = 0
    for index, sentence in enumerate(sentences):
        shown = {p.chunk_id: p for p in answer.passages}
        cited = [shown[c.chunk_id] for c in sentence.citations
                 if c.resolved and c.chunk_id in shown and c.marker <= len(answer.passages)
                 and answer.passages[c.marker - 1].chunk_id == c.chunk_id]
        claim = _MARKERS.sub("", sentence.text).strip()
        exact = [p.chunk_id for p in cited if _plain(claim) and _plain(claim) in _plain(p.text)]
        checks.append(_check("groundedness", "supported" if exact and not sentence.flagged else "unverified",
                             index, claim, "Extractive wording found in a cited passage." if exact and not sentence.flagged
                             else "Citation or claim needs review; numeric agreement alone does not establish entailment.",
                             evidence=exact))
        tickers, years = _scope(claim, parsed, cited)
        metrics = _metrics(claim) or _metrics(answer.question)
        metric = next(iter(metrics)) if len(metrics) == 1 else None
        default_unit = _METRICS[metric][2] if metric else ""
        figures = _figures(claim, unit=default_unit)
        number_count += len(figures)
        for figure in figures:
            matches = []
            # Ambiguous scope cannot become a pass through coincidental values.
            ambiguous = len(tickers) != 1 or len(years) != 1 or len(metrics) > 1
            if not ambiguous:
                for passage in cited:
                    if passage.ticker not in tickers:
                        continue
                    for candidate, year in _passage_values(passage, metric):
                        if year in years and _matches(figure, candidate.value, candidate.unit):
                            matches.append(passage.chunk_id)
            # Most table chunks carry no "(in millions)" caption, so their
            # figures are in an unknown scale: a miss there is unknown, not wrong.
            unscaled = bool(cited) and not matches and not ambiguous and any(
                p.content_type == "table" and not _TABLE_SCALE.search(p.text) for p in cited)
            status = ("unverified" if ambiguous or not cited or unscaled
                      else "supported" if matches else "mismatch")
            reason = ("A cited table declares no scale, so its figures cannot be compared."
                      if unscaled else
                      {"supported": "The figure occurs in cited evidence at the displayed precision.",
                       "unverified": "Cited evidence or a unique company, year and metric scope is missing.",
                       "mismatch": "No matching figure and unit in the cited evidence for this scope."}[status])
            checks.append(_check("passage", status, index, claim, reason, figure, dict.fromkeys(matches)))
            if numeric:
                checks.append(_fact_check(frame, facts_error, figure, metric, tickers, years, cited, index, claim, answer.question))
    if numeric and not number_count:
        checks.append(_check("fact", "unverified", None, answer.text,
                             facts_error or "No supported numeric notation was found to check against facts."))
    return replace(answer, verification=VerificationResult(parsed.question_type, tuple(checks)))


def record_verification(answer: Answer, path: Path, *, question_id: str, run_id: str) -> None:
    """Append one complete, JSON-safe question/run record for evaluation.

    Call from the single writer of an evaluation run. Repeated question IDs are
    allowed across runs; run_id distinguishes model/configuration comparisons.
    I/O failures propagate so an experiment cannot silently lose its results.
    """
    if answer.verification is None:
        raise ValueError("Verify the answer before recording it.")
    if not question_id.strip() or not run_id.strip():
        raise ValueError("question_id and run_id must be non-empty")
    record = {"question_id": question_id, "run_id": run_id, "answer": answer.to_dict()}
    line = json.dumps(record, ensure_ascii=False, allow_nan=False)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


__all__ = ["record_verification", "verify_answer"]
