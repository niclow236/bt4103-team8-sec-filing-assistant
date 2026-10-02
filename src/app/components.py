"""Reusable Streamlit answer, evidence and filter components (#37).

The sidebar returns the Query to pass to ``answer_question(query=...)``.
Render completed Answer records; citation numbers always refer to prompt order.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from hashlib import sha256
from html import escape
import re
from urllib.parse import urlsplit

import streamlit as st

from src.config import read_tickers
from src.pipeline.constants import DEFAULT_FISCAL_YEARS
from src.rag.citations import render_citation
from src.rag.query import ParsedQuestion, parse_question
from src.rag.records import ABSTENTION_MESSAGES, Answer, Citation
from src.retrieval.constants import FINAL_K
from src.retrieval.records import Query, RetrievedPassage


_MARKER = re.compile(r"\[(-?\d+)\]")
_ITEMS = tuple(str(i) + suffix for i in range(1, 17)
               for suffix in ({1: ("", "A", "B", "C"), 4: ("", "A"),
                               7: ("", "A"), 9: ("", "A", "B", "C")}.get(i, ("",))))
_ITEM_MENTION = re.compile(
    r"\bitems?\s+((?:\d{1,2}[a-z]?)(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d{1,2}[a-z]?\b)*)",
    re.IGNORECASE,
)


def _namespace(key: str) -> str:
    return "sec-" + sha256(key.encode()).hexdigest()[:16]


def _safe_url(url: str) -> bool:
    try:
        parsed = urlsplit(url)
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except ValueError:
        return False


def _citation_html(citation: Citation, passages: Sequence[RetrievedPassage], *,
                   namespace: str, expanded: bool = False) -> str:
    label = escape(render_citation(citation, passages))
    target = f"{namespace}-citation-{citation.marker}"
    if not citation.resolved:
        return f'<p id="{target}" class="sec-warning">{label}: no source passage available.</p>'
    passage = passages[citation.marker - 1]
    link = (f'<a href="{escape(passage.url, quote=True)}" target="_blank" '
            'rel="noopener noreferrer">Open filing ↗</a>' if _safe_url(passage.url)
            else '<span>Filing link unavailable</span>')
    return (
        f'<details id="{target}" class="sec-source"{" open" if expanded else ""}>'
        f'<summary>{label}</summary><div class="sec-source-body">'
        f'<p>{label}</p>{link}<blockquote>{escape(passage.text)}</blockquote>'
        f'<small>Chunk: {escape(passage.chunk_id)}</small></div></details>'
    )


_STYLE = """
<style>
.sec-answer {font:inherit;line-height:1.6;color:inherit}
.sec-answer h3 {margin:.2rem 0 1rem}
.sec-answer .sec-claim {padding:.65rem .9rem;border-left:4px solid #78909c;
 margin:.6rem 0;border-radius:4px;white-space:pre-wrap;overflow-wrap:anywhere}
.sec-answer .sec-supported {border-left-color:#198060}
.sec-answer .sec-warning {border-left:4px solid #ba7900;padding:.65rem .9rem;
 background:rgba(220,153,20,.13);border-radius:4px;overflow-wrap:anywhere}
.sec-answer .sec-mismatch {border-left-color:#ce4545;background:rgba(206,69,69,.12)}
.sec-answer .sec-label {display:block;font-size:.8rem;font-weight:600}
.sec-answer a {color:inherit;text-decoration:underline;text-underline-offset:3px}
.sec-answer a:focus-visible,.sec-answer summary:focus-visible {outline:2px solid #448aff}
.sec-answer .sec-unresolved {font-weight:700;text-decoration:underline dotted}
.sec-answer .sec-source {border:1px solid #8886;border-radius:8px;margin:.7rem 0}
.sec-answer summary {padding:.7rem;cursor:pointer;font-weight:600}
.sec-answer .sec-source-body {padding:0 1rem 1rem}
.sec-answer blockquote {white-space:pre-wrap;overflow-wrap:anywhere;margin:1rem 0;
 padding-left:1rem;border-left:3px solid #8886;font:inherit}
</style>
"""

# Fixed code only: answer, source and URL text are escaped and never interpolated
# into JavaScript. A normal link is also keyboard operable (Enter).
_SCRIPT = """
<script>
document.querySelectorAll('.sec-answer').forEach(card => {
  if (card.dataset.citationsBound) return;
  card.dataset.citationsBound = 'true';
  card.addEventListener('click', event => {
    const link = event.target.closest('a[data-citation]');
    if (!link || !card.contains(link)) return;
    const target = document.getElementById(link.getAttribute('href').slice(1));
    if (!target) return;
    event.preventDefault();
    target.open = true;
    target.scrollIntoView({behavior: 'auto', block: 'nearest'});
    target.querySelector('summary').focus({preventScroll: true});
  });
});
</script>
"""


def _inline(text: str, answer: Answer, namespace: str) -> str:
    citations = {c.marker: c for c in answer.citations}
    parts, start = [], 0
    for match in _MARKER.finditer(text):
        parts.append(escape(text[start:match.start()]))
        marker = int(match[1])
        citation = citations.get(marker)
        if citation is not None and citation.resolved:
            parts.append(f'<a data-citation="{marker}" href="#{namespace}-citation-{marker}" '
                         f'aria-label="Show citation {marker}">[{marker}]</a>')
        else:
            parts.append(f'<span class="sec-unresolved" title="Unresolved citation">'
                         f'{escape(match[0])} (unresolved)</span>')
        start = match.end()
    parts.append(escape(text[start:]))
    return "".join(parts)


def answer_card_html(answer: Answer, *, key: str = "answer") -> str:
    """Build escaped markup, also exposed for deterministic rendering tests."""
    namespace = _namespace(key)
    parts = [_STYLE, f'<article class="sec-answer" id="{namespace}">',
             f'<h3>{escape(answer.question)}</h3>']
    if answer.abstained:
        reason = ABSTENTION_MESSAGES.get(
            answer.abstention_reason, "The available evidence does not answer this question.")
        parts.append(f'<p>{escape(answer.text)}</p><p>{escape(reason)}</p>')
    else:
        if answer.truncated or answer.parse_error:
            parts.append('<p class="sec-warning" role="alert">Incomplete or malformed answer. '
                         'Review the sources before using these claims.</p>')
        if answer.verification is None:
            parts.append('<p>Factual verification has not been run. Resolved citations identify '
                         'sources; they do not establish factual support.</p>')
        rows = [(s.raw_text, s.flagged) for s in answer.sentences] or [(answer.text, True)]
        for index, (text, flagged) in enumerate(rows):
            checks = [c for c in (() if answer.verification is None else answer.verification.checks)
                      if c.sentence_index == index]
            warnings = [c for c in checks if c.status in {"unverified", "mismatch"}]
            missing = any(int(m[1]) < 1 or not any(
                c.marker == int(m[1]) and c.resolved for c in answer.citations
            ) for m in _MARKER.finditer(text))
            flagged = flagged or missing or not _MARKER.search(text)
            evidence_supported = any(c.status == "supported" for c in checks)
            if any(c.status == "mismatch" for c in warnings):
                status, label = "sec-warning sec-mismatch", "Mismatch — needs review"
            elif flagged or answer.truncated or answer.parse_error:
                status, label = "sec-warning", "Needs review — unresolved, missing or unverified support"
            elif evidence_supported:
                status, label = "sec-supported", "Completed checks support this claim"
            elif warnings:
                status, label = "sec-warning", "Needs review — unresolved, missing or unverified support"
            else:
                status, label = "", "Citations resolved · factual support not checked"
            parts.append(f'<div class="sec-claim {status}"><span class="sec-label">{label}</span>'
                         f'{_inline(text, answer, namespace)}</div>')
            for check in warnings:
                parts.append(f'<p class="sec-warning">{escape(check.reason)}</p>')
        for check in answer.verification_warnings:
            if check.sentence_index is None or not 0 <= check.sentence_index < len(rows):
                parts.append(f'<p class="sec-warning">{escape(check.claim)} — {escape(check.reason)}</p>')
        for citation in sorted(answer.citations, key=lambda c: c.marker):
            parts.append(_citation_html(citation, answer.passages, namespace=namespace))
    return "\n".join(parts + ["</article>", _SCRIPT])


def answer_card(answer: Answer, *, key: str = "answer") -> None:
    """Render a card. Give each answer on the page a distinct, stable key.

    Inline links open and focus their evidence expander without a rerun or
    another model call. HTML details keep quotes and tables intact as text.
    """
    st.html(answer_card_html(answer, key=key), unsafe_allow_javascript=True)


def filter_sidebar(question: str = "", *, parsed: ParsedQuestion | None = None,
                   key: str = "filters", top_k: int = FINAL_K,
                   companies: Sequence[str] | None = None,
                   fiscal_years: Sequence[int] | None = None,
                   items: Sequence[str] = _ITEMS) -> Query:
    """Return the effective query, with visible, user-editable filters.

    A new question replaces all three axes with its extracted filters. Widget
    reruns preserve manual edits, including clearing an axis (empty means all).
    Call once per key on every run, outside a form. Callers may supply a parse
    already used by their controller; it must describe the same question.
    """
    question = question.strip()
    if parsed is not None and parsed.question != question:
        raise ValueError("parsed must describe the displayed question")
    companies = tuple(dict.fromkeys(c.upper() for c in (
        read_tickers() if companies is None else companies)))
    years = tuple(sorted(set(range(DEFAULT_FISCAL_YEARS[0], DEFAULT_FISCAL_YEARS[1] + 1)
                             if fiscal_years is None else fiscal_years)))
    items = tuple(dict.fromkeys(i.upper() for i in items))
    if top_k < 1:
        raise ValueError("top_k must be positive")
    if parsed is None and question:
        # With the facts store's labels as cues, as the app and answer_question
        # read a question: the Query returned here is searched as it stands, so
        # it has to lean toward tables for every question they read as a figure.
        parsed = parse_question(question, known_tickers=companies,
                                fiscal_years=(min(years), max(years)) if years else DEFAULT_FISCAL_YEARS)
    mentions = tuple(dict.fromkeys(item.upper() for group in _ITEM_MENTION.findall(question)
                                   for item in re.findall(r"\d{1,2}[a-z]?", group, re.IGNORECASE)))
    extracted = {
        "companies": list(parsed.tickers) if parsed else [],
        "years": list(parsed.fiscal_years) if parsed else [],
        "items": list(mentions),
    }
    # Keep extracted values visible even if a caller supplied a narrower menu;
    # never silently turn an unavailable filter into an unrestricted search.
    options = {"companies": list(dict.fromkeys((*companies, *extracted["companies"]))),
               "years": sorted(set(years) | set(extracted["years"])),
               "items": list(dict.fromkeys((*items, *mentions)))}
    changed = st.session_state.get(f"{key}:question") != question
    with st.sidebar:
        st.header("Filter filings")
        st.caption("Empty selections search all values. A new question resets filters to its mentions.")
        reset = st.button("Use question filters", key=f"{key}:reset")
        for axis in extracted:
            state_key = f"{key}:{axis}"
            if changed or reset or state_key not in st.session_state:
                st.session_state[state_key] = extracted[axis]
            options[axis] = list(dict.fromkeys((*options[axis], *st.session_state[state_key])))
        st.session_state[f"{key}:question"] = question
        tickers = st.multiselect("Company", options["companies"], key=f"{key}:companies")
        selected_years = st.multiselect("Fiscal year", options["years"], key=f"{key}:years")
        selected_items = st.multiselect("Item", options["items"], key=f"{key}:items")
        if parsed and parsed.unresolved:
            st.warning("Not in the corpus: " + ", ".join(parsed.unresolved))
        unknown_items = [i for i in mentions if i not in items]
        if unknown_items:
            st.warning("Unrecognised Items (kept as filters): " + ", ".join(unknown_items))
        base = parsed.to_query(top_k=top_k) if parsed else Query(question, top_k=top_k)
        query = replace(base, tickers=tuple(tickers), fiscal_years=tuple(selected_years),
                        items=tuple(selected_items))
        st.caption("Searching: " + ("; ".join(
            f"{axis}: {', '.join(map(str, values))}" for axis, values in query.filters.items()
        ) or "all companies, fiscal years and Items"))
    return query


__all__ = ["answer_card", "answer_card_html", "filter_sidebar"]
