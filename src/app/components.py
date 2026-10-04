"""Reusable Streamlit answer, evidence, trace and filter components (#37, #38).

Each component draws what it is handed. None builds a stack, asks a model or
caches anything: a page reads the question, loads through ``state.py``, asks,
and passes the results here, so a second page that shows an answer shows it
the same way. The sidebar's own selections are the only thing one holds.

The sidebar returns the Query to pass to ``answer_question(query=...)``.
Render completed Answer records; citation numbers always refer to prompt order.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from hashlib import sha256
from html import escape
import re
from urllib.parse import urlsplit

import streamlit as st

from src.config import read_tickers
from src.pipeline.constants import DEFAULT_FISCAL_YEARS
from src.rag.citations import render_citation
from src.rag.constants import FACTS_PROVIDER, MISTRAL, OLLAMA
from src.rag.query import ParsedQuestion, parse_question
from src.rag.records import ABSTENTION_MESSAGES, Answer, Citation
from src.retrieval.constants import FINAL_K
from src.retrieval.records import Query, RetrievedPassage
from src.stack import DEFAULT_STACK, SELECTABLE, STACKS, StackConfig


_MARKER = re.compile(r"\[(-?\d+)\]")
_ITEMS = tuple(str(i) + suffix for i in range(1, 17)
               for suffix in ({1: ("", "A", "B", "C"), 4: ("", "A"),
                               7: ("", "A"), 9: ("", "A", "B", "C")}.get(i, ("",))))
_ITEM_MENTION = re.compile(
    r"\bitems?\s+((?:\d{1,2}[a-z]?)(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d{1,2}[a-z]?\b)*)",
    re.IGNORECASE,
)


def _scope(query: Query) -> list[tuple[list[str], str]]:
    """What ``query`` searches, in words: for each of the three axes, the
    values selected, and what selecting none of them means.

    One wording, for the sidebar's own line and for the filters a page shows,
    so the two cannot describe the same search differently.
    """
    return [
        (list(query.tickers), "every company"),
        ([f"FY{year}" for year in query.fiscal_years], "every fiscal year"),
        ([f"Item {item}" for item in query.items], "every Item"),
    ]


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


def answer_card_html(answer: Answer, *, key: str = "answer", show_question: bool = True) -> str:
    """Build escaped markup, also exposed for deterministic rendering tests.

    For an answer with claims: an abstention is refused, since it has none.
    ``show_question=False`` leaves the heading out, for a page whose question
    box sits directly above the card and already says it.
    """
    if answer.abstained:
        # An abstention has no claims to lay out. Laid out here it was two bare
        # paragraphs under the question, which read as an answer that had not
        # loaded (#38). ``answer_card`` states it with ``abstention_notice``.
        raise ValueError("an abstained answer has no card: draw it with abstention_notice")
    namespace = _namespace(key)
    parts = [_STYLE, f'<article class="sec-answer" id="{namespace}">']
    if show_question:
        parts.append(f'<h3>{escape(answer.question)}</h3>')
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


# What to try after an abstention, for the two kinds a change of filter can
# turn into an answer. Both point at the sidebar, which every page that asks
# has; neither names another part of a page, which a page may not have.
_ABSTENTION_NEXT_STEPS = {
    "filters_excluded_all": "Clear a company, fiscal year or Item in the sidebar and ask again.",
    "company_not_in_corpus": "Pick one of the corpus's companies in the sidebar to search its "
                             "filings for a mention.",
}


def abstention_notice(answer: Answer) -> None:
    """State an abstention as an outcome: that there is no answer, and why (#38).

    An abstention is a result, and on a question the filings cannot answer it
    is the correct one. Drawn as two bare paragraphs it read as an answer
    that had failed to load.
    """
    reason = ABSTENTION_MESSAGES.get(
        answer.abstention_reason, "The available evidence does not answer this question.")
    st.warning(f"**No answer from the filings.** {reason}", icon=":material/do_not_disturb_on:")
    next_step = _ABSTENTION_NEXT_STEPS.get(answer.abstention_reason)
    if next_step:
        st.caption(next_step)


def answer_card(answer: Answer, *, key: str = "answer", show_question: bool = True) -> None:
    """Render a card. Give each answer on the page a distinct, stable key.

    Inline links open and focus their evidence expander without a rerun or
    another model call. HTML details keep quotes and tables intact as text.
    An abstention has no claims to put on a card, and is stated instead.
    """
    if answer.abstained:
        abstention_notice(answer)
        return
    st.html(answer_card_html(answer, key=key, show_question=show_question),
            unsafe_allow_javascript=True)


def resolved_filters(query: Query, parsed: ParsedQuestion | None = None) -> None:
    """Show what a question will be searched in, as soon as it is read (#38).

    The companies, fiscal years and Items of ``query``, which is what the
    sidebar ends up with, so a filter changed by hand shows here too. An axis
    with nothing selected says that every value is searched, so that an
    empty filter cannot be mistaken for a missed one.
    """
    parts = [" ".join(f":blue-badge[{value}]" for value in values) or f":gray-badge[{every}]"
             for values, every in _scope(query)]
    if parsed is not None:
        parts.append(f":violet-badge[{parsed.question_type} question]")
    st.markdown(":material/filter_alt: Searching " + " ".join(parts))


def configuration_picker(runs: Mapping[str, str], *, key: str = "configuration") -> str:
    """Pick one of the configurations ``src/stack.py`` can build, in the sidebar.

    In the registry's order, each labelled with the run that measured it
    (``runs``, id to label, from ``state.measured``), so a page can be set to
    the configuration a reported number came from. Returns the id.
    """
    with st.sidebar:
        config_id = st.selectbox(
            "Configuration", SELECTABLE,
            index=SELECTABLE.index(DEFAULT_STACK),
            format_func=lambda option: runs.get(option, f"{option} — {STACKS[option].name}"),
            help="Named in src/stack.py and built the same way the evaluation "
                 "command builds it. A row measured under results/ is labelled "
                 "with the run that measured it.",
            key=key,
        )
        if config_id not in runs:
            st.caption("No run under results/ has measured this configuration yet.")
        if not STACKS[config_id].metadata_filter:
            st.caption("This configuration was measured without the metadata filter, so "
                       "it searches every filing and the filters above are not applied.")
        st.caption("Uses local processed filings and indexes.")
    return config_id


# Each provider as the picker names it, what choosing it means for the person
# asking (where the model runs, what it needs, how long it takes), and the
# README section that sets it up.
_PROVIDERS = {
    OLLAMA: ("Ollama", "on this computer, with no key. An answer can take minutes on a laptop.",
             "Setting up Ollama"),
    MISTRAL: ("Mistral", "on Mistral's API, with the MISTRAL_API_KEY in your .env. "
                         "An answer takes seconds.", "Setting up Mistral"),
}


def provider_picker(models: Mapping[str, str], default: str, *,
                    unready: Mapping[str, str] | None = None, problem: str | None = None,
                    key: str = "provider") -> str:
    """Pick what writes the answers, in the sidebar: the local model or the hosted one.

    ``models`` is each provider with the model it would answer with,
    ``default`` the provider the page opens on, and ``unready`` each provider
    that could not answer as things stand with why, all from
    ``state.answer_models``. The choice lasts for the browser session and
    changes nothing in ``.env``.

    Two things are said under it, as soon as they are true and not when Ask
    is pressed: what the provider picked lacks, with what to do about it in
    the app, and ``problem``, what was wrong with ``.env``'s own choice. A
    provider that lacks something can still be picked: a figure the facts
    store looks up asks no model. Returns the provider picked.
    """
    with st.sidebar:
        provider = st.segmented_control(
            "Answer model", list(models), default=default, required=True,
            format_func=lambda option: _PROVIDERS[option][0], key=key,
            help="Opens on LLM_PROVIDER from .env, which unset means Ollama. "
                 "LLM_MODEL names the model of the provider .env selects; the "
                 "other one answers with its default.",
        )
        label, means, setup = _PROVIDERS[provider]
        st.caption(f"{models[provider]}, {means}")
        unready = unready or {}
        if provider in unready:
            # Short, and with no icon: the sidebar is narrow, and a warning
            # that ran to twenty lines pushed the Configuration box below it
            # off the screen.
            ready = [_PROVIDERS[other][0] for other in models if other not in unready]
            instead = f", or pick {' or '.join(ready)}" if ready else ""
            st.warning(f"{label} cannot write an answer yet: {unready[provider]}. Fix it in "
                       f"`.env` (README, {setup}) and restart the app{instead}. "
                       "A figure the facts store holds is still answered.")
        if problem:
            st.warning(problem)
    return provider


def _written_by(answer: Answer) -> str:
    """Who wrote the answer: the facts store, or a provider's model."""
    if answer.config.provider == FACTS_PROVIDER:
        return "facts store, no model"
    return f"{answer.config.model} ({answer.config.provider})"


def answer_summary(answer: Answer, config: StackConfig, seconds: float) -> None:
    """One row above an answer: the outcome, what produced it, and how long it took."""
    with st.container(horizontal=True, vertical_alignment="center"):
        if answer.abstained:
            st.badge("No answer", icon=":material/do_not_disturb_on:", color="orange")
        else:
            st.badge("Answered", icon=":material/check_circle:", color="green")
        st.badge(f"{config.id} · {config.name}", icon=":material/tune:", color="gray")
        # Nothing wrote an answer that was refused or found no passage.
        if answer.passages or not answer.abstained:
            st.badge(_written_by(answer), icon=":material/edit_note:", color="gray")
        # The first Ask of a process includes loading the indexes, and a
        # repeated question takes none.
        st.caption(f"Query completed in {seconds:.2f}s", width="content")


def trace_rows(answer: Answer) -> list[dict[str, object]]:
    """One row per passage the answer was written from, in prompt order.

    ``Source`` is the number the passage had in the prompt, which is the
    number a citation carries; ``Rank`` is where its retriever placed it,
    which differs once a question is searched one filing at a time.
    """
    cited = {citation.chunk_id for citation in answer.citations if citation.resolved}
    return [{
        "Source": f"[{number}]",
        "Cited": passage.chunk_id in cited,
        "Rank": passage.rank,
        "Score": passage.score,
        "Found by": ", ".join(passage.sources) or passage.retriever,
        "Company": passage.ticker,
        "Fiscal year": passage.fiscal_year,
        "Item": passage.item,
        "Type": passage.content_type,
        "Section": passage.title,
        "Passage": passage.text,
        "Filing": passage.url if _safe_url(passage.url) else None,
        "Chunk": passage.chunk_id,
    } for number, passage in enumerate(answer.passages, 1)]


def _plural(count: int, noun: str) -> str:
    return f"{count} {noun}{'' if count == 1 else 's'}"


def _search_step(answer: Answer, config: StackConfig, rows: Sequence[Mapping]) -> str:
    """What the search step of the trace says happened."""
    if answer.passages:
        filings = {(passage.ticker, passage.fiscal_year) for passage in answer.passages}
        cited = sum(row["Cited"] for row in rows)
        return (f"Found {_plural(len(answer.passages), 'passage')} in "
                f"{_plural(len(filings), 'filing')} with {config.retriever} retrieval, "
                f"{cited} cited")
    if answer.abstention_reason in {"beyond_the_filings", "company_not_in_corpus"}:
        return "Not searched: no filing could answer this"
    return f"Found no passage to answer from with {config.retriever} retrieval"


def _writing_step(answer: Answer) -> str:
    """What the writing step of the trace says happened."""
    if answer.config.provider == FACTS_PROVIDER:
        return "Answered from the facts store, with no model"
    if not answer.passages:
        return "No model was asked"
    if answer.abstained:
        return f"{answer.config.model} found no answer in the passages"
    return f"Written by {answer.config.model}"


def _writing_detail(answer: Answer) -> list[str]:
    """The lines under the writing step: what wrote the answer, or what was never asked to.

    An answer that asked no model still carries the provider and model that
    were picked. Listed as "Provider" and "Model" under "No model was asked",
    they read as what had answered, so they are said to have been picked and
    not asked. A figure from the facts store names the store.
    """
    config = answer.config
    took = None if answer.latency_ms is None else f"{answer.latency_ms / 1000:.2f}s"
    if config.provider == FACTS_PROVIDER:
        lines = [f"Looked up in the facts store ({config.model}), with no model asked."]
        return lines + ([f"The lookup took {took}"] if took else [])
    if not answer.passages:
        return [f"{config.model} ({config.provider}) was picked and not asked."]
    lines = [f"Provider: {config.provider} · Model: {config.model} · "
             f"Prompt: {config.prompt_template_id}"]
    if took:
        lines.append(f"Generation took {took}")
    if answer.truncated or answer.parse_error:
        lines.append("The output was cut off or malformed, so the answer is incomplete.")
    return lines


def _checking_step(answer: Answer) -> str:
    """What the checking step of the trace says happened."""
    if answer.verification is None:
        return "Not checked"
    counts = [(status, sum(check.status == status for check in answer.verification.checks))
              for status in ("supported", "mismatch", "unverified")]
    found = ", ".join(f"{count} {status}" for status, count in counts if count)
    return f"Checked: {found}" if found else "Checked: nothing to check"


def retrieval_trace(answer: Answer, parsed: ParsedQuestion, config: StackConfig, *,
                    key: str = "trace") -> None:
    """How an answer came about, a step at a time: read, searched, written, checked (#38).

    Drawn from what the ``Answer`` already carries, so nothing is searched or
    asked again. ``parsed`` is the question as read in the scope it was
    searched in, and ``config`` the configuration that answered. Each step's
    label says what happened and opens onto the detail: the passages are
    listed in prompt order with their scores, so a citation can be followed
    back to where its passage ranked. Give each trace on a page its own key.
    """
    rows = trace_rows(answer)
    with st.expander(f"Read as: {parsed.question_type} question", type="step",
                     icon=":material/psychology:"):
        for line in parsed.describe():
            st.caption(line)
        if not config.metadata_filter:
            st.caption(f"{config.id} applies no metadata filter, so every filing was searched.")

    with st.expander(_search_step(answer, config, rows), type="step",
                     icon=":material/search:"):
        if answer.sub_questions:
            st.caption("Searched one filing at a time: " + ", ".join(answer.sub_questions))
        if rows:
            st.dataframe(
                rows, hide_index=True, key=f"{key}:passages",
                column_config={
                    "Source": st.column_config.TextColumn(
                        help="The passage's number in the prompt, as a citation gives it.",
                        pinned=True),
                    "Rank": st.column_config.NumberColumn(
                        help="Where the retriever placed the passage, 1 being the best."),
                    "Score": st.column_config.NumberColumn(
                        format="%.4f", help="On the retriever's own scale: comparable "
                                            "within one method, not across methods."),
                    "Fiscal year": st.column_config.NumberColumn(format="%d"),
                    "Passage": st.column_config.TextColumn(width="large"),
                    "Filing": st.column_config.LinkColumn(display_text="Open filing"),
                },
            )
        else:
            st.caption(ABSTENTION_MESSAGES.get(answer.abstention_reason,
                                               "No passage was retrieved."))

    with st.expander(_writing_step(answer), type="step", icon=":material/edit_note:"):
        for line in _writing_detail(answer):
            st.caption(line)

    checks = () if answer.verification is None else answer.verification.checks
    with st.expander(_checking_step(answer), type="step", icon=":material/fact_check:"):
        if checks:
            st.dataframe(
                [{"Status": check.status, "Kind": check.kind, "Claim": check.claim,
                  "Reason": check.reason} for check in checks],
                hide_index=True, key=f"{key}:checks",
            )
        else:
            st.caption("No figure or claim was checked against the passages.")


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
        st.caption("Searching: " + "; ".join(
            ", ".join(values) or every for values, every in _scope(query)))
    return query


__all__ = [
    "abstention_notice", "answer_card", "answer_card_html", "answer_summary",
    "configuration_picker", "filter_sidebar", "provider_picker", "resolved_filters",
    "retrieval_trace", "trace_rows",
]
