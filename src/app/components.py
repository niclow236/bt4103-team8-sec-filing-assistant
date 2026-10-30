"""Reusable Streamlit answer, trace, filter and corpus components (#37, #38, #40).

Each component draws what it is handed. None builds a stack, asks a model or
caches anything: a page reads the question, loads through ``state.py``, asks,
and passes the results here, so a second page that shows an answer shows it
the same way. The sidebar's own selections are the only thing one holds.
The Results page's before and after view is handed the reader of one
configuration's saved questions rather than the questions, since which two it
reads turns on the pickers it draws; the reader is still ``state.py``'s.

The sidebar returns the Query to pass to ``answer_question(query=...)``.
Render completed Answer records; citation numbers always refer to prompt order.
The corpus controls and passage panels render processed rows without searching
an index or asking a model.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, replace
from hashlib import sha256
from itertools import groupby
from operator import attrgetter
from html import escape
import re
from typing import TYPE_CHECKING, Any, Literal, NamedTuple
from urllib.parse import urlsplit

import altair as alt
import streamlit as st
import pandas as pd

from src.config import read_tickers
from src.evaluation import breakdown, headlines, report
from src.pipeline.constants import DEFAULT_FISCAL_YEARS
from src.rag.citations import render_citation
from src.rag.constants import FACTS_PROVIDER, MISTRAL, OLLAMA, SOCLAAS
from src.rag.query import ParsedQuestion, parse_question
from src.rag.records import Answer, Citation, abstention_text
from src.retrieval.constants import FINAL_K
from src.retrieval.records import Query, RetrievedPassage
from src.stack import DEFAULT_STACK, SELECTABLE, STACKS, StackConfig

if TYPE_CHECKING:
    # For an annotation alone. A component is handed what state.py holds and
    # imports nothing from it when the app runs.
    from src.app.state import AnswerModels, FailureExample, QuestionOutcome, WorkedExample
    from src.rag.generate import ProviderUnavailable


_MARKER = re.compile(r"\[(-?\d+)\]")
_ITEMS = tuple(str(i) + suffix for i in range(1, 17)
               for suffix in ({1: ("", "A", "B", "C"), 4: ("", "A"),
                               7: ("", "A"), 9: ("", "A", "B", "C")}.get(i, ("",))))
_ITEM_MENTION = re.compile(
    r"\bitems?\s+((?:\d{1,2}[a-z]?)(?:\s*(?:,\s*(?:and\s+)?|and\s+|&\s*)\d{1,2}[a-z]?\b)*)",
    re.IGNORECASE,
)


# What each column of the Results page means to a user (#164), keyed by the
# name ``report`` gives it, which the Full metrics section keeps: its label,
# and a tooltip naming the technical metric, so an assessor can match it to
# the report. ``{top}`` and ``{at}`` are the cutoff the run scored at, as in
# "in the top 16" and "Recall@16"; a run that saved none has neither.
_RESULT_METRIC_LABELS = {
    breakdown.FOUND.title: (breakdown.FOUND.title,
                           "Count of questions with at least one supporting passage in the "
                           "top 10, not mean Recall. Questions without labelled evidence or "
                           "readable ranks are excluded."),
    "Headline correctness": ("Answer correctness (0–1)",
                             "Mean saved correctness score, including partial credit and "
                             "automatically scored abstentions; not a count of correct answers."),
    "Headline latency": ("Typical answer time (s)",
                         "Median of saved generation-call latencies in seconds."),
    "Relevance-scored": ("Questions with a right passage",
                         "How many of the questions name a passage that answers them: "
                         "the scores of finding it are worked out over these."),
    "Recall": ("Right passages found {top}",
               "Recall{at}: the share of each question's right passages that were "
               "retrieved, averaged over the questions that have one."),
    report.MEDIAN_RANK: ("Typical rank of the first right passage",
                         "The median of 1 / reciprocal rank, each question's MRR{at} "
                         "score turned back into a rank; 1 is the top. A blank means "
                         "the typical question found no right passage {top}; None means "
                         "the run saved no ranks to work it out from."),
    "Hard negatives": ("Questions with a similar filing",
                       "How many of the questions have hard negatives: passages from the "
                       "wrong company or year that read like the right one."),
    "Hard Negative Accuracy": ("Not fooled by a similar filing (wrong company or year)",
                               "Hard-negative accuracy{at}: the share of each question's hard "
                               "negatives kept out of the passages retrieved, averaged over "
                               "the questions that have them."),
    "Abstention rate": ("Refused rather than guessed",
                        "Abstention rate: the share of questions the configuration declined "
                        "to answer, saying the filings did not support an answer."),
    "Median latency (ms)": ("Typical answer time (ms)",
                            "Median latency: the middle of the configuration's answer times, "
                            "in milliseconds."),
    "LLM questions": ("Answered by the language model",
                      "LLM questions: how many questions went to the language model, rather "
                      "than being answered straight from the facts store."),
    # Answer quality keeps its names, each explained.
    "Outputs": ("Outputs", "How many of the configuration's answers were scored."),
    "Correctness": ("Correctness",
                    "How completely and accurately an answer answers the question against "
                    "the benchmark's answer, as the judge model scores it from 0 to 1."),
    "Faithfulness": ("Faithfulness",
                     "The share of an answer's factual claims that its own cited passages "
                     "support, as the judge model scores it."),
    "Answerability": ("Answerability",
                      "The share of questions where answering or refusing was the right "
                      "call: refusing only the questions the filings cannot answer."),
    "Citation precision": ("Citation precision",
                           "The share of the passages an answer cites that are right passages."),
    "Citation recall": ("Citation recall",
                        "The share of each question's right passages that its answer cites."),
    **{f"{label} scored": (f"{label} scored",
                           f"How many of the outputs were scored on {label.lower()}; the "
                           "mean beside it is over these.")
       for label in report.QUALITY_LABELS.values()},
    "Claim checks": ("Claim checks",
                     "How the answers' claims fared against the verification checks: each "
                     "outcome and how many claims had it."),
}
# The main view's retrieval columns, in order: the scores come first, so the
# page's width shows them before the counts behind them. nDCG and MRR are
# left to the Full metrics section, and the median rank stands in for MRR.
_MAIN_RETRIEVAL_COLUMNS = ("Configuration", "Name", "Recall", report.MEDIAN_RANK,
                           "Hard Negative Accuracy", "Questions", "Relevance-scored",
                           "Hard negatives")


def _cutoff(run: Mapping[str, Any]) -> int | None:
    """How many passages the run scored at, or None if it saved no count:
    json's true reads as an int, and a top 0 retrieves nothing."""
    top_k = run.get("top_k")
    return top_k if report.is_count(top_k) else None


class _Heading(NamedTuple):
    """How the Results page heads one column: its label, and its tooltip."""

    label: str
    tooltip: str


def _headings(top_k: int | None) -> dict[str, _Heading]:
    """Each column's heading, by the name ``report`` gives it, at the run's cutoff."""
    top, at = ((f"in the top {top_k}", f"@{top_k}") if top_k is not None
               else ("among those retrieved", ""))
    return {column: _Heading(label.format(top=top), tooltip.format(top=top, at=at))
            for column, (label, tooltip) in _RESULT_METRIC_LABELS.items()}


def _plain_dataframe(table: pd.DataFrame, top_k: int | None) -> None:
    """Draw ``table`` with each column the labels name in plain words, and its tooltip."""
    headings = _headings(top_k)
    st.dataframe(table.rename(columns={column: heading.label
                                       for column, heading in headings.items()}),
                 column_config={heading.label: st.column_config.Column(help=heading.tooltip)
                                for heading in headings.values()},
                 width="stretch", hide_index=True)


def _results_table(configurations: list[dict[str, Any]], benchmark: str) -> pd.DataFrame:
    return pd.DataFrame(report.retrieval_table(configurations, benchmark))


def _main_results_table(configurations: list[dict[str, Any]], benchmark: str) -> pd.DataFrame:
    """The retrieval table of the page's main view, with the median rank.

    The rank is a number, so the column sorts as one. Past the cutoff it is
    infinite, which sorts after every rank and which the grid draws blank.
    """
    table = _results_table(configurations, benchmark)
    ranks = [item["benchmark_metrics"][benchmark].get(report.MEDIAN_RANK)
             for item in configurations]
    # A configuration whose rows saved no reciprocal rank has none, and the
    # column goes, as the report's tables drop one that no row fills.
    if any(rank is not None for rank in ranks):
        table[report.MEDIAN_RANK] = ranks
    return table[[column for column in _MAIN_RETRIEVAL_COLUMNS if column in table]]


def _results_model_table(configurations: list[dict[str, Any]], benchmark: str) -> pd.DataFrame:
    return pd.DataFrame(report.model_table(configurations, benchmark))


# The two bars the chart's message is about, and the grey the rest are drawn
# in so they recede. Orange against blue rather than green against red, so the
# pair is still two colours to a reader who cannot tell those apart.
#
# One set for both themes, rather than a pair picked by ``_dark_theme()``.
# Streamlit infers the theme from the background and documents that the answer
# is wrong "when the user changes the theme in the settings menu", which is the
# one moment anyone checks a chart in both. Each of these clears 3:1, what
# WCAG asks of a shape, on Streamlit's light and dark backgrounds alike, so
# nothing here has to know which one the reader is on.
_FOUND_COLOURS = {"default": "#3b82f6", "baseline": "#d97706", "other": "#6b7280"}

# The value above each bar, in the one grey that reads on both backgrounds.
# Left uncoloured it comes out black, which disappears on the dark one, and
# Streamlit's theme does not reach a text mark. No colour clears 4.5:1 against
# white and #0e1117 at once -- the arithmetic allows 4.35:1 at best, which
# this is -- so it is a hair under what WCAG asks of small text, and the
# figure is in the tooltip and the table as well.
_FOUND_LABEL = "#797979"


def _found_role(config_id: str) -> str:
    """Which of the chart's three colours ``config_id`` is drawn in."""
    if config_id == DEFAULT_STACK:
        return "default"
    return "baseline" if config_id == breakdown.BASELINE else "other"


def _found_chart(scores: list[breakdown.ConfigurationScore]) -> alt.LayerChart:
    """One bar per configuration, the default and the baseline coloured.

    Drawn on a fixed 0 to 100% axis so two runs can be compared by eye, and
    labelled with each value, since reading a bar against gridlines is not
    what the number is for. Nothing here is told which theme the reader is
    on, because the server can have it wrong: every colour is one that reads
    on either background.
    """
    rows = pd.DataFrame([{"Configuration": item.config_id, "Name": item.name,
                          "Score": item.score, "Questions scored": item.scored,
                          "Role": _found_role(item.config_id)}
                         for item in scores if item.score is not None])
    order = list(rows["Configuration"])
    base = alt.Chart(rows).encode(
        x=alt.X("Configuration:N", sort=order, title=None, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Score:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%")),
        tooltip=[alt.Tooltip("Configuration:N"), alt.Tooltip("Name:N"),
                 alt.Tooltip("Score:Q", format=".1%", title=breakdown.FOUND.title),
                 alt.Tooltip("Questions scored:Q", format="d")],
    )
    bars = base.mark_bar().encode(
        color=alt.Color("Role:N", legend=None,
                        scale=alt.Scale(domain=list(_FOUND_COLOURS),
                                        range=list(_FOUND_COLOURS.values()))))
    # Above the bar rather than inside it, so a short bar still carries its
    # label and the text never has to read against three different fills.
    labels = base.mark_text(dy=-8, color=_FOUND_LABEL, fontWeight="bold").encode(
        text=alt.Text("Score:Q", format=".0%"))
    return bars + labels


def _render_found_chart(run: Mapping[str, Any], benchmark: str,
                        outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]]) -> None:
    """How often each configuration put a right passage in the top 10, under a
    title that says what the bars show (#166)."""
    scores = breakdown.measured_by_configuration(run, benchmark, outcomes)
    if not any(item.score is not None for item in scores):
        # The measure reads a question's own rank, so a run that saved only
        # its means has no chart. Said once, rather than leaving a gap.
        st.caption(f"No chart of {breakdown.FOUND.title.lower()}: this run saved no "
                   "per-question results for this benchmark.")
        return
    headline = breakdown.found_headline(scores)
    if headline:
        st.markdown(f"**{headline}**")
    st.altair_chart(_found_chart(scores), width="stretch")
    st.caption(f"{breakdown.FOUND.title}, over the questions each configuration scored. "
               f"{DEFAULT_STACK} is the app's default and {breakdown.BASELINE} the baseline "
               "every other row adds to.")


def _render_results_benchmark(title: str, retrieval: list[dict[str, Any]],
                              run: dict[str, Any], benchmark: str,
                              outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]],
                              ) -> bool:
    """Draw one benchmark's scores, and say whether the run saved any for it."""
    st.subheader(title)
    headline_shown = _render_headlines(run, benchmark, outcomes)
    model_table = _results_model_table(run["configurations"], benchmark)
    quality = pd.DataFrame(report.answer_quality_table(run, benchmark))
    if not retrieval and model_table.empty and quality.empty and not headline_shown:
        st.info("No saved results for this benchmark.")
        return False
    top_k = _cutoff(run)
    if retrieval:
        _plain_dataframe(_main_results_table(retrieval, benchmark), top_k)
    # #202's table keeps the scores in plain words; its bars of the same three
    # metrics give way to one chart of one of them (#166).
    _render_found_chart(run, benchmark, outcomes)
    if not model_table.empty:
        st.subheader("Answer-model ablation")
        _plain_dataframe(model_table, top_k)
    if not quality.empty:
        st.subheader("Answer quality")
        _plain_dataframe(quality, top_k)
        quality_chart = report.quality_chart_rows(run, benchmark)
        if quality_chart:
            st.bar_chart(pd.DataFrame(quality_chart).pivot(
                index="Configuration", columns="Metric", values="Value"), stack=False)
    return True


_HEADLINE_COLUMNS = {"found": breakdown.FOUND.title,
                     "correctness": "Headline correctness",
                     "refused": "Abstention rate", "latency": "Headline latency"}


def _headline_value(tile: headlines.Headline) -> str:
    """Counts stay counts, correctness stays a mean, milliseconds become seconds."""
    if tile.count is not None:
        return f"{tile.count} of {tile.scored}"
    return f"{tile.value:.2f} s" if tile.key == "latency" else f"{tile.value:.2f} / 1"


def _headline_delta(tile: headlines.Headline) -> str | None:
    """A same-cohort C0 difference, with units that match the displayed measure."""
    if tile.delta is None:
        return None
    if tile.key in {"found", "refused"}:
        return f"{tile.delta * 100:+.1f} pp vs {breakdown.BASELINE}"
    unit = " s" if tile.key == "latency" else ""
    return f"{tile.delta:+.2f}{unit} vs {breakdown.BASELINE}"


def _headline_icons(tile: headlines.Headline) -> str:
    """Ten proportionally filled circles with an exact accessible count.

    Fractional fill avoids claiming 10 out of 10 when 96% were found. The
    real count is printed in the tile and its accessible name, not inferred
    by counting rounded icons. Fixed colours remain visible in either theme.
    """
    icons = []
    for index in range(10):
        filled = min(1.0, max(0.0, tile.value * 10 - index)) * 100
        icons.append('<span aria-hidden="true" style="display:inline-block;'
                     'width:1.1rem;height:1.1rem;border:1px solid #6b7280;'
                     'border-radius:50%;margin-right:0.25rem;'
                     f'background:linear-gradient(to right,#3b82f6 {filled:.2f}%,'
                     f'transparent {filled:.2f}%);"></span>')
    label = (f"Found a right passage in the top {breakdown.FOUND_WITHIN} for "
             f"{tile.count} of {tile.scored} scored questions. Ten circles show the proportion.")
    return f'<div role="img" aria-label="{label}">{"".join(icons)}</div>'


def _render_headlines(run: Mapping[str, Any], benchmark: str,
                      outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]]) -> bool:
    """Headline the selected run's app default before tables; never load live models."""
    tiles = headlines.headline_tiles(run, benchmark, outcomes)
    if not tiles:
        return False
    headings = _headings(_cutoff(run))
    with st.container(key=f"results_headlines_{benchmark}"):
        st.markdown(f"**At a glance — {DEFAULT_STACK}, the app's default configuration**")
        # Two per row: four columns truncate the plain-language labels at
        # the app's normal content width, defeating the headline's purpose.
        columns = []
        for start in range(0, len(tiles), 2):
            columns.extend(st.columns(min(2, len(tiles) - start)))
        for column, tile in zip(columns, tiles):
            heading = headings[_HEADLINE_COLUMNS[tile.key]]
            help_text = heading.tooltip + f" Based on {tile.scored} scored questions."
            if tile.delta is None:
                help_text += " No C0 delta: matching per-question observations are unavailable."
            else:
                help_text += " Compared with C0 on exactly the same scored question IDs; pp means percentage points."
            if tile.key == "latency":
                help_text += " Saved generation-call time only; excludes retrieval and judging, not end-to-end latency."
            if tile.key == "refused":
                help_text += " A refusal is not necessarily the correct decision; a higher rate is not inherently better."
            with column:
                st.metric(heading.label, _headline_value(tile), delta=_headline_delta(tile),
                          delta_color="off" if tile.key == "refused" else
                          "inverse" if tile.key == "latency" else "normal", help=help_text)
        found = next((tile for tile in tiles if tile.key == "found"), None)
        if found is not None:
            st.html(_headline_icons(found))
    return True


# What a configuration is compared against until another baseline is picked:
# naive BM25, the configuration every design choice is measured from. Named
# once, beside the measures, so the chart and this view cannot drift apart.
_DEFAULT_BASELINE = breakdown.BASELINE


class _Measure(NamedTuple):
    """What the before and after view counts changes on. Its title and score
    are the breakdown's (``breakdown.MEASURES``), so the two views score a
    question the same way."""

    title: str
    column: str         # the table column that says how a question changed on it
    shown: str          # what each configuration's own column is headed with
    # What that column shows of a question's outcome, and the column's type.
    value: Callable[[QuestionOutcome], Any]
    dtype: str | None
    # How one question's outcome scores on it, None where it was not scored.
    score: Callable[[QuestionOutcome], float | None]
    unscored: str       # why a question both answered was not counted on it


_TOP_10 = _Measure(breakdown.FOUND.title, f"Top {breakdown.FOUND_WITHIN}",
                   # A rank is a whole number, shown as 2 and not 2.0 beside a missing one.
                   "rank", attrgetter("rank"), "Int64", breakdown.FOUND.score,
                   # No right passage to find, or a row that could not be read.
                   "were not scored on retrieval by both")
_MEASURES = (
    _TOP_10,
    _Measure(breakdown.CORRECTNESS.title, "Correctness", "correctness",
             attrgetter("correctness"), None, breakdown.CORRECTNESS.score,
             "had no scored answer from both"),
)
_SHOW_ALL = "All questions"
# The text colour of a change in the table, matching the green and red of the
# counts above it in each theme: (light, dark).
_IMPROVED, _WORSE, _UNCHANGED = "improved", "worse", "unchanged"
_CHANGE_COLOURS = {_IMPROVED: ("#158237", "#5ce488"), _WORSE: ("#bd4043", "#ff6c6c")}


def _change(before: float, after: float) -> str:
    return _IMPROVED if after > before else _WORSE if after < before else _UNCHANGED


# The table's own column saying a question's text is today's benchmark's,
# counted for the caption and dropped before the table is drawn.
_FROM_CURRENT = "_from_current_benchmark"


def _worded_by(before: QuestionOutcome, after: QuestionOutcome) -> QuestionOutcome:
    """The outcome whose text the table shows: the baseline's, unless only the
    other configuration kept the run's own wording."""
    if after.question and not after.current_text and (before.current_text or not before.question):
        return after
    return before


def _measured_change(measure: _Measure, before: QuestionOutcome,
                     after: QuestionOutcome) -> str | None:
    """How one question changed on ``measure``, None unless both were scored on it."""
    scores = measure.score(before), measure.score(after)
    return None if None in scores else _change(*scores)


def _question_changes(before: Mapping[str, QuestionOutcome], after: Mapping[str, QuestionOutcome],
                      benchmark: str, names: tuple[str, str],
                      ) -> tuple[int, list[tuple[_Measure, Counter]], pd.DataFrame]:
    """What changed from ``before`` to ``after`` on one benchmark's questions.

    Returns how many of its questions both answered, each measure both were
    scored on with its counts, and one row per question with each
    configuration's own value, headed by ``names``, how it changed, and
    whether its text is today's benchmark's (``_FROM_CURRENT``).
    """
    shared = {question_id: (before[question_id], outcome)
              for question_id, outcome in after.items()
              if question_id in before and outcome.benchmark == benchmark}
    changed = {question_id: {measure.column: _measured_change(measure, old, new)
                             for measure in _MEASURES}
               for question_id, (old, new) in shared.items()}
    counts = []
    for measure in _MEASURES:
        changes = Counter(change[measure.column] for change in changed.values()
                          if change[measure.column] is not None)
        if changes:
            counts.append((measure, changes))
    rows = []
    for question_id, (old, new) in shared.items():
        worded = _worded_by(old, new)
        row = {"Question": question_id, "Asked": worded.question,
               _FROM_CURRENT: worded.current_text}
        for measure, _ in counts:
            for name, outcome in zip(names, (old, new)):
                row[f"{name} {measure.shown}"] = measure.value(outcome)
            row[measure.column] = changed[question_id][measure.column]
        if any(row[measure.column] for measure, _ in counts):
            rows.append(row)
    table = pd.DataFrame(rows).astype({
        f"{name} {measure.shown}": measure.dtype
        for measure, _ in counts if measure.dtype for name in names})
    return len(shared), counts, table


def _render_question_changes(run: Mapping[str, Any], benchmark: str,
                             outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]]) -> None:
    """Question by question, what one configuration changed against a baseline (#169)."""
    # Only a configuration whose questions on this benchmark were read can be
    # compared. A summary's own by-benchmark scores do not make it one.
    # Its rows, or a summary keyed as the tabs are, tell which benchmarks it
    # ran; with neither, there is no telling.
    elsewhere = [item["config_id"] for item in run["configurations"]
                 if (item.get("questions_read") or report.keyed_as_tabs(item))
                 and benchmark not in item["benchmark_metrics"]]
    unread = [item["config_id"] for item in run["configurations"]
              if not item.get("questions_read") and item["config_id"] not in elsewhere]
    config_ids = breakdown.configurations_read(run, benchmark)
    # With fewer than two to compare, the section still says which were left
    # out for want of their files; one that ran no question here needs no word.
    if len(config_ids) < 2 and not (unread and len(run["configurations"]) > 1):
        return
    st.subheader("Before and after")
    if unread:
        st.caption("Not offered, since their per-question results were not found on this "
                   "machine: " + ", ".join(unread))
    if elsewhere:
        st.caption("Not offered, since they saved no question on this benchmark: "
                   + ", ".join(elsewhere))
    if len(config_ids) < 2:
        return
    baseline = _existing_choice("Baseline", config_ids, key=f"results_baseline_{benchmark}",
                                format_func=str, default=_DEFAULT_BASELINE)
    others = [config_id for config_id in config_ids if config_id != baseline]
    compared = _existing_choice("Configuration", others, key=f"results_compared_{benchmark}",
                                format_func=str)
    answered, counts, table = _question_changes(outcomes(run["run_dir"], baseline),
                                                outcomes(run["run_dir"], compared),
                                                benchmark, (baseline, compared))
    if not counts:
        st.info(f"{baseline} and {compared} have no saved question in common "
                "that was scored the same way on this benchmark.")
        return
    scored_at = _cutoff(run)
    first_left_out = breakdown.FOUND_WITHIN + 1
    for measure, changes in counts:
        _render_change_counts(measure, changes, answered)
        if measure is _TOP_10 and scored_at is not None and scored_at >= first_left_out:
            # The cutoff stays the issue's top 10, and this says why the
            # counts can disagree with the scores above them (#169).
            ranks = (f"{first_left_out}" if scored_at == first_left_out
                     else f"{first_left_out} to {scored_at}")
            st.caption(f"The scores above count a right passage in this run's top "
                       f"{scored_at}, so one ranked {ranks} counts there but not here.")
    filters = {f"{change.capitalize()}: {measure.column.lower()}": (measure.column, change)
               for measure, _ in counts for change in (_IMPROVED, _WORSE)}
    show = st.selectbox("Show", [_SHOW_ALL, *filters], key=f"results_show_{benchmark}")
    if show in filters:
        column, change = filters[show]
        table = table[table[column] == change]
    from_current_benchmark = int(table[_FROM_CURRENT].sum())
    table = table.drop(columns=_FROM_CURRENT)
    if from_current_benchmark:
        st.caption(f"The text of {_plural(from_current_benchmark, 'question')} is from the "
                   "benchmark as it stands now, since this run kept none, and may not be how "
                   "the run asked it.")
    dark = _dark_theme()

    def colour(change: str | None) -> str:
        shades = _CHANGE_COLOURS.get(change)
        return f"color: {shades[dark]}" if shades else ""

    styled = table.style.map(colour, subset=[measure.column for measure, _ in counts])
    # A styled table is shown as the styler formats it, so say how: scores to
    # two places. A missing value is drawn by the grid itself, as "None",
    # whatever the styler says.
    st.dataframe(styled.format(precision=2), width="stretch", hide_index=True)


def _render_change_counts(measure: _Measure, changes: Counter, answered: int) -> None:
    """One measure's counts, and how many of the questions both answered they cover."""
    st.markdown(f"**{measure.title}**")
    counted = changes.total()
    if counted == answered:
        st.caption(f"Out of the {_plural(answered, 'question')} both configurations answered.")
    else:
        st.caption(f"Out of {counted} of the {answered} questions both configurations answered; "
                   f"the other {answered - counted} {measure.unscored}.")
    improved, worse, unchanged = st.columns(3)
    improved.metric(":green[Improved]", changes[_IMPROVED])
    worse.metric(":red[Worse]", changes[_WORSE])
    unchanged.metric("Unchanged", changes[_UNCHANGED])


_SCORE_BAR = st.column_config.ProgressColumn("Score", format="%.2f", min_value=0, max_value=1)


def _breakdown_table(column: str, groups: list[breakdown.Group]) -> pd.DataFrame:
    """One grouping's rows: a group too small to read keeps its counts but not its score."""
    table = pd.DataFrame([{column: group.name, "Questions": group.questions,
                           "Scored": group.scored,
                           "Score": group.score if group.readable else None,
                           "Too few to read": not group.readable} for group in groups])
    return table if table["Too few to read"].any() else table.drop(columns="Too few to read")


def _render_breakdown(run: Mapping[str, Any], benchmark: str,
                      outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]]) -> None:
    """Where the default configuration is strong and weak: its scores by
    question type and by difficulty, each beside the questions it covers (#170)."""
    splits = breakdown.default_breakdown(run, benchmark, outcomes)
    if not splits:
        return
    st.subheader("By question type and difficulty")
    st.caption(f"{DEFAULT_STACK}, the app's default configuration, on this benchmark. A group "
               f"with fewer than {breakdown.MIN_GROUP_QUESTIONS} scored questions is too few "
               "to read, so its score is not shown.")
    for measure, measured in groupby(splits, key=attrgetter("measure")):
        st.markdown(f"**{measure.title}**")
        for split in measured:
            line = breakdown.headline(split.groups, split.grouping)
            if line:
                st.caption(line)
            st.dataframe(_breakdown_table(split.grouping.capitalize(), split.groups),
                         column_config={"Score": _SCORE_BAR}, width="stretch", hide_index=True)


def _render_full_metrics(retrieval: list[dict[str, Any]], benchmark: str) -> None:
    """The retrieval table under the report's own names, nDCG and MRR among
    them, collapsed below the rest of the tab (#164, #171)."""
    if retrieval:
        with st.expander("Full metrics"):
            st.dataframe(_results_table(retrieval, benchmark), width="stretch", hide_index=True)


def _render_worked_example(run: Mapping[str, Any],
                           examples: Callable[[str, str], Sequence[WorkedExample]]) -> None:
    """Walk one correct saved answer from question through verification (#167)."""
    def plain(text: str) -> str:
        """Keep currency in saved text from becoming Streamlit inline maths."""
        return text.replace("$", r"\$")

    st.subheader("Worked Example")
    run_dir = run.get("run_dir")
    saved = list(examples(str(run_dir), DEFAULT_STACK)) if run_dir else []
    if not saved:
        st.caption(f"This run has no saved correct answer from C6 or {DEFAULT_STACK} with its "
                   "answer text and an EDGAR passage, so there is no worked example to show. "
                   "Runs that only measured retrieval save no answers, and a run whose judge "
                   "failed saved no correctness score.")
        return

    # Each run gets its own cursor. Changing the selected run cannot carry an
    # out-of-range position across, and the page never writes example text.
    cursor_key = "results_example_" + sha256(str(run_dir).encode()).hexdigest()[:16]
    cursor = int(st.session_state.get(cursor_key, 0)) % len(saved)
    if st.button("Next example", key=cursor_key + "_next", disabled=len(saved) == 1):
        cursor = (cursor + 1) % len(saved)
        st.session_state[cursor_key] = cursor
    example = saved[cursor]
    st.caption(f"Example {cursor + 1} of {len(saved)} from {run['run_id']}'s saved "
               f"{example.config_id} rows (question {example.question_id}).")

    st.markdown("**1. Benchmark question**")
    st.write(plain(example.question))
    citation = (f", cited as [{example.citation_marker}]"
                if example.citation_marker is not None else ", not cited by the answer")
    st.markdown(f"**2. Passage {plain(example.config_id)} retrieved{citation}**")
    with st.container(border=True):
        st.markdown(f"**{plain(passage_labels([example.passage])[0])}**")
        _passage_body(example.passage)
    if example.citation_marker is not None and example.other_citation_markers:
        other_markers = ", ".join(f"[{marker}]" for marker in example.other_citation_markers)
        st.caption(f"The answer also cites {other_markers}; only the passage for "
                   f"[{example.citation_marker}] is shown.")
    st.markdown("**3. Saved answer**")
    st.write(plain(example.answer))
    st.caption(f"Benchmark answer: {plain(example.expected_answer)}")
    correctness = "The saved judge marked this answer fully correct."
    if example.correctness_reason:
        correctness += f" {plain(example.correctness_reason)}"
    st.success(correctness)
    st.markdown("**4. Check against the XBRL facts store**")
    if not example.fact_checks:
        st.info("No XBRL facts-store check was saved for this answer.")
    for check in example.fact_checks:
        status = str(check.get("status") or "unverified")
        claim = plain(str(check.get("claim") or "Saved numeric claim"))
        reason = plain(str(check.get("reason") or "No reason was saved."))
        message = f"**{status.replace('_', ' ').title()}:** {claim} — {reason}"
        (st.success if status == "supported" else st.warning)(message)
        details = [f"{label}: {plain(str(check[field]))}" for field, label in
                   (("figure", "figure"), ("value", "XBRL value"), ("unit", "unit"))
                   if check.get(field) not in (None, "")]
        if details:
            st.caption(" · ".join(details))


def _render_failure_example(
        run: Mapping[str, Any],
        failures: Callable[[str, str], Sequence[FailureExample]] | None) -> None:
    """Walk one wrong or refused saved answer through its evidence (#207)."""
    def plain(text: str) -> str:
        """Keep currency in saved text from becoming Streamlit inline maths."""
        return text.replace("$", r"\$")

    st.markdown("### Failure example")
    run_dir = run.get("run_dir")
    saved = list(failures(str(run_dir), DEFAULT_STACK)) if run_dir and failures else []
    if not saved:
        st.caption("This run has no saved answer or refusal scored below 0.5 with the text, "
                   "benchmark answer, and EDGAR passage needed for a walkthrough. Refusing an "
                   "unanswerable question is correct, so those refusals are not shown here.")
        return

    cursor_key = "results_failure_" + sha256(str(run_dir).encode()).hexdigest()[:16]
    cursor = int(st.session_state.get(cursor_key, 0)) % len(saved)
    if st.button("Next failure", key=cursor_key + "_next", disabled=len(saved) == 1):
        cursor = (cursor + 1) % len(saved)
        st.session_state[cursor_key] = cursor
    example = saved[cursor]
    label = "Refused" if example.refused else "Answered wrongly"
    st.error(f"**{label}** — saved failure {cursor + 1} of {len(saved)} from "
             f"{run['run_id']}'s {example.config_id} rows (question {example.question_id}).")

    st.markdown("**1. Benchmark question**")
    st.write(plain(example.question))
    citation = (f", cited as [{example.citation_marker}]"
                if example.citation_marker is not None else ", not cited by the answer")
    st.markdown(f"**2. Passage {plain(example.config_id)} retrieved{citation}**")
    with st.container(border=True):
        st.markdown(f"**{plain(passage_labels([example.passage])[0])}**")
        _passage_body(example.passage)
    if example.citation_marker is not None and example.other_citation_markers:
        other_markers = ", ".join(f"[{marker}]" for marker in example.other_citation_markers)
        st.caption(f"The answer also cites {other_markers}; only the passage for "
                   f"[{example.citation_marker}] is shown.")

    st.markdown("**3. Saved refusal**" if example.refused else "**3. Saved answer**")
    st.write(plain(example.answer))
    if example.refused:
        reason = example.abstention_reason or "not recorded"
        st.caption(f"Saved abstention reason: {plain(reason)} — "
                   f"{plain(abstention_text(example.abstention_reason))}")
    st.caption(f"Benchmark answer: {plain(example.expected_answer)}")
    st.markdown("**4. Saved judge verdict**")
    verdict = ("The saved judge did not record a correctness score."
               if example.correctness is None else
               f"The saved judge gave this answer {example.correctness:.2f} correctness.")
    if example.correctness_reason:
        verdict += f" {plain(example.correctness_reason)}"
    st.error(verdict)


def render_results(runs: list[dict[str, Any]], *,
                   outcomes: Callable[[str, str], Mapping[str, QuestionOutcome]],
                   examples: Callable[[str, str], Sequence[WorkedExample]],
                   failures: Callable[[str, str], Sequence[FailureExample]] | None = None) -> None:
    """Draw the saved Results page; loading and caching remain in state.py.

    ``outcomes(run_dir, config_id)`` reads one configuration's saved questions
    for the breakdown by question type and difficulty and the before and after view.
    """
    if not runs:
        st.info("No completed benchmark runs were found under results/.")
        return
    run_id = _existing_choice("Run", [run["run_id"] for run in runs],
                              key="results_run", format_func=str)
    selected = next(run for run in runs if run["run_id"] == run_id)
    for warning in report.run_warnings(selected):
        st.warning(warning)
    for tab, (title, kind) in zip(st.tabs([title for title, _ in report.BENCHMARKS]),
                                report.BENCHMARKS):
        with tab:
            matching = report.retrieval_configurations(selected["configurations"], kind)
            saved = _render_results_benchmark(title, matching, selected, kind, outcomes)
            if saved:
                _render_breakdown(selected, kind, outcomes)
                _render_question_changes(selected, kind, outcomes)
                _render_full_metrics(matching, kind)
    # Lead with statistics (#165), then let the saved walkthrough explain
    # an individual answer without pushing the headline tiles off screen.
    _render_worked_example(selected, examples)
    _render_failure_example(selected, failures)


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


class CorpusSelection(NamedTuple):
    """One filing scope selected on the Browse page."""

    ticker: str
    fiscal_year: int
    item: str


def _item_order(item: str) -> tuple[int, str]:
    """Sort SEC Item labels in Form 10-K order, with any unknown label after them."""
    return (_ITEMS.index(item), "") if item in _ITEMS else (len(_ITEMS), item)


def _existing_choice(label: str, options: Sequence, *, key: str, format_func=None,
                     default: Any = None):
    """Draw a required selectbox and discard a stale dependent selection.

    A selection that is not one of ``options`` falls back to ``default``, or to
    the first option when ``default`` is not one either.
    """
    if not options:
        raise ValueError(f"{label} has no options")
    if st.session_state.get(key) not in options:
        st.session_state[key] = default if default in options else options[0]
    return st.selectbox(label, options, key=key, format_func=format_func)


def corpus_picker(passages: Sequence[Mapping[str, Any]], *, key: str = "browse") -> CorpusSelection:
    """Pick a company, fiscal year and Item that actually exist in ``passages``.

    Each choice narrows the choices after it. This prevents the Browse page
    from offering a company/year/Item combination that resolves to an empty
    page, and resets a now-invalid downstream value when an upstream widget
    changes.
    """
    usable = [row for row in passages
              if row.get("ticker") and row.get("fiscal_year") is not None and row.get("item")]
    if not usable:
        raise ValueError("The local corpus has no passages with company, fiscal year and Item metadata")
    if skipped := len(passages) - len(usable):
        st.caption(f"{skipped} of {len(passages)} passages have no company, fiscal year or Item "
                   "and are not listed.")

    names: dict[str, str] = {}
    for row in usable:
        names.setdefault(str(row["ticker"]), str(row.get("company") or row["ticker"]))
    tickers = sorted(names)

    with st.container(border=True):
        st.caption("Choose a filing scope. Each menu contains only values present in the corpus.")
        company_col, year_col, item_col = st.columns(3)
        with company_col:
            ticker = _existing_choice(
                "Company", tickers, key=f"{key}:company",
                format_func=lambda value: f"{value} — {names[value]}")
        years = sorted({int(row["fiscal_year"]) for row in usable if row["ticker"] == ticker},
                       reverse=True)
        with year_col:
            fiscal_year = _existing_choice(
                "Fiscal year", years, key=f"{key}:year",
                format_func=lambda value: f"FY{value}")
        items = sorted({str(row["item"]).upper() for row in usable
                        if row["ticker"] == ticker and row["fiscal_year"] == fiscal_year},
                       key=_item_order)
        with item_col:
            item = _existing_choice(
                "Item", items, key=f"{key}:item",
                format_func=lambda value: f"Item {value}")
    return CorpusSelection(ticker, fiscal_year, item)


def select_corpus_passages(passages: Sequence[Mapping[str, Any]],
                           selection: CorpusSelection) -> list[Mapping[str, Any]]:
    """Return the passages in one selected company/year/Item, in corpus order."""
    return [row for row in passages
            if row.get("ticker") == selection.ticker
            and row.get("fiscal_year") == selection.fiscal_year
            and str(row.get("item") or "").upper() == selection.item]


def _passage_kind(row: Mapping[str, Any]) -> tuple[str, str, str]:
    """The visible label, badge colour and icon for a stored content type."""
    if row.get("content_type") == "table":
        return "Table passage", "orange", ":material/table_chart:"
    return "Prose passage", "blue", ":material/article:"


def _table_label(text: Any) -> str:
    """A table passage's own label line, or "" when it opens with the grid.

    The chunker writes a table as its label, a blank line, then rows that each
    start with "|", so the label is the first line when that line is not a
    row. It is the only thing that tells 61 of Apple's 64 FY2025 Item 8 tables
    apart: none of them has a caption or a heading, so all 61 fall back to the
    Item title and read as the same entry (#127).
    """
    for line in str(text or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        return "" if stripped.startswith("|") else stripped
    return ""


def _passage_heading(row: Mapping[str, Any]) -> str:
    """What a collapsed panel calls a passage, before its marker.

    A table prefers its stored caption, then its own label line, then the
    headings a prose passage would use. 7,232 of the corpus's 10,615 tables
    have no caption, and the label line names 4,884 of those better than the
    Item title does.
    """
    if row.get("content_type") == "table":
        heading = (row.get("table_caption") or _table_label(row.get("text"))
                   or row.get("heading") or row.get("title"))
    else:
        heading = row.get("heading") or row.get("title")
    heading = str(heading or "")
    # The selection heading immediately above the panels already states the
    # Item. Some filings repeat it at the start of their first section heading
    # (often separated by several non-breaking spaces), which produces labels
    # such as "Item 7. Management's Discussion ...". Remove only a prefix that
    # agrees with this passage's own metadata, and keep a heading that consists
    # solely of the Item so the panel never becomes nameless.
    item = str(row.get("item") or "").strip()
    if item:
        without_item = re.sub(
            rf"^\s*Item\s+{re.escape(item)}(?![0-9A-Za-z])\s*[.:–—-]?\s*", "", heading,
            count=1, flags=re.IGNORECASE,
        )
        if without_item:
            heading = without_item
    return heading


_PART_SUFFIX = re.compile(r"\s+\(part\s+\d+\s+of\s+\d+\)\s*$", re.IGNORECASE)


def passage_labels(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return panel names, numbering rows that share the same display name.

    Chunk IDs remain inside expanded panels. When an Item contains several
    chunks with the same heading, their collapsed labels instead identify the
    chunk's position as ``Name (Part x of x)``. Counts cover the full selected
    Item, so numbering remains stable when the result is paginated. A table
    fragment's stored label can already end in ``(part x of x)``; that suffix
    is normalized before counting so a panel never shows two competing part
    counters.
    """
    headings = [_passage_heading(row) or _passage_kind(row)[0] for row in rows]
    names = [_PART_SUFFIX.sub("", heading) for heading in headings]
    totals = Counter(names)
    seen: Counter[str] = Counter()
    labels = []
    for heading, name in zip(headings, names):
        seen[name] += 1
        labels.append(
            f"{name} (Part {seen[name]} of {totals[name]})"
            if totals[name] > 1 else heading
        )
    return labels


_TABLE_SEPARATOR_CELL = re.compile(r":?-{3,}:?")


def _table_cells(line: str) -> list[str]:
    """Split one pipe-delimited table row without its outside delimiters."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


# Characters the SEC table extractor splits off a value into cells of their
# own: a currency sign or opening bracket before the amount, and a percent sign,
# closing bracket or comma after it, sometimes several at once such as ``%)``.
_OPENING_FRAGMENT = "$£€(["
_CLOSING_FRAGMENT = "%)],"


def _is_fragment(cell: str, characters: str = _OPENING_FRAGMENT + _CLOSING_FRAGMENT) -> bool:
    return bool(cell) and not cell.strip(characters)


def _is_one_split_value(cells: Sequence[str]) -> bool:
    """Whether a span's cells are one value split into sign, amount and spacer."""
    return sum(1 for cell in cells if cell and not _is_fragment(cell)) <= 1


def _joined_financial_cell(cells: Sequence[str]) -> str:
    """Join pieces that the SEC table extractor split across a column span."""
    value = ""
    for token in (cell for cell in cells if cell):
        if not value:
            value = token
        elif (_is_fragment(token, _CLOSING_FRAGMENT)
              or _is_fragment(value[-1:], _OPENING_FRAGMENT)):
            value += token
        else:
            value += f" {token}"
    return value


def _split_values(rows: list[list[str]], start: int, stop: int) -> list[tuple[int, int]]:
    """Divide one repeated-header span into its separate value columns.

    Each group is the widest run of columns in which every row holds at most
    one value. A column holding only currency signs or opening brackets belongs
    to the value after it, so it never ends a group.
    """
    groups: list[tuple[int, int]] = []
    while start < stop:
        end = start + 1
        while end < stop and all(_is_one_split_value(row[start:end + 1]) for row in rows):
            end += 1
        while end - 1 > start and end < stop and all(
                not row[end - 1] or _is_fragment(row[end - 1], _OPENING_FRAGMENT)
                for row in rows):
            end -= 1
        groups.append((start, end))
        start = end
    return groups


def _collapse_repeated_columns(header: list[str], rows: list[list[str]]) -> tuple[list[str],
                                                                                  list[list[str]]]:
    """Collapse adjacent copies of a header and join their fragmented values.

    EDGAR uses a header such as ``2025`` with ``colspan=3`` above the currency
    sign, amount and an empty spacer. The parser necessarily repeats that
    header in its rectangular representation. Showing all three columns makes
    the browser table technically faithful but visually misleading, so restore
    the span as one useful value column.

    A repeated header can also span several real columns, such as a year over
    "As Reported", "Exchange Rate Effect" and "At Prior Year Rates". Such a
    span becomes one column per value, each still joined with its own currency
    sign and percent sign, so distinct figures are never joined into one cell.
    """
    groups: list[tuple[int, int]] = []
    start = 0
    while start < len(header):
        stop = start + 1
        if header[start]:
            # Some filers repeat the header across the span and others leave
            # the rest of it blank ("2021" over the "$", nothing over the
            # amount). Both are one span. The label column is left alone.
            while stop < len(header) and (header[stop] == header[start]
                                          or (start and not header[stop])):
                stop += 1
        groups.extend(_split_values(rows, start, stop))
        start = stop

    if all(stop - start == 1 for start, stop in groups):
        return header, rows
    collapsed_header = [header[start] for start, _ in groups]
    collapsed_rows = [
        [_joined_financial_cell(row[start:stop]) for start, stop in groups]
        for row in rows
    ]
    return collapsed_header, collapsed_rows


# Streamlit's default page background and body text for each theme. The sticky
# first column needs both: an opaque background so columns do not show through
# it, and a matching text colour so it stays readable if Streamlit reports the
# theme wrongly, as it can on a session's first load.
_TABLE_THEMES = {False: ("#ffffff", "#31333f"), True: ("#0e1117", "#fafafa")}


def _corpus_table_html(text: str, *, dark: bool = False) -> str:
    """Render a stored pipe table as an accessible, horizontally scrolling grid.

    The parser preserves multiple header rows. SEC tables often use one row for
    years and another for subheadings before the Markdown separator, so treating
    only the last one as a header makes wide financial tables harder to read.
    Every value is escaped before it is put in the HTML. ``dark`` picks the
    sticky first column's colours, which Streamlit exposes no CSS variable for.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    first_row = next((index for index, line in enumerate(lines)
                      if line.startswith("|")), None)
    if first_row is None:
        return ""

    caption = " ".join(lines[:first_row])
    rows = [_table_cells(line) for line in lines[first_row:] if line.startswith("|")]
    separator = next((index for index, row in enumerate(rows)
                      if row and all(_TABLE_SEPARATOR_CELL.fullmatch(cell) for cell in row)), None)
    if separator is None or separator == 0:
        return ""

    header_rows = rows[:separator]
    body_rows = rows[separator + 1:]
    width = max(len(row) for row in (*header_rows, *body_rows))

    # Every table in the current corpus has one rectangular header row. When
    # a future format carries a true multi-row header, preserve it verbatim
    # rather than guessing how its spans relate.
    if len(header_rows) == 1:
        header, body_rows = _collapse_repeated_columns(
            header_rows[0] + [""] * (width - len(header_rows[0])),
            [row + [""] * (width - len(row)) for row in body_rows],
        )
        header_rows = [header]
        width = len(header)

    def padded(row: list[str]) -> list[str]:
        return row + [""] * (width - len(row))

    def html_row(row: list[str], element: str) -> str:
        return "<tr>" + "".join(
            f"<{element}>{escape(cell)}</{element}>" for cell in padded(row)
        ) + "</tr>"

    base, ink = _TABLE_THEMES[dark]
    caption_html = f"<caption>{escape(caption)}</caption>" if caption else ""
    head_html = "".join(html_row(row, "th") for row in header_rows)
    body_html = "".join(html_row(row, "td") for row in body_rows)
    return f"""
<style>
.sec-corpus-table {{
  max-width: 100%;
  overflow-x: auto;
  border: 1px solid color-mix(in srgb, currentColor 18%, transparent);
  border-radius: .5rem;
}}
.sec-corpus-table table {{
  width: max-content;
  min-width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  font-size: .9rem;
  font-variant-numeric: tabular-nums;
}}
.sec-corpus-table caption {{
  padding: .65rem .75rem;
  text-align: left;
  font-weight: 600;
}}
.sec-corpus-table th,
.sec-corpus-table td {{
  padding: .45rem .65rem;
  border-top: 1px solid color-mix(in srgb, currentColor 12%, transparent);
  border-right: 1px solid color-mix(in srgb, currentColor 10%, transparent);
  text-align: right;
  vertical-align: top;
  white-space: nowrap;
}}
.sec-corpus-table thead th {{
  max-width: 18rem;
  white-space: normal;
  background: color-mix(in srgb, currentColor 8%, transparent);
  font-weight: 600;
}}
.sec-corpus-table th:first-child,
.sec-corpus-table td:first-child {{
  position: sticky;
  left: 0;
  z-index: 1;
  max-width: 24rem;
  text-align: left;
  white-space: normal;
  /* A sticky cell must be opaque or the columns scroll visibly beneath it. */
  background: var(--sec-table-base);
  color: var(--sec-table-ink);
}}
.sec-corpus-table thead th:first-child {{
  background: linear-gradient(color-mix(in srgb, currentColor 8%, transparent),
                              color-mix(in srgb, currentColor 8%, transparent)),
              var(--sec-table-base);
}}
.sec-corpus-table tbody tr:nth-child(even) td {{
  background: color-mix(in srgb, currentColor 4%, transparent);
}}
.sec-corpus-table tbody tr:nth-child(even) td:first-child {{
  background: linear-gradient(color-mix(in srgb, currentColor 4%, transparent),
                              color-mix(in srgb, currentColor 4%, transparent)),
              var(--sec-table-base);
}}
</style>
<div class="sec-corpus-table" style="--sec-table-base: {base}; --sec-table-ink: {ink}" role="region" aria-label="{escape(caption or 'Filing table', quote=True)}" tabindex="0">
  <table>{caption_html}<thead>{head_html}</thead><tbody>{body_html}</tbody></table>
</div>
""".strip()


def _dark_theme() -> bool:
    """Whether the viewer's Streamlit theme is dark, defaulting to light."""
    return getattr(st.context.theme, "type", None) == "dark"


def corpus_passage(row: Mapping[str, Any], *, label: str = "") -> None:
    """Render one full passage with its chunk ID, type and EDGAR filing link.

    ``label`` is the collapsed label, from :func:`passage_labels` over the
    whole selection, so that panels listed together can be told apart. Without
    one, a passage is labelled on its own and may read like its neighbours.
    """
    _, _, icon = _passage_kind(row)
    passage_title = label or passage_labels([row])[0]
    # Streamlit reads a label as Markdown, where a pair of "$" opens inline math.
    passage_title = passage_title.replace("$", r"\$")

    with st.expander(passage_title, icon=icon):
        _passage_body(row)


def _passage_body(row: Mapping[str, Any]) -> None:
    """The shared stored text/table and safe filing link for Browse and Ask."""
    kind, colour, icon = _passage_kind(row)
    chunk_id = str(row.get("chunk_id") or "Chunk ID unavailable")
    with st.container(horizontal=True, vertical_alignment="center"):
        st.badge(kind, color=colour, icon=icon)
        st.caption(f"Chunk ID: {chunk_id}")
    url = str(row.get("url") or "")
    if _safe_url(url):
        st.link_button("Open filing on EDGAR", url, icon=":material/open_in_new:")
    else:
        st.caption("Filing link unavailable")

    text = str(row.get("text") or "")
    if row.get("content_type") == "table":
        # Financial tables are wider than the page. Keep their columns
        # aligned in a real table and scroll sideways instead of wrapping
        # pipe-delimited rows into an unreadable code block.
        table_html = _corpus_table_html(text, dark=_dark_theme())
        if table_html:
            st.html(table_html)
            # Show what retrieval sees: the exact stored text stays one
            # click away from the rendered table.
            with st.expander("Stored text", icon=":material/code:"):
                st.code(text, language=None, wrap_lines=False)
        else:
            # Defensive fallback for an unrecognised future table format;
            # never hide the stored source text.
            st.code(text, language=None, wrap_lines=False)
    else:
        st.text(text)


def corpus_passage_page(passages: Sequence[Mapping[str, Any]], selection: CorpusSelection,
                        *, key: str = "browse", page_size: int = 25) -> None:
    """Render a paginated selection without hiding how many passages exist."""
    if page_size < 1:
        raise ValueError("page_size must be positive")
    selected = select_corpus_passages(passages, selection)
    if not selected:
        # ``corpus_picker`` only makes existing combinations, so this guards a
        # malformed or concurrently replaced corpus rather than normal use.
        st.warning("No passages exist for this selection.", icon=":material/search_off:")
        return

    filings = {str(row.get("accession_no") or row.get("url") or "") for row in selected}
    page_count = (len(selected) + page_size - 1) // page_size
    scope_key, page_key = f"{key}:scope", f"{key}:page"
    scope = tuple(selection)
    if st.session_state.get(scope_key) != scope:
        st.session_state[scope_key] = scope
        st.session_state[page_key] = 1
    if st.session_state.get(page_key) not in range(1, page_count + 1):
        st.session_state[page_key] = 1

    with st.container(horizontal=True, vertical_alignment="bottom"):
        page = st.selectbox(
            "Page", range(1, page_count + 1), key=page_key,
            format_func=lambda value: f"Page {value} of {page_count}",
            width=180,
        )
        start = (page - 1) * page_size
        stop = min(start + page_size, len(selected))
        st.caption(
            f"Showing passages {start + 1}–{stop} of {len(selected)} "
            f"across {len(filings)} filing{'s' if len(filings) != 1 else ''}.")

    # Labelled over the whole selection, not the page, so a marker is unique
    # however the pages are cut.
    labels = passage_labels(selected)
    for row, label in zip(selected[start:stop], labels[start:stop]):
        corpus_passage(row, label=label)


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
        # The deterministic groundedness check can only confirm wording copied
        # from a passage.  A normal paraphrase is therefore ``unverified`` even
        # when its citation resolves and its figures pass the evidence checks.
        # Keep that diagnostic on the Answer for evaluation, but do not present
        # it as an actionable citation warning on the Ask page.
        warnings = [c for c in checks if c.status in {"unverified", "mismatch"}
                    and not (c.kind == "groundedness" and c.status == "unverified")]
        missing = any(int(m[1]) < 1 or not any(
            c.marker == int(m[1]) and c.resolved for c in answer.citations
        ) for m in _MARKER.finditer(text))
        flagged = flagged or missing or not _MARKER.search(text)
        evidence_supported = any(c.status == "supported" for c in checks)
        if any(c.status == "mismatch" for c in warnings):
            status, label = "sec-warning sec-mismatch", "Verification found a mismatch"
        elif flagged or answer.truncated or answer.parse_error:
            status, label = "sec-warning", "Citation needs attention"
        elif warnings:
            status, label = "sec-warning", "Evidence could not be fully verified"
        elif evidence_supported:
            status, label = "sec-supported", "Verification checks passed"
        else:
            status, label = "", None
        status_label = "" if label is None else f'<span class="sec-label">{label}</span>'
        parts.append(f'<div class="sec-claim {status}">{status_label}'
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
    st.warning(f"**No answer from the filings.** {abstention_text(answer.abstention_reason)}",
               icon=":material/do_not_disturb_on:")
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

    A company or year the question named that the corpus does not hold is
    named here too, as the sidebar names it. Without it "every company" stood
    alone under a question about Tesla, as if the question had named none.
    """
    parts = [" ".join(f":blue-badge[{value}]" for value in values) or f":gray-badge[{every}]"
             for values, every in _scope(query)]
    if parsed is not None:
        if missing := parsed.not_in_corpus:
            # Lower case inside the sentence the badges make.
            parts.append(f":orange-badge[{missing[0].lower() + missing[1:]}]")
        parts.append(f":violet-badge[{parsed.question_type} question]")
    st.markdown(":material/filter_alt: Searching " + " ".join(parts))


def configuration_picker(runs: Mapping[str, str], *, key: str = "configuration",
                         label: str = "Configuration", default: str = DEFAULT_STACK) -> str:
    """Pick one of the configurations ``src/stack.py`` can build, in the sidebar.

    In the registry's order, each labelled with the run that measured it
    (``runs``, id to label, from ``state.measured``), so a page can be set to
    the configuration a reported number came from. Returns the id.
    """
    with st.sidebar:
        config_id = st.selectbox(
            label, SELECTABLE,
            index=SELECTABLE.index(default),
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
    SOCLAAS: ("SOCLAAS", "on NUS SoC's SOCLAAS API, with the SOCLAAS_API_KEY in your .env. "
                         "An answer takes seconds.", "Setting up SOCLAAS"),
}


# The rows whose figure questions the facts store can answer with no model
# (#156).
_FACTS_ROWS = tuple(key for key in SELECTABLE if STACKS[key].answers_figures_from_facts)


def provider_picker(available: AnswerModels, *, key: str = "provider") -> str:
    """Pick what writes the answers, in the sidebar: the local model or a hosted one.

    ``available`` is what ``state.answer_models()`` returns, taken whole so
    that a page does not take it apart to hand it over: each provider with
    the model it would answer with, the provider the page opens on, what any
    of them lacks, and what was wrong with ``.env``'s own choice. The choice
    lasts for the browser session and changes nothing in ``.env``.

    Two things are said under it, as soon as they are true and not when Ask
    is pressed: what the provider picked lacks, with what to do about it in
    the app, and what was wrong with ``.env``. A provider that lacks
    something can still be picked: on a row with the metadata filter and no
    Item filter, a figure the facts store looks up asks no model. Returns the
    provider picked.
    """
    models, unready = available.models, available.unready
    with st.sidebar:
        provider = st.segmented_control(
            "Answer model", list(models), default=available.default, required=True,
            format_func=lambda option: _PROVIDERS[option][0], key=key,
            help="Opens on LLM_PROVIDER from .env, which unset means Ollama. "
                 "LLM_MODEL names the model of the provider .env selects; the "
                 "others answer with their defaults.",
        )
        label, means, setup = _PROVIDERS[provider]
        st.caption(f"{models[provider]}, {means}")
        if provider in unready:
            # Short, and with no icon: the sidebar is narrow, and a warning
            # that ran to twenty lines pushed the Configuration box below it
            # off the screen.
            st.warning(f"{label} cannot write an answer yet: {unready[provider]}. "
                       f"{_fix_in_app(provider, available)} "
                       f"On {' or '.join(_FACTS_ROWS)} with no Item filter, a figure "
                       "the facts store holds is still answered.")
        if available.problem:
            st.warning(available.problem)
    return provider


def _fix_in_app(provider: str, available: AnswerModels) -> str:
    """What to do in the app about a provider's settings: change them and
    restart, or pick a provider that lacks nothing.

    The environment as well as ``.env``: a variable exported in the shell wins
    over the file, so advice to fix ``.env`` alone could be followed to no
    effect.
    """
    ready = [_PROVIDERS[other][0] for other in available.models
             if other != provider and other not in available.unready]
    instead = f", or pick {' or '.join(ready)}" if ready else ""
    return (f"Update it in your environment or `.env` (README, {_PROVIDERS[provider][2]}), "
            f"then restart the app{instead}.")


def provider_message(error: ProviderUnavailable, provider: str, available: AnswerModels) -> str:
    """Explain a provider failure with app-specific advice, as plain data.

    ``available`` is what :func:`provider_picker` was given. What the provider
    lacks, without the advice to restart a notebook or a command that the
    full message goes on to give. Where the sidebar has not said what to do in
    the app, because the settings looked right until a request was sent, as
    a key the API refuses does, the error says it. A failure that names no
    reason of its own, such as a rate limit, says what to do itself, and
    nothing in ``.env`` would fix it, so its whole message is shown.
    """
    shown = error.reason
    if error.reason != str(error) and provider not in available.unready:
        shown += f". {_fix_in_app(provider, available)}"
    return shown


def provider_error(error: ProviderUnavailable, provider: str, available: AnswerModels) -> None:
    """Draw the shared provider failure explanation on an answer page."""
    st.error(provider_message(error, provider, available), icon=":material/error:")


def _writer(answer: Answer) -> Literal["facts", "model"] | None:
    """What wrote ``answer``: the facts store, the model picked, or None where nothing was asked to.

    Asked here and nowhere else. The row above an answer and the two halves
    of the trace's writing step each used to ask it in words of their own:
    the provider's name, ``answer.passages``, ``answer.passages or not
    answer.abstained``. They agree on every answer the pipeline makes, and
    would have stopped agreeing the day one of them was changed alone.

    A model is asked whenever a passage was found to ask it about. One that
    was asked and declined to answer is still "model".
    """
    if answer.config.provider == FACTS_PROVIDER:
        return "facts"
    return "model" if answer.passages else None


def _written_by(answer: Answer) -> str:
    """Who wrote the answer, for the row above it: the facts store, or a provider's model."""
    if _writer(answer) == "facts":
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
        if _writer(answer) is not None:
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


def _passage_ids(answer: Answer | None) -> set[str]:
    """Distinct chunk identities, regardless of rank, score or passage text."""
    return {passage.chunk_id for passage in answer.passages} if answer is not None else set()


def comparison_overlap(left: Answer, right: Answer) -> tuple[int, int, int]:
    """Counts of shared, left-only and right-only passages by chunk identity."""
    left_ids, right_ids = _passage_ids(left), _passage_ids(right)
    return len(left_ids & right_ids), len(left_ids - right_ids), len(right_ids - left_ids)


def comparison_rows(answer: Answer, other: Answer | None,
                    side: Literal["left", "right"]) -> list[dict[str, object]]:
    """Mark overlap by chunk identity, preserving each answer's citation order."""
    other_ids = _passage_ids(other)
    return [{"Presence": ("Not compared" if other is None else
                           "Shared" if row["Chunk"] in other_ids else f"Only {side}"), **row}
            for row in trace_rows(answer)]


def comparison_evidence(answer: Answer, other: Answer | None, *,
                        side: Literal["left", "right"]) -> None:
    """Show one side's passages, marked Shared or Only ``side`` against ``other``."""
    st.subheader("Retrieved passages", anchor=False)
    rows = comparison_rows(answer, other, side)
    if not rows:
        st.caption("No passages were retrieved for this answer.")
        return
    st.dataframe(rows, width="stretch", hide_index=True,
                 column_config={"Filing": st.column_config.LinkColumn(display_text="Open filing"),
                                "Score": st.column_config.NumberColumn(format="%.4f"),
                                "Fiscal year": st.column_config.NumberColumn(format="%d")})
    for row in rows:
        with st.expander(f"{row['Presence']} · {row['Source']} · {row['Company']} · "
                         f"FY{row['Fiscal year']} · Item {row['Item']}"):
            st.text(str(row["Passage"]))


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
    writer = _writer(answer)
    if writer == "facts":
        return "Answered from the facts store, with no model"
    if writer is None:
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
    config, writer = answer.config, _writer(answer)
    took = None if answer.latency_ms is None else f"{answer.latency_ms / 1000:.2f}s"
    if writer == "facts":
        lines = [f"Looked up in the facts store ({config.model}), with no model asked."]
        return lines + ([f"The lookup took {took}"] if took else [])
    if writer is None:
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


def retrieved_evidence(passages: Sequence[RetrievedPassage]) -> None:
    """Show the evidence as soon as it is admitted, before generation (#42).

    Nothing is drawn for no passages: the answer's own notice says why, and
    a refused question was never searched.
    """
    if not passages:
        return
    st.subheader("Retrieved evidence", anchor=False)
    for number, passage in enumerate(passages, 1):
        year = passage.fiscal_year if passage.fiscal_year is not None else "unknown"
        label = (f"[{number}] {passage.ticker} · FY{year} · "
                 f"Item {passage.item or 'unknown'} · score {passage.score:.4f}")
        with st.expander(label):
            _passage_body(asdict(passage))


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
            st.caption(abstention_text(answer.abstention_reason))

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
        if parsed and parsed.not_in_corpus:
            st.warning(parsed.not_in_corpus)
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
    "comparison_evidence", "comparison_overlap", "comparison_rows",
    "configuration_picker", "corpus_passage", "corpus_passage_page", "corpus_picker",
    "CorpusSelection", "filter_sidebar", "passage_labels", "provider_error", "provider_message",
    "provider_picker",
    "resolved_filters", "retrieval_trace", "select_corpus_passages", "trace_rows",
]
