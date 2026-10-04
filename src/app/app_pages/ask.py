"""The Ask page: a question, the filters it resolved to, an answer and its trace (#38).

A page is a script, run from ``src/app/main.py`` on every rerun. It decides
what is asked and in what order the page is drawn; it loads through
``src.app.state`` and draws with ``src.app.components``.
"""

import streamlit as st

from src.app import state
from src.app.components import (
    answer_card,
    answer_summary,
    configuration_picker,
    filter_sidebar,
    provider_picker,
    resolved_filters,
    retrieval_trace,
)
from src.rag.generate import ProviderUnavailable
from src.rag.query import parse_question
from src.rag.verify import verify_answer
from src.stack import STACKS

PAGE = "ask"

st.title("SEC Filing Assistant")
st.caption("Ask about the 10-K filings in the local corpus. "
           "Every answer cites the filing passages it was written from.")

with st.container(horizontal=True, vertical_alignment="bottom"):
    question = st.text_input(
        "Question", placeholder="What was Apple's total revenue in FY2024?", key="ask:question")
    # Never disabled: a click made while the box still holds uncommitted text
    # commits the text and asks in one rerun, and a disabled button lost it.
    asked = st.button("Ask", type="primary", icon=":material/search:")
question = question.strip()

# Read with the facts store's labels as cues, as answer_question and the
# evaluation harness read a question. Read without them, a question that
# names a line item by its label ("What was Apple's gross profit in
# FY2024?") was taken for prose and searched with no lean toward tables.
parsed = parse_question(question) if question else None
query = filter_sidebar(question, parsed=parsed)
available = state.answer_models()
provider = provider_picker(available.models, available.default, problem=available.problem)
config_id = configuration_picker(dict(state.measured()))
config = STACKS[config_id]

# The question as read, in the scope the sidebar ends up with: what the
# answer is checked against and what the trace describes.
scoped = parsed.scoped_to(query) if parsed is not None else None
request = (question, query.tickers, query.fiscal_years, query.items, config_id, provider)

if question:
    # Shown before anything is searched, so the reading can be corrected in
    # the sidebar first.
    resolved_filters(config.scoped(query), scoped)
elif asked:
    st.caption("Type a question first.")

if asked and question:
    # The answer as the model writes it, until the checked answer replaces
    # it. Plain text, because a "$" in prose is not the start of a formula.
    draft, written = st.empty(), []

    def show(token: str) -> None:
        written.append(token)
        draft.text("".join(written))

    try:
        with st.spinner("Searching filings and generating an answer…"), state.Stopwatch() as watch:
            answer = state.load_stack(config_id, provider).answer(
                question, query=query, parsed=parsed, on_token=show)
            answer = verify_answer(answer, parsed=scoped)
    except (FileNotFoundError, ValueError, ProviderUnavailable) as error:
        st.error(str(error), icon=":material/error:")
    else:
        state.keep(PAGE, request, answer, watch.seconds)
    # Also where the provider failed part way, so that half an answer is
    # not left on the page above the error.
    draft.empty()

shown = state.kept(PAGE, request)
if shown is not None:
    # Kept with the answer, so the summary is still there when a rerun draws
    # the answer again.
    st.subheader("Answer", anchor=False)
    answer_summary(shown.answer, config, shown.seconds)
    answer_card(shown.answer, key="ask-answer", show_question=False)
    st.subheader("Retrieval trace", anchor=False)
    retrieval_trace(shown.answer, scoped, config, key="ask-trace")
elif state.has_kept(PAGE) and question:
    st.info("The question, the filters or a setting changed. Select Ask to refresh the answer.",
            icon=":material/refresh:")
