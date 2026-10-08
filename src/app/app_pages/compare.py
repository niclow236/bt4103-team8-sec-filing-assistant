"""One question through two registered configurations, side by side (#39)."""

import streamlit as st

from src.app import state
from src.app.components import (
    answer_card, answer_summary, comparison_evidence, configuration_picker,
    filter_sidebar, provider_picker, resolved_filters,
)
from src.rag.generate import ProviderUnavailable
from src.rag.query import parse_question
from src.rag.verify import verify_answer
from src.stack import SELECTABLE, STACKS


st.title("Compare configurations")
st.caption("Ask one question using two configurations with the same answer model. "
           "Shared passages appear on both sides; unique passages are labelled Only left or Only right.")
with st.container(horizontal=True, vertical_alignment="bottom"):
    question = st.text_input("Question", key="compare:question",
                             placeholder="What risks does Apple describe in FY2024?").strip()
    asked = st.button("Compare", type="primary", icon=":material/compare_arrows:")

parsed = parse_question(question) if question else None
query = filter_sidebar(question, parsed=parsed, key="compare:filters")
available = state.answer_models()
provider = provider_picker(available, key="compare:provider")
runs = dict(state.measured())
left_id = configuration_picker(runs, key="compare:left", label="Left configuration",
                               default=SELECTABLE[0])
right_id = configuration_picker(runs, key="compare:right", label="Right configuration")
requests = (state.Request.of(query, left_id, provider), state.Request.of(query, right_id, provider))
configs = (STACKS[left_id], STACKS[right_id])

if left_id == right_id:
    st.info("Choose two different configurations to compare their behavior.")
elif asked and not question:
    st.caption("Type a question first.")

if question:
    for column, config in zip(st.columns(2), configs):
        with column:
            st.subheader(f"{config.id} — {config.name}", anchor=False)
            resolved_filters(config.scoped(query), parsed.scoped_to(config.scoped(query)))
            if not config.metadata_filter:
                st.caption("This configuration searches all filings; company/year filters are not applied.")

if asked and question and left_id != right_id:
    results = []
    for config in configs:
        completed, error = None, None
        with state.Stopwatch() as watch:
            try:
                with st.spinner(f"Running {config.id}…"):
                    completed = state.load_stack(config.id, provider).answer(
                        question, query=query, parsed=parsed)
                    completed = verify_answer(completed, parsed=parsed.scoped_to(config.scoped(query)))
            except ProviderUnavailable as failure:
                completed = None
                error = failure.reason
            except (OSError, ValueError) as failure:
                completed = None
                error = str(failure)
        results.append(state.ComparisonSide(completed, watch.seconds, error))
    state.keep_comparison(requests, tuple(results))

shown = state.kept_comparison(requests) if question and left_id != right_id else None
if shown is not None:
    left, right = (result.answer for result in shown)
    if left is not None and right is not None:
        left_ids = {passage.chunk_id for passage in left.passages}
        right_ids = {passage.chunk_id for passage in right.passages}
        st.caption(f"{len(left_ids & right_ids)} shared passages · "
                   f"{len(left_ids - right_ids)} only left · {len(right_ids - left_ids)} only right")
    st.caption("Timings include loading, answering and verification for each side. "
               "Configurations run in order and can share warmed indexes or cached answers.")
    for column, config, result, other, side in zip(
        st.columns(2), configs, shown, (right, left), ("left", "right")
    ):
        with column:
            st.subheader(f"{config.id} — {config.name}", anchor=False)
            if result.error is not None:
                st.error(result.error)
                st.caption(f"Request stopped after {result.seconds:.2f}s")
            else:
                answer_summary(result.answer, config, result.seconds)
                answer_card(result.answer, key=f"compare:{side}:answer", show_question=False)
                comparison_evidence(result.answer, other, side=side)
elif question and left_id != right_id:
    st.info("Select Compare to run both configurations for the current question and settings.")
