"""Ask the local filing index: python -m streamlit run src/app/app.py."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from src.app.components import answer_card, filter_sidebar
from src.rag.generate import ProviderUnavailable
from src.rag.query import parse_question
from src.rag.verify import verify_answer
from src.stack import DEFAULT_STACK, SELECTABLE, STACKS, build_stack, measured_runs


@st.cache_resource
def _indexes() -> dict:
    """The BM25 and dense retrievers every configuration shares, once per process.

    Without it, switching from C4 to C3 -- the same hybrid retriever -- read
    both indexes again and loaded a second copy of the embedding model.
    """
    return {}


@st.cache_resource(show_spinner="Checking the local filing indexes…")
def load_stack(config_id: str):
    """Assemble a named configuration once per Streamlit process (#43).

    The same call the evaluation command makes, so what the demo shows is the
    system the numbers in ``results/`` describe, rather than a fourth stack
    assembled here. Cached on the id because building one loads the indexes
    and, for the dense and hybrid rows, the embedding model.
    """
    return build_stack(config_id, parts=_indexes())


@st.cache_data(show_spinner=False)
def measured() -> list[tuple[str, str]]:
    """Each configuration measured under ``results/``, as (id, label).

    Read from the result files rather than listed here, so a configuration is
    offered because a run measured it. Several runs can measure the same row;
    the most recent run's label is the one shown.
    """
    seen: dict[str, str] = {}
    for run in measured_runs():
        seen.setdefault(run.config_id, run.label)
    return list(seen.items())


def main() -> None:
    st.set_page_config(page_title="SEC Filing Assistant", page_icon="📑")
    st.title("SEC Filing Assistant")
    st.caption("Ask questions about the 10-K filings in your local corpus. "
               "Answers cite the filing passages used to generate them.")

    question = st.text_input("Question", placeholder="What did Apple report about revenue in FY2024?")
    # Read with the facts store's labels as cues, as answer_question and the
    # evaluation harness read a question. Read without them, a question that
    # names a line item by its label ("What was Apple's gross profit in
    # FY2024?") was taken for prose and searched with no lean toward tables.
    parsed = parse_question(question) if question.strip() else None
    query = filter_sidebar(question, parsed=parsed)
    with st.sidebar:
        # In the registry's order, each labelled with the run that measured it,
        # so the demo can be set to the configuration a reported number came from.
        runs = dict(measured())
        config_id = st.selectbox(
            "Configuration", SELECTABLE,
            index=SELECTABLE.index(DEFAULT_STACK),
            format_func=lambda key: runs.get(key, f"{key} — {STACKS[key].name}"),
            help="Named in src/stack.py and built the same way the evaluation "
                 "command builds it. A row measured under results/ is labelled "
                 "with the run that measured it.",
        )
        if config_id not in runs:
            st.caption("No run under results/ has measured this configuration yet.")
        if not STACKS[config_id].metadata_filter:
            st.caption("This configuration was measured without the metadata filter, so "
                       "it searches every filing and the filters above are not applied.")
        st.caption("Uses local processed filings and indexes. The answer model "
                   "comes from LLM_PROVIDER / LLM_MODEL in .env.")

    request = (question.strip(), query.tickers, query.fiscal_years, query.items, config_id)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching filings and generating an answer…"):
                stack = load_stack(config_id)
                answer = stack.answer(question.strip(), query=query, parsed=parsed)
                answer = verify_answer(answer, parsed=replace(
                    parsed, tickers=query.tickers, fiscal_years=query.fiscal_years))
        except (FileNotFoundError, ValueError, ProviderUnavailable) as error:
            st.error(str(error))
        else:
            st.session_state["filing_answer"] = answer
            st.session_state["filing_request"] = request

    if st.session_state.get("filing_request") == request:
        answer = st.session_state.get("filing_answer")
        if answer is not None:
            answer_card(answer, key="filing-answer")
    elif "filing_answer" in st.session_state:
        st.info("The question or filters changed. Select Ask to refresh the answer.")


if __name__ == "__main__":
    main()
