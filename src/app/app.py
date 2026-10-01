"""Ask the local filing index: python -m streamlit run src/app/app.py."""

from __future__ import annotations

from dataclasses import replace

import streamlit as st

from src.app.components import answer_card, filter_sidebar
from src.rag.answer import answer_question
from src.rag.generate import ProviderUnavailable, config_from_env
from src.rag.query import parse_question
from src.rag.verify import verify_answer
from src.retrieval.bm25 import BM25Retriever
from src.retrieval.dense import DenseRetriever
from src.retrieval.hybrid import HybridRetriever


@st.cache_resource(show_spinner="Checking the local filing indexes…")
def load_retriever(method: str):
    """Verify indexes against the local corpus once per Streamlit process."""
    bm25 = BM25Retriever.load()
    if method == "BM25":
        return bm25
    if method == "Hybrid":
        return HybridRetriever(bm25, DenseRetriever.load())
    raise ValueError(f"Unknown retrieval method: {method}")


def main() -> None:
    st.set_page_config(page_title="SEC Filing Assistant", page_icon="📑")
    st.title("SEC Filing Assistant")
    st.caption("Ask questions about the 10-K filings in your local corpus. "
               "Answers cite the filing passages used to generate them.")

    question = st.text_input("Question", placeholder="What did Apple report about revenue in FY2024?")
    parsed = parse_question(question, facts_file=None) if question.strip() else None
    query = filter_sidebar(question, parsed=parsed)
    with st.sidebar:
        method = st.selectbox("Retrieval method", ("BM25", "Hybrid"),
                              help="Hybrid also loads the local dense index and embedding model.")
        st.caption("Uses local processed filings and indexes. The answer model "
                   "comes from LLM_PROVIDER / LLM_MODEL in .env (Mistral configured).")

    request = (question.strip(), query.tickers, query.fiscal_years, query.items, method)
    if st.button("Ask", type="primary", disabled=not question.strip()):
        try:
            with st.spinner("Searching filings and generating an answer…"):
                retriever = load_retriever(method)
                answer = answer_question(question.strip(), retriever, query=query,
                                         parsed=parsed, config=config_from_env())
                answer = verify_answer(answer, parsed=parsed)
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
