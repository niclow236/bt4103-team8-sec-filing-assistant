"""The Browse page: read the passages available in the local corpus (#40)."""

import streamlit as st

from src.app import state
from src.app.components import corpus_passage_page, corpus_picker


st.title("Browse filing corpus")
st.caption(
    "Choose a company, fiscal year and SEC Item to inspect the exact local passages "
    "available to retrieval. This page uses no answer model or API.")

try:
    passages = state.corpus_passages()
except (OSError, ValueError, TypeError) as error:
    st.error(f"Could not read the processed filing corpus: {error}", icon=":material/error:")
else:
    if not passages:
        st.warning(
            "No processed passages were found. Run `python -m src.pipeline chunk`, "
            "then reload this page.",
            icon=":material/folder_off:")
    else:
        try:
            selection = corpus_picker(passages)
        except ValueError as error:
            st.warning(str(error), icon=":material/data_alert:")
        else:
            st.subheader(
                f"{selection.ticker} · FY{selection.fiscal_year} · Item {selection.item}",
                anchor=False)
            corpus_passage_page(passages, selection)
