"""The Results page: saved benchmark runs, split by benchmark source (#41)."""

import streamlit as st

from src.app import state
from src.app.components import render_results

st.header("Results")
st.caption("Saved benchmark results are read from results/<run-id>/; no evaluation is rerun.")
if st.button("Reload saved results"):
    state.result_runs.clear()
    state.question_outcomes.clear()
    state.worked_examples.clear()
    state.failure_examples.clear()
    st.rerun()

render_results(state.result_runs(), outcomes=state.question_outcomes,
               examples=state.worked_examples, failures=state.failure_examples)
