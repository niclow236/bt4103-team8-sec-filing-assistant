"""The Results page: saved benchmark runs, split by benchmark source (#41)."""

from src.app import state
from src.app.components import render_results

render_results(state.result_runs())
