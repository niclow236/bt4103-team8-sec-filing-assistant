"""The app's entry point: streamlit run src/app/main.py.

This file is the frame around the pages and nothing else: it makes the
project importable, names the app, and lists the pages. What a page shows is
in ``app_pages/``, what the app keeps between reruns is in ``state.py``, and
what several pages draw the same way is in ``components.py``. A new page is
one more ``st.Page`` in ``PAGES``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# ``streamlit run src/app/main.py`` puts this file's folder on sys.path and
# not the project root, so ``import src`` failed unless the app was started as
# ``python -m streamlit run``, which adds the working directory itself.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Paths are relative to this file. With one page Streamlit shows no menu; the
# second page added here makes it appear.
PAGES = [
    st.Page("app_pages/ask.py", title="Ask", icon=":material/search:", default=True),
]

st.set_page_config(page_title="SEC Filing Assistant", page_icon=":material/find_in_page:")
st.navigation(PAGES).run()
