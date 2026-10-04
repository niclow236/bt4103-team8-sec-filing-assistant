"""The app's entry point: streamlit run src/app/main.py.

This file is the frame around the pages and nothing else: it makes the
project importable, names the app, lists the pages, and keeps Streamlit's
file watcher from setting off a library's lazy imports. What a page shows is
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


def _settle_lazy_aliases() -> None:
    """Give each of transformers' alias modules a ``__path__``, so reading one imports nothing.

    transformers 5 keeps an alias in ``sys.modules`` for every module name it
    retired, 221 of them, and an alias answers for any attribute it lacks by
    importing the module it stands for. After each run Streamlit's file
    watcher asks every loaded module for ``__path__``. So once the embedding
    model had been loaded, the watcher imported every image processor
    transformers ships. The 102 that need torchvision, which this project
    does not install, each failed, and the watcher logged a traceback for
    each: some three hundred "Examining the path of ..." warnings in the
    terminal after the first question, with the app working all the while.

    An alias is known by what transformers gives it and no other module has:
    a ``__file__`` of None and a ``__getattr__`` of its own. Both are read
    from the module's own namespace, because ``getattr`` is what sets the
    import off. The watcher takes an empty ``__path__`` for an answer.
    """
    for name, module in list(sys.modules.items()):
        if not name.startswith("transformers."):
            continue
        own = getattr(module, "__dict__", {})
        if own.get("__file__", "") is None and "__getattr__" in own and "__path__" not in own:
            module.__path__ = []


st.set_page_config(page_title="SEC Filing Assistant", page_icon=":material/find_in_page:")
try:
    st.navigation(PAGES).run()
finally:
    # After the page, which is where the embedding model is first loaded, and
    # before the run ends, which is when the watcher looks. Also when the page
    # stopped early or asked for a rerun.
    _settle_lazy_aliases()
