"""Fixed values that tune the data pipeline.

Kept apart from the modules that use them so the scope of the corpus and the
parser's thresholds can be read and changed in one place, and so each later
package (``src/rag``, ``src/evaluation``) can grow its own constants module the
same way.
"""

from __future__ import annotations

from edgar.company_reports import TenK, TenQ

# --- download scope ---------------------------------------------------------
# The corpus recorded in config/companies.txt: annual reports only, over the
# five filing years the brief asks for. Both are overridable per run with
# --forms and --years.
DEFAULT_FORMS = ["10-K"]
DEFAULT_YEARS = (2021, 2025)  # inclusive range of filing years

# --- parsing --------------------------------------------------------------
# The Items the project brief calls out, per form. Everything else in the
# filing is still parsed and stored; this only marks which sections the
# retrieval work is expected to lean on, so the run summary can report on them.
KEY_ITEMS: dict[str, set[str]] = {
    "10-K": {"1", "1A", "7", "7A", "8"},
    "10-Q": {"1", "2", "3"},
}

# Where an Item commonly carries a cross-reference instead of the disclosure
# itself, and which Item actually holds the text. Oracle and NVIDIA both answer
# Item 8 by pointing at the financial statements filed under Item 15.
INCORPORATION_FALLBACKS: dict[str, dict[str, str]] = {
    "10-K": {"8": "15"},
}

# An Item heading that carries a cross-reference instead of the disclosure
# itself runs to a couple of sentences, while a real Item runs to thousands of
# characters. The gap between the two is wide enough that a flat threshold
# separates them reliably.
STUB_CHAR_LIMIT = 500

# The structures edgartools ships already name every Item, so the
# human-readable titles used in citations come from there rather than a second
# list of our own that could drift out of step with it.
FORM_STRUCTURES = {"10-K": TenK.structure, "10-Q": TenQ.structure}
