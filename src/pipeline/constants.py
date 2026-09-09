"""Fixed values that tune the data pipeline.

Kept apart from the modules that use them so the scope of the corpus and the
parser's thresholds can be read and changed in one place, and so each later
package (``src/rag``, ``src/evaluation``) can grow its own constants module the
same way.
"""

from __future__ import annotations

from edgar.company_reports import TenK, TenQ

# --- download scope ---------------------------------------------------------
# The corpus recorded in config/companies.txt: annual reports only, over five
# fiscal years.
#
# The scope is set in FISCAL years rather than filing years, because that is the
# axis every question is asked on: "revenue in FY2024" has to mean the same year
# for all fifteen companies or the answer is not comparable. The two axes do not
# line up. Adobe, Alphabet, Amazon, Meta, ServiceNow and Texas Instruments close
# their books in November or December and file the following January or
# February, so their fiscal 2025 report is a 2026 filing, while Apple's fiscal
# 2025 report is a 2025 filing. Selecting on filing year would mix fiscal 2020
# into one end of the corpus and drop fiscal 2025 from the other, leaving a set
# that looks complete at five filings per company but cannot answer a single
# question across all fifteen.
DEFAULT_FORMS = ["10-K"]
DEFAULT_FISCAL_YEARS = (2021, 2025)  # inclusive, and the scope of the corpus

# EDGAR indexes filings by the date they were filed, so the search window has to
# run one year past the last fiscal year to reach the December filers. Anything
# outside DEFAULT_FISCAL_YEARS is discarded once its period of report has been
# read, which is what keeps the corpus rectangular. Overridable with --years.
DEFAULT_FILING_YEARS = (DEFAULT_FISCAL_YEARS[0], DEFAULT_FISCAL_YEARS[1] + 1)

# How much of a table's HTML to keep when a rebuild fails and the fragment is
# written out for diagnosis. A financial table runs to tens of kilobytes of
# markup, and the shape of the problem -- a merged cell, a spacer row, a header
# split across two rows -- shows up near the top of it.
TABLE_FRAGMENT_CHARS = 20_000

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

# An Item that carries a cross-reference instead of the disclosure itself, or
# that is genuinely empty, holds nothing worth retrieving. Length alone is not
# enough to tell those apart from an Item that is simply short: Apple answers
# Item 2 Properties in 488 characters of real fact, while Item 12 uses 303
# characters to point at the proxy statement. So two tests are used together.
#
# Below EMPTY_ITEM_CHAR_LIMIT there is no room for content at all: these are the
# "None." and "Not applicable." Items. Between that and STUB_CHAR_LIMIT, an Item
# is only treated as a stub when it actually reads as a cross-reference.
EMPTY_ITEM_CHAR_LIMIT = 200
STUB_CHAR_LIMIT = 500

# The phrases filers use to say "the answer is somewhere else". Matched only
# against short Items, so a long Item that happens to cite a note in passing is
# not mistaken for a cross-reference.
INCORPORATION_PHRASES = [
    r"incorporated\s+(herein\s+)?by\s+reference",
    r"information\s+required\s+by\s+this\s+item",
    r"proxy\s+statement",
    r"see\s+(the\s+)?(consolidated\s+)?financial\s+statements",
    r"reference\s+is\s+made\s+to",
    r"set\s+forth\s+under",
]

# Items some filers use that are not in the standard structure edgartools ships,
# so a citation would otherwise have no title. Item 4A is where several filers
# put their executive officer list.
EXTRA_ITEM_TITLES: dict[str, dict[str, str]] = {
    "10-K": {"4A": "Executive Officers of the Registrant"},
}

# The structures edgartools ships already name every Item, so the
# human-readable titles used in citations come from there rather than a second
# list of our own that could drift out of step with it.
FORM_STRUCTURES = {"10-K": TenK.structure, "10-Q": TenQ.structure}

# --- chunking ---------------------------------------------------------------
# Chunk size is measured in characters rather than tokens because the embedding
# model is not chosen yet, and pulling in a tokenizer now would tie the budget
# to one vendor's idea of a token. English prose runs at roughly four characters
# per token, so 4,000 characters is about 1,000 tokens, which fits inside every
# embedding model under consideration. Each chunk records its own n_chars, so a
# true token count can be added later without re-chunking the corpus.
CHUNK_CHAR_BUDGET = 4000

# Whole paragraphs are carried from the end of one chunk into the start of the
# next, so a point made across a paragraph boundary is still retrievable. This
# is the budget for that carried tail rather than an exact overlap, since only
# whole paragraphs are moved.
CHUNK_CHAR_OVERLAP = 600

# A passage is only closed once it holds this much, so a paragraph that is
# bigger than the whole budget joins the passage in front of it rather than
# stranding it. Without this, an Item whose heading is followed by an oversized
# paragraph emits the heading on its own as a passage nothing can retrieve. A
# section shorter than this in total still yields one short passage, which is
# correct: that is the entire Item.
CHUNK_CHAR_MINIMUM = 500

# A section the filing does not number is not citable as an Item, so it is not
# indexed. In practice this is the signature block, which carries no answerable
# content anyway.
SKIP_UNNUMBERED_SECTIONS = True

# The extractor ends a paragraph at a page boundary, which sometimes lands mid
# sentence. Where one block ends without closing punctuation and the next opens
# lowercase, the two are rejoined before packing, so a passage does not begin
# halfway through a sentence.
REJOIN_PAGE_BREAK_SPLITS = True

# A short block that introduces a longer one is treated as a heading and carried
# onto the chunks beneath it, so a passage taken from the middle of Item 1A
# still knows which risk it sits under. Requiring a body underneath is what
# keeps table row labels such as "Total" out, since those are followed by a
# figure rather than by prose.
HEADING_CHAR_LIMIT = 120
HEADING_BODY_MINIMUM = 200

# Navigation the extractor leaves inline in the text. Company names are
# deliberately not listed here: in Item 8 they head the financial statements
# rather than a page, so dropping them would lose real structure.
FURNITURE_PATTERNS = [
    r"^Table of Contents$",
    r"^[_\-\u2014\s]{3,}$",
    r"\bForm 10-[KQ]\s*\|\s*\d{1,4}$",
]

# A block that is nothing but a number is a page number when it sits between
# paragraphs, but a real figure when it sits in a financial statement, so it is
# only dropped from sections that hold no tables.
PAGE_NUMBER_PATTERN = r"^\d{1,4}$"

# --- tables -----------------------------------------------------------------
# Financial tables are kept as tables rather than flattened into the surrounding
# prose. Flattening detaches every figure from its row and column label, which
# turns "Total revenue, 2024, 245,122" into a bare run of numbers that no reader,
# human or model, can attribute. Each table is stored as its own record and
# chunked on its own, so a passage of figures always arrives with its headers.
#
# A table longer than this is split into several passages, and the header row is
# repeated on each one so no slice is left without its column labels.
TABLE_MAX_ROWS_PER_CHUNK = 30

# Below this a table carries no information worth indexing: a layout table used
# for spacing, or a single stray cell.
TABLE_MIN_CELLS = 4
