"""Settings for the RAG stage, in one place so they are set once and read everywhere.

The tables here are what ``query.py`` reads a question against. They are
tables rather than code because the corpus's scope is a list -- fifteen
companies, five fiscal years -- and the names people use for those companies
are a list too, and a list is easier to check, extend and argue about than a
regular expression that happens to match it.
"""

from __future__ import annotations

# --- companies --------------------------------------------------------------
# The names a question might use for each company in config/companies.txt,
# keyed by ticker. Matched case-insensitively as whole words, so "apple" and
# "Apple's" both resolve to AAPL and "pineapple" does not.
#
# Only the tickers in companies.txt are resolved, whatever this table holds:
# ``query.py`` intersects the two, so a company that is dropped from the scope
# stops resolving without this table having to be edited in step. The reverse
# is not true -- a company added to the scope needs a row here, or it is only
# ever found by its ticker.
#
# The ticker itself is matched separately, in upper case only, so that "now"
# in "how is revenue recognised now" does not resolve to ServiceNow. That
# still leaves the tickers that are also words in their own field: "CRM" in
# "which companies sell CRM software" resolves to Salesforce. It is the wrong
# reading of that question, and the app shows the filter back to the user
# precisely so a wrong reading is visible rather than silent.
COMPANY_ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": ("apple",),
    "MSFT": ("microsoft",),
    "AVGO": ("broadcom",),
    "GOOGL": ("alphabet", "google"),
    "META": ("meta", "meta platforms", "facebook"),
    "AMZN": ("amazon", "aws", "amazon web services"),
    "ORCL": ("oracle",),
    "CRM": ("salesforce",),
    "ADBE": ("adobe",),
    "CSCO": ("cisco", "cisco systems"),
    "TXN": ("texas instruments",),
    "MU": ("micron", "micron technology"),
    "INTU": ("intuit",),
    "NOW": ("servicenow", "service now"),
    "PANW": ("palo alto", "palo alto networks"),
}

# Companies a question about this industry is likely to name that the corpus
# does not hold. companies.txt records why each was dropped. A mention of one
# is reported back as unresolved rather than ignored, so the app can say
# "Intel is not in the corpus" instead of answering about Intel from whatever
# fifteen other filings say about it. A question that names only these is
# unanswerable from the corpus, and is classified as such.
#
# Deliberately short: this is not an attempt to recognise every company in the
# world, only the ones a user of a large-cap tech corpus is likely to reach for.
# An unlisted name is simply not seen, and the search runs unfiltered, which is
# what an unresolved entity degrades to anyway.
OUT_OF_SCOPE_ALIASES: dict[str, tuple[str, ...]] = {
    "Intel": ("intel",),
    "IBM": ("ibm", "international business machines"),
    "NVIDIA": ("nvidia",),
    "AMD": ("amd", "advanced micro devices"),
    "Qualcomm": ("qualcomm",),
    "Applied Materials": ("applied materials",),
    "Netflix": ("netflix",),
    "Tesla": ("tesla",),
}

# --- question types ---------------------------------------------------------
# The five classes the RAG engine and the benchmark agree on. One label per
# question, chosen by the first rule that fires in this order, because a
# question can carry several signals and the engine needs one route:
#
# - unanswerable: names only companies or years the corpus does not hold, or
#   asks for something a 10-K cannot say (a forecast, investment advice).
# - comparative: sets two or more companies against each other.
# - temporal: asks how something moved across years, or names two or more.
# - numeric: asks for a figure.
# - factual: everything else -- what a filing says, in prose.
#
# Comparative outranks temporal so that "compare Apple and Microsoft in 2023
# and 2024" is routed as a comparison across companies, which is the harder
# retrieval problem: it needs passages from two filings, and a temporal read
# would filter to years and let one company crowd out the other.
QUESTION_TYPES = ("factual", "comparative", "temporal", "numeric", "unanswerable")

# Phrases that set two things against each other. Whole-word, lower case.
COMPARATIVE_CUES = (
    "compare", "comparison", "compared", "versus", "vs", "vs.", "relative to",
    "difference between", "differ", "higher than", "lower than", "more than",
    "less than", "outperform", "which company", "which of", "among", "between",
)

# Phrases that ask about movement across time.
TEMPORAL_CUES = (
    "change", "changed", "changes", "changing", "trend", "trends", "over time",
    "grow", "grew", "grown", "growing", "growth", "increase", "increased",
    "decrease", "decreased", "decline", "declined", "rise", "risen", "rose",
    "fall", "fell", "fallen", "shrink", "shrunk", "year over year",
    "year-over-year", "yoy", "since", "evolve", "evolved", "from year to year",
    "each year", "annually", "historically",
)

# Phrases that ask for a figure. Two kinds: the shape of the question ("how
# much") and the line items a 10-K reports as numbers. The latter is what
# makes "what was Apple's revenue" numeric even without "how much".
NUMERIC_CUES = (
    "how much", "how many", "what percentage", "what percent", "what was the total",
    "revenue", "revenues", "net income", "net loss", "operating income",
    "gross margin", "operating margin", "profit margin", "margin", "margins",
    "earnings", "eps", "earnings per share", "cash flow", "free cash flow",
    "capital expenditure", "capex", "r&d", "research and development",
    "operating expenses", "cost of revenue", "cost of sales", "total assets",
    "total liabilities", "debt", "long-term debt", "shares outstanding",
    "headcount", "employees", "dividend", "dividends", "buyback", "repurchase",
    "tax rate", "effective tax", "deferred revenue", "remaining performance",
    "backlog", "goodwill", "impairment", "amount", "figure", "in dollars",
    "in billions", "in millions",
)

# Phrases a 10-K cannot answer: it reports the past, and it is not advice. A
# question built on one of these is unanswerable from the corpus however well
# the retrieval goes, and the engine should abstain rather than guess. "will"
# is not here on purpose: "what risks did Apple say will affect supply" is a
# question the filing answers, so a cue that broad would refuse real questions.
UNANSWERABLE_CUES = (
    "forecast", "predict", "prediction", "next year", "next quarter",
    "should i buy", "should i invest", "should i sell", "good investment",
    "stock price", "share price", "price target", "recommend",
)

# Substrings that show a year is being named. A bare four-digit number is also
# read as a year when it falls inside the corpus's range, which is what "revenue
# in 2024" needs; the prefixes are what let a two-digit year through, since
# "24" on its own is a number and "FY24" is not.
FISCAL_PREFIXES = ("fy", "fiscal year", "fiscal", "fye")
