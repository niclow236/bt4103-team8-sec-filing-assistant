"""Settings for the RAG stage, in one place so they are set once and read everywhere.

The tables here are what ``query.py`` reads a question against. They are
tables rather than code because the corpus's scope is a list -- fifteen
companies, five fiscal years -- and the names people use for those companies
are a list too, and a list is easier to check, extend and argue about than a
regular expression that happens to match it.

The prompt template at the end is what ``prompt.py`` renders. It is here
rather than in the builder for the reason ``GenerationConfig`` records a
template id and not the rendered text: an answer says which template made
it, and the id only means something if the template it names is one fixed
thing in one place, and changes by getting a new id.
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
# "which companies sell CRM software" resolves to Salesforce, and so does
# "NOW" in a question typed entirely in capitals. It is the wrong reading of
# that question, and the app shows the filter back to the user precisely so
# a wrong reading is visible rather than silent.
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

# --- what a 10-K cannot answer ----------------------------------------------
# A 10-K reports the past and gives no advice, so a question is unanswerable
# when it asks for advice, asks for a new prediction, or asks for something
# the filing does not carry -- a current price. What it is NOT is a question
# that merely mentions one of those things: "what risks did Apple disclose
# about its stock price" is answered by Item 1A, and "what did Apple say
# about forecasting risk" by Item 7. So a topic noun is never enough on its
# own; the question has to be asking for the thing rather than asking what
# the filing said about it. ``query.py`` combines these four tables to make
# that call, and the rule is written out there.

# Requests for advice. Always unanswerable: no reading of the filing turns
# "should I buy" into a question about what it says.
ADVICE_CUES = (
    "should i buy", "should i invest", "should i sell", "should i hold",
    "good investment", "worth buying", "worth investing", "do you recommend",
    "would you recommend",
)

# Verbs that, used as a request, ask the engine to produce a prediction. A
# request is the verb at the start of the question, after a clause break, or
# after "can you"/"please" -- so "predict next quarter's revenue" and "based
# on what Apple disclosed, predict ..." both count, while "what does Apple
# predict for its supply chain" asks what the filing predicts and does not.
PREDICTION_VERBS = ("predict", "forecast", "project", "estimate", "extrapolate", "guess")

# Nouns for things a 10-K does not carry. Unanswerable when asked for
# directly, and a topic like any other when the question asks what the filing
# said about them.
BEYOND_FILING_NOUNS = (
    "stock price", "share price", "price target", "market cap", "market capitalisation",
    "market capitalization", "current price", "today's price",
)

# Phrases that put the question in the future. Unanswerable on their own,
# since the corpus ends at FY2025, but not when the question asks what the
# filing said about the future: "what did Apple say it expects next year"
# is a question about Item 7.
FUTURE_CUES = (
    "next year", "next quarter", "next fiscal year", "going forward",
    "in the future", "in the coming year", "in the coming years",
)

# Verbs that make a question about what the filing says rather than about
# the thing itself. Any of these turns a topic noun or a future phrase back
# into an ordinary question. They do NOT excuse an advice request or an
# imperative prediction: "based on what Apple disclosed, predict ..." still
# asks for a prediction.
REPORTING_VERBS = (
    "disclose", "disclosed", "discloses", "disclosure", "disclosures",
    "report", "reported", "reports", "say", "said", "says", "state", "stated",
    "states", "mention", "mentioned", "mentions", "describe", "described",
    "describes", "discuss", "discussed", "discusses", "note", "noted", "notes",
    "warn", "warned", "warns", "expect", "expected", "expects", "anticipate",
    "anticipated", "anticipates", "guidance", "outlook", "according to",
    "in the filing", "in its 10-k", "in the 10-k",
)

# --- numbers that are not years -----------------------------------------------
# A bare four-digit number is read as a year only when nothing marks it as an
# amount, a count, or the date of something other than a filing. "$2024
# million" and "2000 employees" are figures, and a figure read as a year either
# filters to a year the user never asked about or, for "$2000", reports a year
# outside the corpus and refuses the question. A year written with a prefix or
# an apostrophe -- "FY2024", "'24" -- is never in doubt and is not checked.
# Every table here is matched case-insensitively.
CURRENCY_BEFORE = ("$", "us$", "usd", "€", "eur", "£", "gbp", "s$", "sgd", "¥", "jpy")

# A magnitude after the number always makes it a figure: no year is followed
# by "million" or "percent".
MAGNITUDE_AFTER = (
    "million", "millions", "billion", "billions", "trillion", "thousand", "mn", "bn",
    "percent", "per cent", "%",
)

# A count after the number makes it a figure, unless a possessive or "the"
# comes before the number. "2000 employees" is a count, but "Meta's 2023
# headcount" and "Apple's 2024 shares outstanding" name a year, and reading
# those as figures would drop the year filter without telling anyone.
COUNT_AFTER = (
    "employees", "people", "staff", "workers", "headcount", "shares", "units",
    "customers", "stores", "patents", "locations", "countries", "subscribers",
    "users", "cases", "transactions",
)

# A comparison of quantity before the number makes it a figure when a plural
# noun follows, whatever the noun, so "more than 2000 suppliers" is not read
# as the year 2000 and refused as outside the corpus. The plural is required
# so that "was revenue in 2024 more than 2023?" still names two years. "about"
# and "over" are left out on purpose: "what did Apple say about 2024 results"
# is a question about a year.
QUANTITY_BEFORE = (
    "more than", "fewer than", "less than", "at least", "at most",
    "approximately", "roughly", "nearly", "almost",
    "exceed", "exceeds", "exceeded", "exceeding",
)

# --- the grounded prompt ----------------------------------------------------
# The template ``prompt.py`` renders, in the three pieces a chat model takes:
# a system message with the rules, one line per source, and a user message
# holding the sources and the question. ``GenerationConfig.prompt_template_id``
# records this id on every answer, so the id changes whenever any of the text
# below does. Edit the wording and bump it; a results file that says
# "grounded_v1" has to mean this text and no other.
PROMPT_TEMPLATE_ID = "grounded_v1"

# What the model writes, and nothing else, when the sources do not answer the
# question. One fixed sentence rather than "say you don't know", so that #33
# can recognise an abstention by equality rather than by guessing at the
# wording, and the harness can count it.
ABSTAIN_PHRASE = "The filings do not answer this question."

# The rules. The model is shown numbered sources and told to cite by number.
# It is never shown a URL and never asked to name a company, a year or a
# document, because whatever it is allowed to write it will sometimes invent:
# an integer that names no source is caught by the resolver (#31), while an
# invented URL would read as real. The sources' own metadata is in the header
# so the model can tell FY2023 from FY2024 when both are shown, and that is
# the only reason it is there.
#
# Markers are one integer per bracket, "[1][3]" and not "[1, 3]", because that
# is the form the resolver reads. The rest are the failure modes a grounded
# answer has: mixing years, quoting a figure without its period or unit, and
# answering from memory when the sources fall short.
SYSTEM_PROMPT = f"""\
You answer questions about companies' annual reports (Form 10-K filings) using \
only the numbered sources you are given.

Rules:
1. Use only the sources. Do not use any outside knowledge, even if you are sure \
of it. If the sources do not contain the answer, reply with exactly this \
sentence and nothing else: {ABSTAIN_PHRASE}
2. Cite every claim. After each sentence that draws on a source, write the \
source number in square brackets, like [2]. If a sentence draws on more than \
one source, write each number in its own brackets, like [1][3]. Never write a \
number that is not one of the sources given.
3. Do not name a company, a fiscal year, a document or a web address unless \
the source you are citing says it. The citation carries that information.
4. Quote figures exactly as the source gives them, with their unit and the \
period they cover. If the sources give figures for several years, say which \
year each belongs to. Do not calculate a figure the sources do not state \
unless the question asks for one, and then show the figures you used.
5. If the sources disagree with each other or only partly answer the \
question, say so rather than choosing one silently.
6. Be concise: answer the question directly, then stop."""

# One source, as the model sees it. Company and ticker so the model can tell
# fifteen peers apart; the fiscal year so it can tell one company's five
# filings apart; the Item and its title so it knows whether it is reading
# risk factors or the income statement. No URL, no accession number, no date:
# nothing it could copy into the answer.
SOURCE_HEADER = "[{marker}] {company} ({ticker}), fiscal year {fiscal_year}, {section}"
# The Item line inside the header, with and without an Item number.
SOURCE_SECTION = "Item {item}: {title}"
SOURCE_SECTION_NO_ITEM = "{title}"
# What a table passage is labelled, so the model reads its first line as
# column headers rather than as prose.
SOURCE_TABLE_TAG = " (table)"

# The user message: every source, then the question. Sources first so that
# the question, which the model attends to most, sits closest to where it
# starts writing.
USER_PROMPT = """\
Sources:

{sources}

Question: {question}"""

# What separates one source from the next. Two blank lines, so a passage that
# itself contains a blank line does not look like a source boundary.
SOURCE_SEPARATOR = "\n\n\n"
