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

import re
from typing import Literal, NamedTuple, get_args

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

# Aliases that name a part of the company rather than the whole of it. They
# resolve to the company like any other alias, but the search text keeps them
# (#87): "AWS operating income" asks for the AWS segment's row, and taking
# "AWS" out would leave Amazon's total operating income as the best match.
# An alias followed by a capitalised word or a number is kept for the same
# reason -- "Google Cloud", "Microsoft 365" -- without needing a row here.
SEGMENT_ALIASES = ("aws", "amazon web services")

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

# Why a question is unanswerable, which decides what is done about it.
#
# - request: it asks for advice or for a prediction of the engine's own.
#   Refused without a search, however it is worded.
# - company: it asks for a figure of a company the corpus holds no filings
#   for ("What was Intel's total revenue?"). Refused without a search: a
#   search over the other companies only hands a model passages to answer
#   from.
# - topic: it is about something no filing in the corpus reports as its own
#   subject: a share price, next year, a company outside the corpus. Still
#   searched, because the words cannot tell a request for today's price from
#   a question a filing answers. A 10-K prints the average price paid for
#   repurchased shares, the market value of the shares non-affiliates hold
#   and the obligations due next year, and the filings the corpus holds name
#   the companies it does not as competitors, suppliers and customers.
# - year: it names only fiscal years outside the corpus. Still searched: a
#   filing prints the two years before its own beside it, and what falls due
#   in the years after it.
#
# Where a question is searched, a model that finds no answer in the passages
# abstains, which costs a search. A refusal that is wrong costs the answer.
UnanswerableBecause = Literal["request", "company", "topic", "year"]
UNANSWERABLE_BECAUSE: tuple[UnanswerableBecause, ...] = get_args(UnanswerableBecause)

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
#
# Only the first two tables decide a refusal. The nouns and the future
# phrases are a guess from the words, and a filing answers some questions
# that carry them ("What average share price did Apple pay for repurchases?",
# "What were Microsoft's purchase obligations due next year?"), so those are
# searched and left to the model: see ``UNANSWERABLE_BECAUSE``.

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

# The abbreviations a company's legal name ends in. The full stop after one is
# part of the name, not the end of a clause, so the verb after "Apple Inc." is
# not read as an instruction. Lower case, without the stop.
CORPORATE_SUFFIXES = ("inc", "corp", "co", "ltd")

# What a company does to a figure that makes the figure its own: "how many
# employees did NVIDIA have", "what did Intel report". Stems, matched at the
# start of the word after the company's name. Any other verb leaves the
# company as something a filing in the corpus may be asked about, as in "did
# NVIDIA account for more than 10% of any company's revenue", so the list
# errs towards a search.
OWNING_VERBS = (
    "have", "has", "had", "report", "earn", "make", "made", "generate", "spend", "spent",
    "pay", "paid", "employ", "post", "record", "own", "hold", "held", "owe", "invest",
    "book", "recogni", "incur", "lose", "lost",
)

# Nouns that name a relationship with the company after "of", rather than a
# figure of its own: "customers of Intel", "a supplier of NVIDIA".
RELATION_NOUNS = (
    "customer", "customers", "supplier", "suppliers", "competitor", "competitors",
    "partner", "partners", "vendor", "vendors",
)

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
# "grounded_v4" has to mean this text and no other. A test digests a prompt
# rendered from this template and pins the pair, so forgetting the bump
# fails the suite rather than passing silently.
#
# v4 asks for the answer as JSON in the shape of ``records.GroundedAnswer``,
# which the model is also made to decode against. v3 asked for prose with inline
# "[n]" markers and an exact abstain sentence, both of which a small local
# model has to reproduce character for character to be read correctly.
PROMPT_TEMPLATE_ID = "grounded_v4"

# What an abstention reads as, in the app and in ``Generation.text``. The model
# no longer writes it: it sets ``answerable`` to false, and
# ``GroundedAnswer.render`` shows this sentence. It stays one fixed sentence so
# that #33 and the harness can still recognise an abstention in the text by
# equality, as well as from ``GroundedAnswer.abstained``.
ABSTAIN_PHRASE = "The filings do not answer this question."

# The rules. The model is shown numbered sources and told to cite by number.
# It is never shown a URL and never asked to name a document, because whatever
# it is allowed to write it will sometimes invent: a source number is held to
# the ones shown by the output schema, while an invented URL would read as
# real. A company and a year it may name, but only the ones its own source's
# header states, which is also why that metadata is in the header: so the
# model can tell FY2023 from FY2024 when both are shown.
#
# The output format is described in words as well as enforced by the schema,
# because the schema fixes the shape but not the meaning of each field. The
# example's values are "..." on purpose: a small model copies the content of
# an example into its answer. The rest are the failure modes a grounded answer
# has: mixing years, quoting a figure without its period or unit, and
# answering from memory when the sources fall short.
SYSTEM_PROMPT = """\
You answer questions about companies' annual reports (Form 10-K filings) using \
only the numbered sources you are given.

Reply with a JSON object with two fields:
- "answerable": true if the sources contain anything that answers the \
question, even in part. false if nothing in them bears on it.
- "sentences": the answer, one sentence per item. Each item has "text", the \
sentence, and "sources", the numbers of the sources that sentence draws on. \
When "answerable" is false, leave the list empty.

For example: {"answerable": true, "sentences": [{"text": "...", "sources": [2]}, \
{"text": "...", "sources": [1, 3]}]}

Rules:
1. Use only the sources. Do not use any outside knowledge, even if you are sure \
of it.
2. Cite every claim. Put the number of every source a sentence draws on in its \
"sources", and never a number that is not one of the sources given. Do not \
write source numbers inside "text".
3. Name a company or a fiscal year only when the source you cite is that \
company's filing for that year, as its header states, or when the source text \
says it. Never name a document, a filename or a web address.
4. Quote figures exactly as the source gives them, with their unit and the \
period they cover. If the sources give figures for several years, say which \
year each belongs to. Do not calculate a figure the sources do not state \
unless the question asks for one, and then show the figures you used.
5. If the sources bear on the question but answer only part of it, or \
disagree with each other, give what they do support and say what is missing \
or in dispute. That is still an answer, so "answerable" stays true. Do not \
silently pick one side.
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
# What a table passage is labelled, so the model reads it as a rendered grid
# rather than as prose: the text opens with the table's caption, then the grid
# under its own column headers.
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

# --- generation -------------------------------------------------------------
# Two providers, chosen with LLM_PROVIDER in .env (#106). Ollama runs a model on
# the computer running the code: no account, no key and no network, and 2 to 4
# minutes an answer on a laptop's CPU. Mistral's hosted API answers in about 2
# seconds on its free plan, with each teammate's own key from their own account.
# Ollama is the default, so a fresh clone works offline, and a demo whose key or
# connection fails still has something to fall back on.
# ``GenerationConfig.provider`` records which one served an answer, so a results
# row says what produced it.
OLLAMA = "ollama"
MISTRAL = "mistral"
PROVIDERS = (OLLAMA, MISTRAL)
DEFAULT_PROVIDER = OLLAMA

# The model a fresh clone uses, as Ollama names it. Swap it with
# ``ollama pull <model>`` and LLM_MODEL in .env.
DEFAULT_MODEL = "llama3.2:3b"

# The Mistral model used when LLM_MODEL is unset. Ministral 3 8B did best of the
# four suitable chat models the free plan serves. On the 48 test questions at
# FINAL_K 16 it stated 22 of the 28 expected figures, 20 of them to the exact
# digit, and abstained twice, against 22, 17 and 4 for Ministral 3 14B at the
# same 2.2 s median, and the free plan allows it 188 requests a minute to 14B's
# 30. Voxtral Small and Codestral stated figures the passages did not hold. See
# docs/mistral-free-tier-evaluation.md and notebooks/mistral/.
DEFAULT_MISTRAL_MODEL = "ministral-8b-2512"

# Where the provider, the model, the server and the key come from when nothing
# is passed in. All are read from the environment, with the project's .env loaded
# into it first.
LLM_PROVIDER_ENV = "LLM_PROVIDER"
# The model for the provider LLM_PROVIDER names. A provider chosen over it, with
# ``--provider``, uses its own default instead: an Ollama model name means
# nothing to Mistral's API, and the reverse.
LLM_MODEL_ENV = "LLM_MODEL"
LLM_BASE_URL_ENV = "LLM_BASE_URL"
# The Mistral key: each teammate's own, from their own account, never a shared one.
MISTRAL_API_KEY_ENV = "MISTRAL_API_KEY"
# Where Mistral's API is, when not its own address. The client is built with
# it, and the client cache is keyed on it too. Nobody needs to set it: it
# exists for a proxy, and for pointing the test suite at a dead address to
# prove no test reaches the real API.
MISTRAL_BASE_URL_ENV = "MISTRAL_BASE_URL"
# Mistral's own address, ChatMistralAI's default, used when MISTRAL_BASE_URL is
# not set. A connection that fails anywhere else is blamed on the setting.
MISTRAL_API_URL = "https://api.mistral.ai/v1"
# Where a teammate makes their own Mistral key, named in every message that asks
# for one.
MISTRAL_CONSOLE = "https://console.mistral.ai"
# Ollama's own default address, which is always the computer the code runs on:
# every member runs their own Ollama. 127.0.0.1 rather than localhost, which
# Windows can resolve to ::1 first, where Ollama is not listening.
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"

# How many of the model's layers Ollama puts on the GPU. Unset leaves it to
# Ollama, which is right on a machine with a capable GPU or Apple silicon. It is
# wrong on a laptop with a small discrete GPU: on a 2 GB MX450, Ollama put 3 of
# llama3.2:3b's 29 layers on the GPU and generated 1.7 tokens a second, against
# 7.6 with the GPU left out. Set LLM_NUM_GPU=0 in .env on such a machine. It
# changes where the model runs and not what it writes, so it is not recorded
# on the GenerationConfig.
LLM_NUM_GPU_ENV = "LLM_NUM_GPU"

# Ollama's context window, in tokens, set on every request rather than left to
# Ollama. A prompt longer than the window is cut from the front with no error
# to the caller, only a "truncating input prompt" warning in the server's log.
# That drops the rules and the first sources, and leaves an answer that still
# looks fine. The prompt over FINAL_K (16) passages measured about 5,200
# tokens at the median and 6,306 at its largest, and the output ceiling has to
# fit beside it; retrieval/constants.py records how close that comes.
# Ollama's own default depends on the GPU's memory and is 4,096 on a laptop,
# which leaves little room for either.
NUM_CTX = 8192

# The output ceiling. A grounded answer is a few sentences with their source
# numbers (rule 6 says answer directly, then stop), so 1,024 tokens is
# several times the longest answer the benchmark expects, while bounding how
# long a model that ignores rule 6 can run on a laptop CPU. A ``Generation``
# whose ``stop_reason`` says it hit this is marked truncated, so the cut is
# never silent.
MAX_OUTPUT_TOKENS = 1024

# How long to wait on Ollama for the next piece of a response, in seconds.
# This is a read timeout, not a limit on the whole answer, and the longest wait
# is the first one: nothing streams back until the whole prompt has been read.
GENERATION_TIMEOUT_S = 600.0

# The same wait for Mistral's API. Its slowest answer in the free-plan test took
# 23.9 s from question to last word, so two minutes only ever catches a connection
# that has gone quiet. It is also what the test notebook ran with.
HOSTED_TIMEOUT_S = 120

# --- financial metrics ------------------------------------------------------
# The line items a question can name, the XBRL concepts a filer tags them with,
# and the unit each is reported in. One table, read from two directions:
# ``numeric.py`` reads a question's words to find the metric and then looks the
# figure up (#34), and ``verify.py`` reads an answer's words to find the metric
# and then checks the figure against the same store (#32). Two tables would
# drift, and an answer routed on one vocabulary and checked against another
# would be flagged for asking a question its own checker could not.
#
# Aliases are deliberately narrow, and a question matching two metrics is
# treated as matching none: extend this from the benchmark rather than by
# guessing, since a wrong concept answers confidently with the wrong figure.
# ``concepts`` are matched on the part after the taxonomy prefix, so
# "us-gaap:Revenues" matches "Revenues".
#
# An alias is also what a statement row has to be called before the route will
# cite it (``numeric.prints_figure``) or the checker read it, so the names the
# filings use are here beside the ones a question uses: "income from
# operations" (Salesforce, Alphabet, Meta, ServiceNow), "operating profit"
# (Texas Instruments), "cash and equivalents" (Micron), "trade payables"
# (Adobe), and the cash flow statement's line for cash from operations, which
# few filers word alike. Each was read off the rows that print a stored figure
# across the 75 filings. "Inventory", singular, is left out though Alphabet and
# Broadcom print it: the word is in too much prose about purchase commitments
# and risk for a sentence using it to be checked against the balance.
#
# Adobe, Salesforce, Alphabet and Intuit call earnings per share "net income
# per share", which holds another line item's name. ``numeric.metrics_in``
# reads a name inside a longer one as part of the longer one, so "diluted net
# income per share" names earnings per share and not net income. Plain "net
# income per share" is left out on purpose: it does not say basic or diluted,
# so the route would be choosing one for the user. ``metrics_in`` reads it as
# a per-share amount that names no line item here, and not as net income.


class Metric(NamedTuple):
    """One line item: what a question calls it, how a filer tags it, its unit."""

    aliases: tuple[str, ...]
    concepts: tuple[str, ...]
    unit: str


FINANCIAL_METRICS: dict[str, Metric] = {
    "revenue": Metric(("total revenue", "revenues", "revenue", "net sales"),
                      ("Revenues", "Revenue", "RevenueFromContractWithCustomerExcludingAssessedTax",
                       "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet"), "USD"),
    "net_income": Metric(("net income", "net earnings", "net loss"),
                         ("NetIncomeLoss", "ProfitLoss"), "USD"),
    "operating_income": Metric(("operating income", "operating loss", "income from operations",
                                "operating profit"), ("OperatingIncomeLoss",), "USD"),
    "assets": Metric(("total assets",), ("Assets",), "USD"),
    "liabilities": Metric(("total liabilities",), ("Liabilities",), "USD"),
    "accounts_payable": Metric(("accounts payable", "trade payables"),
                               ("AccountsPayableCurrent",), "USD"),
    "inventory": Metric(("inventories",), ("InventoryNet",), "USD"),
    "cash": Metric(("cash and cash equivalents", "cash and equivalents"),
                   ("CashAndCashEquivalentsAtCarryingValue",), "USD"),
    "diluted_eps": Metric(("diluted earnings per share", "diluted eps",
                           "diluted net income per share"),
                          ("EarningsPerShareDiluted",), "USD/shares"),
    "basic_eps": Metric(("basic earnings per share", "basic eps",
                         "basic net income per share"),
                        ("EarningsPerShareBasic",), "USD/shares"),
    "operating_cash": Metric(("cash from operations", "operating cash flow",
                              "net cash from operations",
                              "net cash provided by operating activities",
                              "net cash provided by (used in) operating activities",
                              "cash provided by operating activities",
                              "cash generated by operating activities",
                              "cash flow provided by operating activities",
                              "cash flows from operating activities"),
                             ("NetCashProvidedByUsedInOperatingActivities",), "USD"),
}

# Preserve the established numeric aliases. Newly supported balance-sheet
# items below need a figure request so prose about them keeps its old reading.
NUMERIC_CUES = NUMERIC_CUES + tuple(
    alias
    for key, metric in FINANCIAL_METRICS.items()
    if key not in {"accounts_payable", "inventory"}
    for alias in metric.aliases
    if alias not in NUMERIC_CUES
)

# These newly supported balance-sheet items also occur in prose about risk
# and accounting. Like stored labels, they need a request for a figure.
FIGURE_METRIC_CUES = tuple(
    alias for key in ("accounts_payable", "inventory")
    for alias in FINANCIAL_METRICS[key].aliases
)

# Lines of the three statements that the facts route does not answer, by the
# names a question gives them. "What was Microsoft's accounts receivable in
# fiscal year 2024?" asks for a figure, and none of its words was a cue, so it
# was read as prose and searched with no lean toward tables: four of its
# sixteen passages were tables, where a figure question gets twelve.
# ``notebooks/answers/line_item_figures.py`` asks that question of every line
# item here and every filing of a year.
#
# The XBRL label of a line is not what a question calls it ("Accounts
# Receivable, after Allowance for Credit Loss, Current"), so the labels in the
# facts store do not cover these. Like the two metrics above, each needs a
# request for the figure: "how does Microsoft manage accounts receivable risk"
# stays prose. "Share repurchases" was measured and left out, since a filing
# reports the cash paid and the amount bought under its programme as two
# figures and the question does not say which.
FIGURE_LINE_ITEM_CUES = (
    "income before income taxes", "income tax expense", "provision for income taxes",
    "sales and marketing expense", "sales and marketing expenses",
    "stock-based compensation", "share-based compensation",
    "accounts receivable", "property and equipment", "property, plant and equipment",
    "intangible assets", "capital expenditures", "purchases of property and equipment",
    "interest paid",
)

# How the facts store writes a unit, and what this project calls it. The store
# carries the XBRL unit, and writes a per-share unit as "USD per share" rather
# than the "USD/shares" the taxonomy suggests, so both spellings are here: a
# missing one is not a crash but an earnings-per-share question that silently
# never matches a fact.
UNIT_ALIASES = {"pure": "ratio", "usd": "USD", "usd/shares": "USD/shares",
                "usd per share": "USD/shares"}

# The scope a stored annual figure cannot answer, however well the concept
# matches. Three kinds: a part of the company rather than the whole of it (a
# segment, a product line), a part of the year rather than the year (a
# quarter), and a figure derived from others rather than reported (a margin, a
# change). Read from both directions, like FINANCIAL_METRICS: #34 refuses to
# route the question, and #32 records the answer as unverified.
FACT_SCOPE_UNSUPPORTED = re.compile(
    r"\b(?:segment|iphone|ipad|aws|azure|google cloud|product revenue|services revenue|"
    r"quarter|quarterly|q[1-4]|combined|sum|average|difference|ratio|margin|"
    r"(?:increased?|decreased?|grew|fell) by)\b", re.I,
)

# What the router refuses on top of that, so that the two directions are not
# the same set. A qualifier turns a line item into a different one: "cost of
# revenue" and "deferred revenue" are not revenue, and "net income per diluted
# share" is not net income, so answering any of them with the whole-company
# figure is confidently wrong. Both routes reject these concept qualifiers;
# the router additionally rejects a percentage request. A claim saying
# revenue was "up 2 percent" is a claim about revenue and stays checkable. The
# router refusing a superset of what the checker refuses is the safe
# direction; the reverse would answer what nothing can check. A bare "per
# share" is deliberately absent, since it would block the EPS aliases.
FACT_METRIC_QUALIFIER = re.compile(
    r"\b(?:cost of|deferred|unearned|non-?operating|"
    r"per (?:diluted|basic) share|purchase obligations?|reserves?|write[-\s]?downs?)\b", re.I,
)
# A percentage may describe a valid claim about revenue, while a question
# asking for a percentage cannot be answered with the stored annual total.
METRIC_QUALIFIER = re.compile(
    FACT_METRIC_QUALIFIER.pattern + r"|\bpercent(?:age)?\b", re.I,
)

# The words a question can hold besides the line item it asks for: the frame
# of the question, the scaffolding around a company and a year, and the verbs
# of reporting. A question asking for a stored figure is made of nothing else,
# so anything left over after the metric's own words are removed -- a segment
# ("Services net sales"), a place ("revenue in Greater China"), a second line
# item ("total liabilities and shareholders' equity"), or a verb that asks for
# prose rather than a figure ("how does Apple recognise revenue", "what drove
# net sales") -- means the question is asking for something the whole-company
# annual figure does not answer. See ``numeric.find_metric``.
#
# "total" is here because it never narrows a line item -- it is the income
# statement's own word for the whole-company row ("Total net sales") -- and
# "earn" and "generate" because they are verbs of reporting like "report".
# "value" permits "total value of Accounts Payable" and "Inventories value".
# A possessive counts only where its owner does, so "the company's" and
# "Oracle Corporation's" pass and "LinkedIn's" does not; see
# ``numeric._is_scaffolding``.
#
# This is the positive half of the test. A blocklist alone has to grow by one
# segment name at a time, and the corpus has fifteen companies' worth of them.
QUESTION_SCAFFOLDING = frozenset("""
    what which was were is are be been how much many
    did do does report reported reports say says earn earned generate generated
    the a an this that its their there total
    in for during at on of to
    fy fye fiscal year years ended ending end period periods
    company companies group inc corp corporation plc ltd value
    dollars dollar usd
""".split())

# The accession number a chunk_id opens with, which is how a passage is matched
# to the filing a fact came from. Shared so the router and the checker cannot
# disagree about what counts as the same filing.
ACCESSION_PATTERN = re.compile(r"\d{10}-\d{2}-\d{6}")

# A statement's scale, as a table says it: "(in millions)". What a figure in
# the table has to be multiplied by, and therefore what tells a passage that
# prints "391,035" apart from one that prints a raw 391,035.
TABLE_SCALE = re.compile(r"\bin\s+(thousands|millions|billions)\b", re.I)

# --- answering from the facts store -----------------------------------------
# A numeric question is answered by looking the figure up rather than by asking
# a model to read it out of a passage (#34). The answer is then this project's
# own sentence, not a model's, so it records what produced it the way a
# generated answer does -- the harness reads ``GenerationConfig`` to attribute
# a result, and "facts" is the truthful thing for it to read here.
FACTS_PROVIDER = "facts"
FACTS_SOURCE = "xbrl"
FACTS_TEMPLATE_ID = "facts_v1"

# The sentence a looked-up figure is rendered as. The company, the metric as
# the filing labels it, the figure, and the period it covers -- everything a
# reader needs to check the citation against the filing, and nothing a model
# chose. The source marker is appended by ``render_sentence``.
FACTS_SENTENCE = "{company} reported {label} of {figure} for {period}."

# How many passages to look through for the one that prints the figure. Wider
# than FINAL_K because this is not a ranked answer set: the figure is in one
# specific table of one specific filing, and the search is already narrowed to
# that filing, so the cost of looking further down is a few string comparisons.
#
# 50, which is retrieval's CANDIDATE_K, rather than the 20 it started at. Hybrid
# fetches that many candidates for any smaller request, so looking through all
# of them searches and scores nothing more. At 20 the route found a
# figure in the store and then no passage to cite for questions whose statement
# table sat a little further down: a search for "total revenue" ranks Amazon's
# income statement, which says "net sales", below the prose that uses the word.
# Over the plain question for each filing and line item, 825 of them
# (notebooks/answers/headline_figures.py, with Hybrid), the route answered 660
# at 50 where it answered 651 at 20, and none it had answered before was lost.
FACT_PASSAGE_K = 50

# --- decomposing a question -------------------------------------------------
# How many company-and-year PAIRS a question may be split into (#35). It caps
# the cross product only, not a split along one axis, and the difference is
# what the question enumerates: a question naming five years is asking about
# five years, and a passage from each answers it better than FINAL_K from two
# of them, whereas the pairs of two companies and five years are ten filings
# nobody named. At FINAL_K = 16 four pairs means four passages per filing, and
# the ten pairs above would leave one or two each, too little of each side for
# a comparison. A question naming more pairs than this splits on companies
# alone and keeps its years whole.
#
# Four rather than the eight the budget would now stretch to, because the
# passages per filing are not the only cost: each pair is another search, and
# each filing is another one for the model to hold together in one answer at
# 3B on a laptop. Which of the two matters more is a thing to measure on the
# benchmark (#26) rather than to assume here, in the way the score floors are
# left unset until measured and TABLE_BOOST stayed off until its sweep.
#
# The other cap is not a constant, because it follows from the budget itself:
# no split may leave a sub-question with no passage at all, so a split is only
# made where there are at least as many passages to give out as filings to give
# them to. See ``decompose.decompose``.
MAX_CROSS_SPLIT = 4
