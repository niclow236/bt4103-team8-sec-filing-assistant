"""Fixed values that tune the retrieval stage.

Kept apart from the retrievers the same way ``src/pipeline/constants.py`` is
kept apart from the stages, and for a sharper reason here: BM25, dense, hybrid
and reranking are built by different people at the same time, and every one of
them needs the candidate depth, the model name and the fusion constants. Four
modules holding four copies of ``k = 50`` is four numbers to change when the
sweep says 80, and a comparison that is silently unfair the one time somebody
misses one.

So a retriever imports from here and nothing here imports a retriever, which is
the rule ``records.py`` already follows next door. Paths are the exception: they
live in ``src/config.py`` with every other project path, and are re-exported at
the bottom of this module so a retriever still has one import.

**Nothing here is a magic number.** Each value below says where it came from:
a library default, a published result, or a starting point to be calibrated
once the benchmark in #24 and the harness in #26 can measure it. The ones
marked as starting points are the ones the ablation is expected to move; a
value that was never measured should not be quoted in the report as though it
were.
"""

from __future__ import annotations

from src.config import BM25_INDEX_FILE, CHROMA_DIR, INDEX_DIR

# --- the methods being compared ---------------------------------------------
# The four retrievers the ablation runs, named once here so that the string on
# a RetrievedPassage, the key in FUSION_WEIGHTS and the row label in the results
# table cannot drift apart. These are the exact values RetrievedPassage.retriever
# is documented to take.
BM25 = "bm25"
DENSE = "dense"
HYBRID = "hybrid"
RERANK = "rerank"

RETRIEVERS = (BM25, DENSE, HYBRID, RERANK)

# --- embedding model --------------------------------------------------------
# The corpus was already cut for this class of model. CHUNK_CHAR_BUDGET is 1,800
# characters because that is about 450 tokens at four characters per token, and
# verify.py gates the corpus at EMBED_CHAR_LIMIT = 2,048 characters so that no
# passage exceeds the 512 tokens a sentence-transformer will read. Choosing a
# model with a shorter window than 512 tokens silently truncates passages the
# corpus was built to fit; choosing a longer-window model is safe but wastes the
# calibration. bge-base-en-v1.5 is one of the two the chunker's own comment names.
#
# Changing this means re-embedding. It does not mean re-chunking, as long as the
# replacement also reads 512 tokens -- which is exactly the distinction
# IndexManifest.mismatches() reports, by comparing model and dimensions
# separately from the corpus fingerprint.
EMBED_MODEL = "BAAI/bge-base-en-v1.5"

# Vector width, written into the IndexManifest so a retriever refuses an index
# built by a different checkpoint rather than returning nonsense from vectors
# that do not line up. Fixed by the model above: change one and change both.
EMBED_DIMENSIONS = 768

# The model's context window in tokens, and the reason the chunker's budget is
# what it is. Held here so embed.py can assert against it rather than trusting
# that the corpus on disk was cut by the current settings.
EMBED_MAX_TOKENS = 512

# Passages per forward pass. A throughput knob with no effect on results: raise
# it on a machine with a GPU, lower it if a laptop runs out of memory. 32 is the
# sentence-transformers default and embeds this corpus in minutes on CPU.
EMBED_BATCH_SIZE = 32

# bge is trained for cosine similarity on unit-length vectors, so vectors are
# normalised at encode time and Chroma is told to use cosine distance. These two
# go together: normalised vectors scored by L2 rank differently from cosine, and
# the mismatch shows up as retrieval that is subtly poor rather than broken.
EMBED_NORMALIZE = True
DISTANCE_METRIC = "cosine"

# bge-v1.5 asks for an instruction prefix on the QUERY only, never on the
# passage, and reports a small retrieval gain from it. Omitting it costs a
# little accuracy; applying it to passages as well costs more, because it makes
# every passage slightly more similar to every other. e5 models want the
# opposite convention ("query: " and "passage: " on both sides), so if
# EMBED_MODEL changes to e5, both of these change with it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
PASSAGE_PREFIX = ""

# What is prepended to a passage at embed time and never stored with it (#18).
# A table passage of bare figures embeds poorly alone: "12,345" carries no
# company, year or subject, so it lands nowhere useful in vector space. The
# header is written here rather than inline in embed.py because it is part of
# what an index is, and an index built under a different header is not
# comparable with one built under this. It stays out of data/processed/ so
# citations quote the filing rather than our annotation. The form is a field
# rather than a literal "10-K" so the header still reads correctly when 10-Q
# is layered in; every field named here is a real key on an iter_chunks row.
CONTEXT_HEADER = "{company} ({ticker}) FY{fiscal_year} {form}, Item {item}: {title}\n\n"

# --- BM25 -------------------------------------------------------------------
# The Okapi BM25 defaults, and rank_bm25's own. k1 controls how fast term
# frequency saturates and b how hard long documents are penalised. They are left
# at the defaults deliberately: BM25 is the baseline the dense and hybrid rows
# are measured against, and a baseline tuned harder than the alternatives is not
# a baseline. If the sweep in #26 tunes these, it tunes the others too, and the
# report says so.
#
# Worth knowing when reading the numbers: passages here are near-uniform in
# length by construction (a 1,800-character budget with a 200-character floor),
# so b has less to do on this corpus than it would on raw filings.
BM25_K1 = 1.5
BM25_B = 0.75

# --- how deep to retrieve, and how much to keep -----------------------------
# Broad, then narrow. Each method returns CANDIDATE_K, the union is fused, the
# reranker scores that set, and FINAL_K passages reach the generator. Retrieving
# FINAL_K directly measures worse (Snowflake finance-RAG, cited in #21): the
# passage that answers the question is often outside a first-stage top-8 and
# only a reranker that has seen it can pull it up.
#
# 50 and 8 are the architecture's numbers (§3, §5). FINAL_K is also a context
# budget: 8 passages at up to ~1,800 characters each is roughly 14k characters
# before the prompt, which is what makes 8 rather than 20 the ceiling.
CANDIDATE_K = 50
FINAL_K = 8

# Note for the retrievers: Query.top_k in records.py defaults to 10, which is
# neither of these. A retriever should read k from the Query it was handed and
# treat these as the values the harness passes in, not as a second default to
# apply when the Query already says something.

# --- fusion -----------------------------------------------------------------
# Reciprocal rank fusion: each passage scores sum(weight / (RRF_K + rank)) over
# the methods that returned it. Rank-based rather than score-based, because a
# BM25 score and a cosine similarity are not on the same scale and normalising
# them into agreement is a tuning exercise that RRF avoids entirely.
#
# 60 is the constant from the original RRF paper (Cormack et al., 2009) and the
# value used almost everywhere since. It flattens the difference between the top
# ranks, so a passage found at rank 1 by one method does not automatically beat
# one found at rank 3 by both -- which is the behaviour hybrid retrieval is for.
RRF_K = 60

# Equal by default, so the first hybrid number is an honest fusion of two
# methods rather than a thumb on the scale for whichever we expected to win.
# FinRank finds sub-billion encoders gain at most 3.5 points over BM25 on
# financial text, so there is no prior here strong enough to justify starting
# uneven. A starting point: #26 sweeps these, and the report quotes the swept
# value with the sweep beside it.
FUSION_WEIGHTS = {BM25: 1.0, DENSE: 1.0}

# --- reranking --------------------------------------------------------------
# A cross-encoder reads the question and the passage together and scores the
# pair, which is what lets it catch relevance a bi-encoder misses -- at a cost
# that only makes sense on a candidate set this small. ms-marco-MiniLM-L-6-v2 is
# the standard baseline: 22M parameters, CPU-viable, and the model most reported
# rerank numbers are measured against, which makes ours comparable to theirs.
# bge-reranker-base is the stronger and slower alternative if the interim
# results say reranking is where the gain is.
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Query-passage pairs per forward pass. Throughput only, like EMBED_BATCH_SIZE.
RERANK_BATCH_SIZE = 32

# The pair -- question and passage together -- must fit the cross-encoder's
# window, and the question spends part of it. A 1,800-character passage is about
# 450 tokens, leaving roughly 60 for the question, so a long decomposed
# sub-question can push a pair over and the tail of the passage is dropped.
# Truncation here is the model's own and silent, which is why the limit is
# written down: if reranking underperforms on long questions, look here first.
RERANK_MAX_TOKENS = 512

# --- score thresholds -------------------------------------------------------
# Floors below which a passage is dropped rather than returned weakly.
#
# All three are None to start, which means no floor is applied, and that is the
# deliberate choice rather than an omission: a threshold invented before the
# benchmark exists is a magic number with a confident name, and one set too high
# removes the correct passage and turns a retrieval miss into an answer that
# says nothing. #25 measures recall at these settings first; #26 sets them from
# that measurement, on the benchmark, and the value that ships is the one the
# curve justified.
#
# The scales differ, so they cannot share a value. BM25 scores are unbounded and
# corpus-dependent. RRF scores are tiny and bounded -- one method at rank 1 gives
# 1/61 -- so a threshold here is really a "how many methods agreed" test. The
# cross-encoder emits logits, roughly -11 to +11, where 0 is the natural
# indifference point and the only one of the three with a meaningful prior.
MIN_BM25_SCORE: float | None = None
MIN_FUSED_SCORE: float | None = None
MIN_RERANK_SCORE: float | None = None

# --- filters ----------------------------------------------------------------
# The corpus is fifteen peers in one industry over five years, so metadata is a
# hard pre-filter and not a hint (#22): the retrievers cut the corpus down to
# what the Query allows and score inside it. Apple's FY2023 and FY2024 risk
# factors sit almost on top of each other in embedding space, and filtering
# after scoring returns the right company in the wrong year while reading as
# confident and correct.
#
# One thing the retrievers have to handle themselves: iter_chunks() takes
# tickers, fiscal_years and key_items_only, but NOT item or content_type.
# Those two are filtered above it, as passages.py does. A Query whose items or
# content_type are passed to iter_chunks and forgotten is not an error -- it
# just searches a corpus four times wider than the question asked for.
PREFILTER_FIELDS = ("ticker", "fiscal_year", "item", "content_type", "is_key_section")

# A multiplier on the score of a table passage when the question is numeric, so
# "what was revenue in FY2024" leans toward the passages that keep figures under
# their row and column labels. 1.0 is off, which is where it starts: this is the
# most tempting knob in the file to set by intuition and the easiest to fool
# yourself with, since boosting tables always looks better on the handful of
# numeric questions you happen to try. #24 generates the mechanical XBRL
# benchmark precisely so this can be set from data.
TABLE_BOOST = 1.0

# --- where indexes live -----------------------------------------------------
# Imported from src.config, which owns every project path, and named here so a
# retriever has one import rather than two while the folder layout still changes
# in exactly one place. INDEX_DIR holds both: bm25.pkl is a single file, Chroma
# wants a directory of its own.
__all__ = [
    "BM25", "DENSE", "HYBRID", "RERANK", "RETRIEVERS",
    "EMBED_MODEL", "EMBED_DIMENSIONS", "EMBED_MAX_TOKENS", "EMBED_BATCH_SIZE",
    "EMBED_NORMALIZE", "DISTANCE_METRIC", "QUERY_PREFIX", "PASSAGE_PREFIX",
    "CONTEXT_HEADER",
    "BM25_K1", "BM25_B",
    "CANDIDATE_K", "FINAL_K",
    "RRF_K", "FUSION_WEIGHTS",
    "RERANK_MODEL", "RERANK_BATCH_SIZE", "RERANK_MAX_TOKENS",
    "MIN_BM25_SCORE", "MIN_FUSED_SCORE", "MIN_RERANK_SCORE",
    "PREFILTER_FIELDS", "TABLE_BOOST",
    "INDEX_DIR", "BM25_INDEX_FILE", "CHROMA_DIR",
]
