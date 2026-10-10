# C5 reranking experiment (#154)

The default remains C4. This experiment compares C4 with an opt-in cross-encoder
over the **same** filtered hybrid top-50 pool, returning 16 passages in both rows.
It does not add C5 to the app's selectable configurations. Adoption requires a
retrieval benefit, an answer-quality check, and teammate review.

## Decision criteria declared before the final run

These are practical gates for this experiment, not universal thresholds or
retrospectively chosen cutoffs:

- On each of `handwritten` and mechanical `xbrl`, mean nDCG@16 must improve by
  at least **0.02**, with the lower bound of the paired 95% interval above zero.
  Recall@16 and MRR@16 must not regress by more than **0.01** on either source.
- `llm_assisted` nDCG must not regress by more than **0.02**. A question-type
  slice (overall or within a source) with at least 10 answerable questions must not lose more than **0.02**
  nDCG. Mean hard-negative accuracy must not lose more than **0.01**.
- Added steady-state reranking time, including input-length diagnostics and
  ranking, must have a median at most **500 ms** and p95 at most **1,000 ms**.
  This allows a modest extra stage in an interactive query; cold model load and
  warm-up are recorded separately.
- If these gates pass, perform a paired answer-quality comparison before
  adopting C5. Otherwise retain C4 and document the result.

The decision cutoff is fixed at @16 in the policy. A run or rescore at another
cutoff records metrics with `advance_to_answer_evaluation: null`, even if those
metrics would otherwise pass. Missing sources, including `llm_assisted`, block
advancement. The latency gate excludes empty candidate pools; their number and
the number of timed questions are reported explicitly.

The team benchmark is the accepted 149-question version merged from #152,
containing 124 handwritten and 25 LLM-assisted questions. The mechanical set is
the complete locally generated XBRL benchmark. Its labels use figure substring
matching and are a retrieval proxy, not exhaustively adjudicated evidence.
No model tuning or checkpoint selection uses final benchmark outcomes.

## Fixed experiment settings

The reference is `cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Input pairs contain the full `Query.text`
and stored passage text, without extra headers or prompts. The maximum paired
input is 512 tokens, including special tokens, with the tokenizer's
`longest_first` truncation. Batch size is 32. Scores are raw logits; the query's
table boost is reapplied with the existing sign-aware score transformation.
Ties break by chunk ID. There is no invented cross-encoder score floor.

Both rows use the existing C4 evaluation query policy: parse the question and
apply its benchmark ticker/year where specified. This uses the same filter,
keyword trimming, numeric table boost and fusion settings for both rows. Each
query retrieves once; C4 keeps the first 16 fused passages and C5 reorders the
same 50 candidates. C4 timing is the shared top-50 retrieval measurement and
C5 adds the measured reranking time, rather than pretending these are two
independent end-to-end answer runs. Generation, prompt tokens, and judging are
not measured by this retrieval-only command.

`--candidate-k` may reduce the pool for an exploratory run, but cannot exceed
the shipped fusion depth of 50. A deeper request would change the BM25/dense
rankings fused for the C4 row and would require a separately named baseline.

## Commands and artifacts

With dependencies, the processed corpus, current verified BM25/dense indexes,
and both benchmark files prepared:

```bash
python -m src.evaluation.rerank_ablation run \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --run-id c5-reference-YYYYMMDD --top-k 16 --candidate-k 50 \
  --batch-size 32 --device mps \
  --workload-note "No other model or evaluation jobs; normal desktop apps open"
```

Use `--device cpu` on a machine without MPS, and record that hardware difference.
The checkpoint downloads on first use and is then cached. The default stack
does not load it. `RerankRetriever` is a protocol adapter for explicit experiments
and future answer checks; `CrossEncoderReranker` lets the paired runner reuse
the exact candidate pool.

`results/<run-id>/` holds:

- `config.json`: predeclared policy, settings, source-file hashes, commit,
  dependency/platform versions and input/corpus fingerprints;
- `runtime.json`: model load, warm-up, device and index manifests;
- `runtime-NNNN.json`: an immutable record for each new or resumed segment,
  including its starting question count and current workload note. Existing
  `runtime.json` and legacy `resume-runtime.json` are preserved;
- `pairs.jsonl.gz`: each full benchmark question and parsed query, both ranked
  ID/score lists and metrics, and all candidate logits, token lengths, metadata
  and source provenance;
- `summary.json`: paired metrics, source/type/evidence-kind breakdowns, wins,
  losses, ties, cluster counts, uncertainty, truncation and timing diagnostics,
  plus the decision and any failed criteria.

Outputs stream to disk. Graceful interruptions preserve completed pairs. To
resume, repeat the identical run command with `--resume`; changed inputs,
source files, corpus, policy or configuration are refused. A forcibly killed
process can leave a truncated gzip stream, which is refused rather than silently
dropping evidence. Use a new run ID if that happens.

An interruption before the first pair may leave only a manifest; resume treats
the absent pairs file as an empty prefix. Resume also compares the original
BM25 and dense index manifests, including build identity, before loading models.
Each segment can supply a new workload note without replacing the original
manifest. A process-safe `run.lock` prevents concurrent writers to one run ID
and releases its OS lock on exit, including a killed process.

To recompute metrics and the decision without models or corpus files:

```bash
python -m src.evaluation.rerank_ablation rescore results/c5-reference-YYYYMMDD
```

An optional `--top-k` can rescore a smaller cutoff, up to the stored final k.
It is a sensitivity check, not a replacement for the declared @16 decision.
Incomplete runs cannot be presented as a final rescored benchmark.

Older pairs lack `gold_filings`, so their gold-scope diagnostics are `null`,
never invented zeros. To derive those fields from supporting chunk IDs without
modifying the original pairs, supply the corpus when rescoring:

```bash
python -m src.evaluation.rerank_ablation rescore results/c5-reference-YYYYMMDD \
  --processed-dir data/processed
```

This requires the full corpus records SHA-256 to match the frozen manifest;
changed text or metadata is refused. The output records the derivation digest
and row count. It retains the stored rankings, scores and timing measurements.

Recall, nDCG and MRR exclude questions without supporting IDs, with each metric's
denominator reported. Hard-negative accuracy includes only questions with
labelled hard negatives; unanswerable questions therefore receive that metric,
not fabricated zero relevance scores. Unlabelled negatives cannot measure
abstention or answer safety. Wrong-company/year diagnostics count top-k
passages from a company or fiscal year outside the gold passages' filings; the
query filters already exclude anything outside the question's own scope.
The report retains query-scope counts as integrity checks and adds a joint
company/year count so valid companies and years cannot hide an invalid pairing.
Gold-scope denominators exclude unlabelled questions; unknown gold years cannot
establish a wrong-year count. These are evidence-scope proxies because gold
labels may omit other relevant passages; they do not measure answer correctness.

`by_gold_truncation` separates questions with any truncated supporting passage
in the candidate pool, those with untruncated gold in the pool, those with no
gold in the pool, and unlabelled questions. A missing gold candidate has no
recorded token length and is not asserted to be untruncated. Each slice reports
paired metrics, uncertainty and denominators.

Uncertainty uses 2,000 paired bootstrap draws with seed 154. Questions sharing
a supporting filing form a cluster; multi-filing questions connect those
filings before resampling. Each draw samples clusters with replacement and
computes the question-weighted mean difference. Fewer than two clusters gives
no interval and cannot pass the positive-uncertainty gate. This can yield wide
intervals when a source covers few independent filings; report that limitation.
Filing clustering does not remove every dependence between questions about
the same company or generated from the same template.

## Result and relationship to #44

Final measurements will be recorded here after the complete frozen run. A
negative result is a valid outcome of #154. #44 remains open and still names
C0–C6; the team must agree how its final matrix represents the experimental
C5 result and defines C6. This experiment does not silently rewrite that issue.
