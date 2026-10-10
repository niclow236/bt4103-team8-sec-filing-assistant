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

To recompute metrics and the decision without models or corpus files:

```bash
python -m src.evaluation.rerank_ablation rescore results/c5-reference-YYYYMMDD
```

An optional `--top-k` can rescore a smaller cutoff, up to the stored final k.
It is a sensitivity check, not a replacement for the declared @16 decision.
Incomplete runs cannot be presented as a final rescored benchmark.

Recall, nDCG and MRR exclude questions without supporting IDs, with each metric's
denominator reported. Hard-negative accuracy includes only questions with
labelled hard negatives; unanswerable questions therefore receive that metric,
not fabricated zero relevance scores. Unlabelled negatives cannot measure
abstention or answer safety. Wrong-company/year diagnostics count violations
of the explicit query filters, not violations inferred from gold answer text.

Uncertainty uses 2,000 paired bootstrap draws with seed 154. Questions sharing
a supporting filing form a cluster; multi-filing questions connect those
filings before resampling. Each draw samples clusters with replacement and
computes the question-weighted mean difference. Fewer than two clusters gives
no interval and cannot pass the positive-uncertainty gate. This can yield wide
intervals when a source covers few independent filings; report that limitation.

## Result and relationship to #44

Final measurements will be recorded here after the complete frozen run. A
negative result is a valid outcome of #154. #44 remains open and still names
C0–C6; the team must agree how its final matrix represents the experimental
C5 result and defines C6. This experiment does not silently rewrite that issue.
