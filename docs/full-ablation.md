# Complete C0–C6 matrix (#44)

The experiment measures all 12,728 questions for retrieval: 124 handwritten,
25 LLM-assisted and 12,579 mechanical XBRL questions. Answer generation,
semantic judging and verification cover the complete 149-question team set.
Mechanical labels come from the facts store, so their separate retrieval
results do not claim independent answer quality. The app continues to use C4;
these evaluation rows do not adopt the experimental reranker.

| Row | Evidence construction | Decomposition | Answer verification |
| --- | --- | --- | --- |
| C0 | Naive BM25 over non-overlapping 1,800-character filing windows | Off | Off |
| C1 | BM25 over section-aware passages | Off | Off |
| C2 | Dense retrieval over the same section-aware passages | Off | Off |
| C3 | Hybrid BM25/dense fusion | Off | Off |
| C4 | C3 with company/year/Item/content metadata filters | Off | Off |
| C5 | C4 with the pinned experimental MiniLM reranker | Off | Off |
| C6 | C5 with the existing company/year decomposition and claim checks | On | On |

C2 replaces sparse scoring with dense scoring; C3 combines both. Query parsing,
keyword trimming, figure weighting, score thresholds and the final 16-passage
budget are fixed. C0–C3 remove only metadata filters from the same parsed query;
C3→C4 therefore isolates filtering rather than changing keyword/table policy
at the same time. This controlled evaluation scope is implemented by
`MatrixConfig`; it does not redefine the app's selectable C1–C4 stacks.

C1/C2 keep the first 16 of their shared top-50 component searches. C3 fuses
those exact component results, with its timing calculated from their measured
times plus fusion overhead. C4/C5 can reuse the complete #154 reference only
after checking the exact question sequence and parsed queries, corpus digest,
retrieval settings, library versions, index builds and loaded encoder revision.
The original source/config/pair hashes and runtime are recorded. Their timings
remain from that earlier workload, not a fresh simultaneous timing trial.

C5/C6 use `cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, 512-token inputs and batches of 32.
Each decomposed C6 filing search reranks its own top-50 pool; the existing
interleave gives a shared final budget of 16. A single-filing query has exactly
the C5 ranking. Decomposition is by explicit company/year, not LLM topic
splitting. C6 verification annotates claims; it does not rewrite an answer or
establish semantic correctness. All rows disable the facts-answer shortcut and
automatic refusal so those additional techniques cannot confound the matrix.

C0 uses the existing deterministic reconstruction of filing text from processed
passages, then cuts across section boundaries. It is not a fresh parse of raw
HTML; source overlap and table headers can survive that reconstruction. Its
windows have no claimed Item/table identity. Gold IDs are projected by overlap
with their source passages; a window overlapping gold and a hard negative is
treated as gold, with the conflict count recorded. These are reproducible
relevance proxies in a different chunk namespace, not independently adjudicated
C0 evidence labels. C0 answer citations use that same projected namespace.

## Run and reproduce

```bash
HF_HUB_OFFLINE=1 python -u -m src.evaluation.full_ablation retrieval \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --reference results/c5-reference-20261010 \
  --run-id full-matrix-20261012 --device mps \
  --workload-note "Normal desktop apps; matrix retrieval and hosted answer evaluation"

python -u -m src.evaluation.full_ablation answers results/full-matrix-20261012 \
  --provider mistral --model ministral-8b-2512

python -m src.evaluation.full_ablation rescore results/full-matrix-20261012
```

Omit `--reference` to measure C4/C5 again. Use `--device cpu` without MPS and
record the hardware difference. Retrieval `--resume` requires identical source,
inputs, corpus, policy, reference and runtime identities. It recovers only an
unfinished JSONL tail, never a malformed complete row. One transaction holds
all seven rows, so a question cannot be partly present in the matrix.
Answer evaluation automatically resumes an exact team prefix under identical
source/generation/facts settings. If judging stops after answering, the pending
answer is checkpointed and only its judge is retried; its previous call usage
and timing are retained. Separate locks allow the hosted answer phase to start
once all team questions have recorded retrieval, while mechanical retrieval
continues. Completed outputs are flushed after each question.

`results/<run-id>/` contains the manifest, benchmark snapshot, actual C0 corpus,
transactional per-question matrix and runtime segments. Each `C0/`…`C6/` holds
`questions.jsonl`, retrieval summaries, `answers.jsonl` and answer summaries.
Root and `by-source/` tables separate handwritten, LLM-assisted and XBRL results,
with metric denominators. Model-free `rescore` writes under `rescored/` and
recomputes retrieval from the recorded rankings/gold IDs at the recorded cutoff,
plus answer aggregates from saved judge/citation outputs. It refuses incomplete
retrieval or team answers. It does not ask a new judge to invent past scores.

Generation is at temperature zero with a separate structured judge call using
the versioned `answer-quality-v1` rubric. Judge scores are model assessments,
with reasons and scored/total denominators; self-judging bias and hosted-model
nondeterminism remain limitations. Citation correctness is a chunk-ID proxy.
Verification counts are claim checks, not human correctness labels. Hosted
token usage is recorded, while USD charges remain unknown without explicit
prices. Replay timing measures the answer phase over saved evidence; actual
retrieval timings are recorded separately and must not be replaced by replay
lookup times in an end-to-end latency claim.

## Complete measured result

The run covers **12,728 retrieval questions in every row** and **149 team answers in every row** (1,043 answer outputs, including abstentions). Mechanical answer quality is not inferred. All output counts, question identities, frozen fingerprints and model-free rescore equality were verified.

| Source | Row | Relevance-scored | Recall@16 | nDCG@16 | MRR@16 | Hard-negative accuracy (n) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Handwritten | C0 | 123 | 0.1679 | 0.0967 | 0.1088 | 1.0000 (1) |
| Handwritten | C1 | 123 | 0.3198 | 0.1555 | 0.1160 | 1.0000 (1) |
| Handwritten | C2 | 123 | 0.4173 | 0.2642 | 0.2343 | 0.5000 (1) |
| Handwritten | C3 | 123 | 0.4756 | 0.2655 | 0.2161 | 0.5000 (1) |
| Handwritten | C4 | 123 | 0.8360 | 0.6193 | 0.5686 | 0.0000 (1) |
| Handwritten | C5 | 123 | 0.8496 | 0.5294 | 0.4466 | 0.5000 (1) |
| Handwritten | C6 | 123 | 0.8496 | 0.5248 | 0.4405 | 0.5000 (1) |
| LLM-assisted | C0 | 22 | 0.1705 | 0.1052 | 0.1168 | 0.9467 (25) |
| LLM-assisted | C1 | 22 | 0.2045 | 0.0987 | 0.0695 | 0.9400 (25) |
| LLM-assisted | C2 | 22 | 0.4091 | 0.2550 | 0.2100 | 0.8800 (25) |
| LLM-assisted | C3 | 22 | 0.4545 | 0.2298 | 0.1704 | 0.9000 (25) |
| LLM-assisted | C4 | 22 | 0.7273 | 0.4914 | 0.4565 | 1.0000 (25) |
| LLM-assisted | C5 | 22 | 0.6136 | 0.4314 | 0.4018 | 0.9800 (25) |
| LLM-assisted | C6 | 22 | 0.6136 | 0.4221 | 0.3793 | 0.9800 (25) |
| Mechanical XBRL | C0 | 12579 | 0.0235 | 0.0117 | 0.0122 | — (0) |
| Mechanical XBRL | C1 | 12579 | 0.0667 | 0.0304 | 0.0243 | — (0) |
| Mechanical XBRL | C2 | 12579 | 0.3678 | 0.2315 | 0.2350 | — (0) |
| Mechanical XBRL | C3 | 12579 | 0.2914 | 0.1712 | 0.1676 | — (0) |
| Mechanical XBRL | C4 | 12579 | 0.6517 | 0.4719 | 0.4762 | — (0) |
| Mechanical XBRL | C5 | 12579 | 0.6654 | 0.4165 | 0.3839 | — (0) |
| Mechanical XBRL | C6 | 12579 | 0.6654 | 0.4165 | 0.3839 | — (0) |

C0 uses projected fixed-window relevance labels; compare its proxies with that different namespace in mind. C1–C6 share the original section-aware gold IDs.

| Row | Team outputs | Correctness (scored/total) | Faithfulness (scored/total) | Answerability | Citation precision (n) | Citation recall (n) | Verification counts |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| C0 | 149 | 0.3185 (149/149) | 0.7230 (66/149) | 0.4698 | 0.4303 (66) | 0.1493 (145) | {"not_run": 66} |
| C1 | 149 | 0.3592 (149/149) | 0.7935 (72/149) | 0.5101 | 0.4664 (72) | 0.2678 (145) | {"not_run": 72} |
| C2 | 149 | 0.6107 (149/149) | 0.8625 (118/149) | 0.8188 | 0.3008 (118) | 0.3989 (145) | {"not_run": 118} |
| C3 | 149 | 0.5915 (149/149) | 0.8440 (118/149) | 0.8188 | 0.3656 (118) | 0.4483 (145) | {"not_run": 118} |
| C4 | 149 | 0.8019 (149/149) | 0.9149 (144/149) | 0.9933 | 0.4247 (144) | 0.7851 (145) | {"not_run": 144} |
| C5 | 149 | 0.7797 (149/149) | 0.8933 (143/149) | 0.9866 | 0.4314 (143) | 0.7828 (145) | {"not_run": 143} |
| C6 | 149 | 0.7779 (149/149) | 0.9035 (143/149) | 0.9866 | 0.4239 (143) | 0.7897 (145) | {"mismatch": 39, "supported": 43, "unchecked": 28, "unverified": 33} |

Correctness and faithfulness are recorded model judgements; their denominators expose unscored cases and abstentions. Faithfulness and citation precision have different answered/scored subsets, so compare their coverage before interpreting their means. Citation rates are resolved chunk-ID proxies; n is the number of scored outputs out of 149. Verification counts annotate claims and do not establish independent answer correctness. Separate source-level answer means, token/call usage, timings and known/unknown cost counts are in the [answer summary](evaluation/full-matrix-20261012-answers.json).

Decomposition activates on **10 team questions**; all 12,579 mechanical questions retain exactly the C5 ranks and scores. C6 therefore tests balanced retrieval on that small subset and verification annotations across all 149 team outputs. Verification does not rewrite responses; differences between independently generated C5/C6 answers cannot be attributed to verification itself. Hosted generation/judging variation also limits causal interpretation of small answer-quality differences.

The [retrieval summary](evaluation/full-matrix-20261012-retrieval.json) includes metric denominators and source/type breakdowns. The [manifest](evaluation/full-matrix-20261012-manifest.json) records source/input/corpus/index/model identities and artifact hashes. Large per-question and corpus files remain local in `results/full-matrix-20261012/` and `data/`.

Measurements ran on frozen source `10dff2e`; guarded model-free rescoring ran from the report checkout. C4/C5 rankings and original timings were imported from the independently validated #154 reference on `e74872d`. Current query/corpus/config/index/revision checks passed. C0–C3 and decomposed C6 searches were newly measured; C6 single-filing ranks are exactly C5. The laptop also ran hosted answer bookkeeping and short CPU regression checks, so these are recorded laptop timings rather than isolated throughput. Replay lookup latency is not actual retrieval latency.

The C5 default-adoption result remains the #154 decision: retain C4, with positive prose evidence and negative table evidence explicitly preserved. C6 verification is an annotation stage, not an answer correction algorithm. Review the full source-specific results before attributing changes to a technique or selecting a new default.

## Interpretation and validation

Metadata filtering (C3→C4 under constant query policy) gives the largest observed retrieval improvement. Dense scoring improves over sparse scoring on this benchmark, while the fixed hybrid weighting is not uniformly better than dense alone. C4 has the highest nDCG on each source and the highest mean judged correctness on the team set: **0.8019**, versus **0.7797** for C5 and **0.7779** for C6. These are single-run model judgements, not statistically established human accuracy differences. C5/C6 do not justify changing the default under the recorded protocol.

All **2,112 local tests pass** (58.17 s after the summary-only rescore fix). CI on Python 3.10/3.12 passes for the implementation and integrity/presentation guards; the final report commit is also checked. Real cached-model smoke covered all seven retrieval rows and a decomposition case; Mistral generation/judging ran successfully before the full team study. A retryable Mistral judge failure in C6 recovered under the normal retry policy. Correctness is scored on all 149 outputs in every row.

The measurement source remains `10dff2e`. Later guards and summary-only rescoring were developed separately; they do not alter the saved rankings, responses or scores. Final summaries are marked `evaluation_only` so the controlled query policy cannot label default app stacks as measured. This presentation annotation preserves backed-up original summary hashes and measured configuration equality. A failed initial rescore ran out of disk space while copying per-question files; only its incomplete duplicates were removed, and the final rescore writes aggregates without duplicating the measurements.

Exact reference reproduction uses source `10dff2e` plus the frozen inputs/indexes and #154 reference identified in the manifest. Current-code `rescore` reproduces the archived aggregate numbers without models. Native per-question files remain under `results/full-matrix-20261012/C0/` through `C6/`. #155 remains a dependency until merged; teammate approval is requested before merging this PR to close #44.
