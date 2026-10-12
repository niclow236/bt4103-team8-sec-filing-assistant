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

## Measurements

The complete run, tables, hashes and interpretation are added after both phases
finish and model-free rescoring reproduces the saved aggregates. #155 supplies
the reviewed C5 experiment; this PR depends on it while it remains unmerged.
