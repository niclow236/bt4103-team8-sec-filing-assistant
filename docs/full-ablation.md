# Complete C0–C6 matrix (#44)

The experiment measures all 2,034 questions for retrieval: the 150 team
questions and 1,884 mechanical XBRL questions. Answer generation, semantic
judging and verification cover the complete 150-question team set. The team's
questions carry no `source` tag (#162), so results report one team slice
beside the mechanical set. The #172 measurement machine's facts store held
5,046 facts (built 11 September), so its local `benchmark/generated.jsonl`,
generated from it on 23 September, holds
1,884 questions, not the 12,579 a full facts store gives (see the README).
During #172 verification, regenerating it gave the same 1,884 questions, 34 of
them with different supporting chunks; the runs pin the 23 September file by hash.
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
times plus fusion overhead. C4/C5 can reuse a complete C5 reference run only
after checking the exact question sequence and parsed queries, corpus digest,
retrieval settings, library versions, index builds and loaded encoder revision.
The original source/config/pair hashes and runtime are recorded. Their timings
remain from that earlier workload, not a fresh simultaneous timing trial.
The #172 reranker timings include its extra diagnostic token-counting pass.
Current code times that pass separately, as explained in the
[C5 review report](c5-reranking.md). Reference import compares every ranking
setting and retains the reference's timing descriptor (or records that it was
unrecorded); a reporting-only latency descriptor cannot invalidate identical
saved rankings. Checkpoint/revision, depths, token limit, queries, corpus,
packages and index identities remain guarded. A fresh run must not label
imported diagnostic-inclusive timings as newly measured inference-only latency.

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

Exact reproduction of the archived #172 measurements needs its recorded source
(`13ca6f5` for retrieval and `9f29f9f` for answers), corpus, indexes and frozen
benchmark files. These live on the #172 measurement machine; this checkout's
local corpus and generated set differ. The commands below describe that
protocol. Running current code with a new run ID makes a new measurement;
source/settings guards intentionally prevent resuming the old study after
changing the code or inputs. Model-free rescoring uses the saved per-question
rankings, answers and judge outputs rather than regenerating them.

```bash
python -m src.evaluation.rerank_ablation run \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --run-id c5-reference-20261017 --top-k 16 --candidate-k 50 \
  --batch-size 32 --device mps --workload-note "..."

HF_HUB_OFFLINE=1 python -u -m src.evaluation.full_ablation retrieval \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --reference results/c5-reference-20261017 \
  --run-id full-matrix-20261017 --device mps --workload-note "..."

python -u -m src.evaluation.full_ablation answers results/full-matrix-20261017 \
  --provider mistral --model ministral-8b-2512

python -m src.evaluation.full_ablation rescore results/full-matrix-20261017
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
Root and `by-benchmark/` tables separate the team and XBRL results, with metric
denominators. Model-free `rescore` writes under `rescored/` and
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

The run `full-matrix-20261017` covers **2,034 retrieval questions in every row**
and **150 team answers in every row** (1,050 answer outputs, including
abstentions), on 28,313 passages from 75 filings. Mechanical answer quality is
not inferred. Output counts, question identities, frozen fingerprints and
model-free rescore equality were verified.

It replaces `full-matrix-20261012` (#158), which pinned the earlier
149-question file, split the team's questions into handwritten and
LLM-assisted, and was measured on a different local corpus (28,289 passages)
and XBRL set. Its resume refuses a changed input file, so this run was measured
from scratch (#161); the figures below replace its tables rather than amend
them.

| Benchmark | Row | Relevance-scored | Recall@16 | nDCG@16 | MRR@16 | Hard-negative accuracy (n) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Team | C0 | 146 | 0.1723 | 0.1004 | 0.1111 | 0.9383 (27) |
| Team | C1 | 146 | 0.3025 | 0.1474 | 0.1105 | 0.9074 (27) |
| Team | C2 | 146 | 0.4132 | 0.2610 | 0.2289 | 0.8704 (27) |
| Team | C3 | 146 | 0.4692 | 0.2612 | 0.2113 | 0.8889 (27) |
| Team | C4 | 146 | 0.8185 | 0.5978 | 0.5484 | 0.9630 (27) |
| Team | C5 | 146 | 0.8094 | 0.5195 | 0.4478 | 0.9630 (27) |
| Team | C6 | 146 | 0.8128 | 0.5160 | 0.4411 | 0.9630 (27) |
| Mechanical XBRL | C0 | 1884 | 0.0124 | 0.0064 | 0.0072 | — (0) |
| Mechanical XBRL | C1 | 1884 | 0.0838 | 0.0335 | 0.0237 | — (0) |
| Mechanical XBRL | C2 | 1884 | 0.4628 | 0.2681 | 0.2455 | — (0) |
| Mechanical XBRL | C3 | 1884 | 0.3442 | 0.1926 | 0.1783 | — (0) |
| Mechanical XBRL | C4 | 1884 | 0.7627 | 0.5504 | 0.5340 | — (0) |
| Mechanical XBRL | C5 | 1884 | 0.8217 | 0.5426 | 0.4879 | — (0) |
| Mechanical XBRL | C6 | 1884 | 0.8217 | 0.5426 | 0.4879 | — (0) |

C0 uses projected fixed-window relevance labels; compare its proxies with that
different namespace in mind. One C0 window overlapped both gold and a hard
negative and was scored as gold. C1–C6 share the original section-aware gold
IDs. The four unanswerable team questions have no relevance scores.

| Row | Team outputs | Correctness (scored/total) | Faithfulness (scored/total) | Answerability | Citation precision (n) | Citation recall (n) | Verification counts |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- |
| C0 | 150 | 0.3152 (150/150) | 0.7344 (66/150) | 0.4667 | 0.4046 (66) | 0.1523 (146) | {"not_run": 66} |
| C1 | 150 | 0.3579 (150/150) | 0.7900 (73/150) | 0.5133 | 0.4727 (73) | 0.2614 (146) | {"not_run": 73} |
| C2 | 150 | 0.5925 (150/150) | 0.8417 (121/150) | 0.8333 | 0.2791 (121) | 0.3858 (146) | {"not_run": 121} |
| C3 | 150 | 0.5969 (150/150) | 0.8508 (118/150) | 0.8133 | 0.3680 (118) | 0.4452 (146) | {"not_run": 118} |
| C4 | 150 | 0.8182 (150/150) | 0.9157 (145/150) | 0.9933 | 0.4385 (145) | 0.7945 (146) | {"not_run": 145} |
| C5 | 150 | 0.7782 (150/150) | 0.9142 (144/150) | 0.9867 | 0.4325 (144) | 0.7763 (146) | {"not_run": 144} |
| C6 | 150 | 0.7619 (150/150) | 0.9019 (144/150) | 0.9867 | 0.4289 (144) | 0.7865 (146) | {"mismatch": 22, "supported": 40, "unchecked": 28, "unverified": 54} |

Correctness and faithfulness are recorded model judgements; their denominators
expose unscored cases and abstentions. Faithfulness and citation precision have
different answered/scored subsets, so compare their coverage before
interpreting their means. Citation rates are resolved chunk-ID proxies; n is the
number of scored outputs out of 150. Verification counts annotate claims and do
not establish independent answer correctness. Token/call usage, timings and
known/unknown cost counts are in the
[answer summary](evaluation/full-matrix-20261017-answers.json).

Decomposition activates on **10 team questions**; all 1,884 mechanical
questions keep exactly the C5 ranks and scores. C6 therefore tests balanced
retrieval on that small subset and verification annotations across all 150 team
outputs. Verification does not rewrite responses; differences between
independently generated C5/C6 answers cannot be attributed to verification
itself. Hosted generation/judging variation also limits causal interpretation
of small answer-quality differences.

The [retrieval summary](evaluation/full-matrix-20261017-retrieval.json) includes
each row's metric denominators and benchmark/type breakdowns. The
[manifest](evaluation/full-matrix-20261017-manifest.json) records
source/input/corpus/index/model identities, hardware, runtime segments and
artifact hashes. Large per-question and corpus files remain local in
`results/full-matrix-20261017/` and `data/`.
Those native artifacts are on the #172 measurement machine. The committed
summaries and manifest are incorporated unchanged from #172 at `8d912cd`;
they are not represented as a new full study on the PR #158 review checkout.

Retrieval ran in one segment on source `13ca6f5`, on an Apple M2 / 8 GiB
laptop with MPS; loading and warm-up (7.1 s) are excluded. C4/C5 rankings and
their timings were imported from the [C5 reference](c5-reranking.md)
`c5-reference-20261017`, measured earlier on the same source, corpus, indexes
and loaded encoder revision; the import checks passed. C0–C3 and decomposed C6
searches were newly measured, and C6 single-filing ranks are exactly C5. The
answer phase ran on source `9f29f9f`, which only fixes the benchmark snapshot
check that refused untagged team questions; no retrieval, generation or scoring
code differs. Desktop apps, a Claude Code session and the hosted answer phase
shared the laptop, so these are recorded laptop timings rather than isolated
throughput. Replay lookup latency is not actual retrieval latency.

Three Mistral judge calls (two questions) stopped part-way and recovered under
the normal retry policy; every output was judged or scored as an abstention.

The C5 default-adoption result is the [C5 reference](c5-reranking.md)'s
decision: retain C4, with positive prose evidence and negative table evidence
explicitly preserved. C6 verification is an annotation stage, not an answer
correction algorithm. Review the full benchmark-specific results before
attributing changes to a technique or selecting a new default.

## Interpretation

Metadata filtering (C3→C4 under constant query policy) gives the largest
observed retrieval improvement: team nDCG@16 0.2612 → 0.5978 and XBRL
0.1926 → 0.5504. Dense scoring improves over sparse scoring on both benchmarks,
while the fixed hybrid weighting is not uniformly better than dense alone: it
matches dense on the team set and trails it on XBRL. C4 has the highest nDCG on
each benchmark and the highest mean judged correctness on the team set:
**0.8182**, versus **0.7782** for C5 and **0.7619** for C6. C5 raises XBRL
recall (0.7627 → 0.8217) but lowers nDCG and MRR on both benchmarks. These are
single-run model judgements, not statistically established human accuracy
differences. C5/C6 do not justify changing the default under the recorded
protocol.


## Review integration verification (#158)

PR #172 (`8d912cd`) is incorporated into the #158 feature branch alongside
all current #155 fixes (`2ee4263`). The three `full-matrix-20261017` report
artifacts remain byte-for-byte unchanged. Their benchmark hash matches the
current 150-question team file. All seven retrieval/answer counts, relevance
and correctness denominators, README table values, C4/C5 reference aggregates
and local report links were checked. The superseded full-study links are
replaced; historical C5 boost sensitivity is separately labelled.

The combined source passes **2,147 local tests** (58.31 s). A new regression
reproduced the timing-descriptor reference-import failure introduced by
combining the branches. The fix ignores only reporting-only latency text when
matching ranking settings and preserves that original text in reference
metadata; changed revision, token limit and candidate depth remain refused.

Real cached BGE/MiniLM models on the local M1/MPS ran all C0-C6 rows on seven
team/XBRL questions covering all five question types and `nicholas-21`, plus a
separate Cisco/Palo Alto FY2024 comparison that exercised two C6 filing
searches. Both runs froze source `392325a`: **8 questions / 56 retrieval
outputs**, with every saved-ranking metric recomputed exactly. Completed
native resume of the seven-question run also succeeded. No hosted answer calls
were made. The [smoke report](evaluation/full-matrix-review-20261017-smoke.json)
records identities and artifact hashes; these local integration smokes do not
replace the archived M2 study or claim new full-benchmark answer measurements.
