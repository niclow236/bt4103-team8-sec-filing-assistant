# C0–C6 matrix (#44)

The archived #172 matrix measures its 2,034-question retrieval cohort: the 150 team
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

The accepted C5 decision reference is now the **complete
`c5-full-20261017` study: 150 team + 12,579 mechanical questions (12,729 total)**,
reported in [the full C5 results](c5-reranking.md#complete-benchmark-result-and-relationship-to-44),
[summary](evaluation/c5-full-20261017-summary.json) and
[manifest](evaluation/c5-full-20261017-manifest.json). Team nDCG@16 falls
0.5975 → 0.5163 and XBRL 0.4719 → 0.4165; C4 remains the default.
Issue #173 rebases the matrix runner's C6 on the retained C4 stack. The tables
under **Archived #172 matrix result** remain the frozen, C5-based historical
study and are labelled as such; they are not evidence for the current C6.

The matrix tables below retain the #172 machine's original 150 + 1,884
cohort and measurements. Its C4/C5 imports belong to that frozen matrix;
they are not replaced with rows from the larger M1 study. The complete C5
study supplies the current adoption decision, while source/corpus/cohort
identity checks prevent incompatible results from being spliced together.

| Row | Evidence construction | Decomposition | Answer verification |
| --- | --- | --- | --- |
| C0 | Naive BM25 over non-overlapping 1,800-character filing windows | Off | Off |
| C1 | BM25 over section-aware passages | Off | Off |
| C2 | Dense retrieval over the same section-aware passages | Off | Off |
| C3 | Hybrid BM25/dense fusion | Off | Off |
| C4 | C3 with company/year/Item/content metadata filters | Off | Off |
| C5 | C4 with the pinned experimental MiniLM reranker | Off | Off |
| C6 | C4 with the existing company/year decomposition and claim checks | On | On |

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
retrieval settings, ranking-source hashes, library versions (including
`tokenizers`), index builds and loaded encoder revision.
`--reference-pairs-sha256` is mandatory with `--reference`; its trusted value
comes from the archived measurement manifest, not from a replacement pairs
file. The runner checks it before reading/importing rows, then reconstructs
both complete rankings/scores from the saved candidate pool and logits.
Ranking-source files must match, including retrieval modules, query parsing,
C4 evaluation query policy and stack assembly. Even a harmless edit to one
of those files requires measuring a compatible new reference. The full C5
study remains valid historical evidence at its frozen source; it cannot be
silently imported under changed ranking sources.
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

C5 alone uses `cross-encoder/ms-marco-MiniLM-L-6-v2` revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`, 512-token inputs and batches of 32.
C6 has `rerank: false`: each decomposed filing search keeps C4's first 16
filtered-hybrid passages, and the existing balanced interleave gives a shared
final budget of 16. A single-filing query has exactly the C4 ranking and score.
Decomposition is by explicit company/year, not LLM topic splitting. C6
verification matches the app's post-answer claim check and annotates claims; it
does not rewrite an answer or establish semantic correctness. All rows disable
the facts-answer shortcut and automatic refusal so those additional techniques
cannot confound the matrix.

C0 uses the existing deterministic reconstruction of filing text from processed
passages, then cuts across section boundaries. It is not a fresh parse of raw
HTML; source overlap and table headers can survive that reconstruction. Its
windows have no claimed Item/table identity. Gold IDs are projected by overlap
with their source passages; a window overlapping gold and a hard negative is
treated as gold, with the conflict count recorded. These are reproducible
relevance proxies in a different chunk namespace, not independently adjudicated
C0 evidence labels. C0 answer citations use that same projected namespace.

## Current C4-based C6 evidence (#173)

The runner now records schema `full-matrix-v2`. Its C6 is the configuration the
matrix shows going forward: C4 filtered hybrid retrieval, company/year
decomposition with balanced interleave, and the same post-answer claim
verification the app runs. Its recorded config has `use_decomposition: true`,
`use_verification: true`, and `rerank: false`. The older C5-based C6 remains only
in the explicitly archived tables below.

### Complete paired acceptance study

The fresh `c6-rebase-full-20261019` study completes the #173 acceptance run at
committed source `979add1`: **150 team + all 12,579 frozen XBRL questions
(12,729 per row)**,
plus **150 team answer evaluations for each of C4 and C6 (300 outputs)**.
The input file hashes, ordered canonical question digest, corpus records and
retrieval settings match the complete #155 study. Both rows were retrieved
fresh; no historical ranking or timing was imported. The corpus has **28,289
passages / 75 filings**, measured on Apple M1 / 16 GiB with the BGE encoder on
MPS. C6 has decomposition and verification enabled, with `rerank: false`;
no cross-encoder was constructed or warmed.

| Benchmark | Row | Recall@16 | nDCG@16 | MRR@16 | Hard-negative accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Team | C4 | 0.8185 | 0.5975 | 0.5485 | 0.9630 |
| Team | C6 | 0.8288 | 0.5973 | 0.5411 | 0.9630 |
| Team | C6 − C4 | +0.0103 | -0.0002 | -0.0074 | +0.0000 |
| XBRL | C4 | 0.6517 | 0.4719 | 0.4762 | — |
| XBRL | C6 | 0.6517 | 0.4719 | 0.4762 | — |
| XBRL | C6 − C4 | +0.0000 | +0.0000 | +0.0000 | — |

Relevance-scored denominators are **146 team** (four unanswerable questions
excluded) and **12,579 XBRL**. Hard-negative accuracy covers **27 team
questions** and has no eligible XBRL questions.

Decomposition activates on **10 team questions**. The other
**140 team + 12,579 XBRL questions** preserve C4's ranked IDs, scores,
metrics and measured latency exactly. Decomposed C6 searches use filtered
hybrid evidence per filing, followed by the app's balanced interleave; their
per-filing outputs and provenance remain in the native transactions.
On those 10 decomposed questions, recall is **0.3500 → 0.5000**, nDCG
**0.2070 → 0.2040**, and MRR **0.2314 → 0.1239**: coverage increases without
an observed ranking-quality gain on this subset.

| Row | Correctness (n/150) | Faithfulness (n/150) | Answerability | Citation precision (n) | Citation recall (n) |
| --- | --- | --- | ---: | --- | --- |
| C4 | 0.8024 (150/150) | 0.9151 (145/150) | 0.9933 | 0.4299 (145) | 0.7705 (146) |
| C6 | 0.8067 (150/150) | 0.9099 (145/150) | 0.9933 | 0.4345 (145) | 0.8014 (146) |

Each row has **145 LLM-judged outputs and five automatically scored
abstentions**; correctness is scored on all 150. Generation and separate
judging use Mistral `ministral-8b-2512`, temperature zero and rubric
`answer-quality-v1`, with the facts-answer and refusal shortcuts disabled in
both rows. These are automated single-run assessments, not human accuracy.
Conditional faithfulness/citation denominators are explicit in the table.

C6 records verification on **150 outputs**, versus **0** for C4.
The five C6 abstentions have `not_applicable` checks.
Claim-status counts and paired quality deltas for decomposed and unsplit team
questions are in the report. Verification annotates claims and does not rewrite
answers. Separate hosted generations/judgements are nondeterministic, so small
quality differences, especially on identical retrieved evidence, are not a
causal estimate of verification's benefit.

Completed retrieval resume preserved **23 original native artifact hashes**;
completed answer resume preserved all **11 checked answer/config hashes**.
Model-free CLI rescore reproduced every retrieval and answer aggregate exactly.
The original C5 reports, archived seven-row matrix and earlier team diagnostic
are byte-identical. Timing is a recorded laptop workload: hosted answer
preparation overlapped XBRL retrieval, load/warm-up is excluded, and replay
lookup time is not actual retrieval latency. Token/call accounting is recorded;
USD cost remains unknown without configured prices.

A separate diagnostic checks the fresh C4 against the trusted full #155 pairs:
**12,727 / 12,729 ranked ID lists** and **12,726 score arrays** match exactly.
The maximum absolute score difference among matching ID lists is **0.0001169**.
The fresh run's two changed ID lists are retained; the historical rankings and
timings are not substituted. The controlled C4/C6 comparison uses only this
fresh run, and the summary records the historical pairs' trusted digest.

See the [complete paired summary](evaluation/c6-rebase-full-20261019-summary.json),
[provenance and hashes](evaluation/c6-rebase-full-20261019-manifest.json), and
[report generator](../tools/report_c6_rebase.py). Native per-question evidence,
answers and runtime history stay local under
`results/c6-rebase-full-20261019/`. This is a complete **paired C4/C6 study**,
not a new measurement of C0–C3/C5 or a substitution into the archived M2 matrix.

### Earlier team-only diagnostic

A local retrieval measurement on the complete 150-question team slice
found the following effect relative to C4 (146 relevance-scored questions):

| Row | Recall@16 | nDCG@16 | MRR@16 | Hard-negative accuracy (n) |
| --- | ---: | ---: | ---: | --- |
| C4 | 0.7991 | 0.5785 | 0.5315 | 0.9630 (27) |
| C6 (C4-based) | 0.8128 | 0.5782 | 0.5227 | 0.9630 (27) |
| C6 − C4 | +0.0137 | −0.0003 | −0.0088 | 0.0000 |

Decomposition fired on 10 questions. The other 140 C6 outputs preserve C4's
ranked IDs, scores, metrics, and measured latency exactly; none of the 150 C6
rows contains reranker diagnostics. The native run keeps transactional
per-question outputs under `results/c6-rebase-team-pr182-20261019/`. Its frozen
identities, artifact hashes, aggregate results, and limitations are committed in
[the diagnostic summary](evaluation/c6-rebase-team-20261019-summary.json).

This diagnostic alone does not satisfy the complete #173 acceptance criterion.
Its measurement workspace had a different 12,591-question generated set and
did not have the accepted C5 study's frozen 12,579-question snapshot. Those
datasets must not be spliced. The complete paired study above instead uses
those frozen 150 + 12,579 questions and their matching corpus, plus hosted
generation and judging for all 150 team questions. The archived answer numbers
below remain historical C5-based C6 results and are
not attributed to the current row.

## Run and reproduce

Exact reproduction of the archived #172 measurements needs its recorded source
(`13ca6f5` for retrieval and `9f29f9f` for answers), corpus, indexes and frozen
benchmark files. These live on the #172 measurement machine; this checkout's
local corpus and generated set differ. The commands below describe a new run
with the current code and its required reference-digest flag. Exact archived
reproduction uses the CLI at its recorded source. A new run ID makes a new measurement;
source/settings guards intentionally prevent resuming the old study after
changing the code or inputs. Model-free rescoring uses the saved per-question
rankings, answers and judge outputs rather than regenerating them.

```bash
python -m src.evaluation.rerank_ablation run \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --run-id c5-reference-YYYYMMDD --top-k 16 --candidate-k 50 \
  --batch-size 32 --device mps --workload-note "..."

HF_HUB_OFFLINE=1 python -u -m src.evaluation.full_ablation retrieval \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --reference results/c5-reference-YYYYMMDD \
  --reference-pairs-sha256 "<expected pairs SHA-256 from the trusted manifest>" \
  --run-id full-matrix-YYYYMMDD --device mps --workload-note "..."

python -u -m src.evaluation.full_ablation answers results/full-matrix-YYYYMMDD \
  --provider mistral --model ministral-8b-2512

# Re-evaluate only the rebased row without regenerating C0-C5 answers.
python -u -m src.evaluation.full_ablation answers results/full-matrix-YYYYMMDD \
  --configurations C6 --provider mistral --model ministral-8b-2512

python -m src.evaluation.full_ablation rescore results/full-matrix-YYYYMMDD

# Model-free rescore when only C6 answers were measured.
python -m src.evaluation.full_ablation rescore results/full-matrix-YYYYMMDD \
  --configurations C6

# Fresh paired acceptance run: no C0-C3/C5 measurements or cross-encoder load.
HF_HUB_OFFLINE=1 python -u -m src.evaluation.full_ablation retrieval \
  benchmark/questions.jsonl benchmark/generated.jsonl \
  --configurations C4 C6 --run-id c6-rebase-full-YYYYMMDD \
  --device mps --workload-note "..."

python -u -m src.evaluation.full_ablation answers results/c6-rebase-full-YYYYMMDD \
  --provider mistral --model ministral-8b-2512

python -m src.evaluation.full_ablation rescore results/c6-rebase-full-YYYYMMDD

# Tables and charts for the report, the same ones the app's Results page shows,
# written to docs/evaluation/<run-id>/. Rerun after any re-run or rescore.
python -m src.evaluation.report results/full-matrix-YYYYMMDD
```

Retrieval `--configurations` freezes the selected rows in `config.json`; a
resume cannot add or remove rows. Without the flag all seven rows are measured
as before. Without `--reference`, a C4/C6-only run never constructs or warms a
cross-encoder. Answer
and rescore commands default to the rows whose retrieval this run froze, and
refuse requests for unmeasured rows. Their answer-summary `complete` flag means
all frozen rows have answers; the explicit configuration list distinguishes a
complete paired study from a seven-row study. Within either run, answers can
still be generated for one frozen row at a time and combined on later resumes.

Omit both `--reference` and `--reference-pairs-sha256` to measure C4/C5 again.
Use `--device cpu` without MPS and
record the hardware difference. Retrieval `--resume` requires identical source,
ordered input digests, corpus, policy, reference, Python/packages, architecture
and runtime identities. Workload notes, input/corpus path spellings and OS build
strings may change; immutable runtime segments record the current observations.
An existing canonical runtime is checked even before the first stored row;
without it, only a zero-row resume is allowed. Completed resumes also verify
the C0 baseline and index identity before exporting summaries. It recovers only an
unfinished JSONL tail, never a malformed complete row. One transaction holds
all seven rows, so a question cannot be partly present in the matrix.
Answer evaluation automatically resumes an exact team prefix under identical
source/generation/facts settings; path spellings and OS build observations are
normalized while the current values are retained in `answers-runtime-NNNN.json`.
Before model construction, C0 windows are regenerated from the verified corpus
snapshot and compared against `baseline.jsonl`, including on completed retrieval
resume and answer replay. Retrieval loads one corpus snapshot for benchmark
validation, provenance, C0 and replay maps, and checks the loaded sparse index
against it. If judging stops after answering, the pending
answer is checkpointed and only its judge is retried; its previous call usage
and timing are retained. Separate locks allow the hosted answer phase to start
once all team questions have recorded retrieval, while mechanical retrieval
continues. Completed outputs are flushed after each question.

`results/<run-id>/` contains the manifest, benchmark snapshot, actual C0 corpus,
transactional per-question matrix and runtime segments. Each `C0/`…`C6/` holds
`questions.jsonl`, retrieval summaries, `answers.jsonl` and answer summaries.
Each row records explicit `rerank`, `use_decomposition` and `use_verification`
flags, and the answer loop reads those flags from `MatrixConfig`.
Root and `by-benchmark/` tables separate the team and XBRL results, with metric
denominators. Model-free `rescore` writes under `rescored/` and
recomputes retrieval from the recorded rankings/gold IDs at the recorded cutoff,
plus answer aggregates from saved judge/citation outputs. It refuses incomplete
retrieval or team answers. It does not ask a new judge to invent past scores.

Each retrieval summary also saves `median_rank`: where the median question's
first right passage ranked. The app's Results page shows it as "Typical rank
of the first right passage" (#164). One past the cutoff is saved as `null`
beside a positive `metric_denominators.median_rank`, since JSON has no
infinity. `rescore` saves no per-question rows, so for a rescored run the page
shows only the median that `rescored/summary.json` saves. A run rescored before
#164, such as `full-matrix-20261017`, saved none, and the page shows no rank
column for it until `rescore` is run on it again. That makes no model calls.

Generation is at temperature zero with a separate structured judge call using
the versioned `answer-quality-v1` rubric. Judge scores are model assessments,
with reasons and scored/total denominators; self-judging bias and hosted-model
nondeterminism remain limitations. Citation correctness is a chunk-ID proxy.
Verification counts are claim checks, not human correctness labels. Hosted
token usage is recorded, while USD charges remain unknown without explicit
prices. Replay timing measures the answer phase over saved evidence; actual
retrieval timings are recorded separately and must not be replaced by replay
lookup times in an end-to-end latency claim.

## Archived #172 matrix result (C5-based C6; 150 team + 1,884 XBRL)

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

The [generated report](evaluation/full-matrix-20261017/ablation.md) holds these
tables with their charts, as the app's Results page shows them, written from
the native run by `python -m src.evaluation.report results/full-matrix-20261017`
(#48). The two tables below are a snapshot of it taken on 21 October 2026,
when every figure matched; the generated report is the live source. After a
re-run or rescore, regenerate it rather than editing figures here.

| Benchmark | Row | Relevance-scored | Recall@16 | nDCG@16 | MRR@16 | Hard-negative accuracy (n) |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Team | C0 | 146 | 0.1723 | 0.1004 | 0.1111 | 0.9383 (27) |
| Team | C1 | 146 | 0.3025 | 0.1474 | 0.1105 | 0.9074 (27) |
| Team | C2 | 146 | 0.4132 | 0.2610 | 0.2289 | 0.8704 (27) |
| Team | C3 | 146 | 0.4692 | 0.2612 | 0.2113 | 0.8889 (27) |
| Team | C4 | 146 | 0.8185 | 0.5978 | 0.5484 | 0.9630 (27) |
| Team | C5 | 146 | 0.8094 | 0.5195 | 0.4478 | 0.9630 (27) |
| Team | C6 (archived, C5-based) | 146 | 0.8128 | 0.5160 | 0.4411 | 0.9630 (27) |
| Mechanical XBRL | C0 | 1884 | 0.0124 | 0.0064 | 0.0072 | — (0) |
| Mechanical XBRL | C1 | 1884 | 0.0838 | 0.0335 | 0.0237 | — (0) |
| Mechanical XBRL | C2 | 1884 | 0.4628 | 0.2681 | 0.2455 | — (0) |
| Mechanical XBRL | C3 | 1884 | 0.3442 | 0.1926 | 0.1783 | — (0) |
| Mechanical XBRL | C4 | 1884 | 0.7627 | 0.5504 | 0.5340 | — (0) |
| Mechanical XBRL | C5 | 1884 | 0.8217 | 0.5426 | 0.4879 | — (0) |
| Mechanical XBRL | C6 (archived, C5-based) | 1884 | 0.8217 | 0.5426 | 0.4879 | — (0) |

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
| C6 (archived, C5-based) | 150 | 0.7619 (150/150) | 0.9019 (144/150) | 0.9867 | 0.4289 (144) | 0.7865 (146) | {"mismatch": 22, "supported": 40, "unchecked": 28, "unverified": 54} |

Correctness and faithfulness are recorded model judgements; their denominators
expose unscored cases and abstentions. Faithfulness and citation precision have
different answered/scored subsets, so compare their coverage before
interpreting their means. Citation rates are resolved chunk-ID proxies; n is the
number of scored outputs out of 150. Verification counts annotate claims and do
not establish independent answer correctness. Token/call usage, timings and
known/unknown cost counts are in the
[answer summary](evaluation/full-matrix-20261017-answers.json).

In this archived study, decomposition activates on **10 team questions**; all
1,884 mechanical questions keep exactly the C5 ranks and scores. Its C6 tests
balanced retrieval on that small subset and verification annotations across all 150 team
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
searches were newly measured, and the archived C6 single-filing ranks are
exactly C5. The
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
**0.8182**, versus **0.7782** for C5 and **0.7619** for the archived C5-based
C6. C5 raises XBRL
recall (0.7627 → 0.8217) but lowers nDCG and MRR on both benchmarks. These are
single-run model judgements, not statistically established human accuracy
differences. C5 and the archived C6 do not justify changing the default under
the recorded protocol.


## Earlier integration verification (17 October)

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

## Current review verification (18 October)

The branch includes merged #155 and #174 through main `67a7315`. The current
production source (`31373cc`) passes **2,213 local tests** (61.01 s). Command
tests exercise retrieval and answer orchestration, interrupted and zero-row
resumes, locks, frozen C0 evidence, failed-judge checkpoints and judge-only
retry, completed-checkpoint cleanup, exact prefixes and configuration flags.
Temporarily removing the baseline, zero-row runtime and trusted-reference
digest guards in memory makes five regression cases fail, confirming those
tests detect the reviewed failures. A fixed literal benchmark digest also
guards the untagged-question serialization independently of provenance code.

The same source ran real cached BGE/MiniLM models on the M1 / 16 GiB machine
with MPS, plus hosted Mistral generation/judging, on `nicholas-21` and
`jarren-16`: **14 retrieval outputs and 14 answer evaluations across C0–C6**.
Eleven outputs used the LLM judge; three abstentions used automatic checks.
One judge request retried successfully. The comparison exercised two C6
filing searches, and verification appeared only on the two C6 outputs.
The fresh reference used matching ranking sources and a trusted pairs digest.
Completed retrieval and answer resumes with changed workload notes and path
spelling preserved all **36 checked artifact hashes**. Model-free rescoring
reproduced every retrieval and answer aggregate exactly. This is command
integration validation, not a new answer-quality comparison or full matrix.

The CLI also reproduced diagnostic-tokenization aggregates from isolated
rescore copies of the original native C5 pairs: full-study nonempty denominator
**12,728**, median **4.655 ms**, p95 **5.531 ms**; earlier team timing study
denominator **149**, median **4.939 ms**, p95 **6.648 ms**. Removing the three
additive diagnostic fields leaves the complete original full-study summary
exactly unchanged. The original native runs and published historical reports
were not overwritten.

The [review validation report](evaluation/full-matrix-review-20261018.json)
records source hashes, reference and output hashes, runtime, denominators and
limitations. All three archived matrix artifacts remain byte-identical; the
150 + 1,884 matrix and the separate complete 150 + 12,579 C5 study retain their
original measurements and identities.
