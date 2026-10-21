# C5 reranking experiment (#154)

The default remains C4. This experiment compares C4 with an opt-in cross-encoder
over the **same** filtered hybrid top-50 pool, returning 16 passages in both rows.
It does not add C5 to the app's selectable configurations. Adoption requires a
retrieval benefit, an answer-quality check, and teammate review.

The fresh complete study now covers **150 team + all 12,579 mechanical XBRL
questions (12,729 total)**. It supersedes the partial 2,034-question M2
reference below for #154's complete-set criterion. No smaller-cohort waiver
is used. The predeclared quality gates fail, so C4 remains the default.

## Decision criteria declared before the final run

These are practical gates for this experiment, not universal thresholds or
retrospectively chosen cutoffs:

- On each of the team benchmark and the mechanical `xbrl` set, mean nDCG@16
  must improve by at least **0.02**, with the lower bound of the paired 95%
  interval above zero. Recall@16 and MRR@16 must not regress by more than
  **0.01** on either benchmark.
- A question-type slice (overall or within a benchmark) with at least 10
  answerable questions must not lose more than **0.02** nDCG. Mean
  hard-negative accuracy must not lose more than **0.01**.
- Added steady-state reranking time (model prediction and ranking) must have a
  median at most **500 ms** and p95 at most **1,000 ms**. The extra diagnostic
  token-counting pass is excluded and recorded separately as
  `diagnostic_tokenization_ms`.
  This allows a modest extra stage in an interactive query; cold model load and
  warm-up are recorded separately.
- If these gates pass, perform a paired answer-quality comparison before
  adopting C5. Otherwise retain C4 and document the result.

The decision is fixed at @16 over a configured pool of 50 candidates in the
policy. A run or rescore at another cutoff or a reduced candidate depth records
metrics with `advance_to_answer_evaluation: null`, even if those
metrics would otherwise pass. A missing benchmark blocks advancement. The
latency gate excludes empty candidate pools; their number and the number of
timed questions are reported explicitly. An unmeasurable hard-negative gate
also blocks advancement; missing labels are not evidence that it passed.

Adoption additionally requires the declared complete cohort: 150 team and
12,579 XBRL questions, with canonical question-record SHA-256
`84763df25b8c320ef5f1a60ab8e554db35b4e959eff975bdf81fc898bfc90339`.
Counts and digest are frozen in `POLICY` before the full run. The digest
covers the questions in the order they were run, so the question files are
part of the declaration in the order the `run` command below gives them
(`benchmark/questions.jsonl` then `benchmark/generated.jsonl`). The same 12,729
questions read in the other order are a different cohort. A sample, different
question revision, different order, or legacy policy without this declaration
records `advance_to_answer_evaluation: null`, with a reason that says which
part differs: the benchmark counts, the digest, or a missing declaration.
Quality failures remain diagnostic `quality_gate_reasons`, not an adoption
decision. A new accepted benchmark requires an explicit policy update and
fresh measurement.

The team benchmark is the 150-question `benchmark/questions.jsonl`. Its
questions carry no `source` tag (#162), so they are reported as one `team`
slice; every question tagged `source: xbrl` forms the mechanical slice. The
#154 policy also gated a separate LLM-assisted slice; that split no longer
exists in the benchmark, so its gate was removed with it (#161). The
mechanical set is the frozen 12,579-question `benchmark/generated.jsonl` on
the local M1, pinned by input and canonical question-record hashes in the full
measurement manifest. The historical #172 M2 snapshot held only 5,046 facts
and generated 1,884 questions; regeneration changed 34 supporting-chunk labels.
That partial study remains separately labelled below. The full run does not
mix its labels or results with that snapshot. Generated labels use figure
substring matching and are a retrieval proxy, not exhaustively adjudicated
evidence. No model tuning or checkpoint selection uses final benchmark outcomes.

## Fixed experiment settings

The reference is `cross-encoder/ms-marco-MiniLM-L-6-v2`, pinned to revision
`233902d25c440f23af6f7d6e94d2946bac0bee0a`. Input pairs contain the full `Query.text`
and stored passage text, without extra headers or prompts. The maximum paired
input is 512 tokens, including special tokens, with the tokenizer's
`longest_first` truncation. Batch size is 32. Scores are raw logits; the query's
table boost is reapplied with the existing sign-aware score transformation.
Ties break by chunk ID. `torch.nn.Identity()` is explicit at model construction and each prediction call, including injected models, so these scores remain raw logits. There is no invented cross-encoder score floor.

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
does not load it. The paired and matrix runners use `CrossEncoderReranker`
directly to retain the exact candidate pool and diagnostics; the unused
retriever adapter has been removed.

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
- `summary.json`: paired metrics, benchmark/type/evidence-kind breakdowns, wins,
  losses, ties, cluster counts, uncertainty, truncation and timing diagnostics,
  plus the decision and any failed criteria.

Outputs stream to disk. Graceful interruptions preserve completed pairs. To
resume, repeat the identical run command with `--resume`; changed inputs,
source files, corpus, policy or configuration are refused. A forcibly killed
process can leave a gzip stream without its end marker. Resume recovers every
complete newline-terminated pair, verifies its exact benchmark prefix and
provenance, and atomically repairs the stream before appending from the first
unfinished question. A partial final row is measured again. Malformed complete
rows and corrupt streams are refused; `rescore` always reads strictly. Recovery
does not rewrite a healthy pairs file or mutate a run whose checks fail.

An interruption before the first pair may leave only a manifest; resume treats
the absent pairs file as an empty prefix. Resume also compares the original
BM25 and dense index manifests, including build identity, before loading models.
After warm-up it also requires the loaded dense encoder revision to match the
original runtime when that revision was recorded, before recording new pairs.
Each segment can supply a new workload note without replacing the original
manifest. Identical input digests may be supplied through different paths, and
an OS build-string change is allowed; paths, environment and workload are
retained in immutable runtime segments. Source, ordered input digests, corpus
digests, package/Python versions, machine architecture, settings, index identity
and loaded encoder revision must still match. Warm-up searches until a nonempty
pool exercises the cross-encoder; an entirely empty benchmark explicitly records
`cross_encoder_warmed: false`. A process-safe `run.lock` prevents concurrent writers to one run ID
and releases its OS lock on exit, including a killed process.

To recompute metrics and the decision without models or corpus files:

```bash
python -m src.evaluation.rerank_ablation rescore results/c5-reference-YYYYMMDD
```

An optional `--top-k` can rescore a smaller cutoff, up to the stored final k.
It is a sensitivity check, not a replacement for the declared @16 decision.
Incomplete runs cannot be presented as a final rescored benchmark. Scores are
recomputed from recorded ranks once per question/configuration/cutoff and reused
across metrics and slices. Connected-filing clusters are computed once per
subset, preserving the original bootstrap grouping and confidence intervals.

The summary also reports `diagnostic_tokenization_questions`,
`diagnostic_tokenization_median` and `diagnostic_tokenization_p95` under
`latency_ms`. These are recomputed from nonempty pools with a recorded
`diagnostic_tokenization_ms`. Legacy rows without that field are excluded
from this denominator; an entirely missing diagnostic has null quantiles.
The three additive fields do not change existing latency/quality values.
Rescoring old artifacts can add fields, so preserve the frozen originals.

Both ordinary rescore and table-boost sensitivity first verify the full stored
rankings, even when a smaller rescore cutoff is requested. C4 must reproduce
the saved pool order/hybrid scores; C5 must reproduce its raw logits, original
table boost and chunk-ID ties. Both rows retain exactly `min(stored_k, pool)`
passages and scores. Duplicate IDs, non-finite candidate scores and inconsistent
rankings are refused before output. Resume checks a saved prefix before repair,
runtime writes or scoring. These are consistency checks against recorded model
outputs, not protection against coordinated fabrication of outputs/provenance.

The shared shipped `rank`/`reorder` path rejects NaN and infinite table boosts
as well as nonpositive ones. The normal 1.2 boost is unchanged. Direct tests for
both paths detect restoring the old nonpositive-only check.

To isolate the final table boost without another model call:

```bash
python -m src.evaluation.rerank_ablation table-boost-sensitivity results/c5-reference-YYYYMMDD
```

This first verifies the complete benchmark digest and reproduces the recorded
C5 ranks/scores from the saved logits and original per-query boost. It then
sorts the same logits with boost 1.0, using chunk-ID ties. The separate
`table-boost-sensitivity-k16.json` contains both quality reports, original
policy and artifact hashes. It carries no adoption or latency decision and
never modifies the original pairs/config/summary. It can diagnose historical
source-split runs; normal `rescore` still refuses their obsolete decision policy.

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
intervals when a benchmark covers few independent filings; report that limitation.
Filing clustering does not remove every dependence between questions about
the same company or generated from the same template.

## Complete benchmark result and relationship to #44

**Retain C4.** The fresh native run `c5-full-20261017` completed **all
12,729 paired questions: 150 team + 12,579 mechanical XBRL**, covering all
15 companies and FY2021–2025. Both rows searched the same 28,289-passage /
75-filing corpus, from frozen source `a9f9159`. This is new retrieval and
cross-encoder inference for every nonempty pool, with no historical ranking
reuse. The frozen generated question file was not regenerated or relabelled
between measurement and reporting. Its ordered question digest, over
`benchmark/questions.jsonl` then `benchmark/generated.jsonl`, matches the
complete cohort declared in `POLICY` before the run.

This full study supersedes the partial 1,884-XBRL reference from #172 below
for #154's complete-set acceptance criterion. It requires no partial-cohort
waiver. It supports #44's C4/C5 comparison; it is not a seven-row C0–C6 matrix
or an answer-quality evaluation. The retrieval gates fail, so a paired answer
comparison and app adoption are not triggered.

For #44, the C5 row is **experimental MiniLM reranking of C4, evaluated and
rejected for default adoption**, with this full run retained as the negative
ablation and its prose gains/table regressions recorded explicitly. C6 stays as
#158 defines it, building on this rejected C5; rebasing it on C4 is tracked in
#173. The team [agreed this representation on #44](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/issues/44#issuecomment-6245852597).

| Benchmark | All / relevance-scored | Recall@16 C4 → C5 | nDCG@16 C4 → C5 | MRR@16 C4 → C5 | nDCG delta [paired 95% CI] | nDCG wins / losses / ties |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Team | 150 / 146 | 0.8185 → 0.8128 | 0.5975 → 0.5163 | 0.5485 → 0.4437 | -0.0812 [-0.1416, -0.0259] | 34 / 73 / 39 |
| Mechanical XBRL | 12,579 / 12,579 | 0.6517 → 0.6654 | 0.4719 → 0.4165 | 0.4762 → 0.3839 | -0.0554 [-0.0670, -0.0439] | 3409 / 5801 / 3369 |

The four unanswerable team questions receive no fabricated relevance score.
Hard-negative accuracy is unchanged at **0.9630 → 0.9630** over **27 labelled
questions** (one win, one loss, 25 ties). There are 67 connected filing clusters
for the team relevance metrics and 75 for XBRL. XBRL Recall@16 improves, but
its nDCG and MRR decline; additional coverage does not establish better ranking.

The full-cohort guard is satisfied, and `advance_to_answer_evaluation` is
**false**, rather than an exploratory null. The following criteria fail on
unrounded values:

- team: nDCG gain/uncertainty criterion not met
- team: mrr regression criterion not met
- xbrl: nDCG gain/uncertainty criterion not met
- xbrl: mrr regression criterion not met
- factual: nDCG regression criterion not met
- numeric: nDCG regression criterion not met
- team:factual: nDCG regression criterion not met
- team:numeric: nDCG regression criterion not met
- xbrl:numeric: nDCG regression criterion not met

| Benchmark:type | Relevance-scored | nDCG@16 C4 → C5 | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| team:factual | 44 | 0.7558 → 0.5610 | 5 / 25 / 14 |
| team:numeric | 88 | 0.5554 → 0.4934 | 23 / 46 / 19 |
| team:temporal | 10 | 0.3030 → 0.4694 | 3 / 1 / 6 |
| team:comparative | 4 | 0.5192 → 0.6452 | 3 / 1 / 0 |
| xbrl:numeric | 12,579 | 0.4719 → 0.4165 | 3409 / 5801 / 3369 |

Gold-evidence slices pool both benchmarks. These describe this fixed model,
candidate policy and corpus; they do not establish a weakness of every reranker:

| Gold evidence | Relevance-scored | nDCG@16 C4 → C5 | Delta [paired 95% CI] |
| --- | ---: | ---: | --- |
| prose | 879 | 0.1637 → 0.3488 | +0.1851 [+0.1614, +0.2100] |
| prose+table | 2,523 | 0.3028 → 0.3743 | +0.0715 [+0.0611, +0.0826] |
| table | 9,323 | 0.5487 → 0.4358 | -0.1129 [-0.1290, -0.0976] |

Only **1 of 636,400 candidate inputs** exceeds the 512-token paired limit.
All **11,198 questions with gold evidence in the candidate pool** have
untruncated gold there; **1,527** have no gold in the pool, and **4** are
unlabelled. The truncated nongold passage appears once in C5's top 16 and
never in C4's. Across 203,600 gold-scope eligible top-16 passages per row,
outside-gold-company counts are 0 / 0 and outside-gold-year/company-year
counts are 35 / 34 (C4 / C5). Both rows have zero query company/year-filter
violations. These gold-scope proxies can reflect incomplete labels; they are
not answer-correctness measurements.

### Full-run timing and reproducibility

An Apple M1 / 16 GiB laptop used MPS for both encoders. No other local model
or test job ran; normal desktop apps and the Codex session remained open.
This is a recorded laptop workload, not isolated hardware throughput.
The native run completed one uninterrupted measured segment. Cached
cross-encoder load took **5.405 s**, and warm-up **3.370 s** on `daryl-01`;
both are excluded from steady-state timings.

Across **12,728 nonempty pools** (one empty pool excluded), added model
prediction/reordering took **401.0 ms median / 456.4 ms p95**. The extra
diagnostic token-counting pass is separate: **4.7 / 5.5 ms**. Median shared
C4 retrieval was **519.4 ms**; median per-question C4 retrieval plus C5
reranking was **920.4 ms**. The latency gates pass; the quality gates fail.
These are retrieval-stage measurements, not generation or full-answer latency.
They do not retrospectively correct the older diagnostic-inclusive timings or
establish a speedup across different machines/cohorts.

Both saved rankings and scores were reconstructed at the full stored depth
for all 12,729 pairs. An independent model-free rescore reproduced every
summary field, confidence interval and decision exactly. A completed native
resume verified the input/corpus/source/index identities and preserved all six
original artifact byte hashes, including the compressed pairs and summary.
All 12,729 IDs are unique. The [full summary](evaluation/c5-full-20261017-summary.json)
contains every metric/slice and denominator; the [measurement manifest](evaluation/c5-full-20261017-manifest.json)
records input/source/corpus/model/index identities, environment, hardware,
company/year coverage, runtime history, diagnostic timing and artifact hashes.
Large per-question pairs, indexes and corpus remain local and ignored.

The review changes passed **2,153 local tests** (60.53 s), plus all 115 focused
ranking/runner tests after the final candidate-depth forwarding adjustment.
[Python 3.10 and 3.12 CI both passed on `a9f9159`](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/actions/runs/39043475119).
Restoring the former `<= 0` boost checks in memory causes four new direct
NaN/+infinity tests to fail, covering both shipped ranking paths. The shared
finite-positive validation affects C4 too; valid default `1.2` behavior is
unchanged. Corruption tests reject C4/C5 order/score edits and one-/two-sided
truncation before rescore/sensitivity/resume writes. Four-question samples and
same-count/different-digest cohorts cannot produce an adoption decision.

### Full-cohort table-boost sensitivity

The model-free diagnostic reconstructs every original C5 ranking, then
removes only the final logit table boost, keeping candidates and raw logits
fixed. It supplies no new latency, answer or adoption decision:

| Gold evidence | Questions | C4 nDCG@16 | Original C5 | C5 final boost 1.0 | Change from original C5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| prose | 879 | 0.1637 | 0.3488 | 0.3700 | +0.0212 |
| prose+table | 2,523 | 0.3028 | 0.3743 | 0.3709 | -0.0034 |
| table | 9,323 | 0.5487 | 0.4358 | 0.3821 | -0.0538 |

Removing the boost worsens table-question nDCG by 0.0538, so it mitigates
rather than explains this observed table regression. It slightly improves
prose ranking. This does not select an optimal boost or isolate an intrinsic
model limitation; retrieval-stage boosts remain fixed. The [full sensitivity report](evaluation/c5-full-20261017-table-boost-sensitivity.json)
retains the original policy and artifact fingerprints. Original artifacts are
unchanged. Additive/sigmoid alternatives require a separately declared study.

## Historical partial M2 reference (#172)

**Historical partial evidence; superseded by the full run above.** The
reference run `c5-reference-20261017` covers 2,034 paired
questions (the 150 team questions and 1,884 mechanical XBRL questions) on
28,313 passages from 75 filings, and fails the predeclared retrieval gates. No
answer-quality comparison or app adoption is triggered by this result.

At the time of #172, it replaced the #154 reference `c5-reference-20261010`, which pinned the
earlier 149-question file and split the team's questions by source. That run
can't be resumed or extended: the #172 machine's corpus (28,313 passages, not
28,289) and generated XBRL set differ from it, and the runners refuse to mix
input hashes. Both studies were therefore measured again from scratch (#161).

| Benchmark | All / relevance-scored | Recall@16 C4 → C5 | nDCG@16 C4 → C5 | MRR@16 C4 → C5 | nDCG delta [paired 95% CI] | nDCG wins / losses / ties |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Team | 150 / 146 | 0.8185 → 0.8094 | 0.5978 → 0.5195 | 0.5484 → 0.4478 | -0.0783 [-0.1407, -0.0202] | 35 / 73 / 38 |
| Mechanical (`xbrl`) | 1884 / 1884 | 0.7627 → 0.8217 | 0.5504 → 0.5426 | 0.5340 → 0.4879 | -0.0078 [-0.0336, +0.0142] | 622 / 783 / 479 |

The four unanswerable team questions contribute no fabricated relevance
scores. Labelled hard-negative accuracy is unchanged at 0.9630 over 27
questions (one win, one loss). Each metric denominator, all benchmark/type and
evidence/truncation slices, and filing-cluster counts are in the
[complete summary](evaluation/c5-reference-20261017-summary.json).

Failed criteria (computed from unrounded measurements):

- team: nDCG gain/uncertainty criterion not met
- team: mrr regression criterion not met
- xbrl: nDCG gain/uncertainty criterion not met
- xbrl: mrr regression criterion not met
- factual: nDCG regression criterion not met
- team:factual: nDCG regression criterion not met
- team:numeric: nDCG regression criterion not met

| Question type (team unless noted) | Relevance-scored | nDCG@16 C4 → C5 | Wins / losses / ties |
| --- | ---: | ---: | ---: |
| Factual | 44 | 0.7580 → 0.5609 | 5 / 25 / 14 |
| Numeric | 88 | 0.5547 → 0.5030 | 25 / 45 / 18 |
| Temporal | 10 | 0.3030 → 0.4698 | 3 / 1 / 6 |
| Comparative | 4 | 0.5192 → 0.5511 | 2 / 2 / 0 |
| Numeric (`xbrl`) | 1884 | 0.5504 → 0.5426 | 622 / 783 / 479 |

| Top-16 diagnostic | C4 | C5 |
| --- | ---: | ---: |
| Table passages | 26373 | 20219 |
| Outside gold companies | 0 | 0 |
| Outside gold years | 35 | 34 |
| Outside gold company/year pairs | 35 | 34 |
| Gold-scope eligible passages | 32480 | 32480 |
| Known-year eligible passages | 32480 | 32480 |
| Query company-filter violations | 0 | 0 |
| Query year-filter violations | 0 | 0 |

None of the 101,650 scored candidate pairs exceeded the 512-token paired input
limit. 1,885 questions had untruncated gold in the pool, 145 had no gold in the
pool, and 4 are unlabelled.

Recorded added reranking time: median **279.5 ms**, p95 **442.8 ms**, over
2,033 nonempty pools; 1 empty pool is excluded. Median shared C4 retrieval is
553.9 ms; median C4 retrieval plus C5 reranking is 830.5 ms. The latency gates
passed on that recorded implementation; the quality gates do not. Those timings
include the extra diagnostic tokenization pass flagged in review, as do the
historical M1 figures (417.1 ms / 531.8 ms). They are not corrected inference-only
timings, and cannot be corrected by rescoring saved results. The patched runner
excludes that diagnostic pass; new timing evidence must be measured separately.
These are retrieval-stage measurements, not generation or full answer latency.

The Apple M2 / 8 GiB laptop used MPS for both encoders. Cached cross-encoder
loading took 9.65 s and warm-up 7.72 s, both excluded from steady-state
quantiles. Normal desktop apps and a Claude Code session shared the machine; no
other model job ran. This is recorded laptop timing, not isolated hardware
throughput, and isn't directly comparable to the #154 run's M1 timings.

The run is one uninterrupted segment on source `13ca6f5`. The
[measurement manifest](evaluation/c5-reference-20261017-manifest.json) records
all input/source/corpus fingerprints, the pinned model revision, dependency
versions, index build identities, hardware and artifact hashes. The corpus and
per-question pairs remain local to the #172 measurement machine under `data/`
and `results/`. These report files were imported from PR #172 at `8d912cd`;
they were not measured on the PR #155 review checkout.

Evidence type splits the result, as it did in #154. The pooled gold-evidence
slices across both benchmarks are:

| Gold evidence | Relevance-scored questions | nDCG@16 C4 → C5 | nDCG delta [paired 95% CI] | Wins / losses / ties |
| --- | ---: | ---: | --- | ---: |
| Prose | 113 | 0.4415 → 0.5566 | +0.1150 [+0.0317, +0.1843] | 49 / 27 / 37 |
| Prose + table | 206 | 0.3976 → 0.5100 | +0.1124 [+0.0891, +0.1415] | 125 / 53 / 28 |
| Table | 1711 | 0.5800 → 0.5436 | -0.0364 [-0.0627, -0.0152] | 483 / 776 / 452 |

C5 improves prose and mixed prose/table evidence. Table-evidence questions make
up **84.3% of the 2,030 relevance-scored questions** and drive the overall
regression; table passages in the top 16 fall from 26,373 to 20,219. The
default-adoption decision remains retain C4 under the predeclared gates, but it
does not mean reranking never helps. These are diagnostic slices, not a tested
routing policy. Reranking only prose questions or scoring tables differently
would require a separate experiment, including a routing rule that does not use
gold evidence labels at query time.

This result applies to this fixed MiniLM configuration and corpus. Generated
labels are a retrieval proxy; filing clustering does not remove all
company/template dependence, and gold scope is not exhaustive relevance. No
conclusion about answer correctness, other checkpoints or larger chunk budgets
follows from these retrieval metrics.

## Historical table-boost sensitivity (#155 review)

This model-free diagnostic uses all **12,728** saved pairs from the superseded
`c5-reference-20261010` M1 run (149 team + 12,579 XBRL), not the new #172
benchmark. Every original C5 rank/score was reconstructed exactly from the
saved logits and query-specific boost before removing the final boost.
Original artifacts remain unchanged; hashes and full grouped results are in
[the sensitivity report](evaluation/c5-reference-20261010-table-boost-sensitivity.json).

| Gold evidence | Relevance-scored | C4 nDCG@16 | Original C5 | C5 final boost 1.0 | Change from original C5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prose | 878 | 0.1636 | 0.3483 | 0.3695 | +0.0212 |
| Prose + table | 2523 | 0.3028 | 0.3743 | 0.3709 | -0.0034 |
| Table | 9323 | 0.5487 | 0.4358 | 0.3821 | -0.0538 |

Removing the final logit boost worsens table-question nDCG by 0.0538 on this
historical cohort. Thus that boost mitigated, rather than explained, the
observed table regression in this comparison. It also slightly hurt prose
retrieval. This is evidence about this fixed checkpoint, candidate pool and
corpus; it does not establish an intrinsic weakness of all MiniLM rerankers or
select an optimal table-score transformation. Candidate retrieval still uses
its original table policy. No latency or answer-quality effects are inferred,
and this diagnostic carries no adoption decision. Additive or sigmoid-based
boosts would require separately declared experiments.


## Fresh inference timing validation (#155 review)

Source `5a1c4c0` measured all **150 current team questions** against the local
28,289-passage / 75-filing corpus on an Apple M1 / 16 GiB laptop, with both
encoders on MPS. This is a team-only timing validation, not the complete
team/XBRL study. The full native study above now supplies the complete cohort. The initially sandboxed MPS load failed
before any pair; native resume used the identical manifest and completed one
measured segment. No other model or test job ran; normal desktop apps and the
Codex session remained open.

Across **149 nonempty pools** (one empty pool excluded), added inference and
reordering time was **400.5 ms median / 467.8 ms p95**. The extra diagnostic
tokenization pass was recorded separately at **4.9 ms median / 6.6 ms p95**.
Cached cross-encoder load was 5.10 s; warm-up was 3.51 s on `daryl-01`, with
`cross_encoder_warmed: true`. Load and warm-up are excluded from the quantiles.
These are recorded laptop stage timings, not full answer latency. They do not
replace or retroactively correct either original reference's timings; the
cohorts/hardware differ, so a direct speedup is not inferred.

Native model-free rescore exactly reproduced the complete summary. A completed
native resume with an absolute benchmark path and relative corpus path also
succeeded. The missing XBRL benchmark explicitly blocks advancement; this
validation is not a new adoption decision. The [portable timing report](evaluation/c5-review-20261017-team-timing.json)
records source/input/corpus/runtime identities, quantiles and original artifact
hashes; per-question files remain local.

Earlier review verification on `2ee4263`: **2,126 local tests passed** (58.85 s). The scoring
refactor exactly preserves every aggregate, confidence interval and decision
against the unoptimized #172 summarizer on all 12,728 historical pairs using
the same team/XBRL groups (5.695 s versus 1.063 s on this laptop). Test-first
regressions reproduced the latency, warm-up, unmeasurable-gate, repeated-scoring,
corpus-read and resume problems before their fixes. The strengthened warm-up
coverage exercises both fresh and resumed empty-first/all-empty cases.
