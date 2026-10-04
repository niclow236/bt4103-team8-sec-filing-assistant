# PR 114 review fixes

Checked against issue [46](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/issues/46)
and all 38 inline comments from jarrenoh and niclow236 on
[PR 114](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/pull/114).
The feature branch includes main at `beb9819`.

| Review comments (discussion IDs) | Resolution and regression coverage |
|---|---|
| 4173731637, 4173731664, 4173731677, 4173731691, 4173731702 | Imports live at module scope; SimpleNamespace replaces generated classes; encoder loading no longer branches to accommodate narrow test fakes; the irrelevant evaluate patch is removed. |
| 4173731658, 4173939251 | E5 uses query and passage prefixes. Passage digests and manifest validation include the passage prefix. Real index tests check re-encoding, refusal of incorrect prefixes, and incremental reuse. |
| 4173731662, 4173731666 | C/E/G use shared run-directory validation, run-ID validation and comparison writing. JSON configuration objects remain compatible with measured_runs; E/G rows are skipped by the app. |
| 4173731669, 4173939333 | All stacks are built before measuring or allocating a run directory. A missing later index or key leaves no consumed run ID or earlier measurement. |
| 4173731674, 4173939384 | Provider runs use the stack's passage budget, score floor, decomposition, refusal and facts settings. The default provider comparison records C4 with facts disabled. Tables include provider-routed denominators, abstention, latency and retries. The CLI rejects generation --top-k rather than silently ignoring it. |
| 4173731680, 4173939019 | Invalid/taken run IDs and invalid limits fail before benchmark loading or index preparation. Index preparation is incremental; rebuilding requires an explicit flag. E1 reuses the default BGE index. |
| 4173731684, 4173731686, 4173939503 | EmbeddingConfig groups encoder settings. Legacy arguments remain supported, with explicit validation. Empty prefixes remain empty; omitted prefixes use BGE's default. Shared parts are scoped by corpus and encoder settings, preventing cross-index/model reuse. |
| 4173731694, 4173939551 | The active encoder's actual token limit is stored in the manifest, used by warnings and reports, retained on no-op builds and applied to query encoding. The table includes truncated counts and rates. Token-limit changes require rebuilding rather than relabelling vectors. |
| 4173731696, 4173731705, 4173939039, 4173939042 | Tests assert metric values for every encoder, retrieved IDs/scores, C4 settings, preflight behavior, partial reports, filters and provider statistics. A synthetic integration test runs real Chroma/dense/hybrid retrieval for all three configurations and both views. |
| 4173938981, 4173938994 | README commands name the generated benchmark, show its prerequisites, sample across the full file and pass --no-facts. Documentation distinguishes smoke samples from final measurements. |
| 4173938999, 4173939003 | Embedding queries use the C-row query helper with metadata filtering, including numeric cues. Reports retain passage IDs, scores and timed searches; index preparation time is recorded when measured. |
| 4173939007, 4173939029 | Both stopped and interrupted provider runs save completed rows, report the stop reason and write both summaries before raising. Missing keys produce short CLI errors. Successful runs print the table and path; custom-index failure advice preserves corpus and output paths. |
| 4173939013, 4173939025 | Both matrices share parts within a run, without process-global stale caches. Custom processed directories reach benchmark validation and stack assembly; preparing the fixed E-indexes is restricted to the default corpus, as the follow-up review explains below. |
| 4173939036 | Both provider rows use the application's pinned model constants. |
| 4173939450 | Each E-index is measured through filtered hybrid and filtered dense-only retrieval, with separate rows and reports so fusion cannot hide an encoder comparison. |

The follow-up review on 4 October identified four further comments:

| Review comments (discussion IDs) | Resolution and regression coverage |
|---|---|
| 4176147674, 4176147682 | CLI preparation rejects a non-default corpus before benchmark loading or index mutation. Direct calls to prepare_embedding_indexes use the same guard. A real synthetic Chroma index test checks that its vector IDs and manifest bytes remain unchanged, with and without rebuilding; a relative path resolving to the default corpus is accepted. |
| 4176147684 | Missing-index errors put the E2/E3 preparation command first, label it as applying to those indexes, and explain that subsequent advice applies to the default dense index or BM25. The suggested preparation uses the default corpus, including when an evaluation against another corpus fails. Tests check message order, both error types and shell-quoted paths. |
| 4176147689 | E2/E3 now use sibling directories data/index/chroma-E2 and data/index/chroma-E3, with sidecars outside Chroma's default directory. Registry tests verify the layout; the synthetic three-encoder integration test uses it. The README explains relocating previously built indexes and sidecars without re-encoding. |

Follow-up validation: 1,227 full-suite tests passed, 113 affected-suite tests
passed, and git diff --check passed.

Those regression tests use deterministic encoders and synthetic filings; they
do not measure real model quality. After implementation commit `73ac53b` was
pushed, each of the 28 review threads received a change explanation and was
marked resolved. PR 114 subsequently merged as `a0cb411`.

The 4 October real-data follow-up rebuilt 75 SEC 10-K filings into 28,289
passages, populated `data/raw`, `data/interim`, `data/processed`, BM25 and the
41,710-row facts store, and built all three real encoder indexes. All nine
corpus checks passed; the default dense and BM25 indexes both reported
`current`. The six embedding rows completed over the same 1,500 questions,
with full precision summaries and per-question reports under the ignored local
`results/embeddings-real-20261004/` directory. See the README's
[real corpus measurements](../README.md#real-corpus-measurements--4-october-2026)
for the tables, commands, hardware, sampling limitations and interpretation.
The current application's full suite passed **1,229 tests**.

The real Ollama G1 and hosted Mistral G2 rows each completed the same 30 sampled
questions, with facts disabled and C4's other answer settings preserved. Every
question reached its real provider with retrieved evidence; neither run stopped
or needed a retry. The combined C/E/G-format table and unchanged reports are
under `results/providers-real-20261004/`, with source runs retained under
`results/providers-local-real-20261004/` and
`results/providers-hosted-real-20261004/`. The README now records both provider
rows, latency and abstention interpretation, and numeric verification limits.
All real measurements required by issue 46 are documented in the follow-up
results PR; the issue closes when that PR is merged.
