# PR 114 review fixes

Checked against issue [46](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/issues/46)
and all 34 inline comments from jarrenoh and niclow236 on
[PR 114](https://github.com/niclow236/bt4103-team8-sec-filing-assistant/pull/114).
The feature branch includes the latest main at `4e12c91`.

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
| 4173939013, 4173939025 | Both matrices share parts within a run, without process-global stale caches. Custom processed directories reach benchmark validation, preparation and stack assembly. |
| 4173939036 | Both provider rows use the application's pinned model constants. |
| 4173939450 | Each E-index is measured through filtered hybrid and filtered dense-only retrieval, with separate rows and reports so fusion cannot hide an encoder comparison. |

Tests use deterministic encoders and synthetic filings; they do not measure real
model quality. This checkout has no processed corpus or built indexes, so real
E1–E3/G1–G2 measurements required to close issue 46 remain to be run using the
README commands. No GitHub comments or review threads have been changed.
