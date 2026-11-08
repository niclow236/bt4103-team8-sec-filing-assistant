# Archived results and limitations

The final release archive contains the native per-question files for
`full-matrix-20261021`; aggregate, review, and historical reports remain under
`docs/evaluation/`. Run IDs are measurement identities, not claims that two
hosted-model runs will produce byte-identical answers.

## `full-matrix-20261021` (reported final C0-C6 matrix)

The aggregate [summary](../docs/evaluation/full-matrix-20261021-summary.json),
the [provenance and native-file hashes](../docs/evaluation/full-matrix-20261021-manifest.json),
and the protocol in [the full study](../docs/full-ablation.md) belong together.
The handover archive adds the ignored `results/full-matrix-20261021/` directory
after checking every native file against that provenance manifest.

Limitations affecting these results:

- XBRL retrieval relevance is generated from facts and passage matches, not
  independently labelled by people. The 12,579 XBRL questions measure
  retrieval only; answer quality is evaluated on the 150 team questions.
- C0 reconstructs fixed-window evidence and projects gold labels into a
  different chunk namespace, so its scores are not a like-for-like test of the
  production chunker.
- Correctness and faithfulness are automated Mistral judgements, not
  human-verified accuracy. SOCLAAS generation and Mistral judging are separate
  hosted calls.
- Hosted generation and judging are nondeterministic. A rerun follows the same
  protocol but need not reproduce answer metrics byte for byte, and differences
  on identical evidence do not isolate the effect of verification.
- C6 verification labels claims but does not rewrite an answer. Its result
  cannot be interpreted as evidence that verification corrects prose.
- Retrieval timing was measured on a laptop while hosted answer work and other
  applications could overlap. It is a recorded workload, not isolated
  throughput or end-to-end demo latency.
- Archived Mistral-generator studies use another generator and different
  settings. Their differences from this SOCLAAS study cannot be attributed to
  retrieval alone.

These points are also stored as the `limitations` array in the aggregate
summary so machine readers cannot separate the figures from their caveats.

## Archived and diagnostic studies

- `full-matrix-20261017` covers 150 team plus only 1,884 XBRL questions and uses
  Mistral generation and the former C5-based C6. It is a historical partial
  study, not the final complete matrix.
- `c6-rebase-full-20261019` is a controlled C4/C6 acceptance study, not a
  seven-row replacement for the final matrix. Its hosted generations and
  judgements remain nondeterministic.
- `chunk-sweep-bm25-20261008` and `chunk-sweep-fy2025-20261008` are retrieval
  diagnostics over different corpus cuts. Their summaries are not answer
  accuracy and should not be compared directly with the final C0-C6 matrix.
