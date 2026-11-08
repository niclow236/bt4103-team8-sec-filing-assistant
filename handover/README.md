# Final handover archive

This archive binds the evidence behind the reported results to one Git release
tag. `HANDOVER_MANIFEST.json` names that tag and its full commit, then records
the byte size and SHA-256 of every archived file.

Contents:

- `benchmark/`: the 150 handwritten questions, the 12,579 generated XBRL
  questions, and their schema;
- `results/full-matrix-20261021/`: the native, per-question C0-C6 run whose
  hashes are recorded in the committed evaluation manifest;
- `index-manifests/`: the dense and BM25 build identities. The large index
  binaries are rebuilt from the frozen corpus and are not release assets;
- `docs/evaluation/`: the aggregate summary and provenance manifest;
- `results/README.md`: the limitations attached to each reported result.

Verify the archive from a clean checkout of the release tag before using it.
`SAFE_TAG` is the tag as it appears in the file name, with characters such as
`/` turned into `-` (`final/v1` gives `final-v1`).

```bash
python tools/archive_handover.py verify \
  sec-filing-assistant-handover-SAFE_TAG.tar.gz \
  --tag RELEASE_TAG --commit RELEASE_COMMIT
```

`tools/archive_handover.py` comes from the tagged checkout rather than from the
archive it is verifying. It uses only Python's standard library in `verify`
mode, so project dependencies do not need to be installed. It rejects extra,
missing, renamed, truncated, or changed files and unsafe archive paths, checks
the 48 native results against the archived study manifest, and requires that
manifest to match the checkout's committed copy.

The archive preserves manifests, not the built indexes. After obtaining the
matching frozen corpus, rebuild and check the indexes with:

```bash
python -m src.retrieval bm25
python -m src.retrieval embed
python -m src.retrieval check
```

See the tagged repository README for clean-checkout corpus rebuild and result
reproduction instructions. See `results/README.md` before interpreting any
metric.
