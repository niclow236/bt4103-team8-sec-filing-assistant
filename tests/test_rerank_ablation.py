import gzip
import hashlib
import json
from copy import deepcopy

import pytest

from src.evaluation.records import BenchmarkQuestion
from src.evaluation.rerank_ablation import (
    POLICY, _clusters, main, paired_metric, read_pairs, run_pairs, summarize,
)
from src.retrieval.constants import CANDIDATE_K
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from conftest import RerankInner as Inner, RerankModel as Model, rerank_passage as passage


GATE_POLICY = POLICY | {"decision_top_k": 2}


def question(qid="q", source="handwritten", gold=("a",), kind="numeric"):
    return BenchmarkQuestion(qid, "What was revenue for AAA in FY2024?", "1 USD",
                             gold, ("c",), "AAA", 2024, kind, "medium", source)


def row(q, a=("b", "a"), b=("a", "b"), filings=("filing1",)):
    candidates = [{"chunk_id": cid, "content_type": "table" if cid == "a" else "prose",
                   "ticker": "AAA", "fiscal_year": 2024, "truncated": cid == "b"} for cid in ("a", "b", "c")]
    return {"question": q.to_dict(), "query": {"tickers": ["AAA"], "fiscal_years": [2024]},
            "filings": list(filings), "gold_filings": [["AAA", 2024]] if q.supporting_chunk_ids else [],
            "gold_content": "table" if q.supporting_chunk_ids else "none",
            "C4": {"retrieved_chunk_ids": list(a), "retrieved_scores": [1] * len(a), "latency_ms": 10},
            "C5": {"retrieved_chunk_ids": list(b), "retrieved_scores": [1] * len(b), "latency_ms": 30},
            "rerank": {"latency_ms": 20, "truncated": 1, "candidates": candidates}}


def test_paired_metric_denominators_and_bootstrap():
    rows = [row(question("q1"), filings=("f1",)), row(question("q2"), filings=("f2",)),
            row(question("q3", gold=(), kind="unanswerable"), filings=())]
    m = paired_metric(rows, "mrr", 2)
    assert m["questions"] == 2
    assert m["C4"] == .5 and m["C5"] == 1
    assert m["delta"] == .5 and m["wins"] == 2
    assert m["ci95"] == [.5, .5] and m["clusters"] == 2
    assert paired_metric(rows, "hard_negative_accuracy", 2)["questions"] == 3
    assert paired_metric(rows[:1], "mrr", 2)["ci95"] is None


def test_multifiling_connections_cluster_related_questions():
    rows = [row(question("a"), filings=("f1",)), row(question("b"), filings=("f2",)),
            row(question("c"), filings=("f1", "f2")), row(question("d"), filings=("f3",))]
    clusters = _clusters(rows)
    assert clusters[0] == clusters[1] == clusters[2] != clusters[3]
    assert paired_metric(rows, "ndcg", 2)["clusters"] == 2


def test_source_type_and_truncation_breakdowns():
    rows = [row(question("h", "handwritten")), row(question("m", "xbrl")),
            row(question("l", "llm_assisted")), row(question("u", gold=(), kind="unanswerable"))]
    report = summarize(rows, 2)
    assert report["by_source"]["handwritten"]["questions"] == 2
    assert report["by_source"]["handwritten"]["mrr"]["questions"] == 1
    assert report["by_type"]["unanswerable"]["ndcg"]["C4"] is None
    assert report["truncation"]["rate"] == 1 / 3
    assert report["top_k_diagnostics"]["C5"]["tables"] == 4
    assert report["top_k_diagnostics"]["C5"]["outside_gold_years"] == 0
    assert not report["decision"]["advance_to_answer_evaluation"]  # only one filing per source


def test_predeclared_decision_requires_all_gates():
    rows = [row(question(source + str(i), source), filings=(str(i),))
            for source in ("handwritten", "xbrl", "llm_assisted") for i in range(2)]
    assert summarize(rows, 2, GATE_POLICY)["decision"]["advance_to_answer_evaluation"]
    negative = deepcopy(rows)
    negative[0]["C5"] = negative[0]["C4"]
    negative[1]["C5"] = negative[1]["C4"]
    assert not summarize(negative, 2, GATE_POLICY)["decision"]["advance_to_answer_evaluation"]
    for r in rows:
        r["rerank"]["latency_ms"] = 1500
    assert "reranking median latency criterion not met" in summarize(rows, 2, GATE_POLICY)["decision"]["reasons"]


def test_run_paired_pool_resume_and_rescore(tmp_path):
    qs = [question("h"), question("m", "xbrl")]
    pool = [passage("b"), passage("a", content_type="table")]
    chunks = [{"chunk_id": p.chunk_id, "accession_no": "f", "content_type": p.content_type,
               "ticker": p.ticker, "fiscal_year": p.fiscal_year} for p in pool]
    inner = Inner(pool)
    reranker = CrossEncoderReranker(RerankConfig(candidate_k=2), model=Model([1, 2]))
    out = tmp_path / "run"
    provenance = {"questions": 2, "chunk_settings": [(1800, 300)]}
    report = run_pairs(qs, inner, reranker, output_dir=out, provenance=provenance, chunks=chunks, top_k=1)
    pairs = read_pairs(out / "pairs.jsonl.gz")
    assert [r["C4"]["retrieved_chunk_ids"] for r in pairs] == [["b"], ["b"]]
    assert [r["C5"]["retrieved_chunk_ids"] for r in pairs] == [["a"], ["a"]]
    assert len(inner.calls) == 3  # warmup plus one pool per question
    assert pairs[0]["rerank"]["candidates"][1]["sources"] == ["bm25", "dense"]
    main(["rescore", str(out)])
    rescored = json.loads((out / "rescored-k1.json").read_text())
    assert rescored == report
    # Retain only a completed prefix, as when interrupted between questions.
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(pairs[0]) + "\n")
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])
    run_pairs(qs, inner, reranker, output_dir=out, provenance=provenance, chunks=chunks, top_k=1, resume=True)
    assert len(read_pairs(out / "pairs.jsonl.gz")) == 2
    with pytest.raises(ValueError, match="identical"):
        run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 3}, chunks=chunks, top_k=1, resume=True)
    with pytest.raises(FileExistsError):
        run_pairs(qs, inner, reranker, output_dir=out, provenance=provenance, chunks=chunks, top_k=1)


def test_invalid_inputs_do_not_measure_or_create_output(tmp_path):
    out = tmp_path / "run"
    r = CrossEncoderReranker(RerankConfig(candidate_k=2), model=Model([1]))
    inner = Inner([])
    for qs, k in [([], 1), ([question(), question()], 1), ([question()], 0), ([question()], 3)]:
        with pytest.raises(ValueError):
            run_pairs(qs, inner, r, output_dir=out, provenance={}, chunks=[], top_k=k)
    assert not out.exists() and not inner.calls


def test_cli_invalid_k_and_run_id_do_not_load_models(tmp_path):
    with pytest.raises(SystemExit):
        main(["run", "missing.jsonl", "--run-id", "x", "--top-k", "0", "--workload-note", "test"])
    with pytest.raises(ValueError):
        main(["run", "missing.jsonl", "--run-id", "../bad", "--workload-note", "test"])


def passing_rows():
    return [row(question(source + str(i), source), filings=(source + str(i),))
            for source in ("handwritten", "xbrl", "llm_assisted") for i in range(2)]


def test_each_gate_blocks_on_its_own():
    assert summarize(passing_rows(), 2, GATE_POLICY)["decision"]["reasons"] == []

    regressed = passing_rows()
    for r in regressed[4:]:  # the llm_assisted rows
        r["C4"], r["C5"] = r["C5"], r["C4"]
    assert summarize(regressed, 2, GATE_POLICY)["decision"]["reasons"] == [
        "llm_assisted: nDCG regression criterion not met"]

    assert summarize(passing_rows()[:4], 2, GATE_POLICY)["decision"]["reasons"] == ["missing source llm_assisted"]

    typed = passing_rows() + [
        row(question(f"t{i}", "other", kind="temporal"), a=("a", "b"), b=("b", "a"), filings=(f"t{i}",))
        for i in range(POLICY["min_type_questions"])]
    assert summarize(typed, 2, GATE_POLICY)["decision"]["reasons"] == [
        "temporal: nDCG regression criterion not met", "other:temporal: nDCG regression criterion not met"]

    negatives = passing_rows() + [
        row(question(f"u{i}", "other", gold=(), kind="unanswerable"), a=("a", "b"), b=("c", "b"),
            filings=()) for i in range(2)]
    assert summarize(negatives, 2, GATE_POLICY)["decision"]["reasons"] == [
        "hard-negative accuracy regression criterion not met"]


def test_gold_filing_diagnostics_count_other_years():
    rows = passing_rows()
    rows[0]["rerank"]["candidates"][0]["fiscal_year"] = 2023  # chunk "a", ranked by both
    diagnostics = summarize(rows, 2)["top_k_diagnostics"]
    assert diagnostics["C4"]["outside_gold_years"] == diagnostics["C5"]["outside_gold_years"] == 1
    assert diagnostics["C4"]["outside_gold_tickers"] == 0
    del rows[1]["gold_filings"]
    assert summarize(rows, 2)["top_k_diagnostics"]["C4"]["outside_gold_years"] is None


def completed_run(tmp_path, provenance=None):
    qs = [question("h"), question("m", "xbrl")]
    pool = [passage("b"), passage("a", content_type="table")]
    chunks = [{"chunk_id": p.chunk_id, "accession_no": "f", "content_type": p.content_type,
               "ticker": p.ticker, "fiscal_year": p.fiscal_year} for p in pool]
    reranker = CrossEncoderReranker(RerankConfig(candidate_k=2), model=Model([1, 2]))
    out = tmp_path / "run"
    run_pairs(qs, Inner(pool), reranker, output_dir=out, chunks=chunks, top_k=2,
              provenance=provenance or {"questions": 2})
    return qs, pool, chunks, reranker, out


def test_rescore_at_another_k_carries_no_decision(tmp_path):
    out = completed_run(tmp_path)[-1]
    main(["rescore", str(out), "--top-k", "1"])
    decision = json.loads((out / "rescored-k1.json").read_text())["decision"]
    assert decision["advance_to_answer_evaluation"] is None
    main(["rescore", str(out)])
    # A stored exploratory run at @2 also cannot replace the declared @16 gate.
    assert json.loads((out / "rescored-k2.json").read_text())["decision"]["advance_to_answer_evaluation"] is None


def test_rescore_refuses_questions_that_differ_from_the_frozen_digest(tmp_path):
    digest = hashlib.sha256(json.dumps(
        [question("h").to_dict(), question("m", "xbrl").to_dict()], sort_keys=True).encode()).hexdigest()
    out = completed_run(tmp_path, {"questions": 2, "question_records_sha256": digest})[-1]
    main(["rescore", str(out)])
    pairs = read_pairs(out / "pairs.jsonl.gz")
    pairs[0]["question"]["question"] = "Edited after the run?"
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(p) + "\n" for p in pairs)
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])


def test_resume_after_interruption_before_the_first_pair(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    (out / "pairs.jsonl.gz").unlink()
    run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    assert len(read_pairs(out / "pairs.jsonl.gz")) == 2


def test_candidate_pool_deeper_than_c4_is_refused(tmp_path):
    reranker = CrossEncoderReranker(RerankConfig(candidate_k=CANDIDATE_K + 1), model=Model([1]))
    with pytest.raises(ValueError, match="C4 depth"):
        run_pairs([question()], Inner([]), reranker, output_dir=tmp_path / "run",
                  provenance={}, chunks=[], top_k=1)
    assert not (tmp_path / "run").exists()


def test_cli_rejects_deeper_pool_before_reading_questions():
    with pytest.raises(SystemExit):
        main(["run", "missing.jsonl", "--run-id", "x", "--candidate-k", str(CANDIDATE_K + 1),
              "--workload-note", "test"])


def test_smaller_cutoff_cannot_pass_the_declared_final_gate():
    decision = summarize(passing_rows(), 2)["decision"]
    assert decision["advance_to_answer_evaluation"] is None
    assert decision["reasons"] == [f"sensitivity rescore; decided at k={POLICY['decision_top_k']}"]


def test_legacy_cutoff_does_not_follow_a_retuned_app_default(monkeypatch):
    legacy_policy = {key:value for key,value in POLICY.items() if key != "decision_top_k"}
    monkeypatch.setattr("src.evaluation.rerank_ablation.FINAL_K", 1)
    assert summarize(passing_rows(), 1, legacy_policy)["decision"]["advance_to_answer_evaluation"] is None
    assert summarize(passing_rows(), 16, legacy_policy)["decision"]["advance_to_answer_evaluation"] is True


def test_empty_pools_do_not_dilute_reranking_latency():
    rows = passing_rows()
    for r in rows:
        r["rerank"]["latency_ms"] = 600
    empty = row(question("empty", gold=(), kind="unanswerable"), a=(), b=(), filings=())
    empty["rerank"] = {"latency_ms": 0, "candidates": [], "truncated": 0}
    report = summarize(rows + [deepcopy(empty) for _ in range(100)], 2, GATE_POLICY)
    assert report["latency_ms"]["rerank_median"] == 600
    assert report["latency_ms"]["rerank_p95"] == 600
    assert report["latency_ms"]["rerank_questions"] == 6
    assert report["latency_ms"]["empty_pool_questions"] == 100
    assert "reranking median latency criterion not met" in report["decision"]["reasons"]


def test_gold_truncation_slice_uses_supporting_passages_in_the_pool():
    rows = [row(question("short")), row(question("long")),
            row(question("missing", gold=("outside",))),
            row(question("unlabelled", gold=(), kind="unanswerable"))]
    rows[1]["rerank"]["candidates"][0]["truncated"] = True
    report = summarize(rows, 2)
    groups = report["by_gold_truncation"]
    assert {k: v["questions"] for k, v in groups.items()} == {
        "gold_in_pool_untruncated": 1, "gold_in_pool_truncated": 1,
        "no_gold_in_pool": 1, "unlabelled": 1}
    assert groups["gold_in_pool_truncated"]["mrr"]["delta"] == .5


def test_gold_scope_pairs_and_unknown_years():
    r = row(question())
    r["gold_filings"] = [["AAA", 2023], ["BBB", 2024]]
    d = summarize([r], 2)["top_k_diagnostics"]["C4"]
    assert d["outside_gold_tickers"] == d["outside_gold_years"] == 0
    assert d["outside_gold_filings"] == 2  # AAA/2024 is not either gold pair
    r["gold_filings"] = [["AAA", None]]
    d = summarize([r], 2)["top_k_diagnostics"]["C4"]
    assert d["outside_gold_years"] == d["outside_gold_filings"] == 0
    assert d["gold_scope_passages"] == 2 and d["gold_year_scope_passages"] == 0


def test_resume_preserves_each_runtime_and_allows_a_new_workload_note(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    original_runtime = (out / "runtime.json").read_bytes()
    original_config = (out / "config.json").read_bytes()
    pairs = read_pairs(out / "pairs.jsonl.gz")
    for i, note in enumerate(("quiet laptop", "browser open")):
        with gzip.open(out / "pairs.jsonl.gz", "wt", encoding="utf-8") as stream:
            stream.writelines(json.dumps(r) + "\n" for r in pairs[:i])
        run_pairs(qs, Inner(pool), reranker, output_dir=out,
                  provenance={"questions": 2, "workload_note": note}, chunks=chunks, top_k=2, resume=True)
    history = sorted(out.glob("runtime-*.json"))
    assert len(history) == 3
    assert [json.loads(p.read_text())["resumed_after_questions"] for p in history] == [0, 0, 1]
    assert [json.loads(p.read_text())["workload_note"] for p in history] == [None, "quiet laptop", "browser open"]
    assert (out / "runtime.json").read_bytes() == original_runtime
    assert (out / "config.json").read_bytes() == original_config


def test_resume_refuses_changed_index_manifests_before_loading_models(tmp_path, monkeypatch):
    from dataclasses import dataclass
    from types import SimpleNamespace

    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    @dataclass
    class Manifest:
        built_at: str

    inner = Inner(pool)
    inner.dense = SimpleNamespace(manifest=Manifest("rebuilt"))
    monkeypatch.setattr(reranker, "load", lambda: pytest.fail("must check index before model load"))
    with pytest.raises(ValueError, match="index manifests"):
        run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 2},
                  chunks=chunks, top_k=2, resume=True)


def test_concurrent_resume_is_refused_and_does_not_append(tmp_path):
    from filelock import FileLock

    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    original = (out / "pairs.jsonl.gz").read_bytes()
    with FileLock(out / "run.lock"):
        with pytest.raises(ValueError, match="already locked"):
            run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
                      chunks=chunks, top_k=2, resume=True)
    assert (out / "pairs.jsonl.gz").read_bytes() == original
    # OS lock is released; completed runs can be rescored without warming models.
    inner = Inner(pool)
    run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    assert not inner.calls


def test_rescore_at_declared_cutoff_can_advance_but_sensitivity_cannot(tmp_path):
    rows = passing_rows()
    manifest = {"top_k": 16, "policy": POLICY, "provenance": {"questions": len(rows)}}
    (tmp_path / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    with gzip.open(tmp_path / "pairs.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)
    main(["rescore", str(tmp_path)])
    assert json.loads((tmp_path / "rescored-k16.json").read_text())["decision"]["advance_to_answer_evaluation"] is True
    main(["rescore", str(tmp_path), "--top-k", "1"])
    assert json.loads((tmp_path / "rescored-k1.json").read_text())["decision"]["advance_to_answer_evaluation"] is None


def test_backfill_requires_exact_corpus_and_keeps_recorded_rankings(tmp_path, monkeypatch):
    from src.evaluation.provenance import corpus_records_sha256

    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    manifest = json.loads((out / "config.json").read_text())
    manifest["provenance"]["corpus"] = {"records_sha256": corpus_records_sha256(chunks)}
    (out / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    pairs = read_pairs(out / "pairs.jsonl.gz")
    for r in pairs:
        del r["gold_filings"]
    with gzip.open(out / "pairs.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in pairs)
    original = (out / "pairs.jsonl.gz").read_bytes()
    monkeypatch.setattr("src.evaluation.rerank_ablation.iter_chunks", lambda **kwargs: chunks)
    main(["rescore", str(out), "--processed-dir", str(tmp_path)])
    report = json.loads((out / "rescored-k2.json").read_text())
    assert report["gold_diagnostics_provenance"]["derived_rows"] == 2
    assert report["top_k_diagnostics"]["C5"]["outside_gold_years"] == 0
    assert (out / "pairs.jsonl.gz").read_bytes() == original
    chunks[0]["fiscal_year"] = 2023
    with pytest.raises(ValueError, match="exact frozen corpus"):
        main(["rescore", str(out), "--processed-dir", str(tmp_path)])
    assert (out / "pairs.jsonl.gz").read_bytes() == original
