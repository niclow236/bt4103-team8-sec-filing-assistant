import gzip
import json
from copy import deepcopy
from dataclasses import replace

import pytest

from src.evaluation.records import BenchmarkQuestion
from src.evaluation.rerank_ablation import (
    POLICY, _clusters, main, paired_metric, read_pairs, run_pairs, summarize,
)
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from test_rerank import Inner, Model, passage


def question(qid="q", source="handwritten", gold=("a",), kind="numeric"):
    return BenchmarkQuestion(qid, "What was revenue for AAA in FY2024?", "1 USD",
                             gold, ("c",), "AAA", 2024, kind, "medium", source)


def row(q, a=("b", "a"), b=("a", "b"), filings=("filing1",)):
    candidates = [{"chunk_id": cid, "content_type": "table" if cid == "a" else "prose",
                   "ticker": "AAA", "fiscal_year": 2024, "truncated": cid == "b"} for cid in ("a", "b", "c")]
    return {"question": q.to_dict(), "query": {"tickers": ["AAA"], "fiscal_years": [2024]},
            "filings": list(filings), "gold_content": "table" if q.supporting_chunk_ids else "none",
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
    assert report["top_k_diagnostics"]["C5"]["outside_query_years"] == 0
    assert not report["decision"]["advance_to_answer_evaluation"]  # only one filing per source


def test_predeclared_decision_requires_all_gates():
    rows = [row(question(source + str(i), source), filings=(str(i),))
            for source in ("handwritten", "xbrl", "llm_assisted") for i in range(2)]
    assert summarize(rows, 2)["decision"]["advance_to_answer_evaluation"]
    negative = deepcopy(rows)
    negative[0]["C5"] = negative[0]["C4"]
    negative[1]["C5"] = negative[1]["C4"]
    assert not summarize(negative, 2)["decision"]["advance_to_answer_evaluation"]
    for r in rows:
        r["rerank"]["latency_ms"] = 1500
    assert "reranking median latency criterion not met" in summarize(rows, 2)["decision"]["reasons"]
    assert POLICY["min_ndcg_gain"] == .02


def test_run_paired_pool_resume_and_rescore(tmp_path):
    qs = [question("h"), question("m", "xbrl")]
    pool = [passage("b"), passage("a", content_type="table")]
    chunks = [{"chunk_id": p.chunk_id, "accession_no": "f", "content_type": p.content_type} for p in pool]
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
