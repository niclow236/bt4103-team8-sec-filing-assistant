import gzip
import json
from copy import deepcopy

import pytest

from src.evaluation.provenance import question_records_sha256
from src.evaluation.records import BenchmarkQuestion
from src.evaluation.rerank_ablation import (
    POLICY, _clusters, main, paired_metric, read_pairs, run_pairs, summarize,
)
from src.retrieval.constants import CANDIDATE_K
from src.retrieval.rerank import CrossEncoderReranker, RerankConfig
from conftest import RerankInner as Inner, RerankModel as Model, rerank_passage as passage


GATE_POLICY = POLICY | {"decision_top_k": 2}


def question(qid="q", source=None, gold=("a",), kind="numeric"):
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


def test_benchmark_type_and_truncation_breakdowns():
    rows = [row(question("h")), row(question("m", "xbrl")),
            row(question("l")), row(question("u", gold=(), kind="unanswerable"))]
    report = summarize(rows, 2)
    assert report["by_benchmark"]["team"]["questions"] == 3
    assert report["by_benchmark"]["team"]["mrr"]["questions"] == 2
    assert report["by_type"]["unanswerable"]["ndcg"]["C4"] is None
    assert report["truncation"]["rate"] == 1 / 3
    assert report["top_k_diagnostics"]["C5"]["tables"] == 4
    assert report["top_k_diagnostics"]["C5"]["outside_gold_years"] == 0
    decision = summarize(rows, 2, GATE_POLICY)["decision"]
    assert decision["advance_to_answer_evaluation"] is False
    assert "team: nDCG gain/uncertainty criterion not met" in decision["reasons"]


def test_untagged_team_questions_form_one_slice_beside_the_xbrl_set():
    rows = [row(question("t1", None)), row(question("t2", None)), row(question("m", "xbrl"))]
    report = summarize(rows, 2)
    assert {name: group["questions"] for name, group in report["by_benchmark"].items()} == {"team": 2, "xbrl": 1}
    assert report["by_benchmark_type"]["team:numeric"]["questions"] == 2


def test_predeclared_decision_requires_all_gates():
    rows = passing_rows()
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


def passing_rows(per_benchmark=2):
    return [row(question(f"{name}{i}", source), filings=(f"{name}{i}",))
            for name, source in (("team", None), ("xbrl", "xbrl")) for i in range(per_benchmark)]


def test_unmeasurable_hard_negative_gate_blocks_an_otherwise_passing_run():
    rows = passing_rows()
    for r in rows:
        r["question"]["hard_negative_chunk_ids"] = []
    assert summarize(rows, 2, GATE_POLICY)["decision"] == {
        "advance_to_answer_evaluation": False,
        "reasons": ["hard-negative accuracy could not be measured"],
        "default": "C4", "adoption": "requires paired answer evaluation and team review"}


@pytest.mark.parametrize("all_empty", [False, True])
def test_warmup_skips_empty_pools_and_records_whether_inference_ran(tmp_path, all_empty):
    qs = [question("empty"), question("nonempty")]
    pool = [] if all_empty else [passage("a")]

    class EmptyFirst(Inner):
        def search(self, query, k=None):
            self.calls.append((query, k))
            return [] if "empty first" in query.text else pool

    from dataclasses import replace
    qs[0] = replace(qs[0], question="empty first")
    model = Model([1])
    out = tmp_path / "warmup"
    run_pairs(qs, EmptyFirst(pool), CrossEncoderReranker(model=model), output_dir=out,
              provenance={}, chunks=[{"chunk_id": "a", "accession_no": "f",
                                      "ticker": "AAA", "fiscal_year": 2024, "content_type": "prose"}],
              top_k=1)
    runtime = json.loads((out / "runtime.json").read_text())
    assert runtime["warmup_question_id"] == (None if all_empty else "nonempty")
    assert runtime["cross_encoder_warmed"] is (not all_empty)
    assert len([c for c in model.calls if c[0] == "predict"]) == (0 if all_empty else 2)
    pairs = read_pairs(out / "pairs.jsonl.gz")
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(pairs[0]) + "\n")
    model.calls.clear()
    run_pairs(qs, EmptyFirst(pool), CrossEncoderReranker(model=model), output_dir=out,
              provenance={}, chunks=[{"chunk_id": "a", "accession_no": "f",
                                      "ticker": "AAA", "fiscal_year": 2024, "content_type": "prose"}],
              top_k=1, resume=True)
    resumed = json.loads((out / "runtime-0001.json").read_text())
    assert resumed["warmup_question_id"] == (None if all_empty else "nonempty")
    assert resumed["cross_encoder_warmed"] is (not all_empty)
    assert len([c for c in model.calls if c[0] == "predict"]) == (0 if all_empty else 2)


def test_summarize_scores_each_configuration_once_and_clusters_each_group_once(monkeypatch):
    import src.evaluation.rerank_ablation as module
    rows = passing_rows()
    real_scores, real_clusters = module._scores, module._clusters
    score_calls, cluster_calls = [], []

    def scored(r, config, k):
        score_calls.append((r["question"]["question_id"], config, k))
        return real_scores(r, config, k)

    def clustered(subset):
        cluster_calls.append(tuple(r["question"]["question_id"] for r in subset))
        return real_clusters(subset)

    monkeypatch.setattr(module, "_scores", scored)
    monkeypatch.setattr(module, "_clusters", clustered)
    report = summarize(rows, 2)
    assert len(score_calls) == len(set(score_calls)) == 2 * len(rows)
    group_count = 1 + sum(len(report[key]) for key in (
        "by_benchmark", "by_type", "by_benchmark_type", "by_gold_content", "by_gold_truncation"))
    assert len(cluster_calls) == group_count


def test_cli_invalid_resume_is_actionable_before_loading_inputs(tmp_path, monkeypatch, capsys):
    def unexpected(*args, **kwargs):
        pytest.fail("invalid resume must fail before reading inputs or loading models")
    monkeypatch.setattr("src.evaluation.rerank_ablation.load_questions", unexpected)
    for present in (False, True):
        if present:
            (tmp_path / "typo").mkdir()
        with pytest.raises(SystemExit) as error:
            main(["run", "missing.jsonl", "--run-id", "typo", "--results-root", str(tmp_path),
                  "--resume", "--workload-note", "test"])
        assert error.value.code == 2
        assert "nothing to resume" in capsys.readouterr().err


def test_resume_allows_paths_and_os_build_to_change_but_preserves_history(tmp_path):
    provenance = {"questions": 2, "inputs": [{"path": "relative.jsonl", "sha256": "same"}],
                  "corpus": {"processed_dir": "relative", "records_sha256": "same"},
                  "environment": {"platform": "old OS", "machine": "arm64", "packages": {"torch": "1"}}}
    qs, pool, chunks, reranker, out = completed_run(tmp_path, provenance)
    pairs = read_pairs(out / "pairs.jsonl.gz")
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(pairs[0]) + "\n")
    changed = deepcopy(provenance)
    changed["inputs"][0]["path"] = "/absolute/questions.jsonl"
    changed["corpus"]["processed_dir"] = "/absolute/corpus"
    changed["environment"]["platform"] = "new OS"
    run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance=changed, chunks=chunks,
              top_k=2, resume=True)
    assert len(read_pairs(out / "pairs.jsonl.gz")) == 2
    original, resumed = [json.loads((out / f"runtime-{i:04d}.json").read_text()) for i in range(2)]
    assert original["environment"]["platform"] == "old OS"
    assert resumed["environment"]["platform"] == "new OS"
    assert resumed["inputs"][0]["path"] == "/absolute/questions.jsonl"
    assert json.loads((out / "config.json").read_text())["provenance"] == provenance


@pytest.mark.parametrize("field", ["input", "corpus", "package", "machine", "python"])
def test_resume_still_refuses_changed_result_identity(tmp_path, field):
    provenance = {"questions": 2, "inputs": [{"path": "x", "sha256": "same"}],
                  "corpus": {"records_sha256": "same"},
                  "environment": {"machine": "arm64", "python": "3.13", "packages": {"torch": "1"}}}
    qs, pool, chunks, reranker, out = completed_run(tmp_path, provenance)
    changed = deepcopy(provenance)
    if field == "input":
        changed["inputs"][0]["sha256"] = "different"
    elif field == "corpus":
        changed["corpus"]["records_sha256"] = "different"
    elif field == "package":
        changed["environment"]["packages"]["torch"] = "2"
    else:
        changed["environment"][field] = "different"
    before = (out / "pairs.jsonl.gz").read_bytes()
    with pytest.raises(ValueError, match="identical"):
        run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance=changed, chunks=chunks,
                  top_k=2, resume=True)
    assert (out / "pairs.jsonl.gz").read_bytes() == before


def test_cli_loads_corpus_once_and_passes_the_identical_snapshot_to_provenance_and_runner(
    tmp_path, monkeypatch,
):
    import src.evaluation.rerank_ablation as module
    snapshots = []
    reads = []
    chunks = [{"chunk_id": "a", "accession_no": "f"}]

    def load(**kwargs):
        reads.append(kwargs)
        return iter(chunks)

    def provenance(*args, chunks):
        snapshots.append(chunks)
        return {}

    def run(*args, chunks, **kwargs):
        snapshots.append(chunks)
        return {"decision": {}}

    monkeypatch.setattr(module, "iter_chunks", load)
    monkeypatch.setattr(module, "record_provenance", provenance)
    monkeypatch.setattr(module, "run_pairs", run)
    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [question()])
    monkeypatch.setattr(module, "build_retriever", lambda *args, **kwargs: Inner([]))
    main(["run", "ignored.jsonl", "--run-id", "once", "--results-root", str(tmp_path),
          "--workload-note", "test"])
    assert len(reads) == 1
    assert snapshots[0] is snapshots[1]


def test_cli_runtime_validation_errors_are_reported_without_tracebacks(tmp_path, monkeypatch, capsys):
    import src.evaluation.rerank_ablation as module
    monkeypatch.setattr(module, "load_questions", lambda *args, **kwargs: [question()])
    monkeypatch.setattr(module, "iter_chunks", lambda **kwargs: [])
    monkeypatch.setattr(module, "record_provenance", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "build_retriever", lambda *args, **kwargs: Inner([]))
    def fail(*args, **kwargs):
        raise ValueError("resume requires identical settings")
    monkeypatch.setattr(module, "run_pairs", fail)
    with pytest.raises(SystemExit) as error:
        main(["run", "ignored.jsonl", "--run-id", "bad", "--results-root", str(tmp_path),
              "--workload-note", "test"])
    assert error.value.code == 2
    assert "resume requires identical settings" in capsys.readouterr().err


def test_table_boost_sensitivity_preserves_originals_ties_and_negative_logits():
    from src.evaluation.rerank_ablation import table_boost_sensitivity
    r = row(question(), a=("b", "a"), b=("a", "b"))
    r["query"]["table_boost"] = 1.2
    logits = {"a": -10, "b": -9, "c": -9}
    for p in r["rerank"]["candidates"]:
        p["raw_logit"] = logits[p["chunk_id"]]
    r["C5"]["retrieved_scores"] = [-10 / 1.2, -9]
    before = deepcopy(r)
    report = table_boost_sensitivity([r], 2, {"policy": POLICY, "reranker": {"candidate_k": 50}})
    assert report["original"]["by_gold_content"]["table"]["ndcg"]["C5"] == 1
    assert report["without_table_boost"]["by_gold_content"]["table"]["ndcg"]["C5"] == 0
    assert report["without_table_boost"]["overall"]["hard_negative_accuracy"]["C5"] == 0
    for name in ("original", "without_table_boost"):
        assert report[name]["decision"]["advance_to_answer_evaluation"] is None
        assert "latency_ms" not in report[name]
    assert r == before
    r["C5"]["retrieved_scores"][0] = 99
    with pytest.raises(ValueError, match="do not reproduce"):
        table_boost_sensitivity([r], 2, {"policy": POLICY, "reranker": {"candidate_k": 50}})


def test_table_boost_cli_can_diagnose_a_historical_run_without_relabelling_it(tmp_path):
    out = completed_run(tmp_path)[-1]
    manifest = json.loads((out / "config.json").read_text())
    manifest["policy"]["primary_sources"] = manifest["policy"].pop("primary_benchmarks")
    (out / "config.json").write_text(json.dumps(manifest))
    before = {name: (out / name).read_bytes() for name in ("config.json", "pairs.jsonl.gz", "summary.json")}
    main(["table-boost-sensitivity", str(out)])
    report = json.loads((out / "table-boost-sensitivity-k2.json").read_text())
    assert report["original_policy"] == manifest["policy"]
    assert report["questions"] == 2
    assert report["provenance"]["pairs_sha256"]
    assert before == {name: (out / name).read_bytes() for name in before}


def test_each_gate_blocks_on_its_own():
    assert summarize(passing_rows(), 2, GATE_POLICY)["decision"]["reasons"] == []

    regressed = passing_rows()
    for r in regressed[2:]:  # the xbrl rows
        r["C4"], r["C5"] = r["C5"], r["C4"]
    assert summarize(regressed, 2, GATE_POLICY)["decision"]["reasons"] == [
        "xbrl: nDCG gain/uncertainty criterion not met", "xbrl: mrr regression criterion not met"]

    assert summarize(passing_rows()[:2], 2, GATE_POLICY)["decision"]["reasons"] == ["missing benchmark xbrl"]

    # Enough improving team questions that the benchmark itself still passes.
    typed = passing_rows(30) + [
        row(question(f"t{i}", kind="temporal"), a=("a", "b"), b=("b", "a"), filings=(f"t{i}",))
        for i in range(POLICY["min_type_questions"])]
    assert summarize(typed, 2, GATE_POLICY)["decision"]["reasons"] == [
        "temporal: nDCG regression criterion not met", "team:temporal: nDCG regression criterion not met"]

    negatives = passing_rows() + [
        row(question(f"u{i}", gold=(), kind="unanswerable"), a=("a", "b"), b=("c", "b"),
            filings=()) for i in range(2)]
    assert summarize(negatives, 2, GATE_POLICY)["decision"]["reasons"] == [
        "hard-negative accuracy regression criterion not met"]


def test_gold_filing_diagnostics_count_other_years():
    rows = passing_rows()
    rows[0]["rerank"]["candidates"][0]["fiscal_year"] = 2023  # chunk "a", ranked by both
    rows[0]["query"]["fiscal_years"] = [2023, 2024]
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
    digest = question_records_sha256([question("h"), question("m", "xbrl")])
    out = completed_run(tmp_path, {"questions": 2, "question_records_sha256": digest})[-1]
    main(["rescore", str(out)])
    pairs = read_pairs(out / "pairs.jsonl.gz")
    pairs[0]["question"]["question"] = "Edited after the run?"
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(p) + "\n" for p in pairs)
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])


def test_rescore_accepts_untagged_team_questions_against_their_recorded_digest(tmp_path, monkeypatch):
    from src.evaluation.provenance import record_provenance
    monkeypatch.setattr("src.evaluation.provenance.iter_chunks", lambda **kwargs: [])
    qs = [question("team"), question("xbrl", "xbrl")]
    rows = [row(q, filings=(q.question_id,)) for q in qs]
    manifest = {"top_k": 2, "policy": GATE_POLICY, "reranker": {"candidate_k": 50},
                "provenance": record_provenance([], qs, tmp_path)}
    (tmp_path / "config.json").write_text(json.dumps(manifest))
    with gzip.open(tmp_path / "pairs.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)
    main(["rescore", str(tmp_path)])
    assert json.loads((tmp_path / "rescored-k2.json").read_text())["by_benchmark"]["team"]["questions"] == 1


def test_rescore_refuses_a_malformed_stored_question_as_a_changed_benchmark(tmp_path, capsys):
    digest = question_records_sha256([question("h"), question("m", "xbrl")])
    out = completed_run(tmp_path, {"questions": 2, "question_records_sha256": digest})[-1]
    pairs = read_pairs(out / "pairs.jsonl.gz")
    pairs[0]["question"]["note"] = "added after the run"
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(p) + "\n" for p in pairs)
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])
    assert "do not match the frozen benchmark" in capsys.readouterr().err


def test_rescore_refuses_a_run_recorded_under_the_source_split_policy(tmp_path, capsys):
    out = completed_run(tmp_path)[-1]
    manifest = json.loads((out / "config.json").read_text())
    policy = {k: v for k, v in manifest["policy"].items() if k != "primary_benchmarks"}
    manifest["policy"] = policy | {"primary_sources": ["handwritten", "xbrl"],
                                   "max_llm_assisted_ndcg_regression": 0.02}
    (out / "config.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])
    assert "predates" in capsys.readouterr().err
    assert not (out / "rescored-k2.json").exists()


@pytest.mark.parametrize("lose", [lambda r: r.update(question=None), lambda r: r.pop("question")],
                         ids=["null", "missing"])
def test_rescore_refuses_a_row_without_its_question_as_a_changed_benchmark(tmp_path, capsys, lose):
    digest = question_records_sha256([question("h"), question("m", "xbrl")])
    out = completed_run(tmp_path, {"questions": 2, "question_records_sha256": digest})[-1]
    pairs = read_pairs(out / "pairs.jsonl.gz")
    lose(pairs[0])
    with gzip.open(out / "pairs.jsonl.gz", "wt") as stream:
        stream.writelines(json.dumps(p) + "\n" for p in pairs)
    with pytest.raises(SystemExit):
        main(["rescore", str(out)])
    assert "do not match the frozen benchmark" in capsys.readouterr().err


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
    assert report["latency_ms"]["rerank_questions"] == 4
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
    manifest = {"top_k": 16, "policy": POLICY, "provenance": {"questions": len(rows)},
                "reranker": {"candidate_k": CANDIDATE_K}}
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


@pytest.mark.parametrize("candidate_k", [16, 32])
def test_reduced_candidate_pool_carries_no_decision(tmp_path, candidate_k):
    rows = passing_rows()
    assert summarize(rows, 16)["decision"]["advance_to_answer_evaluation"] is True
    decision = summarize(rows, 16, candidate_k=candidate_k)["decision"]
    assert decision["advance_to_answer_evaluation"] is None
    assert decision["reasons"] == [f"exploratory pool of {candidate_k}; decided over 50 candidates"]
    manifest = {"top_k": 16, "policy": POLICY, "provenance": {"questions": len(rows)},
                "reranker": {"candidate_k": candidate_k}}
    (tmp_path / "config.json").write_text(json.dumps(manifest), encoding="utf-8")
    with gzip.open(tmp_path / "pairs.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.writelines(json.dumps(r) + "\n" for r in rows)
    main(["rescore", str(tmp_path)])
    stored = json.loads((tmp_path / "rescored-k16.json").read_text())
    assert stored["candidate_k"] == candidate_k
    assert stored["decision"] == decision


def test_legacy_pool_does_not_follow_a_retuned_app_default(monkeypatch):
    legacy = {key:value for key,value in POLICY.items() if key != "decision_candidate_k"}
    monkeypatch.setattr("src.evaluation.rerank_ablation.CANDIDATE_K", 8)
    assert summarize(passing_rows(), 16, legacy, candidate_k=8)["decision"]["advance_to_answer_evaluation"] is None
    assert summarize(passing_rows(), 16, legacy, candidate_k=50)["decision"]["advance_to_answer_evaluation"] is True


def test_gold_company_scope_is_distinct_from_query_scope():
    r = row(question())
    r["query"]["tickers"] = ["AAA", "BBB"]
    r["rerank"]["candidates"][0]["ticker"] = "BBB"
    for d in summarize([r], 2)["top_k_diagnostics"].values():
        assert d["outside_gold_tickers"] == d["outside_gold_filings"] == 1
        assert d["outside_query_tickers"] == 0


def test_resume_requires_the_run_directory_and_its_runtime_record(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="existing run directory"):
        run_pairs(qs, Inner(pool), reranker, output_dir=missing, provenance={"questions": 2},
                  chunks=chunks, top_k=2, resume=True)
    assert not missing.exists()
    (out / "runtime.json").unlink()
    with pytest.raises(ValueError, match="original runtime.json"):
        run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
                  chunks=chunks, top_k=2, resume=True)


def test_present_but_unmeasurable_team_benchmark_blocks_advancement():
    rows = passing_rows()[2:] + [row(question("u", gold=(), kind="unanswerable"), filings=())]
    decision = summarize(rows, 2, GATE_POLICY)["decision"]
    assert decision["advance_to_answer_evaluation"] is False
    assert decision["reasons"] == ["team: nDCG gain/uncertainty criterion not met",
                                   "team: recall regression criterion not met",
                                   "team: mrr regression criterion not met"]


def test_resume_refuses_changed_or_unknown_dense_encoder_revision(tmp_path):
    from types import SimpleNamespace

    qs, pool, chunks, reranker, initial = completed_run(tmp_path)
    out = tmp_path / "revision"
    inner = Inner(pool)
    inner.dense = SimpleNamespace(manifest=None, runtime_info={"device": "test", "revision": "original"})
    run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 2}, chunks=chunks, top_k=2)
    pairs = read_pairs(out / "pairs.jsonl.gz")
    with gzip.open(out / "pairs.jsonl.gz", "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(pairs[0]) + "\n")
    original = (out / "pairs.jsonl.gz").read_bytes()
    for revision in ("changed", None):
        inner.dense.runtime_info["revision"] = revision
        with pytest.raises(ValueError, match="original dense encoder revision"):
            run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 2},
                      chunks=chunks, top_k=2, resume=True)
        assert (out / "pairs.jsonl.gz").read_bytes() == original
        assert len(list(out.glob("runtime-*.json"))) == 1
    inner.dense.runtime_info["revision"] = "original"
    run_pairs(qs, inner, reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    assert len(read_pairs(out / "pairs.jsonl.gz")) == 2


def test_fresh_and_completed_resume_propagate_exploratory_pool_depth(tmp_path):
    qs = [question(f"{name}{i}", source) for name, source in (("team", None), ("xbrl", "xbrl")) for i in range(2)]
    pool = [passage("b"), passage("a", content_type="table")]
    chunks = [{"chunk_id": p.chunk_id, "accession_no": "f", "content_type": p.content_type,
               "ticker": p.ticker, "fiscal_year": p.fiscal_year} for p in pool]
    reranker = CrossEncoderReranker(RerankConfig(candidate_k=16), model=Model([1, 2]))
    out = tmp_path / "small-pool"
    for resume in (False, True):
        report = run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": len(qs)},
                           chunks=chunks, top_k=16, resume=resume)
        assert report["candidate_k"] == 16
        assert report["decision"]["advance_to_answer_evaluation"] is None
        assert report["decision"]["reasons"] == ["exploratory pool of 16; decided over 50 candidates"]


@pytest.mark.parametrize("partial", ["", '{"question":', json.dumps({"unfinished": True})])
def test_resume_after_a_kill_keeps_finished_rows_and_remeasures_partial(tmp_path, partial):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    first = read_pairs(out / "pairs.jsonl.gz")[0]
    stream = gzip.open(out / "cut", "wt", encoding="utf-8")
    try:
        stream.write(json.dumps(first) + "\n" + partial)
        stream.flush()
        (out / "pairs.jsonl.gz").write_bytes((out / "cut").read_bytes())
    finally:
        stream.close()
    with pytest.raises(EOFError):
        read_pairs(out / "pairs.jsonl.gz")
    assert read_pairs(out / "pairs.jsonl.gz", recover=True) == [first]
    with pytest.raises(EOFError):
        main(["rescore", str(out)])
    run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    pairs = read_pairs(out / "pairs.jsonl.gz")
    assert [r["question"] for r in pairs] == [q.to_dict() for q in qs]
    assert pairs[0] == first
    assert json.loads((out / "runtime-0001.json").read_text())["resumed_after_questions"] == 1


def test_recovery_does_not_hide_malformed_complete_rows_or_corruption(tmp_path):
    path = tmp_path / "pairs.gz"
    with gzip.open(path, "wt") as stream:
        stream.write('{"broken":\n')
    with pytest.raises(json.JSONDecodeError):
        read_pairs(path, recover=True)
    path.write_bytes(b"not a gzip stream")
    with pytest.raises(gzip.BadGzipFile):
        read_pairs(path, recover=True)


def test_atomic_recovery_failure_preserves_the_original(tmp_path, monkeypatch):
    from src.evaluation.rerank_ablation import _repair_pairs
    path = tmp_path / "pairs.gz"
    path.write_bytes(b"original interrupted stream")
    original = path.read_bytes()
    def interrupted(*args):
        raise OSError("interrupted replacement")
    monkeypatch.setattr(type(path), "replace", interrupted)
    with pytest.raises(OSError, match="interrupted replacement"):
        _repair_pairs(path, [row(question())])
    assert path.read_bytes() == original


def test_completed_resume_does_not_rewrite_healthy_pairs(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    path = out / "pairs.jsonl.gz"
    original = path.read_bytes()
    run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    assert path.read_bytes() == original


def test_completed_resume_repairs_a_cut_stream_that_holds_every_row(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    path = out / "pairs.jsonl.gz"
    pairs = read_pairs(path)
    # All rows were flushed, but a kill prevented the gzip footer being written.
    stream = gzip.open(out / "cut", "wt", encoding="utf-8")
    try:
        stream.writelines(json.dumps(r) + "\n" for r in pairs)
        stream.flush()
        path.write_bytes((out / "cut").read_bytes())
    finally:
        stream.close()
    with pytest.raises(EOFError):
        read_pairs(path)
    run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
              chunks=chunks, top_k=2, resume=True)
    assert read_pairs(path) == pairs
    # Completion must leave a stream strict rescore can actually consume.
    main(["rescore", str(out)])
    summary = json.loads((out / "summary.json").read_text())
    assert json.loads((out / "rescored-k2.json").read_text()) == summary
    assert len(list(out.glob("runtime-*.json"))) == 1


@pytest.mark.parametrize("unfinished", ['{"unfinished":', json.dumps({"unfinished": True})])
def test_recover_reader_drops_nonterminated_line_in_a_closed_gzip(tmp_path, unfinished):
    # Unlike a cut gzip, a closed member returns its final non-newline line.
    # This directly exercises the newline safety check, not the EOF handler.
    first = row(question())
    path = tmp_path / "pairs.gz"
    with gzip.open(path, "wt", encoding="utf-8") as stream:
        stream.write(json.dumps(first) + "\n" + unfinished)
    assert read_pairs(path, recover=True) == [first]


def test_recovery_refuses_a_changed_prefix_without_rewriting_it(tmp_path):
    qs, pool, chunks, reranker, out = completed_run(tmp_path)
    first = read_pairs(out / "pairs.jsonl.gz")[0]
    first["question"]["question"] = "changed"
    stream = gzip.open(out / "cut", "wt")
    try:
        stream.write(json.dumps(first) + "\n")
        stream.flush()
        path = out / "pairs.jsonl.gz"
        path.write_bytes((out / "cut").read_bytes())
    finally:
        stream.close()
    original = path.read_bytes()
    with pytest.raises(ValueError, match="exact prefix"):
        run_pairs(qs, Inner(pool), reranker, output_dir=out, provenance={"questions": 2},
                  chunks=chunks, top_k=2, resume=True)
    assert path.read_bytes() == original
