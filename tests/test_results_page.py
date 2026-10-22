import json
import math
import os
import pytest
from src.config import PROJECT_ROOT

from src.app.state import result_runs
from src.evaluation.report import benchmark_kind
from src.app.components import result_chart_rows


def _read_results_from(monkeypatch, root):
    """Point the page at root, preserving the targeted Reload cache clear."""
    from src.app import state
    result_runs.clear()
    reader = lambda: result_runs(root)
    reader.clear = result_runs.clear
    monkeypatch.setattr(state, "result_runs", reader)


def test_results_page_reads_saved_runs_and_keeps_question_rows_separate(tmp_path):
    run = tmp_path / "demo"
    config = run / "C4"
    config.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "run_id": "demo",
        "configurations": [{
            "config": {"id": "C4", "name": "hybrid"},
            "questions": 2,
            "recall": 0.5,
            "ndcg": 0.4,
            "mrr": 0.3,
            "hard_negative_accuracy": 0.2,
        }],
    }), encoding="utf-8")
    (config / "questions.jsonl").write_text(
        json.dumps({"question_id": "hand"})
        + "\n"
        + json.dumps({"question_id": "xbrl-apple-2024-revenue"})
        + "\n",
        encoding="utf-8",
    )

    result_runs.clear()
    runs = result_runs(tmp_path)

    assert runs[0]["run_id"] == "demo"
    groups = runs[0]["configurations"][0]["benchmark_metrics"]
    assert {kind: group["questions"] for kind, group in groups.items()} == {
        "handwritten": 1, "mechanical": 1}
    assert "question_rows" not in runs[0]["configurations"][0]
    assert "metrics" not in runs[0]["configurations"][0]
    assert benchmark_kind({"question_id": "hand"}) == "handwritten"
    assert benchmark_kind({"question_id": "xbrl-apple-2024-revenue"}) == "mechanical"
    assert benchmark_kind({"source": "llm_assisted"}) == "handwritten"


def test_results_page_reads_model_report_rows(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [{"question_id": "q1"}]}), encoding="utf-8")
    from src.evaluation.report import read_result_rows

    assert read_result_rows(report) == [{"question_id": "q1"}]


def test_old_run_metrics_are_read_from_saved_question_rows(tmp_path):
    run = tmp_path / "old" / "C4"
    run.mkdir(parents=True)
    (tmp_path / "old" / "summary.json").write_text(json.dumps({
        "run_id": "old",
        "configurations": [{"config": {"id": "C4", "name": "hybrid"}, "questions": 2}],
    }), encoding="utf-8")
    (run / "questions.jsonl").write_text(
        json.dumps({"question_id": "q1", "recall": 1.0}) + "\n"
        + json.dumps({"question_id": "q2", "recall": 0.0}) + "\n",
        encoding="utf-8",
    )
    result_runs.clear()
    item = result_runs(tmp_path)[0]["configurations"][0]
    assert item["benchmark_metrics"]["handwritten"]["recall"] == 0.5


def test_result_chart_rows_only_contains_retrieval_metrics():
    rows = result_chart_rows([{
        "config_id": "C4",
        "benchmark_metrics": {"handwritten": {
            "questions": 50, "recall": 0.8, "ndcg": 0.7, "mrr": 0.6, "median_latency_ms": 120,
        }},
    }], "handwritten")
    assert [row["Metric"] for row in rows] == ["Recall", "nDCG", "MRR"]


def test_results_ui_splits_legacy_scores_generation_and_hard_negatives(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.app import state

    run = tmp_path / "mixed"
    (run / "C4").mkdir(parents=True)
    (run / "G3").mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 0.5}}},
        {"config": {"id": "G3", "provider": "ollama"}},
    ]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "manual", "recall": 1.0, "hard_negative_accuracy": 1.0}) + "\n"
        + json.dumps({"question_id": "xbrl-legacy", "source": "handwritten",
                      "difficulty": "easy", "recall": 0.0, "hard_negative_accuracy": 0.0}),
        encoding="utf-8")
    (run / "G3" / "report.json").write_text(json.dumps({"results": [
        {"question_id": "manual", "route": "ollama",
         "answer": {"abstained": False, "latency_ms": 100}},
        {"question_id": "xbrl-fact", "route": "facts",
         "answer": {"abstained": True, "latency_ms": 10}},
    ]}), encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 2
    manual, mechanical = app.tabs
    assert manual.label == "Team benchmark"
    assert manual.dataframe[0].value["Recall"].tolist() == [1.0]
    assert mechanical.dataframe[0].value["Recall"].tolist() == [0.0]
    assert list(manual.dataframe[0].value.columns) == [
        "Configuration", "Name", "Questions", "Relevance-scored", "Recall",
        "Hard negatives", "Hard Negative Accuracy"]
    charts = app.get("vega_lite_chart")
    assert len(charts) == 2
    for chart in charts:
        encoding = json.loads(chart.proto.spec)["encoding"]
        assert "xOffset" in encoding
        assert encoding["y"]["stack"] is False
    assert mechanical.dataframe[0].value["Hard Negative Accuracy"].tolist() == [0.0]
    assert manual.dataframe[1].value["Abstention rate"].tolist() == [0.0]
    assert mechanical.dataframe[1].value["Abstention rate"].tolist() == [1.0]
    assert manual.dataframe[1].value["LLM questions"].tolist() == [1]
    assert mechanical.dataframe[1].value["LLM questions"].tolist() == [0]

    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "manual", "recall": 0.25}), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert app.tabs[0].dataframe[0].value["Recall"].tolist() == [0.25]


def test_malformed_summary_does_not_hide_valid_runs(tmp_path):
    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "summary.json").write_text("[]", encoding="utf-8")
    good = tmp_path / "good"
    good.mkdir()
    (good / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 1}}}
    ]}), encoding="utf-8")
    result_runs.clear()
    assert [run["run_id"] for run in result_runs(tmp_path)] == ["good"]


def test_hard_negative_only_run_is_visible(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.app import state
    monkeypatch.setattr(state, "result_runs", lambda: [{"run_id": "hard", "configurations": [{
        "config_id": "C4", "name": "hybrid", "questions": 1, "metrics": {},
        "benchmark_metrics": {"handwritten": {"questions": 1, "hard_negative_accuracy": 1.0}},
    }]}])
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert app.tabs[0].dataframe[0].value["Hard Negative Accuracy"].tolist() == [1.0]


def test_reload_discovers_first_run(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.app import state
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.button) == 1
    run = tmp_path / "first"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 1}}}
    ]}), encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert app.selectbox[0].value == "first"


def test_summary_only_run_says_why_it_shows_no_scores(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.app import state
    run = tmp_path / "sweep"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": "sweep", "configurations": [
        {"config": {"id": "S1200-bm25"}, "questions": 1490, "recall": 0.5},
    ]}), encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "S1200-bm25" in app.warning[0].value
    assert "questions.jsonl" in app.warning[0].value
    assert "cannot be split" in app.warning[0].value
    assert not app.dataframe and not app.get("vega_lite_chart")


def test_answer_model_only_run_shows_its_results_without_saying_there_are_none(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run = tmp_path / "answers"
    (run / "G1").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "G1", "provider": "ollama"}, "questions": 2}]}), encoding="utf-8")
    (run / "G1" / "report.json").write_text(json.dumps({"results": [
        {"question_id": "manual", "route": "ollama",
         "answer": {"abstained": False, "latency_ms": 100}},
        {"question_id": "xbrl-fact", "route": "facts",
         "answer": {"abstained": True, "latency_ms": 10}},
    ]}), encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception and not app.info
    assert app.tabs[0].dataframe[0].value["Abstention rate"].tolist() == [0.0]
    assert app.tabs[1].dataframe[0].value["Abstention rate"].tolist() == [1.0]


def test_a_stopped_or_unreadable_run_says_its_scores_are_partial(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run = tmp_path / "partial"
    (run / "G1").mkdir(parents=True)
    (run / "C4").mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "G1", "provider": "ollama"}, "questions": 1,
         "stopped": {"question_id": "q002", "error": "quota"}},
        {"config": {"id": "C4"}, "questions": 3},
    ]}), encoding="utf-8")
    (run / "G1" / "report.json").write_text(json.dumps({"results": [
        {"question_id": "q001", "route": "ollama",
         "answer": {"abstained": False, "latency_ms": 100}}]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "q001", "recall": 1.0}) + "\n"
        + json.dumps({"question_id": "q002", "recall": 1.0}) + "\n"
        + '{"question_id": "q003", "rec', encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "G1 (stopped at q002)" in app.warning[0].value
    assert "C4 (2 of 3 question rows could be read)" in app.warning[0].value


def test_rows_from_the_real_writer_split_by_their_saved_source(tmp_path):
    from src.evaluation.records import BenchmarkQuestion
    from src.evaluation.run import run_ablation
    from src.retrieval.records import RetrievedPassage

    def question(question_id, supporting, **fields):
        return BenchmarkQuestion(question_id=question_id, question="What was revenue?",
            expected_answer="$100", supporting_chunk_ids=(supporting,), hard_negative_chunk_ids=(),
            ticker="AAPL", fiscal_year=2024, question_type="numeric", **fields)

    class Retriever:
        name = "bm25"

        def search(self, query):
            return [RetrievedPassage(chunk_id="found", text="found", score=1, rank=1,
                retriever=self.name, ticker="AAPL", company="Apple", fiscal_year=2024, item="7",
                title="MD&A", url="https://example.com")]

    run_ablation([question("q001", "found", difficulty="easy", source="handwritten"),
                  question("m001", "missed", difficulty="mechanical", source="xbrl")],
                 {"bm25": Retriever()}, run_id="mixed", results_root=tmp_path, configurations=("C1",))
    result_runs.clear()
    groups = result_runs(tmp_path)[0]["configurations"][0]["benchmark_metrics"]
    assert (groups["handwritten"]["recall"], groups["mechanical"]["recall"]) == (1.0, 0.0)


@pytest.mark.parametrize("fields", [{"source": "xbrl"}, {"difficulty": "mechanical"}])
def test_benchmark_source_and_difficulty_do_not_depend_on_id_prefix(fields):
    assert benchmark_kind({"question_id": "m001", **fields}) == "mechanical"


def test_runs_are_listed_newest_first(tmp_path):
    for age, name in enumerate(["newest", "middle", "oldest"]):
        run = tmp_path / name
        run.mkdir()
        (run / "summary.json").write_text(json.dumps({"configurations": [
            {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 1}}}
        ]}), encoding="utf-8")
        os.utime(run, (1_000_000 - age, 1_000_000 - age))
    result_runs.clear()
    assert [run["run_id"] for run in result_runs(tmp_path)] == ["newest", "middle", "oldest"]


def test_two_folders_with_one_saved_run_id_can_both_be_opened(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    for name, recall in (("sweep", 1), ("sweep-copy", 0)):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps({"run_id": "sweep",
            "configurations": [{"config": {"id": "C4"},
                                "by_benchmark": {"handwritten": {"recall": recall}}}]}), encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    assert sorted(run["run_id"] for run in result_runs(tmp_path)) == ["sweep", "sweep-copy"]
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    for name, recall in (("sweep", 1), ("sweep-copy", 0)):
        app.selectbox[0].select(name).run()
        assert not app.exception
        assert app.tabs[0].dataframe[0].value["Recall"].tolist() == [recall]


def test_a_configuration_id_cannot_read_outside_its_run(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "questions.jsonl").write_text(json.dumps({"question_id": "q1", "recall": 0.5}), encoding="utf-8")
    run = tmp_path / "results" / "run"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "../../outside"}}, {"config": {"id": str(outside)}},
        {"config": {"id": "..\\..\\outside"}}, {"config": {"id": "C:\\outside"}},
        {"config": {"id": "."}}, {"config": {"id": ".."}},
    ]}), encoding="utf-8")
    result_runs.clear()
    assert result_runs(tmp_path / "results") == []


def test_reload_keeps_the_run_picked_when_a_new_run_arrives(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    def save(name, mtime):
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps({"configurations": [
            {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 1}}}]}), encoding="utf-8")
        os.utime(tmp_path / name, (mtime, mtime))

    save("older", 1000)
    save("newer", 2000)
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    app.selectbox[0].select("older").run()
    save("newest", 3000)
    app.button[0].click().run()
    assert not app.exception
    assert app.selectbox[0].options == ["newest", "newer", "older"]
    assert app.selectbox[0].value == "older"


def test_reload_only_clears_results_not_other_app_caches(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    from src.app import state
    measured = []
    monkeypatch.setattr(state, "measured_runs", lambda: measured.append(True) or [])
    state.measured.clear()
    try:
        assert state.measured() == []
        _read_results_from(monkeypatch, tmp_path)
        app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
        app.button[0].click().run()
        assert not app.exception
        assert state.measured() == []
        assert measured == [True]
    finally:
        state.measured.clear()


def test_reload_falls_back_when_the_selected_run_is_removed(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    for name, mtime in (("older", 1000), ("newer", 2000)):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "summary.json").write_text(json.dumps({"configurations": [
            {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 1}}}]}),
            encoding="utf-8")
        os.utime(folder, (mtime, mtime))
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    app.selectbox[0].select("older").run()
    (tmp_path / "older" / "summary.json").unlink()
    (tmp_path / "older").rmdir()
    app.button[0].click().run()
    assert not app.exception
    assert app.selectbox[0].value == "newer"


def test_matrix_answer_quality_shows_on_the_team_tab(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run = tmp_path / "matrix"
    (run / "C4").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4", "name": "filtered hybrid"}, "questions": 1}]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "q1", "recall": 1.0}) + "\n", encoding="utf-8")
    (run / "answer-summary.json").write_text(json.dumps({"configurations": {"C4": {
        "questions": 150, "checks": {"not_run": 145},
        "quality": {"correctness": {"mean": 0.8182, "scored": 150, "total": 150}}}}}),
        encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    team, mechanical = app.tabs
    assert "Answer quality" in [header.value for header in team.subheader]
    quality = team.dataframe[1].value
    assert quality["Correctness"].tolist() == [0.8182]
    assert quality["Correctness scored"].tolist() == [150]
    assert quality["Claim checks"].tolist() == ["not_run 145"]
    assert "Answer quality" not in [header.value for header in mechanical.subheader]


def test_a_run_with_answer_quality_but_no_retrieval_scores_still_shows_it(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run = tmp_path / "answers-only"
    (run / "C4").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "questions": 1}]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "q1"}) + "\n", encoding="utf-8")
    (run / "answer-summary.json").write_text(json.dumps({"configurations": {"C4": {
        "questions": 150, "quality": {"correctness": {"mean": 0.5, "scored": 150}}}}}),
        encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    team = app.tabs[0]
    assert not team.info
    assert team.dataframe[0].value["Correctness"].tolist() == [0.5]


def test_the_team_tab_charts_answer_quality_beside_retrieval(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    run = tmp_path / "matrix"
    (run / "C4").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "questions": 1}]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "q1", "recall": 1.0}) + "\n", encoding="utf-8")
    (run / "answer-summary.json").write_text(json.dumps({"configurations": {"C4": {
        "questions": 150, "quality": {"correctness": {"mean": 0.8, "scored": 150}}}}}),
        encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    charts = app.tabs[0].get("vega_lite_chart")
    assert len(charts) == 2
    values = json.dumps(json.loads(charts[1].proto.spec).get("datasets", {})) + charts[1].proto.spec
    assert "Correctness" in values


def test_a_run_saves_the_median_rank_of_each_questions_first_right_passage(tmp_path):
    """Worked out from each question's reciprocal rank: 1/2, 1/4 and 1 are
    ranks 2, 4 and 1, so the median is 2. A question with no right passage
    to find does not count, nor does one whose scores could not be read."""
    run = tmp_path / "ranks" / "C4"
    run.mkdir(parents=True)
    (tmp_path / "ranks" / "summary.json").write_text(json.dumps({
        "configurations": [{"config": {"id": "C4"}, "questions": 5}]}), encoding="utf-8")
    (run / "questions.jsonl").write_text("".join(
        json.dumps({"question_id": f"q{n}", "recall": recall, "ndcg": 1.0, "mrr": mrr}) + "\n"
        for n, (recall, mrr) in enumerate(((1.0, 0.5), (1.0, 0.25), (1.0, 1.0), (None, None),
                                           ("unreadable", 0.01)))), encoding="utf-8")
    result_runs.clear()
    team = result_runs(tmp_path)[0]["configurations"][0]["benchmark_metrics"]["handwritten"]
    assert team["median_rank"] == 2


def test_a_question_whose_right_passage_missed_the_cutoff_ranks_below_every_found_one(tmp_path):
    """Two of three found nothing in the top k, so the median question found
    nothing either: its rank is past any number the run ranked."""
    run = tmp_path / "missed" / "C4"
    run.mkdir(parents=True)
    (tmp_path / "missed" / "summary.json").write_text(json.dumps({
        "configurations": [{"config": {"id": "C4"}, "questions": 3}]}), encoding="utf-8")
    (run / "questions.jsonl").write_text("".join(
        json.dumps({"question_id": f"q{n}", "recall": mrr, "ndcg": mrr, "mrr": mrr}) + "\n"
        for n, mrr in enumerate((1.0, 0.0, 0.0))), encoding="utf-8")
    result_runs.clear()
    team = result_runs(tmp_path)[0]["configurations"][0]["benchmark_metrics"]["handwritten"]
    assert team["median_rank"] == math.inf


def test_a_rescored_benchmark_shows_no_median_rank_from_the_rows_rescore_left_behind(tmp_path):
    """rescore rewrites MRR under rescored/ and leaves questions.jsonl as it
    was, so the rows' ranks no longer go with the scores shown beside them."""
    run = tmp_path / "rescored-run"
    (run / "C4").mkdir(parents=True)
    (run / "rescored").mkdir()
    (run / "summary.json").write_text(json.dumps({
        "configurations": [{"config": {"id": "C4"}, "questions": 1}]}), encoding="utf-8")
    (run / "C4" / "questions.jsonl").write_text(json.dumps(
        {"question_id": "q1", "recall": 1.0, "ndcg": 1.0, "mrr": 1.0}) + "\n", encoding="utf-8")
    (run / "rescored" / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"team": {"recall": 1.0, "mrr": 0.5}}}]}),
        encoding="utf-8")
    result_runs.clear()
    team = result_runs(tmp_path)[0]["configurations"][0]["benchmark_metrics"]["handwritten"]
    assert team["mrr"] == 0.5
    assert team["median_rank"] is None
