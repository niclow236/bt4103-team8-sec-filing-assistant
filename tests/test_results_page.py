import json
from src.config import PROJECT_ROOT

from src.app.state import benchmark_kind, result_runs
from src.app.components import result_chart_rows


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
    assert benchmark_kind({"question_id": "hand"}) == "handwritten"
    assert benchmark_kind({"question_id": "xbrl-apple-2024-revenue"}) == "mechanical"
    assert benchmark_kind({"source": "llm_assisted"}) == "handwritten"


def test_results_page_reads_model_report_rows(tmp_path):
    report = tmp_path / "report.json"
    report.write_text(json.dumps({"results": [{"question_id": "q1"}]}), encoding="utf-8")
    from src.app.state import _read_result_rows

    assert _read_result_rows(report) == [{"question_id": "q1"}]


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
    state.result_runs.clear()
    monkeypatch.setattr(state, "result_runs", lambda: result_runs(tmp_path))
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.tabs) == 2
    manual, mechanical = app.tabs
    assert manual.label == "Team benchmark"
    assert manual.dataframe[0].value["Recall"].tolist() == [1.0]
    assert mechanical.dataframe[0].value["Recall"].tolist() == [0.0]
    assert list(manual.dataframe[0].value.columns) == [
        "Configuration", "Name", "Questions", "Recall", "Hard Negative Accuracy"]
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
    result_runs.clear()
    monkeypatch.setattr(state, "result_runs", lambda: result_runs(tmp_path))
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
    result_runs.clear()
    monkeypatch.setattr(state, "result_runs", lambda: result_runs(tmp_path))
    app = AppTest.from_file(PROJECT_ROOT / "src/app/app_pages/results.py", default_timeout=30).run()
    assert not app.exception
    assert len(app.warning) == 1
    assert "S1200-bm25" in app.warning[0].value
    assert "questions.jsonl" in app.warning[0].value
    assert "cannot be split" in app.warning[0].value
    assert not app.dataframe and not app.get("vega_lite_chart")
