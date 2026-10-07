import json

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
    assert [row["question_id"] for row in runs[0]["configurations"][0]["question_rows"]] == [
        "hand", "xbrl-apple-2024-revenue"
    ]
    assert benchmark_kind(runs[0]["configurations"][0]["question_rows"][0]) == "handwritten"
    assert benchmark_kind(runs[0]["configurations"][0]["question_rows"][1]) == "mechanical"


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
            "questions": 50, "recall": 0.8, "median_latency_ms": 120,
        }},
    }], "handwritten")
    assert [row["Metric"] for row in rows] == ["Recall"]
