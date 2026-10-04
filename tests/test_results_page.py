import json

from src.app.results_page import load_result_runs


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
        json.dumps({"question_id": "hand", "difficulty": "hard", "source": "questions"})
        + "\n"
        + json.dumps({"question_id": "xbrl", "difficulty": "mechanical", "source": "xbrl"})
        + "\n",
        encoding="utf-8",
    )

    load_result_runs.clear()
    runs = load_result_runs(tmp_path)

    assert runs[0]["run_id"] == "demo"
    assert [row["question_id"] for row in runs[0]["configurations"][0]["question_rows"]] == [
        "hand", "xbrl"
    ]

