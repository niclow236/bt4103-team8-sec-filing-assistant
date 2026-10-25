"""The Results page's saved-run worked example (#167)."""

import json

from streamlit.testing.v1 import AppTest

from src.app.state import worked_examples
from src.config import PROJECT_ROOT
from tests.sample_answers import sample_answer
from tests.test_results_page import _read_results_from


RESULTS_PAGE = PROJECT_ROOT / "src/app/app_pages/results.py"


def _answer_row(question_id: str, *, correctness: float = 1.0, fact_check: bool = False):
    answer = sample_answer(f"What was {question_id}'s revenue?").to_dict()
    if fact_check:
        answer["verification"]["checks"].append({
            "kind": "fact", "status": "supported", "sentence_index": 0,
            "claim": "Revenue was $5 billion.",
            "reason": "The saved XBRL fact agrees with the answer.",
            "figure": "$5 billion", "value": "5000000000", "unit": "USD",
            "evidence": ["us-gaap:Revenue"],
        })
    return {
        "question_id": question_id,
        "expected_answer": "$5 billion",
        "quality": {"correctness": correctness,
                    "correctness_reason": "It matches the benchmark answer."},
        "answer": answer,
    }


def _save_run(root, answers=()):
    run = root / "fixture-run"
    config = run / "C4"
    config.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({
        "run_id": "fixture-run",
        "configurations": [{"config": {"id": "C4", "name": "default"},
                            "questions": max(1, len(answers)), "recall": 1.0}],
    }), encoding="utf-8")
    (config / "questions.jsonl").write_text(json.dumps({
        "question_id": "q1", "recall": 1.0,
    }) + "\n", encoding="utf-8")
    if answers:
        (config / "answers.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
    return run


def test_chooses_a_correct_saved_example_and_prefers_one_with_an_xbrl_check(tmp_path):
    run = _save_run(tmp_path, [
        _answer_row("wrong", correctness=0.5),
        _answer_row("correct-without-fact"),
        _answer_row("correct-with-fact", fact_check=True),
    ])

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert [example.question_id for example in examples] == [
        "correct-with-fact", "correct-without-fact"]
    assert examples[0].answer == sample_answer(
        "What was correct-with-fact's revenue?").text
    assert examples[0].passage["url"].startswith("https://www.sec.gov/")
    assert examples[0].fact_checks[0]["status"] == "supported"


def test_a_run_local_worked_example_is_the_only_example_and_keeps_its_configuration(tmp_path):
    run = _save_run(tmp_path, [_answer_row("ordinary")])
    curated = _answer_row("daryl-03", fact_check=True) | {
        "matrix_config": {"id": "C6", "name": "C4 + filing decomposition and verification"}}
    (run / "worked-example.jsonl").write_text(json.dumps(curated) + "\n", encoding="utf-8")

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert [(example.question_id, example.config_id) for example in examples] == [("daryl-03", "C6")]


def test_page_walks_the_saved_example_and_links_to_edgar(tmp_path, monkeypatch):
    _save_run(tmp_path, [_answer_row("q1", fact_check=True),
                         _answer_row("q2", fact_check=True)])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert "Worked Example" in [heading.value for heading in app.subheader]
    assert any("1. Benchmark question" in item.value for item in app.markdown)
    links = app.get("link_button")
    assert any(link.label == "Open filing on EDGAR"
               and link.url.startswith("https://www.sec.gov/") for link in links)
    assert any("The saved XBRL fact agrees" in item.value for item in app.success)
    next_example = next(button for button in app.button if button.label == "Next example")
    next_example.click().run()
    assert not app.exception
    assert any("question q2" in caption.value for caption in app.caption)


def test_page_explains_when_the_selected_run_has_no_worked_example(tmp_path, monkeypatch):
    _save_run(tmp_path)
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert "Worked Example" in [heading.value for heading in app.subheader]
    assert any("no saved correct answer" in message.value for message in app.caption)

