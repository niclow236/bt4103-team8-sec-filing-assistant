"""The Results page's saved-run worked example (#167)."""

import json

from streamlit.testing.v1 import AppTest

from src.app.state import failure_examples, worked_examples
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


def _save_run(root, answers=(), *, c6_answers=()):
    run = root / "fixture-run"
    config = run / "C4"
    config.mkdir(parents=True)
    configurations = [{"config": {"id": "C4", "name": "default"},
                       "questions": max(1, len(answers)), "recall": 1.0}]
    if c6_answers:
        configurations.append({"config": {"id": "C6", "name": "verified"},
                               "questions": len(c6_answers), "recall": 1.0})
    (run / "summary.json").write_text(json.dumps({
        "run_id": "fixture-run",
        "configurations": configurations,
    }), encoding="utf-8")
    (config / "questions.jsonl").write_text(json.dumps({
        "question_id": "q1", "recall": 1.0,
    }) + "\n", encoding="utf-8")
    if answers:
        (config / "answers.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in answers), encoding="utf-8")
    if c6_answers:
        verified = run / "C6"
        verified.mkdir()
        (verified / "questions.jsonl").write_text(json.dumps({
            "question_id": "q1", "recall": 1.0,
        }) + "\n", encoding="utf-8")
        (verified / "answers.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in c6_answers), encoding="utf-8")
    return run


def test_prefers_verified_c6_answers_and_orders_supported_checks_first(tmp_path):
    unverified = _answer_row("unverified", fact_check=True)
    unverified["answer"]["verification"]["checks"][-1]["status"] = "unverified"
    run = _save_run(tmp_path, [
        _answer_row("wrong", correctness=0.5),
        _answer_row("correct-without-fact"),
    ], c6_answers=[
        _answer_row("no-check"),
        unverified,
        _answer_row("supported", fact_check=True),
    ])

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert [example.question_id for example in examples] == [
        "supported", "unverified", "no-check"]
    assert examples[0].answer == sample_answer(
        "What was supported's revenue?").text
    assert all(example.config_id == "C6" for example in examples)
    assert examples[0].passage["url"].startswith("https://www.sec.gov/")
    assert examples[0].citation_marker == 1
    assert examples[0].fact_checks[0]["status"] == "supported"


def test_a_run_local_worked_example_is_the_only_example_and_keeps_its_configuration(tmp_path):
    run = _save_run(tmp_path, [_answer_row("ordinary")])
    curated = _answer_row("daryl-03", fact_check=True) | {
        "matrix_config": {"id": "C6", "name": "C4 + filing decomposition and verification"}}
    (run / "worked-example.jsonl").write_text(json.dumps(curated) + "\n", encoding="utf-8")

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert [(example.question_id, example.config_id) for example in examples] == [("daryl-03", "C6")]


def test_an_unusable_run_local_example_falls_back_to_saved_c4_answers(tmp_path):
    run = _save_run(tmp_path, [_answer_row("ordinary")])
    (run / "worked-example.jsonl").write_text(
        json.dumps(_answer_row("not-correct", correctness=0.5)) + "\n", encoding="utf-8")

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert [(example.question_id, example.config_id) for example in examples] == [
        ("ordinary", "C4")]


def test_fallback_passage_is_labelled_as_not_cited(tmp_path):
    row = _answer_row("uncited")
    row["answer"]["citations"] = []
    run = _save_run(tmp_path, [row])

    worked_examples.clear()
    examples = worked_examples(str(run))

    assert examples[0].citation_marker is None


def test_page_labels_an_uncited_fallback_passage(tmp_path, monkeypatch):
    row = _answer_row("uncited")
    row["answer"]["citations"] = []
    _save_run(tmp_path, [row])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert any("2. Passage C4 retrieved, not cited by the answer" in item.value
               for item in app.markdown)


def test_page_walks_the_saved_example_and_links_to_edgar(tmp_path, monkeypatch):
    _save_run(tmp_path, c6_answers=[_answer_row("q1", fact_check=True),
                                    _answer_row("q2", fact_check=True)])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert "Worked Example" in [heading.value for heading in app.subheader]
    assert any("1. Benchmark question" in item.value for item in app.markdown)
    assert any("2. Passage C6 retrieved, cited as [1]" in item.value
               for item in app.markdown)
    links = app.get("link_button")
    assert any(link.label == "Open filing on EDGAR"
               and link.url.startswith("https://www.sec.gov/") for link in links)
    # The passage is shown open under its own heading, not in a collapsed panel.
    assert any("Management's Discussion and Analysis" in item.value for item in app.markdown)
    assert not any(expander.get("link_button") for expander in app.expander)
    assert any("The saved XBRL fact agrees" in item.value for item in app.success)
    next_example = next(button for button in app.button if button.label == "Next example")
    next_example.click().run()
    assert not app.exception
    assert any("question q2" in caption.value for caption in app.caption)


def test_page_names_other_citations_when_only_the_first_passage_is_shown(
        tmp_path, monkeypatch):
    row = _answer_row("comparison", fact_check=True)
    first = row["answer"]["passages"][0]
    row["answer"]["passages"].append(first | {
        "chunk_id": "comparison-passage", "fiscal_year": 2023,
        "text": "Comparison passage: Revenue was $4 billion.",
    })
    row["answer"]["citations"] = [
        {"marker": 1, "chunk_id": first["chunk_id"], "resolved": True},
        {"marker": 2, "chunk_id": "comparison-passage", "resolved": True},
    ]
    row["answer"]["text"] = (
        "Revenue was $5 billion in FY2024 [1], compared with $4 billion in FY2023 [2].")
    _save_run(tmp_path, c6_answers=[row])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert any("The answer also cites [2]; only the passage for [1] is shown."
               in caption.value for caption in app.caption)


def test_page_renders_saved_currency_as_literal_text(tmp_path, monkeypatch):
    _save_run(tmp_path, c6_answers=[_answer_row("q1", fact_check=True)])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert any(r"\$5 billion" in item.value for item in app.markdown)
    assert any(item.value.startswith("Benchmark answer:") and r"\$5 billion" in item.value
               for item in app.caption)
    assert any(r"\$5 billion" in item.value for item in app.success)


def test_page_explains_when_the_selected_run_has_no_worked_example(tmp_path, monkeypatch):
    run = _save_run(tmp_path)
    failed = _answer_row("judge-failed")
    failed["quality"]["correctness"] = None
    (run / "C4" / "answers.jsonl").write_text(
        json.dumps(failed) + "\n", encoding="utf-8")
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert "Worked Example" in [heading.value for heading in app.subheader]
    assert any("no saved correct answer" in message.value and "judge failed" in message.value
               for message in app.caption)


def test_chooses_only_answers_below_the_wrong_answer_cutoff(tmp_path):
    run = _save_run(tmp_path, [
        _answer_row("partial", correctness=0.5),
        _answer_row("wrong", correctness=0.49),
        _answer_row("correct"),
    ])

    failure_examples.clear()
    examples = failure_examples(str(run))

    assert [(example.question_id, example.correctness, example.refused)
            for example in examples] == [("wrong", 0.49, False)]


def test_chooses_a_refusal_and_keeps_its_saved_reason(tmp_path):
    refused = _answer_row("refused")
    refused["answer"].update({
        "abstained": True,
        "abstention_reason": "model_declined",
        "text": "I cannot support the requested $5 billion figure.",
    })
    refused["quality"]["correctness"] = None
    run = _save_run(tmp_path, [refused])

    failure_examples.clear()
    examples = failure_examples(str(run))

    assert [(example.question_id, example.refused, example.abstention_reason)
            for example in examples] == [("refused", True, "model_declined")]


def test_a_run_with_neither_a_wrong_answer_nor_refusal_has_no_failure(tmp_path):
    run = _save_run(tmp_path, [
        _answer_row("partial", correctness=0.5),
        _answer_row("correct"),
    ])

    failure_examples.clear()

    assert failure_examples(str(run)) == ()


def test_failure_prefers_run_local_then_c6_before_c4(tmp_path):
    run = _save_run(
        tmp_path,
        [_answer_row("c4", correctness=0.0)],
        c6_answers=[_answer_row("c6", correctness=0.0)],
    )
    local = _answer_row("local", correctness=0.0) | {
        "matrix_config": {"id": "C7", "name": "curated"}}
    (run / "worked-example.jsonl").write_text(json.dumps(local) + "\n", encoding="utf-8")

    failure_examples.clear()
    assert [(example.question_id, example.config_id)
            for example in failure_examples(str(run))] == [("local", "C7")]

    (run / "worked-example.jsonl").unlink()
    failure_examples.clear()
    assert [(example.question_id, example.config_id)
            for example in failure_examples(str(run))] == [("c6", "C6")]


def test_page_labels_and_steps_between_saved_failures(tmp_path, monkeypatch):
    wrong = _answer_row("wrong", correctness=0.0)
    wrong["answer"]["text"] = "The answer was $4 billion [1]."
    refused = _answer_row("refused")
    refused["answer"].update({
        "abstained": True,
        "abstention_reason": "model_declined",
        "text": "I cannot support the $5 billion answer.",
    })
    refused["quality"]["correctness"] = None
    _save_run(tmp_path, [_answer_row("correct"), wrong, refused])
    _read_results_from(monkeypatch, tmp_path)
    worked_examples.clear()
    failure_examples.clear()

    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()

    assert not app.exception
    assert any("Answered wrongly" in message.value for message in app.error)
    assert any(r"\$4 billion" in item.value for item in app.markdown)
    assert any("4. Saved judge verdict" in item.value for item in app.markdown)
    next_failure = next(button for button in app.button if button.label == "Next failure")
    next_failure.click().run()
    assert not app.exception
    assert any("Refused" in message.value for message in app.error)
    assert any("model_declined" in caption.value for caption in app.caption)

