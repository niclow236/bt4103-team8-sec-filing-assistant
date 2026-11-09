"""The chat-assistant and closed-book baselines (#220).

They read and write their files as UTF-8 on every platform, say which line of
an answers file is broken, record the judge as it scores (with its rubric), and
keep a person's overruled judgements apart from the judged mean.
"""
from __future__ import annotations

import json
from importlib import import_module
from pathlib import Path

import pytest

from tools import llm_only_baseline as closed_book
from tools.score_chat_baseline import matrix_means, read_answers, write_source_check

# What a real model writes: curly quotes, whose UTF-8 bytes cp1252 cannot decode.
ANSWER = "Apple’s total net sales were “$391.0 billion” in fiscal 2024."
QUESTIONS = Path(closed_book.__file__).resolve().parents[1] / "benchmark" / "questions.jsonl"


@pytest.fixture
def utf8_only(monkeypatch):
    """Fail any read or write that leaves the encoding to the platform. Windows'
    default is cp1252, so without this the tests could only fail there."""
    read, write = Path.read_text, Path.write_text

    def strict_read(self, encoding=None, errors=None):
        assert encoding is not None, f"{self.name} read without an encoding"
        return read(self, encoding=encoding, errors=errors)

    def strict_write(self, data, encoding=None, errors=None, newline=None):
        assert encoding is not None, f"{self.name} written without an encoding"
        return write(self, data, encoding=encoding, errors=errors, newline=newline)

    monkeypatch.setattr(Path, "read_text", strict_read)
    monkeypatch.setattr(Path, "write_text", strict_write)


@pytest.fixture
def offline_run(tmp_path, monkeypatch):
    """The closed-book command with its model and judge replaced, so no call
    leaves the machine. The judge keeps LLMJudge's own record of itself."""
    # By module path: src.rag re-exports a function named generate.
    generate, quality = import_module("src.rag.generate"), import_module("src.evaluation.quality")

    class Model:
        def invoke(self, messages):
            return ANSWER

    class Judge(quality.LLMJudge):
        def __call__(self, question, answer):
            return {"correctness": 0.5, "correctness_reason": "partly matches the gold answer"}

    monkeypatch.setattr(closed_book, "ROOT", tmp_path)
    monkeypatch.setenv("SOCLAAS_API_KEY", "test-key")
    monkeypatch.setenv("MISTRAL_API_KEY", "test-key")
    monkeypatch.setattr(closed_book, "plain_model", lambda config, model: Model())
    monkeypatch.setattr(generate, "checked_model", lambda config, llm=None: None)
    monkeypatch.setattr(quality, "checked_model", lambda config, llm=None: None)
    monkeypatch.setattr(quality, "LLMJudge", Judge)

    def run(*extra):
        closed_book.main(["--run-id", "offline", "--limit", "2",
                          "--questions", str(QUESTIONS), *extra])
        out = tmp_path / "results" / "offline"
        return {name: out / name for name in ("answers.jsonl", "config.json", "summary.json")}

    return run


def test_a_matrix_run_is_read_as_utf8(tmp_path, utf8_only):
    row = {"question_id": "daryl-28", "answer": ANSWER, "quality": {"correctness": 0.5}}
    (tmp_path / "C4").mkdir()
    (tmp_path / "C4" / "answers.jsonl").write_bytes(
        (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8"))
    assert matrix_means(tmp_path) == {"C4": {"mean_correctness": 0.5, "scored": 1, "of": 20}}


def test_a_closed_book_answer_with_curly_quotes_is_saved_resumed_and_summarized(
        offline_run, utf8_only):
    offline_run()
    files = offline_run()   # a second run resumes: it reads back what the first wrote
    rows = [json.loads(line)
            for line in files["answers.jsonl"].read_text(encoding="utf-8").splitlines()]
    assert [row["answer"] for row in rows] == [ANSWER, ANSWER]
    summary = json.loads(files["summary.json"].read_text(encoding="utf-8"))
    assert summary["questions"] == 2


def test_the_closed_book_run_records_the_rubric_the_judge_scored_with(offline_run):
    judge = json.loads(offline_run()["config.json"].read_text(encoding="utf-8"))["judge"]
    assert judge["rubric"] == "answer-quality-v1"
    assert judge["config"]["prompt_template_id"] == "answer-quality-v1"
    assert judge["config"]["model"] == "ministral-8b-2512"


def test_overruled_judgements_are_reported_beside_the_judged_mean(offline_run, tmp_path):
    qid = json.loads(QUESTIONS.read_text(encoding="utf-8").splitlines()[0])["question_id"]
    overrides = tmp_path / "adjudications.json"
    overrides.write_text(json.dumps([{"question_id": qid, "correctness": 1.0,
                                      "reason": "$200.58 billion is $200,583 million"}]),
                         encoding="utf-8")
    summary = json.loads(offline_run("--adjudications", str(overrides))["summary.json"]
                         .read_text(encoding="utf-8"))
    assert summary["mean_correctness"] == 0.5               # the judge's, unchanged
    assert summary["adjudicated"]["mean_correctness"] == 0.75
    assert summary["adjudicated"]["changes"] == [
        {"question_id": qid, "judged": 0.5, "correctness": 1.0,
         "reason": "$200.58 billion is $200,583 million"}]


def test_an_adjudication_of_a_question_the_run_did_not_answer_is_refused(offline_run, tmp_path):
    overrides = tmp_path / "adjudications.json"
    overrides.write_text(json.dumps([{"question_id": "no-such-question", "correctness": 1.0,
                                      "reason": "r"}]), encoding="utf-8")
    with pytest.raises(SystemExit, match="no-such-question"):
        offline_run("--adjudications", str(overrides))


def test_the_source_check_sheet_opens_in_excel_with_its_quotes(tmp_path):
    row = {"id": "Q01", "question": "q", "gold_answer": "g", "answer": ANSWER, "correctness": 1.0,
           "sources": [{"url": "https://www.sec.gov/x", "section": "Item 8"}]}
    write_source_check(tmp_path / "sheet.csv", [row])
    # Excel reads a CSV as UTF-8 only when it starts with the byte-order mark.
    assert (tmp_path / "sheet.csv").read_bytes().startswith(b"\xef\xbb\xbf")


def test_a_malformed_answer_line_is_named(tmp_path):
    path = tmp_path / "answers.jsonl"
    path.write_bytes(b'{"id": "Q01", "answer": "x"}\n{not json\n')
    with pytest.raises(SystemExit, match="line 2 is not JSON"):
        read_answers(path)
