"""The results report command: one run directory in, tables and charts out (#48)."""

import json
import os
import subprocess
import sys
from xml.etree import ElementTree

import pytest

from src.config import PROJECT_ROOT
from src.evaluation import report


def _save_run(root, name, rows_by_config, **summaries):
    """A saved run as the evaluation commands leave it: a manifest and per-question rows."""
    run = root / name
    run.mkdir(parents=True)
    configurations = []
    for config_id, rows in rows_by_config.items():
        (run / config_id).mkdir()
        (run / config_id / "questions.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        configurations.append({"config": {"id": config_id, "name": f"row {config_id}"},
                               "questions": len(rows), **summaries.get(config_id, {})})
    (run / "summary.json").write_text(json.dumps({"run_id": name, "top_k": 16,
                                                  "configurations": configurations}),
                                      encoding="utf-8")
    return run


def test_the_command_writes_the_team_ablation_table_from_a_run_directory(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {
        "C1": [{"question_id": "q1", "recall": 1.0, "ndcg": 0.5},
               {"question_id": "q2", "recall": 0.0, "ndcg": 0.25}],
        "C4": [{"question_id": "q1", "recall": 1.0, "ndcg": 1.0},
               {"question_id": "q2", "recall": 1.0, "ndcg": 0.5}],
    })
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| --- | --- | ---: | ---: | ---: |" in table
    assert "| C1 | row C1 | 2 | 2 | 0.5000 | 0.3750 |" in table
    assert "| C4 | row C4 | 2 | 2 | 1.0000 | 0.7500 |" in table


def test_a_score_one_configuration_lacks_shows_as_a_dash_and_draws_no_bar(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {
        "C1": [{"question_id": "q1", "recall": 0.5, "ndcg": 0.5}],
        "C4": [{"question_id": "q1", "recall": 1.0}],
    })
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 1 | 1 | 1.0000 | — |" in (out / "ablation.md").read_text(encoding="utf-8")
    assert set(_bars(out / "handwritten-retrieval.svg")) == {
        "C1 Recall: 0.5000", "C1 nDCG: 0.5000", "C4 Recall: 1.0000"}


def test_without_out_the_report_lands_beside_the_committed_evaluation_summaries(tmp_path, monkeypatch):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    monkeypatch.setattr(report, "REPORT_ROOT", tmp_path / "docs" / "evaluation")

    report.main([str(run)])

    written = tmp_path / "docs" / "evaluation" / "matrix" / "ablation.md"
    assert written.is_file()
    assert "| C4 | row C4 | 1 | 1 | 1.0000 |" in written.read_text(encoding="utf-8")


def test_a_folder_without_a_saved_run_is_refused_without_writing_anything(tmp_path, capsys):
    (tmp_path / "empty").mkdir()
    out = tmp_path / "report"

    with pytest.raises(SystemExit) as stopped:
        report.main([str(tmp_path / "empty"), "--out", str(out)])

    assert stopped.value.code != 0
    assert "summary.json" in capsys.readouterr().err
    assert not out.exists()


SVG = "{http://www.w3.org/2000/svg}"


def _bars(chart):
    """Each bar's hover title and drawn height, read from the saved SVG."""
    root = ElementTree.parse(chart).getroot()
    return {rect.find(SVG + "title").text: float(rect.get("height"))
            for rect in root.iter(SVG + "rect") if rect.find(SVG + "title") is not None}


def test_each_benchmark_gets_a_metric_chart_the_table_embeds(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {
        "C1": [{"question_id": "q1", "recall": 0.5, "mrr": 0.25}],
        "C4": [{"question_id": "q1", "recall": 1.0, "mrr": 0.0}],
    })
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "![Team benchmark retrieval metrics](handwritten-retrieval.svg)" in (
        out / "ablation.md").read_text(encoding="utf-8")
    bars = _bars(out / "handwritten-retrieval.svg")
    assert set(bars) == {"C1 Recall: 0.5000", "C1 MRR: 0.2500",
                         "C4 Recall: 1.0000", "C4 MRR: 0.0000"}
    assert bars["C4 Recall: 1.0000"] == 2 * bars["C1 Recall: 0.5000"] > 0
    assert bars["C1 MRR: 0.2500"] * 2 == bars["C1 Recall: 0.5000"]
    assert bars["C4 MRR: 0.0000"] == 0
    bottoms = {float(rect.get("y")) + float(rect.get("height"))
               for rect in ElementTree.parse(out / "handwritten-retrieval.svg").getroot().iter(SVG + "rect")
               if rect.find(SVG + "title") is not None}
    assert len(bottoms) == 1, "every bar stands on the same zero line"
    assert not (out / "mechanical-retrieval.svg").exists()


def test_regenerating_after_a_rerun_replaces_the_report_with_no_stale_chart(tmp_path):
    results = tmp_path / "results"
    out = tmp_path / "report"
    run = _save_run(results, "matrix", {"C4": [
        {"question_id": "q1", "recall": 0.5},
        {"question_id": "xbrl-aapl-2024-revenue", "recall": 1.0}]})
    report.main([str(run), "--out", str(out)])
    assert (out / "mechanical-retrieval.svg").is_file()
    (out / "notes.txt").write_text("kept", encoding="utf-8")

    (run / "C4" / "questions.jsonl").write_text(
        json.dumps({"question_id": "q1", "recall": 0.75}) + "\n", encoding="utf-8")
    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 1 | 1 | 0.7500 |" in table
    assert "Mechanical XBRL benchmark" not in table
    assert not (out / "mechanical-retrieval.svg").exists()
    assert (out / "notes.txt").read_text(encoding="utf-8") == "kept"


def test_a_stopped_run_is_marked_partial_in_the_report(tmp_path):
    run = _save_run(tmp_path / "results", "matrix",
                    {"C4": [{"question_id": "q1", "recall": 1.0}]},
                    C4={"stopped": {"question_id": "q2", "error": "quota"}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "Incomplete results: C4 (stopped at q2)." in table
    assert table.index("Incomplete results") < table.index("## Team benchmark")


def test_an_answer_model_run_reports_its_abstention_and_latency_table(tmp_path):
    run = tmp_path / "results" / "generators"
    (run / "G1").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "G1", "name": "local", "provider": "ollama"}, "questions": 2}]}),
        encoding="utf-8")
    (run / "G1" / "report.json").write_text(json.dumps({"results": [
        {"question_id": "q1", "route": "ollama", "answer": {"abstained": False, "latency_ms": 100}},
        {"question_id": "q2", "route": "facts", "answer": {"abstained": True, "latency_ms": 300}},
    ]}), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "### Answer-model ablation" in table
    assert "| G1 | local | 2 | 0.5000 | 200.0000 | 1 |" in table
    assert not (out / "handwritten-retrieval.svg").exists()


def _quality(mean, scored, total=150):
    return {"mean": mean, "scored": scored, "total": total}


def _save_answers(run, **configurations):
    """The answer summary full_ablation's answers and rescore commands write."""
    (run / "answer-summary.json").write_text(json.dumps({
        "complete": True, "team_questions": 150, "configurations": configurations}),
        encoding="utf-8")


def test_matrix_answer_quality_is_reported_with_its_scored_counts(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 150, "verification_enabled": False,
                           "checks": {"unverified": 54, "mismatch": 22},
                           "quality": {"correctness": _quality(0.8182, 150),
                                       "faithfulness": _quality(0.91572, 145),
                                       "answerability_correct": _quality(0.99333, 150),
                                       "citation_precision": _quality(0.43848, 145),
                                       "citation_recall": _quality(0.79452, 146)}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "### Answer quality" in table
    assert ("| C4 | row C4 | 150 | 0.8182 | 150 | 0.9157 | 145 | 0.9933 | 150 "
            "| 0.4385 | 145 | 0.7945 | 146 | mismatch 22, unverified 54 |") in table


def test_a_summary_only_matrix_reports_its_saved_team_and_xbrl_split(tmp_path):
    run = tmp_path / "results" / "matrix"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4", "name": "filtered hybrid"}, "questions": 2034,
        "by_benchmark": {"team": {"questions": 150, "recall": 0.8185},
                         "xbrl": {"questions": 1884, "recall": 0.7627}}}]}), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "## Mechanical XBRL benchmark" in table
    team, mechanical = table.split("## Mechanical XBRL benchmark")
    assert "| C4 | filtered hybrid | 150 | 0.8185 |" in team
    assert "| C4 | filtered hybrid | 1884 | 0.7627 |" in mechanical


def test_the_report_is_one_python_dash_m_command(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    out = tmp_path / "report"

    subprocess.run([sys.executable, "-m", "src.evaluation.report", str(run), "--out", str(out)],
                   cwd=PROJECT_ROOT, check=True)

    assert (out / "ablation.md").is_file()


def test_answer_quality_alone_still_gets_a_team_section(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1"}]})
    _save_answers(run, C4={"questions": 150,
                           "quality": {"correctness": _quality(0.5, 150)}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "## Team benchmark" in table
    assert "| C4 | row C4 | 150 | 0.5000 | 150 |" in table
    assert "Claim checks" not in table
    assert not (out / "handwritten-retrieval.svg").exists()


@pytest.mark.parametrize("saved", ["not json", "[]", '{"configurations": []}',
                                   '{"configurations": {"C4": "broken"}}'])
def test_an_unreadable_answer_summary_leaves_the_retrieval_report_intact(tmp_path, saved):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    (run / "answer-summary.json").write_text(saved, encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 1 | 1 | 1.0000 |" in table
    assert "Answer quality" not in table


def test_a_pipe_in_a_configuration_name_does_not_split_its_table_cell(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["configurations"][0]["config"]["name"] = "BM25 | dense"
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | BM25 \\| dense | 1 | 1 | 1.0000 |" in (out / "ablation.md").read_text(encoding="utf-8")


def test_a_line_break_in_a_name_stays_inside_its_table_row(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["configurations"][0]["config"]["name"] = "BM25\r\ndense"
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | BM25 dense | 1 | 1 | 1.0000 |" in (out / "ablation.md").read_text(encoding="utf-8")


def _answer_row(question_id, correctness, *, abstained=False):
    """One row of a matrix configuration's answers.jsonl, as answers and rescore read it."""
    return {"question_id": question_id,
            "quality": {"faithfulness": None if abstained else 1.0, "correctness": correctness,
                        "answerability_correct": 1.0, "citation_precision": None,
                        "citation_recall": None},
            "answer": {"abstained": abstained, "verification": None}}


def test_answer_quality_follows_the_saved_answers_not_a_stale_summary(tmp_path):
    """A rescore leaves answer-summary.json behind; the per-question answers are current."""
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 2, "quality": {"correctness": _quality(0.9, 2, 2)}})
    (run / "C4" / "answers.jsonl").write_text(
        json.dumps(_answer_row("q1", 1.0)) + "\n"
        + json.dumps(_answer_row("q2", 0.0, abstained=True)) + "\n", encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert ("| C4 | row C4 | 2 | 0.5000 | 2 | 1.0000 | 1 | 1.0000 | 2 "
            "| — | 0 | — | 0 | not_run 1 |") in (
        out / "ablation.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("answer", [
    {"question_id": "q1"},
    {**_answer_row("q1", 1.0), "answer": {"abstained": False, "verification": {"checks": ["x"]}}},
], ids=["no quality", "checks not objects"])
def test_answers_that_cannot_be_scored_fall_back_to_the_saved_summary(tmp_path, answer):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 150, "quality": {"correctness": _quality(0.9, 150)}})
    (run / "C4" / "answers.jsonl").write_text(json.dumps(answer) + "\n", encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 150 | 0.9000 | 150 |" in (out / "ablation.md").read_text(encoding="utf-8")


def test_the_run_id_is_the_run_folder_however_its_path_is_spelled(tmp_path, monkeypatch):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    monkeypatch.setattr(report, "REPORT_ROOT", tmp_path / "docs" / "evaluation")

    report.main([str(run / "C4" / "..")])

    written = tmp_path / "docs" / "evaluation" / "matrix" / "ablation.md"
    assert written.is_file()
    assert written.read_text(encoding="utf-8").startswith("# Results: matrix\n")


def test_an_out_path_that_is_a_file_is_refused_with_a_usage_error(tmp_path, capsys):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    out = tmp_path / "report.md"
    out.write_text("mine", encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", str(out)])

    assert stopped.value.code == 2
    assert "is not a folder" in capsys.readouterr().err
    assert out.read_text(encoding="utf-8") == "mine"


def test_a_run_path_that_does_not_exist_says_so(tmp_path, capsys):
    with pytest.raises(SystemExit) as stopped:
        report.main([str(tmp_path / "missing"), "--out", str(tmp_path / "report")])

    assert stopped.value.code == 2
    assert "does not exist" in capsys.readouterr().err


@pytest.mark.parametrize("where", ["under a file", "a broken link"])
def test_an_out_folder_that_cannot_be_made_is_a_usage_error(tmp_path, capsys, where):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    (tmp_path / "a-file").write_text("mine", encoding="utf-8")
    if where == "under a file":
        out = tmp_path / "a-file" / "report"
    else:
        out = tmp_path / "link"
        out.symlink_to(tmp_path / "gone")

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", str(out)])

    assert stopped.value.code == 2
    assert "cannot write" in capsys.readouterr().err


def test_answers_for_fewer_team_questions_than_were_retrieved_are_marked_partial(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {
        "C3": [{"question_id": "q1", "recall": 1.0}, {"question_id": "q2", "recall": 1.0}],
        "C4": [{"question_id": "q1", "recall": 1.0}, {"question_id": "q2", "recall": 1.0}]})
    for config_id, answered in (("C3", 2), ("C4", 1)):
        (run / config_id / "answers.jsonl").write_text("".join(
            json.dumps(_answer_row(f"q{n}", 1.0)) + "\n" for n in range(1, answered + 1)),
            encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "Incomplete answers: C4 (1 of 2 team questions answered)." in table
    assert "C3 (" not in table


def test_a_rescored_runs_retrieval_figures_reach_the_report(tmp_path):
    """rescore writes recomputed retrieval under rescored/, leaving questions.jsonl as it was."""
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 1.0}, {"question_id": "xbrl-q2", "recall": 1.0}]})
    (run / "rescored").mkdir()
    (run / "rescored" / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4"}, "by_benchmark": {"team": {"questions": 1, "recall": 0.25},
                                                  "xbrl": {"questions": 1, "recall": 0.5}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    team, mechanical = table.split("## Mechanical XBRL benchmark")
    assert "| C4 | row C4 | 1 | 1 | 0.2500 |" in team
    assert "| C4 | row C4 | 1 | 1 | 0.5000 |" in mechanical
    assert set(_bars(out / "handwritten-retrieval.svg")) == {"C4 Recall: 0.2500"}


@pytest.mark.parametrize("saved", [
    "not json", "[]", '{"configurations": null}', '{"configurations": 5}',
    '{"configurations": ["broken"]}',
    '{"configurations": [{"config": "C4"}]}',
    '{"configurations": [{"config": {"id": "C4"}, "by_benchmark": {"team": 0.25}}]}'])
def test_an_unreadable_rescore_summary_leaves_the_saved_rows_in_charge(tmp_path, saved):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    (run / "rescored").mkdir()
    (run / "rescored" / "summary.json").write_text(saved, encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 1 | 1 | 1.0000 |" in (out / "ablation.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("saved_summary", [True, False], ids=["saved summary", "no summary"])
@pytest.mark.parametrize("answers", ["", '{"question_id": "q1", "qual', '{"question_id": "q1"}\n'],
                         ids=["empty", "unfinished line", "unscorable row"])
def test_answers_that_exist_but_cannot_be_scored_are_flagged(tmp_path, answers, saved_summary):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    if saved_summary:
        _save_answers(run, C4={"questions": 1, "quality": {"correctness": _quality(0.9, 1, 1)}})
    (run / "C4" / "answers.jsonl").write_text(answers, encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "Unscored answers: C4." in (out / "ablation.md").read_text(encoding="utf-8")


def test_a_configuration_without_an_answers_file_is_not_flagged(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 1, "quality": {"correctness": _quality(0.9, 1, 1)}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "Unscored answers" not in (out / "ablation.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("summary", [
    {"questions": 150, "quality": "corrupt"},
    {"questions": 150, "quality": {"correctness": 0.8}},
    {"questions": 150, "quality": {"correctness": _quality(0.8, 150)}, "checks": ["not_run"]},
], ids=["quality not a record", "metric not a record", "checks not a record"])
def test_an_oddly_shaped_answer_summary_does_not_stop_the_report(tmp_path, summary):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4=summary)
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 1 | 1 | 1.0000 |" in table
    assert "| C4 | row C4 | 150 |" in table


@pytest.mark.parametrize("inside", [".", "report", "C4/../report"])
def test_the_report_is_never_written_inside_the_run_it_reads(tmp_path, capsys, inside):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    before = sorted(path.relative_to(run) for path in run.rglob("*"))

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", str(run / inside)])

    assert stopped.value.code == 2
    assert "inside the run" in capsys.readouterr().err
    assert sorted(path.relative_to(run) for path in run.rglob("*")) == before


def test_the_retrieval_table_says_how_many_questions_each_mean_covers(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 1.0, "ndcg": 1.0, "hard_negative_accuracy": 1.0},
        {"question_id": "q2", "recall": 0.0, "ndcg": 0.5},
        {"question_id": "q3"}]})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert ("| Configuration | Name | Questions | Relevance-scored | Recall | nDCG "
            "| Hard negatives | Hard Negative Accuracy |") in table
    assert "| C4 | row C4 | 3 | 2 | 0.5000 | 0.7500 | 1 | 1.0000 |" in table


def test_a_summary_only_run_reports_its_saved_denominators(tmp_path):
    run = tmp_path / "results" / "matrix"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4", "name": "filtered hybrid"}, "questions": 150,
        "by_benchmark": {"xbrl": {"questions": 1884, "recall": 0.7627,
                                  "hard_negative_accuracy": None,
                                  "metric_denominators": {"recall": 1884,
                                                          "hard_negative_accuracy": 0}}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| Configuration | Name | Questions | Relevance-scored | Recall |" in table
    assert "| C4 | filtered hybrid | 1884 | 1884 | 0.7627 |" in table


def test_answer_quality_gets_its_own_chart(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 150, "quality": {"correctness": _quality(0.8, 150),
                                                         "faithfulness": _quality(0.4, 145)}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "![Answer quality](handwritten-answer-quality.svg)" in (
        out / "ablation.md").read_text(encoding="utf-8")
    bars = _bars(out / "handwritten-answer-quality.svg")
    assert set(bars) == {"C4 Correctness: 0.8000", "C4 Faithfulness: 0.4000"}
    assert bars["C4 Correctness: 0.8000"] == 2 * bars["C4 Faithfulness: 0.4000"]

    _save_answers(run)
    report.main([str(run), "--out", str(out)])
    assert not (out / "handwritten-answer-quality.svg").exists()


@pytest.mark.parametrize("row, absent", [
    ({"question_id": "q1", "recall": 1.0}, "Hard negatives"),
    ({"question_id": "q1", "hard_negative_accuracy": 1.0}, "Relevance-scored"),
])
def test_a_count_column_goes_with_the_scores_it_counts(tmp_path, row, absent):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [row]})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 1 | 1 | 1.0000 |" in table
    assert absent not in table


@pytest.mark.parametrize("mean", ["high", [0.5], {"value": 0.5}, True, float("nan"), float("inf"),
                                  10 ** 400, -0.5, 5],
                         ids=["text", "list", "record", "true", "nan", "infinity",
                              "huge integer", "below zero", "above one"])
def test_a_saved_mean_that_is_not_a_number_is_left_blank(tmp_path, mean):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 150, "quality": {
        "correctness": {"mean": mean, "scored": 1}, "faithfulness": _quality(0.5, 150)}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 150 | — | 1 | 0.5000 | 150 |" in (out / "ablation.md").read_text(encoding="utf-8")
    assert set(_bars(out / "handwritten-answer-quality.svg")) == {"C4 Faithfulness: 0.5000"}


def test_a_rescored_score_that_is_not_a_number_is_left_blank(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    (run / "rescored").mkdir()
    (run / "rescored" / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4"}, "by_benchmark": {"team": {"recall": "high", "ndcg": 0.5}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| Relevance-scored | nDCG |" in table
    assert "| C4 | row C4 | 1 | 1 | 0.5000 |" in table
    assert set(_bars(out / "handwritten-retrieval.svg")) == {"C4 nDCG: 0.5000"}


def test_a_case_variant_spelling_of_the_run_is_still_inside_it(tmp_path, capsys):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    variant = run.parent / "MATRIX"
    if not variant.exists():
        pytest.skip("this filesystem tells MATRIX from matrix")

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", str(variant / "report")])

    assert stopped.value.code == 2
    assert "inside the run" in capsys.readouterr().err
    assert not (run / "report").exists()


def test_an_empty_out_is_a_usage_error(tmp_path, capsys, monkeypatch):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", ""])

    assert stopped.value.code == 2
    assert "--out" in capsys.readouterr().err
    assert not (tmp_path / "ablation.md").exists()


@pytest.mark.skipif(os.name != "posix" or os.geteuid() == 0, reason="needs a folder the user cannot read")
def test_an_out_under_an_unreadable_folder_is_a_usage_error(tmp_path, capsys):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    locked = tmp_path / "locked"
    locked.mkdir()
    locked.chmod(0)
    try:
        with pytest.raises(SystemExit) as stopped:
            report.main([str(run), "--out", str(locked / "report")])
    finally:
        locked.chmod(0o755)

    assert stopped.value.code == 2
    assert "cannot write" in capsys.readouterr().err


@pytest.mark.parametrize("recall", ["high", 10 ** 400, 5, -3, True],
                         ids=["text", "huge integer", "above one", "below zero", "true"])
def test_a_question_row_score_that_is_not_a_score_is_left_out_of_its_mean(tmp_path, recall):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 0.5}, {"question_id": "q2", "recall": recall}]})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 2 | 1 | 0.5000 |" in (out / "ablation.md").read_text(encoding="utf-8")


def test_an_answer_row_too_large_to_average_leaves_only_its_own_score(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    row = _answer_row("q1", 1.0)
    row["quality"]["correctness"] = 10 ** 400
    (run / "C4" / "answers.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 1 | — | 0 | 1.0000 | 1 |" in table
    assert "Unscored answers" not in table


def test_negative_zero_is_shown_as_zero(tmp_path):
    run = tmp_path / "results" / "matrix"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4"}, "by_benchmark": {"team": {"questions": 1, "recall": -0.0}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | C4 | 1 | 0.0000 |" in (out / "ablation.md").read_text(encoding="utf-8")
    assert 'height="-0"' not in (out / "handwritten-retrieval.svg").read_text(encoding="utf-8")


@pytest.mark.parametrize("saved", [{"recall": 5, "ndcg": 0.5}, {"recall": -0.5, "ndcg": 0.5},
                                   {"hard_negative_accuracy": 2, "ndcg": 0.5}],
                         ids=["recall above one", "recall below zero", "hard negatives above one"])
def test_a_saved_retrieval_score_outside_zero_to_one_is_left_blank(tmp_path, saved):
    run = tmp_path / "results" / "matrix"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [{
        "config": {"id": "C4"}, "by_benchmark": {"team": {"questions": 1, **saved}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| Configuration | Name | Questions | nDCG |" in table
    assert set(_bars(out / "handwritten-retrieval.svg")) == {"C4 nDCG: 0.5000"}


def test_a_configuration_whose_saved_scores_are_all_invalid_is_not_listed(tmp_path):
    run = tmp_path / "results" / "matrix"
    run.mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"team": {"questions": 1, "recall": 0.5}}},
        {"config": {"id": "C5"}, "by_benchmark": {"team": {"questions": 1, "recall": 2}}}]}),
        encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 |" in table
    assert "| C5 |" not in table


def test_the_report_is_never_written_into_another_saved_run(tmp_path, capsys):
    results = tmp_path / "results"
    run = _save_run(results, "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    other = _save_run(results, "other", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    before = sorted(path.relative_to(other) for path in other.rglob("*"))

    for out in (other, other / "report"):
        with pytest.raises(SystemExit) as stopped:
            report.main([str(run), "--out", str(out)])
        assert stopped.value.code == 2
        assert "inside a saved run" in capsys.readouterr().err
    assert sorted(path.relative_to(other) for path in other.rglob("*")) == before

    report.main([str(run), "--out", str(results / "matrix-report")])
    assert (results / "matrix-report" / "ablation.md").is_file()


@pytest.mark.parametrize("inside", ["C4", "C4/report", "rescored/C4/x"])
def test_a_folder_of_the_run_being_read_is_named_as_that_run(tmp_path, capsys, inside):
    """Real runs keep a summary.json in each configuration folder and under rescored/."""
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    (run / "C4" / "summary.json").write_text("{}", encoding="utf-8")
    (run / "rescored" / "C4").mkdir(parents=True)
    (run / "rescored" / "C4" / "summary.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit) as stopped:
        report.main([str(run), "--out", str(run / inside)])

    assert stopped.value.code == 2
    assert "inside the run it reads" in capsys.readouterr().err


def test_recall_ndcg_and_mrr_share_the_questions_they_count(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 0.5, "ndcg": 0.5, "mrr": 0.5},
        {"question_id": "q2", "recall": "high", "ndcg": 0.7, "mrr": 0.7}]})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 2 | 1 | 0.5000 | 0.5000 | 0.5000 |" in (
        out / "ablation.md").read_text(encoding="utf-8")


def test_an_answer_row_score_outside_zero_to_one_is_left_out_of_its_mean(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 1.0}, {"question_id": "q2", "recall": 1.0}]})
    rows = [_answer_row("q1", 0.5), _answer_row("q2", 5)]
    rows[1]["quality"]["faithfulness"] = 10 ** 400
    (run / "C4" / "answers.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    table = (out / "ablation.md").read_text(encoding="utf-8")
    assert "| C4 | row C4 | 2 | 0.5000 | 1 | 1.0000 | 1 | 1.0000 | 2 |" in table
    assert "Unscored answers" not in table


def test_every_score_the_tables_can_show_gets_a_bar_of_its_own_colour(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [
        {"question_id": "q1", "recall": 0.5, "ndcg": 0.5, "mrr": 0.5,
         "hard_negative_accuracy": 0.5}]})
    _save_answers(run, C4={"questions": 1, "quality": {
        metric: _quality(0.5, 1, 1) for metric in report.QUALITY_LABELS}})
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    fills = set()
    labels = set()
    for chart in ("handwritten-retrieval.svg", "handwritten-answer-quality.svg"):
        root = ElementTree.parse(out / chart).getroot()
        for rect in root.iter(SVG + "rect"):
            if rect.find(SVG + "title") is not None:
                labels.add(rect.find(SVG + "title").text.split(":")[0].removeprefix("C4 "))
                fills.add(rect.get("fill"))
    assert labels == {*report.METRIC_LABELS.values(), *report.QUALITY_LABELS.values()}
    assert len(fills) == len(labels)
