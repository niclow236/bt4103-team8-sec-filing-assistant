"""The results report command: one run directory in, tables and charts out (#48)."""

import json
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
    assert "| C1 | row C1 | 2 | 0.5000 | 0.3750 |" in table
    assert "| C4 | row C4 | 2 | 1.0000 | 0.7500 |" in table


def test_a_score_one_configuration_lacks_shows_as_a_dash_and_draws_no_bar(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {
        "C1": [{"question_id": "q1", "recall": 0.5, "ndcg": 0.5}],
        "C4": [{"question_id": "q1", "recall": 1.0}],
    })
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 1 | 1.0000 | — |" in (out / "ablation.md").read_text(encoding="utf-8")
    assert set(_bars(out / "handwritten-retrieval.svg")) == {
        "C1 Recall: 0.5000", "C1 nDCG: 0.5000", "C4 Recall: 1.0000"}


def test_without_out_the_report_lands_beside_the_committed_evaluation_summaries(tmp_path, monkeypatch):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    monkeypatch.setattr(report, "REPORT_ROOT", tmp_path / "docs" / "evaluation")

    report.main([str(run)])

    written = tmp_path / "docs" / "evaluation" / "matrix" / "ablation.md"
    assert written.is_file()
    assert "| C4 | row C4 | 1 | 1.0000 |" in written.read_text(encoding="utf-8")


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
    assert "| C4 | row C4 | 1 | 0.7500 |" in table
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
    assert "| C4 | row C4 | 1 | 1.0000 |" in table
    assert "Answer quality" not in table


def test_a_pipe_in_a_configuration_name_does_not_split_its_table_cell(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    summary["configurations"][0]["config"]["name"] = "BM25 | dense"
    (run / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | BM25 \\| dense | 1 | 1.0000 |" in (out / "ablation.md").read_text(encoding="utf-8")


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


def test_answers_that_cannot_be_scored_fall_back_to_the_saved_summary(tmp_path):
    run = _save_run(tmp_path / "results", "matrix", {"C4": [{"question_id": "q1", "recall": 1.0}]})
    _save_answers(run, C4={"questions": 150, "quality": {"correctness": _quality(0.9, 150)}})
    (run / "C4" / "answers.jsonl").write_text(json.dumps({"question_id": "q1"}) + "\n",
                                              encoding="utf-8")
    out = tmp_path / "report"

    report.main([str(run), "--out", str(out)])

    assert "| C4 | row C4 | 150 | 0.9000 | 150 |" in (out / "ablation.md").read_text(encoding="utf-8")
