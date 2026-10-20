"""The Results page's before and after view of two configurations (#169)."""

import json

from streamlit.testing.v1 import AppTest

from src.config import PROJECT_ROOT
from tests.test_results_page import _read_results_from

RESULTS_PAGE = PROJECT_ROOT / "src/app/app_pages/results.py"


def _retrieval_row(question_id, rank, *, question="What was revenue?", **fields):
    """A saved retrieval row whose right passage sits at ``rank``, or nowhere for None.

    ``rank`` "unscored" leaves the question without a right passage at all,
    as an unanswerable question is saved.
    """
    retrieved = [f"{question_id}-other-{index}" for index in range(16)]
    supporting = [] if rank == "unscored" else [f"{question_id}-gold"]
    if isinstance(rank, int):
        retrieved[rank - 1] = f"{question_id}-gold"
    return {"question_id": question_id, "difficulty": "medium", "source": None,
            "query": {"text": question}, "retrieved_chunk_ids": retrieved,
            "scoring_supporting_chunk_ids": supporting, **fields}


def _save_run(root, configurations, *, name="run"):
    """Write a run folder: ``configurations`` maps an id to its saved rows."""
    run = root / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": name, "configurations": [
        {"config": {"id": config_id, "name": config_id}, "questions": len(files["questions"])}
        for config_id, files in configurations.items()]}), encoding="utf-8")
    for config_id, files in configurations.items():
        (run / config_id).mkdir()
        for file_name, rows in files.items():
            (run / config_id / f"{file_name}.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return run


def _open(monkeypatch, root):
    _read_results_from(monkeypatch, root)
    app = AppTest.from_file(RESULTS_PAGE, default_timeout=30).run()
    assert not app.exception
    return app


def _counts(tab):
    """Each count the comparison shows in ``tab``, as (label, value) in order."""
    return [(metric.label, metric.value) for metric in tab.metric]


def test_counts_which_questions_found_the_right_passage_in_the_top_10_against_c0(
        tmp_path, monkeypatch):
    _save_run(tmp_path, {
        "C0": {"questions": [
            _retrieval_row("q1", None), _retrieval_row("q2", 1), _retrieval_row("q3", 3),
            _retrieval_row("q4", 12), _retrieval_row("q5", "unscored"),
            _retrieval_row("q6", 1)]},
        "C4": {"questions": [
            _retrieval_row("q1", 2), _retrieval_row("q2", None), _retrieval_row("q3", 1),
            _retrieval_row("q4", 5), _retrieval_row("q5", "unscored")]},
    })
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "2"), (":red[Worse]", "1"), ("Unchanged", "1")]


def test_two_identical_configurations_leave_every_question_both_answered_unchanged(
        tmp_path, monkeypatch):
    rows = [_retrieval_row("q1", None), _retrieval_row("q2", 1), _retrieval_row("q3", 12),
            _retrieval_row("q4", "unscored")]
    _save_run(tmp_path, {"C0": {"questions": rows}, "C0-copy": {"questions": rows}})
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "0"), (":red[Worse]", "0"), ("Unchanged", "3")]
    assert any("3 questions both configurations answered" in caption.value
               for caption in app.tabs[0].caption)


def test_the_baseline_opens_on_c0_and_any_two_configurations_can_be_compared(
        tmp_path, monkeypatch):
    _save_run(tmp_path, {
        "C4": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", 1)]},
        "C0": {"questions": [_retrieval_row("q1", None), _retrieval_row("q2", None)]},
        "C2": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", None)]},
    })
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]
    assert (team.selectbox[0].value, team.selectbox[1].value) == ("C0", "C4")
    assert _counts(team)[:2] == [(":green[Improved]", "2"), (":red[Worse]", "0")]

    team.selectbox[0].select("C4").run()
    team = app.tabs[0]
    team.selectbox[1].select("C2").run()
    assert not app.exception
    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "0"), (":red[Worse]", "1"), ("Unchanged", "1")]


def test_each_benchmark_tab_counts_only_its_own_questions(tmp_path, monkeypatch):
    _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", None), _retrieval_row("xbrl-a", 1),
                             _retrieval_row("xbrl-b", 1)]},
        "C4": {"questions": [_retrieval_row("q1", 1), _retrieval_row("xbrl-a", None),
                             _retrieval_row("xbrl-b", 1)]},
    })
    app = _open(monkeypatch, tmp_path)
    team, mechanical = app.tabs

    assert _counts(team) == [(":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "0")]
    assert _counts(mechanical) == [
        (":green[Improved]", "0"), (":red[Worse]", "1"), ("Unchanged", "1")]


def test_rows_saved_without_passage_ids_are_compared_by_their_reciprocal_rank(
        tmp_path, monkeypatch):
    # How src.evaluation.run saves a row: its scores, and no right passage ids.
    def row(question_id, mrr):
        return {"question_id": question_id, "difficulty": "easy", "source": "handwritten",
                "retrieved_chunk_ids": ["a", "b"], "recall": mrr, "mrr": mrr}

    _save_run(tmp_path, {
        "C1": {"questions": [row("q1", 0.0), row("q2", 1.0), row("q3", 0.5), row("q4", None)]},
        "C4": {"questions": [row("q1", 0.25), row("q2", 0.0), row("q3", 1.0), row("q4", None)]},
    })
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "1"), (":red[Worse]", "1"), ("Unchanged", "1")]


def test_reload_rereads_the_questions_a_comparison_counts(tmp_path, monkeypatch):
    run = _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", None)]},
        "C4": {"questions": [_retrieval_row("q1", None)]},
    })
    app = _open(monkeypatch, tmp_path)
    assert _counts(app.tabs[0])[0] == (":green[Improved]", "0")

    (run / "C4" / "questions.jsonl").write_text(
        json.dumps(_retrieval_row("q1", 1)) + "\n", encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert _counts(app.tabs[0])[0] == (":green[Improved]", "1")
