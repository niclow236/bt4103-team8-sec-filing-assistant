"""The Results page's before and after view of two configurations (#169)."""

import json

import pandas as pd
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
    # Scored at 10, as the full matrix scores them.
    found = isinstance(rank, int) and rank <= 10
    score = None if rank == "unscored" else float(found)
    return {"question_id": question_id, "difficulty": "medium", "source": None,
            "query": {"text": question}, "retrieved_chunk_ids": retrieved,
            "scoring_supporting_chunk_ids": supporting, "recall": score,
            "mrr": None if score is None else (1 / rank if found else 0.0), **fields}


def _answer_row(question_id, correctness, *, question="What was revenue?"):
    """A saved answer row the judge gave ``correctness``."""
    return {"question_id": question_id, "source": None, "answer": {"question": question},
            "quality": {"correctness": correctness, "method": "llm_judge"}}


def _save_run(root, configurations, *, name="run"):
    """Write a run folder: ``configurations`` maps an id to its saved rows."""
    run = root / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": name, "configurations": [
        {"config": {"id": config_id, "name": config_id},
         "questions": len(next(iter(files.values())))}
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
            _retrieval_row("q6", 1), _retrieval_row("q7", 10)]},
        "C4": {"questions": [
            _retrieval_row("q1", 2), _retrieval_row("q2", None), _retrieval_row("q3", 1),
            _retrieval_row("q4", 5), _retrieval_row("q5", "unscored"), _retrieval_row("q7", 11)]},
    })
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "2"), (":red[Worse]", "2"), ("Unchanged", "1")]


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


def test_judged_correctness_is_counted_too_when_the_run_has_it(tmp_path, monkeypatch):
    retrieval = [_retrieval_row(question_id, 1) for question_id in ("q1", "q2", "q3", "q4")]
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [
            _answer_row("q1", 0.5), _answer_row("q2", 1.0), _answer_row("q3", 0.75),
            _answer_row("q4", None)]},
        "C4": {"questions": retrieval, "answers": [
            _answer_row("q1", 1.0), _answer_row("q2", 0.8), _answer_row("q3", 0.75),
            _answer_row("q4", 1.0)]},
    })
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]

    assert _counts(team)[3:] == [
        (":green[Improved]", "1"), (":red[Worse]", "1"), ("Unchanged", "1")]
    assert any("3 questions both configurations answered" in caption.value
               for caption in team.caption)


def test_answer_model_configurations_are_compared_on_judged_correctness_alone(
        tmp_path, monkeypatch):
    run = tmp_path / "generators"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": config_id, "provider": "ollama"}, "questions": 2}
        for config_id in ("G1", "G3")]}), encoding="utf-8")
    for config_id, scores in (("G1", (0.5, 1.0, 1.0)), ("G3", (1.0, 1.0, 0.0))):
        (run / config_id).mkdir()
        (run / config_id / "report.json").write_text(json.dumps({"results": [
            {**_answer_row(question_id, score), "route": "ollama",
             "answer": {"question": f"Asked {question_id}?", "abstained": False,
                        "latency_ms": 1}}
            for question_id, score in zip(("q1", "q2", "xbrl-a"), scores)]}), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    team, mechanical = app.tabs

    assert _counts(team) == [(":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "1")]
    assert not any("top 10" in markdown.value for markdown in team.markdown)
    assert _questions_table(team).set_index("Question").loc["q1", "Asked"] == "Asked q1?"
    assert _counts(mechanical) == [
        (":green[Improved]", "0"), (":red[Worse]", "1"), ("Unchanged", "0")]


def test_a_benchmark_with_no_saved_results_offers_no_comparison(tmp_path, monkeypatch):
    rows = [_retrieval_row("q1", 1)]
    _save_run(tmp_path, {"C0": {"questions": rows}, "C4": {"questions": rows}})
    app = _open(monkeypatch, tmp_path)
    team, mechanical = app.tabs

    assert team.metric
    assert not mechanical.selectbox and not mechanical.metric


def test_two_configurations_with_no_question_in_common_say_so_instead_of_counting(
        tmp_path, monkeypatch):
    _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", 1)]},
                         "C4": {"questions": [_retrieval_row("q2", 1)]}})
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]

    assert not team.metric
    assert any("C0 and C4" in info.value for info in team.info)


def _questions_table(tab):
    """The comparison's table: the last one in the tab, after the run's scores."""
    return tab.dataframe[-1].value


def test_each_question_shows_both_configurations_rank_and_can_be_filtered_to_the_worse(
        tmp_path, monkeypatch):
    _save_run(tmp_path, {
        "C0": {"questions": [
            _retrieval_row("q1", None, question="Who audits Apple?"), _retrieval_row("q2", 1),
            _retrieval_row("q3", 3), _retrieval_row("q4", 12)]},
        "C4": {"questions": [
            _retrieval_row("q1", 2, question="Who audits Apple?"), _retrieval_row("q2", None),
            _retrieval_row("q3", 1), _retrieval_row("q4", 5)]},
    })
    app = _open(monkeypatch, tmp_path)
    table = _questions_table(app.tabs[0])

    ranks = {row["Question"]: (row["C0 rank"], row["C4 rank"]) for _, row in table.iterrows()}
    assert {question: tuple(None if pd.isna(rank) else rank for rank in pair)
            for question, pair in ranks.items()} == {
        "q1": (None, 2), "q2": (1, None), "q3": (3, 1), "q4": (12, 5)}
    assert table.set_index("Question").loc["q1", "Asked"] == "Who audits Apple?"
    # A rank is a whole number, shown as 2 and not 2.0, even beside a missing one.
    assert str(table["C4 rank"].dtype) == "Int64"

    show = next(box for box in app.tabs[0].selectbox if box.label == "Show")
    show.select("Worse: top 10").run()
    assert not app.exception
    assert _questions_table(app.tabs[0])["Question"].tolist() == ["q2"]


def test_questions_can_be_filtered_to_those_whose_judged_correctness_improved(
        tmp_path, monkeypatch):
    retrieval = [_retrieval_row("q1", 1), _retrieval_row("q2", 1), _retrieval_row("q3", "unscored")]
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [_answer_row("q1", 0.5), _answer_row("q2", 1.0)]},
        "C4": {"questions": retrieval, "answers": [_answer_row("q1", 0.9), _answer_row("q2", 0.4)]},
    })
    app = _open(monkeypatch, tmp_path)
    table = _questions_table(app.tabs[0]).set_index("Question")

    assert sorted(table.index) == ["q1", "q2"]
    assert (table.loc["q1", "C0 correctness"], table.loc["q1", "C4 correctness"]) == (0.5, 0.9)
    assert table.loc["q2", "Correctness"] == "worse"

    show = next(box for box in app.tabs[0].selectbox if box.label == "Show")
    show.select("Improved: correctness").run()
    assert not app.exception
    assert _questions_table(app.tabs[0])["Question"].tolist() == ["q1"]


def test_a_filter_the_new_pair_cannot_offer_falls_back_to_every_question(tmp_path, monkeypatch):
    retrieval = [_retrieval_row("q1", 1), _retrieval_row("q2", None)]
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [_answer_row("q1", 0.5), _answer_row("q2", 1.0)]},
        "C4": {"questions": retrieval, "answers": [_answer_row("q1", 0.9), _answer_row("q2", 1.0)]},
        "C2": {"questions": retrieval},
    })
    app = _open(monkeypatch, tmp_path)
    next(box for box in app.tabs[0].selectbox if box.label == "Show").select(
        "Improved: correctness").run()
    app.tabs[0].selectbox[1].select("C2").run()

    assert not app.exception
    assert next(box for box in app.tabs[0].selectbox if box.label == "Show").value == "All questions"
    assert sorted(_questions_table(app.tabs[0])["Question"]) == ["q1", "q2"]


def test_embedding_configurations_are_compared_on_the_retrieval_rows_their_report_saved(
        tmp_path, monkeypatch):
    # How src.evaluation.model_ablation saves an E row: in report.json, with
    # its scores and retrieved ids and no right passage ids.
    def row(question_id, mrr):
        return {"question_id": question_id, "question_type": "factual", "difficulty": "easy",
                "source": "handwritten", "k": 10, "recall": mrr, "ndcg": mrr, "mrr": mrr,
                "hard_negative_accuracy": None, "retrieved_chunk_ids": ["a", "b"]}

    run = tmp_path / "embeddings"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": config_id}, "questions": 2} for config_id in ("E1-dense", "E2-dense")]}),
        encoding="utf-8")
    for config_id, scores in (("E1-dense", (0.0, 1.0)), ("E2-dense", (0.5, 1.0))):
        (run / config_id).mkdir()
        (run / config_id / "report.json").write_text(json.dumps({"results": [
            row(question_id, score) for question_id, score in zip(("q1", "q2"), scores)]}),
            encoding="utf-8")
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "1")]


def test_a_row_whose_retrieved_passages_are_unreadable_is_left_unscored(tmp_path, monkeypatch):
    broken = {**_retrieval_row("q2", 1), "retrieved_chunk_ids": None}
    _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", None), _retrieval_row("q2", 1)]},
                         "C4": {"questions": [_retrieval_row("q1", 1), broken]}})
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "0")]
