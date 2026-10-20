"""The Results page's before and after view of two configurations (#169)."""

import json
import re

import pandas as pd
import pyarrow as pa
import pytest
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
    # Scored at 16, as the full matrix scores them, so a right passage ranked
    # 11 to 16 counts in recall and MRR though it is outside the top 10.
    retrieved_at_all = isinstance(rank, int)
    score = None if rank == "unscored" else float(retrieved_at_all)
    return {"question_id": question_id, "difficulty": "medium", "source": None,
            "query": {"text": question}, "retrieved_chunk_ids": retrieved,
            "scoring_supporting_chunk_ids": supporting, "recall": score,
            "mrr": None if score is None else (1 / rank if retrieved_at_all else 0.0), **fields}


def _answer_row(question_id, correctness, *, question="What was revenue?"):
    """A saved answer row the judge gave ``correctness``."""
    return {"question_id": question_id, "source": None, "answer": {"question": question},
            "quality": {"correctness": correctness, "method": "llm_judge"}}


def _save_run(root, configurations, *, name="run", **manifest):
    """Write a run folder: ``configurations`` maps an id to its saved rows."""
    run = root / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"run_id": name, **manifest, "configurations": [
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
    # q4 has no right passage to find, so it is answered but not counted.
    assert any("3 of the 4 questions both configurations answered" in caption.value
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


def test_answer_correctness_is_counted_too_when_the_run_has_it(tmp_path, monkeypatch):
    retrieval = [_retrieval_row(question_id, 1) for question_id in ("q1", "q2", "q3", "q4")]
    abstained = {**_answer_row("q3", 0.75),
                 "quality": {"correctness": 0.75, "method": "abstention"}}
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [
            _answer_row("q1", 0.5), _answer_row("q2", 1.0), abstained,
            _answer_row("q4", None)]},
        "C4": {"questions": retrieval, "answers": [
            _answer_row("q1", 1.0), _answer_row("q2", 0.8), _answer_row("q3", 0.75),
            _answer_row("q4", 1.0)]},
    })
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]

    assert _counts(team)[3:] == [
        (":green[Improved]", "1"), (":red[Worse]", "1"), ("Unchanged", "1")]
    # q4's baseline answer was never scored, so it is answered but not counted.
    assert any("3 of the 4 questions both configurations answered" in caption.value
               for caption in team.caption)
    # q3's baseline answer was scored by the abstention rule, not the judge,
    # so the heading claims no more than the answer's correctness.
    assert [markdown.value for markdown in team.markdown][-1] == "**Answer correctness**"


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
    assert [caption.value for caption in team.caption] == [
        "Out of the 2 questions both configurations answered."]
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


def _embedding_row(question_id, mrr):
    """How src.evaluation.model_ablation saves an E row: in report.json, with
    its scores and retrieved ids, and neither right passage ids nor its text."""
    return {"question_id": question_id, "question_type": "factual", "difficulty": "easy",
            "source": "handwritten", "k": 10, "recall": mrr, "ndcg": mrr, "mrr": mrr,
            "hard_negative_accuracy": None, "retrieved_chunk_ids": ["a", "b"]}


def test_embedding_configurations_are_compared_on_the_retrieval_rows_their_report_saved(
        tmp_path, monkeypatch):
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
    # q2 has a right passage, so the caption must not say it had none.
    assert [caption.value for caption in app.tabs[0].caption] == [
        "Out of 1 of the 2 questions both configurations answered; "
        "the other 1 were not scored on retrieval by both."]


def _cell_colours(tab, column):
    """Each question's text colour in ``column`` of the comparison's table, None where plain."""
    element = tab.dataframe[-1]
    styles = element.proto.arrow_data.styler.styles
    position = list(element.value.columns).index(column)
    colours = {}
    # One rule per style, its selector listing every cell that has it.
    for selectors, colour in re.findall(r"([^{}]+)\{\s*color:\s*([^;}\s]+)", styles):
        for row in re.findall(rf"row(\d+)_col{position}\b", selectors):
            colours[int(row)] = colour
    return {question: colours.get(row) for row, question in enumerate(element.value["Question"])}


def test_each_question_is_coloured_by_whether_it_improved_or_got_worse(tmp_path, monkeypatch):
    _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", None), _retrieval_row("q2", 1),
                             _retrieval_row("q3", 3), _retrieval_row("q4", 12)]},
        "C4": {"questions": [_retrieval_row("q1", 2), _retrieval_row("q2", None),
                             _retrieval_row("q3", 1), _retrieval_row("q4", 5)]},
    })
    app = _open(monkeypatch, tmp_path)
    colours = _cell_colours(app.tabs[0], "Top 10")
    # The light theme's green and red, the shades its :green[] and :red[] use.
    green, red = "#158237", "#bd4043"

    assert colours == {"q1": green, "q2": red, "q3": None, "q4": green}


def test_the_table_shows_scores_to_two_places(tmp_path, monkeypatch):
    retrieval = [_retrieval_row("q1", None)]
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [_answer_row("q1", 0.5)]},
        "C4": {"questions": [_retrieval_row("q1", 3)], "answers": [_answer_row("q1", 1 / 3)]},
    })
    app = _open(monkeypatch, tmp_path)
    styler = app.tabs[0].dataframe[-1].proto.arrow_data.styler
    shown = pa.ipc.open_stream(styler.display_values).read_pandas().iloc[0]

    assert shown["C4 rank"] == "3"
    assert (shown["C0 correctness"], shown["C4 correctness"]) == ("0.50", "0.33")


def test_the_change_colours_follow_the_viewers_theme(tmp_path, monkeypatch):
    from src.app import components

    _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", None)]},
                         "C4": {"questions": [_retrieval_row("q1", 1)]}})
    light = _cell_colours(_open(monkeypatch, tmp_path).tabs[0], "Top 10")["q1"]
    monkeypatch.setattr(components, "_dark_theme", lambda: True)
    dark = _cell_colours(_open(monkeypatch, tmp_path).tabs[0], "Top 10")["q1"]

    assert light is not None and dark is not None and light != dark


def test_a_run_scored_past_the_top_10_says_its_recall_counts_passages_the_view_does_not(
        tmp_path, monkeypatch):
    configurations = {
        "C0": {"questions": [_retrieval_row("q1", 12)], "answers": [_answer_row("q1", 0.5)]},
        "C4": {"questions": [_retrieval_row("q1", 2)], "answers": [_answer_row("q1", 1.0)]}}
    _save_run(tmp_path, configurations, name="at-16", top_k=16)
    _save_run(tmp_path, configurations, name="at-10", top_k=10)
    app = _open(monkeypatch, tmp_path)

    def says_why(app):
        return [caption.value for caption in app.tabs[0].caption if "Recall" in caption.value]

    app.selectbox[0].select("at-16").run()
    assert len(says_why(app)) == 1 and "11 to 16" in says_why(app)[0]
    app.selectbox[0].select("at-10").run()
    assert says_why(app) == []


@pytest.mark.parametrize(("field", "unchanged"), [
    # Its retrieved passages cannot be read, so it is not scored at all.
    ("retrieved_chunk_ids", "0"),
    # Its right passages cannot be read, so it is ranked by its saved MRR.
    ("scoring_supporting_chunk_ids", "1"),
])
def test_a_row_whose_passage_ids_are_not_text_does_not_stop_the_page(
        tmp_path, monkeypatch, field, unchanged):
    garbled = {**_retrieval_row("q2", 1), field: [{"id": "q2-gold"}]}
    _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", None), _retrieval_row("q2", 1)]},
                         "C4": {"questions": [_retrieval_row("q1", 1), garbled]}})
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0]) == [
        (":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", unchanged)]


def test_a_question_whose_rows_kept_no_text_is_shown_with_the_benchmarks(tmp_path, monkeypatch):
    from src.app import state

    benchmark = tmp_path / "questions.jsonl"
    benchmark.write_text(json.dumps({"question_id": "q1", "question": "Who audits Apple?"}) + "\n",
                         encoding="utf-8")
    monkeypatch.setattr(state, "BENCHMARK_FILES", (benchmark,))
    results = tmp_path / "results"
    results.mkdir()
    run = results / "embeddings"
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": config_id}, "questions": 1} for config_id in ("E1", "E2")]}),
        encoding="utf-8")
    for config_id, score in (("E1", 0.0), ("E2", 1.0)):
        (run / config_id).mkdir()
        (run / config_id / "report.json").write_text(
            json.dumps({"results": [_embedding_row("q1", score)]}), encoding="utf-8")
    app = _open(monkeypatch, results)

    assert _questions_table(app.tabs[0]).set_index("Question").loc["q1", "Asked"] == (
        "Who audits Apple?")
