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


def _save_report_run(root, name, configurations, **config):
    """Write a run whose configurations saved their rows in report.json, as an
    answer-model or embedding ablation does: ``configurations`` maps an id to its rows."""
    run = root / name
    run.mkdir()
    (run / "summary.json").write_text(json.dumps({"configurations": [
        {"config": {"id": config_id, **config}, "questions": len(rows)}
        for config_id, rows in configurations.items()]}), encoding="utf-8")
    for config_id, rows in configurations.items():
        (run / config_id).mkdir()
        (run / config_id / "report.json").write_text(json.dumps({"results": rows}),
                                                     encoding="utf-8")
    return run


def _embedding_row(question_id, mrr):
    """How src.evaluation.model_ablation saves an E row: in report.json, with
    its scores and retrieved ids, and neither right passage ids nor its text."""
    return {"question_id": question_id, "question_type": "factual", "difficulty": "easy",
            "source": "handwritten", "k": 10, "recall": mrr, "ndcg": mrr, "mrr": mrr,
            "hard_negative_accuracy": None, "retrieved_chunk_ids": ["a", "b"]}


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
    def rows(scores):
        return [{**_answer_row(question_id, score), "route": "ollama",
                 "answer": {"question": f"Asked {question_id}?", "abstained": False,
                            "latency_ms": 1}}
                for question_id, score in zip(("q1", "q2", "xbrl-a"), scores)]

    _save_report_run(tmp_path, "generators", {"G1": rows((0.5, 1.0, 1.0)),
                                              "G3": rows((1.0, 1.0, 0.0))}, provider="ollama")
    app = _open(monkeypatch, tmp_path)
    team, mechanical = app.tabs

    assert _counts(team) == [(":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "1")]
    assert [caption.value for caption in team.caption] == [
        # An answer-model run saves no retrieval ranks, so it has no chart (#166).
        "No chart of found the right passage in the top 10: this run saved no "
        "per-question results for this benchmark.",
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


def _comparison_dataframe(tab):
    """The comparison's table: the last one in the tab, after the run's scores,
    outside the Full metrics section at the tab's foot."""
    full = [table for expander in tab.expander if expander.label == "Full metrics"
            for table in expander.dataframe]
    return [table for table in tab.dataframe if not any(table is other for other in full)][-1]


def _questions_table(tab):
    return _comparison_dataframe(tab).value


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
    _save_report_run(tmp_path, "embeddings", {
        "E1-dense": [_embedding_row("q1", 0.0), _embedding_row("q2", 1.0)],
        "E2-dense": [_embedding_row("q1", 0.5), _embedding_row("q2", 1.0)]})
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
        "Found the right passage in the top 10, over the questions each configuration "
        "scored. C4 is the app's default and C0 the baseline every other row adds to.",
        "Out of 1 of the 2 questions both configurations answered; "
        "the other 1 were not scored on retrieval by both."]


def _cell_colours(tab, column):
    """Each question's text colour in ``column`` of the comparison's table, None where plain."""
    element = _comparison_dataframe(tab)
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
    styler = _comparison_dataframe(app.tabs[0]).proto.arrow_data.styler
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
        return [caption.value for caption in app.tabs[0].caption
                if "but not here" in caption.value]

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
    _save_report_run(results, "embeddings", {"E1": [_embedding_row("q1", 0.0)],
                                             "E2": [_embedding_row("q1", 1.0)]})
    app = _open(monkeypatch, results)

    assert _questions_table(app.tabs[0]).set_index("Question").loc["q1", "Asked"] == (
        "Who audits Apple?")
    # The run froze no benchmark of its own, so the page says whose wording it is.
    assert any("1 question" in caption.value and "as it stands now" in caption.value
               for caption in app.tabs[0].caption)


def test_a_run_that_froze_its_benchmark_is_shown_with_its_own_wording(tmp_path, monkeypatch):
    from src.app import state

    current = tmp_path / "questions.jsonl"
    current.write_text(json.dumps({"question_id": "q1", "question": "Who audits Apple now?"})
                       + "\n", encoding="utf-8")
    monkeypatch.setattr(state, "BENCHMARK_FILES", (current,))
    results = tmp_path / "results"
    results.mkdir()
    run = _save_report_run(results, "embeddings", {"E1": [_embedding_row("q1", 0.0)],
                                                   "E2": [_embedding_row("q1", 1.0)]})
    (run / "benchmark.jsonl").write_text(
        json.dumps({"question_id": "q1", "question": "Who audits Apple?"}) + "\n",
        encoding="utf-8")
    app = _open(monkeypatch, results)

    assert _questions_table(app.tabs[0]).set_index("Question").loc["q1", "Asked"] == (
        "Who audits Apple?")
    assert not any("as it stands now" in caption.value for caption in app.tabs[0].caption)


def test_a_run_scored_at_11_names_the_one_rank_the_view_leaves_out(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", 11)]},
                         "C4": {"questions": [_retrieval_row("q1", 2)]}}, top_k=11)
    app = _open(monkeypatch, tmp_path)

    assert [caption.value for caption in app.tabs[0].caption
            if "but not here" in caption.value] == [
        "The scores above count a right passage in this run's top 11, so one ranked 11 "
        "counts there but not here."]


# Text is a malformed row. A NaN, which json reads, would count as unchanged
# against any score. The Answer quality table counts none of them either.
@pytest.mark.parametrize("saved", ["0.5", float("nan"), 1.5, True],
                         ids=["text", "nan", "above_one", "true"])
def test_a_correctness_that_is_not_a_score_is_not_counted(tmp_path, monkeypatch, saved):
    retrieval = [_retrieval_row("q1", 1), _retrieval_row("q2", 1)]
    _save_run(tmp_path, {
        "C0": {"questions": retrieval, "answers": [_answer_row("q1", saved), _answer_row("q2", 0.5)]},
        "C4": {"questions": retrieval, "answers": [_answer_row("q1", 1.0), _answer_row("q2", 1.0)]},
    })
    app = _open(monkeypatch, tmp_path)

    assert _counts(app.tabs[0])[3:] == [
        (":green[Improved]", "1"), (":red[Worse]", "0"), ("Unchanged", "0")]


@pytest.mark.parametrize(("by_benchmark", "scores_shown"), [
    (None, False),
    # As the full matrix saves a summary: its "team" and "xbrl" keys are read
    # as the tabs' benchmarks (#48), so its scores are shown.
    ({"team": {"recall": 0.2}, "xbrl": {"recall": 0.1}}, True),
    # As the writers before 0b49c77 saved one, keyed as the page's tabs are.
    ({"handwritten": {"recall": 0.2}, "mechanical": {"recall": 0.1}}, True),
])
def test_only_configurations_that_saved_their_questions_are_offered(
        tmp_path, monkeypatch, by_benchmark, scores_shown):
    run = _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", None)]},
                               "C4": {"questions": [_retrieval_row("q1", 1)]},
                               "C5": {"questions": [_retrieval_row("q1", 1)]}})
    # C0 saved a summary but its question rows never reached this machine.
    (run / "C0" / "questions.jsonl").unlink()
    if by_benchmark:
        manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
        manifest["configurations"][0]["by_benchmark"] = by_benchmark
        (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]

    assert team.selectbox[0].options == ["C4", "C5"]
    assert _counts(team) == [(":green[Improved]", "0"), (":red[Worse]", "0"), ("Unchanged", "1")]
    # Whichever shape its summary has, the page says beside the pickers why C0
    # is not offered, and the run's warning names it when its scores are not
    # shown at all.
    # C0's files are missing whatever its summary is keyed by: none of the three
    # shapes says it never ran the team questions.
    assert [caption.value for caption in team.caption if "Not offered" in caption.value] == [
        "Not offered, since their per-question results were not found on this machine: C0"]
    assert any("C0" in warning.value for warning in app.warning) is not scores_shown


def test_a_configuration_with_no_question_on_a_benchmark_is_not_said_to_be_missing_files(
        tmp_path, monkeypatch):
    both = [_retrieval_row("q1", 1), _retrieval_row("xbrl-a", 1)]
    # C5 stopped before it reached the mechanical questions: its rows were read.
    _save_run(tmp_path, {"C0": {"questions": both}, "C4": {"questions": both},
                         "C5": {"questions": [_retrieval_row("q1", 1)]}})
    app = _open(monkeypatch, tmp_path)
    mechanical = app.tabs[1]

    assert mechanical.selectbox[0].options == ["C0", "C4"]
    assert [caption.value for caption in mechanical.caption if "Not offered" in caption.value] == [
        "Not offered, since they saved no question on this benchmark: C5"]


def test_a_lone_readable_configuration_still_says_why_the_other_is_not_offered(
        tmp_path, monkeypatch):
    run = _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", None)]},
                               "C4": {"questions": [_retrieval_row("q1", 1)]}})
    (run / "C0" / "questions.jsonl").unlink()
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    # As writers before 0b49c77 saved a summary, keyed as the page's tabs are.
    manifest["configurations"][0]["by_benchmark"] = {"handwritten": {"recall": 0.2}}
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    team = app.tabs[0]

    assert not team.selectbox and not team.metric
    assert any("Not offered" in caption.value and "C0" in caption.value
               for caption in team.caption)


def test_the_table_asks_in_the_words_a_configuration_of_the_run_kept(tmp_path, monkeypatch):
    from src.app import state

    current = tmp_path / "questions.jsonl"
    current.write_text(json.dumps({"question_id": "q1", "question": "Who audits Apple now?"})
                       + "\n", encoding="utf-8")
    monkeypatch.setattr(state, "BENCHMARK_FILES", (current,))
    results = tmp_path / "results"
    results.mkdir()
    run = _save_report_run(results, "mixed", {"E1": [_embedding_row("q1", 0.0)]})
    # C4 saved its rows with their text, E1 without.
    (run / "C4").mkdir()
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps(_retrieval_row("q1", 1, question="Who audits Apple?")) + "\n", encoding="utf-8")
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest["configurations"].append({"config": {"id": "C4"}, "questions": 1})
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, results)
    team = app.tabs[0]

    assert team.selectbox[0].value == "E1"
    assert _questions_table(team).set_index("Question").loc["q1", "Asked"] == "Who audits Apple?"
    assert not any("as it stands now" in caption.value for caption in team.caption)


def test_the_text_caption_counts_only_the_questions_shown(tmp_path, monkeypatch):
    from src.app import state

    current = tmp_path / "questions.jsonl"
    current.write_text("".join(json.dumps({"question_id": question_id, "question": "Asked?"}) + "\n"
                               for question_id in ("q1", "q2")), encoding="utf-8")
    monkeypatch.setattr(state, "BENCHMARK_FILES", (current,))
    results = tmp_path / "results"
    results.mkdir()
    _save_report_run(results, "embeddings", {
        "E1": [_embedding_row("q1", 0.0), _embedding_row("q2", 1.0)],
        "E2": [_embedding_row("q1", 1.0), _embedding_row("q2", 1.0)]})
    app = _open(monkeypatch, results)
    next(box for box in app.tabs[0].selectbox if box.label == "Show").select(
        "Improved: top 10").run()

    assert any("The text of 1 question " in caption.value for caption in app.tabs[0].caption)


def test_a_summary_that_names_only_the_other_benchmark_is_not_said_to_be_missing_files(
        tmp_path, monkeypatch):
    both = [_retrieval_row("q1", 1), _retrieval_row("xbrl-a", 1)]
    run = _save_run(tmp_path, {"C0": {"questions": both}, "C4": {"questions": both},
                               "C5": {"questions": both}})
    (run / "C0" / "questions.jsonl").unlink()
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    # As writers before 0b49c77 saved a summary: C0 ran the team questions only.
    manifest["configurations"][0]["by_benchmark"] = {"handwritten": {"recall": 0.2}}
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    team, mechanical = app.tabs

    def not_offered(tab):
        return [caption.value for caption in tab.caption if "Not offered" in caption.value]

    assert not_offered(team) == [
        "Not offered, since their per-question results were not found on this machine: C0"]
    assert not_offered(mechanical) == [
        "Not offered, since they saved no question on this benchmark: C0"]


def test_a_baseline_with_no_text_anywhere_asks_in_the_other_configurations_words(
        tmp_path, monkeypatch):
    from src.app import state

    # Neither the run nor the checkout has the question's text.
    monkeypatch.setattr(state, "BENCHMARK_FILES", (tmp_path / "missing.jsonl",))
    results = tmp_path / "results"
    results.mkdir()
    run = _save_report_run(results, "mixed", {"E1": [_embedding_row("q1", 0.0)]})
    (run / "C4").mkdir()
    (run / "C4" / "questions.jsonl").write_text(
        json.dumps(_retrieval_row("q1", 1, question="Who audits Apple?")) + "\n", encoding="utf-8")
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest["configurations"].append({"config": {"id": "C4"}, "questions": 1})
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, results)

    assert app.tabs[0].selectbox[0].value == "E1"
    assert _questions_table(app.tabs[0]).set_index("Question").loc["q1", "Asked"] == (
        "Who audits Apple?")


def test_a_lone_configuration_without_its_rows_draws_no_comparison(tmp_path, monkeypatch):
    run = _save_run(tmp_path, {"C0": {"questions": [_retrieval_row("q1", 1)]}})
    (run / "C0" / "questions.jsonl").unlink()
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    # Its summary's scores are shown, so only the comparison is in question.
    manifest["configurations"][0]["by_benchmark"] = {"handwritten": {"recall": 0.2}}
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    assert app.tabs[0].dataframe[0].value[
        "Right passages found among those retrieved"].tolist() == [0.2]

    assert not any(header.value == "Before and after"
                   for tab in app.tabs for header in tab.subheader)
