"""The Results page's breakdown by question type and difficulty (#170)."""

import json

import pandas as pd

from src.app.state import question_outcomes
from src.evaluation.breakdown import CORRECTNESS, FOUND, Group, default_breakdown
from src.evaluation.report import MECHANICAL, TEAM, read_runs
from tests.test_results_comparison import _answer_row, _open, _retrieval_row, _save_run

BREAKDOWN = "By question type and difficulty"


def _row(question_id, rank, question_type, difficulty, **fields):
    return _retrieval_row(question_id, rank, question_type=question_type,
                          difficulty=difficulty, **fields)


def _fixture_run(root, *, with_c4=True, answers=None):
    """A team run of 26 questions: 12 easy numeric (9 found in the top 10),
    10 medium factual (3 found) and 4 hard comparative (all found), plus two
    XBRL questions. C0 found nothing, so any number shown is C4's."""
    c4 = ([_row(f"n{index:02}", 1 if index < 9 else None, "numeric", "easy")
           for index in range(12)]
          + [_row(f"f{index:02}", 2 if index < 3 else None, "factual", "medium")
             for index in range(10)]
          + [_row(f"c{index:02}", 1, "comparative", "hard") for index in range(4)]
          + [_row(f"xbrl-{index}", None, "numeric", "mechanical", source="xbrl")
             for index in range(2)])
    c0 = [{**row, "retrieved_chunk_ids": [f"{row['question_id']}-miss"] * 16} for row in c4]
    configurations = {"C0": {"questions": c0}}
    if with_c4:
        configurations["C4"] = {"questions": c4, **({"answers": answers} if answers else {})}
    return _save_run(root, configurations)


def _tables(tab):
    """The breakdown's tables in ``tab``, keyed by the grouping they split on."""
    tables = {}
    for element in tab.dataframe:
        frame = element.value
        for column in ("Question type", "Difficulty"):
            if isinstance(frame, pd.DataFrame) and column in frame.columns:
                tables.setdefault(column, []).append(frame)
    return tables


def _records(frame):
    """A table's rows, a missing score read as None (pandas holds it as NaN)."""
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def test_the_default_configuration_is_scored_by_question_type_with_each_groups_count(
        tmp_path, monkeypatch):
    _fixture_run(tmp_path)
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert BREAKDOWN in [header.value for header in team.subheader]
    by_type = _tables(team)["Question type"][0]
    assert _records(by_type) == [
        {"Question type": "numeric", "Questions": 12, "Scored": 12, "Score": 0.75,
         "Too few to read": False},
        {"Question type": "factual", "Questions": 10, "Scored": 10, "Score": 0.3,
         "Too few to read": False},
        # Too few to read: counted, but its 1.0 is not shown.
        {"Question type": "comparative", "Questions": 4, "Scored": 4, "Score": None,
         "Too few to read": True}]
    captions = [caption.value for caption in team.caption]
    assert ("By question type, numeric questions score highest (0.75 over 12) "
            "and factual lowest (0.30 over 10).") in captions
    assert ("By difficulty, easy questions score highest (0.75 over 12) "
            "and medium lowest (0.30 over 10).") in captions
    assert any(caption.startswith("C4, the app's default configuration") for caption in captions)
    # The run saved no answers, so there is no correctness section of unscored groups.
    assert "**Answer correctness**" not in [markdown.value for markdown in team.markdown]


def test_each_tab_breaks_down_only_its_own_benchmarks_questions(tmp_path, monkeypatch):
    _fixture_run(tmp_path)
    app = _open(monkeypatch, tmp_path)

    team_numeric = _tables(app.tabs[0])["Question type"][0].iloc[0]
    assert (team_numeric["Question type"], team_numeric["Questions"]) == ("numeric", 12)
    # The XBRL questions are all numeric and mechanical: one group each, so
    # there is no strong or weak side to show.
    assert BREAKDOWN not in [header.value for header in app.tabs[1].subheader]


def test_judged_answers_add_answer_correctness_by_group(tmp_path, monkeypatch):
    answers = ([_answer_row(f"n{index:02}", 1.0) for index in range(12)]
               + [_answer_row(f"f{index:02}", 0.5) for index in range(10)])
    _fixture_run(tmp_path, answers=answers)
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert "**Answer correctness**" in [markdown.value for markdown in team.markdown]
    correctness = _tables(team)["Question type"][1]
    # The comparative answers were never judged, so that group has no score.
    assert _records(correctness[["Question type", "Scored", "Score"]]) == [
        {"Question type": "numeric", "Scored": 12, "Score": 1.0},
        {"Question type": "factual", "Scored": 10, "Score": 0.5},
        {"Question type": "comparative", "Scored": 0, "Score": None}]


def test_a_run_without_the_default_configuration_shows_no_breakdown(tmp_path, monkeypatch):
    _fixture_run(tmp_path, with_c4=False)
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert BREAKDOWN not in [header.value for header in team.subheader]
    assert not _tables(team)


def test_a_c4_folder_the_runs_summary_does_not_list_is_not_broken_down(tmp_path, monkeypatch):
    run = _fixture_run(tmp_path)
    # As a run stopped before its summary named C4 would leave it.
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest["configurations"] = [entry for entry in manifest["configurations"]
                                  if entry["config"]["id"] != "C4"]
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert BREAKDOWN not in [header.value for header in team.subheader]


def test_a_question_whose_label_is_not_text_is_in_no_group(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C4": {"questions": (
        [_row(f"n{index:02}", 1, "numeric", "easy") for index in range(10)]
        + [_row(f"f{index:02}", None, "factual", "easy") for index in range(10)]
        + [_row("odd", 1, 7, "")])}})
    team = _open(monkeypatch, tmp_path).tabs[0]

    tables = _tables(team)
    assert tables["Question type"][0]["Questions"].tolist() == [10, 10]
    # "" is no difficulty either, so every labelled question is easy: one group.
    assert "Difficulty" not in tables


def test_a_breakdown_with_every_group_readable_has_no_too_few_column(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C4": {"questions": (
        [_row(f"e{index:02}", 1, "numeric", "easy") for index in range(10)]
        + [_row(f"m{index:02}", None, "numeric", "medium") for index in range(10)])}})
    team = _open(monkeypatch, tmp_path).tabs[0]

    tables = _tables(team)
    # One question type is no split at all; difficulty has two readable groups.
    assert "Question type" not in tables
    assert list(tables["Difficulty"][0].columns) == ["Difficulty", "Questions", "Scored", "Score"]


def test_the_grouping_function_splits_a_fixture_run_as_the_page_shows_it(tmp_path):
    _fixture_run(tmp_path, answers=[_answer_row(f"n{index:02}", 1.0) for index in range(12)])
    run = read_runs(tmp_path)[0]

    splits = default_breakdown(run, TEAM, question_outcomes)

    assert [(split.measure, split.grouping) for split in splits] == [
        (FOUND, "question type"), (FOUND, "difficulty"),
        (CORRECTNESS, "question type"), (CORRECTNESS, "difficulty")]
    assert splits[0].groups == [Group("numeric", 12, 12, 0.75), Group("factual", 10, 10, 0.3),
                                Group("comparative", 4, 4, 1.0)]
    # Only the numeric answers were judged.
    assert splits[2].groups == [Group("numeric", 12, 12, 1.0), Group("factual", 10, 0, None),
                                Group("comparative", 4, 0, None)]
    # The XBRL questions are one group each way, so there is nothing to split.
    assert default_breakdown(run, MECHANICAL, question_outcomes) == []


def test_the_grouping_function_gives_nothing_for_a_run_without_the_default_configuration(
        tmp_path):
    _fixture_run(tmp_path, with_c4=False)

    assert default_breakdown(read_runs(tmp_path)[0], TEAM, question_outcomes) == []
