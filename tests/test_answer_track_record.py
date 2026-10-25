"""The line under an answer saying how questions like it scored in testing (#168)."""

import pytest
from streamlit.testing.v1 import AppTest

from src.app.state import question_outcomes
from src.evaluation.breakdown import (
    CORRECTNESS,
    FOUND,
    MIN_GROUP_QUESTIONS,
    configuration_breakdown,
    default_breakdown,
    track_record,
)
from src.evaluation.report import TEAM, read_runs
from src.stack import DEFAULT_STACK
from tests.test_app_live import APP, _standing_in
from tests.test_results_comparison import _answer_row, _retrieval_row, _save_run


def _typed(prefix, count, question_type, correctness, *, source=None):
    """``count`` questions of one type, each answered and judged alike."""
    rows = [_retrieval_row(f"{prefix}{index:02}", 1, question_type=question_type,
                           **({"source": source} if source else {}))
            for index in range(count)]
    answers = [_answer_row(f"{prefix}{index:02}", correctness) for index in range(count)]
    return rows, answers


def _run(root, *, numeric=12, factual=12, config_ids=(DEFAULT_STACK,), correctness=0.75):
    """A saved run where each configuration answered the same questions."""
    numeric_rows, numeric_answers = _typed("n", numeric, "numeric", correctness)
    factual_rows, factual_answers = _typed("f", factual, "factual", 0.25)
    return _save_run(root, {
        config_id: {"questions": numeric_rows + factual_rows,
                    "answers": numeric_answers + factual_answers}
        for config_id in config_ids})


def _record(root, *, config_id=DEFAULT_STACK, question_type="numeric", measure=CORRECTNESS):
    run = read_runs(root)[0]
    return track_record(run, TEAM, question_outcomes,
                        config_id=config_id, question_type=question_type, measure=measure)


# --- the calculation ------------------------------------------------------

def test_a_type_with_enough_scored_questions_has_a_record(tmp_path):
    _run(tmp_path)
    record = _record(tmp_path)
    assert (record.name, record.questions, record.scored) == ("numeric", 12, 12)
    assert record.score == pytest.approx(0.75)


def test_a_type_the_benchmark_does_not_hold_has_none(tmp_path):
    _run(tmp_path)
    assert _record(tmp_path, question_type="comparative") is None


@pytest.mark.parametrize("numeric, shown", [(MIN_GROUP_QUESTIONS - 1, False),
                                            (MIN_GROUP_QUESTIONS, True)])
def test_a_type_needs_enough_scored_questions_for_a_record(tmp_path, numeric, shown):
    """Counted on the Results page either way, but one question must not be
    able to move the mean that an answer is trusted on."""
    _run(tmp_path, numeric=numeric)
    assert (_record(tmp_path) is not None) is shown


def test_a_configuration_that_saved_no_questions_has_none(tmp_path):
    _run(tmp_path, config_ids=("C1",))
    assert _record(tmp_path, config_id=DEFAULT_STACK) is None
    assert _record(tmp_path, config_id="C1") is not None


def test_each_configuration_is_scored_on_its_own_answers(tmp_path):
    """The Compare page puts two configurations side by side, so one record
    cannot stand under both."""
    numeric_rows, _ = _typed("n", 12, "numeric", 0.0)
    _save_run(tmp_path, {
        DEFAULT_STACK: {"questions": numeric_rows,
                        "answers": [_answer_row(f"n{index:02}", 0.9) for index in range(12)]},
        "C1": {"questions": numeric_rows,
               "answers": [_answer_row(f"n{index:02}", 0.1) for index in range(12)]}})

    assert _record(tmp_path, config_id=DEFAULT_STACK).score == pytest.approx(0.9)
    assert _record(tmp_path, config_id="C1").score == pytest.approx(0.1)


def test_the_mechanical_benchmark_is_not_read_for_a_typed_record(tmp_path):
    """Its questions are generated and numeric throughout, so they say little
    about a numeric question someone typed."""
    rows, answers = _typed("xbrl-", 12, "numeric", 1.0, source="xbrl")
    _save_run(tmp_path, {DEFAULT_STACK: {"questions": rows, "answers": answers}})
    assert _record(tmp_path) is None


def test_the_default_breakdown_is_the_shared_one(tmp_path):
    """#170's view and this line read the same function, so a question cannot
    be scored one way on the Results page and another under an answer."""
    _run(tmp_path)
    run = read_runs(tmp_path)[0]
    assert (default_breakdown(run, TEAM, question_outcomes)
            == configuration_breakdown(run, TEAM, question_outcomes, DEFAULT_STACK))


def test_another_measure_can_be_asked_for(tmp_path):
    """Correctness is a judged mean; the top-10 rate is a count. The line uses
    the first, and the second is here so that choice stays a choice."""
    _run(tmp_path)
    assert _record(tmp_path, measure=FOUND).score == pytest.approx(1.0)


# --- on the page ----------------------------------------------------------

def _ask(monkeypatch, root, question="What was Apple's total revenue in FY2024?"):
    """Ask one question of the real page, answered by the stand-in stack the
    live tests use, with ``root`` as the saved runs it reads."""
    import src.app.state as state_module
    _standing_in(monkeypatch)
    state_module.result_runs.clear()
    state_module.question_outcomes.clear()
    monkeypatch.setattr(state_module, "result_runs", lambda: read_runs(root))
    ui = AppTest.from_file(APP, default_timeout=30).run()
    assert not ui.exception
    ui.text_input[0].set_value(question).run()
    ui.button[0].click().run()
    assert not ui.exception
    return ui


def _said(ui):
    return [caption.value for caption in ui.caption if "In testing" in caption.value]


def test_the_ask_page_shows_the_record_under_the_answer(tmp_path, monkeypatch):
    _run(tmp_path)
    said = _said(_ask(monkeypatch, tmp_path))

    assert said
    assert "numeric questions" in said[0]
    assert "0.75 out of 1 for correctness" in said[0]
    assert "averaged over the 12 scored" in said[0]
    # The acceptance criteria: the line names the configuration and the run.
    assert DEFAULT_STACK in said[0] and "run run" in said[0]


def test_the_ask_page_says_nothing_without_a_saved_run(tmp_path, monkeypatch):
    assert not _said(_ask(monkeypatch, tmp_path))


def test_each_side_of_compare_carries_its_own_record(tmp_path, monkeypatch):
    """The two sides are different configurations. One record under both would
    credit one of them with the other's answers."""
    import src.app.state as state_module
    from tests.test_compare_page import PAGE

    # The page opens on C1 against C4, which is what the fixture saves.
    numeric, _ = _typed("n", 12, "numeric", 0.0)
    _save_run(tmp_path, {
        "C1": {"questions": numeric,
               "answers": [_answer_row(f"n{index:02}", 0.10) for index in range(12)]},
        DEFAULT_STACK: {"questions": numeric,
                        "answers": [_answer_row(f"n{index:02}", 0.90) for index in range(12)]}})
    _standing_in(monkeypatch)
    state_module.result_runs.clear()
    state_module.question_outcomes.clear()
    monkeypatch.setattr(state_module, "result_runs", lambda: read_runs(tmp_path))

    ui = AppTest.from_file(PAGE, default_timeout=30).run()
    # A numeric question, since that is the type the run holds a record for.
    ui.text_input[0].set_value("What was Apple's total revenue in FY2024?").run()
    ui.button[0].click().run()
    assert not ui.exception

    said = _said(ui)
    assert len(said) == 2, said
    assert "C1's answers" in said[0] and "0.10 out of 1" in said[0]
    assert f"{DEFAULT_STACK}'s answers" in said[1] and "0.90 out of 1" in said[1]


def test_the_ask_page_says_nothing_for_a_type_with_no_record(tmp_path, monkeypatch):
    """The run holds numeric and factual questions; this one reads as
    comparative, so there is no record to stand behind."""
    _run(tmp_path)
    ui = _ask(monkeypatch, tmp_path,
              question="How did Apple's revenue compare with Microsoft's in FY2024?")
    assert not _said(ui)
