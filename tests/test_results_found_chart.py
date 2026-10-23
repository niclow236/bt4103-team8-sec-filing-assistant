"""The Results page's one chart, and the title it builds from its own numbers (#166)."""

import json

import pytest

from src.app.components import _found_chart, _found_role
from src.app.state import question_outcomes
from src.evaluation.breakdown import (
    BASELINE,
    FOUND,
    ConfigurationScore,
    found_headline,
    in_ten,
    measured_by_configuration,
)
from src.evaluation.report import TEAM, read_runs
from src.stack import DEFAULT_STACK
from tests.test_results_comparison import _open, _retrieval_row, _save_run

CAPTION = "No chart of found the right passage in the top 10"


def _score(config_id, score, *, name=None, scored=10):
    return ConfigurationScore(config_id, name or config_id, scored, scored, score)


def _luminance(colour):
    """A hex colour's relative luminance, as WCAG defines it."""
    channels = [int(colour[index:index + 2], 16) / 255 for index in (1, 3, 5)]
    linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4
              for value in channels]
    return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]


def _contrast(foreground, background):
    """The contrast ratio between two hex colours, lighter over darker."""
    first, second = _luminance(foreground), _luminance(background)
    lighter, darker = max(first, second), min(first, second)
    return (lighter + 0.05) / (darker + 0.05)


def _charts(tab):
    """Every Vega-Lite spec drawn in ``tab``, as dictionaries."""
    return [json.loads(chart.proto.spec) for chart in tab.get("vega_lite_chart")]


def _bars(spec):
    """The charted rows: the configuration, its score, and the colour role."""
    rows = next(iter(spec["datasets"].values()))
    return [(row["Configuration"], row["Score"], row["Role"]) for row in rows]


# --- the title ------------------------------------------------------------

def test_the_title_says_the_default_beat_the_baseline():
    line = found_headline([_score(BASELINE, 0.2, name="naive BM25"),
                           _score(DEFAULT_STACK, 0.8, name="hybrid with filters")])
    assert line == ("Hybrid with filters finds the right passage for 8 in 10 questions, "
                    "up from 2 in 10 with naive BM25.")


def test_the_title_says_the_default_lost_to_the_baseline():
    line = found_headline([_score(BASELINE, 0.8, name="naive BM25"),
                           _score(DEFAULT_STACK, 0.2, name="hybrid with filters")])
    assert "down from 8 in 10 with naive BM25." in line


def test_the_title_says_the_two_are_level():
    line = found_headline([_score(BASELINE, 0.4, name="naive BM25"),
                           _score(DEFAULT_STACK, 0.4, name="hybrid with filters")])
    assert line.endswith("for 4 in 10 questions, the same as naive BM25.")
    assert "up from" not in line and "down from" not in line


def test_a_configuration_ahead_of_the_default_is_named_not_hidden():
    line = found_headline([_score(BASELINE, 0.1), _score(DEFAULT_STACK, 0.5),
                           _score("C5", 0.9, name="C4 plus reranking")])
    assert line.endswith("C4 plus reranking finds more, at 9 in 10 against 5 in 10.")


def test_every_configuration_tied_at_the_top_is_named():
    line = found_headline([_score(BASELINE, 0.1), _score(DEFAULT_STACK, 0.5),
                           _score("C5", 0.9), _score("C6", 0.9)])
    assert "C5 and C6 find more" in line


def test_a_gap_too_small_for_the_chart_to_show_is_not_announced():
    """The bars are labelled to the point, so a 0.2-point gap is not a difference."""
    line = found_headline([_score(BASELINE, 0.5), _score(DEFAULT_STACK, 0.5),
                           _score("C5", 0.502)])
    assert line.endswith("the same as C0.")
    assert "finds more" not in line


def test_two_scores_alike_in_tenths_are_stated_as_percentages():
    """0.763 and 0.822 are both "8 in 10"; saying so twice would read as a tie."""
    line = found_headline([_score(BASELINE, 0.1), _score(DEFAULT_STACK, 0.763),
                           _score("C5", 0.822)])
    assert "at 82% against 76%." in line


def test_the_title_is_dropped_when_the_default_has_no_score():
    assert found_headline([_score(BASELINE, 0.5), _score("C5", 0.9)]) is None
    assert found_headline([_score(DEFAULT_STACK, None)]) is None


def test_the_baseline_alone_still_gets_a_title():
    line = found_headline([_score(DEFAULT_STACK, 0.6)])
    assert line == "C4 finds the right passage for 6 in 10 questions."


@pytest.mark.parametrize("score, said", [
    (0.0, "fewer than 1 in 10"), (0.04, "fewer than 1 in 10"),
    (0.05, "fewer than 1 in 10"), (0.15, "2 in 10"), (0.8, "8 in 10"),
    # 96% rounds to ten, which would claim every question.
    (0.96, "9 in 10"), (1.0, "10 in 10"),
])
def test_a_score_is_worded_without_overstating_it(score, said):
    assert in_ten(score) == said


# --- the chart ------------------------------------------------------------

def test_the_default_and_the_baseline_are_coloured_and_the_rest_are_grey():
    assert _found_role(DEFAULT_STACK) == "default"
    assert _found_role(BASELINE) == "baseline"
    assert _found_role("C2") == "other"


def test_the_three_roles_are_drawn_in_three_colours():
    scores = [_score(BASELINE, 0.2), _score("C2", 0.5), _score(DEFAULT_STACK, 0.8)]
    bars = _found_chart(scores).to_dict()["layer"][0]
    colours = bars["encoding"]["color"]["scale"]["range"]
    assert len(set(colours)) == 3


def test_each_bar_is_labelled_with_its_value():
    labels = _found_chart([_score(DEFAULT_STACK, 0.8)]).to_dict()["layer"][1]
    assert labels["mark"]["type"] == "text"
    assert labels["encoding"]["text"]["format"] == ".0%"


def test_every_colour_reads_on_a_light_and_a_dark_background():
    """Streamlit infers the theme from the background and documents that it
    gets this wrong just after the reader changes it in the settings menu,
    which is the one moment anyone compares the two. Picking colours by that
    reading put white labels on white. So no colour here is picked by it, and
    each has to carry both backgrounds on its own.

    Bars are shapes, which WCAG asks 3:1 of. The label is text, which it asks
    4.5:1 of; no colour reaches that against both, and this is the best of
    them, so the bar is the lower one it has to clear.
    """
    spec = _found_chart([_score(DEFAULT_STACK, 0.8)]).to_dict()
    bars, labels = spec["layer"]
    for colour in bars["encoding"]["color"]["scale"]["range"]:
        assert min(_contrast(colour, "#ffffff"), _contrast(colour, "#0e1117")) >= 3
    label = labels["mark"]["color"]
    assert min(_contrast(label, "#ffffff"), _contrast(label, "#0e1117")) >= 4.3


def test_the_chart_keeps_the_runs_order_and_a_fixed_axis():
    scores = [_score(BASELINE, 0.2), _score("C2", 0.5), _score(DEFAULT_STACK, 0.8)]
    spec = _found_chart(scores).to_dict()
    encoding = spec["layer"][0]["encoding"]
    assert encoding["x"]["sort"] == [BASELINE, "C2", DEFAULT_STACK]
    # Fixed, so one run's bars can be read against another's.
    assert encoding["y"]["scale"]["domain"] == [0, 1]


def test_a_configuration_without_a_score_is_left_off_the_chart():
    spec = _found_chart([_score(BASELINE, 0.2), _score("C2", None)]).to_dict()
    assert [row[0] for row in _bars(spec)] == [BASELINE]


# --- on the page ----------------------------------------------------------

def _run_with_ranks(root, ranks):
    """A run whose configurations found their right passage at ``ranks``."""
    return _save_run(root, {
        config_id: {"questions": [_retrieval_row(f"q{index}", rank)
                                  for index, rank in enumerate(found)]}
        for config_id, found in ranks.items()})


def test_the_page_charts_how_often_each_configuration_found_the_passage(tmp_path, monkeypatch):
    """C0 found 1 of 4 in the top 10 and C4 found 3, rank 12 being outside it.

    The rows themselves are checked in the unit tests above: AppTest hands
    them over as Arrow, while the title states both scores in words.
    """
    _run_with_ranks(tmp_path, {"C0": [1, None, None, None],
                               "C4": [1, 2, 3, 12]})
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert len(_charts(team)) == 1, "one chart per message, not one per metric"
    assert any("finds the right passage for 8 in 10 questions, up from 2 in 10 with C0"
               in markdown.value for markdown in team.markdown)


def test_a_run_that_saved_only_its_means_says_why_it_has_no_chart(tmp_path, monkeypatch):
    run = tmp_path / "means"
    (run / "C4").mkdir(parents=True)
    (run / "summary.json").write_text(json.dumps({"run_id": "means", "configurations": [
        {"config": {"id": "C4"}, "by_benchmark": {"handwritten": {"recall": 0.8}}}]}),
        encoding="utf-8")
    team = _open(monkeypatch, tmp_path).tabs[0]

    assert not _charts(team)
    assert any(CAPTION in caption.value for caption in team.caption)


def test_the_measure_is_the_breakdowns_so_both_count_a_question_alike(tmp_path):
    """#170's tables and this chart must not disagree about the same questions.

    Rank 11 is retrieved and counts toward the run's recall, and is still not
    found in the top 10, which is the distinction the two share.
    """
    _run_with_ranks(tmp_path, {"C4": [1, 11, None]})
    saved = read_runs(tmp_path)[0]

    scored = measured_by_configuration(saved, TEAM, question_outcomes)
    counted = [FOUND.score(outcome)
               for outcome in question_outcomes(saved["run_dir"], "C4").values()]
    assert scored[0].score == sum(counted) / len(counted) == pytest.approx(1 / 3)
