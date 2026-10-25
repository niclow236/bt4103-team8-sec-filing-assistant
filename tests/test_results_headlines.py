"""Selected-run headline tiles and their honest denominators (#165)."""

import json

from bs4 import BeautifulSoup
import pytest

from src.app import state
from src.app.components import _headline_icons
from src.evaluation import breakdown, headlines, report
from tests.test_results_comparison import _answer_row, _open, _retrieval_row, _save_run


def _answer(qid, correctness, abstained=False, latency=1000, **fields):
    row = _answer_row(qid, correctness)
    row["answer"].update(abstained=abstained, latency_ms=latency)
    return {**row, **fields}


def _fixture(root, *, name="run"):
    return _save_run(root, {
        "C0": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", 12),
                              _retrieval_row("q3", None), _retrieval_row("q4", "unscored"),
                              _retrieval_row("xbrl-a", 1)],
               "answers": [_answer("q1", 0.0, latency=4000),
                           _answer("q2", 0.5, latency=6000), _answer("q3", 1, latency=8000)]},
        "C4": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", 10),
                              _retrieval_row("q3", 11), _retrieval_row("q4", "unscored"),
                              _retrieval_row("xbrl-a", None)],
               "answers": [_answer("q1", 0.5, latency=1000),
                           _answer("q2", 1, abstained=True, latency=2000),
                           _answer("q3", 0.75, latency=3000)]}}, name=name, top_k=16)


def _tiles(root, benchmark=report.TEAM):
    state.question_outcomes.clear()
    return {tile.key: tile for tile in headlines.headline_tiles(
        report.read_runs(root)[0], benchmark, state.question_outcomes)}


def _metrics(tab):
    labels = {breakdown.FOUND.title, "Answer correctness (0–1)",
              "Refused rather than guessed", "Typical answer time (s)"}
    return {item.label: item for item in tab.metric if item.label in labels}


def test_tiles_use_default_rows_not_mean_recall_and_compare_the_same_cohort(tmp_path):
    _fixture(tmp_path)
    tiles = _tiles(tmp_path)
    assert list(tiles) == ["found", "correctness", "refused", "latency"]
    assert tiles["found"].count == 2 and tiles["found"].scored == 3
    assert tiles["found"].delta == pytest.approx(1 / 3)
    assert tiles["correctness"].value == 0.75
    assert tiles["correctness"].count is None
    assert tiles["correctness"].delta == 0.25
    assert tiles["refused"].count == 1
    assert tiles["refused"].delta == pytest.approx(1 / 3)
    assert tiles["latency"].value == 2 and tiles["latency"].delta == -4


def test_headlines_render_before_tables_with_help_deltas_and_ten_icons(tmp_path, monkeypatch):
    _fixture(tmp_path)
    app = _open(monkeypatch, tmp_path)
    metrics = _metrics(app.tabs[0])
    assert {label: item.value for label, item in metrics.items()} == {
        breakdown.FOUND.title: "2 of 3", "Answer correctness (0–1)": "0.75 / 1",
        "Refused rather than guessed": "1 of 3", "Typical answer time (s)": "2.00 s"}
    assert metrics[breakdown.FOUND.title].delta == "+33.3 pp vs C0"
    assert metrics["Answer correctness (0–1)"].delta == "+0.25 vs C0"
    assert metrics["Typical answer time (s)"].delta == "-4.00 s vs C0"
    assert "not end-to-end" in metrics["Typical answer time (s)"].proto.help
    assert all(item.proto.help for item in metrics.values())
    # Traversal order pins the tile section above the first table.
    kinds = [element.type for element in app.tabs[0]]
    assert kinds.index("metric") < kinds.index(app.tabs[0].dataframe[0].type)
    assert app.tabs[0].get("html")


def test_retrieval_only_run_hides_unavailable_answer_metrics(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C4": {"questions": [_retrieval_row("q1", None)]}})
    metrics = _metrics(_open(monkeypatch, tmp_path).tabs[0])
    assert list(metrics) == [breakdown.FOUND.title]
    assert metrics[breakdown.FOUND.title].value == "0 of 1"
    assert not metrics[breakdown.FOUND.title].delta


def test_absent_default_never_substitutes_another_configuration(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C1": {"questions": [_retrieval_row("q1", 1)]}})
    assert _tiles(tmp_path) == {}
    assert _metrics(_open(monkeypatch, tmp_path).tabs[0]) == {}


def test_mechanical_and_team_statistics_never_mix(tmp_path):
    _fixture(tmp_path)
    tiles = _tiles(tmp_path, report.MECHANICAL)
    assert list(tiles) == ["found"]
    assert tiles["found"].count == 0 and tiles["found"].scored == 1
    assert tiles["found"].delta == -1


def test_unequal_question_ids_hide_deltas_even_when_counts_match(tmp_path):
    _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", 1)], "answers": [_answer("q1", 0.5)]},
        "C4": {"questions": [_retrieval_row("q2", 1)], "answers": [_answer("q2", 1)]}})
    assert all(tile.delta is None for tile in _tiles(tmp_path).values())


def test_partial_answer_scores_have_their_own_denominators_and_no_unmatched_delta(tmp_path):
    _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", 1)],
               "answers": [_answer("q1", 1), _answer("q2", 1)]},
        "C4": {"questions": [_retrieval_row("q1", 1), _retrieval_row("q2", 1)],
               "answers": [_answer("q1", 0.4)]}})
    tiles = _tiles(tmp_path)
    assert tiles["found"].scored == 2
    assert tiles["correctness"].scored == tiles["refused"].scored == tiles["latency"].scored == 1
    assert tiles["correctness"].value == 0.4 and tiles["correctness"].delta is None


@pytest.mark.parametrize("bad", [None, "1000", True, -1, float("nan"), float("inf"), 10 ** 400])
def test_missing_or_invalid_latency_is_not_zero(tmp_path, bad):
    _save_run(tmp_path, {"C4": {"answers": [_answer("q1", 0, latency=bad)]}})
    tiles = _tiles(tmp_path)
    assert "latency" not in tiles
    assert tiles["correctness"].value == 0 and tiles["refused"].count == 0


def test_actual_zero_latency_is_available(tmp_path):
    _save_run(tmp_path, {"C4": {"answers": [_answer("q1", 0, latency=0)]}})
    assert _tiles(tmp_path)["latency"].value == 0


def test_answer_observations_without_retrieval_do_not_claim_no_results(tmp_path, monkeypatch):
    _save_run(tmp_path, {"C4": {"answers": [_answer("q1", 0.5, latency=0)]}})
    app = _open(monkeypatch, tmp_path)
    assert _metrics(app.tabs[0])["Typical answer time (s)"].value == "0.00 s"
    assert not any("No saved results" in info.value for info in app.tabs[0].info)


def test_missing_refusal_flag_is_not_counted_as_an_answer(tmp_path):
    row = _answer_row("q1", 0.5)
    _save_run(tmp_path, {"C4": {"answers": [row]}})
    assert list(_tiles(tmp_path)) == ["correctness"]


def test_saved_correctness_summary_is_a_mean_not_a_fabricated_correct_count(tmp_path):
    run = _save_run(tmp_path, {"C4": {"questions": [_retrieval_row("q1", 1)]}})
    (run / "answer-summary.json").write_text(json.dumps({"configurations": {"C4": {
        "questions": 150, "quality": {"correctness": {"mean": 0.3145333, "scored": 150}}}}}),
        encoding="utf-8")
    tile = _tiles(tmp_path)["correctness"]
    assert tile.value == 0.3145333 and tile.scored == 150
    assert tile.count is None and tile.delta is None
    assert "correctness" not in _tiles(tmp_path, report.MECHANICAL)


def test_model_report_rows_supply_refusals_and_timing_without_judging(tmp_path):
    run = _save_run(tmp_path, {"C4": {"questions": [_retrieval_row("q1", 1)]}})
    row = {"question_id": "q1", "answer": {"abstained": True, "latency_ms": 2500}}
    (run / "C4" / "report.json").write_text(json.dumps({"results": [row]}), encoding="utf-8")
    tiles = _tiles(tmp_path)
    assert tiles["refused"].count == 1 and tiles["latency"].value == 2.5
    assert "correctness" not in tiles


def test_icons_show_exact_count_accessibly_and_do_not_round_96_percent_to_all(tmp_path):
    html = _headline_icons(headlines.Headline("found", 0.96, 100, 96))
    soup = BeautifulSoup(html, "html.parser")
    assert "96 of 100" in soup.div["aria-label"]
    assert len(soup.select("span")) == 10
    assert "60.00%" in soup.select("span")[-1]["style"]
    assert sum("100.00%" in item["style"] for item in soup.select("span")) == 9


def test_reload_and_switching_runs_refresh_the_headlines(tmp_path, monkeypatch):
    run = _fixture(tmp_path, name="first")
    _save_run(tmp_path, {"C4": {"questions": [_retrieval_row("other", None)]}}, name="second")
    app = _open(monkeypatch, tmp_path)
    app.selectbox[0].select("first").run()
    assert _metrics(app.tabs[0])[breakdown.FOUND.title].value == "2 of 3"
    (run / "C4" / "questions.jsonl").write_text(json.dumps(_retrieval_row("q1", None)) + "\n",
                                               encoding="utf-8")
    app.button[0].click().run()
    assert not app.exception
    assert _metrics(app.tabs[0])[breakdown.FOUND.title].value == "0 of 1"
    app.selectbox[0].select("second").run()
    assert not app.exception
    assert _metrics(app.tabs[0])[breakdown.FOUND.title].value == "0 of 1"


@pytest.mark.parametrize("before,after,delta", [(1, 0, -1), (0, 1, 1), (0, 0, 0), (1, 1, 0)])
def test_deltas_include_negative_positive_and_real_ties(tmp_path, before, after, delta):
    _save_run(tmp_path, {
        "C0": {"questions": [_retrieval_row("q1", 1 if before else None)]},
        "C4": {"questions": [_retrieval_row("q1", 1 if after else None)]}})
    assert _tiles(tmp_path)["found"].delta == delta


def test_partial_run_warning_stays_above_the_headlines(tmp_path, monkeypatch):
    run = _fixture(tmp_path)
    manifest = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    manifest["configurations"][1]["stopped"] = {"question_id": "q5"}
    (run / "summary.json").write_text(json.dumps(manifest), encoding="utf-8")
    app = _open(monkeypatch, tmp_path)
    assert any("partial" in warning.value for warning in app.warning)
    order = [element.type for element in app.main]
    assert order.index("warning") < order.index("metric")
