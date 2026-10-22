"""Grouping a run's questions by type and difficulty (#170)."""

from operator import itemgetter

from src.evaluation.breakdown import MIN_GROUP_QUESTIONS, Group, group_scores, headline


def _items(kind, scores):
    return [{"kind": kind, "score": score} for score in scores]


def test_each_group_counts_its_questions_and_averages_only_the_scored_ones():
    items = (_items("numeric", [1.0, 0.5, None]) + _items("factual", [0.0, 1.0])
             + [{"kind": None, "score": 1.0}])

    groups = group_scores(items, itemgetter("kind"), itemgetter("score"))

    # Largest group first; the unlabelled item is in no group.
    assert groups == [Group("numeric", 3, 2, 0.75), Group("factual", 2, 2, 0.5)]


def test_groups_of_one_size_are_ordered_by_name_and_a_group_with_no_score_has_none():
    groups = group_scores(_items("temporal", [None]) + _items("comparative", [None]),
                          itemgetter("kind"), itemgetter("score"))

    assert groups == [Group("comparative", 1, 0, None), Group("temporal", 1, 0, None)]


def test_a_true_or_false_score_is_averaged_as_a_rate():
    groups = group_scores(_items("numeric", [True, True, False, None]),
                          itemgetter("kind"), itemgetter("score"))

    assert groups == [Group("numeric", 4, 3, 2 / 3)]


def test_a_nan_or_infinite_score_counts_toward_its_group_but_not_its_mean():
    groups = group_scores(_items("numeric", [1.0, float("nan"), float("inf"), None]),
                          itemgetter("kind"), itemgetter("score"))

    assert groups == [Group("numeric", 4, 1, 1.0)]


def test_one_nan_score_leaves_the_headline_worked_out_from_the_other_scores():
    items = (_items("numeric", [1.0] * 10 + [float("nan")]) + _items("factual", [0.5] * 10)
             + _items("temporal", [0.0] * 10))

    groups = group_scores(items, itemgetter("kind"), itemgetter("score"))

    assert headline(groups, "question type") == (
        "By question type, numeric questions score highest (1.00 over 10) "
        "and temporal lowest (0.00 over 10).")


def test_a_group_is_readable_from_the_threshold_of_scored_questions_up():
    assert MIN_GROUP_QUESTIONS == 10
    assert Group("hard", 12, MIN_GROUP_QUESTIONS, 0.5).readable
    # Questions without a score do not make a group readable.
    assert not Group("hard", 12, MIN_GROUP_QUESTIONS - 1, 0.5).readable


def test_the_headline_names_the_strongest_and_weakest_readable_groups():
    groups = [Group("numeric", 88, 88, 0.8), Group("factual", 44, 44, 0.55),
              Group("temporal", 10, 10, 0.4),
              # Higher and lower than every readable group, but too few to read.
              Group("comparative", 4, 4, 1.0), Group("unanswerable", 4, 4, 0.0)]

    assert headline(groups, "question type") == (
        "By question type, numeric questions score highest (0.80 over 88) "
        "and temporal lowest (0.40 over 10).")


def test_no_headline_without_two_readable_groups_to_set_side_by_side():
    assert headline([Group("easy", 79, 79, 0.9), Group("hard", 9, 9, 0.1)], "difficulty") is None
    assert headline([Group("easy", 79, 0, None), Group("medium", 61, 61, 0.5)],
                    "difficulty") is None
    assert headline([], "difficulty") is None


def test_readable_groups_that_score_the_same_are_not_called_strongest_and_weakest():
    assert headline([Group("easy", 10, 10, 0.5), Group("medium", 10, 10, 0.5)],
                    "difficulty") == "By difficulty, every group with enough questions scores 0.50."


def test_every_group_tied_at_the_top_is_named():
    groups = [Group("numeric", 12, 12, 0.9), Group("factual", 11, 11, 0.9),
              Group("temporal", 10, 10, 0.5)]

    assert headline(groups, "question type") == (
        "By question type, numeric and factual questions score highest "
        "(0.90 each, over 12 and 11) and temporal lowest (0.50 over 10).")


def test_every_group_tied_at_the_bottom_is_named():
    groups = [Group("easy", 40, 40, 0.8), Group("medium", 30, 30, 0.4),
              Group("hard", 20, 20, 0.4), Group("expert", 10, 10, 0.4)]

    assert headline(groups, "difficulty") == (
        "By difficulty, easy questions score highest (0.80 over 40) "
        "and medium, hard and expert lowest (0.40 each, over 30, 20 and 10).")


def test_groups_that_show_the_same_score_to_two_places_are_tied():
    # Both are shown as 0.90, so naming one of them alone would disagree with the table.
    groups = [Group("numeric", 12, 12, 0.904), Group("factual", 11, 11, 0.896),
              Group("temporal", 10, 10, 0.5)]

    assert headline(groups, "question type") == (
        "By question type, numeric and factual questions score highest "
        "(0.90 each, over 12 and 11) and temporal lowest (0.50 over 10).")
