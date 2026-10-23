"""A saved run's scores split by question type and by difficulty (#170).

The Results page shows each group's score beside its question count, and the
Ask page's track record (#168) reads the same groups through
:func:`default_breakdown`, so both pick the same configuration, the same
benchmark's questions and the same scores. Nothing here reads files: the
caller hands over a run's summary and its reader of one configuration's
saved questions, as ``src/app/state.py`` provides them.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from operator import attrgetter
from typing import Any, NamedTuple, TypeVar

from src.stack import DEFAULT_STACK

Item = TypeVar("Item")

# A group with fewer scored questions than this is marked too few to read. On
# the team benchmark one question moves a group of 4 by 25 points, and a group
# of 10 by 10. Its comparative and unanswerable questions (4 each) fall below
# the line; temporal and hard (10 each) are the smallest groups it scores.
MIN_GROUP_QUESTIONS = 10

# The cutoff "found the right passage" is judged at, the top 10 the issue
# asks for (#169). It is not the run's own: the full matrix scores its Recall
# and MRR at 16 (FINAL_K), so a right passage ranked 11 to 16 counts there
# and not here. Ranks are worked out from the saved passages, not from MRR.
FOUND_WITHIN = 10

# What every configuration is measured from: naive BM25, the row each design
# choice was added to. Both the before and after view and the chart compare
# against it, so neither can be reading a different baseline from the other.
BASELINE = "C0"

# The smallest gap the chart's title calls a difference. Its bars are labelled
# to the whole point, so two scores inside one of each other are drawn the
# same and are not worth a word either.
_A_POINT = 0.005


class Group(NamedTuple):
    """One question type or difficulty, and how the questions in it scored."""

    name: str
    questions: int        # how many questions are in the group
    scored: int           # how many of them have a score on the measure
    score: float | None   # the mean of those scores, None when none has one

    @property
    def readable(self) -> bool:
        """Whether enough of its questions were scored for the mean to mean something."""
        return self.scored >= MIN_GROUP_QUESTIONS


def found(outcome: Any) -> bool | None:
    """Whether the right passage ranked in the top 10, or None for an unscored question."""
    if not outcome.scored:
        return None
    return outcome.rank is not None and outcome.rank <= FOUND_WITHIN


class Measure(NamedTuple):
    """One way a saved question is scored: its heading, and its score, None where it has none."""

    title: str
    score: Callable[[Any], float | None]


FOUND = Measure(f"Found the right passage in the top {FOUND_WITHIN}", found)
# Scored by the LLM judge, or for an abstention by whether abstaining was
# right, so not "judged" correctness alone.
CORRECTNESS = Measure("Answer correctness", attrgetter("correctness"))
MEASURES = (FOUND, CORRECTNESS)
# How the questions are split: the words a headline uses, and a question's label.
GROUPINGS = (("question type", attrgetter("question_type")),
             ("difficulty", attrgetter("difficulty")))


class Split(NamedTuple):
    """One measure's groups under one grouping, such as correctness by question type."""

    measure: Measure
    grouping: str
    groups: list[Group]


def group_scores(items: Iterable[Item], group: Callable[[Item], str | None],
                 score: Callable[[Item], float | None]) -> list[Group]:
    """Each group's question count and mean score, the largest group first.

    An item ``group`` names no group for is left out. One ``score`` gives
    None, NaN or an infinity counts toward its group's questions but not
    toward its mean.
    """
    members: dict[str, list[float | None]] = {}
    for item in items:
        name = group(item)
        if name is not None:
            members.setdefault(name, []).append(score(item))
    groups = []
    for name, scores in members.items():
        measured = [float(value) for value in scores
                    if value is not None and math.isfinite(value)]
        groups.append(Group(name, len(scores), len(measured),
                            sum(measured) / len(measured) if measured else None))
    return sorted(groups, key=lambda group: (-group.questions, group.name))


def configurations_read(run: Mapping[str, Any], benchmark: str) -> list[str]:
    """The run's configurations, in its order, whose saved questions on
    ``benchmark`` were read. A summary's own scores do not make one."""
    return [item["config_id"] for item in run["configurations"]
            if item.get("questions_read") and benchmark in item["benchmark_metrics"]]


def default_breakdown(run: Mapping[str, Any], benchmark: str,
                      outcomes: Callable[[str, str], Mapping[str, Any]]) -> list[Split]:
    """The default configuration's questions on ``benchmark``, split by each
    grouping and scored on each measure, in that order.

    ``outcomes(run_dir, config_id)`` reads one configuration's saved
    questions. Empty unless the default configuration's questions on
    ``benchmark`` were read. A split with a single group, or with no question
    scored, is left out, since it has no strong or weak side.
    """
    if DEFAULT_STACK not in configurations_read(run, benchmark):
        return []
    questions = [outcome for outcome in outcomes(run["run_dir"], DEFAULT_STACK).values()
                 if outcome.benchmark == benchmark]
    splits = []
    for measure in MEASURES:
        for grouping, label in GROUPINGS:
            groups = group_scores(questions, label, measure.score)
            if len(groups) > 1 and any(group.scored for group in groups):
                splits.append(Split(measure, grouping, groups))
    return splits


class ConfigurationScore(NamedTuple):
    """How one configuration scored on one measure, over one benchmark's questions."""

    config_id: str
    name: str             # as the run saved it, the id where it saved none
    questions: int
    scored: int
    score: float | None   # the mean of those scores, None when none has one


def measured_by_configuration(run: Mapping[str, Any], benchmark: str,
                              outcomes: Callable[[str, str], Mapping[str, Any]],
                              measure: Measure = FOUND) -> list[ConfigurationScore]:
    """Each configuration's score on ``measure``, in the run's own order (#166).

    Only a configuration whose saved questions were read is scored here: the
    measure reads a question's own outcome, which a summary's means do not
    carry. Scored through :func:`group_scores`, so a configuration here and a
    group in :func:`default_breakdown` count a question the same way.
    """
    names = {item["config_id"]: item.get("name") or item["config_id"]
             for item in run["configurations"]}
    scores = []
    for config_id in configurations_read(run, benchmark):
        questions = [outcome for outcome in outcomes(run["run_dir"], config_id).values()
                     if outcome.benchmark == benchmark]
        for group in group_scores(questions, lambda _, named=config_id: named, measure.score):
            scores.append(ConfigurationScore(config_id, names.get(config_id, config_id),
                                             group.questions, group.scored, group.score))
    return scores


def in_ten(score: float) -> str:
    """A score as the chart's title states it: "8 in 10".

    Never "10 in 10" short of every question, since 96% rounded up reads as a
    claim the number does not make, and "fewer than 1 in 10" rather than
    "0 in 10", which reads as none at all. Exactly none is fewer than one in
    ten as well, and is worded the same.
    """
    if score >= 1:
        return "10 in 10"
    rounded = min(round(score * 10), 9)
    return f"{rounded} in 10" if rounded else "fewer than 1 in 10"


def _worded(first: float, second: float) -> tuple[str, str]:
    """Two scores worded so that one sentence can state both and still tell
    them apart: in tenths, or as percentages where tenths would collide."""
    if in_ten(first) == in_ten(second):
        return f"{first:.0%}", f"{second:.0%}"
    return in_ten(first), in_ten(second)


def _sentence_case(name: str) -> str:
    """``name`` to open a sentence, leaving the rest as the run wrote it, so
    that "naive BM25" does not come back as "Naive bm25"."""
    return name[:1].upper() + name[1:]


def found_headline(scores: Sequence[ConfigurationScore], *,
                   default: str = DEFAULT_STACK, baseline: str = BASELINE) -> str | None:
    """The chart's title, built from the numbers the chart draws, or None
    where the default configuration has no score to headline.

    The default is compared against ``baseline`` and then against the best
    score on the chart, and a configuration beating the default is named
    rather than left for the reader to spot. Both comparisons are made on the
    scores themselves, and two that would read alike in tenths are stated as
    percentages instead, so the sentence never says "up from" and then gives
    the same figure twice.

    A configuration is ahead only by more than a point, since the chart is
    drawn to the point and a difference it cannot show is not one to announce.
    """
    scored = {item.config_id: item for item in scores if item.score is not None}
    if default not in scored:
        return None
    mine = scored[default].score
    said = f"{_sentence_case(scored[default].name)} finds the right passage for "
    against = scored.get(baseline) if baseline != default else None
    if against is None:
        sentence = f"{said}{in_ten(mine)} questions."
    elif abs(mine - against.score) <= _A_POINT:
        sentence = f"{said}{in_ten(mine)} questions, the same as {against.name}."
    else:
        here, there = _worded(mine, against.score)
        moved = "up" if mine > against.score else "down"
        sentence = f"{said}{here} questions, {moved} from {there} with {against.name}."
    # Named whoever they are: the baseline itself scoring highest is worth
    # saying twice over, since the reader has just been told the default is
    # the one to use.
    ahead = [item for item in scored.values() if item.score > mine + _A_POINT]
    if ahead:
        best = max(item.score for item in ahead)
        top = [item for item in ahead if item.score >= best - _A_POINT]
        here, there = _worded(best, mine)
        finds = "finds" if len(top) == 1 else "find"
        sentence += (f" {_sentence_case(_names([item.name for item in top]))} {finds} more, "
                     f"at {here} against {there}.")
    return sentence


def _names(names: Sequence[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _counts(groups: Sequence[Group]) -> str:
    """A score's support: "over 12", or "each, over 12 and 10" for a tie."""
    over = _names([str(group.scored) for group in groups])
    return f"over {over}" if len(groups) == 1 else f"each, over {over}"


def headline(groups: Sequence[Group], grouping: str) -> str | None:
    """The strongest and weakest readable groups in words, or None with fewer than two.

    ``grouping`` names what the groups are, such as "question type". Groups
    are compared at the two places the page shows, and every group tied at
    the top or the bottom is named.
    """
    readable = [group for group in groups if group.readable and group.score is not None]
    if len(readable) < 2:
        return None
    shown = {group.name: round(group.score, 2) for group in readable}
    high, low = max(shown.values()), min(shown.values())
    if high == low:
        return f"By {grouping}, every group with enough questions scores {high:.2f}."
    strongest = [group for group in readable if shown[group.name] == high]
    weakest = [group for group in readable if shown[group.name] == low]
    return (f"By {grouping}, {_names([group.name for group in strongest])} questions score "
            f"highest ({high:.2f} {_counts(strongest)}) and "
            f"{_names([group.name for group in weakest])} lowest "
            f"({low:.2f} {_counts(weakest)}).")
