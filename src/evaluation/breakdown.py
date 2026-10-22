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


def _names(groups: Sequence[Group]) -> str:
    names = [group.name for group in groups]
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def _counts(groups: Sequence[Group]) -> str:
    """A score's support: "over 12", or "each, over 12 and 10" for a tie."""
    over = _names([group._replace(name=str(group.scored)) for group in groups])
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
    return (f"By {grouping}, {_names(strongest)} questions score highest "
            f"({high:.2f} {_counts(strongest)}) and {_names(weakest)} lowest "
            f"({low:.2f} {_counts(weakest)}).")
