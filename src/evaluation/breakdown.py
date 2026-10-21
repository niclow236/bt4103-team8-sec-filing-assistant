"""A saved run's scores split by question type and by difficulty (#170).

The Results page shows each group's score beside its question count, and the
Ask page's track record (#168) reads the same groups, so both say the same
thing about a kind of question. Nothing here reads files or knows a metric:
the caller says how to name an item's group and how to score it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import NamedTuple, TypeVar

Item = TypeVar("Item")

# A group with fewer scored questions than this is marked too few to read. On
# the team benchmark one question moves a group of 4 by 25 points, and a group
# of 10 by 10. Its comparative and unanswerable questions (4 each) fall below
# the line; temporal and hard (10 each) are the smallest groups it scores.
MIN_GROUP_QUESTIONS = 10


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


def group_scores(items: Iterable[Item], group: Callable[[Item], str | None],
                 score: Callable[[Item], float | None]) -> list[Group]:
    """Each group's question count and mean score, the largest group first.

    An item ``group`` names no group for is left out. One ``score`` gives
    None counts toward its group's questions but not toward its mean.
    """
    members: dict[str, list[float | None]] = {}
    for item in items:
        name = group(item)
        if name is not None:
            members.setdefault(name, []).append(score(item))
    groups = []
    for name, scores in members.items():
        measured = [float(value) for value in scores if value is not None]
        groups.append(Group(name, len(scores), len(measured),
                            sum(measured) / len(measured) if measured else None))
    return sorted(groups, key=lambda group: (-group.questions, group.name))


def headline(groups: Sequence[Group], grouping: str) -> str | None:
    """The strongest and weakest readable group in words, or None with fewer than two.

    ``grouping`` names what the groups are, such as "question type".
    """
    readable = [group for group in groups if group.readable and group.score is not None]
    if len(readable) < 2:
        return None
    strongest = max(readable, key=lambda group: group.score)
    weakest = min(readable, key=lambda group: group.score)
    if strongest.score == weakest.score:
        return f"By {grouping}, every group with enough questions scores {strongest.score:.2f}."
    return (f"By {grouping}, {strongest.name} questions score highest "
            f"({strongest.score:.2f} over {strongest.scored}) and {weakest.name} lowest "
            f"({weakest.score:.2f} over {weakest.scored}).")
