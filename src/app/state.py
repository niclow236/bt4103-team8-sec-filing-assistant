"""What the app keeps between one Ask and the next (#36).

Streamlit runs the page's script again on every widget change, so whatever is
worth keeping has to be held outside it. ``app.py`` keeps the indexes, the
embedding model and each built configuration with ``st.cache_resource``. This
module holds what a repeated question should not pay for twice, the answer it
was given, and the stopwatch behind the line that says how long one took.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any

# What a caller passes with a question that does not change its answer:
# ``parsed`` is read from the question, and ``on_token`` only shows the
# answer arriving.
NOT_PART_OF_THE_ANSWER = ("parsed", "on_token")


class Remembered:
    """A built configuration that gives a repeated question the answer it gave before.

    The app asks through one of these in place of the ``Stack`` it wraps.
    ``load_stack`` makes one per configuration and ``st.cache_resource`` keeps
    it for the process, so the answers are shared by every browser session and
    last until the app restarts. An answer is kept by the question and by what
    it was asked with: the ``Query``, which carries the sidebar's companies,
    years and Items, and any setting the caller overrode. A failure is not
    kept, so a question that met a busy provider is asked again.

    The answers are held in a dict and not by ``st.cache_data``, which #36
    names. A function cached that way replays the page elements drawn while
    it ran, and raises ``CacheReplayClosureError`` on its second call when
    they were drawn on a placeholder made outside it. That is how an answer
    is streamed to the page, through the ``on_token`` a caller passes here.
    """

    def __init__(self, stack: Any) -> None:
        self.stack = stack
        self._answers: dict[tuple[Any, ...], Any] = {}

    def answer(self, question: str, **overrides: Any) -> Any:
        """The answer to ``question``, asked of the stack only the first time.

        ``overrides`` go to ``Stack.answer`` with that first request. A
        remembered answer comes back at once, so nothing is streamed to
        ``on_token`` for it.
        """
        asked_with = tuple(sorted(
            (name, value) for name, value in overrides.items()
            if name not in NOT_PART_OF_THE_ANSWER
        ))
        key = (question, asked_with)
        if key not in self._answers:
            self._answers[key] = self.stack.answer(question, **overrides)
        return self._answers[key]


class Stopwatch:
    """How long a ``with`` block took: ``seconds``, once the block has ended."""

    seconds: float | None = None

    def __enter__(self) -> Stopwatch:
        self._started = perf_counter()
        return self

    def __exit__(self, *error: object) -> None:
        self.seconds = perf_counter() - self._started
