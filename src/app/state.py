"""What the app keeps between one Ask and the next (#36).

Streamlit runs the page's script again on every widget change, so whatever is
worth keeping has to be held outside it. ``app.py`` keeps the indexes, the
embedding model and each built configuration with ``st.cache_resource``. This
module holds what a repeated question should not pay for twice, the answer it
was given, and the stopwatch behind the line that says how long one took.
"""

from __future__ import annotations

from collections import OrderedDict
from threading import Event, Lock
from time import perf_counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.rag.records import Answer
    from src.stack import Stack

# What a caller passes with a question that does not change its answer:
# ``parsed`` is read from the question, and ``on_token`` only shows the
# answer arriving.
NOT_PART_OF_THE_ANSWER = ("parsed", "on_token")

# How many answers one configuration keeps. Each holds the passages it was
# written from, sixteen of them, so the memory of a process left running
# would otherwise grow with every new question. Past this many the answer
# asked for longest ago is dropped, and asking it again asks the model again.
ANSWERS_KEPT = 128


class Remembered:
    """A built configuration that gives a repeated question the answer it gave before.

    The app asks through one of these in place of the ``Stack`` it wraps.
    ``load_stack`` makes one per configuration and ``st.cache_resource`` keeps
    it for the process, so the answers are shared by every browser session and
    last until the app restarts. An answer is kept by the question and by what
    it was asked with: the ``Query``, which carries the sidebar's companies,
    years and Items, and any setting the caller overrode.

    Only a finished answer is kept. A failure is not, so a question that met a
    busy provider is asked again, and neither is an answer the model left
    malformed or cut short (``parse_error``, ``truncated``): it is shown for
    what it is, and the next Ask gets another try and not the same broken
    text until the app restarts. At most ``limit`` answers are kept, the ones
    asked for most recently.

    The sessions run on threads of one process, so two can ask the same
    question at once. The second waits for the first and is given its answer:
    asked twice, a slow local model would do minutes of work twice and could
    give the two sessions different answers. Different questions do not wait
    for each other. If the first fails, or its answer is not kept, a session
    that was waiting asks in its turn.

    The answers are held in a dict and not by ``st.cache_data``, which #36
    names. A function cached that way replays the page elements drawn while
    it ran, and raises ``CacheReplayClosureError`` on its second call when
    they were drawn on a placeholder made outside it. That is how an answer
    is streamed to the page, through the ``on_token`` a caller passes here.
    """

    def __init__(self, stack: Stack, limit: int = ANSWERS_KEPT) -> None:
        if limit < 1:
            raise ValueError("limit must be at least 1")
        self.stack = stack
        self.limit = limit
        self._answers: OrderedDict[tuple[Any, ...], Answer] = OrderedDict()
        # The questions being answered now, each with the event its asker sets
        # when it is done. Both dicts are only touched while holding the lock,
        # and the lock is never held while the stack answers.
        self._being_answered: dict[tuple[Any, ...], Event] = {}
        self._lock = Lock()

    def answer(self, question: str, **overrides: Any) -> Answer:
        """The answer to ``question``, asked of the stack only the first time.

        ``overrides`` go to ``Stack.answer`` with that first request. A
        remembered answer comes back at once, and so does one another session
        was already being given, so nothing is streamed to ``on_token`` for
        either.
        """
        asked_with = tuple(sorted(
            (name, value) for name, value in overrides.items()
            if name not in NOT_PART_OF_THE_ANSWER
        ))
        key = (question, asked_with)
        while True:
            with self._lock:
                if key in self._answers:
                    self._answers.move_to_end(key)
                    return self._answers[key]
                done = self._being_answered.get(key)
                if done is None:
                    done = self._being_answered[key] = Event()
                    break
            # Another session is asking this now. Wait for it, then look again:
            # its answer is there, or it failed and this one asks in its turn.
            done.wait()
        try:
            answer = self.stack.answer(question, **overrides)
            if _finished(answer):
                with self._lock:
                    self._answers[key] = answer
                    while len(self._answers) > self.limit:
                        self._answers.popitem(last=False)
            return answer
        finally:
            # Whatever happened, the question is no longer being answered and
            # the sessions waiting for it are woken.
            with self._lock:
                del self._being_answered[key]
            done.set()


def _finished(answer: Answer) -> bool:
    """Whether the model wrote the whole of a well-formed answer."""
    return answer.parse_error is None and not answer.truncated


class Stopwatch:
    """How long a ``with`` block took: ``seconds``, once the block has ended."""

    seconds: float | None = None

    def __enter__(self) -> Stopwatch:
        self._started = perf_counter()
        return self

    def __exit__(self, *error: object) -> None:
        self.seconds = perf_counter() - self._started
