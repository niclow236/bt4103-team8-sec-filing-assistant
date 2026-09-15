"""The data records the RAG stage hands to the app and the evaluation harness.

Kept apart from the prompt builder, the generator and the citation resolver
that fill them, the same way ``src/retrieval/records.py`` is kept apart from
the retrievers, so the app, the harness and the verifier can all agree on one
shape without importing each other. A generator imports this module; nothing
here imports a generator.

Three records, each answering a question the stage would otherwise answer in
several places:

``GenerationConfig`` is what produced an answer: provider, model, temperature
and which prompt template was rendered. It rides on the ``Answer`` rather than
being tracked alongside it, because the evaluation harness runs several
configurations over the same question and has to attribute every answer it is
handed without threading extra state through the call -- the same reason
``RetrievedPassage`` carries ``retriever``.

``Citation`` is one ``[n]`` marker the model wrote, resolved back to the
passage it was shown as source ``n``. The model never writes a company, a year
or a URL; it is shown numbered passages and can only emit an integer, so a
citation is the integer plus what it resolved to. An integer that matches no
source is kept with ``resolved`` False rather than dropped, so a claim that
cites nothing real is shown as unsupported rather than as supported.

``Answer`` is the whole result for one question: the text, its citations, the
passages the model was shown, whether it declined to answer, and what
generated it.

Every record is frozen and holds no mutable default, so an answer handed to
the app, the verifier and the harness cannot be edited by one of them behind
the others' backs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ..retrieval.records import RetrievedPassage


@dataclass(frozen=True)
class GenerationConfig:
    """What generated an answer, as the harness reports results against it.

    The four fields are the ones a comparison in the ablation matrix varies:
    swap the provider or model, turn the temperature up, or render a different
    template, and the answers are no longer comparable to the last run's. Each
    is recorded as it was actually used, not as the constants say when someone
    later reads the results file, for the same reason ``ChunkedFiling`` records
    the budget its passages were cut with.
    """

    provider: str          # "anthropic", "openai", "ollama", ...; what the generator dispatches on
    # The model exactly as the provider names it, since two checkpoints of one
    # family answer differently and a results row has to say which one spoke.
    model: str
    # The id of the template in ``rag/constants.py`` that was rendered, rather
    # than the rendered text: the text is the size of the passages, and the
    # same id always renders the same prompt from the same passages.
    prompt_template_id: str
    temperature: float = 0.0   # 0 by default, so a rerun reproduces the answer

    def __post_init__(self) -> None:
        # Refused here rather than on the first call, so a bad setting is an
        # error where the config was built and not inside a provider's client.
        if self.temperature < 0:
            raise ValueError(f"temperature must not be negative, got {self.temperature}")
        object.__setattr__(self, "temperature", float(self.temperature))

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation for results files."""
        return asdict(self)


@dataclass(frozen=True)
class Citation:
    """One ``[n]`` marker in an answer, and the passage it resolved to.

    ``marker`` is the integer the model wrote. ``chunk_id`` is the passage that
    was numbered ``n`` in the prompt, or None when no passage was: a model can
    write ``[9]`` after being shown five sources, and that marker is kept here
    as unresolved rather than being dropped, so that the sentence carrying it
    can be flagged. The two fields agree by construction: a citation is
    resolved exactly when it names a passage.
    """

    marker: int
    chunk_id: str | None   # as the chunker assigned it, so the citation joins back to the corpus
    resolved: bool

    def __post_init__(self) -> None:
        if isinstance(self.marker, bool) or not isinstance(self.marker, int) or self.marker < 1:
            raise ValueError(f"marker must be a positive integer, got {self.marker!r}")
        if self.resolved != (self.chunk_id is not None):
            raise ValueError(
                f"citation [{self.marker}] is "
                f"{'resolved' if self.resolved else 'unresolved'} but "
                f"{'has no' if self.chunk_id is None else 'names a'} chunk_id"
            )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation for results files."""
        return asdict(self)


@dataclass(frozen=True)
class Answer:
    """One answered question, with everything needed to show, check and score it.

    ``passages`` are the sources the model was shown, in the order they were
    numbered, so ``passages[n - 1]`` is what marker ``[n]`` refers to. They are
    carried whole rather than as ids because the app renders each citation from
    the passage's own stored metadata -- company, fiscal year, Item, URL -- and
    the verifier checks the text against them, and neither should go back to
    the index for what the retriever already returned.

    ``citations`` holds one entry per distinct marker the model wrote, resolved
    or not. A resolved one must name a passage in ``passages``, which is the
    invariant that makes a rendered citation trustworthy: the record cannot say
    a claim is supported by a source the model was never shown.

    ``abstained`` is True when the model declined to answer from the passages
    it was given. An abstention is an answer, not an error: the harness counts
    it, and on a question the corpus cannot answer it is the correct output.
    """

    question: str
    text: str
    citations: tuple[Citation, ...]
    passages: tuple[RetrievedPassage, ...]
    abstained: bool
    config: GenerationConfig
    latency_ms: float | None = None   # wall-clock time of the generation call, for the metrics track

    def __post_init__(self) -> None:
        # Tuples rather than lists, so a caller that built them as lists gets
        # the same immutable record as one that did not.
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "passages", tuple(self.passages))

        shown = [passage.chunk_id for passage in self.passages]
        if len(set(shown)) != len(shown):
            raise ValueError("passages must not contain duplicate chunk_ids")

        markers = [citation.marker for citation in self.citations]
        if len(set(markers)) != len(markers):
            raise ValueError("citations must hold one entry per marker")

        # A resolved citation says "this claim rests on that passage", and the
        # only passages it may rest on are the ones the model was shown.
        unseen = sorted(
            citation.chunk_id for citation in self.citations
            if citation.resolved and citation.chunk_id not in shown
        )
        if unseen:
            raise ValueError(
                "resolved citations name passages the model was not shown: "
                + ", ".join(unseen)
            )

    @property
    def cited_passages(self) -> tuple[RetrievedPassage, ...]:
        """The passages the answer actually rests on, in marker order.

        A subset of ``passages``: the ones some resolved marker points at. The
        app lists these under the answer, and groundedness is measured against
        these rather than against everything retrieved.
        """
        by_id = {passage.chunk_id: passage for passage in self.passages}
        return tuple(
            by_id[citation.chunk_id]
            for citation in sorted(self.citations, key=lambda c: c.marker)
            if citation.resolved
        )

    @property
    def unresolved_markers(self) -> tuple[int, ...]:
        """The markers the model wrote that matched no source, in order."""
        return tuple(
            citation.marker
            for citation in sorted(self.citations, key=lambda c: c.marker)
            if not citation.resolved
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation for results files."""
        return {
            "question": self.question,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
            "passages": [
                asdict(passage) | {"sources": list(passage.sources)}
                for passage in self.passages
            ],
            "abstained": self.abstained,
            "config": self.config.to_dict(),
            "latency_ms": self.latency_ms,
        }
