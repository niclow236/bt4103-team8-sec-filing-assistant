"""The data records the RAG stage hands to the app and the evaluation harness.

Kept apart from the prompt builder, the generator and the citation resolver
that fill them, the same way ``src/retrieval/records.py`` is kept apart from
the retrievers, so the app, the harness and the verifier can all agree on one
shape without importing each other. A generator imports this module; nothing
here imports a generator.

Four records, each answering a question the stage would otherwise answer in
several places, and the schema the model's output is held to:

``GenerationConfig`` is what produced an answer: provider, model, temperature
and which prompt template was rendered. It rides on the ``Answer`` rather than
being tracked alongside it, because the evaluation harness runs several
configurations over the same question and has to attribute every answer it is
handed without threading extra state through the call -- the same reason
``RetrievedPassage`` carries ``retriever``.

``GroundedAnswer`` is the shape the model is made to answer in: whether the
sources answer the question, then the answer as sentences, each listing the
source numbers it draws on. It is a Pydantic model rather than a dataclass
because it is the one record here that is filled by something untrusted. Its
JSON schema is handed to Ollama as the output format, so decoding is
constrained to it, and the same model validates what comes back. Everything
else in this module is built by our own code and checks itself in
``__post_init__``.

``Generation`` is what the model wrote, before anything is made of it: the
answer as parsed, the same answer as prose with its markers, what produced it,
how long it took and how many tokens it used. The citation resolver reads the
answer out of it and the metrics track reads the rest, and neither needs the
client that filled it.

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

from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import lru_cache
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, create_model

from ..retrieval.records import RetrievedPassage
from .constants import ABSTAIN_PHRASE


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

    provider: str          # "ollama", the local runtime that served the model; what the generator dispatches on
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


def render_sentence(text: str, sources: Iterable[int]) -> str:
    """One sentence as the app shows it and the resolver (#31) reads it.

    The text, then a marker per source in the form the resolver matches:
    "Revenue was $5 billion. [1][3]". A sentence that cites nothing is shown
    bare, so that it reads as unsupported rather than as supported.
    """
    markers = "".join(f"[{number}]" for number in sources)
    return f"{text.strip()} {markers}" if markers else text.strip()


class CitedSentence(BaseModel):
    """One sentence of a model's answer, and the numbers of the sources it draws on.

    ``sources`` may be empty. Rule 5 asks the model to say what the sources do
    not cover, and a sentence saying so draws on no source; forcing a number
    onto it would dress an unsupported sentence in a citation, which is worse
    than leaving it bare for the verifier (#32) to flag.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str
    sources: tuple[int, ...]


class GroundedAnswer(BaseModel):
    """The shape a model answers in, and what its answer was once validated.

    ``answerable`` comes first, so the model commits to whether the sources
    bear on the question before it writes anything, and a model that says
    they do not is not then asked to fill ``sentences`` from memory. Both
    fields are required, so the schema Ollama decodes against always has the
    citations field in it.

    Use :meth:`for_sources` rather than this class to talk to a model: it
    narrows ``sources`` to the numbers the prompt actually showed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    answerable: bool
    sentences: tuple[CitedSentence, ...]

    @classmethod
    def for_sources(cls, n_sources: int) -> type[GroundedAnswer]:
        """This model with every source number held to 1..``n_sources``.

        Its JSON schema is what Ollama decodes against, so a model shown eight
        sources cannot write a ninth: the constraint is on the tokens it may
        produce, not a check made after it has produced them. Validating with
        the same class then refuses an out-of-range number from a runtime
        that ignored the schema. Parse with it, then :meth:`model_validate`
        the result into a plain ``GroundedAnswer`` to hand on.
        """
        if n_sources < 1:
            raise ValueError(f"n_sources must be at least 1, got {n_sources}")
        return _for_sources(n_sources)

    @property
    def abstained(self) -> bool:
        """Whether this is a refusal to answer from the sources.

        True when the model said the sources do not bear on the question, and
        also when it said they do and then wrote nothing but blank sentences:
        an empty answer is not an answer. Sentences written after
        ``answerable`` came back false are not shown, since by the model's own
        account they cannot be from the sources.
        """
        return not self.answerable or not any(s.text.strip() for s in self.sentences)

    def render(self) -> str:
        """The answer as prose with inline markers, or the abstain sentence.

        Blank sentences are skipped rather than shown as a stray marker.
        """
        if self.abstained:
            return ABSTAIN_PHRASE
        return " ".join(
            render_sentence(s.text, s.sources) for s in self.sentences if s.text.strip()
        )


@lru_cache(maxsize=None)
def _for_sources(n_sources: int) -> type[GroundedAnswer]:
    """Build :meth:`GroundedAnswer.for_sources` once per source count."""
    number = Annotated[int, Field(ge=1, le=n_sources)]
    sentence = create_model(
        "CitedSentence", __base__=CitedSentence, sources=(tuple[number, ...], ...)
    )
    return create_model(
        "GroundedAnswer", __base__=GroundedAnswer, sentences=(tuple[sentence, ...], ...)
    )


@dataclass(frozen=True)
class Generation:
    """One model response to one prompt: what it wrote, and what writing it cost.

    ``raw`` is the JSON exactly as the model emitted it, kept so a results row
    can be traced back to the output that produced it. ``answer`` is that JSON
    validated into a ``GroundedAnswer``, or None when it could not be, and
    ``parse_error`` then says why: almost always because the output was cut
    off at the token limit and the JSON never closed. ``text`` is the answer
    as prose with its markers, which is what the app shows and what the
    resolver (#31) reads. For an answer that did not parse it holds what
    arrived before the cut rather than nothing: the closed sentences with
    their markers, then the text of the one being written, without, so a
    truncated answer is still shown for what it is.

    ``input_tokens`` and ``output_tokens`` are Ollama's counts, None where it
    did not report one, which is the truth of it rather than a zero that would
    average in as free.

    ``stop_reason`` is Ollama's word for why it stopped, kept as given.
    :attr:`truncated` is the reading every consumer needs: an answer cut off
    at the token limit has lost its last citation, and the resolver should
    know that before it flags the final sentence as unsupported.
    """

    text: str
    answer: GroundedAnswer | None
    raw: str
    config: GenerationConfig
    latency_ms: float          # wall-clock time of the call, request out to last token in
    input_tokens: int | None
    output_tokens: int | None
    stop_reason: str | None
    parse_error: str | None = None

    # Ollama's word for "hit the output limit".
    _TRUNCATED_REASONS = frozenset({"length"})

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must not be negative, got {self.latency_ms}")
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        if (self.answer is None) == (self.parse_error is None):
            raise ValueError("a generation has either a parsed answer or a parse_error, exactly one")

    @property
    def truncated(self) -> bool:
        """Whether the output was cut off at the token limit rather than finished."""
        return self.stop_reason in self._TRUNCATED_REASONS

    def to_dict(self) -> dict[str, Any]:
        """Return the JSON-compatible representation for results files."""
        return {
            "text": self.text,
            "answer": None if self.answer is None else self.answer.model_dump(mode="json"),
            "raw": self.raw,
            "config": self.config.to_dict(),
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "stop_reason": self.stop_reason,
            "truncated": self.truncated,
            "parse_error": self.parse_error,
        }


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
class SentenceCitations:
    """Citation checks for one sentence, not a judgement of factual support.

    ``raw_text`` keeps the model's markers; ``text`` drops unresolved ones.
    Non-positive markers cannot be Citations and are kept in ``invalid_markers``.
    A fallback block from unparseable output is conservatively checked as one
    unit, with ``incomplete`` True, rather than guessing its sentence boundaries.
    """

    raw_text: str
    text: str
    citations: tuple[Citation, ...]
    invalid_markers: tuple[str, ...] = ()
    incomplete: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "invalid_markers", tuple(self.invalid_markers))

    @property
    def flagged(self) -> bool:
        """Whether this sentence needs a visible citation warning."""
        return (
            self.incomplete
            or bool(self.invalid_markers)
            or not self.citations
            or any(not citation.resolved for citation in self.citations)
        )

    @property
    def warning_text(self) -> str:
        """What a citation warning shows: the cleaned text, or the model's own
        text when cleaning left nothing, so a warning is never blank."""
        return self.text or self.raw_text

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_text": self.raw_text,
            "text": self.text,
            "citations": [citation.to_dict() for citation in self.citations],
            "invalid_markers": list(self.invalid_markers),
            "incomplete": self.incomplete,
            "flagged": self.flagged,
        }


@dataclass(frozen=True)
class VerificationCheck:
    """One auditable check; unknown evidence is never counted as support."""

    kind: str
    status: str
    sentence_index: int | None
    claim: str
    reason: str
    figure: str | None = None
    value: str | None = None
    unit: str | None = None
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"passage", "fact", "groundedness", "output"}:
            raise ValueError(f"Unknown verification kind: {self.kind}")
        if self.status not in {"supported", "mismatch", "unverified", "not_applicable"}:
            raise ValueError(f"Unknown verification status: {self.status}")
        object.__setattr__(self, "evidence", tuple(self.evidence))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self) | {"evidence": list(self.evidence)}


@dataclass(frozen=True)
class VerificationResult:
    """Per-question checks, with explicit denominators for evaluation.

    The numeric support rate is a consistency proxy, not semantic faithfulness.
    Unverified checks remain in its denominator; abstentions have no score.
    """

    question_type: str
    checks: tuple[VerificationCheck, ...]
    version: str = "numeric-grounding-v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))

    @property
    def warnings(self) -> tuple[VerificationCheck, ...]:
        return tuple(c for c in self.checks if c.status in {"mismatch", "unverified"})

    @property
    def numeric_support_rate(self) -> float | None:
        checks = [c for c in self.checks if c.kind in {"passage", "fact"}
                  and c.status != "not_applicable"]
        return sum(c.status == "supported" for c in checks) / len(checks) if checks else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "question_type": self.question_type,
            "checks": [c.to_dict() for c in self.checks],
            "counts": {s: sum(c.status == s for c in self.checks) for s in
                       ("supported", "mismatch", "unverified", "not_applicable")},
            "numeric_support_rate": self.numeric_support_rate,
        }


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
    or not. Each must agree with its position: ``[n]`` resolves to
    ``passages[n - 1]`` when that source exists and is unresolved when it does
    not. That is the invariant that makes a rendered citation trustworthy: the
    record cannot show ``[1]`` with source 3's metadata, claim support from a
    source the model was never shown, or flag a real source as unsupported.

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
    sentences: tuple[SentenceCitations, ...] = ()
    parse_error: str | None = None
    truncated: bool = False

    verification: VerificationResult | None = None

    def __post_init__(self) -> None:
        # Tuples rather than lists, so a caller that built them as lists gets
        # the same immutable record as one that did not.
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "passages", tuple(self.passages))
        object.__setattr__(self, "sentences", tuple(self.sentences))

        shown = [passage.chunk_id for passage in self.passages]
        if len(set(shown)) != len(shown):
            raise ValueError("passages must not contain duplicate chunk_ids")

        markers = [citation.marker for citation in self.citations]
        if len(set(markers)) != len(markers):
            raise ValueError("citations must hold one entry per marker")

        # Marker [n] names passages[n - 1], so each citation must say exactly
        # what that position holds: its chunk_id when the marker is in range,
        # None when it is not. Checking membership alone would let [1] render
        # with source 3's metadata, or flag a real source as unsupported.
        mismatched = [
            f"[{citation.marker}] names {citation.chunk_id} but source is {numbered}"
            for citation in self.citations
            if citation.chunk_id != (numbered := (
                shown[citation.marker - 1] if citation.marker <= len(shown) else None
            ))
        ]
        if mismatched:
            raise ValueError(
                "citations do not match the numbered passages: " + "; ".join(mismatched)
            )

    @property
    def flagged_sentences(self) -> tuple[SentenceCitations, ...]:
        """Sentences with missing, unresolved, invalid or incomplete citations."""
        return tuple(sentence for sentence in self.sentences if sentence.flagged)

    @property
    def verification_warnings(self) -> tuple[VerificationCheck, ...]:
        return () if self.verification is None else self.verification.warnings

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
            "sentences": [sentence.to_dict() for sentence in self.sentences],
            "verification": None if self.verification is None else self.verification.to_dict(),
            "parse_error": self.parse_error,
            "truncated": self.truncated,
        }
