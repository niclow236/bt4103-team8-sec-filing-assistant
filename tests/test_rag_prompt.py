"""The grounded prompt: what it shows the model, what it refuses, and that it is stable."""

import dataclasses
import hashlib
import re

import pytest

from src.rag.constants import (
    ABSTAIN_PHRASE,
    PROMPT_TEMPLATE_ID,
    SOURCE_HEADER,
    SOURCE_SECTION,
    SOURCE_SECTION_NO_ITEM,
    SOURCE_SEPARATOR,
    SOURCE_TABLE_TAG,
    SYSTEM_PROMPT,
    USER_PROMPT,
)
from src.rag.prompt import GroundedPrompt, build_prompt, render_source
from src.rag.records import Answer, Citation, GenerationConfig
from src.retrieval.records import RetrievedPassage


def _passage(chunk_id: str, rank: int, **overrides) -> RetrievedPassage:
    fields = dict(
        chunk_id=chunk_id, text=f"Text of {chunk_id}.", score=1.0 / rank, rank=rank,
        retriever="hybrid", ticker="AAA", company="Alpha Corp", fiscal_year=2024,
        item="7", title="Management's Discussion", url="https://example.test/" + chunk_id,
    )
    fields.update(overrides)
    return RetrievedPassage(**fields)


P1, P2, P3 = _passage("p1", 1), _passage("p2", 2, fiscal_year=2023), _passage("p3", 3, ticker="BBB", company="Beta Inc.")
QUESTION = "What was revenue?"


# --- shape ---------------------------------------------------------------------

def test_build_prompt_returns_the_template_id_the_rules_and_the_passages():
    prompt = build_prompt(QUESTION, [P1, P2, P3])
    assert isinstance(prompt, GroundedPrompt)
    assert prompt.template_id == PROMPT_TEMPLATE_ID
    assert prompt.system == SYSTEM_PROMPT
    assert prompt.passages == (P1, P2, P3)
    assert prompt.n_sources == 3


def test_the_user_message_holds_every_source_and_then_the_question():
    prompt = build_prompt(QUESTION, [P1, P2, P3])
    assert prompt.user.startswith("Sources:")
    assert prompt.user.endswith(f"Question: {QUESTION}")
    for marker, passage in enumerate((P1, P2, P3), start=1):
        assert f"[{marker}] " in prompt.user
        assert passage.text in prompt.user
    assert prompt.user.index("[1] ") < prompt.user.index("[2] ") < prompt.user.index("[3] ")


def test_to_messages_is_a_system_turn_then_a_user_turn():
    prompt = build_prompt(QUESTION, [P1])
    assert prompt.to_messages() == [
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.user},
    ]


def test_as_text_joins_the_two_for_a_single_string_model():
    prompt = build_prompt(QUESTION, [P1])
    assert prompt.as_text() == prompt.system + "\n\n" + prompt.user


def test_the_prompt_is_frozen():
    prompt = build_prompt(QUESTION, [P1])
    with pytest.raises(dataclasses.FrozenInstanceError):
        prompt.user = "changed"


# --- the source header --------------------------------------------------------

def test_a_source_header_names_company_ticker_year_and_item():
    assert render_source(2, P1).splitlines()[0] == (
        "[2] Alpha Corp (AAA), fiscal year 2024, Item 7: Management's Discussion"
    )


def test_a_source_shows_its_text_under_the_header_stripped():
    rendered = render_source(1, _passage("p", 1, text="\n  Some text.  \n\n"))
    assert rendered.splitlines()[1:] == ["Some text."]


def test_a_source_without_an_item_number_still_names_its_section():
    rendered = render_source(1, _passage("p", 1, item=None, title="Signatures"))
    assert rendered.splitlines()[0] == "[1] Alpha Corp (AAA), fiscal year 2024, Signatures"


def test_a_source_without_a_fiscal_year_says_so():
    rendered = render_source(1, _passage("p", 1, fiscal_year=None))
    assert "fiscal year unknown" in rendered.splitlines()[0]


def test_a_table_source_is_tagged():
    rendered = render_source(1, _passage("p", 1, content_type="table"))
    assert rendered.splitlines()[0].endswith("(table)")
    assert "(table)" not in render_source(1, P1)


def test_the_stored_url_is_never_rendered():
    # The metadata URL stays out of the prompt. A passage's own text is
    # rendered verbatim and may quote an address; that is content, not leakage.
    prompt = build_prompt(QUESTION, [P1, P2, P3])
    assert "example.test" not in prompt.as_text()
    quoting = _passage("q", 1, text="See www.example.org for details.")
    assert "www.example.org" in build_prompt(QUESTION, [quoting]).user
    assert "example.test" not in build_prompt(QUESTION, [quoting]).user


def test_a_passage_with_braces_or_a_blank_line_renders_verbatim():
    text = "Revenue {in millions}\n\n$100 [see note 1]"
    prompt = build_prompt("q {x}", [_passage("p", 1, text=text)])
    assert text in prompt.user
    assert prompt.user.endswith("Question: q {x}")


# --- the rules -----------------------------------------------------------------

def test_the_rules_tell_the_model_to_cite_by_number_and_when_to_abstain():
    assert "[1][3]" in SYSTEM_PROMPT
    assert "only the sources" in SYSTEM_PROMPT
    assert ABSTAIN_PHRASE in SYSTEM_PROMPT


def test_the_abstain_phrase_is_one_sentence_the_harness_can_match():
    assert ABSTAIN_PHRASE == ABSTAIN_PHRASE.strip()
    assert ABSTAIN_PHRASE.endswith(".")
    assert "\n" not in ABSTAIN_PHRASE


def test_the_template_id_is_bound_to_the_template_text():
    """A wording change fails here until the id is bumped alongside it.

    The id only means something if two results files that both say
    "grounded_v2" were produced by the same prompt. Nothing else ties the id
    to the text, so an edit would otherwise pass silently.
    """
    digest = hashlib.sha256(
        "\0".join(
            (
                SYSTEM_PROMPT,
                USER_PROMPT,
                SOURCE_HEADER,
                SOURCE_SECTION,
                SOURCE_SECTION_NO_ITEM,
                SOURCE_TABLE_TAG,
                SOURCE_SEPARATOR,
            )
        ).encode()
    ).hexdigest()[:16]
    assert (PROMPT_TEMPLATE_ID, digest) == ("grounded_v2", "b2cb6c5bc3aa94f5")


# --- numbering and determinism ----------------------------------------------------

def test_sources_are_numbered_by_rank_whatever_order_they_arrive_in():
    prompt = build_prompt(QUESTION, [P3, P1, P2])
    assert prompt.passages == (P1, P2, P3)
    assert prompt.user.index("Text of p1.") < prompt.user.index("Text of p2.") < prompt.user.index("Text of p3.")


def test_the_same_set_of_passages_gives_an_identical_prompt():
    a = build_prompt(QUESTION, [P1, P2, P3])
    b = build_prompt(QUESTION, (P3, P2, P1))
    c = build_prompt(QUESTION, [P2, P3, P1])
    assert a == b == c
    assert a.user == b.user == c.user


def test_the_prompt_is_stable_across_calls():
    assert build_prompt(QUESTION, [P1, P2]) == build_prompt(QUESTION, [P1, P2])


def test_tied_ranks_are_ordered_by_chunk_id():
    x, y = _passage("x", 1), _passage("y", 1)
    assert build_prompt(QUESTION, [y, x]).passages == (x, y)


def test_marker_n_is_passages_n_minus_1():
    prompt = build_prompt(QUESTION, [P2, P3, P1])
    for match in re.finditer(r"^\[(\d+)\] ", prompt.user, re.MULTILINE):
        marker = int(match.group(1))
        header_line = prompt.user[match.start():].splitlines()[0]
        assert prompt.passages[marker - 1].ticker in header_line


# --- refusals ----------------------------------------------------------------

def test_no_passages_is_refused():
    with pytest.raises(ValueError, match="no passages"):
        build_prompt(QUESTION, [])


def test_an_empty_generator_is_refused_too():
    # A generator is truthy however little it yields, so the guard has to
    # look at what arrived rather than at the container.
    with pytest.raises(ValueError, match="no passages"):
        build_prompt(QUESTION, (p for p in []))


def test_a_blank_question_is_refused():
    with pytest.raises(ValueError, match="blank"):
        build_prompt("  ", [P1])


def test_a_repeated_passage_is_refused():
    with pytest.raises(ValueError, match="repeat a chunk_id: p1"):
        build_prompt(QUESTION, [P1, _passage("p1", 5)])


# --- fits the records -----------------------------------------------------------

def test_the_prompt_passages_satisfy_answer_numbering():
    # The generator hands prompt.passages to the Answer as-is; a citation to
    # [2] must then resolve to whatever the prompt showed as source 2.
    prompt = build_prompt(QUESTION, [P3, P1, P2])
    answer = Answer(
        question=QUESTION, text="Revenue was $100 [2].",
        citations=(Citation(2, prompt.passages[1].chunk_id, True),),
        passages=prompt.passages, abstained=False,
        config=GenerationConfig(provider="ollama", model="m", prompt_template_id=prompt.template_id),
    )
    assert answer.cited_passages == (P2,)
    assert answer.config.prompt_template_id == PROMPT_TEMPLATE_ID
