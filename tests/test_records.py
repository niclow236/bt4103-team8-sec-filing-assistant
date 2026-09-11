"""The shared fingerprint, and the Query every retriever reads its filters from."""

from src.retrieval.records import Query, corpus_fingerprint, fingerprint_of, passage_digest

ROWS = [
    {"chunk_id": "a", "text": "one", "company": "A"},
    {"chunk_id": "b", "text": "two", "company": "B"},
]


def test_fingerprint_ignores_order():
    assert corpus_fingerprint(ROWS) == corpus_fingerprint(list(reversed(ROWS)))


def test_fingerprint_sees_text_ids_and_membership():
    base = corpus_fingerprint(ROWS)
    assert corpus_fingerprint([{**ROWS[0], "text": "ONE"}, ROWS[1]]) != base
    assert corpus_fingerprint([{**ROWS[0], "chunk_id": "z"}, ROWS[1]]) != base
    assert corpus_fingerprint(ROWS[:1]) != base


def test_text_of_decides_what_counts_as_content():
    header = lambda row: f"{row['company']}: {row['text']}"   # noqa: E731
    assert corpus_fingerprint(ROWS, text_of=header) != corpus_fingerprint(ROWS)
    renamed = [{**ROWS[0], "company": "Alpha"}, ROWS[1]]
    # Metadata that reaches the content moves the digest; metadata that does not, does not.
    assert corpus_fingerprint(renamed, text_of=header) != corpus_fingerprint(ROWS, text_of=header)
    assert corpus_fingerprint(renamed) == corpus_fingerprint(ROWS)


def test_fold_of_passage_digests_is_the_fingerprint():
    digests = [passage_digest(row["chunk_id"], row["text"]) for row in ROWS]
    assert fingerprint_of(digests) == corpus_fingerprint(ROWS)


def test_query_normalises_filter_values():
    query = Query("q", tickers=("aapl", "Msft"), fiscal_years=("2024", 2023), items=("1a", "7"))
    assert query.tickers == ("AAPL", "MSFT")
    assert query.fiscal_years == (2024, 2023)
    assert query.items == ("1A", "7")
    assert query.filters == {"ticker": ["AAPL", "MSFT"], "fiscal_year": [2024, 2023],
                             "item": ["1A", "7"]}


def test_query_without_filters_is_unrestricted():
    assert Query("q").filters == {}
