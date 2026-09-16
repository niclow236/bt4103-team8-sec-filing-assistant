# Benchmark question schema

`benchmark/questions.jsonl` contains one JSON object per line. Blank lines are
ignored. Every object must contain exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `question_id` | non-empty string | Stable identifier for the question |
| `question` | non-empty string | User-style question sent to a retriever |
| `expected_answer` | non-empty string | Ground-truth answer used by evaluation |
| `supporting_chunk_ids` | array of strings | Chunks that contain evidence for the answer. Empty for an `unanswerable` question, and only then |
| `hard_negative_chunk_ids` | array of strings | Relevant-looking chunks that should not answer the question |
| `ticker` | non-empty string or `null` | Company ticker, for example `AAPL`. `null` when the question does not name exactly one company in the corpus, as in a comparison |
| `fiscal_year` | integer or `null` | Fiscal year reported by the filing. `null` when the question does not name exactly one year, as in a change across years |
| `question_type` | one of `factual`, `comparative`, `temporal`, `numeric`, `unanswerable` | The labels `src/rag/query.py` assigns, defined in `src/rag/constants.py`, so results can be broken down by the type the engine read a question as |
| `difficulty` | non-empty string | Dataset difficulty label |
| `source` | non-empty string | For example `handwritten` or `xbrl` |

An `unanswerable` question is one the corpus cannot answer: it asks about a
company or year outside the corpus, or for something a 10-K does not contain.
It has no supporting chunks. Chunks that look relevant to it belong in
`hard_negative_chunk_ids`, which is how the benchmark checks that the system
abstains rather than answering from them.

Supporting and hard-negative IDs must not overlap. Every referenced ID must be
present in the current `data/processed/` corpus. Question IDs must be unique.
The loader rejects missing fields, unknown fields, malformed JSON, duplicate
questions, empty benchmark files, unresolved chunk IDs, and a `question_type`
outside the five above.
