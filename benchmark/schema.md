# Benchmark question schema

`benchmark/questions.jsonl` contains one JSON object per line. Blank lines are
ignored. Every object must contain exactly these fields:

| Field | Type | Meaning |
| --- | --- | --- |
| `question_id` | non-empty string | Stable identifier for the question |
| `question` | non-empty string | User-style question sent to a retriever |
| `expected_answer` | non-empty string | Ground-truth answer used by evaluation |
| `supporting_chunk_ids` | non-empty array of strings | Chunks that contain evidence for the answer |
| `hard_negative_chunk_ids` | array of strings | Relevant-looking chunks that should not answer the question |
| `ticker` | non-empty string | Company ticker, for example `AAPL` |
| `fiscal_year` | integer | Fiscal year reported by the filing |
| `question_type` | non-empty string | For example `numeric`, `comparison`, or `narrative` |
| `difficulty` | non-empty string | Dataset difficulty label |
| `source` | non-empty string | For example `handwritten` or `xbrl` |

Supporting and hard-negative IDs must not overlap. Every referenced ID must be
present in the current `data/processed/` corpus. Question IDs must be unique.
The loader rejects missing fields, unknown fields, malformed JSON, duplicate
questions, empty benchmark files, and unresolved chunk IDs.