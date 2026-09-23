# Open-source Mistral models: latency test

`mistral_generation_test.ipynb` runs the questions in `../test_data/test_questions.csv` through the project's real RAG path. The model is an open-source one on Mistral's API instead of the local Ollama model. The notebook times every answer against the target of 10–30 seconds per question.

Question parsing, hybrid retrieval over `data/index/`, the prompt and the answer schema are the project's own code, imported from `src/` unchanged. Only the answer is generated on Mistral's servers. The full findings are in [`docs/mistral-free-tier-evaluation.md`](../../docs/mistral-free-tier-evaluation.md).

## Results so far

The test ran on 18 September 2026 on Mistral's free plan, with all 48 questions on each of four models. Every model answered all 48 within the 30-second target, so speed is no longer a problem. The models differ in how they handle figures.

| Rank | Model | Right figures (of 28, rounding allowed) | Wrong figures stated | Said the filings don't answer | Median time | Within 10 s |
|---|---|---|---|---|---|---|
| 1 | **Ministral 3 14B** (`ministral-14b-2512`) | 13 | 5 | 10 | 2.2 s | 48/48 |
| 2 | **Ministral 3 8B** (`ministral-8b-2512`) | 13 | 5 | 10 | 2.1 s | 47/48 |
| 3 | Voxtral Small 24B (`voxtral-small-2507`) | 11 | 13 | 4 | 1.4 s | 48/48 |
| 4 | Codestral 25.08 (`codestral-2508`) | 10 | 14 | 4 | 1.7 s | 48/48 |

- **The two Ministral models are the strongest.** In 12 of the 28 figure questions, retrieval gave the model the passage holding the figure. Both Ministral models stated all 12 correctly. When the figure wasn't there, they mostly said the filings don't answer, which is what the prompt asks for.
- **Ministral 3 14B ranks first.** The two are tied on these checks. 14B is the larger model and never took more than 10 s. 8B allows 188 requests a minute on the free plan against 30 for 14B, so it's the better choice if the app needs more throughput.
- **Voxtral Small and Codestral answer fast but invent figures** when the passages don't hold them. For example, Amazon's FY2025 net sales came out as "$514 billion" (really $716,924 million). Codestral is also tuned for code, and its licence restricts commercial use.
- **Retrieval, not the model, now limits accuracy.** The figure was in the retrieved passages for only 12 of 28 figure questions. Issues #85–#89 cover the fixes.

### 23 September 2026: `FINAL_K` at 8 against 16 (#85)

Both Ministral models were re-run over the same 48 questions at the old `FINAL_K` of 8 and at the 16 that #85 settled on. The two `20260923-*` files in `results/` are the record.

| | 14B @ 8 | 14B @ 16 | 8B @ 8 | 8B @ 16 |
|---|---|---|---|---|
| Figure in the retrieved passages | 18/28 | 22/28 | 18/28 | 22/28 |
| Figure stated, rounding allowed | 18/28 | 22/28 | 18/28 | 22/28 |
| Figure stated, exact digits | 13/28 | 17/28 | 16/28 | 20/28 |
| Abstained | 7 | 4 | 5 | 2 |
| Expected prose terms, mean | 90% | 92% | 93% | 93% |
| Prompt tokens, median | 3,139 | 5,751 | 3,139 | 5,751 |
| End to end, median | 2.0 s | 2.2 s | 2.2 s | 2.2 s |

The first two rows agree at both cutoffs: each model stated every figure it was given, so the deeper cut is the whole of the gain and a longer prompt did not distract either model. Prose did not regress, all 96 answers parsed, and the extra 2,600 prompt tokens cost about two tenths of a second. The figure in the passages rose from 12 of 28 in September to 18 at the same `FINAL_K` of 8, because #86, #87 and #88 landed in between.
- **These are automatic checks.** Hand grading in the `manual_correct` column is still to do.

## Running it

You need:
- **The packages:** the notebook's packages are in the project's `requirements.txt`. Run `pip install -r requirements.txt`.
- **The corpus and indexes:** `data/processed/` and `data/index/`, from the same build. Git ignores both. Either build them with the pipeline and the retrieval stage's `embed` and `bm25` commands (see the README), or copy both folders from a teammate.
- **Your own Mistral API key:** put it in the project's `.env` as `MISTRAL_API_KEY=...`, which git ignores.
  - On the free plan, turn off "Allow the use of your API calls to train Mistral's AI models" in the admin panel's privacy settings.
  - Never paste the key into a notebook cell.

Then:
1. **Choose the two models** in the settings cell. The free plan's models are listed there.
2. **Run all cells**, in VS Code with the project's `.venv` as the kernel, or from a terminal:
   ```bash
   jupyter nbconvert --to notebook --execute notebooks/mistral/mistral_generation_test.ipynb --output-dir <a folder outside the repo>
   ```
   48 questions on two models takes about 4–5 minutes. To try a few questions first, set `LIMIT = 5`.

**Running it with an AI agent** (Claude Code, Codex):
- **The key must already be in `.env`.** Nobody is there to answer the notebook's hidden key prompt.
- **Keep the agent out of `.env`.** In Claude Code, add `"permissions": {"deny": ["Read(**/.env)"]}` to your own `.claude/settings.local.json`. The notebook never prints the key.
- **Don't commit the executed notebook.** The run's CSV in `results/` is the record.

## Results files

Each run adds one file to `results/`, named for its time and models, such as `20260918-1239_ministral-14b-2512_ministral-8b-2512.csv`.

- **Each run file has one row per answer:** the question, the expected answer, the model's answer, its timings and the automatic checks. Since #85 each row also carries `final_k`, the retrieval cutoff the run used, and `input_tokens`, the prompt's length as the provider counted it. The September files predate both columns, and the summary leaves them blank rather than guessing.
- **`manual_correct` starts empty.** Mark each answer's correctness there by hand.
- **`comparison.csv` is rebuilt after every run.** It puts each model's latest run side by side, and it's where the table above comes from.

## The test questions

`../test_data/test_questions.csv` is a copy of the `Questions` sheet of the team's question bank, `BT4103_Team8_Evaluation_Questions.xlsx`. It holds 48 questions:

- 40 original questions and 8 paraphrases (`Paraphrase` is `TRUE`), each rewording an original with the same answer.
- 8 companies across FY2021–FY2025, on Items 1, 1A, 7 and 8.

The workbook stays the master copy. When it changes, refresh the copy from the project root:

```bash
python -c "import pandas as pd; pd.read_excel('../BT4103_Team8_Evaluation_Questions.xlsx', sheet_name='Questions').to_csv('notebooks/test_data/test_questions.csv', index=False, encoding='utf-8-sig')"
```
