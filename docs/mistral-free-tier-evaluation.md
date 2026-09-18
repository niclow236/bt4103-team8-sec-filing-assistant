# Open-source Mistral models on the free plan: latency and first accuracy check

**Date:** 18 September 2026
**Branch:** `testing-mistral`, started from `feature/rag-generate` at `d1548ff`
**Notebook:** [`notebooks/mistral/mistral_generation_test.ipynb`](../notebooks/mistral/mistral_generation_test.ipynb)
**Results:** [`notebooks/mistral/results/`](../notebooks/mistral/results/)

## Summary

- **Speed is no longer a problem.** All four models answered all 48 test questions within the 30-second target, and 191 of the 192 answers came back within 10 seconds. That time covers the whole question, from parsing through retrieval to the full answer. The median was 1.4–2.2 s per question. On a team laptop, `llama3.2:3b` on Ollama took 2.3–3.4 minutes.
- **Ministral 3 14B is the model to use, with Ministral 3 8B as the fallback.**
  - Twelve of the 28 figure questions had the expected figure among the retrieved passages. Both Ministral models stated all 12 correctly.
  - When the passages didn't hold the figure, the Ministral models mostly said the filings don't answer the question, as the prompt asks.
  - Voxtral Small and Codestral answered more often, but by inventing figures. For example, they gave Amazon's FY2025 net sales as "$514 billion" (really $716,924 million) and Salesforce's goodwill as "$337 million" (really $48,568 million).
- **Retrieval now limits accuracy, not the model.** The expected figure was among the 8 retrieved passages for only 12 of the 28 figure questions. All four models fail on the same questions.
- **Mistral's free plan serves only four suitable chat models.** Its stronger open-weight models, Mistral Small 4, Mistral Medium 3.5 and Mistral Large 3, are not usable on the free plan.

## Why this test

The RAG stage generates answers with a local model on Ollama. On the team's laptops (8–16 GB RAM, CPU only), one answer takes minutes. The planned chatbot is a hosted web app, so it can't depend on anyone's laptop.

The team has no budget, so the question was whether a free, open-source model served in the cloud can answer inside 10–30 seconds per question. Mistral's free plan includes $10 of API usage a month, and it serves open-weight models. This test runs the project's real pipeline against those models.

## What was tested

### The pipeline

Everything except the model is the project's own code, imported unchanged from `src/`:

1. `parse_question` reads the question.
2. The hybrid retriever (BM25 and dense, fused by reciprocal rank, top 8, no reranker yet) finds passages.
3. `build_prompt` renders the `grounded_v4` prompt.
4. The answer is constrained to the `GroundedAnswer` JSON schema, which holds every cited source number to 1–8.
5. The answer is validated and rendered as prose with `[n]` markers.

The only replacement is the model call inside `generate.stream`. That call passes Ollama-only settings, so the notebook repeats the same loop with a call to Mistral's API instead.

All four models accepted the schema in strict mode and streamed their answers.

### The models

Mistral's API was checked with a free account's key on 18 September 2026. These are the chat models it served:

| Model | API id | Licence | Free-plan limit |
|---|---|---|---|
| Ministral 3 14B | `ministral-14b-2512` | Apache 2.0 | 30 requests/min, 937,500 tokens/min |
| Ministral 3 8B | `ministral-8b-2512` | Apache 2.0 | 188 requests/min, 625,000 tokens/min |
| Voxtral Small 24B | `voxtral-small-2507` | Apache 2.0 | 60 requests/min, 50,000 tokens/min |
| Codestral 25.08 | `codestral-2508` | Open weights; licence restricts commercial use; tuned for code | 125 requests/min, 625,000 tokens/min |
| Mistral Small 4 | `mistral-small-2603` | Apache 2.0 | **0 requests/min**: listed but unusable |
| Mistral Medium 3.5 | `mistral-medium-2604` | Modified MIT | **0 requests/min**: listed but unusable |
| Mistral Large 3 | `mistral-large-2512` | Apache 2.0 | **Refused**: "not available in your subscription tier" |

The first four were tested; Ministral 3 3B was also served but left out as the weakest. Voxtral Small is an audio-and-text model built on Mistral Small 3.1, and matches it on text. Some names in the model list point to Mistral Small 4 for Mistral's own tools, such as `mistral-vibe-cli-fast`. Those weren't used, since that would sidestep the free plan's limit.

### The questions

There are 48 questions, in [`notebooks/test_data/test_questions.csv`](../notebooks/test_data/test_questions.csv), a copy of the team's question bank:

- **Originals and paraphrases:** 40 originals and 8 paraphrases. A paraphrase rewords an original and keeps the same answer.
- **Companies and years:** eight companies (AAPL, MSFT, AVGO, GOOGL, META, AMZN, ORCL and CRM) across FY2021–FY2025.
- **Sections:** Items 1, 1A, 7 and 8 of the 10-K.
- **Answer types:** 28 questions expect a dollar figure or a percentage, and 20 expect a list or a description.

### What was measured

- **End to end:** parsing, retrieval and the whole answer. This is what a chatbot user waits for.
- **First words:** when the answer starts to appear on screen.
- **Output speed:** output tokens per second once writing has started.
- **Left out of the times:**
  - The one-off start-up. Loading the indexes took 9–12 s, and the first search, which loads the embedding model, took 13–15 s (41 s on a cold start). A hosted app pays these once at boot, not per question.
  - Waits caused by the free plan's rate limits, which the notebook reports as they happen.
- **Where it ran:** retrieval ran on a Windows laptop with 8 CPUs, and the models on Mistral's servers. Both models in a run answered each question back to back, over the same passages.

## Results

### Speed

| | Ministral 3 14B | Ministral 3 8B | Voxtral Small 24B | Codestral 25.08 |
|---|---|---|---|---|
| Within 30 s end to end | 48/48 | 48/48 | 48/48 | 48/48 |
| Within 10 s end to end | 48/48 | 47/48 | 48/48 | 48/48 |
| End to end, median | 2.2 s | 2.1 s | 1.4 s | 1.7 s |
| End to end, 90th percentile | 4.8 s | 3.4 s | 2.7 s | 3.8 s |
| End to end, slowest | 6.0 s | 10.8 s | 3.4 s | 5.5 s |
| First words on screen, median | 1.3 s | 1.2 s | 1.2 s | 1.3 s |
| Output speed, median | 83 tokens/s | 119 tokens/s | 176 tokens/s | 147 tokens/s |
| Answer length, median | 111 tokens | 107 tokens | 54 tokens | 54 tokens |

- **Retrieval** took 0.4 s per question (median). The rest is the model.
- **Every answer streamed.** Every answer was valid JSON under the strict schema, and none was cut off.
- **The one answer over 10 s was just long.** Ministral 3 8B on Q14 wrote 391 tokens.
- **Answer length explains the speed differences.** The Ministral models write about twice as much, often hedging or giving context, so they take a little longer.

### Answers

These checks are automatic. Final grading is by hand (see [Next steps](#next-steps)).

**The 28 figure questions:**

| | Ministral 3 14B | Ministral 3 8B | Voxtral Small 24B | Codestral 25.08 |
|---|---|---|---|---|
| Right figure, rounding allowed | **13** | **13** | 11 | 10 |
| Right, of the 12 where retrieval found the figure | **12/12** | **12/12** | 10/12 | 9/12 |
| Said the filings don't answer | 10 | 10 | 4 | 4 |
| Stated a wrong figure | **5** | **5** | 13 | 14 |
| Right figure, exact digits | 9 | 11 | 9 | 9 |

- **"Rounding allowed"** counts "$49.6 billion" as the same figure as "$49,584 million" (within 0.1%).
- **Most of the Ministral models' wrong figures were hedged.** They added that "the exact figure is not provided in the sources". Voxtral and Codestral stated wrong figures plainly, for example:
  - $101,700 million for Apple's accounts payable (really $64,115 million).
  - $3.0 billion for Microsoft's net income (really $72,361 million).
  - 18% for Microsoft's effective tax rate (really 17.6%).
  - 46% for Google's US revenue share, a figure from another year (really 48%).
- **Two recurring mistakes cut across models:**
  - For AWS's operating income, all four gave Amazon's total operating income ($68.6 billion) instead of AWS's ($39,834 million).
  - For Amazon's FY2025 net sales, several gave the growth in net sales instead of the total.
- **All models round despite being told not to.** Rule 4 of the prompt says to quote figures exactly as the source gives them. Ministral 3 14B also puts figures in markdown bold.

**The other questions:**

| | Ministral 3 14B | Ministral 3 8B | Voxtral Small 24B | Codestral 25.08 |
|---|---|---|---|---|
| List answers: expected items mentioned (mean of 20) | 88% | 88% | 86% | 87% |
| Paraphrase pairs with the same result | 5/8 | 6/8 | 5/8 | 6/8 |

The list answers were close across models. The low scores all come from Q13, Q14 and Q23. Retrieval missed the section for Q14, and Q23's expected answer is a long passage that a short answer can't match word for word.

### Retrieval

- **Right section:** the retrieved passages included the right company, year and 10-K Item for 44 of 48 questions. The misses were Q14 (Broadcom tax risks, reworded), Q38 (Oracle FY2022), and Q41 and Q42 (Oracle FY2025 net income).
- **Right figure:** the expected figure was among the retrieved passages for only 12 of the 28 figure questions. Most of the missing ones are balance-sheet and income-statement totals from Item 8, such as total assets, liabilities, net income, goodwill and net sales. The right section is retrieved, but not the table row that holds the number.

### Rate limits and cost

- **Rate limits:** only Voxtral Small hit one. Its 50,000-tokens-a-minute limit caused 20 s of waiting over the run, which is not counted in its times. At about 3,300 input tokens per question, 50,000 tokens a minute allows roughly 15 questions a minute.
- **Cost:** by mid-run, the Ministral run had used about $0.02 of the $10 monthly allowance. At these sizes the allowance covers tens of thousands of answers a month. With no payment method on the account, running out stops requests rather than charging.

## Conclusions

1. **A free, open-source model on Mistral's API meets the latency target by a wide margin.** Generation is no longer the bottleneck: first words appear after about 1.2 s, and whole answers arrive in about 2 s.
2. **Ministral 3 14B is the model to use, with Ministral 3 8B as the fallback.**
   - Both are Apache 2.0.
   - Both stated every figure that retrieval surfaced correctly, and mostly abstained when it didn't. In a finance assistant, saying "the filings do not answer this" is far better than a confident wrong number.
   - 8B has six times the request limit (188 a minute) and gives nearly identical results.
3. **Don't use Voxtral Small or Codestral for this task.** They are slightly faster, but they invent figures when the passages don't hold them. Codestral's licence also restricts commercial use.
4. **Retrieval is what to work on next.** Until the passages carry the figure, no model can state it.

## Next steps

1. **Improve retrieval of financial-statement figures.** Options, with this notebook re-run after each change to measure it:
   - Reranking.
   - The table boost (`TABLE_BOOST`, still 1.0 pending #24).
   - Answering figure questions from the XBRL facts table that `src/retrieval/facts.py` builds.
2. **Grade by hand.** Fill in the `manual_correct` column of the two run files in [`notebooks/mistral/results/`](../notebooks/mistral/results/) to confirm the automatic checks.
3. **Tighten how figures are written.** Either strengthen rule 4 or check the figures after generation. The app also needs to render or strip the markdown bold.
4. **Add Mistral as a provider in `src/rag/generate.py`,** after `feature/rag-generate` merges:
   - Use `ChatMistralAI` behind the existing chat-model interface, with the answer schema sent as a strict `json_schema` response format.
   - Configure the provider, the model and `MISTRAL_API_KEY` in `.env`.
   - Keep Ollama for offline use.
   - The hosted app should use one agreed account's key. Teammates keep their own keys for development.
5. **Re-check the free plan before the demo.** Model availability and limits can change. The notebook's connection check reports any model the plan no longer serves.

## Limitations

- **The answer checks are proxies.** The figure check matches numbers, and the list check matches words. Neither judges whether an answer is right in context; that needs the hand grading in step 2 above.
- **Each model was run once**, from one network, at one time of day. The free plan's latency may vary with load.
- **Retrieval was timed on a laptop.** The app's server may be faster or slower.
- **The question set is narrow.** Every question is about one company and one year, and all 48 are answerable. There are no comparison, change-over-time or unanswerable questions yet.
- **The free plan was checked on 18 September 2026.** Its models and limits may have changed since.

## Reproducing

See [`notebooks/mistral/MISTRAL_TEST_OVERVIEW.md`](../notebooks/mistral/MISTRAL_TEST_OVERVIEW.md) for setup, including the data the notebook needs, adding your own API key to `.env`, and turning off training on your data. Each run sets its two models in the notebook's settings cell.

| File | Contents |
|---|---|
| [`20260918-1239_ministral-14b-2512_ministral-8b-2512.csv`](../notebooks/mistral/results/20260918-1239_ministral-14b-2512_ministral-8b-2512.csv) | First run: every answer, with its timings and checks |
| [`20260918-2000_voxtral-small-2507_codestral-2508.csv`](../notebooks/mistral/results/20260918-2000_voxtral-small-2507_codestral-2508.csv) | Second run, same columns |
| [`comparison.csv`](../notebooks/mistral/results/comparison.csv) | The four models side by side: the tables in this report |
