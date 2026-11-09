"""LLM-only (closed-book) baseline on the 150 team questions.

The same answer model as the complete C0-C6 matrix answers each team question
with no retrieved passages, from what it knows. The same judge, rubric and
abstention rules as the matrix then score it, so its mean correctness sits
beside C0 (0.3145) and C4 (0.8710) from full-matrix-20261021.

Run from the repository root, with SOCLAAS_API_KEY and MISTRAL_API_KEY in .env:
    python tools/llm_only_baseline.py

Defaults match the matrix: SOCLAAS qwen3.6:35b answers, Mistral
ministral-8b-2512 judges with answer-quality-v1, temperature 0, 1,024 output
tokens. Rows are appended one question at a time under
results/llm-only-<date>/, so a stopped run picks up where it left off when
the same command is run again.

Scoring follows src.evaluation.harness:
- the model declines -> correctness 1 on an unanswerable question, 0 otherwise;
- the model answers an unanswerable question -> correctness 0;
- otherwise the judge's correctness.
Faithfulness and citation scores do not apply: a closed-book answer cites
nothing. Correctness is a model judgement, as in the matrix.

--adjudications names a JSON list of {"question_id", "correctness", "reason"}
for judgements a person has overruled. The summary then also gives the mean
with those applied, beside the judged mean. The judged mean stays the one to
put beside the matrix, whose rows were not adjudicated.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.records import BenchmarkQuestion, TEAM, UNANSWERABLE  # noqa: E402
from src.rag.constants import MAX_OUTPUT_TOKENS, MISTRAL, SOCLAAS  # noqa: E402

DECLINE = "I cannot answer this."
SYSTEM = f"""You are a financial analyst answering questions about US public
companies' annual reports (Form 10-K filed with the SEC). You have no
documents: answer from your own knowledge only. Be concise. Give exact figures
with units and the fiscal year where the question asks for a number. If you do
not know the answer, or the question cannot be answered from a 10-K filing,
reply with exactly: {DECLINE}"""

MATRIX_REFERENCE = {  # full-matrix-20261021, team benchmark, judged correctness
    "C0 naive BM25 + LLM": 0.3145,
    "C4 hybrid + filters + LLM (app default)": 0.8710,
}


def read_jsonl(path: Path) -> list[dict]:
    # As written, UTF-8: Windows' default, cp1252, cannot decode a curly quote,
    # so a run there could neither finish nor resume.
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def load_team(path: Path) -> list[BenchmarkQuestion]:
    rows = read_jsonl(path)
    questions = [BenchmarkQuestion.from_mapping(row) for row in rows]
    return [q for q in questions if q.benchmark == TEAM]


def plain_model(config, model):
    """The chat model with the recorded settings and no JSON schema attached."""
    from pydantic import BaseModel

    from src.rag.generate import _provider

    class _Unused(BaseModel):
        text: str

    options = _provider(config).request(config, model, _Unused, MAX_OUTPUT_TOKENS)
    for key in ("format", "response_format", "extra_body"):
        options.pop(key, None)
    return model.model_copy(update=options.get("options", options))


def answer_text(raw) -> str:
    content = getattr(raw, "content", raw)
    if isinstance(content, list):
        content = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                          for part in content)
    text = re.sub(r"<think>.*?</think>", "", str(content), flags=re.S)
    return text.strip()


def declined(text: str) -> bool:
    return text.strip().rstrip(".").lower().startswith(DECLINE.rstrip(".").lower())


def with_retries(call, label: str, attempts: int = 6):
    from src.rag.generate import ProviderBusy

    for attempt in range(1, attempts + 1):
        try:
            return call()
        except ProviderBusy as error:
            wait = error.retry_after or min(60, 5 * attempt)
            print(f"  {label}: provider busy ({error.reason}); retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
        except Exception as error:  # rate limits surface differently per client
            if attempt == attempts or "429" not in str(error):
                raise
            wait = min(60, 5 * attempt)
            print(f"  {label}: rate limited; retrying in {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"{label}: gave up after {attempts} attempts")


def score(question: BenchmarkQuestion, text: str, judge, config) -> dict:
    from src.rag.records import Answer

    answerable = question.question_type != UNANSWERABLE
    if declined(text):
        return {"abstained": True, "method": "abstention",
                "correctness": float(not answerable),
                "correctness_reason": "Benchmark answerability"}
    answer = Answer(question=question.question, text=text, citations=(), passages=(),
                    abstained=False, config=config)
    result = with_retries(lambda: judge(question, answer), f"{question.question_id} judge")
    row = {"abstained": False, "method": "llm_judge",
           "correctness": result["correctness"],
           "correctness_reason": result["correctness_reason"],
           "judge_input_tokens": result.get("input_tokens"),
           "judge_output_tokens": result.get("output_tokens")}
    if not answerable:
        row.update(correctness=0.0, correctness_reason="Answered an unanswerable question")
    return row


def read_adjudications(path: Path, rows: list[dict]) -> dict[str, dict]:
    """A person's overrides of the judge, keyed by question ID."""
    entries = json.loads(path.read_text(encoding="utf-8"))
    known = {r["question_id"] for r in rows}
    out = {}
    for entry in entries:
        qid = entry["question_id"]
        if qid not in known:
            raise SystemExit(f"{path} adjudicates {qid}, which this run did not answer")
        correctness = float(entry["correctness"])
        if not 0.0 <= correctness <= 1.0 or not str(entry.get("reason", "")).strip():
            raise SystemExit(f"{path}: {qid} needs a correctness from 0 to 1 and a reason")
        out[qid] = {"correctness": correctness, "reason": entry["reason"]}
    return out


def summarize(rows: list[dict], adjudications: dict[str, dict] | None = None) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        by_type[row["question_type"]].append(row["correctness"])
    summary = {
        "questions": len(rows),
        "mean_correctness": mean(r["correctness"] for r in rows) if rows else None,
        "abstained": sum(r["abstained"] for r in rows),
        "by_question_type": {k: {"n": len(v), "mean_correctness": mean(v)}
                             for k, v in sorted(by_type.items())},
        "matrix_reference": MATRIX_REFERENCE,
    }
    if adjudications:
        adjusted = [adjudications[r["question_id"]]["correctness"]
                    if r["question_id"] in adjudications else r["correctness"] for r in rows]
        summary["adjudicated"] = {
            "mean_correctness": mean(adjusted),
            "changes": [{"question_id": r["question_id"], "judged": r["correctness"]}
                        | adjudications[r["question_id"]]
                        for r in rows if r["question_id"] in adjudications],
        }
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark" / "questions.jsonl")
    parser.add_argument("--run-id", default=f"llm-only-{date.today():%Y%m%d}")
    parser.add_argument("--provider", default=SOCLAAS)
    parser.add_argument("--model", default="qwen3.6:35b")
    parser.add_argument("--judge-provider", default=MISTRAL)
    parser.add_argument("--judge-model", default="ministral-8b-2512")
    parser.add_argument("--limit", type=int, help="Answer only the first N questions (smoke test)")
    parser.add_argument("--adjudications", type=Path,
                        help="JSON list of judgements a person has overruled")
    args = parser.parse_args(argv)

    from src.evaluation.quality import LLMJudge
    from src.rag.generate import check_provider, checked_model, config_from_env

    questions = load_team(args.questions)
    if args.limit:
        questions = questions[: args.limit]
    generation = config_from_env(provider=args.provider, model=args.model)
    judging = config_from_env(provider=args.judge_provider, model=args.judge_model)
    check_provider(generation)  # refuse a missing key before any call
    check_provider(judging)

    model = plain_model(generation, checked_model(generation))
    judge = LLMJudge(judging)

    run_dir = ROOT / "results" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # The judge's own record: LLMJudge swaps in the rubric as its prompt template.
    config = {"generation": generation.to_dict(), "judge": judge.to_dict(),
              "system_prompt": SYSTEM, "max_output_tokens": MAX_OUTPUT_TOKENS,
              "questions_file": str(args.questions)}
    config_path = run_dir / "config.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise SystemExit(f"{config_path} records different settings; use a new --run-id")
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    path = run_dir / "answers.jsonl"
    done = {r["question_id"] for r in read_jsonl(path)} if path.exists() else set()

    with path.open("a", encoding="utf-8") as out:
        for n, q in enumerate(questions, 1):
            if q.question_id in done:
                continue
            started = time.perf_counter()
            raw = with_retries(lambda: model.invoke([("system", SYSTEM), ("human", q.question)]),
                               f"{q.question_id} answer")
            latency_ms = (time.perf_counter() - started) * 1000
            text = answer_text(raw)
            row = {"question_id": q.question_id, "question_type": q.question_type,
                   "question": q.question, "gold_answer": q.expected_answer,
                   "answer": text, "generation_latency_ms": round(latency_ms)}
            row |= score(q, text, judge, generation)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            out.flush()
            print(f"{n}/{len(questions)} {q.question_id}: correctness {row['correctness']:.2f}",
                  flush=True)

    rows = read_jsonl(path)
    adjudications = read_adjudications(args.adjudications, rows) if args.adjudications else None
    summary = summarize(rows, adjudications)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nLLM-only mean correctness: {summary['mean_correctness']:.4f} "
          f"over {summary['questions']} questions ({summary['abstained']} declined)")
    if adjudications:
        print(f"With {len(adjudications)} judgements overruled: "
              f"{summary['adjudicated']['mean_correctness']:.4f}")
    for name, value in MATRIX_REFERENCE.items():
        print(f"{name}: {value:.4f}")
    for kind, s in summary["by_question_type"].items():
        print(f"  {kind:<12} n={s['n']:<3} {s['mean_correctness']:.3f}")
    print(f"\nSaved to {run_dir}")


if __name__ == "__main__":
    main()
