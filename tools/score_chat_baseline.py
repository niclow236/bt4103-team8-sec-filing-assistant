"""Score a general-purpose chat assistant's answers on the 20-question subset.

The chat session (GPT-5.6 in ChatGPT) answers the prompt in the benchmark
design doc and returns one JSON Lines block. Save it as a file, then run from
the repository root, with MISTRAL_API_KEY in .env. The 8 November session's
answers are committed, and were compared with the archived matrix run:

    python tools/score_chat_baseline.py docs/evaluation/chat-baseline-20261108-answers.jsonl \
        --matrix-dir results/full-matrix-20261017

--matrix-dir needs a run that saved each configuration's answers.jsonl.

Scoring is the C0-C6 study's: the same Mistral ministral-8b-2512 judge and
answer-quality-v1 rubric, and the harness's refusal rules (a refusal scores 1 on
an unanswerable question and 0 otherwise; answering an unanswerable question
scores 0). --matrix-dir, the complete matrix run, puts each configuration's
mean correctness on the same 20 questions beside the chat assistant's.

It also writes source-check.csv, one row per answer and its sources, with
blank columns for the person who opens each source and checks it. A second
run's file (--run2) is scored the same way, and the two are compared question
by question.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from llm_only_baseline import DECLINE, declined, with_retries  # noqa: E402
from src.evaluation.records import BenchmarkQuestion, UNANSWERABLE  # noqa: E402
from src.rag.constants import MISTRAL  # noqa: E402

SUBSET = {  # prompt ID -> benchmark question_id, as in the design doc
    "Q01": "daryl-28", "Q02": "anqi-14", "Q03": "callista-07", "Q04": "tingyi-06",
    "Q05": "daryl-06", "Q06": "callista-12", "Q07": "callista-23", "Q08": "anqi-05",
    "Q09": "nicholas-06", "Q10": "callista-05", "Q11": "tingyi-18", "Q12": "daryl-17",
    "Q13": "nicholas-03", "Q14": "callista-22", "Q15": "jarren-12", "Q16": "jarren-16",
    "Q17": "jarren-17", "Q18": "jarren-18", "Q19": "jarren-23", "Q20": "callista-25",
}


def read_answers(path: Path) -> dict[str, dict]:
    """The chat's JSON Lines block, tolerating code fences and blank lines."""
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("```"):
            continue
        row = json.loads(line)
        rows[row["id"].strip().upper()] = row
    missing = sorted(set(SUBSET) - set(rows))
    if missing:
        raise SystemExit(f"{path} has no answer for {', '.join(missing)}")
    return rows


def score_run(path: Path, questions: dict[str, BenchmarkQuestion], judge, config) -> list[dict]:
    from src.rag.records import Answer

    answers = read_answers(path)
    scored = []
    for pid, qid in SUBSET.items():
        q, row = questions[qid], answers[pid]
        text = str(row.get("answer", "")).strip()
        answerable = q.question_type != UNANSWERABLE
        refused = bool(row.get("declined")) or declined(text)
        if refused:
            result = {"method": "abstention", "correctness": float(not answerable),
                      "correctness_reason": "Benchmark answerability"}
        else:
            answer = Answer(question=q.question, text=text, citations=(), passages=(),
                            abstained=False, config=config)
            judged = with_retries(lambda: judge(q, answer), f"{pid} judge")
            result = {"method": "llm_judge", "correctness": judged["correctness"],
                      "correctness_reason": judged["correctness_reason"]}
            if not answerable:
                result.update(correctness=0.0, correctness_reason="Answered an unanswerable question")
        sources = row.get("sources") or []
        scored.append({"id": pid, "question_id": qid, "question_type": q.question_type,
                       "question": q.question, "gold_answer": q.expected_answer,
                       "answer": text, "declined": refused, "sources": sources} | result)
        print(f"{pid} {qid:<12} {q.question_type:<12} correctness {result['correctness']:.2f}",
              flush=True)
    return scored


def summary(rows: list[dict]) -> dict:
    by_type: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_type[r["question_type"]].append(r["correctness"])
    answered = [r for r in rows if not r["declined"]]
    sec = [r for r in answered
           if any("sec.gov" in str(s.get("url", "")) for s in r["sources"])]
    return {
        "questions": len(rows),
        "mean_correctness": mean(r["correctness"] for r in rows),
        "declined": len(rows) - len(answered),
        "answered_with_a_source": sum(bool(r["sources"]) for r in answered),
        "answered_citing_sec_gov": len(sec),
        "by_question_type": {k: {"n": len(v), "mean_correctness": mean(v)}
                             for k, v in sorted(by_type.items())},
    }


def matrix_means(matrix_dir: Path) -> dict[str, dict]:
    """Each configuration's mean correctness on the same 20 questions."""
    wanted = set(SUBSET.values())
    out = {}
    for config_dir in sorted(p for p in matrix_dir.iterdir() if (p / "answers.jsonl").exists()):
        rows = [json.loads(l) for l in (config_dir / "answers.jsonl").read_text().splitlines()
                if l.strip()]
        values = {r["question_id"]: r["quality"]["correctness"] for r in rows
                  if r["question_id"] in wanted}
        scored = [v for v in values.values() if v is not None]
        out[config_dir.name] = {"mean_correctness": mean(scored) if scored else None,
                                "scored": len(scored), "of": len(wanted)}
    return out


def write_source_check(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "question", "gold_answer", "chat_answer", "judged_correctness",
                    "source_urls", "source_sections",
                    "url_opens (y/n)", "right_company_and_year_10K (y/n)",
                    "source_states_the_answer (y/n)", "verifiable (all three y)", "notes"])
        for r in rows:
            w.writerow([r["id"], r["question"], r["gold_answer"], r["answer"],
                        f"{r['correctness']:.2f}",
                        " | ".join(str(s.get("url", "")) for s in r["sources"]),
                        " | ".join(str(s.get("section", "")) for s in r["sources"]),
                        "", "", "", "", ""])


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("answers", type=Path, help="The chat's JSON Lines answers")
    parser.add_argument("--run2", type=Path, help="A second session's answers, for consistency")
    parser.add_argument("--matrix-dir", type=Path,
                        help="Complete matrix run, e.g. results/full-matrix-20261021")
    parser.add_argument("--questions", type=Path, default=ROOT / "benchmark" / "questions.jsonl")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "chat-baseline")
    parser.add_argument("--judge-provider", default=MISTRAL)
    parser.add_argument("--judge-model", default="ministral-8b-2512")
    args = parser.parse_args(argv)

    from src.evaluation.quality import LLMJudge
    from src.rag.generate import check_provider, config_from_env

    lines = [l for l in args.questions.read_text(encoding="utf-8").splitlines() if l.strip()]
    questions = {q.question_id: q for q in
                 (BenchmarkQuestion.from_mapping(json.loads(l)) for l in lines)}
    read_answers(args.answers)  # refuse an incomplete file before any judge call
    if args.run2:
        read_answers(args.run2)
    judging = config_from_env(provider=args.judge_provider, model=args.judge_model)
    check_provider(judging)
    judge = LLMJudge(judging)

    args.out.mkdir(parents=True, exist_ok=True)
    result = {"judge": judging.to_dict(), "subset": SUBSET}
    rows = score_run(args.answers, questions, judge, judging)
    (args.out / "scored.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_source_check(args.out / "source-check.csv", rows)
    result["chat"] = summary(rows)

    if args.run2:
        rows2 = score_run(args.run2, questions, judge, judging)
        (args.out / "scored-run2.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows2), encoding="utf-8")
        result["chat_run2"] = summary(rows2)
        changed = [a["id"] for a, b in zip(rows, rows2)
                   if abs(a["correctness"] - b["correctness"]) >= 0.5 or a["declined"] != b["declined"]]
        result["consistency"] = {"questions_whose_result_changed": changed,
                                 "changed": len(changed), "of": len(rows)}

    if args.matrix_dir:
        result["matrix_same_20"] = matrix_means(args.matrix_dir)

    (args.out / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    chat = result["chat"]
    print(f"\nGPT-5.6 chat: mean correctness {chat['mean_correctness']:.3f} on 20 questions "
          f"({chat['declined']} declined; {chat['answered_citing_sec_gov']} of "
          f"{20 - chat['declined']} answers cite sec.gov)")
    for kind, s in chat["by_question_type"].items():
        print(f"  {kind:<12} n={s['n']:<2} {s['mean_correctness']:.3f}")
    if args.run2:
        c = result["consistency"]
        print(f"Second session: mean {result['chat_run2']['mean_correctness']:.3f}; "
              f"{c['changed']} of 20 questions changed result")
    for name, m in result.get("matrix_same_20", {}).items():
        if m["mean_correctness"] is not None:
            print(f"{name}: mean correctness {m['mean_correctness']:.3f} "
                  f"({m['scored']}/{m['of']} scored)")
    print(f"\nSaved to {args.out}; fill in {args.out / 'source-check.csv'} by hand.")


if __name__ == "__main__":
    main()
