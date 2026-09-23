"""Evaluate one retrieval/generation configuration and report abstention rates."""

import argparse
import json
from math import isfinite
from pathlib import Path

from src.config import PROCESSED_DIR
from src.rag import config_from_env
from src.retrieval.constants import FINAL_K
from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import evaluate


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--min-score", type=float, help="Additional floor on final retrieval scores")
    parser.add_argument("--top-k", type=int, default=FINAL_K)
    parser.add_argument("--model")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report including rates and answers")
    parser.add_argument("--answers", type=Path, help="Optional answer JSONL for the browser viewer")
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.min_score is not None and not isfinite(args.min_score):
        parser.error("--min-score must be finite")
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if args.answers and args.answers.resolve() == args.output.resolve():
        parser.error("--output and --answers must name different files")
    questions = load_questions(args.questions, processed_dir=args.processed_dir)
    from src.retrieval.bm25 import BM25Retriever
    from src.retrieval.dense import DenseRetriever
    from src.retrieval.hybrid import HybridRetriever

    if args.retriever == "bm25":
        retriever = BM25Retriever.load(processed_dir=args.processed_dir)
    elif args.retriever == "dense":
        retriever = DenseRetriever.load(processed_dir=args.processed_dir)
    else:
        retriever = HybridRetriever(BM25Retriever.load(processed_dir=args.processed_dir),
                                    DenseRetriever.load(processed_dir=args.processed_dir))
    report = evaluate(questions, retriever, config_from_env(model=args.model),
                      run_id=args.run_id, min_score=args.min_score, top_k=args.top_k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.answers:
        args.answers.parent.mkdir(parents=True, exist_ok=True)
        args.answers.write_text("".join(json.dumps(row) + "\n" for row in report["results"]),
                                encoding="utf-8")
    print(json.dumps({"summary": report["summary"],
                      "by_answerability": report["by_answerability"]}, indent=2))
