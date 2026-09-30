"""Evaluate one retrieval/generation configuration and report abstention rates."""

import argparse
import json
import sys
from math import isfinite
from pathlib import Path

from src.config import PROCESSED_DIR
from src.rag import ProviderUnavailable, chat_model, config_from_env
from src.rag.constants import LLM_MODEL_ENV, LLM_PROVIDER_ENV, PROVIDERS
from src.retrieval.constants import FINAL_K
from .benchmark import DEFAULT_QUESTIONS_PATH, load_questions
from .harness import RunInterrupted, RunStopped, evaluate


def _for_this_command(message: str) -> str:
    """A provider error as this command prints it.

    Its advice names the .env settings that the app and notebooks read, and
    here --provider and --model win over those when given, so setting .env
    alone would change nothing. The message says so wherever it names one.
    """
    if LLM_PROVIDER_ENV in message or LLM_MODEL_ENV in message:
        return (f"{message} (on this command, --provider and --model do the same as "
                f"{LLM_PROVIDER_ENV} and {LLM_MODEL_ENV}, and win over .env when given)")
    return message


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("questions", type=Path, nargs="?", default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument("--processed-dir", type=Path, default=PROCESSED_DIR)
    parser.add_argument("--retriever", choices=("bm25", "dense", "hybrid"), default="hybrid")
    parser.add_argument("--min-score", type=float, help="Additional floor on final retrieval scores")
    parser.add_argument("--top-k", type=int, default=FINAL_K)
    parser.add_argument(
        "--provider", choices=PROVIDERS,
        help="Which provider answers: ollama, the local model, or mistral, the hosted API "
             "with your own MISTRAL_API_KEY. Defaults to LLM_PROVIDER in .env, else ollama.",
    )
    parser.add_argument(
        "--model",
        help="The model, as the provider names it. Defaults to LLM_MODEL in .env when "
             "--provider is LLM_PROVIDER's, else the provider's default.",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path, required=True, help="JSON report including rates and answers")
    parser.add_argument("--answers", type=Path, help="Optional answer JSONL for the browser viewer")
    parser.add_argument(
        "--no-facts", dest="use_facts", action="store_false",
        help="Send every question to retrieval and generation, instead of looking a "
             "numeric one up in the XBRL facts store first. The without half of the "
             "#34 ablation, and the honest setting for the generated XBRL benchmark, "
             "whose questions come from that same store.",
    )
    parser.add_argument(
        "--no-decompose", dest="use_decomposition", action="store_false",
        help="Search each question once, instead of once per filing for a question "
             "naming several companies or years. The without half of the #35 ablation.",
    )
    args = parser.parse_args(argv)
    if args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.min_score is not None and not isfinite(args.min_score):
        parser.error("--min-score must be finite")
    if not args.run_id.strip():
        parser.error("--run-id must be non-empty")
    if args.answers and args.answers.resolve() == args.output.resolve():
        parser.error("--output and --answers must name different files")
    # Built before the questions and indexes load, so a mistyped LLM_PROVIDER
    # or a missing MISTRAL_API_KEY stops the run at once.
    try:
        config = config_from_env(model=args.model, provider=args.provider)
        llm = chat_model(config)
    except (ValueError, ProviderUnavailable) as error:
        parser.error(_for_this_command(str(error)))
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
    try:
        report = evaluate(questions, retriever, config, llm=llm, run_id=args.run_id,
                          min_score=args.min_score, top_k=args.top_k,
                          use_facts=args.use_facts, use_decomposition=args.use_decomposition)
        stopped = None
    except (RunStopped, RunInterrupted) as error:
        # The answers before the question the run stopped at, whether a
        # provider failure or Ctrl-C stopped it, are written as for a complete
        # run, and the command then fails.
        report, stopped = error.report, error
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if args.answers:
        args.answers.parent.mkdir(parents=True, exist_ok=True)
        args.answers.write_text("".join(json.dumps(row) + "\n" for row in report["results"]),
                                encoding="utf-8")
    print(json.dumps({"summary": report["summary"],
                      "by_answerability": report["by_answerability"]}, indent=2))
    if stopped is not None:
        # The note goes right after the advice it is about, not after the path.
        answered = len(report["results"])
        sys.exit(f"{_for_this_command(str(stopped))}; {answered} answered before it, "
                 f"written to {args.output}")
