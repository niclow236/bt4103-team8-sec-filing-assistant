"""Benchmark records, loaders, and ranking metrics for retrieval evaluation."""

from .benchmark import (
    DEFAULT_GENERATED_QUESTIONS_PATH,
    DEFAULT_QUESTIONS_PATH,
    generate_xbrl_questions,
    load_questions,
)
from .metrics import hard_negative_accuracy, mrr, ndcg_at_k, recall_at_k, score_question
from .records import BenchmarkQuestion, RunResult
from .harness import evaluate

__all__ = [
    "BenchmarkQuestion",
    "DEFAULT_GENERATED_QUESTIONS_PATH",
    "DEFAULT_QUESTIONS_PATH",
    "RunResult",
    "generate_xbrl_questions",
    "hard_negative_accuracy",
    "load_questions",
    "evaluate",
    "mrr",
    "ndcg_at_k",
    "recall_at_k",
    "score_question",
]
