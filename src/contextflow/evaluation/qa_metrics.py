from __future__ import annotations

import re
import string
from collections import Counter
from typing import Any


ARTICLES_PATTERN = re.compile(r"\b(a|an|the)\b", flags=re.IGNORECASE)
PUNCTUATION_TRANSLATION = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Normalize a QA answer using common extractive-QA conventions.

    The normalization lowercases text, removes punctuation, removes English
    articles, and collapses whitespace.
    """

    lowered = text.lower()
    without_punctuation = lowered.translate(PUNCTUATION_TRANSLATION)
    without_articles = ARTICLES_PATTERN.sub(" ", without_punctuation)
    return " ".join(without_articles.split())


def exact_match_score(prediction: str, gold: str) -> float:
    """Return 1.0 when normalized prediction exactly matches normalized gold."""

    return float(normalize_answer(prediction) == normalize_answer(gold))


def best_exact_match(prediction: str, gold_answers: list[str]) -> float:
    """Return the best exact-match score over multiple gold answers."""

    if not gold_answers:
        return 0.0
    return max(exact_match_score(prediction, gold) for gold in gold_answers)


def tokenize_normalized_answer(text: str) -> list[str]:
    normalized = normalize_answer(text)
    if not normalized:
        return []
    return normalized.split()


def f1_score(prediction: str, gold: str) -> float:
    """Return token-level F1 between normalized prediction and normalized gold."""

    prediction_tokens = tokenize_normalized_answer(prediction)
    gold_tokens = tokenize_normalized_answer(gold)
    if not prediction_tokens and not gold_tokens:
        return 1.0
    if not prediction_tokens or not gold_tokens:
        return 0.0

    common = Counter(prediction_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def best_f1_score(prediction: str, gold_answers: list[str]) -> float:
    """Return the best token-level F1 score over multiple gold answers."""

    if not gold_answers:
        return 0.0
    return max(f1_score(prediction, gold) for gold in gold_answers)


def evaluate_qa_prediction(prediction: str, gold_answers: list[str]) -> dict[str, float]:
    """Evaluate one predicted answer against one or more gold answers."""

    return {
        "exact_match": best_exact_match(prediction, gold_answers),
        "f1": best_f1_score(prediction, gold_answers),
    }


def aggregate_qa_metrics(metrics: list[dict[str, Any]]) -> dict[str, float | int | None]:
    """Aggregate per-example QA metric dictionaries.

    Empty input returns count 0 and None means.
    """

    count = len(metrics)
    if count == 0:
        return {
            "count": 0,
            "mean_exact_match": None,
            "mean_f1": None,
        }

    exact_match_total = sum(float(metric.get("exact_match", 0.0)) for metric in metrics)
    f1_total = sum(float(metric.get("f1", 0.0)) for metric in metrics)
    return {
        "count": count,
        "mean_exact_match": exact_match_total / count,
        "mean_f1": f1_total / count,
    }


def compute_normalized_score(
    method_score: float,
    lower_score: float,
    upper_score: float,
) -> float | None:
    """Normalize a method score between a lower and upper reference.

    The intended use is generic quality recovery:
    (method_score - lower_score) / (upper_score - lower_score).

    Returns None when upper_score == lower_score because the reference interval
    has zero width and quality recovery is undefined.
    """

    denominator = upper_score - lower_score
    if denominator == 0:
        return None
    return (method_score - lower_score) / denominator
