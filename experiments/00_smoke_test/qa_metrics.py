from __future__ import annotations

from contextflow.evaluation import (
    aggregate_qa_metrics,
    best_exact_match,
    best_f1_score,
    compute_normalized_score,
    evaluate_qa_prediction,
    exact_match_score,
    f1_score,
    normalize_answer,
)


def assert_close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    assert abs(actual - expected) <= tolerance, f"Expected {expected}, got {actual}."


def main() -> None:
    assert normalize_answer("The, QUICK! Brown fox.") == "quick brown fox"
    assert normalize_answer("  A   Tale\t of   Two Cities  ") == "tale of two cities"

    assert exact_match_score("Jane Austen", "jane austen") == 1.0
    assert exact_match_score("The Jane Austen!", "Jane Austen") == 1.0
    assert exact_match_score("Jane Austen", "Charlotte Bronte") == 0.0

    assert_close(f1_score("Jane Austen novelist", "Jane Austen"), 0.8)
    assert f1_score("Charles Dickens", "Jane Austen") == 0.0
    assert f1_score("", "") == 1.0
    assert f1_score("", "Jane Austen") == 0.0

    gold_answers = ["Charlotte Bronte", "Jane Austen"]
    assert best_exact_match("jane austen", gold_answers) == 1.0
    assert best_f1_score("Austen", gold_answers) == 2 / 3

    metrics = [
        evaluate_qa_prediction("Jane Austen", ["Jane Austen"]),
        evaluate_qa_prediction("Austen", ["Jane Austen"]),
        evaluate_qa_prediction("Charles Dickens", ["Jane Austen"]),
    ]
    aggregate = aggregate_qa_metrics(metrics)
    assert aggregate["count"] == 3
    assert_close(float(aggregate["mean_exact_match"]), 1 / 3)
    assert_close(float(aggregate["mean_f1"]), (1.0 + (2 / 3) + 0.0) / 3)

    assert compute_normalized_score(method_score=0.75, lower_score=0.5, upper_score=1.0) == 0.5
    assert compute_normalized_score(method_score=0.8, lower_score=0.7, upper_score=0.7) is None

    print("qa_metrics smoke test passed")
    print(f"aggregate: {aggregate}")
    print(
        "normalized_score: "
        f"{compute_normalized_score(method_score=0.75, lower_score=0.5, upper_score=1.0)}"
    )


if __name__ == "__main__":
    main()
