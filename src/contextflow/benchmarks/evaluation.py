from __future__ import annotations

from typing import Any

from contextflow.benchmarks.constants import (
    PREDICTION_PARSER_CACHEBLEND_QA,
    PREDICTION_PARSER_NONE,
    STATUS_SUCCESS,
    SUPPORTED_PREDICTION_PARSERS,
)
from contextflow.evaluation import evaluate_qa_prediction, parse_cacheblend_generation


def parse_prediction_for_evaluation(raw_generated_text: str, prediction_parser: str) -> str:
    if prediction_parser == PREDICTION_PARSER_NONE:
        return raw_generated_text
    if prediction_parser == PREDICTION_PARSER_CACHEBLEND_QA:
        return parse_cacheblend_generation(raw_generated_text)
    raise ValueError(
        f"Unsupported prediction_parser={prediction_parser!r}. Supported parsers: "
        f"{sorted(SUPPORTED_PREDICTION_PARSERS)}."
    )


def evaluate_method_generation(
    generated_ids: list[int],
    generated_text: str,
    answers: list[str],
    prediction_parser: str,
    *,
    include_status: bool = False,
) -> dict[str, Any]:
    evaluated_prediction = parse_prediction_for_evaluation(
        generated_text,
        prediction_parser=prediction_parser,
    )
    metrics = evaluate_qa_prediction(evaluated_prediction, answers)
    record: dict[str, Any] = {
        "generated_ids": generated_ids,
        "raw_generated_text": generated_text,
        "generated_text": generated_text,
        "evaluated_prediction": evaluated_prediction,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
    }
    if include_status:
        record["status"] = STATUS_SUCCESS
    return record
