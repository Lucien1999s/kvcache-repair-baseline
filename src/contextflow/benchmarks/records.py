from __future__ import annotations

import json
from typing import Any

from contextflow.benchmarks.constants import (
    DEFAULT_METHODS,
    METHOD_CACHEBLEND_REPAIR,
    METHOD_FULL_RECOMPUTE,
    METHOD_NAIVE_REUSE,
    SUPPORTED_METHODS,
)
from contextflow.benchmarks.resources import aggregate_resource_metrics
from contextflow.evaluation import (
    aggregate_qa_metrics,
    compute_normalized_score,
)


def parse_methods(raw_methods: str) -> list[str]:
    methods = [method.strip() for method in raw_methods.split(",") if method.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(
            f"Unsupported methods: {invalid}. Supported methods: {sorted(SUPPORTED_METHODS)}."
        )
    return methods


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def build_dataset_example_record(
    example_index: int,
    example: Any,
    dataset_key: str,
    model_name: str,
    model_family: str,
    max_new_tokens: int,
    prompt_policy: str,
    prediction_parser: str,
    repair_planner: str,
) -> dict[str, Any]:
    return {
        "example_index": example_index,
        "example_id": example.example_id,
        "dataset": dataset_key,
        "question": example.question,
        "answers": example.answers,
        "num_ctxs": len(example.ctxs),
        "model": model_name,
        "model_family": model_family,
        "max_new_tokens": max_new_tokens,
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "repair_planner": repair_planner,
        "methods": {},
    }


def summarize_dataset_results(
    dataset_key: str,
    model_name: str,
    model_family: str,
    prompt_policy: str,
    prediction_parser: str,
    repair_planner: str,
    metrics_by_method: dict[str, list[dict[str, float]]],
    resources_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    method_summaries = {
        method: aggregate_qa_metrics(metrics)
        for method, metrics in metrics_by_method.items()
    }
    for method, method_summary in method_summaries.items():
        method_summary.update(aggregate_resource_metrics(resources_by_method.get(method, [])))

    normalized_f1 = None
    if all(method in method_summaries for method in DEFAULT_METHODS):
        naive_f1 = method_summaries[METHOD_NAIVE_REUSE]["mean_f1"]
        full_f1 = method_summaries[METHOD_FULL_RECOMPUTE]["mean_f1"]
        repair_f1 = method_summaries[METHOD_CACHEBLEND_REPAIR]["mean_f1"]
        if naive_f1 is not None and full_f1 is not None and repair_f1 is not None:
            normalized_f1 = compute_normalized_score(
                method_score=float(repair_f1),
                lower_score=float(naive_f1),
                upper_score=float(full_f1),
            )

    return {
        "dataset": dataset_key,
        "model": model_name,
        "model_family": model_family,
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "repair_planner": repair_planner,
        "count": max((summary["count"] for summary in method_summaries.values()), default=0),
        "methods": method_summaries,
        "cacheblend_normalized_f1": normalized_f1,
    }


def write_jsonl_record(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False))
    file.write("\n")
    file.flush()
