from __future__ import annotations

from collections import defaultdict
from typing import Any

from contextflow.benchmarks.constants import STATUS_ERROR, STATUS_OOM, STATUS_SUCCESS
from contextflow.benchmarks.resources import mean_optional_float
from contextflow.evaluation import aggregate_qa_metrics


def method_metric_record(method_record: dict[str, Any]) -> dict[str, float]:
    return {
        "exact_match": float(method_record["exact_match"]),
        "f1": float(method_record["f1"]),
    }


def summarize_success_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    metric_records = [
        method_metric_record(record)
        for record in records
        if record.get("status") == STATUS_SUCCESS
    ]
    summary = aggregate_qa_metrics(metric_records)
    summary.update(
        {
            "mean_latency_seconds": mean_optional_float(records, "latency_seconds"),
            "mean_total_latency_seconds": mean_optional_float(records, "total_latency_seconds"),
            "mean_peak_gpu_memory_mb": mean_optional_float(records, "peak_gpu_memory_mb"),
            "mean_planning_latency_seconds": mean_optional_float(
                records,
                "planning_latency_seconds",
            ),
            "mean_reuse_precompute_latency_seconds": mean_optional_float(
                records,
                "reuse_precompute_latency_seconds",
            ),
            "mean_enriched_precompute_latency_seconds": mean_optional_float(
                records,
                "enriched_precompute_latency_seconds",
            ),
            "mean_selection_latency_seconds": mean_optional_float(
                records,
                "selection_latency_seconds",
            ),
            "mean_repair_latency_seconds": mean_optional_float(
                records,
                "repair_latency_seconds",
            ),
            "mean_decode_latency_seconds": mean_optional_float(
                records,
                "decode_latency_seconds",
            ),
            "mean_execution_latency_seconds": mean_optional_float(
                records,
                "execution_latency_seconds",
            ),
        }
    )
    return summary


def summarize_oom_boundary(records: list[dict[str, Any]]) -> dict[str, Any]:
    success_records = [
        record for record in records if record.get("status") == STATUS_SUCCESS
    ]
    oom_records = [record for record in records if record.get("status") == STATUS_OOM]

    return {
        "max_success_chunk_count": (
            max(record["chunk_count"] for record in success_records)
            if success_records
            else None
        ),
        "max_success_doc_token_count": (
            max(record["doc_token_count"] for record in success_records)
            if success_records
            else None
        ),
        "min_oom_chunk_count": (
            min(record["chunk_count"] for record in oom_records)
            if oom_records
            else None
        ),
        "min_oom_doc_token_count": (
            min(record["doc_token_count"] for record in oom_records)
            if oom_records
            else None
        ),
    }


def summarize_by_chunk_count(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["chunk_count"])].append(record)

    summary: dict[str, Any] = {}
    for chunk_count in sorted(grouped):
        chunk_records = grouped[chunk_count]
        success_records = [
            record for record in chunk_records if record.get("status") == STATUS_SUCCESS
        ]
        metric_summary = summarize_success_records(success_records)
        summary[str(chunk_count)] = {
            "case_count": len(chunk_records),
            "success_count": len(success_records),
            "oom_count": sum(
                1 for record in chunk_records if record.get("status") == STATUS_OOM
            ),
            "error_count": sum(
                1 for record in chunk_records if record.get("status") == STATUS_ERROR
            ),
            "mean_doc_token_count": mean_optional_float(chunk_records, "doc_token_count"),
            "mean_total_prefill_token_count": mean_optional_float(
                chunk_records,
                "total_prefill_token_count",
            ),
            **metric_summary,
        }
    return summary


def summarize_sweep_results(
    dataset_key: str,
    model_name: str,
    model_family: str,
    prompt_policy: str,
    prediction_parser: str,
    repair_planner: str,
    chunk_count_specs: list[int | str],
    method_outcomes: dict[str, list[dict[str, Any]]],
    chunking_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    method_summaries: dict[str, Any] = {}
    for method, records in method_outcomes.items():
        success_records = [
            record for record in records if record.get("status") == STATUS_SUCCESS
        ]
        method_summary = summarize_success_records(success_records)
        method_summary.update(
            {
                "case_count": len(records),
                "success_count": len(success_records),
                "oom_count": sum(
                    1 for record in records if record.get("status") == STATUS_OOM
                ),
                "error_count": sum(
                    1 for record in records if record.get("status") == STATUS_ERROR
                ),
                **summarize_oom_boundary(records),
                "by_chunk_count": summarize_by_chunk_count(records),
            }
        )
        method_summaries[method] = method_summary

    return {
        "record_type": "chunk_sweep_summary",
        "dataset": dataset_key,
        "model": model_name,
        "model_family": model_family,
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "repair_planner": repair_planner,
        **dict(chunking_config or {}),
        "chunk_count_specs": chunk_count_specs,
        "methods": method_summaries,
    }
