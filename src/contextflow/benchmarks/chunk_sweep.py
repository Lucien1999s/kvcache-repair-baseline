from __future__ import annotations

from typing import Any

from contextflow.benchmarks.chunk_cases import (
    parse_chunk_count_specs,
    resolve_example_chunk_cases,
    slice_example_contexts,
    token_count_record,
)
from contextflow.benchmarks.constants import (
    METHOD_CACHEBLEND_REPAIR,
    METHOD_FUSIONRAG_REPAIR,
    METHOD_FULL_RECOMPUTE,
    METHOD_NAIVE_REUSE,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
    REPAIR_PLANNER_ORACLE_HKVD,
    SUPPORTED_REPAIR_PLANNERS,
)
from contextflow.benchmarks.errors import (
    build_case_failure_record,
    handle_phase_error,
    run_phase_capture,
)
from contextflow.benchmarks.evaluation import evaluate_method_generation
from contextflow.benchmarks.method_runner import (
    build_cacheblend_repair_record,
    build_fusionrag_repair_record,
    build_online_cacheblend_repair_record,
    decode_with_past_key_values,
    precompute_reuse_doc_kv,
)
from contextflow.benchmarks.resources import (
    max_phase_peak_memory_mb,
    phase_latency_seconds,
    total_phase_latency_seconds,
)
from contextflow.benchmarks.sweep_summary import summarize_sweep_results
from contextflow.data import InputExample
from contextflow.kv_cache import rope_position_correction_enabled
from contextflow.methods.cacheblend_repair import (
    prepare_cacheblend_repair_plan_with_artifacts,
    run_cacheblend_style_online_partial_repair,
    run_cacheblend_style_partial_repair_from_plan,
)
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.methods.fusionrag_repair import run_fusionrag_style_repair_generation


def run_full_recompute_method(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    prediction_parser: str,
    enable_profiling: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    phase_metrics: dict[str, dict[str, Any]] = {}
    result, generation_profile, error = run_phase_capture(
        "generation",
        lambda: run_token_aligned_full_recompute_greedy_generation(
            tokenized_example,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["generation"] = generation_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="generation",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    method_record = evaluate_method_generation(
        generated_ids=result.generation.generated_ids,
        generated_text=result.generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
        include_status=True,
    )
    method_record.update(
        {
            "phase_metrics": phase_metrics,
            "latency_seconds": total_phase_latency_seconds(phase_metrics),
            "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
            "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
        }
    )
    return method_record


def run_naive_reuse_method(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    model_family: str,
    prediction_parser: str,
    enable_profiling: bool,
    continue_on_error: bool,
) -> dict[str, Any]:
    phase_metrics: dict[str, dict[str, Any]] = {}
    reuse_doc_kv, reuse_profile, error = run_phase_capture(
        "reuse_precompute",
        lambda: precompute_reuse_doc_kv(
            model=model,
            tokenized_example=tokenized_example,
            model_family=model_family,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["reuse_precompute"] = reuse_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="reuse_precompute",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    generation, decode_profile, error = run_phase_capture(
        "decode",
        lambda: decode_with_past_key_values(
            model=model,
            tokenizer=tokenizer,
            tokenized_example=tokenized_example,
            past_key_values=reuse_doc_kv,
            max_new_tokens=max_new_tokens,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["decode"] = decode_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="decode",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    method_record = evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
        include_status=True,
    )
    method_record.update(
        {
            "phase_metrics": phase_metrics,
            "reuse_precompute_latency_seconds": phase_latency_seconds(reuse_profile),
            "decode_latency_seconds": phase_latency_seconds(decode_profile),
            "latency_seconds": total_phase_latency_seconds(phase_metrics),
            "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
            "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
            "assembled_kv_layers": len(reuse_doc_kv),
            "rope_position_correction_applied": rope_position_correction_enabled(
                model_family
            ),
        }
    )
    return method_record


def run_cacheblend_repair_method(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    initial_top_k: int,
    top_k: int,
    model_family: str,
    prediction_parser: str,
    enable_profiling: bool,
    continue_on_error: bool,
    repair_planner: str,
) -> dict[str, Any]:
    phase_metrics: dict[str, dict[str, Any]] = {}
    if repair_planner == REPAIR_PLANNER_ONLINE_GRADUAL_HKVD:
        reuse_doc_kv, reuse_profile, error = run_phase_capture(
            "reuse_precompute",
            lambda: precompute_reuse_doc_kv(
                model=model,
                tokenized_example=tokenized_example,
                model_family=model_family,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics["reuse_precompute"] = reuse_profile
        if error is not None:
            return handle_phase_error(
                error,
                failed_phase="reuse_precompute",
                phase_metrics=phase_metrics,
                continue_on_error=continue_on_error,
            )

        partial_repair, online_repair_profile, error = run_phase_capture(
            "online_repair",
            lambda: run_cacheblend_style_online_partial_repair(
                model=model,
                tokenized_example=tokenized_example,
                initial_top_k=initial_top_k,
                top_k=top_k,
                model_family=model_family,
                reuse_past_key_values=reuse_doc_kv,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics["online_repair"] = online_repair_profile
        if error is not None:
            return handle_phase_error(
                error,
                failed_phase="online_repair",
                phase_metrics=phase_metrics,
                continue_on_error=continue_on_error,
            )

        generation, decode_profile, error = run_phase_capture(
            "decode",
            lambda: decode_with_past_key_values(
                model=model,
                tokenizer=tokenizer,
                tokenized_example=tokenized_example,
                past_key_values=partial_repair.repaired_past_key_values,
                max_new_tokens=max_new_tokens,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics["decode"] = decode_profile
        if error is not None:
            return handle_phase_error(
                error,
                failed_phase="decode",
                phase_metrics=phase_metrics,
                continue_on_error=continue_on_error,
            )

        return build_online_cacheblend_repair_record(
            generation=generation,
            answers=answers,
            prediction_parser=prediction_parser,
            partial_repair_metadata=partial_repair.metadata,
            phase_metrics=phase_metrics,
            include_status=True,
        )

    if repair_planner != REPAIR_PLANNER_ORACLE_HKVD:
        raise ValueError(
            f"Unsupported repair_planner={repair_planner!r}. "
            f"Supported planners: {sorted(SUPPORTED_REPAIR_PLANNERS)}."
        )

    planning_artifacts, plan_profile, error = run_phase_capture(
        "plan",
        lambda: prepare_cacheblend_repair_plan_with_artifacts(
            model=model,
            tokenized_example=tokenized_example,
            initial_top_k=initial_top_k,
            top_k=top_k,
            model_family=model_family,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["plan"] = plan_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="plan",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    repair_plan = planning_artifacts.plan
    repair_reuse_kv = planning_artifacts.reuse_doc_kv
    del planning_artifacts

    partial_repair, repair_profile, error = run_phase_capture(
        "repair",
        lambda: run_cacheblend_style_partial_repair_from_plan(
            model=model,
            tokenized_example=tokenized_example,
            repair_plan=repair_plan,
            model_family=model_family,
            reuse_past_key_values=repair_reuse_kv,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["repair"] = repair_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="repair",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    generation, decode_profile, error = run_phase_capture(
        "decode",
        lambda: decode_with_past_key_values(
            model=model,
            tokenizer=tokenizer,
            tokenized_example=tokenized_example,
            past_key_values=partial_repair.repaired_past_key_values,
            max_new_tokens=max_new_tokens,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["decode"] = decode_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="decode",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    return build_cacheblend_repair_record(
        generation=generation,
        answers=answers,
        prediction_parser=prediction_parser,
        repair_plan=repair_plan,
        partial_repair_metadata=partial_repair.metadata,
        phase_metrics=phase_metrics,
        include_status=True,
    )


def run_fusionrag_repair_method(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    model_family: str,
    prediction_parser: str,
    enable_profiling: bool,
    continue_on_error: bool,
    fusionrag_neighbor_top_n: int,
    fusionrag_recompute_ratio: float,
) -> dict[str, Any]:
    phase_metrics: dict[str, dict[str, Any]] = {}
    result, fusionrag_profile, error = run_phase_capture(
        "fusionrag_repair_generation",
        lambda: run_fusionrag_style_repair_generation(
            model=model,
            tokenizer=tokenizer,
            tokenized_example=tokenized_example,
            max_new_tokens=max_new_tokens,
            neighbor_top_n=fusionrag_neighbor_top_n,
            recompute_ratio=fusionrag_recompute_ratio,
            model_family=model_family,
        ),
        enable_profiling=enable_profiling,
    )
    phase_metrics["fusionrag_repair_generation"] = fusionrag_profile
    if error is not None:
        return handle_phase_error(
            error,
            failed_phase="fusionrag_repair_generation",
            phase_metrics=phase_metrics,
            continue_on_error=continue_on_error,
        )

    return build_fusionrag_repair_record(
        result=result,
        answers=answers,
        prediction_parser=prediction_parser,
        phase_metrics=phase_metrics,
        include_status=True,
    )


def run_methods_for_sweep_case(
    methods: list[str],
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    initial_top_k: int,
    top_k: int,
    model_family: str,
    prediction_parser: str,
    enable_profiling: bool,
    continue_on_error: bool,
    repair_planner: str,
    fusionrag_neighbor_top_n: int = 5,
    fusionrag_recompute_ratio: float = 0.15,
) -> dict[str, dict[str, Any]]:
    method_records: dict[str, dict[str, Any]] = {}
    if METHOD_FULL_RECOMPUTE in methods:
        method_records[METHOD_FULL_RECOMPUTE] = run_full_recompute_method(
            tokenized_example=tokenized_example,
            model=model,
            tokenizer=tokenizer,
            answers=answers,
            max_new_tokens=max_new_tokens,
            prediction_parser=prediction_parser,
            enable_profiling=enable_profiling,
            continue_on_error=continue_on_error,
        )
    if METHOD_NAIVE_REUSE in methods:
        method_records[METHOD_NAIVE_REUSE] = run_naive_reuse_method(
            tokenized_example=tokenized_example,
            model=model,
            tokenizer=tokenizer,
            answers=answers,
            max_new_tokens=max_new_tokens,
            model_family=model_family,
            prediction_parser=prediction_parser,
            enable_profiling=enable_profiling,
            continue_on_error=continue_on_error,
        )
    if METHOD_CACHEBLEND_REPAIR in methods:
        method_records[METHOD_CACHEBLEND_REPAIR] = run_cacheblend_repair_method(
            tokenized_example=tokenized_example,
            model=model,
            tokenizer=tokenizer,
            answers=answers,
            max_new_tokens=max_new_tokens,
            initial_top_k=initial_top_k,
            top_k=top_k,
            model_family=model_family,
            prediction_parser=prediction_parser,
            enable_profiling=enable_profiling,
            continue_on_error=continue_on_error,
            repair_planner=repair_planner,
        )
    if METHOD_FUSIONRAG_REPAIR in methods:
        method_records[METHOD_FUSIONRAG_REPAIR] = run_fusionrag_repair_method(
            tokenized_example=tokenized_example,
            model=model,
            tokenizer=tokenizer,
            answers=answers,
            max_new_tokens=max_new_tokens,
            model_family=model_family,
            prediction_parser=prediction_parser,
            enable_profiling=enable_profiling,
            continue_on_error=continue_on_error,
            fusionrag_neighbor_top_n=fusionrag_neighbor_top_n,
            fusionrag_recompute_ratio=fusionrag_recompute_ratio,
        )
    return method_records


def build_sweep_case_record(
    example_index: int,
    example: InputExample,
    dataset_key: str,
    model_name: str,
    model_family: str,
    max_new_tokens: int,
    prompt_policy: str,
    prediction_parser: str,
    repair_planner: str,
    chunk_case: dict[str, Any],
    token_counts: dict[str, int],
) -> dict[str, Any]:
    return {
        "record_type": "chunk_sweep_case",
        "example_index": example_index,
        "example_id": example.example_id,
        "dataset": dataset_key,
        "question": example.question,
        "answers": example.answers,
        "available_chunk_count": chunk_case["available_chunk_count"],
        "requested_chunk_count": chunk_case["requested_chunk_count"],
        "chunk_count": chunk_case["chunk_count"],
        "model": model_name,
        "model_family": model_family,
        "max_new_tokens": max_new_tokens,
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "repair_planner": repair_planner,
        **token_counts,
        "methods": {},
    }


def add_sweep_context_to_method_records(
    method_records: dict[str, dict[str, Any]],
    *,
    example_index: int,
    example_id: str,
    chunk_case: dict[str, Any],
    token_counts: dict[str, int],
) -> None:
    for method_record in method_records.values():
        method_record.update(
            {
                "example_index": example_index,
                "example_id": example_id,
                "requested_chunk_count": chunk_case["requested_chunk_count"],
                "chunk_count": chunk_case["chunk_count"],
                "available_chunk_count": chunk_case["available_chunk_count"],
                **token_counts,
            }
        )


__all__ = [
    "add_sweep_context_to_method_records",
    "build_case_failure_record",
    "build_sweep_case_record",
    "parse_chunk_count_specs",
    "resolve_example_chunk_cases",
    "run_methods_for_sweep_case",
    "slice_example_contexts",
    "summarize_sweep_results",
    "token_count_record",
]
