from __future__ import annotations

import argparse
import gc
import json
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from contextflow.data import (
    PROMPT_POLICY_CACHEBLEND_QA,
    SUPPORTED_CACHEBLEND_PROMPT_POLICIES,
    InputExample,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
    tokenize_prompt_example,
)
from contextflow.evaluation import (
    aggregate_qa_metrics,
    evaluate_qa_prediction,
    parse_cacheblend_generation,
)
from contextflow.kv_cache import (
    assemble_chunk_kvs,
    correct_doc_chunk_kvs_for_model_family,
    precompute_doc_chunk_kvs,
    rope_position_correction_enabled,
)
from contextflow.methods.cacheblend_repair import (
    CacheBlendRepairPlan,
    prepare_cacheblend_repair_plan_with_artifacts,
    run_cacheblend_style_online_partial_repair,
    run_cacheblend_style_partial_repair_from_plan,
)
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.profiling import (
    get_peak_memory_mb_if_available,
    reset_peak_memory_stats_if_available,
    synchronize_cuda_if_available,
)
from contextflow.runtime import load_hf_causal_lm
from contextflow.runtime.hf_cached_generation import generate_with_past_key_values


METHOD_FULL_RECOMPUTE = "full_recompute"
METHOD_NAIVE_REUSE = "naive_reuse"
METHOD_CACHEBLEND_REPAIR = "cacheblend_repair"
DEFAULT_METHODS = [
    METHOD_FULL_RECOMPUTE,
    METHOD_NAIVE_REUSE,
    METHOD_CACHEBLEND_REPAIR,
]
SUPPORTED_METHODS = set(DEFAULT_METHODS)

PREDICTION_PARSER_NONE = "none"
PREDICTION_PARSER_CACHEBLEND_QA = "cacheblend_qa"
SUPPORTED_PREDICTION_PARSERS = {
    PREDICTION_PARSER_NONE,
    PREDICTION_PARSER_CACHEBLEND_QA,
}

STATUS_SUCCESS = "success"
STATUS_OOM = "oom"
STATUS_ERROR = "error"
REPAIR_PLANNER_ORACLE_HKVD = "oracle_hkvd"
REPAIR_PLANNER_ONLINE_GRADUAL_HKVD = "online_gradual_hkvd"
SUPPORTED_REPAIR_PLANNERS = {
    REPAIR_PLANNER_ORACLE_HKVD,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk-count sweep runner for the CacheBlend-style HF/PyTorch reference baseline."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--model-family", required=True, choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to sweep.")
    parser.add_argument(
        "--chunk-counts",
        default="1,2,4,8,16,32,all",
        help="Comma-separated context counts to sweep. Use 'all' for all contexts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--initial-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--repair-planner",
        choices=sorted(SUPPORTED_REPAIR_PLANNERS),
        default=REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
        help=(
            "Token-selection planner for repair. online_gradual_hkvd is the measured "
            "baseline; oracle_hkvd is full-reference diagnostic mode."
        ),
    )
    parser.add_argument(
        "--prompt-policy",
        choices=sorted(SUPPORTED_CACHEBLEND_PROMPT_POLICIES),
        default=PROMPT_POLICY_CACHEBLEND_QA,
    )
    parser.add_argument(
        "--prediction-parser",
        choices=sorted(SUPPORTED_PREDICTION_PARSERS),
        default=PREDICTION_PARSER_CACHEBLEND_QA,
    )
    parser.add_argument("--output-jsonl", required=True, help="Path for sweep JSONL records.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype passed to model load.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to model load.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated subset of full_recompute,naive_reuse,cacheblend_repair.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Record non-OOM method errors and continue. OOM is always recorded and continued.",
    )
    parser.add_argument(
        "--enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record synchronized latency and peak GPU memory by phase when available.",
    )
    return parser.parse_args()


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


def parse_chunk_count_specs(raw_counts: str) -> list[int | str]:
    specs: list[int | str] = []
    for raw_part in raw_counts.split(","):
        part = raw_part.strip().lower()
        if not part:
            continue
        if part == "all":
            specs.append("all")
            continue
        count = int(part)
        if count <= 0:
            raise ValueError("--chunk-counts values must be positive integers or 'all'.")
        specs.append(count)
    if not specs:
        raise ValueError("--chunk-counts must contain at least one value.")
    return specs


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def resolve_example_chunk_cases(
    example: InputExample,
    chunk_count_specs: list[int | str],
) -> list[dict[str, Any]]:
    available_count = len(example.ctxs)
    if available_count <= 0:
        raise ValueError(f"Example {example.example_id!r} has no contexts to sweep.")

    cases: list[dict[str, Any]] = []
    seen_counts: set[int] = set()
    for spec in chunk_count_specs:
        chunk_count = available_count if spec == "all" else int(spec)
        if chunk_count > available_count:
            continue
        if chunk_count in seen_counts:
            continue
        seen_counts.add(chunk_count)
        cases.append(
            {
                "requested_chunk_count": spec,
                "chunk_count": chunk_count,
                "available_chunk_count": available_count,
            }
        )

    if not cases:
        cases.append(
            {
                "requested_chunk_count": "all",
                "chunk_count": available_count,
                "available_chunk_count": available_count,
            }
        )
    return cases


def slice_example_contexts(example: InputExample, chunk_count: int) -> InputExample:
    return InputExample(
        question=example.question,
        answers=list(example.answers),
        ctxs=list(example.ctxs[:chunk_count]),
        example_id=example.example_id,
        metadata={
            **dict(example.metadata),
            "sweep_original_num_ctxs": len(example.ctxs),
            "sweep_chunk_count": chunk_count,
        },
    )


def doc_token_count(tokenized_example: Any) -> int:
    return sum(len(doc_ids) for doc_ids in tokenized_example.doc_chunk_ids)


def token_count_record(tokenized_example: Any) -> dict[str, int]:
    doc_count = doc_token_count(tokenized_example)
    query_count = len(tokenized_example.q_ids)
    return {
        "doc_token_count": doc_count,
        "query_token_count": query_count,
        "total_prefill_token_count": doc_count + query_count,
    }


def safe_synchronize_cuda() -> None:
    try:
        synchronize_cuda_if_available()
    except Exception:
        return


def clear_cuda_cache_after_failure() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom_error(error: Exception) -> bool:
    error_name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "outofmemory" in error_name
        or "out of memory" in message
        or "cuda oom" in message
        or "cuda error: out of memory" in message
    )


def serialize_error(error: Exception) -> dict[str, str]:
    return {
        "type": type(error).__name__,
        "message": str(error),
    }


def run_phase_capture(
    phase_name: str,
    fn: Callable[[], Any],
    *,
    enable_profiling: bool,
) -> tuple[Any | None, dict[str, Any], Exception | None]:
    if enable_profiling:
        reset_peak_memory_stats_if_available()
        safe_synchronize_cuda()

    start = time.perf_counter()
    try:
        result = fn()
        if enable_profiling:
            safe_synchronize_cuda()
        profile_record: dict[str, Any] = {
            "latency_seconds": time.perf_counter() - start,
        }
        if enable_profiling:
            peak_gpu_memory_mb = get_peak_memory_mb_if_available()
            if peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_mb"] = peak_gpu_memory_mb
        return result, profile_record, None
    except Exception as error:
        if enable_profiling:
            safe_synchronize_cuda()
        profile_record = {
            "latency_seconds": time.perf_counter() - start,
            "failed": True,
            "failed_phase": phase_name,
        }
        if enable_profiling:
            peak_gpu_memory_mb = get_peak_memory_mb_if_available()
            if peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_mb"] = peak_gpu_memory_mb
        if is_oom_error(error):
            clear_cuda_cache_after_failure()
        return None, profile_record, error


def phase_latency_seconds(profile_record: dict[str, Any]) -> float:
    return float(profile_record.get("latency_seconds", 0.0))


def total_phase_latency_seconds(phase_metrics: dict[str, dict[str, Any]]) -> float:
    return sum(phase_latency_seconds(phase_record) for phase_record in phase_metrics.values())


def max_phase_peak_memory_mb(phase_metrics: dict[str, dict[str, Any]]) -> float | None:
    peaks = [
        float(phase_record["peak_gpu_memory_mb"])
        for phase_record in phase_metrics.values()
        if phase_record.get("peak_gpu_memory_mb") is not None
    ]
    if not peaks:
        return None
    return max(peaks)


def build_failure_record(
    error: Exception,
    failed_phase: str,
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    status = STATUS_OOM if is_oom_error(error) else STATUS_ERROR
    return {
        "status": status,
        "error_is_oom": status == STATUS_OOM,
        "failed_phase": failed_phase,
        "error": serialize_error(error),
        "phase_metrics": phase_metrics,
        "latency_seconds": total_phase_latency_seconds(phase_metrics),
        "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
        "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
    }


def handle_phase_error(
    error: Exception,
    *,
    failed_phase: str,
    phase_metrics: dict[str, dict[str, Any]],
    continue_on_error: bool,
) -> dict[str, Any]:
    if not is_oom_error(error) and not continue_on_error:
        raise error
    return build_failure_record(
        error=error,
        failed_phase=failed_phase,
        phase_metrics=phase_metrics,
    )


def build_case_failure_record(
    error: Exception,
    *,
    example_index: int,
    example: InputExample,
    dataset_key: str,
    model_name: str,
    model_family: str,
    prompt_policy: str,
    prediction_parser: str,
    repair_planner: str,
    chunk_case: dict[str, Any],
    token_counts: dict[str, int] | None = None,
) -> dict[str, Any]:
    status = STATUS_OOM if is_oom_error(error) else STATUS_ERROR
    if status == STATUS_OOM:
        clear_cuda_cache_after_failure()
    return {
        "record_type": "chunk_sweep_case",
        "status": status,
        "error_is_oom": status == STATUS_OOM,
        "failed_scope": "sweep_case_setup_or_unhandled_method_error",
        "example_index": example_index,
        "example_id": example.example_id,
        "dataset": dataset_key,
        "requested_chunk_count": chunk_case["requested_chunk_count"],
        "chunk_count": chunk_case["chunk_count"],
        "available_chunk_count": chunk_case["available_chunk_count"],
        "model": model_name,
        "model_family": model_family,
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "repair_planner": repair_planner,
        **dict(token_counts or {}),
        "error": serialize_error(error),
    }


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
) -> dict[str, Any]:
    evaluated_prediction = parse_prediction_for_evaluation(
        generated_text,
        prediction_parser=prediction_parser,
    )
    metrics = evaluate_qa_prediction(evaluated_prediction, answers)
    return {
        "status": STATUS_SUCCESS,
        "generated_ids": generated_ids,
        "raw_generated_text": generated_text,
        "generated_text": generated_text,
        "evaluated_prediction": evaluated_prediction,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
    }


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


def precompute_reuse_doc_kv(model: Any, tokenized_example: Any, model_family: str) -> Any:
    chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
    chunk_kvs = correct_doc_chunk_kvs_for_model_family(
        model=model,
        chunk_kvs=chunk_kvs,
        model_family=model_family,
    )
    return assemble_chunk_kvs(chunk_kvs)


def decode_with_past_key_values(
    model: Any,
    tokenizer: Any,
    tokenized_example: Any,
    past_key_values: Any,
    max_new_tokens: int,
) -> Any:
    return generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
    )


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


def build_cacheblend_repair_record(
    generation: Any,
    answers: list[str],
    prediction_parser: str,
    repair_plan: CacheBlendRepairPlan,
    partial_repair_metadata: dict[str, Any],
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_record = evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
    )
    repair_plan_metadata = repair_plan.metadata
    repair_latency_seconds = phase_latency_seconds(phase_metrics["repair"])
    decode_latency_seconds = phase_latency_seconds(phase_metrics["decode"])
    execution_latency_seconds = repair_latency_seconds + decode_latency_seconds
    method_record.update(
        {
            "repair_plan_strategy": repair_plan.strategy,
            "planning_latency_seconds": phase_latency_seconds(phase_metrics["plan"]),
            "uses_full_recompute_reference": repair_plan_metadata.get(
                "uses_full_recompute_reference"
            ),
            "rope_position_correction_applied": repair_plan_metadata.get(
                "rope_position_correction_applied"
            ),
            "execution_mode": "from_plan",
            "execution_uses_full_recompute_reference": partial_repair_metadata.get(
                "execution_uses_full_recompute_reference"
            ),
            "planning_included_in_total_latency": True,
            "execution_latency_seconds": execution_latency_seconds,
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "latency_seconds": total_phase_latency_seconds(phase_metrics),
            "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
            "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
            "phase_metrics": phase_metrics,
            "reuse_past_key_values_source": partial_repair_metadata.get(
                "reuse_past_key_values_source"
            ),
            "runtime_selected_count": len(repair_plan.runtime_selected_indices),
            "runtime_selection_mode": repair_plan.runtime_selection_mode,
            "layer_selected_counts": list(repair_plan.layer_selected_counts),
        }
    )
    return method_record


def build_online_cacheblend_repair_record(
    generation: Any,
    answers: list[str],
    prediction_parser: str,
    partial_repair_metadata: dict[str, Any],
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_record = evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
    )
    repair_latency_seconds = phase_latency_seconds(phase_metrics["online_repair"])
    decode_latency_seconds = phase_latency_seconds(phase_metrics["decode"])
    execution_latency_seconds = repair_latency_seconds + decode_latency_seconds
    method_record.update(
        {
            "repair_plan_strategy": partial_repair_metadata.get("repair_plan_strategy"),
            "repair_planner": REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
            "planning_latency_seconds": None,
            "uses_full_recompute_reference": partial_repair_metadata.get(
                "repair_plan_metadata",
                {},
            ).get("uses_full_recompute_reference"),
            "rope_position_correction_applied": partial_repair_metadata.get(
                "rope_position_correction_applied"
            ),
            "execution_mode": "online_gradual_hkvd",
            "execution_uses_full_recompute_reference": partial_repair_metadata.get(
                "execution_uses_full_recompute_reference"
            ),
            "planning_included_in_total_latency": False,
            "execution_latency_seconds": execution_latency_seconds,
            "reuse_precompute_latency_seconds": phase_latency_seconds(
                phase_metrics["reuse_precompute"]
            ),
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "latency_seconds": total_phase_latency_seconds(phase_metrics),
            "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
            "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
            "phase_metrics": phase_metrics,
            "reuse_past_key_values_source": partial_repair_metadata.get(
                "reuse_past_key_values_source"
            ),
            "runtime_selected_count": len(
                partial_repair_metadata.get("runtime_selected_indices", [])
            ),
            "runtime_selection_mode": partial_repair_metadata.get(
                "runtime_selection_mode"
            ),
            "layer_selected_counts": partial_repair_metadata.get(
                "layer_selected_counts",
                [],
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
    return method_records


def write_jsonl_record(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False))
    file.write("\n")
    file.flush()


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


def method_metric_record(method_record: dict[str, Any]) -> dict[str, float]:
    return {
        "exact_match": float(method_record["exact_match"]),
        "f1": float(method_record["f1"]),
    }


def mean_optional_float(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


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
        "chunk_count_specs": chunk_count_specs,
        "methods": method_summaries,
    }


def add_sweep_context_to_method_records(
    method_records: dict[str, dict[str, Any]],
    *,
    example_index: int,
    example_id: str | None,
    chunk_case: dict[str, Any],
    token_counts: dict[str, int],
) -> None:
    for method_record in method_records.values():
        method_record["example_index"] = example_index
        method_record["example_id"] = example_id
        method_record["requested_chunk_count"] = chunk_case["requested_chunk_count"]
        method_record["chunk_count"] = chunk_case["chunk_count"]
        method_record["available_chunk_count"] = chunk_case["available_chunk_count"]
        method_record.update(token_counts)


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    methods = parse_methods(args.methods)
    chunk_count_specs = parse_chunk_count_specs(args.chunk_counts)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    examples = load_qa_dataset_examples(dataset_key, args.input)
    if args.limit is not None:
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("Expected at least one dataset example.")

    bundle = load_hf_causal_lm(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=normalize_optional_device_map(args.device_map),
    )
    bundle.model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    method_outcomes: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, full_example in enumerate(examples):
            chunk_cases = resolve_example_chunk_cases(full_example, chunk_count_specs)
            for chunk_case in chunk_cases:
                sliced_example = slice_example_contexts(
                    full_example,
                    chunk_count=int(chunk_case["chunk_count"]),
                )
                token_counts: dict[str, int] = {}
                try:
                    prompt = build_cacheblend_prompt(
                        sliced_example,
                        prompt_policy=args.prompt_policy,
                        dataset=dataset_key,
                    )
                    tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
                    token_counts = token_count_record(tokenized)
                    record = build_sweep_case_record(
                        example_index=example_index,
                        example=sliced_example,
                        dataset_key=dataset_key,
                        model_name=args.model,
                        model_family=args.model_family,
                        max_new_tokens=args.max_new_tokens,
                        prompt_policy=args.prompt_policy,
                        prediction_parser=args.prediction_parser,
                        repair_planner=args.repair_planner,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    with torch.inference_mode():
                        method_records = run_methods_for_sweep_case(
                            methods=methods,
                            tokenized_example=tokenized,
                            model=bundle.model,
                            tokenizer=bundle.tokenizer,
                            answers=sliced_example.answers,
                            max_new_tokens=args.max_new_tokens,
                            initial_top_k=args.initial_top_k,
                            top_k=args.top_k,
                            model_family=args.model_family,
                            prediction_parser=args.prediction_parser,
                            enable_profiling=args.enable_profiling,
                            continue_on_error=args.continue_on_error,
                            repair_planner=args.repair_planner,
                        )
                    add_sweep_context_to_method_records(
                        method_records,
                        example_index=example_index,
                        example_id=sliced_example.example_id,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    record["methods"] = method_records
                    for method, method_record in method_records.items():
                        method_outcomes[method].append(method_record)
                except Exception as error:
                    record = build_case_failure_record(
                        error,
                        example_index=example_index,
                        example=full_example,
                        dataset_key=dataset_key,
                        model_name=args.model,
                        model_family=args.model_family,
                        prompt_policy=args.prompt_policy,
                        prediction_parser=args.prediction_parser,
                        repair_planner=args.repair_planner,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    write_jsonl_record(output_file, record)
                    if args.continue_on_error or record["status"] == STATUS_OOM:
                        continue
                    raise

                write_jsonl_record(output_file, record)

    summary = summarize_sweep_results(
        dataset_key=dataset_key,
        model_name=args.model,
        model_family=args.model_family,
        prompt_policy=args.prompt_policy,
        prediction_parser=args.prediction_parser,
        repair_planner=args.repair_planner,
        chunk_count_specs=chunk_count_specs,
        method_outcomes=method_outcomes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
