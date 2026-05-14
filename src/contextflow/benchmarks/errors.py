from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

from contextflow.benchmarks.constants import (
    FUSIONRAG_PROFILE_PHASES,
    STATUS_ERROR,
    STATUS_OOM,
)
from contextflow.benchmarks.resources import (
    memory_summary_record,
    selected_phase_profile_fields,
    total_phase_latency_seconds,
)
from contextflow.data import InputExample
from contextflow.profiling import (
    get_allocated_memory_mb_if_available,
    get_peak_memory_mb_if_available,
    reset_peak_memory_stats_if_available,
    synchronize_cuda_if_available,
)


def safe_synchronize_cuda() -> None:
    try:
        synchronize_cuda_if_available()
    except Exception:
        return


def clear_cuda_cache_after_failure() -> None:
    gc.collect()
    try:
        import torch
    except ModuleNotFoundError:
        return
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        except Exception:
            return


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
    baseline_memory_mb = None
    if enable_profiling:
        safe_synchronize_cuda()
        baseline_memory_mb = get_allocated_memory_mb_if_available()
        reset_peak_memory_stats_if_available()

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
            if baseline_memory_mb is not None:
                profile_record["baseline_gpu_memory_mb"] = baseline_memory_mb
            if peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            if baseline_memory_mb is not None and peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_delta_mb"] = max(
                    0.0,
                    peak_gpu_memory_mb - baseline_memory_mb,
                )
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
            if baseline_memory_mb is not None:
                profile_record["baseline_gpu_memory_mb"] = baseline_memory_mb
            if peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_mb"] = peak_gpu_memory_mb
            if baseline_memory_mb is not None and peak_gpu_memory_mb is not None:
                profile_record["peak_gpu_memory_delta_mb"] = max(
                    0.0,
                    peak_gpu_memory_mb - baseline_memory_mb,
                )
        if is_oom_error(error):
            clear_cuda_cache_after_failure()
        return None, profile_record, error


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
        **memory_summary_record(phase_metrics),
        **selected_phase_profile_fields(phase_metrics, FUSIONRAG_PROFILE_PHASES),
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
    chunking_config: dict[str, Any] | None = None,
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
        **dict(chunking_config or {}),
        **dict(token_counts or {}),
        "error": serialize_error(error),
    }
