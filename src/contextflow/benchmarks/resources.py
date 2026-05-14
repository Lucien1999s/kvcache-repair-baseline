from __future__ import annotations

import gc
import time
from collections.abc import Callable
from typing import Any

from contextflow.benchmarks.constants import FUSIONRAG_PROFILE_PHASES
from contextflow.profiling import timed_call as profiling_timed_call


def run_profiled_phase(
    fn: Callable[[], Any],
    *,
    enable_profiling: bool,
) -> tuple[Any, dict[str, Any]]:
    if enable_profiling:
        result, profile_record = profiling_timed_call(
            fn,
            synchronize_cuda=True,
            collect_peak_memory=True,
        )
        return result, dict(profile_record)

    start = time.perf_counter()
    result = fn()
    return result, {"latency_seconds": time.perf_counter() - start}


def cleanup_cuda_between_methods() -> None:
    """Release Python references and cached CUDA blocks between method runs."""

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


def run_with_method_cleanup(fn: Callable[[], Any]) -> Any:
    """Run one benchmark method with cleanup before and after it."""

    cleanup_cuda_between_methods()
    try:
        return fn()
    finally:
        cleanup_cuda_between_methods()


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


def first_phase_baseline_memory_mb(
    phase_metrics: dict[str, dict[str, Any]],
) -> float | None:
    for phase_record in phase_metrics.values():
        if phase_record.get("baseline_gpu_memory_mb") is not None:
            return float(phase_record["baseline_gpu_memory_mb"])
    return None


def peak_memory_delta_mb(phase_metrics: dict[str, dict[str, Any]]) -> float | None:
    baseline = first_phase_baseline_memory_mb(phase_metrics)
    peak = max_phase_peak_memory_mb(phase_metrics)
    if baseline is None or peak is None:
        return None
    return max(0.0, peak - baseline)


def memory_summary_record(phase_metrics: dict[str, dict[str, Any]]) -> dict[str, float | None]:
    return {
        "baseline_gpu_memory_mb": first_phase_baseline_memory_mb(phase_metrics),
        "peak_gpu_memory_mb": max_phase_peak_memory_mb(phase_metrics),
        "peak_gpu_memory_delta_mb": peak_memory_delta_mb(phase_metrics),
    }


def phase_profile_fields(
    phase_name: str,
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, float | None]:
    """Return JSON-friendly latency and memory fields for one named phase."""

    phase_record = phase_metrics.get(phase_name, {})
    if not phase_record:
        return {
            f"{phase_name}_latency_seconds": None,
            f"{phase_name}_baseline_gpu_memory_mb": None,
            f"{phase_name}_peak_gpu_memory_mb": None,
            f"{phase_name}_peak_gpu_memory_delta_mb": None,
        }
    return {
        f"{phase_name}_latency_seconds": phase_latency_seconds(phase_record),
        f"{phase_name}_baseline_gpu_memory_mb": (
            float(phase_record["baseline_gpu_memory_mb"])
            if phase_record.get("baseline_gpu_memory_mb") is not None
            else None
        ),
        f"{phase_name}_peak_gpu_memory_mb": (
            float(phase_record["peak_gpu_memory_mb"])
            if phase_record.get("peak_gpu_memory_mb") is not None
            else None
        ),
        f"{phase_name}_peak_gpu_memory_delta_mb": (
            float(phase_record["peak_gpu_memory_delta_mb"])
            if phase_record.get("peak_gpu_memory_delta_mb") is not None
            else None
        ),
    }


def selected_phase_profile_fields(
    phase_metrics: dict[str, dict[str, Any]],
    phase_names: list[str],
) -> dict[str, float | None]:
    fields: dict[str, float | None] = {}
    for phase_name in phase_names:
        fields.update(phase_profile_fields(phase_name, phase_metrics))
    return fields


def mean_optional_float(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_resource_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_latency_seconds": mean_optional_float(records, "latency_seconds"),
        "mean_total_latency_seconds": mean_optional_float(records, "total_latency_seconds"),
        "mean_peak_gpu_memory_mb": mean_optional_float(records, "peak_gpu_memory_mb"),
        "mean_baseline_gpu_memory_mb": mean_optional_float(
            records,
            "baseline_gpu_memory_mb",
        ),
        "mean_peak_gpu_memory_delta_mb": mean_optional_float(
            records,
            "peak_gpu_memory_delta_mb",
        ),
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
        **{
            f"mean_{phase_name}_latency_seconds": mean_optional_float(
                records,
                f"{phase_name}_latency_seconds",
            )
            for phase_name in FUSIONRAG_PROFILE_PHASES
        },
        **{
            f"mean_{phase_name}_peak_gpu_memory_delta_mb": mean_optional_float(
                records,
                f"{phase_name}_peak_gpu_memory_delta_mb",
            )
            for phase_name in FUSIONRAG_PROFILE_PHASES
        },
    }
