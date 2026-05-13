from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

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
        "mean_planning_latency_seconds": mean_optional_float(
            records,
            "planning_latency_seconds",
        ),
        "mean_reuse_precompute_latency_seconds": mean_optional_float(
            records,
            "reuse_precompute_latency_seconds",
        ),
        "mean_repair_latency_seconds": mean_optional_float(records, "repair_latency_seconds"),
        "mean_decode_latency_seconds": mean_optional_float(records, "decode_latency_seconds"),
        "mean_execution_latency_seconds": mean_optional_float(
            records,
            "execution_latency_seconds",
        ),
    }
