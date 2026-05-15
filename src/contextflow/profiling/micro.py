from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from typing import Any

from contextflow.profiling.memory import (
    get_allocated_memory_mb_if_available,
    get_peak_memory_mb_if_available,
    get_torch_if_available,
    reset_peak_memory_stats_if_available,
)
from contextflow.profiling.timing import synchronize_cuda_if_available


STATUS_SUCCESS = "success"
STATUS_ERROR = "error"
STATUS_OOM = "oom"


def cuda_available() -> bool:
    torch = get_torch_if_available()
    return bool(torch is not None and torch.cuda.is_available())


def is_oom_error(error: Exception) -> bool:
    error_name = type(error).__name__.lower()
    message = str(error).lower()
    return (
        "outofmemory" in error_name
        or "out of memory" in message
        or "cuda oom" in message
        or "cuda error: out of memory" in message
    )


def safe_synchronize_cuda() -> None:
    try:
        synchronize_cuda_if_available()
    except Exception:
        return


class MicroProfiler:
    """Capture latency and CUDA peak-memory deltas for small repair sub-steps."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._records: list[dict[str, Any]] = []

    @property
    def records(self) -> list[dict[str, Any]]:
        return self._records

    @contextmanager
    def profile(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return

        has_cuda = cuda_available()
        baseline_memory_mb = None
        if has_cuda:
            safe_synchronize_cuda()
            try:
                baseline_memory_mb = get_allocated_memory_mb_if_available()
                reset_peak_memory_stats_if_available()
            except Exception:
                baseline_memory_mb = None

        start = time.perf_counter()
        try:
            yield
        except Exception as error:
            self._records.append(
                self._build_record(
                    name=name,
                    start=start,
                    status=STATUS_OOM if is_oom_error(error) else STATUS_ERROR,
                    baseline_memory_mb=baseline_memory_mb,
                    error=error,
                )
            )
            raise
        else:
            self._records.append(
                self._build_record(
                    name=name,
                    start=start,
                    status=STATUS_SUCCESS,
                    baseline_memory_mb=baseline_memory_mb,
                    error=None,
                )
            )

    def _build_record(
        self,
        *,
        name: str,
        start: float,
        status: str,
        baseline_memory_mb: float | None,
        error: Exception | None,
    ) -> dict[str, Any]:
        has_cuda = cuda_available()
        if has_cuda:
            safe_synchronize_cuda()
        try:
            peak_memory_mb = get_peak_memory_mb_if_available() if has_cuda else None
        except Exception:
            peak_memory_mb = None
        record: dict[str, Any] = {
            "name": name,
            "latency_seconds": time.perf_counter() - start,
            "baseline_gpu_memory_mb": baseline_memory_mb,
            "peak_gpu_memory_mb": peak_memory_mb,
            "peak_gpu_memory_delta_mb": (
                max(0.0, peak_memory_mb - baseline_memory_mb)
                if baseline_memory_mb is not None and peak_memory_mb is not None
                else None
            ),
            "status": status,
            "error_type": type(error).__name__ if error is not None else None,
            "error_message": str(error) if error is not None else None,
            "cuda_available": has_cuda,
        }
        return record

    def add_aggregate_record(
        self,
        *,
        name: str,
        child_records: list[dict[str, Any]],
        latency_seconds: float,
    ) -> None:
        if not self.enabled:
            return
        failed = [record for record in child_records if record.get("status") != STATUS_SUCCESS]
        status = STATUS_SUCCESS
        error_type = None
        error_message = None
        if failed:
            first_failure = failed[0]
            status = str(first_failure.get("status", STATUS_ERROR))
            error_type = first_failure.get("error_type")
            error_message = first_failure.get("error_message")
        peaks = [
            float(record["peak_gpu_memory_mb"])
            for record in child_records
            if record.get("peak_gpu_memory_mb") is not None
        ]
        deltas = [
            float(record["peak_gpu_memory_delta_mb"])
            for record in child_records
            if record.get("peak_gpu_memory_delta_mb") is not None
        ]
        baselines = [
            float(record["baseline_gpu_memory_mb"])
            for record in child_records
            if record.get("baseline_gpu_memory_mb") is not None
        ]
        self._records.append(
            {
                "name": name,
                "latency_seconds": latency_seconds,
                "baseline_gpu_memory_mb": baselines[0] if baselines else None,
                "peak_gpu_memory_mb": max(peaks) if peaks else None,
                "peak_gpu_memory_delta_mb": max(deltas) if deltas else None,
                "status": status,
                "error_type": error_type,
                "error_message": error_message,
                "cuda_available": cuda_available(),
            }
        )

    def to_records(self) -> list[dict[str, Any]]:
        return [dict(record) for record in self._records]

    def summary(self) -> dict[str, Any]:
        if not self._records:
            return {
                "micro_step_count": 0,
                "failed_micro_step": None,
                "max_peak_gpu_memory_delta_step": None,
                "max_peak_gpu_memory_delta_mb": None,
                "max_peak_gpu_memory_step": None,
                "max_peak_gpu_memory_mb": None,
                "total_micro_latency_seconds": 0.0,
            }

        failed = [record for record in self._records if record.get("status") != STATUS_SUCCESS]
        delta_records = [
            record
            for record in self._records
            if record.get("peak_gpu_memory_delta_mb") is not None
        ]
        peak_records = [
            record for record in self._records if record.get("peak_gpu_memory_mb") is not None
        ]
        max_delta_record = (
            max(delta_records, key=lambda record: float(record["peak_gpu_memory_delta_mb"]))
            if delta_records
            else None
        )
        max_peak_record = (
            max(peak_records, key=lambda record: float(record["peak_gpu_memory_mb"]))
            if peak_records
            else None
        )
        return {
            "micro_step_count": len(self._records),
            "failed_micro_step": failed[0]["name"] if failed else None,
            "max_peak_gpu_memory_delta_step": (
                max_delta_record["name"] if max_delta_record is not None else None
            ),
            "max_peak_gpu_memory_delta_mb": (
                float(max_delta_record["peak_gpu_memory_delta_mb"])
                if max_delta_record is not None
                else None
            ),
            "max_peak_gpu_memory_step": (
                max_peak_record["name"] if max_peak_record is not None else None
            ),
            "max_peak_gpu_memory_mb": (
                float(max_peak_record["peak_gpu_memory_mb"])
                if max_peak_record is not None
                else None
            ),
            "total_micro_latency_seconds": sum(
                float(record.get("latency_seconds", 0.0)) for record in self._records
            ),
        }


def profile_micro_step(
    micro_profiler: MicroProfiler | None,
    name: str,
) -> Iterator[None]:
    if micro_profiler is None:
        return nullcontext()
    return micro_profiler.profile(name)
