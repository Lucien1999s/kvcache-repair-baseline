from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Optional, TypeVar

from contextflow.profiling.memory import (
    get_peak_memory_mb_if_available,
    get_torch_if_available,
    reset_peak_memory_stats_if_available,
)


T = TypeVar("T")
ProfileRecord = dict[str, Optional[float]]


def synchronize_cuda_if_available() -> None:
    """Synchronize CUDA when available so wall-clock timing includes queued work."""

    torch = get_torch_if_available()
    if torch is None:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextmanager
def synchronized_timer(
    *,
    synchronize_cuda: bool = True,
) -> Iterator[ProfileRecord]:
    """Context manager that records synchronized wall-clock latency seconds."""

    if synchronize_cuda:
        synchronize_cuda_if_available()
    start = time.perf_counter()
    record: ProfileRecord = {}
    try:
        yield record
    finally:
        if synchronize_cuda:
            synchronize_cuda_if_available()
        record["latency_seconds"] = time.perf_counter() - start


def timed_call(
    fn: Callable[[], T],
    *,
    synchronize_cuda: bool = True,
    collect_peak_memory: bool = False,
) -> tuple[T, ProfileRecord]:
    """Run a callable and return its result plus a JSON-serializable profile record."""

    if collect_peak_memory:
        reset_peak_memory_stats_if_available()
    with synchronized_timer(synchronize_cuda=synchronize_cuda) as record:
        result = fn()
    if collect_peak_memory:
        peak_memory_mb = get_peak_memory_mb_if_available()
        if peak_memory_mb is not None:
            record["peak_gpu_memory_mb"] = peak_memory_mb
    return result, record
