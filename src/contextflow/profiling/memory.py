from __future__ import annotations

from typing import Any


def get_torch_if_available() -> Any | None:
    try:
        import torch
    except ModuleNotFoundError:
        return None
    return torch


def reset_peak_memory_stats_if_available() -> None:
    """Reset CUDA peak memory stats when CUDA is available."""

    torch = get_torch_if_available()
    if torch is None:
        return
    if not torch.cuda.is_available():
        return
    torch.cuda.reset_peak_memory_stats()


def get_peak_memory_mb_if_available() -> float | None:
    """Return peak allocated CUDA memory in MB, or None on CPU-only runs."""

    torch = get_torch_if_available()
    if torch is None:
        return None
    if not torch.cuda.is_available():
        return None
    return float(torch.cuda.max_memory_allocated()) / (1024 * 1024)
