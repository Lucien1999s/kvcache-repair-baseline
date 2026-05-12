from __future__ import annotations

import json

from contextflow.profiling import (
    get_peak_memory_mb_if_available,
    reset_peak_memory_stats_if_available,
    timed_call,
)


def main() -> None:
    reset_peak_memory_stats_if_available()
    peak_memory_mb = get_peak_memory_mb_if_available()
    assert peak_memory_mb is None or peak_memory_mb >= 0

    result, profile_record = timed_call(
        lambda: {"ok": True},
        synchronize_cuda=True,
        collect_peak_memory=True,
    )
    assert result == {"ok": True}
    assert profile_record["latency_seconds"] >= 0
    if "peak_gpu_memory_mb" in profile_record:
        assert profile_record["peak_gpu_memory_mb"] is not None
        assert profile_record["peak_gpu_memory_mb"] >= 0

    json.dumps(
        {
            "latency_seconds": profile_record["latency_seconds"],
            "peak_gpu_memory_mb": profile_record.get("peak_gpu_memory_mb"),
        }
    )

    print("profiling smoke test passed")
    print(f"profile_record: {profile_record}")


if __name__ == "__main__":
    main()
