from __future__ import annotations

import math
from typing import Any

from contextflow.kv_cache.precompute import normalize_past_key_values


def get_layer_key_value(layer_kv: Any) -> tuple[Any, Any]:
    if len(layer_kv) < 2:
        raise ValueError("Each KV layer must contain key and value tensors.")
    return layer_kv[0], layer_kv[1]


def validate_kv_pair_shapes(reuse_tensor: Any, full_tensor: Any) -> None:
    reuse_shape = tuple(reuse_tensor.shape)
    full_shape = tuple(full_tensor.shape)
    if len(reuse_shape) < 3 or len(full_shape) < 3:
        raise ValueError(
            f"KV tensors must have rank >= 3 with sequence dimension at -2; "
            f"got {reuse_shape} and {full_shape}."
        )
    if reuse_shape != full_shape:
        raise ValueError(
            f"Reuse and full KV tensor shapes must match; got {reuse_shape} and {full_shape}."
        )
    if reuse_tensor.device != full_tensor.device:
        raise ValueError(
            "Reuse and full KV tensors must be on the same device; "
            f"got {reuse_tensor.device} and {full_tensor.device}."
        )


def l2_norm_by_token(tensor: Any) -> Any:
    seq_dim = tensor.ndim - 2
    by_token = tensor.float().movedim(seq_dim, 0)
    return by_token.reshape(by_token.shape[0], -1).norm(p=2, dim=1)


def compute_layer_kv_deviation(reuse_layer_kv: Any, full_layer_kv: Any) -> Any:
    """Compute CacheBlend-style token-level KV deviation for one transformer layer."""

    reuse_key, reuse_value = get_layer_key_value(reuse_layer_kv)
    full_key, full_value = get_layer_key_value(full_layer_kv)
    validate_kv_pair_shapes(reuse_key, full_key)
    validate_kv_pair_shapes(reuse_value, full_value)

    key_diff = l2_norm_by_token(reuse_key - full_key)
    value_diff = l2_norm_by_token(reuse_value - full_value)
    return key_diff + value_diff


def compute_kv_deviation(reuse_past_key_values: Any, full_past_key_values: Any) -> list[Any]:
    """Compute per-layer token-level KV deviation between reused and full-compute KVs."""

    reuse_past_key_values = normalize_past_key_values(reuse_past_key_values)
    full_past_key_values = normalize_past_key_values(full_past_key_values)

    if len(reuse_past_key_values) != len(full_past_key_values):
        raise ValueError(
            "Reuse and full past_key_values must have the same number of layers; "
            f"got {len(reuse_past_key_values)} and {len(full_past_key_values)}."
        )
    if not reuse_past_key_values:
        raise ValueError("past_key_values must contain at least one layer.")

    return [
        compute_layer_kv_deviation(reuse_layer_kv, full_layer_kv)
        for reuse_layer_kv, full_layer_kv in zip(reuse_past_key_values, full_past_key_values)
    ]


def resolve_topk_count(seq_len: int, ratio: float | None, top_k: int | None) -> int:
    if seq_len <= 0:
        raise ValueError("layer_deviation must be non-empty.")
    if (ratio is None) == (top_k is None):
        raise ValueError("Exactly one of ratio or top_k must be provided.")
    if top_k is not None:
        if top_k <= 0:
            raise ValueError("top_k must be positive.")
        return min(top_k, seq_len)
    if ratio is None or ratio <= 0 or ratio > 1:
        raise ValueError("ratio must be in the range (0, 1].")
    return max(1, min(seq_len, math.ceil(seq_len * ratio)))


def select_topk_hkvd_tokens(
    layer_deviation: Any,
    ratio: float | None = None,
    top_k: int | None = None,
) -> list[int]:
    """Select token positions with highest KV deviation for one layer."""

    if layer_deviation.ndim != 1:
        raise ValueError(
            f"layer_deviation must be 1D with shape [seq_len], got {tuple(layer_deviation.shape)}."
        )

    k = resolve_topk_count(int(layer_deviation.shape[0]), ratio=ratio, top_k=top_k)
    selected = layer_deviation.topk(k=k, largest=True).indices.tolist()
    return sorted(int(index) for index in selected)


def select_hkvd_tokens_by_layer(
    deviations: list[Any],
    ratio: float | None = None,
    top_k: int | None = None,
) -> dict[int, list[int]]:
    """Select HKVD token positions independently for every transformer layer."""

    if not deviations:
        raise ValueError("deviations must contain at least one layer.")

    return {
        layer_index: select_topk_hkvd_tokens(
            layer_deviation,
            ratio=ratio,
            top_k=top_k,
        )
        for layer_index, layer_deviation in enumerate(deviations)
    }
