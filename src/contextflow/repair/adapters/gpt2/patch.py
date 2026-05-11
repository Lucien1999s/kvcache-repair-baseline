from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.kv_cache.precompute import normalize_past_key_values
from contextflow.repair.adapters.gpt2.adapter import (
    recompute_and_patch_gpt2_layer_kv,
    validate_gpt2_like_model,
)
from contextflow.repair.cacheblend_selector import (
    compute_kv_deviation,
    select_gradual_hkvd_tokens_by_layer,
    select_hkvd_tokens_by_layer,
)


@dataclass(slots=True)
class GPT2HKVDPatchResult:
    patched_past_key_values: tuple[tuple[Any, Any], ...]
    selected_indices_by_layer: dict[int, list[int]]
    deviations: list[Any]


def validate_hidden_states_by_layer(
    hidden_states_by_layer: list[Any] | tuple[Any, ...],
    num_layers: int,
    seq_len: int,
) -> None:
    if len(hidden_states_by_layer) != num_layers:
        raise ValueError(
            "hidden_states_by_layer must contain one input hidden-state tensor per layer; "
            f"expected {num_layers}, got {len(hidden_states_by_layer)}."
        )
    for layer_index, hidden_states in enumerate(hidden_states_by_layer):
        if hidden_states.ndim != 3:
            raise ValueError(
                f"hidden_states_by_layer[{layer_index}] must have shape "
                f"[batch, seq_len, hidden_dim], got {hidden_states.shape}."
            )
        if int(hidden_states.shape[1]) != seq_len:
            raise ValueError(
                f"hidden_states_by_layer[{layer_index}] seq_len must match KV seq_len; "
                f"expected {seq_len}, got {hidden_states.shape[1]}."
            )


def validate_layer_kv_shape(layer_index: int, reference_layer_kv: Any, patched_layer_kv: Any) -> None:
    reference_key, reference_value = reference_layer_kv[0], reference_layer_kv[1]
    patched_key, patched_value = patched_layer_kv[0], patched_layer_kv[1]
    if tuple(patched_key.shape) != tuple(reference_key.shape):
        raise ValueError(
            f"Layer {layer_index} patched key shape must match reuse key shape; "
            f"got {patched_key.shape} and {reference_key.shape}."
        )
    if tuple(patched_value.shape) != tuple(reference_value.shape):
        raise ValueError(
            f"Layer {layer_index} patched value shape must match reuse value shape; "
            f"got {patched_value.shape} and {reference_value.shape}."
        )


def patch_gpt2_layers_with_hkvd_tokens(
    model: Any,
    hidden_states_by_layer: list[Any] | tuple[Any, ...],
    reuse_past_key_values: Any,
    full_past_key_values: Any,
    top_k: int | None = None,
    ratio: float | None = None,
) -> GPT2HKVDPatchResult:
    """Patch every GPT2 layer at HKVD-selected token positions.

    This is a GPT2-only primitive. The caller is responsible for passing doc-prefix
    hidden states that correspond to each layer's input hidden states.
    """

    validate_gpt2_like_model(model)
    reuse_past_key_values = normalize_past_key_values(reuse_past_key_values)
    full_past_key_values = normalize_past_key_values(full_past_key_values)
    deviations = compute_kv_deviation(reuse_past_key_values, full_past_key_values)

    if not reuse_past_key_values:
        raise ValueError("reuse_past_key_values must contain at least one layer.")
    num_layers = len(reuse_past_key_values)
    seq_len = int(reuse_past_key_values[0][0].shape[-2])
    validate_hidden_states_by_layer(hidden_states_by_layer, num_layers=num_layers, seq_len=seq_len)

    selected_indices_by_layer = select_hkvd_tokens_by_layer(
        deviations,
        ratio=ratio,
        top_k=top_k,
    )

    patched_past_key_values = patch_gpt2_layers_with_selected_tokens(
        model=model,
        hidden_states_by_layer=hidden_states_by_layer,
        reuse_past_key_values=reuse_past_key_values,
        selected_indices_by_layer=selected_indices_by_layer,
    )

    return GPT2HKVDPatchResult(
        patched_past_key_values=patched_past_key_values,
        selected_indices_by_layer=selected_indices_by_layer,
        deviations=deviations,
    )


def patch_gpt2_layers_with_selected_tokens(
    model: Any,
    hidden_states_by_layer: list[Any] | tuple[Any, ...],
    reuse_past_key_values: Any,
    selected_indices_by_layer: dict[int, list[int]],
) -> tuple[tuple[Any, Any], ...]:
    patched_layers = []
    for layer_index, layer_kv in enumerate(reuse_past_key_values):
        if layer_index not in selected_indices_by_layer:
            raise ValueError(f"Missing selected indices for layer {layer_index}.")
        patched_layer_kv = recompute_and_patch_gpt2_layer_kv(
            model=model,
            layer_index=layer_index,
            hidden_states=hidden_states_by_layer[layer_index],
            layer_kv=layer_kv,
            selected_indices=selected_indices_by_layer[layer_index],
        )
        validate_layer_kv_shape(layer_index, layer_kv, patched_layer_kv)
        patched_layers.append(patched_layer_kv)
    return tuple(patched_layers)


def patch_gpt2_layers_with_gradual_hkvd_tokens(
    model: Any,
    hidden_states_by_layer: list[Any] | tuple[Any, ...],
    reuse_past_key_values: Any,
    full_past_key_values: Any,
    initial_top_k: int | None = None,
    initial_ratio: float | None = None,
    top_k: int | None = None,
    ratio: float | None = None,
) -> GPT2HKVDPatchResult:
    """Patch GPT2 layers with CacheBlend-style gradual HKVD filtering."""

    validate_gpt2_like_model(model)
    reuse_past_key_values = normalize_past_key_values(reuse_past_key_values)
    full_past_key_values = normalize_past_key_values(full_past_key_values)
    deviations = compute_kv_deviation(reuse_past_key_values, full_past_key_values)

    if not reuse_past_key_values:
        raise ValueError("reuse_past_key_values must contain at least one layer.")
    num_layers = len(reuse_past_key_values)
    seq_len = int(reuse_past_key_values[0][0].shape[-2])
    validate_hidden_states_by_layer(hidden_states_by_layer, num_layers=num_layers, seq_len=seq_len)

    selected_indices_by_layer = select_gradual_hkvd_tokens_by_layer(
        deviations,
        initial_top_k=initial_top_k,
        initial_ratio=initial_ratio,
        top_k=top_k,
        ratio=ratio,
    )
    patched_past_key_values = patch_gpt2_layers_with_selected_tokens(
        model=model,
        hidden_states_by_layer=hidden_states_by_layer,
        reuse_past_key_values=reuse_past_key_values,
        selected_indices_by_layer=selected_indices_by_layer,
    )

    return GPT2HKVDPatchResult(
        patched_past_key_values=patched_past_key_values,
        selected_indices_by_layer=selected_indices_by_layer,
        deviations=deviations,
    )
