from __future__ import annotations

import math
from typing import Any

from contextflow.kv_cache.precompute import normalize_past_key_values
from contextflow.repair.adapters.gpt2.adapter import (
    compute_gpt2_layer_qkv,
    get_gpt2_layer,
    patch_selected_kv,
    validate_selected_indices,
)
from contextflow.repair.adapters.selection import (
    normalize_selected_indices_by_layer,
    select_hidden_states_for_indices,
)


def _merge_gpt2_heads(tensor: Any) -> Any:
    """Merge GPT2 attention heads from [batch, heads, seq, dim] to [batch, seq, hidden]."""

    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    batch_size, seq_len, num_heads, head_dim = tensor.shape
    return tensor.view(batch_size, seq_len, num_heads * head_dim)


def _apply_selected_causal_mask(attn_weights: Any, selected_indices: list[int]) -> Any:
    import torch

    full_seq_len = int(attn_weights.shape[-1])
    selected_positions = torch.tensor(
        selected_indices,
        dtype=torch.long,
        device=attn_weights.device,
    ).view(1, 1, len(selected_indices), 1)
    key_positions = torch.arange(
        full_seq_len,
        dtype=torch.long,
        device=attn_weights.device,
    ).view(1, 1, 1, full_seq_len)
    causal_mask = key_positions <= selected_positions
    mask_value = torch.finfo(attn_weights.dtype).min
    return torch.where(causal_mask, attn_weights, mask_value)


def _apply_attention_mask(attn_weights: Any, attention_mask: Any | None) -> Any:
    import torch

    if attention_mask is None:
        return attn_weights

    if attention_mask.ndim == 2:
        if int(attention_mask.shape[-1]) != int(attn_weights.shape[-1]):
            raise ValueError(
                "2D attention_mask length must match KV seq_len; "
                f"got {attention_mask.shape[-1]} and {attn_weights.shape[-1]}."
            )
        mask = attention_mask[:, None, None, :].to(dtype=attn_weights.dtype)
        mask_value = torch.finfo(attn_weights.dtype).min
        return attn_weights + (1.0 - mask) * mask_value

    if attention_mask.ndim == 4:
        if int(attention_mask.shape[-1]) != int(attn_weights.shape[-1]):
            raise ValueError(
                "4D attention_mask length must match KV seq_len; "
                f"got {attention_mask.shape[-1]} and {attn_weights.shape[-1]}."
            )
        return attn_weights + attention_mask.to(dtype=attn_weights.dtype)

    raise ValueError(
        "attention_mask must have shape [batch, seq_len] or [batch, 1, selected, seq_len], "
        f"got {attention_mask.shape}."
    )


def run_gpt2_selected_attention(
    layer: Any,
    selected_hidden_states: Any,
    selected_indices: list[int],
    layer_kv: tuple[Any, Any],
    attention_mask: Any | None = None,
    head_mask: Any | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Run GPT2 attention for selected token positions against a full patched KV cache.

    The selected query positions may be sparse, so causal masking is built directly from
    selected_indices instead of using GPT2Attention's contiguous suffix mask.
    """

    import torch

    if selected_hidden_states.ndim != 3:
        raise ValueError(
            "selected_hidden_states must have shape [batch, selected, hidden_dim], "
            f"got {selected_hidden_states.shape}."
        )
    if len(layer_kv) < 2:
        raise ValueError("layer_kv must contain key and value tensors.")

    key, value = layer_kv[0], layer_kv[1]
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match, got {key.shape} and {value.shape}.")
    if selected_hidden_states.device != key.device:
        raise ValueError("selected_hidden_states and layer_kv must be on the same device.")
    if int(selected_hidden_states.shape[0]) != int(key.shape[0]):
        raise ValueError(
            "selected_hidden_states batch size must match layer_kv batch size; "
            f"got {selected_hidden_states.shape[0]} and {key.shape[0]}."
        )
    if int(selected_hidden_states.shape[1]) != len(selected_indices):
        raise ValueError(
            "selected_hidden_states seq_len must match len(selected_indices); "
            f"got {selected_hidden_states.shape[1]} and {len(selected_indices)}."
        )

    full_seq_len = int(key.shape[-2])
    validate_selected_indices(selected_indices, full_seq_len)

    selected_query, selected_key, selected_value = compute_gpt2_layer_qkv(
        layer,
        selected_hidden_states,
    )
    patched_layer_kv = patch_selected_kv(
        layer_kv=layer_kv,
        selected_indices=selected_indices,
        selected_key=selected_key,
        selected_value=selected_value,
    )
    patched_key, patched_value = patched_layer_kv

    attn_weights = torch.matmul(selected_query, patched_key.transpose(-1, -2))
    if getattr(layer.attn, "scale_attn_weights", True):
        attn_weights = attn_weights / math.sqrt(float(patched_value.size(-1)))
    if getattr(layer.attn, "scale_attn_by_inverse_layer_idx", False):
        layer_idx = getattr(layer.attn, "layer_idx", None)
        if layer_idx is None:
            raise ValueError(
                "GPT2 attention requested inverse-layer scaling but layer_idx is unset."
            )
        attn_weights = attn_weights / float(layer_idx + 1)

    attn_weights = _apply_selected_causal_mask(attn_weights, selected_indices)
    attn_weights = _apply_attention_mask(attn_weights, attention_mask)
    attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
    attn_weights = attn_weights.type(patched_value.dtype)
    attn_weights = layer.attn.attn_dropout(attn_weights)
    if head_mask is not None:
        attn_weights = attn_weights * head_mask

    attn_output = torch.matmul(attn_weights, patched_value)
    attn_output = _merge_gpt2_heads(attn_output)
    attn_output = layer.attn.c_proj(attn_output)
    attn_output = layer.attn.resid_dropout(attn_output)
    return attn_output, patched_layer_kv


def run_gpt2_partial_layer(
    model: Any,
    layer_index: int,
    selected_hidden_states: Any,
    selected_indices: list[int],
    layer_kv: tuple[Any, Any],
    attention_mask: Any | None = None,
    head_mask: Any | None = None,
) -> tuple[Any, tuple[Any, Any]]:
    """Propagate selected GPT2 hidden states through one transformer layer."""

    layer = get_gpt2_layer(model, layer_index)
    attn_output, patched_layer_kv = run_gpt2_selected_attention(
        layer=layer,
        selected_hidden_states=selected_hidden_states,
        selected_indices=selected_indices,
        layer_kv=layer_kv,
        attention_mask=attention_mask,
        head_mask=head_mask,
    )
    hidden_after_attn = selected_hidden_states + attn_output
    mlp_output = layer.mlp(layer.ln_2(hidden_after_attn))
    next_selected_hidden_states = hidden_after_attn + mlp_output
    return next_selected_hidden_states, patched_layer_kv


def _select_layer_head_mask(head_mask: Any | None, layer_index: int) -> Any | None:
    if head_mask is None:
        return None
    if isinstance(head_mask, (list, tuple)):
        return head_mask[layer_index]
    if getattr(head_mask, "ndim", 0) >= 1:
        return head_mask[layer_index]
    return head_mask


def run_gpt2_partial_layers(
    model: Any,
    selected_hidden_states: Any,
    selected_indices: list[int],
    past_key_values: Any,
    start_layer_index: int = 0,
    end_layer_index: int | None = None,
    attention_mask: Any | None = None,
    head_mask: Any | None = None,
    selected_indices_by_layer: dict[int, list[int]] | None = None,
) -> tuple[Any, tuple[tuple[Any, Any], ...]]:
    """Propagate selected GPT2 hidden states through multiple transformer layers.

    This runs GPT2 blocks only. It intentionally does not apply GPT2's final ln_f.
    """

    normalized_past_key_values = normalize_past_key_values(past_key_values)
    num_layers = len(normalized_past_key_values)
    if num_layers == 0:
        raise ValueError("past_key_values must contain at least one layer.")
    if end_layer_index is None:
        end_layer_index = num_layers
    if start_layer_index < 0 or start_layer_index > num_layers:
        raise ValueError(
            f"start_layer_index must be in [0, {num_layers}], got {start_layer_index}."
        )
    if end_layer_index < start_layer_index or end_layer_index > num_layers:
        raise ValueError(
            "end_layer_index must be in [start_layer_index, num_layers]; "
            f"got {end_layer_index} with start_layer_index={start_layer_index} "
            f"and num_layers={num_layers}."
        )

    full_seq_len = int(normalized_past_key_values[0][0].shape[-2])
    layer_selected_indices = normalize_selected_indices_by_layer(
        selected_indices=selected_indices,
        selected_indices_by_layer=selected_indices_by_layer,
        start_layer_index=start_layer_index,
        end_layer_index=end_layer_index,
        full_seq_len=full_seq_len,
    )

    patched_past_key_values = list(normalized_past_key_values)
    current_selected_hidden_states = selected_hidden_states
    current_selected_indices = list(selected_indices)
    for layer_index in range(start_layer_index, end_layer_index):
        selected_for_layer = layer_selected_indices[layer_index]
        current_selected_hidden_states = select_hidden_states_for_indices(
            hidden_states=current_selected_hidden_states,
            current_indices=current_selected_indices,
            target_indices=selected_for_layer,
        )
        current_selected_indices = selected_for_layer
        current_selected_hidden_states, patched_layer_kv = run_gpt2_partial_layer(
            model=model,
            layer_index=layer_index,
            selected_hidden_states=current_selected_hidden_states,
            selected_indices=selected_for_layer,
            layer_kv=patched_past_key_values[layer_index],
            attention_mask=attention_mask,
            head_mask=_select_layer_head_mask(head_mask, layer_index),
        )
        original_key, original_value = normalized_past_key_values[layer_index][:2]
        patched_key, patched_value = patched_layer_kv[:2]
        if tuple(patched_key.shape) != tuple(original_key.shape):
            raise ValueError(
                f"Layer {layer_index} patched key shape must match original key shape; "
                f"got {patched_key.shape} and {original_key.shape}."
            )
        if tuple(patched_value.shape) != tuple(original_value.shape):
            raise ValueError(
                f"Layer {layer_index} patched value shape must match original value shape; "
                f"got {patched_value.shape} and {original_value.shape}."
            )
        patched_past_key_values[layer_index] = patched_layer_kv

    return current_selected_hidden_states, tuple(patched_past_key_values)
