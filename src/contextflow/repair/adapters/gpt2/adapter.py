from __future__ import annotations

from typing import Any


def validate_gpt2_like_model(model: Any) -> None:
    """Validate the minimal GPT2LMHeadModel structure needed by this adapter."""

    if not hasattr(model, "transformer") or not hasattr(model.transformer, "h"):
        raise ValueError("Expected a GPT2-like model with model.transformer.h layers.")
    if len(model.transformer.h) == 0:
        raise ValueError("Expected model.transformer.h to contain at least one layer.")
    layer = model.transformer.h[0]
    if not hasattr(layer, "attn"):
        raise ValueError("Expected GPT2-like layer to have an attn module.")
    if not hasattr(layer.attn, "c_attn"):
        raise ValueError("Expected GPT2-like attention module to have attn.c_attn.")
    if not hasattr(layer.attn, "c_proj"):
        raise ValueError("Expected GPT2-like attention module to have attn.c_proj.")
    if not hasattr(layer, "ln_1"):
        raise ValueError("Expected GPT2-like layer to have ln_1 before attention.")


def get_gpt2_layer(model: Any, layer_index: int) -> Any:
    """Return model.transformer.h[layer_index] after GPT2-like validation."""

    validate_gpt2_like_model(model)
    if layer_index < 0 or layer_index >= len(model.transformer.h):
        raise ValueError(
            f"layer_index must be in [0, {len(model.transformer.h)}), got {layer_index}."
        )
    return model.transformer.h[layer_index]


def infer_gpt2_num_heads(layer: Any, hidden_dim: int) -> int:
    num_heads = getattr(layer.attn, "num_heads", None)
    if num_heads is None:
        raise ValueError("Expected GPT2 attention module to expose num_heads.")
    if hidden_dim % int(num_heads) != 0:
        raise ValueError(
            f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}."
        )
    return int(num_heads)


def split_gpt2_heads(tensor: Any, num_heads: int) -> Any:
    batch_size, seq_len, hidden_dim = tensor.shape
    if hidden_dim % num_heads != 0:
        raise ValueError(
            f"hidden_dim must be divisible by num_heads, got {hidden_dim} and {num_heads}."
        )
    head_dim = hidden_dim // num_heads
    return tensor.view(batch_size, seq_len, num_heads, head_dim).permute(0, 2, 1, 3).contiguous()


def compute_gpt2_layer_qkv(layer: Any, hidden_states: Any) -> tuple[Any, Any, Any]:
    """Compute GPT2 attention Q/K/V from layer input hidden states.

    This is GPT2-specific: GPT2Block applies ln_1 before attn.c_attn, and GPT2 cache
    K/V tensors use shape [batch, heads, seq_len, head_dim].
    """

    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq_len, hidden_dim], got {hidden_states.shape}."
        )
    if not hasattr(layer, "ln_1") or not hasattr(layer, "attn") or not hasattr(layer.attn, "c_attn"):
        raise ValueError("Expected a GPT2-like layer with ln_1 and attn.c_attn.")

    normalized_hidden_states = layer.ln_1(hidden_states)
    qkv = layer.attn.c_attn(normalized_hidden_states)
    if qkv.shape[-1] % 3 != 0:
        raise ValueError(f"attn.c_attn output last dimension must be divisible by 3, got {qkv.shape}.")

    split_size = qkv.shape[-1] // 3
    query, key, value = qkv.split(split_size, dim=-1)
    num_heads = infer_gpt2_num_heads(layer, split_size)
    return (
        split_gpt2_heads(query, num_heads),
        split_gpt2_heads(key, num_heads),
        split_gpt2_heads(value, num_heads),
    )


def validate_selected_indices(selected_indices: list[int], seq_len: int) -> None:
    if not selected_indices:
        raise ValueError("selected_indices must be non-empty.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError("selected_indices must not contain duplicates.")
    invalid = [index for index in selected_indices if index < 0 or index >= seq_len]
    if invalid:
        raise ValueError(
            f"selected_indices must be in [0, {seq_len}); invalid indices: {invalid}."
        )


def validate_selected_kv_shape(reference: Any, selected: Any, selected_count: int, name: str) -> None:
    reference_shape = tuple(reference.shape)
    selected_shape = tuple(selected.shape)
    if len(reference_shape) != len(selected_shape):
        raise ValueError(
            f"{name} rank must match original KV rank; got {selected_shape} and {reference_shape}."
        )
    comparable_reference = reference_shape[:-2] + reference_shape[-1:]
    comparable_selected = selected_shape[:-2] + selected_shape[-1:]
    if comparable_reference != comparable_selected:
        raise ValueError(
            f"{name} shape must match original KV except seq_len; "
            f"got {selected_shape} and {reference_shape}."
        )
    if int(selected.shape[-2]) != selected_count:
        raise ValueError(
            f"{name} seq_len must equal len(selected_indices), got {selected.shape[-2]} "
            f"and {selected_count}."
        )
    if selected.device != reference.device:
        raise ValueError(f"{name} device must match original KV device.")


def patch_selected_kv(
    layer_kv: tuple[Any, Any],
    selected_indices: list[int],
    selected_key: Any,
    selected_value: Any,
) -> tuple[Any, Any]:
    """Clone a layer KV cache and replace selected token positions."""

    import torch

    if len(layer_kv) < 2:
        raise ValueError("layer_kv must contain key and value tensors.")

    key, value = layer_kv[0], layer_kv[1]
    if key.shape != value.shape:
        raise ValueError(f"key/value shapes must match, got {key.shape} and {value.shape}.")

    seq_dim = key.ndim - 2
    seq_len = int(key.shape[seq_dim])
    validate_selected_indices(selected_indices, seq_len)
    validate_selected_kv_shape(key, selected_key, len(selected_indices), "selected_key")
    validate_selected_kv_shape(value, selected_value, len(selected_indices), "selected_value")

    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=key.device)
    patched_key = key.clone()
    patched_value = value.clone()
    patched_key.index_copy_(seq_dim, index_tensor, selected_key)
    patched_value.index_copy_(seq_dim, index_tensor, selected_value)
    return patched_key, patched_value


def recompute_and_patch_gpt2_layer_kv(
    model: Any,
    layer_index: int,
    hidden_states: Any,
    layer_kv: tuple[Any, Any],
    selected_indices: list[int],
) -> tuple[Any, Any]:
    """Recompute selected GPT2 layer K/V states and patch them into a reused layer KV."""

    import torch

    layer = get_gpt2_layer(model, layer_index)
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq_len, hidden_dim], got {hidden_states.shape}."
        )
    validate_selected_indices(selected_indices, int(hidden_states.shape[1]))

    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=hidden_states.device)
    selected_hidden_states = hidden_states.index_select(dim=1, index=index_tensor)
    _, selected_key, selected_value = compute_gpt2_layer_qkv(layer, selected_hidden_states)
    return patch_selected_kv(
        layer_kv=layer_kv,
        selected_indices=selected_indices,
        selected_key=selected_key,
        selected_value=selected_value,
    )
