from __future__ import annotations

from typing import Any


def validate_mistral_like_model(model: Any) -> None:
    """Validate the minimal MistralForCausalLM structure needed by this adapter."""

    decoder = get_mistral_decoder(model)
    if not hasattr(decoder, "layers"):
        raise ValueError("Expected a Mistral-like model with model.model.layers.")
    if len(decoder.layers) == 0:
        raise ValueError("Expected model.model.layers to contain at least one layer.")

    layer = decoder.layers[0]
    if not hasattr(layer, "self_attn"):
        raise ValueError("Expected Mistral-like layer to have self_attn.")
    for projection_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        if not hasattr(layer.self_attn, projection_name):
            raise ValueError(f"Expected Mistral attention to expose {projection_name}.")
    if not hasattr(layer, "input_layernorm"):
        raise ValueError("Expected Mistral-like layer to have input_layernorm.")
    if not hasattr(layer, "post_attention_layernorm"):
        raise ValueError("Expected Mistral-like layer to have post_attention_layernorm.")
    if not hasattr(layer, "mlp"):
        raise ValueError("Expected Mistral-like layer to have mlp.")


def get_mistral_decoder(model: Any) -> Any:
    decoder = getattr(model, "model", None)
    if decoder is None:
        raise ValueError("Expected a Mistral-like model with a .model decoder module.")
    return decoder


def get_mistral_layer(model: Any, layer_index: int) -> Any:
    validate_mistral_like_model(model)
    layers = get_mistral_decoder(model).layers
    if layer_index < 0 or layer_index >= len(layers):
        raise ValueError(f"layer_index must be in [0, {len(layers)}), got {layer_index}.")
    return layers[layer_index]


def compute_mistral_selected_initial_hidden_states(
    model: Any,
    input_ids: Any,
    selected_indices: list[int],
) -> Any:
    """Compute Mistral layer-0 input hidden states for selected absolute positions."""

    import torch

    validate_mistral_like_model(model)
    if input_ids.ndim != 2:
        raise ValueError(f"input_ids must have shape [batch, seq_len], got {input_ids.shape}.")
    if int(input_ids.shape[0]) != 1:
        raise ValueError("Only batch size 1 is currently supported for selected repair.")

    seq_len = int(input_ids.shape[1])
    validate_selected_indices(selected_indices, seq_len)
    decoder = get_mistral_decoder(model)
    if not hasattr(decoder, "embed_tokens"):
        raise ValueError("Expected Mistral-like decoder to expose embed_tokens.")

    input_ids = input_ids.to(decoder.embed_tokens.weight.device)
    hidden_states = decoder.embed_tokens(input_ids)
    index_tensor = torch.tensor(selected_indices, dtype=torch.long, device=input_ids.device)
    return hidden_states.index_select(dim=1, index=index_tensor)


def infer_mistral_attention_geometry(layer: Any) -> tuple[int, int, int, int]:
    attn = layer.self_attn
    config = getattr(attn, "config", None)
    num_heads = getattr(attn, "num_heads", None)
    if num_heads is None and config is not None:
        num_heads = getattr(config, "num_attention_heads", None)

    num_key_value_heads = getattr(attn, "num_key_value_heads", None)
    if num_key_value_heads is None and config is not None:
        num_key_value_heads = getattr(config, "num_key_value_heads", None)

    head_dim = getattr(attn, "head_dim", None)
    if head_dim is None and config is not None:
        head_dim = getattr(config, "head_dim", None)

    if num_heads is None:
        raise ValueError(
            "Expected Mistral attention module/config to expose num_heads or "
            "num_attention_heads."
        )
    if num_key_value_heads is None:
        raise ValueError(
            "Expected Mistral attention module/config to expose num_key_value_heads."
        )
    if head_dim is None:
        hidden_size = getattr(attn, "hidden_size", None)
        if hidden_size is None and config is not None:
            hidden_size = getattr(config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Expected Mistral attention module/config to expose head_dim or hidden_size."
            )
        head_dim = int(hidden_size) // int(num_heads)

    num_heads = int(num_heads)
    num_key_value_heads = int(num_key_value_heads)
    head_dim = int(head_dim)
    if num_heads % num_key_value_heads != 0:
        raise ValueError(
            "num_heads must be divisible by num_key_value_heads for Mistral GQA; "
            f"got {num_heads} and {num_key_value_heads}."
        )
    return num_heads, num_key_value_heads, num_heads // num_key_value_heads, head_dim


def split_mistral_qkv_heads(
    layer: Any,
    query: Any,
    key: Any,
    value: Any,
) -> tuple[Any, Any, Any]:
    num_heads, num_key_value_heads, _, head_dim = infer_mistral_attention_geometry(layer)
    batch_size, seq_len, _ = query.shape
    query = query.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).contiguous()
    key = key.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()
    value = value.view(batch_size, seq_len, num_key_value_heads, head_dim).transpose(1, 2).contiguous()
    return query, key, value


def compute_mistral_layer_qkv(layer: Any, hidden_states: Any) -> tuple[Any, Any, Any]:
    if hidden_states.ndim != 3:
        raise ValueError(
            f"hidden_states must have shape [batch, seq_len, hidden_dim], got {hidden_states.shape}."
        )

    normalized_hidden_states = layer.input_layernorm(hidden_states)
    query = layer.self_attn.q_proj(normalized_hidden_states)
    key = layer.self_attn.k_proj(normalized_hidden_states)
    value = layer.self_attn.v_proj(normalized_hidden_states)
    return split_mistral_qkv_heads(layer, query, key, value)


def compute_mistral_layer_query(layer: Any, hidden_states: Any) -> Any:
    """Compute Mistral attention query states before RoPE application."""

    query, _, _ = compute_mistral_layer_qkv(layer, hidden_states)
    return query


def repeat_mistral_kv(hidden_states: Any, num_key_value_groups: int) -> Any:
    """Repeat KV heads for grouped-query attention."""

    if num_key_value_groups == 1:
        return hidden_states
    batch_size, num_key_value_heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size,
        num_key_value_heads,
        num_key_value_groups,
        seq_len,
        head_dim,
    )
    return hidden_states.reshape(
        batch_size,
        num_key_value_heads * num_key_value_groups,
        seq_len,
        head_dim,
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
    """Clone a Mistral layer KV cache and replace selected token positions."""

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
