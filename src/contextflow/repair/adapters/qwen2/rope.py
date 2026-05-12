from __future__ import annotations

from typing import Any

from contextflow.kv_cache.types import ChunkKV
from contextflow.repair.adapters.qwen2.adapter import (
    get_qwen2_decoder,
    get_qwen2_layer,
    validate_qwen2_like_model,
)
from contextflow.repair.adapters.rope import (
    correct_rope_key_positions,
    select_rotary_positions,
)


def compute_qwen2_rotary_embeddings_for_positions(
    model: Any,
    layer: Any,
    reference_tensor: Any,
    positions: list[int],
) -> tuple[Any, Any]:
    import torch

    if not positions:
        raise ValueError("positions must be non-empty.")

    decoder = get_qwen2_decoder(model)
    rotary_emb = getattr(decoder, "rotary_emb", None)
    if rotary_emb is None:
        rotary_emb = getattr(layer.self_attn, "rotary_emb", None)
    if rotary_emb is None:
        raise ValueError("Expected Qwen2 model or attention module to expose rotary_emb.")

    position_ids = torch.tensor(
        [positions],
        dtype=torch.long,
        device=reference_tensor.device,
    )
    try:
        cos, sin = rotary_emb(reference_tensor, position_ids)
    except TypeError:
        try:
            cos, sin = rotary_emb(reference_tensor, position_ids=position_ids)
        except TypeError:
            cos, sin = rotary_emb(reference_tensor, seq_len=max(positions) + 1)

    cos = select_rotary_positions(cos, positions, selected_count=len(positions))
    sin = select_rotary_positions(sin, positions, selected_count=len(positions))
    return cos.to(dtype=reference_tensor.dtype), sin.to(dtype=reference_tensor.dtype)


def correct_qwen2_chunk_kv_rope_positions(
    model: Any,
    chunk_kvs: list[ChunkKV],
) -> list[ChunkKV]:
    """Correct chunk-local Qwen2 RoPE keys to full-context absolute positions."""

    validate_qwen2_like_model(model)
    if not chunk_kvs:
        return []

    corrected_chunks: list[ChunkKV] = []
    absolute_offset = 0
    for chunk_kv in chunk_kvs:
        local_positions = list(range(chunk_kv.num_tokens))
        absolute_positions = list(range(absolute_offset, absolute_offset + chunk_kv.num_tokens))
        corrected_layers = []

        for layer_index, layer_kv in enumerate(chunk_kv.past_key_values):
            key, value = layer_kv[0], layer_kv[1]
            if absolute_positions == local_positions:
                corrected_key = key
            else:
                layer = get_qwen2_layer(model, layer_index)
                local_cos, local_sin = compute_qwen2_rotary_embeddings_for_positions(
                    model=model,
                    layer=layer,
                    reference_tensor=key,
                    positions=local_positions,
                )
                absolute_cos, absolute_sin = compute_qwen2_rotary_embeddings_for_positions(
                    model=model,
                    layer=layer,
                    reference_tensor=key,
                    positions=absolute_positions,
                )
                corrected_key = correct_rope_key_positions(
                    key=key,
                    local_cos=local_cos,
                    local_sin=local_sin,
                    absolute_cos=absolute_cos,
                    absolute_sin=absolute_sin,
                )
            corrected_layers.append((corrected_key, value))

        corrected_chunks.append(
            ChunkKV(
                chunk_id=chunk_kv.chunk_id,
                input_ids=list(chunk_kv.input_ids),
                past_key_values=tuple(corrected_layers),
                num_tokens=chunk_kv.num_tokens,
                num_layers=chunk_kv.num_layers,
                device=chunk_kv.device,
            )
        )
        absolute_offset += chunk_kv.num_tokens

    return corrected_chunks
