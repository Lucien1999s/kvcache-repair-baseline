from __future__ import annotations

from typing import Any

from contextflow.kv_cache.types import ChunkKV


def kv_sequence_length(layer_kv: tuple[Any, Any]) -> int:
    key, _ = layer_kv
    return int(key.shape[-2])


def validate_layer_shapes(reference: Any, candidate: Any) -> None:
    reference_shape = tuple(reference.shape)
    candidate_shape = tuple(candidate.shape)
    if len(reference_shape) != len(candidate_shape):
        raise ValueError(
            f"KV tensor rank mismatch: expected {reference_shape}, got {candidate_shape}."
        )
    comparable_reference = reference_shape[:-2] + reference_shape[-1:]
    comparable_candidate = candidate_shape[:-2] + candidate_shape[-1:]
    if comparable_reference != comparable_candidate:
        raise ValueError(
            "KV tensor shapes must match except sequence length dimension: "
            f"expected {reference_shape}, got {candidate_shape}."
        )


def assemble_chunk_kvs(chunk_kvs: list[ChunkKV]) -> Any:
    """Layer-wise concatenate precomputed chunk KVs along the sequence dimension."""

    import torch

    if not chunk_kvs:
        raise ValueError("chunk_kvs must be non-empty.")

    num_layers = chunk_kvs[0].num_layers
    expected_total_len = sum(chunk_kv.num_tokens for chunk_kv in chunk_kvs)
    for chunk_kv in chunk_kvs:
        if chunk_kv.num_layers != num_layers:
            raise ValueError(
                f"All chunks must have the same num_layers: expected {num_layers}, "
                f"got {chunk_kv.num_layers} for {chunk_kv.chunk_id}."
            )

    assembled_layers = []
    for layer_index in range(num_layers):
        layer_keys = []
        layer_values = []
        reference_key = chunk_kvs[0].past_key_values[layer_index][0]
        reference_value = chunk_kvs[0].past_key_values[layer_index][1]

        for chunk_kv in chunk_kvs:
            key, value = chunk_kv.past_key_values[layer_index]
            validate_layer_shapes(reference_key, key)
            validate_layer_shapes(reference_value, value)
            layer_keys.append(key)
            layer_values.append(value)

        assembled_key = torch.cat(layer_keys, dim=-2)
        assembled_value = torch.cat(layer_values, dim=-2)
        if int(assembled_key.shape[-2]) != expected_total_len:
            raise ValueError(
                f"Assembled key seq_len mismatch at layer {layer_index}: "
                f"expected {expected_total_len}, got {assembled_key.shape[-2]}."
            )
        if int(assembled_value.shape[-2]) != expected_total_len:
            raise ValueError(
                f"Assembled value seq_len mismatch at layer {layer_index}: "
                f"expected {expected_total_len}, got {assembled_value.shape[-2]}."
            )
        assembled_layers.append((assembled_key, assembled_value))

    return tuple(assembled_layers)
