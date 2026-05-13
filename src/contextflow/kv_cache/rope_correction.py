from __future__ import annotations

from typing import Any

from contextflow.kv_cache.types import ChunkKV


ROPE_CORRECTION_MODEL_FAMILIES = {"mistral", "qwen2"}


def rope_position_correction_enabled(model_family: str) -> bool:
    return model_family.lower() in ROPE_CORRECTION_MODEL_FAMILIES


def correct_doc_chunk_kvs_for_model_family(
    model: Any,
    chunk_kvs: list[ChunkKV],
    model_family: str,
) -> list[ChunkKV]:
    """Correct reusable document chunk KVs into full-context positional space.

    GPT2 has learned absolute position embeddings before K/V projection, so there
    is no RoPE key correction to apply here. RoPE families precompute chunks at
    chunk-local positions and need their keys rotated into concatenated document
    absolute positions before assembly.
    """

    normalized_model_family = model_family.lower()
    if normalized_model_family == "gpt2":
        return chunk_kvs

    if normalized_model_family == "mistral":
        from contextflow.repair.adapters.mistral.rope import (
            correct_mistral_chunk_kv_rope_positions,
        )

        return correct_mistral_chunk_kv_rope_positions(model, chunk_kvs)

    if normalized_model_family == "qwen2":
        from contextflow.repair.adapters.qwen2.rope import (
            correct_qwen2_chunk_kv_rope_positions,
        )

        return correct_qwen2_chunk_kv_rope_positions(model, chunk_kvs)

    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. RoPE position correction currently "
        "supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )


def correct_chunk_kv_rope_source_positions_for_model_family(
    model: Any,
    chunk_kv: ChunkKV,
    model_family: str,
    source_positions: list[int],
    target_positions: list[int],
) -> ChunkKV:
    """Correct one chunk KV from its precompute source positions into target positions.

    This is an opt-in FusionRAG-style path. The existing CacheBlend/naive reuse
    full-document correction remains unchanged.
    """

    if len(source_positions) != chunk_kv.num_tokens:
        raise ValueError(
            "source_positions length must match chunk token count; "
            f"got {len(source_positions)} and {chunk_kv.num_tokens}."
        )
    if len(target_positions) != chunk_kv.num_tokens:
        raise ValueError(
            "target_positions length must match chunk token count; "
            f"got {len(target_positions)} and {chunk_kv.num_tokens}."
        )

    normalized_model_family = model_family.lower()
    if normalized_model_family == "gpt2":
        return chunk_kv

    if normalized_model_family == "mistral":
        from contextflow.repair.adapters.mistral.rope import (
            correct_mistral_chunk_kv_rope_source_positions,
        )

        return correct_mistral_chunk_kv_rope_source_positions(
            model=model,
            chunk_kv=chunk_kv,
            source_positions=source_positions,
            target_positions=target_positions,
        )

    if normalized_model_family == "qwen2":
        from contextflow.repair.adapters.qwen2.rope import (
            correct_qwen2_chunk_kv_rope_source_positions,
        )

        return correct_qwen2_chunk_kv_rope_source_positions(
            model=model,
            chunk_kv=chunk_kv,
            source_positions=source_positions,
            target_positions=target_positions,
        )

    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. RoPE source-position correction "
        "currently supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )
