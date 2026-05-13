from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache.assembly import assemble_chunk_kvs
from contextflow.kv_cache.precompute import (
    infer_model_input_device,
    infer_past_key_values_device,
    normalize_past_key_values,
    precompute_doc_chunk_kvs,
    precompute_kv_for_input_ids,
)
from contextflow.kv_cache.rope_correction import (
    correct_chunk_kv_rope_source_positions_for_model_family,
    correct_doc_chunk_kvs_for_model_family,
    rope_position_correction_enabled,
)
from contextflow.kv_cache.types import ChunkKV
from contextflow.retrieval.similarity import (
    ChunkNeighborPlan,
    ChunkSimilarityFn,
    build_chunk_neighbor_plan,
)
from contextflow.runtime.hf_cached_generation import (
    convert_legacy_cache_for_model,
    forward_with_optional_cache_position,
)


@dataclass(slots=True)
class EnrichedDocKVPrecomputeResult:
    """FusionRAG-style SGP precompute output for one tokenized QA example."""

    enriched_chunk_kvs: list[ChunkKV]
    plain_chunk_kvs: list[ChunkKV]
    neighbor_plans: list[ChunkNeighborPlan]
    metadata: dict[str, Any]


def doc_chunk_token_offsets(doc_chunk_ids: list[list[int]]) -> list[int]:
    offsets: list[int] = []
    current_offset = 0
    for chunk_ids in doc_chunk_ids:
        offsets.append(current_offset)
        current_offset += len(chunk_ids)
    return offsets


def validate_prefix_layer_count(prefix_past_key_values: Any, expected_layers: int) -> None:
    normalized = normalize_past_key_values(prefix_past_key_values)
    if len(normalized) != expected_layers:
        raise ValueError(
            "prefix_past_key_values layer count must match expected_layers; "
            f"got {len(normalized)} and {expected_layers}."
        )


def slice_target_kv_from_prefixed_outputs(
    past_key_values: Any,
    prefix_len: int,
    target_len: int,
) -> tuple[tuple[Any, Any], ...]:
    normalized_past_key_values = normalize_past_key_values(past_key_values)
    sliced_layers = []
    for layer_index, layer_kv in enumerate(normalized_past_key_values):
        key, value = layer_kv[0], layer_kv[1]
        seq_len = int(key.shape[-2])
        if seq_len >= prefix_len + target_len:
            target_start = prefix_len
            target_end = prefix_len + target_len
        elif seq_len == target_len:
            target_start = 0
            target_end = target_len
        else:
            raise ValueError(
                f"Layer {layer_index} KV seq_len cannot be sliced into target KV; "
                f"seq_len={seq_len}, prefix_len={prefix_len}, target_len={target_len}."
            )
        sliced_layers.append(
            (
                key[..., target_start:target_end, :],
                value[..., target_start:target_end, :],
            )
        )
    return tuple(sliced_layers)


def precompute_target_chunk_kv_with_prefix(
    model: Any,
    target_input_ids: list[int],
    prefix_past_key_values: Any | None,
    prefix_len: int,
    chunk_id: str,
) -> ChunkKV:
    """Precompute target chunk KV while allowing it to attend to a prefix cache."""

    if not target_input_ids:
        raise ValueError("target_input_ids must be non-empty.")
    if prefix_len < 0:
        raise ValueError("prefix_len must be non-negative.")
    if prefix_past_key_values is None:
        if prefix_len != 0:
            raise ValueError("prefix_len must be 0 when prefix_past_key_values is None.")
        return precompute_kv_for_input_ids(
            model=model,
            input_ids=target_input_ids,
            chunk_id=chunk_id,
        )

    import torch

    device = infer_model_input_device(model)
    prefix_cache = convert_legacy_cache_for_model(model, prefix_past_key_values)
    input_tensor = torch.tensor([target_input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones(
        (1, prefix_len + len(target_input_ids)),
        dtype=torch.long,
        device=device,
    )
    cache_position = torch.arange(
        prefix_len,
        prefix_len + len(target_input_ids),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        outputs = forward_with_optional_cache_position(
            model,
            input_ids=input_tensor,
            attention_mask=attention_mask,
            past_key_values=prefix_cache,
            cache_position=cache_position,
            use_cache=True,
        )

    if outputs.past_key_values is None:
        raise RuntimeError(
            "Model did not return past_key_values for enriched chunk precompute."
        )
    target_past_key_values = slice_target_kv_from_prefixed_outputs(
        outputs.past_key_values,
        prefix_len=prefix_len,
        target_len=len(target_input_ids),
    )
    return ChunkKV(
        chunk_id=chunk_id,
        input_ids=list(target_input_ids),
        past_key_values=target_past_key_values,
        num_tokens=len(target_input_ids),
        num_layers=len(target_past_key_values),
        device=infer_past_key_values_device(target_past_key_values),
    )


def build_enriched_chunk_metadata(
    plan: ChunkNeighborPlan,
    source_positions: list[int],
    target_absolute_positions: list[int],
    neighbor_token_count: int,
    rope_source_position_correction_enabled: bool,
    rope_source_position_correction_applied: bool,
) -> dict[str, Any]:
    return {
        "fusionrag_enriched": True,
        "target_chunk_index": plan.target_chunk_index,
        "neighbor_indices": list(plan.neighbor_indices),
        "neighbor_scores": list(plan.neighbor_scores),
        "neighbor_token_count": neighbor_token_count,
        "source_positions": list(source_positions),
        "target_absolute_positions": list(target_absolute_positions),
        "source_position_basis": "neighbor_prefix_absolute",
        "target_position_basis": "document_absolute",
        "rope_source_position_correction_enabled": rope_source_position_correction_enabled,
        "rope_source_position_correction_applied": rope_source_position_correction_applied,
    }


def precompute_enriched_doc_chunk_kvs(
    model: Any,
    example: TokenizedExample,
    neighbor_top_n: int = 5,
    similarity_fn: ChunkSimilarityFn | None = None,
    model_family: str = "gpt2",
    apply_rope_source_position_correction: bool = True,
    chunk_id_prefix: str | None = None,
) -> EnrichedDocKVPrecomputeResult:
    """Precompute FusionRAG-style neighbor-enriched document chunk KV caches.

    This implements the offline SGP substrate only. It does not run query-guided
    selection, repair, or decode.
    """

    if neighbor_top_n < 0:
        raise ValueError("neighbor_top_n must be non-negative.")
    if not example.doc_chunk_ids:
        raise ValueError("example.doc_chunk_ids must contain at least one chunk.")

    model_family = model_family.lower()
    prefix = chunk_id_prefix or example.example_id or "example"
    plain_chunk_kvs = precompute_doc_chunk_kvs(
        model=model,
        example=example,
        chunk_id_prefix=prefix,
    )
    neighbor_plans = build_chunk_neighbor_plan(
        example.doc_chunk_ids,
        neighbor_top_n=neighbor_top_n,
        similarity_fn=similarity_fn,
    )
    chunk_offsets = doc_chunk_token_offsets(example.doc_chunk_ids)
    rope_correction_enabled = (
        apply_rope_source_position_correction
        and rope_position_correction_enabled(model_family)
    )

    enriched_chunk_kvs: list[ChunkKV] = []
    for plan in neighbor_plans:
        target_index = plan.target_chunk_index
        target_input_ids = example.doc_chunk_ids[target_index]
        neighbor_chunk_kvs = [plain_chunk_kvs[index] for index in plan.neighbor_indices]
        neighbor_token_count = sum(chunk_kv.num_tokens for chunk_kv in neighbor_chunk_kvs)
        if neighbor_chunk_kvs:
            prefix_chunk_kvs = correct_doc_chunk_kvs_for_model_family(
                model=model,
                chunk_kvs=neighbor_chunk_kvs,
                model_family=model_family,
            )
            prefix_past_key_values = assemble_chunk_kvs(prefix_chunk_kvs)
            validate_prefix_layer_count(
                prefix_past_key_values,
                expected_layers=plain_chunk_kvs[target_index].num_layers,
            )
        else:
            prefix_past_key_values = None

        source_positions = list(
            range(neighbor_token_count, neighbor_token_count + len(target_input_ids))
        )
        target_absolute_positions = list(
            range(
                chunk_offsets[target_index],
                chunk_offsets[target_index] + len(target_input_ids),
            )
        )
        target_chunk_kv = precompute_target_chunk_kv_with_prefix(
            model=model,
            target_input_ids=target_input_ids,
            prefix_past_key_values=prefix_past_key_values,
            prefix_len=neighbor_token_count,
            chunk_id=f"{prefix}:fusionrag_enriched_doc:{target_index}",
        )
        if target_chunk_kv.num_layers != plain_chunk_kvs[target_index].num_layers:
            raise ValueError(
                "Enriched target KV layer count must match plain chunk KV layer count; "
                f"got {target_chunk_kv.num_layers} and {plain_chunk_kvs[target_index].num_layers}."
            )

        if rope_correction_enabled:
            target_chunk_kv = correct_chunk_kv_rope_source_positions_for_model_family(
                model=model,
                chunk_kv=target_chunk_kv,
                model_family=model_family,
                source_positions=source_positions,
                target_positions=target_absolute_positions,
            )

        target_chunk_kv.metadata.update(
            build_enriched_chunk_metadata(
                plan=plan,
                source_positions=source_positions,
                target_absolute_positions=target_absolute_positions,
                neighbor_token_count=neighbor_token_count,
                rope_source_position_correction_enabled=apply_rope_source_position_correction,
                rope_source_position_correction_applied=rope_correction_enabled,
            )
        )
        enriched_chunk_kvs.append(target_chunk_kv)

    return EnrichedDocKVPrecomputeResult(
        enriched_chunk_kvs=enriched_chunk_kvs,
        plain_chunk_kvs=plain_chunk_kvs,
        neighbor_plans=neighbor_plans,
        metadata={
            "example_id": example.example_id,
            "model_family": model_family,
            "neighbor_top_n": neighbor_top_n,
            "similarity_backend": (
                "custom" if similarity_fn is not None else "token_id_cosine"
            ),
            "apply_rope_source_position_correction": apply_rope_source_position_correction,
            "rope_source_position_correction_applied": rope_correction_enabled,
            "num_doc_chunks": len(example.doc_chunk_ids),
            "doc_token_count": sum(len(chunk_ids) for chunk_ids in example.doc_chunk_ids),
        },
    )
