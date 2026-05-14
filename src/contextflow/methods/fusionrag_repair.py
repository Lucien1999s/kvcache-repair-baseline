from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache import assemble_chunk_kvs, precompute_enriched_doc_chunk_kvs
from contextflow.kv_cache.rope_correction import rope_position_correction_enabled
from contextflow.methods.cacheblend_repair import (
    CacheBlendRepairPlan,
    build_cacheblend_repair_plan,
    run_cacheblend_style_partial_repair_from_plan,
)
from contextflow.retrieval.similarity import ChunkSimilarityFn
from contextflow.repair.fusionrag_selector import (
    FusionRAGQuerySelectionResult,
    select_fusionrag_query_guided_tokens,
)
from contextflow.runtime.hf_cached_generation import (
    HFCachedGenerationResult,
    generate_with_past_key_values,
)


FUSIONRAG_REPAIR_STRATEGY = "fusionrag_query_guided_fixed"
FUSIONRAG_RUNTIME_SELECTION_MODE = "fusionrag_query_guided_fixed_selected_indices"


@dataclass(slots=True)
class FusionRAGRepairResult:
    generated_ids: list[int]
    output_text: str
    generation: HFCachedGenerationResult
    repaired_past_key_values: Any
    enriched_past_key_values: Any
    repair_plan: CacheBlendRepairPlan
    query_selection: FusionRAGQuerySelectionResult
    metadata: dict[str, Any]


def build_fusionrag_repair_plan(
    query_selection: FusionRAGQuerySelectionResult,
    doc_total_len: int,
    num_layers: int,
    metadata: dict[str, Any],
) -> CacheBlendRepairPlan:
    """Build the fixed selected-token repair plan used by FusionRAG QGS."""

    if not query_selection.selected_indices:
        raise ValueError("FusionRAG query selection must select at least one token.")
    selected_indices_by_layer = {
        layer_index: list(query_selection.selected_indices)
        for layer_index in range(num_layers)
    }
    return build_cacheblend_repair_plan(
        selected_indices_by_layer=selected_indices_by_layer,
        doc_total_len=doc_total_len,
        num_layers=num_layers,
        metadata=metadata,
        strategy=FUSIONRAG_REPAIR_STRATEGY,
        runtime_selection_mode=FUSIONRAG_RUNTIME_SELECTION_MODE,
    )


def validate_fusionrag_rope_correction_setting(
    model_family: str,
    apply_rope_source_position_correction: bool,
) -> None:
    if rope_position_correction_enabled(model_family) and not apply_rope_source_position_correction:
        raise ValueError(
            "FusionRAG repair on RoPE model families requires source-position correction. "
            "Use apply_rope_source_position_correction=True for measured repair; disabling "
            "is only intended for isolated enriched-precompute diagnostics."
        )


def run_fusionrag_style_repair_generation(
    model: Any,
    tokenizer: Any,
    tokenized_example: TokenizedExample,
    max_new_tokens: int = 16,
    neighbor_top_n: int = 5,
    recompute_ratio: float = 0.15,
    model_family: str = "gpt2",
    similarity_fn: ChunkSimilarityFn | None = None,
    apply_rope_source_position_correction: bool = True,
) -> FusionRAGRepairResult:
    """Run FusionRAG-style neighbor-enriched repair and greedy generation.

    This is the method-level baseline path: SGP enriched KV precompute, QGS token
    selection, selected-token partial repair, then decode from repaired KV.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if neighbor_top_n < 0:
        raise ValueError("neighbor_top_n must be non-negative.")
    model_family = model_family.lower()
    validate_fusionrag_rope_correction_setting(
        model_family=model_family,
        apply_rope_source_position_correction=apply_rope_source_position_correction,
    )

    total_start = time.perf_counter()

    enriched_start = time.perf_counter()
    enriched_result = precompute_enriched_doc_chunk_kvs(
        model=model,
        example=tokenized_example,
        neighbor_top_n=neighbor_top_n,
        similarity_fn=similarity_fn,
        model_family=model_family,
        apply_rope_source_position_correction=apply_rope_source_position_correction,
    )
    enriched_past_key_values = assemble_chunk_kvs(enriched_result.enriched_chunk_kvs)
    enriched_precompute_latency_seconds = time.perf_counter() - enriched_start

    doc_chunk_lengths = [len(doc_ids) for doc_ids in tokenized_example.doc_chunk_ids]
    doc_total_len = sum(doc_chunk_lengths)
    num_layers = len(enriched_past_key_values)
    if num_layers <= 0:
        raise ValueError("enriched_past_key_values must contain at least one layer.")

    selection_start = time.perf_counter()
    query_selection = select_fusionrag_query_guided_tokens(
        model=model,
        q_ids=tokenized_example.q_ids,
        doc_past_key_values=enriched_past_key_values,
        doc_chunk_lengths=doc_chunk_lengths,
        recompute_ratio=recompute_ratio,
        model_family=model_family,
    )
    selection_latency_seconds = time.perf_counter() - selection_start

    repair_plan_metadata = {
        "uses_full_recompute_reference": False,
        "selection_algorithm": "fusionrag_query_guided_selection",
        "neighbor_top_n": neighbor_top_n,
        "recompute_ratio": recompute_ratio,
        "selected_count": query_selection.recompute_count,
        "enriched_precompute_latency_seconds": enriched_precompute_latency_seconds,
        "selection_latency_seconds": selection_latency_seconds,
        "enriched_precompute_metadata": dict(enriched_result.metadata),
        "query_selection_metadata": query_selection.to_metadata(),
        "rope_position_correction_applied": enriched_result.metadata.get(
            "rope_source_position_correction_applied",
            False,
        ),
        "reuse_doc_kv_position_basis": (
            "fusionrag_enriched_full_context_absolute"
            if enriched_result.metadata.get("rope_source_position_correction_applied", False)
            else "fusionrag_enriched_native"
        ),
    }
    repair_plan = build_fusionrag_repair_plan(
        query_selection=query_selection,
        doc_total_len=doc_total_len,
        num_layers=num_layers,
        metadata=repair_plan_metadata,
    )

    repair_start = time.perf_counter()
    partial_repair = run_cacheblend_style_partial_repair_from_plan(
        model=model,
        tokenized_example=tokenized_example,
        repair_plan=repair_plan,
        model_family=model_family,
        reuse_past_key_values=enriched_past_key_values,
    )
    repair_latency_seconds = time.perf_counter() - repair_start

    decode_start = time.perf_counter()
    generation = generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=partial_repair.repaired_past_key_values,
        max_new_tokens=max_new_tokens,
    )
    decode_latency_seconds = time.perf_counter() - decode_start
    total_latency_seconds = time.perf_counter() - total_start

    metadata = {
        "method": "fusionrag_repair",
        "model_family": model_family,
        "neighbor_top_n": neighbor_top_n,
        "recompute_ratio": recompute_ratio,
        "selected_count": query_selection.recompute_count,
        "selected_indices": list(query_selection.selected_indices),
        "selected_indices_by_score": list(query_selection.selected_indices_by_score),
        "enriched_precompute_latency_seconds": enriched_precompute_latency_seconds,
        "selection_latency_seconds": selection_latency_seconds,
        "repair_latency_seconds": repair_latency_seconds,
        "decode_latency_seconds": decode_latency_seconds,
        "total_latency_seconds": total_latency_seconds,
        "execution_uses_full_recompute_reference": False,
        "repair_plan_strategy": repair_plan.strategy,
        "runtime_selection_mode": repair_plan.runtime_selection_mode,
        "layer_selected_counts": list(repair_plan.layer_selected_counts),
        "enriched_precompute_metadata": dict(enriched_result.metadata),
        "query_selection_metadata": query_selection.to_metadata(),
        "partial_repair_metadata": dict(partial_repair.metadata),
    }
    return FusionRAGRepairResult(
        generated_ids=generation.generated_ids,
        output_text=generation.output_text,
        generation=generation,
        repaired_past_key_values=partial_repair.repaired_past_key_values,
        enriched_past_key_values=enriched_past_key_values,
        repair_plan=repair_plan,
        query_selection=query_selection,
        metadata=metadata,
    )
