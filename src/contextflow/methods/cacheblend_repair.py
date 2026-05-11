from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from contextflow.data.assembly import assemble_token_aligned_full_prefill_input_ids
from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache import assemble_chunk_kvs, precompute_doc_chunk_kvs
from contextflow.kv_cache.precompute import infer_model_input_device, normalize_past_key_values
from contextflow.repair.cacheblend_selector import (
    compute_kv_deviation,
    select_gradual_hkvd_tokens_by_layer,
)
from contextflow.runtime.hf_cached_generation import (
    HFCachedGenerationResult,
    generate_with_past_key_values,
)


@dataclass(slots=True)
class CacheBlendRepairResult:
    generated_ids: list[int]
    output_text: str
    generation: HFCachedGenerationResult
    repaired_past_key_values: Any
    metadata: dict[str, Any]


def slice_past_key_values_prefix(past_key_values: Any, seq_len: int) -> tuple[tuple[Any, Any], ...]:
    sliced_layers = []
    for layer_index, layer_kv in enumerate(past_key_values):
        key, value = layer_kv[0], layer_kv[1]
        if int(key.shape[-2]) < seq_len or int(value.shape[-2]) < seq_len:
            raise ValueError(
                f"Layer {layer_index} KV seq_len is shorter than requested prefix length {seq_len}."
            )
        sliced_layers.append((key[..., :seq_len, :], value[..., :seq_len, :]))
    return tuple(sliced_layers)


def index_select_sequence(tensor: Any, indices: list[int]) -> Any:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim=tensor.ndim - 2, index=index_tensor)


def max_abs_diff(left: Any, right: Any) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().detach().cpu())


def layer_kv_max_diff_at_indices(
    layer_a: tuple[Any, Any],
    layer_b: tuple[Any, Any],
    indices: list[int],
) -> float:
    key_a, value_a = layer_a[0], layer_a[1]
    key_b, value_b = layer_b[0], layer_b[1]
    key_diff = max_abs_diff(
        index_select_sequence(key_a, indices),
        index_select_sequence(key_b, indices),
    )
    value_diff = max_abs_diff(
        index_select_sequence(value_a, indices),
        index_select_sequence(value_b, indices),
    )
    return max(key_diff, value_diff)


def resolve_unselected_indices(seq_len: int, selected_indices: list[int]) -> list[int]:
    selected = set(selected_indices)
    return [token_index for token_index in range(seq_len) if token_index not in selected]


def compute_repair_diagnostics(
    reuse_past_key_values: tuple[tuple[Any, Any], ...],
    repaired_past_key_values: tuple[tuple[Any, Any], ...],
    full_past_key_values: tuple[tuple[Any, Any], ...],
    runtime_selected_indices: list[int],
    seq_len: int,
) -> dict[str, dict[int, float] | dict[int, bool]]:
    if len(reuse_past_key_values) != len(repaired_past_key_values):
        raise ValueError("reuse and repaired KV must have the same layer count.")
    if len(reuse_past_key_values) != len(full_past_key_values):
        raise ValueError("reuse and full KV must have the same layer count.")

    unselected_indices = resolve_unselected_indices(seq_len, runtime_selected_indices)
    selected_before: dict[int, float] = {}
    selected_after: dict[int, float] = {}
    unselected_after_vs_reuse: dict[int, float] = {}
    shape_matches: dict[int, bool] = {}

    for layer_index, (reuse_layer_kv, repaired_layer_kv, full_layer_kv) in enumerate(
        zip(reuse_past_key_values, repaired_past_key_values, full_past_key_values)
    ):
        reuse_key, reuse_value = reuse_layer_kv[0], reuse_layer_kv[1]
        repaired_key, repaired_value = repaired_layer_kv[0], repaired_layer_kv[1]
        selected_before[layer_index] = layer_kv_max_diff_at_indices(
            reuse_layer_kv,
            full_layer_kv,
            runtime_selected_indices,
        )
        selected_after[layer_index] = layer_kv_max_diff_at_indices(
            repaired_layer_kv,
            full_layer_kv,
            runtime_selected_indices,
        )
        unselected_after_vs_reuse[layer_index] = layer_kv_max_diff_at_indices(
            repaired_layer_kv,
            reuse_layer_kv,
            unselected_indices,
        )
        shape_matches[layer_index] = (
            tuple(repaired_key.shape) == tuple(reuse_key.shape)
            and tuple(repaired_value.shape) == tuple(reuse_value.shape)
        )

    return {
        "selected_kv_max_diff_before_by_layer": selected_before,
        "selected_kv_max_diff_after_by_layer": selected_after,
        "unselected_kv_max_diff_after_vs_reuse_by_layer": unselected_after_vs_reuse,
        "repaired_kv_shape_matches_reuse_by_layer": shape_matches,
    }


def validate_gradual_selection(
    selected_indices_by_layer: dict[int, list[int]],
    num_layers: int,
    seq_len: int,
) -> None:
    if not selected_indices_by_layer:
        raise ValueError("selected_indices_by_layer must be non-empty.")
    if not selected_indices_by_layer[0]:
        raise ValueError("Layer 0 HKVD selection must be non-empty.")

    for layer_index in range(num_layers):
        if layer_index not in selected_indices_by_layer:
            raise ValueError(f"Missing HKVD selection for layer {layer_index}.")
        selected = selected_indices_by_layer[layer_index]
        if not selected:
            raise ValueError(f"Layer {layer_index} HKVD selection must be non-empty.")
        if selected != sorted(selected):
            raise ValueError(f"Layer {layer_index} HKVD selection must be sorted.")
        invalid = [index for index in selected if index < 0 or index >= seq_len]
        if invalid:
            raise ValueError(f"Layer {layer_index} HKVD selection has invalid indices: {invalid}.")
        if layer_index == 0:
            continue
        if not set(selected).issubset(selected_indices_by_layer[layer_index - 1]):
            raise ValueError(f"Layer {layer_index} HKVD selection must be gradual.")


def run_model_family_partial_repair(
    model: Any,
    model_family: str,
    selected_hidden_states: Any,
    selected_indices: list[int],
    reuse_past_key_values: Any,
    num_layers: int,
    attention_mask: Any,
) -> tuple[Any, tuple[tuple[Any, Any], ...]]:
    normalized_model_family = model_family.lower()

    if normalized_model_family == "gpt2":
        from contextflow.repair.adapters.gpt2 import (
            run_gpt2_partial_layers,
            validate_gpt2_like_model,
        )

        validate_gpt2_like_model(model)
        return run_gpt2_partial_layers(
            model=model,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            past_key_values=reuse_past_key_values,
            start_layer_index=0,
            end_layer_index=num_layers,
            attention_mask=attention_mask,
        )

    if normalized_model_family == "mistral":
        from contextflow.repair.adapters.mistral import (
            run_mistral_partial_layers,
            validate_mistral_like_model,
        )

        validate_mistral_like_model(model)
        return run_mistral_partial_layers(
            model=model,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            past_key_values=reuse_past_key_values,
            start_layer_index=0,
            end_layer_index=num_layers,
            attention_mask=attention_mask,
        )

    if normalized_model_family == "qwen2":
        from contextflow.repair.adapters.qwen2 import (
            run_qwen2_partial_layers,
            validate_qwen2_like_model,
        )

        validate_qwen2_like_model(model)
        return run_qwen2_partial_layers(
            model=model,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            past_key_values=reuse_past_key_values,
            start_layer_index=0,
            end_layer_index=num_layers,
            attention_mask=attention_mask,
        )

    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. CacheBlend-style repair currently "
        "supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )


def run_cacheblend_style_repair_generation(
    model: Any,
    tokenizer: Any,
    tokenized_example: TokenizedExample,
    max_new_tokens: int = 16,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
    include_repair_diagnostics: bool = False,
) -> CacheBlendRepairResult:
    """Run CacheBlend-style repaired-KV greedy generation.

    The current reference path uses full recompute KV as a diagnostic repair target.
    Model-specific partial prefill is dispatched by model_family.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if initial_top_k <= 0 or top_k <= 0:
        raise ValueError("initial_top_k and top_k must be positive.")

    model_family = model_family.lower()
    total_start = time.perf_counter()
    repair_start = total_start

    device = infer_model_input_device(model)
    full_input_ids = assemble_token_aligned_full_prefill_input_ids(tokenized_example)
    doc_total_len = sum(len(doc_ids) for doc_ids in tokenized_example.doc_chunk_ids)
    if doc_total_len <= 0:
        raise ValueError("Expected tokenized_example.doc_chunk_ids to contain document tokens.")

    full_input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=device)
    full_attention_mask = torch.ones_like(full_input_tensor)
    with torch.inference_mode():
        full_outputs = model(
            input_ids=full_input_tensor,
            attention_mask=full_attention_mask,
            output_hidden_states=True,
            use_cache=True,
        )
    if full_outputs.hidden_states is None:
        raise RuntimeError("Model did not return hidden_states for repair diagnostics.")
    if full_outputs.past_key_values is None:
        raise RuntimeError("Model did not return past_key_values for repair diagnostics.")

    full_past_key_values = normalize_past_key_values(full_outputs.past_key_values)
    full_doc_kv = slice_past_key_values_prefix(full_past_key_values, seq_len=doc_total_len)

    chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
    reuse_doc_kv = assemble_chunk_kvs(chunk_kvs)
    deviations = compute_kv_deviation(reuse_doc_kv, full_doc_kv)
    selected_indices_by_layer = select_gradual_hkvd_tokens_by_layer(
        deviations,
        initial_top_k=initial_top_k,
        top_k=top_k,
    )
    num_layers = len(full_doc_kv)
    validate_gradual_selection(
        selected_indices_by_layer,
        num_layers=num_layers,
        seq_len=doc_total_len,
    )

    # Current runtime primitive uses one fixed selected index set across layers.
    # Per-layer variable selected-index runtime repair is not implemented yet.
    runtime_selected_indices = selected_indices_by_layer[0]
    doc_attention_mask = torch.ones((1, doc_total_len), dtype=torch.long, device=device)
    selected_hidden_states = full_outputs.hidden_states[0][:, runtime_selected_indices, :]
    _, repaired_doc_kv = run_model_family_partial_repair(
        model=model,
        model_family=model_family,
        selected_hidden_states=selected_hidden_states,
        selected_indices=runtime_selected_indices,
        reuse_past_key_values=reuse_doc_kv,
        num_layers=num_layers,
        attention_mask=doc_attention_mask,
    )
    repair_diagnostics = (
        compute_repair_diagnostics(
            reuse_past_key_values=reuse_doc_kv,
            repaired_past_key_values=repaired_doc_kv,
            full_past_key_values=full_doc_kv,
            runtime_selected_indices=runtime_selected_indices,
            seq_len=doc_total_len,
        )
        if include_repair_diagnostics
        else {}
    )
    repair_latency_seconds = time.perf_counter() - repair_start

    decode_start = time.perf_counter()
    generation = generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=repaired_doc_kv,
        max_new_tokens=max_new_tokens,
    )
    decode_latency_seconds = time.perf_counter() - decode_start
    total_latency_seconds = time.perf_counter() - total_start

    metadata = {
        "model_family": model_family,
        "runtime_selected_indices": runtime_selected_indices,
        "runtime_selection_mode": "fixed_layer0_selected_indices",
        "selected_indices_by_layer": selected_indices_by_layer,
        "layer_selected_counts": [
            len(selected_indices_by_layer[layer_index]) for layer_index in range(num_layers)
        ],
        "repair_latency_seconds": repair_latency_seconds,
        "decode_latency_seconds": decode_latency_seconds,
        "total_latency_seconds": total_latency_seconds,
    }
    metadata.update(repair_diagnostics)
    return CacheBlendRepairResult(
        generated_ids=generation.generated_ids,
        output_text=generation.output_text,
        generation=generation,
        repaired_past_key_values=repaired_doc_kv,
        metadata=metadata,
    )
