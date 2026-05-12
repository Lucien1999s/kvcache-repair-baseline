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
class CacheBlendRepairPlan:
    """Selected-token plan for CacheBlend-style partial KV repair.

    The current planner is an oracle diagnostic planner: it compares reused doc KV
    against full recompute doc KV to select gradual HKVD tokens. Execution code can
    later consume this plan without owning the selection/reference computation.
    """

    selected_indices_by_layer: dict[int, list[int]]
    runtime_selected_indices: list[int]
    layer_selected_counts: list[int]
    doc_total_len: int
    num_layers: int
    strategy: str
    runtime_selection_mode: str
    metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "repair_plan_strategy": self.strategy,
            "runtime_selection_mode": self.runtime_selection_mode,
            "runtime_selected_indices": list(self.runtime_selected_indices),
            "selected_indices_by_layer": {
                int(layer_index): list(selected_indices)
                for layer_index, selected_indices in self.selected_indices_by_layer.items()
            },
            "layer_selected_counts": list(self.layer_selected_counts),
            "doc_total_len": self.doc_total_len,
            "num_layers": self.num_layers,
            "repair_plan_metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class CacheBlendRepairResult:
    generated_ids: list[int]
    output_text: str
    generation: HFCachedGenerationResult
    repaired_past_key_values: Any
    metadata: dict[str, Any]


@dataclass(slots=True)
class CacheBlendRepairPlanningArtifacts:
    plan: CacheBlendRepairPlan
    full_outputs: Any
    full_doc_kv: tuple[tuple[Any, Any], ...]
    reuse_doc_kv: tuple[tuple[Any, Any], ...]


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


def tensor_max_float(tensor: Any) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.max().detach().cpu())


def tensor_min_float(tensor: Any) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.min().detach().cpu())


def tensor_mean_float(tensor: Any) -> float:
    if tensor.numel() == 0:
        return 0.0
    return float(tensor.float().mean().detach().cpu())


def compute_plan_diagnostic_metadata(
    deviations: list[Any],
    selected_indices_by_layer: dict[int, list[int]],
    initial_top_k: int,
    top_k: int,
    selection_latency_seconds: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "diagnostic_source": "full_recompute_reference",
        "uses_full_recompute_reference": True,
        "selection_algorithm": "gradual_hkvd",
        "initial_top_k": initial_top_k,
        "top_k": top_k,
        "selection_latency_seconds": selection_latency_seconds,
        "deviation_max_by_layer": {},
        "deviation_mean_by_layer": {},
        "selected_deviation_max_by_layer": {},
        "selected_deviation_min_by_layer": {},
        "selected_deviation_mean_by_layer": {},
    }
    for layer_index, layer_deviation in enumerate(deviations):
        selected_indices = selected_indices_by_layer[layer_index]
        selected_deviation = layer_deviation[selected_indices]
        metadata["deviation_max_by_layer"][layer_index] = tensor_max_float(layer_deviation)
        metadata["deviation_mean_by_layer"][layer_index] = tensor_mean_float(layer_deviation)
        metadata["selected_deviation_max_by_layer"][layer_index] = tensor_max_float(
            selected_deviation
        )
        metadata["selected_deviation_min_by_layer"][layer_index] = tensor_min_float(
            selected_deviation
        )
        metadata["selected_deviation_mean_by_layer"][layer_index] = tensor_mean_float(
            selected_deviation
        )
    return metadata


def build_cacheblend_repair_plan(
    selected_indices_by_layer: dict[int, list[int]],
    doc_total_len: int,
    num_layers: int,
    metadata: dict[str, Any],
) -> CacheBlendRepairPlan:
    validate_gradual_selection(
        selected_indices_by_layer,
        num_layers=num_layers,
        seq_len=doc_total_len,
    )
    # Current runtime primitive uses one fixed selected index set across layers.
    # Per-layer variable selected-index runtime repair is not implemented yet.
    runtime_selected_indices = list(selected_indices_by_layer[0])
    return CacheBlendRepairPlan(
        selected_indices_by_layer={
            int(layer_index): list(selected_indices)
            for layer_index, selected_indices in selected_indices_by_layer.items()
        },
        runtime_selected_indices=runtime_selected_indices,
        layer_selected_counts=[
            len(selected_indices_by_layer[layer_index]) for layer_index in range(num_layers)
        ],
        doc_total_len=doc_total_len,
        num_layers=num_layers,
        strategy="oracle_hkvd_gradual",
        runtime_selection_mode="fixed_layer0_selected_indices",
        metadata=metadata,
    )


def _prepare_cacheblend_repair_plan_and_artifacts(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
) -> CacheBlendRepairPlanningArtifacts:
    if initial_top_k <= 0 or top_k <= 0:
        raise ValueError("initial_top_k and top_k must be positive.")

    selection_start = time.perf_counter()
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
    selection_latency_seconds = time.perf_counter() - selection_start
    plan_metadata = compute_plan_diagnostic_metadata(
        deviations=deviations,
        selected_indices_by_layer=selected_indices_by_layer,
        initial_top_k=initial_top_k,
        top_k=top_k,
        selection_latency_seconds=selection_latency_seconds,
    )
    plan = build_cacheblend_repair_plan(
        selected_indices_by_layer=selected_indices_by_layer,
        doc_total_len=doc_total_len,
        num_layers=num_layers,
        metadata=plan_metadata,
    )
    return CacheBlendRepairPlanningArtifacts(
        plan=plan,
        full_outputs=full_outputs,
        full_doc_kv=full_doc_kv,
        reuse_doc_kv=reuse_doc_kv,
    )


def prepare_cacheblend_repair_plan(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
) -> CacheBlendRepairPlan:
    """Prepare an oracle-HKVD repair plan for CacheBlend-style partial repair.

    This planner intentionally uses full recompute KV as a diagnostic reference.
    It should be measured separately from repair execution when reporting resource
    numbers.
    """

    return _prepare_cacheblend_repair_plan_and_artifacts(
        model=model,
        tokenized_example=tokenized_example,
        initial_top_k=initial_top_k,
        top_k=top_k,
    ).plan


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
    artifacts = _prepare_cacheblend_repair_plan_and_artifacts(
        model=model,
        tokenized_example=tokenized_example,
        initial_top_k=initial_top_k,
        top_k=top_k,
    )
    plan = artifacts.plan
    repair_start = time.perf_counter()
    device = infer_model_input_device(model)
    runtime_selected_indices = plan.runtime_selected_indices
    doc_attention_mask = torch.ones((1, plan.doc_total_len), dtype=torch.long, device=device)
    selected_hidden_states = artifacts.full_outputs.hidden_states[0][:, runtime_selected_indices, :]
    _, repaired_doc_kv = run_model_family_partial_repair(
        model=model,
        model_family=model_family,
        selected_hidden_states=selected_hidden_states,
        selected_indices=runtime_selected_indices,
        reuse_past_key_values=artifacts.reuse_doc_kv,
        num_layers=plan.num_layers,
        attention_mask=doc_attention_mask,
    )
    repair_diagnostics = (
        compute_repair_diagnostics(
            reuse_past_key_values=artifacts.reuse_doc_kv,
            repaired_past_key_values=repaired_doc_kv,
            full_past_key_values=artifacts.full_doc_kv,
            runtime_selected_indices=runtime_selected_indices,
            seq_len=plan.doc_total_len,
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
        "repair_latency_seconds": repair_latency_seconds,
        "decode_latency_seconds": decode_latency_seconds,
        "total_latency_seconds": total_latency_seconds,
    }
    metadata.update(plan.to_metadata())
    metadata.update(repair_diagnostics)
    return CacheBlendRepairResult(
        generated_ids=generation.generated_ids,
        output_text=generation.output_text,
        generation=generation,
        repaired_past_key_values=repaired_doc_kv,
        metadata=metadata,
    )
