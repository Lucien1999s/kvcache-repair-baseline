from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from contextflow.data.assembly import assemble_token_aligned_full_prefill_input_ids
from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache import (
    assemble_chunk_kvs,
    correct_doc_chunk_kvs_for_model_family,
    precompute_doc_chunk_kvs,
    rope_position_correction_enabled,
)
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
class CacheBlendPartialRepairResult:
    final_selected_hidden_states: Any
    repaired_past_key_values: Any
    metadata: dict[str, Any]


@dataclass(slots=True)
class CacheBlendRepairPlanningArtifacts:
    plan: CacheBlendRepairPlan
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


def assemble_doc_input_ids(tokenized_example: TokenizedExample) -> list[int]:
    return [
        token_id
        for doc_chunk_ids in tokenized_example.doc_chunk_ids
        for token_id in doc_chunk_ids
    ]


def validate_repair_plan_for_doc_input(
    plan: CacheBlendRepairPlan,
    doc_input_ids: list[int],
) -> None:
    if plan.doc_total_len != len(doc_input_ids):
        raise ValueError(
            "repair_plan.doc_total_len must match concatenated document token length; "
            f"got {plan.doc_total_len} and {len(doc_input_ids)}."
        )
    validate_gradual_selection(
        plan.selected_indices_by_layer,
        num_layers=plan.num_layers,
        seq_len=plan.doc_total_len,
    )
    if plan.runtime_selected_indices != plan.selected_indices_by_layer[0]:
        raise ValueError(
            "repair_plan.runtime_selected_indices must match layer 0 selected indices."
        )
    if plan.runtime_selected_indices != sorted(plan.runtime_selected_indices):
        raise ValueError("repair_plan.runtime_selected_indices must be sorted.")
    invalid = [
        index
        for index in plan.runtime_selected_indices
        if index < 0 or index >= plan.doc_total_len
    ]
    if invalid:
        raise ValueError(
            "repair_plan.runtime_selected_indices contains invalid indices: "
            f"{invalid}."
        )


def validate_reuse_past_key_values_for_plan(
    past_key_values: Any,
    plan: CacheBlendRepairPlan,
) -> tuple[tuple[Any, Any], ...]:
    normalized_past_key_values = normalize_past_key_values(past_key_values)
    if len(normalized_past_key_values) != plan.num_layers:
        raise ValueError(
            "reuse_past_key_values layer count must match repair plan; "
            f"got {len(normalized_past_key_values)} and {plan.num_layers}."
        )

    for layer_index, layer_kv in enumerate(normalized_past_key_values):
        key, value = layer_kv[0], layer_kv[1]
        if int(key.shape[-2]) != plan.doc_total_len:
            raise ValueError(
                f"Layer {layer_index} reuse key seq_len must match repair plan doc_total_len; "
                f"got {key.shape[-2]} and {plan.doc_total_len}."
            )
        if int(value.shape[-2]) != plan.doc_total_len:
            raise ValueError(
                f"Layer {layer_index} reuse value seq_len must match repair plan "
                f"doc_total_len; got {value.shape[-2]} and {plan.doc_total_len}."
            )
    return normalized_past_key_values


def compute_repair_diagnostics(
    reuse_past_key_values: tuple[tuple[Any, Any], ...],
    repaired_past_key_values: tuple[tuple[Any, Any], ...],
    full_past_key_values: tuple[tuple[Any, Any], ...],
    runtime_selected_indices: list[int],
    seq_len: int,
    selected_indices_by_layer: dict[int, list[int]] | None = None,
) -> dict[str, dict[int, float] | dict[int, bool]]:
    """Compare reused/repaired doc KV against full recompute doc KV.

    These values are diagnostics. With true per-layer partial repair, layers after
    layer 0 consume hidden states propagated through mixed repaired/reused KV, so
    selected KV deviation from full recompute is not guaranteed to decrease
    monotonically at every layer.
    """

    if len(reuse_past_key_values) != len(repaired_past_key_values):
        raise ValueError("reuse and repaired KV must have the same layer count.")
    if len(reuse_past_key_values) != len(full_past_key_values):
        raise ValueError("reuse and full KV must have the same layer count.")

    selected_before: dict[int, float] = {}
    selected_after: dict[int, float] = {}
    unselected_after_vs_reuse: dict[int, float] = {}
    shape_matches: dict[int, bool] = {}

    for layer_index, (reuse_layer_kv, repaired_layer_kv, full_layer_kv) in enumerate(
        zip(reuse_past_key_values, repaired_past_key_values, full_past_key_values)
    ):
        selected_indices = (
            selected_indices_by_layer[layer_index]
            if selected_indices_by_layer is not None
            else runtime_selected_indices
        )
        unselected_indices = resolve_unselected_indices(seq_len, selected_indices)
        reuse_key, reuse_value = reuse_layer_kv[0], reuse_layer_kv[1]
        repaired_key, repaired_value = repaired_layer_kv[0], repaired_layer_kv[1]
        selected_before[layer_index] = layer_kv_max_diff_at_indices(
            reuse_layer_kv,
            full_layer_kv,
            selected_indices,
        )
        selected_after[layer_index] = layer_kv_max_diff_at_indices(
            repaired_layer_kv,
            full_layer_kv,
            selected_indices,
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
    planning_latency_seconds: float,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "diagnostic_source": "full_recompute_reference",
        "uses_full_recompute_reference": True,
        "selection_algorithm": "gradual_hkvd",
        "initial_top_k": initial_top_k,
        "top_k": top_k,
        "planning_latency_seconds": planning_latency_seconds,
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
    metadata: dict[str, Any] | None = None,
    strategy: str = "oracle_hkvd_gradual",
    runtime_selection_mode: str = "gradual_selected_indices_by_layer",
) -> CacheBlendRepairPlan:
    validate_gradual_selection(
        selected_indices_by_layer,
        num_layers=num_layers,
        seq_len=doc_total_len,
    )
    # The layer-0 set is the initial active set; gradual filtering can shrink it
    # in later layers through selected_indices_by_layer.
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
        strategy=strategy,
        runtime_selection_mode=runtime_selection_mode,
        metadata=dict(metadata or {}),
    )


def _prepare_cacheblend_repair_plan_and_artifacts(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
) -> CacheBlendRepairPlanningArtifacts:
    if initial_top_k <= 0 or top_k <= 0:
        raise ValueError("initial_top_k and top_k must be positive.")

    planning_start = time.perf_counter()
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
            use_cache=True,
        )
    if full_outputs.past_key_values is None:
        raise RuntimeError("Model did not return past_key_values for repair diagnostics.")

    full_past_key_values = normalize_past_key_values(full_outputs.past_key_values)
    full_doc_kv = slice_past_key_values_prefix(full_past_key_values, seq_len=doc_total_len)
    chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
    chunk_kvs = correct_doc_chunk_kvs_for_model_family(
        model=model,
        chunk_kvs=chunk_kvs,
        model_family=model_family,
    )
    reuse_doc_kv = assemble_chunk_kvs(chunk_kvs)
    deviations = compute_kv_deviation(reuse_doc_kv, full_doc_kv)
    selected_indices_by_layer = select_gradual_hkvd_tokens_by_layer(
        deviations,
        initial_top_k=initial_top_k,
        top_k=top_k,
    )
    num_layers = len(full_doc_kv)
    planning_latency_seconds = time.perf_counter() - planning_start
    plan_metadata = compute_plan_diagnostic_metadata(
        deviations=deviations,
        selected_indices_by_layer=selected_indices_by_layer,
        initial_top_k=initial_top_k,
        top_k=top_k,
        planning_latency_seconds=planning_latency_seconds,
    )
    applies_rope_correction = rope_position_correction_enabled(model_family)
    plan_metadata["rope_position_correction_applied"] = applies_rope_correction
    plan_metadata["reuse_doc_kv_position_basis"] = (
        "full_context_absolute" if applies_rope_correction else "native"
    )
    plan = build_cacheblend_repair_plan(
        selected_indices_by_layer=selected_indices_by_layer,
        doc_total_len=doc_total_len,
        num_layers=num_layers,
        metadata=plan_metadata,
    )
    return CacheBlendRepairPlanningArtifacts(
        plan=plan,
        full_doc_kv=full_doc_kv,
        reuse_doc_kv=reuse_doc_kv,
    )


def prepare_cacheblend_repair_plan(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
) -> CacheBlendRepairPlan:
    """Prepare an oracle-HKVD repair plan for CacheBlend-style partial repair.

    This planner intentionally uses full recompute KV as a diagnostic reference.
    It should be measured separately from repair execution when reporting resource
    numbers. RoPE model families correct reused doc KVs into full-context
    absolute positions before HKVD selection.
    """

    return _prepare_cacheblend_repair_plan_and_artifacts(
        model=model,
        tokenized_example=tokenized_example,
        initial_top_k=initial_top_k,
        top_k=top_k,
        model_family=model_family,
    ).plan


def prepare_cacheblend_repair_plan_with_artifacts(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
) -> CacheBlendRepairPlanningArtifacts:
    """Prepare an oracle-HKVD repair plan plus reusable planning artifacts.

    Dataset-level diagnostics use this to measure planning as one phase and then
    reuse the assembled document KV during clean repair execution without
    recomputing it. Returned reuse_doc_kv is already position-corrected for
    RoPE model families.
    """

    return _prepare_cacheblend_repair_plan_and_artifacts(
        model=model,
        tokenized_example=tokenized_example,
        initial_top_k=initial_top_k,
        top_k=top_k,
        model_family=model_family,
    )


def compute_model_family_selected_initial_hidden_states(
    model: Any,
    model_family: str,
    doc_input_ids: list[int],
    selected_indices: list[int],
) -> Any:
    if not doc_input_ids:
        raise ValueError("doc_input_ids must be non-empty.")

    normalized_model_family = model_family.lower()
    device = infer_model_input_device(model)
    input_tensor = torch.tensor([doc_input_ids], dtype=torch.long, device=device)

    if normalized_model_family == "gpt2":
        from contextflow.repair.adapters.gpt2 import (
            compute_gpt2_selected_initial_hidden_states,
            validate_gpt2_like_model,
        )

        validate_gpt2_like_model(model)
        return compute_gpt2_selected_initial_hidden_states(
            model=model,
            input_ids=input_tensor,
            selected_indices=selected_indices,
        )

    if normalized_model_family == "mistral":
        from contextflow.repair.adapters.mistral import (
            compute_mistral_selected_initial_hidden_states,
            validate_mistral_like_model,
        )

        validate_mistral_like_model(model)
        return compute_mistral_selected_initial_hidden_states(
            model=model,
            input_ids=input_tensor,
            selected_indices=selected_indices,
        )

    if normalized_model_family == "qwen2":
        from contextflow.repair.adapters.qwen2 import (
            compute_qwen2_selected_initial_hidden_states,
            validate_qwen2_like_model,
        )

        validate_qwen2_like_model(model)
        return compute_qwen2_selected_initial_hidden_states(
            model=model,
            input_ids=input_tensor,
            selected_indices=selected_indices,
        )

    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. CacheBlend-style repair currently "
        "supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )


def run_model_family_partial_repair(
    model: Any,
    model_family: str,
    selected_hidden_states: Any,
    selected_indices: list[int],
    reuse_past_key_values: Any,
    num_layers: int,
    attention_mask: Any,
    selected_indices_by_layer: dict[int, list[int]] | None = None,
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
            selected_indices_by_layer=selected_indices_by_layer,
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
            selected_indices_by_layer=selected_indices_by_layer,
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
            selected_indices_by_layer=selected_indices_by_layer,
        )

    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. CacheBlend-style repair currently "
        "supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )


def run_cacheblend_style_partial_repair_from_plan(
    model: Any,
    tokenized_example: TokenizedExample,
    repair_plan: CacheBlendRepairPlan,
    model_family: str = "gpt2",
    reuse_past_key_values: Any | None = None,
) -> CacheBlendPartialRepairResult:
    """Execute selected-token partial KV repair from a precomputed repair plan.

    This path does not compute or inspect full-recompute reference KV, and it does
    not decode. Dataset runners can therefore measure repair and decode phases
    separately.
    """

    model_family = model_family.lower()
    doc_input_ids = assemble_doc_input_ids(tokenized_example)
    validate_repair_plan_for_doc_input(repair_plan, doc_input_ids)
    expected_rope_correction = rope_position_correction_enabled(model_family)
    planned_rope_correction = repair_plan.metadata.get("rope_position_correction_applied")
    if (
        planned_rope_correction is not None
        and bool(planned_rope_correction) != expected_rope_correction
    ):
        raise ValueError(
            "repair_plan rope_position_correction_applied does not match model_family; "
            f"plan has {planned_rope_correction}, model_family={model_family!r} "
            f"expects {expected_rope_correction}."
        )

    reuse_precompute_latency_seconds: float | None = None
    if reuse_past_key_values is None:
        reuse_precompute_start = time.perf_counter()
        chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
        chunk_kvs = correct_doc_chunk_kvs_for_model_family(
            model=model,
            chunk_kvs=chunk_kvs,
            model_family=model_family,
        )
        reuse_doc_kv = assemble_chunk_kvs(chunk_kvs)
        reuse_precompute_latency_seconds = time.perf_counter() - reuse_precompute_start
        reuse_past_key_values_source = "computed_from_doc_chunks"
    else:
        reuse_doc_kv = reuse_past_key_values
        reuse_past_key_values_source = "provided"
    reuse_doc_kv = validate_reuse_past_key_values_for_plan(reuse_doc_kv, repair_plan)

    runtime_selected_indices = repair_plan.runtime_selected_indices
    repair_start = time.perf_counter()
    selected_hidden_states = compute_model_family_selected_initial_hidden_states(
        model=model,
        model_family=model_family,
        doc_input_ids=doc_input_ids,
        selected_indices=runtime_selected_indices,
    )
    doc_attention_mask = torch.ones(
        (1, repair_plan.doc_total_len),
        dtype=torch.long,
        device=selected_hidden_states.device,
    )
    final_selected_hidden_states, repaired_doc_kv = run_model_family_partial_repair(
        model=model,
        model_family=model_family,
        selected_hidden_states=selected_hidden_states,
        selected_indices=runtime_selected_indices,
        reuse_past_key_values=reuse_doc_kv,
        num_layers=repair_plan.num_layers,
        attention_mask=doc_attention_mask,
        selected_indices_by_layer=repair_plan.selected_indices_by_layer,
    )
    repair_latency_seconds = time.perf_counter() - repair_start

    metadata = {
        "model_family": model_family,
        "execution_mode": "partial_repair_from_plan",
        "execution_uses_full_recompute_reference": False,
        "planning_included_in_total_latency": False,
        "selected_initial_hidden_source": "model_embedding_path",
        "reuse_past_key_values_source": reuse_past_key_values_source,
        "rope_position_correction_applied": expected_rope_correction,
        "reuse_precompute_latency_seconds": reuse_precompute_latency_seconds,
        "repair_latency_seconds": repair_latency_seconds,
    }
    metadata.update(repair_plan.to_metadata())
    return CacheBlendPartialRepairResult(
        final_selected_hidden_states=final_selected_hidden_states,
        repaired_past_key_values=repaired_doc_kv,
        metadata=metadata,
    )


def run_cacheblend_style_repair_generation_from_plan(
    model: Any,
    tokenizer: Any,
    tokenized_example: TokenizedExample,
    repair_plan: CacheBlendRepairPlan,
    max_new_tokens: int = 16,
    model_family: str = "gpt2",
    reuse_past_key_values: Any | None = None,
) -> CacheBlendRepairResult:
    """Execute CacheBlend-style repair and decode from a precomputed repair plan.

    This execution path does not compute or inspect full-recompute reference KV.
    If reuse_past_key_values is provided, repair/decode measurement can exclude both
    oracle planning and reusable document-KV construction.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    total_start = time.perf_counter()
    partial_repair = run_cacheblend_style_partial_repair_from_plan(
        model=model,
        tokenized_example=tokenized_example,
        repair_plan=repair_plan,
        model_family=model_family,
        reuse_past_key_values=reuse_past_key_values,
    )

    decode_start = time.perf_counter()
    generation = generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=partial_repair.repaired_past_key_values,
        max_new_tokens=max_new_tokens,
    )
    decode_latency_seconds = time.perf_counter() - decode_start
    repair_latency_seconds = float(partial_repair.metadata["repair_latency_seconds"])
    execution_latency_seconds = repair_latency_seconds + decode_latency_seconds
    total_latency_seconds = time.perf_counter() - total_start

    metadata = dict(partial_repair.metadata)
    metadata.update(
        {
            "execution_mode": "from_plan",
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "execution_latency_seconds": execution_latency_seconds,
            "total_latency_seconds": total_latency_seconds,
        }
    )
    return CacheBlendRepairResult(
        generated_ids=generation.generated_ids,
        output_text=generation.output_text,
        generation=generation,
        repaired_past_key_values=partial_repair.repaired_past_key_values,
        metadata=metadata,
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
        model_family=model_family,
    )
    plan = artifacts.plan
    execution_result = run_cacheblend_style_repair_generation_from_plan(
        model=model,
        tokenizer=tokenizer,
        tokenized_example=tokenized_example,
        repair_plan=plan,
        max_new_tokens=max_new_tokens,
        model_family=model_family,
        reuse_past_key_values=artifacts.reuse_doc_kv,
    )
    repair_diagnostics = (
        compute_repair_diagnostics(
            reuse_past_key_values=artifacts.reuse_doc_kv,
            repaired_past_key_values=execution_result.repaired_past_key_values,
            full_past_key_values=artifacts.full_doc_kv,
            runtime_selected_indices=plan.runtime_selected_indices,
            seq_len=plan.doc_total_len,
            selected_indices_by_layer=plan.selected_indices_by_layer,
        )
        if include_repair_diagnostics
        else {}
    )
    total_latency_seconds = time.perf_counter() - total_start

    metadata = dict(execution_result.metadata)
    metadata["total_latency_seconds"] = total_latency_seconds
    metadata["planning_included_in_total_latency"] = True
    metadata.update(repair_diagnostics)
    return CacheBlendRepairResult(
        generated_ids=execution_result.generated_ids,
        output_text=execution_result.output_text,
        generation=execution_result.generation,
        repaired_past_key_values=execution_result.repaired_past_key_values,
        metadata=metadata,
    )
