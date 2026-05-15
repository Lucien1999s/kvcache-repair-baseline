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
from contextflow.profiling.micro import MicroProfiler, profile_micro_step
from contextflow.repair.cacheblend_selector import (
    compute_kv_deviation,
    l2_norm_by_token,
    select_gradual_hkvd_tokens_by_layer,
)
from contextflow.repair.adapters.selection import select_hidden_states_for_indices
from contextflow.runtime.hf_cached_generation import (
    HFCachedGenerationResult,
    generate_with_past_key_values,
)


REPAIR_PLANNER_ORACLE_HKVD = "oracle_hkvd"
REPAIR_PLANNER_ONLINE_GRADUAL_HKVD = "online_gradual_hkvd"
SUPPORTED_REPAIR_PLANNERS = {
    REPAIR_PLANNER_ORACLE_HKVD,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
}


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
class CacheBlendOnlinePartialRepairResult:
    repair_plan: CacheBlendRepairPlan
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


def layer_kv_deviation_scores_at_indices(
    layer_a: tuple[Any, Any],
    layer_b: tuple[Any, Any],
    indices: list[int],
) -> Any:
    """Return per-token KV deviation scores for selected absolute token indices."""

    key_a, value_a = layer_a[0], layer_a[1]
    key_b, value_b = layer_b[0], layer_b[1]
    selected_key_a = index_select_sequence(key_a, indices)
    selected_key_b = index_select_sequence(key_b, indices)
    selected_value_a = index_select_sequence(value_a, indices)
    selected_value_b = index_select_sequence(value_b, indices)
    return (
        l2_norm_by_token(selected_key_a - selected_key_b)
        + l2_norm_by_token(selected_value_a - selected_value_b)
    )


def select_topk_indices_from_scores(candidate_indices: list[int], scores: Any, top_k: int) -> list[int]:
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if not candidate_indices:
        raise ValueError("candidate_indices must be non-empty.")
    if int(scores.shape[0]) != len(candidate_indices):
        raise ValueError(
            "scores length must match candidate_indices length; "
            f"got {scores.shape[0]} and {len(candidate_indices)}."
        )

    k = min(top_k, len(candidate_indices))
    selected_offsets = scores.topk(k=k, largest=True).indices.tolist()
    return sorted(candidate_indices[int(offset)] for offset in selected_offsets)


def resolve_repair_diagnostics_tolerance(
    requested_tolerance: float,
    repaired_past_key_values: tuple[tuple[Any, Any], ...],
) -> float:
    if requested_tolerance <= 0:
        raise ValueError("repair diagnostics tolerance must be positive.")
    if not repaired_past_key_values:
        return requested_tolerance

    dtype = repaired_past_key_values[0][0].dtype
    if dtype.is_floating_point:
        return max(requested_tolerance, float(torch.finfo(dtype).eps) * 4.0)
    return requested_tolerance


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
    tolerance: float = 1e-5,
) -> dict[str, Any]:
    """Compare reused/repaired doc KV against full recompute doc KV.

    These values are diagnostics. Hard invariants cover patch scope, shape, and
    gradual selection. Full-reference closeness is reported separately because
    with true per-layer partial repair, layers after layer 0 consume hidden states
    propagated through mixed repaired/reused KV, so selected KV deviation from
    full recompute is not guaranteed to decrease monotonically at every layer.
    """

    if len(reuse_past_key_values) != len(repaired_past_key_values):
        raise ValueError("reuse and repaired KV must have the same layer count.")
    if len(reuse_past_key_values) != len(full_past_key_values):
        raise ValueError("reuse and full KV must have the same layer count.")

    effective_tolerance = resolve_repair_diagnostics_tolerance(
        requested_tolerance=tolerance,
        repaired_past_key_values=repaired_past_key_values,
    )
    selected_before: dict[int, float] = {}
    selected_after: dict[int, float] = {}
    selected_after_vs_reuse: dict[int, float] = {}
    selected_oracle_close: dict[int, bool] = {}
    selected_oracle_not_worse: dict[int, bool] = {}
    unselected_after_vs_reuse: dict[int, float] = {}
    unselected_unchanged: dict[int, bool] = {}
    shape_matches: dict[int, bool] = {}
    patch_scope_valid: dict[int, bool] = {}

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
        selected_after_vs_reuse[layer_index] = layer_kv_max_diff_at_indices(
            repaired_layer_kv,
            reuse_layer_kv,
            selected_indices,
        )
        selected_oracle_close[layer_index] = selected_after[layer_index] <= effective_tolerance
        selected_oracle_not_worse[layer_index] = (
            selected_after[layer_index] <= selected_before[layer_index] + effective_tolerance
        )
        unselected_after_vs_reuse[layer_index] = layer_kv_max_diff_at_indices(
            repaired_layer_kv,
            reuse_layer_kv,
            unselected_indices,
        )
        unselected_unchanged[layer_index] = unselected_after_vs_reuse[layer_index] == 0.0
        shape_matches[layer_index] = (
            tuple(repaired_key.shape) == tuple(reuse_key.shape)
            and tuple(repaired_value.shape) == tuple(reuse_value.shape)
        )
        patch_scope_valid[layer_index] = (
            shape_matches[layer_index] and unselected_unchanged[layer_index]
        )

    if selected_indices_by_layer is None:
        gradual_selected_sets_valid = True
    else:
        try:
            validate_gradual_selection(
                selected_indices_by_layer,
                num_layers=len(reuse_past_key_values),
                seq_len=seq_len,
            )
            gradual_selected_sets_valid = True
        except ValueError:
            gradual_selected_sets_valid = False

    layer0_selected_oracle_close = selected_oracle_close.get(0, False)
    layer0_selected_oracle_not_worse = selected_oracle_not_worse.get(0, False)
    hard_invariants = {
        "shape_matches_all_layers": all(shape_matches.values()),
        "unselected_kv_unchanged_all_layers": all(unselected_unchanged.values()),
        "patch_scope_valid_all_layers": all(patch_scope_valid.values()),
        "gradual_selected_sets_valid": gradual_selected_sets_valid,
        "layer0_selected_kv_not_worse_than_reuse": layer0_selected_oracle_not_worse,
    }
    hard_invariants["all_passed"] = all(hard_invariants.values())
    oracle_reference_checks = {
        "layer0_selected_kv_close_to_full": layer0_selected_oracle_close,
        "layer0_selected_kv_not_worse_than_reuse": layer0_selected_oracle_not_worse,
    }

    return {
        "repair_diagnostics_mode": "oracle_full_reference",
        "repair_diagnostics_tolerance": effective_tolerance,
        "repair_diagnostics_requested_tolerance": tolerance,
        "selected_kv_max_diff_before_by_layer": selected_before,
        "selected_kv_max_diff_after_by_layer": selected_after,
        "selected_kv_max_diff_after_vs_reuse_by_layer": selected_after_vs_reuse,
        "selected_kv_oracle_close_by_layer": selected_oracle_close,
        "selected_kv_oracle_not_worse_by_layer": selected_oracle_not_worse,
        "unselected_kv_max_diff_after_vs_reuse_by_layer": unselected_after_vs_reuse,
        "unselected_kv_unchanged_by_layer": unselected_unchanged,
        "repaired_kv_shape_matches_reuse_by_layer": shape_matches,
        "repair_patch_scope_valid_by_layer": patch_scope_valid,
        "repair_hard_invariants": hard_invariants,
        "repair_oracle_reference_checks": oracle_reference_checks,
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


def run_model_family_partial_layer(
    model: Any,
    model_family: str,
    layer_index: int,
    selected_hidden_states: Any,
    selected_indices: list[int],
    layer_kv: tuple[Any, Any],
    attention_mask: Any,
) -> tuple[Any, tuple[Any, Any]]:
    normalized_model_family = model_family.lower()

    if normalized_model_family == "gpt2":
        from contextflow.repair.adapters.gpt2 import (
            run_gpt2_partial_layer,
            validate_gpt2_like_model,
        )

        validate_gpt2_like_model(model)
        return run_gpt2_partial_layer(
            model=model,
            layer_index=layer_index,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            layer_kv=layer_kv,
            attention_mask=attention_mask,
        )

    if normalized_model_family == "mistral":
        from contextflow.repair.adapters.mistral import (
            run_mistral_partial_layer,
            validate_mistral_like_model,
        )

        validate_mistral_like_model(model)
        return run_mistral_partial_layer(
            model=model,
            layer_index=layer_index,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            layer_kv=layer_kv,
            attention_mask=attention_mask,
        )

    if normalized_model_family == "qwen2":
        from contextflow.repair.adapters.qwen2 import (
            run_qwen2_partial_layer,
            validate_qwen2_like_model,
        )

        validate_qwen2_like_model(model)
        return run_qwen2_partial_layer(
            model=model,
            layer_index=layer_index,
            selected_hidden_states=selected_hidden_states,
            selected_indices=selected_indices,
            layer_kv=layer_kv,
            attention_mask=attention_mask,
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


def build_repair_micro_summary_metadata(
    *,
    doc_total_len: int,
    runtime_selected_count: int,
    num_layers: int,
    layer_selected_counts: list[int],
    repair_plan_strategy: str | None = None,
    runtime_selection_mode: str | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "doc_total_len": doc_total_len,
        "runtime_selected_count": runtime_selected_count,
        "selected_ratio": (
            float(runtime_selected_count) / float(doc_total_len)
            if doc_total_len > 0
            else None
        ),
        "num_layers": num_layers,
        "layer_selected_counts": list(layer_selected_counts),
    }
    if repair_plan_strategy is not None:
        metadata["repair_plan_strategy"] = repair_plan_strategy
    if runtime_selection_mode is not None:
        metadata["runtime_selection_mode"] = runtime_selection_mode
    return metadata


def attach_micro_profile_metadata(
    metadata: dict[str, Any],
    micro_profiler: MicroProfiler | None,
    micro_summary_metadata: dict[str, Any],
) -> None:
    if micro_profiler is None:
        return
    metadata["micro_phase_metrics"] = micro_profiler.to_records()
    summary = micro_profiler.summary()
    summary.update(micro_summary_metadata)
    metadata["repair_micro_summary"] = summary


def run_model_family_partial_repair_with_micro(
    model: Any,
    model_family: str,
    selected_hidden_states: Any,
    selected_indices: list[int],
    reuse_past_key_values: Any,
    num_layers: int,
    attention_mask: Any,
    selected_indices_by_layer: dict[int, list[int]],
    micro_profiler: MicroProfiler | None,
    micro_prefix: str,
) -> tuple[Any, tuple[tuple[Any, Any], ...]]:
    patched_past_key_values = list(reuse_past_key_values)
    current_selected_hidden_states = selected_hidden_states
    current_selected_indices = list(selected_indices)
    loop_start = time.perf_counter()
    loop_record_start = len(micro_profiler.records) if micro_profiler is not None else 0

    for layer_index in range(num_layers):
        selected_for_layer = selected_indices_by_layer[layer_index]
        with profile_micro_step(
            micro_profiler,
            f"{micro_prefix}_partial_layer_{layer_index:02d}",
        ):
            current_selected_hidden_states = select_hidden_states_for_indices(
                hidden_states=current_selected_hidden_states,
                current_indices=current_selected_indices,
                target_indices=selected_for_layer,
            )
            current_selected_indices = selected_for_layer
            current_selected_hidden_states, patched_layer_kv = run_model_family_partial_layer(
                model=model,
                model_family=model_family,
                layer_index=layer_index,
                selected_hidden_states=current_selected_hidden_states,
                selected_indices=selected_for_layer,
                layer_kv=patched_past_key_values[layer_index],
                attention_mask=attention_mask,
            )

        with profile_micro_step(micro_profiler, f"{micro_prefix}_kv_patch_update"):
            original_key, original_value = reuse_past_key_values[layer_index][:2]
            patched_key, patched_value = patched_layer_kv[:2]
            if tuple(patched_key.shape) != tuple(original_key.shape):
                raise ValueError(
                    f"Layer {layer_index} patched key shape must match reused key shape; "
                    f"got {patched_key.shape} and {original_key.shape}."
                )
            if tuple(patched_value.shape) != tuple(original_value.shape):
                raise ValueError(
                    f"Layer {layer_index} patched value shape must match reused value shape; "
                    f"got {patched_value.shape} and {original_value.shape}."
                )
            patched_past_key_values[layer_index] = patched_layer_kv

    if micro_profiler is not None:
        micro_profiler.add_aggregate_record(
            name=f"{micro_prefix}_partial_layer_loop_total",
            child_records=micro_profiler.records[loop_record_start:],
            latency_seconds=time.perf_counter() - loop_start,
        )

    with profile_micro_step(micro_profiler, f"{micro_prefix}_finalize_repaired_kv"):
        repaired_past_key_values = tuple(patched_past_key_values)
    return current_selected_hidden_states, repaired_past_key_values


def run_cacheblend_style_partial_repair_from_plan(
    model: Any,
    tokenized_example: TokenizedExample,
    repair_plan: CacheBlendRepairPlan,
    model_family: str = "gpt2",
    reuse_past_key_values: Any | None = None,
    micro_profiler: MicroProfiler | None = None,
    micro_phase_prefix: str = "cacheblend",
) -> CacheBlendPartialRepairResult:
    """Execute selected-token partial KV repair from a precomputed repair plan.

    This path does not compute or inspect full-recompute reference KV, and it does
    not decode. Dataset runners can therefore measure repair and decode phases
    separately.
    """

    model_family = model_family.lower()
    if micro_phase_prefix == "cacheblend":
        with profile_micro_step(micro_profiler, "cacheblend_assemble_doc_input_ids"):
            doc_input_ids = assemble_doc_input_ids(tokenized_example)
    else:
        doc_input_ids = assemble_doc_input_ids(tokenized_example)

    with profile_micro_step(micro_profiler, f"{micro_phase_prefix}_validate_reuse_kv"):
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
    with profile_micro_step(
        micro_profiler,
        f"{micro_phase_prefix}_selected_initial_hidden_states",
    ):
        selected_hidden_states = compute_model_family_selected_initial_hidden_states(
            model=model,
            model_family=model_family,
            doc_input_ids=doc_input_ids,
            selected_indices=runtime_selected_indices,
        )
    with profile_micro_step(micro_profiler, f"{micro_phase_prefix}_doc_attention_mask"):
        doc_attention_mask = torch.ones(
            (1, repair_plan.doc_total_len),
            dtype=torch.long,
            device=selected_hidden_states.device,
        )
    if micro_profiler is None:
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
    else:
        final_selected_hidden_states, repaired_doc_kv = (
            run_model_family_partial_repair_with_micro(
                model=model,
                model_family=model_family,
                selected_hidden_states=selected_hidden_states,
                selected_indices=runtime_selected_indices,
                reuse_past_key_values=reuse_doc_kv,
                num_layers=repair_plan.num_layers,
                attention_mask=doc_attention_mask,
                selected_indices_by_layer=repair_plan.selected_indices_by_layer,
                micro_profiler=micro_profiler,
                micro_prefix=micro_phase_prefix,
            )
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
    attach_micro_profile_metadata(
        metadata,
        micro_profiler,
        build_repair_micro_summary_metadata(
            doc_total_len=repair_plan.doc_total_len,
            runtime_selected_count=len(runtime_selected_indices),
            num_layers=repair_plan.num_layers,
            layer_selected_counts=repair_plan.layer_selected_counts,
            repair_plan_strategy=repair_plan.strategy,
            runtime_selection_mode=repair_plan.runtime_selection_mode,
        ),
    )
    return CacheBlendPartialRepairResult(
        final_selected_hidden_states=final_selected_hidden_states,
        repaired_past_key_values=repaired_doc_kv,
        metadata=metadata,
    )


def run_cacheblend_style_online_partial_repair(
    model: Any,
    tokenized_example: TokenizedExample,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
    reuse_past_key_values: Any | None = None,
    micro_profiler: MicroProfiler | None = None,
) -> CacheBlendOnlinePartialRepairResult:
    """Run non-oracle gradual HKVD selection and partial repair together.

    This follows the practical CacheBlend idea more closely than oracle-HKVD
    planning: layer 0 recomputes all document tokens, measures fresh-vs-reused KV
    deviation on that layer, then gradually filters the active selected token set
    for later layers. It does not compute or inspect full-recompute reference KV.
    """

    if initial_top_k <= 0 or top_k <= 0:
        raise ValueError("initial_top_k and top_k must be positive.")

    model_family = model_family.lower()
    with profile_micro_step(micro_profiler, "cacheblend_assemble_doc_input_ids"):
        doc_input_ids = assemble_doc_input_ids(tokenized_example)
        doc_total_len = len(doc_input_ids)
        if doc_total_len <= 0:
            raise ValueError(
                "Expected tokenized_example.doc_chunk_ids to contain document tokens."
            )

    with profile_micro_step(micro_profiler, "cacheblend_validate_reuse_kv"):
        expected_rope_correction = rope_position_correction_enabled(model_family)
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

        reuse_doc_kv = normalize_past_key_values(reuse_doc_kv)
        num_layers = len(reuse_doc_kv)
        if num_layers == 0:
            raise ValueError("reuse_past_key_values must contain at least one layer.")
        for layer_index, layer_kv in enumerate(reuse_doc_kv):
            key, value = layer_kv[0], layer_kv[1]
            if int(key.shape[-2]) != doc_total_len:
                raise ValueError(
                    f"Layer {layer_index} reuse key seq_len must match doc length; "
                    f"got {key.shape[-2]} and {doc_total_len}."
                )
            if int(value.shape[-2]) != doc_total_len:
                raise ValueError(
                    f"Layer {layer_index} reuse value seq_len must match doc length; "
                    f"got {value.shape[-2]} and {doc_total_len}."
                )

    repair_start = time.perf_counter()
    active_indices = list(range(doc_total_len))
    with profile_micro_step(
        micro_profiler,
        "cacheblend_selected_initial_hidden_states",
    ):
        selected_hidden_states = compute_model_family_selected_initial_hidden_states(
            model=model,
            model_family=model_family,
            doc_input_ids=doc_input_ids,
            selected_indices=active_indices,
        )
    with profile_micro_step(micro_profiler, "cacheblend_doc_attention_mask"):
        doc_attention_mask = torch.ones(
            (1, doc_total_len),
            dtype=torch.long,
            device=selected_hidden_states.device,
        )

    patched_past_key_values = list(reuse_doc_kv)
    selected_indices_by_layer: dict[int, list[int]] = {}
    online_deviation_max_by_layer: dict[int, float] = {}
    online_deviation_mean_by_layer: dict[int, float] = {}
    online_deviation_selected_next_by_layer: dict[int, list[int]] = {}

    current_hidden_states = selected_hidden_states
    current_indices = active_indices
    loop_start = time.perf_counter()
    loop_record_start = len(micro_profiler.records) if micro_profiler is not None else 0
    for layer_index in range(num_layers):
        selected_for_layer = list(current_indices)
        selected_indices_by_layer[layer_index] = selected_for_layer
        with profile_micro_step(
            micro_profiler,
            f"cacheblend_partial_layer_{layer_index:02d}",
        ):
            next_hidden_states, patched_layer_kv = run_model_family_partial_layer(
                model=model,
                model_family=model_family,
                layer_index=layer_index,
                selected_hidden_states=current_hidden_states,
                selected_indices=selected_for_layer,
                layer_kv=patched_past_key_values[layer_index],
                attention_mask=doc_attention_mask,
            )

        with profile_micro_step(micro_profiler, "cacheblend_kv_patch_update"):
            original_key, original_value = reuse_doc_kv[layer_index][:2]
            patched_key, patched_value = patched_layer_kv[:2]
            if tuple(patched_key.shape) != tuple(original_key.shape):
                raise ValueError(
                    f"Layer {layer_index} patched key shape must match reused key shape; "
                    f"got {patched_key.shape} and {original_key.shape}."
                )
            if tuple(patched_value.shape) != tuple(original_value.shape):
                raise ValueError(
                    f"Layer {layer_index} patched value shape must match reused value shape; "
                    f"got {patched_value.shape} and {original_value.shape}."
                )
            patched_past_key_values[layer_index] = patched_layer_kv

        deviation_scores = layer_kv_deviation_scores_at_indices(
            patched_layer_kv,
            reuse_doc_kv[layer_index],
            selected_for_layer,
        )
        online_deviation_max_by_layer[layer_index] = tensor_max_float(deviation_scores)
        online_deviation_mean_by_layer[layer_index] = tensor_mean_float(deviation_scores)

        if layer_index == num_layers - 1:
            current_hidden_states = next_hidden_states
            current_indices = selected_for_layer
            continue

        next_top_k = initial_top_k if layer_index == 0 else top_k
        with profile_micro_step(
            micro_profiler,
            "cacheblend_repair_plan_build_or_selection",
        ):
            next_indices = select_topk_indices_from_scores(
                candidate_indices=selected_for_layer,
                scores=deviation_scores,
                top_k=next_top_k,
            )
            online_deviation_selected_next_by_layer[layer_index] = next_indices
            current_hidden_states = select_hidden_states_for_indices(
                hidden_states=next_hidden_states,
                current_indices=selected_for_layer,
                target_indices=next_indices,
            )
            current_indices = next_indices

    if micro_profiler is not None:
        micro_profiler.add_aggregate_record(
            name="cacheblend_partial_layer_loop_total",
            child_records=micro_profiler.records[loop_record_start:],
            latency_seconds=time.perf_counter() - loop_start,
        )
    repair_latency_seconds = time.perf_counter() - repair_start
    with profile_micro_step(micro_profiler, "cacheblend_repair_plan_build_or_selection"):
        plan_metadata = {
            "diagnostic_source": "online_fresh_vs_reuse",
            "uses_full_recompute_reference": False,
            "selection_algorithm": "online_gradual_hkvd",
            "initial_top_k": initial_top_k,
            "top_k": top_k,
            "rope_position_correction_applied": expected_rope_correction,
            "reuse_doc_kv_position_basis": (
                "full_context_absolute" if expected_rope_correction else "native"
            ),
            "online_initial_active_set": "all_doc_tokens",
            "online_deviation_metric": "fresh_kv_vs_reused_kv_l2",
            "online_deviation_max_by_layer": online_deviation_max_by_layer,
            "online_deviation_mean_by_layer": online_deviation_mean_by_layer,
            "online_deviation_selected_next_by_layer": online_deviation_selected_next_by_layer,
        }
        plan = build_cacheblend_repair_plan(
            selected_indices_by_layer=selected_indices_by_layer,
            doc_total_len=doc_total_len,
            num_layers=num_layers,
            metadata=plan_metadata,
            strategy="online_gradual_hkvd",
            runtime_selection_mode="online_gradual_selected_indices_by_layer",
        )
    with profile_micro_step(micro_profiler, "cacheblend_finalize_repaired_kv"):
        repaired_past_key_values = tuple(patched_past_key_values)
    metadata = {
        "model_family": model_family,
        "execution_mode": "online_gradual_hkvd",
        "execution_uses_full_recompute_reference": False,
        "planning_included_in_total_latency": False,
        "selected_initial_hidden_source": "model_embedding_path",
        "reuse_past_key_values_source": reuse_past_key_values_source,
        "rope_position_correction_applied": expected_rope_correction,
        "reuse_precompute_latency_seconds": reuse_precompute_latency_seconds,
        "repair_latency_seconds": repair_latency_seconds,
    }
    metadata.update(plan.to_metadata())
    attach_micro_profile_metadata(
        metadata,
        micro_profiler,
        build_repair_micro_summary_metadata(
            doc_total_len=doc_total_len,
            runtime_selected_count=len(plan.runtime_selected_indices),
            num_layers=num_layers,
            layer_selected_counts=plan.layer_selected_counts,
            repair_plan_strategy=plan.strategy,
            runtime_selection_mode=plan.runtime_selection_mode,
        ),
    )
    return CacheBlendOnlinePartialRepairResult(
        repair_plan=plan,
        final_selected_hidden_states=current_hidden_states,
        repaired_past_key_values=repaired_past_key_values,
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


def run_cacheblend_style_online_repair_generation(
    model: Any,
    tokenizer: Any,
    tokenized_example: TokenizedExample,
    max_new_tokens: int = 16,
    initial_top_k: int = 10,
    top_k: int = 5,
    model_family: str = "gpt2",
    reuse_past_key_values: Any | None = None,
) -> CacheBlendRepairResult:
    """Run online gradual-HKVD repair and decode without full-reference planning."""

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    total_start = time.perf_counter()
    partial_repair = run_cacheblend_style_online_partial_repair(
        model=model,
        tokenized_example=tokenized_example,
        initial_top_k=initial_top_k,
        top_k=top_k,
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
            "execution_mode": "online_gradual_hkvd",
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
    repair_planner: str = REPAIR_PLANNER_ORACLE_HKVD,
) -> CacheBlendRepairResult:
    """Run CacheBlend-style repaired-KV greedy generation.

    By default this keeps the oracle-HKVD diagnostic behavior for backward
    compatibility. Use repair_planner="online_gradual_hkvd" for the non-oracle
    measured baseline path that does not compute full-reference KV.
    """

    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if initial_top_k <= 0 or top_k <= 0:
        raise ValueError("initial_top_k and top_k must be positive.")

    model_family = model_family.lower()
    repair_planner = repair_planner.lower()
    if repair_planner not in SUPPORTED_REPAIR_PLANNERS:
        raise ValueError(
            f"Unsupported repair_planner={repair_planner!r}. "
            f"Supported planners: {sorted(SUPPORTED_REPAIR_PLANNERS)}."
        )
    if repair_planner == REPAIR_PLANNER_ONLINE_GRADUAL_HKVD:
        if include_repair_diagnostics:
            raise ValueError(
                "include_repair_diagnostics=True requires oracle_hkvd planning. "
                "Run oracle diagnostics separately from measured online repair."
            )
        return run_cacheblend_style_online_repair_generation(
            model=model,
            tokenizer=tokenizer,
            tokenized_example=tokenized_example,
            max_new_tokens=max_new_tokens,
            initial_top_k=initial_top_k,
            top_k=top_k,
            model_family=model_family,
        )

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
