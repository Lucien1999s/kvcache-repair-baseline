from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from contextflow.kv_cache.precompute import infer_model_input_device, normalize_past_key_values


@dataclass(slots=True)
class FusionRAGQuerySelectionResult:
    """Query-guided critical-token selection for FusionRAG-style repair."""

    selected_indices: list[int]
    selected_indices_by_score: list[int]
    selected_scores: list[float]
    token_scores: Any
    recompute_ratio: float
    recompute_count: int
    metadata: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "selection_algorithm": "fusionrag_query_guided_selection",
            "selected_indices": list(self.selected_indices),
            "selected_indices_by_score": list(self.selected_indices_by_score),
            "selected_scores": list(self.selected_scores),
            "recompute_ratio": self.recompute_ratio,
            "recompute_count": self.recompute_count,
            **dict(self.metadata),
        }


def validate_recompute_ratio(recompute_ratio: float) -> None:
    if recompute_ratio <= 0.0 or recompute_ratio > 1.0:
        raise ValueError("recompute_ratio must be in the range (0, 1].")


def resolve_recompute_count(doc_token_count: int, recompute_ratio: float) -> int:
    validate_recompute_ratio(recompute_ratio)
    if doc_token_count <= 0:
        raise ValueError("doc_token_count must be positive.")
    return max(1, min(doc_token_count, math.ceil(doc_token_count * recompute_ratio)))


def validate_doc_chunk_lengths(doc_chunk_lengths: list[int], doc_token_count: int) -> None:
    if not doc_chunk_lengths:
        raise ValueError("doc_chunk_lengths must contain at least one chunk length.")
    invalid = [length for length in doc_chunk_lengths if length <= 0]
    if invalid:
        raise ValueError(f"doc_chunk_lengths must be positive; invalid values: {invalid}.")
    if sum(doc_chunk_lengths) != doc_token_count:
        raise ValueError(
            "sum(doc_chunk_lengths) must match document KV sequence length; "
            f"got {sum(doc_chunk_lengths)} and {doc_token_count}."
        )


def select_top_recompute_indices_from_scores(
    token_scores: Any,
    recompute_ratio: float,
) -> tuple[list[int], list[int], list[float]]:
    if token_scores.ndim != 1:
        raise ValueError(f"token_scores must be 1D, got shape {tuple(token_scores.shape)}.")
    recompute_count = resolve_recompute_count(
        doc_token_count=int(token_scores.shape[0]),
        recompute_ratio=recompute_ratio,
    )
    score_values = [float(score) for score in token_scores.detach().cpu().tolist()]
    ranked = sorted(
        enumerate(score_values),
        key=lambda item: (-item[1], item[0]),
    )
    selected_items = ranked[:recompute_count]
    selected_indices_by_score = [int(index) for index, _ in selected_items]
    selected_scores = [float(score) for _, score in selected_items]
    return sorted(selected_indices_by_score), selected_indices_by_score, selected_scores


def get_final_layer_input_hidden_states(outputs: Any, num_layers: int) -> Any:
    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("Model did not return hidden_states for FusionRAG QGS.")
    if len(hidden_states) <= num_layers - 1:
        raise RuntimeError(
            "Model hidden_states length is shorter than expected for final layer input; "
            f"got {len(hidden_states)} and num_layers={num_layers}."
        )
    return hidden_states[num_layers - 1]


def compute_gpt2_final_layer_query(
    model: Any,
    final_layer_input_hidden_states: Any,
    final_layer_index: int,
) -> tuple[Any, float]:
    from contextflow.repair.adapters.gpt2 import (
        compute_gpt2_layer_query,
        get_gpt2_layer,
        validate_gpt2_like_model,
    )

    validate_gpt2_like_model(model)
    layer = get_gpt2_layer(model, final_layer_index)
    query = compute_gpt2_layer_query(layer, final_layer_input_hidden_states)
    scale = 1.0
    if getattr(layer.attn, "scale_attn_weights", True):
        scale = 1.0 / math.sqrt(float(query.shape[-1]))
    if getattr(layer.attn, "scale_attn_by_inverse_layer_idx", False):
        layer_idx = getattr(layer.attn, "layer_idx", None)
        if layer_idx is None:
            raise ValueError("GPT2 attention inverse-layer scaling requested but layer_idx is unset.")
        scale = scale / float(layer_idx + 1)
    return query, scale


def compute_mistral_final_layer_query(
    model: Any,
    final_layer_input_hidden_states: Any,
    final_layer_index: int,
    query_positions: list[int],
) -> tuple[Any, float]:
    from contextflow.repair.adapters.mistral import (
        compute_mistral_layer_query,
        get_mistral_layer,
        infer_mistral_attention_geometry,
        validate_mistral_like_model,
    )
    from contextflow.repair.adapters.mistral.rope import (
        compute_mistral_rotary_embeddings_for_positions,
    )
    from contextflow.repair.adapters.rope import apply_rope

    validate_mistral_like_model(model)
    layer = get_mistral_layer(model, final_layer_index)
    query = compute_mistral_layer_query(layer, final_layer_input_hidden_states)
    if len(query_positions) != int(query.shape[-2]):
        raise ValueError(
            "query_positions length must match query sequence length; "
            f"got {len(query_positions)} and {query.shape[-2]}."
        )
    cos, sin = compute_mistral_rotary_embeddings_for_positions(
        model=model,
        layer=layer,
        reference_tensor=query,
        positions=query_positions,
    )
    query = apply_rope(query, cos, sin)
    _, _, _, head_dim = infer_mistral_attention_geometry(layer)
    scale = getattr(layer.self_attn, "scaling", None)
    if scale is None:
        scale = 1.0 / math.sqrt(float(head_dim))
    return query, float(scale)


def compute_qwen2_final_layer_query(
    model: Any,
    final_layer_input_hidden_states: Any,
    final_layer_index: int,
    query_positions: list[int],
) -> tuple[Any, float]:
    from contextflow.repair.adapters.qwen2 import (
        compute_qwen2_layer_query,
        get_qwen2_layer,
        infer_qwen2_attention_geometry,
        validate_qwen2_like_model,
    )
    from contextflow.repair.adapters.qwen2.rope import (
        compute_qwen2_rotary_embeddings_for_positions,
    )
    from contextflow.repair.adapters.rope import apply_rope

    validate_qwen2_like_model(model)
    layer = get_qwen2_layer(model, final_layer_index)
    query = compute_qwen2_layer_query(layer, final_layer_input_hidden_states)
    if len(query_positions) != int(query.shape[-2]):
        raise ValueError(
            "query_positions length must match query sequence length; "
            f"got {len(query_positions)} and {query.shape[-2]}."
        )
    cos, sin = compute_qwen2_rotary_embeddings_for_positions(
        model=model,
        layer=layer,
        reference_tensor=query,
        positions=query_positions,
    )
    query = apply_rope(query, cos, sin)
    _, _, _, head_dim = infer_qwen2_attention_geometry(layer)
    scale = getattr(layer.self_attn, "scaling", None)
    if scale is None:
        scale = 1.0 / math.sqrt(float(head_dim))
    return query, float(scale)


def compute_model_family_final_layer_query(
    model: Any,
    q_ids: list[int],
    model_family: str,
    num_layers: int,
    query_position_offset: int = 0,
) -> tuple[Any, float]:
    """Run query-only prefill and return final-layer attention query states."""

    import torch

    if not q_ids:
        raise ValueError("q_ids must be non-empty.")
    if num_layers <= 0:
        raise ValueError("num_layers must be positive.")
    if query_position_offset < 0:
        raise ValueError("query_position_offset must be non-negative.")

    device = infer_model_input_device(model)
    input_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)
    position_ids = torch.arange(
        query_position_offset,
        query_position_offset + len(q_ids),
        dtype=torch.long,
        device=device,
    ).unsqueeze(0)
    with torch.inference_mode():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
            use_cache=False,
        )
    final_layer_input_hidden_states = get_final_layer_input_hidden_states(
        outputs,
        num_layers=num_layers,
    )
    final_layer_index = num_layers - 1
    query_positions = position_ids.squeeze(0).tolist()
    normalized_model_family = model_family.lower()
    if normalized_model_family == "gpt2":
        return compute_gpt2_final_layer_query(
            model,
            final_layer_input_hidden_states,
            final_layer_index,
        )
    if normalized_model_family == "mistral":
        return compute_mistral_final_layer_query(
            model,
            final_layer_input_hidden_states,
            final_layer_index,
            query_positions=query_positions,
        )
    if normalized_model_family == "qwen2":
        return compute_qwen2_final_layer_query(
            model,
            final_layer_input_hidden_states,
            final_layer_index,
            query_positions=query_positions,
        )
    raise NotImplementedError(
        f"Unsupported model_family={model_family!r}. FusionRAG QGS currently "
        "supports model_family in {'gpt2', 'mistral', 'qwen2'}."
    )


def repeat_kv_heads_to_match_query(key_states: Any, query_heads: int) -> Any:
    if int(key_states.shape[1]) == query_heads:
        return key_states
    if query_heads % int(key_states.shape[1]) != 0:
        raise ValueError(
            "query head count must be divisible by key/value head count; "
            f"got {query_heads} and {key_states.shape[1]}."
        )
    repeat_count = query_heads // int(key_states.shape[1])
    batch_size, kv_heads, seq_len, head_dim = key_states.shape
    key_states = key_states[:, :, None, :, :].expand(
        batch_size,
        kv_heads,
        repeat_count,
        seq_len,
        head_dim,
    )
    return key_states.reshape(batch_size, kv_heads * repeat_count, seq_len, head_dim)


def compute_query_guided_token_scores(
    query_states: Any,
    doc_final_layer_key: Any,
    doc_chunk_lengths: list[int],
    attention_scale: float,
) -> Any:
    """Score document tokens from final-layer query-key attention weights.

    Scores are computed per chunk, matching FusionRAG's QGS description: multiply
    query Q by each chunk's K, form attention weights over that chunk, then sum
    each token column over query tokens and heads.
    """

    import torch

    if query_states.ndim != 4:
        raise ValueError(
            f"query_states must have shape [batch, heads, q_len, head_dim], got {query_states.shape}."
        )
    if doc_final_layer_key.ndim != 4:
        raise ValueError(
            "doc_final_layer_key must have shape [batch, heads, doc_len, head_dim], "
            f"got {doc_final_layer_key.shape}."
        )
    if int(query_states.shape[0]) != int(doc_final_layer_key.shape[0]):
        raise ValueError("query and document key batch size must match.")
    if int(query_states.shape[-1]) != int(doc_final_layer_key.shape[-1]):
        raise ValueError("query and document key head_dim must match.")

    doc_len = int(doc_final_layer_key.shape[-2])
    validate_doc_chunk_lengths(doc_chunk_lengths, doc_token_count=doc_len)
    attention_key = repeat_kv_heads_to_match_query(
        doc_final_layer_key,
        query_heads=int(query_states.shape[1]),
    )

    chunk_scores = []
    offset = 0
    for chunk_len in doc_chunk_lengths:
        chunk_key = attention_key[..., offset : offset + chunk_len, :]
        raw_scores = torch.matmul(
            query_states.float(),
            chunk_key.float().transpose(-1, -2),
        ) * float(attention_scale)
        attention_weights = torch.nn.functional.softmax(raw_scores, dim=-1)
        token_scores = attention_weights.sum(dim=(0, 1, 2))
        chunk_scores.append(token_scores)
        offset += chunk_len

    return torch.cat(chunk_scores, dim=0)


def select_fusionrag_query_guided_tokens(
    model: Any,
    q_ids: list[int],
    doc_past_key_values: Any,
    doc_chunk_lengths: list[int],
    recompute_ratio: float = 0.15,
    model_family: str = "gpt2",
    query_position_offset: int | None = None,
) -> FusionRAGQuerySelectionResult:
    """Select FusionRAG critical tokens using final-layer query-key attention."""

    normalized_past_key_values = normalize_past_key_values(doc_past_key_values)
    if not normalized_past_key_values:
        raise ValueError("doc_past_key_values must contain at least one layer.")
    final_layer_key = normalized_past_key_values[-1][0]
    doc_token_count = int(final_layer_key.shape[-2])
    validate_doc_chunk_lengths(doc_chunk_lengths, doc_token_count=doc_token_count)
    resolved_query_position_offset = (
        doc_token_count if query_position_offset is None else query_position_offset
    )

    query_states, attention_scale = compute_model_family_final_layer_query(
        model=model,
        q_ids=q_ids,
        model_family=model_family,
        num_layers=len(normalized_past_key_values),
        query_position_offset=resolved_query_position_offset,
    )
    token_scores = compute_query_guided_token_scores(
        query_states=query_states,
        doc_final_layer_key=final_layer_key,
        doc_chunk_lengths=doc_chunk_lengths,
        attention_scale=attention_scale,
    )
    selected_indices, selected_indices_by_score, selected_scores = (
        select_top_recompute_indices_from_scores(
            token_scores,
            recompute_ratio=recompute_ratio,
        )
    )
    recompute_count = len(selected_indices)
    return FusionRAGQuerySelectionResult(
        selected_indices=selected_indices,
        selected_indices_by_score=selected_indices_by_score,
        selected_scores=selected_scores,
        token_scores=token_scores,
        recompute_ratio=recompute_ratio,
        recompute_count=recompute_count,
        metadata={
            "model_family": model_family.lower(),
            "doc_token_count": doc_token_count,
            "doc_chunk_lengths": list(doc_chunk_lengths),
            "query_token_count": len(q_ids),
            "query_source": "query_only_final_layer_prefill",
            "query_position_offset": resolved_query_position_offset,
            "query_position_basis": "document_absolute",
            "score_source": "final_layer_query_key_attention_weights",
            "attention_score_normalization": "per_chunk_softmax",
            "attention_scale": float(attention_scale),
        },
    )
