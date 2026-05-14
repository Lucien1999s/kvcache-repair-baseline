from __future__ import annotations

from typing import Any

from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache.types import ChunkKV


def infer_model_input_device(model: Any) -> Any:
    if hasattr(model, "device"):
        return model.device
    return next(model.parameters()).device


def normalize_past_key_values(past_key_values: Any) -> Any:
    """Normalize HF cache objects to legacy tuple-of-tuples format."""

    if isinstance(past_key_values, (tuple, list)):
        return tuple(past_key_values)

    if hasattr(past_key_values, "to_legacy_cache"):
        legacy = past_key_values.to_legacy_cache()
        if isinstance(legacy, (tuple, list)):
            return tuple(legacy)

    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        return tuple(zip(past_key_values.key_cache, past_key_values.value_cache))

    if hasattr(past_key_values, "layers"):
        legacy_layers = []
        for layer in past_key_values.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values"):
                legacy_layers.append((layer.keys, layer.values))
        if legacy_layers:
            return tuple(legacy_layers)

    try:
        legacy = tuple(past_key_values)
    except TypeError:
        legacy = ()
    if legacy:
        return legacy

    raise TypeError(
        "Unsupported past_key_values type. Expected tuple/list, an object with "
        "to_legacy_cache(), a DynamicCache-style object with key_cache/value_cache "
        "or layers, or an iterable cache yielding per-layer key/value pairs; "
        f"got {type(past_key_values).__name__}."
    )


def infer_past_key_values_device(past_key_values: Any) -> str | None:
    if not past_key_values:
        return None
    first_layer = past_key_values[0]
    if not first_layer:
        return None
    first_tensor = first_layer[0]
    return str(first_tensor.device)


def precompute_kv_for_input_ids(
    model: Any,
    input_ids: list[int],
    chunk_id: str | None = None,
) -> ChunkKV:
    """Extract HF/PyTorch past_key_values for one input-id chunk."""

    import torch

    if not input_ids:
        raise ValueError("input_ids must be non-empty.")

    resolved_chunk_id = chunk_id or "chunk-0"
    device = infer_model_input_device(model)
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_tensor)

    with torch.inference_mode():
        outputs = model(
            input_ids=input_tensor,
            attention_mask=attention_mask,
            use_cache=True,
        )

    past_key_values = outputs.past_key_values
    if past_key_values is None:
        raise RuntimeError(
            "Model did not return past_key_values. Make sure use_cache=True is supported."
        )
    past_key_values = normalize_past_key_values(past_key_values)

    return ChunkKV(
        chunk_id=resolved_chunk_id,
        input_ids=list(input_ids),
        past_key_values=past_key_values,
        num_tokens=len(input_ids),
        num_layers=len(past_key_values),
        device=infer_past_key_values_device(past_key_values),
    )


def precompute_doc_chunk_kvs(
    model: Any,
    example: TokenizedExample,
    chunk_id_prefix: str | None = None,
) -> list[ChunkKV]:
    """Precompute reusable KV caches for document chunks only."""

    prefix = chunk_id_prefix or example.example_id or "example"
    return [
        precompute_kv_for_input_ids(
            model=model,
            input_ids=doc_chunk_ids,
            chunk_id=f"{prefix}:doc:{chunk_index}",
        )
        for chunk_index, doc_chunk_ids in enumerate(example.doc_chunk_ids)
    ]
