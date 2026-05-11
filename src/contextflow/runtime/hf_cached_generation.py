from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.kv_cache.precompute import infer_model_input_device


@dataclass(slots=True)
class HFCachedGenerationResult:
    generated_ids: list[int]
    output_text: str
    final_past_key_values: Any


def infer_past_sequence_length(past_key_values: Any) -> int:
    if not past_key_values:
        return 0
    if hasattr(past_key_values, "get_seq_length"):
        return int(past_key_values.get_seq_length())
    if hasattr(past_key_values, "key_cache") and past_key_values.key_cache:
        return int(past_key_values.key_cache[0].shape[-2])
    if hasattr(past_key_values, "layers") and past_key_values.layers:
        first_layer = past_key_values.layers[0]
        if hasattr(first_layer, "keys") and first_layer.keys is not None:
            return int(first_layer.keys.shape[-2])
    return int(past_key_values[0][0].shape[-2])


def convert_legacy_cache_for_model(model: Any, past_key_values: Any) -> Any:
    """Convert legacy tuple cache to the Cache object expected by newer Transformers models."""

    if hasattr(past_key_values, "get_seq_length"):
        return past_key_values
    if not isinstance(past_key_values, (tuple, list)):
        return past_key_values

    from transformers import DynamicCache

    if hasattr(DynamicCache, "from_legacy_cache"):
        return DynamicCache.from_legacy_cache(tuple(past_key_values))

    try:
        cache = DynamicCache(config=model.config)
    except TypeError:
        cache = DynamicCache()

    for layer_index, layer_past in enumerate(past_key_values):
        key_states, value_states = layer_past[:2]
        cache.update(key_states, value_states, layer_index)
    return cache


def forward_with_optional_cache_position(model: Any, **kwargs: Any) -> Any:
    try:
        return model(**kwargs)
    except TypeError as error:
        if "cache_position" not in str(error):
            raise
        kwargs.pop("cache_position", None)
        return model(**kwargs)


def generate_with_past_key_values(
    model: Any,
    tokenizer: Any,
    q_ids: list[int],
    past_key_values: Any,
    max_new_tokens: int = 16,
) -> HFCachedGenerationResult:
    """Reference greedy decoding loop from precomputed document past_key_values."""

    import torch

    if not q_ids:
        raise ValueError("q_ids must be non-empty.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    device = infer_model_input_device(model)
    past_key_values = convert_legacy_cache_for_model(model, past_key_values)
    past_len = infer_past_sequence_length(past_key_values)

    current_input = torch.tensor([q_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, past_len + len(q_ids)), dtype=torch.long, device=device)
    cache_position = torch.arange(
        past_len,
        past_len + len(q_ids),
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        generated_ids: list[int] = []
        for _ in range(max_new_tokens):
            outputs = forward_with_optional_cache_position(
                model,
                input_ids=current_input,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                cache_position=cache_position,
                use_cache=True,
            )
            if outputs.past_key_values is None:
                raise RuntimeError("Model did not return past_key_values.")
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

            token_id = int(next_token.item())
            generated_ids.append(token_id)

            current_input = next_token
            past_len = infer_past_sequence_length(past_key_values)
            attention_mask = torch.ones((1, past_len + 1), dtype=torch.long, device=device)
            cache_position = torch.tensor([past_len], dtype=torch.long, device=device)

    return HFCachedGenerationResult(
        generated_ids=generated_ids,
        output_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        final_past_key_values=past_key_values,
    )
