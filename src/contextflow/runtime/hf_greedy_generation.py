from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.kv_cache.precompute import infer_model_input_device
from contextflow.runtime.hf_cached_generation import (
    forward_with_optional_cache_position,
    infer_past_sequence_length,
)


@dataclass(slots=True)
class HFGreedyGenerationResult:
    generated_ids: list[int]
    output_text: str
    full_text: str
    input_ids: list[int]
    final_past_key_values: Any


def generate_greedy_from_input_ids(
    model: Any,
    tokenizer: Any,
    input_ids: list[int],
    max_new_tokens: int = 16,
) -> HFGreedyGenerationResult:
    """Reference full-prefill greedy decoding loop without model.generate()."""

    import torch

    if not input_ids:
        raise ValueError("input_ids must be non-empty.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")

    device = infer_model_input_device(model)
    current_input = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones((1, len(input_ids)), dtype=torch.long, device=device)
    cache_position = torch.arange(0, len(input_ids), dtype=torch.long, device=device)
    past_key_values = None

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

    full_ids = list(input_ids) + generated_ids
    return HFGreedyGenerationResult(
        generated_ids=generated_ids,
        output_text=tokenizer.decode(generated_ids, skip_special_tokens=True),
        full_text=tokenizer.decode(full_ids, skip_special_tokens=True),
        input_ids=list(input_ids),
        final_past_key_values=past_key_values,
    )
