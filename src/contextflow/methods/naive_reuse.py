from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache import (
    ChunkKV,
    assemble_chunk_kvs,
    correct_doc_chunk_kvs_for_model_family,
    precompute_doc_chunk_kvs,
    rope_position_correction_enabled,
)
from contextflow.runtime.hf_cached_generation import (
    HFCachedGenerationResult,
    generate_with_past_key_values,
)


@dataclass(slots=True)
class NaiveReuseResult:
    chunk_kvs: list[ChunkKV]
    assembled_past_key_values: Any
    generation: HFCachedGenerationResult
    rope_position_correction_applied: bool


def run_naive_reuse_generation(
    example: TokenizedExample,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 16,
    model_family: str = "gpt2",
) -> NaiveReuseResult:
    """Run the lower-bound baseline that reuses concatenated doc chunk KVs.

    RoPE model families correct reusable chunk keys into full-context absolute
    positions before assembly; GPT2 keeps the native chunk KV unchanged.
    """

    chunk_kvs = precompute_doc_chunk_kvs(model, example)
    chunk_kvs = correct_doc_chunk_kvs_for_model_family(
        model=model,
        chunk_kvs=chunk_kvs,
        model_family=model_family,
    )
    assembled_past_key_values = assemble_chunk_kvs(chunk_kvs)
    generation = generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=example.q_ids,
        past_key_values=assembled_past_key_values,
        max_new_tokens=max_new_tokens,
    )
    return NaiveReuseResult(
        chunk_kvs=chunk_kvs,
        assembled_past_key_values=assembled_past_key_values,
        generation=generation,
        rope_position_correction_applied=rope_position_correction_enabled(model_family),
    )
