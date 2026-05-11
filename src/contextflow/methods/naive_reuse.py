from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.schema import TokenizedExample
from contextflow.kv_cache import ChunkKV, assemble_chunk_kvs, precompute_doc_chunk_kvs
from contextflow.runtime.hf_cached_generation import (
    HFCachedGenerationResult,
    generate_with_past_key_values,
)


@dataclass(slots=True)
class NaiveReuseResult:
    chunk_kvs: list[ChunkKV]
    assembled_past_key_values: Any
    generation: HFCachedGenerationResult


def run_naive_reuse_generation(
    example: TokenizedExample,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 16,
) -> NaiveReuseResult:
    """Run the lower-bound baseline that directly reuses concatenated doc chunk KVs."""

    chunk_kvs = precompute_doc_chunk_kvs(model, example)
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
    )
