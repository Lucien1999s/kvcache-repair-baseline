from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.assembly import AssembledInput, assemble_token_aligned_full_prefill_input
from contextflow.data.schema import TokenizedExample
from contextflow.runtime.hf_greedy_generation import (
    HFGreedyGenerationResult,
    generate_greedy_from_input_ids,
)


@dataclass(slots=True)
class FullRecomputeGreedyResult:
    assembled: AssembledInput
    generation: HFGreedyGenerationResult


def run_token_aligned_full_recompute_greedy_generation(
    example: TokenizedExample,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int = 16,
) -> FullRecomputeGreedyResult:
    """Run full-prefill greedy generation on concat(doc_chunk_ids) + q_ids."""

    assembled = assemble_token_aligned_full_prefill_input(
        example,
        tokenizer=tokenizer,
    )
    generation = generate_greedy_from_input_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=assembled.input_ids,
        max_new_tokens=max_new_tokens,
    )
    return FullRecomputeGreedyResult(
        assembled=assembled,
        generation=generation,
    )
