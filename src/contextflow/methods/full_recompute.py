from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.assembly import (
    AssembledInput,
    CacheBlendMistralAssemblyConfig,
    FullPrefillAssemblyConfig,
    assemble_cacheblend_mistral_input,
    assemble_full_prefill_input_from_prompt,
)
from contextflow.data.schema import PromptExample, TokenizedExample
from contextflow.runtime.hf_generation import (
    HFGenerationConfig,
    HFGenerationResult,
    generate_from_input_ids,
)


@dataclass(slots=True)
class FullRecomputeResult:
    assembled: AssembledInput
    generation: HFGenerationResult


def run_full_recompute_generation(
    example: PromptExample,
    model: Any,
    tokenizer: Any,
    assembly_config: FullPrefillAssemblyConfig | None = None,
    generation_config: HFGenerationConfig | None = None,
) -> FullRecomputeResult:
    """Run the general full-prefill baseline with HF model.generate()."""

    assembled = assemble_full_prefill_input_from_prompt(
        example,
        tokenizer=tokenizer,
        config=assembly_config,
    )
    generation = generate_from_input_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=assembled.input_ids,
        config=generation_config,
    )
    return FullRecomputeResult(
        assembled=assembled,
        generation=generation,
    )


def run_cacheblend_mistral_full_recompute_generation(
    example: TokenizedExample,
    model: Any,
    tokenizer: Any,
    assembly_config: CacheBlendMistralAssemblyConfig | None = None,
    generation_config: HFGenerationConfig | None = None,
) -> FullRecomputeResult:
    """Run the CacheBlend Mistral-7B-Instruct exact reproduction full-prefill path."""

    assembled = assemble_cacheblend_mistral_input(
        example,
        tokenizer=tokenizer,
        config=assembly_config,
    )
    generation = generate_from_input_ids(
        model=model,
        tokenizer=tokenizer,
        input_ids=assembled.input_ids,
        config=generation_config,
    )
    return FullRecomputeResult(
        assembled=assembled,
        generation=generation,
    )
