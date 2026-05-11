from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Callable

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.methods.cacheblend_repair import run_cacheblend_style_repair_generation
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.methods.naive_reuse import run_naive_reuse_generation
from contextflow.repair.adapters.gpt2 import validate_gpt2_like_model
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, object]] = [
    {
        "id": "sample-0",
        "question": "Who wrote the novel Pride and Prejudice",
        "answers": ["Jane Austen"],
        "ctxs": [
            {
                "title": "Pride and Prejudice",
                "text": "Pride and Prejudice is a novel by Jane Austen.",
            },
            {
                "title": "Jane Austen",
                "text": "Jane Austen was an English novelist known for six major novels.",
            },
        ],
    }
]


@dataclass(slots=True)
class MethodResult:
    name: str
    generated_ids: list[int]
    generated_text: str
    latency_seconds: float
    note: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal GPT2 CacheBlend-style reproduction runner."
    )
    parser.add_argument("--input", default=None, help="Optional CacheBlend-style JSON list.")
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace GPT2-like causal LM name or local path.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Greedy decode length.")
    parser.add_argument("--initial-top-k", type=int, default=10, help="Gradual HKVD initial top-k.")
    parser.add_argument("--top-k", type=int, default=5, help="Gradual HKVD follow-up top-k.")
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def timed_call(
    name: str,
    fn: Callable[[], tuple[list[int], str, str | None]],
) -> MethodResult:
    start = time.perf_counter()
    generated_ids, generated_text, note = fn()
    latency_seconds = time.perf_counter() - start
    return MethodResult(
        name=name,
        generated_ids=generated_ids,
        generated_text=generated_text,
        latency_seconds=latency_seconds,
        note=note,
    )


def print_method_result(result: MethodResult, full_generated_ids: list[int]) -> None:
    print(f"method: {result.name}")
    print(f"generated token ids: {result.generated_ids}")
    print(f"generated text: {result.generated_text!r}")
    print(f"total latency seconds: {result.latency_seconds:.6f}")
    print(f"token ids match Full Recompute: {result.generated_ids == full_generated_ids}")
    if result.note is not None:
        print(f"note: {result.note}")


def main() -> None:
    args = parse_args()
    examples = load_examples(args.input)
    if not examples:
        raise ValueError("Expected at least one input example.")

    bundle = load_hf_causal_lm(args.model)
    validate_gpt2_like_model(bundle.model)

    for example_index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)

        print(f"[example {example_index}]")
        print(f"example_id: {example.example_id}")

        full_result = timed_call(
            "Full Recompute",
            lambda: (
                lambda result: (
                    result.generation.generated_ids,
                    result.generation.output_text,
                    None,
                )
            )(
                run_token_aligned_full_recompute_greedy_generation(
                    tokenized,
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
            ),
        )
        full_generated_ids = full_result.generated_ids
        print_method_result(full_result, full_generated_ids)

        naive_result = timed_call(
            "Naive KV Reuse",
            lambda: (
                lambda result: (
                    result.generation.generated_ids,
                    result.generation.output_text,
                    None,
                )
            )(
                run_naive_reuse_generation(
                    tokenized,
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    max_new_tokens=args.max_new_tokens,
                )
            ),
        )
        print_method_result(naive_result, full_generated_ids)

        repair_result = timed_call(
            "CacheBlend-style Repair",
            lambda: (
                lambda result: (
                    result.generated_ids,
                    result.output_text,
                    "entered decode with repaired doc KV; "
                    f"runtime_selected_count={len(result.metadata['runtime_selected_indices'])}; "
                    f"layer_selected_counts={result.metadata['layer_selected_counts']}; "
                    f"repair_latency_seconds={result.metadata['repair_latency_seconds']:.6f}; "
                    f"decode_latency_seconds={result.metadata['decode_latency_seconds']:.6f}; "
                    f"method_total_latency_seconds={result.metadata['total_latency_seconds']:.6f}",
                )
            )(
                run_cacheblend_style_repair_generation(
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    tokenized_example=tokenized,
                    max_new_tokens=args.max_new_tokens,
                    initial_top_k=args.initial_top_k,
                    top_k=args.top_k,
                    model_family="gpt2",
                )
            ),
        )
        print_method_result(repair_result, full_generated_ids)


if __name__ == "__main__":
    main()
