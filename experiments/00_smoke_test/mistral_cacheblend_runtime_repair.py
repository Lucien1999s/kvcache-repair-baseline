from __future__ import annotations

import argparse
from typing import Any

import torch

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.methods.cacheblend_repair import run_cacheblend_style_repair_generation
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "mistral-sample-0",
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test Mistral CacheBlend-style runtime repair."
    )
    parser.add_argument("--input", default=None, help="Optional CacheBlend-style JSON list.")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace Mistral causal LM name or local path.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Greedy decode length.")
    parser.add_argument("--initial-top-k", type=int, default=10, help="Gradual HKVD initial top-k.")
    parser.add_argument("--top-k", type=int, default=5, help="Gradual HKVD follow-up top-k.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype passed to model load.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to model load.")
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def main() -> None:
    args = parse_args()
    model_family = "mistral"

    try:
        examples = load_examples(args.input)
        if not examples:
            raise ValueError("Expected at least one input example.")

        bundle = load_hf_causal_lm(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        bundle.model.eval()

        for example_index, example in enumerate(examples[: args.limit]):
            prompt = build_cacheblend_prompt(example)
            tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
            with torch.inference_mode():
                result = run_cacheblend_style_repair_generation(
                    model=bundle.model,
                    tokenizer=bundle.tokenizer,
                    tokenized_example=tokenized,
                    max_new_tokens=args.max_new_tokens,
                    initial_top_k=args.initial_top_k,
                    top_k=args.top_k,
                    model_family=model_family,
                )

            assert result.generated_ids, "generated_ids must be non-empty."
            assert result.metadata["runtime_selected_indices"], (
                "runtime_selected_indices must be non-empty."
            )
            assert "layer_selected_counts" in result.metadata, (
                "metadata must include layer_selected_counts."
            )
            assert (
                result.metadata["runtime_selection_mode"]
                == "fixed_layer0_selected_indices"
            ), "runtime_selection_mode must describe the fixed layer-0 selection path."
            assert result.metadata["repair_latency_seconds"] >= 0, (
                "repair_latency_seconds must be non-negative."
            )
            assert result.metadata["decode_latency_seconds"] >= 0, (
                "decode_latency_seconds must be non-negative."
            )

            print(f"[example {example_index}]")
            print(f"example_id: {example.example_id}")
            print(f"model_name: {args.model}")
            print(f"model_family: {model_family}")
            print(f"generated_ids: {result.generated_ids}")
            print(f"generated_text: {result.output_text!r}")
            print(f"metadata: {result.metadata}")
    except Exception as error:
        print(f"exception_type: {type(error).__name__}")
        print(f"exception_message: {error}")
        raise


if __name__ == "__main__":
    main()
