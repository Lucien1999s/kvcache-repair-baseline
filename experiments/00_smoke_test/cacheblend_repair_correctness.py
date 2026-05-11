from __future__ import annotations

import argparse
from typing import Any

import torch

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.methods.cacheblend_repair import run_cacheblend_style_repair_generation
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "correctness-sample-0",
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
        description="Smoke test CacheBlend-style repair correctness diagnostics."
    )
    parser.add_argument("--model", required=True, help="HuggingFace causal LM name or local path.")
    parser.add_argument(
        "--model-family",
        required=True,
        choices=["gpt2", "mistral", "qwen2"],
        help="Adapter family to use for runtime repair.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16, help="Greedy decode length.")
    parser.add_argument("--initial-top-k", type=int, default=10, help="Gradual HKVD initial top-k.")
    parser.add_argument("--top-k", type=int, default=5, help="Gradual HKVD follow-up top-k.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype passed to model load.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to model load.")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1e-2,
        help="Tolerance for repair correctness diagnostics.",
    )
    return parser.parse_args()


def load_inline_example() -> InputExample:
    return parse_input_examples(INLINE_SAMPLE)[0]


def assert_repair_diagnostics(metadata: dict[str, Any], tolerance: float) -> None:
    generated_mode = metadata["runtime_selection_mode"]
    assert generated_mode == "fixed_layer0_selected_indices", (
        "runtime_selection_mode must describe the fixed layer-0 selection path."
    )
    assert metadata["runtime_selected_indices"], "runtime_selected_indices must be non-empty."

    shape_matches = metadata["repaired_kv_shape_matches_reuse_by_layer"]
    unselected_diffs = metadata["unselected_kv_max_diff_after_vs_reuse_by_layer"]
    selected_before = metadata["selected_kv_max_diff_before_by_layer"]
    selected_after = metadata["selected_kv_max_diff_after_by_layer"]

    assert all(shape_matches.values()), "Every repaired KV layer shape must match reuse KV."
    for layer_index, diff in unselected_diffs.items():
        assert diff <= tolerance, (
            f"Layer {layer_index} unselected K/V changed after repair: {diff} > {tolerance}."
        )
    for layer_index, after_diff in selected_after.items():
        before_diff = selected_before[layer_index]
        assert after_diff <= before_diff + tolerance, (
            f"Layer {layer_index} selected K/V did not improve within tolerance: "
            f"after={after_diff}, before={before_diff}, tolerance={tolerance}."
        )


def main() -> None:
    args = parse_args()

    try:
        bundle = load_hf_causal_lm(
            args.model,
            torch_dtype=args.torch_dtype,
            device_map=args.device_map,
        )
        bundle.model.eval()

        example = load_inline_example()
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
                model_family=args.model_family,
                include_repair_diagnostics=True,
            )

        assert result.generated_ids, "generated_ids must be non-empty."
        assert_repair_diagnostics(result.metadata, tolerance=args.tolerance)

        print(f"model_name: {args.model}")
        print(f"model_family: {args.model_family}")
        print(f"generated_ids: {result.generated_ids}")
        print(f"generated_text: {result.output_text!r}")
        print(f"runtime_selected_indices: {result.metadata['runtime_selected_indices']}")
        print(f"runtime_selection_mode: {result.metadata['runtime_selection_mode']}")
        print(f"layer_selected_counts: {result.metadata['layer_selected_counts']}")
        print(f"repair_latency_seconds: {result.metadata['repair_latency_seconds']}")
        print(f"decode_latency_seconds: {result.metadata['decode_latency_seconds']}")
        print(
            "selected_kv_max_diff_before_by_layer: "
            f"{result.metadata['selected_kv_max_diff_before_by_layer']}"
        )
        print(
            "selected_kv_max_diff_after_by_layer: "
            f"{result.metadata['selected_kv_max_diff_after_by_layer']}"
        )
        print(
            "unselected_kv_max_diff_after_vs_reuse_by_layer: "
            f"{result.metadata['unselected_kv_max_diff_after_vs_reuse_by_layer']}"
        )
        print(
            "repaired_kv_shape_matches_reuse_by_layer: "
            f"{result.metadata['repaired_kv_shape_matches_reuse_by_layer']}"
        )
    except Exception as error:
        print(f"exception_type: {type(error).__name__}")
        print(f"exception_message: {error}")
        raise


if __name__ == "__main__":
    main()
