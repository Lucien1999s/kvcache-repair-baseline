from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch

from contextflow.data import (
    build_cacheblend_prompt,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.evaluation import evaluate_qa_prediction, parse_cacheblend_generation
from contextflow.methods.fusionrag_repair import run_fusionrag_style_repair_generation
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "fusionrag-method-smoke-0",
        "question": "Who wrote the novel Pride and Prejudice",
        "answers": ["Jane Austen"],
        "ctxs": [
            {
                "title": "Pride and Prejudice",
                "text": "Pride and Prejudice is a novel by Jane Austen.",
            },
            {
                "title": "Jane Austen",
                "text": "Jane Austen wrote novels including Pride and Prejudice.",
            },
            {
                "title": "Charles Dickens",
                "text": "Charles Dickens wrote Oliver Twist and Great Expectations.",
            },
        ],
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test FusionRAG-style repair generation on one inline QA example."
    )
    parser.add_argument("--model", default="sshleifer/tiny-gpt2")
    parser.add_argument("--model-family", default="gpt2", choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--neighbor-top-n", type=int, default=1)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    return parser.parse_args()


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def main() -> None:
    args = parse_args()
    example = parse_input_examples(INLINE_SAMPLE)[0]
    bundle = load_hf_causal_lm(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=normalize_optional_device_map(args.device_map),
    )
    bundle.model.eval()

    prompt = build_cacheblend_prompt(
        example,
        prompt_policy="cacheblend_qa",
        dataset="musique",
    )
    tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
    doc_token_count = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)
    expected_selected_count = max(
        1,
        min(doc_token_count, math.ceil(doc_token_count * args.recompute_ratio)),
    )

    with torch.inference_mode():
        result = run_fusionrag_style_repair_generation(
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            tokenized_example=tokenized,
            max_new_tokens=args.max_new_tokens,
            neighbor_top_n=args.neighbor_top_n,
            recompute_ratio=args.recompute_ratio,
            model_family=args.model_family,
        )

    evaluated_prediction = parse_cacheblend_generation(result.output_text)
    metrics = evaluate_qa_prediction(evaluated_prediction, example.answers)
    metadata = result.metadata

    assert result.generated_ids, "FusionRAG method must generate at least one token."
    assert metadata["method"] == "fusionrag_repair"
    assert metadata["execution_uses_full_recompute_reference"] is False
    assert metadata["neighbor_top_n"] == args.neighbor_top_n
    assert metadata["recompute_ratio"] == args.recompute_ratio
    assert metadata["selected_count"] == expected_selected_count
    assert len(metadata["selected_indices"]) == expected_selected_count
    assert metadata["runtime_selection_mode"] == "fusionrag_query_guided_fixed_selected_indices"
    assert metadata["repair_plan_strategy"] == "fusionrag_query_guided_fixed"
    assert metadata["enriched_precompute_latency_seconds"] >= 0
    assert metadata["selection_latency_seconds"] >= 0
    assert metadata["repair_latency_seconds"] >= 0
    assert metadata["decode_latency_seconds"] >= 0
    assert metadata["total_latency_seconds"] >= 0
    assert metadata["query_selection_metadata"]["query_position_offset"] == doc_token_count
    assert metadata["partial_repair_metadata"]["execution_uses_full_recompute_reference"] is False

    summary: dict[str, Any] = {
        "example_id": example.example_id,
        "model": args.model,
        "model_family": args.model_family,
        "generated_ids": result.generated_ids,
        "raw_generated_text": result.output_text,
        "evaluated_prediction": evaluated_prediction,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
        "metadata": {
            "neighbor_top_n": metadata["neighbor_top_n"],
            "recompute_ratio": metadata["recompute_ratio"],
            "selected_count": metadata["selected_count"],
            "selected_indices": metadata["selected_indices"],
            "enriched_precompute_latency_seconds": metadata[
                "enriched_precompute_latency_seconds"
            ],
            "selection_latency_seconds": metadata["selection_latency_seconds"],
            "repair_latency_seconds": metadata["repair_latency_seconds"],
            "decode_latency_seconds": metadata["decode_latency_seconds"],
            "total_latency_seconds": metadata["total_latency_seconds"],
            "repair_plan_strategy": metadata["repair_plan_strategy"],
            "runtime_selection_mode": metadata["runtime_selection_mode"],
        },
    }
    print("smoke_fusionrag_method passed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
