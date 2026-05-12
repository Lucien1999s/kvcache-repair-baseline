from __future__ import annotations

import argparse
import json
from typing import Any

import torch

from contextflow.data import (
    build_cacheblend_prompt,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.evaluation import (
    compute_normalized_score,
    evaluate_qa_prediction,
    parse_cacheblend_generation,
)
from contextflow.methods.cacheblend_repair import run_cacheblend_style_repair_generation
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.methods.naive_reuse import run_naive_reuse_generation
from contextflow.profiling import timed_call
from contextflow.runtime import load_hf_causal_lm


METHOD_FULL_RECOMPUTE = "full_recompute"
METHOD_NAIVE_REUSE = "naive_reuse"
METHOD_CACHEBLEND_REPAIR = "cacheblend_repair"

INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "cacheblend-methods-smoke-0",
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
        description="Smoke test the three CacheBlend comparison methods on one inline QA example."
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace causal LM name or local path.",
    )
    parser.add_argument(
        "--model-family",
        default="gpt2",
        choices=["gpt2", "mistral", "qwen2"],
        help="Adapter family used by CacheBlend-style repair.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=4)
    parser.add_argument("--initial-top-k", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    return parser.parse_args()


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def evaluate_generated_text(generated_text: str, answers: list[str]) -> dict[str, Any]:
    evaluated_prediction = parse_cacheblend_generation(generated_text)
    metrics = evaluate_qa_prediction(evaluated_prediction, answers)
    return {
        "raw_generated_text": generated_text,
        "generated_text": generated_text,
        "evaluated_prediction": evaluated_prediction,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
    }


def assert_method_record(method_name: str, record: dict[str, Any]) -> None:
    assert record["generated_ids"], f"{method_name} must generate at least one token."
    assert isinstance(record["generated_text"], str), f"{method_name} generated text must be a string."
    assert record["latency_seconds"] >= 0, f"{method_name} latency must be non-negative."
    assert 0.0 <= record["exact_match"] <= 1.0
    assert 0.0 <= record["f1"] <= 1.0


def build_method_record(generated_ids: list[int], generated_text: str, answers: list[str]) -> dict[str, Any]:
    record = {
        "generated_ids": generated_ids,
        **evaluate_generated_text(generated_text, answers),
    }
    return record


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

    method_records: dict[str, dict[str, Any]] = {}

    with torch.inference_mode():
        full_result, full_profile = timed_call(
            lambda: run_token_aligned_full_recompute_greedy_generation(
                tokenized,
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                max_new_tokens=args.max_new_tokens,
            ),
            synchronize_cuda=True,
            collect_peak_memory=True,
        )
        method_records[METHOD_FULL_RECOMPUTE] = build_method_record(
            full_result.generation.generated_ids,
            full_result.generation.output_text,
            example.answers,
        )
        method_records[METHOD_FULL_RECOMPUTE].update(full_profile)

        naive_result, naive_profile = timed_call(
            lambda: run_naive_reuse_generation(
                tokenized,
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                max_new_tokens=args.max_new_tokens,
            ),
            synchronize_cuda=True,
            collect_peak_memory=True,
        )
        method_records[METHOD_NAIVE_REUSE] = build_method_record(
            naive_result.generation.generated_ids,
            naive_result.generation.output_text,
            example.answers,
        )
        method_records[METHOD_NAIVE_REUSE].update(naive_profile)
        method_records[METHOD_NAIVE_REUSE]["assembled_kv_layers"] = len(
            naive_result.assembled_past_key_values
        )

        repair_result, repair_profile = timed_call(
            lambda: run_cacheblend_style_repair_generation(
                model=bundle.model,
                tokenizer=bundle.tokenizer,
                tokenized_example=tokenized,
                max_new_tokens=args.max_new_tokens,
                initial_top_k=args.initial_top_k,
                top_k=args.top_k,
                model_family=args.model_family,
                include_repair_diagnostics=True,
            ),
            synchronize_cuda=True,
            collect_peak_memory=True,
        )
        method_records[METHOD_CACHEBLEND_REPAIR] = build_method_record(
            repair_result.generated_ids,
            repair_result.output_text,
            example.answers,
        )
        method_records[METHOD_CACHEBLEND_REPAIR].update(repair_profile)
        repair_metadata = repair_result.metadata
        method_records[METHOD_CACHEBLEND_REPAIR].update(
            {
                "execution_mode": repair_metadata["execution_mode"],
                "execution_uses_full_recompute_reference": repair_metadata[
                    "execution_uses_full_recompute_reference"
                ],
                "execution_latency_seconds": repair_metadata["execution_latency_seconds"],
                "repair_latency_seconds": repair_metadata["repair_latency_seconds"],
                "decode_latency_seconds": repair_metadata["decode_latency_seconds"],
                "runtime_selected_count": len(repair_metadata["runtime_selected_indices"]),
                "runtime_selection_mode": repair_metadata["runtime_selection_mode"],
                "layer_selected_counts": repair_metadata["layer_selected_counts"],
            }
        )

    for method_name, record in method_records.items():
        assert_method_record(method_name, record)

    assert repair_metadata["runtime_selected_indices"], "repair must select runtime token indices."
    assert repair_metadata["repair_plan_strategy"] == "oracle_hkvd_gradual"
    assert repair_metadata["repair_plan_metadata"]["uses_full_recompute_reference"] is True
    assert repair_metadata["repair_plan_metadata"]["selection_algorithm"] == "gradual_hkvd"
    assert repair_metadata["repair_plan_metadata"]["planning_latency_seconds"] >= 0
    assert repair_metadata["execution_mode"] == "from_plan"
    assert repair_metadata["execution_uses_full_recompute_reference"] is False
    assert repair_metadata["selected_initial_hidden_source"] == "model_embedding_path"
    assert repair_metadata["execution_latency_seconds"] >= 0
    assert repair_metadata["planning_included_in_total_latency"] is True
    assert repair_metadata["layer_selected_counts"], "repair must report selected counts by layer."
    assert repair_metadata["runtime_selection_mode"] == "gradual_selected_indices_by_layer"
    assert (
        repair_metadata["runtime_selected_indices"]
        == repair_metadata["selected_indices_by_layer"][0]
    ), "runtime selection should start from layer-0 HKVD selection."
    for layer_index in range(1, len(repair_metadata["selected_indices_by_layer"])):
        current = set(repair_metadata["selected_indices_by_layer"][layer_index])
        previous = set(repair_metadata["selected_indices_by_layer"][layer_index - 1])
        assert current.issubset(previous), (
            f"Layer {layer_index} selected tokens must be gradual subset of previous layer."
        )
    assert all(repair_metadata["repaired_kv_shape_matches_reuse_by_layer"].values()), (
        "repaired KV shapes must match reused KV shapes."
    )
    for layer_index, unselected_diff in repair_metadata[
        "unselected_kv_max_diff_after_vs_reuse_by_layer"
    ].items():
        assert unselected_diff == 0.0, (
            f"Layer {layer_index} unselected K/V should remain unchanged."
        )
    for layer_index, selected_after in repair_metadata[
        "selected_kv_max_diff_after_by_layer"
    ].items():
        selected_before = repair_metadata["selected_kv_max_diff_before_by_layer"][layer_index]
        assert selected_after <= selected_before + 1e-5, (
            f"Layer {layer_index} selected K/V repair should not increase deviation."
        )

    normalized_f1 = compute_normalized_score(
        method_score=float(method_records[METHOD_CACHEBLEND_REPAIR]["f1"]),
        lower_score=float(method_records[METHOD_NAIVE_REUSE]["f1"]),
        upper_score=float(method_records[METHOD_FULL_RECOMPUTE]["f1"]),
    )
    summary = {
        "example_id": example.example_id,
        "model": args.model,
        "model_family": args.model_family,
        "max_new_tokens": args.max_new_tokens,
        "methods": method_records,
        "cacheblend_normalized_f1": normalized_f1,
    }

    print("smoke_cacheblend_methods passed")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
