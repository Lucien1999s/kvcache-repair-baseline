from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from contextflow.data import (
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
    tokenize_prompt_example,
)
from contextflow.evaluation import (
    aggregate_qa_metrics,
    compute_normalized_score,
    evaluate_qa_prediction,
)
from contextflow.methods.cacheblend_repair import run_cacheblend_style_repair_generation
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.methods.naive_reuse import run_naive_reuse_generation
from contextflow.runtime import load_hf_causal_lm


METHOD_FULL_RECOMPUTE = "full_recompute"
METHOD_NAIVE_REUSE = "naive_reuse"
METHOD_CACHEBLEND_REPAIR = "cacheblend_repair"
DEFAULT_METHODS = [
    METHOD_FULL_RECOMPUTE,
    METHOD_NAIVE_REUSE,
    METHOD_CACHEBLEND_REPAIR,
]
SUPPORTED_METHODS = set(DEFAULT_METHODS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dataset-level CacheBlend reproduction runner for local QA data."
    )
    parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--model-family", required=True, choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--initial-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-jsonl", required=True, help="Path for per-example JSONL results.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype passed to model load.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to model load.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated subset of full_recompute,naive_reuse,cacheblend_repair.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error record and continue when an example fails.",
    )
    return parser.parse_args()


def parse_methods(raw_methods: str) -> list[str]:
    methods = [method.strip() for method in raw_methods.split(",") if method.strip()]
    if not methods:
        raise ValueError("--methods must contain at least one method.")
    invalid = [method for method in methods if method not in SUPPORTED_METHODS]
    if invalid:
        raise ValueError(f"Unsupported methods: {invalid}. Supported methods: {sorted(SUPPORTED_METHODS)}.")
    return methods


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def timed_call(fn: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    start = time.perf_counter()
    result = fn()
    result["latency_seconds"] = time.perf_counter() - start
    return result


def evaluate_method_output(
    generated_ids: list[int],
    generated_text: str,
    answers: list[str],
) -> dict[str, Any]:
    metrics = evaluate_qa_prediction(generated_text, answers)
    return {
        "generated_ids": generated_ids,
        "generated_text": generated_text,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
    }


def run_full_recompute(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    result = run_token_aligned_full_recompute_greedy_generation(
        tokenized_example,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return evaluate_method_output(
        generated_ids=result.generation.generated_ids,
        generated_text=result.generation.output_text,
        answers=answers,
    )


def run_naive_reuse(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
) -> dict[str, Any]:
    result = run_naive_reuse_generation(
        tokenized_example,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )
    return evaluate_method_output(
        generated_ids=result.generation.generated_ids,
        generated_text=result.generation.output_text,
        answers=answers,
    )


def run_cacheblend_repair(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    initial_top_k: int,
    top_k: int,
    model_family: str,
) -> dict[str, Any]:
    result = run_cacheblend_style_repair_generation(
        model=model,
        tokenizer=tokenizer,
        tokenized_example=tokenized_example,
        max_new_tokens=max_new_tokens,
        initial_top_k=initial_top_k,
        top_k=top_k,
        model_family=model_family,
    )
    method_record = evaluate_method_output(
        generated_ids=result.generated_ids,
        generated_text=result.output_text,
        answers=answers,
    )
    metadata = result.metadata
    method_record.update(
        {
            "repair_latency_seconds": metadata.get("repair_latency_seconds"),
            "decode_latency_seconds": metadata.get("decode_latency_seconds"),
            "runtime_selected_count": len(metadata.get("runtime_selected_indices", [])),
            "runtime_selection_mode": metadata.get("runtime_selection_mode"),
            "layer_selected_counts": metadata.get("layer_selected_counts", []),
        }
    )
    return method_record


def build_example_record(
    example_index: int,
    example: Any,
    dataset_key: str,
    model_name: str,
    model_family: str,
    max_new_tokens: int,
) -> dict[str, Any]:
    return {
        "example_index": example_index,
        "example_id": example.example_id,
        "dataset": dataset_key,
        "question": example.question,
        "answers": example.answers,
        "num_ctxs": len(example.ctxs),
        "model": model_name,
        "model_family": model_family,
        "max_new_tokens": max_new_tokens,
        "methods": {},
    }


def run_methods_for_example(
    methods: list[str],
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    initial_top_k: int,
    top_k: int,
    model_family: str,
) -> dict[str, dict[str, Any]]:
    method_records: dict[str, dict[str, Any]] = {}
    if METHOD_FULL_RECOMPUTE in methods:
        method_records[METHOD_FULL_RECOMPUTE] = timed_call(
            lambda: run_full_recompute(
                tokenized_example,
                model=model,
                tokenizer=tokenizer,
                answers=answers,
                max_new_tokens=max_new_tokens,
            )
        )
    if METHOD_NAIVE_REUSE in methods:
        method_records[METHOD_NAIVE_REUSE] = timed_call(
            lambda: run_naive_reuse(
                tokenized_example,
                model=model,
                tokenizer=tokenizer,
                answers=answers,
                max_new_tokens=max_new_tokens,
            )
        )
    if METHOD_CACHEBLEND_REPAIR in methods:
        method_records[METHOD_CACHEBLEND_REPAIR] = timed_call(
            lambda: run_cacheblend_repair(
                tokenized_example,
                model=model,
                tokenizer=tokenizer,
                answers=answers,
                max_new_tokens=max_new_tokens,
                initial_top_k=initial_top_k,
                top_k=top_k,
                model_family=model_family,
            )
        )
    return method_records


def write_jsonl_record(file: Any, record: dict[str, Any]) -> None:
    file.write(json.dumps(record, ensure_ascii=False))
    file.write("\n")
    file.flush()


def summarize_results(
    dataset_key: str,
    model_name: str,
    model_family: str,
    metrics_by_method: dict[str, list[dict[str, float]]],
) -> dict[str, Any]:
    method_summaries = {
        method: aggregate_qa_metrics(metrics)
        for method, metrics in metrics_by_method.items()
    }

    normalized_f1 = None
    if all(method in method_summaries for method in DEFAULT_METHODS):
        naive_f1 = method_summaries[METHOD_NAIVE_REUSE]["mean_f1"]
        full_f1 = method_summaries[METHOD_FULL_RECOMPUTE]["mean_f1"]
        repair_f1 = method_summaries[METHOD_CACHEBLEND_REPAIR]["mean_f1"]
        if naive_f1 is not None and full_f1 is not None and repair_f1 is not None:
            normalized_f1 = compute_normalized_score(
                method_score=float(repair_f1),
                lower_score=float(naive_f1),
                upper_score=float(full_f1),
            )

    return {
        "dataset": dataset_key,
        "model": model_name,
        "model_family": model_family,
        "count": max((summary["count"] for summary in method_summaries.values()), default=0),
        "methods": method_summaries,
        "cacheblend_normalized_f1": normalized_f1,
    }


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    methods = parse_methods(args.methods)
    examples = load_qa_dataset_examples(dataset_key, args.input)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive when provided.")
        examples = examples[: args.limit]
    if not examples:
        raise ValueError("Expected at least one dataset example.")

    bundle = load_hf_causal_lm(
        args.model,
        torch_dtype=args.torch_dtype,
        device_map=normalize_optional_device_map(args.device_map),
    )
    bundle.model.eval()

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_by_method: dict[str, list[dict[str, float]]] = {method: [] for method in methods}

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, example in enumerate(examples):
            record = build_example_record(
                example_index=example_index,
                example=example,
                dataset_key=dataset_key,
                model_name=args.model,
                model_family=args.model_family,
                max_new_tokens=args.max_new_tokens,
            )
            try:
                prompt = build_cacheblend_prompt(example)
                tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
                with torch.inference_mode():
                    method_records = run_methods_for_example(
                        methods=methods,
                        tokenized_example=tokenized,
                        model=bundle.model,
                        tokenizer=bundle.tokenizer,
                        answers=example.answers,
                        max_new_tokens=args.max_new_tokens,
                        initial_top_k=args.initial_top_k,
                        top_k=args.top_k,
                        model_family=args.model_family,
                    )
                record["methods"] = method_records
                for method, method_record in method_records.items():
                    metrics_by_method[method].append(
                        {
                            "exact_match": float(method_record["exact_match"]),
                            "f1": float(method_record["f1"]),
                        }
                    )
            except Exception as error:
                record["error"] = {
                    "type": type(error).__name__,
                    "message": str(error),
                }
                write_jsonl_record(output_file, record)
                if args.continue_on_error:
                    continue
                raise

            write_jsonl_record(output_file, record)

    summary = summarize_results(
        dataset_key=dataset_key,
        model_name=args.model,
        model_family=args.model_family,
        metrics_by_method=metrics_by_method,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
