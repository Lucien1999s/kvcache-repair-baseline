from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from contextflow.benchmarks.constants import (
    DEFAULT_METHODS,
    PREDICTION_PARSER_CACHEBLEND_QA,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
    SUPPORTED_PREDICTION_PARSERS,
    SUPPORTED_REPAIR_PLANNERS,
)
from contextflow.benchmarks.method_runner import run_methods_for_example
from contextflow.benchmarks.records import (
    build_dataset_example_record,
    normalize_optional_device_map,
    parse_methods,
    summarize_dataset_results,
    write_jsonl_record,
)
from contextflow.data import (
    PROMPT_POLICY_CACHEBLEND_QA,
    SUPPORTED_CACHEBLEND_PROMPT_POLICIES,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
    tokenize_prompt_example,
)
from contextflow.runtime import load_hf_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dataset-level runner for HF/PyTorch reference baselines."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--model-family", required=True, choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--limit", type=int, default=None, help="Number of examples to run.")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--initial-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--fusionrag-neighbor-top-n",
        type=int,
        default=5,
        help="Top-n similar neighbor chunks for FusionRAG enriched KV precompute.",
    )
    parser.add_argument(
        "--fusionrag-recompute-ratio",
        type=float,
        default=0.15,
        help="Fraction of document tokens selected by FusionRAG QGS for repair.",
    )
    parser.add_argument(
        "--repair-planner",
        choices=sorted(SUPPORTED_REPAIR_PLANNERS),
        default=REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
        help=(
            "Token-selection planner for repair. online_gradual_hkvd is the measured "
            "baseline; oracle_hkvd is full-reference diagnostic mode."
        ),
    )
    parser.add_argument(
        "--prompt-policy",
        choices=sorted(SUPPORTED_CACHEBLEND_PROMPT_POLICIES),
        default=PROMPT_POLICY_CACHEBLEND_QA,
        help="Prompt protocol used when formatting CacheBlend-style QA inputs.",
    )
    parser.add_argument(
        "--prediction-parser",
        choices=sorted(SUPPORTED_PREDICTION_PARSERS),
        default=PREDICTION_PARSER_CACHEBLEND_QA,
        help="Post-processing parser applied before EM/F1 evaluation.",
    )
    parser.add_argument("--output-jsonl", required=True, help="Path for per-example JSONL results.")
    parser.add_argument("--torch-dtype", default="auto", help="torch_dtype passed to model load.")
    parser.add_argument("--device-map", default="auto", help="device_map passed to model load.")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help=(
            "Comma-separated subset of full_recompute,naive_reuse,"
            "cacheblend_repair,fusionrag_repair."
        ),
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Write an error record and continue when an example fails.",
    )
    parser.add_argument(
        "--enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record synchronized latency and peak GPU memory when available.",
    )
    return parser.parse_args()


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
    resources_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, example in enumerate(examples):
            record = build_dataset_example_record(
                example_index=example_index,
                example=example,
                dataset_key=dataset_key,
                model_name=args.model,
                model_family=args.model_family,
                max_new_tokens=args.max_new_tokens,
                prompt_policy=args.prompt_policy,
                prediction_parser=args.prediction_parser,
                repair_planner=args.repair_planner,
            )
            try:
                prompt = build_cacheblend_prompt(
                    example,
                    prompt_policy=args.prompt_policy,
                    dataset=dataset_key,
                )
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
                        prediction_parser=args.prediction_parser,
                        enable_profiling=args.enable_profiling,
                        repair_planner=args.repair_planner,
                        fusionrag_neighbor_top_n=args.fusionrag_neighbor_top_n,
                        fusionrag_recompute_ratio=args.fusionrag_recompute_ratio,
                    )
                record["methods"] = method_records
                for method, method_record in method_records.items():
                    metrics_by_method[method].append(
                        {
                            "exact_match": float(method_record["exact_match"]),
                            "f1": float(method_record["f1"]),
                        }
                    )
                    resources_by_method[method].append(method_record)
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

    summary = summarize_dataset_results(
        dataset_key=dataset_key,
        model_name=args.model,
        model_family=args.model_family,
        prompt_policy=args.prompt_policy,
        prediction_parser=args.prediction_parser,
        repair_planner=args.repair_planner,
        metrics_by_method=metrics_by_method,
        resources_by_method=resources_by_method,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
