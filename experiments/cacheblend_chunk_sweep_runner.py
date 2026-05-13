from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from contextflow.benchmarks.chunk_sweep import (
    add_sweep_context_to_method_records,
    build_case_failure_record,
    build_sweep_case_record,
    parse_chunk_count_specs,
    resolve_example_chunk_cases,
    run_methods_for_sweep_case,
    slice_example_contexts,
    summarize_sweep_results,
    token_count_record,
)
from contextflow.benchmarks.constants import (
    DEFAULT_METHODS,
    PREDICTION_PARSER_CACHEBLEND_QA,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
    STATUS_OOM,
    SUPPORTED_PREDICTION_PARSERS,
    SUPPORTED_REPAIR_PLANNERS,
)
from contextflow.benchmarks.records import (
    normalize_optional_device_map,
    parse_methods,
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
            "Chunk-count sweep runner for the CacheBlend-style HF/PyTorch reference baseline."
        )
    )
    parser.add_argument("--dataset", required=True, choices=["musique", "2wiki"])
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--model-family", required=True, choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to sweep.")
    parser.add_argument(
        "--chunk-counts",
        default="1,2,4,8,16,32,all",
        help="Comma-separated context counts to sweep. Use 'all' for all contexts.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--initial-top-k", type=int, default=10)
    parser.add_argument("--top-k", type=int, default=5)
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
    )
    parser.add_argument(
        "--prediction-parser",
        choices=sorted(SUPPORTED_PREDICTION_PARSERS),
        default=PREDICTION_PARSER_CACHEBLEND_QA,
    )
    parser.add_argument("--output-jsonl", required=True, help="Path for sweep JSONL records.")
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
        help="Record non-OOM method errors and continue. OOM is always recorded and continued.",
    )
    parser.add_argument(
        "--enable-profiling",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record synchronized latency and peak GPU memory by phase when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    methods = parse_methods(args.methods)
    chunk_count_specs = parse_chunk_count_specs(args.chunk_counts)
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive.")

    examples = load_qa_dataset_examples(dataset_key, args.input)
    if args.limit is not None:
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
    method_outcomes: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, full_example in enumerate(examples):
            chunk_cases = resolve_example_chunk_cases(full_example, chunk_count_specs)
            for chunk_case in chunk_cases:
                sliced_example = slice_example_contexts(
                    full_example,
                    chunk_count=int(chunk_case["chunk_count"]),
                )
                token_counts: dict[str, int] = {}
                try:
                    prompt = build_cacheblend_prompt(
                        sliced_example,
                        prompt_policy=args.prompt_policy,
                        dataset=dataset_key,
                    )
                    tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
                    token_counts = token_count_record(tokenized)
                    record = build_sweep_case_record(
                        example_index=example_index,
                        example=sliced_example,
                        dataset_key=dataset_key,
                        model_name=args.model,
                        model_family=args.model_family,
                        max_new_tokens=args.max_new_tokens,
                        prompt_policy=args.prompt_policy,
                        prediction_parser=args.prediction_parser,
                        repair_planner=args.repair_planner,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    with torch.inference_mode():
                        method_records = run_methods_for_sweep_case(
                            methods=methods,
                            tokenized_example=tokenized,
                            model=bundle.model,
                            tokenizer=bundle.tokenizer,
                            answers=sliced_example.answers,
                            max_new_tokens=args.max_new_tokens,
                            initial_top_k=args.initial_top_k,
                            top_k=args.top_k,
                            model_family=args.model_family,
                            prediction_parser=args.prediction_parser,
                            enable_profiling=args.enable_profiling,
                            continue_on_error=args.continue_on_error,
                            repair_planner=args.repair_planner,
                        )
                    add_sweep_context_to_method_records(
                        method_records,
                        example_index=example_index,
                        example_id=sliced_example.example_id,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    record["methods"] = method_records
                    for method, method_record in method_records.items():
                        method_outcomes[method].append(method_record)
                except Exception as error:
                    record = build_case_failure_record(
                        error,
                        example_index=example_index,
                        example=full_example,
                        dataset_key=dataset_key,
                        model_name=args.model,
                        model_family=args.model_family,
                        prompt_policy=args.prompt_policy,
                        prediction_parser=args.prediction_parser,
                        repair_planner=args.repair_planner,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                    )
                    write_jsonl_record(output_file, record)
                    if args.continue_on_error or record["status"] == STATUS_OOM:
                        continue
                    raise

                write_jsonl_record(output_file, record)

    summary = summarize_sweep_results(
        dataset_key=dataset_key,
        model_name=args.model,
        model_family=args.model_family,
        prompt_policy=args.prompt_policy,
        prediction_parser=args.prediction_parser,
        repair_planner=args.repair_planner,
        chunk_count_specs=chunk_count_specs,
        method_outcomes=method_outcomes,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
