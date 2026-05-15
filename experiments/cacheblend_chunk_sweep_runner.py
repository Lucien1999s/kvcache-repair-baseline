from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from contextflow.benchmarks.chunk_sweep import (
    CHUNKING_MODE_CONTEXT,
    CHUNKING_MODE_TOKEN,
    SUPPORTED_CHUNKING_MODES,
    add_sweep_context_to_method_records,
    build_case_failure_record,
    build_sweep_case_record,
    chunking_config_record,
    parse_chunk_count_specs,
    resolve_chunk_cases_for_available_count,
    resolve_example_chunk_cases,
    run_methods_for_sweep_case,
    slice_example_contexts,
    summarize_sweep_results,
    token_count_record,
    validate_chunking_mode,
)
from contextflow.benchmarks.constants import (
    DEFAULT_METHODS,
    PREDICTION_PARSER_CACHEBLEND_QA,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
    STATUS_OOM,
    STATUS_SKIPPED_TOO_LONG,
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
    SUPPORTED_QA_DATASET_KEYS,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
    tokenize_prompt_example,
)
from contextflow.chunking import (
    TokenChunkingConfig,
    slice_tokenized_doc_chunks,
    token_chunk_tokenized_example,
    token_chunking_metadata,
)
from contextflow.runtime import load_hf_causal_lm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Chunk-count sweep runner for HF/PyTorch reference baselines."
        )
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_QA_DATASET_KEYS))
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--model", required=True, help="HuggingFace model name or local path.")
    parser.add_argument("--model-family", required=True, choices=["gpt2", "mistral", "qwen2"])
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to sweep.")
    parser.add_argument(
        "--chunk-counts",
        default="1,2,4,8,16,32,all",
        help="Comma-separated context counts to sweep. Use 'all' for all contexts.",
    )
    parser.add_argument(
        "--chunking",
        choices=sorted(SUPPORTED_CHUNKING_MODES),
        default=CHUNKING_MODE_CONTEXT,
        help="Use dataset contexts directly or rechunk the tokenized document into fixed token chunks.",
    )
    parser.add_argument(
        "--chunk-size-tokens",
        type=int,
        default=1024,
        help="Token chunk size used when --chunking token.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=0,
        help="Token overlap between chunks used when --chunking token.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Maximum produced token chunks when --chunking token.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help=(
            "Skip sweep cases whose doc+query prefill tokens exceed this budget. "
            "No truncation is applied."
        ),
    )
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
        help=(
            "Comma-separated subset of full_recompute,naive_reuse,"
            "cacheblend_repair,fusionrag_repair."
        ),
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
    parser.add_argument(
        "--enable-micro-profiling",
        action="store_true",
        help=(
            "Record repair-internal micro phase latency and peak GPU memory for "
            "CacheBlend-style and FusionRAG-style repair."
        ),
    )
    return parser.parse_args()


def build_too_long_method_records(
    methods: list[str],
    *,
    max_total_tokens: int,
    token_counts: dict[str, int],
) -> dict[str, dict[str, Any]]:
    return {
        method: {
            "method": method,
            "status": STATUS_SKIPPED_TOO_LONG,
            "skip_reason": "total_prefill_token_count_exceeds_max_total_tokens",
            "max_total_tokens": max_total_tokens,
            **token_counts,
        }
        for method in methods
    }


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    methods = parse_methods(args.methods)
    chunk_count_specs = parse_chunk_count_specs(args.chunk_counts)
    chunking_mode = validate_chunking_mode(args.chunking)
    summary_chunking_config = chunking_config_record(
        chunking_mode,
        chunk_size_tokens=(
            args.chunk_size_tokens if chunking_mode == CHUNKING_MODE_TOKEN else None
        ),
        chunk_overlap_tokens=(
            args.chunk_overlap_tokens if chunking_mode == CHUNKING_MODE_TOKEN else None
        ),
        max_chunks=args.max_chunks if chunking_mode == CHUNKING_MODE_TOKEN else None,
    )
    if args.max_total_tokens is not None:
        if args.max_total_tokens <= 0:
            raise ValueError("--max-total-tokens must be positive when provided.")
        summary_chunking_config["max_total_tokens"] = args.max_total_tokens
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
            if chunking_mode == CHUNKING_MODE_CONTEXT:
                chunk_cases = resolve_example_chunk_cases(full_example, chunk_count_specs)
                prepared_cases = [
                    {
                        "chunk_case": chunk_case,
                        "example": slice_example_contexts(
                            full_example,
                            chunk_count=int(chunk_case["chunk_count"]),
                        ),
                        "tokenized": None,
                        "chunking_config": chunking_config_record(CHUNKING_MODE_CONTEXT),
                    }
                    for chunk_case in chunk_cases
                ]
            elif chunking_mode == CHUNKING_MODE_TOKEN:
                token_chunking_config = TokenChunkingConfig(
                    chunk_size_tokens=args.chunk_size_tokens,
                    chunk_overlap_tokens=args.chunk_overlap_tokens,
                    max_chunks=args.max_chunks,
                )
                full_prompt = build_cacheblend_prompt(
                    full_example,
                    prompt_policy=args.prompt_policy,
                    dataset=dataset_key,
                )
                full_tokenized = tokenize_prompt_example(full_prompt, bundle.tokenizer)
                source_doc_token_count = sum(
                    len(doc_ids) for doc_ids in full_tokenized.doc_chunk_ids
                )
                token_chunked = token_chunk_tokenized_example(
                    full_tokenized,
                    chunk_size_tokens=args.chunk_size_tokens,
                    chunk_overlap_tokens=args.chunk_overlap_tokens,
                    max_chunks=args.max_chunks,
                )
                token_chunking_config_record = token_chunking_metadata(
                    token_chunking_config,
                    source_doc_token_count=source_doc_token_count,
                    produced_chunk_count=len(token_chunked.doc_chunk_ids),
                )
                chunk_cases = resolve_chunk_cases_for_available_count(
                    available_count=len(token_chunked.doc_chunk_ids),
                    chunk_count_specs=chunk_count_specs,
                )
                prepared_cases = [
                    {
                        "chunk_case": chunk_case,
                        "example": full_example,
                        "tokenized": slice_tokenized_doc_chunks(
                            token_chunked,
                            chunk_count=int(chunk_case["chunk_count"]),
                        ),
                        "chunking_config": token_chunking_config_record,
                    }
                    for chunk_case in chunk_cases
                ]
            else:
                raise AssertionError(f"Unhandled chunking mode: {chunking_mode}")

            for prepared_case in prepared_cases:
                chunk_case = prepared_case["chunk_case"]
                sliced_example = prepared_case["example"]
                chunking_config = dict(prepared_case["chunking_config"])
                if args.max_total_tokens is not None:
                    chunking_config["max_total_tokens"] = args.max_total_tokens
                token_counts: dict[str, int] = {}
                try:
                    tokenized = prepared_case["tokenized"]
                    if tokenized is None:
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
                        chunking_config=chunking_config,
                    )
                    if (
                        args.max_total_tokens is not None
                        and token_counts["total_prefill_token_count"] > args.max_total_tokens
                    ):
                        method_records = build_too_long_method_records(
                            methods,
                            max_total_tokens=args.max_total_tokens,
                            token_counts=token_counts,
                        )
                        add_sweep_context_to_method_records(
                            method_records,
                            example_index=example_index,
                            example_id=sliced_example.example_id,
                            chunk_case=chunk_case,
                            token_counts=token_counts,
                            chunking_config=chunking_config,
                        )
                        record.update(
                            {
                                "status": STATUS_SKIPPED_TOO_LONG,
                                "skip_reason": (
                                    "total_prefill_token_count_exceeds_max_total_tokens"
                                ),
                                "max_total_tokens": args.max_total_tokens,
                                "methods": method_records,
                            }
                        )
                        for method, method_record in method_records.items():
                            method_outcomes[method].append(method_record)
                        write_jsonl_record(output_file, record)
                        continue

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
                            enable_micro_profiling=args.enable_micro_profiling,
                            continue_on_error=args.continue_on_error,
                            repair_planner=args.repair_planner,
                            fusionrag_neighbor_top_n=args.fusionrag_neighbor_top_n,
                            fusionrag_recompute_ratio=args.fusionrag_recompute_ratio,
                        )
                    add_sweep_context_to_method_records(
                        method_records,
                        example_index=example_index,
                        example_id=sliced_example.example_id,
                        chunk_case=chunk_case,
                        token_counts=token_counts,
                        chunking_config=chunking_config,
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
                        chunking_config=chunking_config,
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
        chunking_config=summary_chunking_config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
