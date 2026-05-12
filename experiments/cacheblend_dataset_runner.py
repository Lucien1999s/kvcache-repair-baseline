from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch

from contextflow.data import (
    PROMPT_POLICY_CACHEBLEND_QA,
    SUPPORTED_CACHEBLEND_PROMPT_POLICIES,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
    tokenize_prompt_example,
)
from contextflow.evaluation import (
    aggregate_qa_metrics,
    compute_normalized_score,
    evaluate_qa_prediction,
    parse_cacheblend_generation,
)
from contextflow.kv_cache import assemble_chunk_kvs, precompute_doc_chunk_kvs
from contextflow.methods.cacheblend_repair import (
    CacheBlendRepairPlan,
    prepare_cacheblend_repair_plan_with_artifacts,
    run_cacheblend_style_partial_repair_from_plan,
)
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.profiling import timed_call as profiling_timed_call
from contextflow.runtime.hf_cached_generation import generate_with_past_key_values
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
PREDICTION_PARSER_NONE = "none"
PREDICTION_PARSER_CACHEBLEND_QA = "cacheblend_qa"
SUPPORTED_PREDICTION_PARSERS = {
    PREDICTION_PARSER_NONE,
    PREDICTION_PARSER_CACHEBLEND_QA,
}


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
        help="Comma-separated subset of full_recompute,naive_reuse,cacheblend_repair.",
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


def run_profiled_phase(
    fn: Callable[[], Any],
    *,
    enable_profiling: bool,
) -> tuple[Any, dict[str, Any]]:
    if enable_profiling:
        result, profile_record = profiling_timed_call(
            fn,
            synchronize_cuda=True,
            collect_peak_memory=True,
        )
        return result, dict(profile_record)

    start = time.perf_counter()
    result = fn()
    return result, {"latency_seconds": time.perf_counter() - start}


def phase_latency_seconds(profile_record: dict[str, Any]) -> float:
    return float(profile_record.get("latency_seconds", 0.0))


def max_phase_peak_memory_mb(phase_metrics: dict[str, dict[str, Any]]) -> float | None:
    peaks = [
        float(phase_record["peak_gpu_memory_mb"])
        for phase_record in phase_metrics.values()
        if phase_record.get("peak_gpu_memory_mb") is not None
    ]
    if not peaks:
        return None
    return max(peaks)


def mean_optional_float(records: list[dict[str, Any]], key: str) -> float | None:
    values = [
        float(record[key])
        for record in records
        if record.get(key) is not None
    ]
    if not values:
        return None
    return sum(values) / len(values)


def aggregate_resource_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_latency_seconds": mean_optional_float(records, "latency_seconds"),
        "mean_total_latency_seconds": mean_optional_float(records, "total_latency_seconds"),
        "mean_peak_gpu_memory_mb": mean_optional_float(records, "peak_gpu_memory_mb"),
        "mean_planning_latency_seconds": mean_optional_float(
            records,
            "planning_latency_seconds",
        ),
        "mean_reuse_precompute_latency_seconds": mean_optional_float(
            records,
            "reuse_precompute_latency_seconds",
        ),
        "mean_repair_latency_seconds": mean_optional_float(records, "repair_latency_seconds"),
        "mean_decode_latency_seconds": mean_optional_float(records, "decode_latency_seconds"),
        "mean_execution_latency_seconds": mean_optional_float(
            records,
            "execution_latency_seconds",
        ),
    }


def parse_prediction_for_evaluation(raw_generated_text: str, prediction_parser: str) -> str:
    if prediction_parser == PREDICTION_PARSER_NONE:
        return raw_generated_text
    if prediction_parser == PREDICTION_PARSER_CACHEBLEND_QA:
        return parse_cacheblend_generation(raw_generated_text)
    raise ValueError(
        f"Unsupported prediction_parser={prediction_parser!r}. Supported parsers: "
        f"{sorted(SUPPORTED_PREDICTION_PARSERS)}."
    )


def evaluate_method_output(
    generated_ids: list[int],
    generated_text: str,
    answers: list[str],
    prediction_parser: str,
) -> dict[str, Any]:
    evaluated_prediction = parse_prediction_for_evaluation(
        generated_text,
        prediction_parser=prediction_parser,
    )
    metrics = evaluate_qa_prediction(evaluated_prediction, answers)
    return {
        "generated_ids": generated_ids,
        "raw_generated_text": generated_text,
        "generated_text": generated_text,
        "evaluated_prediction": evaluated_prediction,
        "exact_match": metrics["exact_match"],
        "f1": metrics["f1"],
    }


def run_full_recompute(
    tokenized_example: Any,
    model: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    prediction_parser: str,
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
        prediction_parser=prediction_parser,
    )


def run_naive_reuse(
    tokenized_example: Any,
    model: Any,
    past_key_values: Any,
    tokenizer: Any,
    answers: list[str],
    max_new_tokens: int,
    prediction_parser: str,
) -> dict[str, Any]:
    generation = generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
    )
    return evaluate_method_output(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
    )


def decode_with_past_key_values(
    model: Any,
    tokenizer: Any,
    tokenized_example: Any,
    past_key_values: Any,
    max_new_tokens: int,
) -> Any:
    return generate_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        q_ids=tokenized_example.q_ids,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
    )


def build_cacheblend_repair_record(
    generation: Any,
    answers: list[str],
    prediction_parser: str,
    repair_plan: CacheBlendRepairPlan,
    partial_repair_metadata: dict[str, Any],
    phase_metrics: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    method_record = evaluate_method_output(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
    )
    repair_plan_metadata = repair_plan.metadata
    planning_latency_seconds = phase_latency_seconds(phase_metrics["plan"])
    reuse_precompute_latency_seconds = (
        phase_latency_seconds(phase_metrics["reuse_precompute"])
        if "reuse_precompute" in phase_metrics
        else None
    )
    repair_latency_seconds = phase_latency_seconds(phase_metrics["repair"])
    decode_latency_seconds = phase_latency_seconds(phase_metrics["decode"])
    execution_latency_seconds = repair_latency_seconds + decode_latency_seconds
    total_latency_seconds = sum(
        phase_latency_seconds(phase_record)
        for phase_record in phase_metrics.values()
    )
    peak_gpu_memory_mb = max_phase_peak_memory_mb(phase_metrics)
    method_record.update(
        {
            "repair_plan_strategy": repair_plan.strategy,
            "planning_latency_seconds": planning_latency_seconds,
            "uses_full_recompute_reference": repair_plan_metadata.get(
                "uses_full_recompute_reference"
            ),
            "execution_mode": "from_plan",
            "execution_uses_full_recompute_reference": partial_repair_metadata.get(
                "execution_uses_full_recompute_reference"
            ),
            "planning_included_in_total_latency": True,
            "execution_latency_seconds": execution_latency_seconds,
            "reuse_precompute_latency_seconds": reuse_precompute_latency_seconds,
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "total_latency_seconds": total_latency_seconds,
            "latency_seconds": total_latency_seconds,
            "peak_gpu_memory_mb": peak_gpu_memory_mb,
            "phase_metrics": phase_metrics,
            "reuse_past_key_values_source": partial_repair_metadata.get(
                "reuse_past_key_values_source"
            ),
            "runtime_selected_count": len(repair_plan.runtime_selected_indices),
            "runtime_selection_mode": repair_plan.runtime_selection_mode,
            "layer_selected_counts": list(repair_plan.layer_selected_counts),
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
    prompt_policy: str,
    prediction_parser: str,
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
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
        "methods": {},
    }


def precompute_reuse_doc_kv(model: Any, tokenized_example: Any) -> Any:
    chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
    return assemble_chunk_kvs(chunk_kvs)


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
    prediction_parser: str,
    enable_profiling: bool,
) -> dict[str, dict[str, Any]]:
    method_records: dict[str, dict[str, Any]] = {}
    if METHOD_FULL_RECOMPUTE in methods:
        full_record, full_profile = run_profiled_phase(
            lambda: run_full_recompute(
                tokenized_example,
                model=model,
                tokenizer=tokenizer,
                answers=answers,
                max_new_tokens=max_new_tokens,
                prediction_parser=prediction_parser,
            ),
            enable_profiling=enable_profiling,
        )
        full_record["phase_metrics"] = {"generation": full_profile}
        full_record["latency_seconds"] = phase_latency_seconds(full_profile)
        full_record["total_latency_seconds"] = phase_latency_seconds(full_profile)
        full_record["peak_gpu_memory_mb"] = max_phase_peak_memory_mb(
            full_record["phase_metrics"]
        )
        method_records[METHOD_FULL_RECOMPUTE] = full_record

    if METHOD_NAIVE_REUSE in methods:
        naive_reuse_kv, naive_reuse_profile = run_profiled_phase(
            lambda: precompute_reuse_doc_kv(
                model=model,
                tokenized_example=tokenized_example,
            ),
            enable_profiling=enable_profiling,
        )
        naive_record, naive_decode_profile = run_profiled_phase(
            lambda: run_naive_reuse(
                tokenized_example,
                model=model,
                past_key_values=naive_reuse_kv,
                tokenizer=tokenizer,
                answers=answers,
                max_new_tokens=max_new_tokens,
                prediction_parser=prediction_parser,
            ),
            enable_profiling=enable_profiling,
        )
        naive_phase_metrics = {
            "reuse_precompute": naive_reuse_profile,
            "decode": naive_decode_profile,
        }
        naive_reuse_latency_seconds = phase_latency_seconds(naive_reuse_profile)
        naive_decode_latency_seconds = phase_latency_seconds(naive_decode_profile)
        naive_total_latency_seconds = naive_reuse_latency_seconds + naive_decode_latency_seconds
        naive_record.update(
            {
                "phase_metrics": naive_phase_metrics,
                "reuse_precompute_latency_seconds": naive_reuse_latency_seconds,
                "decode_latency_seconds": naive_decode_latency_seconds,
                "total_latency_seconds": naive_total_latency_seconds,
                "latency_seconds": naive_total_latency_seconds,
                "peak_gpu_memory_mb": max_phase_peak_memory_mb(naive_phase_metrics),
                "assembled_kv_layers": len(naive_reuse_kv),
            }
        )
        method_records[METHOD_NAIVE_REUSE] = naive_record

    if METHOD_CACHEBLEND_REPAIR in methods:
        planning_artifacts, plan_profile = run_profiled_phase(
            lambda: prepare_cacheblend_repair_plan_with_artifacts(
                model=model,
                tokenized_example=tokenized_example,
                initial_top_k=initial_top_k,
                top_k=top_k,
            ),
            enable_profiling=enable_profiling,
        )
        repair_plan = planning_artifacts.plan
        repair_reuse_kv = planning_artifacts.reuse_doc_kv
        del planning_artifacts
        partial_repair, repair_profile = run_profiled_phase(
            lambda: run_cacheblend_style_partial_repair_from_plan(
                model=model,
                tokenized_example=tokenized_example,
                repair_plan=repair_plan,
                model_family=model_family,
                reuse_past_key_values=repair_reuse_kv,
            ),
            enable_profiling=enable_profiling,
        )
        repair_generation, repair_decode_profile = run_profiled_phase(
            lambda: decode_with_past_key_values(
                model=model,
                tokenizer=tokenizer,
                tokenized_example=tokenized_example,
                past_key_values=partial_repair.repaired_past_key_values,
                max_new_tokens=max_new_tokens,
            ),
            enable_profiling=enable_profiling,
        )
        repair_phase_metrics = {
            "plan": plan_profile,
            "repair": repair_profile,
            "decode": repair_decode_profile,
        }
        method_records[METHOD_CACHEBLEND_REPAIR] = build_cacheblend_repair_record(
            generation=repair_generation,
            answers=answers,
            prediction_parser=prediction_parser,
            repair_plan=repair_plan,
            partial_repair_metadata=partial_repair.metadata,
            phase_metrics=repair_phase_metrics,
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
    prompt_policy: str,
    prediction_parser: str,
    metrics_by_method: dict[str, list[dict[str, float]]],
    resources_by_method: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    method_summaries = {
        method: aggregate_qa_metrics(metrics)
        for method, metrics in metrics_by_method.items()
    }
    for method, method_summary in method_summaries.items():
        method_summary.update(
            aggregate_resource_metrics(resources_by_method.get(method, []))
        )

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
        "prompt_policy": prompt_policy,
        "prediction_parser": prediction_parser,
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
    resources_by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, example in enumerate(examples):
            record = build_example_record(
                example_index=example_index,
                example=example,
                dataset_key=dataset_key,
                model_name=args.model,
                model_family=args.model_family,
                max_new_tokens=args.max_new_tokens,
                prompt_policy=args.prompt_policy,
                prediction_parser=args.prediction_parser,
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

    summary = summarize_results(
        dataset_key=dataset_key,
        model_name=args.model,
        model_family=args.model_family,
        prompt_policy=args.prompt_policy,
        prediction_parser=args.prediction_parser,
        metrics_by_method=metrics_by_method,
        resources_by_method=resources_by_method,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
