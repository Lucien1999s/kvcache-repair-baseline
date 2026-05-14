from __future__ import annotations

from typing import Any

from contextflow.benchmarks.constants import (
    FUSIONRAG_PHASE_DECODE,
    FUSIONRAG_PHASE_ENRICHED_PRECOMPUTE,
    FUSIONRAG_PHASE_QUERY_SELECTION,
    FUSIONRAG_PHASE_REPAIR,
    FUSIONRAG_PROFILE_PHASES,
    METHOD_CACHEBLEND_REPAIR,
    METHOD_FUSIONRAG_REPAIR,
    METHOD_FULL_RECOMPUTE,
    METHOD_NAIVE_REUSE,
    REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
    REPAIR_PLANNER_ORACLE_HKVD,
    SUPPORTED_REPAIR_PLANNERS,
)
from contextflow.benchmarks.evaluation import evaluate_method_generation
from contextflow.benchmarks.resources import (
    memory_summary_record,
    phase_latency_seconds,
    run_profiled_phase,
    run_with_method_cleanup,
    selected_phase_profile_fields,
    total_phase_latency_seconds,
)
from contextflow.kv_cache import (
    assemble_chunk_kvs,
    correct_doc_chunk_kvs_for_model_family,
    precompute_doc_chunk_kvs,
    rope_position_correction_enabled,
)
from contextflow.methods.cacheblend_repair import (
    CacheBlendRepairPlan,
    prepare_cacheblend_repair_plan_with_artifacts,
    run_cacheblend_style_online_partial_repair,
    run_cacheblend_style_partial_repair_from_plan,
)
from contextflow.methods.full_recompute import run_token_aligned_full_recompute_greedy_generation
from contextflow.methods.fusionrag_repair import (
    FusionRAGRepairResult,
    prepare_fusionrag_enriched_context,
    prepare_fusionrag_query_repair_plan,
)
from contextflow.runtime.hf_cached_generation import generate_with_past_key_values


def precompute_reuse_doc_kv(model: Any, tokenized_example: Any, model_family: str) -> Any:
    chunk_kvs = precompute_doc_chunk_kvs(model, tokenized_example)
    chunk_kvs = correct_doc_chunk_kvs_for_model_family(
        model=model,
        chunk_kvs=chunk_kvs,
        model_family=model_family,
    )
    return assemble_chunk_kvs(chunk_kvs)


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
    return evaluate_method_generation(
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
    generation = decode_with_past_key_values(
        model=model,
        tokenizer=tokenizer,
        tokenized_example=tokenized_example,
        past_key_values=past_key_values,
        max_new_tokens=max_new_tokens,
    )
    return evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
    )


def build_cacheblend_repair_record(
    generation: Any,
    answers: list[str],
    prediction_parser: str,
    repair_plan: CacheBlendRepairPlan,
    partial_repair_metadata: dict[str, Any],
    phase_metrics: dict[str, dict[str, Any]],
    *,
    include_status: bool = False,
) -> dict[str, Any]:
    method_record = evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
        include_status=include_status,
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
    total_latency_seconds = total_phase_latency_seconds(phase_metrics)
    method_record.update(
        {
            "repair_plan_strategy": repair_plan.strategy,
            "planning_latency_seconds": planning_latency_seconds,
            "uses_full_recompute_reference": repair_plan_metadata.get(
                "uses_full_recompute_reference"
            ),
            "rope_position_correction_applied": repair_plan_metadata.get(
                "rope_position_correction_applied"
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
            **memory_summary_record(phase_metrics),
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


def build_online_cacheblend_repair_record(
    generation: Any,
    answers: list[str],
    prediction_parser: str,
    partial_repair_metadata: dict[str, Any],
    phase_metrics: dict[str, dict[str, Any]],
    *,
    include_status: bool = False,
) -> dict[str, Any]:
    method_record = evaluate_method_generation(
        generated_ids=generation.generated_ids,
        generated_text=generation.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
        include_status=include_status,
    )
    repair_latency_seconds = phase_latency_seconds(phase_metrics["online_repair"])
    decode_latency_seconds = phase_latency_seconds(phase_metrics["decode"])
    execution_latency_seconds = repair_latency_seconds + decode_latency_seconds
    total_latency_seconds = total_phase_latency_seconds(phase_metrics)
    method_record.update(
        {
            "repair_plan_strategy": partial_repair_metadata.get("repair_plan_strategy"),
            "repair_planner": REPAIR_PLANNER_ONLINE_GRADUAL_HKVD,
            "planning_latency_seconds": None,
            "uses_full_recompute_reference": partial_repair_metadata.get(
                "repair_plan_metadata",
                {},
            ).get("uses_full_recompute_reference"),
            "rope_position_correction_applied": partial_repair_metadata.get(
                "rope_position_correction_applied"
            ),
            "execution_mode": "online_gradual_hkvd",
            "execution_uses_full_recompute_reference": partial_repair_metadata.get(
                "execution_uses_full_recompute_reference"
            ),
            "planning_included_in_total_latency": False,
            "execution_latency_seconds": execution_latency_seconds,
            "reuse_precompute_latency_seconds": phase_latency_seconds(
                phase_metrics["reuse_precompute"]
            ),
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "total_latency_seconds": total_latency_seconds,
            "latency_seconds": total_latency_seconds,
            **memory_summary_record(phase_metrics),
            "phase_metrics": phase_metrics,
            "reuse_past_key_values_source": partial_repair_metadata.get(
                "reuse_past_key_values_source"
            ),
            "runtime_selected_count": len(
                partial_repair_metadata.get("runtime_selected_indices", [])
            ),
            "runtime_selection_mode": partial_repair_metadata.get(
                "runtime_selection_mode"
            ),
            "layer_selected_counts": partial_repair_metadata.get(
                "layer_selected_counts",
                [],
            ),
        }
    )
    return method_record


def build_fusionrag_repair_record(
    result: FusionRAGRepairResult,
    answers: list[str],
    prediction_parser: str,
    phase_metrics: dict[str, dict[str, Any]],
    *,
    include_status: bool = False,
) -> dict[str, Any]:
    method_record = evaluate_method_generation(
        generated_ids=result.generated_ids,
        generated_text=result.output_text,
        answers=answers,
        prediction_parser=prediction_parser,
        include_status=include_status,
    )
    metadata = result.metadata
    enriched_precompute_latency_seconds = (
        phase_latency_seconds(phase_metrics[FUSIONRAG_PHASE_ENRICHED_PRECOMPUTE])
        if FUSIONRAG_PHASE_ENRICHED_PRECOMPUTE in phase_metrics
        else metadata.get("enriched_precompute_latency_seconds")
    )
    selection_latency_seconds = (
        phase_latency_seconds(phase_metrics[FUSIONRAG_PHASE_QUERY_SELECTION])
        if FUSIONRAG_PHASE_QUERY_SELECTION in phase_metrics
        else metadata.get("selection_latency_seconds")
    )
    repair_latency_seconds = (
        phase_latency_seconds(phase_metrics[FUSIONRAG_PHASE_REPAIR])
        if FUSIONRAG_PHASE_REPAIR in phase_metrics
        else metadata.get("repair_latency_seconds")
    )
    decode_latency_seconds = (
        phase_latency_seconds(phase_metrics[FUSIONRAG_PHASE_DECODE])
        if FUSIONRAG_PHASE_DECODE in phase_metrics
        else metadata.get("decode_latency_seconds")
    )
    execution_latency_seconds = None
    if repair_latency_seconds is not None and decode_latency_seconds is not None:
        execution_latency_seconds = float(repair_latency_seconds) + float(
            decode_latency_seconds
        )
    total_latency_seconds = total_phase_latency_seconds(phase_metrics)
    method_record.update(
        {
            "method": metadata.get("method", METHOD_FUSIONRAG_REPAIR),
            "neighbor_top_n": metadata.get("neighbor_top_n"),
            "recompute_ratio": metadata.get("recompute_ratio"),
            "selected_count": metadata.get("selected_count"),
            "selected_indices": metadata.get("selected_indices", []),
            "selected_indices_by_score": metadata.get("selected_indices_by_score", []),
            "repair_plan_strategy": metadata.get("repair_plan_strategy"),
            "runtime_selection_mode": metadata.get("runtime_selection_mode"),
            "layer_selected_counts": metadata.get("layer_selected_counts", []),
            "execution_uses_full_recompute_reference": metadata.get(
                "execution_uses_full_recompute_reference"
            ),
            "planning_included_in_total_latency": False,
            "enriched_precompute_latency_seconds": enriched_precompute_latency_seconds,
            "selection_latency_seconds": selection_latency_seconds,
            "repair_latency_seconds": repair_latency_seconds,
            "decode_latency_seconds": decode_latency_seconds,
            "execution_latency_seconds": execution_latency_seconds,
            "method_internal_total_latency_seconds": metadata.get("total_latency_seconds"),
            "total_latency_seconds": total_latency_seconds,
            "latency_seconds": total_latency_seconds,
            **memory_summary_record(phase_metrics),
            **selected_phase_profile_fields(phase_metrics, FUSIONRAG_PROFILE_PHASES),
            "phase_metrics": phase_metrics,
            "enriched_precompute_metadata": metadata.get(
                "enriched_precompute_metadata",
                {},
            ),
            "query_selection_metadata": metadata.get("query_selection_metadata", {}),
            "partial_repair_metadata": metadata.get("partial_repair_metadata", {}),
        }
    )
    return method_record


def build_fusionrag_result_from_profiled_phases(
    generation: Any,
    partial_repair: Any,
    enriched_context: Any,
    plan_artifacts: Any,
    model_family: str,
    neighbor_top_n: int,
    recompute_ratio: float,
    phase_metrics: dict[str, dict[str, Any]],
) -> FusionRAGRepairResult:
    query_selection = plan_artifacts.query_selection
    repair_plan = plan_artifacts.repair_plan
    metadata = {
        "method": METHOD_FUSIONRAG_REPAIR,
        "model_family": model_family,
        "neighbor_top_n": neighbor_top_n,
        "recompute_ratio": recompute_ratio,
        "selected_count": query_selection.recompute_count,
        "selected_indices": list(query_selection.selected_indices),
        "selected_indices_by_score": list(query_selection.selected_indices_by_score),
        "enriched_precompute_latency_seconds": phase_latency_seconds(
            phase_metrics[FUSIONRAG_PHASE_ENRICHED_PRECOMPUTE]
        ),
        "selection_latency_seconds": phase_latency_seconds(
            phase_metrics[FUSIONRAG_PHASE_QUERY_SELECTION]
        ),
        "repair_latency_seconds": phase_latency_seconds(
            phase_metrics[FUSIONRAG_PHASE_REPAIR]
        ),
        "decode_latency_seconds": phase_latency_seconds(
            phase_metrics[FUSIONRAG_PHASE_DECODE]
        ),
        "total_latency_seconds": total_phase_latency_seconds(phase_metrics),
        "execution_uses_full_recompute_reference": False,
        "repair_plan_strategy": repair_plan.strategy,
        "runtime_selection_mode": repair_plan.runtime_selection_mode,
        "layer_selected_counts": list(repair_plan.layer_selected_counts),
        "enriched_precompute_metadata": dict(enriched_context.metadata),
        "query_selection_metadata": query_selection.to_metadata(),
        "partial_repair_metadata": dict(partial_repair.metadata),
    }
    return FusionRAGRepairResult(
        generated_ids=generation.generated_ids,
        output_text=generation.output_text,
        generation=generation,
        repaired_past_key_values=partial_repair.repaired_past_key_values,
        enriched_past_key_values=enriched_context.enriched_past_key_values,
        repair_plan=repair_plan,
        query_selection=query_selection,
        metadata=metadata,
    )


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
    repair_planner: str,
    fusionrag_neighbor_top_n: int = 5,
    fusionrag_recompute_ratio: float = 0.15,
) -> dict[str, dict[str, Any]]:
    method_records: dict[str, dict[str, Any]] = {}

    def run_full_recompute_record() -> dict[str, Any]:
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
        full_record.update(memory_summary_record(full_record["phase_metrics"]))
        return full_record

    def run_naive_reuse_record() -> dict[str, Any]:
        naive_reuse_kv, naive_reuse_profile = run_profiled_phase(
            lambda: precompute_reuse_doc_kv(
                model=model,
                tokenized_example=tokenized_example,
                model_family=model_family,
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
                **memory_summary_record(naive_phase_metrics),
                "assembled_kv_layers": len(naive_reuse_kv),
                "rope_position_correction_applied": rope_position_correction_enabled(
                    model_family
                ),
            }
        )
        return naive_record

    def run_oracle_cacheblend_record() -> dict[str, Any]:
        planning_artifacts, plan_profile = run_profiled_phase(
            lambda: prepare_cacheblend_repair_plan_with_artifacts(
                model=model,
                tokenized_example=tokenized_example,
                initial_top_k=initial_top_k,
                top_k=top_k,
                model_family=model_family,
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
        method_record = build_cacheblend_repair_record(
            generation=repair_generation,
            answers=answers,
            prediction_parser=prediction_parser,
            repair_plan=repair_plan,
            partial_repair_metadata=partial_repair.metadata,
            phase_metrics=repair_phase_metrics,
        )
        method_record["repair_planner"] = repair_planner
        return method_record

    def run_online_cacheblend_record() -> dict[str, Any]:
        online_reuse_kv, online_reuse_profile = run_profiled_phase(
            lambda: precompute_reuse_doc_kv(
                model=model,
                tokenized_example=tokenized_example,
                model_family=model_family,
            ),
            enable_profiling=enable_profiling,
        )
        partial_repair, online_repair_profile = run_profiled_phase(
            lambda: run_cacheblend_style_online_partial_repair(
                model=model,
                tokenized_example=tokenized_example,
                initial_top_k=initial_top_k,
                top_k=top_k,
                model_family=model_family,
                reuse_past_key_values=online_reuse_kv,
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
            "reuse_precompute": online_reuse_profile,
            "online_repair": online_repair_profile,
            "decode": repair_decode_profile,
        }
        return build_online_cacheblend_repair_record(
            generation=repair_generation,
            answers=answers,
            prediction_parser=prediction_parser,
            partial_repair_metadata=partial_repair.metadata,
            phase_metrics=repair_phase_metrics,
        )

    def run_fusionrag_record() -> dict[str, Any]:
        phase_metrics: dict[str, dict[str, Any]] = {}
        enriched_context, enriched_profile = run_profiled_phase(
            lambda: prepare_fusionrag_enriched_context(
                model=model,
                tokenized_example=tokenized_example,
                neighbor_top_n=fusionrag_neighbor_top_n,
                model_family=model_family,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics[FUSIONRAG_PHASE_ENRICHED_PRECOMPUTE] = enriched_profile

        plan_artifacts, selection_profile = run_profiled_phase(
            lambda: prepare_fusionrag_query_repair_plan(
                model=model,
                tokenized_example=tokenized_example,
                enriched_context=enriched_context,
                neighbor_top_n=fusionrag_neighbor_top_n,
                recompute_ratio=fusionrag_recompute_ratio,
                model_family=model_family,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics[FUSIONRAG_PHASE_QUERY_SELECTION] = selection_profile

        partial_repair, repair_profile = run_profiled_phase(
            lambda: run_cacheblend_style_partial_repair_from_plan(
                model=model,
                tokenized_example=tokenized_example,
                repair_plan=plan_artifacts.repair_plan,
                model_family=model_family,
                reuse_past_key_values=enriched_context.enriched_past_key_values,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics[FUSIONRAG_PHASE_REPAIR] = repair_profile

        generation, decode_profile = run_profiled_phase(
            lambda: decode_with_past_key_values(
                model=model,
                tokenizer=tokenizer,
                tokenized_example=tokenized_example,
                past_key_values=partial_repair.repaired_past_key_values,
                max_new_tokens=max_new_tokens,
            ),
            enable_profiling=enable_profiling,
        )
        phase_metrics[FUSIONRAG_PHASE_DECODE] = decode_profile

        result = build_fusionrag_result_from_profiled_phases(
            generation=generation,
            partial_repair=partial_repair,
            enriched_context=enriched_context,
            plan_artifacts=plan_artifacts,
            model_family=model_family,
            neighbor_top_n=fusionrag_neighbor_top_n,
            recompute_ratio=fusionrag_recompute_ratio,
            phase_metrics=phase_metrics,
        )
        return build_fusionrag_repair_record(
            result=result,
            answers=answers,
            prediction_parser=prediction_parser,
            phase_metrics=phase_metrics,
        )

    if METHOD_FULL_RECOMPUTE in methods:
        method_records[METHOD_FULL_RECOMPUTE] = run_with_method_cleanup(
            run_full_recompute_record
        )
    if METHOD_NAIVE_REUSE in methods:
        method_records[METHOD_NAIVE_REUSE] = run_with_method_cleanup(
            run_naive_reuse_record
        )
    if METHOD_CACHEBLEND_REPAIR in methods:
        if repair_planner == REPAIR_PLANNER_ORACLE_HKVD:
            method_records[METHOD_CACHEBLEND_REPAIR] = run_with_method_cleanup(
                run_oracle_cacheblend_record
            )
        elif repair_planner == REPAIR_PLANNER_ONLINE_GRADUAL_HKVD:
            method_records[METHOD_CACHEBLEND_REPAIR] = run_with_method_cleanup(
                run_online_cacheblend_record
            )
        else:
            raise ValueError(
                f"Unsupported repair_planner={repair_planner!r}. "
                f"Supported planners: {sorted(SUPPORTED_REPAIR_PLANNERS)}."
            )
    if METHOD_FUSIONRAG_REPAIR in methods:
        method_records[METHOD_FUSIONRAG_REPAIR] = run_with_method_cleanup(
            run_fusionrag_record
        )
    return method_records
