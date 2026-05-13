from __future__ import annotations

import argparse
import json
from typing import Any

from contextflow.data import (
    build_cacheblend_prompt,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.kv_cache import precompute_enriched_doc_chunk_kvs
from contextflow.retrieval import build_chunk_neighbor_plan


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "fusionrag-enriched-smoke-0",
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
        description="Smoke test FusionRAG-style neighbor-enriched KV precompute."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional HuggingFace causal LM name or local path. Omit to test only neighbor plans.",
    )
    parser.add_argument(
        "--model-family",
        default="gpt2",
        choices=["gpt2", "mistral", "qwen2"],
        help="Adapter family used for optional RoPE source-position correction.",
    )
    parser.add_argument("--neighbor-top-n", type=int, default=1)
    parser.add_argument("--torch-dtype", default="auto")
    parser.add_argument("--device-map", default=None)
    parser.add_argument(
        "--apply-rope-source-position-correction",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser.parse_args()


def normalize_optional_device_map(raw_device_map: str | None) -> str | None:
    if raw_device_map is None:
        return None
    if raw_device_map.lower() in {"none", "null", ""}:
        return None
    return raw_device_map


def assert_neighbor_plans(
    doc_chunk_ids: list[list[int]],
    neighbor_top_n: int,
) -> list[dict[str, Any]]:
    plans = build_chunk_neighbor_plan(doc_chunk_ids, neighbor_top_n=neighbor_top_n)
    repeated_plans = build_chunk_neighbor_plan(doc_chunk_ids, neighbor_top_n=neighbor_top_n)
    assert plans == repeated_plans, "neighbor planning must be deterministic."

    expected_count = min(neighbor_top_n, max(0, len(doc_chunk_ids) - 1))
    summary = []
    for plan in plans:
        assert plan.target_chunk_index not in plan.neighbor_indices, (
            "neighbor plan must not select the target chunk itself."
        )
        assert len(plan.neighbor_indices) == expected_count
        assert len(plan.neighbor_scores) == expected_count
        summary.append(
            {
                "target_chunk_index": plan.target_chunk_index,
                "neighbor_indices": plan.neighbor_indices,
                "neighbor_scores": plan.neighbor_scores,
            }
        )
    return summary


def first_layer_seq_len(chunk_kv: Any) -> int:
    key, value = chunk_kv.past_key_values[0][:2]
    assert int(key.shape[-2]) == int(value.shape[-2])
    return int(key.shape[-2])


def run_model_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    from contextflow.runtime import load_hf_causal_lm

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
    with torch.inference_mode():
        result = precompute_enriched_doc_chunk_kvs(
            model=bundle.model,
            example=tokenized,
            neighbor_top_n=args.neighbor_top_n,
            model_family=args.model_family,
            apply_rope_source_position_correction=(
                args.apply_rope_source_position_correction
            ),
        )

    assert len(result.enriched_chunk_kvs) == len(tokenized.doc_chunk_ids)
    assert len(result.plain_chunk_kvs) == len(tokenized.doc_chunk_ids)
    assert len(result.neighbor_plans) == len(tokenized.doc_chunk_ids)

    chunk_summaries = []
    for chunk_index, chunk_kv in enumerate(result.enriched_chunk_kvs):
        assert first_layer_seq_len(chunk_kv) == len(tokenized.doc_chunk_ids[chunk_index])
        metadata = chunk_kv.metadata
        for key in (
            "target_chunk_index",
            "neighbor_indices",
            "neighbor_scores",
            "neighbor_token_count",
            "source_positions",
            "target_absolute_positions",
            "rope_source_position_correction_applied",
        ):
            assert key in metadata, f"enriched chunk metadata must include {key!r}."
        assert metadata["target_chunk_index"] == chunk_index
        assert len(metadata["source_positions"]) == len(tokenized.doc_chunk_ids[chunk_index])
        assert len(metadata["target_absolute_positions"]) == len(
            tokenized.doc_chunk_ids[chunk_index]
        )
        if (
            args.model_family in {"mistral", "qwen2"}
            and args.apply_rope_source_position_correction
        ):
            assert metadata["rope_source_position_correction_applied"] is True
        chunk_summaries.append(
            {
                "chunk_index": chunk_index,
                "target_seq_len": first_layer_seq_len(chunk_kv),
                "neighbor_indices": metadata["neighbor_indices"],
                "neighbor_token_count": metadata["neighbor_token_count"],
                "rope_source_position_correction_applied": metadata[
                    "rope_source_position_correction_applied"
                ],
            }
        )

    return {
        "model": args.model,
        "model_family": args.model_family,
        "metadata": result.metadata,
        "chunks": chunk_summaries,
    }


def main() -> None:
    args = parse_args()
    if args.neighbor_top_n < 0:
        raise ValueError("--neighbor-top-n must be non-negative.")

    synthetic_doc_chunk_ids = [
        [1, 2, 3, 2, 1],
        [1, 2, 4, 2, 1],
        [8, 9, 10, 9],
    ]
    neighbor_summary = assert_neighbor_plans(
        synthetic_doc_chunk_ids,
        neighbor_top_n=args.neighbor_top_n,
    )
    output: dict[str, Any] = {
        "neighbor_plan_smoke": neighbor_summary,
    }
    if args.model is not None:
        output["enriched_precompute_smoke"] = run_model_smoke(args)

    print("smoke_fusionrag_enriched_precompute passed")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
