from __future__ import annotations

import argparse
import json
import math
from typing import Any

from contextflow.data import (
    build_cacheblend_prompt,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.kv_cache import assemble_chunk_kvs, precompute_enriched_doc_chunk_kvs
from contextflow.repair.fusionrag_selector import (
    compute_query_guided_token_scores,
    select_fusionrag_query_guided_tokens,
    select_top_recompute_indices_from_scores,
)


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "fusionrag-qgs-smoke-0",
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
        description="Smoke test FusionRAG-style query-guided token selection."
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional HuggingFace causal LM name or local path. Omit to run tensor-only checks.",
    )
    parser.add_argument(
        "--model-family",
        default="gpt2",
        choices=["gpt2", "mistral", "qwen2"],
    )
    parser.add_argument("--neighbor-top-n", type=int, default=1)
    parser.add_argument("--recompute-ratio", type=float, default=0.15)
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


def run_tensor_only_smoke(recompute_ratio: float) -> dict[str, Any]:
    import torch

    query_states = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    doc_key = torch.tensor([[[[3.0, 0.0], [1.0, 0.0], [0.0, 4.0], [0.0, 1.0]]]])
    token_scores = compute_query_guided_token_scores(
        query_states=query_states,
        doc_final_layer_key=doc_key,
        doc_chunk_lengths=[2, 2],
        attention_scale=1.0,
    )
    selected_indices, selected_by_score, selected_scores = select_top_recompute_indices_from_scores(
        token_scores,
        recompute_ratio=recompute_ratio,
    )
    expected_count = max(1, min(4, math.ceil(4 * recompute_ratio)))
    assert len(selected_indices) == expected_count
    assert selected_indices == sorted(selected_indices)
    assert len(selected_by_score) == expected_count
    assert len(selected_scores) == expected_count
    assert all(0 <= index < 4 for index in selected_indices)
    return {
        "token_scores": [float(score) for score in token_scores.tolist()],
        "selected_indices": selected_indices,
        "selected_indices_by_score": selected_by_score,
        "selected_scores": selected_scores,
    }


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
        enriched = precompute_enriched_doc_chunk_kvs(
            model=bundle.model,
            example=tokenized,
            neighbor_top_n=args.neighbor_top_n,
            model_family=args.model_family,
            apply_rope_source_position_correction=(
                args.apply_rope_source_position_correction
            ),
        )
        assembled_enriched_kv = assemble_chunk_kvs(enriched.enriched_chunk_kvs)
        doc_chunk_lengths = [len(chunk_ids) for chunk_ids in tokenized.doc_chunk_ids]
        selection = select_fusionrag_query_guided_tokens(
            model=bundle.model,
            q_ids=tokenized.q_ids,
            doc_past_key_values=assembled_enriched_kv,
            doc_chunk_lengths=doc_chunk_lengths,
            recompute_ratio=args.recompute_ratio,
            model_family=args.model_family,
        )

    doc_token_count = sum(doc_chunk_lengths)
    expected_count = max(1, min(doc_token_count, math.ceil(doc_token_count * args.recompute_ratio)))
    assert selection.recompute_count == expected_count
    assert len(selection.selected_indices) == expected_count
    assert selection.selected_indices == sorted(selection.selected_indices)
    assert len(selection.selected_indices_by_score) == expected_count
    assert len(selection.selected_scores) == expected_count
    assert int(selection.token_scores.shape[0]) == doc_token_count
    assert all(0 <= index < doc_token_count for index in selection.selected_indices)
    assert selection.metadata["score_source"] == "final_layer_query_key_attention_weights"
    assert selection.metadata["attention_score_normalization"] == "per_chunk_softmax"
    assert selection.metadata["query_position_offset"] == doc_token_count
    assert selection.metadata["query_position_basis"] == "document_absolute"

    return {
        "model": args.model,
        "model_family": args.model_family,
        "doc_token_count": doc_token_count,
        "doc_chunk_lengths": doc_chunk_lengths,
        "recompute_ratio": args.recompute_ratio,
        "recompute_count": selection.recompute_count,
        "selected_indices": selection.selected_indices,
        "selected_indices_by_score": selection.selected_indices_by_score,
        "selected_scores": selection.selected_scores,
        "metadata": selection.to_metadata(),
    }


def main() -> None:
    args = parse_args()
    output: dict[str, Any] = {
        "tensor_only_smoke": run_tensor_only_smoke(args.recompute_ratio),
    }
    if args.model is not None:
        output["model_smoke"] = run_model_smoke(args)

    print("smoke_fusionrag_query_selection passed")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
