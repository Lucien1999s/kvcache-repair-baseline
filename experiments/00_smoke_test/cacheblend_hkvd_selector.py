from __future__ import annotations

import argparse
from typing import Any

from contextflow.data import (
    InputExample,
    assemble_token_aligned_full_prefill_input_ids,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.kv_cache import assemble_chunk_kvs, precompute_doc_chunk_kvs, precompute_kv_for_input_ids
from contextflow.repair import compute_kv_deviation, select_hkvd_tokens_by_layer
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "sample-0",
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
        description="Smoke test CacheBlend-style HKVD token selection diagnostic."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to a CacheBlend-style JSON list.",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace causal LM name or local path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of examples to check.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of HKVD token positions to select per layer.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Optional HKVD selection ratio per layer. If set, --top-k is ignored.",
    )
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def slice_past_key_values_prefix(past_key_values: Any, seq_len: int) -> tuple[tuple[Any, Any], ...]:
    sliced_layers = []
    for layer_index, layer_kv in enumerate(past_key_values):
        if len(layer_kv) < 2:
            raise ValueError(f"Layer {layer_index} must contain key and value tensors.")
        key, value = layer_kv[0], layer_kv[1]
        if int(key.shape[-2]) < seq_len or int(value.shape[-2]) < seq_len:
            raise ValueError(
                f"Layer {layer_index} KV seq_len is shorter than requested prefix length {seq_len}."
            )
        sliced_layers.append((key[..., :seq_len, :], value[..., :seq_len, :]))
    return tuple(sliced_layers)


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."

    bundle = load_hf_causal_lm(args.model)

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
        full_input_ids = assemble_token_aligned_full_prefill_input_ids(tokenized)
        doc_total_len = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)

        full_kv = precompute_kv_for_input_ids(
            bundle.model,
            full_input_ids,
            chunk_id=f"{example.example_id or index}:full",
        )
        full_doc_past_key_values = slice_past_key_values_prefix(
            full_kv.past_key_values,
            seq_len=doc_total_len,
        )

        chunk_kvs = precompute_doc_chunk_kvs(bundle.model, tokenized)
        reuse_doc_past_key_values = assemble_chunk_kvs(chunk_kvs)

        deviations = compute_kv_deviation(
            reuse_doc_past_key_values,
            full_doc_past_key_values,
        )
        selected = select_hkvd_tokens_by_layer(
            deviations,
            ratio=args.ratio,
            top_k=None if args.ratio is not None else args.top_k,
        )

        assert len(deviations) > 0, "Expected at least one deviation layer."
        for layer_index, layer_deviation in enumerate(deviations):
            assert int(layer_deviation.shape[0]) == doc_total_len, (
                f"Layer {layer_index} deviation seq_len must match doc_total_len."
            )
            assert selected[layer_index], f"Layer {layer_index} selected indices must be non-empty."
            assert all(0 <= token_index < doc_total_len for token_index in selected[layer_index]), (
                f"Layer {layer_index} selected indices must be in [0, doc_total_len)."
            )

        first_layer_pairs = sorted(
            (
                (token_index, float(deviations[0][token_index].detach().cpu()))
                for token_index in selected[0]
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"doc_total_len: {doc_total_len}")
        print(f"num_layers: {len(deviations)}")
        print(f"first_layer_top_indices: {[token_index for token_index, _ in first_layer_pairs]}")
        print(f"first_layer_top_deviation_scores: {[score for _, score in first_layer_pairs]}")


if __name__ == "__main__":
    main()
