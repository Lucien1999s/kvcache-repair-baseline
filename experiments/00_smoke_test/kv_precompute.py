from __future__ import annotations

import argparse
from typing import Any

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.kv_cache import precompute_doc_chunk_kvs
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
        description="Smoke test HF/PyTorch chunk-level KV precompute."
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
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def get_layer_key_value(layer_past: Any) -> tuple[Any, Any]:
    assert len(layer_past) >= 2, "Each KV layer must contain key and value tensors."
    return layer_past[0], layer_past[1]


def sequence_length(tensor: Any) -> int:
    assert len(tensor.shape) >= 3, f"Expected KV tensor rank >= 3, got shape {tensor.shape}."
    return int(tensor.shape[-2])


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."

    bundle = load_hf_causal_lm(args.model)

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
        chunk_kvs = precompute_doc_chunk_kvs(bundle.model, tokenized)

        assert len(chunk_kvs) == len(tokenized.doc_chunk_ids), (
            "chunk_kvs must match the number of doc_chunk_ids."
        )

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"number_of_chunks: {len(chunk_kvs)}")

        for chunk_index, chunk_kv in enumerate(chunk_kvs):
            expected_len = len(tokenized.doc_chunk_ids[chunk_index])
            assert chunk_kv.past_key_values, "past_key_values must be non-empty."
            assert chunk_kv.num_layers > 0, "num_layers must be positive."
            assert chunk_kv.num_tokens == expected_len, "num_tokens must match chunk token length."

            first_key, first_value = get_layer_key_value(chunk_kv.past_key_values[0])
            assert first_key is not None, "First layer key must exist."
            assert first_value is not None, "First layer value must exist."

            for layer_index, layer_past in enumerate(chunk_kv.past_key_values):
                key, value = get_layer_key_value(layer_past)
                assert sequence_length(key) == expected_len, (
                    f"Layer {layer_index} key sequence length must match chunk length."
                )
                assert sequence_length(value) == expected_len, (
                    f"Layer {layer_index} value sequence length must match chunk length."
                )

            print(f"chunk[{chunk_index}] id: {chunk_kv.chunk_id}")
            print(f"chunk[{chunk_index}] token_length: {chunk_kv.num_tokens}")
            print(f"chunk[{chunk_index}] num_layers: {chunk_kv.num_layers}")
            print(f"chunk[{chunk_index}] first_layer_key_shape: {tuple(first_key.shape)}")
            print(f"chunk[{chunk_index}] first_layer_value_shape: {tuple(first_value.shape)}")


if __name__ == "__main__":
    main()
