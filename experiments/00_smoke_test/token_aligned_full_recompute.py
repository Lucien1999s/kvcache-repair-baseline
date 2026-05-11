from __future__ import annotations

import argparse
from typing import Any

from transformers import AutoTokenizer

from contextflow.data import (
    InputExample,
    assemble_token_aligned_full_prefill_input_ids,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.methods import (
    run_naive_reuse_generation,
    run_token_aligned_full_recompute_greedy_generation,
)
from contextflow.runtime import load_hf_causal_lm
from contextflow.runtime.hf_cached_generation import infer_past_sequence_length


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
        description=(
            "Smoke test token-aligned full recompute greedy generation and its layout "
            "alignment with naive KV reuse."
        )
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to a CacheBlend-style JSON list.",
    )
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace tokenizer/model name or local path.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1,
        help="Number of examples to check.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens when --run-generation is enabled.",
    )
    parser.add_argument(
        "--run-generation",
        action="store_true",
        help="Also run token-aligned full recompute greedy and naive reuse generation.",
    )
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def concat_doc_chunk_ids(doc_chunk_ids: list[list[int]]) -> list[int]:
    input_ids: list[int] = []
    for doc_ids in doc_chunk_ids:
        input_ids.extend(doc_ids)
    return input_ids


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    bundle = load_hf_causal_lm(args.model) if args.run_generation else None

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, tokenizer)

        expected_input_ids = concat_doc_chunk_ids(tokenized.doc_chunk_ids) + tokenized.q_ids
        token_aligned_input_ids = assemble_token_aligned_full_prefill_input_ids(tokenized)

        assert token_aligned_input_ids == expected_input_ids, (
            "token-aligned full-prefill input must equal concat(doc_chunk_ids) + q_ids."
        )

        doc_token_count = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)
        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"doc_chunk_count: {len(tokenized.doc_chunk_ids)}")
        print(f"doc_token_count: {doc_token_count}")
        print(f"q_ids_length: {len(tokenized.q_ids)}")
        print(f"token_aligned_input_ids_length: {len(token_aligned_input_ids)}")
        print("token_aligned_layout: ok")

        if bundle is None:
            continue

        full_result = run_token_aligned_full_recompute_greedy_generation(
            tokenized,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )
        naive_result = run_naive_reuse_generation(
            tokenized,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

        assembled_kv_seq_len = infer_past_sequence_length(
            naive_result.assembled_past_key_values
        )
        assert assembled_kv_seq_len == doc_token_count, (
            "naive reuse assembled KV seq_len must equal summed doc token length."
        )
        assert full_result.assembled.input_ids == token_aligned_input_ids, (
            "token-aligned full-prefill method must use the checked input ids."
        )

        print(f"token_aligned_full_recompute_output_text: {full_result.generation.output_text!r}")
        print(f"naive_reuse_output_text: {naive_result.generation.output_text!r}")
        print(f"naive_reuse_assembled_kv_seq_len: {assembled_kv_seq_len}")


if __name__ == "__main__":
    main()
