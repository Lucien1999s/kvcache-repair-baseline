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
from contextflow.methods.naive_reuse import run_naive_reuse_generation
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
    parser = argparse.ArgumentParser(description="Smoke test naive chunk KV reuse generation.")
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
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to greedily decode.",
    )
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."

    bundle = load_hf_causal_lm(args.model)

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
        result = run_naive_reuse_generation(
            tokenized,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            max_new_tokens=args.max_new_tokens,
        )

        assembled_seq_len = infer_past_sequence_length(result.assembled_past_key_values)
        expected_seq_len = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)
        assert result.assembled_past_key_values, "assembled KV must be non-empty."
        assert assembled_seq_len == expected_seq_len, (
            "assembled KV seq_len must match the sum of doc chunk lengths."
        )
        assert result.generation.generated_ids, "generated_ids must be non-empty."
        assert isinstance(result.generation.output_text, str), "output_text must be a string."

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"doc_chunk_count: {len(tokenized.doc_chunk_ids)}")
        print(f"assembled_kv_seq_len: {assembled_seq_len}")
        print(f"q_ids_length: {len(tokenized.q_ids)}")
        print(f"generated_ids_length: {len(result.generation.generated_ids)}")
        print(f"output_text: {result.generation.output_text!r}")


if __name__ == "__main__":
    main()
