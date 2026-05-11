from __future__ import annotations

import argparse
from typing import Any

from transformers import AutoTokenizer

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)


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
        description="Smoke test the CacheBlend-style data preprocessing flow."
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional path to a CacheBlend-style JSON list.",
    )
    parser.add_argument(
        "--tokenizer",
        default="facebook/opt-125m",
        help="HuggingFace tokenizer name or local path.",
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


def assert_token_ids(name: str, token_ids: list[int]) -> None:
    assert token_ids, f"{name} must be non-empty."
    assert all(isinstance(token_id, int) for token_id in token_ids), (
        f"{name} must contain only int token ids."
    )


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, tokenizer)

        assert len(prompt.doc_prompts) == len(example.ctxs), (
            "doc_prompts must match the number of input ctxs."
        )
        assert len(tokenized.doc_chunk_ids) == len(prompt.doc_prompts), (
            "doc_chunk_ids must match the number of doc_prompts."
        )
        assert prompt.q_prompt.endswith("\nAnswer:"), "q_prompt must end with '\\nAnswer:'."
        assert_token_ids("q_ids", tokenized.q_ids)
        for doc_index, doc_chunk_ids in enumerate(tokenized.doc_chunk_ids):
            assert_token_ids(f"doc_chunk_ids[{doc_index}]", doc_chunk_ids)

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"question: {example.question}")
        print(f"answers: {example.answers}")
        print(f"number_of_docs: {len(prompt.doc_prompts)}")
        print(f"q_ids_length: {len(tokenized.q_ids)}")
        for doc_index, doc_chunk_ids in enumerate(tokenized.doc_chunk_ids):
            print(f"doc_chunk_ids[{doc_index}]_length: {len(doc_chunk_ids)}")


if __name__ == "__main__":
    main()
