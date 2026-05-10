from __future__ import annotations

import argparse
from typing import Any

from contextflow.data import (
    InputExample,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
)
from contextflow.methods import run_full_recompute_generation
from contextflow.runtime import HFGenerationConfig, load_hf_causal_lm


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
        description="Smoke test the general full-prefill HF generation path."
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
        help="Number of examples to generate for.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=16,
        help="Maximum number of new tokens to generate.",
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
    generation_config = HFGenerationConfig(
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        result = run_full_recompute_generation(
            prompt,
            model=bundle.model,
            tokenizer=bundle.tokenizer,
            generation_config=generation_config,
        )

        assert result.assembled.input_ids, "assembled.input_ids must be non-empty."
        assert result.generation.generated_ids, "generation.generated_ids must be non-empty."
        assert isinstance(result.generation.output_text, str), "output_text must be a string."
        assert isinstance(result.generation.full_text, str), "full_text must be a string."

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"question: {example.question}")
        print(f"number_of_docs: {len(prompt.doc_prompts)}")
        print(f"input_ids_length: {len(result.assembled.input_ids)}")
        print(f"generated_ids_length: {len(result.generation.generated_ids)}")
        print(f"output_text: {result.generation.output_text!r}")


if __name__ == "__main__":
    main()
