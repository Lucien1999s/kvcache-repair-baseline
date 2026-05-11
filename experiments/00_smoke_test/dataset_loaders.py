from __future__ import annotations

import argparse
from typing import Any

from contextflow.data import (
    InputExample,
    load_qa_dataset_examples,
    normalize_dataset_key,
    parse_qa_dataset_examples,
)


INLINE_MUSIQUE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "musique-sample-0",
        "question": "Who wrote the novel that features Elizabeth Bennet?",
        "answer": "Jane Austen",
        "paragraphs": [
            {
                "title": "Pride and Prejudice",
                "paragraph_text": "Pride and Prejudice features Elizabeth Bennet.",
                "is_supporting": True,
            },
            {
                "title": "Jane Austen",
                "paragraph_text": "Jane Austen wrote Pride and Prejudice.",
                "is_supporting": True,
            },
        ],
    }
]


INLINE_2WIKI_SAMPLE: list[dict[str, Any]] = [
    {
        "_id": "2wiki-sample-0",
        "question": "Who wrote the novel that features Elizabeth Bennet?",
        "answer": "Jane Austen",
        "context": [
            [
                "Pride and Prejudice",
                ["Pride and Prejudice is a novel.", "It features Elizabeth Bennet."],
            ],
            [
                "Jane Austen",
                ["Jane Austen was an English novelist.", "She wrote Pride and Prejudice."],
            ],
        ],
        "supporting_facts": [["Pride and Prejudice", 1], ["Jane Austen", 1]],
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test MuSiQue and 2Wiki dataset loaders into InputExample."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["musique", "2wiki"],
        help=(
            "Dataset key. '2wiki' refers to 2WikiMultiHopQA / 2WikiMQA-style "
            "multi-hop QA data."
        ),
    )
    parser.add_argument("--input", default=None, help="Optional local JSON or JSONL path.")
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to print.")
    return parser.parse_args()


def load_examples(dataset: str, input_path: str | None) -> list[InputExample]:
    dataset_key = normalize_dataset_key(dataset)
    if input_path is not None:
        return load_qa_dataset_examples(dataset_key, input_path)
    if dataset_key == "musique":
        return parse_qa_dataset_examples(dataset_key, INLINE_MUSIQUE_SAMPLE)
    if dataset_key == "2wiki":
        return parse_qa_dataset_examples(dataset_key, INLINE_2WIKI_SAMPLE)
    raise ValueError(f"Unsupported dataset key: {dataset!r}.")


def preview_text(text: str, max_chars: int = 80) -> str:
    collapsed = " ".join(text.split())
    if len(collapsed) <= max_chars:
        return collapsed
    return f"{collapsed[: max_chars - 3]}..."


def main() -> None:
    args = parse_args()
    examples = load_examples(args.dataset, args.input)
    assert examples, "Expected at least one parsed example."

    for index, example in enumerate(examples[: args.limit]):
        first_ctx = example.ctxs[0] if example.ctxs else None
        assert example.question, "question must be non-empty."
        assert example.ctxs, "ctxs must be non-empty."

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"question: {example.question}")
        print(f"answers: {example.answers}")
        print(f"num_ctxs: {len(example.ctxs)}")
        print(f"first_ctx_title: {first_ctx.title if first_ctx is not None else ''}")
        print(
            "first_ctx_text_preview: "
            f"{preview_text(first_ctx.text) if first_ctx is not None else ''}"
        )


if __name__ == "__main__":
    main()
