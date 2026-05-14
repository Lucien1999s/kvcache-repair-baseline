from __future__ import annotations

import argparse
import json

from contextflow.data import (
    MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT,
    SUPPORTED_QA_DATASET_KEYS,
    TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    parse_qa_dataset_examples,
)
from contextflow.evaluation import (
    aggregate_qa_metrics,
    compute_normalized_score,
    evaluate_qa_prediction,
    parse_cacheblend_generation,
)
from contextflow.profiling import (
    get_peak_memory_mb_if_available,
    reset_peak_memory_stats_if_available,
    timed_call,
)


MUSIQUE_SAMPLE = {
    "id": "musique-smoke-0",
    "question": "Who wrote Pride and Prejudice",
    "answer": "Jane Austen",
    "paragraphs": [
        {
            "title": "Pride and Prejudice",
            "paragraph_text": "Pride and Prejudice is a novel by Jane Austen.",
            "is_supporting": True,
        },
        {
            "title": "English novels",
            "paragraph_text": "Many English novels were published in the nineteenth century.",
            "is_supporting": False,
        },
    ],
}


TWOWIKI_SAMPLE = {
    "_id": "2wiki-smoke-0",
    "question": "Who wrote Pride and Prejudice",
    "answer": "Jane Austen",
    "context": [
        ["Pride and Prejudice", ["Pride and Prejudice is a novel by Jane Austen."]],
        ["Jane Austen", ["Jane Austen was an English novelist."]],
    ],
}


LONGBOOK_SAMPLE = {
    "id": "longbook-smoke-0",
    "input": "Who wrote Pride and Prejudice",
    "answers": ["Jane Austen"],
    "context": (
        "Pride and Prejudice is a novel by Jane Austen. "
        "The book follows Elizabeth Bennet and Fitzwilliam Darcy."
    ),
    "title": "Pride and Prejudice",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test data, prompt, evaluation, and profiling helpers."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(SUPPORTED_QA_DATASET_KEYS),
        default=None,
        help="Optional local dataset key to parse after inline checks.",
    )
    parser.add_argument(
        "--input",
        default=None,
        help="Optional local JSON/JSONL path to validate with --dataset.",
    )
    parser.add_argument("--limit", type=int, default=3)
    return parser.parse_args()


def assert_dataset_parsing() -> None:
    musique = parse_qa_dataset_examples("musique", [MUSIQUE_SAMPLE])[0]
    twowiki = parse_qa_dataset_examples("2wiki", [TWOWIKI_SAMPLE])[0]
    longbook = parse_qa_dataset_examples("longbook_qa_en", [LONGBOOK_SAMPLE])[0]

    assert musique.example_id == "musique-smoke-0"
    assert musique.question == "Who wrote Pride and Prejudice"
    assert musique.answers == ["Jane Austen"]
    assert len(musique.ctxs) == 2
    assert musique.ctxs[0].title == "Pride and Prejudice"
    assert musique.ctxs[0].text == "Pride and Prejudice is a novel by Jane Austen."

    assert twowiki.example_id == "2wiki-smoke-0"
    assert twowiki.answers == ["Jane Austen"]
    assert len(twowiki.ctxs) == 2
    assert twowiki.ctxs[0].title == "Pride and Prejudice"
    assert twowiki.ctxs[0].text == "Pride and Prejudice is a novel by Jane Austen."

    assert longbook.example_id == "longbook-smoke-0"
    assert longbook.question == "Who wrote Pride and Prejudice"
    assert longbook.answers == ["Jane Austen"]
    assert len(longbook.ctxs) == 1
    assert longbook.ctxs[0].title == "Pride and Prejudice"
    assert "Jane Austen" in longbook.ctxs[0].text


def assert_prompting() -> None:
    musique = parse_qa_dataset_examples("musique", [MUSIQUE_SAMPLE])[0]
    twowiki = parse_qa_dataset_examples("2wiki", [TWOWIKI_SAMPLE])[0]

    default_prompt = build_cacheblend_prompt(musique, prompt_policy="default")
    musique_prompt = build_cacheblend_prompt(
        musique,
        prompt_policy="cacheblend_qa",
        dataset="musique",
    )
    twowiki_prompt = build_cacheblend_prompt(
        twowiki,
        prompt_policy="cacheblend_qa",
        dataset="2wiki",
    )

    expected_context = "Pride and Prejudice\n\nPride and Prejudice is a novel by Jane Austen.\n\n"
    assert default_prompt.doc_prompts[0] == expected_context
    assert default_prompt.q_prompt == "who wrote Pride and Prejudice?\nAnswer:"
    assert musique_prompt.doc_prompts[0] == expected_context
    assert musique_prompt.q_prompt == (
        f"{MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT}who wrote Pride and Prejudice?\nAnswer:"
    )
    assert twowiki_prompt.q_prompt == (
        f"{TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT}who wrote Pride and Prejudice?\nAnswer:"
    )


def assert_evaluation() -> None:
    assert parse_cacheblend_generation("\n\nJane Austen\nbecause the context says so") == "Jane Austen"
    assert parse_cacheblend_generation("Yes, it is") == "Yes"
    assert parse_cacheblend_generation("no, it is not") == "No"

    exact = evaluate_qa_prediction("The Jane Austen!", ["Jane Austen"])
    partial = evaluate_qa_prediction("Austen", ["Jane Austen"])
    miss = evaluate_qa_prediction("Charles Dickens", ["Jane Austen"])
    aggregate = aggregate_qa_metrics([exact, partial, miss])

    assert exact["exact_match"] == 1.0
    assert partial["f1"] == 2 / 3
    assert miss["f1"] == 0.0
    assert aggregate["count"] == 3
    assert aggregate["mean_exact_match"] == 1 / 3
    assert compute_normalized_score(
        method_score=0.75,
        lower_score=0.5,
        upper_score=1.0,
    ) == 0.5
    assert compute_normalized_score(
        method_score=0.8,
        lower_score=0.7,
        upper_score=0.7,
    ) is None


def assert_profiling() -> None:
    reset_peak_memory_stats_if_available()
    peak_memory_mb = get_peak_memory_mb_if_available()
    assert peak_memory_mb is None or peak_memory_mb >= 0

    result, profile_record = timed_call(
        lambda: {"ok": True},
        synchronize_cuda=True,
        collect_peak_memory=True,
    )
    assert result == {"ok": True}
    assert profile_record["latency_seconds"] >= 0
    if "peak_gpu_memory_mb" in profile_record:
        assert profile_record["peak_gpu_memory_mb"] is not None
        assert profile_record["peak_gpu_memory_mb"] >= 0
    json.dumps(profile_record)


def validate_local_dataset_input(dataset: str | None, input_path: str | None, limit: int) -> None:
    if dataset is None and input_path is None:
        return
    if dataset is None or input_path is None:
        raise ValueError("--dataset and --input must be provided together.")
    if limit <= 0:
        raise ValueError("--limit must be positive.")

    examples = load_qa_dataset_examples(dataset, input_path)
    assert examples, "local dataset input must contain at least one example."
    for index, example in enumerate(examples[:limit]):
        assert example.question, "question must be non-empty."
        assert example.ctxs, "ctxs must be non-empty."
        print(
            json.dumps(
                {
                    "index": index,
                    "example_id": example.example_id,
                    "question": example.question,
                    "answers": example.answers,
                    "num_ctxs": len(example.ctxs),
                    "first_ctx_title": example.ctxs[0].title,
                    "first_ctx_text_preview": example.ctxs[0].text[:120],
                },
                ensure_ascii=False,
            )
        )


def main() -> None:
    args = parse_args()
    assert_dataset_parsing()
    assert_prompting()
    assert_evaluation()
    assert_profiling()
    validate_local_dataset_input(args.dataset, args.input, args.limit)

    print("smoke_data_eval passed")
    print("covered: dataset loaders, prompt policies, QA parser, QA metrics, profiling helpers")


if __name__ == "__main__":
    main()
