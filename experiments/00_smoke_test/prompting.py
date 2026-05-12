from __future__ import annotations

from contextflow.data import (
    Context,
    InputExample,
    build_cacheblend_prompt,
    format_cacheblend_context,
)


def build_example() -> InputExample:
    return InputExample(
        example_id="prompting-sample-0",
        question="Who wrote the novel Pride and Prejudice",
        answers=["Jane Austen"],
        ctxs=[
            Context(
                title="Pride and Prejudice",
                text="Pride and Prejudice is a novel by Jane Austen.",
            )
        ],
    )


def main() -> None:
    example = build_example()
    default_prompt = build_cacheblend_prompt(example, prompt_policy="default")
    cacheblend_qa_prompt = build_cacheblend_prompt(example, prompt_policy="cacheblend_qa")

    expected_context = "Pride and Prejudice\n\nPride and Prejudice is a novel by Jane Austen.\n\n"
    expected_default_query = "who wrote the novel Pride and Prejudice?\nAnswer:"
    expected_cacheblend_qa_query = (
        "who wrote the novel Pride and Prejudice?\n"
        "Answer within 5 words.\n"
        "Answer:"
    )

    assert format_cacheblend_context(example.ctxs[0]) == expected_context
    assert default_prompt.doc_prompts == [expected_context]
    assert default_prompt.q_prompt == expected_default_query
    assert cacheblend_qa_prompt.doc_prompts == default_prompt.doc_prompts
    assert cacheblend_qa_prompt.q_prompt == expected_cacheblend_qa_query
    assert "Answer within 5 words." in cacheblend_qa_prompt.q_prompt
    assert cacheblend_qa_prompt.q_prompt.startswith(
        "who wrote the novel Pride and Prejudice?\n"
    )
    assert cacheblend_qa_prompt.q_prompt.endswith("\nAnswer:")

    print("prompting smoke test passed")
    print(f"default_q_prompt: {default_prompt.q_prompt!r}")
    print(f"cacheblend_qa_q_prompt: {cacheblend_qa_prompt.q_prompt!r}")


if __name__ == "__main__":
    main()
