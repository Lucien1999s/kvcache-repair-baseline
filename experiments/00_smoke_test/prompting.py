from __future__ import annotations

from contextflow.data import (
    Context,
    InputExample,
    MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT,
    TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT,
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
    musique_prompt = build_cacheblend_prompt(
        example,
        prompt_policy="cacheblend_qa",
        dataset="musique",
    )
    twowiki_prompt = build_cacheblend_prompt(
        example,
        prompt_policy="cacheblend_qa",
        dataset="2wiki",
    )

    expected_context = "Pride and Prejudice\n\nPride and Prejudice is a novel by Jane Austen.\n\n"
    expected_default_query = "who wrote the novel Pride and Prejudice?\nAnswer:"
    expected_question_and_answer = "who wrote the novel Pride and Prejudice?\nAnswer:"

    assert format_cacheblend_context(example.ctxs[0]) == expected_context
    assert default_prompt.doc_prompts == [expected_context]
    assert default_prompt.q_prompt == expected_default_query

    assert musique_prompt.doc_prompts == default_prompt.doc_prompts
    assert musique_prompt.q_prompt == (
        f"{MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT}{expected_question_and_answer}"
    )
    assert "Do NOT repeat the question." in musique_prompt.q_prompt
    assert "within 5 words" in musique_prompt.q_prompt

    assert twowiki_prompt.doc_prompts == default_prompt.doc_prompts
    assert twowiki_prompt.q_prompt == (
        f"{TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT}{expected_question_and_answer}"
    )
    assert "output any other words" in twowiki_prompt.q_prompt
    assert twowiki_prompt.q_prompt.endswith("\nAnswer:")

    print("prompting smoke test passed")
    print(f"default_q_prompt: {default_prompt.q_prompt!r}")
    print(f"musique_q_prompt: {musique_prompt.q_prompt!r}")
    print(f"twowiki_q_prompt: {twowiki_prompt.q_prompt!r}")


if __name__ == "__main__":
    main()
