from __future__ import annotations

from contextflow.data.schema import Context, InputExample, PromptExample


PROMPT_POLICY_DEFAULT = "default"
PROMPT_POLICY_CACHEBLEND_QA = "cacheblend_qa"
MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT = (
    "Answer the question directly based on the given passages. "
    "Do NOT repeat the question. The answer should be within 5 words. \n"
    "Question:"
)
TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT = (
    "Answer the question based on the given passages. "
    "Answer the question within 5 words. "
    "Do NOT repeat the question or output any other words. Question:"
)
GENERIC_CACHEBLEND_QA_QUERY_PROMPT = (
    "Answer the question within 5 words. Question:"
)
SUPPORTED_CACHEBLEND_PROMPT_POLICIES = {
    PROMPT_POLICY_DEFAULT,
    PROMPT_POLICY_CACHEBLEND_QA,
}


def format_cacheblend_context(ctx: Context) -> str:
    """Format one context passage the way CacheBlend prepares doc_prompts."""

    return f"{ctx.title}\n\n{ctx.text}\n\n"


def normalize_cacheblend_question(question: str) -> str:
    """Normalize a question following CacheBlend's QA prompt convention."""

    normalized = question
    if not normalized.endswith("?"):
        normalized = f"{normalized}?"
    if normalized:
        normalized = normalized[0].lower() + normalized[1:]
    return normalized


def validate_cacheblend_prompt_policy(prompt_policy: str) -> str:
    normalized_policy = prompt_policy.lower()
    if normalized_policy not in SUPPORTED_CACHEBLEND_PROMPT_POLICIES:
        raise ValueError(
            f"Unsupported prompt_policy={prompt_policy!r}. Supported policies: "
            f"{sorted(SUPPORTED_CACHEBLEND_PROMPT_POLICIES)}."
        )
    return normalized_policy


def normalize_prompt_dataset(dataset: str | None) -> str | None:
    if dataset is None:
        return None
    normalized = dataset.lower()
    if normalized in {"2wikimultihopqa", "2wikimqa", "wikimqa"}:
        return "2wiki"
    return normalized


def get_cacheblend_qa_query_prompt(dataset: str | None = None) -> str:
    """Return the CacheBlend QA query prompt for a supported benchmark."""

    normalized_dataset = normalize_prompt_dataset(dataset)
    if normalized_dataset == "musique":
        return MUSIQUE_CACHEBLEND_QA_QUERY_PROMPT
    if normalized_dataset == "2wiki":
        return TWOWIKI_CACHEBLEND_QA_QUERY_PROMPT
    return GENERIC_CACHEBLEND_QA_QUERY_PROMPT


def build_cacheblend_query_prompt(
    question: str,
    query_prompt: str = "",
    prompt_policy: str = PROMPT_POLICY_DEFAULT,
    dataset: str | None = None,
) -> str:
    """Build the query prompt for a CacheBlend-style QA example."""

    policy = validate_cacheblend_prompt_policy(prompt_policy)
    q = normalize_cacheblend_question(question)
    if policy == PROMPT_POLICY_DEFAULT:
        return f"{query_prompt}{q}\nAnswer:"
    if policy == PROMPT_POLICY_CACHEBLEND_QA:
        resolved_query_prompt = query_prompt or get_cacheblend_qa_query_prompt(dataset)
        return f"{resolved_query_prompt}{q}\nAnswer:"
    raise AssertionError(f"Unhandled prompt policy: {policy}")


def build_cacheblend_prompt(
    example: InputExample,
    query_prompt: str = "",
    prompt_policy: str = PROMPT_POLICY_DEFAULT,
    dataset: str | None = None,
) -> PromptExample:
    """Convert a raw QA example into CacheBlend-style document and query prompts."""

    return PromptExample(
        doc_prompts=[format_cacheblend_context(ctx) for ctx in example.ctxs],
        q_prompt=build_cacheblend_query_prompt(
            example.question,
            query_prompt=query_prompt,
            prompt_policy=prompt_policy,
            dataset=dataset,
        ),
        answers=list(example.answers),
        example_id=example.example_id,
    )


def build_cacheblend_prompts(
    examples: list[InputExample],
    query_prompt: str = "",
    prompt_policy: str = PROMPT_POLICY_DEFAULT,
    dataset: str | None = None,
) -> list[PromptExample]:
    """Format multiple QA examples with the same query prompt prefix."""

    return [
        build_cacheblend_prompt(
            example,
            query_prompt=query_prompt,
            prompt_policy=prompt_policy,
            dataset=dataset,
        )
        for example in examples
    ]
