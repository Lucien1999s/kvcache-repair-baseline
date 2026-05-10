from __future__ import annotations

from contextflow.data.schema import Context, InputExample, PromptExample


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


def build_cacheblend_prompt(example: InputExample, query_prompt: str = "") -> PromptExample:
    """Convert a raw QA example into CacheBlend-style document and query prompts."""

    q = normalize_cacheblend_question(example.question)
    return PromptExample(
        doc_prompts=[format_cacheblend_context(ctx) for ctx in example.ctxs],
        q_prompt=f"{query_prompt}{q}\nAnswer:",
        answers=list(example.answers),
        example_id=example.example_id,
    )


def build_cacheblend_prompts(
    examples: list[InputExample],
    query_prompt: str = "",
) -> list[PromptExample]:
    """Format multiple QA examples with the same query prompt prefix."""

    return [build_cacheblend_prompt(example, query_prompt=query_prompt) for example in examples]
