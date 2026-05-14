from __future__ import annotations

from typing import Protocol

from contextflow.data.schema import PromptExample, TokenizedExample


class EncodeTokenizer(Protocol):
    """Minimal tokenizer protocol needed by the CacheBlend preprocessing flow."""

    def encode(self, text: str) -> list[int]:
        ...


def encode_cacheblend_text(
    tokenizer: EncodeTokenizer,
    text: str,
    strip_first_token: bool = True,
) -> list[int]:
    """Encode text with CacheBlend's default tokenizer.encode(text)[1:] convention."""

    token_ids = list(tokenizer.encode(text))
    if strip_first_token:
        return token_ids[1:]
    return token_ids


def tokenize_prompt_example(
    example: PromptExample,
    tokenizer: EncodeTokenizer,
    strip_first_token: bool = True,
) -> TokenizedExample:
    """Tokenize CacheBlend-style doc_prompts and q_prompt."""

    return TokenizedExample(
        doc_chunk_ids=[
            encode_cacheblend_text(tokenizer, doc_prompt, strip_first_token=strip_first_token)
            for doc_prompt in example.doc_prompts
        ],
        q_ids=encode_cacheblend_text(tokenizer, example.q_prompt, strip_first_token=strip_first_token),
        answers=list(example.answers),
        example_id=example.example_id,
    )


def tokenize_prompt_examples(
    examples: list[PromptExample],
    tokenizer: EncodeTokenizer,
    strip_first_token: bool = True,
) -> list[TokenizedExample]:
    """Tokenize multiple prompt-formatted examples."""

    return [
        tokenize_prompt_example(
            example,
            tokenizer,
            strip_first_token=strip_first_token,
        )
        for example in examples
    ]
