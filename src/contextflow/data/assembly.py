from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from contextflow.data.schema import PromptExample, TokenizedExample


class AssemblyTokenizer(Protocol):
    """Minimal tokenizer protocol needed for full-prefill prompt assembly."""

    def encode(self, text: str, **kwargs: Any) -> list[int]:
        ...

    def decode(self, token_ids: Sequence[int], **kwargs: Any) -> str:
        ...


@dataclass(slots=True)
class FullPrefillAssemblyConfig:
    """Model-agnostic full-prefill assembly settings."""

    prefix_prompt: str = ""
    use_chat_template: bool = True
    chat_role: str = "user"
    add_generation_prompt: bool = True


@dataclass(slots=True)
class AssembledInput:
    """Fully assembled model input for normal/full-prefill generation."""

    input_ids: list[int]
    input_text: str | None
    example_id: str | None = None
    suffix_len: int | None = None


def build_full_prefill_content(
    example: PromptExample,
    config: FullPrefillAssemblyConfig | None = None,
) -> str:
    """Build model-agnostic full-prefill user content from prompt-formatted inputs."""

    assembly_config = config or FullPrefillAssemblyConfig()
    return f"{assembly_config.prefix_prompt}{''.join(example.doc_prompts)}{example.q_prompt}"


def assemble_full_prefill_input_ids(
    example: PromptExample,
    tokenizer: AssemblyTokenizer,
    config: FullPrefillAssemblyConfig | None = None,
) -> list[int]:
    """Assemble prompt-level full-prefill input ids without model-specific hardcoded tokens."""

    assembly_config = config or FullPrefillAssemblyConfig()
    content = build_full_prefill_content(example, config=assembly_config)

    if assembly_config.use_chat_template and getattr(tokenizer, "chat_template", None):
        messages = [{"role": assembly_config.chat_role, "content": content}]
        try:
            input_ids = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=assembly_config.add_generation_prompt,
            )
            if hasattr(input_ids, "tolist"):
                input_ids = input_ids.tolist()
            if input_ids and isinstance(input_ids[0], list):
                input_ids = input_ids[0]
            return list(input_ids)
        except Exception:
            pass

    return list(tokenizer.encode(content, add_special_tokens=True))


def assemble_full_prefill_input_from_prompt(
    example: PromptExample,
    tokenizer: AssemblyTokenizer,
    config: FullPrefillAssemblyConfig | None = None,
) -> AssembledInput:
    """Assemble a PromptExample into full-prefill input ids and decoded input text."""

    input_ids = assemble_full_prefill_input_ids(example, tokenizer=tokenizer, config=config)
    return AssembledInput(
        input_ids=input_ids,
        input_text=tokenizer.decode(input_ids),
        example_id=example.example_id,
    )


def assemble_token_aligned_full_prefill_input_ids(example: TokenizedExample) -> list[int]:
    """Assemble the logical token layout used by naive KV reuse: docs followed by query."""

    input_ids: list[int] = []
    for doc_chunk_ids in example.doc_chunk_ids:
        input_ids.extend(doc_chunk_ids)
    input_ids.extend(example.q_ids)
    return input_ids


def assemble_token_aligned_full_prefill_input(
    example: TokenizedExample,
    tokenizer: AssemblyTokenizer | None = None,
) -> AssembledInput:
    """Assemble a TokenizedExample into the token-aligned full-prefill reference input."""

    input_ids = assemble_token_aligned_full_prefill_input_ids(example)
    input_text = tokenizer.decode(input_ids) if tokenizer is not None else None
    return AssembledInput(
        input_ids=input_ids,
        input_text=input_text,
        example_id=example.example_id,
        suffix_len=len(example.q_ids),
    )

