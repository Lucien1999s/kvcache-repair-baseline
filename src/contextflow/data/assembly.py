from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

from contextflow.data.schema import PromptExample, TokenizedExample
from contextflow.data.tokenization import encode_cacheblend_text


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
class CacheBlendMistralAssemblyConfig:
    """Token-level preset matching the CacheBlend repo's Mistral-7B-Instruct example."""

    prefix_start_ids: list[int] = field(default_factory=lambda: [733, 16289, 28793])
    chunk_start_ids: list[int] = field(default_factory=list)
    suffix_ids: list[int] = field(default_factory=lambda: [733, 28748, 16289, 28793])
    prefix_prompt: str = ""
    strip_first_token: bool = True


@dataclass(slots=True)
class AssembledInput:
    """Fully assembled model input for normal/full-prefill generation."""

    input_ids: list[int]
    input_text: str | None
    example_id: str | None = None
    suffix_len: int | None = None


def get_cacheblend_mistral_instruct_config() -> CacheBlendMistralAssemblyConfig:
    """Return the CacheBlend repo's hardcoded Mistral-7B-Instruct assembly preset."""

    return CacheBlendMistralAssemblyConfig()


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


def encode_cacheblend_mistral_prefix_prompt(
    tokenizer: AssemblyTokenizer | None,
    prefix_prompt: str,
    strip_first_token: bool,
) -> list[int]:
    if not prefix_prompt:
        return []
    if tokenizer is None:
        raise ValueError(
            "A tokenizer is required when CacheBlendMistralAssemblyConfig.prefix_prompt is set."
        )
    return encode_cacheblend_text(
        tokenizer,
        prefix_prompt,
        strip_first_token=strip_first_token,
    )


def assemble_cacheblend_mistral_input_ids(
    example: TokenizedExample,
    tokenizer: AssemblyTokenizer | None = None,
    config: CacheBlendMistralAssemblyConfig | None = None,
) -> list[int]:
    """Assemble token ids using the CacheBlend repo's Mistral-7B-Instruct preset."""

    assembly_config = config or get_cacheblend_mistral_instruct_config()
    prefix_prompt_ids = encode_cacheblend_mistral_prefix_prompt(
        tokenizer,
        assembly_config.prefix_prompt,
        strip_first_token=assembly_config.strip_first_token,
    )

    input_ids: list[int] = []
    input_ids.extend(assembly_config.prefix_start_ids)
    input_ids.extend(prefix_prompt_ids)
    for doc_chunk_ids in example.doc_chunk_ids:
        input_ids.extend(assembly_config.chunk_start_ids)
        input_ids.extend(doc_chunk_ids)
    input_ids.extend(assembly_config.chunk_start_ids)
    input_ids.extend(example.q_ids)
    input_ids.extend(assembly_config.suffix_ids)
    return input_ids


def assemble_cacheblend_mistral_input(
    example: TokenizedExample,
    tokenizer: AssemblyTokenizer | None = None,
    config: CacheBlendMistralAssemblyConfig | None = None,
) -> AssembledInput:
    """Assemble a TokenizedExample with the CacheBlend Mistral exact reproduction preset."""

    assembly_config = config or get_cacheblend_mistral_instruct_config()
    input_ids = assemble_cacheblend_mistral_input_ids(
        example,
        tokenizer=tokenizer,
        config=assembly_config,
    )
    input_text = tokenizer.decode(input_ids) if tokenizer is not None else None
    suffix_len = len(assembly_config.chunk_start_ids) + len(example.q_ids) + len(
        assembly_config.suffix_ids
    )
    return AssembledInput(
        input_ids=input_ids,
        input_text=input_text,
        example_id=example.example_id,
        suffix_len=suffix_len,
    )
