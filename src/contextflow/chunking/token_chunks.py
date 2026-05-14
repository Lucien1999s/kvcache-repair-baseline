from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from contextflow.data.schema import TokenizedExample


@dataclass(frozen=True, slots=True)
class TokenChunkingConfig:
    chunk_size_tokens: int
    chunk_overlap_tokens: int = 0
    max_chunks: int | None = None

    def to_metadata(self) -> dict[str, int | None | str]:
        return {
            "chunking_mode": "token",
            "chunk_size_tokens": self.chunk_size_tokens,
            "chunk_overlap_tokens": self.chunk_overlap_tokens,
            "max_chunks": self.max_chunks,
        }


def validate_token_chunking_config(config: TokenChunkingConfig) -> None:
    if config.chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive.")
    if config.chunk_overlap_tokens < 0:
        raise ValueError("chunk_overlap_tokens must be non-negative.")
    if config.chunk_overlap_tokens >= config.chunk_size_tokens:
        raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens.")
    if config.max_chunks is not None and config.max_chunks <= 0:
        raise ValueError("max_chunks must be positive when provided.")


def chunk_token_ids(
    token_ids: list[int],
    chunk_size_tokens: int,
    chunk_overlap_tokens: int = 0,
    max_chunks: int | None = None,
) -> list[list[int]]:
    """Split a flat token sequence into fixed-size chunks with optional overlap."""

    config = TokenChunkingConfig(
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        max_chunks=max_chunks,
    )
    validate_token_chunking_config(config)
    if not token_ids:
        return []

    chunks: list[list[int]] = []
    step = chunk_size_tokens - chunk_overlap_tokens
    start = 0
    while start < len(token_ids):
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        end = min(start + chunk_size_tokens, len(token_ids))
        chunks.append(list(token_ids[start:end]))
        if end >= len(token_ids):
            break
        start += step
    return chunks


def flatten_doc_chunk_ids(doc_chunk_ids: list[list[int]]) -> list[int]:
    flat_ids: list[int] = []
    for chunk_ids in doc_chunk_ids:
        flat_ids.extend(chunk_ids)
    return flat_ids


def token_chunk_tokenized_example(
    tokenized_example: TokenizedExample,
    chunk_size_tokens: int,
    chunk_overlap_tokens: int = 0,
    max_chunks: int | None = None,
) -> TokenizedExample:
    """Return a TokenizedExample whose document side is token-size chunked."""

    flat_doc_ids = flatten_doc_chunk_ids(tokenized_example.doc_chunk_ids)
    token_chunks = chunk_token_ids(
        flat_doc_ids,
        chunk_size_tokens=chunk_size_tokens,
        chunk_overlap_tokens=chunk_overlap_tokens,
        max_chunks=max_chunks,
    )
    if not token_chunks:
        raise ValueError("Token chunking produced no document chunks.")
    return TokenizedExample(
        doc_chunk_ids=token_chunks,
        q_ids=list(tokenized_example.q_ids),
        answers=list(tokenized_example.answers),
        example_id=tokenized_example.example_id,
    )


def slice_tokenized_doc_chunks(
    tokenized_example: TokenizedExample,
    chunk_count: int,
) -> TokenizedExample:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be positive.")
    if chunk_count > len(tokenized_example.doc_chunk_ids):
        raise ValueError(
            "chunk_count cannot exceed available token chunks; "
            f"got {chunk_count} and {len(tokenized_example.doc_chunk_ids)}."
        )
    return TokenizedExample(
        doc_chunk_ids=[list(chunk) for chunk in tokenized_example.doc_chunk_ids[:chunk_count]],
        q_ids=list(tokenized_example.q_ids),
        answers=list(tokenized_example.answers),
        example_id=tokenized_example.example_id,
    )


def token_chunking_metadata(
    config: TokenChunkingConfig,
    *,
    source_doc_token_count: int,
    produced_chunk_count: int,
) -> dict[str, Any]:
    metadata = config.to_metadata()
    metadata.update(
        {
            "source_doc_token_count": source_doc_token_count,
            "produced_chunk_count": produced_chunk_count,
        }
    )
    return metadata
