from __future__ import annotations

from typing import Any

from contextflow.data import InputExample


CHUNKING_MODE_CONTEXT = "context"
CHUNKING_MODE_TOKEN = "token"
SUPPORTED_CHUNKING_MODES = {CHUNKING_MODE_CONTEXT, CHUNKING_MODE_TOKEN}


def validate_chunking_mode(chunking_mode: str) -> str:
    normalized_mode = chunking_mode.strip().lower()
    if normalized_mode not in SUPPORTED_CHUNKING_MODES:
        raise ValueError(
            f"Unsupported chunking mode {chunking_mode!r}. "
            f"Supported modes: {sorted(SUPPORTED_CHUNKING_MODES)}."
        )
    return normalized_mode


def parse_chunk_count_specs(raw_counts: str) -> list[int | str]:
    specs: list[int | str] = []
    for part in raw_counts.split(","):
        value = part.strip()
        if not value:
            continue
        if value == "all":
            specs.append(value)
            continue
        count = int(value)
        if count <= 0:
            raise ValueError("--chunk-counts values must be positive.")
        specs.append(count)
    if not specs:
        raise ValueError("--chunk-counts must contain at least one value.")
    return specs


def resolve_example_chunk_cases(
    example: InputExample,
    chunk_count_specs: list[int | str],
) -> list[dict[str, Any]]:
    return resolve_chunk_cases_for_available_count(
        available_count=len(example.ctxs),
        chunk_count_specs=chunk_count_specs,
    )


def resolve_chunk_cases_for_available_count(
    available_count: int,
    chunk_count_specs: list[int | str],
) -> list[dict[str, Any]]:
    if available_count <= 0:
        raise ValueError("available_count must be positive.")

    cases: list[dict[str, Any]] = []
    seen_counts: set[int] = set()
    for spec in chunk_count_specs:
        chunk_count = available_count if spec == "all" else int(spec)
        if chunk_count > available_count:
            continue
        if chunk_count in seen_counts:
            continue
        seen_counts.add(chunk_count)
        cases.append(
            {
                "requested_chunk_count": spec,
                "chunk_count": chunk_count,
                "available_chunk_count": available_count,
            }
        )

    if not cases:
        cases.append(
            {
                "requested_chunk_count": "all",
                "chunk_count": available_count,
                "available_chunk_count": available_count,
            }
        )
    return cases


def slice_example_contexts(example: InputExample, chunk_count: int) -> InputExample:
    return InputExample(
        question=example.question,
        answers=list(example.answers),
        ctxs=list(example.ctxs[:chunk_count]),
        example_id=example.example_id,
        metadata={
            **dict(example.metadata),
            "sweep_original_num_ctxs": len(example.ctxs),
            "sweep_chunk_count": chunk_count,
        },
    )


def doc_token_count(tokenized_example: Any) -> int:
    return sum(len(doc_ids) for doc_ids in tokenized_example.doc_chunk_ids)


def token_count_record(tokenized_example: Any) -> dict[str, int]:
    doc_count = doc_token_count(tokenized_example)
    query_count = len(tokenized_example.q_ids)
    return {
        "doc_token_count": doc_count,
        "query_token_count": query_count,
        "total_prefill_token_count": doc_count + query_count,
    }


def chunking_config_record(
    chunking_mode: str,
    *,
    chunk_size_tokens: int | None = None,
    chunk_overlap_tokens: int | None = None,
    max_chunks: int | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "chunking_mode": validate_chunking_mode(chunking_mode),
    }
    if record["chunking_mode"] == CHUNKING_MODE_TOKEN:
        record.update(
            {
                "chunk_size_tokens": chunk_size_tokens,
                "chunk_overlap_tokens": chunk_overlap_tokens,
                "max_chunks": max_chunks,
            }
        )
    return record
