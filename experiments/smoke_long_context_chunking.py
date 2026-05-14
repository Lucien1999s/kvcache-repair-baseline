from __future__ import annotations

import json
from typing import Any

from contextflow.chunking import (
    TokenChunkingConfig,
    chunk_token_ids,
    slice_tokenized_doc_chunks,
    token_chunk_tokenized_example,
    token_chunking_metadata,
)
from contextflow.data import (
    build_cacheblend_prompt,
    parse_qa_dataset_examples,
    tokenize_prompt_example,
)


LONG_CONTEXT_SAMPLE = {
    "id": "long-context-chunking-smoke-0",
    "question": "Who wrote Pride and Prejudice",
    "answers": ["Jane Austen"],
    "context": " ".join(
        [
            "Pride and Prejudice is a novel by Jane Austen.",
            "Elizabeth Bennet and Fitzwilliam Darcy are central characters.",
            "Jane Austen wrote several major English novels.",
            "This extra sentence makes the document long enough for token chunks.",
        ]
        * 4
    ),
    "title": "Pride and Prejudice",
}


class WhitespaceTokenizer:
    def encode(self, text: str) -> list[int]:
        # Include a synthetic leading token so tokenization exercises the existing
        # CacheBlend tokenizer.encode(text)[1:] convention.
        token_ids = [0]
        for token in text.split():
            token_ids.append(abs(hash(token)) % 10000 + 1)
        return token_ids


def main() -> None:
    example = parse_qa_dataset_examples("longbook_qa_en", [LONG_CONTEXT_SAMPLE])[0]
    prompt = build_cacheblend_prompt(
        example,
        prompt_policy="cacheblend_qa",
        dataset="longbook_qa_en",
    )
    tokenized = tokenize_prompt_example(prompt, WhitespaceTokenizer())
    source_doc_token_count = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)
    assert source_doc_token_count > 20

    chunks = chunk_token_ids(
        list(range(20)),
        chunk_size_tokens=6,
        chunk_overlap_tokens=2,
        max_chunks=3,
    )
    assert chunks == [
        [0, 1, 2, 3, 4, 5],
        [4, 5, 6, 7, 8, 9],
        [8, 9, 10, 11, 12, 13],
    ]

    config = TokenChunkingConfig(
        chunk_size_tokens=12,
        chunk_overlap_tokens=3,
        max_chunks=4,
    )
    token_chunked = token_chunk_tokenized_example(
        tokenized,
        chunk_size_tokens=config.chunk_size_tokens,
        chunk_overlap_tokens=config.chunk_overlap_tokens,
        max_chunks=config.max_chunks,
    )
    assert 1 <= len(token_chunked.doc_chunk_ids) <= 4
    assert all(len(chunk) <= 12 for chunk in token_chunked.doc_chunk_ids)
    assert token_chunked.q_ids == tokenized.q_ids

    chunk_count = min(2, len(token_chunked.doc_chunk_ids))
    sliced = slice_tokenized_doc_chunks(
        token_chunked,
        chunk_count=chunk_count,
    )
    token_counts = {
        "doc_token_count": sum(len(doc_ids) for doc_ids in sliced.doc_chunk_ids),
        "query_token_count": len(sliced.q_ids),
    }
    token_counts["total_prefill_token_count"] = (
        token_counts["doc_token_count"] + token_counts["query_token_count"]
    )
    metadata = token_chunking_metadata(
        config,
        source_doc_token_count=source_doc_token_count,
        produced_chunk_count=len(token_chunked.doc_chunk_ids),
    )

    output: dict[str, Any] = {
        "example_id": example.example_id,
        "source_doc_token_count": source_doc_token_count,
        "produced_chunk_count": len(token_chunked.doc_chunk_ids),
        "sliced_chunk_count": chunk_count,
        "token_counts": token_counts,
        "chunking_metadata": metadata,
    }
    print("smoke_long_context_chunking passed")
    print(json.dumps(output, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
