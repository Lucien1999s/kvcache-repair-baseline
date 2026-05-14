from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Context:
    """One raw context passage from a CacheBlend-style QA example."""

    title: str
    text: str

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Context":
        return cls(
            title=str(data.get("title", "")),
            text=str(data.get("text", "")),
        )


@dataclass(slots=True)
class InputExample:
    """Raw QA example before prompt formatting.

    This mirrors the CacheBlend-style input shape:
    question, answers, and ctxs, where each ctx has title and text.
    """

    question: str
    answers: list[str]
    ctxs: list[Context]
    example_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "InputExample":
        raw_example_id = data.get("id")
        if raw_example_id is None:
            raw_example_id = data.get("example_id")

        return cls(
            question=str(data["question"]),
            answers=[str(answer) for answer in data.get("answers", [])],
            ctxs=[Context.from_dict(ctx) for ctx in data.get("ctxs", [])],
            example_id=str(raw_example_id) if raw_example_id is not None else None,
            metadata={
                key: value
                for key, value in data.items()
                if key not in {"question", "answers", "ctxs", "id", "example_id"}
            },
        )


@dataclass(slots=True)
class PromptExample:
    """Prompt-formatted example before tokenization."""

    doc_prompts: list[str]
    q_prompt: str
    answers: list[str]
    example_id: str | None = None


@dataclass(slots=True)
class TokenizedExample:
    """Tokenized form consumed by KV reuse and repair workflows."""

    doc_chunk_ids: list[list[int]]
    q_ids: list[int]
    answers: list[str]
    example_id: str | None = None
