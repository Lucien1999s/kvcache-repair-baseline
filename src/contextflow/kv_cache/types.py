from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class ChunkKV:
    """HF/PyTorch past_key_values extracted for one reusable document chunk."""

    chunk_id: str
    input_ids: list[int]
    past_key_values: Any
    num_tokens: int
    num_layers: int
    device: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
