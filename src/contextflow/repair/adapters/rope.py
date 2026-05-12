from __future__ import annotations

from typing import Any


def rotate_half(tensor: Any) -> Any:
    import torch

    first_half, second_half = tensor.chunk(2, dim=-1)
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(tensor: Any, cos: Any, sin: Any) -> Any:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (tensor * cos) + (rotate_half(tensor) * sin)


def undo_rope(tensor: Any, cos: Any, sin: Any) -> Any:
    cos = cos[:, None, :, :]
    sin = sin[:, None, :, :]
    return (tensor * cos) - (rotate_half(tensor) * sin)


def select_rotary_positions(embedding: Any, positions: list[int], selected_count: int) -> Any:
    import torch

    if embedding.ndim == 4 and int(embedding.shape[1]) == 1:
        return select_rotary_positions(
            embedding[:, 0, :, :],
            positions=positions,
            selected_count=selected_count,
        )

    if embedding.ndim == 2:
        if int(embedding.shape[0]) == selected_count:
            return embedding.unsqueeze(0)
        index_tensor = torch.tensor(positions, dtype=torch.long, device=embedding.device)
        return embedding.index_select(dim=0, index=index_tensor).unsqueeze(0)

    if embedding.ndim == 3:
        if int(embedding.shape[1]) == selected_count:
            return embedding
        index_tensor = torch.tensor(positions, dtype=torch.long, device=embedding.device)
        return embedding.index_select(dim=1, index=index_tensor)

    raise ValueError(f"Unsupported rotary embedding shape: {embedding.shape}.")


def correct_rope_key_positions(
    key: Any,
    local_cos: Any,
    local_sin: Any,
    absolute_cos: Any,
    absolute_sin: Any,
) -> Any:
    """Convert a RoPE-rotated key from local positions to absolute positions."""

    unrotated_key = undo_rope(key, local_cos, local_sin)
    return apply_rope(unrotated_key, absolute_cos, absolute_sin)
