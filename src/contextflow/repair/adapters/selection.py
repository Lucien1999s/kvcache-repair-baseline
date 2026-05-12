from __future__ import annotations

from typing import Any


def validate_selected_indices_for_sequence(
    selected_indices: list[int],
    seq_len: int,
    name: str,
    require_sorted: bool = False,
) -> None:
    if not selected_indices:
        raise ValueError(f"{name} must be non-empty.")
    if require_sorted and selected_indices != sorted(selected_indices):
        raise ValueError(f"{name} must be sorted.")
    if len(set(selected_indices)) != len(selected_indices):
        raise ValueError(f"{name} must not contain duplicates.")
    invalid = [index for index in selected_indices if index < 0 or index >= seq_len]
    if invalid:
        raise ValueError(f"{name} contains invalid indices for seq_len={seq_len}: {invalid}.")


def normalize_selected_indices_by_layer(
    selected_indices: list[int],
    selected_indices_by_layer: dict[int, list[int]] | None,
    start_layer_index: int,
    end_layer_index: int,
    full_seq_len: int,
) -> dict[int, list[int]]:
    """Resolve fixed or per-layer selected token sets for gradual repair.

    When selected_indices_by_layer is provided, every layer's set must be a subset
    of the currently propagated hidden-state set. This matches CacheBlend-style
    gradual filtering, where the active repair set shrinks across layers.
    """

    validate_selected_indices_for_sequence(
        selected_indices,
        seq_len=full_seq_len,
        name="selected_indices",
        require_sorted=False,
    )

    if selected_indices_by_layer is None:
        return {
            layer_index: list(selected_indices)
            for layer_index in range(start_layer_index, end_layer_index)
        }

    layer_indices: dict[int, list[int]] = {}
    current_indices = list(selected_indices)
    for layer_index in range(start_layer_index, end_layer_index):
        if layer_index not in selected_indices_by_layer:
            raise ValueError(f"Missing selected_indices_by_layer entry for layer {layer_index}.")
        selected_for_layer = list(selected_indices_by_layer[layer_index])
        validate_selected_indices_for_sequence(
            selected_for_layer,
            seq_len=full_seq_len,
            name=f"selected_indices_by_layer[{layer_index}]",
            require_sorted=True,
        )
        missing = [index for index in selected_for_layer if index not in current_indices]
        if missing:
            raise ValueError(
                f"selected_indices_by_layer[{layer_index}] must be a subset of the "
                f"previous active selected set; missing indices: {missing}."
            )
        layer_indices[layer_index] = selected_for_layer
        current_indices = selected_for_layer
    return layer_indices


def select_hidden_states_for_indices(
    hidden_states: Any,
    current_indices: list[int],
    target_indices: list[int],
) -> Any:
    """Select target absolute token positions from current selected hidden states."""

    import torch

    if target_indices == current_indices:
        return hidden_states
    if int(hidden_states.shape[1]) != len(current_indices):
        raise ValueError(
            "hidden_states selected dimension must match len(current_indices); "
            f"got {hidden_states.shape[1]} and {len(current_indices)}."
        )

    index_to_position = {
        token_index: selected_position
        for selected_position, token_index in enumerate(current_indices)
    }
    local_positions = []
    for token_index in target_indices:
        if token_index not in index_to_position:
            raise ValueError(
                f"target index {token_index} is not present in current selected indices."
            )
        local_positions.append(index_to_position[token_index])

    index_tensor = torch.tensor(
        local_positions,
        dtype=torch.long,
        device=hidden_states.device,
    )
    return hidden_states.index_select(dim=1, index=index_tensor)
