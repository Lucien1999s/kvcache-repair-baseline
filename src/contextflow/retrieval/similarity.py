from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass


ChunkSimilarityFn = Callable[[list[int], list[int]], float]


@dataclass(slots=True)
class ChunkNeighborPlan:
    """Example-local neighbor plan for FusionRAG-style SGP preprocessing."""

    target_chunk_index: int
    neighbor_indices: list[int]
    neighbor_scores: list[float]


def token_id_cosine_similarity(left: list[int], right: list[int]) -> float:
    """Deterministic lexical cosine similarity over token-id counts.

    This is the dependency-free default. The API accepts a pluggable scorer so a
    dense embedding model can replace this later without changing KV precompute.
    """

    if not left or not right:
        return 0.0

    left_counts = Counter(left)
    right_counts = Counter(right)
    shared_token_ids = left_counts.keys() & right_counts.keys()
    dot = sum(left_counts[token_id] * right_counts[token_id] for token_id in shared_token_ids)
    if dot == 0:
        return 0.0

    left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
    right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return float(dot / (left_norm * right_norm))


def validate_doc_chunk_ids(doc_chunk_ids: list[list[int]]) -> None:
    if not doc_chunk_ids:
        raise ValueError("doc_chunk_ids must contain at least one chunk.")
    for chunk_index, chunk_ids in enumerate(doc_chunk_ids):
        if not chunk_ids:
            raise ValueError(f"doc_chunk_ids[{chunk_index}] must be non-empty.")


def build_chunk_neighbor_plan(
    doc_chunk_ids: list[list[int]],
    neighbor_top_n: int,
    similarity_fn: ChunkSimilarityFn | None = None,
) -> list[ChunkNeighborPlan]:
    """Build deterministic top-n neighbor plans for every document chunk.

    Neighbors are selected within the current example only. Ties are broken by
    chunk index to keep smoke tests and benchmark records reproducible.
    """

    validate_doc_chunk_ids(doc_chunk_ids)
    if neighbor_top_n < 0:
        raise ValueError("neighbor_top_n must be non-negative.")

    scorer = similarity_fn or token_id_cosine_similarity
    plans: list[ChunkNeighborPlan] = []
    for target_index, target_ids in enumerate(doc_chunk_ids):
        scored_neighbors = []
        for candidate_index, candidate_ids in enumerate(doc_chunk_ids):
            if candidate_index == target_index:
                continue
            score = float(scorer(target_ids, candidate_ids))
            scored_neighbors.append((candidate_index, score))

        scored_neighbors.sort(key=lambda item: (-item[1], item[0]))
        selected_neighbors = scored_neighbors[:neighbor_top_n]
        plans.append(
            ChunkNeighborPlan(
                target_chunk_index=target_index,
                neighbor_indices=[index for index, _ in selected_neighbors],
                neighbor_scores=[score for _, score in selected_neighbors],
            )
        )

    return plans
