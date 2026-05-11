from __future__ import annotations

import argparse
from typing import Any

import torch

from contextflow.data import (
    InputExample,
    assemble_token_aligned_full_prefill_input_ids,
    build_cacheblend_prompt,
    load_json_examples,
    parse_input_examples,
    tokenize_prompt_example,
)
from contextflow.kv_cache import assemble_chunk_kvs, precompute_doc_chunk_kvs
from contextflow.kv_cache.precompute import infer_model_input_device, normalize_past_key_values
from contextflow.repair.adapters.gpt2 import (
    patch_gpt2_layers_with_gradual_hkvd_tokens,
    validate_gpt2_like_model,
)
from contextflow.runtime import load_hf_causal_lm


INLINE_SAMPLE: list[dict[str, Any]] = [
    {
        "id": "sample-0",
        "question": "Who wrote the novel Pride and Prejudice",
        "answers": ["Jane Austen"],
        "ctxs": [
            {
                "title": "Pride and Prejudice",
                "text": "Pride and Prejudice is a novel by Jane Austen.",
            },
            {
                "title": "Jane Austen",
                "text": "Jane Austen was an English novelist known for six major novels.",
            },
        ],
    }
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test GPT2 gradual HKVD-selected multi-layer KV patching."
    )
    parser.add_argument("--input", default=None, help="Optional CacheBlend-style JSON list.")
    parser.add_argument(
        "--model",
        default="sshleifer/tiny-gpt2",
        help="HuggingFace GPT2-like causal LM name or local path.",
    )
    parser.add_argument("--limit", type=int, default=1, help="Number of examples to check.")
    parser.add_argument(
        "--initial-top-k",
        type=int,
        default=None,
        help="Initial layer HKVD candidate count. Defaults to --top-k when omitted.",
    )
    parser.add_argument(
        "--initial-ratio",
        type=float,
        default=None,
        help="Initial layer HKVD candidate ratio. Defaults to --ratio when omitted.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Follow-up layer HKVD count within previous layer candidates.",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=None,
        help="Follow-up layer HKVD ratio within previous layer candidates.",
    )
    return parser.parse_args()


def load_examples(input_path: str | None) -> list[InputExample]:
    if input_path is not None:
        return load_json_examples(input_path)
    return parse_input_examples(INLINE_SAMPLE)


def slice_past_key_values_prefix(past_key_values: Any, seq_len: int) -> tuple[tuple[Any, Any], ...]:
    sliced_layers = []
    for layer_index, layer_kv in enumerate(past_key_values):
        key, value = layer_kv[0], layer_kv[1]
        if int(key.shape[-2]) < seq_len or int(value.shape[-2]) < seq_len:
            raise ValueError(
                f"Layer {layer_index} KV seq_len is shorter than requested prefix length {seq_len}."
            )
        sliced_layers.append((key[..., :seq_len, :], value[..., :seq_len, :]))
    return tuple(sliced_layers)


def build_doc_hidden_states_by_layer(
    hidden_states: tuple[Any, ...],
    num_layers: int,
    doc_total_len: int,
) -> list[Any]:
    if len(hidden_states) < num_layers:
        raise ValueError(
            f"Expected at least {num_layers} hidden-state tensors, got {len(hidden_states)}."
        )
    return [hidden_states[layer_index][:, :doc_total_len, :] for layer_index in range(num_layers)]


def index_select_sequence(tensor: Any, indices: list[int]) -> Any:
    index_tensor = torch.tensor(indices, dtype=torch.long, device=tensor.device)
    return tensor.index_select(dim=tensor.ndim - 2, index=index_tensor)


def max_abs_diff(left: Any, right: Any) -> float:
    if left.numel() == 0:
        return 0.0
    return float((left - right).abs().max().detach().cpu())


def layer_diff_at_indices(layer_a: tuple[Any, Any], layer_b: tuple[Any, Any], indices: list[int]) -> float:
    key_a, value_a = layer_a[0], layer_a[1]
    key_b, value_b = layer_b[0], layer_b[1]
    key_diff = max_abs_diff(index_select_sequence(key_a, indices), index_select_sequence(key_b, indices))
    value_diff = max_abs_diff(
        index_select_sequence(value_a, indices),
        index_select_sequence(value_b, indices),
    )
    return max(key_diff, value_diff)


def validate_selection_schedule(
    selected_indices_by_layer: dict[int, list[int]],
    num_layers: int,
    doc_total_len: int,
) -> None:
    assert selected_indices_by_layer[0], "Layer 0 selected indices must be non-empty."
    for layer_index in range(num_layers):
        selected_indices = selected_indices_by_layer[layer_index]
        assert selected_indices == sorted(selected_indices), (
            f"Layer {layer_index} selected indices must be sorted."
        )
        assert all(0 <= token_index < doc_total_len for token_index in selected_indices), (
            f"Layer {layer_index} selected indices must be in doc prefix range."
        )
        if layer_index == 0:
            continue
        previous = set(selected_indices_by_layer[layer_index - 1])
        current = set(selected_indices)
        assert current.issubset(previous), (
            f"Layer {layer_index} selected indices must be a subset of previous layer."
        )
        assert len(selected_indices) <= len(selected_indices_by_layer[layer_index - 1]), (
            f"Layer {layer_index} selected token count must not increase."
        )


def main() -> None:
    args = parse_args()

    examples = load_examples(args.input)
    assert examples, "Expected at least one input example."
    if args.ratio is None and args.top_k <= 0:
        raise ValueError("top_k must be positive when ratio is not set.")

    bundle = load_hf_causal_lm(args.model)
    validate_gpt2_like_model(bundle.model)
    device = infer_model_input_device(bundle.model)

    for index, example in enumerate(examples[: args.limit]):
        prompt = build_cacheblend_prompt(example)
        tokenized = tokenize_prompt_example(prompt, bundle.tokenizer)
        full_input_ids = assemble_token_aligned_full_prefill_input_ids(tokenized)
        doc_total_len = sum(len(doc_ids) for doc_ids in tokenized.doc_chunk_ids)

        input_tensor = torch.tensor([full_input_ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_tensor)
        with torch.inference_mode():
            outputs = bundle.model(
                input_ids=input_tensor,
                attention_mask=attention_mask,
                output_hidden_states=True,
                use_cache=True,
            )
        if outputs.hidden_states is None:
            raise RuntimeError("Model did not return hidden_states.")
        if outputs.past_key_values is None:
            raise RuntimeError("Model did not return past_key_values.")

        full_past_key_values = normalize_past_key_values(outputs.past_key_values)
        full_doc_kv = slice_past_key_values_prefix(full_past_key_values, seq_len=doc_total_len)
        doc_hidden_states_by_layer = build_doc_hidden_states_by_layer(
            outputs.hidden_states,
            num_layers=len(full_doc_kv),
            doc_total_len=doc_total_len,
        )

        chunk_kvs = precompute_doc_chunk_kvs(bundle.model, tokenized)
        reuse_doc_kv = assemble_chunk_kvs(chunk_kvs)
        patch_result = patch_gpt2_layers_with_gradual_hkvd_tokens(
            model=bundle.model,
            hidden_states_by_layer=doc_hidden_states_by_layer,
            reuse_past_key_values=reuse_doc_kv,
            full_past_key_values=full_doc_kv,
            initial_top_k=args.initial_top_k,
            initial_ratio=args.initial_ratio,
            top_k=None if args.ratio is not None else args.top_k,
            ratio=args.ratio,
        )

        patched_doc_kv = patch_result.patched_past_key_values
        assert len(patched_doc_kv) == len(reuse_doc_kv), (
            "patched layer count must match reuse layer count."
        )
        validate_selection_schedule(
            patch_result.selected_indices_by_layer,
            num_layers=len(patched_doc_kv),
            doc_total_len=doc_total_len,
        )

        print(f"[example {index}]")
        print(f"example_id: {example.example_id}")
        print(f"doc_total_len: {doc_total_len}")
        print(f"num_layers: {len(patched_doc_kv)}")
        print(f"initial_top_k: {args.initial_top_k}")
        print(f"initial_ratio: {args.initial_ratio}")
        print(f"top_k: {None if args.ratio is not None else args.top_k}")
        print(f"ratio: {args.ratio}")

        for layer_index, (patched_layer_kv, reuse_layer_kv, full_layer_kv) in enumerate(
            zip(patched_doc_kv, reuse_doc_kv, full_doc_kv)
        ):
            patched_key, patched_value = patched_layer_kv
            reuse_key, reuse_value = reuse_layer_kv
            assert tuple(patched_key.shape) == tuple(reuse_key.shape), (
                f"Layer {layer_index} patched key shape must match reuse key shape."
            )
            assert tuple(patched_value.shape) == tuple(reuse_value.shape), (
                f"Layer {layer_index} patched value shape must match reuse value shape."
            )

            selected_indices = patch_result.selected_indices_by_layer[layer_index]
            unselected_indices = [
                token_index
                for token_index in range(doc_total_len)
                if token_index not in set(selected_indices)
            ]

            selected_diff_before = layer_diff_at_indices(
                reuse_layer_kv,
                full_layer_kv,
                selected_indices,
            )
            selected_diff_after = layer_diff_at_indices(
                patched_layer_kv,
                full_layer_kv,
                selected_indices,
            )
            unselected_diff_after_vs_reuse = layer_diff_at_indices(
                patched_layer_kv,
                reuse_layer_kv,
                unselected_indices,
            )

            tolerance = 1e-6
            if selected_diff_before > tolerance:
                assert selected_diff_after < selected_diff_before, (
                    f"Layer {layer_index} selected positions must be closer to full KV after patch."
                )
            else:
                assert selected_diff_after <= tolerance, (
                    f"Layer {layer_index} selected positions must remain close to full KV after patch."
                )
            assert unselected_diff_after_vs_reuse == 0.0, (
                f"Layer {layer_index} unselected positions must remain unchanged from reuse KV."
            )

            print(f"layer_index: {layer_index}")
            print(f"selected_hkvd_indices: {selected_indices}")
            print(f"selected_count: {len(selected_indices)}")
            print(f"selected_diff_before_patch_vs_full: {selected_diff_before}")
            print(f"selected_diff_after_patch_vs_full: {selected_diff_after}")
            print(f"unselected_diff_after_patch_vs_reuse: {unselected_diff_after_vs_reuse}")


if __name__ == "__main__":
    main()
