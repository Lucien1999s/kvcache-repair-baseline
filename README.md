# ContextFlow

Research prototype for RAG KV cache reuse, repair, and memory workflows.

Current main comparison path:

1. Data pipeline: `JSON -> InputExample -> PromptExample -> TokenizedExample`
2. Token-aligned full recompute greedy: `concat(doc_chunk_ids) + q_ids -> greedy decode`
3. Naive KV reuse greedy: `doc_chunk_ids -> precomputed doc KV -> concat KV -> feed q_ids -> greedy decode`
4. CacheBlend-style HKVD diagnostic: compare naive reused doc KV against full recompute doc KV and select high-deviation tokens
5. GPT2 gradual HKVD patch feasibility: patch selected doc KV positions without running repaired generation

The current baseline substrate does not yet include retrieval, chunking, repaired generation, runtime partial-prefill hidden-state propagation, vLLM hooks, or scheduling.

## Smoke Tests

Run from the repo root after installing the package in editable mode.

```bash
python3 experiments/00_smoke_test/data_pipeline.py --limit 1
python3 experiments/00_smoke_test/kv_precompute.py --model sshleifer/tiny-gpt2 --limit 1
python3 experiments/00_smoke_test/token_aligned_full_recompute.py --model sshleifer/tiny-gpt2 --limit 1 --run-generation
python3 experiments/00_smoke_test/naive_kv_reuse.py --model sshleifer/tiny-gpt2 --limit 1
python3 experiments/00_smoke_test/cacheblend_hkvd_selector.py --model sshleifer/tiny-gpt2 --limit 1
python3 experiments/00_smoke_test/gpt2_gradual_hkvd_patch.py --model sshleifer/tiny-gpt2 --limit 1 --initial-top-k 10 --top-k 5
```

- `data_pipeline.py`: checks CacheBlend-style JSON loading, prompt formatting, and tokenization.
- `kv_precompute.py`: checks `doc_chunk_ids -> ChunkKV` extraction with HF/PyTorch `use_cache=True`.
- `token_aligned_full_recompute.py`: checks `concat(doc_chunk_ids) + q_ids` layout and optionally runs token-aligned full recompute greedy generation.
- `naive_kv_reuse.py`: checks direct doc KV reuse with concatenated chunk KVs and cached greedy decoding.
- `cacheblend_hkvd_selector.py`: checks CacheBlend-style HKVD token selection by comparing naive reused doc KV against full recompute doc KV.
- `gpt2_gradual_hkvd_patch.py`: checks GPT2-only CacheBlend-style gradual HKVD filtering where selected token sets shrink layer by layer.
