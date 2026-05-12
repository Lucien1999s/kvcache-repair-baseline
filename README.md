# ContextFlow

Research prototype for RAG KV cache reuse, repair, and memory workflows.

Current main comparison path:

1. Data pipeline: `JSON -> InputExample -> PromptExample -> TokenizedExample`
2. Token-aligned full recompute greedy: `concat(doc_chunk_ids) + q_ids -> greedy decode`
3. Naive KV reuse greedy: `doc_chunk_ids -> precomputed doc KV -> concat KV -> feed q_ids -> greedy decode`
4. CacheBlend-style repair: compare reused doc KV against full recompute doc KV, select HKVD tokens, run adapter-specific partial repair, then decode from repaired KV

The current baseline substrate does not include retrieval, reranking, chunking,
vLLM hooks, or scheduling. Dataset inputs are expected to provide passages.

## Experiments

Run from the repo root after installing the package in editable mode.

```bash
python3 experiments/smoke_data_eval.py
python3 experiments/smoke_cacheblend_methods.py --model sshleifer/tiny-gpt2 --model-family gpt2
```

- `experiments/smoke_data_eval.py`: no-model smoke for dataset parsing,
  CacheBlend QA prompt policy, prediction parsing, QA metrics, and profiling helpers.
- `experiments/smoke_cacheblend_methods.py`: tiny-model smoke that runs the
  three comparison methods on the same inline QA example: full recompute,
  naive KV reuse, and CacheBlend-style repair.
- `experiments/cacheblend_dataset_runner.py`: dataset-level runner for local
  MuSiQue / 2Wiki JSONL files. It writes per-example JSONL and prints aggregate
  EM/F1 plus CacheBlend normalized F1.
