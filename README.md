# ContextFlow

Research prototype for RAG KV cache reuse, repair, and memory workflows.

This repository implements a **CacheBlend-style HF/PyTorch reference
baseline** for controlled research comparisons. It is not an official
CacheBlend/vLLM serving-system reproduction.

Current main comparison path:

1. Data pipeline: `JSON -> InputExample -> PromptExample -> TokenizedExample`
2. Token-aligned full recompute greedy: `concat(doc_chunk_ids) + q_ids -> greedy decode`
3. Naive KV reuse greedy: `doc_chunk_ids -> precomputed doc KV -> concat KV -> feed q_ids -> greedy decode`
4. CacheBlend-style repair:
   - measured baseline: online gradual HKVD selection and adapter-specific
     partial repair, without full-reference KV during measured execution
   - diagnostic mode: oracle HKVD selection from reused-vs-full doc KV, used
     only to inspect repair correctness and upper-bound selection behavior

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
- `experiments/cacheblend_chunk_sweep_runner.py`: diagnostic runner that sweeps
  context chunk counts, records token counts, latency, peak GPU memory, and OOM
  boundaries for full recompute, naive KV reuse, and CacheBlend-style repair.

The repair runner option `--repair-planner online_gradual_hkvd` is the default
measured baseline. Use `--repair-planner oracle_hkvd` only for diagnostics that
explicitly need full-reference KV.
