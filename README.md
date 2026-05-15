# ContextFlow

ContextFlow is a research artifact for studying retrieval-augmented generation
under KV-cache reuse, repair, and memory constraints. The current repository
contains a CacheBlend-style HF/PyTorch reference baseline, dataset preparation
utilities for MuSiQue and 2Wiki-style QA data, QA evaluation metrics, and
diagnostic runners for latency, peak GPU memory, and chunk-count/OOM sweeps.
It also includes a FusionRAG-style HF/PyTorch baseline built on the same data,
method, evaluation, and profiling substrate.

This is not an official CacheBlend/vLLM serving-system reproduction. The
baseline reproduces the core KV reuse and selective repair workflow in a
controlled HF/PyTorch environment so CacheBlend-style, FusionRAG-style, and
future ContextFlow methods can be compared on the same data, model, prompt, and
metric pipeline.

## Repository Layout

```text
src/contextflow/
  data/          Input schemas, dataset loaders, prompt formatting, tokenization
  chunking/      Token-based long-document chunking utilities
  methods/       Full recompute, naive reuse, CacheBlend-style, FusionRAG-style
  repair/        HKVD/QGS selectors and model-family repair adapters
  kv_cache/      KV precompute, enriched precompute, assembly, RoPE correction
  evaluation/    QA answer parsing and EM/F1 metrics
  profiling/     Timing and peak-memory helpers
  benchmarks/    Shared benchmark execution, records, and sweep utilities

experiments/     Thin CLI entrypoints for smoke checks and baseline runs
configs/         Dataset preparation configs
scripts/         Dataset preparation utilities
docs/            Setup, baseline notes, and ContextFlow design notes
```

## Quick Start

Install the package in editable mode from the repository root:

```bash
pip install -e .
```

Run the lightweight no-model smoke check:

```bash
python experiments/smoke_data_eval.py
```

Run the baseline method smoke check with a local or HuggingFace model:

```bash
python experiments/smoke_cacheblend_methods.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral \
  --device-map auto \
  --torch-dtype auto
```

Run the FusionRAG-style method smoke check:

```bash
python experiments/smoke_fusionrag_method.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-family qwen2 \
  --device-map auto \
  --torch-dtype auto
```

Prepare a small MuSiQue subset:

```bash
pip install -e ".[datasets]"
python scripts/prepare_dataset.py \
  --config configs/datasets/musique.yaml \
  --limit 100 \
  --overwrite
```

Prepare a LongBook-QA-English subset for token-chunked memory diagnostics:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/longbook_qa_en.yaml \
  --limit 10 \
  --overwrite
```

Run a dataset-level baseline comparison:

```bash
python experiments/cacheblend_dataset_runner.py \
  --dataset musique \
  --input data/raw/musique/validation.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral \
  --limit 2 \
  --max-new-tokens 16 \
  --methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair \
  --fusionrag-neighbor-top-n 1 \
  --fusionrag-recompute-ratio 0.15 \
  --output-jsonl results/cacheblend_musique_mistral_limit2.jsonl \
  --device-map auto \
  --torch-dtype auto
```

Run a chunk-count diagnostic sweep:

```bash
python experiments/cacheblend_chunk_sweep_runner.py \
  --dataset musique \
  --input data/raw/musique/validation.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral \
  --limit 1 \
  --chunk-counts 1,2,4,8,16,all \
  --max-new-tokens 16 \
  --methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair \
  --fusionrag-neighbor-top-n 1 \
  --fusionrag-recompute-ratio 0.15 \
  --output-jsonl results/chunk_sweep_musique_mistral_limit1.jsonl \
  --device-map auto \
  --torch-dtype auto
```

Run a token-chunked long-context sweep:

```bash
python experiments/cacheblend_chunk_sweep_runner.py \
  --dataset longbook_qa_en \
  --input data/raw/longbook_qa_en/test.jsonl \
  --model Qwen/Qwen2.5-7B-Instruct \
  --model-family qwen2 \
  --limit 1 \
  --chunking token \
  --chunk-size-tokens 1024 \
  --chunk-overlap-tokens 0 \
  --max-chunks 128 \
  --max-total-tokens 131072 \
  --chunk-counts 8,16,32,64,128,all \
  --methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair \
  --output-jsonl results/longbook_qwen2_7b_token_sweep.jsonl \
  --device-map auto \
  --torch-dtype auto
```

Chunk sweeps record `baseline_gpu_memory_mb`, `peak_gpu_memory_mb`, and
`peak_gpu_memory_delta_mb`; memory-constrained plots should generally use the
delta field so model weights do not dominate the KV/cache signal. Use
`--max-total-tokens` to skip over-budget cases explicitly rather than silently
truncating prompts. FusionRAG-style records additionally expose synchronized
sub-phase latency and memory delta for enriched precompute, query selection,
repair, and decode. Add `--enable-micro-profiling` only when diagnosing repair
internals; it appends `micro_phase_metrics` and `repair_micro_summary` without
changing the default JSONL schema.

## Documentation

- [Setup](docs/setup.md): environment, dataset preparation, and smoke checks.
- [Baseline Notes](docs/baseline_notes.md): Full Recompute, Naive KV Reuse,
  CacheBlend-style repair, and FusionRAG-style repair baseline definitions.
- [ContextFlow Notes](docs/contextflow_notes.md): notes for upcoming
  memory-constrained ContextFlow development.

## Scope

The current codebase does not include retrieval/reranking pipelines, vLLM hooks,
request scheduling, or CPU/GPU KV-store switching. Dataset inputs may provide
passages/chunks directly, or long-context rows can be token-chunked by the
diagnostic runner. KV-store/offload components are reserved for the next
ContextFlow development phases.
