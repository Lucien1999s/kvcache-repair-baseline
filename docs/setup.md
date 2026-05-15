# Setup

This document describes environment setup, dataset preparation, and smoke checks
for the ContextFlow research artifact.

## Environment

Use Python 3.10 or newer:

```bash
pip install -e .
```

Install dataset preparation dependencies only when downloading or converting
datasets:

```bash
pip install -e ".[datasets]"
```

Large model runs should be executed in a GPU environment with compatible
`torch`, `transformers`, and optional `accelerate` support for `device_map=auto`.

## Dataset Preparation

Full datasets are intentionally not committed to GitHub. Keep raw and processed
dataset files under `data/raw/` or `data/processed/`; those paths are ignored by
Git. Dataset configs live under `configs/datasets/`.

Prepare a MuSiQue subset:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/musique.yaml \
  --limit 100 \
  --overwrite
```

Prepare a 2WikiMultiHopQA / 2WikiMQA-style subset:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/2wiki.yaml \
  --limit 100 \
  --overwrite
```

Prepare a LongBook-QA-English / InfiniteBench subset for token-chunked memory
diagnostics:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/longbook_qa_en.yaml \
  --limit 10 \
  --overwrite
```

Prepare the configured full split by omitting `--limit`:

```bash
python scripts/prepare_dataset.py --config configs/datasets/musique.yaml --overwrite
python scripts/prepare_dataset.py --config configs/datasets/2wiki.yaml --overwrite
python scripts/prepare_dataset.py --config configs/datasets/longbook_qa_en.yaml --overwrite
```

The repo dataset key `2wiki` refers to 2WikiMultiHopQA / 2WikiMQA-style
multi-hop QA data.

The repo dataset key `longbook_qa_en` refers to LongBook-QA-English /
InfiniteBench `longbook_qa_eng`, and also supports compatible local long-context
QA JSON/JSONL rows. Expected fields are flexible, but a row should provide a
question-like field (`question`, `query`, or `input`), an answer field
(`answers` or `answer`), and a long context field (`context`, `document`,
`article`, `book`, or `text`).
The LongBook-QA-English config uses HuggingFace streaming so small subsets can
be prepared without first materializing the full Arrow dataset.

The preparation script writes local JSONL only. It does not run retrieval,
reranking, chunking, tokenization, or model execution.

## Smoke Checks

Run the no-model smoke check:

```bash
python experiments/smoke_data_eval.py
```

Optionally validate local dataset parsing:

```bash
python experiments/smoke_data_eval.py \
  --dataset musique \
  --input data/raw/musique/validation.jsonl \
  --limit 3

python experiments/smoke_data_eval.py \
  --dataset 2wiki \
  --input data/raw/2wiki/validation.jsonl \
  --limit 3
```

Run the CacheBlend-style method smoke check:

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

Run oracle-HKVD diagnostic mode when checking full-reference repair diagnostics:

```bash
python experiments/smoke_cacheblend_methods.py \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral \
  --repair-planner oracle_hkvd \
  --device-map auto \
  --torch-dtype auto
```

## Baseline Runs

Dataset-level comparison:

```bash
python experiments/cacheblend_dataset_runner.py \
  --dataset musique \
  --input data/raw/musique/validation.jsonl \
  --model mistralai/Mistral-7B-Instruct-v0.3 \
  --model-family mistral \
  --limit 10 \
  --max-new-tokens 16 \
  --methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair \
  --fusionrag-neighbor-top-n 1 \
  --fusionrag-recompute-ratio 0.15 \
  --output-jsonl results/cacheblend_musique_mistral_limit10.jsonl \
  --device-map auto \
  --torch-dtype auto
```

Chunk-count / memory diagnostic sweep:

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

Token-chunked long-context sweep:

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
  --max-new-tokens 16 \
  --methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair \
  --fusionrag-neighbor-top-n 1 \
  --fusionrag-recompute-ratio 0.15 \
  --output-jsonl results/longbook_qwen2_7b_token_sweep.jsonl \
  --device-map auto \
  --torch-dtype auto
```

For memory diagnostic figures, prefer `peak_gpu_memory_delta_mb` over raw
`peak_gpu_memory_mb`; the raw peak includes model weights, while the delta is
measured relative to allocated GPU memory before each method run. If a case
exceeds `--max-total-tokens`, the runner records `status=skipped_too_long`
instead of silently truncating.

FusionRAG records expose sub-phase bottlenecks directly:

```text
fusionrag_enriched_precompute_latency_seconds
fusionrag_enriched_precompute_peak_gpu_memory_delta_mb
fusionrag_query_selection_latency_seconds
fusionrag_query_selection_peak_gpu_memory_delta_mb
fusionrag_repair_latency_seconds
fusionrag_repair_peak_gpu_memory_delta_mb
fusionrag_decode_latency_seconds
fusionrag_decode_peak_gpu_memory_delta_mb
```

On failures, `failed_phase` uses the same sub-phase names so OOM boundaries can
be attributed to enriched precompute, query selection, repair, or decode.

For repair-internal OOM diagnosis, add `--enable-micro-profiling` to the chunk
sweep runner. When enabled, CacheBlend-style and FusionRAG-style repair records
include `micro_phase_metrics` and `repair_micro_summary`, which identify the
largest memory micro step and the failing micro step when an exception occurs.

Both runners default to `--repair-planner online_gradual_hkvd`. Use
`--repair-planner oracle_hkvd` only for diagnostic checks that intentionally use
full-reference KV.

FusionRAG is not part of the default method list yet. Add
`fusionrag_repair` explicitly through `--methods` when comparing all four
baselines.
