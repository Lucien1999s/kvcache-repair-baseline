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

Prepare the configured full split by omitting `--limit`:

```bash
python scripts/prepare_dataset.py --config configs/datasets/musique.yaml --overwrite
python scripts/prepare_dataset.py --config configs/datasets/2wiki.yaml --overwrite
```

The repo dataset key `2wiki` refers to 2WikiMultiHopQA / 2WikiMQA-style
multi-hop QA data.

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
  --output-jsonl results/chunk_sweep_musique_mistral_limit1.jsonl \
  --device-map auto \
  --torch-dtype auto
```

Both runners default to `--repair-planner online_gradual_hkvd`. Use
`--repair-planner oracle_hkvd` only for diagnostic checks that intentionally use
full-reference KV.
