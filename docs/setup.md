# Setup

## Dataset Setup

Full datasets are intentionally not committed to GitHub. Keep raw and processed
dataset files under `data/raw/` or `data/processed/`; those paths are ignored by
Git. The committed `configs/datasets/*.yaml` files describe where data comes
from and where local JSONL outputs should be written.

Install optional dataset preparation dependencies when you need to download from
HuggingFace:

```bash
pip install -e ".[datasets]"
```

Prepare a small MuSiQue subset:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/musique.yaml \
  --limit 100 \
  --overwrite
```

Prepare a small 2WikiMultiHopQA / 2WikiMQA-style subset:

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

The MuSiQue config should point to a source with `question`, `answer`, and
`paragraphs` fields. Do not use CoRAG-style `query/context_doc_ids` exports for
this loader unless you also provide passage text, because `context_doc_ids`
alone cannot become `InputExample.ctxs`.

The script writes local JSONL only. It does not run retrieval, reranking,
chunking, tokenization, or model execution.

Validate local dataset parsing into `InputExample`:

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

The repo dataset key `2wiki` refers to 2WikiMultiHopQA / 2WikiMQA-style
multi-hop QA data.

## CacheBlend-Style Runs

The default repair planner is the measured online path:

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

For memory-constrained diagnostics and OOM boundaries:

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
`--repair-planner oracle_hkvd` only for diagnostic checks that intentionally
use full-reference KV.
