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
python experiments/00_smoke_test/dataset_loaders.py \
  --dataset musique \
  --input data/raw/musique/validation.jsonl \
  --limit 3

python experiments/00_smoke_test/dataset_loaders.py \
  --dataset 2wiki \
  --input data/raw/2wiki/validation.jsonl \
  --limit 3
```

The repo dataset key `2wiki` refers to 2WikiMultiHopQA / 2WikiMQA-style
multi-hop QA data.
