# Scripts

Scripts are operational entrypoints for preparing local artifacts. They should
not contain model methods, benchmark logic, or retrieval algorithms.

## `prepare_dataset.py`

Prepare local JSONL files from dataset configs:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/musique.yaml \
  --limit 100 \
  --overwrite
```

Long-context memory diagnostics use LongBook-QA-English / InfiniteBench:

```bash
python scripts/prepare_dataset.py \
  --config configs/datasets/longbook_qa_en.yaml \
  --limit 10 \
  --overwrite
```

Supported source types:

- `huggingface`: uses the `datasets` package to download a configured split.
  Set `source.streaming: true` for large or schema-irregular HuggingFace
  datasets where preparing a limited local JSONL subset should not materialize
  the full Arrow dataset first.
- `local`: reads an existing local JSON or JSONL file and rewrites a limited
  JSONL subset.

The script only prepares local data files. It does not perform retrieval,
reranking, chunking, tokenization, or model execution.
