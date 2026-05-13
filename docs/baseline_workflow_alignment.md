# Baseline Workflow Alignment

The repo keeps all comparison methods on the same logical input layout:

```text
doc_chunk_ids[0] + ... + doc_chunk_ids[n] + q_ids
```

`full_recompute` uses that full token sequence directly.

`naive_reuse` precomputes document chunk KV independently, assembles document
KV layer-wise, then decodes the query from the cached document KV.

`cacheblend_repair` starts from the same reused document KV as `naive_reuse`,
repairs selected document-token KV positions, then decodes the query from the
repaired document KV.

## CacheBlend Repair Modes

`online_gradual_hkvd` is the default measured repair mode. It does not run full
recompute reference KV for token selection.

`oracle_hkvd` is diagnostic mode. It compares reused document KV with
full-reference document KV to select HKVD tokens. Use it for correctness checks,
not for measured resource claims.

## Reporting

When reporting results, distinguish:

- full recompute upper-bound method
- naive KV reuse lower-bound method
- CacheBlend-style online gradual HKVD repair method
- oracle-HKVD diagnostics

Avoid presenting this repo as an official CacheBlend/vLLM serving
implementation.
