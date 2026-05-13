# CacheBlend-Style Baseline Notes

This repo implements a CacheBlend-style HF/PyTorch reference baseline for
controlled research comparisons. It intentionally does not claim to reproduce
the official CacheBlend vLLM serving system.

## Implemented Comparison Methods

- `full_recompute`: token-aligned full prefill over `concat(doc_chunk_ids) + q_ids`,
  followed by greedy decode.
- `naive_reuse`: precompute each document chunk KV independently, apply RoPE
  position correction for RoPE model families, concatenate document KV, then
  decode the query from the reused cache.
- `cacheblend_repair`: repair reused document KV before query decode.

## Repair Planner Modes

`online_gradual_hkvd` is the measured CacheBlend-style baseline path.

- Does not compute full-reference document KV.
- Starts layer 0 from all document tokens.
- Computes fresh selected-token KV and compares it with reused KV.
- Gradually filters the active selected-token set layer by layer.
- Runs adapter-specific partial repair and decodes from the repaired KV.

`oracle_hkvd` is diagnostic mode.

- Computes full-reference document KV.
- Selects HKVD tokens from reused-vs-full KV deviation.
- Runs the same partial repair execution from the resulting plan.
- Should be used for correctness checks and upper-bound selection studies, not
  as the measured resource baseline.

## Terminology

- **CacheBlend-style HF/PyTorch reference baseline**: this repo's controlled
  implementation of the core CacheBlend-inspired KV reuse/repair workflow.
- **Measured repair execution**: online gradual HKVD repair and decode without
  full-reference KV.
- **Oracle-HKVD repair plan**: a diagnostic plan generated with full-reference
  KV; useful for inspection but not a deployable planner.
- **Diagnostic mode**: any run that uses `oracle_hkvd` or full-reference repair
  diagnostics.

## Scope Boundaries

The current baseline does not include retrieval, reranking, chunking, vLLM
integration, request scheduling, or CPU/GPU KV-store switching. Dataset inputs
are expected to already contain passages/chunks.
