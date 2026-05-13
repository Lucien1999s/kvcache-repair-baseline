# Baseline Notes

This document defines the baseline methods currently implemented in ContextFlow.
These baselines are used for controlled comparison before adding FusionRAG or
ContextFlow-specific scheduling and KV-store policies.

## Scope

The current baseline stack runs in HuggingFace/PyTorch. It intentionally does
not claim to reproduce the official CacheBlend vLLM serving system. In
particular, it does not include vLLM block management, request scheduling,
production batching, CPU/GPU KV-store switching, or compute/load overlap.

The implemented baseline is best described as:

```text
CacheBlend-style HF/PyTorch reference baseline
```

## Shared Input Layout

All comparison methods operate on the same logical token layout:

```text
doc_chunk_ids[0] + ... + doc_chunk_ids[n] + q_ids
```

Dataset loaders produce `InputExample`; prompt formatting and tokenization
produce `TokenizedExample`; methods consume the same tokenized example.

## Methods

### Full Recompute

`full_recompute` concatenates all document chunks and the query, then runs
standard greedy generation over the full prefill:

```text
concat(doc_chunk_ids) + q_ids -> greedy decode
```

This is the quality upper-bound method in the current comparison setup.

### Naive KV Reuse

`naive_reuse` precomputes each document chunk KV independently, assembles the
document KV cache layer-wise, and decodes the query from the assembled cache:

```text
doc_chunk_ids -> per-chunk KV -> assembled document KV -> q_ids decode
```

For RoPE model families such as Mistral and Qwen2, reusable chunk keys are
corrected from chunk-local positions into full-context absolute positions before
assembly.

### CacheBlend-Style Repair

`cacheblend_repair` starts from the same reused document KV as `naive_reuse`.
It then recomputes selected document-token hidden states layer by layer, patches
selected K/V positions into the reused cache, and decodes the query from the
repaired document KV.

Supported repair adapters currently include:

- GPT-2-like models
- Mistral-like models
- Qwen2/Qwen2.5-like models

## Repair Planner Modes

### `online_gradual_hkvd`

This is the default measured baseline.

- Does not compute full-reference document KV.
- Starts from a large layer-0 active set.
- Computes fresh selected-token KV and compares it with reused KV.
- Gradually filters selected tokens layer by layer.
- Runs adapter-specific partial repair and decodes from repaired KV.

This is the mode to use for resource measurements.

### `oracle_hkvd`

This is diagnostic mode.

- Computes full-reference document KV.
- Selects HKVD tokens from reused-vs-full KV deviation.
- Runs the same partial repair execution from the resulting plan.
- Supports correctness checks and upper-bound token-selection diagnostics.

Do not use `oracle_hkvd` for measured resource claims.

## QA Protocol

For MuSiQue and 2Wiki-style QA, the benchmark runners default to the
CacheBlend-aligned QA prompt policy:

```text
--prompt-policy cacheblend_qa
--prediction-parser cacheblend_qa
```

The parser follows the minimal CacheBlend-style rule: strip leading newlines,
take the first generated line, and normalize leading Yes/No answers.

## Reporting

When reporting results, distinguish:

- Full Recompute: full-context upper-bound method
- Naive KV Reuse: unrepaired reuse baseline
- CacheBlend-style Repair: online gradual HKVD repair baseline
- Oracle-HKVD: diagnostic mode only

Dataset-level runners report EM/F1, normalized F1, latency, phase latency, and
peak GPU memory. Chunk sweep runners additionally report token counts and OOM
boundaries.
