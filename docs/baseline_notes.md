# Baseline Notes

This document defines the baseline methods currently implemented in ContextFlow.
These baselines are used for controlled comparison before adding
ContextFlow-specific scheduling and KV-store policies.

## Scope

The current baseline stack runs in HuggingFace/PyTorch. It intentionally does
not claim to reproduce the official CacheBlend vLLM serving system. In
particular, it does not include vLLM block management, request scheduling,
production batching, CPU/GPU KV-store switching, or compute/load overlap.

The implemented baselines are best described as:

```text
HF/PyTorch reference baselines for KV reuse and repair
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

### FusionRAG-Style Repair

`fusionrag_repair` is implemented as a method-level HF/PyTorch reference
baseline on top of the same tokenized QA inputs and partial-repair primitives.
It follows the core FusionRAG-style path currently needed for baseline
comparison:

```text
doc chunks -> neighbor-enriched KV precompute -> QGS token selection
           -> selected-token repair -> q_ids decode
```

The current implementation includes:

- example-local top-n neighbor planning for enriched chunk precompute
- neighbor-prefix target chunk KV precompute
- RoPE source-position correction for Mistral/Qwen2-style models
- query-guided token selection from final-layer query-key attention scores
- recompute-ratio controlled selected-token repair

It intentionally does not include FusionRAG serving-system components such as
Alternative Path, async KV-cache scheduling, Q-Sparse-Attn kernels, corpus-level
KV stores, batching, or CPU/GPU KV placement.

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
- FusionRAG-style Repair: neighbor-enriched KV plus QGS repair baseline
- Oracle-HKVD: diagnostic mode only

Dataset-level runners report EM/F1, normalized F1, latency, phase latency, raw
peak GPU memory, and peak GPU memory delta. Chunk sweep runners additionally
report token counts, chunking config, explicit too-long skips, and OOM
boundaries. For memory plots, prefer `peak_gpu_memory_delta_mb` over raw
`peak_gpu_memory_mb` because the raw peak includes model weights.

FusionRAG-style repair is profiled as four synchronized outer phases:

- `fusionrag_enriched_precompute`
- `fusionrag_query_selection`
- `fusionrag_repair`
- `fusionrag_decode`

Each FusionRAG method record includes flat latency and memory-delta fields such
as `fusionrag_enriched_precompute_latency_seconds` and
`fusionrag_enriched_precompute_peak_gpu_memory_delta_mb`. If an OOM or error
occurs, `failed_phase` points to the failing sub-phase. The method's internal
metadata latencies are retained for debugging, but diagnostic plots should use
the outer phase profiling fields.

When chunk sweeps are run with `--enable-micro-profiling`, repair records also
include `micro_phase_metrics` and `repair_micro_summary`. These optional fields
break CacheBlend-style and FusionRAG-style repair into selected-hidden-state
construction, attention-mask allocation, per-layer partial forwards, KV patch
updates, and finalization. They are intended only for bottleneck diagnosis and
are omitted by default to preserve the stable JSONL schema.

Use FusionRAG explicitly in runners:

```text
--methods full_recompute,naive_reuse,cacheblend_repair,fusionrag_repair
--fusionrag-neighbor-top-n 1
--fusionrag-recompute-ratio 0.15
```
