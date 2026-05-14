# ContextFlow Notes

The current repository has four HF/PyTorch reference baselines: Full Recompute,
Naive KV Reuse, CacheBlend-style Repair, and FusionRAG-style Repair. They share
the same data, prompt, tokenization, evaluation, profiling, and benchmark
pipeline.

The next ContextFlow phase is separate from those baselines. It should focus on
memory-constrained long-context diagnostics first, then add ContextFlow-specific
memory-system components.

## Next Diagnostic Work

Before adding a KV store or CPU/GPU offload system, the benchmark substrate
should support:

- token-based long-document chunking
- long-context QA data such as LongBookQA-style examples
- chunk sweeps over configurable `chunk_size_tokens`, `chunk_overlap_tokens`,
  and `max_chunks`
- peak GPU memory delta reporting in addition to absolute peak memory
- OOM-safe sweep records with method, chunk count, token count, and failed phase

This keeps the baseline diagnostic clean: it measures how the four reference
methods behave as retrieved/chunked context grows, without mixing in a new
ContextFlow memory system.

## Later ContextFlow System Work

After the long-context diagnostic is in place, ContextFlow-specific components
can be introduced as new modules:

- KV-store abstractions for reusable document chunks
- CPU/GPU KV placement and switching policies
- memory-constrained chunk loading strategies
- accuracy-preserving and trade-off execution modes
- overlap between KV transfer and repair computation

These components should not be folded into the baseline definitions. They are
the system layer that ContextFlow will compare against the existing baselines.
