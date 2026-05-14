# vLLM RAG Pressure Diagnostic

`experiments/vllm_rag_pressure_runner.py` measures native vLLM full-context
serving pressure as retrieved/chunked context grows. It is intentionally separate
from the HF/PyTorch reference baselines:

- it does not run Full Recompute, Naive KV Reuse, CacheBlend-style Repair, or
  FusionRAG-style Repair;
- it does not evaluate EM/F1;
- it sends full-context prompts directly to a running vLLM OpenAI-compatible
  server.

The purpose is to answer one system question: as the number of retrieved chunks
increases, how do TTFT, TPOT, E2E latency, vLLM KV cache usage, nvidia-smi memory,
and context/OOM boundaries change under native vLLM serving?

## Start vLLM

Example:

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --dtype auto \
  --generation-config vllm \
  --gpu-memory-utilization 0.90 \
  --max-model-len 131072 \
  --host 0.0.0.0 \
  --port 8000
```

The runner expects:

- OpenAI-compatible completions endpoint: `http://HOST:PORT/v1/completions`
- tokenizer endpoint: `http://HOST:PORT/tokenize`
- Prometheus metrics endpoint: `http://HOST:PORT/metrics`

## Run A Small Probe

```bash
python experiments/vllm_rag_pressure_runner.py \
  --dataset longbook_qa_en \
  --input data/raw/longbook_qa_en/test.jsonl \
  --server-url http://localhost:8000 \
  --metrics-url http://localhost:8000/metrics \
  --model Qwen/Qwen2.5-7B-Instruct \
  --limit 1 \
  --chunk-size-tokens 1024 \
  --chunk-overlap-tokens 0 \
  --max-chunks 16 \
  --max-total-tokens 32768 \
  --chunk-counts 1,2,4,8,16,all \
  --max-new-tokens 16 \
  --temperature 0 \
  --poll-interval-ms 50 \
  --nvidia-smi-gpu-index 0 \
  --output-jsonl results/vllm_pressure_longbook_qwen2_7b_probe.jsonl
```

If the server rejects token-id prompts, retry with:

```bash
--allow-detokenize-fallback
```

That fallback asks vLLM `/detokenize` to turn the exact token ids back into text,
then sends a text prompt to `/v1/completions`.

## Output

The JSONL contains one record per example and chunk count with:

- `prompt_tokens`, `doc_token_count`, `query_token_count`
- `ttft_seconds`
- `tpot_seconds`
- `e2e_latency_seconds`
- `completion_tokens`
- `vllm_kv_cache_usage_peak`
- `nvidia_smi_memory_peak_mb`
- `status`: `success`, `context_limit`, `oom`, `server_error`,
  `request_timeout`, or `skipped_too_long`

The printed summary aggregates those fields by `chunk_count` and reports
boundaries such as `max_success_chunk_count`, `min_context_limit_chunk_count`,
and `min_oom_chunk_count`.

## Interpretation

Use `vllm_kv_cache_usage_peak` as the primary KV pressure signal. The
`nvidia_smi_memory_*` fields are useful sanity checks, but vLLM often preallocates
GPU/KV memory, so absolute nvidia-smi memory can be flat even when KV cache usage
changes.

The runner measures native full-context serving pressure. It should be compared
alongside, not mixed into, the HF/PyTorch reference baseline chunk sweeps.
