from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any

from contextflow.benchmarks.chunk_sweep import (
    CHUNKING_MODE_TOKEN,
    chunking_config_record,
    parse_chunk_count_specs,
    resolve_chunk_cases_for_available_count,
    token_count_record,
)
from contextflow.benchmarks.records import write_jsonl_record
from contextflow.chunking import (
    TokenChunkingConfig,
    flatten_doc_chunk_ids,
    slice_tokenized_doc_chunks,
    token_chunk_tokenized_example,
    token_chunking_metadata,
)
from contextflow.data import (
    PROMPT_POLICY_CACHEBLEND_QA,
    SUPPORTED_CACHEBLEND_PROMPT_POLICIES,
    SUPPORTED_QA_DATASET_KEYS,
    TokenizedExample,
    build_cacheblend_prompt,
    load_qa_dataset_examples,
    normalize_dataset_key,
)


STATUS_SUCCESS = "success"
STATUS_CONTEXT_LIMIT = "context_limit"
STATUS_OOM = "oom"
STATUS_SERVER_ERROR = "server_error"
STATUS_REQUEST_TIMEOUT = "request_timeout"
STATUS_SKIPPED_TOO_LONG = "skipped_too_long"


class VLLMHTTPError(RuntimeError):
    def __init__(self, status_code: int, body: str) -> None:
        super().__init__(f"HTTP {status_code}: {body}")
        self.status_code = status_code
        self.body = body


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="vLLM full-context RAG pressure chunk-count sweep runner."
    )
    parser.add_argument("--dataset", required=True, choices=sorted(SUPPORTED_QA_DATASET_KEYS))
    parser.add_argument("--input", required=True, help="Local JSON or JSONL dataset path.")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--metrics-url", default=None)
    parser.add_argument("--model", required=True, help="Served vLLM model name.")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument(
        "--chunk-counts",
        default="8,16,32,64,128,all",
        help="Comma-separated token chunk counts. Use 'all' for all produced chunks.",
    )
    parser.add_argument("--chunk-size-tokens", type=int, default=1024)
    parser.add_argument("--chunk-overlap-tokens", type=int, default=0)
    parser.add_argument("--max-chunks", type=int, default=None)
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Skip cases whose prompt token count exceeds this budget.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--prompt-policy",
        choices=sorted(SUPPORTED_CACHEBLEND_PROMPT_POLICIES),
        default=PROMPT_POLICY_CACHEBLEND_QA,
    )
    parser.add_argument(
        "--poll-interval-ms",
        type=int,
        default=50,
        help="Metrics/nvidia-smi polling interval during each request.",
    )
    parser.add_argument(
        "--request-timeout-seconds",
        type=float,
        default=600.0,
        help="HTTP request timeout per case.",
    )
    parser.add_argument(
        "--nvidia-smi-gpu-index",
        type=int,
        default=None,
        help="Optional GPU index for nvidia-smi memory sampling.",
    )
    parser.add_argument(
        "--allow-detokenize-fallback",
        action="store_true",
        help=(
            "If token-id prompts are rejected by the server, detokenize through "
            "vLLM /detokenize and retry with a text prompt."
        ),
    )
    parser.add_argument("--output-jsonl", required=True)
    return parser.parse_args()


def normalize_server_url(server_url: str) -> str:
    return server_url.rstrip("/")


def resolve_metrics_url(server_url: str, metrics_url: str | None) -> str:
    if metrics_url:
        return metrics_url
    return f"{normalize_server_url(server_url)}/metrics"


def http_json(
    url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            raw = response.read().decode("utf-8")
    except urllib.error.HTTPError as error:
        raise_vllm_http_error(error)
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from {url}, got {type(parsed).__name__}.")
    return parsed


def tokenize_with_vllm(
    server_url: str,
    *,
    model: str,
    prompt: str,
    timeout_seconds: float,
) -> list[int]:
    response = http_json(
        f"{normalize_server_url(server_url)}/tokenize",
        {"model": model, "prompt": prompt, "add_special_tokens": False},
        timeout_seconds=timeout_seconds,
    )
    token_ids = response.get("token_ids")
    if token_ids is None:
        token_ids = response.get("tokens")
    if token_ids is None and isinstance(response.get("data"), dict):
        token_ids = response["data"].get("token_ids", response["data"].get("tokens"))
    if not isinstance(token_ids, list):
        raise ValueError(f"Could not find token id list in /tokenize response: {response}")
    return [int(token_id) for token_id in token_ids]


def detokenize_with_vllm(
    server_url: str,
    *,
    model: str,
    token_ids: list[int],
    timeout_seconds: float,
) -> str:
    response = http_json(
        f"{normalize_server_url(server_url)}/detokenize",
        {"model": model, "tokens": token_ids, "skip_special_tokens": False},
        timeout_seconds=timeout_seconds,
    )
    text = response.get("prompt")
    if text is None:
        text = response.get("text")
    if text is None and isinstance(response.get("data"), dict):
        text = response["data"].get("prompt", response["data"].get("text"))
    if not isinstance(text, str):
        raise ValueError(f"Could not find text in /detokenize response: {response}")
    return text


def tokenize_prompt_with_vllm(
    server_url: str,
    *,
    model: str,
    doc_prompts: list[str],
    q_prompt: str,
    timeout_seconds: float,
) -> TokenizedExample:
    return TokenizedExample(
        doc_chunk_ids=[
            tokenize_with_vllm(
                server_url,
                model=model,
                prompt=doc_prompt,
                timeout_seconds=timeout_seconds,
            )
            for doc_prompt in doc_prompts
        ],
        q_ids=tokenize_with_vllm(
            server_url,
            model=model,
            prompt=q_prompt,
            timeout_seconds=timeout_seconds,
        ),
        answers=[],
        example_id=None,
    )


def prompt_token_ids_from_tokenized(tokenized_example: TokenizedExample) -> list[int]:
    return flatten_doc_chunk_ids(tokenized_example.doc_chunk_ids) + list(tokenized_example.q_ids)


def build_completion_payload(
    *,
    model: str,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt_token_ids,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }


def build_text_completion_payload(
    *,
    model: str,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_new_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }


def read_http_error_body(error: urllib.error.HTTPError) -> str:
    try:
        return error.read().decode("utf-8")
    except Exception:
        return str(error)


def raise_vllm_http_error(error: urllib.error.HTTPError) -> None:
    raise VLLMHTTPError(error.code, read_http_error_body(error)) from error


def classify_request_error(error: Exception) -> str:
    message = str(error).lower()
    if isinstance(error, TimeoutError):
        return STATUS_REQUEST_TIMEOUT
    if isinstance(error, VLLMHTTPError):
        if error.status_code == 400 and any(
            marker in message
            for marker in ("maximum context", "context length", "too long", "max model len")
        ):
            return STATUS_CONTEXT_LIMIT
        if error.status_code in {408, 504}:
            return STATUS_REQUEST_TIMEOUT
    if isinstance(error, urllib.error.HTTPError):
        if error.code in {408, 504}:
            return STATUS_REQUEST_TIMEOUT
    if "out of memory" in message or "cuda oom" in message or "cuda error: out of memory" in message:
        return STATUS_OOM
    if any(marker in message for marker in ("maximum context", "context length", "too long", "max model len")):
        return STATUS_CONTEXT_LIMIT
    if "timed out" in message or "timeout" in message:
        return STATUS_REQUEST_TIMEOUT
    return STATUS_SERVER_ERROR


def serialize_error(error: Exception) -> dict[str, Any]:
    message = str(error)
    status_code = None
    if isinstance(error, VLLMHTTPError):
        status_code = error.status_code
        message = error.body
    elif isinstance(error, urllib.error.HTTPError):
        status_code = error.code
        message = read_http_error_body(error)
    return {
        "type": type(error).__name__,
        "status_code": status_code,
        "message": message,
    }


def iter_sse_json_lines(response: Any) -> Any:
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[len("data:") :].strip()
        if data == "[DONE]":
            break
        yield json.loads(data)


def extract_delta_text_and_token_ids(event: dict[str, Any]) -> tuple[str, list[int]]:
    choices = event.get("choices")
    if not isinstance(choices, list) or not choices:
        return "", []
    choice = choices[0]
    text = choice.get("text", "")
    token_ids = choice.get("token_ids")
    if token_ids is None:
        token_ids = choice.get("tokens")
    if token_ids is None:
        token_ids = []
    return str(text), [int(token_id) for token_id in token_ids]


def post_streaming_completion(
    server_url: str,
    payload: dict[str, Any],
    *,
    timeout_seconds: float,
) -> dict[str, Any]:
    url = f"{normalize_server_url(server_url)}/v1/completions"
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_time = None
    last_token_time = None
    generated_text_parts: list[str] = []
    generated_token_ids: list[int] = []
    final_usage = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            for event in iter_sse_json_lines(response):
                now = time.perf_counter()
                if event.get("usage") is not None:
                    final_usage = event["usage"]
                delta_text, token_ids = extract_delta_text_and_token_ids(event)
                if delta_text:
                    generated_text_parts.append(delta_text)
                if token_ids:
                    generated_token_ids.extend(token_ids)
                if delta_text or token_ids:
                    if first_token_time is None:
                        first_token_time = now
                    last_token_time = now
    except urllib.error.HTTPError as error:
        raise_vllm_http_error(error)
    end = time.perf_counter()
    completion_tokens = len(generated_token_ids)
    if completion_tokens == 0 and isinstance(final_usage, dict):
        completion_tokens = int(final_usage.get("completion_tokens", 0) or 0)
    ttft_seconds = None if first_token_time is None else first_token_time - start
    e2e_latency_seconds = end - start
    tpot_seconds = None
    if first_token_time is not None and last_token_time is not None and completion_tokens > 1:
        tpot_seconds = (last_token_time - first_token_time) / (completion_tokens - 1)
    return {
        "generated_text": "".join(generated_text_parts),
        "generated_token_ids": generated_token_ids,
        "completion_tokens": completion_tokens,
        "ttft_seconds": ttft_seconds,
        "tpot_seconds": tpot_seconds,
        "e2e_latency_seconds": e2e_latency_seconds,
        "usage": final_usage,
    }


def parse_prometheus_metrics(text: str) -> dict[str, float]:
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if " " not in line:
            continue
        name_and_labels, raw_value = line.rsplit(" ", 1)
        metric_name = name_and_labels.split("{", 1)[0]
        try:
            value = float(raw_value)
        except ValueError:
            continue
        if metric_name in {"vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"}:
            values["vllm_kv_cache_usage_perc"] = max(
                value,
                values.get("vllm_kv_cache_usage_perc", 0.0),
            )
        elif metric_name == "vllm:num_requests_running":
            values["vllm_num_requests_running"] = max(
                value,
                values.get("vllm_num_requests_running", 0.0),
            )
        elif metric_name == "vllm:num_requests_waiting":
            values["vllm_num_requests_waiting"] = max(
                value,
                values.get("vllm_num_requests_waiting", 0.0),
            )
    return values


def fetch_metrics(metrics_url: str, timeout_seconds: float = 2.0) -> dict[str, float]:
    with urllib.request.urlopen(metrics_url, timeout=timeout_seconds) as response:
        text = response.read().decode("utf-8")
    return parse_prometheus_metrics(text)


def read_nvidia_smi_memory_mb(gpu_index: int | None) -> float | None:
    cmd = [
        "nvidia-smi",
        "--query-gpu=memory.used",
        "--format=csv,noheader,nounits",
    ]
    if gpu_index is not None:
        cmd.extend(["-i", str(gpu_index)])
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    values = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        try:
            values.append(float(stripped))
        except ValueError:
            continue
    if not values:
        return None
    return max(values)


class PressureSampler:
    def __init__(
        self,
        *,
        metrics_url: str,
        poll_interval_seconds: float,
        nvidia_smi_gpu_index: int | None,
    ) -> None:
        self.metrics_url = metrics_url
        self.poll_interval_seconds = poll_interval_seconds
        self.nvidia_smi_gpu_index = nvidia_smi_gpu_index
        self.samples: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.poll_interval_seconds * 4))

    def _run(self) -> None:
        while not self._stop.is_set():
            sample: dict[str, Any] = {"timestamp": time.time()}
            try:
                sample.update(fetch_metrics(self.metrics_url))
            except Exception as error:
                sample["metrics_error"] = str(error)
            nvidia_memory_mb = read_nvidia_smi_memory_mb(self.nvidia_smi_gpu_index)
            if nvidia_memory_mb is not None:
                sample["nvidia_smi_memory_mb"] = nvidia_memory_mb
            self.samples.append(sample)
            self._stop.wait(self.poll_interval_seconds)

    def summary(self) -> dict[str, Any]:
        return {
            "vllm_kv_cache_usage_peak": max_optional(
                self.samples,
                "vllm_kv_cache_usage_perc",
            ),
            "vllm_kv_cache_usage_before": first_optional(
                self.samples,
                "vllm_kv_cache_usage_perc",
            ),
            "vllm_kv_cache_usage_after": last_optional(
                self.samples,
                "vllm_kv_cache_usage_perc",
            ),
            "nvidia_smi_memory_peak_mb": max_optional(
                self.samples,
                "nvidia_smi_memory_mb",
            ),
            "nvidia_smi_memory_before_mb": first_optional(
                self.samples,
                "nvidia_smi_memory_mb",
            ),
            "nvidia_smi_memory_after_mb": last_optional(
                self.samples,
                "nvidia_smi_memory_mb",
            ),
            "sampler_num_samples": len(self.samples),
        }


def first_optional(records: list[dict[str, Any]], key: str) -> float | None:
    for record in records:
        if record.get(key) is not None:
            return float(record[key])
    return None


def last_optional(records: list[dict[str, Any]], key: str) -> float | None:
    for record in reversed(records):
        if record.get(key) is not None:
            return float(record[key])
    return None


def max_optional(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return max(values)


def mean_optional(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if record.get(key) is not None]
    if not values:
        return None
    return sum(values) / len(values)


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["chunk_count"])].append(record)

    by_chunk_count: dict[str, Any] = {}
    for chunk_count in sorted(grouped):
        chunk_records = grouped[chunk_count]
        success_records = [
            record for record in chunk_records if record.get("status") == STATUS_SUCCESS
        ]
        by_chunk_count[str(chunk_count)] = {
            "case_count": len(chunk_records),
            "success_count": len(success_records),
            "context_limit_count": count_status(chunk_records, STATUS_CONTEXT_LIMIT),
            "oom_count": count_status(chunk_records, STATUS_OOM),
            "server_error_count": count_status(chunk_records, STATUS_SERVER_ERROR),
            "request_timeout_count": count_status(chunk_records, STATUS_REQUEST_TIMEOUT),
            "skipped_too_long_count": count_status(chunk_records, STATUS_SKIPPED_TOO_LONG),
            "mean_prompt_tokens": mean_optional(chunk_records, "prompt_tokens"),
            "mean_doc_token_count": mean_optional(chunk_records, "doc_token_count"),
            "mean_ttft_seconds": mean_optional(success_records, "ttft_seconds"),
            "mean_tpot_seconds": mean_optional(success_records, "tpot_seconds"),
            "mean_e2e_latency_seconds": mean_optional(
                success_records,
                "e2e_latency_seconds",
            ),
            "mean_completion_tokens": mean_optional(success_records, "completion_tokens"),
            "mean_vllm_kv_cache_usage_peak": mean_optional(
                success_records,
                "vllm_kv_cache_usage_peak",
            ),
            "mean_nvidia_smi_memory_peak_mb": mean_optional(
                success_records,
                "nvidia_smi_memory_peak_mb",
            ),
        }

    success_records = [record for record in records if record.get("status") == STATUS_SUCCESS]
    context_limit_records = [
        record for record in records if record.get("status") == STATUS_CONTEXT_LIMIT
    ]
    oom_records = [record for record in records if record.get("status") == STATUS_OOM]
    return {
        "case_count": len(records),
        "success_count": len(success_records),
        "context_limit_count": len(context_limit_records),
        "oom_count": len(oom_records),
        "server_error_count": count_status(records, STATUS_SERVER_ERROR),
        "request_timeout_count": count_status(records, STATUS_REQUEST_TIMEOUT),
        "skipped_too_long_count": count_status(records, STATUS_SKIPPED_TOO_LONG),
        "max_success_chunk_count": (
            max(record["chunk_count"] for record in success_records)
            if success_records
            else None
        ),
        "min_context_limit_chunk_count": (
            min(record["chunk_count"] for record in context_limit_records)
            if context_limit_records
            else None
        ),
        "min_oom_chunk_count": (
            min(record["chunk_count"] for record in oom_records) if oom_records else None
        ),
        "by_chunk_count": by_chunk_count,
    }


def count_status(records: list[dict[str, Any]], status: str) -> int:
    return sum(1 for record in records if record.get("status") == status)


def build_case_base_record(
    *,
    example_index: int,
    example_id: str | None,
    dataset_key: str,
    model: str,
    server_url: str,
    chunk_case: dict[str, Any],
    token_counts: dict[str, int],
    chunking_config: dict[str, Any],
    max_new_tokens: int,
    temperature: float,
) -> dict[str, Any]:
    return {
        "record_type": "vllm_rag_pressure_case",
        "example_index": example_index,
        "example_id": example_id,
        "dataset": dataset_key,
        "model": model,
        "server_url": server_url,
        "available_chunk_count": chunk_case["available_chunk_count"],
        "requested_chunk_count": chunk_case["requested_chunk_count"],
        "chunk_count": chunk_case["chunk_count"],
        **chunking_config,
        **token_counts,
        "prompt_tokens": token_counts["total_prefill_token_count"],
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
    }


def run_vllm_case(
    *,
    server_url: str,
    metrics_url: str,
    model: str,
    prompt_token_ids: list[int],
    max_new_tokens: int,
    temperature: float,
    poll_interval_seconds: float,
    request_timeout_seconds: float,
    nvidia_smi_gpu_index: int | None,
    allow_detokenize_fallback: bool,
) -> dict[str, Any]:
    sampler = PressureSampler(
        metrics_url=metrics_url,
        poll_interval_seconds=poll_interval_seconds,
        nvidia_smi_gpu_index=nvidia_smi_gpu_index,
    )
    payload = build_completion_payload(
        model=model,
        prompt_token_ids=prompt_token_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    sampler.start()
    try:
        result = post_streaming_completion(
            server_url,
            payload,
            timeout_seconds=request_timeout_seconds,
        )
        prompt_transport = "token_ids"
    except VLLMHTTPError as error:
        if not allow_detokenize_fallback:
            sampler.stop()
            raise
        error_body = error.body.lower()
        if "prompt" not in error_body or "token" not in error_body:
            sampler.stop()
            raise
        prompt_text = detokenize_with_vllm(
            server_url,
            model=model,
            token_ids=prompt_token_ids,
            timeout_seconds=request_timeout_seconds,
        )
        text_payload = build_text_completion_payload(
            model=model,
            prompt=prompt_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        result = post_streaming_completion(
            server_url,
            text_payload,
            timeout_seconds=request_timeout_seconds,
        )
        prompt_transport = "detokenized_text_fallback"
    finally:
        sampler.stop()
    result.update(sampler.summary())
    result["prompt_transport"] = prompt_transport
    return result


def main() -> None:
    args = parse_args()
    dataset_key = normalize_dataset_key(args.dataset)
    if args.limit <= 0:
        raise ValueError("--limit must be positive.")
    if args.max_total_tokens is not None and args.max_total_tokens <= 0:
        raise ValueError("--max-total-tokens must be positive when provided.")
    if args.poll_interval_ms <= 0:
        raise ValueError("--poll-interval-ms must be positive.")

    server_url = normalize_server_url(args.server_url)
    metrics_url = resolve_metrics_url(server_url, args.metrics_url)
    chunk_count_specs = parse_chunk_count_specs(args.chunk_counts)
    examples = load_qa_dataset_examples(dataset_key, args.input)[: args.limit]
    if not examples:
        raise ValueError("Expected at least one dataset example.")

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_case_records: list[dict[str, Any]] = []
    summary_chunking_config = chunking_config_record(
        CHUNKING_MODE_TOKEN,
        chunk_size_tokens=args.chunk_size_tokens,
        chunk_overlap_tokens=args.chunk_overlap_tokens,
        max_chunks=args.max_chunks,
    )
    if args.max_total_tokens is not None:
        summary_chunking_config["max_total_tokens"] = args.max_total_tokens

    with output_path.open("w", encoding="utf-8") as output_file:
        for example_index, example in enumerate(examples):
            prompt = build_cacheblend_prompt(
                example,
                prompt_policy=args.prompt_policy,
                dataset=dataset_key,
            )
            tokenized = tokenize_prompt_with_vllm(
                server_url,
                model=args.model,
                doc_prompts=prompt.doc_prompts,
                q_prompt=prompt.q_prompt,
                timeout_seconds=args.request_timeout_seconds,
            )
            tokenized = TokenizedExample(
                doc_chunk_ids=tokenized.doc_chunk_ids,
                q_ids=tokenized.q_ids,
                answers=list(example.answers),
                example_id=example.example_id,
            )
            source_doc_token_count = sum(len(ids) for ids in tokenized.doc_chunk_ids)
            token_chunking_config = TokenChunkingConfig(
                chunk_size_tokens=args.chunk_size_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
                max_chunks=args.max_chunks,
            )
            token_chunked = token_chunk_tokenized_example(
                tokenized,
                chunk_size_tokens=args.chunk_size_tokens,
                chunk_overlap_tokens=args.chunk_overlap_tokens,
                max_chunks=args.max_chunks,
            )
            chunking_config = token_chunking_metadata(
                token_chunking_config,
                source_doc_token_count=source_doc_token_count,
                produced_chunk_count=len(token_chunked.doc_chunk_ids),
            )
            if args.max_total_tokens is not None:
                chunking_config["max_total_tokens"] = args.max_total_tokens
            chunk_cases = resolve_chunk_cases_for_available_count(
                available_count=len(token_chunked.doc_chunk_ids),
                chunk_count_specs=chunk_count_specs,
            )
            for chunk_case in chunk_cases:
                sliced = slice_tokenized_doc_chunks(
                    token_chunked,
                    chunk_count=int(chunk_case["chunk_count"]),
                )
                token_counts = token_count_record(sliced)
                record = build_case_base_record(
                    example_index=example_index,
                    example_id=example.example_id,
                    dataset_key=dataset_key,
                    model=args.model,
                    server_url=server_url,
                    chunk_case=chunk_case,
                    token_counts=token_counts,
                    chunking_config=chunking_config,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                )
                if (
                    args.max_total_tokens is not None
                    and token_counts["total_prefill_token_count"] > args.max_total_tokens
                ):
                    record.update(
                        {
                            "status": STATUS_SKIPPED_TOO_LONG,
                            "skip_reason": (
                                "total_prefill_token_count_exceeds_max_total_tokens"
                            ),
                        }
                    )
                    all_case_records.append(record)
                    write_jsonl_record(output_file, record)
                    continue

                prompt_token_ids = prompt_token_ids_from_tokenized(sliced)
                try:
                    case_result = run_vllm_case(
                        server_url=server_url,
                        metrics_url=metrics_url,
                        model=args.model,
                        prompt_token_ids=prompt_token_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        poll_interval_seconds=args.poll_interval_ms / 1000.0,
                        request_timeout_seconds=args.request_timeout_seconds,
                        nvidia_smi_gpu_index=args.nvidia_smi_gpu_index,
                        allow_detokenize_fallback=args.allow_detokenize_fallback,
                    )
                    record.update({"status": STATUS_SUCCESS, **case_result})
                except Exception as error:
                    record.update(
                        {
                            "status": classify_request_error(error),
                            "error": serialize_error(error),
                        }
                    )
                all_case_records.append(record)
                write_jsonl_record(output_file, record)

    summary = {
        "record_type": "vllm_rag_pressure_summary",
        "dataset": dataset_key,
        "model": args.model,
        "server_url": server_url,
        "metrics_url": metrics_url,
        **summary_chunking_config,
        "chunk_count_specs": chunk_count_specs,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        **summarize_records(all_case_records),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
