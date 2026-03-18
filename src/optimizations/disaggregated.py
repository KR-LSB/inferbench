"""Disaggregated inference measurement module.

Measures Prefill and Decode phases separately by extracting TTFT from
streaming responses. This is the core differentiator of InferBench:
arXiv:2601.09527 only measured E2E latency.

GTC 2026 context: NVIDIA's Vera Rubin platform uses disaggregated inference
(Rubin GPU for prefill, Groq LPU for decode). This module provides the
empirical data to understand *why* that separation is beneficial.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass

import httpx

from src.bench.metrics import RequestMetrics


@dataclass
class DisaggregatedResult:
    """Results from a disaggregated inference measurement.

    Extends RequestMetrics with explicit prefill/decode breakdown.
    """

    request_metrics: RequestMetrics
    prefill_ratio: float  # prefill_time / total_time (0.0 ~ 1.0)

    @property
    def prefill_time_ms(self) -> float:
        return self.request_metrics.ttft_ms

    @property
    def decode_time_ms(self) -> float:
        return self.request_metrics.decode_time_ms

    @property
    def prefill_tokens_per_sec(self) -> float:
        return self.request_metrics.prefill_tokens_per_sec

    @property
    def decode_tokens_per_sec(self) -> float:
        return self.request_metrics.decode_tokens_per_sec


async def measure_single_request(
    prompt: str,
    max_tokens: int = 512,
    server_url: str = "http://localhost:8000",
    model: str = "default",
    system_prompt: str | None = None,
) -> DisaggregatedResult:
    """Measure prefill and decode times for a single request via streaming.

    Uses the OpenAI-compatible streaming API to extract precise TTFT
    (≈ prefill time) by measuring time to first SSE data chunk.

    Args:
        prompt: The user prompt to send.
        max_tokens: Maximum tokens to generate.
        server_url: Base URL of the inference server.
        model: Model name for the API request.
        system_prompt: Optional system prompt.

    Returns:
        DisaggregatedResult with separated prefill/decode metrics.
    """
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    start = time.perf_counter()
    first_token_time: float | None = None
    output_tokens = 0
    token_timestamps: list[float] = []

    async with httpx.AsyncClient() as client:
        async with client.stream(
            "POST",
            f"{server_url}/v1/chat/completions",
            json={
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            },
            timeout=120.0,
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue

                now = time.perf_counter()
                try:
                    chunk = json.loads(line[6:])
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue

                if content:
                    if first_token_time is None:
                        first_token_time = now
                    output_tokens += 1
                    token_timestamps.append((now - start) * 1000)

    end = time.perf_counter()

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0.0
    total_ms = (end - start) * 1000

    # Estimate input tokens (~1.3 tokens per word for English)
    total_text = (system_prompt or "") + " " + prompt
    input_tokens_est = int(len(total_text.split()) * 1.3)

    metrics = RequestMetrics(
        ttft_ms=ttft_ms,
        total_time_ms=total_ms,
        input_tokens=input_tokens_est,
        output_tokens=output_tokens,
        token_timestamps_ms=token_timestamps,
    )

    prefill_ratio = ttft_ms / total_ms if total_ms > 0 else 0.0

    return DisaggregatedResult(
        request_metrics=metrics,
        prefill_ratio=prefill_ratio,
    )


async def run_disaggregated_experiment(
    prompts: list[str],
    concurrency: int = 1,
    max_tokens: int = 512,
    server_url: str = "http://localhost:8000",
    model: str = "default",
    system_prompt: str | None = None,
) -> list[DisaggregatedResult]:
    """Run disaggregated measurement across multiple requests.

    Sends requests at the specified concurrency level and collects
    per-request prefill/decode breakdown.

    Args:
        prompts: List of prompts to benchmark.
        concurrency: Number of concurrent requests.
        max_tokens: Maximum tokens per request.
        server_url: Inference server URL.
        model: Model name.
        system_prompt: Optional shared system prompt.

    Returns:
        List of DisaggregatedResult for each request.
    """
    semaphore = asyncio.Semaphore(concurrency)
    results: list[DisaggregatedResult] = []

    async def _bounded_request(prompt: str) -> DisaggregatedResult:
        async with semaphore:
            return await measure_single_request(
                prompt=prompt,
                max_tokens=max_tokens,
                server_url=server_url,
                model=model,
                system_prompt=system_prompt,
            )

    tasks = [_bounded_request(p) for p in prompts]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    return list(results)
