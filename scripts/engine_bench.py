#!/usr/bin/env python3
"""
engine_bench.py — vLLM vs SGLang Concurrent Benchmark

Measures TTFT, TPS, ITL, and aggregate throughput across concurrency levels
for any OpenAI-compatible inference server. Designed for fair comparison
between vLLM and SGLang on RTX 5070 Ti (SM_120).

Usage (PowerShell):
    python engine_bench.py --engine sglang --port 30000 --concurrency 1 8 16 32
    python engine_bench.py --engine vllm --port 8000 --concurrency 1 8 16 32

Requirements:
    pip install httpx aiohttp
"""

import asyncio
import json
import time
import argparse
import statistics
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx


# ──────────────────── Configuration ────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-8B-AWQ"
DEFAULT_PROMPT = "Explain the concept of KV cache in transformer inference in detail."
DEFAULT_MAX_TOKENS = 256
WARMUP_REQUESTS = 2
REQUESTS_PER_CONCURRENCY = 10


# ──────────────────── Data Classes ────────────────────

@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    ttft_ms: float          # Time to first token
    total_time_ms: float    # End-to-end latency
    output_tokens: int      # Number of generated tokens
    tps: float              # Tokens per second (per request)
    itl_ms: float           # Inter-token latency (median)
    itl_values: list[float] = field(default_factory=list)


@dataclass
class ConcurrencyResult:
    """Aggregated results for one concurrency level."""
    concurrency: int
    engine: str
    num_requests: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    tps_mean: float
    tps_p50: float
    aggregate_tps: float
    itl_p50_ms: float
    itl_p95_ms: float
    total_time_mean_ms: float
    output_tokens_mean: float


# ──────────────────── Core Benchmark ────────────────────

def percentile(data: list[float], p: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


async def measure_single_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    enable_thinking: bool = True,
) -> Optional[RequestMetrics]:
    """
    Send a single streaming request and measure TTFT, TPS, ITL.

    Uses SSE streaming to precisely capture first-token timing.
    """
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    token_times: list[float] = []
    output_tokens = 0

    try:
        async with client.stream(
            "POST",
            url,
            json=body,
            timeout=120.0,
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choices = chunk.get("choices", [])
                if not choices:
                    continue

                delta = choices[0].get("delta", {})
                content = delta.get("content", "")

                if content:
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    token_times.append(now)
                    output_tokens += 1

    except Exception as e:
        print(f"  [ERROR] Request failed: {e}")
        return None

    end = time.perf_counter()

    if first_token_time is None or output_tokens == 0:
        return None

    ttft_ms = (first_token_time - start) * 1000
    total_ms = (end - start) * 1000

    # Inter-token latency
    itl_values: list[float] = []
    for i in range(1, len(token_times)):
        itl_values.append((token_times[i] - token_times[i - 1]) * 1000)

    itl_median = statistics.median(itl_values) if itl_values else 0.0
    tps = output_tokens / ((end - first_token_time)) if end > first_token_time else 0.0

    return RequestMetrics(
        ttft_ms=ttft_ms,
        total_time_ms=total_ms,
        output_tokens=output_tokens,
        tps=tps,
        itl_ms=itl_median,
        itl_values=itl_values,
    )


async def run_concurrency_level(
    url: str,
    model: str,
    prompt: str,
    max_tokens: int,
    concurrency: int,
    num_requests: int,
    engine: str,
    enable_thinking: bool = True,
) -> ConcurrencyResult:
    """Run benchmark at a specific concurrency level."""
    print(f"\n{'='*60}")
    print(f"  Engine: {engine} | Concurrency: {concurrency} | Requests: {num_requests}")
    print(f"{'='*60}")

    # Warmup
    print(f"  Warming up ({WARMUP_REQUESTS} requests)...")
    async with httpx.AsyncClient() as client:
        for _ in range(WARMUP_REQUESTS):
            await measure_single_request(
                client, url, model, prompt, max_tokens, enable_thinking
            )

    # Actual benchmark
    print(f"  Running {num_requests} requests at c={concurrency}...")
    results: list[RequestMetrics] = []

    async with httpx.AsyncClient() as client:
        semaphore = asyncio.Semaphore(concurrency)

        async def bounded_request() -> Optional[RequestMetrics]:
            async with semaphore:
                return await measure_single_request(
                    client, url, model, prompt, max_tokens, enable_thinking
                )

        tasks = [bounded_request() for _ in range(num_requests)]
        bench_start = time.perf_counter()
        raw_results = await asyncio.gather(*tasks)
        bench_end = time.perf_counter()

    results = [r for r in raw_results if r is not None]

    if not results:
        print("  [ERROR] All requests failed!")
        return ConcurrencyResult(
            concurrency=concurrency, engine=engine, num_requests=0,
            ttft_p50_ms=0, ttft_p95_ms=0, ttft_p99_ms=0, ttft_mean_ms=0,
            tps_mean=0, tps_p50=0, aggregate_tps=0,
            itl_p50_ms=0, itl_p95_ms=0,
            total_time_mean_ms=0, output_tokens_mean=0,
        )

    # Aggregate metrics
    ttfts = [r.ttft_ms for r in results]
    tps_values = [r.tps for r in results]
    itl_all = [v for r in results for v in r.itl_values]
    total_times = [r.total_time_ms for r in results]
    total_tokens = sum(r.output_tokens for r in results)
    bench_duration = bench_end - bench_start

    result = ConcurrencyResult(
        concurrency=concurrency,
        engine=engine,
        num_requests=len(results),
        ttft_p50_ms=round(percentile(ttfts, 50), 1),
        ttft_p95_ms=round(percentile(ttfts, 95), 1),
        ttft_p99_ms=round(percentile(ttfts, 99), 1),
        ttft_mean_ms=round(statistics.mean(ttfts), 1),
        tps_mean=round(statistics.mean(tps_values), 1),
        tps_p50=round(percentile(tps_values, 50), 1),
        aggregate_tps=round(total_tokens / bench_duration, 1),
        itl_p50_ms=round(percentile(itl_all, 50), 1) if itl_all else 0.0,
        itl_p95_ms=round(percentile(itl_all, 95), 1) if itl_all else 0.0,
        total_time_mean_ms=round(statistics.mean(total_times), 1),
        output_tokens_mean=round(statistics.mean([r.output_tokens for r in results]), 1),
    )

    # Print summary
    print(f"\n  Results (c={concurrency}):")
    print(f"    TTFT P50:      {result.ttft_p50_ms:>8.1f} ms")
    print(f"    TTFT P95:      {result.ttft_p95_ms:>8.1f} ms")
    print(f"    TTFT mean:     {result.ttft_mean_ms:>8.1f} ms")
    print(f"    TPS mean:      {result.tps_mean:>8.1f} tok/s")
    print(f"    Agg TPS:       {result.aggregate_tps:>8.1f} tok/s")
    print(f"    ITL P50:       {result.itl_p50_ms:>8.1f} ms")
    print(f"    ITL P95:       {result.itl_p95_ms:>8.1f} ms")
    print(f"    Avg tokens:    {result.output_tokens_mean:>8.1f}")
    print(f"    Valid reqs:    {result.num_requests}/{num_requests}")

    return result


# ──────────────────── Main ────────────────────

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM vs SGLang Concurrent Benchmark"
    )
    parser.add_argument(
        "--engine", type=str, required=True,
        choices=["vllm", "sglang"],
        help="Inference engine name (for labeling results)",
    )
    parser.add_argument(
        "--port", type=int, default=30000,
        help="Server port (default: 30000 for SGLang, use 8000 for vLLM)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=[1, 8, 16, 32],
        help="Concurrency levels to test (default: 1 8 16 32)",
    )
    parser.add_argument(
        "--requests", type=int, default=REQUESTS_PER_CONCURRENCY,
        help=f"Requests per concurrency level (default: {REQUESTS_PER_CONCURRENCY})",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=DEFAULT_MAX_TOKENS,
        help=f"Max output tokens (default: {DEFAULT_MAX_TOKENS})",
    )
    parser.add_argument(
        "--no-thinking", action="store_true",
        help="Disable Qwen3 thinking mode",
    )
    parser.add_argument(
        "--prompt", type=str, default=DEFAULT_PROMPT,
        help="Prompt to use for benchmark",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save results (default: results/)",
    )

    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    enable_thinking = not args.no_thinking

    print(f"\n{'#'*60}")
    print(f"  Engine Benchmark: {args.engine.upper()}")
    print(f"  URL: {url}")
    print(f"  Model: {args.model}")
    print(f"  Concurrency levels: {args.concurrency}")
    print(f"  Requests per level: {args.requests}")
    print(f"  Max tokens: {args.max_tokens}")
    print(f"  Thinking: {'ON' if enable_thinking else 'OFF'}")
    print(f"{'#'*60}")

    all_results: list[ConcurrencyResult] = []

    for c in args.concurrency:
        result = await run_concurrency_level(
            url=url,
            model=args.model,
            prompt=args.prompt,
            max_tokens=args.max_tokens,
            concurrency=c,
            num_requests=args.requests,
            engine=args.engine,
            enable_thinking=enable_thinking,
        )
        all_results.append(result)

    # Print comparison table
    print(f"\n\n{'='*80}")
    print(f"  SUMMARY — {args.engine.upper()} (Qwen3-8B-AWQ, RTX 5070 Ti)")
    print(f"{'='*80}")
    header = f"{'c':>4} | {'TTFT P50':>10} | {'TTFT P95':>10} | {'TPS mean':>10} | {'Agg TPS':>10} | {'ITL P50':>10}"
    print(f"  {header}")
    print(f"  {'-'*len(header)}")
    for r in all_results:
        row = f"{r.concurrency:>4} | {r.ttft_p50_ms:>8.1f}ms | {r.ttft_p95_ms:>8.1f}ms | {r.tps_mean:>8.1f}  | {r.aggregate_tps:>8.1f}  | {r.itl_p50_ms:>8.1f}ms"
        print(f"  {row}")

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    thinking_tag = "think_on" if enable_thinking else "think_off"
    filename = f"engine_{args.engine}_{thinking_tag}_{timestamp}.json"
    filepath = output_dir / filename

    output_data = {
        "engine": args.engine,
        "model": args.model,
        "port": args.port,
        "thinking": enable_thinking,
        "max_tokens": args.max_tokens,
        "requests_per_level": args.requests,
        "prompt": args.prompt,
        "timestamp": timestamp,
        "results": [asdict(r) for r in all_results],
    }

    with open(filepath, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n  Results saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())