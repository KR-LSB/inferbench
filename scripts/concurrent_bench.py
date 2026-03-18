"""InferBench Concurrency Benchmark — Standalone Script

Measures TTFT, TPS, ITL under varying concurrency levels (c=1,8,16,32).
Uses asyncio + http.client for concurrent request execution.
No external dependencies required (stdlib only).

Usage:
    python scripts/concurrent_bench.py
    python scripts/concurrent_bench.py --concurrency 1,8,16,32 --requests-per-level 10
"""

from __future__ import annotations

import argparse
import asyncio
import http.client
import json
import statistics
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlparse


@dataclass
class RequestResult:
    """Result of a single streaming inference request."""

    ttft_ms: float
    total_time_ms: float
    output_tokens: int
    token_timestamps_ms: list[float]
    success: bool = True
    error: str = ""

    @property
    def decode_time_ms(self) -> float:
        return max(0.0, self.total_time_ms - self.ttft_ms)

    @property
    def tps(self) -> float:
        decode_s = self.decode_time_ms / 1000
        return self.output_tokens / decode_s if decode_s > 0 else 0.0

    @property
    def itl_ms(self) -> float:
        if len(self.token_timestamps_ms) < 2:
            return 0.0
        deltas = [
            self.token_timestamps_ms[i] - self.token_timestamps_ms[i - 1]
            for i in range(1, len(self.token_timestamps_ms))
        ]
        return statistics.mean(deltas)


def _sync_streaming_request(
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a single streaming request (blocking). Runs in a thread."""
    parsed = urlparse(server_url)
    try:
        conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=180)
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": True,
            "temperature": 0.7,
        })

        start = time.perf_counter()
        first_token_time: float | None = None
        output_tokens = 0
        token_timestamps: list[float] = []

        conn.request("POST", "/v1/chat/completions", body=body, headers={
            "Content-Type": "application/json",
        })
        response = conn.getresponse()
        buffer = ""

        while True:
            chunk = response.read(1024)
            if not chunk:
                break
            buffer += chunk.decode("utf-8", errors="replace")
            while "\n" in buffer:
                line, buffer = buffer.split("\n", 1)
                line = line.strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                now = time.perf_counter()
                try:
                    data = json.loads(line[6:])
                    delta = data.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                except (json.JSONDecodeError, IndexError, KeyError):
                    continue
                if content:
                    if first_token_time is None:
                        first_token_time = now
                    output_tokens += 1
                    token_timestamps.append((now - start) * 1000)

        end = time.perf_counter()
        conn.close()

        ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0.0
        total_ms = (end - start) * 1000

        return RequestResult(
            ttft_ms=ttft_ms,
            total_time_ms=total_ms,
            output_tokens=output_tokens,
            token_timestamps_ms=token_timestamps,
        )
    except Exception as e:
        return RequestResult(
            ttft_ms=0, total_time_ms=0, output_tokens=0,
            token_timestamps_ms=[], success=False, error=str(e),
        )


async def run_concurrent_requests(
    server_url: str,
    model: str,
    prompts: list[str],
    max_tokens: int,
    concurrency: int,
) -> list[RequestResult]:
    """Run requests at a given concurrency level using thread pool."""
    loop = asyncio.get_event_loop()
    semaphore = asyncio.Semaphore(concurrency)
    results: list[RequestResult] = []

    async def _bounded(prompt: str) -> RequestResult:
        async with semaphore:
            return await loop.run_in_executor(
                None, _sync_streaming_request, server_url, model, prompt, max_tokens,
            )

    tasks = [_bounded(p) for p in prompts]
    results = await asyncio.gather(*tasks)
    return list(results)


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def print_level_summary(
    concurrency: int,
    results: list[RequestResult],
    duration_s: float,
) -> dict:
    """Print and return summary for one concurrency level."""
    successful = [r for r in results if r.success]
    if not successful:
        print(f"    c={concurrency}: ALL FAILED")
        return {}

    ttfts = [r.ttft_ms for r in successful]
    tps_vals = [r.tps for r in successful]
    itls = [r.itl_ms for r in successful if r.itl_ms > 0]
    total_tokens = sum(r.output_tokens for r in successful)
    aggregate_tps = total_tokens / duration_s if duration_s > 0 else 0

    print(f"  ┌─ c={concurrency} ({len(successful)}/{len(results)} ok)")
    print(f"  │  TTFT   P50={percentile(ttfts,50):7.0f}ms  P95={percentile(ttfts,95):7.0f}ms  mean={statistics.mean(ttfts):7.0f}ms")
    print(f"  │  TPS    P50={percentile(tps_vals,50):7.1f}     mean={statistics.mean(tps_vals):7.1f}")
    print(f"  │  ITL    P50={percentile(itls,50):7.1f}ms" if itls else "  │  ITL    N/A")
    print(f"  │  Total  {total_tokens} tokens in {duration_s:.1f}s = {aggregate_tps:.1f} tok/s (aggregate)")
    print(f"  └─")

    return {
        "concurrency": concurrency,
        "num_requests": len(results),
        "num_successful": len(successful),
        "duration_s": round(duration_s, 2),
        "ttft_p50_ms": round(percentile(ttfts, 50), 1),
        "ttft_p95_ms": round(percentile(ttfts, 95), 1),
        "ttft_mean_ms": round(statistics.mean(ttfts), 1),
        "tps_p50": round(percentile(tps_vals, 50), 1),
        "tps_mean": round(statistics.mean(tps_vals), 1),
        "itl_p50_ms": round(percentile(itls, 50), 1) if itls else None,
        "aggregate_tps": round(aggregate_tps, 1),
        "total_tokens": total_tokens,
    }


async def main_async(args: argparse.Namespace) -> None:
    concurrency_levels = [int(c) for c in args.concurrency.split(",")]

    prompts_pool = [
        "Explain how KV cache works in transformer inference. Be concise.",
        "What is the difference between prefill and decode phases in LLM serving?",
        "Describe the benefits of quantization for inference optimization.",
        "How does continuous batching improve GPU utilization in LLM serving?",
        "What is prefix caching and when is it most effective?",
        "Explain PagedAttention and why it matters for LLM inference.",
        "What are the tradeoffs between FP16, INT8, and FP4 quantization?",
        "How does speculative decoding work and when does it help?",
        "What is disaggregated inference and why did NVIDIA build it?",
        "Explain the relationship between TTFT and user experience in chat apps.",
    ]

    print(f"\n{'='*65}")
    print(f"  InferBench Concurrency Benchmark")
    print(f"{'='*65}")
    print(f"  Server:       {args.server}")
    print(f"  Model:        {args.model}")
    print(f"  Concurrency:  {concurrency_levels}")
    print(f"  Requests/lvl: {args.requests_per_level}")
    print(f"  Max tokens:   {args.max_tokens}")
    print(f"{'='*65}")

    # Warmup: 2 sequential requests
    print(f"\n  Warming up (2 sequential requests)...")
    for i in range(2):
        _sync_streaming_request(args.server, args.model, prompts_pool[i], args.max_tokens)
    print(f"  Warmup done.\n")

    all_summaries: list[dict] = []

    for c in concurrency_levels:
        n = args.requests_per_level
        prompts = [prompts_pool[i % len(prompts_pool)] for i in range(n)]

        print(f"  Running c={c}, {n} requests...")
        start = time.perf_counter()
        results = await run_concurrent_requests(
            args.server, args.model, prompts, args.max_tokens, c,
        )
        duration = time.perf_counter() - start

        summary = print_level_summary(c, results, duration)
        if summary:
            all_summaries.append(summary)
        print()

    # Print comparison table
    print(f"{'='*65}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*65}")
    print(f"  {'c':>4}  {'TTFT P50':>10}  {'TPS mean':>10}  {'Agg TPS':>10}  {'ITL P50':>10}")
    print(f"  {'─'*4}  {'─'*10}  {'─'*10}  {'─'*10}  {'─'*10}")
    for s in all_summaries:
        itl_str = f"{s['itl_p50_ms']:.1f} ms" if s.get('itl_p50_ms') else "N/A"
        print(
            f"  {s['concurrency']:>4}  "
            f"{s['ttft_p50_ms']:>7.0f} ms  "
            f"{s['tps_mean']:>7.1f}     "
            f"{s['aggregate_tps']:>7.1f}     "
            f"{itl_str:>10}"
        )
    print(f"{'='*65}")

    # Save JSON
    import os
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "benchmark": "concurrency",
        "timestamp": timestamp,
        "config": {
            "server": args.server,
            "model": args.model,
            "concurrency_levels": concurrency_levels,
            "requests_per_level": args.requests_per_level,
            "max_tokens": args.max_tokens,
        },
        "hardware": {
            "gpu": "NVIDIA RTX 5070 Ti 16GB",
            "cpu": "AMD Ryzen 9 9900X",
            "driver": "595.79",
            "cuda": "13.2",
            "vllm": "0.13.0",
        },
        "levels": all_summaries,
    }
    outfile = f"results/concurrency_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outfile}")


def main() -> None:
    parser = argparse.ArgumentParser(description="InferBench Concurrency Benchmark")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--concurrency", default="1,8,16,32", help="Comma-separated concurrency levels")
    parser.add_argument("--requests-per-level", "-n", type=int, default=16)
    parser.add_argument("--max-tokens", "-t", type=int, default=256, help="Shorter tokens for concurrency test")
    args = parser.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()