"""InferBench Quick Benchmark — Standalone Script

Measures TTFT, TPS, ITL, and Prefill/Decode breakdown via streaming API.
No external dependencies required (uses only stdlib + urllib).

Usage (new PowerShell window, while vLLM server is running):
    python scripts/quick_bench.py
    python scripts/quick_bench.py --num-requests 10 --max-tokens 512
"""

from __future__ import annotations

import argparse
import http.client
import json
import statistics
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

    @property
    def prefill_ratio(self) -> float:
        return self.ttft_ms / self.total_time_ms if self.total_time_ms > 0 else 0.0


def measure_streaming_request(
    server_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> RequestResult:
    """Send a streaming chat completion request and measure timing.

    Uses only stdlib http.client for zero-dependency execution.
    """
    parsed = urlparse(server_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=120)

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


def percentile(data: list[float], p: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (k - f) * (s[c] - s[f])


def main() -> None:
    parser = argparse.ArgumentParser(description="InferBench Quick Benchmark")
    parser.add_argument("--server", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ", help="Model name")
    parser.add_argument("--num-requests", "-n", type=int, default=5, help="Number of requests")
    parser.add_argument("--max-tokens", "-t", type=int, default=512, help="Max tokens per request")
    args = parser.parse_args()

    prompts = [
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

    print(f"\n{'='*60}")
    print(f"  InferBench Quick Benchmark")
    print(f"{'='*60}")
    print(f"  Server:   {args.server}")
    print(f"  Model:    {args.model}")
    print(f"  Requests: {args.num_requests}")
    print(f"  Max Tok:  {args.max_tokens}")
    print(f"{'='*60}\n")

    results: list[RequestResult] = []
    run_start = time.perf_counter()

    for i in range(args.num_requests):
        prompt = prompts[i % len(prompts)]
        short_prompt = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"  [{i+1}/{args.num_requests}] {short_prompt}")

        try:
            r = measure_streaming_request(
                server_url=args.server,
                model=args.model,
                prompt=prompt,
                max_tokens=args.max_tokens,
            )
            results.append(r)
            print(
                f"           TTFT={r.ttft_ms:7.0f}ms | "
                f"TPS={r.tps:6.1f} | "
                f"tokens={r.output_tokens:4d} | "
                f"total={r.total_time_ms:7.0f}ms | "
                f"prefill={r.prefill_ratio:4.1%}"
            )
        except Exception as e:
            print(f"           FAILED: {e}")

    run_end = time.perf_counter()
    total_duration = run_end - run_start

    if not results:
        print("\nNo successful requests. Check server connection.")
        return

    # Aggregate
    ttfts = [r.ttft_ms for r in results]
    tps_vals = [r.tps for r in results]
    itls = [r.itl_ms for r in results if r.itl_ms > 0]
    e2es = [r.total_time_ms for r in results]
    total_tokens = sum(r.output_tokens for r in results)

    print(f"\n{'='*60}")
    print(f"  RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"  Successful requests:  {len(results)}/{args.num_requests}")
    print(f"  Total duration:       {total_duration:.1f}s")
    print(f"  Total output tokens:  {total_tokens}")
    print(f"  Throughput:           {total_tokens/total_duration:.1f} tok/s (aggregate)")
    print(f"{'─'*60}")
    print(f"  TTFT (Time to First Token):")
    print(f"    P50:  {percentile(ttfts, 50):8.0f} ms")
    print(f"    P95:  {percentile(ttfts, 95):8.0f} ms")
    print(f"    P99:  {percentile(ttfts, 99):8.0f} ms")
    print(f"    Mean: {statistics.mean(ttfts):8.0f} ms")
    print(f"{'─'*60}")
    print(f"  TPS (Tokens Per Second, per request):")
    print(f"    P50:  {percentile(tps_vals, 50):8.1f} tok/s")
    print(f"    Mean: {statistics.mean(tps_vals):8.1f} tok/s")
    print(f"{'─'*60}")
    if itls:
        print(f"  ITL (Inter-Token Latency):")
        print(f"    P50:  {percentile(itls, 50):8.1f} ms")
        print(f"    P95:  {percentile(itls, 95):8.1f} ms")
        print(f"{'─'*60}")
    print(f"  E2E Latency:")
    print(f"    P50:  {percentile(e2es, 50):8.0f} ms")
    print(f"    P95:  {percentile(e2es, 95):8.0f} ms")
    print(f"{'─'*60}")
    print(f"  Prefill/Decode Breakdown (mean):")
    print(f"    Prefill ratio: {statistics.mean([r.prefill_ratio for r in results]):.1%}")
    print(f"    TTFT (≈prefill): {statistics.mean(ttfts):.0f} ms")
    print(f"    Decode time:     {statistics.mean([r.decode_time_ms for r in results]):.0f} ms")
    print(f"{'='*60}")

    # Save JSON
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "benchmark": "quick",
        "timestamp": timestamp,
        "config": {
            "server": args.server,
            "model": args.model,
            "num_requests": args.num_requests,
            "max_tokens": args.max_tokens,
        },
        "hardware": {
            "gpu": "NVIDIA RTX 5070 Ti 16GB",
            "cpu": "AMD Ryzen 9 9900X",
            "driver": "595.79",
            "cuda": "13.2",
            "vllm": "0.13.0",
        },
        "summary": {
            "ttft_p50_ms": round(percentile(ttfts, 50), 1),
            "ttft_p95_ms": round(percentile(ttfts, 95), 1),
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "tps_p50": round(percentile(tps_vals, 50), 1),
            "tps_mean": round(statistics.mean(tps_vals), 1),
            "itl_p50_ms": round(percentile(itls, 50), 1) if itls else None,
            "e2e_p50_ms": round(percentile(e2es, 50), 1),
            "total_tokens": total_tokens,
            "aggregate_tps": round(total_tokens / total_duration, 1),
            "prefill_ratio_mean": round(statistics.mean([r.prefill_ratio for r in results]), 3),
        },
        "per_request": [
            {
                "ttft_ms": round(r.ttft_ms, 1),
                "total_time_ms": round(r.total_time_ms, 1),
                "output_tokens": r.output_tokens,
                "tps": round(r.tps, 1),
                "itl_ms": round(r.itl_ms, 1),
                "prefill_ratio": round(r.prefill_ratio, 3),
            }
            for r in results
        ],
    }

    import os
    os.makedirs("results", exist_ok=True)
    outfile = f"results/quick_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outfile}")


if __name__ == "__main__":
    main()