#!/usr/bin/env python3
"""
Experiment 5A: EXAONE-Deep-7.8B Concurrency Benchmark on RTX 5070 Ti

Mirrors engine_bench.py but adapted for EXAONE-Deep-7.8B-AWQ:
- No system prompt (EXAONE recommendation: instruction in user message)
- <thought> reasoning token handling
- Comparison-ready output format vs Qwen3-8B-AWQ baseline

Usage (PowerShell):
  # vLLM (port 8000)
  python scripts/exaone_bench.py --port 8000 --engine vllm
  # SGLang (port 30000)
  python scripts/exaone_bench.py --port 30000 --engine sglang
  # Skip warmup (debugging)
  python scripts/exaone_bench.py --port 8000 --engine vllm --no-warmup
"""

import asyncio
import argparse
import json
import time
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_NAME = "LGAI-EXAONE/EXAONE-Deep-7.8B-AWQ"
CONCURRENCY_LEVELS = [1, 8, 16, 32]
MAX_TOKENS = 256
WARMUP_REQUESTS = 3
REQUESTS_PER_LEVEL = 20  # n=20 per concurrency level (same as engine_bench)

# EXAONE: no system prompt. Instruction goes in user message.
PROMPT = (
    "Explain the key differences between Transformer encoder and decoder "
    "architectures. Cover attention mechanisms, use cases, and computational "
    "trade-offs. Be concise but thorough."
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Result of a single inference request."""
    ttft_ms: float
    total_time_ms: float
    output_tokens: int
    decode_tps: float
    prefill_ratio: float  # ttft / total_time
    error: Optional[str] = None


@dataclass
class ConcurrencyResult:
    """Aggregated result for one concurrency level."""
    concurrency: int
    engine: str
    model: str
    n_requests: int
    agg_tps: float
    per_request_tps_mean: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    total_time_p50_ms: float
    decode_tps_mean: float
    prefill_ratio_mean: float
    errors: int


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    url: str,
    engine: str,
) -> RequestResult:
    """Send a single streaming chat completion request and measure metrics."""
    messages = [{"role": "user", "content": PROMPT}]

    body: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "stream": True,
    }

    # EXAONE repetition_penalty <= 1.0 recommended
    if engine == "vllm":
        body["repetition_penalty"] = 1.0
    elif engine == "sglang":
        body["repetition_penalty"] = 1.0

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    output_tokens = 0

    try:
        async with client.stream("POST", url, json=body, timeout=120.0) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                        output_tokens += 1
                except json.JSONDecodeError:
                    continue

        end = time.perf_counter()

        if first_token_time is None:
            return RequestResult(0, 0, 0, 0, 0, error="No tokens received")

        ttft_ms = (first_token_time - start) * 1000
        total_ms = (end - start) * 1000
        decode_ms = total_ms - ttft_ms
        decode_tps = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
        prefill_ratio = ttft_ms / total_ms if total_ms > 0 else 0

        return RequestResult(
            ttft_ms=round(ttft_ms, 1),
            total_time_ms=round(total_ms, 1),
            output_tokens=output_tokens,
            decode_tps=round(decode_tps, 1),
            prefill_ratio=round(prefill_ratio, 4),
        )

    except Exception as e:
        return RequestResult(0, 0, 0, 0, 0, error=str(e))


async def run_concurrency_level(
    concurrency: int,
    engine: str,
    port: int,
) -> ConcurrencyResult:
    """Run n=REQUESTS_PER_LEVEL requests at a given concurrency level."""
    url = f"http://localhost:{port}/v1/chat/completions"
    semaphore = asyncio.Semaphore(concurrency)

    async def bounded_request(client: httpx.AsyncClient) -> RequestResult:
        async with semaphore:
            return await send_request(client, url, engine)

    async with httpx.AsyncClient() as client:
        tasks = [bounded_request(client) for _ in range(REQUESTS_PER_LEVEL)]
        wall_start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        wall_end = time.perf_counter()

    wall_time = wall_end - wall_start

    # Filter successful results
    ok = [r for r in results if r.error is None]
    errors = len(results) - len(ok)

    if not ok:
        return ConcurrencyResult(
            concurrency=concurrency, engine=engine, model=MODEL_NAME,
            n_requests=REQUESTS_PER_LEVEL, agg_tps=0,
            per_request_tps_mean=0, ttft_p50_ms=0, ttft_p95_ms=0,
            ttft_p99_ms=0, total_time_p50_ms=0, decode_tps_mean=0,
            prefill_ratio_mean=0, errors=errors,
        )

    total_tokens = sum(r.output_tokens for r in ok)
    agg_tps = total_tokens / wall_time

    ttfts = sorted(r.ttft_ms for r in ok)
    totals = sorted(r.total_time_ms for r in ok)

    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    return ConcurrencyResult(
        concurrency=concurrency,
        engine=engine,
        model=MODEL_NAME,
        n_requests=REQUESTS_PER_LEVEL,
        agg_tps=round(agg_tps, 1),
        per_request_tps_mean=round(statistics.mean(r.decode_tps for r in ok), 1),
        ttft_p50_ms=round(percentile(ttfts, 50), 1),
        ttft_p95_ms=round(percentile(ttfts, 95), 1),
        ttft_p99_ms=round(percentile(ttfts, 99), 1),
        total_time_p50_ms=round(percentile(totals, 50), 1),
        decode_tps_mean=round(statistics.mean(r.decode_tps for r in ok), 1),
        prefill_ratio_mean=round(statistics.mean(r.prefill_ratio for r in ok), 4),
        errors=errors,
    )


async def warmup(port: int, engine: str) -> None:
    """Send warmup requests to stabilize the model server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "Hello, respond briefly."}],
        "max_tokens": 32,
        "temperature": 0.6,
        "stream": False,
    }
    if engine in ("vllm", "sglang"):
        body["repetition_penalty"] = 1.0

    print(f"  Warming up ({WARMUP_REQUESTS} requests)...", flush=True)
    async with httpx.AsyncClient() as client:
        for i in range(WARMUP_REQUESTS):
            try:
                resp = await client.post(url, json=body, timeout=60.0)
                resp.raise_for_status()
                print(f"    Warmup {i+1}/{WARMUP_REQUESTS} OK", flush=True)
            except Exception as e:
                print(f"    Warmup {i+1}/{WARMUP_REQUESTS} FAILED: {e}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5A: EXAONE-Deep-7.8B concurrency benchmark"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument(
        "--concurrency", type=str, default=None,
        help="Comma-separated concurrency levels (default: 1,8,16,32)"
    )
    args = parser.parse_args()

    levels = (
        [int(x) for x in args.concurrency.split(",")]
        if args.concurrency
        else CONCURRENCY_LEVELS
    )

    print("=" * 70)
    print(f"  EXAONE-Deep-7.8B Benchmark — {args.engine.upper()} (port {args.port})")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Concurrency levels: {levels}")
    print(f"  Requests per level: {REQUESTS_PER_LEVEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print("=" * 70)

    if not args.no_warmup:
        await warmup(args.port, args.engine)

    all_results: list[ConcurrencyResult] = []

    for c in levels:
        print(f"\n--- Concurrency = {c} ---", flush=True)
        result = await run_concurrency_level(c, args.engine, args.port)
        all_results.append(result)
        print(
            f"  Agg TPS: {result.agg_tps:>8.1f} | "
            f"TTFT P50: {result.ttft_p50_ms:>7.1f} ms | "
            f"P95: {result.ttft_p95_ms:>7.1f} ms | "
            f"Decode TPS: {result.decode_tps_mean:>7.1f} | "
            f"Errors: {result.errors}"
        )

    # Print summary table
    print("\n" + "=" * 70)
    print("  SUMMARY: EXAONE-Deep-7.8B-AWQ")
    print("=" * 70)
    print(
        f"  {'c':>3} | {'Agg TPS':>9} | {'TTFT P50':>9} | "
        f"{'TTFT P95':>9} | {'Decode TPS':>10} | {'Prefill%':>8} | {'Err':>3}"
    )
    print("-" * 70)
    for r in all_results:
        print(
            f"  {r.concurrency:>3} | {r.agg_tps:>9.1f} | "
            f"{r.ttft_p50_ms:>7.1f} ms | {r.ttft_p95_ms:>7.1f} ms | "
            f"{r.decode_tps_mean:>10.1f} | {r.prefill_ratio_mean:>7.2%} | {r.errors:>3}"
        )

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exaone_bench_{args.engine}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "5A_exaone_concurrency",
                "engine": args.engine,
                "model": MODEL_NAME,
                "max_tokens": MAX_TOKENS,
                "requests_per_level": REQUESTS_PER_LEVEL,
                "results": [asdict(r) for r in all_results],
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
