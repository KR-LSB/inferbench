#!/usr/bin/env python3
"""
prefix_cache_bench.py — Prefix Caching Benchmark (vLLM vs SGLang)

Tests how effectively each engine caches a shared system context (RAG scenario).
Sends the same long context with different questions across 2 rounds:
  Round 1 = cold cache (first time seeing context)
  Round 2 = warm cache (context should be cached)

A warmup request primes the cache before Round 1.

Usage (PowerShell):
    python prefix_cache_bench.py --engine sglang --port 30000
    python prefix_cache_bench.py --engine vllm --port 8000

Requirements:
    pip install httpx
"""

import asyncio
import json
import time
import argparse
import statistics
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import httpx


# ──────────────────── Shared RAG Context ────────────────────

SYSTEM_CONTEXT = """You are an expert AI assistant. Use the following technical document to answer questions accurately and concisely.

---
DOCUMENT: LLM Inference Optimization — Technical Reference

1. ATTENTION MECHANISMS AND KV CACHE

The Key-Value (KV) cache is a fundamental optimization in transformer inference. During autoregressive generation, each new token attends to all previous tokens. Without caching, this would require recomputing the key and value projections for every previous token at each generation step, resulting in O(n²) compute complexity.

The KV cache stores previously computed key-value pairs, reducing per-step computation to O(n). However, the memory cost is significant: for a model with L layers, H attention heads, and dimension D, the KV cache for a sequence of length S requires 2 × L × H × D × S × sizeof(dtype) bytes.

For Qwen3-8B with 36 layers, 8 KV heads, and dimension 128 per head, a single 8192-token sequence in FP16 requires: 2 × 36 × 8 × 128 × 8192 × 2 bytes = 1.15 GB. This scales linearly with sequence length and batch size, making KV cache the primary memory bottleneck in inference serving.

2. PAGED ATTENTION (vLLM)

PagedAttention, introduced by vLLM, applies virtual memory concepts to KV cache management. Instead of allocating contiguous memory for each sequence, it divides the KV cache into fixed-size blocks (pages) that can be allocated non-contiguously.

Benefits: (a) Eliminates memory fragmentation — pre-allocated contiguous buffers waste 60-80% of memory on average. (b) Enables memory sharing between sequences with common prefixes. (c) Allows dynamic memory allocation as sequences grow.

The block size is typically 16 tokens. Each block contains the KV pairs for 16 consecutive tokens across all layers. The block table maps logical block indices to physical memory locations, similar to a page table in OS virtual memory.

3. RADIX ATTENTION (SGLang)

RadixAttention extends prefix caching using a radix tree (prefix tree) data structure. Unlike PagedAttention's block-level sharing, RadixAttention operates at the token level and can share arbitrary common prefixes across requests.

The radix tree stores the KV cache indexed by token sequences. When a new request arrives, the engine traverses the tree to find the longest matching prefix, reusing cached KV pairs and only computing the novel suffix. This is particularly effective for: (a) Multi-turn conversations sharing system prompts. (b) Few-shot prompting where examples are reused. (c) RAG workloads where retrieved documents are partially shared.

The key advantage over PagedAttention's prefix caching is granularity: RadixAttention can match at any token boundary, while block-level caching only matches at block boundaries (every 16 tokens).

4. CONTINUOUS BATCHING

Static batching groups requests into fixed-size batches and waits for all to complete. This wastes compute on padding and blocks early-finishing requests. Continuous batching (also called iteration-level scheduling) adds and removes requests at each decode step.

When a request finishes, a new request from the queue immediately takes its slot. This maximizes GPU utilization and reduces queuing delays. Both vLLM and SGLang implement continuous batching, but their scheduling policies differ in how they prioritize prefill vs decode operations.

5. CHUNKED PREFILL

Long input sequences can monopolize the GPU during prefill, blocking decode operations for other requests. Chunked prefill splits the prefill computation into smaller chunks (e.g., 512 or 2048 tokens) that are interleaved with decode steps.

This prevents head-of-line blocking: while one request's prefill chunk is being processed, other requests can still make progress on their decode steps. The trade-off is slightly higher prefill latency due to chunking overhead, but significantly better tail latency for concurrent requests.

6. QUANTIZATION METHODS

FP16/BF16: Full precision baseline. 2 bytes per parameter.
INT8 (W8A8): Both weights and activations quantized. ~2x compression with minimal quality loss.
INT4/W4A16 (AWQ, GPTQ): Weights only quantized to 4 bits, activations remain in FP16. ~4x weight compression. AWQ (Activation-aware Weight Quantization) preserves salient weight channels.
FP4 (NVFP4): NVIDIA's native FP4 format for Blackwell GPUs. Hardware-accelerated 4-bit operations on SM_120 tensor cores. Similar compression to INT4 but potentially better quality due to floating-point representation.
MXFP4: Microscaling FP4 with shared exponents per block. Better dynamic range than fixed FP4.

7. SPECULATIVE DECODING

Uses a small draft model to propose multiple tokens in parallel, then the main model verifies them in a single forward pass. If the draft model's predictions match, multiple tokens are generated per step, increasing throughput.

Effective when: (a) The draft model has high acceptance rate (>70%). (b) The main model's single-token latency is high. (c) The workload is latency-sensitive rather than throughput-sensitive. Less effective for batch serving where throughput matters more.

8. DISAGGREGATED INFERENCE

NVIDIA's approach separates prefill (input processing) and decode (token generation) onto different hardware optimized for each phase. Prefill is compute-bound and benefits from high-FLOPS accelerators. Decode is memory-bandwidth-bound and benefits from high-bandwidth memory.

The Vera Rubin platform implements this with Rubin GPUs handling prefill and Groq LPUs handling decode, orchestrated by NVIDIA Dynamo. This separation allows each phase to run at optimal efficiency rather than compromising on shared hardware.

9. PREFIX CACHING ECONOMICS

For RAG workloads with shared system prompts, prefix caching dramatically reduces TTFT. Consider a scenario with 8,000 tokens of shared context:
- Without caching: 8,000 tokens must be processed in prefill for every request (~1,700ms on RTX 5070 Ti)
- With caching: Only novel tokens (question + few tokens) need prefill (~60ms)
- Speedup: ~27x TTFT reduction

The economic impact scales with context length. At 16,000 tokens, the speedup reaches 57x because prefill cost grows super-linearly while cached lookups remain near-constant.

10. SERVING ARCHITECTURE PATTERNS

Production LLM serving systems typically employ: (a) Load balancing across multiple model replicas. (b) Request queuing with priority scheduling. (c) KV cache offloading to CPU/disk for memory pressure relief. (d) Health monitoring with automatic replica restart. (e) A/B testing infrastructure for model version comparison.

The choice between vLLM and SGLang depends on workload characteristics. vLLM excels in stable, high-throughput scenarios with its mature PagedAttention implementation. SGLang's RadixAttention provides advantages in prefix-heavy workloads like multi-turn chat and RAG.
---"""

QUESTIONS = [
    "What is the KV cache and why does it become a memory bottleneck in LLM inference?",
    "How does PagedAttention solve memory fragmentation compared to traditional allocation?",
    "Explain the difference between RadixAttention and PagedAttention for prefix caching.",
    "Why does chunked prefill improve tail latency for concurrent requests?",
    "Compare AWQ and NVFP4 quantization methods for Blackwell GPUs.",
    "When is speculative decoding effective and when is it not?",
    "How does disaggregated inference separate prefill and decode phases?",
    "What is the economic impact of prefix caching on RAG workloads?",
]


# ──────────────────── Data Classes ────────────────────

@dataclass
class SingleRequestResult:
    """Result from a single request."""
    question_idx: int
    round_num: int
    ttft_ms: float
    decode_time_ms: float
    total_time_ms: float
    output_tokens: int
    tps: float
    prefill_ratio: float  # ttft / total


@dataclass
class RoundSummary:
    """Summary for one round."""
    round_num: int
    ttft_p50_ms: float
    ttft_mean_ms: float
    decode_p50_ms: float
    tps_mean: float
    prefill_ratio_mean: float
    num_requests: int


# ──────────────────── Core Benchmark ────────────────────

def percentile(data: list[float], p: float) -> float:
    """Calculate percentile."""
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[f]
    return s[f] + (k - f) * (s[c] - s[f])


async def send_rag_request(
    client: httpx.AsyncClient,
    url: str,
    model: str,
    context: str,
    question: str,
    max_tokens: int = 256,
    enable_thinking: bool = True,
) -> Optional[dict]:
    """Send a RAG request with shared context + unique question."""
    messages = [
        {"role": "system", "content": context},
        {"role": "user", "content": question},
    ]

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": True,
    }

    if not enable_thinking:
        body["chat_template_kwargs"] = {"enable_thinking": False}

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    output_tokens = 0

    try:
        async with client.stream(
            "POST", url, json=body, timeout=180.0,
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
                if delta.get("content", ""):
                    now = time.perf_counter()
                    if first_token_time is None:
                        first_token_time = now
                    output_tokens += 1
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

    end = time.perf_counter()
    if first_token_time is None or output_tokens == 0:
        return None

    ttft_ms = (first_token_time - start) * 1000
    total_ms = (end - start) * 1000
    decode_ms = total_ms - ttft_ms
    tps = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0

    return {
        "ttft_ms": ttft_ms,
        "decode_time_ms": decode_ms,
        "total_time_ms": total_ms,
        "output_tokens": output_tokens,
        "tps": tps,
        "prefill_ratio": ttft_ms / total_ms if total_ms > 0 else 0,
    }


async def run_prefix_cache_bench(
    url: str,
    model: str,
    engine: str,
    max_tokens: int = 256,
    enable_thinking: bool = True,
) -> dict:
    """
    Run the full prefix caching benchmark:
    1. Warmup request (primes cache)
    2. Round 1: 8 questions (cold → warming cache)
    3. Round 2: same 8 questions (warm cache)
    """
    print(f"\n{'#'*60}")
    print(f"  Prefix Caching Benchmark: {engine.upper()}")
    print(f"  URL: {url}")
    print(f"  Context: ~{len(SYSTEM_CONTEXT.split())} words")
    print(f"  Questions: {len(QUESTIONS)}")
    print(f"  Rounds: 2 (cold → warm)")
    print(f"  Thinking: {'ON' if enable_thinking else 'OFF'}")
    print(f"{'#'*60}")

    all_results: list[SingleRequestResult] = []

    async with httpx.AsyncClient() as client:
        # Warmup — prime the cache
        print(f"\n  [Warmup] Priming cache with first request...")
        warmup = await send_rag_request(
            client, url, model, SYSTEM_CONTEXT,
            "Summarize the key concepts in this document.",
            max_tokens, enable_thinking,
        )
        if warmup:
            print(f"    Warmup TTFT: {warmup['ttft_ms']:.0f}ms (cold prefill)")
        else:
            print(f"    Warmup failed!")

        # Round 1 and Round 2
        for round_num in range(1, 3):
            label = "cold cache" if round_num == 1 else "warm cache"
            print(f"\n  [Round {round_num}] {label} — {len(QUESTIONS)} questions")

            for i, question in enumerate(QUESTIONS):
                result = await send_rag_request(
                    client, url, model, SYSTEM_CONTEXT,
                    question, max_tokens, enable_thinking,
                )
                if result:
                    sr = SingleRequestResult(
                        question_idx=i,
                        round_num=round_num,
                        ttft_ms=result["ttft_ms"],
                        decode_time_ms=result["decode_time_ms"],
                        total_time_ms=result["total_time_ms"],
                        output_tokens=result["output_tokens"],
                        tps=result["tps"],
                        prefill_ratio=result["prefill_ratio"],
                    )
                    all_results.append(sr)
                    print(
                        f"    Q{i+1}: TTFT={result['ttft_ms']:>7.0f}ms | "
                        f"Decode={result['decode_time_ms']:>7.0f}ms | "
                        f"TPS={result['tps']:>6.1f} | "
                        f"Prefill={result['prefill_ratio']*100:>4.1f}%"
                    )
                else:
                    print(f"    Q{i+1}: FAILED")

    # ── Summaries ──
    def summarize_round(rnd: int) -> RoundSummary:
        rr = [r for r in all_results if r.round_num == rnd]
        if not rr:
            return RoundSummary(rnd, 0, 0, 0, 0, 0, 0)
        return RoundSummary(
            round_num=rnd,
            ttft_p50_ms=round(percentile([r.ttft_ms for r in rr], 50), 1),
            ttft_mean_ms=round(statistics.mean([r.ttft_ms for r in rr]), 1),
            decode_p50_ms=round(percentile([r.decode_time_ms for r in rr], 50), 1),
            tps_mean=round(statistics.mean([r.tps for r in rr]), 1),
            prefill_ratio_mean=round(statistics.mean([r.prefill_ratio for r in rr]) * 100, 1),
            num_requests=len(rr),
        )

    r1 = summarize_round(1)
    r2 = summarize_round(2)
    r_all = [r for r in all_results]
    all_summary = RoundSummary(
        round_num=0,
        ttft_p50_ms=round(percentile([r.ttft_ms for r in r_all], 50), 1),
        ttft_mean_ms=round(statistics.mean([r.ttft_ms for r in r_all]), 1),
        decode_p50_ms=round(percentile([r.decode_time_ms for r in r_all], 50), 1),
        tps_mean=round(statistics.mean([r.tps for r in r_all]), 1),
        prefill_ratio_mean=round(statistics.mean([r.prefill_ratio for r in r_all]) * 100, 1),
        num_requests=len(r_all),
    )

    print(f"\n{'='*70}")
    print(f"  SUMMARY — {engine.upper()} Prefix Caching")
    print(f"{'='*70}")
    print(f"  {'':>10} | {'TTFT P50':>10} | {'TTFT mean':>10} | {'TPS mean':>10} | {'Prefill %':>10}")
    print(f"  {'-'*60}")
    print(f"  {'Round 1':>10} | {r1.ttft_p50_ms:>8.1f}ms | {r1.ttft_mean_ms:>8.1f}ms | {r1.tps_mean:>8.1f}  | {r1.prefill_ratio_mean:>8.1f}%")
    print(f"  {'Round 2':>10} | {r2.ttft_p50_ms:>8.1f}ms | {r2.ttft_mean_ms:>8.1f}ms | {r2.tps_mean:>8.1f}  | {r2.prefill_ratio_mean:>8.1f}%")
    print(f"  {'All':>10} | {all_summary.ttft_p50_ms:>8.1f}ms | {all_summary.ttft_mean_ms:>8.1f}ms | {all_summary.tps_mean:>8.1f}  | {all_summary.prefill_ratio_mean:>8.1f}%")

    warmup_ttft = warmup["ttft_ms"] if warmup else 0

    return {
        "engine": engine,
        "model": model,
        "thinking": enable_thinking,
        "max_tokens": max_tokens,
        "context_words": len(SYSTEM_CONTEXT.split()),
        "warmup_ttft_ms": round(warmup_ttft, 1),
        "round_1": asdict(r1),
        "round_2": asdict(r2),
        "all": asdict(all_summary),
        "raw_results": [asdict(r) for r in all_results],
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
    }


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefix Caching Benchmark (vLLM vs SGLang)"
    )
    parser.add_argument("--engine", type=str, required=True, choices=["vllm", "sglang"])
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    url = f"http://localhost:{args.port}/v1/chat/completions"
    enable_thinking = not args.no_thinking

    output = await run_prefix_cache_bench(
        url=url,
        model=args.model,
        engine=args.engine,
        max_tokens=args.max_tokens,
        enable_thinking=enable_thinking,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    thinking_tag = "think_on" if enable_thinking else "think_off"
    filename = f"prefix_cache_{args.engine}_{thinking_tag}_{output['timestamp']}.json"
    filepath = output_dir / filename

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved to: {filepath}")


if __name__ == "__main__":
    asyncio.run(main())