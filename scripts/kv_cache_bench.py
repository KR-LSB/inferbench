"""
Experiment 3: KV Cache Economics
Measures how context length impacts inference performance on RTX 5070 Ti.

Hypothesis: Doubling context length roughly halves throughput and doubles TTFT.
Prefix caching dramatically reduces this cost for repeated-context RAG workloads.

Usage (PowerShell, with vLLM server running):
    python scripts/kv_cache_bench.py --server http://localhost:8000
    python scripts/kv_cache_bench.py --server http://localhost:8000 --output results/kv_cache_results.json
"""

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import httpx


# ---------------------------------------------------------------------------
# Context generator — creates prompts of target token lengths
# ---------------------------------------------------------------------------

# ~1.3 tokens per word for English text (rough estimate)
TOKENS_PER_WORD = 1.3

# Base technical passage (~500 words ≈ 650 tokens), repeated to reach target
BASE_PASSAGE = (
    "Large language models use the transformer architecture with self-attention "
    "mechanisms that compute relationships between all tokens in a sequence. "
    "During inference, the key-value cache stores intermediate attention states "
    "to avoid redundant computation. As context length grows, KV cache memory "
    "consumption increases linearly, directly competing with model weights for "
    "GPU VRAM. On a 16GB consumer GPU like the RTX 5070 Ti, this creates a "
    "fundamental tradeoff between context capacity and batch size. "
    "The prefill phase processes all input tokens in parallel, leveraging the "
    "GPU's compute throughput. The decode phase generates tokens one at a time, "
    "bottlenecked by memory bandwidth. Prefix caching allows the KV cache from "
    "shared context prefixes to be reused across requests, eliminating redundant "
    "prefill computation. This is especially valuable for RAG workloads where "
    "the same retrieved documents are queried with different questions. "
    "NVIDIA's disaggregated inference architecture, introduced at GTC 2026, "
    "separates prefill and decode onto different hardware optimized for each "
    "phase. Rubin GPUs handle compute-heavy prefill while Groq LPUs handle "
    "memory-bandwidth-bound decode. Understanding prefill vs decode costs at "
    "each context length is essential for capacity planning. "
    "Quantization reduces model size but does not reduce KV cache size, which "
    "is always stored in the model's compute precision. An 8B parameter model "
    "quantized to 4-bit still generates 16-bit KV cache entries. This means "
    "KV cache becomes the dominant memory consumer at longer contexts. "
    "Continuous batching in vLLM dynamically schedules requests, but when KV "
    "cache memory is exhausted, new requests must wait. Monitoring KV cache "
    "utilization is critical for maintaining low tail latency under load. "
)

# Questions to ask about the context (cycling through)
QUESTIONS = [
    "What is the main tradeoff when running long-context inference on a 16GB GPU?",
    "How does prefix caching help with RAG workloads?",
    "Why does NVIDIA separate prefill and decode in disaggregated inference?",
    "What happens to KV cache memory as context length doubles?",
    "Why does quantization not reduce KV cache size?",
    "How does continuous batching handle KV cache exhaustion?",
    "What is the difference between prefill and decode phases?",
    "Why is memory bandwidth the bottleneck during decode?",
]


def build_prompt(target_tokens: int, question_idx: int = 0) -> str:
    """Build a prompt with approximately target_tokens of context + a question.

    The context is a repeated technical passage. The question is appended at the end.
    This simulates a RAG workload: long context + short query.
    """
    question = QUESTIONS[question_idx % len(QUESTIONS)]
    question_part = f"\n\nBased on the above context, answer concisely: {question}"
    question_tokens = int(len(question_part.split()) * TOKENS_PER_WORD)

    context_tokens_needed = target_tokens - question_tokens
    if context_tokens_needed < 100:
        context_tokens_needed = 100

    words_needed = int(context_tokens_needed / TOKENS_PER_WORD)
    base_words = BASE_PASSAGE.split()

    # Repeat base passage to reach target length
    repeated = []
    while len(repeated) < words_needed:
        repeated.extend(base_words)
    context = " ".join(repeated[:words_needed])

    return context + question_part


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class KVCacheMetrics:
    """Metrics for a single request in the KV Cache Economics experiment."""
    target_context_tokens: int
    actual_prompt_words: int
    estimated_prompt_tokens: int
    max_output_tokens: int
    output_tokens: int
    ttft_ms: float
    decode_time_ms: float
    total_time_ms: float
    tps: float
    prefill_ratio: float  # ttft / total_time
    prefix_caching: bool
    question_idx: int


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

async def run_single_request(
    client: httpx.AsyncClient,
    server_url: str,
    prompt: str,
    max_tokens: int,
    target_tokens: int,
    prefix_caching: bool,
    question_idx: int,
) -> KVCacheMetrics:
    """Run a single streaming request and measure TTFT + decode separately."""

    start = time.perf_counter()
    first_token_time: Optional[float] = None
    output_tokens = 0

    async with client.stream(
        "POST",
        f"{server_url}/v1/chat/completions",
        json={
            "model": "Qwen/Qwen3-8B-AWQ",
            "messages": [
                {"role": "user", "content": prompt},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7,
            "stream": True,
        },
        timeout=300.0,
    ) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data: ") and line != "data: [DONE]":
                if first_token_time is None:
                    first_token_time = time.perf_counter()
                output_tokens += 1

    end = time.perf_counter()

    ttft_ms = (first_token_time - start) * 1000 if first_token_time else 0
    total_ms = (end - start) * 1000
    decode_ms = total_ms - ttft_ms

    prompt_words = len(prompt.split())
    est_tokens = int(prompt_words * TOKENS_PER_WORD)

    return KVCacheMetrics(
        target_context_tokens=target_tokens,
        actual_prompt_words=prompt_words,
        estimated_prompt_tokens=est_tokens,
        max_output_tokens=max_tokens,
        output_tokens=output_tokens,
        ttft_ms=round(ttft_ms, 1),
        decode_time_ms=round(decode_ms, 1),
        total_time_ms=round(total_ms, 1),
        tps=round(output_tokens / (decode_ms / 1000), 1) if decode_ms > 0 else 0,
        prefill_ratio=round(ttft_ms / total_ms * 100, 1) if total_ms > 0 else 0,
        prefix_caching=prefix_caching,
        question_idx=question_idx,
    )


async def run_context_length_sweep(
    server_url: str,
    context_lengths: list[int],
    max_tokens: int,
    num_requests: int,
    prefix_caching: bool,
) -> list[KVCacheMetrics]:
    """Run benchmark across multiple context lengths.

    For each context length:
    - Send 1 warmup request (populates prefix cache if enabled)
    - Send num_requests measurement requests with different questions
    """
    all_results = []

    async with httpx.AsyncClient() as client:
        # Verify server is up
        try:
            resp = await client.get(f"{server_url}/v1/models", timeout=10.0)
            resp.raise_for_status()
            models = resp.json()
            model_id = models["data"][0]["id"] if models.get("data") else "unknown"
            print(f"Connected to server. Model: {model_id}")
        except Exception as e:
            print(f"ERROR: Cannot connect to {server_url}: {e}")
            return []

        for ctx_len in context_lengths:
            print(f"\n{'='*60}")
            print(f"Context: ~{ctx_len} tokens | Prefix Caching: {'ON' if prefix_caching else 'OFF'}")
            print(f"{'='*60}")

            # Warmup — populates prefix cache, not counted in results
            warmup_prompt = build_prompt(ctx_len, question_idx=0)
            print(f"  Warmup ({len(warmup_prompt.split())} words)...", end=" ", flush=True)
            warmup = await run_single_request(
                client, server_url, warmup_prompt, max_tokens,
                ctx_len, prefix_caching, question_idx=0,
            )
            print(f"TTFT={warmup.ttft_ms:.0f}ms, TPS={warmup.tps:.1f}")

            # Measurement requests — different questions, same context
            results_for_length = []
            for i in range(num_requests):
                q_idx = (i + 1) % len(QUESTIONS)  # Skip question 0 (used in warmup)
                prompt = build_prompt(ctx_len, question_idx=q_idx)

                result = await run_single_request(
                    client, server_url, prompt, max_tokens,
                    ctx_len, prefix_caching, question_idx=q_idx,
                )
                results_for_length.append(result)

                print(
                    f"  [{i+1}/{num_requests}] "
                    f"TTFT={result.ttft_ms:>8.1f}ms | "
                    f"TPS={result.tps:>6.1f} | "
                    f"Prefill={result.prefill_ratio:>5.1f}% | "
                    f"Total={result.total_time_ms:>8.1f}ms"
                )

            all_results.extend(results_for_length)

            # Summary for this context length
            ttfts = [r.ttft_ms for r in results_for_length]
            tpss = [r.tps for r in results_for_length]
            prefill_ratios = [r.prefill_ratio for r in results_for_length]
            totals = [r.total_time_ms for r in results_for_length]

            print(f"\n  Summary (~{ctx_len} tokens):")
            print(f"    TTFT    — mean: {sum(ttfts)/len(ttfts):>8.1f}ms, "
                  f"P50: {sorted(ttfts)[len(ttfts)//2]:>8.1f}ms")
            print(f"    TPS     — mean: {sum(tpss)/len(tpss):>6.1f}")
            print(f"    Prefill — mean: {sum(prefill_ratios)/len(prefill_ratios):>5.1f}%")
            print(f"    Total   — mean: {sum(totals)/len(totals):>8.1f}ms")

    return all_results


def print_comparison_table(results: list[KVCacheMetrics], context_lengths: list[int]):
    """Print a summary comparison table across all context lengths."""
    print(f"\n{'='*80}")
    print("SUMMARY: KV Cache Economics")
    print(f"{'='*80}")

    for caching in [True, False]:
        label = "Prefix Caching ON" if caching else "Prefix Caching OFF"
        print(f"\n  {label}:")
        print(f"  {'Context':>10} | {'TTFT mean':>10} | {'TPS mean':>10} | {'Prefill %':>10} | {'Total mean':>12}")
        print(f"  {'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*10}-+-{'-'*12}")

        for ctx_len in context_lengths:
            subset = [r for r in results
                      if r.target_context_tokens == ctx_len and r.prefix_caching == caching]
            if not subset:
                continue

            ttft_mean = sum(r.ttft_ms for r in subset) / len(subset)
            tps_mean = sum(r.tps for r in subset) / len(subset)
            prefill_mean = sum(r.prefill_ratio for r in subset) / len(subset)
            total_mean = sum(r.total_time_ms for r in subset) / len(subset)

            print(
                f"  {f'~{ctx_len}':>10} | "
                f"{ttft_mean:>8.1f}ms | "
                f"{tps_mean:>8.1f} | "
                f"{prefill_mean:>8.1f}% | "
                f"{total_mean:>10.1f}ms"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main():
    parser = argparse.ArgumentParser(description="Experiment 3: KV Cache Economics")
    parser.add_argument("--server", default="http://localhost:8000", help="vLLM server URL")
    parser.add_argument("--output", default=None, help="Output JSON file path")
    parser.add_argument("--max-tokens", type=int, default=256, help="Max output tokens")
    parser.add_argument("--num-requests", type=int, default=5, help="Requests per context length")
    parser.add_argument(
        "--context-lengths",
        nargs="+",
        type=int,
        default=[2048, 4096, 8192, 16384],
        help="Target context lengths in tokens",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("Experiment 3: KV Cache Economics")
    print(f"Server: {args.server}")
    print(f"Context lengths: {args.context_lengths}")
    print(f"Max output tokens: {args.max_tokens}")
    print(f"Requests per context length: {args.num_requests}")
    print("=" * 60)

    all_results = []

    # NOTE: Prefix caching ON/OFF requires restarting vLLM with different flags.
    # This script measures whichever mode the server is currently running.
    # Run twice (once with caching, once without) and merge results.

    # Detect if prefix caching is likely on (we'll label based on user input)
    print("\nIs prefix caching enabled on the current server?")
    print("  (Check your docker run command for --no-enable-prefix-caching)")
    caching_input = input("  Prefix caching ON? [Y/n]: ").strip().lower()
    prefix_caching = caching_input != "n"

    results = await run_context_length_sweep(
        server_url=args.server,
        context_lengths=args.context_lengths,
        max_tokens=args.max_tokens,
        num_requests=args.num_requests,
        prefix_caching=prefix_caching,
    )
    all_results.extend(results)

    # Print summary
    print_comparison_table(all_results, args.context_lengths)

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cache_label = "cache_on" if prefix_caching else "cache_off"
        output_path = Path(f"results/kv_cache_{cache_label}_{timestamp}.json")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "experiment": "kv_cache_economics",
        "prefix_caching": prefix_caching,
        "context_lengths": args.context_lengths,
        "max_output_tokens": args.max_tokens,
        "num_requests_per_length": args.num_requests,
        "results": [asdict(r) for r in all_results],
    }

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())