"""InferBench Experiment 2: Prefill/Decode Disaggregation + Prefix Caching

Measures Prefill vs Decode breakdown on RAG-8k workloads, and compares
Prefix Caching ON vs OFF to quantify TTFT reduction from cache hits.

This is the core differentiator of InferBench vs arXiv:2601.09527,
which only measured E2E latency.

Key hypothesis:
  - RAG workloads (long input + short output) are prefill-dominated (>80%)
  - Prefix Caching reduces TTFT by 50%+ for repeated system prompts

Usage:
    # Step 1: Start vLLM WITH prefix caching (default in v0.13.0)
    # Step 2: Run this script
    python scripts/prefill_decode_bench.py --model Qwen/Qwen3-8B-AWQ

    # Step 3: Restart vLLM WITHOUT prefix caching
    #   Add --no-enable-prefix-caching to vLLM args
    # Step 4: Run again with --label no-cache
    python scripts/prefill_decode_bench.py --model Qwen/Qwen3-8B-AWQ --label no-cache
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import statistics
import time
from dataclasses import dataclass
from urllib.parse import urlparse


# ~8k tokens of context (approx 6000 words ≈ 8000 tokens)
SYSTEM_PROMPT = """You are a technical AI assistant specialized in LLM inference optimization.
Based on the provided technical context, answer the user's question accurately and concisely.
Always cite specific details from the context. If the information is not in the context, say so."""

# Generate a long technical context (~7500 tokens worth of text)
def _generate_rag_context(target_words: int = 5800) -> str:
    """Generate a synthetic RAG context of approximately target_words words.

    Simulates a technical document about LLM inference optimization.
    The content is repetitive by design — this tests KV cache efficiency.
    """
    sections = [
        """LLM INFERENCE OPTIMIZATION: A COMPREHENSIVE TECHNICAL REPORT
Version: 2026-Q1 | Authors: InferBench Research

1. INTRODUCTION TO INFERENCE OPTIMIZATION

Large Language Model inference optimization has become one of the most critical challenges
in deploying AI systems at scale. As models grow larger and demand increases, the cost of
serving these models has become a dominant factor in the economics of AI applications.
The key insight driving modern inference optimization is that the inference workload can
be decomposed into two fundamentally different computational phases: the prefill phase
and the decode phase. Understanding this decomposition is essential for designing efficient
serving systems. The prefill phase processes all input tokens in parallel, making it
compute-bound on modern GPU hardware. In contrast, the decode phase generates tokens
one at a time autoregressively, making it memory-bandwidth-bound. This fundamental
asymmetry motivates the disaggregated inference approach championed by NVIDIA at GTC 2026,
where different hardware is optimized for each phase.""",

        """2. KV CACHE MANAGEMENT AND OPTIMIZATION

The Key-Value cache is the central data structure in transformer inference. During the
attention computation, each token produces key and value vectors that must be stored for
use by all subsequent tokens. For a model with L layers, H attention heads, and dimension
D per head, the KV cache for a single sequence of length S requires 2 * L * H * D * S
elements. For a typical 8B parameter model with 32 layers, 8 KV heads, and dimension 128,
this amounts to 32 * 2 * 8 * 128 * S = 65,536 * S bytes in FP16.

For a sequence of 8,192 tokens, this is approximately 512MB of KV cache per sequence.
On a 16GB GPU like the RTX 5070 Ti, this severely limits the number of concurrent
sequences that can be served. PagedAttention, introduced by vLLM, addresses the memory
fragmentation problem by managing KV cache in fixed-size blocks (pages), similar to
virtual memory in operating systems. This allows non-contiguous memory allocation and
eliminates the need to pre-allocate maximum sequence length memory for each request.

Prefix caching is another critical optimization for RAG workloads. When multiple requests
share a common prefix (such as a system prompt or a retrieved document), the KV cache
for that prefix can be computed once and reused across all requests. This is particularly
effective when the shared prefix is long relative to the unique suffix. For example,
in a RAG application where the retrieved context is 8,000 tokens and the user question
is 50 tokens, prefix caching can eliminate 99.4 percent of redundant prefill computation
for subsequent requests with the same context.""",

        """3. QUANTIZATION FOR INFERENCE

Quantization reduces the numerical precision of model weights and activations to decrease
memory footprint and increase computational throughput. The key quantization formats for
inference are as follows.

BF16 (Brain Float 16): The baseline format for most modern models. Uses 16 bits per
parameter, providing a good balance between precision and memory efficiency. An 8B
parameter model requires approximately 16GB in BF16, which barely fits on a 16GB GPU
with minimal KV cache headroom.

AWQ (Activation-Aware Weight Quantization): Quantizes weights to 4-bit integers while
keeping activations in 16-bit format (W4A16). AWQ identifies and preserves salient
weight channels that contribute most to model quality. The awq_marlin kernel in vLLM
provides highly optimized matrix multiplication for this format, achieving near-FP16
throughput on many workloads. An 8B model in AWQ requires approximately 5GB of memory.

NVFP4 (NVIDIA FP4): A 4-bit floating point format native to the Blackwell architecture
(SM_120 and above). Unlike integer quantization, FP4 uses a floating point representation
with 1 sign bit, 2 exponent bits, and 1 mantissa bit. This provides better dynamic range
than INT4 for the same bit width. NVFP4 quantizes both weights and activations to 4-bit,
enabling hardware-accelerated computation on Blackwell tensor cores.

MXFP4 (Microscaling FP4): Similar to NVFP4 but uses a microscaling approach where groups
of elements share a common scale factor. This can provide better accuracy for some model
architectures at the cost of slightly more complex memory access patterns.""",

        """4. CONTINUOUS BATCHING AND SCHEDULING

Static batching, where all requests in a batch must complete before new requests can be
added, leads to significant GPU underutilization. This is because different requests in
a batch may have vastly different output lengths, and the batch is limited by the slowest
request. Continuous batching, also known as iteration-level scheduling, addresses this by
allowing new requests to join and completed requests to leave the batch at each iteration
step. This dramatically improves GPU utilization and reduces average latency.

In vLLM's V1 engine (version 0.13.0), the scheduler implements chunked prefill, which
breaks long prefill operations into smaller chunks that can be interleaved with decode
iterations. This prevents long prefill requests from blocking decode operations for
existing requests, maintaining low inter-token latency for streaming users.

The scheduling algorithm must balance several competing objectives: maximizing throughput
by keeping the GPU busy, minimizing time-to-first-token for new requests, maintaining
low inter-token latency for ongoing generations, and staying within the GPU memory budget.
The optimal scheduling strategy depends on the workload characteristics, particularly
the ratio of prefill to decode computation and the distribution of request lengths.""",

        """5. DISAGGREGATED INFERENCE

At GTC 2026, NVIDIA introduced the concept of disaggregated inference as a core principle
of their Vera Rubin inference platform. The key insight is that prefill and decode have
fundamentally different computational characteristics. Prefill is compute-bound, requiring
high FLOPS to process all input tokens through the model. It benefits from hardware with
high arithmetic intensity, such as GPU tensor cores. Decode is memory-bandwidth-bound,
as each token generation requires reading the entire model weights but performs relatively
little computation per byte read. It benefits from hardware with high memory bandwidth
relative to compute, such as the Groq LPU.

By separating these two phases onto different hardware optimized for each, disaggregated
inference can achieve better overall efficiency than running both phases on the same device.
NVIDIA's Dynamo framework orchestrates the routing of requests between prefill and decode
hardware, handling the transfer of KV cache data between devices.

On consumer hardware like the RTX 5070 Ti, true disaggregated inference across different
devices is not possible. However, understanding the prefill/decode decomposition is still
valuable for optimizing single-GPU inference. By measuring the time spent in each phase,
we can identify bottlenecks and apply targeted optimizations. For prefill-dominated
workloads (long context RAG), prefix caching and chunked prefill are most impactful.
For decode-dominated workloads (chatbot with short inputs), continuous batching and
speculative decoding provide the greatest benefits.""",

        """6. BENCHMARKING METHODOLOGY

Proper inference benchmarking requires careful attention to measurement methodology.
The key metrics for inference performance are Time to First Token (TTFT), which measures
the latency from request submission to the first generated token, Tokens Per Second (TPS)
which measures the decode throughput, Inter-Token Latency (ITL) which measures the time
between consecutive tokens during streaming, and tail latencies at P95 and P99 percentiles.

TTFT is approximately equal to the prefill time for a single request on an unloaded server.
Under load, TTFT includes queuing delay in addition to prefill computation. Measuring TTFT
accurately requires streaming responses and recording the timestamp of the first received
token with high-resolution timers like time.perf_counter in Python.

TPS during decode should be measured separately from prefill to avoid including prefill
time in the throughput calculation. The aggregate throughput under concurrent load is the
most relevant metric for capacity planning, while per-request TPS affects individual user
experience. These two metrics can diverge significantly: a system may achieve high aggregate
throughput through batching while individual requests experience lower per-request TPS
due to batch interference.

Energy efficiency is increasingly important as inference costs scale. Measuring power
consumption per token enables cost comparisons between quantization formats and hardware
configurations. The NVIDIA RTX 5070 Ti has a TDP of 300W, and measuring actual power
draw during inference provides a more accurate picture of operational costs.""",
    ]

    full_text = "\n\n".join(sections)
    words = full_text.split()

    # Repeat sections until we reach target_words
    while len(words) < target_words:
        for section in sections:
            words.extend(section.split())
            if len(words) >= target_words:
                break

    full_text = " ".join(words[:target_words])
    return full_text


# Different questions about the same context (simulates RAG with shared prefix)
RAG_QUESTIONS = [
    "What is the KV cache memory formula for an 8B parameter model at 8192 tokens?",
    "Explain the difference between prefill and decode computational characteristics.",
    "How does prefix caching reduce redundant computation in RAG workloads?",
    "Compare AWQ and NVFP4 quantization formats in terms of memory and precision.",
    "What scheduling tradeoffs does the vLLM V1 engine need to balance?",
    "Why did NVIDIA introduce disaggregated inference at GTC 2026?",
    "What are the key metrics for proper inference benchmarking?",
    "How does continuous batching improve GPU utilization over static batching?",
]


@dataclass
class PrefillDecodeResult:
    """Result with explicit prefill/decode breakdown."""

    ttft_ms: float
    total_time_ms: float
    output_tokens: int
    token_timestamps_ms: list[float]
    question: str

    @property
    def decode_time_ms(self) -> float:
        return max(0.0, self.total_time_ms - self.ttft_ms)

    @property
    def tps(self) -> float:
        decode_s = self.decode_time_ms / 1000
        return self.output_tokens / decode_s if decode_s > 0 else 0.0

    @property
    def prefill_ratio(self) -> float:
        return self.ttft_ms / self.total_time_ms if self.total_time_ms > 0 else 0.0

    @property
    def itl_ms(self) -> float:
        if len(self.token_timestamps_ms) < 2:
            return 0.0
        deltas = [
            self.token_timestamps_ms[i] - self.token_timestamps_ms[i - 1]
            for i in range(1, len(self.token_timestamps_ms))
        ]
        return statistics.mean(deltas)


def send_rag_request(
    server_url: str,
    model: str,
    context: str,
    question: str,
    system_prompt: str,
    max_tokens: int,
) -> PrefillDecodeResult:
    """Send a RAG-style request with long context and measure prefill/decode."""
    parsed = urlparse(server_url)
    conn = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=300)

    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
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

    return PrefillDecodeResult(
        ttft_ms=(first_token_time - start) * 1000 if first_token_time else 0.0,
        total_time_ms=(end - start) * 1000,
        output_tokens=output_tokens,
        token_timestamps_ms=token_timestamps,
        question=question,
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
    parser = argparse.ArgumentParser(description="InferBench Prefill/Decode Experiment")
    parser.add_argument("--server", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen3-8B-AWQ")
    parser.add_argument("--max-tokens", "-t", type=int, default=256)
    parser.add_argument("--num-rounds", "-n", type=int, default=2,
                        help="Number of rounds through all questions")
    parser.add_argument("--label", default="default",
                        help="Label for this run (e.g. 'prefix-cache-on', 'no-cache')")
    args = parser.parse_args()

    context = _generate_rag_context()
    context_words = len(context.split())
    est_context_tokens = int(context_words * 1.3)
    est_system_tokens = int(len(SYSTEM_PROMPT.split()) * 1.3)
    est_total_input = est_context_tokens + est_system_tokens

    print(f"\n{'='*65}")
    print(f"  InferBench Experiment 2: Prefill/Decode Disaggregation")
    print(f"{'='*65}")
    print(f"  Server:          {args.server}")
    print(f"  Model:           {args.model}")
    print(f"  Label:           {args.label}")
    print(f"  Context words:   ~{context_words}")
    print(f"  Est input tokens:~{est_total_input} (system + context + question)")
    print(f"  Max output:      {args.max_tokens} tokens")
    print(f"  Questions:       {len(RAG_QUESTIONS)} x {args.num_rounds} rounds = {len(RAG_QUESTIONS)*args.num_rounds} requests")
    print(f"  Shared prefix:   system prompt + context (same across all questions)")
    print(f"{'='*65}")

    # Warmup with first question
    print(f"\n  Warming up (1 request)...")
    warmup = send_rag_request(
        args.server, args.model, context,
        RAG_QUESTIONS[0], SYSTEM_PROMPT, args.max_tokens,
    )
    print(f"  Warmup: TTFT={warmup.ttft_ms:.0f}ms, tokens={warmup.output_tokens}")

    # Run benchmark
    results: list[PrefillDecodeResult] = []
    first_round_results: list[PrefillDecodeResult] = []
    subsequent_results: list[PrefillDecodeResult] = []

    for round_num in range(args.num_rounds):
        print(f"\n  ── Round {round_num + 1}/{args.num_rounds} ──")
        for i, question in enumerate(RAG_QUESTIONS):
            short_q = question[:55] + "..." if len(question) > 55 else question
            r = send_rag_request(
                args.server, args.model, context,
                question, SYSTEM_PROMPT, args.max_tokens,
            )
            results.append(r)
            if round_num == 0:
                first_round_results.append(r)
            else:
                subsequent_results.append(r)

            print(
                f"    [{i+1}/{len(RAG_QUESTIONS)}] TTFT={r.ttft_ms:7.0f}ms | "
                f"decode={r.decode_time_ms:7.0f}ms | "
                f"TPS={r.tps:5.1f} | "
                f"prefill={r.prefill_ratio:5.1%} | "
                f"{short_q}"
            )

    # Summary
    print(f"\n{'='*65}")
    print(f"  RESULTS SUMMARY — {args.label}")
    print(f"{'='*65}")

    def _print_group(name: str, group: list[PrefillDecodeResult]) -> dict:
        if not group:
            return {}
        ttfts = [r.ttft_ms for r in group]
        decodes = [r.decode_time_ms for r in group]
        tps_vals = [r.tps for r in group]
        prefill_ratios = [r.prefill_ratio for r in group]
        itls = [r.itl_ms for r in group if r.itl_ms > 0]

        print(f"\n  {name} ({len(group)} requests):")
        print(f"  ┌─ Prefill (TTFT ≈ prefill time)")
        print(f"  │  P50:  {percentile(ttfts, 50):7.0f} ms")
        print(f"  │  P95:  {percentile(ttfts, 95):7.0f} ms")
        print(f"  │  Mean: {statistics.mean(ttfts):7.0f} ms")
        print(f"  ├─ Decode")
        print(f"  │  P50:  {percentile(decodes, 50):7.0f} ms")
        print(f"  │  Mean: {statistics.mean(decodes):7.0f} ms")
        print(f"  ├─ TPS (decode phase)")
        print(f"  │  P50:  {percentile(tps_vals, 50):7.1f}")
        print(f"  │  Mean: {statistics.mean(tps_vals):7.1f}")
        if itls:
            print(f"  ├─ ITL")
            print(f"  │  P50:  {percentile(itls, 50):7.1f} ms")
        print(f"  ├─ Prefill/Decode ratio")
        print(f"  │  Mean: {statistics.mean(prefill_ratios):5.1%} prefill / {1-statistics.mean(prefill_ratios):5.1%} decode")
        print(f"  └─")

        return {
            "name": name,
            "count": len(group),
            "ttft_p50_ms": round(percentile(ttfts, 50), 1),
            "ttft_p95_ms": round(percentile(ttfts, 95), 1),
            "ttft_mean_ms": round(statistics.mean(ttfts), 1),
            "decode_p50_ms": round(percentile(decodes, 50), 1),
            "decode_mean_ms": round(statistics.mean(decodes), 1),
            "tps_p50": round(percentile(tps_vals, 50), 1),
            "tps_mean": round(statistics.mean(tps_vals), 1),
            "itl_p50_ms": round(percentile(itls, 50), 1) if itls else None,
            "prefill_ratio_mean": round(statistics.mean(prefill_ratios), 4),
        }

    all_summary = _print_group("All requests", results)
    r1_summary = _print_group("Round 1 (cold cache)", first_round_results)
    r2_summary = {}
    if subsequent_results:
        r2_summary = _print_group("Round 2+ (warm cache)", subsequent_results)

        # Cache hit analysis
        r1_ttft = statistics.mean([r.ttft_ms for r in first_round_results])
        r2_ttft = statistics.mean([r.ttft_ms for r in subsequent_results])
        reduction = (r1_ttft - r2_ttft) / r1_ttft * 100 if r1_ttft > 0 else 0

        print(f"\n  ── Prefix Cache Effect ──")
        print(f"  Round 1 TTFT mean: {r1_ttft:7.0f} ms")
        print(f"  Round 2 TTFT mean: {r2_ttft:7.0f} ms")
        print(f"  TTFT reduction:    {reduction:5.1f}%")
        if reduction > 5:
            print(f"  → Prefix caching is EFFECTIVE ({reduction:.0f}% TTFT reduction)")
        else:
            print(f"  → Prefix caching shows MINIMAL effect (may be disabled or context too varied)")

    # Save JSON
    os.makedirs("results", exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output = {
        "experiment": "prefill_decode_disaggregation",
        "label": args.label,
        "timestamp": timestamp,
        "config": {
            "server": args.server,
            "model": args.model,
            "max_tokens": args.max_tokens,
            "num_rounds": args.num_rounds,
            "est_input_tokens": est_total_input,
            "context_words": context_words,
        },
        "hardware": {
            "gpu": "NVIDIA RTX 5070 Ti 16GB",
            "cpu": "AMD Ryzen 9 9900X",
            "driver": "595.79",
            "cuda": "13.2",
            "vllm": "0.13.0",
        },
        "summary": {
            "all": all_summary,
            "round1_cold": r1_summary,
            "round2_warm": r2_summary,
        },
        "per_request": [
            {
                "question": r.question[:60],
                "ttft_ms": round(r.ttft_ms, 1),
                "decode_time_ms": round(r.decode_time_ms, 1),
                "total_time_ms": round(r.total_time_ms, 1),
                "output_tokens": r.output_tokens,
                "tps": round(r.tps, 1),
                "prefill_ratio": round(r.prefill_ratio, 4),
            }
            for r in results
        ],
    }
    outfile = f"results/prefill_decode_{args.label}_{timestamp}.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to: {outfile}")


if __name__ == "__main__":
    main()