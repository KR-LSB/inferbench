#!/usr/bin/env python3
"""
Experiment 5B: EXAONE-Deep-7.8B Korean RAG Prefix Caching Benchmark

Tests prefix caching effectiveness with Korean technical document context.
EXAONE does not support system prompts — context goes in the user message.

Usage (PowerShell):
  # vLLM prefix caching ON (default in 0.13.0)
  python scripts/exaone_rag_bench.py --port 8000 --engine vllm --label cache_on

  # vLLM prefix caching OFF (restart server with --no-enable-prefix-caching)
  python scripts/exaone_rag_bench.py --port 8000 --engine vllm --label cache_off

  # SGLang
  python scripts/exaone_rag_bench.py --port 30000 --engine sglang --label cache_on
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
MAX_TOKENS = 256
WARMUP_REQUESTS = 3
N_REQUESTS = 20  # Per question

# ---------------------------------------------------------------------------
# Korean RAG context (~1k words, technical AI/ML document)
# EXAONE: no system prompt → context embedded in user message
# ---------------------------------------------------------------------------

KOREAN_RAG_CONTEXT = """다음은 참고 문서입니다. 이 문서를 기반으로 질문에 답변해주세요.

---

## 대규모 언어 모델의 추론 최적화 기술 개요

### 1. KV 캐시와 메모리 병목

트랜스포머 기반 대규모 언어 모델(LLM)에서 추론 시 가장 큰 병목 중 하나는 KV(Key-Value) 캐시의 메모리 사용량이다. 자기회귀(autoregressive) 생성 과정에서 모델은 이전에 생성한 모든 토큰의 키와 밸류 벡터를 저장해야 하며, 이 캐시는 시퀀스 길이에 비례하여 선형적으로 증가한다.

예를 들어, 8B 파라미터 모델에서 16K 컨텍스트를 처리할 경우, KV 캐시만으로 수 기가바이트의 GPU 메모리를 소비할 수 있다. 이는 특히 소비자용 GPU(16GB VRAM)에서 동시 처리 가능한 요청 수를 심각하게 제한한다.

### 2. 프리필(Prefill)과 디코드(Decode) 단계 분리

LLM 추론은 크게 두 단계로 나뉜다. 프리필 단계에서는 입력 프롬프트의 모든 토큰을 병렬로 처리하여 KV 캐시를 구축한다. 이 단계는 컴퓨트 바운드(compute-bound)이며, GPU의 연산 능력에 의해 성능이 결정된다.

디코드 단계에서는 한 번에 하나의 토큰을 순차적으로 생성한다. 이 단계는 메모리 바운드(memory-bound)이며, KV 캐시에 대한 메모리 대역폭이 병목이 된다. GTC 2026에서 NVIDIA가 발표한 분리형 추론(Disaggregated Inference) 아키텍처는 이 두 단계의 특성 차이를 활용하여, 프리필에는 높은 연산 성능의 GPU를, 디코드에는 낮은 지연시간의 전용 프로세서(LPU)를 할당한다.

### 3. 프리픽스 캐싱(Prefix Caching)

RAG(Retrieval-Augmented Generation) 워크로드에서는 동일한 시스템 프롬프트나 검색된 문서가 여러 쿼리에 걸쳐 반복된다. 프리픽스 캐싱은 이러한 공통 프리픽스의 KV 캐시를 재사용하여 프리필 시간을 극적으로 단축하는 기법이다.

vLLM의 자동 프리픽스 캐싱은 프롬프트의 해시 값을 기반으로 캐시 히트를 판단하며, 캐시가 히트할 경우 프리필 시간이 수십 밀리초 수준으로 감소할 수 있다. 이는 16K 컨텍스트에서 캐시 미스 시 6초 이상 걸리는 프리필이 100밀리초 이내로 단축되는 것을 의미한다.

### 4. 양자화(Quantization) 기법

모델 가중치를 낮은 비트로 양자화하면 메모리 사용량과 연산량을 줄일 수 있다. 주요 양자화 포맷은 다음과 같다:

- **AWQ (Activation-aware Weight Quantization):** 가중치를 4비트로 양자화하되, 활성화 분포를 고려하여 중요한 채널은 높은 정밀도를 유지한다. 성숙한 커널(awq_marlin)이 있어 실제 처리량이 높다.
- **NVFP4:** NVIDIA Blackwell 아키텍처의 네이티브 FP4 포맷으로, 하드웨어 가속을 통해 효율적인 저비트 연산이 가능하다. 다만 커널 성숙도는 AWQ 대비 아직 발전 중이다.
- **GPTQ:** 초기 양자화 기법으로, 레이어별 최적화를 통해 양자화 오차를 최소화한다.

### 5. 연속 배칭(Continuous Batching)과 PagedAttention

전통적인 정적 배칭은 배치 내 모든 요청이 완료될 때까지 대기해야 한다. vLLM이 도입한 연속 배칭은 개별 요청의 생성이 완료되는 즉시 새로운 요청을 배치에 추가하여, GPU 활용률을 극대화한다.

PagedAttention은 KV 캐시를 가상 메모리의 페이지처럼 관리하여, 메모리 단편화를 최소화하고 더 많은 동시 요청을 처리할 수 있게 한다. 이 두 기술의 조합은 vLLM이 높은 처리량을 달성하는 핵심 원리다.

### 6. 한국어 모델의 추론 특성

한국어는 영어 대비 토큰당 정보 밀도가 다르다. 한국어 텍스트는 동일한 의미를 표현하는 데 더 많은 토큰을 소비하는 경향이 있으며, 이는 KV 캐시 크기와 추론 비용에 직접적으로 영향을 미친다. EXAONE과 같은 한국어 특화 모델의 토크나이저는 한국어에 최적화되어 있어 범용 모델 대비 효율적인 토큰화를 수행한다.

---"""

# Different questions over the SAME context (tests prefix caching)
KOREAN_QUESTIONS = [
    "KV 캐시가 추론에서 병목이 되는 이유를 설명하고, 이를 해결하기 위한 기술적 접근법을 정리해주세요.",
    "프리필과 디코드 단계의 차이점을 설명하고, NVIDIA의 분리형 추론이 이를 어떻게 활용하는지 설명해주세요.",
    "프리픽스 캐싱이 RAG 워크로드에서 왜 효과적인지, 구체적인 성능 개선 수치와 함께 설명해주세요.",
    "AWQ와 NVFP4 양자화의 차이점을 비교하고, 소비자 GPU에서의 실용성을 평가해주세요.",
    "연속 배칭과 PagedAttention의 작동 원리를 설명하고, 이 기술이 vLLM의 처리량에 기여하는 방식을 설명해주세요.",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Result of a single inference request."""
    question_idx: int
    ttft_ms: float
    total_time_ms: float
    output_tokens: int
    decode_tps: float
    prefill_ratio: float
    error: Optional[str] = None


@dataclass
class ExperimentResult:
    """Aggregated result for the prefix caching experiment."""
    engine: str
    label: str
    model: str
    context_words_approx: int
    n_questions: int
    n_requests_per_question: int
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_mean_ms: float
    decode_tps_mean: float
    prefill_ratio_mean: float
    total_time_p50_ms: float
    errors: int


# ---------------------------------------------------------------------------
# Core benchmark logic
# ---------------------------------------------------------------------------

async def send_request(
    client: httpx.AsyncClient,
    url: str,
    question: str,
    question_idx: int,
    engine: str,
) -> RequestResult:
    """Send a single request with Korean RAG context + question."""
    # EXAONE: no system prompt. Context + question in user message.
    user_content = f"{KOREAN_RAG_CONTEXT}\n\n질문: {question}"
    messages = [{"role": "user", "content": user_content}]

    body: dict = {
        "model": MODEL_NAME,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "temperature": 0.6,
        "stream": True,
        "repetition_penalty": 1.0,
    }

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
            return RequestResult(question_idx, 0, 0, 0, 0, 0, error="No tokens received")

        ttft_ms = (first_token_time - start) * 1000
        total_ms = (end - start) * 1000
        decode_ms = total_ms - ttft_ms
        decode_tps = output_tokens / (decode_ms / 1000) if decode_ms > 0 else 0
        prefill_ratio = ttft_ms / total_ms if total_ms > 0 else 0

        return RequestResult(
            question_idx=question_idx,
            ttft_ms=round(ttft_ms, 1),
            total_time_ms=round(total_ms, 1),
            output_tokens=output_tokens,
            decode_tps=round(decode_tps, 1),
            prefill_ratio=round(prefill_ratio, 4),
        )

    except Exception as e:
        return RequestResult(question_idx, 0, 0, 0, 0, 0, error=str(e))


async def warmup(port: int, engine: str) -> None:
    """Send warmup requests to stabilize the model server."""
    url = f"http://localhost:{port}/v1/chat/completions"
    body = {
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "안녕하세요, 간단히 응답해주세요."}],
        "max_tokens": 32,
        "temperature": 0.6,
        "stream": False,
        "repetition_penalty": 1.0,
    }

    print(f"  Warming up ({WARMUP_REQUESTS} requests)...", flush=True)
    async with httpx.AsyncClient() as client:
        for i in range(WARMUP_REQUESTS):
            try:
                resp = await client.post(url, json=body, timeout=60.0)
                resp.raise_for_status()
                print(f"    Warmup {i+1}/{WARMUP_REQUESTS} OK", flush=True)
            except Exception as e:
                print(f"    Warmup {i+1}/{WARMUP_REQUESTS} FAILED: {e}", flush=True)


async def run_prefix_caching_bench(
    port: int,
    engine: str,
    label: str,
) -> ExperimentResult:
    """
    Run prefix caching benchmark.

    Strategy: Send requests sequentially with different questions but the
    SAME context. If prefix caching is ON, the 2nd+ requests should have
    dramatically lower TTFT since the context KV cache is reused.
    """
    url = f"http://localhost:{port}/v1/chat/completions"
    all_results: list[RequestResult] = []

    async with httpx.AsyncClient() as client:
        for q_idx, question in enumerate(KOREAN_QUESTIONS):
            print(f"\n  Question {q_idx + 1}/{len(KOREAN_QUESTIONS)}: {question[:50]}...")
            for req_num in range(N_REQUESTS):
                result = await send_request(client, url, question, q_idx, engine)
                all_results.append(result)
                if req_num < 3 or req_num == N_REQUESTS - 1:
                    status = "OK" if result.error is None else f"ERR: {result.error}"
                    print(
                        f"    [{req_num+1:>2}/{N_REQUESTS}] "
                        f"TTFT: {result.ttft_ms:>7.1f} ms | "
                        f"Decode: {result.decode_tps:>6.1f} tps | "
                        f"{status}"
                    )

    # Aggregate
    ok = [r for r in all_results if r.error is None]
    errors = len(all_results) - len(ok)

    if not ok:
        return ExperimentResult(
            engine=engine, label=label, model=MODEL_NAME,
            context_words_approx=len(KOREAN_RAG_CONTEXT),
            n_questions=len(KOREAN_QUESTIONS),
            n_requests_per_question=N_REQUESTS,
            ttft_p50_ms=0, ttft_p95_ms=0, ttft_mean_ms=0,
            decode_tps_mean=0, prefill_ratio_mean=0,
            total_time_p50_ms=0, errors=errors,
        )

    ttfts = sorted(r.ttft_ms for r in ok)

    def percentile(data: list[float], p: float) -> float:
        idx = int(len(data) * p / 100)
        idx = min(idx, len(data) - 1)
        return data[idx]

    # Also compute first-request vs subsequent TTFT (cache warm analysis)
    first_ttfts = [r.ttft_ms for r in ok if r.question_idx == 0][:3]
    later_ttfts = [r.ttft_ms for r in ok if r.question_idx > 0]

    print(f"\n  --- Cache Warmth Analysis ---")
    if first_ttfts:
        print(f"  First few requests TTFT: {[round(t, 1) for t in first_ttfts]}")
    if later_ttfts:
        print(
            f"  Subsequent requests TTFT P50: "
            f"{round(percentile(sorted(later_ttfts), 50), 1)} ms"
        )

    return ExperimentResult(
        engine=engine,
        label=label,
        model=MODEL_NAME,
        context_words_approx=len(KOREAN_RAG_CONTEXT),
        n_questions=len(KOREAN_QUESTIONS),
        n_requests_per_question=N_REQUESTS,
        ttft_p50_ms=round(percentile(ttfts, 50), 1),
        ttft_p95_ms=round(percentile(ttfts, 95), 1),
        ttft_mean_ms=round(statistics.mean(r.ttft_ms for r in ok), 1),
        decode_tps_mean=round(statistics.mean(r.decode_tps for r in ok), 1),
        prefill_ratio_mean=round(statistics.mean(r.prefill_ratio for r in ok), 4),
        total_time_p50_ms=round(percentile(sorted(r.total_time_ms for r in ok), 50), 1),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Experiment 5B: EXAONE Korean RAG prefix caching benchmark"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument(
        "--label", type=str, default="cache_on",
        help="Label for this run (e.g., cache_on, cache_off)"
    )
    parser.add_argument("--no-warmup", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print(f"  EXAONE Korean RAG Prefix Caching Benchmark")
    print(f"  Engine: {args.engine.upper()} (port {args.port})")
    print(f"  Label: {args.label}")
    print(f"  Model: {MODEL_NAME}")
    print(f"  Context: ~{len(KOREAN_RAG_CONTEXT)} chars Korean technical document")
    print(f"  Questions: {len(KOREAN_QUESTIONS)} (x{N_REQUESTS} requests each)")
    print("=" * 70)

    if not args.no_warmup:
        await warmup(args.port, args.engine)

    result = await run_prefix_caching_bench(args.port, args.engine, args.label)

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  TTFT P50:      {result.ttft_p50_ms:>8.1f} ms")
    print(f"  TTFT P95:      {result.ttft_p95_ms:>8.1f} ms")
    print(f"  TTFT Mean:     {result.ttft_mean_ms:>8.1f} ms")
    print(f"  Decode TPS:    {result.decode_tps_mean:>8.1f}")
    print(f"  Prefill Ratio: {result.prefill_ratio_mean:>7.2%}")
    print(f"  Errors:        {result.errors}")

    # Save results
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"exaone_rag_{args.engine}_{args.label}_{ts}.json"

    with open(out_path, "w") as f:
        json.dump(
            {
                "experiment": "5B_exaone_korean_rag",
                "engine": args.engine,
                "label": args.label,
                "model": MODEL_NAME,
                "context_chars": len(KOREAN_RAG_CONTEXT),
                "max_tokens": MAX_TOKENS,
                "n_questions": len(KOREAN_QUESTIONS),
                "n_requests_per_question": N_REQUESTS,
                "result": asdict(result),
            },
            f,
            indent=2,
        )

    print(f"\n  Results saved: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
