# InferBench

> **Don't just run models. Understand why they run the way they do.**

Benchmarking LLM inference on NVIDIA RTX 5070 Ti (16GB, Blackwell SM_120). Filling the gaps that [arXiv:2601.09527](https://arxiv.org/abs/2601.09527) left open.

**Hardware:** RTX 5070 Ti 16GB GDDR7 · Ryzen 9 9900X · vLLM 0.13.0 · SGLang latest  
**Blog:** [kr-lsb.github.io](https://kr-lsb.github.io/)  
**License:** MIT

---

## Key Results

### Experiment 1: AWQ vs NVFP4 Quantization

Qwen3-8B on vLLM 0.13.0, API-short workload (256 in / 256 out).

| Concurrency | AWQ Agg TPS | NVFP4 Agg TPS | AWQ TTFT | NVFP4 TTFT |
|---|---|---|---|---|
| 1 | 119.9 | 97.4 | 33 ms | 41 ms |
| 8 | 885.5 | 524.2 | 54 ms | 57 ms |
| 16 | **1,587** | 1,015 | 128 ms | **72 ms** |
| 32 | 1,576 | 1,016 | 126 ms | 70 ms |

**Finding:** AWQ (`awq_marlin` kernel) delivers 56% higher throughput than NVFP4 (`flashinfer-cutlass`). Kernel maturity matters more than bit width. NVFP4 wins on TTFT at high concurrency due to smaller memory footprint.

📝 [Blog: AWQ vs NVFP4 and Why Prefix Caching Changes Everything](https://kr-lsb.github.io/posts/inferbench-awq-vs-nvfp4/)

### Experiment 2: Prefix Caching on RAG Workload

Qwen3-8B-AWQ, ~7,593 token context, 8 different questions, 256 max output.

| Metric | Cache ON | Cache OFF | Improvement |
|---|---|---|---|
| TTFT P50 | 66 ms | 1,787 ms | **27x faster** |
| TTFT mean | 67 ms | 2,373 ms | **35x faster** |
| Prefill % of total | 1.9% | 38.2% | Nearly eliminated |
| Decode TPS | 73.6 | 70.6 | ~Same |

**Finding:** Prefix caching reduces TTFT by 96.5% on a 7.5k token RAG context. Decode throughput is unaffected — it's a pure prefill optimization.

📝 [Blog: AWQ vs NVFP4 and Why Prefix Caching Changes Everything](https://kr-lsb.github.io/posts/inferbench-awq-vs-nvfp4/)

### Experiment 3: KV Cache Economics

Qwen3-8B-AWQ, context sweep 2k → 4k → 8k → 16k, c=1, 256 max output.

**Without Prefix Caching:**

| Context | TTFT | TPS | Prefill % | Total Latency |
|---|---|---|---|---|
| ~2k | 505 ms | 74.4 | 12.7% | 4,148 ms |
| ~4k | 1,200 ms | 65.3 | 23.6% | 5,119 ms |
| ~8k | 2,547 ms | 62.7 | 38.3% | 6,668 ms |
| ~16k | 6,306 ms | 61.0 | 59.9% | 10,517 ms |

**With Prefix Caching:**

| Context | TTFT | TPS | Prefill % | Total Latency |
|---|---|---|---|---|
| ~2k | 62 ms | 95.6 | 2.2% | 2,751 ms |
| ~4k | 72 ms | 94.7 | 2.7% | 2,725 ms |
| ~8k | 78 ms | 85.5 | 2.5% | 3,084 ms |
| ~16k | 111 ms | 69.1 | 3.2% | 3,553 ms |

**Prefix Caching Speedup by Context Length:**

| Context | Cache OFF TTFT | Cache ON TTFT | Speedup |
|---|---|---|---|
| ~2k | 505 ms | 62 ms | **8x** |
| ~4k | 1,200 ms | 72 ms | **17x** |
| ~8k | 2,547 ms | 78 ms | **33x** |
| ~16k | 6,306 ms | 111 ms | **57x** |

**Finding:** TTFT scales super-linearly with context length. At 16k tokens, prefill is 60% of total latency. Prefix caching speedup accelerates from 8x at 2k to 57x at 16k.

📝 [Blog: KV Cache is the New Memory Wall](https://kr-lsb.github.io/posts/kv-cache-economics/)

### Experiment 4: vLLM vs SGLang — Engine Comparison

Qwen3-8B-AWQ, identical benchmark script for both engines, thinking ON, max_tokens=256.

**Concurrency Benchmark:**

| Metric | c=1 | c=8 | c=16 | c=32 |
|---|---|---|---|---|
| **Agg TPS — vLLM** | 121.5 | 551.7 | 968.8 | 969.4 |
| **Agg TPS — SGLang** | 130.5 | 810.2 | 978.4 | **1,027.8** |
| **TTFT P50 — vLLM** | **15.4 ms** | **37.3 ms** | **40.5 ms** | 42.2 ms |
| **TTFT P50 — SGLang** | 20.0 ms | 37.4 ms | 69.2 ms | **39.7 ms** |

**Prefix Caching (RAG scenario, ~1k words context):**

| Metric | vLLM | SGLang |
|---|---|---|
| Cached TTFT P50 | **19.8 ms** | 20.7 ms |
| Decode TPS | 115.6 | **127.7** |
| Prefill ratio | 0.9% | 1.0% |

**Findings:**
1. **SGLang wins throughput** — 6-47% higher aggregate TPS across concurrency levels.
2. **vLLM wins latency stability** — TTFT P95/P50 ratio stays below 1.1x; SGLang shows intermittent spikes (P95=299ms at c=8).
3. **Prefix caching equally effective** — Both achieve ~20ms cached TTFT. PagedAttention ≈ RadixAttention at this scale.
4. **SGLang 10% faster decode** — Consistent 127.7 vs 115.6 TPS across all prefix caching runs.
5. **Both saturate at c=16** — No meaningful throughput gain beyond 16 concurrent requests.

📝 [Blog: vLLM vs SGLang on Blackwell: Which Engine Wins?](https://kr-lsb.github.io/posts/vllm-vs-sglang/)

---

## What This Project Adds Over arXiv:2601.09527

| Dimension | arXiv:2601.09527 | InferBench |
|---|---|---|
| Prefill/Decode separation | ❌ E2E only | ✅ Streaming-based measurement |
| Prefix caching | ❌ Not tested | ✅ Up to 57x TTFT reduction |
| Kernel comparison | ❌ NVFP4 only | ✅ AWQ vs NVFP4 |
| Context length scaling | ❌ Fixed | ✅ 2k → 4k → 8k → 16k sweep |
| Engine comparison | ❌ vLLM only | ✅ vLLM vs SGLang |
| Memory pressure effects | ❌ Not observed | ✅ Degradation documented |
| Reproducibility | Docker image | ✅ Docker Compose + scripts |

---

## Quick Start

### Prerequisites

- NVIDIA GPU with driver 595+ (CUDA 12.9+)
- Docker with NVIDIA runtime
- Python 3.11+ with `httpx`

### Run a Benchmark

```bash
# 1a. Start vLLM server
docker run --gpus all --ipc=host -p 8000:8000 \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  --model Qwen/Qwen3-8B-AWQ \
  --gpu-memory-utilization 0.90 --dtype auto

# 1b. Or start SGLang server
docker run --gpus all --ipc=host -p 30000:30000 --shm-size 16g \
  -v inferbench-hf-cache:/root/.cache/huggingface \
  lmsysorg/sglang:latest \
  python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-8B-AWQ \
  --host 0.0.0.0 --port 30000 --mem-fraction-static 0.90

# 2. Run concurrency benchmark
python scripts/engine_bench.py --engine vllm --port 8000 --concurrency 1 8 16 32
python scripts/engine_bench.py --engine sglang --port 30000 --concurrency 1 8 16 32

# 3. Run prefix caching comparison
python scripts/prefix_cache_bench.py --engine vllm --port 8000
python scripts/prefix_cache_bench.py --engine sglang --port 30000

# 4. Run KV cache economics sweep (vLLM)
python scripts/kv_cache_bench.py --server http://localhost:8000
```

---

## Project Structure

```
inferbench/
├── configs/
│   ├── models.yaml          # Model configs
│   ├── workloads.yaml       # Workload definitions
│   ├── engines.yaml         # Engine configs
│   └── prometheus.yml       # Prometheus scrape config
├── scripts/
│   ├── quick_bench.py       # Sequential benchmark (c=1)
│   ├── concurrent_bench.py  # Concurrency scaling (c=1,8,16,32)
│   ├── prefill_decode_bench.py  # Prefix caching experiment
│   ├── kv_cache_bench.py    # Context length sweep (2k-16k)
│   ├── engine_bench.py      # vLLM vs SGLang concurrency comparison
│   └── prefix_cache_bench.py    # vLLM vs SGLang prefix caching comparison
├── src/
│   ├── bench/
│   │   ├── metrics.py       # TTFT, TPS, P50/P95/P99
│   │   ├── runner.py        # CLI benchmark runner
│   │   └── workloads.py     # Workload generators
│   └── optimizations/
│       └── disaggregated.py # Prefill/Decode measurement
├── results/                 # JSON benchmark results
├── docs/
│   └── engine_comparison.md # Experiment 4 full analysis
└── tests/
```

---

## Hardware & Software

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| OS | Windows 11 + WSL2 (Ubuntu 24.04) |
| Driver | 595.79 (CUDA 13.2) |
| vLLM | 0.13.0 (`vllm/vllm-openai:latest`) |
| SGLang | latest (`lmsysorg/sglang:latest`) |
| Model (AWQ) | `Qwen/Qwen3-8B-AWQ` — awq_marlin kernel, 5.7 GiB |
| Model (NVFP4) | `RedHatAI/Qwen3-8B-NVFP4` — flashinfer-cutlass, ~4.5 GiB |

---

## Roadmap

- [x] Experiment 1: AWQ vs NVFP4 quantization comparison
- [x] Experiment 2: Prefill/Decode disaggregation + prefix caching
- [x] Experiment 3: KV cache economics (context length scaling)
- [x] Experiment 4: vLLM vs SGLang engine comparison
- [x] Blog 3: vLLM vs SGLang on Blackwell
- [ ] EXAONE-Deep-7.8B benchmarks (Korean-language model)
- [ ] Prometheus + Grafana monitoring dashboard
- [ ] TensorRT-LLM comparison (optional)

---

## Blog Series

1. [Benchmarking LLM Inference on RTX 5070 Ti: AWQ vs NVFP4 and Why Prefix Caching Changes Everything](https://kr-lsb.github.io/posts/inferbench-awq-vs-nvfp4/)
2. [KV Cache is the New Memory Wall: 57x TTFT Reduction at 16k Context](https://kr-lsb.github.io/posts/kv-cache-economics/)
3. [vLLM vs SGLang on Blackwell: Which Engine Wins on RTX 5070 Ti?](https://kr-lsb.github.io/posts/vllm-vs-sglang/)

---

## Author

**SeungByeong** — ML engineer focused on LLM inference optimization.

- [GitHub](https://github.com/KR-LSB)
- [Blog](https://kr-lsb.github.io/)
- Previously: M.A.R.S. medical AI (SNUBH Datathon 6th/100, F1 0.92), [LangChain contributor](https://github.com/langchain-ai/langchain/pull/34997)