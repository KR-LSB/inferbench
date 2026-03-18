# InferBench

> **"Don't just run models. Understand why they run the way they do."**

LLM inference benchmarking on consumer Blackwell GPU (NVIDIA RTX 5070 Ti, 16GB).

InferBench fills the gaps left by existing benchmarks ([arXiv:2601.09527](https://arxiv.org/abs/2601.09527)) by measuring what matters for production inference:

1. **Prefill vs Decode disaggregation** — stage-level latency breakdown, not just E2E
2. **KV cache optimization impact** — Prefix Caching ON/OFF with cost analysis
3. **Cross-engine comparison** — vLLM vs SGLang under identical conditions
4. **Korean language models** — EXAONE-Deep-7.8B on Blackwell (first known benchmark)

## Hardware

| Component | Spec |
|---|---|
| GPU | NVIDIA RTX 5070 Ti 16GB GDDR7 (SM_120, Blackwell) |
| CPU | AMD Ryzen 9 9900X (16C/32T) |
| CUDA | 12.8+ required |
| Usable VRAM | ~14.5GB |

## Key Experiments

### Experiment 1: Quantization Ladder
BF16 → AWQ (W4A16) → NVFP4 → MXFP4 performance/quality tradeoff on Qwen3-8B.

### Experiment 2: Prefill/Decode Disaggregation
Stage-level latency measurement via streaming TTFT extraction. Analyzes why NVIDIA built disaggregated inference (GTC 2026).

### Experiment 3: KV Cache Economics
Context length scaling (2k → 16k) cost analysis with Prefix Caching ON/OFF.

## Models

| Model | Params | Why |
|---|---|---|
| Qwen3-8B | 8B | Baseline (paper reference model) |
| EXAONE-Deep-7.8B | 7.8B | Korean LLM, differentiator |
| Gemma3-12B | 12B | NVFP4 required to fit 16GB |
| Llama-3.1-8B-Instruct | 8B | Industry standard |

## Metrics

- **TTFT** — Time to First Token (user-perceived latency)
- **TPS** — Tokens Per Second (throughput)
- **ITL** — Inter-Token Latency (streaming UX quality)
- **P50/P95/P99** — Tail latency percentiles
- **VRAM** — Peak GPU memory usage
- **Energy** — Power consumption per token
- **Prefill/Decode Time** — Disaggregated stage timing

## Quick Start

```bash
# Verify GPU (SM_120 support)
docker run --gpus all nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04 nvidia-smi

# Launch full stack (vLLM + Prometheus + Grafana)
docker compose up -d

# Run benchmark
docker compose run benchmark-runner --config configs/workloads.yaml
```

## Project Structure

```
inferbench/
├── src/
│   ├── bench/          # Benchmark orchestrator, metrics, workloads
│   ├── engines/        # vLLM, SGLang, TRT-LLM adapters
│   ├── optimizations/  # KV cache, quantization, disaggregated experiments
│   └── serving/        # FastAPI wrapper, Prometheus exporter
├── configs/            # Model, workload, engine YAML configs
├── dashboards/         # Grafana dashboard JSON
├── docs/               # Blog drafts and technical docs
└── tests/
```

## Blog Series

1. **"Setting Up vLLM on RTX 5070 Ti: A Blackwell SM_120 Guide"**
2. **"Disaggregated Inference on Consumer GPU: Prefill vs Decode Analysis"**
3. **"vLLM vs SGLang on Blackwell: Which Engine Wins?"**

## License

MIT

## Author

SeungByeong — [GitHub](https://github.com/seungbyeong)
