#!/usr/bin/env python3
"""
Experiment 5: EXAONE vs Qwen3 Results Comparison

Loads benchmark JSON results and generates:
1. Console comparison table
2. Markdown report (docs/exaone_comparison.md)

Usage:
  python scripts/compare_exaone_qwen.py \
    --exaone results/exaone_bench_vllm_*.json \
    --qwen results/exaone_bench_vllm_*.json  # or use Qwen3 baseline data

  # Or use hardcoded Qwen3 baseline (from context v3):
  python scripts/compare_exaone_qwen.py --exaone results/exaone_bench_vllm_*.json --use-baseline
"""

import argparse
import json
import sys
from pathlib import Path

# Qwen3-8B-AWQ baseline from Experiment 4 (engine_bench.py, vLLM)
QWEN3_VLLM_BASELINE = {
    1:  {"agg_tps": 121.5, "ttft_p50_ms": 15.4, "ttft_p95_ms": 15.8, "decode_tps_mean": 121.5},
    8:  {"agg_tps": 551.7, "ttft_p50_ms": 37.3, "ttft_p95_ms": 37.8, "decode_tps_mean": 70.0},
    16: {"agg_tps": 968.8, "ttft_p50_ms": 40.5, "ttft_p95_ms": 42.0, "decode_tps_mean": 62.0},
    32: {"agg_tps": 969.4, "ttft_p50_ms": 42.2, "ttft_p95_ms": 45.0, "decode_tps_mean": 31.0},
}

QWEN3_SGLANG_BASELINE = {
    1:  {"agg_tps": 130.5, "ttft_p50_ms": 20.0, "ttft_p95_ms": 25.0, "decode_tps_mean": 130.5},
    8:  {"agg_tps": 810.2, "ttft_p50_ms": 37.4, "ttft_p95_ms": 299.0, "decode_tps_mean": 102.0},
    16: {"agg_tps": 978.4, "ttft_p50_ms": 69.2, "ttft_p95_ms": 75.0, "decode_tps_mean": 62.0},
    32: {"agg_tps": 1027.8, "ttft_p50_ms": 39.7, "ttft_p95_ms": 55.0, "decode_tps_mean": 33.0},
}


def load_results(path: str) -> dict:
    """Load a benchmark result JSON file."""
    with open(path) as f:
        return json.load(f)


def format_delta(exaone_val: float, qwen_val: float) -> str:
    """Format the percentage delta between EXAONE and Qwen3."""
    if qwen_val == 0:
        return "N/A"
    delta = ((exaone_val - qwen_val) / qwen_val) * 100
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.1f}%"


def print_comparison(exaone_data: dict, engine: str) -> str:
    """Print and return markdown comparison table."""
    baseline = QWEN3_VLLM_BASELINE if engine == "vllm" else QWEN3_SGLANG_BASELINE
    exaone_results = {r["concurrency"]: r for r in exaone_data["results"]}

    lines: list[str] = []

    header = (
        f"## EXAONE-Deep-7.8B vs Qwen3-8B — {engine.upper()} on RTX 5070 Ti"
    )
    lines.append(header)
    lines.append("")
    lines.append(
        "| c | Model | Agg TPS | TTFT P50 (ms) | TTFT P95 (ms) | Decode TPS |"
    )
    lines.append(
        "|--:|-------|--------:|--------------:|--------------:|-----------:|"
    )

    for c in sorted(set(list(baseline.keys()) + list(exaone_results.keys()))):
        q = baseline.get(c)
        e = exaone_results.get(c)

        if q:
            lines.append(
                f"| {c} | Qwen3-8B | {q['agg_tps']:,.1f} | "
                f"{q['ttft_p50_ms']:.1f} | {q['ttft_p95_ms']:.1f} | "
                f"{q['decode_tps_mean']:.1f} |"
            )
        if e:
            delta_tps = format_delta(e["agg_tps"], q["agg_tps"]) if q else "—"
            delta_ttft = format_delta(e["ttft_p50_ms"], q["ttft_p50_ms"]) if q else "—"
            lines.append(
                f"| {c} | **EXAONE-7.8B** | **{e['agg_tps']:,.1f}** ({delta_tps}) | "
                f"**{e['ttft_p50_ms']:.1f}** ({delta_ttft}) | "
                f"**{e['ttft_p95_ms']:.1f}** | "
                f"**{e['decode_tps_mean']:.1f}** |"
            )
        lines.append(
            f"|   |       |         |               |               |            |"
        )

    md = "\n".join(lines)
    print(md)
    return md


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare EXAONE vs Qwen3 results")
    parser.add_argument("--exaone", required=True, help="EXAONE results JSON path")
    parser.add_argument("--engine", choices=["vllm", "sglang"], default="vllm")
    parser.add_argument(
        "--output", default="docs/exaone_comparison.md",
        help="Output markdown path"
    )
    args = parser.parse_args()

    exaone_data = load_results(args.exaone)
    engine = args.engine or exaone_data.get("engine", "vllm")

    print("=" * 70)
    print(f"  EXAONE vs Qwen3 Comparison — {engine.upper()}")
    print("=" * 70)

    md = print_comparison(exaone_data, engine)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Experiment 5: EXAONE-Deep-7.8B Benchmark Results\n\n")
        f.write(f"**Hardware:** RTX 5070 Ti 16GB (SM_120, Blackwell)\n")
        f.write(f"**Engine:** {engine}\n")
        f.write(f"**Quantization:** AWQ (W4A16)\n\n")
        f.write(md)
        f.write("\n")

    print(f"\n  Report saved: {out_path}")


if __name__ == "__main__":
    main()
