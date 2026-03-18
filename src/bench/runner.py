"""InferBench benchmark runner.

CLI entrypoint for running inference benchmarks. Orchestrates workload
generation, request execution, metrics collection, and report generation.

Usage:
    python -m src.bench.runner --config configs/workloads.yaml
    python -m src.bench.runner --quick  # Quick smoke test
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from src.bench.metrics import RequestMetrics, Timer, aggregate_metrics
from src.optimizations.disaggregated import measure_single_request

app = typer.Typer(help="InferBench: LLM Inference Benchmarking CLI")
console = Console()


async def _run_quick_benchmark(
    server_url: str,
    model: str,
    num_requests: int,
    max_tokens: int,
) -> None:
    """Run a quick smoke-test benchmark and print results.

    Sends sequential requests to measure baseline TTFT and TPS
    without concurrency. Useful for verifying server is running
    and getting initial numbers.

    Args:
        server_url: Base URL of the inference server.
        model: Model name for API requests.
        num_requests: Number of sequential requests.
        max_tokens: Max tokens per request.
    """
    console.print(f"\n[bold]InferBench Quick Benchmark[/bold]")
    console.print(f"Server: {server_url}")
    console.print(f"Model: {model}")
    console.print(f"Requests: {num_requests} (sequential)\n")

    prompts = [
        "Explain how KV cache works in transformer inference.",
        "What is the difference between prefill and decode in LLM serving?",
        "Describe the benefits of quantization for inference optimization.",
        "How does continuous batching improve GPU utilization?",
        "What is prefix caching and when is it useful?",
    ]

    results: list[RequestMetrics] = []
    timer = Timer()
    timer.start()

    for i in range(num_requests):
        prompt = prompts[i % len(prompts)]
        console.print(f"  Request {i + 1}/{num_requests}...", end=" ")

        try:
            result = await measure_single_request(
                prompt=prompt,
                max_tokens=max_tokens,
                server_url=server_url,
                model=model,
            )
            metrics = result.request_metrics
            results.append(metrics)
            console.print(
                f"TTFT={metrics.ttft_ms:.0f}ms  "
                f"TPS={metrics.tps:.1f}  "
                f"tokens={metrics.output_tokens}  "
                f"prefill_ratio={result.prefill_ratio:.1%}"
            )
        except Exception as e:
            console.print(f"[red]FAILED: {e}[/red]")

    timer.stop()

    if not results:
        console.print("\n[red]No successful requests. Check server connection.[/red]")
        return

    agg = aggregate_metrics(results, timer.elapsed_s)

    # Print summary table
    table = Table(title="\nBenchmark Summary")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Requests", str(agg.num_requests))
    table.add_row("Total Duration", f"{agg.total_duration_s:.1f}s")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("TTFT P50", f"{agg.ttft_p50_ms:.0f} ms")
    table.add_row("TTFT P95", f"{agg.ttft_p95_ms:.0f} ms")
    table.add_row("TTFT P99", f"{agg.ttft_p99_ms:.0f} ms")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("TPS (mean)", f"{agg.tps_mean:.1f} tok/s")
    table.add_row("TPS (total)", f"{agg.tps_total:.1f} tok/s")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("ITL P50", f"{agg.itl_p50_ms:.1f} ms")
    table.add_row("ITL P95", f"{agg.itl_p95_ms:.1f} ms")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("Prefill TPS (mean)", f"{agg.prefill_tps_mean:.0f} tok/s")
    table.add_row("Decode TPS (mean)", f"{agg.decode_tps_mean:.1f} tok/s")
    table.add_row("─" * 20, "─" * 15)
    table.add_row("Total Input Tokens", str(agg.total_input_tokens))
    table.add_row("Total Output Tokens", str(agg.total_output_tokens))

    console.print(table)

    # Save results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"quick_{timestamp}.json"

    result_data = {
        "benchmark_type": "quick",
        "timestamp": timestamp,
        "server_url": server_url,
        "model": model,
        "num_requests": num_requests,
        "max_tokens": max_tokens,
        "summary": {
            "ttft_p50_ms": agg.ttft_p50_ms,
            "ttft_p95_ms": agg.ttft_p95_ms,
            "ttft_p99_ms": agg.ttft_p99_ms,
            "tps_mean": agg.tps_mean,
            "tps_total": agg.tps_total,
            "itl_p50_ms": agg.itl_p50_ms,
            "prefill_tps_mean": agg.prefill_tps_mean,
            "decode_tps_mean": agg.decode_tps_mean,
        },
        "per_request": [
            {
                "ttft_ms": r.ttft_ms,
                "total_time_ms": r.total_time_ms,
                "tps": r.tps,
                "input_tokens": r.input_tokens,
                "output_tokens": r.output_tokens,
            }
            for r in results
        ],
    }

    output_file.write_text(json.dumps(result_data, indent=2))
    console.print(f"\nResults saved to: {output_file}")


@app.command()
def quick(
    server_url: str = typer.Option(
        "http://localhost:8000",
        "--server",
        "-s",
        help="Inference server URL",
    ),
    model: str = typer.Option(
        "default",
        "--model",
        "-m",
        help="Model name for API requests",
    ),
    num_requests: int = typer.Option(
        5,
        "--num-requests",
        "-n",
        help="Number of requests to send",
    ),
    max_tokens: int = typer.Option(
        256,
        "--max-tokens",
        "-t",
        help="Max tokens per request",
    ),
) -> None:
    """Run a quick smoke-test benchmark (sequential requests)."""
    asyncio.run(_run_quick_benchmark(server_url, model, num_requests, max_tokens))


@app.command()
def run(
    config: str = typer.Option(
        "configs/workloads.yaml",
        "--config",
        "-c",
        help="Path to workload configuration YAML",
    ),
) -> None:
    """Run full benchmark suite from configuration file."""
    console.print("[yellow]Full benchmark suite not yet implemented.[/yellow]")
    console.print("Use 'inferbench quick' for a smoke test.")
    console.print(f"Config: {config}")


if __name__ == "__main__":
    app()
