"""Inference metrics collection and aggregation.

Measures TTFT, TPS, ITL, percentile latencies, VRAM usage, and energy
consumption for LLM inference benchmarking.
"""

from __future__ import annotations

import statistics
import time
from dataclasses import dataclass, field


@dataclass
class RequestMetrics:
    """Metrics for a single inference request."""

    ttft_ms: float = 0.0
    total_time_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    token_timestamps_ms: list[float] = field(default_factory=list)

    @property
    def tps(self) -> float:
        """Output tokens per second."""
        if self.total_time_ms <= 0 or self.output_tokens <= 0:
            return 0.0
        decode_time_s = (self.total_time_ms - self.ttft_ms) / 1000
        return self.output_tokens / decode_time_s if decode_time_s > 0 else 0.0

    @property
    def itl_ms(self) -> float:
        """Average inter-token latency in milliseconds."""
        if len(self.token_timestamps_ms) < 2:
            return 0.0
        deltas = [
            self.token_timestamps_ms[i] - self.token_timestamps_ms[i - 1]
            for i in range(1, len(self.token_timestamps_ms))
        ]
        return statistics.mean(deltas)

    @property
    def prefill_tokens_per_sec(self) -> float:
        """Input tokens processed per second during prefill."""
        if self.ttft_ms <= 0 or self.input_tokens <= 0:
            return 0.0
        return self.input_tokens / (self.ttft_ms / 1000)

    @property
    def decode_time_ms(self) -> float:
        """Time spent in decode phase (total - prefill)."""
        return max(0.0, self.total_time_ms - self.ttft_ms)

    @property
    def decode_tokens_per_sec(self) -> float:
        """Output tokens generated per second during decode."""
        decode_s = self.decode_time_ms / 1000
        if decode_s <= 0 or self.output_tokens <= 0:
            return 0.0
        return self.output_tokens / decode_s


@dataclass
class AggregatedMetrics:
    """Aggregated metrics across multiple requests."""

    num_requests: int = 0
    total_duration_s: float = 0.0

    ttft_p50_ms: float = 0.0
    ttft_p95_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    ttft_mean_ms: float = 0.0

    tps_mean: float = 0.0
    tps_total: float = 0.0

    itl_p50_ms: float = 0.0
    itl_p95_ms: float = 0.0
    itl_p99_ms: float = 0.0

    e2e_p50_ms: float = 0.0
    e2e_p95_ms: float = 0.0
    e2e_p99_ms: float = 0.0

    prefill_tps_mean: float = 0.0
    decode_tps_mean: float = 0.0

    total_input_tokens: int = 0
    total_output_tokens: int = 0


def _percentile(data: list[float], p: float) -> float:
    """Calculate percentile from sorted data."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[f]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


def aggregate_metrics(
    request_metrics: list[RequestMetrics],
    total_duration_s: float,
) -> AggregatedMetrics:
    """Aggregate individual request metrics into summary statistics.

    Args:
        request_metrics: List of per-request measurements.
        total_duration_s: Wall-clock duration of the entire benchmark run.

    Returns:
        Aggregated summary with percentiles and means.
    """
    if not request_metrics:
        return AggregatedMetrics()

    ttfts = [r.ttft_ms for r in request_metrics if r.ttft_ms > 0]
    tps_values = [r.tps for r in request_metrics if r.tps > 0]
    itls = [r.itl_ms for r in request_metrics if r.itl_ms > 0]
    e2es = [r.total_time_ms for r in request_metrics if r.total_time_ms > 0]
    prefill_tps = [r.prefill_tokens_per_sec for r in request_metrics if r.prefill_tokens_per_sec > 0]
    decode_tps = [r.decode_tokens_per_sec for r in request_metrics if r.decode_tokens_per_sec > 0]

    total_output = sum(r.output_tokens for r in request_metrics)

    return AggregatedMetrics(
        num_requests=len(request_metrics),
        total_duration_s=total_duration_s,
        ttft_p50_ms=_percentile(ttfts, 50),
        ttft_p95_ms=_percentile(ttfts, 95),
        ttft_p99_ms=_percentile(ttfts, 99),
        ttft_mean_ms=statistics.mean(ttfts) if ttfts else 0.0,
        tps_mean=statistics.mean(tps_values) if tps_values else 0.0,
        tps_total=total_output / total_duration_s if total_duration_s > 0 else 0.0,
        itl_p50_ms=_percentile(itls, 50),
        itl_p95_ms=_percentile(itls, 95),
        itl_p99_ms=_percentile(itls, 99),
        e2e_p50_ms=_percentile(e2es, 50),
        e2e_p95_ms=_percentile(e2es, 95),
        e2e_p99_ms=_percentile(e2es, 99),
        prefill_tps_mean=statistics.mean(prefill_tps) if prefill_tps else 0.0,
        decode_tps_mean=statistics.mean(decode_tps) if decode_tps else 0.0,
        total_input_tokens=sum(r.input_tokens for r in request_metrics),
        total_output_tokens=total_output,
    )


class Timer:
    """Simple high-resolution timer for benchmarking."""

    def __init__(self) -> None:
        self._start: float = 0.0
        self._end: float = 0.0

    def start(self) -> None:
        self._start = time.perf_counter()

    def stop(self) -> None:
        self._end = time.perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (self._end - self._start) * 1000

    @property
    def elapsed_s(self) -> float:
        return self._end - self._start
