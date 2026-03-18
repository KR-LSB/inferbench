"""Tests for the metrics module."""

from __future__ import annotations

from src.bench.metrics import RequestMetrics, aggregate_metrics, _percentile


class TestRequestMetrics:
    """Tests for individual request metric calculations."""

    def test_tps_calculation(self) -> None:
        m = RequestMetrics(
            ttft_ms=100.0,
            total_time_ms=1100.0,
            input_tokens=50,
            output_tokens=100,
        )
        # decode_time = 1100 - 100 = 1000ms = 1s
        # tps = 100 / 1 = 100
        assert m.tps == 100.0

    def test_tps_zero_when_no_output(self) -> None:
        m = RequestMetrics(ttft_ms=100.0, total_time_ms=200.0, output_tokens=0)
        assert m.tps == 0.0

    def test_itl_calculation(self) -> None:
        m = RequestMetrics(
            token_timestamps_ms=[100.0, 110.0, 120.0, 130.0],
        )
        assert m.itl_ms == 10.0

    def test_itl_empty_timestamps(self) -> None:
        m = RequestMetrics()
        assert m.itl_ms == 0.0

    def test_prefill_tokens_per_sec(self) -> None:
        m = RequestMetrics(ttft_ms=200.0, input_tokens=1000)
        # 1000 / 0.2s = 5000 tok/s
        assert m.prefill_tokens_per_sec == 5000.0

    def test_decode_time_ms(self) -> None:
        m = RequestMetrics(ttft_ms=100.0, total_time_ms=500.0)
        assert m.decode_time_ms == 400.0

    def test_decode_tokens_per_sec(self) -> None:
        m = RequestMetrics(
            ttft_ms=100.0,
            total_time_ms=1100.0,
            output_tokens=50,
        )
        # decode_time = 1000ms = 1s, 50 tokens / 1s = 50
        assert m.decode_tokens_per_sec == 50.0


class TestPercentile:
    """Tests for percentile calculation."""

    def test_p50(self) -> None:
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        assert _percentile(data, 50) == 3.0

    def test_p95_small_dataset(self) -> None:
        data = [10.0, 20.0, 30.0]
        result = _percentile(data, 95)
        assert 28.0 <= result <= 30.0

    def test_empty_data(self) -> None:
        assert _percentile([], 50) == 0.0

    def test_single_element(self) -> None:
        assert _percentile([42.0], 99) == 42.0


class TestAggregateMetrics:
    """Tests for metrics aggregation."""

    def test_basic_aggregation(self) -> None:
        requests = [
            RequestMetrics(
                ttft_ms=100.0,
                total_time_ms=1000.0,
                input_tokens=50,
                output_tokens=100,
            ),
            RequestMetrics(
                ttft_ms=200.0,
                total_time_ms=2000.0,
                input_tokens=50,
                output_tokens=200,
            ),
        ]
        agg = aggregate_metrics(requests, total_duration_s=3.0)

        assert agg.num_requests == 2
        assert agg.total_duration_s == 3.0
        assert agg.total_input_tokens == 100
        assert agg.total_output_tokens == 300
        assert agg.tps_total == 100.0  # 300 tokens / 3s

    def test_empty_requests(self) -> None:
        agg = aggregate_metrics([], total_duration_s=0.0)
        assert agg.num_requests == 0
        assert agg.ttft_p50_ms == 0.0
