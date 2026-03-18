"""Tests for the workloads module."""

from __future__ import annotations

from src.bench.workloads import WorkloadConfig, generate_requests, _build_prompt


class TestBuildPrompt:
    """Tests for prompt generation."""

    def test_prompt_not_empty(self) -> None:
        prompt = _build_prompt(256, "You are a helpful assistant.")
        assert len(prompt) > 0
        assert "Context:" in prompt
        assert "Question:" in prompt

    def test_prompt_scales_with_target(self) -> None:
        short = _build_prompt(256, "")
        long = _build_prompt(8192, "")
        assert len(long) > len(short)

    def test_prompt_includes_question(self) -> None:
        prompt = _build_prompt(512, "")
        assert "?" in prompt


class TestGenerateRequests:
    """Tests for request generation."""

    def test_generates_correct_count(self) -> None:
        config = WorkloadConfig(
            name="test",
            description="Test workload",
            context_length=256,
            max_output_tokens=64,
            concurrency_levels=[2, 4],
            num_requests_per_level=10,
            system_prompt="You are a test assistant.",
        )
        requests = generate_requests(config)
        # max(concurrency_levels) * num_requests_per_level = 4 * 10 = 40
        assert len(requests) == 40

    def test_request_fields(self) -> None:
        config = WorkloadConfig(
            name="api-short",
            description="Short API test",
            context_length=128,
            max_output_tokens=32,
            concurrency_levels=[1],
            num_requests_per_level=3,
            system_prompt="Test.",
        )
        requests = generate_requests(config)
        assert all(r.workload_name == "api-short" for r in requests)
        assert all(r.max_tokens == 32 for r in requests)
        assert all(r.system_prompt == "Test." for r in requests)
