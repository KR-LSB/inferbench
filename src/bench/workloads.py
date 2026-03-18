"""Workload generators for inference benchmarking.

Generates prompts of varying context lengths to simulate real-world
serving scenarios: RAG, chatbot API, and agentic workflows.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class WorkloadConfig:
    """Configuration for a single workload scenario."""

    name: str
    description: str
    context_length: int
    max_output_tokens: int
    concurrency_levels: list[int]
    num_requests_per_level: int
    system_prompt: str


@dataclass
class BenchmarkRequest:
    """A single benchmark request to send to the inference server."""

    prompt: str
    system_prompt: str
    max_tokens: int
    workload_name: str
    request_id: int


# Filler text fragments for building prompts of target lengths
_FILLER_PARAGRAPHS = [
    "The transformer architecture has fundamentally changed how we approach sequence modeling. "
    "Self-attention mechanisms allow the model to weigh the importance of different parts of "
    "the input when producing each element of the output.",
    "Inference optimization is critical for deploying large language models in production. "
    "Key techniques include quantization, KV cache management, continuous batching, and "
    "speculative decoding to reduce latency and improve throughput.",
    "Retrieval-augmented generation combines the strengths of parametric models with "
    "non-parametric retrieval systems. By grounding responses in retrieved documents, "
    "RAG systems can provide more factual and up-to-date information.",
    "The KV cache stores key and value tensors from previous tokens to avoid redundant "
    "computation during autoregressive generation. Managing this cache efficiently is "
    "essential for serving long-context workloads within memory constraints.",
    "Prefix caching allows reuse of computed KV cache entries across requests that share "
    "a common prefix, such as a system prompt. This dramatically reduces time-to-first-token "
    "for workloads with repeated context.",
    "Disaggregated inference separates the prefill (input processing) and decode (token "
    "generation) phases onto different hardware optimized for each task. Prefill is "
    "compute-bound while decode is memory-bandwidth-bound.",
    "Quantization reduces model precision from 16-bit to 8-bit or 4-bit representations. "
    "Blackwell GPUs natively support FP4 computation through dedicated tensor cores, "
    "enabling significant throughput improvements with minimal quality degradation.",
    "Continuous batching dynamically adds and removes requests from a running batch as "
    "they arrive and complete. This prevents the batch from being limited by the slowest "
    "request and significantly improves GPU utilization.",
]


def _build_prompt(target_token_count: int, system_prompt: str) -> str:
    """Build a prompt of approximately the target token count.

    Uses a simple heuristic: ~1.3 tokens per word for English text.

    Args:
        target_token_count: Approximate number of tokens to generate.
        system_prompt: System prompt (its tokens are subtracted from target).

    Returns:
        A prompt string of approximately the target length.
    """
    system_tokens_est = len(system_prompt.split()) * 1.3
    remaining_tokens = max(64, target_token_count - int(system_tokens_est))
    target_words = int(remaining_tokens / 1.3)

    paragraphs: list[str] = []
    word_count = 0
    while word_count < target_words:
        para = random.choice(_FILLER_PARAGRAPHS)
        paragraphs.append(para)
        word_count += len(para.split())

    context = "\n\n".join(paragraphs)

    # Trim to approximate target
    words = context.split()
    if len(words) > target_words:
        context = " ".join(words[:target_words])

    question = random.choice([
        "Based on the context above, summarize the key technical concepts discussed.",
        "What are the main optimization techniques mentioned in the text?",
        "Explain the relationship between the concepts described above.",
        "What are the tradeoffs discussed in the context?",
    ])

    return f"Context:\n{context}\n\nQuestion: {question}"


def load_workloads(config_path: str | Path) -> list[WorkloadConfig]:
    """Load workload configurations from YAML file.

    Args:
        config_path: Path to the workloads YAML configuration.

    Returns:
        List of WorkloadConfig objects.
    """
    with open(config_path) as f:
        raw = yaml.safe_load(f)

    return [
        WorkloadConfig(
            name=w["name"],
            description=w["description"],
            context_length=w["context_length"],
            max_output_tokens=w["max_output_tokens"],
            concurrency_levels=w["concurrency_levels"],
            num_requests_per_level=w["num_requests_per_level"],
            system_prompt=w["system_prompt"],
        )
        for w in raw["workloads"]
    ]


def generate_requests(workload: WorkloadConfig) -> list[BenchmarkRequest]:
    """Generate benchmark requests for a given workload configuration.

    Args:
        workload: The workload configuration.

    Returns:
        List of BenchmarkRequest objects ready to send.
    """
    max_requests = max(workload.concurrency_levels) * workload.num_requests_per_level
    requests = []

    for i in range(max_requests):
        prompt = _build_prompt(workload.context_length, workload.system_prompt)
        requests.append(
            BenchmarkRequest(
                prompt=prompt,
                system_prompt=workload.system_prompt,
                max_tokens=workload.max_output_tokens,
                workload_name=workload.name,
                request_id=i,
            )
        )

    return requests
