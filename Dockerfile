# InferBench Dockerfile
# Optimized for NVIDIA RTX 5070 Ti (SM_120, Blackwell architecture)
# Requires: CUDA 12.8+, Docker with NVIDIA runtime
#
# Build:  docker build -t inferbench .
# Run:    docker run --gpus all -p 8000:8000 inferbench

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

# SM_120 (Blackwell) support
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV CUDA_HOME=/usr/local/cuda
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3-pip \
    git \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 12.8 support
RUN pip install --break-system-packages --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cu128

# Install vLLM (SM_120 compatible)
RUN pip install --break-system-packages --no-cache-dir vllm

# Install InferBench dependencies
RUN pip install --break-system-packages --no-cache-dir \
    httpx \
    fastapi \
    uvicorn \
    prometheus-client \
    pyyaml \
    pandas \
    matplotlib \
    seaborn \
    rich \
    typer

WORKDIR /app
COPY . /app

# Expose ports: 8000 (vLLM API), 8001 (SGLang API), 9100 (metrics)
EXPOSE 8000 8001 9100

# Default: run quick benchmark
CMD ["python3", "-m", "src.bench.runner", "quick"]
