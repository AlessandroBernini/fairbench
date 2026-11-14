# FairBench multi-stage Docker image
#
# Usage:
#   docker build -t fairbench .
#   docker run --rm -v ~/models:/app/models:ro fairbench verify
#   docker run --rm --gpus all \
#     -v ~/models:/app/models:ro \
#     -v ~/.fairbench:/root/.fairbench \
#     fairbench run /app/models/model.gguf --category all --lite

FROM python:3.12-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir --prefix=/install .

# Final stage
FROM python:3.12-slim

LABEL org.opencontainers.image.source="https://github.com/AlessandroBernini/fairbench"
LABEL org.opencontainers.image.description="FairBench — AI model benchmarking suite"

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas-dev \
    liblapack-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY --from=builder /install /usr/local
COPY . /app

# Create data directories
RUN mkdir -p /root/.fairbench/results/cache /app/models

CMD ["fairbench", "verify"]
