# Default image keeps the environment lightweight and mock-capable.
# Override BASE_IMAGE with an Isaac Sim image to enable the real runtime in-container:
#   docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0 -t isaaclab-arena-env .

ARG BASE_IMAGE=python:3.11-slim
FROM ${BASE_IMAGE}

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY . /app/env

WORKDIR /app/env

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -e .

ENV PYTHONPATH="/app/env:${PYTHONPATH}"
ENV ISAACLAB_ARENA_RUNTIME_MODE="mock"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["sh", "-c", "cd /app/env && uvicorn server.app:app --host 0.0.0.0 --port 8000"]
