# IsaacLab-Arena OpenEnv Environment
#
# Two-layer build strategy:
#   Layer 1 (base): Isaac Sim 5.0.0 + IsaacLab + IsaacLab-Arena + pre-warmed shaders
#                   → pushed as ghcr.io/meta-pytorch/isaaclab-arena-base:5.0.0
#   Layer 2 (this): OpenEnv code on top of the pre-built base (fast iteration)
#
# Requirements:
#   - GPU: NVIDIA A10G (24GB VRAM) minimum; A100 for multi-env
#   - HF Spaces hardware: A10G GPU (~$1.10/hr)
#   - Startup time: 2-3 min (shaders pre-warmed in base), 10-15 min cold
#   - Disk: ~20-25 GB
#
# Build (env image, after base is pushed):
#   docker build -t isaaclab-arena-openenv:latest -f envs/isaaclab_arena_env/server/Dockerfile .
#
# Run (mock mode, no GPU needed):
#   docker run -e ISAACLAB_MOCK_MODE=true -p 8000:8000 isaaclab-arena-openenv:latest
#
# Run (real GPU mode):
#   docker run --gpus all -e ISAACLAB_TASK=pick_and_place -p 8000:8000 isaaclab-arena-openenv:latest

# ── Layer 1: pre-built base with Isaac Sim + IsaacLab + IsaacLab-Arena ─────────
# Build the base image separately (30-60 min) and push to registry:
#   docker build -f envs/isaaclab_arena_env/server/Dockerfile.base -t ghcr.io/meta-pytorch/isaaclab-arena-base:5.0.0 .
#   docker push ghcr.io/meta-pytorch/isaaclab-arena-base:5.0.0
FROM ghcr.io/meta-pytorch/isaaclab-arena-base:5.0.0

# ── Layer 2: OpenEnv environment code (fast to iterate) ────────────────────────
WORKDIR /app/isaaclab_arena_env

# Install OpenEnv and server dependencies using Isaac Sim's Python
# (all runtime execution must use /isaac-sim/python.sh)
RUN /isaac-sim/python.sh -m pip install --upgrade pip && \
    /isaac-sim/python.sh -m pip install \
    "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git" \
    fastapi>=0.115.0 \
    uvicorn>=0.24.0 \
    pydantic>=2.0.0 \
    requests>=2.31.0

# Copy environment code
COPY envs/isaaclab_arena_env/ /app/isaaclab_arena_env/

# ── Health check ───────────────────────────────────────────────────────────────
# Generous start period: shader pre-warm in base cuts this from 15min to 3min
HEALTHCHECK --interval=30s --timeout=10s --start-period=180s --retries=10 \
    CMD curl -f http://localhost:8000/health || exit 1

EXPOSE 8000

CMD ["/app/isaaclab_arena_env/server/start.sh"]
