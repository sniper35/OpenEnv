#!/usr/bin/env bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Startup script for IsaacLab-Arena OpenEnv server.
# Runtime EULA acceptance and environment setup (cannot be in Dockerfile ENV).

set -euo pipefail

# ── EULA acknowledgment ────────────────────────────────────────────────────────
# Must be set at runtime, not Dockerfile build time, per NVIDIA policy
export ACCEPT_EULA=Y
export OMNI_KIT_ACCEPT_EULA=YES

# ── Isaac Sim cache directories ────────────────────────────────────────────────
export OMNI_USER_DATA_PATH="${OMNI_USER_DATA_PATH:-/tmp/omni_data}"
export OMNI_CACHE_PATH="${OMNI_CACHE_PATH:-/tmp/omni_cache}"
mkdir -p "${OMNI_USER_DATA_PATH}" "${OMNI_CACHE_PATH}"

# ── Task configuration (with defaults) ────────────────────────────────────────
export ISAACLAB_TASK="${ISAACLAB_TASK:-pick_and_place}"
export ISAACLAB_EMBODIMENT="${ISAACLAB_EMBODIMENT:-franka}"
export ISAACLAB_ACTION_MODE="${ISAACLAB_ACTION_MODE:-ik}"
export ISAACLAB_SCENE="${ISAACLAB_SCENE:-kitchen}"
export ISAACLAB_NUM_ENVS="${ISAACLAB_NUM_ENVS:-1}"
export ISAACLAB_HEADLESS="${ISAACLAB_HEADLESS:-true}"
export ISAACLAB_ENABLE_CAMERA="${ISAACLAB_ENABLE_CAMERA:-false}"
export ISAACLAB_MOCK_MODE="${ISAACLAB_MOCK_MODE:-false}"

echo "=== IsaacLab-Arena OpenEnv Server ==="
echo "  Task:       ${ISAACLAB_TASK}"
echo "  Embodiment: ${ISAACLAB_EMBODIMENT}"
echo "  Action mode:${ISAACLAB_ACTION_MODE}"
echo "  Mock mode:  ${ISAACLAB_MOCK_MODE}"
echo "======================================="

if [ "${ISAACLAB_MOCK_MODE}" = "true" ]; then
    # Mock mode: use standard Python (no Isaac Sim needed)
    echo "Starting in mock mode (no GPU required)..."
    exec python3 /app/isaaclab_arena_env/server/main.py
else
    # Real mode: must use Isaac Sim's bundled Python runtime
    echo "Starting Isaac Sim simulation (GPU required)..."
    exec /isaac-sim/python.sh /app/isaaclab_arena_env/server/main.py
fi
