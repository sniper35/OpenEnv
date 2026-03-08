# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
FastAPI application factory for the IsaacLab-Arena Environment.

In mock mode (ISAACLAB_MOCK_MODE=true), can be launched directly via uvicorn:
    ISAACLAB_MOCK_MODE=true uvicorn server.app:app --host 0.0.0.0 --port 8000

In real mode, this app is started by main.py on a background thread after
the Isaac Sim main-thread loop is initialized.

Usage:
    # Mock mode (CI, no GPU):
    ISAACLAB_MOCK_MODE=true uvicorn envs.isaaclab_arena_env.server.app:app

    # Real mode (via main.py):
    /isaac-sim/python.sh envs/isaaclab_arena_env/server/main.py
"""

try:
    from openenv.core.env_server.http_server import create_app

    from ..models import IsaacLabArenaAction, IsaacLabArenaObservation
    from .environment import IsaacLabArenaEnvironment
except ImportError:
    import sys
    from pathlib import Path

    from openenv.core.env_server.http_server import create_app

    _parent = str(Path(__file__).parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from models import IsaacLabArenaAction, IsaacLabArenaObservation
    from server.environment import IsaacLabArenaEnvironment


def create_mock_app():
    """Create app in mock mode (no GPU required)."""
    return create_app(
        IsaacLabArenaEnvironment,
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        env_name="isaaclab_arena_env",
    )


def create_bridged_app(bridge):
    """
    Create app with an injected IsaacSimBridge (real GPU mode).

    Called from main.py after the sim loop is initialized on the main thread.
    The bridge is pre-constructed with the action/obs queues.
    """

    def environment_factory():
        return IsaacLabArenaEnvironment(bridge=bridge)

    return create_app(
        environment_factory,
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        env_name="isaaclab_arena_env",
    )


# Default app instance: auto-detects mock vs real mode
# In real mode, main.py replaces this with create_bridged_app(bridge)
app = create_mock_app()


def main():
    """Entry point for mock-mode direct execution."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
