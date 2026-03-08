# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
IsaacLab-Arena Environment for OpenEnv.

Wraps Isaac Sim + IsaacLab-Arena physics simulation with the OpenEnv
Environment interface for GPU-accelerated RL training.

Supports two modes:
1. Real mode: Isaac Sim GPU simulation (requires NVIDIA GPU + Isaac Sim)
2. Mock mode: Synthetic observations for CI (ISAACLAB_MOCK_MODE=true)

The thread-bridge pattern (IsaacSimBridge / MockIsaacSimBridge) is used
to forward actions from uvicorn's background thread to the sim loop on
the main thread. In mock mode, the bridge is self-contained and handles
everything directly without a main-thread sim loop.
"""

import uuid
from typing import Optional

try:
    from openenv.core.env_server.interfaces import Environment

    from ..models import (
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        IsaacLabArenaState,
        MOCK_OBS_SPECS,
    )
    from .bridge import MockIsaacSimBridge
    from .config import get_env_config
except ImportError:
    import sys
    from pathlib import Path

    from openenv.core.env_server.interfaces import Environment

    _parent = str(Path(__file__).parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from models import (
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        IsaacLabArenaState,
        MOCK_OBS_SPECS,
    )
    from server.bridge import MockIsaacSimBridge
    from server.config import get_env_config


# Try to detect whether Isaac Sim is available (only relevant for non-mock mode)
try:
    import isaacsim  # noqa: F401

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False


class IsaacLabArenaEnvironment(
    Environment[IsaacLabArenaAction, IsaacLabArenaObservation, IsaacLabArenaState]
):
    """
    OpenEnv environment wrapping IsaacLab-Arena GPU robotics simulation.

    This environment delegates all simulation calls to either a real Isaac Sim
    bridge (main-thread GPU sim loop) or a mock bridge (for CI without GPU).

    In real mode, the bridge communicates with the sim loop via thread-safe
    queues. The sim loop must be started before this environment is used
    (see server/main.py).

    In mock mode, the bridge is fully self-contained and runs in the same
    thread as uvicorn.

    Configuration is driven by environment variables (see server/config.py).

    Example (mock mode):
        >>> os.environ["ISAACLAB_MOCK_MODE"] = "true"
        >>> env = IsaacLabArenaEnvironment()
        >>> obs = env.reset()
        >>> for _ in range(10):
        ...     obs = env.step(IsaacLabArenaAction(values=[0.0] * 7))
    """

    # Each session gets its own environment instance (no shared state between sessions)
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(
        self,
        bridge: Optional[object] = None,
    ):
        """
        Initialize IsaacLab-Arena environment.

        Args:
            bridge: IsaacSimBridge or MockIsaacSimBridge. If None, auto-selects
                based on ISAACLAB_MOCK_MODE env var and Isaac Sim availability.
        """
        cfg = get_env_config()
        self._cfg = cfg
        self._episode_id: str = ""
        self._step_count: int = 0
        self._total_reward: float = 0.0
        self._success_count: int = 0
        self._episode_count: int = 0

        self._action_dim = cfg["action_dim"]
        self._obs_spec = MOCK_OBS_SPECS.get(cfg["embodiment"], MOCK_OBS_SPECS["franka"])

        if bridge is not None:
            self._bridge = bridge
        elif cfg["mock_mode"] or not ISAACLAB_AVAILABLE:
            self._bridge = MockIsaacSimBridge(
                embodiment=cfg["embodiment"],
                max_steps=500,
            )
        else:
            # Real mode: bridge must be injected from main.py after sim is running
            raise RuntimeError(
                "Isaac Sim is available but no bridge was injected. "
                "Use mock mode (ISAACLAB_MOCK_MODE=true) or start via server/main.py "
                "which sets up the thread-bridge."
            )

    def reset(self, seed: Optional[int] = None, **kwargs) -> IsaacLabArenaObservation:
        """Reset the environment and return the initial observation."""
        self._episode_id = str(uuid.uuid4())
        self._step_count = 0
        self._total_reward = 0.0
        obs = self._bridge.reset(seed=seed)
        return obs

    def step(self, action: IsaacLabArenaAction) -> IsaacLabArenaObservation:
        """Execute one step in the environment."""
        obs = self._bridge.step(action)
        self._step_count += 1
        if obs.reward is not None:
            self._total_reward += obs.reward
        if obs.success:
            self._success_count += 1
        if obs.terminated or obs.truncated:
            self._episode_count += 1
        return obs

    @property
    def state(self) -> IsaacLabArenaState:
        """Return current environment state and metadata."""
        success_rate = (
            self._success_count / self._episode_count
            if self._episode_count > 0
            else 0.0
        )
        action_bounds = {
            "low": [-1.0] * self._action_dim,
            "high": [1.0] * self._action_dim,
        }
        observation_spec = {group: dim for group, dim in self._obs_spec.items()}
        return IsaacLabArenaState(
            episode_id=self._episode_id,
            step_count=self._step_count,
            task_name=self._cfg["task"],
            embodiment_name=self._cfg["embodiment"],
            action_mode=self._cfg["action_mode"],
            action_dim=self._action_dim,
            action_bounds=action_bounds,
            observation_spec=observation_spec,
            total_reward=self._total_reward,
            success_rate=success_rate,
        )
