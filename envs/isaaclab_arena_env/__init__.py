# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
IsaacLab-Arena Environment for OpenEnv.

GPU-accelerated robotics simulation via NVIDIA Isaac Sim + IsaacLab-Arena.
Supports RL training on manipulation tasks (Franka, G1 humanoid) via
the standard WebSocket API.

Supported tasks (configured via ISAACLAB_TASK env var):
    - pick_and_place (Franka, 7-DOF IK or 9-DOF joint_pos)
    - open_door (Franka)
    - press_button (Franka)
    - g1_locomanip_pick_and_place (G1 humanoid, 35-DOF)

Example:
    >>> from isaaclab_arena_env import IsaacLabArenaEnv, IsaacLabArenaAction
    >>>
    >>> with IsaacLabArenaEnv(base_url="http://localhost:8000") as env:
    ...     result = env.reset()
    ...     obs_keys = list(result.observation.observations.keys())
    ...     print(f"Observation groups: {obs_keys}")
    ...     action = IsaacLabArenaAction(values=[0.0] * 7)
    ...     result = env.step(action)
    ...     print(f"Reward: {result.reward}")
"""

from .client import IsaacLabArenaEnv
from .models import IsaacLabArenaAction, IsaacLabArenaObservation, IsaacLabArenaState

__all__ = [
    "IsaacLabArenaEnv",
    "IsaacLabArenaAction",
    "IsaacLabArenaObservation",
    "IsaacLabArenaState",
]
