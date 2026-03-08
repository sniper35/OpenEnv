# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the IsaacLab-Arena OpenEnv Environment.

Wraps IsaacLab-Arena (NVIDIA Isaac Sim) physics-based robotics simulation
with the OpenEnv interface for GPU-accelerated RL training.
"""

from typing import Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv.core.env_server.types import Action, Observation, State


class IsaacLabArenaAction(Action):
    """
    Action for IsaacLab-Arena environments.

    Action dimensions by embodiment and mode:
      - Franka IK (default): 7 (3 EEF pos + 3 EEF rot + 1 gripper)
      - Franka joint_pos: 9 (7 arm joints + 2 gripper)
      - G1 humanoid: 35 (full-body joint positions)

    Example (Franka IK):
        >>> action = IsaacLabArenaAction(values=[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0])
    """

    values: List[float] = Field(
        default_factory=list,
        description="Flat action vector. Dim depends on embodiment+mode (7 Franka IK, 9 Franka joint, 35 G1)",
    )


class IsaacLabArenaObservation(Observation):
    """
    Observation from IsaacLab-Arena environments.

    Observations are organized as named groups matching IsaacLab's
    obs_buf["policy"] structure. Each group is a flat list of floats.

    Common groups (Franka pick-and-place):
      - "joint_pos": Robot joint positions (9,)
      - "joint_vel": Robot joint velocities (9,)
      - "eef_pos": End-effector position (3,)
      - "eef_quat": End-effector quaternion (4,)
      - "object_pos": Target object position (3,)
      - "object_quat": Target object quaternion (4,)
      - "goal_pos": Goal/destination position (3,)

    Example:
        >>> obs = IsaacLabArenaObservation(
        ...     observations={"joint_pos": [0.0]*9, "eef_pos": [0.5, 0.0, 0.4]},
        ...     reward=0.1,
        ...     terminated=False,
        ...     truncated=False,
        ...     success=False,
        ... )
    """

    observations: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Named observation groups from obs_buf['policy']. Each value is a flat list of floats.",
    )
    terminated: bool = Field(
        default=False,
        description="Whether the episode ended due to a terminal condition (success or failure)",
    )
    truncated: bool = Field(
        default=False,
        description="Whether the episode was truncated (e.g., max steps reached)",
    )
    success: bool = Field(
        default=False,
        description="Whether the task was completed successfully",
    )
    rgb_image: Optional[str] = Field(
        default=None,
        description="Base64-encoded PNG camera image. Only present when ISAACLAB_ENABLE_CAMERA=true",
    )


class IsaacLabArenaState(State):
    """
    Extended state for IsaacLab-Arena environments.

    Provides metadata about the current task, embodiment, action space,
    and episode statistics.
    """

    task_name: str = Field(
        default="pick_and_place",
        description="Active task name (pick_and_place, open_door, press_button, g1_locomanip_pick_and_place)",
    )
    embodiment_name: str = Field(
        default="franka",
        description="Active embodiment (franka, g1)",
    )
    action_mode: str = Field(
        default="ik",
        description="Action mode: 'ik' (end-effector) or 'joint_pos'",
    )
    action_dim: int = Field(
        default=7,
        description="Action vector dimension",
    )
    action_bounds: Dict[str, List[float]] = Field(
        default_factory=dict,
        description="Action bounds: {'low': [...], 'high': [...]}",
    )
    observation_spec: Dict[str, int] = Field(
        default_factory=dict,
        description="Observation group sizes: {'joint_pos': 9, 'eef_pos': 3, ...}",
    )
    total_reward: float = Field(
        default=0.0,
        description="Cumulative reward for the current episode",
    )
    success_rate: float = Field(
        default=0.0,
        description="Running success rate over completed episodes",
    )


# Mock observation specs by embodiment (group_name -> dim)
MOCK_OBS_SPECS = {
    "franka": {
        "joint_pos": 9,
        "joint_vel": 9,
        "eef_pos": 3,
        "eef_quat": 4,
        "object_pos": 3,
        "object_quat": 4,
        "goal_pos": 3,
    },
    "g1": {
        "joint_pos": 35,
        "joint_vel": 35,
        "base_lin_vel": 3,
        "base_ang_vel": 3,
        "projected_gravity": 3,
        "object_pos": 3,
        "goal_pos": 3,
    },
}

# Task registry: task_name -> (supported_embodiments, default_action_dim_by_mode)
TASK_REGISTRY = {
    "pick_and_place": {
        "embodiments": ["franka"],
        "action_dims": {"ik": 7, "joint_pos": 9},
        "default_scene": "kitchen",
    },
    "open_door": {
        "embodiments": ["franka"],
        "action_dims": {"ik": 7, "joint_pos": 9},
        "default_scene": "kitchen",
    },
    "press_button": {
        "embodiments": ["franka"],
        "action_dims": {"ik": 7, "joint_pos": 9},
        "default_scene": "kitchen",
    },
    "g1_locomanip_pick_and_place": {
        "embodiments": ["g1"],
        "action_dims": {"joint_pos": 35},
        "default_scene": "kitchen",
    },
}
