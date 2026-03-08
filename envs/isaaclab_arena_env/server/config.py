# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Task and embodiment configuration for IsaacLab-Arena environments.

Environment variables:
    ISAACLAB_TASK: Task name (default: pick_and_place)
    ISAACLAB_EMBODIMENT: Embodiment name (default: franka)
    ISAACLAB_ACTION_MODE: Action mode - 'ik' or 'joint_pos' (default: ik)
    ISAACLAB_SCENE: Scene name (default: kitchen)
    ISAACLAB_NUM_ENVS: Number of parallel envs (default: 1)
    ISAACLAB_ENABLE_CAMERA: Enable RGB camera obs (default: false)
    ISAACLAB_MOCK_MODE: Use mock bridge without GPU (default: false)
    ISAACLAB_HEADLESS: Run headless (default: true)
"""

import os

# ── Task registry ──────────────────────────────────────────────────────────────
# Maps task_name → IsaacLab task registration string
TASK_REGISTRY = {
    "pick_and_place": "Isaac-Arena-Franka-Pick-And-Place-v0",
    "open_door": "Isaac-Arena-Franka-Open-Door-v0",
    "press_button": "Isaac-Arena-Franka-Press-Button-v0",
    "g1_locomanip_pick_and_place": "Isaac-Arena-G1-LocomanipPickAndPlace-v0",
}

# ── Embodiment registry ────────────────────────────────────────────────────────
EMBODIMENT_REGISTRY = {
    "franka": {
        "action_dims": {"ik": 7, "joint_pos": 9},
        "default_action_mode": "ik",
    },
    "g1": {
        "action_dims": {"joint_pos": 35},
        "default_action_mode": "joint_pos",
    },
}

# ── Runtime configuration ─────────────────────────────────────────────────────


def get_env_config() -> dict:
    """Read configuration from environment variables."""
    task = os.environ.get("ISAACLAB_TASK", "pick_and_place")
    embodiment = os.environ.get("ISAACLAB_EMBODIMENT", "franka")
    scene = os.environ.get("ISAACLAB_SCENE", "kitchen")

    emb_cfg = EMBODIMENT_REGISTRY.get(embodiment, EMBODIMENT_REGISTRY["franka"])
    default_mode = emb_cfg["default_action_mode"]
    action_mode = os.environ.get("ISAACLAB_ACTION_MODE", default_mode)
    action_dim = emb_cfg["action_dims"].get(
        action_mode, list(emb_cfg["action_dims"].values())[0]
    )

    return {
        "task": task,
        "embodiment": embodiment,
        "scene": scene,
        "action_mode": action_mode,
        "action_dim": action_dim,
        "num_envs": int(os.environ.get("ISAACLAB_NUM_ENVS", "1")),
        "enable_camera": os.environ.get("ISAACLAB_ENABLE_CAMERA", "false").lower()
        == "true",
        "mock_mode": os.environ.get("ISAACLAB_MOCK_MODE", "false").lower() == "true",
        "headless": os.environ.get("ISAACLAB_HEADLESS", "true").lower() == "true",
        "task_id": TASK_REGISTRY.get(task, f"Isaac-Arena-{task}-v0"),
    }
