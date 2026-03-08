# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tensor-to-Pydantic conversion utilities for IsaacLab observations.

IsaacLab's ManagerBasedRLEnv returns observations as a dict of named groups,
each being a (num_envs, dim) tensor. This module converts them to the
Dict[str, List[float]] format used by IsaacLabArenaObservation.

Supports single-env (num_envs=1) inference by squeezing the batch dimension.
"""

from typing import Dict, List, Union


def tensor_obs_to_dict(obs_buf) -> Dict[str, List[float]]:
    """
    Convert IsaacLab obs_buf to Dict[str, List[float]].

    Handles:
        - obs_buf = {"policy": {"joint_pos": Tensor(1, 9), ...}}  (standard)
        - obs_buf = {"policy": Tensor(1, N)}  (flat policy obs)
        - obs_buf = Tensor(1, N)  (bare tensor fallback)

    All tensors are squeezed along dim=0 (batch) and converted to Python lists.
    This works for num_envs=1 (single-env training).

    Args:
        obs_buf: Raw observation buffer from ManagerBasedRLEnv.step() / .reset()

    Returns:
        Dict mapping group names to flat float lists.
    """
    if isinstance(obs_buf, dict) and "policy" in obs_buf:
        policy_obs = obs_buf["policy"]
    else:
        policy_obs = obs_buf

    if isinstance(policy_obs, dict):
        return {key: _tensor_to_list(val) for key, val in policy_obs.items()}

    # Flat tensor or array fallback
    return {"obs": _tensor_to_list(policy_obs)}


def _tensor_to_list(val) -> List[float]:
    """Squeeze batch dimension and convert tensor/array to flat Python list."""
    try:
        # PyTorch tensor
        import torch

        if isinstance(val, torch.Tensor):
            return val.squeeze(0).cpu().tolist()
    except ImportError:
        pass

    try:
        # NumPy array
        import numpy as np

        if isinstance(val, np.ndarray):
            return val.squeeze(0).tolist()
    except ImportError:
        pass

    # Plain Python list or scalar
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    return [float(val)]


def extract_success_from_info(info: Union[dict, None]) -> bool:
    """
    Extract per-env success flag from IsaacLab step info dict.

    IsaacLab's info dict may contain 'success', 'log', or termination signals.
    For single-env mode (num_envs=1), checks index [0].

    Args:
        info: Info dict returned by ManagerBasedRLEnv.step()

    Returns:
        Boolean success flag for env 0.
    """
    if not info:
        return False

    for key in ("success", "is_success", "task_success"):
        if key in info:
            val = info[key]
            if hasattr(val, "__getitem__"):
                return bool(val[0])
            return bool(val)

    # Check nested under "log" (IsaacLab metrics format)
    log = info.get("log", {})
    if "success_rate" in log:
        return float(log["success_rate"]) > 0.5

    return False
