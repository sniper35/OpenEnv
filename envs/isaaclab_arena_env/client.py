# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
IsaacLab-Arena Environment Client.

Maintains a persistent WebSocket connection to the IsaacLab-Arena server,
enabling efficient multi-step RL training interactions.
"""

from typing import Any, Dict, Optional

try:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    from .models import (
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        IsaacLabArenaState,
    )
except ImportError:
    from openenv.core.client_types import StepResult
    from openenv.core.env_client import EnvClient

    try:
        from models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )
    except ImportError:
        from isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )


class IsaacLabArenaEnv(
    EnvClient[IsaacLabArenaAction, IsaacLabArenaObservation, IsaacLabArenaState]
):
    """
    Client for IsaacLab-Arena GPU robotics simulation environments.

    Connects to a running IsaacLab-Arena server and provides a gymnasium-like
    interface for RL training.

    Supported tasks (via ISAACLAB_TASK env var on the server):
        - pick_and_place (default, Franka, IK control)
        - open_door (Franka)
        - press_button (Franka)
        - g1_locomanip_pick_and_place (G1 humanoid)

    Example:
        >>> with IsaacLabArenaEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(f"Obs keys: {list(result.observation.observations.keys())}")
        ...     for _ in range(100):
        ...         action = IsaacLabArenaAction(values=[0.0] * 7)
        ...         result = client.step(action)
        ...         if result.done:
        ...             break
    """

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 30.0,
        message_timeout_s: float = 60.0,
        provider: Optional[Any] = None,
    ):
        """
        Initialize IsaacLab-Arena environment client.

        Args:
            base_url: Base URL of the IsaacLab-Arena server (http:// or ws://).
            connect_timeout_s: Timeout for WebSocket connection (longer for GPU warmup).
            message_timeout_s: Timeout per step response (GPU sim can be slow on first step).
            provider: Optional container/runtime provider for lifecycle management.
        """
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )

    def _step_payload(self, action: IsaacLabArenaAction) -> Dict:
        payload: Dict[str, Any] = {"values": action.values}
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[IsaacLabArenaObservation]:
        obs_data = payload.get("observation", {})
        observation = IsaacLabArenaObservation(
            observations=obs_data.get("observations", {}),
            terminated=obs_data.get("terminated", False),
            truncated=obs_data.get("truncated", False),
            success=obs_data.get("success", False),
            rgb_image=obs_data.get("rgb_image"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> IsaacLabArenaState:
        return IsaacLabArenaState(
            episode_id=payload.get("episode_id", ""),
            step_count=payload.get("step_count", 0),
            task_name=payload.get("task_name", "pick_and_place"),
            embodiment_name=payload.get("embodiment_name", "franka"),
            action_mode=payload.get("action_mode", "ik"),
            action_dim=payload.get("action_dim", 7),
            action_bounds=payload.get("action_bounds", {}),
            observation_spec=payload.get("observation_spec", {}),
            total_reward=payload.get("total_reward", 0.0),
            success_rate=payload.get("success_rate", 0.0),
        )

    def reset(
        self,
        seed: Optional[int] = None,
        **kwargs,
    ) -> StepResult[IsaacLabArenaObservation]:
        """Reset the environment and return the initial observation."""
        reset_kwargs = dict(kwargs)
        if seed is not None:
            reset_kwargs["seed"] = seed
        return super().reset(**reset_kwargs)

    def step(
        self,
        action: IsaacLabArenaAction,
        **kwargs,
    ) -> StepResult[IsaacLabArenaObservation]:
        """Execute one step in the environment."""
        return super().step(action, **kwargs)
