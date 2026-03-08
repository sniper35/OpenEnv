"""Client for the IsaacLab Arena OpenEnv wrapper."""

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
    try:
        from openenv_core.client_types import StepResult
        from openenv_core.env_client import EnvClient

        from isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )
    except ImportError:
        from envs.isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )
        from openenv.core.client_types import StepResult
        from openenv.core.env_client import EnvClient


class IsaacLabArenaEnv(
    EnvClient[IsaacLabArenaAction, IsaacLabArenaObservation, IsaacLabArenaState]
):
    """Client for a running IsaacLab Arena OpenEnv server."""

    def __init__(
        self,
        base_url: str,
        connect_timeout_s: float = 10.0,
        message_timeout_s: float = 300.0,
        provider: Optional[Any] = None,
    ):
        super().__init__(
            base_url=base_url,
            connect_timeout_s=connect_timeout_s,
            message_timeout_s=message_timeout_s,
            provider=provider,
        )

    def _step_payload(self, action: IsaacLabArenaAction) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"values": action.values}
        if action.metadata:
            payload["metadata"] = action.metadata
        return payload

    def _parse_result(
        self, payload: Dict[str, Any]
    ) -> StepResult[IsaacLabArenaObservation]:
        obs_data = payload.get("observation", {})
        observation = IsaacLabArenaObservation(
            observations=obs_data.get("observations", {}),
            reward_vector=obs_data.get("reward_vector", []),
            terminated=obs_data.get("terminated", []),
            truncated=obs_data.get("truncated", []),
            info=obs_data.get("info", {}),
            example_environment=obs_data.get("example_environment", "press_button"),
            runtime_mode=obs_data.get("runtime_mode", "mock"),
            action_space=obs_data.get("action_space", {}),
            observation_space=obs_data.get("observation_space", {}),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> IsaacLabArenaState:
        return IsaacLabArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            runtime_mode=payload.get("runtime_mode", "auto"),
            resolved_runtime_mode=payload.get("resolved_runtime_mode", "mock"),
            worker_alive=payload.get("worker_alive", False),
            supports_real_runtime=payload.get("supports_real_runtime", False),
            example_environment=payload.get("example_environment"),
            available_example_environments=payload.get(
                "available_example_environments", []
            ),
            num_envs=payload.get("num_envs", 1),
            device=payload.get("device", "cuda:0"),
            headless=payload.get("headless", True),
            enable_cameras=payload.get("enable_cameras", False),
            arena_repo_path=payload.get("arena_repo_path"),
            isaacsim_repo_path=payload.get("isaacsim_repo_path"),
            isaac_python_path=payload.get("isaac_python_path"),
            action_space=payload.get("action_space", {}),
            observation_space=payload.get("observation_space", {}),
            last_info=payload.get("last_info", {}),
        )

    def reset(
        self,
        *,
        example_environment: Optional[str] = None,
        runtime_mode: Optional[str] = None,
        num_envs: Optional[int] = None,
        device: Optional[str] = None,
        headless: Optional[bool] = None,
        enable_cameras: Optional[bool] = None,
        arena_repo_path: Optional[str] = None,
        isaacsim_repo_path: Optional[str] = None,
        isaac_python_path: Optional[str] = None,
        example_options: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> StepResult[IsaacLabArenaObservation]:
        reset_kwargs = dict(kwargs)
        if example_environment is not None:
            reset_kwargs["example_environment"] = example_environment
        if runtime_mode is not None:
            reset_kwargs["runtime_mode"] = runtime_mode
        if num_envs is not None:
            reset_kwargs["num_envs"] = num_envs
        if device is not None:
            reset_kwargs["device"] = device
        if headless is not None:
            reset_kwargs["headless"] = headless
        if enable_cameras is not None:
            reset_kwargs["enable_cameras"] = enable_cameras
        if arena_repo_path is not None:
            reset_kwargs["arena_repo_path"] = arena_repo_path
        if isaacsim_repo_path is not None:
            reset_kwargs["isaacsim_repo_path"] = isaacsim_repo_path
        if isaac_python_path is not None:
            reset_kwargs["isaac_python_path"] = isaac_python_path
        if example_options is not None:
            reset_kwargs["example_options"] = example_options
        return super().reset(**reset_kwargs)
