"""OpenEnv server environment for IsaacLab Arena."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

try:
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.types import EnvironmentMetadata

    from ..models import (
        IsaacLabArenaAction,
        IsaacLabArenaObservation,
        IsaacLabArenaState,
    )
    from .worker_process import IsaacLabArenaWorkerProcess, WorkerLaunchConfig
except ImportError:
    try:
        from isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )
        from isaaclab_arena_env.server.worker_process import (
            IsaacLabArenaWorkerProcess,
            WorkerLaunchConfig,
        )
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import EnvironmentMetadata
    except ImportError:
        from envs.isaaclab_arena_env.models import (
            IsaacLabArenaAction,
            IsaacLabArenaObservation,
            IsaacLabArenaState,
        )
        from envs.isaaclab_arena_env.server.worker_process import (
            IsaacLabArenaWorkerProcess,
            WorkerLaunchConfig,
        )
        from openenv.core.env_server.interfaces import Environment
        from openenv.core.env_server.types import EnvironmentMetadata


DEFAULT_EXAMPLE_ENVIRONMENT = "press_button"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_RUNTIME_MODE = "auto"
KNOWN_EXAMPLE_ENVIRONMENTS = [
    "galileo_g1_locomanip_pick_and_place",
    "galileo_pick_and_place",
    "gr1_open_microwave",
    "kitchen_pick_and_place",
    "press_button",
]


def _openenv_repo_root() -> Optional[Path]:
    try:
        return Path(__file__).resolve().parents[3]
    except IndexError:
        return None


def _detect_sibling_repo(repo_name: str) -> Optional[str]:
    root = _openenv_repo_root()
    if root is None:
        return None
    candidate = root.parent / repo_name
    return str(candidate) if candidate.exists() else None


def _resolve_executable(candidate: Optional[str]) -> Optional[str]:
    if not candidate:
        return None
    path = Path(candidate)
    if path.exists():
        return str(path)
    resolved = shutil.which(candidate)
    return resolved


def _dedupe_strings(values: Iterable[Optional[str]]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _candidate_isaac_python_paths(
    *,
    arena_repo_path: Optional[str],
    isaacsim_repo_path: Optional[str],
    preferred: Optional[str] = None,
) -> list[str]:
    candidates: list[str] = _dedupe_strings(
        [
            preferred,
            os.environ.get("ISAACSIM_PYTHON"),
            "/isaac-sim/python.sh",
        ]
    )

    if arena_repo_path:
        arena_repo = Path(arena_repo_path)
        candidates.extend(
            _dedupe_strings(
                [
                    str(
                        arena_repo
                        / "submodules"
                        / "IsaacLab"
                        / "_isaac_sim"
                        / "python.sh"
                    ),
                ]
            )
        )

    if isaacsim_repo_path:
        isaacsim_repo = Path(isaacsim_repo_path)
        build_candidates = sorted(
            isaacsim_repo.glob("_build/linux-*/release/python.sh")
        )
        candidates.extend(
            _dedupe_strings(
                [
                    str(isaacsim_repo / "python.sh"),
                    *[str(path) for path in build_candidates],
                ]
            )
        )

    return candidates


def _detect_isaac_python_executable(
    *,
    arena_repo_path: Optional[str],
    isaacsim_repo_path: Optional[str],
    preferred: Optional[str] = None,
) -> Optional[str]:
    for candidate in _candidate_isaac_python_paths(
        arena_repo_path=arena_repo_path,
        isaacsim_repo_path=isaacsim_repo_path,
        preferred=preferred,
    ):
        resolved = _resolve_executable(candidate)
        if resolved:
            return resolved
    return None


class IsaacLabArenaEnvironment(
    Environment[IsaacLabArenaAction, IsaacLabArenaObservation, IsaacLabArenaState]
):
    """Worker-backed OpenEnv environment that wraps IsaacLab Arena."""

    SUPPORTS_CONCURRENT_SESSIONS = False

    def __init__(self):
        super().__init__()
        default_arena_repo = os.environ.get(
            "ISAACLAB_ARENA_REPO_PATH", _detect_sibling_repo("IsaacLab-Arena")
        )
        default_isaacsim_repo = os.environ.get(
            "ISAACSIM_REPO_PATH", _detect_sibling_repo("IsaacSim")
        )
        default_isaac_python = _detect_isaac_python_executable(
            arena_repo_path=default_arena_repo,
            isaacsim_repo_path=default_isaacsim_repo,
        )
        supports_real_runtime = bool(default_isaac_python)

        self._worker: Optional[IsaacLabArenaWorkerProcess] = None
        self._worker_config: Optional[WorkerLaunchConfig] = None
        self._state = IsaacLabArenaState(
            episode_id=str(uuid4()),
            step_count=0,
            runtime_mode=os.environ.get(
                "ISAACLAB_ARENA_RUNTIME_MODE", DEFAULT_RUNTIME_MODE
            ),
            resolved_runtime_mode="mock",
            worker_alive=False,
            supports_real_runtime=supports_real_runtime,
            example_environment=None,
            available_example_environments=list(KNOWN_EXAMPLE_ENVIRONMENTS),
            num_envs=int(os.environ.get("ISAACLAB_ARENA_NUM_ENVS", "1")),
            device=os.environ.get("ISAACLAB_ARENA_DEVICE", DEFAULT_DEVICE),
            headless=os.environ.get("ISAACLAB_ARENA_HEADLESS", "1").lower()
            not in ("0", "false", "no"),
            enable_cameras=os.environ.get("ISAACLAB_ARENA_ENABLE_CAMERAS", "0")
            .lower()
            in ("1", "true", "yes"),
            arena_repo_path=default_arena_repo,
            isaacsim_repo_path=default_isaacsim_repo,
            isaac_python_path=default_isaac_python,
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> IsaacLabArenaObservation:
        launch_config, request_payload = self._build_reset_config(
            seed=seed,
            episode_id=episode_id,
            **kwargs,
        )
        self._ensure_worker(launch_config)
        assert self._worker is not None
        response = self._worker.request("reset", request_payload, timeout_s=300.0)
        self._apply_worker_state(response.get("state", {}))
        return self._payload_to_observation(response.get("observation", {}))

    def step(
        self,
        action: IsaacLabArenaAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> IsaacLabArenaObservation:
        if self._worker is None or not self._worker.is_alive:
            raise RuntimeError("IsaacLab Arena worker is not running. Call reset() first.")

        payload: Dict[str, Any] = {"values": action.values}
        if action.metadata:
            payload["metadata"] = action.metadata
        payload.update(kwargs)
        response = self._worker.request("step", payload, timeout_s=timeout_s or 300.0)
        self._apply_worker_state(response.get("state", {}))
        return self._payload_to_observation(response.get("observation", {}))

    @property
    def state(self) -> IsaacLabArenaState:
        self._state.worker_alive = self._worker.is_alive if self._worker else False
        return self._state

    def close(self) -> None:
        if self._worker is not None:
            self._worker.close()
            self._worker = None
        self._worker_config = None
        self._state.worker_alive = False

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="IsaacLabArenaEnvironment",
            description=(
                "OpenEnv wrapper for IsaacLab Arena example environments. "
                "Uses a lightweight mock runtime by default and can delegate "
                "to a real Isaac Sim + IsaacLab Arena worker when an external "
                "Isaac runtime is configured."
            ),
            version="0.1.0",
        )

    def _build_reset_config(
        self,
        *,
        seed: Optional[int],
        episode_id: Optional[str],
        **kwargs: Any,
    ) -> tuple[WorkerLaunchConfig, Dict[str, Any]]:
        requested_runtime_mode = kwargs.get("runtime_mode", self._state.runtime_mode)
        headless = kwargs.get("headless", self._state.headless)
        enable_cameras = kwargs.get("enable_cameras", self._state.enable_cameras)
        arena_repo_path = kwargs.get("arena_repo_path", self._state.arena_repo_path)
        isaacsim_repo_path = kwargs.get(
            "isaacsim_repo_path", self._state.isaacsim_repo_path
        )
        explicit_isaac_python = kwargs.get("isaac_python_path")
        isaac_python_path = _resolve_executable(explicit_isaac_python)
        if isaac_python_path is None and explicit_isaac_python is None:
            isaac_python_path = _detect_isaac_python_executable(
                arena_repo_path=arena_repo_path,
                isaacsim_repo_path=isaacsim_repo_path,
                preferred=self._state.isaac_python_path,
            )

        supports_real_runtime = bool(isaac_python_path)
        if requested_runtime_mode == "auto":
            resolved_runtime_mode = "arena" if supports_real_runtime else "mock"
        else:
            resolved_runtime_mode = requested_runtime_mode

        if resolved_runtime_mode == "arena" and not isaac_python_path:
            raise RuntimeError(
                "runtime_mode='arena' was requested but no Isaac runtime executable "
                "was found. Set ISAACSIM_PYTHON or pass isaac_python_path=... ."
            )
        if resolved_runtime_mode not in {"mock", "arena"}:
            raise ValueError(
                f"Unsupported runtime_mode '{resolved_runtime_mode}'. "
                "Expected auto, mock, or arena."
            )

        launch_config = WorkerLaunchConfig(
            worker_mode=resolved_runtime_mode,
            headless=bool(headless),
            enable_cameras=bool(enable_cameras),
            arena_repo_path=arena_repo_path,
            isaacsim_repo_path=isaacsim_repo_path,
            isaac_python_path=isaac_python_path,
        )
        payload = {
            "seed": seed,
            "episode_id": episode_id,
            "example_environment": kwargs.get(
                "example_environment",
                self._state.example_environment or DEFAULT_EXAMPLE_ENVIRONMENT,
            ),
            "example_options": kwargs.get("example_options", {}),
            "num_envs": int(kwargs.get("num_envs", self._state.num_envs)),
            "device": kwargs.get("device", self._state.device),
            "headless": bool(headless),
            "enable_cameras": bool(enable_cameras),
        }
        self._state.runtime_mode = requested_runtime_mode
        self._state.resolved_runtime_mode = resolved_runtime_mode
        self._state.supports_real_runtime = supports_real_runtime
        self._state.headless = bool(headless)
        self._state.enable_cameras = bool(enable_cameras)
        self._state.arena_repo_path = arena_repo_path
        self._state.isaacsim_repo_path = isaacsim_repo_path
        self._state.isaac_python_path = isaac_python_path
        return launch_config, payload

    def _ensure_worker(self, launch_config: WorkerLaunchConfig) -> None:
        if self._worker_config == launch_config and self._worker and self._worker.is_alive:
            return

        self.close()
        self._worker = IsaacLabArenaWorkerProcess(launch_config)
        ready = self._worker.start(timeout_s=120.0)
        self._worker_config = launch_config
        self._state.worker_alive = True
        self._state.resolved_runtime_mode = ready.get(
            "worker_mode", launch_config.worker_mode
        )
        self._state.available_example_environments = ready.get(
            "available_example_environments", list(KNOWN_EXAMPLE_ENVIRONMENTS)
        )

    def _apply_worker_state(self, state_payload: Dict[str, Any]) -> None:
        self._state.episode_id = state_payload.get("episode_id", self._state.episode_id)
        self._state.step_count = state_payload.get("step_count", self._state.step_count)
        self._state.example_environment = state_payload.get(
            "example_environment", self._state.example_environment
        )
        self._state.num_envs = state_payload.get("num_envs", self._state.num_envs)
        self._state.device = state_payload.get("device", self._state.device)
        self._state.action_space = state_payload.get("action_space", {})
        self._state.observation_space = state_payload.get("observation_space", {})
        self._state.last_info = state_payload.get("last_info", {})
        self._state.worker_alive = state_payload.get("worker_alive", True)

    def _payload_to_observation(
        self, payload: Dict[str, Any]
    ) -> IsaacLabArenaObservation:
        reward = payload.get("reward")
        done = payload.get("done", False)
        return IsaacLabArenaObservation(
            observations=payload.get("observations", {}),
            reward_vector=payload.get("reward_vector", []),
            terminated=payload.get("terminated", []),
            truncated=payload.get("truncated", []),
            info=payload.get("info", {}),
            example_environment=payload.get(
                "example_environment",
                self._state.example_environment or DEFAULT_EXAMPLE_ENVIRONMENT,
            ),
            runtime_mode=payload.get(
                "runtime_mode", self._state.resolved_runtime_mode
            ),
            action_space=payload.get("action_space", self._state.action_space),
            observation_space=payload.get(
                "observation_space", self._state.observation_space
            ),
            reward=reward,
            done=done,
            metadata=payload.get("metadata", {}),
        )
