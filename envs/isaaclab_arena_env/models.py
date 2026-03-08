"""Data models for the IsaacLab Arena OpenEnv wrapper."""

from typing import Any, Dict, List, Optional

from pydantic import Field

try:
    from openenv.core.env_server.types import Action, Observation, State
except ImportError:
    from openenv_core.env_server.types import Action, Observation, State


class IsaacLabArenaAction(Action):
    """Continuous control action for an IsaacLab Arena environment."""

    values: List[float] = Field(
        default_factory=list,
        description=(
            "Flat continuous action vector. When num_envs > 1 this may either "
            "match the full action tensor size or a single-env action that is "
            "broadcast to every environment."
        ),
    )


class IsaacLabArenaObservation(Observation):
    """Observation returned by the IsaacLab Arena environment."""

    observations: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized policy observation returned by Isaac Lab.",
    )
    reward_vector: List[float] = Field(
        default_factory=list,
        description="Per-environment reward values from the last step.",
    )
    terminated: List[bool] = Field(
        default_factory=list,
        description="Per-environment termination flags.",
    )
    truncated: List[bool] = Field(
        default_factory=list,
        description="Per-environment truncation flags.",
    )
    info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized info payload from the underlying environment.",
    )
    example_environment: str = Field(
        default="press_button",
        description="IsaacLab Arena example environment identifier.",
    )
    runtime_mode: str = Field(
        default="mock",
        description="Resolved runtime mode used for execution: mock or arena.",
    )
    action_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized action space for the active environment.",
    )
    observation_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized observation space for the active environment.",
    )


class IsaacLabArenaState(State):
    """Extended state for the IsaacLab Arena OpenEnv wrapper."""

    runtime_mode: str = Field(
        default="auto",
        description="Requested runtime mode: auto, mock, or arena.",
    )
    resolved_runtime_mode: str = Field(
        default="mock",
        description="Resolved runtime mode used by the worker process.",
    )
    worker_alive: bool = Field(
        default=False,
        description="Whether the backing worker process is currently alive.",
    )
    supports_real_runtime: bool = Field(
        default=False,
        description="Whether a real Isaac runtime appears to be available.",
    )
    example_environment: Optional[str] = Field(
        default=None,
        description="Currently selected IsaacLab Arena example environment.",
    )
    available_example_environments: List[str] = Field(
        default_factory=list,
        description="Known IsaacLab Arena example environments.",
    )
    num_envs: int = Field(
        default=1,
        ge=1,
        description="Number of vectorized environments inside Isaac Lab.",
    )
    device: str = Field(
        default="cuda:0",
        description="Requested Isaac Lab device string.",
    )
    headless: bool = Field(
        default=True,
        description="Whether the backing Isaac app runs headless.",
    )
    enable_cameras: bool = Field(
        default=False,
        description="Whether camera extensions are enabled.",
    )
    arena_repo_path: Optional[str] = Field(
        default=None,
        description="Resolved path to the local IsaacLab-Arena checkout.",
    )
    isaacsim_repo_path: Optional[str] = Field(
        default=None,
        description="Resolved path to the local IsaacSim checkout.",
    )
    isaac_python_path: Optional[str] = Field(
        default=None,
        description="Executable used to start the real Isaac worker.",
    )
    action_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized action space for the active environment.",
    )
    observation_space: Dict[str, Any] = Field(
        default_factory=dict,
        description="Serialized observation space for the active environment.",
    )
    last_info: Dict[str, Any] = Field(
        default_factory=dict,
        description="Last info payload received from the worker.",
    )
