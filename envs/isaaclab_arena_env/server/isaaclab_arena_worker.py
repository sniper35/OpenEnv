"""Worker process for the IsaacLab Arena OpenEnv wrapper."""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


KNOWN_EXAMPLE_ENVIRONMENTS = [
    "galileo_g1_locomanip_pick_and_place",
    "galileo_pick_and_place",
    "gr1_open_microwave",
    "kitchen_pick_and_place",
    "press_button",
]


def _write_message(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _serialize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _serialize_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
    return str(value)


def _serialize_space(space: Any) -> Dict[str, Any]:
    if space is None:
        return {}
    payload: Dict[str, Any] = {
        "type": type(space).__name__,
    }
    for attr in ("shape", "dtype", "n"):
        if hasattr(space, attr):
            payload[attr] = _serialize_value(getattr(space, attr))
    for attr in ("low", "high"):
        if hasattr(space, attr):
            payload[attr] = _serialize_value(getattr(space, attr))
    if hasattr(space, "spaces"):
        spaces = getattr(space, "spaces")
        if isinstance(spaces, dict):
            payload["spaces"] = {
                key: _serialize_space(value) for key, value in spaces.items()
            }
        else:
            payload["spaces"] = [_serialize_space(value) for value in spaces]
    return payload


def _ensure_dir_on_syspath(path: Optional[str]) -> None:
    if not path:
        return
    path_obj = Path(path)
    if path_obj.exists():
        path_str = str(path_obj)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def _iter_existing(paths: Iterable[Path]) -> Iterable[str]:
    for path in paths:
        if path.exists():
            yield str(path)


@dataclass
class RuntimeConfig:
    headless: bool
    enable_cameras: bool
    arena_repo_path: Optional[str]
    isaacsim_repo_path: Optional[str]


class MockArenaRuntime:
    """Deterministic runtime used for local tests and validation."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self.episode_id = "mock-episode"
        self.step_count = 0
        self.num_envs = 1
        self.device = "cuda:0"
        self.example_environment = "press_button"
        self.action_dim = 4
        self.max_steps = 25
        self.positions = [[0.0] * self.action_dim]
        self.target = [1.0, 0.5, 0.25, -0.25]
        self._rng = random.Random(0)

    def capabilities(self) -> Dict[str, Any]:
        return {
            "event": "ready",
            "worker_mode": "mock",
            "available_example_environments": list(KNOWN_EXAMPLE_ENVIRONMENTS),
        }

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self.episode_id = payload.get("episode_id") or "mock-episode"
        seed = payload.get("seed")
        if seed is not None:
            self._rng.seed(seed)
        self.step_count = 0
        self.num_envs = int(payload.get("num_envs", 1))
        self.device = payload.get("device", "cuda:0")
        self.example_environment = payload.get(
            "example_environment", "press_button"
        )
        options = payload.get("example_options", {})
        self.action_dim = int(options.get("action_dim", 4))
        self.max_steps = int(options.get("max_steps", 25))
        self.positions = [[0.0] * self.action_dim for _ in range(self.num_envs)]
        self.target = [
            1.0 if i == 0 else round(self._rng.uniform(-0.75, 0.75), 3)
            for i in range(self.action_dim)
        ]
        return self._build_response(info={"reset": True}, force_zero_reward=True)

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        values = list(payload.get("values", []))
        reshaped = self._reshape_actions(values)
        for env_id, action in enumerate(reshaped):
            self.positions[env_id] = [
                max(-1.0, min(1.0, pos + 0.2 * delta))
                for pos, delta in zip(self.positions[env_id], action)
            ]
        self.step_count += 1
        return self._build_response(info={"step": self.step_count})

    def close(self) -> None:
        return

    def _reshape_actions(self, values: list[float]) -> list[list[float]]:
        total_dim = self.num_envs * self.action_dim
        if not values:
            return [[0.0] * self.action_dim for _ in range(self.num_envs)]
        if len(values) == self.action_dim and self.num_envs > 1:
            return [list(values) for _ in range(self.num_envs)]
        if len(values) != total_dim:
            raise ValueError(
                f"Expected {self.action_dim} values (broadcast) or {total_dim} "
                f"values for {self.num_envs} envs, got {len(values)}"
            )
        return [
            values[idx * self.action_dim : (idx + 1) * self.action_dim]
            for idx in range(self.num_envs)
        ]

    def _build_response(
        self,
        info: Dict[str, Any],
        *,
        force_zero_reward: bool = False,
    ) -> Dict[str, Any]:
        reward_vector = []
        terminated = []
        truncated = []
        for position in self.positions:
            if force_zero_reward:
                reward = 0.0
                done = False
                cutoff = False
            else:
                mean_abs_error = sum(
                    abs(current - target)
                    for current, target in zip(position, self.target)
                ) / max(1, self.action_dim)
                reward = max(0.0, 1.0 - mean_abs_error)
                done = reward >= 0.95
                cutoff = self.step_count >= self.max_steps and not done
            reward_vector.append(round(reward, 6))
            terminated.append(done)
            truncated.append(cutoff)

        done = any(terminated) or any(truncated)
        scalar_reward = reward_vector[0] if len(reward_vector) == 1 else sum(
            reward_vector
        ) / len(reward_vector)
        observation = {
            "observations": {
                "policy": _serialize_value(self.positions),
                "target": _serialize_value(self.target),
            },
            "reward": scalar_reward,
            "reward_vector": reward_vector,
            "terminated": terminated,
            "truncated": truncated,
            "done": done,
            "info": {
                **info,
                "mock": True,
            },
            "runtime_mode": "mock",
            "example_environment": self.example_environment,
            "action_space": {
                "type": "Box",
                "shape": [self.num_envs, self.action_dim],
                "low": -1.0,
                "high": 1.0,
                "dtype": "float32",
            },
            "observation_space": {
                "type": "Dict",
                "spaces": {
                    "policy": {
                        "type": "Box",
                        "shape": [self.num_envs, self.action_dim],
                        "dtype": "float32",
                    },
                    "target": {
                        "type": "Box",
                        "shape": [self.action_dim],
                        "dtype": "float32",
                    },
                },
            },
        }
        state = {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "worker_alive": True,
            "example_environment": self.example_environment,
            "num_envs": self.num_envs,
            "device": self.device,
            "action_space": observation["action_space"],
            "observation_space": observation["observation_space"],
            "last_info": observation["info"],
        }
        return {
            "ok": True,
            "observation": observation,
            "state": state,
        }


class ArenaRuntime:
    """Real IsaacLab Arena runtime backed by an external Isaac interpreter."""

    def __init__(self, config: RuntimeConfig):
        self.config = config
        self._app_launcher = None
        self._env = None
        self._torch = None
        self.example_environment = "press_button"
        self.device = "cuda:0"
        self.num_envs = 1
        self.action_space: Dict[str, Any] = {}
        self.observation_space: Dict[str, Any] = {}
        self.episode_id = "arena-episode"
        self.step_count = 0
        self._install_workspace_paths()

    def capabilities(self) -> Dict[str, Any]:
        from isaaclab_arena.examples.example_environments.cli import ExampleEnvironments

        return {
            "event": "ready",
            "worker_mode": "arena",
            "available_example_environments": sorted(ExampleEnvironments.keys()),
        }

    def reset(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        self._close_env()
        self.step_count = 0
        self.episode_id = payload.get("episode_id") or "arena-episode"
        self.example_environment = payload.get(
            "example_environment", "press_button"
        )
        self.num_envs = int(payload.get("num_envs", 1))
        self.device = payload.get("device", "cuda:0")
        self._ensure_app(self.device, self.num_envs)
        env = self._build_env(payload)
        self._env = env
        reset_output = env.reset()
        observation, info = self._normalize_reset_output(reset_output)
        return self._build_response(
            observation=observation,
            reward=None,
            terminated=[False] * self.num_envs,
            truncated=[False] * self.num_envs,
            info=info,
        )

    def step(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._env is None:
            raise RuntimeError("Environment has not been reset")
        assert self._torch is not None
        flat_values = list(payload.get("values", []))
        action_shape = tuple(self._env.action_space.shape)
        action_tensor = self._reshape_actions(flat_values, action_shape)
        obs, reward, terminated, truncated, info = self._env.step(action_tensor)
        self.step_count += 1
        return self._build_response(
            observation=obs,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def close(self) -> None:
        self._close_env()
        if self._app_launcher is not None:
            self._safe_teardown()
            try:
                self._app_launcher.app.close()
            finally:
                self._app_launcher = None

    def _install_workspace_paths(self) -> None:
        arena_repo = Path(self.config.arena_repo_path) if self.config.arena_repo_path else None
        isaacsim_repo = (
            Path(self.config.isaacsim_repo_path)
            if self.config.isaacsim_repo_path
            else None
        )

        if arena_repo is not None:
            _ensure_dir_on_syspath(str(arena_repo))
            isaaclab_root = arena_repo / "submodules" / "IsaacLab"
            for path in _iter_existing(
                [
                    isaaclab_root / "source" / "isaaclab",
                    isaaclab_root / "source" / "isaaclab_assets",
                    isaaclab_root / "source" / "isaaclab_tasks",
                    isaaclab_root / "source" / "isaaclab_rl",
                    isaaclab_root / "source" / "isaaclab_mimic",
                ]
            ):
                _ensure_dir_on_syspath(path)

        if isaacsim_repo is not None:
            for path in _iter_existing(
                [
                    isaacsim_repo / "source" / "python_packages",
                    isaacsim_repo / "source" / "extensions",
                ]
            ):
                _ensure_dir_on_syspath(path)

    def _ensure_app(self, device: str, num_envs: int) -> None:
        if self._app_launcher is not None:
            return

        from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser
        from isaaclab_arena.utils.isaaclab_utils.simulation_app import get_app_launcher

        args_parser = get_isaaclab_arena_cli_parser()
        args_cli = args_parser.parse_args([])
        args_cli.headless = self.config.headless
        args_cli.enable_cameras = self.config.enable_cameras
        args_cli.device = device
        args_cli.num_envs = num_envs
        self._app_launcher = get_app_launcher(args_cli)

        import torch

        self._torch = torch

    def _build_env(self, payload: Dict[str, Any]) -> Any:
        from isaaclab_arena.examples.example_environments.cli import (
            get_arena_builder_from_cli,
            get_isaaclab_arena_example_environment_cli_parser,
        )

        args = []
        args.extend(["--num_envs", str(self.num_envs), "--device", self.device])
        args.append(self.example_environment)
        example_options = payload.get("example_options", {})
        for key, value in example_options.items():
            flag = "--" + str(key).replace("_", "-")
            if isinstance(value, bool):
                if value:
                    args.append(flag)
                continue
            if isinstance(value, (list, tuple)):
                for item in value:
                    args.extend([flag, str(item)])
                continue
            args.extend([flag, str(value)])

        parser = get_isaaclab_arena_example_environment_cli_parser()
        args_cli = parser.parse_args(args)
        builder = get_arena_builder_from_cli(args_cli)
        env = builder.make_registered()
        self.action_space = _serialize_space(env.action_space)
        self.observation_space = _serialize_space(env.observation_space)
        return env

    def _normalize_reset_output(self, reset_output: Any) -> tuple[Any, Dict[str, Any]]:
        if isinstance(reset_output, tuple) and len(reset_output) == 2:
            return reset_output
        return reset_output, {}

    def _reshape_actions(self, flat_values: list[float], shape: tuple[int, ...]):
        assert self._torch is not None
        total_dim = 1
        for item in shape:
            total_dim *= item
        if not flat_values:
            return self._torch.zeros(shape, device=self._env.device)
        if len(flat_values) == shape[-1] and total_dim != len(flat_values):
            flat_values = flat_values * (total_dim // len(flat_values))
        if len(flat_values) != total_dim:
            raise ValueError(
                f"Expected {shape[-1]} values (broadcast) or {total_dim} values "
                f"for action_space.shape={shape}, got {len(flat_values)}"
            )
        tensor = self._torch.tensor(flat_values, device=self._env.device)
        return tensor.reshape(shape)

    def _build_response(
        self,
        *,
        observation: Any,
        reward: Any,
        terminated: Any,
        truncated: Any,
        info: Any,
    ) -> Dict[str, Any]:
        reward_vector = self._flatten_float_vector(reward)
        terminated_vector = self._flatten_bool_vector(terminated)
        truncated_vector = self._flatten_bool_vector(truncated)
        if not reward_vector:
            reward_scalar = None
        elif len(reward_vector) == 1:
            reward_scalar = reward_vector[0]
        else:
            reward_scalar = sum(reward_vector) / len(reward_vector)

        observation_payload = {
            "observations": _serialize_value(observation),
            "reward": reward_scalar,
            "reward_vector": reward_vector,
            "terminated": terminated_vector,
            "truncated": truncated_vector,
            "done": any(terminated_vector) or any(truncated_vector),
            "info": _serialize_value(info) or {},
            "runtime_mode": "arena",
            "example_environment": self.example_environment,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
        }
        state = {
            "episode_id": self.episode_id,
            "step_count": self.step_count,
            "worker_alive": True,
            "example_environment": self.example_environment,
            "num_envs": self.num_envs,
            "device": self.device,
            "action_space": self.action_space,
            "observation_space": self.observation_space,
            "last_info": observation_payload["info"],
        }
        return {
            "ok": True,
            "observation": observation_payload,
            "state": state,
        }

    def _flatten_float_vector(self, value: Any) -> list[float]:
        if value is None:
            return []
        serialized = _serialize_value(value)
        if isinstance(serialized, list):
            if serialized and isinstance(serialized[0], list):
                return [float(item) for sublist in serialized for item in sublist]
            return [float(item) for item in serialized]
        return [float(serialized)]

    def _flatten_bool_vector(self, value: Any) -> list[bool]:
        if value is None:
            return [False] * self.num_envs
        serialized = _serialize_value(value)
        if isinstance(serialized, list):
            if serialized and isinstance(serialized[0], list):
                return [bool(item) for sublist in serialized for item in sublist]
            return [bool(item) for item in serialized]
        return [bool(serialized)]

    def _close_env(self) -> None:
        if self._env is not None:
            try:
                self._env.close()
            finally:
                self._env = None
                self._safe_teardown()

    def _safe_teardown(self) -> None:
        try:
            from isaaclab.sim import SimulationContext

            simulation_context = SimulationContext.instance()
            if simulation_context is not None:
                simulation_context._disable_app_control_on_stop_handle = True
                simulation_context.stop()
                simulation_context.clear_instance()
        except Exception:
            pass

        try:
            import omni.timeline

            omni.timeline.get_timeline_interface().stop()
        except Exception:
            pass

        try:
            import omni.usd

            omni.usd.get_context().new_stage()
        except Exception:
            pass


def _build_runtime(args: argparse.Namespace):
    config = RuntimeConfig(
        headless=bool(args.headless),
        enable_cameras=bool(args.enable_cameras),
        arena_repo_path=args.arena_repo_path,
        isaacsim_repo_path=args.isaacsim_repo_path,
    )
    if args.mode == "mock":
        return MockArenaRuntime(config)
    if args.mode == "arena":
        return ArenaRuntime(config)
    raise ValueError(f"Unsupported worker mode: {args.mode}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["mock", "arena"], required=True)
    parser.add_argument("--headless", type=int, default=1)
    parser.add_argument("--enable-cameras", type=int, default=0)
    parser.add_argument("--arena-repo-path", type=str, default=None)
    parser.add_argument("--isaacsim-repo-path", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    runtime = _build_runtime(args)
    _write_message(runtime.capabilities())
    try:
        for line in sys.stdin:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                message = json.loads(stripped)
                method = message.get("method")
                payload = message.get("payload", {})
                if method == "reset":
                    _write_message(runtime.reset(payload))
                elif method == "step":
                    _write_message(runtime.step(payload))
                elif method == "close":
                    runtime.close()
                    _write_message({"ok": True})
                    break
                else:
                    _write_message(
                        {
                            "ok": False,
                            "error": f"Unsupported worker method: {method}",
                        }
                    )
            except Exception as exc:
                _write_message({"ok": False, "error": str(exc)})
    finally:
        runtime.close()


if __name__ == "__main__":
    main()
