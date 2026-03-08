# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Thread-safe bridge between uvicorn (background thread) and Isaac Sim (main thread).

Isaac Sim requires the simulation loop on the main thread due to GPU context
ownership. This bridge enables the uvicorn server running on a background thread
to communicate with the sim loop via thread-safe queues.

Architecture:
    [Main Thread]  AppLauncher → ManagerBasedRLEnv (sim loop)
                     polls action_queue → gym_env.step() → puts to obs_queue

    [uvicorn Thread]  IsaacLabArenaEnvironment.step()
                        → puts action in action_queue
                        → blocks on obs_queue.get(timeout=30)
                        → returns IsaacLabArenaObservation
"""

import queue
import time
from typing import Optional, Tuple

try:
    from ..models import IsaacLabArenaAction, IsaacLabArenaObservation, MOCK_OBS_SPECS
    from .obs_utils import tensor_obs_to_dict
except ImportError:
    import sys
    from pathlib import Path

    _parent = str(Path(__file__).parent.parent)
    if _parent not in sys.path:
        sys.path.insert(0, _parent)
    from models import IsaacLabArenaAction, IsaacLabArenaObservation, MOCK_OBS_SPECS
    from server.obs_utils import tensor_obs_to_dict


class IsaacSimBridge:
    """Thread-safe bridge from uvicorn thread to Isaac Sim main thread."""

    def __init__(
        self,
        action_queue: "queue.SimpleQueue[Tuple]",
        obs_queue: "queue.SimpleQueue[Tuple]",
        timeout_s: float = 30.0,
    ):
        self._action_q = action_queue
        self._obs_q = obs_queue
        self._timeout = timeout_s

    def reset(self, seed: Optional[int] = None) -> IsaacLabArenaObservation:
        """Send reset command and wait for initial observation."""
        self._action_q.put(("reset", seed))
        return self._recv()

    def step(self, action: IsaacLabArenaAction) -> IsaacLabArenaObservation:
        """Send action and wait for resulting observation."""
        self._action_q.put(("step", action.values))
        return self._recv()

    def _recv(self) -> IsaacLabArenaObservation:
        """Block until obs_queue has a result or timeout fires."""
        try:
            result = self._obs_q.get(block=True, timeout=self._timeout)
        except queue.Empty:
            raise TimeoutError(
                f"Isaac Sim did not respond within {self._timeout}s. "
                "Check that the simulation loop is running."
            )
        msg_type, *payload = result
        if msg_type in ("reset_result", "step_result"):
            obs_dict, reward, terminated, truncated = payload
            return _obs_from_dict(obs_dict, reward, terminated, truncated)
        raise RuntimeError(f"Unexpected message from sim: {msg_type!r}")


class MockIsaacSimBridge:
    """
    Synthetic bridge for CI testing without GPU or Isaac Sim.

    Returns fixed-shape observations per MOCK_OBS_SPECS[embodiment].
    Auto-terminates after max_steps with reward=1.0, success=True.
    """

    def __init__(self, embodiment: str = "franka", max_steps: int = 100):
        self._embodiment = embodiment
        self._max_steps = max_steps
        self._step_count = 0
        self._obs_spec = MOCK_OBS_SPECS.get(embodiment, MOCK_OBS_SPECS["franka"])

    def reset(self, seed: Optional[int] = None) -> IsaacLabArenaObservation:
        self._step_count = 0
        return self._make_obs(reward=0.0, terminated=False, truncated=False)

    def step(self, action: IsaacLabArenaAction) -> IsaacLabArenaObservation:
        self._step_count += 1
        terminated = self._step_count >= self._max_steps
        reward = 1.0 if terminated else 0.01
        return self._make_obs(
            reward=reward, terminated=terminated, truncated=False, success=terminated
        )

    def _make_obs(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        success: bool = False,
    ) -> IsaacLabArenaObservation:
        observations = {
            group: [float(self._step_count) * 0.001] * dim
            for group, dim in self._obs_spec.items()
        }
        return IsaacLabArenaObservation(
            observations=observations,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            success=success,
            done=terminated or truncated,
        )


def _obs_from_dict(
    obs_dict: dict,
    reward: float,
    terminated: bool,
    truncated: bool,
    success: bool = False,
) -> IsaacLabArenaObservation:
    """Convert raw obs_dict from sim loop to IsaacLabArenaObservation."""
    # obs_dict is already Dict[str, List[float]] from obs_utils.tensor_to_obs_dict
    return IsaacLabArenaObservation(
        observations=obs_dict,
        reward=reward,
        terminated=terminated,
        truncated=truncated,
        success=success,
        done=terminated or truncated,
    )


def run_sim_loop(
    gym_env,
    action_queue: "queue.SimpleQueue[Tuple]",
    obs_queue: "queue.SimpleQueue[Tuple]",
    shutdown_event,
    sim_app=None,
) -> None:
    """
    Main-thread sim loop. Call from the main thread after AppLauncher.

    Polls action_queue for commands, steps the gym environment, and puts
    results in obs_queue. Calls sim_app.update() to keep Omniverse alive
    between actions.

    Args:
        gym_env: Gymnasium-compatible IsaacLab ManagerBasedRLEnv.
        action_queue: Receives ("reset", seed) or ("step", values) tuples.
        obs_queue: Sends ("reset_result"|"step_result", obs_dict, rew, term, trunc).
        shutdown_event: threading.Event; set to stop the loop.
        sim_app: Isaac Sim SimulationApp (called .update() between steps).
    """
    import torch

    while not shutdown_event.is_set():
        try:
            msg_type, payload = action_queue.get_nowait()
        except queue.Empty:
            if sim_app is not None:
                sim_app.update()
            time.sleep(0.001)
            continue

        if msg_type == "reset":
            seed = payload
            reset_kwargs = {"seed": seed} if seed is not None else {}
            obs_buf, _ = gym_env.reset(**reset_kwargs)
            obs_dict = tensor_obs_to_dict(obs_buf)
            obs_queue.put(("reset_result", obs_dict, 0.0, False, False))

        elif msg_type == "step":
            values = payload
            action_t = torch.tensor([values], dtype=torch.float32)
            obs_buf, rew, term, trunc, info = gym_env.step(action_t)
            obs_dict = tensor_obs_to_dict(obs_buf)
            obs_queue.put(
                ("step_result", obs_dict, float(rew[0]), bool(term[0]), bool(trunc[0]))
            )

        elif msg_type == "shutdown":
            break
