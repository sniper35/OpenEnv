# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for IsaacLab-Arena environment.

All tests run in mock mode (no GPU or Isaac Sim required).
Real GPU tests are skipped unless ISAACLAB_AVAILABLE is True.
"""

import json
import os
import pytest

# Force mock mode for all tests
os.environ.setdefault("ISAACLAB_MOCK_MODE", "true")

from isaaclab_arena_env.models import (
    MOCK_OBS_SPECS,
    TASK_REGISTRY,
    IsaacLabArenaAction,
    IsaacLabArenaObservation,
    IsaacLabArenaState,
)
from isaaclab_arena_env.server.bridge import MockIsaacSimBridge
from isaaclab_arena_env.server.config import get_env_config
from isaaclab_arena_env.server.environment import IsaacLabArenaEnvironment

try:
    import isaacsim  # noqa: F401

    ISAACLAB_AVAILABLE = True
except ImportError:
    ISAACLAB_AVAILABLE = False


# ── Model serialization tests ──────────────────────────────────────────────────


class TestModels:
    """Test Pydantic model round-trip serialization."""

    def test_action_roundtrip(self):
        """IsaacLabArenaAction serializes and deserializes correctly."""
        action = IsaacLabArenaAction(values=[0.1, -0.2, 0.3, 0.0, -0.1, 0.2, 1.0])
        data = action.model_dump()
        restored = IsaacLabArenaAction(**data)
        assert restored.values == action.values

    def test_action_json_roundtrip(self):
        """Action survives JSON serialization."""
        action = IsaacLabArenaAction(values=[0.5] * 7)
        json_str = action.model_dump_json()
        parsed = json.loads(json_str)
        restored = IsaacLabArenaAction(**parsed)
        assert restored.values == action.values

    def test_observation_roundtrip(self):
        """IsaacLabArenaObservation serializes correctly."""
        obs = IsaacLabArenaObservation(
            observations={"joint_pos": [0.0] * 9, "eef_pos": [0.5, 0.0, 0.4]},
            reward=0.1,
            terminated=False,
            truncated=False,
            success=False,
        )
        data = obs.model_dump()
        restored = IsaacLabArenaObservation(**data)
        assert restored.observations["joint_pos"] == [0.0] * 9
        assert restored.reward == pytest.approx(0.1)
        assert not restored.terminated

    def test_state_roundtrip(self):
        """IsaacLabArenaState serializes correctly."""
        state = IsaacLabArenaState(
            task_name="pick_and_place",
            embodiment_name="franka",
            action_mode="ik",
            action_dim=7,
            action_bounds={"low": [-1.0] * 7, "high": [1.0] * 7},
            observation_spec={"joint_pos": 9, "eef_pos": 3},
            total_reward=1.5,
            success_rate=0.8,
        )
        data = state.model_dump()
        restored = IsaacLabArenaState(**data)
        assert restored.action_dim == 7
        assert restored.success_rate == pytest.approx(0.8)

    def test_obs_with_optional_rgb(self):
        """rgb_image field is optional and defaults to None."""
        obs = IsaacLabArenaObservation(observations={})
        assert obs.rgb_image is None

    def test_mock_obs_specs_completeness(self):
        """MOCK_OBS_SPECS covers all registered embodiments."""
        assert "franka" in MOCK_OBS_SPECS
        assert "g1" in MOCK_OBS_SPECS
        # Each spec should have at least joint_pos
        for emb, spec in MOCK_OBS_SPECS.items():
            assert "joint_pos" in spec, f"{emb} missing joint_pos"

    def test_task_registry_completeness(self):
        """TASK_REGISTRY has all expected tasks."""
        expected = {"pick_and_place", "open_door", "press_button", "g1_locomanip_pick_and_place"}
        assert expected.issubset(set(TASK_REGISTRY.keys()))


# ── Mock bridge tests ──────────────────────────────────────────────────────────


class TestMockBridge:
    """Test MockIsaacSimBridge behavior."""

    def test_reset_returns_observation(self):
        """reset() returns IsaacLabArenaObservation with correct structure."""
        bridge = MockIsaacSimBridge(embodiment="franka")
        obs = bridge.reset()
        assert isinstance(obs, IsaacLabArenaObservation)
        assert "joint_pos" in obs.observations
        assert len(obs.observations["joint_pos"]) == 9

    def test_step_returns_observation(self):
        """step() returns observation with reward."""
        bridge = MockIsaacSimBridge(embodiment="franka")
        bridge.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        obs = bridge.step(action)
        assert isinstance(obs, IsaacLabArenaObservation)
        assert obs.reward is not None

    def test_auto_terminates_at_max_steps(self):
        """Bridge terminates episode after max_steps."""
        bridge = MockIsaacSimBridge(embodiment="franka", max_steps=5)
        bridge.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        obs = None
        for _ in range(5):
            obs = bridge.step(action)
        assert obs is not None
        assert obs.terminated
        assert obs.success

    def test_g1_obs_shape(self):
        """G1 embodiment returns correct observation dimensions."""
        bridge = MockIsaacSimBridge(embodiment="g1")
        obs = bridge.reset()
        assert len(obs.observations["joint_pos"]) == 35

    def test_step_count_tracked(self):
        """Step count increments correctly."""
        bridge = MockIsaacSimBridge(embodiment="franka", max_steps=100)
        bridge.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        for _ in range(10):
            bridge.step(action)
        assert bridge._step_count == 10

    def test_reset_clears_step_count(self):
        """reset() resets internal step counter."""
        bridge = MockIsaacSimBridge(embodiment="franka", max_steps=100)
        bridge.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        for _ in range(10):
            bridge.step(action)
        bridge.reset()
        assert bridge._step_count == 0


# ── Environment tests ──────────────────────────────────────────────────────────


class TestIsaacLabArenaEnvironmentMock:
    """Test IsaacLabArenaEnvironment in mock mode."""

    def _make_env(self, embodiment="franka"):
        bridge = MockIsaacSimBridge(embodiment=embodiment, max_steps=100)
        os.environ["ISAACLAB_MOCK_MODE"] = "true"
        os.environ["ISAACLAB_EMBODIMENT"] = embodiment
        return IsaacLabArenaEnvironment(bridge=bridge)

    def test_creation(self):
        """Environment creates without error in mock mode."""
        env = self._make_env()
        assert env is not None

    def test_reset(self):
        """reset() returns valid initial observation."""
        env = self._make_env()
        obs = env.reset()
        assert isinstance(obs, IsaacLabArenaObservation)
        assert len(obs.observations) > 0
        assert not obs.terminated
        assert not obs.truncated

    def test_step(self):
        """step() advances the environment."""
        env = self._make_env()
        env.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        obs = env.step(action)
        assert isinstance(obs, IsaacLabArenaObservation)
        assert obs.reward is not None

    def test_state_after_reset(self):
        """State reflects correct task/embodiment metadata after reset."""
        env = self._make_env()
        env.reset()
        state = env.state
        assert isinstance(state, IsaacLabArenaState)
        assert state.task_name == "pick_and_place"
        assert state.action_dim == 7
        assert state.step_count == 0

    def test_state_step_count_increments(self):
        """State.step_count tracks steps correctly."""
        env = self._make_env()
        env.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        for _ in range(5):
            env.step(action)
        assert env.state.step_count == 5

    def test_state_total_reward_accumulates(self):
        """State.total_reward accumulates over steps."""
        env = self._make_env()
        env.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        for _ in range(10):
            env.step(action)
        assert env.state.total_reward > 0.0

    def test_episode_terminates(self):
        """Episode terminates when bridge signals done."""
        bridge = MockIsaacSimBridge(embodiment="franka", max_steps=5)
        env = IsaacLabArenaEnvironment(bridge=bridge)
        env.reset()
        action = IsaacLabArenaAction(values=[0.0] * 7)
        done = False
        for _ in range(10):
            obs = env.step(action)
            if obs.terminated or obs.truncated:
                done = True
                break
        assert done

    def test_action_bounds_in_state(self):
        """State reports action bounds for the current embodiment."""
        env = self._make_env()
        env.reset()
        state = env.state
        assert "low" in state.action_bounds
        assert "high" in state.action_bounds
        assert len(state.action_bounds["low"]) == 7

    def test_observation_spec_in_state(self):
        """State reports observation group sizes."""
        env = self._make_env()
        env.reset()
        state = env.state
        assert "joint_pos" in state.observation_spec
        assert state.observation_spec["joint_pos"] == 9


# ── Config tests ───────────────────────────────────────────────────────────────


class TestConfig:
    """Test environment configuration from env vars."""

    def test_default_config(self):
        """Default config uses franka pick_and_place with IK."""
        os.environ.pop("ISAACLAB_TASK", None)
        os.environ.pop("ISAACLAB_EMBODIMENT", None)
        os.environ.pop("ISAACLAB_ACTION_MODE", None)
        cfg = get_env_config()
        assert cfg["task"] == "pick_and_place"
        assert cfg["embodiment"] == "franka"
        assert cfg["action_mode"] == "ik"
        assert cfg["action_dim"] == 7

    def test_joint_pos_mode(self):
        """joint_pos mode gives action_dim=9 for Franka."""
        os.environ["ISAACLAB_ACTION_MODE"] = "joint_pos"
        cfg = get_env_config()
        assert cfg["action_dim"] == 9
        os.environ.pop("ISAACLAB_ACTION_MODE", None)

    def test_mock_mode_env_var(self):
        """ISAACLAB_MOCK_MODE=true is parsed correctly."""
        os.environ["ISAACLAB_MOCK_MODE"] = "true"
        cfg = get_env_config()
        assert cfg["mock_mode"] is True
        os.environ["ISAACLAB_MOCK_MODE"] = "false"


# ── GPU tests (skipped without Isaac Sim) ─────────────────────────────────────


@pytest.mark.skipif(not ISAACLAB_AVAILABLE, reason="Requires NVIDIA GPU + Isaac Sim")
class TestIsaacLabArenaGPU:
    """Integration tests that require real Isaac Sim."""

    def test_real_env_reset(self):
        """Placeholder: real env reset on GPU."""
        # This test only runs on hardware with Isaac Sim installed
        pass
