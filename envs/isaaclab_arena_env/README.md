# IsaacLab-Arena Environment for OpenEnv

GPU-accelerated robotics simulation via NVIDIA Isaac Sim + IsaacLab-Arena, exposed through OpenEnv's standard WebSocket API for RL training.

## Overview

This environment wraps [IsaacLab-Arena](https://github.com/isaac-sim/IsaacLab-Arena) — a composable robotics simulation framework built on NVIDIA Isaac Sim. It enables:

- **RL training** via the standard gym API through OpenEnv's WebSocket infrastructure (TRL, SkyRL, etc.)
- **Composable task/embodiment combinations** (Franka manipulation, G1 humanoid)
- **Single Docker image** parametrized by environment variables (no separate images per task)

## Supported Tasks

| `ISAACLAB_TASK` | `ISAACLAB_EMBODIMENT` | Action Dim | Description |
|---|---|---|---|
| `pick_and_place` (default) | `franka` | 7 (IK) or 9 (joint) | Pick object and place at goal |
| `open_door` | `franka` | 7 | Open articulated door |
| `press_button` | `franka` | 7 | Press button in scene |
| `g1_locomanip_pick_and_place` | `g1` | 35 | G1 humanoid loco-manipulation |

## Quick Start

### Mock Mode (CI, no GPU)

```bash
# Install mock-mode dependencies
pip install openenv-core fastapi uvicorn pydantic

# Start server in mock mode
ISAACLAB_MOCK_MODE=true uvicorn envs.isaaclab_arena_env.server.app:app --port 8000

# Test with client
python -c "
from isaaclab_arena_env import IsaacLabArenaEnv, IsaacLabArenaAction
with IsaacLabArenaEnv(base_url='http://localhost:8000') as env:
    result = env.reset()
    print('Obs keys:', list(result.observation.observations.keys()))
    for _ in range(5):
        result = env.step(IsaacLabArenaAction(values=[0.0] * 7))
        print(f'Reward: {result.reward:.3f}')
"
```

### Real GPU Mode (A10G+)

```bash
# Build the env image (requires base image, see Dockerfile.base)
docker build -t isaaclab-arena-openenv:latest -f envs/isaaclab_arena_env/server/Dockerfile .

# Run pick-and-place with Franka (IK control)
docker run --gpus all \
    -e ISAACLAB_TASK=pick_and_place \
    -e ISAACLAB_EMBODIMENT=franka \
    -e ISAACLAB_ACTION_MODE=ik \
    -p 8000:8000 \
    isaaclab-arena-openenv:latest
```

## Configuration

All configuration is via environment variables:

| Variable | Default | Description |
|---|---|---|
| `ISAACLAB_TASK` | `pick_and_place` | Task to run |
| `ISAACLAB_EMBODIMENT` | `franka` | Robot embodiment |
| `ISAACLAB_ACTION_MODE` | `ik` | `ik` (end-effector) or `joint_pos` |
| `ISAACLAB_SCENE` | `kitchen` | Scene name |
| `ISAACLAB_NUM_ENVS` | `1` | Parallel environments |
| `ISAACLAB_ENABLE_CAMERA` | `false` | Enable RGB observations |
| `ISAACLAB_MOCK_MODE` | `false` | Use mock bridge (no GPU) |
| `ISAACLAB_HEADLESS` | `true` | Headless rendering |

## Observation Space

Observations are named groups from IsaacLab's `obs_buf["policy"]`:

**Franka (pick_and_place):**
```python
{
    "joint_pos": [9 floats],   # Robot joint positions
    "joint_vel": [9 floats],   # Robot joint velocities
    "eef_pos":   [3 floats],   # End-effector position
    "eef_quat":  [4 floats],   # End-effector orientation
    "object_pos": [3 floats],  # Object position
    "object_quat":[4 floats],  # Object orientation
    "goal_pos":  [3 floats],   # Goal position
}
```

## Action Space

- **Franka IK** (`action_dim=7`): `[dx, dy, dz, droll, dpitch, dyaw, gripper]`
- **Franka joint_pos** (`action_dim=9`): 7 arm joints + 2 gripper joints
- **G1 joint_pos** (`action_dim=35`): Full-body joint positions

## Architecture

```
[Main Thread]  AppLauncher → ManagerBasedRLEnv (Isaac Sim GPU loop)
                 ↑ action_queue            ↓ obs_queue

[uvicorn Thread]  FastAPI WebSocket ↔ IsaacLabArenaEnvironment
                    → IsaacSimBridge (thread-safe queue bridge)
```

Isaac Sim requires the simulation loop on the main thread (GPU context ownership). The thread-queue bridge pattern solves this constraint cleanly.

## Hardware Requirements

- **GPU**: NVIDIA A10G (24GB VRAM) minimum
- **HF Spaces cost**: A10G ~\$1.10/hr
- **Build time**: 30-60 min (base image: Isaac Sim + IsaacLab + shader pre-warm)
- **Startup time**: 2-3 min (shaders pre-warmed) / 10-15 min (cold)
- **Disk**: ~20-25 GB

## Tests (Mock Mode)

```bash
# Run mock-mode tests (no GPU required)
PYTHONPATH=src:envs ISAACLAB_MOCK_MODE=true \
    uv run pytest tests/envs/test_isaaclab_arena_environment.py -v
```
