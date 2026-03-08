---
title: IsaacLab Arena Environment Server
emoji: 🤖
colorFrom: orange
colorTo: gray
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - isaac-sim
  - isaaclab
  - robotics
---

# IsaacLab Arena Environment

OpenEnv wrapper for NVIDIA IsaacLab Arena example environments. The server uses a worker process so the OpenEnv API stays lightweight while the actual simulator can run in a separate Isaac runtime when available.

## Runtime Modes

- `mock`: deterministic CPU-only worker for local testing, CI, and OpenEnv validation.
- `arena`: real Isaac Sim + IsaacLab Arena execution through an external runtime executable such as `/isaac-sim/python.sh`.
- `auto`: chooses `arena` when it can resolve a real Isaac runtime from `ISAACSIM_PYTHON`, the sibling `IsaacSim` staged build, or the IsaacLab `_isaac_sim` link; otherwise it falls back to `mock`.

## Quick Start

### Mock Mode

```python
from isaaclab_arena_env import IsaacLabArenaAction, IsaacLabArenaEnv

with IsaacLabArenaEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(runtime_mode="mock", example_environment="press_button")
    result = env.step(IsaacLabArenaAction(values=[1.0, 0.0, 0.0, 0.0]))
    print(result.observation.reward_vector)
```

Run the server locally:

```bash
cd envs/isaaclab_arena_env
uv run server
```

### Real Isaac Runtime

Provide the Isaac runtime executable and local repo paths at reset time or through environment variables:

```bash
export ISAACSIM_PYTHON=/isaac-sim/python.sh
export ISAACLAB_ARENA_REPO_PATH=/workspace/IsaacLab-Arena
export ISAACSIM_REPO_PATH=/workspace/IsaacSim
```

For a source-built Isaac Sim runtime, the practical setup sequence is:

```bash
cd /workspace/IsaacSim
./build.sh -r

cd /workspace/IsaacLab-Arena/submodules/IsaacLab
ln -s /workspace/IsaacSim/_build/linux-x86_64/release _isaac_sim
TERM=xterm ./isaaclab.sh -i none

/workspace/IsaacSim/_build/linux-x86_64/release/python.sh -m pip install --no-build-isolation flatdict==4.0.1
/workspace/IsaacSim/_build/linux-x86_64/release/python.sh -m pip install -e /workspace/IsaacLab-Arena
```

```python
from isaaclab_arena_env import IsaacLabArenaEnv

with IsaacLabArenaEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset(
        runtime_mode="arena",
        example_environment="press_button",
        num_envs=1,
        device="cuda:0",
        headless=True,
        example_options={"embodiment": "franka"},
    )
```

## Reset Parameters

- `example_environment`: one of the built-in IsaacLab Arena example environment names.
- `runtime_mode`: `auto`, `mock`, or `arena`.
- `num_envs`: number of vectorized Isaac Lab environments.
- `device`: e.g. `cuda:0`.
- `headless`: launch the simulator without a visible UI.
- `enable_cameras`: enable camera extensions for camera-based examples.
- `arena_repo_path`: path to the local `IsaacLab-Arena` checkout.
- `isaacsim_repo_path`: path to the local `IsaacSim` checkout.
- `isaac_python_path`: runtime executable used for the real worker.
- `example_options`: extra CLI-style options forwarded to the example environment builder.

## Docker Notes

The included Dockerfile defaults to a lightweight Python base image and runs in `mock` mode. To run the real simulator in-container, build with an Isaac Sim base image and set `ISAACSIM_PYTHON` appropriately:

```bash
docker build \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/isaac-sim:5.1.0 \
  -t isaaclab-arena-env .
```

## Validation

This environment is intentionally structured like other OpenEnv envs:

- `models.py`, `client.py`, `server/app.py`, `server/Dockerfile`
- `openenv.yaml`, `pyproject.toml`, `README.md`
- `uv run server` entry point
- `openenv validate` compatibility

For fast correctness checks, use `runtime_mode="mock"` and run the tests under `tests/envs/test_isaaclab_arena_environment.py`.

## Platform Notes

- `mock` mode is the reliable validation path on hosts that do not satisfy Isaac Sim graphics requirements.
