# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Entry point for real GPU mode: Isaac Sim main-thread + uvicorn background thread.

Isaac Sim requires the simulation loop on the main thread (GPU context ownership).
This module:
  1. Initializes AppLauncher (must be FIRST before any isaaclab imports)
  2. Creates the ManagerBasedRLEnv on the main thread
  3. Starts uvicorn on a background daemon thread
  4. Runs the sim polling loop on the main thread

Architecture:
    [Main Thread]  AppLauncher → ManagerBasedRLEnv
                     sim_loop() polls action_queue, calls gym.step(), puts to obs_queue

    [Background]   uvicorn → IsaacLabArenaEnvironment
                     .step() puts to action_queue, blocks on obs_queue

Usage (inside Docker / on GPU machine):
    /isaac-sim/python.sh envs/isaaclab_arena_env/server/main.py
"""

import queue
import threading


def main():
    from server.config import get_env_config

    cfg = get_env_config()

    if cfg["mock_mode"]:
        # Mock mode: just run uvicorn directly (no AppLauncher needed)
        import uvicorn
        from server.app import app

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
        return

    # ── Real mode: Isaac Sim must be initialized FIRST ─────────────────────────
    # AppLauncher must be called before any isaaclab/omni imports
    from isaaclab.app import AppLauncher

    app_launcher_args = {
        "headless": cfg["headless"],
        "num_envs": cfg["num_envs"],
    }
    sim_app = AppLauncher(app_launcher_args).app

    import gymnasium as gym
    # Now safe to import IsaacLab
    from isaaclab_tasks.utils import parse_env_cfg

    # ── Build gym environment ───────────────────────────────────────────────────
    env_cfg = parse_env_cfg(
        cfg["task_id"],
        num_envs=cfg["num_envs"],
        use_gpu=True,
    )
    gym_env = gym.make(cfg["task_id"], cfg=env_cfg)

    # ── Set up thread queues ────────────────────────────────────────────────────
    action_queue: queue.SimpleQueue = queue.SimpleQueue()
    obs_queue: queue.SimpleQueue = queue.SimpleQueue()
    shutdown_event = threading.Event()

    from server.app import create_bridged_app
    # ── Start uvicorn on background thread ──────────────────────────────────────
    from server.bridge import IsaacSimBridge

    bridge = IsaacSimBridge(action_queue, obs_queue, timeout_s=30.0)
    fastapi_app = create_bridged_app(bridge)

    import uvicorn

    uvicorn_config = uvicorn.Config(
        fastapi_app, host="0.0.0.0", port=8000, log_level="info"
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    def _run_uvicorn():
        uvicorn_server.run()

    uvicorn_thread = threading.Thread(target=_run_uvicorn, daemon=True, name="uvicorn")
    uvicorn_thread.start()

    # ── Run sim loop on main thread (blocks until shutdown) ─────────────────────
    try:
        from server.bridge import run_sim_loop

        run_sim_loop(
            gym_env=gym_env,
            action_queue=action_queue,
            obs_queue=obs_queue,
            shutdown_event=shutdown_event,
            sim_app=sim_app,
        )
    except KeyboardInterrupt:
        pass
    finally:
        shutdown_event.set()
        uvicorn_server.should_exit = True
        gym_env.close()
        sim_app.close()


if __name__ == "__main__":
    main()
