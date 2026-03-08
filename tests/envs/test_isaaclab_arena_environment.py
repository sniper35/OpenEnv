"""End-to-end tests for the IsaacLab Arena OpenEnv wrapper."""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest
import requests

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_PATH = os.path.join(ROOT_DIR, "src")

sys.path.insert(0, ROOT_DIR)

from envs.isaaclab_arena_env.client import IsaacLabArenaEnv
from envs.isaaclab_arena_env.models import IsaacLabArenaAction, IsaacLabArenaState
from envs.isaaclab_arena_env.server.isaaclab_arena_environment import (
    IsaacLabArenaEnvironment,
)
from envs.isaaclab_arena_env.server.worker_process import (
    IsaacLabArenaWorkerProcess,
    WorkerLaunchConfig,
)


@pytest.fixture(scope="module")
def server():
    server_env = {
        **os.environ,
        "PYTHONPATH": f"{SRC_PATH}:{ROOT_DIR}",
        "ISAACLAB_ARENA_RUNTIME_MODE": "mock",
        "NO_PROXY": "localhost,127.0.0.1",
        "no_proxy": "localhost,127.0.0.1",
    }

    process = None
    localhost = ""
    start_port = 18017 + (os.getpid() % 1000)
    last_output = ""
    log_file = tempfile.TemporaryFile(mode="w+")

    try:
        for port in range(start_port, start_port + 20):
            localhost = f"http://localhost:{port}"
            command = [
                sys.executable,
                "-m",
                "uvicorn",
                "envs.isaaclab_arena_env.server.app:app",
                "--host",
                "0.0.0.0",
                "--port",
                str(port),
            ]

            log_file.seek(0)
            log_file.truncate(0)
            process = subprocess.Popen(
                command,
                cwd=ROOT_DIR,
                env=server_env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )

            is_healthy = False
            for _ in range(20):
                if process.poll() is not None:
                    break
                try:
                    response = requests.get(
                        f"{localhost}/health",
                        timeout=5,
                        proxies={"http": None, "https": None},
                    )
                    if response.status_code == 200:
                        is_healthy = True
                        break
                except requests.RequestException:
                    time.sleep(0.5)

            if is_healthy:
                break

            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
            log_file.seek(0)
            last_output = log_file.read()
            process = None

        if process is None:
            pytest.fail(f"Server failed to start.\n\n{last_output}")

        yield localhost
    finally:
        if process is not None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        log_file.close()


def test_reset_auto_mode_falls_back_to_mock(server):
    with IsaacLabArenaEnv(base_url=server).sync() as env:
        result = env.reset(runtime_mode="auto", example_environment="press_button")
        assert result.observation.runtime_mode == "mock"
        assert result.observation.example_environment == "press_button"
        assert len(result.observation.reward_vector) == 1
        assert result.done is False

        state = env.state()
        assert isinstance(state, IsaacLabArenaState)
        assert state.resolved_runtime_mode == "mock"
        assert state.example_environment == "press_button"
        assert state.worker_alive is True


def test_step_updates_reward_and_step_count(server):
    with IsaacLabArenaEnv(base_url=server).sync() as env:
        env.reset(runtime_mode="mock", example_environment="press_button")
        result = env.step(IsaacLabArenaAction(values=[1.0, 0.0, 0.0, 0.0]))

        assert result.reward is not None
        assert result.observation.reward_vector[0] > 0.0
        assert result.observation.terminated == [False]
        assert result.observation.truncated == [False]

        state = env.state()
        assert state.step_count == 1
        assert state.action_space["shape"] == [1, 4]


def test_reset_supports_multiple_envs_in_mock_mode(server):
    with IsaacLabArenaEnv(base_url=server).sync() as env:
        result = env.reset(
            runtime_mode="mock",
            example_environment="kitchen_pick_and_place",
            num_envs=2,
            example_options={"action_dim": 3},
        )

        assert result.observation.reward_vector == [0.0, 0.0]

        result = env.step(IsaacLabArenaAction(values=[1.0, 0.0, -1.0]))
        assert len(result.observation.reward_vector) == 2
        assert result.observation.action_space["shape"] == [2, 3]

        state = env.state()
        assert state.num_envs == 2
        assert state.action_space["shape"] == [2, 3]


def test_invalid_action_dimension_returns_error(server):
    with IsaacLabArenaEnv(base_url=server).sync() as env:
        env.reset(
            runtime_mode="mock",
            example_environment="press_button",
            num_envs=2,
            example_options={"action_dim": 3},
        )

        with pytest.raises(RuntimeError):
            env.step(IsaacLabArenaAction(values=[1.0, 2.0]))


def test_arena_mode_requires_runtime_executable(server):
    with IsaacLabArenaEnv(base_url=server).sync() as env:
        with pytest.raises(RuntimeError, match="ISAACSIM_PYTHON|isaac_python_path"):
            env.reset(runtime_mode="arena", example_environment="press_button")


def test_detects_staged_isaacsim_runtime_from_repo(monkeypatch, tmp_path):
    isaacsim_repo = tmp_path / "IsaacSim"
    staged_runtime = isaacsim_repo / "_build" / "linux-x86_64" / "release"
    staged_runtime.mkdir(parents=True)
    python_sh = staged_runtime / "python.sh"
    python_sh.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
    python_sh.chmod(0o755)

    monkeypatch.delenv("ISAACSIM_PYTHON", raising=False)
    monkeypatch.setenv("ISAACSIM_REPO_PATH", str(isaacsim_repo))
    monkeypatch.setenv("ISAACLAB_ARENA_REPO_PATH", str(tmp_path / "missing-arena"))

    env = IsaacLabArenaEnvironment()
    try:
        assert env.state.supports_real_runtime is True
        assert Path(env.state.isaac_python_path).resolve() == python_sh.resolve()
    finally:
        env.close()


def test_worker_process_ignores_non_json_stdout_lines():
    class _FakeProcess:
        @staticmethod
        def poll():
            return None

    worker = IsaacLabArenaWorkerProcess(
        WorkerLaunchConfig(
            worker_mode="mock",
            headless=True,
            enable_cameras=False,
            arena_repo_path=None,
            isaacsim_repo_path=None,
            isaac_python_path=None,
        )
    )
    worker._process = _FakeProcess()  # type: ignore[assignment]
    worker._stdout_queue.put("[Info] Isaac Sim log noise")
    worker._stdout_queue.put('{"ok": true, "payload": "ready"}')

    message = worker._read_message(timeout_s=0.1)

    assert message == {"ok": True, "payload": "ready"}
    assert worker._stdout_noise_lines == ["[Info] Isaac Sim log noise"]
