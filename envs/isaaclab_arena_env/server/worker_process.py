"""Worker process management for the IsaacLab Arena environment."""

from __future__ import annotations

import json
import queue
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class WorkerLaunchConfig:
    """Arguments that require a worker restart when changed."""

    worker_mode: str
    headless: bool
    enable_cameras: bool
    arena_repo_path: Optional[str]
    isaacsim_repo_path: Optional[str]
    isaac_python_path: Optional[str]


class IsaacLabArenaWorkerProcess:
    """Thin JSON-RPC-like wrapper around the worker subprocess."""

    def __init__(self, launch_config: WorkerLaunchConfig):
        self.launch_config = launch_config
        self._process: Optional[subprocess.Popen[str]] = None
        self._stdout_queue: queue.Queue[str] = queue.Queue()
        self._stderr_lines: list[str] = []
        self._stdout_noise_lines: list[str] = []
        self._stdout_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._ready_payload: Dict[str, Any] = {}

    @property
    def ready_payload(self) -> Dict[str, Any]:
        return dict(self._ready_payload)

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self, timeout_s: float = 60.0) -> Dict[str, Any]:
        if self.is_alive:
            return self.ready_payload

        worker_script = Path(__file__).with_name("isaaclab_arena_worker.py")
        python_cmd = (
            self.launch_config.isaac_python_path
            if self.launch_config.worker_mode == "arena"
            and self.launch_config.isaac_python_path
            else sys.executable
        )
        cmd = [
            python_cmd,
            str(worker_script),
            "--mode",
            self.launch_config.worker_mode,
            "--headless",
            "1" if self.launch_config.headless else "0",
            "--enable-cameras",
            "1" if self.launch_config.enable_cameras else "0",
        ]
        if self.launch_config.arena_repo_path:
            cmd.extend(["--arena-repo-path", self.launch_config.arena_repo_path])
        if self.launch_config.isaacsim_repo_path:
            cmd.extend(["--isaacsim-repo-path", self.launch_config.isaacsim_repo_path])

        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self._stdout_thread = threading.Thread(
            target=self._pump_stdout,
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._pump_stderr,
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        ready = self._read_message(timeout_s=timeout_s)
        if ready.get("event") != "ready":
            self.close()
            raise RuntimeError(
                f"Worker failed to start cleanly: {ready.get('error', ready)}"
            )
        self._ready_payload = ready
        return self.ready_payload

    def request(
        self,
        method: str,
        payload: Dict[str, Any],
        timeout_s: float = 300.0,
    ) -> Dict[str, Any]:
        if not self.is_alive:
            raise RuntimeError("Worker process is not running")
        assert self._process is not None
        assert self._process.stdin is not None

        message = {"method": method, "payload": payload}
        self._process.stdin.write(json.dumps(message) + "\n")
        self._process.stdin.flush()
        response = self._read_message(timeout_s=timeout_s)
        if not response.get("ok", False):
            output_tail = self._tail_output()
            error = response.get("error", "Worker returned an unknown error")
            if output_tail:
                error = f"{error}\n\nWorker output:\n{output_tail}"
            raise RuntimeError(error)
        return response

    def close(self) -> None:
        if self._process is None:
            return

        if self.is_alive and self._process.stdin is not None:
            try:
                self._process.stdin.write(
                    json.dumps({"method": "close", "payload": {}}) + "\n"
                )
                self._process.stdin.flush()
            except Exception:
                pass

        try:
            self._process.terminate()
            self._process.wait(timeout=10)
        except Exception:
            try:
                self._process.kill()
            except Exception:
                pass
        finally:
            self._process = None

    def _pump_stdout(self) -> None:
        assert self._process is not None
        assert self._process.stdout is not None
        for line in self._process.stdout:
            stripped = line.strip()
            if stripped:
                self._stdout_queue.put(stripped)

    def _pump_stderr(self) -> None:
        assert self._process is not None
        assert self._process.stderr is not None
        for line in self._process.stderr:
            stripped = line.rstrip()
            if stripped:
                self._stderr_lines.append(stripped)

    def _read_message(self, timeout_s: float) -> Dict[str, Any]:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            remaining = max(0.0, deadline - time.time())
            try:
                raw = self._stdout_queue.get(timeout=min(0.5, remaining))
            except queue.Empty:
                if self._process is not None and self._process.poll() is not None:
                    output_tail = self._tail_output()
                    raise RuntimeError(
                        "Worker exited unexpectedly"
                        + (f"\n\nWorker output:\n{output_tail}" if output_tail else "")
                    )
                continue

            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                self._stdout_noise_lines.append(raw)
                continue

        raise TimeoutError(
            f"Timed out waiting for worker response after {timeout_s:.1f}s"
        )

    def _tail_output(self) -> str:
        lines = [*self._stderr_lines[-20:], *self._stdout_noise_lines[-20:]]
        return "\n".join(lines)
