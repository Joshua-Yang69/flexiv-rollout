from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Any

from rollout.control.executor import make_control_executor_from_config
from rollout.control.pregrasp import PregraspConfig, run_pregrasp
from rollout.models.server import make_model_server_from_config
from rollout.perception.server import make_perceiver_from_config
from utils.config import load_yaml
from utils.latest_buffer import LatestBuffer

@dataclass
class RolloutRuntime:
    observation_buffer: LatestBuffer
    action_buffer: LatestBuffer
    result_buffer: LatestBuffer
    perceiver: Any
    model_server: Any
    control_executor: Any
    pregrasp_config: PregraspConfig | None = None

    def __post_init__(self) -> None:
        self._stop = Event()
        self._threads: list[Thread] = []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "RolloutRuntime":
        observation_buffer = LatestBuffer()
        action_buffer = LatestBuffer()
        result_buffer = LatestBuffer()

        perceiver = make_perceiver_from_config(config.get("perception", {}), output_buffer=observation_buffer)
        # Wire the perceiver's rolling history buffer into the model server so
        # temporal policies (VTMusePolicyAdapter) can sample strided windows.
        model_server = make_model_server_from_config(
            config.get("model_server", {}),
            observation_buffer=observation_buffer,
            history_buffer=perceiver.history_buffer,
            action_buffer=action_buffer,
        )
        control_executor = make_control_executor_from_config(
            config.get("control", {}),
            action_buffer=action_buffer,
            observation_buffer=observation_buffer,
            result_buffer=result_buffer,
            arm=perceiver.arm,
            gripper=perceiver.gripper,
        )

        # Optional pregrasp config
        pregrasp_config: PregraspConfig | None = None
        pg_data = config.get("pregrasp")
        if pg_data:
            pregrasp_config = PregraspConfig(**pg_data)

        return cls(
            observation_buffer=observation_buffer,
            action_buffer=action_buffer,
            result_buffer=result_buffer,
            perceiver=perceiver,
            model_server=model_server,
            control_executor=control_executor,
            pregrasp_config=pregrasp_config,
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> "RolloutRuntime":
        return cls.from_config(load_yaml(path))

    def pregrasp(self) -> None:
        """Execute pre-grasp (close gripper) before starting the rollout loops.

        Uses the ``pregrasp`` section from the rollout config. If no config is
        provided this is a no-op so that mock / simulation runs are unaffected.
        """
        if self.pregrasp_config is None:
            return
        run_pregrasp(self.perceiver.gripper, self.pregrasp_config)

    def start(self) -> None:
        self._threads = [
            Thread(target=self.perceiver.serve_forever, args=(self._stop,), name="perception", daemon=True),
            Thread(target=self.model_server.serve_forever, args=(self._stop,), name="model_server", daemon=True),
            Thread(target=self.control_executor.serve_forever, args=(self._stop,), name="control", daemon=True),
        ]
        for thread in self._threads:
            thread.start()

    def run(self, duration_s: float | None = None) -> None:
        self.pregrasp()   # pre-grasp before starting inference loops
        self.start()
        try:
            if duration_s is None:
                while not self._stop.is_set():
                    time.sleep(0.25)
            else:
                deadline = time.time() + duration_s
                while time.time() < deadline and not self._stop.is_set():
                    time.sleep(0.05)
        finally:
            self.stop()

    def stop(self) -> None:
        self._stop.set()
        for owner in (self.model_server, self.control_executor, self.perceiver):
            stop = getattr(owner, "stop", None)
            if callable(stop):
                stop()
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=2.0)

