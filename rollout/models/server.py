from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any

from rollout.models.base import BasePolicy, build_policy_from_config
from rollout.types import Observation, PolicyAction
from utils.latest_buffer import LatestBuffer
from utils.rolling_buffer import RollingBuffer
from utils.timing import RateLimiter, now_ms


@dataclass
class ModelServerConfig:
    fps: float = 30.0
    observation_timeout_s: float = 1.0
    publish_hold_on_error: bool = False


class PolicyModelServer:
    """Persistent policy inference loop.

    Parameters
    ----------
    policy:
        A ``BasePolicy`` subclass (ACTPolicyAdapter, VTMusePolicyAdapter, …).
    observation_buffer:
        Latest-only buffer; updated by the perception thread every frame.
    history_buffer:
        Rolling buffer of past observations used by temporal policies (e.g.
        VTMusePolicyAdapter) that need a strided frame window.  Optional; if
        provided it is passed to the policy's ``infer_with_history`` method
        when that method exists, otherwise ``infer`` is called normally.
    action_buffer:
        Latest-only buffer receiving the most recent PolicyAction.
    config:
        Runtime parameters (fps, timeout, …).
    """

    def __init__(
        self,
        policy: BasePolicy,
        observation_buffer: LatestBuffer,
        history_buffer: RollingBuffer | None = None,
        action_buffer: LatestBuffer | None = None,
        config: ModelServerConfig | None = None,
    ) -> None:
        self.policy = policy
        self.observation_buffer = observation_buffer
        self.history_buffer = history_buffer
        self.action_buffer = action_buffer or LatestBuffer()
        self.config = config or ModelServerConfig()
        self._stop = Event()
        self._last_observation_version = 0

    @property
    def latest_action(self) -> PolicyAction | None:
        value = self.action_buffer.get()
        return value if isinstance(value, PolicyAction) else None

    def infer_once(self, wait: bool = False) -> PolicyAction | None:
        if wait:
            version, value = self.observation_buffer.wait_next(
                self._last_observation_version,
                timeout=self.config.observation_timeout_s,
            )
            self._last_observation_version = version
        else:
            value = self.observation_buffer.get()
        if not isinstance(value, Observation):
            return None

        try:
            # If the policy supports temporal-history inference, use it.
            infer_with_history = getattr(self.policy, "infer_with_history", None)
            if callable(infer_with_history) and self.history_buffer is not None:
                action = infer_with_history(value, self.history_buffer)
            else:
                action = self.policy.infer(value)
        except Exception as exc:
            if not self.config.publish_hold_on_error:
                raise
            action = PolicyAction(
                mode="hold",
                timestamp_ms=now_ms(),
                target_gripper_width=value.robot.gripper.width_m,
                metadata={"error": repr(exc), "source": "model_server"},
            )
        self.action_buffer.put(action)
        return action

    def serve_forever(self, stop_event: Event | None = None) -> None:
        self.policy.load()
        limiter = RateLimiter(self.config.fps)
        stop_event = stop_event or self._stop
        while not stop_event.is_set():
            limiter.mark_start()
            self.infer_once(wait=False)
            limiter.sleep()

    def stop(self) -> None:
        self._stop.set()


def make_model_server_from_config(
    config: dict[str, Any],
    observation_buffer: LatestBuffer,
    history_buffer: RollingBuffer | None = None,
    action_buffer: LatestBuffer | None = None,
) -> PolicyModelServer:
    policy = build_policy_from_config(config.get("policy", {}))
    server_config = ModelServerConfig(**config.get("runtime", {}))
    return PolicyModelServer(
        policy=policy,
        observation_buffer=observation_buffer,
        history_buffer=history_buffer,
        action_buffer=action_buffer,
        config=server_config,
    )
