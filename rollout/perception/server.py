from __future__ import annotations

from dataclasses import dataclass
from threading import Event
from typing import Any

from rollout.perception.devices import (
    ArmClient,
    GripperClient,
    TactileClient,
    VisualClient,
    make_arm_client,
    make_gripper_client,
    make_tactile_client,
    make_visual_client,
)
from rollout.types import Observation, RobotState
from utils.latest_buffer import LatestBuffer
from utils.rolling_buffer import RollingBuffer
from utils.timing import RateLimiter, now_ms


@dataclass
class PerceptionRuntimeConfig:
    fps: float = 30.0
    max_state_skew_ms: float = 5.0
    publish_visual: bool = True
    publish_tactile: bool = True
    history_capacity: int = 200  # frames to keep in rolling history buffer


class StatePerceiver:
    """Capture robot state and sensors into one latest observation stream.

    In addition to ``output_buffer`` (latest-only), a ``history_buffer``
    (RollingBuffer) is maintained so that temporal-model adapters (e.g.
    VTMusePolicyAdapter) can sample strided windows of past observations.
    """

    def __init__(
        self,
        arm: ArmClient,
        gripper: GripperClient,
        visual: VisualClient | None = None,
        tactile: TactileClient | None = None,
        output_buffer: LatestBuffer | None = None,
        config: PerceptionRuntimeConfig | None = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.visual = visual
        self.tactile = tactile
        self.output_buffer = output_buffer or LatestBuffer()
        self.config = config or PerceptionRuntimeConfig()
        self._stop = Event()

        # Rolling history: keeps the last `history_capacity` observations so
        # that inference threads can reconstruct temporal windows.
        self.history_buffer: RollingBuffer = RollingBuffer(
            capacity=self.config.history_capacity
        )

    @property
    def latest(self) -> Observation | None:
        value = self.output_buffer.get()
        return value if isinstance(value, Observation) else None

    def read_robot_state(self) -> RobotState:
        arm_state = self.arm.read_state()
        gripper_state = self.gripper.read_state()
        return RobotState(arm=arm_state, gripper=gripper_state, timestamp_ms=now_ms())

    def capture_once(self) -> Observation:
        robot_state = self.read_robot_state()
        visual_frame = None
        tactile_frame = None
        if self.config.publish_visual and self.visual is not None:
            visual_frame = self.visual.read_frame()
        if self.config.publish_tactile and self.tactile is not None:
            tactile_frame = self.tactile.read_frame()

        extras: dict[str, Any] = {
            "state_skew_ms": robot_state.skew_ms,
            "capture_timestamp_ms": now_ms(),
        }
        if robot_state.skew_ms > self.config.max_state_skew_ms:
            extras["warning"] = f"arm/gripper state skew {robot_state.skew_ms:.3f} ms"

        observation = Observation(
            robot=robot_state,
            visual=visual_frame,
            tactile=tactile_frame,
            extras=extras,
        )
        self.output_buffer.put(observation)
        self.history_buffer.put(observation)  # keep every frame in rolling history
        return observation

    def serve_forever(self, stop_event: Event | None = None) -> None:
        limiter = RateLimiter(self.config.fps)
        stop_event = stop_event or self._stop
        while not stop_event.is_set():
            limiter.mark_start()
            self.capture_once()
            limiter.sleep()

    def stop(self) -> None:
        self._stop.set()
        for device in (self.visual, self.tactile, self.gripper, self.arm):
            if device is not None:
                device.stop()


def make_perceiver_from_config(config: dict[str, Any], output_buffer: LatestBuffer | None = None) -> StatePerceiver:
    runtime = PerceptionRuntimeConfig(**config.get("runtime", {}))
    arm = make_arm_client(config.get("arm", {}))
    gripper = make_gripper_client(config.get("gripper", {}))
    visual = make_visual_client(config.get("visual", {}))
    tactile = make_tactile_client(config.get("tactile", {}))
    return StatePerceiver(
        arm=arm,
        gripper=gripper,
        visual=visual,
        tactile=tactile,
        output_buffer=output_buffer,
        config=runtime,
    )


