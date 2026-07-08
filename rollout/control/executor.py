from __future__ import annotations

from dataclasses import dataclass, field
from threading import Event
from typing import Any

import numpy as np

from rollout.control.safety import SafetyLimits
from rollout.perception.devices import ArmClient, GripperClient, make_arm_client, make_gripper_client
from rollout.types import ActionResult, Observation, PolicyAction, RobotState
from utils.latest_buffer import LatestBuffer
from utils.timing import RateLimiter, now_ms


@dataclass
class ControlRuntimeConfig:
    fps: float = 50.0
    max_action_age_ms: float = 250.0
    # Both TCP and joint modes are enabled by default.
    # The active NRT mode on the Flexiv RDK is switched automatically when the
    # policy changes its output mode (tcp <-> joint), so no manual flag is needed.
    max_linear_vel: float = 0.1
    max_linear_acc: float = 0.5
    max_angular_vel: float = 0.1
    max_angular_acc: float = 0.5
    gripper_velocity_m_s: float = 0.08
    gripper_force_n: float = 27.0
    safety: SafetyLimits = field(default_factory=SafetyLimits)


class ControlExecutor:
    """Consume latest model action and send bounded commands to robot devices.

    Supports two NRT control paths transparently:
      - ``mode="tcp"``   → NRT_CARTESIAN_MOTION_FORCE via arm.send_cartesian_motion_force()
      - ``mode="joint"`` → NRT_JOINT_IMPEDANCE       via arm.send_joint_positions()

    The FlexivRizonClient switches the RDK mode automatically and only when it
    changes, so the policy can freely alternate between tcp and joint outputs
    without any manual flag.
    """

    def __init__(
        self,
        arm: ArmClient,
        gripper: GripperClient,
        action_buffer: LatestBuffer,
        observation_buffer: LatestBuffer | None = None,
        result_buffer: LatestBuffer | None = None,
        config: ControlRuntimeConfig | None = None,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.action_buffer = action_buffer
        self.observation_buffer = observation_buffer
        self.result_buffer = result_buffer or LatestBuffer()
        self.config = config or ControlRuntimeConfig()
        self._stop = Event()
        self._last_action_version = 0

    def apply_once(self) -> ActionResult:
        action_version = self.action_buffer.version
        action = self.action_buffer.get()
        if not isinstance(action, PolicyAction):
            return self._publish(ActionResult(False, now_ms(), "no action available"))
        if action_version == self._last_action_version:
            return self._publish(ActionResult(False, now_ms(), "no new action"))
        self._last_action_version = action_version

        age_ms = now_ms() - action.timestamp_ms
        if age_ms > self.config.max_action_age_ms:
            return self._publish(ActionResult(False, now_ms(), f"stale action {age_ms:.1f} ms"))

        current_state = self._current_robot_state()
        safe_action, messages = self.config.safety.enforce(action, current_state)

        if safe_action.mode == "hold":
            return self._publish(
                ActionResult(True, now_ms(), "; ".join(messages), applied_action=safe_action)
            )
        if safe_action.mode == "tcp":
            self._apply_tcp(safe_action)
            return self._publish(
                ActionResult(True, now_ms(), "; ".join(messages), applied_action=safe_action)
            )
        if safe_action.mode == "joint":
            self._apply_joint(safe_action)
            return self._publish(
                ActionResult(True, now_ms(), "; ".join(messages), applied_action=safe_action)
            )
        return self._publish(ActionResult(False, now_ms(), f"unsupported action mode {safe_action.mode}"))

    def _apply_tcp(self, action: PolicyAction) -> None:
        if action.target_tcp_pose is None:
            raise ValueError("TCP action is missing target_tcp_pose.")
        wrench = np.zeros(6, dtype=np.float64) if action.wrench is None else action.wrench
        self.arm.send_cartesian_motion_force(
            action.target_tcp_pose,
            wrench,
            max_linear_vel=self.config.max_linear_vel,
            max_linear_acc=self.config.max_linear_acc,
            max_angular_vel=self.config.max_angular_vel,
            max_angular_acc=self.config.max_angular_acc,
        )
        if action.target_gripper_width is not None:
            self.gripper.move(
                action.target_gripper_width,
                velocity_m_s=self.config.gripper_velocity_m_s,
                force_n=self.config.gripper_force_n,
            )

    def _apply_joint(self, action: PolicyAction) -> None:
        if action.target_joints is None:
            raise ValueError("Joint action is missing target_joints.")
        self.arm.send_joint_positions(action.target_joints)
        if action.target_gripper_width is not None:
            self.gripper.move(
                action.target_gripper_width,
                velocity_m_s=self.config.gripper_velocity_m_s,
                force_n=self.config.gripper_force_n,
            )

    def _current_robot_state(self) -> RobotState | None:
        if self.observation_buffer is not None:
            observation = self.observation_buffer.get()
            if isinstance(observation, Observation):
                return observation.robot
        try:
            arm_state = self.arm.read_state()
            gripper_state = self.gripper.read_state()
            return RobotState(arm=arm_state, gripper=gripper_state, timestamp_ms=now_ms())
        except Exception:
            return None

    def _publish(self, result: ActionResult) -> ActionResult:
        self.result_buffer.put(result)
        return result

    def serve_forever(self, stop_event: Event | None = None) -> None:
        limiter = RateLimiter(self.config.fps)
        stop_event = stop_event or self._stop
        while not stop_event.is_set():
            limiter.mark_start()
            self.apply_once()
            limiter.sleep()

    def stop(self) -> None:
        self._stop.set()
        self.gripper.stop()
        self.arm.stop()


def _runtime_config_from_mapping(data: dict[str, Any]) -> ControlRuntimeConfig:
    data = dict(data)
    safety_data = data.pop("safety", {})
    return ControlRuntimeConfig(safety=SafetyLimits(**safety_data), **data)


def make_control_executor_from_config(
    config: dict[str, Any],
    action_buffer: LatestBuffer,
    observation_buffer: LatestBuffer | None = None,
    result_buffer: LatestBuffer | None = None,
    arm: ArmClient | None = None,
    gripper: GripperClient | None = None,
) -> ControlExecutor:
    arm = arm or make_arm_client(config.get("arm", {}))
    gripper = gripper or make_gripper_client(config.get("gripper", {}))
    runtime = _runtime_config_from_mapping(config.get("runtime", {}))
    return ControlExecutor(
        arm=arm,
        gripper=gripper,
        action_buffer=action_buffer,
        observation_buffer=observation_buffer,
        result_buffer=result_buffer,
        config=runtime,
    )

