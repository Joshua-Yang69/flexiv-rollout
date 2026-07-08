from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from rollout.types import PolicyAction, RobotState
from utils.timing import now_ms


@dataclass
class SafetyLimits:
    max_tcp_translation_step_m: float = 0.02
    max_joint_step_rad: float = 0.05
    gripper_min_width_m: float = 0.0
    gripper_max_width_m: float = 0.085
    max_wrench_abs: float = 30.0

    def enforce(self, action: PolicyAction, current: RobotState | None) -> tuple[PolicyAction, list[str]]:
        messages: list[str] = []
        if action.mode == "hold":
            return action, messages

        target_tcp_pose = action.target_tcp_pose
        target_joints = action.target_joints
        wrench = action.wrench
        target_gripper_width = action.target_gripper_width

        if target_gripper_width is not None:
            clipped = float(np.clip(target_gripper_width, self.gripper_min_width_m, self.gripper_max_width_m))
            if clipped != target_gripper_width:
                messages.append("clipped gripper width")
            target_gripper_width = clipped

        if wrench is not None:
            clipped_wrench = np.clip(wrench, -self.max_wrench_abs, self.max_wrench_abs)
            if not np.array_equal(clipped_wrench, wrench):
                messages.append("clipped wrench")
            wrench = clipped_wrench

        if current is not None and target_tcp_pose is not None:
            current_xyz = current.arm.tcp_pose[:3, 3]
            target_xyz = target_tcp_pose[:3, 3]
            delta = target_xyz - current_xyz
            norm = float(np.linalg.norm(delta))
            if norm > self.max_tcp_translation_step_m:
                target_tcp_pose = target_tcp_pose.copy()
                target_tcp_pose[:3, 3] = current_xyz + delta / norm * self.max_tcp_translation_step_m
                messages.append("clipped tcp translation step")

        if current is not None and target_joints is not None:
            delta = np.clip(
                target_joints - current.arm.joints,
                -self.max_joint_step_rad,
                self.max_joint_step_rad,
            )
            clipped_joints = current.arm.joints + delta
            if not np.allclose(clipped_joints, target_joints):
                messages.append("clipped joint step")
            target_joints = clipped_joints

        clipped_action = PolicyAction(
            mode=action.mode,
            timestamp_ms=action.timestamp_ms,
            target_tcp_pose=target_tcp_pose,
            target_joints=target_joints,
            target_gripper_width=target_gripper_width,
            wrench=wrench,
            metadata={**action.metadata, "safety_messages": messages, "safety_timestamp_ms": now_ms()},
        )
        return clipped_action, messages

