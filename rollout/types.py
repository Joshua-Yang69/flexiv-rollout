from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np


def _array(value: Any, dtype: np.dtype | type = np.float64) -> np.ndarray:
    return np.asarray(value, dtype=dtype)


@dataclass(frozen=True)
class ArmState:
    joints: np.ndarray
    tcp_pose: np.ndarray
    timestamp_ms: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "joints", _array(self.joints))
        object.__setattr__(self, "tcp_pose", _array(self.tcp_pose))
        if self.joints.shape != (7,):
            raise ValueError(f"Arm joints must have shape (7,), got {self.joints.shape}")
        if self.tcp_pose.shape != (4, 4):
            raise ValueError(f"TCP pose must have shape (4, 4), got {self.tcp_pose.shape}")


@dataclass(frozen=True)
class GripperState:
    width_m: float
    timestamp_ms: float


@dataclass(frozen=True)
class RobotState:
    arm: ArmState
    gripper: GripperState
    timestamp_ms: float

    @property
    def qpos8(self) -> np.ndarray:
        return np.concatenate([self.arm.joints, np.array([self.gripper.width_m], dtype=np.float64)])

    @property
    def skew_ms(self) -> float:
        return abs(self.arm.timestamp_ms - self.gripper.timestamp_ms)


@dataclass(frozen=True)
class VisualFrame:
    color: np.ndarray | None
    depth: np.ndarray | None
    timestamp_ms: float
    intrinsics: np.ndarray | None = None
    depth_scale: float | None = None


@dataclass(frozen=True)
class TactileFrame:
    rectify: np.ndarray | None
    timestamp_ms: float


@dataclass(frozen=True)
class Observation:
    robot: RobotState
    visual: VisualFrame | None = None
    tactile: TactileFrame | None = None
    extras: dict[str, Any] = field(default_factory=dict)


ActionMode = Literal["tcp", "joint", "hold"]


@dataclass(frozen=True)
class PolicyAction:
    mode: ActionMode
    timestamp_ms: float
    target_tcp_pose: np.ndarray | None = None
    target_joints: np.ndarray | None = None
    target_gripper_width: float | None = None
    wrench: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.target_tcp_pose is not None:
            object.__setattr__(self, "target_tcp_pose", _array(self.target_tcp_pose))
            if self.target_tcp_pose.shape != (4, 4):
                raise ValueError(f"target_tcp_pose must have shape (4, 4), got {self.target_tcp_pose.shape}")
        if self.target_joints is not None:
            object.__setattr__(self, "target_joints", _array(self.target_joints))
            if self.target_joints.shape != (7,):
                raise ValueError(f"target_joints must have shape (7,), got {self.target_joints.shape}")
        if self.wrench is not None:
            object.__setattr__(self, "wrench", _array(self.wrench))
            if self.wrench.shape != (6,):
                raise ValueError(f"wrench must have shape (6,), got {self.wrench.shape}")


@dataclass(frozen=True)
class ActionResult:
    accepted: bool
    timestamp_ms: float
    reason: str = ""
    applied_action: PolicyAction | None = None

