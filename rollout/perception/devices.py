from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from rollout.types import ArmState, GripperState, TactileFrame, VisualFrame
from utils.imports import optional_import
from utils.timing import now_ms
from utils.transforms import flexiv_pose_to_mat, mat_to_flexiv_pose


class ArmClient(Protocol):
    def read_state(self) -> ArmState:
        ...

    def send_cartesian_motion_force(
        self,
        pose: np.ndarray,
        wrench: np.ndarray,
        max_linear_vel: float,
        max_linear_acc: float,
        max_angular_vel: float,
        max_angular_acc: float,
    ) -> None:
        ...

    def send_joint_positions(self, joints: np.ndarray, velocities: np.ndarray | None = None) -> None:
        ...

    def stop(self) -> None:
        ...


class GripperClient(Protocol):
    def read_state(self) -> GripperState:
        ...

    def move(self, width_m: float, velocity_m_s: float, force_n: float) -> bool:
        ...

    def stop(self) -> None:
        ...


class VisualClient(Protocol):
    def read_frame(self) -> VisualFrame:
        ...

    def stop(self) -> None:
        ...


class TactileClient(Protocol):
    def read_frame(self) -> TactileFrame:
        ...

    def stop(self) -> None:
        ...


@dataclass
class FlexivRizonConfig:
    robot_id: str = "Rizon4s-063231"
    tool_name: str = "Flange"
    auto_enable: bool = True
    operational_wait_s: float = 10.0
    tcp_mode_on_start: bool = True
    joint_max_vel: float = 1.0
    joint_max_acc: float = 1.0


class FlexivRizonClient:
    """Minimal Flexiv Rizon wrapper for state read and NRT TCP commands."""

    def __init__(self, config: FlexivRizonConfig) -> None:
        self.config = config
        self._rdk = optional_import("flexivrdk", "Install Flexiv RDK on the robot deployment machine.")
        self.robot = self._rdk.Robot(config.robot_id)
        if config.auto_enable:
            self._enable_robot()
        if config.tcp_mode_on_start:
            self.switch_tcp_mode()

    def _enable_robot(self) -> None:
        if self.robot.fault() and not self.robot.ClearFault():
            raise RuntimeError(f"Failed to clear Flexiv fault for {self.config.robot_id}")
        self.robot.Enable()
        deadline = time.time() + self.config.operational_wait_s
        while not self.robot.operational():
            if time.time() > deadline:
                raise RuntimeError(f"Flexiv robot {self.config.robot_id} did not become operational.")
            time.sleep(0.1)
        self.robot.SwitchMode(self._rdk.Mode.IDLE)
        tool = self._rdk.Tool(self.robot)
        if not tool.exist(self.config.tool_name):
            raise RuntimeError(f"Flexiv tool '{self.config.tool_name}' not found.")
        tool.Switch(self.config.tool_name)

    def switch_tcp_mode(self) -> None:
        self.robot.SwitchMode(self._rdk.Mode.NRT_CARTESIAN_MOTION_FORCE)

    def switch_joint_mode(self) -> None:
        self.robot.SwitchMode(self._rdk.Mode.NRT_JOINT_IMPEDANCE)

    def read_state(self) -> ArmState:
        state = self.robot.states()
        return ArmState(
            joints=np.asarray(state.q, dtype=np.float64),
            tcp_pose=flexiv_pose_to_mat(np.asarray(state.tcp_pose, dtype=np.float64)),
            timestamp_ms=now_ms(),
        )

    def send_cartesian_motion_force(
        self,
        pose: np.ndarray,
        wrench: np.ndarray,
        max_linear_vel: float,
        max_linear_acc: float,
        max_angular_vel: float,
        max_angular_acc: float,
    ) -> None:
        self.robot.SendCartesianMotionForce(
            mat_to_flexiv_pose(pose).tolist(),
            np.asarray(wrench, dtype=np.float64).tolist(),
            max_linear_vel=float(max_linear_vel),
            max_linear_acc=float(max_linear_acc),
            max_angular_vel=float(max_angular_vel),
            max_angular_acc=float(max_angular_acc),
        )

    def send_joint_positions(self, joints: np.ndarray, velocities: np.ndarray | None = None) -> None:
        joints = np.asarray(joints, dtype=np.float64)
        velocities = np.zeros(7, dtype=np.float64) if velocities is None else np.asarray(velocities, dtype=np.float64)
        max_vel = np.full(7, self.config.joint_max_vel, dtype=np.float64)
        max_acc = np.full(7, self.config.joint_max_acc, dtype=np.float64)
        self.robot.SendJointPosition(joints.tolist(), velocities.tolist(), max_vel.tolist(), max_acc.tolist())

    def stop(self) -> None:
        self.robot.Stop()


@dataclass
class XenseGripperConfig:
    device_id: str = "5e77ff097831"
    max_width_m: float = 0.085
    max_velocity_m_s: float = 0.35
    max_force_n: float = 60.0
    blocking_timeout_s: float = -1.0
    blocking_tolerance_mm: float = 1.0


class XenseGripperClient:
    def __init__(self, config: XenseGripperConfig) -> None:
        self.config = config
        module = optional_import("xensegripper", "Install the Xense gripper SDK on the robot deployment machine.")
        self.gripper = module.XenseGripper.create(config.device_id)

    def read_state(self) -> GripperState:
        status = self.gripper.get_gripper_status()
        return GripperState(width_m=float(status["position"]) / 1000.0, timestamp_ms=now_ms())

    def move(self, width_m: float, velocity_m_s: float = 0.08, force_n: float = 27.0) -> bool:
        target_width_mm = np.clip(width_m, 0.0, self.config.max_width_m) * 1000.0
        target_velocity_mm_s = np.clip(velocity_m_s, 0.0, self.config.max_velocity_m_s) * 1000.0
        target_force_n = np.clip(force_n, 0.0, self.config.max_force_n)
        if self.config.blocking_timeout_s <= 0:
            self.gripper.set_position(target_width_mm, target_velocity_mm_s, target_force_n)
            return True
        return bool(
            self.gripper.set_position_sync(
                target_width_mm,
                target_velocity_mm_s,
                target_force_n,
                timeout=self.config.blocking_timeout_s,
                tolerance=self.config.blocking_tolerance_mm,
            )
        )

    def set_led_color(self, rgb: tuple[int, int, int]) -> None:
        self.gripper.set_led_color(*rgb)

    def stop(self) -> None:
        # Xense SDK exposes no explicit stop in the reference code.
        return None


@dataclass
class RealSenseD415Config:
    serial: str | None = None
    width: int = 640
    height: int = 480
    fps: int = 30
    enable_color: bool = True
    enable_depth: bool = True
    align_depth_to_color: bool = True


class RealSenseD415Client:
    def __init__(self, config: RealSenseD415Config) -> None:
        self.config = config
        self._rs = optional_import("pyrealsense2", "Install librealsense/pyrealsense2 for D415 streaming.")
        self.pipeline = self._rs.pipeline()
        self.rs_config = self._rs.config()
        if config.serial:
            self.rs_config.enable_device(config.serial)
        if config.enable_depth:
            self.rs_config.enable_stream(
                self._rs.stream.depth,
                config.width,
                config.height,
                self._rs.format.z16,
                config.fps,
            )
        if config.enable_color:
            self.rs_config.enable_stream(
                self._rs.stream.color,
                config.width,
                config.height,
                self._rs.format.bgr8,
                config.fps,
            )
        profile = self.pipeline.start(self.rs_config)
        self.align = self._rs.align(self._rs.stream.color) if config.enable_depth and config.enable_color else None
        self.depth_scale: float | None = None
        if config.enable_depth:
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        self.intrinsics: np.ndarray | None = None
        if config.enable_color:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            intr = color_frame.get_profile().as_video_stream_profile().get_intrinsics()
            self.intrinsics = np.array([intr.ppx, intr.ppy, intr.fx, intr.fy], dtype=np.float64)

    def read_frame(self) -> VisualFrame:
        frames = self.pipeline.wait_for_frames()
        if self.align is not None and self.config.align_depth_to_color:
            frames = self.align.process(frames)
        color = None
        depth = None
        if self.config.enable_color:
            color = np.asanyarray(frames.get_color_frame().get_data(), dtype=np.uint8).copy()
        if self.config.enable_depth:
            depth = np.asanyarray(frames.get_depth_frame().get_data(), dtype=np.uint16).copy()
        return VisualFrame(
            color=color,
            depth=depth,
            timestamp_ms=now_ms(),
            intrinsics=None if self.intrinsics is None else self.intrinsics.copy(),
            depth_scale=self.depth_scale,
        )

    def stop(self) -> None:
        self.pipeline.stop()


@dataclass
class XenseTactileConfig:
    device_id: str | None = None
    ip_address: str | None = None
    mac_addr: str | None = None
    config_path: str | None = None
    video_path: str | None = None
    use_gpu: bool = False


class XenseTactileClient:
    def __init__(self, config: XenseTactileConfig) -> None:
        self.config = config
        module = optional_import("xensesdk", "Install the Xense sensor SDK for tactile streaming.")
        self._sensor_cls = module.Sensor
        kwargs: dict[str, Any] = {}
        for key in ("ip_address", "mac_addr", "config_path", "video_path", "use_gpu"):
            value = getattr(config, key)
            if value is not None:
                kwargs[key] = value
        self.sensor = self._sensor_cls.create(config.device_id, **kwargs)

    def read_frame(self) -> TactileFrame:
        rectify = self.sensor.selectSensorInfo(self._sensor_cls.OutputType.Rectify)
        return TactileFrame(rectify=np.asarray(rectify).copy(), timestamp_ms=now_ms())

    def stop(self) -> None:
        self.sensor.release()


class MockArmClient:
    def __init__(self) -> None:
        self._joints = np.zeros(7, dtype=np.float64)
        self._pose = np.eye(4, dtype=np.float64)

    def read_state(self) -> ArmState:
        return ArmState(self._joints.copy(), self._pose.copy(), now_ms())

    def send_cartesian_motion_force(
        self,
        pose: np.ndarray,
        wrench: np.ndarray,
        max_linear_vel: float,
        max_linear_acc: float,
        max_angular_vel: float,
        max_angular_acc: float,
    ) -> None:
        del wrench, max_linear_vel, max_linear_acc, max_angular_vel, max_angular_acc
        self._pose = np.asarray(pose, dtype=np.float64).copy()

    def send_joint_positions(self, joints: np.ndarray, velocities: np.ndarray | None = None) -> None:
        del velocities
        self._joints = np.asarray(joints, dtype=np.float64).copy()

    def stop(self) -> None:
        return None


class MockGripperClient:
    def __init__(self, width_m: float = 0.04) -> None:
        self._width_m = float(width_m)

    def read_state(self) -> GripperState:
        return GripperState(width_m=self._width_m, timestamp_ms=now_ms())

    def move(self, width_m: float, velocity_m_s: float = 0.08, force_n: float = 27.0) -> bool:
        del velocity_m_s, force_n
        self._width_m = float(width_m)
        return True

    def stop(self) -> None:
        return None


class MockVisualClient:
    def __init__(self, shape: tuple[int, int, int] = (480, 640, 3)) -> None:
        self.shape = shape

    def read_frame(self) -> VisualFrame:
        return VisualFrame(
            color=np.zeros(self.shape, dtype=np.uint8),
            depth=np.zeros(self.shape[:2], dtype=np.uint16),
            timestamp_ms=now_ms(),
            intrinsics=np.array([self.shape[1] / 2, self.shape[0] / 2, 600.0, 600.0], dtype=np.float64),
            depth_scale=0.001,
        )

    def stop(self) -> None:
        return None


class MockTactileClient:
    def __init__(self, shape: tuple[int, int, int] = (240, 320, 3)) -> None:
        self.shape = shape

    def read_frame(self) -> TactileFrame:
        return TactileFrame(rectify=np.zeros(self.shape, dtype=np.uint8), timestamp_ms=now_ms())

    def stop(self) -> None:
        return None


def make_arm_client(config: dict[str, Any]) -> ArmClient:
    driver = config.get("driver", "mock")
    if driver == "mock":
        return MockArmClient()
    if driver == "flexiv":
        return FlexivRizonClient(FlexivRizonConfig(**config.get("flexiv", {})))
    raise ValueError(f"Unknown arm driver: {driver}")


def make_gripper_client(config: dict[str, Any]) -> GripperClient:
    driver = config.get("driver", "mock")
    if driver == "mock":
        return MockGripperClient(width_m=float(config.get("mock_width_m", 0.04)))
    if driver == "xense":
        return XenseGripperClient(XenseGripperConfig(**config.get("xense", {})))
    raise ValueError(f"Unknown gripper driver: {driver}")


def make_visual_client(config: dict[str, Any]) -> VisualClient | None:
    if not config.get("enabled", True):
        return None
    driver = config.get("driver", "mock")
    if driver == "mock":
        return MockVisualClient()
    if driver == "realsense_d415":
        return RealSenseD415Client(RealSenseD415Config(**config.get("realsense", {})))
    raise ValueError(f"Unknown visual driver: {driver}")


def make_tactile_client(config: dict[str, Any]) -> TactileClient | None:
    if not config.get("enabled", True):
        return None
    driver = config.get("driver", "mock")
    if driver == "mock":
        return MockTactileClient()
    if driver == "xense":
        return XenseTactileClient(XenseTactileConfig(**config.get("xense", {})))
    raise ValueError(f"Unknown tactile driver: {driver}")

