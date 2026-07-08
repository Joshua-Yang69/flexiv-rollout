"""
test_rollout_sim.py — Policy rollout 模拟测试（关节空间）

功能
----
预先定义一个 joint pose 序列，按照指定频率（--hz）逐帧注入 action_buffer，
驱动 ControlExecutor 以完整 rollout 流水线的方式消费指令，精确复现真机部署
时的节奏。

设计原则
--------
- 使用项目现有的 LatestBuffer / RateLimiter / ControlExecutor，与真机路径
  共享同一套代码，模拟结果可直接对照真机行为。
- 默认使用 mock 驱动（MockArmClient / MockGripperClient），无需任何硬件。
- 可选 --real 切换为真机硬件（FlexivRizonClient + XenseGripperClient），
  此时 joint pose 序列会经过 safety.py 的步长裁剪再发送。
- action 线程（publisher）和 control 线程（executor）独立运行，中间通过
  LatestBuffer 解耦，与真机 rollout 完全一致。
- 每帧打印时间戳、target joints、实际 mock 状态及 safety messages。

运行方式
--------
  # mock 模拟（默认）
  python test_rollout_sim.py

  # 指定频率和步数
  python test_rollout_sim.py --hz 10 --steps 30

  # 真机模式（需要硬件就绪）
  python test_rollout_sim.py --real

  # 使用 tcp 模式的预设序列
  python test_rollout_sim.py --mode tcp
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from threading import Event, Thread

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from rollout.control.executor import ControlExecutor, ControlRuntimeConfig
from rollout.control.safety import SafetyLimits
from rollout.perception.devices import (
    FlexivRizonClient,
    FlexivRizonConfig,
    MockArmClient,
    MockGripperClient,
    XenseGripperClient,
    XenseGripperConfig,
)
from rollout.types import PolicyAction
from utils.latest_buffer import LatestBuffer
from utils.timing import RateLimiter, now_ms

# ── HOME 常量 ──────────────────────────────────────────────────────────────────
HOME_JOINTS = np.deg2rad([0, -40, 0, 90, 0, 40, 0])   # shape (7,)
HOME_POSE = np.array([
    [0.,  1.,  0.,  0.683],
    [1.,  0.,  0., -0.110],
    [0.,  0., -1.,  0.283],
    [0.,  0.,  0.,  1.   ],
], dtype=np.float64)

# ── 控制参数 ───────────────────────────────────────────────────────────────────
REAL_ARM_CONFIG = FlexivRizonConfig(
    robot_id="Rizon4s-063586",
    tool_name="xense",
    auto_enable=True,
    operational_wait_s=10.0,
    tcp_mode_on_start=False,   # 由 ensure_*_mode() 按需切换
)
REAL_GRIPPER_CONFIG = XenseGripperConfig(
    device_id="1659f0e0dde0",
    blocking_timeout_s=5.0,
    blocking_tolerance_mm=10.0,
)


# ── 预设 Joint Pose 序列 ───────────────────────────────────────────────────────

def make_joint_sequence(n_steps: int) -> np.ndarray:
    """
    构造 n_steps 帧的关节序列：以 HOME_JOINTS 为中心，
    J1（腰部）做 ±5° 正弦摆动，J4（肘部）做 ±3° 正弦摆动，
    其余关节保持 HOME 不动。

    返回 shape (n_steps, 7) 的数组，单位 rad。
    """
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    seq = np.tile(HOME_JOINTS.copy(), (n_steps, 1))   # (n_steps, 7)
    seq[:, 0] += np.deg2rad(5.0) * np.sin(t)          # J1 ±5°
    seq[:, 3] += np.deg2rad(3.0) * np.sin(2 * t)      # J4 ±3°
    return seq


def make_tcp_sequence(n_steps: int) -> list[np.ndarray]:
    """
    构造 n_steps 帧的 TCP pose 序列：以 HOME_POSE 为中心，
    X 方向做 ±10 mm 正弦运动。

    返回长度 n_steps 的 list，每个元素是 4×4 矩阵。
    """
    t = np.linspace(0, 2 * np.pi, n_steps, endpoint=False)
    poses = []
    for i in range(n_steps):
        p = HOME_POSE.copy()
        p[0, 3] += 0.010 * np.sin(t[i])    # X ±10 mm
        poses.append(p)
    return poses


# ── 核心：模拟 rollout ─────────────────────────────────────────────────────────

class SimRollout:
    """
    模拟 policy rollout 的完整循环。

    Publisher 线程：按 hz 频率逐帧从预设序列取 action → put 进 action_buffer。
    Executor 线程：serve_forever() 按同样频率消费 action_buffer → 调用 arm/gripper。
    主线程：等待结束，汇总统计。
    """

    def __init__(
        self,
        arm,
        gripper,
        sequence: list[PolicyAction],
        hz: float = 20.0,
        max_action_age_ms: float = 500.0,
    ) -> None:
        self.arm = arm
        self.gripper = gripper
        self.sequence = sequence
        self.hz = hz

        self.action_buffer = LatestBuffer()
        self.observation_buffer = LatestBuffer()
        self.result_buffer = LatestBuffer()

        self.executor = ControlExecutor(
            arm=arm,
            gripper=gripper,
            action_buffer=self.action_buffer,
            observation_buffer=self.observation_buffer,
            result_buffer=self.result_buffer,
            config=ControlRuntimeConfig(
                fps=hz,
                max_action_age_ms=max_action_age_ms,
                max_linear_vel=0.1,
                max_linear_acc=0.5,
                max_angular_vel=0.1,
                max_angular_acc=0.5,
                gripper_velocity_m_s=0.05,
                gripper_force_n=20.0,
                safety=SafetyLimits(
                    max_tcp_translation_step_m=0.015,
                    max_joint_step_rad=0.08,
                    gripper_min_width_m=0.0,
                    gripper_max_width_m=0.085,
                    max_wrench_abs=30.0,
                ),
            ),
        )

        self._stop = Event()
        self._results: list[dict] = []

    # ── publisher ─────────────────────────────────────────────────────────────

    def _publish_loop(self) -> None:
        limiter = RateLimiter(self.hz)
        for step, action in enumerate(self.sequence):
            if self._stop.is_set():
                break
            limiter.mark_start()

            # 更新时间戳，保证 max_action_age_ms 检查通过
            fresh = PolicyAction(
                mode=action.mode,
                timestamp_ms=now_ms(),
                target_tcp_pose=action.target_tcp_pose,
                target_joints=action.target_joints,
                target_gripper_width=action.target_gripper_width,
                wrench=action.wrench,
                metadata={**action.metadata, "step": step},
            )
            self.action_buffer.put(fresh)

            limiter.sleep()

        # 序列发完，发一个 hold 让 executor 安全停下
        self.action_buffer.put(
            PolicyAction(mode="hold", timestamp_ms=now_ms(),
                         metadata={"step": "hold_final"})
        )
        time.sleep(0.5)
        self._stop.set()

    # ── result collector ──────────────────────────────────────────────────────

    def _collect_loop(self) -> None:
        last_ver = 0
        while not self._stop.is_set():
            ver, result = self.result_buffer.wait_next(last_ver, timeout=0.1)
            if ver == last_ver:
                continue
            last_ver = ver
            if result is None:
                continue
            action = result.applied_action
            safety_msgs = (action.metadata.get("safety_messages", [])
                           if action is not None else [])
            step = (action.metadata.get("step", "?")
                    if action is not None else "?")
            record = {
                "step": step,
                "accepted": result.accepted,
                "reason": result.reason,
                "safety_clipped": bool(safety_msgs),
                "safety_messages": safety_msgs,
                "target_joints": (action.target_joints.copy()
                                  if action is not None and action.target_joints is not None
                                  else None),
                "target_tcp_xyz": (action.target_tcp_pose[:3, 3].copy()
                                   if action is not None and action.target_tcp_pose is not None
                                   else None),
                "gripper_width": (action.target_gripper_width
                                  if action is not None else None),
            }
            self._results.append(record)
            self._print_step(record)

    @staticmethod
    def _print_step(r: dict) -> None:
        tag = "✓" if r["accepted"] else "✗"
        step = r["step"]
        clip = " [CLIPPED]" if r["safety_clipped"] else ""
        reason = f"  reason={r['reason']!r}" if not r["accepted"] else ""

        if r["target_joints"] is not None:
            joints_deg = np.rad2deg(r["target_joints"])
            body = f"joints=[{', '.join(f'{v:+.2f}' for v in joints_deg)}]°"
        elif r["target_tcp_xyz"] is not None:
            xyz = r["target_tcp_xyz"] * 1000
            body = f"tcp=[{xyz[0]:+.1f}, {xyz[1]:+.1f}, {xyz[2]:+.1f}] mm"
        else:
            body = f"mode=hold"

        gripper = (f"  gripper={r['gripper_width']*1000:.1f}mm"
                   if r["gripper_width"] is not None else "")
        print(f"  [{tag}] step={step:>3}  {body}{gripper}{clip}{reason}")

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self) -> None:
        n = len(self.sequence)
        period_ms = 1000.0 / self.hz
        print(f"\n{'='*60}")
        print(f"  Rollout sim: {n} steps @ {self.hz:.0f} Hz "
              f"(Δt={period_ms:.1f} ms, total≈{n/self.hz:.1f}s)")
        print(f"  mode: {self.sequence[0].mode if self.sequence else 'n/a'}")
        print(f"{'='*60}")

        publisher = Thread(target=self._publish_loop, name="publisher", daemon=True)
        collector = Thread(target=self._collect_loop, name="collector", daemon=True)
        executor_thread = Thread(
            target=self.executor.serve_forever, args=(self._stop,),
            name="executor", daemon=True,
        )

        publisher.start()
        collector.start()
        executor_thread.start()

        publisher.join()
        self._stop.set()
        executor_thread.join(timeout=2.0)
        collector.join(timeout=1.0)

        self._print_summary()
        self.executor.stop()

    def _print_summary(self) -> None:
        total = len(self._results)
        accepted = sum(1 for r in self._results if r["accepted"])
        clipped = sum(1 for r in self._results if r["safety_clipped"])
        rejected = total - accepted

        print(f"\n{'='*60}")
        print(f"  Summary: {total} results collected")
        print(f"  Accepted : {accepted}")
        print(f"  Rejected : {rejected}")
        print(f"  Clipped  : {clipped}  (accepted but safety-trimmed)")
        print(f"{'='*60}\n")


# ── 构建 PolicyAction 序列 ─────────────────────────────────────────────────────

def build_joint_sequence(n_steps: int, gripper_width: float = 0.04) -> list[PolicyAction]:
    joints_seq = make_joint_sequence(n_steps)
    actions = []
    for i, joints in enumerate(joints_seq):
        # 后半段轻微开合夹爪
        g = gripper_width + 0.02 * np.sin(2 * np.pi * i / n_steps)
        actions.append(PolicyAction(
            mode="joint",
            timestamp_ms=now_ms(),
            target_joints=joints.astype(np.float64),
            target_gripper_width=float(np.clip(g, 0.0, 0.085)),
            metadata={"source": "sim_joint"},
        ))
    return actions


def build_tcp_sequence(n_steps: int, gripper_width: float = 0.04) -> list[PolicyAction]:
    poses = make_tcp_sequence(n_steps)
    actions = []
    for i, pose in enumerate(poses):
        g = gripper_width + 0.02 * np.sin(2 * np.pi * i / n_steps)
        actions.append(PolicyAction(
            mode="tcp",
            timestamp_ms=now_ms(),
            target_tcp_pose=pose,
            target_gripper_width=float(np.clip(g, 0.0, 0.085)),
            wrench=np.zeros(6, dtype=np.float64),
            metadata={"source": "sim_tcp"},
        ))
    return actions


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Policy rollout 模拟测试（joint / tcp 两种模式）"
    )
    parser.add_argument(
        "--mode", choices=["joint", "tcp"], default="joint",
        help="动作空间模式（默认 joint）",
    )
    parser.add_argument(
        "--hz", type=float, default=20.0,
        help="rollout 频率 Hz（默认 20）",
    )
    parser.add_argument(
        "--steps", type=int, default=40,
        help="序列帧数（默认 40，即 @ 20 Hz 共 2 秒）",
    )
    parser.add_argument(
        "--real", action="store_true",
        help="使用真机硬件（默认 mock）",
    )
    args = parser.parse_args()

    # ── 硬件 ──────────────────────────────────────────────────────────────────
    if args.real:
        print("连接真机硬件…")
        arm = FlexivRizonClient(REAL_ARM_CONFIG)
        gripper = XenseGripperClient(REAL_GRIPPER_CONFIG)
    else:
        print("使用 mock 硬件（无需机器人连接）")
        arm = MockArmClient()
        gripper = MockGripperClient(width_m=0.04)

    # ── 序列 ──────────────────────────────────────────────────────────────────
    if args.mode == "joint":
        sequence = build_joint_sequence(args.steps)
    else:
        sequence = build_tcp_sequence(args.steps)

    # ── 运行 ──────────────────────────────────────────────────────────────────
    sim = SimRollout(arm, gripper, sequence, hz=args.hz)
    try:
        sim.run()
    except KeyboardInterrupt:
        print("\n⚠️  用户中断（Ctrl-C）")
        sim._stop.set()
        sim.executor.stop()


if __name__ == "__main__":
    main()
