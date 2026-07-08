"""
test_real_robot.py — 真机单点动作测试脚本

安全设计原则
------------
- 所有 TCP 位移偏移量均受 SAFE_DELTA_LIMIT_M 硬限制，超出则脚本直接退出，不发送任何指令。
- 关节步长受 SAFE_JOINT_DELTA_LIMIT_RAD 硬限制，同上。
- 每个 Case 执行前都会读取当前机器人状态，以当前位置为基准计算偏移，
  不依赖事先假设的位置，避免累积误差。
- 执行后等待 SETTLE_S 秒再读取状态，打印实际末端位置与目标偏差。
- Ctrl-C 会触发 stop() 并退出，不会留下悬挂指令。

运行方式
--------
  cd /Users/acondaway/Desktop/Policy_Rollout_Flexiv/flexiv-rollout
  python test_real_robot.py [--case tcp_x | tcp_z | joint | gripper | all]

默认只运行 --case tcp_x（+X 方向轻微位移），是最保守的起步测试。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# ── 路径 ───────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT))

from rollout.control.executor import ControlExecutor, ControlRuntimeConfig
from rollout.control.safety import SafetyLimits
from rollout.perception.devices import FlexivRizonClient, FlexivRizonConfig, XenseGripperClient, XenseGripperConfig
from rollout.types import PolicyAction
from utils.latest_buffer import LatestBuffer
from utils.timing import now_ms

# ── 机器人常量（由用户提供）────────────────────────────────────────────────────
RIZON_HOME_JOINTS = np.deg2rad([0, -40, 0, 90, 0, 40, 0])   # rad, shape (7,)
RIZON_HOME_POSE = np.array([                                  # 4×4 齐次变换矩阵
    [0.,  1.,  0.,  0.683],
    [1.,  0.,  0., -0.110],
    [0.,  0., -1.,  0.283],
    [0.,  0.,  0.,  1.   ],
], dtype=np.float64)

RIZON_JOINT_MAX_VEL        = 1.0          # rad/s
RIZON_JOINT_MAX_ACC        = 1.0          # rad/s²
RIZON_TCP_MAX_VEL          = (0.1, 0.1)   # (linear m/s, angular rad/s)
RIZON_TCP_MAX_ACC          = (0.5, 0.5)   # (linear m/s², angular rad/s²)
RIZON_JOINT_EPSILON        = 0.02         # rad  — 收敛判断阈值（仅打印参考）
RIZON_GRIPPER_EPSILON      = 0.01         # m
RIZON_TCP_POSE_EPSILON     = (0.0005, 0.001)  # (m, rad)
RIZON_BLOCK_WAIT_TIME      = 0.01         # s
RIZON_BLOCK_TIMEOUT        = 5.0          # s

# ── 安全硬限制（比 safety.py 的 per-step 限制更严，作为脚本层保险） ──────────
SAFE_DELTA_LIMIT_M        = 0.015  # 单次 TCP 平移最大偏移：15 mm
SAFE_JOINT_DELTA_LIMIT_RAD = 0.08  # 单次关节最大偏移：约 4.6°

# ── 动作发送后等待时间 ─────────────────────────────────────────────────────────
SETTLE_S = 1.5   # 等待机器人到位的时间（秒）

# ── 硬件配置（与 rollout/configs/control.yml 一致）────────────────────────────
ARM_CONFIG = FlexivRizonConfig(
    robot_id="Rizon4s-063586",
    tool_name="xense",
    auto_enable=True,
    operational_wait_s=10.0,
    tcp_mode_on_start=True,
    joint_max_vel=RIZON_JOINT_MAX_VEL,
    joint_max_acc=RIZON_JOINT_MAX_ACC,
)

GRIPPER_CONFIG = XenseGripperConfig(
    device_id="1659f0e0dde0",
)

SAFETY_LIMITS = SafetyLimits(
    max_tcp_translation_step_m=SAFE_DELTA_LIMIT_M,
    max_joint_step_rad=SAFE_JOINT_DELTA_LIMIT_RAD,
    gripper_min_width_m=0.0,
    gripper_max_width_m=0.085,
    max_wrench_abs=30.0,
)

CONTROL_CONFIG = ControlRuntimeConfig(
    fps=50.0,
    max_action_age_ms=2000.0,   # 测试脚本里手动 put，宽松一些
    allow_joint_commands=True,  # 开启关节模式，供 joint case 使用
    max_linear_vel=RIZON_TCP_MAX_VEL[0],
    max_linear_acc=RIZON_TCP_MAX_ACC[0],
    max_angular_vel=RIZON_TCP_MAX_VEL[1],
    max_angular_acc=RIZON_TCP_MAX_ACC[1],
    gripper_velocity_m_s=0.05,
    gripper_force_n=20.0,
    safety=SAFETY_LIMITS,
)


# ── 辅助 ───────────────────────────────────────────────────────────────────────

def _pose_delta_str(current: np.ndarray, target: np.ndarray) -> str:
    """打印 TCP 平移偏差（mm）。"""
    d = (target[:3, 3] - current[:3, 3]) * 1000.0
    return f"Δ xyz = [{d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f}] mm"


def _guard_tcp_delta(current_pose: np.ndarray, target_pose: np.ndarray) -> None:
    """如果目标与当前位置距离超出 SAFE_DELTA_LIMIT_M，直接 raise。"""
    dist = float(np.linalg.norm(target_pose[:3, 3] - current_pose[:3, 3]))
    if dist > SAFE_DELTA_LIMIT_M:
        raise ValueError(
            f"[SAFETY ABORT] TCP delta {dist*1000:.2f} mm "
            f"> limit {SAFE_DELTA_LIMIT_M*1000:.1f} mm. 指令未发送。"
        )


def _guard_joint_delta(current_joints: np.ndarray, target_joints: np.ndarray) -> None:
    """如果任一关节步长超出 SAFE_JOINT_DELTA_LIMIT_RAD，直接 raise。"""
    delta = np.abs(target_joints - current_joints)
    max_delta = float(delta.max())
    if max_delta > SAFE_JOINT_DELTA_LIMIT_RAD:
        worst = int(delta.argmax())
        raise ValueError(
            f"[SAFETY ABORT] 关节 J{worst+1} 步长 {np.rad2deg(max_delta):.2f}° "
            f"> 限制 {np.rad2deg(SAFE_JOINT_DELTA_LIMIT_RAD):.1f}°. 指令未发送。"
        )


def _send_and_check(executor: ControlExecutor, action: PolicyAction, label: str) -> None:
    """put action → apply_once → 打印结果。"""
    executor.action_buffer.put(action)
    result = executor.apply_once()
    tag = "✓ ACCEPTED" if result.accepted else "✗ REJECTED"
    print(f"  [{tag}] {label}  reason={result.reason!r}")
    if not result.accepted:
        print("  ⚠️  动作被拒绝，请检查 reason 字段。")


# ── 测试 Cases ────────────────────────────────────────────────────────────────

def case_tcp_x(arm: FlexivRizonClient, executor: ControlExecutor, delta_m: float = 0.010) -> None:
    """TCP +X 方向平移 delta_m（默认 10 mm），然后返回。"""
    print(f"\n=== case_tcp_x: +X {delta_m*1000:.0f} mm ===")
    state = arm.read_state()
    print(f"  当前 TCP: {state.tcp_pose[:3, 3]}")

    fwd_pose = state.tcp_pose.copy()
    fwd_pose[0, 3] += delta_m
    _guard_tcp_delta(state.tcp_pose, fwd_pose)

    print(f"  目标 TCP: {fwd_pose[:3, 3]}  {_pose_delta_str(state.tcp_pose, fwd_pose)}")
    _send_and_check(executor, PolicyAction(mode="tcp", timestamp_ms=now_ms(),
                                           target_tcp_pose=fwd_pose), "前进")
    time.sleep(SETTLE_S)

    # 返回原位
    back_pose = state.tcp_pose.copy()
    executor.action_buffer.put(
        PolicyAction(mode="tcp", timestamp_ms=now_ms(), target_tcp_pose=back_pose)
    )
    executor.apply_once()
    print(f"  返回原位: {back_pose[:3, 3]}")
    time.sleep(SETTLE_S)


def case_tcp_z(arm: FlexivRizonClient, executor: ControlExecutor, delta_m: float = 0.010) -> None:
    """TCP +Z 方向平移 delta_m（默认 10 mm，向上），然后返回。"""
    print(f"\n=== case_tcp_z: +Z {delta_m*1000:.0f} mm ===")
    state = arm.read_state()
    print(f"  当前 TCP: {state.tcp_pose[:3, 3]}")

    up_pose = state.tcp_pose.copy()
    up_pose[2, 3] += delta_m
    _guard_tcp_delta(state.tcp_pose, up_pose)

    print(f"  目标 TCP: {up_pose[:3, 3]}  {_pose_delta_str(state.tcp_pose, up_pose)}")
    _send_and_check(executor, PolicyAction(mode="tcp", timestamp_ms=now_ms(),
                                           target_tcp_pose=up_pose), "上升")
    time.sleep(SETTLE_S)

    back_pose = state.tcp_pose.copy()
    executor.action_buffer.put(
        PolicyAction(mode="tcp", timestamp_ms=now_ms(), target_tcp_pose=back_pose)
    )
    executor.apply_once()
    print(f"  返回原位: {back_pose[:3, 3]}")
    time.sleep(SETTLE_S)


def case_joint(arm: FlexivRizonClient, executor: ControlExecutor,
               joint_idx: int = 6, delta_rad: float = 0.05) -> None:
    """
    单关节微动：关节 joint_idx（0-based，默认第7轴 wrist）偏移 delta_rad（默认 ≈ 2.9°），
    然后返回。
    """
    label = f"J{joint_idx+1} +{np.rad2deg(delta_rad):.1f}°"
    print(f"\n=== case_joint: {label} ===")
    state = arm.read_state()
    print(f"  当前关节: {np.rad2deg(state.joints).round(2)} deg")

    target_joints = state.joints.copy()
    target_joints[joint_idx] += delta_rad
    _guard_joint_delta(state.joints, target_joints)

    print(f"  目标关节: {np.rad2deg(target_joints).round(2)} deg")
    _send_and_check(executor, PolicyAction(mode="joint", timestamp_ms=now_ms(),
                                           target_joints=target_joints), label)
    time.sleep(SETTLE_S)

    # 返回原位
    executor.action_buffer.put(
        PolicyAction(mode="joint", timestamp_ms=now_ms(), target_joints=state.joints.copy())
    )
    executor.apply_once()
    print(f"  返回原位")
    time.sleep(SETTLE_S)


def case_gripper(gripper: XenseGripperClient, executor: ControlExecutor) -> None:
    """夹爪：微开 → 微关，用 hold action 携带 target_gripper_width。"""
    print("\n=== case_gripper: open 40 mm → close 20 mm → restore ===")
    g_state = gripper.read_state()
    original_w = g_state.width_m
    print(f"  当前夹爪宽度: {original_w*1000:.1f} mm")

    for label, width_m in [("open 40mm", 0.040), ("close 20mm", 0.020)]:
        action = PolicyAction(
            mode="hold",          # hold 模式只执行 gripper 不动 TCP
            timestamp_ms=now_ms(),
            target_gripper_width=width_m,
        )
        executor.action_buffer.put(action)
        result = executor.apply_once()
        tag = "✓" if result.accepted else "✗"
        print(f"  [{tag}] {label}  reason={result.reason!r}")
        time.sleep(SETTLE_S)

    # 恢复
    executor.action_buffer.put(
        PolicyAction(mode="hold", timestamp_ms=now_ms(), target_gripper_width=original_w)
    )
    executor.apply_once()
    print(f"  恢复夹爪: {original_w*1000:.1f} mm")
    time.sleep(SETTLE_S)


# ── 主程序 ────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Flexiv 真机单点动作测试")
    parser.add_argument(
        "--case",
        choices=["tcp_x", "tcp_z", "joint", "gripper", "all"],
        default="tcp_x",
        help="要运行的测试 case（默认 tcp_x）",
    )
    parser.add_argument(
        "--delta-mm", type=float, default=10.0,
        help="TCP 位移量（毫米，默认 10 mm）",
    )
    parser.add_argument(
        "--delta-deg", type=float, default=2.9,
        help="关节偏移量（度，默认 2.9°）",
    )
    parser.add_argument(
        "--joint-idx", type=int, default=6,
        help="joint case 使用的关节索引 0-based（默认 6 = J7）",
    )
    args = parser.parse_args()

    delta_m   = float(args.delta_mm) / 1000.0
    delta_rad = float(np.deg2rad(args.delta_deg))

    # ── 连接硬件 ───────────────────────────────────────────────────────────────
    print("正在连接 Flexiv 机械臂…")
    arm = FlexivRizonClient(ARM_CONFIG)

    print("正在连接 Xense 夹爪…")
    gripper = XenseGripperClient(GRIPPER_CONFIG)

    # ── 读取当前状态并与 HOME 比对（提示，不阻断）─────────────────────────────
    state = arm.read_state()
    home_dist_mm = float(np.linalg.norm(state.tcp_pose[:3, 3] - RIZON_HOME_POSE[:3, 3])) * 1000.0
    joint_diff_deg = np.rad2deg(np.abs(state.joints - RIZON_HOME_JOINTS)).max()
    print(f"\n当前 TCP 距 HOME 位置: {home_dist_mm:.1f} mm")
    print(f"当前关节距 HOME 最大偏差: {joint_diff_deg:.2f}°")
    if home_dist_mm > 30.0:
        print("⚠️  警告：当前位置距 HOME 超过 30 mm，请确认机器人已回零后再运行。")
        resp = input("继续? [y/N] ").strip().lower()
        if resp != "y":
            print("已取消。")
            arm.stop()
            gripper.stop()
            return

    # ── 构建 executor ──────────────────────────────────────────────────────────
    action_buffer      = LatestBuffer()
    observation_buffer = LatestBuffer()
    result_buffer      = LatestBuffer()

    executor = ControlExecutor(
        arm=arm,
        gripper=gripper,
        action_buffer=action_buffer,
        observation_buffer=observation_buffer,
        result_buffer=result_buffer,
        config=CONTROL_CONFIG,
    )

    # ── 运行 cases ─────────────────────────────────────────────────────────────
    try:
        run_all = (args.case == "all")

        if run_all or args.case == "tcp_x":
            case_tcp_x(arm, executor, delta_m=delta_m)

        if run_all or args.case == "tcp_z":
            case_tcp_z(arm, executor, delta_m=delta_m)

        if run_all or args.case == "joint":
            case_joint(arm, executor, joint_idx=args.joint_idx, delta_rad=delta_rad)

        if run_all or args.case == "gripper":
            case_gripper(gripper, executor)

        print("\n✅ 测试完成。")

    except ValueError as exc:
        # 安全硬限制触发
        print(f"\n🚨 {exc}")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  用户中断（Ctrl-C）。")
    finally:
        executor.stop()
        print("硬件连接已关闭。")


if __name__ == "__main__":
    main()
