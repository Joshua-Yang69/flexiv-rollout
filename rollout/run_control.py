from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.control.executor import make_control_executor_from_config
from rollout.perception.server import make_perceiver_from_config
from rollout.types import PolicyAction
from utils.config import load_yaml
from utils.latest_buffer import LatestBuffer
from utils.timing import now_ms


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Apply a single action through the control executor.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "control.yml"))
    parser.add_argument(
        "--mode",
        choices=["hold", "tcp", "joint"],
        default="hold",
        help=(
            "Control mode for the test action (default: hold):\n"
            "  hold  — no motion, keep current pose\n"
            "         → NRT mode unchanged\n"
            "  tcp   — send current TCP pose as target\n"
            "         → NRT_CARTESIAN_MOTION_FORCE\n"
            "  joint — send current joint positions as target\n"
            "         → NRT_JOINT_IMPEDANCE"
        ),
    )
    args = parser.parse_args()

    observation_buffer = LatestBuffer()
    action_buffer = LatestBuffer()
    result_buffer = LatestBuffer()

    perceiver = make_perceiver_from_config({}, output_buffer=observation_buffer)
    observation = perceiver.capture_once()

    # Build the test action according to the requested mode.
    # For tcp/joint the target is the robot's *current* state so no motion
    # occurs — we are verifying that the executor accepts the action and
    # routes it to the correct NRT path, not that the robot actually moves.
    mode = args.mode
    if mode == "hold":
        action = PolicyAction(
            mode="hold",
            timestamp_ms=now_ms(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"source": "run_control", "mode": mode},
        )
    elif mode == "tcp":
        action = PolicyAction(
            mode="tcp",
            timestamp_ms=now_ms(),
            target_tcp_pose=observation.robot.arm.tcp_pose.copy(),
            target_gripper_width=observation.robot.gripper.width_m,
            wrench=np.zeros(6, dtype=np.float64),
            metadata={"source": "run_control", "mode": mode},
        )
    else:  # joint
        action = PolicyAction(
            mode="joint",
            timestamp_ms=now_ms(),
            target_joints=observation.robot.arm.joints.copy(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"source": "run_control", "mode": mode},
        )

    action_buffer.put(action)

    executor = make_control_executor_from_config(
        load_yaml(args.config),
        action_buffer=action_buffer,
        observation_buffer=observation_buffer,
        result_buffer=result_buffer,
        arm=perceiver.arm,
        gripper=perceiver.gripper,
    )

    print(f"mode={mode}")
    print(executor.apply_once())
    executor.stop()
    perceiver.stop()


if __name__ == "__main__":
    main()
