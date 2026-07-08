from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.control.executor import make_control_executor_from_config
from rollout.perception.server import make_perceiver_from_config
from rollout.types import PolicyAction
from utils.config import load_yaml
from utils.latest_buffer import LatestBuffer
from utils.timing import now_ms


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply a single hold action through the control executor.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "control.yml"))
    args = parser.parse_args()

    observation_buffer = LatestBuffer()
    action_buffer = LatestBuffer()
    result_buffer = LatestBuffer()
    perceiver = make_perceiver_from_config({}, output_buffer=observation_buffer)
    observation = perceiver.capture_once()
    action_buffer.put(
        PolicyAction(
            mode="hold",
            timestamp_ms=now_ms(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"source": "run_control"},
        )
    )
    executor = make_control_executor_from_config(
        load_yaml(args.config),
        action_buffer=action_buffer,
        observation_buffer=observation_buffer,
        result_buffer=result_buffer,
        arm=perceiver.arm,
        gripper=perceiver.gripper,
    )
    print(executor.apply_once())
    executor.stop()
    perceiver.stop()


if __name__ == "__main__":
    main()
