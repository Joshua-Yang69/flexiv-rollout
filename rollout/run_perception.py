from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.perception.server import make_perceiver_from_config
from utils.config import load_yaml
from utils.latest_buffer import LatestBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or probe the perception loop.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "perception.yml"))
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()

    buffer = LatestBuffer()
    perceiver = make_perceiver_from_config(load_yaml(args.config), output_buffer=buffer)
    try:
        if args.once:
            observation = perceiver.capture_once()
            print(
                {
                    "qpos8": observation.robot.qpos8.tolist(),
                    "state_skew_ms": observation.robot.skew_ms,
                    "has_visual": observation.visual is not None,
                    "has_tactile": observation.tactile is not None,
                }
            )
            return

        if args.duration_s is None:
            perceiver.serve_forever()
        else:
            deadline = time.time() + args.duration_s
            while time.time() < deadline:
                perceiver.capture_once()
                time.sleep(1.0 / perceiver.config.fps)
    finally:
        perceiver.stop()


if __name__ == "__main__":
    main()
