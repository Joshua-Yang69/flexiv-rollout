from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.runtime import RolloutRuntime


def main() -> None:
    parser = argparse.ArgumentParser(description="Run perception, model, and control loops in one process.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "rollout.yml"))
    parser.add_argument("--duration-s", type=float, default=None)
    args = parser.parse_args()

    runtime = RolloutRuntime.from_yaml(args.config)
    runtime.run(duration_s=args.duration_s)


if __name__ == "__main__":
    main()
