from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.models.server import make_model_server_from_config
from rollout.perception.server import make_perceiver_from_config
from utils.config import load_yaml
from utils.latest_buffer import LatestBuffer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one model inference on a mock observation.")
    parser.add_argument("--config", default=str(Path(__file__).parent / "configs" / "model_server.yml"))
    args = parser.parse_args()

    observation_buffer = LatestBuffer()
    action_buffer = LatestBuffer()
    perceiver = make_perceiver_from_config({}, output_buffer=observation_buffer)
    perceiver.capture_once()
    server = make_model_server_from_config(load_yaml(args.config), observation_buffer, action_buffer)
    action = server.infer_once()
    print(action)
    perceiver.stop()


if __name__ == "__main__":
    main()
