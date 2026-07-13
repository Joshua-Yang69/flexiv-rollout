from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rollout.runtime import RolloutRuntime
from utils.config import load_yaml

# Real-robot task order must match the ViTacDreamer encoder's num_tasks embedding.
TASK_ORDER = ["insert_tube", "wipe_board"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run perception, model, and control loops in one process."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).parent / "configs" / "rollout.yml"),
        help="Path to rollout YAML config.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Run duration in seconds (default: run until Ctrl-C).",
    )
    parser.add_argument(
        "--task",
        choices=TASK_ORDER,
        default=None,
        help=(
            "Task name for ViTacDreamer encoder task embedding "
            f"({', '.join(f'{i}={t}' for i, t in enumerate(TASK_ORDER))}). "
            "Overrides policy.vt_muse.task_id in the config."
        ),
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help="Task ID integer (alternative to --task). Overrides config.",
    )
    args = parser.parse_args()

    config = load_yaml(args.config)

    # Resolve task_id from CLI and inject into vt_muse config
    task_id: int | None = None
    if args.task is not None:
        task_id = TASK_ORDER.index(args.task)
    elif args.task_id is not None:
        task_id = args.task_id

    if task_id is not None:
        vt_muse_cfg = (
            config
            .setdefault("model_server", {})
            .setdefault("policy", {})
            .setdefault("vt_muse", {})
        )
        vt_muse_cfg["task_id"] = task_id
        print(f"[run_rollout] task_id={task_id} ({TASK_ORDER[task_id] if task_id < len(TASK_ORDER) else '?'})")

    runtime = RolloutRuntime.from_config(config)
    runtime.run(duration_s=args.duration_s)


if __name__ == "__main__":
    main()
