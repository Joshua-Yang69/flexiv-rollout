import argparse
import json
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.append(str(REPO_ROOT))

from vitacdreamer.policy_wrapper import ViTacDreamerFeatureExtractor  # noqa: E402


TARGET_HW = (224, 224)


def preprocess_image_sequence(images: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float().div_(255.0)
    return F.interpolate(tensor, size=TARGET_HW, mode="bilinear", align_corners=False)


def compose_bilateral_tactile(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    if left.shape != right.shape:
        raise ValueError(f"Left/right tactile shapes must match, got {left.shape} vs {right.shape}")
    tactile = torch.cat([left, right], dim=2)
    if tactile.shape[-2:] != TARGET_HW:
        tactile = F.interpolate(
            tactile.unsqueeze(0),
            size=TARGET_HW,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    return tactile


def build_batch_history_indices(start: int, end: int, history_len: int, sample_stride: int) -> torch.Tensor:
    ts = torch.arange(start, end, dtype=torch.long)
    offsets = torch.arange(history_len - 1, -1, -1, dtype=torch.long) * sample_stride
    return ts.unsqueeze(1) - offsets.unsqueeze(0)


def task_stem_from_name(task_name: str) -> str:
    if not task_name.startswith("sim-"):
        return task_name
    core = task_name[4:]

    # Task names are formatted as sim-{task_stem}-{task_config}-{num_episodes}.
    # Strip the episode count first so default-balanced-50/100 share the same
    # parsing path and map to the same multitask task embedding.
    parts = core.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        core = parts[0]

    for suffix in ("-default-ee-balanced", "-default-balanced", "-default-ee", "-default"):
        if core.endswith(suffix):
            return core[: -len(suffix)]

    # Backward compatibility for any legacy call sites that still pass names
    # with a hard-coded episode suffix in the config portion.
    for suffix in ("-default-ee-balanced-50", "-default-balanced-50", "-default-ee-50", "-default-50"):
        if core.endswith(suffix):
            return core[: -len(suffix)]

    parts = core.rsplit("-", 2)
    if len(parts) < 3:
        return core
    return parts[0]


def precompute_episode(extractor, hdf5_path, output_path, history_len, sample_stride, batch_size, task_id):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as root:
        episode_len = root["/action"].shape[0]
        features = []
        visual_seq = preprocess_image_sequence(root["/observations/images/cam_high"][()])
        left_tactile_seq = preprocess_image_sequence(root["/observations/images/tac_left"][()])
        right_tactile_path = "/observations/images/tac_right"
        if right_tactile_path not in root:
            raise KeyError(
                f"{hdf5_path} is missing {right_tactile_path}; "
                "bilateral ViTacDreamer feature precompute requires both tactile sides."
            )
        right_tactile_seq = preprocess_image_sequence(root[right_tactile_path][()])
        tactile_seq = torch.stack(
            [compose_bilateral_tactile(left_tactile_seq[i], right_tactile_seq[i]) for i in range(left_tactile_seq.shape[0])],
            dim=0,
        )
        action_seq = torch.from_numpy(root["/action"][()][:, :7]).float()

        zero_visual = torch.zeros((1, 3, *TARGET_HW), dtype=visual_seq.dtype)
        zero_tactile = torch.zeros((1, 3, *TARGET_HW), dtype=tactile_seq.dtype)
        zero_action = torch.zeros((1, 7), dtype=action_seq.dtype)

        visual_padded = torch.cat([zero_visual, visual_seq], dim=0)
        tactile_padded = torch.cat([zero_tactile, tactile_seq], dim=0)
        action_padded = torch.cat([zero_action, action_seq], dim=0)

        for start in tqdm(range(0, episode_len, batch_size), desc=hdf5_path.name, leave=False):
            end = min(start + batch_size, episode_len)
            history_indices = build_batch_history_indices(start, end, history_len, sample_stride)
            gather_indices = history_indices.clamp_min(-1).add(1)
            prev_history_indices = torch.where(
                history_indices < 0,
                torch.full_like(history_indices, -1),
                (history_indices - 1).clamp_min(0),
            )
            prev_gather_indices = prev_history_indices.add(1)
            prev_current_indices = torch.arange(start, end, dtype=torch.long).sub(1).clamp_min(0)

            current_tactile = tactile_seq[start:end]
            prev_current_tactile = tactile_seq[prev_current_indices]
            visual_history = visual_padded[gather_indices]
            tactile_history = tactile_padded[gather_indices]
            prev_tactile_history = tactile_padded[prev_gather_indices]
            action_history = action_padded[gather_indices]

            with torch.inference_mode():
                batch_features = extractor.extract_features_from_history(
                    current_tactile=current_tactile,
                    visual_history=visual_history,
                    tactile_history=tactile_history,
                    action_history=action_history,
                    prev_tactile_history=prev_tactile_history,
                    prev_current_tactile=prev_current_tactile,
                    task_id=torch.full((end - start,), task_id, dtype=torch.long),
                )
            features.append(batch_features.detach().cpu().float().numpy())

    np.save(output_path, np.concatenate(features, axis=0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", required=True, help="Example: sim-lift_bottle-default-50")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--task_order",
        nargs="+",
        default=["insert_HDMI", "insert_hole", "lift_bottle", "pull_out_key"],
        help="Task order used by ViTacDreamer multitask training task embeddings.",
    )
    args = parser.parse_args()

    with open("SIM_TASK_CONFIGS.json", "r") as f:
        task_configs = json.load(f)
    if args.task_name not in task_configs:
        raise KeyError(f"{args.task_name} not found in SIM_TASK_CONFIGS.json")

    dataset_dir = Path(task_configs[args.task_name]["dataset_dir"])
    num_episodes = task_configs[args.task_name]["num_episodes"]
    output_dir = Path(args.output_dir)
    task_stem = task_stem_from_name(args.task_name)
    if task_stem not in args.task_order:
        raise ValueError(f"Task {task_stem!r} is not in task_order={args.task_order}")
    task_id = args.task_order.index(task_stem)

    extractor = ViTacDreamerFeatureExtractor(
        checkpoint_path=args.checkpoint,
        freeze_encoder=True,
        device=args.device,
    )
    extractor.eval()
    if extractor.num_tasks and extractor.num_tasks != len(args.task_order):
        raise ValueError(
            f"Checkpoint has {extractor.num_tasks} task embeddings, "
            f"but task_order has {len(args.task_order)} entries: {args.task_order}"
        )

    for episode_idx in tqdm(range(num_episodes), desc="episodes"):
        hdf5_path = dataset_dir / f"episode_{episode_idx}.hdf5"
        output_path = output_dir / f"episode_{episode_idx}.npy"
        if output_path.exists() and not args.overwrite:
            continue
        precompute_episode(
            extractor=extractor,
            hdf5_path=hdf5_path,
            output_path=output_path,
            history_len=args.history_len,
            sample_stride=args.sample_stride,
            batch_size=args.batch_size,
            task_id=task_id,
        )

    print(f"Saved cached ViTacDreamer features to {output_dir}")


if __name__ == "__main__":
    main()
