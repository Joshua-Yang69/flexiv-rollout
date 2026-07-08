import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from references.models.ACT.act_policy import ACTPolicy
from references.models.ACT.utils import TacArenaDataset


def _task_template_vars(task_name: str):
    name = task_name
    if name.startswith("sim-"):
        name = name[4:]
    parts = name.rsplit("-", 2)
    if len(parts) >= 3:
        task_stem, task_config, expert_data_num = parts[0], parts[1], parts[2]
        task_config_ep = f"{task_config}-{expert_data_num}"
    else:
        task_stem = name
        task_config = ""
        expert_data_num = ""
        task_config_ep = ""
    return {
        "task_name": task_name,
        "task_stem": task_stem,
        "task_config": task_config,
        "expert_data_num": expert_data_num,
        "task_config_ep": task_config_ep,
    }


def _resolve_path_template(path_value, task_name: str):
    if not path_value:
        return path_value
    return path_value.format(**_task_template_vars(task_name))


def _resolve_task_id(cfg, task_name: str):
    order = cfg.get("vitacdreamer_task_order")
    if order is None:
        return None
    name = task_name[4:] if task_name.startswith("sim-") else task_name
    task_stem = next((stem for stem in order if name == stem or name.startswith(f"{stem}-")), None)
    if task_stem is None:
        task_stem = _task_template_vars(task_name)["task_stem"]
    if task_stem not in order:
        raise ValueError(f"Task {task_stem!r} is not in vitacdreamer_task_order={order}")
    return order.index(task_stem)


def _move_to_device(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    return value


def _select_subset(dataset, max_windows: int, seed: int):
    if max_windows <= 0 or max_windows >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(dataset), size=max_windows, replace=False))
    return Subset(dataset, indices.tolist())


def _accumulate_metrics(pred, target, valid, action_mean, action_std, sums):
    # pred/target: [B, Q, D], valid: [B, Q]
    valid3 = valid.unsqueeze(-1)
    pred_valid = pred[valid]
    target_valid = target[valid]
    if pred_valid.numel() == 0:
        return

    diff_norm = pred_valid - target_valid
    pred_raw = pred_valid * action_std + action_mean
    target_raw = target_valid * action_std + action_mean
    diff_raw = pred_raw - target_raw

    n = pred_valid.shape[0]
    sums["n"] += n
    sums["norm_abs"] += diff_norm.abs().sum(dim=0).cpu()
    sums["norm_sq"] += (diff_norm ** 2).sum(dim=0).cpu()
    sums["raw_abs"] += diff_raw.abs().sum(dim=0).cpu()
    sums["raw_sq"] += (diff_raw ** 2).sum(dim=0).cpu()

    first_mask = valid[:, 0]
    if first_mask.any():
        fp = pred[:, 0][first_mask]
        ft = target[:, 0][first_mask]
        fd_norm = fp - ft
        fd_raw = fp * action_std + action_mean - (ft * action_std + action_mean)
        fn = fp.shape[0]
        sums["first_n"] += fn
        sums["first_norm_abs"] += fd_norm.abs().sum(dim=0).cpu()
        sums["first_norm_sq"] += (fd_norm ** 2).sum(dim=0).cpu()
        sums["first_raw_abs"] += fd_raw.abs().sum(dim=0).cpu()
        sums["first_raw_sq"] += (fd_raw ** 2).sum(dim=0).cpu()


def _finalize(prefix, n, abs_sum, sq_sum):
    if n == 0:
        return {}
    l1_dim = abs_sum / n
    mse_dim = sq_sum / n
    return {
        f"{prefix}_count": int(n),
        f"{prefix}_l1_mean": float(l1_dim.mean()),
        f"{prefix}_mse_mean": float(mse_dim.mean()),
        f"{prefix}_rmse_mean": float(torch.sqrt(mse_dim).mean()),
        f"{prefix}_gripper_l1": float(l1_dim[-1]),
        f"{prefix}_gripper_mse": float(mse_dim[-1]),
        f"{prefix}_gripper_rmse": float(torch.sqrt(mse_dim[-1])),
        f"{prefix}_per_dim_l1": [float(x) for x in l1_dim],
        f"{prefix}_per_dim_mse": [float(x) for x in mse_dim],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate offline action prediction error on ACT-format HDF5 training data."
    )
    parser.add_argument("--task_name", required=True, help="e.g. sim-lift_bottle-default-balanced-100")
    parser.add_argument("--config_path", required=True)
    parser.add_argument("--ckpt_dir", required=True)
    parser.add_argument("--policy_checkpoint", default=None)
    parser.add_argument("--stats_path", default=None)
    parser.add_argument("--vitacdreamer_checkpoint", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--max_windows", type=int, default=4096)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    act_dir = Path(__file__).resolve().parent
    sim_cfg_path = act_dir / "SIM_TASK_CONFIGS.json"
    with open(sim_cfg_path, "r") as f:
        sim_cfg = json.load(f)
    task_cfg = sim_cfg[args.task_name]

    ckpt_dir = Path(args.ckpt_dir)
    stats_path = Path(args.stats_path) if args.stats_path else ckpt_dir / "dataset_stats.pkl"
    ckpt_path = Path(args.policy_checkpoint) if args.policy_checkpoint else ckpt_dir / "policy_best.ckpt"

    with open(stats_path, "rb") as f:
        stats = pickle.load(f)

    if args.vitacdreamer_checkpoint:
        cfg["vitacdreamer_checkpoint"] = args.vitacdreamer_checkpoint

    cfg.update(
        {
            "task_name": args.task_name,
            "ckpt_dir": str(ckpt_dir),
            "policy_checkpoint": str(ckpt_path),
            "stats_path": str(stats_path),
            "seed": 0,
            "device": args.device,
        }
    )

    dataset_dir = Path(task_cfg["dataset_dir"])
    if not dataset_dir.is_absolute():
        dataset_dir = act_dir / dataset_dir
    num_episodes = int(task_cfg["num_episodes"])
    if not dataset_dir.exists() or not any(dataset_dir.glob("episode_*.hdf5")):
        raise FileNotFoundError(
            f"ACT dataset not found or empty: {dataset_dir}. "
            "Run process_data.sh first or sync the processed ACT HDF5 directory."
        )
    task_id = _resolve_task_id(cfg, args.task_name)

    dataset = TacArenaDataset(
        episode_ids=np.arange(num_episodes),
        dataset_dir=str(dataset_dir),
        camera_names=cfg["camera_names"],
        tactile_names=cfg.get("tactile_names", []),
        norm_stats=stats,
        chunk_size=cfg["chunk_size"],
        use_vitacdreamer_feature=cfg.get("use_vitacdreamer_feature", False),
        vitacdreamer_history_len=cfg.get("vitacdreamer_history_len", 5),
        vitacdreamer_sample_stride=cfg.get("vitacdreamer_sample_stride", 5),
        vitacdreamer_feature_cache_dir=_resolve_path_template(
            cfg.get("vitacdreamer_feature_cache_dir"), args.task_name
        ),
        vitacdreamer_task_id=task_id,
    )
    eval_dataset = _select_subset(dataset, args.max_windows, args.sample_seed)
    loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device.startswith("cuda"),
    )

    policy = ACTPolicy(cfg).to(args.device)
    load_status = policy.load_state_dict(torch.load(ckpt_path, map_location=args.device))
    policy.eval()
    for param in policy.parameters():
        param.requires_grad_(False)
    if policy.feature_extractor is not None:
        policy.feature_extractor.freeze_encoder = True
        policy.feature_extractor.model.eval()

    action_mean = torch.as_tensor(stats["action_mean"], dtype=torch.float32, device=args.device)
    action_std = torch.as_tensor(stats["action_std"], dtype=torch.float32, device=args.device)
    action_dim = action_mean.numel()
    sums = {
        "n": 0,
        "first_n": 0,
        "norm_abs": torch.zeros(action_dim),
        "norm_sq": torch.zeros(action_dim),
        "raw_abs": torch.zeros(action_dim),
        "raw_sq": torch.zeros(action_dim),
        "first_norm_abs": torch.zeros(action_dim),
        "first_norm_sq": torch.zeros(action_dim),
        "first_raw_abs": torch.zeros(action_dim),
        "first_raw_sq": torch.zeros(action_dim),
    }

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 6:
                cam, tac, qpos, actions, is_pad, vitac_inputs = batch
            else:
                cam, tac, qpos, actions, is_pad = batch
                vitac_inputs = None
            cam = cam.to(args.device, non_blocking=True)
            tac = tac.to(args.device, non_blocking=True)
            qpos = qpos.to(args.device, non_blocking=True)
            actions = actions.to(args.device, non_blocking=True)
            is_pad = is_pad.to(args.device, non_blocking=True)
            vitac_inputs = _move_to_device(vitac_inputs, args.device)

            pred = policy(qpos, cam, tac, vitac_inputs=vitac_inputs)
            target = actions[:, : pred.shape[1]]
            valid = ~is_pad[:, : pred.shape[1]]
            _accumulate_metrics(pred, target, valid, action_mean, action_std, sums)

    result = {
        "task_name": args.task_name,
        "dataset_dir": str(dataset_dir),
        "num_episodes": num_episodes,
        "dataset_windows": len(dataset),
        "evaluated_windows": len(eval_dataset),
        "ckpt_dir": str(ckpt_dir),
        "policy_checkpoint": str(ckpt_path),
        "stats_path": str(stats_path),
        "load_status": str(load_status),
        "use_vitacdreamer_feature": bool(cfg.get("use_vitacdreamer_feature", False)),
        "finetune_vitacdreamer_encoder": bool(cfg.get("finetune_vitacdreamer_encoder", False)),
        "tactile_names": cfg.get("tactile_names", []),
    }
    result.update(
        _finalize(
            "all_queries_norm",
            sums["n"],
            sums["norm_abs"],
            sums["norm_sq"],
        )
    )
    result.update(
        _finalize(
            "all_queries_raw",
            sums["n"],
            sums["raw_abs"],
            sums["raw_sq"],
        )
    )
    result.update(
        _finalize(
            "first_query_norm",
            sums["first_n"],
            sums["first_norm_abs"],
            sums["first_norm_sq"],
        )
    )
    result.update(
        _finalize(
            "first_query_raw",
            sums["first_n"],
            sums["first_raw_abs"],
            sums["first_raw_sq"],
        )
    )

    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")


if __name__ == "__main__":
    main()
