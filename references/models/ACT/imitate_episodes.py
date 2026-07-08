import os
import torch
import torch.distributed as dist
import numpy as np
import pickle
import argparse
import json
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from copy import deepcopy
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from references.models.ACT.utils import load_data  # data functions
from references.models.ACT.utils import compute_dict_mean, set_seed, detach_dict  # helper functions
from references.models.ACT.act_policy import ACTPolicy, CNNMLPPolicy

import IPython
e = IPython.embed
_METRIC_PROCESS_GROUP = None


def is_dist_enabled():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_dist_enabled() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def setup_distributed(args):
    global _METRIC_PROCESS_GROUP
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        args["distributed"] = False
        args["rank"] = 0
        args["local_rank"] = 0
        return

    args["distributed"] = True
    args["rank"] = int(os.environ["RANK"])
    args["local_rank"] = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args["local_rank"])
    dist.init_process_group(backend="nccl")
    _METRIC_PROCESS_GROUP = dist.new_group(backend="gloo")
    args["device"] = f"cuda:{args['local_rank']}"


def cleanup_distributed():
    if is_dist_enabled():
        dist.destroy_process_group()


def reduce_loss_dict(loss_dict, device):
    if not is_dist_enabled():
        return loss_dict
    keys = sorted(loss_dict.keys())
    values = torch.stack([loss_dict[key].detach().cpu().to(dtype=torch.float64) for key in keys])
    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=_METRIC_PROCESS_GROUP)
    values /= dist.get_world_size()
    return {key: value.to(device) for key, value in zip(keys, values)}


def reduce_weighted_loss_dict(loss_sum_dict, count, device):
    if not is_dist_enabled():
        return {
            key: value / max(count, 1)
            for key, value in loss_sum_dict.items()
        }

    keys = sorted(loss_sum_dict.keys())
    values = torch.stack([loss_sum_dict[key].detach().cpu().to(dtype=torch.float64) for key in keys])
    count_tensor = torch.tensor(float(count), dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM, group=_METRIC_PROCESS_GROUP)
    dist.all_reduce(count_tensor, op=dist.ReduceOp.SUM, group=_METRIC_PROCESS_GROUP)
    count_tensor = count_tensor.clamp_min(1.0)
    return {key: (value / count_tensor).to(device) for key, value in zip(keys, values)}


def policy_state_dict_for_save(policy):
    state_dict = policy.state_dict()
    cleaned = {}
    for key, value in state_dict.items():
        key = key.replace("module.", "", 1)
        key = key.replace("model.module.", "model.", 1)
        cleaned[key] = value
    return cleaned


def _task_template_vars(task_name):
    if not task_name.startswith("sim-"):
        return {
            "task_name_full": task_name,
            "task_stem": task_name,
            "task_config": "",
            "expert_data_num": "",
            "task_config_ep": "",
        }
    core = task_name[4:]

    # Task names are sim-{task_stem}-{task_config}-{num_episodes}. Parse the
    # episode count independently so path templates work for both 50 and 100
    # demo policies without splitting default-balanced into default/balanced.
    parts = core.rsplit("-", 1)
    if len(parts) == 2 and parts[1].isdigit():
        config_core, expert_data_num = parts
        for suffix in ("-default-ee-balanced", "-default-balanced", "-default-ee", "-default"):
            if config_core.endswith(suffix):
                task_config = suffix[1:]
                return {
                    "task_name_full": task_name,
                    "task_stem": config_core[: -len(suffix)],
                    "task_config": task_config,
                    "expert_data_num": expert_data_num,
                    "task_config_ep": f"{task_config}-{expert_data_num}",
                }

    # Backward compatibility for legacy hard-coded 50-demo task names.
    for suffix in ("-default-ee-balanced-50", "-default-balanced-50", "-default-ee-50", "-default-50"):
        if core.endswith(suffix):
            expert_data_num = suffix.rsplit("-", 1)[-1]
            task_config_ep = suffix[1:]
            task_config = task_config_ep[: -(len(expert_data_num) + 1)]
            return {
                "task_name_full": task_name,
                "task_stem": core[: -len(suffix)],
                "task_config": task_config,
                "expert_data_num": expert_data_num,
                "task_config_ep": task_config_ep,
            }
    parts = core.rsplit("-", 2)
    if len(parts) < 3:
        return {
            "task_name_full": task_name,
            "task_stem": core,
            "task_config": "",
            "expert_data_num": "",
            "task_config_ep": "",
        }
    task_stem, task_config, expert_data_num = parts
    return {
        "task_name_full": task_name,
        "task_stem": task_stem,
        "task_config": task_config,
        "expert_data_num": expert_data_num,
        "task_config_ep": f"{task_config}-{expert_data_num}",
    }


def _resolve_path_template(path_value, task_name):
    if not path_value:
        return path_value
    return path_value.format(**_task_template_vars(task_name))


def _resolve_vitacdreamer_task_id(args, task_name):
    task_order = args.get("vitacdreamer_task_order", None)
    if task_order is None:
        return None
    task_stem = _task_template_vars(task_name)["task_stem"]
    if task_stem not in task_order:
        raise ValueError(f"Task {task_stem!r} is not in vitacdreamer_task_order={task_order}")
    return task_order.index(task_stem)


def main(args):
    setup_distributed(args)
    set_seed(1 + int(args.get("rank", 0)))
    # command line parameters
    is_eval = args["eval"]
    ckpt_dir = args["ckpt_dir"]
    policy_class = args["policy_class"]
    onscreen_render = args["onscreen_render"]
    task_name = args["task_name"]
    batch_size_train = args["batch_size"]
    batch_size_val = args["batch_size"]

    # get task parameters
    is_sim = task_name[:4] == "sim-"
    if is_sim:
        # TacArena: load from JSON file generated by process_data.py
        SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
        task_config = SIM_TASK_CONFIGS[task_name]
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
    
    dataset_dir = task_config["dataset_dir"]
    num_episodes = task_config["num_episodes"]
    episode_len = task_config["episode_len"]
    camera_names = args["camera_names"]

    # fixed parameters
    if policy_class == "CNNMLP":
        policy_config = {
            "lr": args["lr"],
            "lr_backbone": args["lr_backbone"],
            "backbone": args["backbone"],
            "num_queries": 1,
            "camera_names": camera_names,
        }
    elif policy_class != "ACT":
        raise NotImplementedError

    state_dim = args["state_dim"]
    tactile_names = args["tactile_names"]
    chunk_size = args["chunk_size"]
    config = {
        "num_epochs": 6000,
        "ckpt_dir": ckpt_dir,
        "episode_len": episode_len,
        "state_dim": state_dim,
        "lr": args["lr"],
        "policy_class": policy_class,
        "onscreen_render": onscreen_render,
        "policy_config": args,
        "task_name": task_name,
        "seed": args["seed"],
        "temporal_agg": args["temporal_agg"],
        "camera_names": camera_names,
        "real_robot": not is_sim,
        "save_freq": args['save_freq'],
        "num_steps": args['num_steps'],
    }

    if is_eval:
        print("=" * 60)
        print("TacArena ACT Policy Evaluation")
        print("=" * 60)
        print("Please use the unified evaluation script:")
        print("  python scripts/eval_policy.py policy/ACT/deploy_policy_{task_name}.yml")
        print("")
        print("Note: TacArena uses IsaacLab simulation environment for evaluation.")
        print("      The eval_bc() function is for RoboTwin's MuJoCo environment.")
        print("=" * 60)
        exit()

    train_dataloader, val_dataloader, stats, _, train_sampler, _ = load_data(
        dataset_dir, num_episodes, camera_names, tactile_names, batch_size_train, batch_size_val, chunk_size,
        num_workers=args.get("num_workers", 0),
        use_vitacdreamer_feature=args.get("use_vitacdreamer_feature", False),
        vitacdreamer_history_len=args.get("vitacdreamer_history_len", 5),
        vitacdreamer_feature_cache_dir=_resolve_path_template(
            args.get("vitacdreamer_feature_cache_dir", None),
            task_name,
        ),
        vitacdreamer_task_id=_resolve_vitacdreamer_task_id(args, task_name),
        distributed=args.get("distributed", False),
    )

    # save dataset stats
    if is_main_process() and not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f"dataset_stats.pkl")
    if is_main_process():
        with open(stats_path, "wb") as f:
            pickle.dump(stats, f)
    config["train_sampler"] = train_sampler
    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    if is_main_process():
        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_best.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}")
    cleanup_distributed()


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == "ACT":
        optimizer = policy.configure_optimizers()
    elif policy_class == "CNNMLP":
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def forward_pass(data, policy):
    if len(data) == 6:
        cam_data, tac_data, qpos_data, action_data, is_pad, vitac_data = data
    else:
        cam_data, tac_data, qpos_data, action_data, is_pad = data
        vitac_data = None
    device = next(policy.parameters()).device
    cam_data, tac_data, qpos_data, action_data, is_pad = (
        cam_data.to(device),
        tac_data.to(device),
        qpos_data.to(device),
        action_data.to(device),
        is_pad.to(device),
    )
    vitac_inputs = None
    vitac_feature = None
    if vitac_data is not None:
        if isinstance(vitac_data, dict):
            vitac_inputs = {
                key: value.to(device)
                for key, value in vitac_data.items()
            }
        else:
            vitac_feature = vitac_data.to(device)
    return policy(
        qpos_data,
        cam_data,
        tac_data,
        action_data,
        is_pad,
        vitac_inputs=vitac_inputs,
        vitac_feature=vitac_feature,
    )


def train_bc(train_dataloader, val_dataloader, config):
    ckpt_dir = config["ckpt_dir"]
    seed = config["seed"]
    policy_class = config["policy_class"]
    policy_config = config["policy_config"]

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    local_rank = int(config.get("local_rank", 0))
    if config.get("distributed", False):
        if getattr(policy, "finetune_vitacdreamer_encoder", False):
            policy = DDP(
                policy,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=True,
            )
        else:
            policy.model = DDP(
                policy.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
    optimizer = make_optimizer(policy_class, unwrap_model(policy))

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None

    step_count = 0
    num_steps = config['num_steps']
    epoch = 0
    
    pbar = tqdm(range(num_steps), total=num_steps, leave=False, disable=not is_main_process())
    while step_count < num_steps:
        policy.train()
        optimizer.zero_grad()
        if isinstance(train_dataloader.sampler, DistributedSampler):
            train_dataloader.sampler.set_epoch(epoch)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict["loss"]
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if is_main_process():
                train_history.append(detach_dict(forward_dict))

            if is_main_process():
                pbar.set_postfix({'epoch': epoch, 'loss': loss.item()})
                pbar.update(1)

            step_count += 1
            if is_main_process() and step_count % config['save_freq'] == 0:
                ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{epoch + 1}_seed_{seed}.ckpt")
                torch.save(policy_state_dict_for_save(policy), ckpt_path)
                plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

            if step_count >= num_steps:
                break

        stop_after_train = step_count >= num_steps
        if is_dist_enabled():
            stop_tensor = torch.tensor(float(stop_after_train), dtype=torch.float64)
            dist.all_reduce(stop_tensor, op=dist.ReduceOp.MAX, group=_METRIC_PROCESS_GROUP)
            stop_after_train = bool(stop_tensor.item())

        if stop_after_train:
            break

        if is_main_process():
            epoch_train_start = epoch * len(train_dataloader)
            epoch_train_dicts = train_history[epoch_train_start:]
            epoch_summary = compute_dict_mean(epoch_train_dicts) if epoch_train_dicts else {}
            train_summary_string = ""
            for k, v in epoch_summary.items():
                train_summary_string += f"{k}: {v.item():.3f} "
        else:
            epoch_summary = {}
            train_summary_string = ""

        with torch.inference_mode():
            policy.eval()
            epoch_loss_sums = None
            epoch_count = 0
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                batch_size = int(data[2].shape[0])
                batch_loss_sums = {
                    key: value.detach().to(next(policy.parameters()).device, dtype=torch.float64) * batch_size
                    for key, value in forward_dict.items()
                }
                if epoch_loss_sums is None:
                    epoch_loss_sums = batch_loss_sums
                else:
                    for key in epoch_loss_sums:
                        epoch_loss_sums[key] += batch_loss_sums[key]
                epoch_count += batch_size

            epoch_summary = reduce_weighted_loss_dict(epoch_loss_sums, epoch_count, next(policy.parameters()).device)
            if is_main_process():
                validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary["loss"]
            if is_main_process() and epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy_state_dict_for_save(policy)))

        eval_summary_string = ""
        for k, v in epoch_summary.items():
            eval_summary_string += f"{k}: {v.item():.3f} "

        epoch += 1

    if is_main_process():
        ckpt_path = os.path.join(ckpt_dir, f"policy_last.ckpt")
        torch.save(policy_state_dict_for_save(policy), ckpt_path)

        if best_ckpt_info is None:
            best_ckpt_info = (epoch, float("nan"), deepcopy(policy_state_dict_for_save(policy)))

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        ckpt_path = os.path.join(ckpt_dir, f"policy_epoch_{best_epoch}_seed_{seed}.ckpt")
        torch.save(best_state_dict, ckpt_path)
        print(f"Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")

        # save training curves
        plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f"train_val_{key}_seed_{seed}.png")
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(
            np.linspace(0, num_epochs - 1, len(train_history)),
            train_values,
            label="train",
        )
        plt.plot(
            np.linspace(0, num_epochs - 1, len(validation_history)),
            val_values,
            label="validation",
        )
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f"Saved plots to {ckpt_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--onscreen_render", action="store_true")
    parser.add_argument("--ckpt_dir", action="store", type=str, help="ckpt_dir", required=True)
    parser.add_argument("--task_name", action="store", type=str, help="task_name", required=True)
    parser.add_argument("--config_path", action="store", type=str, help="config_path", required=True)
    parser.add_argument("--seed", action="store", type=int, help="seed", required=True)

    args = parser.parse_args()
    with open(args.config_path, 'r') as f:
        config_args = yaml.load(f, Loader=yaml.FullLoader)
    config_args.update(vars(args))
    main(config_args)
