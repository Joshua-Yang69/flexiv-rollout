import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from .._base_policy import BasePolicy

import os
import cv2
import yaml
import numpy as np
import torch
from .act_policy import ACT
# from act_policy import ACT
from torchvision import transforms


DEFAULT_QPOS_DELTA_LIMIT = np.array(
    [0.016, 0.006, 0.014, 0.011, 0.0045, 0.026, 0.003, 0.0015],
    dtype=np.float32,
)


def _load_qpos_delta_limit():
    raw = os.environ.get("DEPLOY_QPOS_DELTA_LIMIT")
    if raw is None:
        return DEFAULT_QPOS_DELTA_LIMIT.copy()
    values = [float(item) for item in raw.split(",") if item.strip()]
    if len(values) != 8:
        raise ValueError("DEPLOY_QPOS_DELTA_LIMIT must contain 8 comma-separated values.")
    return np.asarray(values, dtype=np.float32)


def _get_tactile_marker(observation, side: str):
    tactile_obs = observation["tactile"]
    for sensor_name in (f"{side}_tactile", f"{side}_gsmini"):
        sensor = tactile_obs.get(sensor_name)
        if sensor is not None and "rgb_marker" in sensor:
            return sensor["rgb_marker"]
    raise KeyError(f"Could not find tactile rgb_marker for side '{side}'")


def _to_chw_float01(img: torch.Tensor) -> torch.Tensor:
    if not isinstance(img, torch.Tensor):
        img = torch.as_tensor(img)
    if img.dim() != 3:
        raise ValueError(f"Expected HWC/CHW image tensor, got shape {tuple(img.shape)}")
    if img.shape[0] in (1, 3):
        img = img.float()
    else:
        img = img.permute(2, 0, 1).float()
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _resolve_repo_checkpoint_path(path_value: str | None) -> str | None:
    if not path_value:
        return path_value
    path = Path(path_value)
    if path.exists():
        return str(path)
    repo_root = Path(__file__).resolve().parents[3]
    parts = path.parts
    if "checkpoints" in parts:
        checkpoint_idx = parts.index("checkpoints")
        candidate = repo_root.joinpath(*parts[checkpoint_idx:])
        if candidate.exists():
            return str(candidate)
    candidate = repo_root / path_value
    if candidate.exists():
        return str(candidate)
    return path_value

class Policy(BasePolicy):
# class Policy:
    def __init__(self, args):
        """Initialize ACT policy for TacArena deployment"""
        # Construct checkpoint directory path
        self.train_config_name = args.get('train_config_name', os.environ.get('TRAIN_CONFIG', 'train_config'))
        self.ep_num = os.environ.get('EP_NUM', str(args.get('expert_data_num', '50')))
        # Official UniVTAC ACT eval sends the predicted qpos target directly to
        # task.take_action(..., action_type="qpos"). Keep delta limiting off by
        # default so deployment matches both ACT training targets and upstream
        # UniVTAC evaluation. Set DEPLOY_LIMIT_QPOS_DELTA=1 only for explicit
        # debugging/safety experiments.
        self.limit_qpos_delta = os.environ.get("DEPLOY_LIMIT_QPOS_DELTA", "0") != "0"
        self.qpos_delta_limit = _load_qpos_delta_limit()
        self.gripper_max_qpos = float(os.environ.get("DEPLOY_GRIPPER_MAX_QPOS", "0.039"))
        ckpt_setting = args.get('ckpt_setting')
        if ckpt_setting is None:
            ckpt_setting = f"{args['task_config']}-{self.ep_num}"
        elif isinstance(ckpt_setting, int):
            ckpt_setting = str(ckpt_setting)
        ckpt_dir = Path(__file__).parent / "act_ckpt" / f"act-{args['task_name']}" / ckpt_setting / self.train_config_name
 
        self.task_name = args['task_name']
        with open(Path(__file__).parent.parent / 'task_settings.json', 'r') as f:
            task_settings = json.load(f)
        assert self.task_name in task_settings, f"Task '{self.task_name}' not found in task_settings.json"
        self.camera_type = task_settings[self.task_name].get('camera_type', 'head')
        print(f"Using camera type '{self.camera_type}' for task '{self.task_name}'")

        with open(Path(__file__).parent / f'{self.train_config_name}.yml', 'r') as f:
            train_config = yaml.load(f, Loader=yaml.FullLoader)
        self.control_mode = train_config.get("control_mode", "qpos")
        self.state_repr = train_config.get("state_repr", "qpos")
        self.use_vitacdreamer_feature = bool(train_config.get("use_vitacdreamer_feature", False))
        self.vitacdreamer_task_id = None
        task_order = train_config.get("vitacdreamer_task_order")
        if self.use_vitacdreamer_feature:
            if "vitacdreamer_feature_cache_dir" in train_config:
                # Cached features are only valid for offline ACT training.
                # Online TacArena eval must recompute the frozen encoder feature
                # from the current observation stream using the same preprocessing.
                train_config.pop("vitacdreamer_feature_cache_dir", None)
            train_config["vitacdreamer_checkpoint"] = _resolve_repo_checkpoint_path(
                train_config.get("vitacdreamer_checkpoint")
            )
            if task_order is None or self.task_name not in task_order:
                raise ValueError(
                    f"Task {self.task_name!r} is not in vitacdreamer_task_order={task_order}"
                )
            self.vitacdreamer_task_id = int(task_order.index(self.task_name))
        
        train_config.update({
            'task_name': f"sim-{args['task_name']}-{args['task_config']}-{self.ep_num}",
            'task_config': args['task_config'],
            'ckpt_dir': str(ckpt_dir),
            'policy_checkpoint': str((Path(__file__).parent / args['policy_checkpoint']).resolve())
                if args.get('policy_checkpoint') and not Path(args['policy_checkpoint']).is_absolute()
                else args.get('policy_checkpoint'),
            'stats_path': str((Path(__file__).parent / args['stats_path']).resolve())
                if args.get('stats_path') and not Path(args['stats_path']).is_absolute()
                else args.get('stats_path'),
            "seed": args.get('seed', 0),
            "num_epochs": 1
        })
        
        # Initialize ACT model (RoboTwin_Config=None for TacArena)
        self.model = ACT(train_config)
        print(f"ACT policy loaded from {ckpt_dir}")

    def _encode_vitacdreamer_feature(self, observation):
        if not self.use_vitacdreamer_feature:
            return None
        extractor = getattr(getattr(self.model, "policy", None), "feature_extractor", None)
        if extractor is None:
            raise RuntimeError(
                "Online eval requires a ViTacDreamer feature extractor. "
                "Check that vitacdreamer_feature_cache_dir is not used in deploy."
            )
        obs_for_encoder = dict(observation)
        obs_for_encoder["task_id"] = torch.tensor([self.vitacdreamer_task_id], dtype=torch.long)
        with torch.no_grad():
            return extractor(obs_for_encoder).detach().cpu()

    def _limit_action(self, task, obs_qpos, raw_action):
        obs_qpos = np.asarray(obs_qpos, dtype=np.float32)
        raw_action = np.asarray(raw_action, dtype=np.float32).copy()
        if not self.limit_qpos_delta:
            return raw_action, raw_action - obs_qpos, raw_action - obs_qpos, False

        raw_delta = raw_action - obs_qpos
        limited_delta = np.clip(raw_delta, -self.qpos_delta_limit, self.qpos_delta_limit)
        action = obs_qpos + limited_delta
        gripper_max = float(getattr(task._robot_manager, "gripper_max_qpos", self.gripper_max_qpos))
        action[7] = np.clip(action[7], 0.0, gripper_max)
        clipped = bool(np.any(np.abs(raw_delta - limited_delta) > 1e-8))
        return action, raw_delta, limited_delta, clipped

    def encode_obs(self, task, observation):
        """
        Encode TacArena observation to ACT input format
        
        Input (TacArena):
            observation = {
                "observation": {"head": {"rgb": torch.Tensor([H, W, 3])}},  # HWC, 0-255
                "joint_action": torch.Tensor([9])  # [arm(7), gripper(1), extra(1)]
            }
            camera: 480x270
            tactile: 320x240
        
        Output (ACT):
            obs = {
                "qpos": torch.Tensor([8])  # [arm(7), gripper(1)]
                "cam_high": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_left": torch.Tensor([3, 256, 256]),  # CHW, 0-1
                "tac_right": torch.Tensor([3, 256, 256]),  # CHW, 0-1
            }
        """
        # Debug: observation structure validated
        # observation['embodiment']['joint'] contains joint state
        def camera_transform(img: torch.Tensor):
            img = _to_chw_float01(img)
            img = transforms.Resize((256, 256))(img)
            img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
            return img
        
        def tactile_transform(img: torch.Tensor):
            img = _to_chw_float01(img)
            img = transforms.Resize((256, 256))(img)
            return img

        vitac_feature = self._encode_vitacdreamer_feature(observation)

        if self.camera_type == 'all':
            cam_high = camera_transform(observation["observation"]["head"]["rgb"])
            cam_wrist = camera_transform(observation["observation"]["wrist"]["rgb"])
        else:
            cam_high = camera_transform(observation["observation"][self.camera_type]["rgb"])

        left_tac = tactile_transform(_get_tactile_marker(observation, "left"))
        right_tac = tactile_transform(_get_tactile_marker(observation, "right"))
        
        if self.state_repr == "ee":
            if "ee" in observation["embodiment"]:
                ee_pose = observation["embodiment"]["ee"][:7].detach().cpu().numpy()
            else:
                ee_pose_obj = task._robot_manager.get_ee_pose()
                ee_pose = np.asarray([float(ee_pose_obj[i]) for i in range(7)], dtype=np.float32)
            gripper_qpos = float(observation["embodiment"]["joint"][7].detach().cpu().item())
            qpos = np.concatenate([ee_pose.astype(np.float32), np.array([gripper_qpos], dtype=np.float32)], axis=0)
        else:
            # Extract joint positions (8D: 7 arm + 1 gripper)
            qpos = observation["embodiment"]["joint"][:8].cpu().numpy()

        ret = {
            "cam_high": cam_high,
            "tac_left": left_tac,
            "tac_right": right_tac,
            "qpos": qpos
        }
        if vitac_feature is not None:
            ret["vitac_feature"] = vitac_feature
        if self.camera_type == 'all':
            ret["cam_wrist"] = cam_wrist
        return ret

    def eval(self, task, observation):
        """
        Evaluate ACT policy on TacArena task
        
        Args:
            task: TacArena BaseTask instance
            observation: Current observation from environment
        """
        
        # Get action from ACT model (returns (1, 8) numpy array)
        obs = self.encode_obs(task, observation)
        if self.model.t % 10 == 0:
            self.save(task.get_frame_shot(observation), task.take_action_cnt)
        raw_action = self.model.get_action(obs).reshape(-1)
        if self.control_mode == "ee":
            action = raw_action.astype(np.float32)
            quat_norm = np.linalg.norm(action[3:7])
            if quat_norm > 1e-6:
                action[3:7] = action[3:7] / quat_norm
            gripper_max = float(getattr(task._robot_manager, "gripper_max_qpos", self.gripper_max_qpos))
            action[7] = np.clip(action[7], 0.0, gripper_max)
            action_type = "ee"
        else:
            action, raw_action_delta, action_delta, action_clipped = self._limit_action(
                task, obs["qpos"], raw_action
            )
            action_type = "qpos"
        action = torch.from_numpy(action).to(task.device).float()
        exec_succ, eval_succ = task.take_action(action, action_type=action_type)

    def reset(self):
        """Reset ACT model state (temporal aggregation and timestep counter)"""
        if hasattr(self.model, 'reset'):
            self.model.reset()

    def save(self, img, t):
        from PIL import Image
        from PIL import ImageDraw, ImageFont
        
        obs = Image.fromarray(img.cpu().numpy())

        draw = ImageDraw.Draw(obs)
        font = ImageFont.load_default()

        draw.text((obs.width-100, obs.height-60), f'{t:03d}', fill=(255, 0, 0), font=font)
        obs.save(f'ACT_{self.task_name}_{self.train_config_name}.png')
