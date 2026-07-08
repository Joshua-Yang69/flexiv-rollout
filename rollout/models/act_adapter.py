from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rollout.models.base import BasePolicy
from rollout.types import Observation, PolicyAction
from utils.timing import now_ms
from utils.transforms import xyzquat_xyzw_to_mat


class ACTPolicyAdapter(BasePolicy):
    """Adapter around references/models/ACT for online rollout.

    The adapter keeps model loading persistent and converts this project's
    Observation contract into ACT's expected dict: qpos plus camera/tactile
    tensors. It deliberately stays thin so ACT-specific training choices remain
    in the reference model config.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.model_args = dict(config.get("model_args", {}))
        self.control_mode = config.get("control_mode", self.model_args.get("control_mode", "joint"))
        self.camera_names = list(config.get("camera_names", self.model_args.get("camera_names", ["cam_high"])))
        self.tactile_names = list(config.get("tactile_names", self.model_args.get("tactile_names", ["tac_left", "tac_right"])))
        self.image_size = int(config.get("image_size", 256))
        self.normalize_visual = bool(config.get("normalize_visual", True))
        self.device = config.get("device", self.model_args.get("device", "cuda:0"))
        self.model = None

    def load(self) -> None:
        try:
            from references.models.ACT.act_policy import ACT
        except ImportError as exc:
            raise RuntimeError("Could not import references.models.ACT.act_policy.ACT.") from exc

        self.model_args.setdefault("device", self.device)
        self.model_args.setdefault("camera_names", self.camera_names)
        self.model_args.setdefault("tactile_names", self.tactile_names)
        self.model = ACT(self.model_args)

    def reset(self) -> None:
        if self.model is not None and hasattr(self.model, "reset"):
            self.model.reset()

    def infer(self, observation: Observation) -> PolicyAction:
        if self.model is None:
            self.load()
        obs_dict = self.encode_observation(observation)
        raw_action = np.asarray(self.model.get_action(obs_dict), dtype=np.float32).reshape(-1)  # type: ignore[union-attr]
        return self.decode_action(raw_action)

    def encode_observation(self, observation: Observation) -> dict[str, Any]:
        try:
            import torch
            import torch.nn.functional as functional
        except ImportError as exc:
            raise RuntimeError("PyTorch is required for ACT policy inference.") from exc

        encoded: dict[str, Any] = {"qpos": observation.robot.qpos8.astype(np.float32)}

        if self.camera_names:
            if observation.visual is None or observation.visual.color is None:
                raise ValueError("ACT camera input requested but observation has no color frame.")
            visual = self._prepare_image_tensor(
                torch,
                functional,
                observation.visual.color,
                normalize=self.normalize_visual,
            )
            for name in self.camera_names:
                encoded[name] = visual

        if self.tactile_names:
            for name in self.tactile_names:
                frame = self._resolve_tactile_frame(observation, name)
                if frame is None or frame.rectify is None:
                    raise ValueError(f"ACT tactile input '{name}' requested but observation has no matching frame.")
                encoded[name] = self._prepare_image_tensor(
                    torch,
                    functional,
                    frame.rectify,
                    normalize=False,
                )

        return encoded

    def _resolve_tactile_frame(self, observation: Observation, act_name: str) -> Any:
        aliases = {
            "tac_left": ("xense_left", "left", "left_tactile"),
            "tac_right": ("xense_right", "right", "right_tactile"),
        }
        for candidate in (act_name, *aliases.get(act_name, ())):
            frame = observation.tactile_frame(candidate)
            if frame is not None:
                return frame
        return None

    def _prepare_image_tensor(self, torch: Any, functional: Any, image: np.ndarray, normalize: bool) -> Any:
        tensor = torch.as_tensor(image)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1).repeat(1, 1, 3)
        if tensor.ndim != 3:
            raise ValueError(f"Expected HWC/CHW image, got shape {tuple(tensor.shape)}")
        if tensor.shape[0] in (1, 3):
            tensor = tensor.float()
        else:
            tensor = tensor.permute(2, 0, 1).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        if tensor.shape[-2:] != (self.image_size, self.image_size):
            tensor = functional.interpolate(
                tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406], dtype=tensor.dtype).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], dtype=tensor.dtype).view(3, 1, 1)
            tensor = (tensor - mean) / std
        return tensor

    def decode_action(self, raw_action: np.ndarray) -> PolicyAction:
        if raw_action.shape[0] < 8:
            raise ValueError(f"ACT action must contain at least 8 values, got {raw_action.shape[0]}")

        gripper_width = float(raw_action[7])
        metadata = {"policy": "act", "raw_action": raw_action.copy()}
        if self.control_mode in {"ee", "tcp", "cartesian"}:
            xyz = raw_action[:3].astype(np.float64)
            quat = raw_action[3:7].astype(np.float64)
            quat_order = self.config.get("ee_quat_order", "xyzw")
            if quat_order == "wxyz":
                quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
            target_pose = xyzquat_xyzw_to_mat(xyz, quat)
            return PolicyAction(
                mode="tcp",
                timestamp_ms=now_ms(),
                target_tcp_pose=target_pose,
                target_gripper_width=gripper_width,
                wrench=np.zeros(6, dtype=np.float64),
                metadata=metadata,
            )

        return PolicyAction(
            mode="joint",
            timestamp_ms=now_ms(),
            target_joints=raw_action[:7].astype(np.float64),
            target_gripper_width=gripper_width,
            metadata=metadata,
        )


def resolve_reference_path(path_value: str | None) -> str | None:
    if not path_value:
        return None
    path = Path(path_value)
    if path.exists():
        return str(path)
    candidate = Path(__file__).resolve().parents[2] / path_value
    return str(candidate) if candidate.exists() else path_value
