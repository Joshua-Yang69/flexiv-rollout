from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rollout.models.base import BasePolicy
from rollout.types import Observation, PolicyAction
from utils.rolling_buffer import RollingBuffer
from utils.timing import now_ms
from utils.transforms import xyzquat_xyzw_to_mat


class VTMusePolicyAdapter(BasePolicy):
    """ViTacACT policy adapter: ViTacDreamer feature extractor + ACT action policy.

    This follows the ViTacACT deployment pattern from code_reference/ViTacACT:
    - A trained ViTacDreamerFeatureExtractor runs online to obtain latent features
      from the current tactile + a strided visual/tactile history window.
    - The features are passed as ``vitac_feature`` to the ACT model (which was
      trained with those cached features from the same encoder).
    - History is sourced from the ``RollingBuffer`` maintained by
      ``StatePerceiver``, sampled every ``sample_stride`` perception frames to
      match training (default stride=5 @ 50 Hz perception = 10 Hz samples).

    Inference entry points
    ----------------------
    ``infer_with_history(observation, history_buffer)``
        Called by ``PolicyModelServer`` when a ``RollingBuffer`` is available.
        Samples a strided window from the buffer and calls
        ``extract_features_from_history()`` — the same stateless batch API used
        during offline precompute.

    ``infer(observation)``
        Fallback when no history buffer is wired (e.g. standalone tests).
        Delegates to the internal stateful ``extract_features()`` which
        maintains its own rolling list via ``update_history()``.

    Config keys
    -----------
    checkpoint            : path to ViTacDreamer encoder checkpoint  (required)
    freeze_encoder        : bool, default True
    device                : "cuda:0" | "cpu"
    control_mode          : "joint" | "tcp"  (default "joint")
    camera_names          : list[str] — visual keys in observation.visual
    tactile_names         : list[str] — two tactile names for bilateral composition
    image_size            : int, default 224 (ViTacDreamer native resolution)
    sample_stride         : int, default 5  — perception frames between history slots

    model_args            : dict passed directly to ACT.__init__  (required for ACT)
      Required sub-keys: ckpt_dir (or policy_checkpoint), state_dim, chunk_size, etc.
      ``use_vitacdreamer_feature`` is forced to True.
      ``vitacdreamer_checkpoint`` is filled from ``checkpoint`` if omitted.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = config.get("device", "cuda:0")
        self.control_mode = config.get("control_mode", "joint")
        self.camera_names: list[str] = list(config.get("camera_names", ["cam_high"]))
        self.tactile_names: list[str] = list(
            config.get("tactile_names", ["tac_left", "tac_right"])
        )
        self.image_size = int(config.get("image_size", 224))
        # Stride between sampled history frames; 5 matches ViTacACT training
        self.sample_stride = int(config.get("sample_stride", 5))

        self.feature_extractor = None
        self.act_model = None

    # ── lifecycle ──────────────────────────────────────────────────────────────

    def load(self) -> None:
        from rollout.models.act.act_policy import ACT
        from rollout.models.vtmuse.policy_wrapper import ViTacDreamerFeatureExtractor

        checkpoint = self.config.get("checkpoint")
        if not checkpoint:
            raise RuntimeError(
                "vt_muse config requires 'checkpoint': path to ViTacDreamer encoder."
            )

        self.feature_extractor = ViTacDreamerFeatureExtractor(
            checkpoint_path=checkpoint,
            freeze_encoder=bool(self.config.get("freeze_encoder", True)),
            device=self.device,
        )

        model_args: dict[str, Any] = dict(self.config.get("model_args", {}))
        model_args.setdefault("device", self.device)
        model_args.setdefault("camera_names", self.camera_names)
        model_args.setdefault("tactile_names", self.tactile_names)
        model_args["use_vitacdreamer_feature"] = True
        model_args.setdefault("vitacdreamer_checkpoint", checkpoint)
        model_args.pop("vitacdreamer_feature_cache_dir", None)

        self.act_model = ACT(model_args)

    def reset(self) -> None:
        if self.feature_extractor is not None:
            self.feature_extractor.reset()
        if self.act_model is not None:
            self.act_model.reset()

    # ── primary inference path: uses external RollingBuffer ───────────────────

    def infer_with_history(
        self, observation: Observation, history_buffer: RollingBuffer
    ) -> PolicyAction:
        """Infer using a strided window sampled from the external history buffer.

        This is the production path.  It matches the offline ViTacDreamer
        precompute convention used during training:
          - ``history_len`` slots, spaced ``sample_stride`` perception frames apart
          - Current tactile is composed bilaterally and replaces the last slot
          - Visual history slots are provided but the most recent is masked by
            the encoder (Stage 2 training setup)

        The stateless ``extract_features_from_history()`` batch API is used so
        that the result is reproducible regardless of call frequency.
        """
        if self.feature_extractor is None or self.act_model is None:
            self.load()
        assert self.feature_extractor is not None
        assert self.act_model is not None

        import torch

        history_len = self.feature_extractor.history_len
        stride = self.sample_stride

        # ── 1. Sample strided history from the rolling buffer ─────────────────
        past_frames: list[Observation] = history_buffer.sample_strided(
            history_len=history_len, stride=stride
        )

        # ── 2. Build visual and tactile history tensors ───────────────────────
        # Episode-start zero-padding: missing slots receive zero tensors, matching
        # the precompute convention in precompute_vitacdreamer_features.py where
        # indices [t-20, t-15, …] that fall before t=0 are zero-filled, NOT
        # repeated from real observations.
        H = self.image_size
        zero_visual   = torch.zeros(3, H, H)
        zero_tactile  = torch.zeros(3, H, H)

        visual_hist_list: list[torch.Tensor] = []
        tactile_hist_list: list[torch.Tensor] = []
        prev_tactile_hist_list: list[torch.Tensor] = []

        n_real = len(past_frames)   # may be < history_len at episode start
        n_pad  = history_len - n_real

        # Zero-pad oldest slots first
        for _ in range(n_pad):
            visual_hist_list.append(zero_visual.clone())
            tactile_hist_list.append(zero_tactile.clone())
            prev_tactile_hist_list.append(zero_tactile.clone())

        # Fill in real frames (oldest → newest)
        for i, frame in enumerate(past_frames):
            v = self._get_visual_tensor(frame)      # (3, H, W)
            t = self._get_bilateral_tactile(frame)  # (3, H, W)
            visual_hist_list.append(v)
            tactile_hist_list.append(t)
            # prev_tactile for slot i = tactile of the previous slot in the list
            slot_idx = n_pad + i
            if slot_idx == 0:
                prev_tactile_hist_list.append(zero_tactile.clone())
            else:
                prev_tactile_hist_list.append(tactile_hist_list[-2].clone())

        # Stack: (1, T, 3, H, W)
        visual_hist   = torch.stack(visual_hist_list,       dim=0).unsqueeze(0)
        tactile_hist  = torch.stack(tactile_hist_list,      dim=0).unsqueeze(0)
        prev_tac_hist = torch.stack(prev_tactile_hist_list, dim=0).unsqueeze(0)

        # Action history: always zeros.
        # Training used no_action_conditioning=True (use_action_conditioning=False),
        # so the encoder ignores action tokens; zeros are correct for deployment.
        action_hist = torch.zeros(
            1, history_len, self.feature_extractor.action_dim, dtype=torch.float32
        )

        # Current tactile (most recent, for the final slot)
        current_tactile = self._get_bilateral_tactile(observation)  # (3, H, W)

        # ── 3. Feature extraction (stateless batch API) ───────────────────────
        # task_id is required: real encoder was trained with num_tasks=2
        # (insert_tube=0, wipe_board=1). Must be set in config.
        task_id = self._get_task_id()
        vitac_feature = self.feature_extractor.extract_features_from_history(
            current_tactile=current_tactile,
            visual_history=visual_hist,
            tactile_history=tactile_hist,
            action_history=action_hist,
            prev_tactile_history=prev_tac_hist,
            task_id=task_id,
        ).detach().cpu()  # (1, latent_dim) or (latent_dim,)

        # ── 4. ACT inference ──────────────────────────────────────────────────
        obs_dict = self._encode_observation(observation, current_tactile, vitac_feature)
        raw_action = np.asarray(
            self.act_model.get_action(obs_dict), dtype=np.float32
        ).reshape(-1)

        return self._decode_action(raw_action)

    # ── fallback inference path: uses internal stateful history ───────────────

    def infer(self, observation: Observation) -> PolicyAction:
        """Fallback: use the feature extractor's internal rolling history.

        Called when no external RollingBuffer is provided (e.g. unit tests or
        single-shot evaluation).  The internal history is maintained via
        ``update_history()`` after each step.
        """
        if self.feature_extractor is None or self.act_model is None:
            self.load()
        assert self.feature_extractor is not None
        assert self.act_model is not None

        import torch

        current_tactile = self._get_bilateral_tactile(observation)
        vitac_feature = self.feature_extractor.extract_features(
            current_tactile=current_tactile,
        ).detach().cpu()

        obs_dict = self._encode_observation(observation, current_tactile, vitac_feature)
        raw_action = np.asarray(
            self.act_model.get_action(obs_dict), dtype=np.float32
        ).reshape(-1)

        # Update internal history for the next step.
        # Training used no_action_conditioning=True, so the encoder ignores
        # action tokens. Pass zeros to match precompute convention.
        current_visual = self._get_visual_tensor(observation)
        self.feature_extractor.update_history(
            current_visual=current_visual,
            current_tactile=current_tactile,
            action_for_history=None,  # encoder does not use action conditioning
        )

        return self._decode_action(raw_action)

    # ── observation helpers ───────────────────────────────────────────────────

    def _to_chw_float(self, img) -> "torch.Tensor":
        import torch
        t = torch.as_tensor(img).float()
        if t.dim() == 3 and t.shape[-1] in (1, 3):
            t = t.permute(2, 0, 1)
        if t.max() > 1.0:
            t = t / 255.0
        return t

    def _resize(self, img: "torch.Tensor", size: tuple[int, int]) -> "torch.Tensor":
        import torch.nn.functional as functional
        if img.shape[-2:] == size:
            return img
        return functional.interpolate(
            img.unsqueeze(0), size=size, mode="bilinear", align_corners=False
        ).squeeze(0)

    def _get_bilateral_tactile(self, observation: Observation) -> "torch.Tensor":
        """Compose left+right tactile into one bilateral image (ViTacACT convention)."""
        import torch

        _aliases = {
            "tac_left":  ("xense_left",  "left",  "left_tactile"),
            "tac_right": ("xense_right", "right", "right_tactile"),
        }

        sz = (self.image_size, self.image_size)

        def _resolve(name: str) -> np.ndarray:
            for candidate in (name, *_aliases.get(name, ())):
                frame = observation.tactile_frame(candidate)
                if frame is not None and frame.rectify is not None:
                    return frame.rectify
            raise ValueError(
                f"vt_muse: cannot resolve tactile frame '{name}' in observation."
            )

        if len(self.tactile_names) >= 2:
            left  = self._resize(self._to_chw_float(_resolve(self.tactile_names[0])), sz)
            right = self._resize(self._to_chw_float(_resolve(self.tactile_names[1])), sz)
            return self._resize(torch.cat([left, right], dim=2), sz)

        if len(self.tactile_names) == 1:
            return self._resize(self._to_chw_float(_resolve(self.tactile_names[0])), sz)

        if isinstance(observation.tactile, dict):
            for v in observation.tactile.values():
                if v.rectify is not None:
                    return self._resize(self._to_chw_float(v.rectify), sz)
        raise ValueError("vt_muse: no tactile frame found in observation.")

    def _get_visual_tensor(self, observation: Observation) -> "torch.Tensor":
        import torch
        sz = (self.image_size, self.image_size)
        if observation.visual is not None and observation.visual.color is not None:
            return self._resize(self._to_chw_float(observation.visual.color), sz)
        return torch.zeros(3, self.image_size, self.image_size)

    def _get_task_id(self):
        """Return task_id tensor if configured, else None."""
        task_id_val = self.config.get("task_id")
        if task_id_val is None:
            return None
        import torch
        return torch.tensor([int(task_id_val)], dtype=torch.long)

    def _encode_observation(
        self,
        observation: Observation,
        current_tactile,
        vitac_feature,
    ) -> dict[str, Any]:
        """Build the obs dict expected by ACT.get_action()."""
        import torch
        from torchvision import transforms

        def camera_transform(img):
            t = self._resize(self._to_chw_float(img), (256, 256))
            return transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )(t)

        def tactile_transform(img):
            return self._resize(self._to_chw_float(img), (256, 256))

        qpos = observation.robot.qpos8.astype(np.float32)

        _aliases = {
            "tac_left":  ("xense_left",  "left",  "left_tactile"),
            "tac_right": ("xense_right", "right", "right_tactile"),
        }

        obs: dict[str, Any] = {"qpos": qpos, "vitac_feature": vitac_feature}

        # camera inputs
        if observation.visual is not None and observation.visual.color is not None:
            visual_tensor = camera_transform(observation.visual.color)
        else:
            visual_tensor = torch.zeros(3, 256, 256)
        for cam in self.camera_names:
            obs[cam] = visual_tensor

        # per-side tactile inputs
        for name in self.tactile_names:
            found = None
            for candidate in (name, *_aliases.get(name, ())):
                frame = observation.tactile_frame(candidate)
                if frame is not None and frame.rectify is not None:
                    found = tactile_transform(frame.rectify)
                    break
            obs[name] = found if found is not None else torch.zeros(3, 256, 256)

        # bilateral composite
        obs["tac_bilateral"] = tactile_transform(current_tactile)

        return obs

    # ── action decoding ───────────────────────────────────────────────────────

    def _decode_action(self, raw: np.ndarray) -> PolicyAction:
        if raw.shape[0] < 8:
            raise ValueError(
                f"vt_muse action must have at least 8 values, got {raw.shape[0]}"
            )

        gripper_width = float(raw[7])
        metadata = {"policy": "vt_muse", "raw_action": raw.copy()}

        if self.control_mode in {"ee", "tcp", "cartesian"}:
            xyz  = raw[:3].astype(np.float64)
            quat = raw[3:7].astype(np.float64)
            quat_order = self.config.get("ee_quat_order", "xyzw")
            if quat_order == "wxyz":
                quat = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)
            quat_norm = float(np.linalg.norm(quat))
            if quat_norm > 1e-6:
                quat = quat / quat_norm
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
            target_joints=raw[:7].astype(np.float64),
            target_gripper_width=gripper_width,
            metadata=metadata,
        )


class VTMuseHoldPolicy(BasePolicy):
    def infer(self, observation: Observation) -> PolicyAction:
        return PolicyAction(
            mode="hold",
            timestamp_ms=now_ms(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"policy": "vt_muse_hold"},
        )
