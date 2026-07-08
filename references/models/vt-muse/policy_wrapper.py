"""
ViTacDreamer Policy Wrapper for UniVTAC

This module integrates the trained ViTacDreamer encoder as a feature extractor
for downstream policies (Diffusion Policy, ACT).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Dict, Optional
from collections import OrderedDict
import sys

sys.path.append(str(Path(__file__).parent.parent))
from vitacdreamer.model import ViTacDreamer
from vitacdreamer.model_v2 import ViTacDreamerV2
from vitacdreamer.tactile_flow import build_tactile_temporal_features


def _convert_legacy_vit_key(key: str) -> str:
    """Map legacy ViT layer keys to HuggingFace ViTModel keys."""
    marker = ".vit.layers."
    if marker not in key:
        return key

    prefix, rest = key.split(marker, 1)
    layer_idx, sep, suffix = rest.partition(".")
    if not sep:
        return key

    replacements = (
        ("attention.q_proj.", "attention.attention.query."),
        ("attention.k_proj.", "attention.attention.key."),
        ("attention.v_proj.", "attention.attention.value."),
        ("attention.o_proj.", "attention.output.dense."),
        ("layernorm_before.", "layernorm_before."),
        ("layernorm_after.", "layernorm_after."),
        ("mlp.fc1.", "intermediate.dense."),
        ("mlp.fc2.", "output.dense."),
    )
    for old, new in replacements:
        if suffix.startswith(old):
            return f"{prefix}.vit.encoder.layer.{layer_idx}.{new}{suffix[len(old):]}"
    return key


def _normalize_vit_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    normalized = OrderedDict()
    converted = 0
    for key, value in state_dict.items():
        new_key = _convert_legacy_vit_key(key)
        if new_key in normalized and new_key != key:
            raise RuntimeError(f"Duplicate key after ViT checkpoint key conversion: {new_key}")
        normalized[new_key] = value
        converted += int(new_key != key)
    if converted:
        print(f"Converted {converted} legacy ViT checkpoint keys for HuggingFace ViTModel compatibility.")
    return normalized


def _is_allowed_encoder_only_missing(key: str) -> bool:
    # Decoder heads are intentionally absent from encoder-only checkpoints.
    # HuggingFace ViT pooler is not used by ViTacDreamer, which consumes token
    # sequences / CLS states directly.
    return (
        key.startswith("decoder.")
        or key.startswith("tactile_flow_decoder.")
        or key.startswith("depth_delta_decoder.")
        or key.startswith("marker_flow_decoder.")
        or ".vit.pooler." in key
    )


class ViTacDreamerFeatureExtractor(nn.Module):
    """
    Wraps trained ViTacDreamer encoder for use as a feature extractor
    in downstream manipulation policies.
    """

    def __init__(
        self,
        checkpoint_path: str,
        freeze_encoder: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()
        self.device = device

        checkpoint = torch.load(checkpoint_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        state_dict = _normalize_vit_state_dict_keys(state_dict)
        self.model_type = self._infer_model_type(state_dict)
        decoder_prefixes = ("decoder.", "tactile_flow_decoder.", "depth_delta_decoder.", "marker_flow_decoder.")
        encoder_only = not any(key.startswith(decoder_prefixes) for key in state_dict)
        config = self._sanitize_model_config(checkpoint.get("config", {}), self.model_type, state_dict)
        model_config = dict(config)
        sample_stride = model_config.pop("sample_stride", 1)
        tactile_temporal_mode = model_config.pop("tactile_temporal_mode", "raw")
        tactile_flow_clip = model_config.pop("tactile_flow_clip", 0.25)
        tactile_delta_clip = model_config.pop("tactile_delta_clip", 0.25)
        use_action_conditioning = model_config.pop("use_action_conditioning", True)

        if self.model_type == "v2":
            self.model = ViTacDreamerV2(**model_config).to(device)
        else:
            model_config.pop("num_tail_frames", None)
            self.model = ViTacDreamer(**model_config).to(device)

        load_result = self.model.load_state_dict(state_dict, strict=not encoder_only)
        if encoder_only:
            missing = [
                key for key in load_result.missing_keys
                if not _is_allowed_encoder_only_missing(key)
            ]
            if missing:
                raise RuntimeError(
                    f"Unexpected missing non-decoder keys when loading encoder-only checkpoint: {missing[:10]}"
                )

        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()
        else:
            trainable_prefixes = ["visual_encoder.", "tactile_encoder.", "encoder."]
            if use_action_conditioning:
                trainable_prefixes.append("action_tokenizer.")
            for name, param in self.model.named_parameters():
                # Downstream policies only call encode_window()/encode(); decoder
                # heads from Stage 2 are not in the policy graph and must stay
                # frozen or DDP will wait for gradients that never appear.
                param.requires_grad = any(
                    name.startswith(prefix) for prefix in trainable_prefixes
                )
            self.model.train()

        self.history_len = config['history_len']
        self.latent_dim = config['latent_dim']
        self.visual_image_size = config['visual_image_size']
        self.tactile_image_size = config['tactile_image_size']
        self.action_dim = config['action_dim']
        self.sample_stride = sample_stride
        self.num_tail_frames = config.get("num_tail_frames", 1)
        self.num_tasks = config.get("num_tasks", 0)
        self.tactile_temporal_mode = tactile_temporal_mode
        self.tactile_flow_clip = tactile_flow_clip
        self.tactile_delta_clip = tactile_delta_clip
        self.use_action_conditioning = use_action_conditioning
        self.freeze_encoder = freeze_encoder
        # Keep enough past transitions to reconstruct the causal window used by
        # Stage 1/2 training: [t-(history_len-1)*stride, ..., t-stride, t].
        # The current visual slot is masked, so online inference can fill it
        # with a placeholder while still using the current tactile observation.
        self.history_capacity = max(1, (self.history_len - 1) * max(self.sample_stride, 1))

        # History buffers
        self.visual_history = []
        self.tactile_history = []
        self.prev_tactile_history = []
        self.action_history = []

    def _infer_model_type(self, state_dict: Dict[str, torch.Tensor]) -> str:
        if any(
            "visual_mask_token" in key
            or "memory_encoder" in key
            or "masked_visual_embed" in key
            for key in state_dict
        ):
            return "v2"
        return "v1"

    def _sanitize_model_config(self, raw_config: Dict, model_type: str, state_dict: Dict[str, torch.Tensor]) -> Dict:
        if model_type == "v2":
            max_delta_t = raw_config.get("max_delta_t")
            if max_delta_t is None and "encoder.delta_t_embed.weight" in state_dict:
                max_delta_t = int(state_dict["encoder.delta_t_embed.weight"].shape[0] - 1)
            if max_delta_t is None:
                max_delta_t = 64

            history_len = raw_config.get("history_len")
            if history_len is None and "encoder.temporal_slot_embed" in state_dict:
                history_len = int(state_dict["encoder.temporal_slot_embed"].shape[1])
            if history_len is None:
                history_len = 5

            num_tail_frames = raw_config.get("num_tail_frames")
            if num_tail_frames is None:
                for decoder_prefix in (
                    "decoder.",
                    "tactile_flow_decoder.",
                    "depth_delta_decoder.",
                    "marker_flow_decoder.",
                ):
                    frame_embed_key = f"{decoder_prefix}frame_embed"
                    if frame_embed_key in state_dict:
                        num_tail_frames = int(state_dict[frame_embed_key].shape[1])
                        break
            if num_tail_frames is None:
                num_tail_frames = 2

            num_tasks = raw_config.get("num_tasks")
            if num_tasks is None and "encoder.task_embed.weight" in state_dict:
                num_tasks = int(state_dict["encoder.task_embed.weight"].shape[0])
            if num_tasks is None:
                num_tasks = 0

            config = {
                "visual_image_size": raw_config.get("visual_image_size", 224),
                "tactile_image_size": raw_config.get("tactile_image_size", 224),
                "action_dim": raw_config.get("action_dim", 7),
                "latent_dim": raw_config.get("latent_dim", 256),
                "history_len": history_len,
                "pretrained_encoders": raw_config.get("pretrained_encoders", True),
                "max_delta_t": max_delta_t,
                "num_memory_layers": raw_config.get("num_memory_layers", 2),
                "num_latent_layers": raw_config.get("num_latent_layers", 2),
                "num_tail_frames": num_tail_frames,
                "num_tasks": num_tasks,
                "reconstruct_tactile_flow": any(
                    key.startswith("tactile_flow_decoder.") for key in state_dict
                ),
                "reconstruct_depth_delta": any(
                    key.startswith("depth_delta_decoder.") for key in state_dict
                ),
                "reconstruct_marker_flow": any(
                    key.startswith("marker_flow_decoder.") for key in state_dict
                ),
            }
            config["sample_stride"] = raw_config.get("sample_stride", 5)
            config["tactile_temporal_mode"] = raw_config.get("tactile_temporal_mode", "raw")
            config["tactile_flow_clip"] = raw_config.get("tactile_flow_clip", 0.25)
            config["tactile_delta_clip"] = raw_config.get("tactile_delta_clip", 0.25)
            config["use_action_conditioning"] = raw_config.get("use_action_conditioning", True)
            return config

        return {
            "visual_image_size": raw_config.get("visual_image_size", 224),
            "tactile_image_size": raw_config.get("tactile_image_size", 224),
            "action_dim": raw_config.get("action_dim", 7),
            "latent_dim": raw_config.get("latent_dim", 256),
            "history_len": raw_config.get("history_len", 5),
            "pretrained_encoders": raw_config.get("pretrained_encoders", True),
            "sample_stride": raw_config.get("sample_stride", 1),
            "num_tail_frames": raw_config.get("num_tail_frames", 1),
            "tactile_temporal_mode": raw_config.get("tactile_temporal_mode", "raw"),
            "tactile_flow_clip": raw_config.get("tactile_flow_clip", 0.25),
            "tactile_delta_clip": raw_config.get("tactile_delta_clip", 0.25),
            "use_action_conditioning": raw_config.get("use_action_conditioning", True),
        }

    def reset(self):
        """Reset history buffers."""
        self.visual_history = []
        self.tactile_history = []
        self.prev_tactile_history = []
        self.action_history = []

    def _prepare_image(
        self,
        image: torch.Tensor,
        image_size: int
    ) -> torch.Tensor:
        """Convert HWC/CHW images to BCHW float tensors resized for ViTacDreamer."""
        if image.dim() == 3:
            if image.shape[0] in (1, 3):
                image = image.unsqueeze(0)
            else:
                image = image.permute(2, 0, 1).unsqueeze(0)
        elif image.dim() == 4:
            if image.shape[-1] in (1, 3):
                image = image.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"Unsupported image rank: {image.shape}")

        image = image.float()
        if image.max() > 1.0:
            image = image / 255.0

        if image.shape[-2:] != (image_size, image_size):
            image = F.interpolate(
                image,
                size=(image_size, image_size),
                mode='bilinear',
                align_corners=False
            )
        return image

    def _prepare_action(self, action: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if action is None:
            return None
        if action.dim() == 1:
            action = action.unsqueeze(0)
        action = action.float()
        if action.shape[-1] > self.action_dim:
            action = action[..., :self.action_dim]
        elif action.shape[-1] < self.action_dim:
            pad = self.action_dim - action.shape[-1]
            action = F.pad(action, (0, pad))
        return action

    def _compose_bilateral_tactile(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError(f"Left/right tactile shapes must match, got {left.shape} vs {right.shape}")
        left = self._prepare_image(left, self.tactile_image_size)
        right = self._prepare_image(right, self.tactile_image_size)
        tactile = torch.cat([left, right], dim=3)
        if tactile.shape[-2:] != (self.tactile_image_size, self.tactile_image_size):
            tactile = F.interpolate(
                tactile,
                size=(self.tactile_image_size, self.tactile_image_size),
                mode="bilinear",
                align_corners=False,
            )
        return tactile

    def extract_features_from_history(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor,
        prev_tactile_history: Optional[torch.Tensor] = None,
        prev_current_tactile: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Stateless batch feature extraction for offline policy training.
        """
        current_tactile = self._prepare_image(current_tactile, self.tactile_image_size).to(self.device)

        if visual_history.dim() == 4:
            visual_history = visual_history.unsqueeze(0)
        if tactile_history.dim() == 4:
            tactile_history = tactile_history.unsqueeze(0)
        if prev_tactile_history is not None and prev_tactile_history.dim() == 4:
            prev_tactile_history = prev_tactile_history.unsqueeze(0)
        if prev_current_tactile is not None:
            prev_current_tactile = self._prepare_image(prev_current_tactile, self.tactile_image_size).to(self.device)
        if action_history.dim() == 2:
            action_history = action_history.unsqueeze(0)

        B, T = visual_history.shape[:2]
        visual_history = self._prepare_image(
            visual_history.reshape(B * T, *visual_history.shape[2:]),
            self.visual_image_size
        ).reshape(B, T, 3, self.visual_image_size, self.visual_image_size).to(self.device)
        tactile_history = self._prepare_image(
            tactile_history.reshape(B * T, *tactile_history.shape[2:]),
            self.tactile_image_size
        ).reshape(B, T, 3, self.tactile_image_size, self.tactile_image_size).to(self.device)
        if prev_tactile_history is not None:
            prev_tactile_history = self._prepare_image(
                prev_tactile_history.reshape(B * T, *prev_tactile_history.shape[2:]),
                self.tactile_image_size
            ).reshape(B, T, 3, self.tactile_image_size, self.tactile_image_size).to(self.device)

        action_history = action_history.float()
        if action_history.shape[-1] > self.action_dim:
            action_history = action_history[..., :self.action_dim]
        elif action_history.shape[-1] < self.action_dim:
            pad = self.action_dim - action_history.shape[-1]
            action_history = F.pad(action_history, (0, pad))
        action_history = action_history.to(self.device)

        grad_context = torch.no_grad() if self.freeze_encoder else torch.enable_grad()
        with grad_context:
            features = self._encode_features(
                current_tactile=current_tactile,
                visual_history=visual_history,
                tactile_history=tactile_history,
                action_history=action_history,
                prev_tactile_history=prev_tactile_history,
                prev_current_tactile=prev_current_tactile,
                task_id=task_id,
            )
        return features

    def _build_delta_steps(self, batch_size: int, seq_len: int, device: torch.device) -> torch.Tensor:
        stride = max(int(self.sample_stride), 1)
        delta_steps = torch.arange(seq_len - 1, -1, -1, device=device, dtype=torch.long) * stride
        return delta_steps.unsqueeze(0).expand(batch_size, -1)

    def _encode_features(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor,
        prev_tactile_history: Optional[torch.Tensor] = None,
        prev_current_tactile: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.model_type == "v2":
            batch_size, seq_len = visual_history.shape[:2]
            visual_mask = torch.zeros(
                batch_size,
                seq_len,
                dtype=torch.bool,
                device=visual_history.device,
            )
            visual_mask[:, -min(self.num_tail_frames, seq_len):] = True
            tactile_seq = tactile_history.clone()
            tactile_seq[:, -1] = current_tactile
            if prev_tactile_history is not None:
                if self.tactile_temporal_mode != "raw" and prev_current_tactile is None:
                    raise ValueError(
                        f"{self.tactile_temporal_mode} tactile mode requires prev_current_tactile "
                        "for the replaced current tactile frame"
                    )
                prev_tactile_seq = prev_tactile_history.clone()
                if prev_current_tactile is not None:
                    prev_tactile_seq[:, -1] = prev_current_tactile
            else:
                prev_tactile_seq = None
            tactile_seq = build_tactile_temporal_features(
                tactile_seq,
                prev_tactile_seq=prev_tactile_seq,
                mode=self.tactile_temporal_mode,
                flow_clip=self.tactile_flow_clip,
                delta_clip=self.tactile_delta_clip,
            )
            delta_steps = self._build_delta_steps(batch_size, seq_len, visual_history.device)
            if task_id is not None:
                task_id = task_id.to(visual_history.device)
            return self.model.encode_window(
                visual_seq=visual_history,
                tactile_seq=tactile_seq,
                action_seq=action_history if self.use_action_conditioning else None,
                visual_mask=visual_mask,
                delta_steps=delta_steps,
                task_id=task_id,
            )

        return self.model.encode(
            current_tactile=current_tactile,
            visual_history=visual_history,
            tactile_history=tactile_history,
            action_history=action_history,
            task_id=task_id,
        )

    def _build_history_tensors(self, batch_size: int):
        stride = max(int(self.sample_stride), 1)
        visual_hist = []
        tactile_hist = []
        prev_tactile_hist = []
        action_hist = []
        # Match the Stage 1/2 training and offline precompute convention.
        # For history_len=5 and stride=5, the temporal slots are:
        #   [t - 20, t - 15, t - 10, t - 5, t]
        # The final visual slot is masked by the encoder, so online inference
        # inserts a placeholder for current visual and replaces the final
        # tactile slot with the current tactile observation in _encode_features.
        for offset in range(self.history_len - 1, 0, -1):
            hist_idx = len(self.visual_history) - offset * stride
            if hist_idx >= 0:
                visual_hist.append(self.visual_history[hist_idx])
                tactile_hist.append(self.tactile_history[hist_idx])
                prev_tactile_hist.append(self.prev_tactile_history[hist_idx])
                action_hist.append(self.action_history[hist_idx])

        visual_hist.append(torch.zeros(batch_size, 3, self.visual_image_size, self.visual_image_size))
        tactile_hist.append(torch.zeros(batch_size, 3, self.tactile_image_size, self.tactile_image_size))
        prev_tactile_hist.append(torch.zeros(batch_size, 3, self.tactile_image_size, self.tactile_image_size))
        action_hist.append(torch.zeros(batch_size, self.action_dim))

        while len(visual_hist) < self.history_len:
            visual_hist.insert(
                0,
                torch.zeros(batch_size, 3, self.visual_image_size, self.visual_image_size),
            )
            tactile_hist.insert(
                0,
                torch.zeros(batch_size, 3, self.tactile_image_size, self.tactile_image_size),
            )
            prev_tactile_hist.insert(
                0,
                torch.zeros(batch_size, 3, self.tactile_image_size, self.tactile_image_size),
            )
            action_hist.insert(0, torch.zeros(batch_size, self.action_dim))

        visual_hist = torch.stack(visual_hist, dim=1).to(self.device)
        tactile_hist = torch.stack(tactile_hist, dim=1).to(self.device)
        prev_tactile_hist = torch.stack(prev_tactile_hist, dim=1).to(self.device)
        action_hist = torch.stack(action_hist, dim=1).to(self.device)
        return visual_hist, tactile_hist, prev_tactile_hist, action_hist

    def update_history(
        self,
        current_visual: torch.Tensor,
        current_tactile: torch.Tensor,
        action_for_history: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Append the current transition to history for the next inference step.
        """
        current_visual = self._prepare_image(current_visual, self.visual_image_size)
        current_tactile = self._prepare_image(current_tactile, self.tactile_image_size)
        action_for_history = self._prepare_action(action_for_history)
        if self.tactile_history:
            prev_tactile = self.tactile_history[-1]
        else:
            prev_tactile = current_tactile.clone()

        if action_for_history is None:
            action_for_history = torch.zeros(
                current_visual.shape[0], self.action_dim, dtype=current_visual.dtype
            )

        self.visual_history.append(current_visual.cpu())
        self.tactile_history.append(current_tactile.cpu())
        self.prev_tactile_history.append(prev_tactile.cpu())
        self.action_history.append(action_for_history.cpu())

        while len(self.visual_history) > self.history_capacity:
            self.visual_history.pop(0)
            self.tactile_history.pop(0)
            self.prev_tactile_history.pop(0)
            self.action_history.pop(0)

    def extract_features(
        self,
        current_tactile: torch.Tensor,
        current_visual: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract features causally from current tactile input and stored history.

        The current visual frame is intentionally excluded to match Stage 2
        inference, where only prior history plus current tactile are available.
        """
        del current_visual
        squeeze_output = current_tactile.dim() == 3
        current_tactile = self._prepare_image(current_tactile, self.tactile_image_size)
        batch_size = current_tactile.shape[0]
        visual_hist, tactile_hist, prev_tactile_hist, action_hist = self._build_history_tensors(batch_size)
        if self.tactile_history:
            prev_current_tactile = self.tactile_history[-1].to(self.device)
        else:
            prev_current_tactile = current_tactile.to(self.device)

        with torch.no_grad():
            features = self._encode_features(
                current_tactile=current_tactile.to(self.device),
                visual_history=visual_hist,
                tactile_history=tactile_hist,
                action_history=action_hist,
                prev_tactile_history=prev_tactile_hist,
                prev_current_tactile=prev_current_tactile,
                task_id=task_id,
            )

        if squeeze_output:
            features = features.squeeze(0)

        return features

    def forward(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass compatible with UniVTAC observation format.

        Args:
            obs: Dictionary with 'observation' (visual) and 'tactile' keys

        Returns:
            Extracted features
        """
        # Extract visual observation (assume first camera)
        if 'observation' in obs and isinstance(obs['observation'], dict):
            camera_keys = list(obs['observation'].keys())
            if len(camera_keys) > 0:
                current_visual = obs['observation'][camera_keys[0]]['rgb']
            else:
                raise ValueError("No camera observations found")
        else:
            raise ValueError("Invalid observation format")

        # Extract bilateral tactile observations and concatenate them into one
        # composite tactile image so the encoder sees both sides.
        if 'tactile' in obs and isinstance(obs['tactile'], dict):
            tactile_obs = obs['tactile']
            left_key = next((k for k in ("left_tactile", "left_gsmini") if k in tactile_obs), None)
            right_key = next((k for k in ("right_tactile", "right_gsmini") if k in tactile_obs), None)
            if left_key is None or right_key is None:
                raise KeyError(
                    "Bilateral ViTacDreamer feature extraction requires both left and right tactile observations."
                )
            current_tactile = self._compose_bilateral_tactile(
                tactile_obs[left_key]['rgb_marker' if 'rgb_marker' in tactile_obs[left_key] else 'rgb'],
                tactile_obs[right_key]['rgb_marker' if 'rgb_marker' in tactile_obs[right_key] else 'rgb'],
            )
        else:
            raise KeyError("Missing tactile observations for bilateral ViTacDreamer feature extraction.")

        task_id = obs.get("task_id")
        if task_id is not None and not isinstance(task_id, torch.Tensor):
            task_id = torch.as_tensor(task_id, dtype=torch.long)
        if isinstance(task_id, torch.Tensor) and task_id.dim() == 0:
            task_id = task_id.unsqueeze(0)
        features = self.extract_features(current_tactile, task_id=task_id)
        self.update_history(
            current_visual=current_visual,
            current_tactile=current_tactile,
            action_for_history=obs.get('action', None),
        )
        return features


def create_vitacdreamer_policy(
    base_policy_class,
    vitacdreamer_checkpoint: str,
    freeze_encoder: bool = True,
    **policy_kwargs
):
    """
    Factory function to create a policy that uses ViTacDreamer features.

    Args:
        base_policy_class: Base policy class (e.g., ACT, DiffusionPolicy)
        vitacdreamer_checkpoint: Path to trained ViTacDreamer checkpoint
        freeze_encoder: Whether to freeze the ViTacDreamer encoder
        **policy_kwargs: Additional arguments for base policy

    Returns:
        Policy instance with ViTacDreamer feature extractor
    """

    class ViTacDreamerPolicy(base_policy_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Add ViTacDreamer feature extractor
            self.feature_extractor = ViTacDreamerFeatureExtractor(
                checkpoint_path=vitacdreamer_checkpoint,
                freeze_encoder=freeze_encoder,
                device=self.device if hasattr(self, 'device') else 'cuda'
            )

        def reset(self):
            """Reset policy and feature extractor."""
            super().reset()
            self.feature_extractor.reset()

        def encode_observations(self, obs):
            """Override observation encoding to use ViTacDreamer features."""
            return self.feature_extractor(obs)

    return ViTacDreamerPolicy(**policy_kwargs)
