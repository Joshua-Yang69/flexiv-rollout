from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

from rollout.models.base import BasePolicy
from rollout.types import Observation, PolicyAction
from utils.timing import now_ms


class VTMusePolicyAdapter(BasePolicy):
    """Stub adapter for vt-muse / ViTacDreamer-based deployments.

    The checked-in vt-muse reference currently exposes feature extraction. This
    adapter loads that feature extractor persistently and provides a clear hook
    for a downstream action head when one is available.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.feature_extractor = None
        self.action_head = None

    def load(self) -> None:
        checkpoint = self.config.get("checkpoint")
        if not checkpoint:
            raise RuntimeError("vt-muse config requires a 'checkpoint' for feature extraction.")

        repo_root = Path(__file__).resolve().parents[2]
        wrapper_path = repo_root / "references" / "models" / "vt-muse" / "policy_wrapper.py"
        spec = importlib.util.spec_from_file_location("vt_muse_policy_wrapper", wrapper_path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Could not load vt-muse wrapper from {wrapper_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.feature_extractor = module.ViTacDreamerFeatureExtractor(
            checkpoint_path=checkpoint,
            freeze_encoder=bool(self.config.get("freeze_encoder", True)),
            device=self.config.get("device", "cuda:0"),
        )

    def reset(self) -> None:
        if self.feature_extractor is not None and hasattr(self.feature_extractor, "reset"):
            self.feature_extractor.reset()

    def infer(self, observation: Observation) -> PolicyAction:
        if self.feature_extractor is None:
            self.load()
        raise NotImplementedError(
            "vt-muse feature extraction is loaded, but no downstream action head "
            "is configured yet. Add an action head and decode it into PolicyAction."
        )


class VTMuseHoldPolicy(BasePolicy):
    def infer(self, observation: Observation) -> PolicyAction:
        return PolicyAction(
            mode="hold",
            timestamp_ms=now_ms(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"policy": "vt_muse_hold"},
        )

