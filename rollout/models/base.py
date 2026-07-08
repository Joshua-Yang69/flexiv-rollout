from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from rollout.types import Observation, PolicyAction
from utils.timing import now_ms


class BasePolicy(ABC):
    """Persistent policy interface used by the model server."""

    def load(self) -> None:
        return None

    def reset(self) -> None:
        return None

    @abstractmethod
    def infer(self, observation: Observation) -> PolicyAction:
        ...


class ConstantHoldPolicy(BasePolicy):
    """Development policy that publishes a hold action without hardware motion."""

    def infer(self, observation: Observation) -> PolicyAction:
        return PolicyAction(
            mode="hold",
            timestamp_ms=now_ms(),
            target_gripper_width=observation.robot.gripper.width_m,
            metadata={"policy": "constant_hold"},
        )


def build_policy_from_config(config: dict[str, Any]) -> BasePolicy:
    policy_type = config.get("type", "constant_hold")
    if policy_type == "constant_hold":
        return ConstantHoldPolicy()
    if policy_type == "act":
        from rollout.models.act_adapter import ACTPolicyAdapter

        return ACTPolicyAdapter(config.get("act", {}))
    if policy_type in {"vt_muse", "vt-muse", "vitacdreamer"}:
        from rollout.models.vtmuse_adapter import VTMusePolicyAdapter

        return VTMusePolicyAdapter(config.get("vt_muse", {}))
    raise ValueError(f"Unknown policy type: {policy_type}")

