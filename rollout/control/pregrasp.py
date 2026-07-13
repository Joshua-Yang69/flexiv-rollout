"""
rollout/control/pregrasp.py
───────────────────────────
Pre-grasp step: close the gripper to the target width and wait until the
finger position converges (or the timeout expires) before the policy rollout
loop starts.

Usage
-----
In a standalone script::

    from rollout.control.pregrasp import run_pregrasp, PregraspConfig

    with FlexivRizonClient(arm_cfg) as arm, XenseGripperClient(g_cfg) as gripper:
        success = run_pregrasp(gripper, PregraspConfig(grasp_width_m=0.03))

Or inside RolloutRuntime via ``runtime.pregrasp()``.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from rollout.perception.devices import GripperClient


@dataclass
class PregraspConfig:
    """Parameters that control the pre-grasp behaviour.

    Attributes
    ----------
    grasp_width_m:
        Target gripper opening (metres).  0.0 = fully closed, ~0.085 = fully open.
        Set this to the width that just surrounds the object before grasping.
    velocity_m_s:
        Finger closing speed (m/s).  Slow (0.02–0.05) for delicate objects.
    force_n:
        Maximum contact force (N).  Keep moderate (15–30 N) for fragile items.
    wait_after_s:
        Seconds to hold still after the gripper finishes closing, allowing the
        arm to settle before inference begins.
    blocking_timeout_s:
        How long to wait for the gripper to report convergence.  A warning is
        printed if this expires; rollout continues anyway unless
        ``require_success=True``.
    require_success:
        If True, raise ``RuntimeError`` when the gripper fails to converge.
    """

    grasp_width_m: float = 0.0
    velocity_m_s: float = 0.04
    force_n: float = 20.0
    wait_after_s: float = 0.5
    blocking_timeout_s: float = 5.0
    require_success: bool = False


def run_pregrasp(gripper: GripperClient, config: PregraspConfig | None = None) -> bool:
    """Close the gripper and wait for it to settle before the rollout begins.

    Parameters
    ----------
    gripper:
        An open, ready GripperClient instance.
    config:
        Pre-grasp parameters.  Defaults to PregraspConfig() if omitted.

    Returns
    -------
    bool
        ``True`` if the gripper reported convergence; ``False`` on timeout.

    Raises
    ------
    RuntimeError
        Only if ``config.require_success`` is ``True`` and the gripper times out.
    """
    if config is None:
        config = PregraspConfig()

    width_m = float(config.grasp_width_m)
    print(
        f"[pregrasp] closing gripper to {width_m * 1000:.1f} mm "
        f"@ {config.velocity_m_s * 1000:.0f} mm/s, force={config.force_n:.0f} N …"
    )

    reached = gripper.move(
        width_m,
        velocity_m_s=config.velocity_m_s,
        force_n=config.force_n,
    )

    if reached:
        print(f"[pregrasp] ✓ gripper converged to {width_m * 1000:.1f} mm")
    else:
        msg = (
            f"[pregrasp] ⚠ gripper did not converge within "
            f"{config.blocking_timeout_s:.1f} s (target={width_m * 1000:.1f} mm)"
        )
        if config.require_success:
            raise RuntimeError(msg)
        print(msg)

    if config.wait_after_s > 0.0:
        print(f"[pregrasp] holding for {config.wait_after_s:.2f} s …")
        time.sleep(config.wait_after_s)

    return reached
