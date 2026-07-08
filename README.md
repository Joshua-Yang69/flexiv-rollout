# Flexiv Policy Rollout

Real-world policy rollout scaffold for a Flexiv Rizon 4s arm with an Xense
gripper, Xense tactile sensor, and RealSense D415 camera.

The project is organized around a low-latency three-stage deployment pipeline:

1. `perception`: reads robot state and sensor frames into a latest-observation buffer.
2. `model_server`: keeps the policy model loaded and publishes the latest action.
3. `control`: consumes actions, applies safety limits, and sends commands to the robot.

All implementation code lives under `rollout/` and `utils/`. The `references/`
directory keeps trimmed upstream code used as implementation reference for ACT,
vt-muse/ViTacDreamer, Flexiv, Xense, and RealSense integration.

## Current Status

Implemented:

- 7+1 robot state contract: 7 Flexiv joints plus 1 Xense gripper width.
- Mock devices for local development without robot hardware.
- Flexiv, Xense gripper, Xense tactile, and RealSense D415 wrapper classes.
- Latest-value buffers to avoid queue backlog and reduce rollout latency.
- Persistent model server interface.
- ACT adapter scaffold.
- vt-muse feature-extractor adapter scaffold.
- TCP control executor for Flexiv `NRT_CARTESIAN_MOTION_FORCE` mode.
- Safety clipping for TCP step size, joint step size, wrench, and gripper width.
- YAML configs for perception, model server, control, and full rollout.

The default configs use mock devices and a hold policy, so the full pipeline can
run on a development machine.

## Repository Layout

```text
rollout/
  configs/             YAML configs for each deployment stage
  perception/          Hardware wrappers and observation capture loop
  models/              Policy interface, server, ACT and vt-muse adapters
  control/             Action safety and robot command execution
  runtime.py           Single-process orchestrator for all three loops
  run_*.py             CLI entrypoints
utils/
  config.py            YAML loading helpers
  latest_buffer.py     Latest-value and shared-array buffers
  timing.py            Fixed-rate loop utilities
  transforms.py        Pose and quaternion conversion utilities
references/
  models/              ACT and vt-muse reference code
  robot-api/           Flexiv, Xense, and RealSense reference code
```

## Quick Start With Mock Devices

From the repository root:

```bash
python rollout/run_perception.py --once
python rollout/run_model_server.py
python rollout/run_control.py
python -B rollout/run_rollout.py --duration-s 1
```

Expected behavior:

- `run_perception.py --once` prints an 8D `qpos8` state.
- `run_model_server.py` prints a hold `PolicyAction`.
- `run_control.py` applies a hold action through mock hardware.
- `run_rollout.py` starts perception, model, and control loops together.

## Configuration

Main config:

```text
rollout/configs/rollout.yml
```

Stage-specific configs:

```text
rollout/configs/perception.yml
rollout/configs/model_server.yml
rollout/configs/control.yml
```

By default, all device drivers are `mock`:

```yaml
perception:
  arm:
    driver: mock
  gripper:
    driver: mock
  visual:
    driver: mock
  tactile:
    driver: mock
```

For real hardware, switch drivers in `rollout/configs/rollout.yml`:

```yaml
perception:
  arm:
    driver: flexiv
  gripper:
    driver: xense
  visual:
    driver: realsense_d415
  tactile:
    driver: xense
    xense:
      fps: 50
    sensors:
      xense_left:
        device_id: OG001452
        mac_addr: 1659f0e0dde0
      xense_right:
        device_id: OG001454
        mac_addr: 1659f0e0dde0
```

The control stage reuses the perception arm and gripper instances when running
through `RolloutRuntime`, so only one hardware connection is created per device.

## Hardware Dependencies

The hardware SDKs are imported lazily. Mock rollout does not require them.

Real deployment requires the relevant SDKs installed on the robot machine:

- `flexivrdk` for Flexiv Rizon.
- `xensegripper` for Xense gripper control.
- `xensesdk` for Xense tactile sensor frames.
- `pyrealsense2` for RealSense D415.
- `PyYAML` for YAML config loading.
- `torch` and model-specific dependencies for ACT or vt-muse inference.

## Policy Modes

Available policy types in `rollout/configs/model_server.yml`:

```yaml
policy:
  type: constant_hold
```

Supported values:

- `constant_hold`: development policy that publishes hold actions.
- `act`: loads the ACT adapter scaffold from `references/models/ACT`.
- `vt_muse`: loads the vt-muse/ViTacDreamer feature extractor scaffold.

The ACT adapter decodes either joint actions or TCP/end-effector actions into
the shared `PolicyAction` contract. The vt-muse adapter currently loads feature
extraction but still needs a downstream action head before it can command the
robot.

## Control Safety

The control executor enforces limits before sending commands:

```yaml
runtime:
  max_action_age_ms: 250
  allow_joint_commands: false
  safety:
    max_tcp_translation_step_m: 0.02
    max_joint_step_rad: 0.05
    gripper_min_width_m: 0.0
    gripper_max_width_m: 0.085
    max_wrench_abs: 30
```

Joint commands are disabled by default because the intended real robot path is
Flexiv TCP force/motion control via `NRT_CARTESIAN_MOTION_FORCE`.

## Development Checks

Syntax check:

```bash
python -m compileall -q rollout utils
```

Mock full-stack smoke test:

```bash
python -B rollout/run_rollout.py --duration-s 1
```

On Windows, the same commands also work if `/` is replaced with `\`.

## Notes

- This repository does not include trained checkpoints.
- Large generated artifacts such as videos, logs, runs, checkpoints, and model
  weights are ignored by `.gitignore`.
- Keep deployment-specific IDs, IP addresses, and checkpoint paths in YAML
  config files rather than hard-coding them in rollout modules.
