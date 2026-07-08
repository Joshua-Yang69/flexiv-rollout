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
- ACT adapter scaffold (`control_mode: joint` or `ee`/`tcp`/`cartesian`).
- vt-muse feature-extractor adapter scaffold (action head not yet wired).
- Dual NRT control paths on the Flexiv RDK, selected automatically from
  `PolicyAction.mode`: `NRT_CARTESIAN_MOTION_FORCE` (TCP) and
  `NRT_JOINT_IMPEDANCE` (joint).
- Safety clipping for TCP step size, joint step size, wrench, and gripper width.
- YAML configs for perception, model server, control, and full rollout.
- Single-point real-robot test script with per-case pre-flight safety guards.
- Policy rollout simulation script (joint / TCP, configurable Hz and steps).

The default configs use mock devices and a hold policy, so the full pipeline can
run on a development machine without any hardware.

## Repository Layout

```text
rollout/
  configs/
    perception.yml       Perception loop: drivers, FPS, sensor toggles
    model_server.yml     Policy type and model args
    control.yml          Executor FPS, NRT velocity limits, safety limits
    rollout.yml          Full-pipeline config (merges all three stages)
  perception/
    devices.py           ArmClient, GripperClient, Visual/Tactile clients + Mock variants
    server.py            StatePerceiver — observation capture loop
  models/
    base.py              BasePolicy ABC + ConstantHoldPolicy
    act_adapter.py       ACT policy adapter (joint and TCP output modes)
    vtmuse_adapter.py    vt-muse feature extractor adapter (stub action head)
    server.py            PolicyModelServer — persistent inference loop
  control/
    safety.py            SafetyLimits — per-step clipping for TCP/joint/gripper/wrench
    executor.py          ControlExecutor — action routing + NRT command dispatch
  runtime.py             RolloutRuntime — single-process three-loop orchestrator
  run_perception.py      CLI — capture observations (once / stream / timed)
  run_model_server.py    CLI — run one policy inference on a mock observation
  run_control.py         CLI — send a single test action (hold / tcp / joint)
  run_rollout.py         CLI — start the full three-loop pipeline
utils/
  config.py              YAML loading helpers
  latest_buffer.py       LatestBuffer (thread-safe) + SharedArrayBuffer (lock-free IPC)
  timing.py              RateLimiter, now_ms
  transforms.py          Pose ↔ quaternion conversions, Flexiv pose ↔ 4×4 matrix
references/
  models/                ACT and vt-muse reference code
  robot-api/             Flexiv, Xense, and RealSense reference code
test_real_robot.py       Real-robot single-point test (tcp_x / tcp_z / joint / gripper)
test_rollout_sim.py      Policy rollout simulation (mock or real, joint / tcp)
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  perception loop  (50 Hz)                                       │
│  StatePerceiver.capture_once()                                  │
│    arm.read_state() + gripper.read_state() + camera + tactile  │
│                        │                                        │
│                        ▼                                        │
│               observation_buffer (LatestBuffer)                 │
└───────────────────────────────────────────────────────────────┬─┘
                                                                │
┌───────────────────────────────────────────────────────────────▼─┐
│  model server loop  (30 Hz)                                     │
│  PolicyModelServer.infer_once()                                 │
│    policy.infer(observation) → PolicyAction                     │
│      mode="joint"  → target_joints  (7-DoF, rad)               │
│      mode="tcp"    → target_tcp_pose (4×4) + wrench (6-D)      │
│      mode="hold"   → no motion                                  │
│                        │                                        │
│                        ▼                                        │
│               action_buffer (LatestBuffer)                      │
└───────────────────────────────────────────────────────────────┬─┘
                                                                │
┌───────────────────────────────────────────────────────────────▼─┐
│  control loop  (50 Hz)                                          │
│  ControlExecutor.apply_once()                                   │
│    SafetyLimits.enforce() → clip TCP step / joint step          │
│    mode="tcp"   → FlexivRizonClient.send_cartesian_motion_force │
│                   NRT_CARTESIAN_MOTION_FORCE                    │
│    mode="joint" → FlexivRizonClient.send_joint_positions        │
│                   NRT_JOINT_IMPEDANCE                           │
│    + XenseGripperClient.move(target_gripper_width)              │
│                        │                                        │
│                        ▼                                        │
│               result_buffer (LatestBuffer)                      │
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start With Mock Devices

From the repository root:

```bash
# 1. Capture one observation (prints qpos8, sensor flags)
python rollout/run_perception.py --once

# 2. Run one policy inference and print the resulting PolicyAction
python rollout/run_model_server.py

# 3. Send a single test action through the control executor
python rollout/run_control.py --mode hold

# 4. Run the full three-loop pipeline for 1 second
python -B rollout/run_rollout.py --duration-s 1
```

Expected behavior:

- `run_perception.py --once` prints an 8D `qpos8` state, skew, and sensor flags.
- `run_model_server.py` prints a hold `PolicyAction`.
- `run_control.py --mode hold` prints `ActionResult(accepted=True, ...)`.
- `run_rollout.py` starts all three loops and exits cleanly after 1 s.

## Single-Stage CLI Reference

### run_perception.py

```bash
# Capture once and print observation summary
python rollout/run_perception.py --once

# Stream continuously until Ctrl-C (uses perception.yml by default)
python rollout/run_perception.py

# Stream for a fixed duration
python rollout/run_perception.py --duration-s 5

# Custom config
python rollout/run_perception.py --config rollout/configs/perception.yml --once
```

Output fields from `--once`:

```python
{
  "qpos8": [...],          # 7 joint angles (rad) + gripper width (m)
  "state_skew_ms": 1.2,    # arm vs gripper timestamp skew
  "has_visual": True,
  "has_tactile": True,
  "tactile_keys": ["xense_left", "xense_right"]
}
```

### run_model_server.py

```bash
# Run one inference on a mock observation (uses model_server.yml by default)
python rollout/run_model_server.py

# Custom config (e.g. switch to ACT policy)
python rollout/run_model_server.py --config rollout/configs/model_server.yml
```

### run_control.py

Sends a single test action and prints the `ActionResult`. The `--mode` flag
selects `PolicyAction.mode` and therefore the NRT path exercised on the
Flexiv RDK. The target is always the **current** robot state, so no physical
motion occurs on real hardware.

```bash
# hold — no motion, NRT mode unchanged (default)
python rollout/run_control.py --mode hold

# tcp — current TCP pose as target → NRT_CARTESIAN_MOTION_FORCE
python rollout/run_control.py --mode tcp

# joint — current joint angles as target → NRT_JOINT_IMPEDANCE
python rollout/run_control.py --mode joint

# Custom config
python rollout/run_control.py --mode tcp --config rollout/configs/control.yml
```

### run_rollout.py

```bash
# Run until Ctrl-C
python -B rollout/run_rollout.py

# Run for a fixed duration
python -B rollout/run_rollout.py --duration-s 10

# Custom config
python -B rollout/run_rollout.py --config rollout/configs/rollout.yml
```

## Real-Robot Single-Point Test

`test_real_robot.py` connects to live hardware and runs isolated motion cases.
Each case reads the **current** robot state as its base, applies a small delta,
then returns to the original position. A pre-flight safety guard aborts before
sending any command if the requested delta exceeds the hard limit.

```bash
# TCP +X direction, 10 mm (default — most conservative starting point)
python test_real_robot.py --case tcp_x

# TCP +Z direction (upward), custom delta
python test_real_robot.py --case tcp_z --delta-mm 5

# Single-joint nudge — J7 (wrist) +2.9° by default
python test_real_robot.py --case joint

# Custom joint index and angle
python test_real_robot.py --case joint --joint-idx 5 --delta-deg 2.0

# Gripper open/close with blocking readback
python test_real_robot.py --case gripper

# All cases in sequence
python test_real_robot.py --case all --delta-mm 8
```

Pre-flight safety hard limits (applied **before** any command is sent):

| Limit | Value |
|---|---|
| Max TCP translation per call | 15 mm |
| Max joint step per call | 4.6° (0.08 rad) |
| HOME distance check | warn + confirm if > 30 mm from HOME TCP |

If the distance to HOME exceeds 30 mm on startup, the script prompts for
manual confirmation before proceeding.

Gripper case uses `set_position_sync` (blocking, 5 s timeout) and prints the
actual width readback after each move, matching the reference implementation in
`third_parties/r3kit/r3kit/devices/gripper/xense/xense.py`.

## Policy Rollout Simulation

`test_rollout_sim.py` runs the complete publisher → `LatestBuffer` →
`ControlExecutor` loop at a configurable rate, using a deterministic joint/TCP
pose sequence to simulate a real policy rollout without a trained model.

Architecture mirrors the real three-loop pipeline:

```
Publisher thread      action_buffer      Executor thread
────────────────       ─────────────      ───────────────
sequence[step].put() ──────────────────► apply_once()
@ RateLimiter(hz)                        @ RateLimiter(hz)
                                                │
                       result_buffer ◄──────────┘
                            │
                      Collector thread
                      prints per-frame result
```

```bash
# Mock hardware, joint mode, 20 Hz, 40 steps — 2 s total (default)
python test_rollout_sim.py

# Custom frequency and step count
python test_rollout_sim.py --hz 10 --steps 30

# TCP mode
python test_rollout_sim.py --mode tcp --hz 20 --steps 60

# Real hardware, joint mode
python test_rollout_sim.py --real --hz 20 --steps 40

# Real hardware, TCP mode
python test_rollout_sim.py --real --mode tcp --hz 20 --steps 40
```

Built-in motion profiles:

| `--mode` | Motion | Gripper |
|---|---|---|
| `joint` (default) | J1 ±5° sine + J4 ±3° double-freq sine | ±20 mm sine |
| `tcp` | X-axis ±10 mm sine | ±20 mm sine |

Each frame prints step index, target pose (joints in degrees or TCP XYZ in mm),
gripper width, and any safety clipping messages. A summary of accepted /
rejected / safety-clipped counts is printed at the end.

## Configuration Reference

### Stage configs vs rollout.yml

Each stage has a standalone config used when running `run_*.py` individually:

```
rollout/configs/perception.yml    ← used by run_perception.py
rollout/configs/model_server.yml  ← used by run_model_server.py
rollout/configs/control.yml       ← used by run_control.py
rollout/configs/rollout.yml       ← used by run_rollout.py (merges all three)
```

### Mock vs real hardware

By default `rollout.yml` uses `driver: mock` for all devices. Switch to real
hardware by editing the driver fields:

```yaml
# rollout/configs/rollout.yml
perception:
  arm:
    driver: flexiv               # or: mock
    flexiv:
      robot_id: Rizon4s-063586
      tool_name: xense
      auto_enable: true
      operational_wait_s: 10
      tcp_mode_on_start: false   # NRT mode is switched per-action automatically
  gripper:
    driver: xense                # or: mock
    xense:
      device_id: 1659f0e0dde0
      blocking_timeout_s: -1     # -1 = non-blocking (rollout default)
      blocking_tolerance_mm: 1
  visual:
    driver: realsense_d415       # or: mock
    realsense:
      serial: "314522062078"
      width: 640
      height: 480
      fps: 30
  tactile:
    driver: xense                # or: mock
    xense:
      fps: 30
    sensors:
      xense_left:
        device_id: OG001452
        mac_addr: 1659f0e0dde0
      xense_right:
        device_id: OG001454
        mac_addr: 1659f0e0dde0
```

### Gripper blocking mode

The gripper `blocking_timeout_s` field controls whether `gripper.move()` waits
for the gripper to reach the target position:

| Value | Behaviour |
|---|---|
| `-1` (default) | Non-blocking — `set_position()`, returns immediately |
| `> 0` (e.g. `5.0`) | Blocking — `set_position_sync()`, waits up to N seconds |

Use `-1` during rollout (non-blocking keeps the control loop running at full
rate). Use `5.0` in `test_real_robot.py` (blocking with readback verification).

### Policy type

```yaml
# rollout/configs/model_server.yml
policy:
  type: constant_hold   # development default — no motion
  # type: act
  # type: vt_muse
```

### ACT policy

```yaml
policy:
  type: act
  act:
    control_mode: joint      # joint → PolicyAction(mode="joint", target_joints=...)
    # control_mode: ee       # ee / tcp / cartesian → PolicyAction(mode="tcp", ...)
    camera_names:
      - cam_high
    tactile_names:
      - tac_left
      - tac_right
    image_size: 256
    normalize_visual: true
    model_args:
      device: cuda:0
      chunk_size: 1
      state_dim: 8
```

### Control safety limits

```yaml
# rollout/configs/control.yml
runtime:
  fps: 50
  max_action_age_ms: 250         # reject actions older than this
  max_linear_vel: 0.1            # m/s  (NRT_CARTESIAN_MOTION_FORCE)
  max_linear_acc: 0.5            # m/s²
  max_angular_vel: 0.1           # rad/s
  max_angular_acc: 0.5           # rad/s²
  gripper_velocity_m_s: 0.08
  gripper_force_n: 27
  safety:
    max_tcp_translation_step_m: 0.02   # clip per-step TCP translation
    max_joint_step_rad: 0.05           # clip per-step joint delta
    gripper_min_width_m: 0.0
    gripper_max_width_m: 0.085
    max_wrench_abs: 30                 # N or N·m per component
```

## Dual NRT Control Modes

The control executor selects the Flexiv NRT mode automatically from
`PolicyAction.mode`. No config flag or manual switch is needed.

| `PolicyAction.mode` | Flexiv RDK mode | RDK call |
|---|---|---|
| `"tcp"` | `NRT_CARTESIAN_MOTION_FORCE` | `SendCartesianMotionForce` |
| `"joint"` | `NRT_JOINT_IMPEDANCE` | `SendJointPosition` |
| `"hold"` | unchanged | — |

Mode switching is idempotent: `FlexivRizonClient` tracks the active mode and
only calls `SwitchMode` when it actually changes, so consecutive same-mode
actions have zero switching overhead.

## Policy Modes

| `type` | Description |
|---|---|
| `constant_hold` | Always publishes `mode="hold"`. No motion. Default for development. |
| `act` | ACT policy adapter. `control_mode: joint` → joint output; `ee`/`tcp`/`cartesian` → TCP output. |
| `vt_muse` | vt-muse / ViTacDreamer feature extractor. Action head not yet wired — raises `NotImplementedError` on `infer()`. |

## Hardware Dependencies

All SDKs are imported lazily. Mock rollout does not require any of them.

| SDK | Used for |
|---|---|
| `flexivrdk` | Flexiv Rizon arm — `Robot`, `Mode`, `Tool` |
| `xensegripper` | Xense gripper — `XenseGripper` |
| `xensesdk` | Xense tactile sensor — `Sensor` |
| `pyrealsense2` | RealSense D415 — pipeline and frame capture |
| `PyYAML` | YAML config loading |
| `torch` | ACT / vt-muse model inference |

## Development Checks

```bash
# Syntax check all rollout and utils modules
python -m compileall -q rollout utils

# Mock full-stack smoke test (no hardware)
python -B rollout/run_rollout.py --duration-s 1

# Rollout simulation (no hardware, joint mode)
python test_rollout_sim.py --hz 10 --steps 20

# Rollout simulation (no hardware, TCP mode)
python test_rollout_sim.py --mode tcp --hz 10 --steps 20
```

On Windows, replace `/` with `\` in all paths.

## Notes

- This repository does not include trained model checkpoints.
- Large generated artifacts (videos, logs, runs, checkpoints, model weights)
  are excluded by `.gitignore`.
- Keep deployment-specific values (robot IDs, IP addresses, serial numbers,
  checkpoint paths) in YAML config files — do not hard-code them in Python.
- The control stage reuses the perception `arm` and `gripper` instances when
  running through `RolloutRuntime`, so only one hardware connection per device
  is created.
