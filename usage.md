Just FYI

# Initialization

All the code uses under the `~/MUSE_workspace/flexiv-rollout`

So first:

```bash
cd ~/MUSE_workspace/flexiv-rollout
```

Every time start inference, please use homing process to homing the robot arm and gripper:

```bash
python utils/homing.py --id 2
```

# Test

We make many keypoint test to check what problem happens when error occurs

## Test Robot Controller

We use joint pose control, so we only start with joint mode control test.

```bash
python rollout/run_control.py --config rollout/configs/control.yml --mode joint
```

## Test Perception

We test if the sensors work (Camera, XenseGripper, Arm)

```bash
python python rollout/run_perception.py --config rollout/configs/perception.yml --once
```

# Inference

## VT-MUSE Inference

```bash
python rollout/run
```