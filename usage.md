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

# Test (Optional)

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

Before every inference, we should first home the robot, here is the SOP.

```bash
python utils/homing.py --id 2
python rollout/run_rollout.py --config /path/to/config
```

Every time in inference, will have the first pregrasp process, you should put your object under the gripper and press 'Enter' button, and then after the object is set, press the 'Enter' button again to start inference.

## VT-MUSE Inference

All configs are set in the `rollout/configs/`

If you want to make a new inference template, please copy and create a new one.

Some setting for reference:

```yml
# Pregrasp
pregrasp:
  grasp_width_m: 0.02        # metres — finger gap at grasp; tune per object This is important if the gripper not hold the object tight, you can set
  velocity_m_s: 0.1         # closing speed slow for delicate, if you want to let the process quick enough you can increase, don't exceed 0.3
  ...

model_server:
  policy:
    type: vt_muse # here used for change the type of the policy you use, it influence what setting below will be used (ACT or vt-MUSE)

    vt_muse:
      # Path to trained ViTacDreamer Stage-2 encoder-only checkpoint (required)
      checkpoint: "/home/xense/MUSE_workspace/flexiv-rollout/rollout/models/checkpoints/stage2_prior512_hlen5_depthdelta_80train20val/stage2_v2_encoder_only.pth" # Trained Encoder
      ...
      task_id: 0 # Important! Used for task changing
      model_args:
        # For down stream policy
        ...
        vitacdreamer_fusion_mode: feature_query_policy_kv
        vitacdreamer_cross_attn_layers: middle
        finetune_vitacdreamer_encoder: false
        # use_vitacdreamer_feature is forced to true by VTMusePolicyAdapter.load()
        # ── checkpoint ───────────────────────────────────────────────────────
        ckpt_dir: "/home/xense/MUSE_workspace/flexiv-rollout/rollout/models/checkpoints/train_config_vitacdreamer_real2_stage2_depthdelta_insert_tube_100_16k"          # directory with policy_best.ckpt + dataset_stats.pkl, be careful! should check twice!
        ...
        task_name: insert_tube   # used for internal ACT logging; set per task


```

```bash
export VITACDREAMER_VIT_CHECKPOINT="/home/xense/MUSE_workspace/flexiv-rollout/rollout/models/checkpoints/google_vit_base_patch16_224" # Every time start a new terminal
python rollout/run_rollout.py --config rollout/configs/rollout_muse.yml
```

## Baselines

### ACT Inference

```bash
python rollout/run_rollout.py --config rollout/configs/rollout_act.yml # Pure Vision
```

```bash
python rollout/run_rollout.py --config rollout/configs/rollout_act_tac.yml # Vision and Tactile
```