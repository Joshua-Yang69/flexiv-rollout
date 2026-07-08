#!/bin/bash
task_name=${1}
task_config=${2}
expert_data_num=${3}
seed=${4}
gpu_id=${5}
train_config=${6:-"train_config"}
if [ -z "${NPROC_PER_NODE:-}" ]; then
    IFS=',' read -r -a _gpu_ids <<< "${gpu_id}"
    NPROC_PER_NODE=${#_gpu_ids[@]}
fi

DEBUG=False
save_ckpt=True

export CUDA_VISIBLE_DEVICES=${gpu_id}
export TORCH_HOME=${TORCH_HOME:-/inspire/hdd/global_user/yangyi-253108120173/inspire_shared/mount/advanced-machine-learning-and-deep-learning-applications/lzh/ksq/joshua/torch_cache}
export NCCL_SHM_DISABLE=${NCCL_SHM_DISABLE:-1}

if [ "${NPROC_PER_NODE}" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" imitate_episodes.py \
        --task_name sim-${task_name}-${task_config}-${expert_data_num} \
        --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${train_config} \
        --config_path ./${train_config}.yml \
        --seed ${seed}
else
    python3 imitate_episodes.py \
        --task_name sim-${task_name}-${task_config}-${expert_data_num} \
        --ckpt_dir ./act_ckpt/act-${task_name}/${task_config}-${expert_data_num}/${train_config} \
        --config_path ./${train_config}.yml \
        --seed ${seed}
fi
