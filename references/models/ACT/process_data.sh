task_name=${1}
task_config=${2}
expert_data_num=${3}
control_mode=${ACT_CONTROL_MODE:-joint}
selection=${ACT_DEMO_SELECTION:-first}
selection_seed=${ACT_DEMO_SELECTION_SEED:-0}

python process_data.py "$task_name" "$task_config" "$expert_data_num" \
    --control_mode "$control_mode" \
    --selection "$selection" \
    --selection_seed "$selection_seed"
