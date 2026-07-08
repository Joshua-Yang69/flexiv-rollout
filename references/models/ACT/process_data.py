import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import os
import h5py
import numpy as np
import argparse
import json
import importlib.util
from tqdm import tqdm

_DATA_UTIL_PATH = Path(__file__).resolve().parents[2] / "envs" / "utils" / "data.py"
_DATA_UTIL_SPEC = importlib.util.spec_from_file_location("univtac_env_data", _DATA_UTIL_PATH)
_DATA_UTIL_MODULE = importlib.util.module_from_spec(_DATA_UTIL_SPEC)
assert _DATA_UTIL_SPEC.loader is not None
_DATA_UTIL_SPEC.loader.exec_module(_DATA_UTIL_MODULE)
HDF5Handler = _DATA_UTIL_MODULE.HDF5Handler
LEGACY_RGB_JPEG_ENCODING = _DATA_UTIL_MODULE.LEGACY_RGB_JPEG_ENCODING
RGB_JPEG_ENCODING_ATTR = _DATA_UTIL_MODULE.RGB_JPEG_ENCODING_ATTR


def load_hdf5(dataset_paths, camera_type, downsample_factor, control_mode="joint"):
    data_paths = [
        'embodiment/joint',
    ]
    if control_mode == "ee":
        data_paths.append('embodiment/ee')
    if camera_type == 'all':
        data_paths.append(f'observation/head/rgb')
        data_paths.append(f'observation/wrist/rgb')
    else:
        data_paths.append(f'observation/{camera_type}/rgb')    
    
    with h5py.File(str(dataset_paths[0]), 'r') as f:
        try:
            f['tactile/left_tactile/rgb_marker']
            data_paths.append('tactile/left_tactile/rgb_marker')
            data_paths.append('tactile/right_tactile/rgb_marker')
        except:
            data_paths.append('tactile/left_gsmini/rgb_marker')
            data_paths.append('tactile/right_gsmini/rgb_marker')

    with h5py.File(str(dataset_paths[0]), 'r') as f:
        source_encoding = f.attrs.get(RGB_JPEG_ENCODING_ATTR, LEGACY_RGB_JPEG_ENCODING)
        if isinstance(source_encoding, bytes):
            source_encoding = source_encoding.decode("utf-8")

    if len(dataset_paths) == 1:
        data = HDF5Handler().load_hdf5(
            dataset_paths[0],
            data_paths=data_paths,
            resize=False,
            convert_channels=False,
            downsample_factor=downsample_factor,
        )
        data["embodiment/joint_action"] = data["embodiment/joint"][1:][np.arange(0, len(data["embodiment/joint"]) - 1, downsample_factor)]
        data["embodiment/joint_state"] = data["embodiment/joint"][:-1][np.arange(0, len(data["embodiment/joint"]) - 1, downsample_factor)]
        if control_mode == "ee":
            data["embodiment/ee_action"] = data["embodiment/ee"][1:][np.arange(0, len(data["embodiment/ee"]) - 1, downsample_factor)]
            data["embodiment/ee_state"] = data["embodiment/ee"][:-1][np.arange(0, len(data["embodiment/ee"]) - 1, downsample_factor)]
        data["episode_ends"] = np.array([len(data["embodiment/joint_state"])], dtype=np.int64)
    else:
        data = HDF5Handler().batch_gather_hdf5(
            dataset_paths,
            data_paths=data_paths,
            workers=1,
            resize=False,
            convert_channels=False,
            downsample_factor=downsample_factor,
        )

    # TacArena source episodes were historically JPEG-encoded through OpenCV
    # without an RGB->BGR conversion. For downstream ACT/DP training we want
    # processed episode_{i}.hdf5 files to store correct RGB uint8 arrays so
    # they align with the color-fix ViTacDreamer stage1/stage2 checkpoints.
    if source_encoding == LEGACY_RGB_JPEG_ENCODING and _DATA_UTIL_MODULE.cv2 is not None:
        for key in list(data.keys()):
            if key.endswith("/rgb") or key.endswith("/rgb_marker"):
                data[key] = data[key][..., ::-1].copy()
 
    return data


def _select_hdf5_files(hdf5_files, episode_num, selection, seed):
    if selection == "first":
        selected_indices = np.arange(episode_num)
    elif selection == "random":
        rng = np.random.default_rng(seed)
        selected_indices = np.sort(rng.choice(len(hdf5_files), size=episode_num, replace=False))
    elif selection == "stratified":
        # Deterministically cover the full ordered seed range while keeping
        # exactly episode_num demonstrations. This avoids training only on
        # a contiguous prefix such as seeds 0..49 when 100 demos are available.
        bins = np.array_split(np.arange(len(hdf5_files)), episode_num)
        selected_indices = np.array([bucket[len(bucket) // 2] for bucket in bins], dtype=int)
    else:
        raise ValueError(f"Unknown demo selection mode: {selection}")
    return [hdf5_files[i] for i in selected_indices], selected_indices


def data_transform(path, episode_num, save_path, control_mode="joint", selection="first", selection_seed=0):
    hdf5_dir = Path(path) / 'hdf5'
    if not hdf5_dir.exists():
        hdf5_dir = Path(path)
        if len(list(hdf5_dir.glob('*.hdf5'))) == 0:
            print(f"HDF5 directory does not exist at \n{hdf5_dir}\n")
            raise FileNotFoundError(f"HDF5 directory not found: {hdf5_dir}")
    
    # 获取所有 episode 文件
    hdf5_files = sorted(hdf5_dir.glob('*.hdf5'), key=lambda x: int(x.stem))
    assert episode_num <= len(hdf5_files), f"data num not enough: requested {episode_num}, found {len(hdf5_files)}"
    selected_files, selected_indices = _select_hdf5_files(
        hdf5_files,
        episode_num,
        selection=selection,
        seed=selection_seed,
    )

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    global task_name
    with open('../task_settings.json', 'r') as f:
        task_settings = json.load(f)
    assert task_name in task_settings, f"Task '{task_name}' not found in task_settings.json"
    camera_type = task_settings[task_name].get('camera_type', 'head')
    downsample_factor = task_settings[task_name].get('downsample', 1)
    print(
        f"Loading {episode_num} episodes with camera type '{camera_type}', "
        f"downsample factor {downsample_factor}, selection '{selection}', "
        f"selection_seed {selection_seed}."
    )
    print("Selected raw seeds:", " ".join(path.stem for path in selected_files))

    for i in tqdm(range(episode_num), desc='Writing episodes'):
        source_file = selected_files[i]
        data = load_hdf5([str(source_file)], camera_type, downsample_factor, control_mode=control_mode)

        joint_state = data['embodiment/joint_state'][:, 0:8]
        joint_action = data['embodiment/joint_action'][:, 0:8]
        if control_mode == "ee":
            if 'embodiment/ee_state' not in data:
                raise KeyError(
                    "control_mode='ee' requires raw dataset key 'embodiment/ee'. "
                    "Regenerate/download UniVTAC raw demonstrations with EE observations."
                )
            gripper_state = joint_state[:, 7:8]
            gripper_action = joint_action[:, 7:8]
            policy_state = np.concatenate([data['embodiment/ee_state'][:, 0:7], gripper_state], axis=1)
            policy_action = np.concatenate([data['embodiment/ee_action'][:, 0:7], gripper_action], axis=1)
        else:
            policy_state = joint_state
            policy_action = joint_action
        if camera_type == 'all':
            head_cam = data['observation/head/rgb']
            wrist_cam = data['observation/wrist/rgb']
        else:
            head_cam = data[f'observation/{camera_type}/rgb']
        if 'tactile/left_tactile/rgb_marker' in data:
            left_tac = data['tactile/left_tactile/rgb_marker']
            right_tac = data['tactile/right_tactile/rgb_marker']
        else:
            left_tac = data['tactile/left_gsmini/rgb_marker']
            right_tac = data['tactile/right_gsmini/rgb_marker']

        # 保存为 ACT 格式的 HDF5
        hdf5path = os.path.join(save_path, f"episode_{i}.hdf5")
        with h5py.File(hdf5path, "w") as f:
            f.attrs["control_mode"] = control_mode
            f.attrs["state_action_layout"] = "ee_xyzquat_gripper" if control_mode == "ee" else "joint7_gripper"
            f.attrs["source_file"] = str(source_file)
            f.attrs["source_seed"] = source_file.stem
            f.attrs["source_index"] = int(selected_indices[i])
            f.attrs["demo_selection"] = selection
            f.attrs["demo_selection_seed"] = int(selection_seed)
            f.create_dataset("action", data=np.array(policy_action))
            obs = f.create_group("observations")
            obs.create_dataset("qpos", data=np.array(policy_state))
            image = obs.create_group("images")
            # 只保存头部相机和触觉传感器
            if camera_type == 'all':
                image.create_dataset("cam_high", data=np.stack(head_cam), dtype=np.uint8)
                image.create_dataset("cam_wrist", data=np.stack(wrist_cam), dtype=np.uint8)
            else:
                image.create_dataset("cam_high", data=np.stack(head_cam), dtype=np.uint8)
            image.create_dataset("tac_left", data=np.stack(left_tac), dtype=np.uint8)
            image.create_dataset("tac_right", data=np.stack(right_tac), dtype=np.uint8)

    return episode_num, camera_type


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process TacArena episodes for ACT training.")
    parser.add_argument(
        "task_name",
        type=str,
        help="The name of the task (e.g., insert_hole)",
    )
    parser.add_argument("task_config", type=str, help="Task config (e.g., demo)")
    parser.add_argument("expert_data_num", type=int, help="Number of episodes to process")
    parser.add_argument(
        "--control_mode",
        choices=["joint", "ee"],
        default=os.environ.get("ACT_CONTROL_MODE", "joint"),
        help="Representation used for policy state/action. 'ee' stores [xyz, quat, gripper].",
    )
    parser.add_argument(
        "--selection",
        choices=["first", "random", "stratified"],
        default=os.environ.get("ACT_DEMO_SELECTION", "first"),
        help="Raw demonstration selection strategy when more demos exist than requested.",
    )
    parser.add_argument(
        "--selection_seed",
        type=int,
        default=int(os.environ.get("ACT_DEMO_SELECTION_SEED", "0")),
        help="Seed for --selection random.",
    )

    args = parser.parse_args()

    task_name = args.task_name
    task_config = args.task_config
    expert_data_num = args.expert_data_num
    control_mode = args.control_mode
    selection = args.selection
    selection_seed = args.selection_seed
    input_task_config = task_config
    if control_mode == "ee":
        if task_config == "default-ee" or task_config.startswith("default-ee-"):
            input_task_config = "default"
        elif task_config.endswith("-ee"):
            input_task_config = task_config[:-3]

    candidate_paths = [
        Path("../../data") / task_name / input_task_config,
        Path("../../data") / task_name / "clean",
        Path(__file__).resolve().parents[4] / "univtac_dataset" / task_name / input_task_config,
        Path(__file__).resolve().parents[4] / "univtac_dataset" / task_name / "clean",
    ]
    existing_input = next((p for p in candidate_paths if p.exists()), None)
    if existing_input is None:
        raise FileNotFoundError(
            "Could not locate input dataset. Tried:\n" +
            "\n".join(str(p) for p in candidate_paths)
        )
    input_path = str(existing_input)
    output_path = f"./data/sim-{task_name}/{task_config}-{expert_data_num}"
    
    begin, cam_type = data_transform(
        input_path,
        expert_data_num,
        output_path,
        control_mode=control_mode,
        selection=selection,
        selection_seed=selection_seed,
    )

    SIM_TASK_CONFIGS_PATH = "./SIM_TASK_CONFIGS.json"

    try:
        with open(SIM_TASK_CONFIGS_PATH, "r") as f:
            SIM_TASK_CONFIGS = json.load(f)
    except Exception:
        SIM_TASK_CONFIGS = {}

    SIM_TASK_CONFIGS[f"sim-{task_name}-{task_config}-{expert_data_num}"] = {
        "dataset_dir": f"./data/sim-{task_name}/{task_config}-{expert_data_num}",
        "num_episodes": expert_data_num,
        "episode_len": 1000,
        "control_mode": control_mode,
        "state_action_layout": "ee_xyzquat_gripper" if control_mode == "ee" else "joint7_gripper",
        "demo_selection": selection,
        "demo_selection_seed": selection_seed,
        "camera_names": ["cam_high", "tac_left", "tac_right"] if cam_type != 'all' \
            else ["cam_high", "cam_wrist", "tac_left", "tac_right"],
    }

    with open(SIM_TASK_CONFIGS_PATH, "w") as f:
        json.dump(SIM_TASK_CONFIGS, f, indent=4)
    
