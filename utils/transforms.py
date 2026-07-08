from __future__ import annotations

import math

import numpy as np


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-12:
        raise ValueError("Quaternion norm is too small.")
    return quat / norm


def xyzquat_xyzw_to_mat(xyz: np.ndarray, quat_xyzw: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat_xyzw)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    mat = np.eye(4, dtype=np.float64)
    mat[:3, :3] = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    mat[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return mat


def mat_to_xyzquat_xyzw(mat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mat = np.asarray(mat, dtype=np.float64)
    rot = mat[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rot[2, 1] - rot[1, 2]) / s
        qy = (rot[0, 2] - rot[2, 0]) / s
        qz = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
    return mat[:3, 3].copy(), normalize_quat_xyzw(np.array([qx, qy, qz, qw], dtype=np.float64))


def flexiv_pose_to_mat(tcp_pose: np.ndarray) -> np.ndarray:
    """Convert Flexiv [x, y, z, qw, qx, qy, qz] to 4x4 matrix."""
    pose = np.asarray(tcp_pose, dtype=np.float64)
    return xyzquat_xyzw_to_mat(pose[:3], np.array([pose[4], pose[5], pose[6], pose[3]], dtype=np.float64))


def mat_to_flexiv_pose(mat: np.ndarray) -> np.ndarray:
    xyz, quat_xyzw = mat_to_xyzquat_xyzw(mat)
    return np.concatenate([xyz, [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]]).astype(np.float64)

