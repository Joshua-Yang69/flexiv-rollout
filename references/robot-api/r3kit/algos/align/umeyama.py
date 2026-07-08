from typing import Tuple, Optional
import numpy as np

from r3kit.utils.transformation import transform_pc


def umeyama_align(sources:np.ndarray, targets:np.ndarray, with_scale:bool=False, return_aligned:bool=False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    '''
    sources, targets: (N, 3) with one-to-one correspondence
    '''
    assert sources.shape == targets.shape
    mu_src = np.mean(sources, axis=0)
    mu_tgt = np.mean(targets, axis=0)
    src_centered = sources - mu_src
    tgt_centered = targets - mu_tgt

    cov = src_centered.T @ tgt_centered / sources.shape[0]
    U, D, Vt = np.linalg.svd(cov)
    S_mat = np.eye(3)
    if np.linalg.det(U @ Vt) < 0:
        S_mat[2, 2] = -1

    R = (U @ S_mat @ Vt).T

    if with_scale:
        var_src = np.sum(src_centered ** 2) / sources.shape[0]
        s = np.sum(D * np.diag(S_mat)) / var_src
    else:
        s = 1.0

    t = mu_tgt - s * R @ mu_src

    align_transformation = np.eye(4)
    align_transformation[:3, :3] = s * R
    align_transformation[:3, 3] = t
    if return_aligned:
        aligned_sources = transform_pc(sources, align_transformation)
    else:
        aligned_sources = None
    return align_transformation, aligned_sources


if __name__ == "__main__":
    import open3d as o3d
    from r3kit.utils.vis import vis_pc
    from r3kit.utils.transformation import delta_smat

    mesh = o3d.geometry.TriangleMesh.create_cone(radius=1.0, height=3.0, resolution=20)
    pcd = mesh.sample_points_uniformly(number_of_points=1000)
    sources = np.asarray(pcd.points).copy()
    pcd.scale(0.5, center=(0, 0, 0))
    pcd.rotate(np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]]), center=(0, 0, 0))
    sources = np.concatenate([sources, np.asarray(pcd.points)], axis=0)
    targets = sources.copy() + np.random.randn(*sources.shape) * 0.01
    transformation = np.eye(4)
    transformation[:3, :3] = 0.5 * np.array([[0.6124, -0.7891, 0.0474], [0.6124, 0.4356, -0.6597], [0.5, 0.433, 0.75]])
    transformation[:3, 3] = np.array([0.5, -0.25, 2.0])
    targets = transform_pc(targets, transformation)
    vis_pc(np.concatenate([sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(sources)), np.array([[0, 1, 0]] * len(targets))]))

    align_transformation, aligned_sources = umeyama_align(sources, targets, with_scale=True, return_aligned=True)
    vis_pc(np.concatenate([aligned_sources, targets]),
           np.concatenate([np.array([[1, 0, 0]] * len(aligned_sources)), np.array([[0, 1, 0]] * len(targets))]))
    print(delta_smat(align_transformation, transformation))
