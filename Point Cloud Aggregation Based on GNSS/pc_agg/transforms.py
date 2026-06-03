import numpy as np


def quat_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    Convert quaternion [w, x, y, z] to 3x3 rotation matrix.
    nuScenes stores quaternions as [w, x, y, z].
    """
    w, x, y, z = q_wxyz
    # Normalize to be safe
    n = np.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n

    R = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Build 4x4 homogeneous transform from R (3x3) and t (3,)."""
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def invert_T(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transform 4x4."""
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def transform_points(T: np.ndarray, pts_xyz: np.ndarray) -> np.ndarray:
    """
    Apply 4x4 transform to Nx3 points.
    """
    assert pts_xyz.ndim == 2 and pts_xyz.shape[1] == 3
    ones = np.ones((pts_xyz.shape[0], 1), dtype=pts_xyz.dtype)
    pts_h = np.hstack([pts_xyz, ones])
    out = (T @ pts_h.T).T
    return out[:, :3]
