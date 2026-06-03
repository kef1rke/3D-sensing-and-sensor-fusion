from __future__ import annotations
from typing import List, Tuple, Dict
import numpy as np
import cv2

from transforms import invert_T, transform_points


def _project_points(
    K: np.ndarray, pts_cam: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    pts_cam: Nx3 in camera frame
    Returns:
      u (float), v (float), z (float)
    """
    X = pts_cam[:, 0]
    Y = pts_cam[:, 1]
    Z = pts_cam[:, 2]
    eps = 1e-6

    # Only valid if Z > 0
    Zc = np.maximum(Z, eps)
    uvw = (K @ pts_cam.T).T  # Nx3
    u = uvw[:, 0] / Zc
    v = uvw[:, 1] / Zc
    return u, v, Z


def colorize_points_from_cameras(
    xyz_global: np.ndarray,
    camera_frames,
    nusc,
    max_time_diff_us: int = 80_000,  # ~0.08s
) -> np.ndarray:
    """
    Assign RGB to each point by projecting into the nearest-in-time camera frames.
    Uses per-image z-buffer to reduce "color through walls".

    Returns colors uint8 Nx3. Uncolored points -> [0,0,0].
    """
    N = xyz_global.shape[0]
    colors = np.zeros((N, 3), dtype=np.uint8)

    if N == 0 or len(camera_frames) == 0:
        return colors

    # Sort camera frames by timestamp for fast nearest search
    cam_frames = sorted(camera_frames, key=lambda f: f.timestamp_us)
    cam_ts = np.array([f.timestamp_us for f in cam_frames], dtype=np.int64)

    # For simplicity, I colorize in chunks by picking a small set of candidate camera frames.
    # Here I do: for each camera frame, project all points and update colors if closer in z-buffer.
    # This is O(#cams * #points), okay for keyframes; for huge clouds, voxel-downsample first.

    for frame in cam_frames:
        # Load image
        img_path = nusc.get_sample_data_path(frame.sample_data_token)
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        h, w = img.shape[:2]

        # Build transform: global -> ego(cam) -> cam
        T_ego_to_global = frame.T_ego_to_global
        T_cam_to_ego = frame.T_cam_to_ego
        T_global_to_ego = invert_T(T_ego_to_global)
        T_ego_to_cam = invert_T(T_cam_to_ego)
        T_global_to_cam = T_ego_to_cam @ T_global_to_ego

        pts_cam = transform_points(T_global_to_cam, xyz_global)

        u, v, Z = _project_points(frame.K, pts_cam)

        valid = (Z > 0.5) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = np.nonzero(valid)[0]
        if idx.size == 0:
            continue

        uu = u[idx].astype(np.int32)
        vv = v[idx].astype(np.int32)
        zz = Z[idx].astype(np.float32)

        # Z-buffer per pixel
        depth = np.full((h, w), np.inf, dtype=np.float32)

        # First pass: compute nearest depth per pixel
        pix = vv * w + uu
        order = np.argsort(pix)
        pix_s = pix[order]
        zz_s = zz[order]
        idx_s = idx[order]

        # group by pixel
        boundaries = np.flatnonzero(np.diff(pix_s)) + 1
        starts = np.concatenate(([0], boundaries))
        ends = np.concatenate((boundaries, [pix_s.shape[0]]))

        for a, b in zip(starts, ends):
            p = pix_s[a]
            y = p // w
            x = p % w
            zmin = zz_s[a:b].min()
            depth[y, x] = zmin

        # Second pass: assign colors only for points that match z-buffer (within tolerance)
        tol = 0.20  # meters
        keep = []
        for i in range(idx_s.shape[0]):
            p = pix_s[i]
            y = p // w
            x = p % w
            if zz_s[i] <= depth[y, x] + tol:
                keep.append(idx_s[i])

        if not keep:
            continue
        keep = np.array(keep, dtype=np.int64)
        uu2 = u[keep].astype(np.int32)
        vv2 = v[keep].astype(np.int32)

        # Sample BGR -> RGB
        bgr = img[vv2, uu2, :]
        rgb = bgr[:, ::-1]
        colors[keep] = rgb

    return colors
