from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from transforms import transform_points


@dataclass
class AggregatedCloud:
    xyz_global: np.ndarray  # Mx3
    sweep_ids: np.ndarray  # M  (int sweep index per point)
    timestamps_us: np.ndarray  # M  (timestamp per point)


def aggregate_sweeps_to_global(sweeps) -> AggregatedCloud:
    """
    For each sweep:
      lidar pts -> ego -> global
    Accumulate points and keep sweep id + timestamp for filtering later.
    """
    all_xyz = []
    all_sid = []
    all_ts = []

    for sid, sw in enumerate(sweeps):
        # lidar->ego->global
        pts_ego = transform_points(sw.T_lidar_to_ego, sw.pts_lidar)
        pts_glb = transform_points(sw.T_ego_to_global, pts_ego)

        all_xyz.append(pts_glb.astype(np.float32))
        all_sid.append(np.full((pts_glb.shape[0],), sid, dtype=np.int32))
        all_ts.append(np.full((pts_glb.shape[0],), sw.timestamp_us, dtype=np.int64))

    xyz = np.vstack(all_xyz) if all_xyz else np.zeros((0, 3), dtype=np.float32)
    sid = np.concatenate(all_sid) if all_sid else np.zeros((0,), dtype=np.int32)
    ts = np.concatenate(all_ts) if all_ts else np.zeros((0,), dtype=np.int64)

    return AggregatedCloud(xyz_global=xyz, sweep_ids=sid, timestamps_us=ts)


def voxel_downsample(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """
    Simple voxel downsampling by keeping one point per voxel (first occurrence).
    """
    if xyz.shape[0] == 0:
        return xyz
    grid = np.floor(xyz / voxel_size).astype(np.int64)
    # Unique voxel keys
    keys = grid[:, 0] * 73856093 ^ grid[:, 1] * 19349663 ^ grid[:, 2] * 83492791
    _, idx = np.unique(keys, return_index=True)
    return xyz[idx]
