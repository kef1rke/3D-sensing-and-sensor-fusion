from __future__ import annotations
from typing import Tuple
import numpy as np


def temporal_voxel_static_mask(
    xyz: np.ndarray,
    sweep_ids: np.ndarray,
    voxel_size: float = 0.20,
    min_unique_sweeps: int = 4,
) -> np.ndarray:
    """
    Returns boolean mask of points considered STATIC using temporal voxel support.

    Steps:
      - Assign each point to a voxel in global space
      - For each voxel, compute number of unique sweeps contributing
      - Keep points in voxels with >= min_unique_sweeps

    This removes moving objects because they don't repeatedly occupy the same global voxels.
    """
    assert xyz.shape[0] == sweep_ids.shape[0]
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    vox = np.floor(xyz / voxel_size).astype(np.int64)

    # Hash voxels to 1D keys (fast dictionary-like grouping)
    keys = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791

    # Sort by voxel key for grouping
    order = np.argsort(keys)
    keys_s = keys[order]
    sid_s = sweep_ids[order]

    # Find group boundaries
    boundaries = np.flatnonzero(np.diff(keys_s)) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [keys_s.shape[0]]))

    # For each group, compute unique sweep count
    unique_counts = np.zeros_like(keys_s, dtype=np.int16)
    for a, b in zip(starts, ends):
        uc = np.unique(sid_s[a:b]).size
        unique_counts[a:b] = uc

    # Map back to original order
    uc_full = np.empty_like(unique_counts)
    uc_full[order] = unique_counts

    return uc_full >= min_unique_sweeps


def radius_outlier_mask(
    xyz: np.ndarray, radius: float = 0.6, min_neighbors: int = 4
) -> np.ndarray:
    """
    Lightweight radius-based outlier removal using a voxel neighbor search.
    """
    if xyz.shape[0] == 0:
        return np.zeros((0,), dtype=bool)

    # Use voxel grid for neighbor count approx
    vox = np.floor(xyz / radius).astype(np.int64)
    keys = vox[:, 0] * 73856093 ^ vox[:, 1] * 19349663 ^ vox[:, 2] * 83492791

    # Count points per voxel
    uniq, counts = np.unique(keys, return_counts=True)
    count_map = dict(zip(uniq.tolist(), counts.tolist()))
    neigh_counts = np.array([count_map[k] for k in keys], dtype=np.int32)

    return neigh_counts >= min_neighbors
