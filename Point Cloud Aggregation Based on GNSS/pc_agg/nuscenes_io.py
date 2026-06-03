from __future__ import annotations
from dataclasses import dataclass
from typing import Iterator, List, Dict, Optional, Tuple

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from tqdm import tqdm

from transforms import quat_to_rotmat, make_T


@dataclass
class Sweep:
    sample_data_token: str
    timestamp_us: int
    pts_lidar: np.ndarray  # Nx3 in lidar frame
    T_lidar_to_ego: np.ndarray  # 4x4
    T_ego_to_global: np.ndarray  # 4x4


@dataclass
class CameraFrame:
    channel: str
    sample_data_token: str
    timestamp_us: int
    filename: str
    width: int
    height: int
    K: np.ndarray  # 3x3
    T_cam_to_ego: np.ndarray  # 4x4
    T_ego_to_global: np.ndarray  # 4x4


def _get_T_from_record(rec: Dict) -> np.ndarray:
    R = quat_to_rotmat(np.array(rec["rotation"], dtype=np.float64))
    t = np.array(rec["translation"], dtype=np.float64)
    return make_T(R, t)


def load_scene_sweeps(
    nusc: NuScenes,
    scene_name: str,
    lidar_channel: str = "LIDAR_TOP",
    show_progress: bool = True,
) -> List[Sweep]:
    """
    Iterate over keyframes (2Hz samples), and for each sample take its LIDAR_TOP sample_data.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    sample_token = scene["first_sample_token"]

    sweeps: List[Sweep] = []
    pbar = tqdm(
        total=scene["nbr_samples"],
        disable=not show_progress,
        desc=f"Loading sweeps {scene_name}",
    )

    while sample_token:
        sample = nusc.get("sample", sample_token)
        sd_token = sample["data"][lidar_channel]
        sd = nusc.get("sample_data", sd_token)

        # Load points
        pc = LidarPointCloud.from_file(nusc.get_sample_data_path(sd_token))
        pts = pc.points[:3, :].T.astype(np.float32)  # Nx3

        # Calibrated sensor (lidar->ego)
        cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        T_lidar_to_ego = _get_T_from_record(cs)

        # Ego pose (ego->global)
        ep = nusc.get("ego_pose", sd["ego_pose_token"])
        T_ego_to_global = _get_T_from_record(ep)

        sweeps.append(
            Sweep(
                sample_data_token=sd_token,
                timestamp_us=sd["timestamp"],
                pts_lidar=pts,
                T_lidar_to_ego=T_lidar_to_ego,
                T_ego_to_global=T_ego_to_global,
            )
        )

        pbar.update(1)
        sample_token = sample["next"]

    pbar.close()
    return sweeps


def load_scene_camera_frames(
    nusc: NuScenes,
    scene_name: str,
    camera_channels: List[str],
    show_progress: bool = True,
) -> List[CameraFrame]:
    """
    Load camera keyframes aligned to sample timestamps.
    """
    scene = next(s for s in nusc.scene if s["name"] == scene_name)
    sample_token = scene["first_sample_token"]

    frames: List[CameraFrame] = []
    pbar = tqdm(
        total=scene["nbr_samples"],
        disable=not show_progress,
        desc=f"Loading cameras {scene_name}",
    )

    while sample_token:
        sample = nusc.get("sample", sample_token)

        for ch in camera_channels:
            sd_token = sample["data"][ch]
            sd = nusc.get("sample_data", sd_token)
            cs = nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep = nusc.get("ego_pose", sd["ego_pose_token"])

            K = np.array(cs["camera_intrinsic"], dtype=np.float64)
            T_cam_to_ego = _get_T_from_record(cs)
            T_ego_to_global = _get_T_from_record(ep)

            frames.append(
                CameraFrame(
                    channel=ch,
                    sample_data_token=sd_token,
                    timestamp_us=sd["timestamp"],
                    filename=sd["filename"],
                    width=sd["width"],
                    height=sd["height"],
                    K=K,
                    T_cam_to_ego=T_cam_to_ego,
                    T_ego_to_global=T_ego_to_global,
                )
            )

        pbar.update(1)
        sample_token = sample["next"]

    pbar.close()
    return frames
