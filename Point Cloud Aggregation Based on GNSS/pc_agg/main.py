import argparse
import numpy as np
import open3d as o3d

from nuscenes.nuscenes import NuScenes

from nuscenes_io import load_scene_sweeps, load_scene_camera_frames
from aggregate import aggregate_sweeps_to_global, voxel_downsample
from dynamic_filter import temporal_voxel_static_mask
from colorize import colorize_points_from_cameras
from ply_utils import save_ply_xyz, save_ply_xyzrgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataroot", required=True, help="Path to nuScenes data root folder."
    )
    ap.add_argument(
        "--version",
        default="v1.0-mini",
        help="nuScenes version, e.g. v1.0-mini, v1.0-trainval",
    )
    ap.add_argument("--scene", default="scene-0061", help="Scene name, e.g. scene-0061")
    ap.add_argument("--outdir", default="out", help="Output directory")
    ap.add_argument(
        "--voxel_ds",
        type=float,
        default=0.15,
        help="Downsample voxel size for aggregation (0 disables)",
    )
    ap.add_argument(
        "--static_voxel",
        type=float,
        default=0.20,
        help="Voxel size for temporal static filtering",
    )
    ap.add_argument(
        "--min_sweeps",
        type=int,
        default=4,
        help="Min unique sweeps per voxel to keep as static",
    )
    ap.add_argument("--colorize", action="store_true", help="Run colorization step")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # --- Task 1: Aggregate ---
    sweeps = load_scene_sweeps(
        nusc, args.scene, lidar_channel="LIDAR_TOP", show_progress=True
    )
    agg = aggregate_sweeps_to_global(sweeps)
    xyz = agg.xyz_global

    if args.voxel_ds > 0:
        xyz_ds = voxel_downsample(xyz, args.voxel_ds)
        save_ply_xyz(
            f"{args.outdir}/aggregated_preview_vox{args.voxel_ds:.2f}.ply", xyz_ds
        )

    save_ply_xyz(f"{args.outdir}/aggregated_raw.ply", xyz)

    # --- Task 2: Moving object filtering (static voxels) ---
    static_mask = temporal_voxel_static_mask(
        xyz=agg.xyz_global,
        sweep_ids=agg.sweep_ids,
        voxel_size=args.static_voxel,
        min_unique_sweeps=args.min_sweeps,
    )
    xyz_static = agg.xyz_global[static_mask]
    save_ply_xyz(f"{args.outdir}/static_filtered.ply", xyz_static)

    # Optional Open3D cleanup
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_static.astype(np.float64))
    pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    xyz_static_clean = np.asarray(pcd.points).astype(np.float32)
    save_ply_xyz(f"{args.outdir}/static_filtered_clean.ply", xyz_static_clean)

    # --- Task 3: Colorization ---
    if args.colorize:
        camera_channels = [
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
        ]
        cams = load_scene_camera_frames(
            nusc, args.scene, camera_channels, show_progress=True
        )

        # For speed: downsample before colorization
        xyz_col = voxel_downsample(xyz_static_clean, voxel_size=0.10)
        rgb = colorize_points_from_cameras(xyz_col, cams, nusc)
        save_ply_xyzrgb(f"{args.outdir}/static_colored.ply", xyz_col, rgb)


if __name__ == "__main__":
    main()
