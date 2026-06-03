import argparse
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

import open3d as o3d

try:
    from pyproj import Proj, Transformer

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False
    print("Warning: pyproj not found. Falling back to simple local ENU approximation.")


#  Data structures


@dataclass
class Pose2D:
    """2D pose with heading (yaw) in radians in some world frame."""

    x: float
    y: float
    yaw: float  # heading in radians


#  Timestamp utilities


def timestamp_to_float(secs: int, nsecs: int) -> float:
    """Convert ROS-style secs + nsecs to float seconds."""
    return float(secs) + float(nsecs) * 1e-9


def parse_timestamp_from_filename(fname: str) -> float:
    """
    Parse LiDAR timestamp from PCD filename.

    Supports:
      - "<secs>_<nsecs>.pcd"
      - "<secs><nsecs>.pcd"  (concatenated)
    Returns:
      float seconds
    """
    base = os.path.basename(fname)
    name, _ = os.path.splitext(base)

    # Try with underscore: <secs>_<nsecs>
    if "_" in name:
        parts = name.split("_")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            secs = int(parts[0])
            nsecs = int(parts[1])
            return timestamp_to_float(secs, nsecs)

    # Fallback: pure concatenation <secs><nsecs> with fixed 9-digit nsecs
    if name.isdigit() and len(name) > 9:
        secs = int(name[:-9])
        nsecs = int(name[-9:])
        return timestamp_to_float(secs, nsecs)

    raise ValueError(f"Cannot parse timestamp from filename: {fname}")


#  GNSS utilities


def load_gnss_fixes(csv_path: str) -> pd.DataFrame:
    """
    Load GNSS fixes from fix.csv and create:
      - 't'  : float timestamp in seconds
    """
    df = pd.read_csv(csv_path)

    required_cols = [
        "header_stamp_secs",
        "header_stamp_nsecs",
        "latitude",
        "longitude",
        "altitude",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing from GNSS CSV: {csv_path}")

    df["t"] = (
        df["header_stamp_secs"].astype(float)
        + df["header_stamp_nsecs"].astype(float) * 1e-9
    )
    df = df.sort_values("t").reset_index(drop=True)
    return df


def latlon_to_local_xy(lat, lon, lat0, lon0):
    """
    Convert (lat, lon) to local (x,y) in meters relative to (lat0, lon0).
    Uses pyproj if available, otherwise a simple equirect approximation.
    """
    if HAS_PYPROJ:
        # WGS84 -> local UTM (approx) using pyproj transformer
        proj_wgs84 = Proj("epsg:4326")
        proj_utm = Proj(proj="utm", zone=int((lon0 + 180) / 6) + 1, ellps="WGS84")
        transformer = Transformer.from_proj(proj_wgs84, proj_utm, always_xy=True)

        x0, y0 = transformer.transform(lon0, lat0)
        x, y = transformer.transform(lon, lat)

        return x - x0, y - y0
    else:
        # Simple approximation (works OK for small areas)
        R = 6378137.0
        lat0_rad = np.deg2rad(lat0)
        dlat = np.deg2rad(lat - lat0)
        dlon = np.deg2rad(lon - lon0)

        x = R * dlon * np.cos(lat0_rad)
        y = R * dlat
        return x, y


def build_gnss_trajectory(df: pd.DataFrame) -> dict:
    """
    Build GNSS trajectory with:
      - times      : np.array of float seconds
      - xy         : np.array shape (N, 2) in meters
      - yaw        : np.array shape (N,) heading in radians (approx from trajectory)
    """
    times = df["t"].to_numpy()
    lat = df["latitude"].to_numpy()
    lon = df["longitude"].to_numpy()

    lat0 = lat[0]
    lon0 = lon[0]

    xs = np.zeros_like(lat, dtype=float)
    ys = np.zeros_like(lat, dtype=float)

    for i in range(len(lat)):
        xs[i], ys[i] = latlon_to_local_xy(lat[i], lon[i], lat0, lon0)

    # Estimate yaw from forward difference of trajectory
    yaw = np.zeros_like(xs)
    dx = np.diff(xs)
    dy = np.diff(ys)
    heading = np.arctan2(dy, dx)
    yaw[1:-1] = (heading[:-1] + heading[1:]) / 2.0
    yaw[0] = heading[0]
    yaw[-1] = heading[-1]

    traj = {
        "times": times,
        "xy": np.stack([xs, ys], axis=1),
        "yaw": yaw,
        "lat0": lat0,
        "lon0": lon0,
    }
    return traj


def interpolate_pose(t_query: float, traj: dict) -> Pose2D:
    """
    Linearly interpolate position and yaw from GNSS trajectory for time t_query.

    If t_query is outside the time range, clamp to the closest endpoint.
    """
    times = traj["times"]
    xy = traj["xy"]
    yaw = traj["yaw"]

    if t_query <= times[0]:
        x, y = xy[0]
        psi = yaw[0]
        return Pose2D(x, y, psi)

    if t_query >= times[-1]:
        x, y = xy[-1]
        psi = yaw[-1]
        return Pose2D(x, y, psi)

    # Find interval [i, i+1] such that times[i] <= t_query <= times[i+1]
    idx = np.searchsorted(times, t_query) - 1
    idx = np.clip(idx, 0, len(times) - 2)

    t0 = times[idx]
    t1 = times[idx + 1]
    alpha = (t_query - t0) / (t1 - t0)

    p0 = xy[idx]
    p1 = xy[idx + 1]
    x = (1 - alpha) * p0[0] + alpha * p1[0]
    y = (1 - alpha) * p0[1] + alpha * p1[1]

    # Interpolate yaw, careful with wrap-around
    y0 = yaw[idx]
    y1 = yaw[idx + 1]
    dyaw = ((y1 - y0 + np.pi) % (2 * np.pi)) - np.pi
    psi = y0 + alpha * dyaw

    return Pose2D(float(x), float(y), float(psi))


#  Point cloud utilities


def load_pcd(path: str) -> np.ndarray:
    """
    Load PCD file with open3d and return Nx3 numpy array of points.
    """
    pc = o3d.io.read_point_cloud(path)
    if pc.is_empty():
        return np.zeros((0, 3), dtype=np.float32)
    return np.asarray(pc.points, dtype=np.float32)


def transform_points_2d(points: np.ndarray, pose: Pose2D) -> np.ndarray:
    """
    Apply 2D SE(2) transform to LiDAR points.

    Assumes LiDAR points are in its local frame, lying in XY plane (Z possibly 0).
    We apply:
        [x']   [cos -sin tx] [x]
        [y'] = [sin  cos ty] [y]
        [z']   [ 0    0   1] [z]
    """
    if points.shape[0] == 0:
        return points

    R = np.array(
        [
            [np.cos(pose.yaw), -np.sin(pose.yaw), 0.0],
            [np.sin(pose.yaw), np.cos(pose.yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    t = np.array([pose.x, pose.y, 0.0], dtype=np.float32)

    return (points @ R.T) + t  # (N,3)


#  Main merging logic


def merge_pointclouds(pcd_dir: str, fix_csv: str) -> o3d.geometry.PointCloud:
    """
    Main function:
      - load GNSS fixes and build trajectory
      - iterate over PCD files
      - for each, estimate pose by timestamp
      - transform and accumulate into one merged point cloud
    """
    print(f"Loading GNSS fixes from {fix_csv} ...")
    df = load_gnss_fixes(fix_csv)
    traj = build_gnss_trajectory(df)

    pattern = os.path.join(pcd_dir, "*.pcd")
    pcd_files = sorted(glob.glob(pattern))
    if not pcd_files:
        raise FileNotFoundError(f"No PCD files found in: {pcd_dir}")

    print(f"Found {len(pcd_files)} PCD files in {pcd_dir}")

    all_points = []

    for i, pcd_path in enumerate(pcd_files):
        try:
            t_lidar = parse_timestamp_from_filename(pcd_path)
        except ValueError as e:
            print(f"Skipping file (timestamp parse failed): {pcd_path} | {e}")
            continue

        pose = interpolate_pose(t_lidar, traj)
        pts_local = load_pcd(pcd_path)
        pts_world = transform_points_2d(pts_local, pose)

        all_points.append(pts_world)

        if (i + 1) % 20 == 0 or (i + 1) == len(pcd_files):
            print(
                f"Processed {i + 1}/{len(pcd_files)} scans, "
                f"accumulated points: {sum(p.shape[0] for p in all_points)}"
            )

    if not all_points:
        raise RuntimeError("No points accumulated – check filenames and parsing.")

    merged_points = np.vstack(all_points)
    merged_pc = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(merged_points.astype(np.float64))
    )

    print(f"Merged point cloud has {len(merged_points)} points.")
    # Build GNSS trajectory visualization (as 3D points)
    gnss_xy = traj["xy"]
    gnss_z = np.zeros(len(gnss_xy))  # or traj["alt"] if you added altitude
    gnss_points = np.column_stack([gnss_xy, gnss_z])

    # LiDAR cloud (gray)
    lidar_pc = o3d.geometry.PointCloud()
    lidar_pc.points = o3d.utility.Vector3dVector(merged_points)
    lidar_pc.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[0.5, 0.5, 0.5]]), (len(merged_points), 1))
    )

    # Trajectory cloud (red)
    traj_pc = o3d.geometry.PointCloud()
    traj_pc.points = o3d.utility.Vector3dVector(gnss_points)
    traj_pc.colors = o3d.utility.Vector3dVector(
        np.tile(np.array([[1.0, 0.0, 0.0]]), (len(gnss_points), 1))
    )

    # Combined
    final_pc = lidar_pc + traj_pc

    # Save PLY
    o3d.io.write_point_cloud("merged_with_trajectory.ply", final_pc)
    print("Saved PLY with trajectory: merged_with_trajectory.ply")

    # IMPORTANT: return point cloud WITH trajectory
    return final_pc


#  CLI
def main():
    parser = argparse.ArgumentParser(
        description="Merge 2D LiDAR pointclouds using GNSS trajectory."
    )
    parser.add_argument(
        "--pcd_dir",
        type=str,
        required=True,
        help="Directory with PCD files (e.g., Parkolo1/pcd)",
    )
    parser.add_argument(
        "--fix_csv", type=str, required=True, help="Path to fix.csv file"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output merged PCD filename"
    )

    args = parser.parse_args()

    merged_pc = merge_pointclouds(args.pcd_dir, args.fix_csv)

    print(f"Saving merged point cloud to {args.output} ...")
    o3d.io.write_point_cloud(args.output, merged_pc)
    print("Done.")


if __name__ == "__main__":
    main()
