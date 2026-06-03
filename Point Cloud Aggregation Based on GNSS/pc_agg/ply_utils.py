import numpy as np
import open3d as o3d


def save_ply_xyz(filename: str, xyz: np.ndarray):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False)


def save_ply_xyzrgb(filename: str, xyz: np.ndarray, rgb_u8: np.ndarray):
    assert xyz.shape[0] == rgb_u8.shape[0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector((rgb_u8.astype(np.float32) / 255.0))
    o3d.io.write_point_cloud(filename, pcd, write_ascii=False)
