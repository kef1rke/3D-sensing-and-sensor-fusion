import argparse
import os
import sys
import time
import numpy as np
import open3d as o3d


def load_point_cloud(
    ply_path: str, voxel: float = 0.0, max_points: int = 0, seed: int = 0
) -> o3d.geometry.PointCloud:
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    pcd = o3d.io.read_point_cloud(ply_path)
    if pcd.is_empty():
        raise ValueError(f"Loaded point cloud is empty: {ply_path}")

    if voxel and voxel > 0:
        pcd = pcd.voxel_down_sample(voxel)

    if max_points and max_points > 0 and len(pcd.points) > max_points:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pcd.points), size=max_points, replace=False)
        pcd = pcd.select_by_index(idx)

    return pcd


def setup_camera(
    vis: o3d.visualization.Visualizer, pcd: o3d.geometry.PointCloud, zoom: float = 0.65
):
    """Fit camera to geometry and set a nice initial view."""
    ctr = vis.get_view_control()
    bbox = pcd.get_axis_aligned_bounding_box()
    center = bbox.get_center()

    ctr.set_lookat(center)
    ctr.set_front([0.0, -1.0, 0.25])
    ctr.set_up([0.0, 0.0, 1.0])
    ctr.set_zoom(zoom)


def render_frames_visible_window(
    pcd: o3d.geometry.PointCloud,
    outdir: str,
    n_frames: int = 300,
    width: int = 1280,
    height: int = 720,
    zoom: float = 0.65,
    rotate_step: float = 5.0,
    point_size: int = 2,
    sleep_s: float = 0.01,
):
    """
    Render frames by orbiting the camera and capturing screenshots.
    """
    os.makedirs(outdir, exist_ok=True)

    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name="Open3D Render", width=width, height=height, visible=True
    )

    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.background_color = np.asarray([0.0, 0.0, 0.0])

    vis.poll_events()
    vis.update_renderer()
    setup_camera(vis, pcd, zoom=zoom)

    # Warm-up renders
    for _ in range(5):
        vis.poll_events()
        vis.update_renderer()
        time.sleep(sleep_s)

    for i in range(n_frames):
        ctr = vis.get_view_control()
        ctr.rotate(rotate_step, 0.0)  # orbit horizontally

        vis.poll_events()
        vis.update_renderer()

        path = os.path.join(outdir, f"frame_{i:06d}.png")
        vis.capture_screen_image(path, do_render=True)

        # Small sleep helps macOS event loop stability
        time.sleep(sleep_s)

    vis.destroy_window()


def parse_args():
    ap = argparse.ArgumentParser(
        description="Render PLY to frames (macOS compatible, visible window)."
    )
    ap.add_argument("--ply", required=True, help="Input .ply point cloud file")
    ap.add_argument("--out", required=True, help="Output directory for frames (PNG)")
    ap.add_argument("--n_frames", type=int, default=300)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30, help="FPS for ffmpeg hint")
    ap.add_argument(
        "--voxel",
        type=float,
        default=0.0,
        help="Optional voxel downsample (0 disables)",
    )
    ap.add_argument(
        "--max_points",
        type=int,
        default=0,
        help="Optional random subsample cap (0 disables)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zoom", type=float, default=0.65)
    ap.add_argument("--rotate_step", type=float, default=5.0)
    ap.add_argument("--point_size", type=int, default=2)
    ap.add_argument(
        "--sleep",
        type=float,
        default=0.01,
        help="Sleep per frame to improve stability on macOS",
    )
    return ap.parse_args()


def main():
    args = parse_args()
    pcd = load_point_cloud(
        args.ply, voxel=args.voxel, max_points=args.max_points, seed=args.seed
    )

    render_frames_visible_window(
        pcd=pcd,
        outdir=args.out,
        n_frames=args.n_frames,
        width=args.width,
        height=args.height,
        zoom=args.zoom,
        rotate_step=args.rotate_step,
        point_size=args.point_size,
        sleep_s=args.sleep,
    )

    print("\nFrames rendered successfully.")
    print("To create a video with ffmpeg, run:\n")
    print(
        f'  ffmpeg -r {args.fps} -i "{args.out}/frame_%06d.png" -c:v libx264 -pix_fmt yuv420p output.mp4\n'
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
