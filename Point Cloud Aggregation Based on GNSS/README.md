# Point Cloud Aggregation Based on GNSS-INS Data and Moving Object Filtering

## 1. Introduction

This project builds a static, globally consistent 3D representation of an urban environment using LiDAR, GNSS-INS ego-motion, and camera data.

The main goals of the assignment are:

1. Aggregate frame-by-frame LiDAR measurements into a global coordinate system.
2. Remove measurements belonging to moving objects without using manual annotations.
3. Enhance the resulting static point cloud by assigning RGB color values from camera images.

All methods are implemented in Python and validated using the `nuScenes v1.0-mini` dataset.  
The mini split provides the same sensor configuration, calibration data, and ego-pose format as the full nuScenes dataset.

---

## 2. Dataset and Sensor Configuration

The nuScenes dataset contains synchronized multi-sensor data recorded in urban environments.

This project uses the following sensors and data sources:

- **LiDAR:** `LIDAR_TOP`
- **Cameras:** six RGB cameras covering a 360° field of view
- **Ego-motion:** GNSS-INS fused pose estimates from `ego_pose` records

All sensors are intrinsically and extrinsically calibrated, and timestamps are synchronized across modalities.

---

## 3. Coordinate Frames and Transformations

Three coordinate frames are used throughout the pipeline:

- **LiDAR frame:** raw point cloud coordinates from the LiDAR sensor
- **Ego vehicle frame:** vehicle-centric reference frame
- **Global frame:** world coordinate system provided by nuScenes

For each LiDAR point, the following transformation chain is applied:

```text
p_global = T_ego→global · T_lidar→ego · p_lidar
```

Rigid transformations are represented as `4 × 4` homogeneous transformation matrices constructed from:

- rotation quaternions
- translation vectors

---

## 4. Task 1 – Point Cloud Aggregation

### 4.1 Method

Each LiDAR sweep is processed independently and accumulated into a global point cloud.

The aggregation process is:

1. Load LiDAR points from disk.
2. Transform points from the LiDAR frame to the ego vehicle frame using calibrated sensor extrinsics.
3. Transform points from the ego vehicle frame to the global frame using ego-pose data.
4. Accumulate transformed points across all sweeps.

To support later temporal filtering, each point is stored together with:

- its originating sweep index
- its timestamp

### 4.2 Implementation

The aggregation is implemented in the following function:

```python
aggregate_sweeps_to_global(sweeps)
```

The output is an `AggregatedCloud` structure containing:

- `xyz_global`: global 3D coordinates
- `sweep_ids`: sweep index for each point
- `timestamps_us`: timestamp for each point

A lightweight voxel downsampling function is also provided for visualization and performance purposes.

### 4.3 Output

Task 1 produces:

- Aggregated global point cloud: `.ply`
- Downsampled preview for visualization and video generation

---

## 5. Task 2 – Moving Object Filtering

### 5.1 Motivation

Moving objects such as vehicles and pedestrians do not remain in the same spatial location across multiple LiDAR sweeps.

In contrast, static structures such as roads, buildings, poles, and signs are observed repeatedly in the same global regions.

This observation makes it possible to remove moving objects without using:

- semantic labels
- bounding boxes
- manual annotations

---

### 5.2 Temporal Voxel Consistency Filtering

The filtering strategy is based on temporal consistency.

The process is:

1. Discretize the global point cloud into voxels of fixed size.
2. For each voxel, count how many unique LiDAR sweeps contributed points to it.
3. Mark a voxel as static if it is observed in at least `K` different sweeps.
4. Keep only points belonging to static voxels.

This removes transient measurements caused by moving objects.

---

### 5.3 Implementation

The filtering is implemented in:

```python
temporal_voxel_static_mask(
    xyz,
    sweep_ids,
    voxel_size,
    min_unique_sweeps
)
```

An optional radius-based outlier removal step is applied to further suppress isolated noise.

---

### 5.4 Output

Task 2 produces:

- `static_filtered.ply` – static point cloud after temporal voxel filtering
- `static_filtered_clean.ply` – cleaned static point cloud after outlier removal

---

## 6. Task 3 – Point Cloud Colorization

### 6.1 Overview

To improve interpretability and visual realism, the static point cloud is enriched with RGB color values from camera images.

---

### 6.2 Projection and Color Assignment

For each camera frame, the following steps are performed:

1. Transform global points into the camera coordinate frame.
2. Project 3D points onto the image plane using camera intrinsics.
3. Discard points behind the camera or outside the image bounds.
4. Use a per-image z-buffer to resolve occlusions.
5. Assign RGB values to visible points by sampling image pixels.

Points that are not visible in any camera remain uncolored.

---

### 6.3 Implementation

Colorization is implemented in:

```python
colorize_points_from_cameras(
    xyz_global,
    camera_frames,
    nusc
)
```

Camera metadata and calibration are loaded using:

```python
load_scene_camera_frames(...)
```

---

### 6.4 Output

Task 3 produces:

- `static_colored.ply` – colorized static point cloud

---

## 7. Visualization and Video Generation

To visually demonstrate the result of each task, videos are generated by rendering point clouds using Open3D.

Because of platform limitations on macOS, a visible OpenGL window is used for rendering.  
The camera is smoothly rotated around the scene, and each rendered frame is saved as a PNG image.

The image sequences are then converted into MP4 videos using FFmpeg.

Generated videos include:

- **Task 1:** Aggregated point cloud
- **Task 2:** Static point cloud after moving object removal
- **Task 3:** Colorized static point cloud

---

## 8. Notes on Dataset Usage

All experiments were conducted on the `nuScenes mini` split.

This split contains the same:

- sensor setup
- calibration parameters
- ego-motion format

as the full nuScenes dataset.

Therefore, the implemented pipeline can be applied directly to the full nuScenes dataset without modification.

---

## 9. Output Files

| File | Description |
|---|---|
| `aggregated_global.ply` | Aggregated global point cloud |
| `static_filtered.ply` | Static point cloud after temporal voxel filtering |
| `static_filtered_clean.ply` | Cleaned static point cloud after outlier removal |
| `static_colored.ply` | Colorized static point cloud |
| Task videos | Rotating Open3D visualizations of the generated point clouds |

---

## 10. Conclusion

This project demonstrates a complete LiDAR-based mapping pipeline using real autonomous driving data.

By combining GNSS-INS ego-motion, temporal consistency filtering, and multi-view camera color projection, the system produces a clean and visually enriched static 3D representation of an urban environment.

The pipeline removes moving objects without relying on manual annotations and produces point clouds suitable for visualization, analysis, and further processing.
