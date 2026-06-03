# Merging 2D LiDAR Point Clouds Using GNSS

## 1. Method Overview
    
This project merges multiple 2D LiDAR point clouds into a single world-frame point cloud by using GNSS position data as the reference trajectory.

The workflow consists of:

1. Loading and converting GNSS data
2. Estimating vehicle orientation
3. Interpolating the vehicle pose for each LiDAR scan
4. Transforming each scan into the world frame
5. Merging all transformed scans
6. Visualizing the GNSS trajectory together with the merged point cloud

---

## 1.1 Loading GNSS Data

The `fix.csv` file is parsed to extract the following fields:

- Timestamp
- Latitude
- Longitude
- Altitude

The timestamp is computed as:

```text
timestamp = header_stamp_secs + header_stamp_nsecs * 1e-9
```

The GNSS positions are originally stored as GPS coordinates in the WGS84 coordinate system.

These coordinates are converted into a local Cartesian `XY` coordinate system using `pyproj` with a UTM projection.

After conversion, the GNSS timestamps are sorted and used as the reference trajectory for the vehicle.

---

## 1.2 Estimating Vehicle Orientation

The GNSS data does not directly contain orientation information.

Therefore, the vehicle yaw angle is estimated from the direction of forward motion:

```text
yaw ≈ arctan2(Δy, Δx)
```

This gives a reasonable approximation of the vehicle heading while the vehicle is moving.

---

## 1.3 Interpolating the Pose for Each LiDAR Scan

Each `.pcd` file is named using its timestamp.

For each LiDAR scan timestamp, the corresponding vehicle pose is estimated by linearly interpolating between the two nearest GNSS measurements.

The interpolated pose contains:

```text
x, y, yaw
```

where:

- `x` and `y` represent the vehicle position in the local Cartesian coordinate system
- `yaw` represents the estimated vehicle heading

---

## 1.4 Transforming and Merging LiDAR Points

Each LiDAR scan is loaded using Open3D.

The scan points are transformed from the local LiDAR frame into the world frame using a 2D SE(2) transformation:

```text
[x']   [cos(yaw)  -sin(yaw)   tx] [x]
[y'] = [sin(yaw)   cos(yaw)   ty] [y]
[z']   [   0          0        1] [z]
```

Where:

- `tx` and `ty` are the interpolated GNSS-based vehicle position
- `yaw` is the estimated vehicle orientation
- `x, y, z` are the original LiDAR point coordinates
- `x', y', z'` are the transformed world-frame coordinates

All transformed scans are concatenated into a single merged point cloud:

```text
merged_pc
```

---

## 2. Trajectory Visualization

To make the final result easier to inspect, the GNSS trajectory is added to the merged point cloud.

The visualization uses colors to distinguish between LiDAR points and trajectory points:

- LiDAR points are colored gray
- GNSS trajectory points are colored red

Both sets of points are merged into a single colored Open3D point cloud and saved as:

```text
merged_with_trajectory.ply
```

This allows visual inspection of the vehicle path inside the merged point cloud.

---

## 3. Saving the Output

The script saves the following output files:

| File | Description |
|---|---|
| `merged_with_trajectory.ply` | Colored point cloud containing both the merged LiDAR points and the GNSS trajectory |
| `merged.xyz` | Plain ASCII XYZ point cloud |
| User-specified output file | Optional custom output path for the merged XYZ point cloud |

---

## 4. Output Summary

The final output contains:

- A merged LiDAR point cloud in world coordinates
- A GNSS trajectory overlay for visual validation
- A colored `.ply` file for inspection in point cloud viewers
- A plain `.xyz` file for compatibility with simple point cloud processing tools
