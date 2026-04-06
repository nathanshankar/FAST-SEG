# FAST-SEG: Point Cloud Clustering and Tracking Package

`FAST-SEG` is a ROS 2 package centered around the `my_cluster_odom` package, designed for efficient point cloud segmentation using DBSCAN and robust cluster tracking with Kalman Filters. It provides a full pipeline from raw point cloud filtering to stable, tracked object clusters.



https://github.com/user-attachments/assets/c650a529-6b12-4f3f-8a95-87a96c517003


## Features

- **Adaptive Point Cloud Filtering**: Automatically adjusts filtering thresholds for X, Y, and Z axes to isolate objects of interest (e.g., removing ground, ceiling, or distant walls).
- **Voxel Grid Downsampling**: Efficiently reduces point cloud density for faster processing.
- **DBSCAN Clustering**: Robust, density-based spatial clustering to group points into discrete objects.
- **Kalman Filter Tracking**: Tracks clusters across frames with an adaptive velocity noise model to handle various object dynamics.
- **Cluster Re-Identification**: Re-associates lost tracks using both position and dimension-based re-identification logic.
- **Rich Visualizations**: Publishes cluster point clouds, bounding boxes, and ID labels for RVIZ.
- **Custom Data Output**: Provides a comprehensive `ClusterArray` message for downstream tasks like obstacle avoidance or SLAM.

## Dependencies

- **ROS 2** (Humble/Foxy/Galactic)
- **PCL** (Point Cloud Library)
- **Eigen3**
- **Standard ROS 2 Messages**: `sensor_msgs`, `visualization_msgs`, `geometry_msgs`, `nav_msgs`, `std_msgs`
- **Other**: `pcl_conversions`, `pcl_ros`, `tf2`, `tf2_ros`, `tf2_eigen`, `cv_bridge`

## Installation

Assuming you have a ROS 2 workspace:

```bash
cd ~/ros2_ws/src
# Clone or copy the FAST-SEG directory here
cd ~/ros2_ws
colcon build --packages-select my_cluster_odom
source install/setup.bash
```

## Usage

Run the main clustering and tracking node:

```bash
ros2 run my_cluster_odom dbscan_clusterer_node
```

### Subscribed Topics

- **`/points`** (`sensor_msgs/msg/PointCloud2`): The input point cloud to be processed.

### Published Topics

- **`/detected_clusters`** (`my_cluster_odom/msg/ClusterArray`): Detailed data for all stable, tracked clusters.
- **`/clusters`** (`visualization_msgs/msg/MarkerArray`): Point cloud visualizations of stable clusters.
- **`/bounding_boxes`** (`visualization_msgs/msg/MarkerArray`): Cube markers representing cluster bounding boxes (faded for active/lost tracks).
- **`/cluster_ids`** (`visualization_msgs/msg/MarkerArray`): Text labels showing ID, frame count, and status.

### Parameters

The node exposes several parameters to tune the clustering and tracking:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| `dbscan_eps` | 0.20 | DBSCAN search radius. |
| `dbscan_min_pts` | 10 | Minimum points to form a cluster. |
| `n_stable_frames` | 3 | Frames a cluster must be seen to be considered "stable". |
| `n_missed_frames` | 20 | Frames a cluster can be missed before it's moved to "recently lost". |
| `max_lost_frames` | 50 | Total frames a cluster can be lost before the track is deleted. |
| `voxel_leaf_size` | 0.01 | Leaf size for voxel grid downsampling. |
| `active_track_match_distance_threshold` | 1.5 | Max distance for matching active tracks. |
| `reid_combined_threshold` | 7.0 | Combined score threshold for Re-ID of lost tracks. |
| `kalman_pos_noise_q` | 0.1 | Kalman filter process noise for position. |
| `kalman_vel_noise_q` | 2.0 | Initial Kalman filter process noise for velocity. |

## Custom Messages

### `Cluster.msg`
Represents a single tracked object with:
- `id`: Unique identifier.
- `centroid`: Kalman-filtered 3D position.
- `dimensions`: Bounding box size (width, depth, height).
- `kalman_state`: 6D state vector `[px, py, pz, vx, vy, vz]`.
- `kalman_covariance`: 6x6 uncertainty matrix.
- `frame_count` & `missed_count`: Tracking status.

### `ClusterArray.msg`
A standard header followed by an array of `Cluster` messages.
