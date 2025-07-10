#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, PointCloud2, PointField, CameraInfo
from stereo_msgs.msg import DisparityImage
from visualization_msgs.msg import Marker, MarkerArray # NEW import
from cv_bridge import CvBridge
import numpy as np
import cv2
from scipy import ndimage
from sklearn.cluster import DBSCAN
import open3d as o3d
import struct
import matplotlib.pyplot as plt # Used for colormap in markers

# Constants for point cloud fields
_DATATYPES = {}
_DATATYPES[PointField.INT8] = ('b', 1)
_DATATYPES[PointField.UINT8] = ('B', 1)
_DATATYPES[PointField.INT16] = ('h', 2)
_DATATYPES[PointField.UINT16] = ('H', 2)
_DATATYPES[PointField.INT32] = ('i', 4)
_DATATYPES[PointField.UINT32] = ('I', 4)
_DATATYPES[PointField.FLOAT32] = ('f', 4)
_DATATYPES[PointField.FLOAT64] = ('d', 8)

class DisparityClusterer(Node):
    def __init__(self):
        super().__init__('disparity_clusterer')
        self.get_logger().info('DisparityClusterer node starting...')

        # Declare ROS Parameters
        self.declare_parameter('disparity_topic', '/disparity')
        self.declare_parameter('camera_info_topic', '/left/camera_info')
        self.declare_parameter('clustered_points_topic', '/clustered_points')
        self.declare_parameter('clustered_markers_topic', '/clustered_markers') # NEW parameter
        self.declare_parameter('debug_depth_image_topic', '/debug/depth_image')
        self.declare_parameter('bilateral_filter_d', 9)
        self.declare_parameter('bilateral_filter_sigma_color', 75.0)
        self.declare_parameter('bilateral_filter_sigma_space', 75.0)
        self.declare_parameter('fill_holes', True)
        self.declare_parameter('max_hole_size', 500)
        self.declare_parameter('dbscan_eps', 0.05)   # meters
        self.declare_parameter('dbscan_min_samples', 50)
        
        # Parameters for adaptive plane detection (ground/ceiling filtering)
        self.declare_parameter('max_plane_iterations', 1000)
        self.declare_parameter('plane_distance_threshold', 0.03) # meters (3 cm tolerance for points to be on plane)
        self.declare_parameter('plane_ransac_n', 3) # Minimum points to form a plane
        self.declare_parameter('plane_normal_threshold', 0.15) # Max deviation (cosine) from vertical/horizontal normal (e.g., 0.15 allows ~8 degrees off axis)
        self.declare_parameter('min_plane_points', 1000) # Minimum points for a valid plane to be considered ground/ceiling
        self.declare_parameter('ground_height_tolerance_m', 0.15) # Max vertical distance from lowest plane to be considered part of "ground"
        self.declare_parameter('ceiling_height_tolerance_m', 0.20) # Max vertical distance from highest plane to be considered part of "ceiling"

        # Camera Intrinsics (initialized to None, will be populated from CameraInfo)
        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None
        self.baseline = None 

        self.bridge = CvBridge()

        # Create subscribers
        self.disparity_subscription = self.create_subscription(
            DisparityImage,
            self.get_parameter('disparity_topic').get_parameter_value().string_value,
            self.disparity_callback,
            10
        )
        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            self.get_parameter('camera_info_topic').get_parameter_value().string_value,
            self.camera_info_callback,
            1 
        )

        # Create publishers
        self.clustered_points_publisher = self.create_publisher(
            PointCloud2,
            self.get_parameter('clustered_points_topic').get_parameter_value().string_value,
            10
        )
        self.clustered_markers_publisher = self.create_publisher( # NEW publisher
            MarkerArray,
            self.get_parameter('clustered_markers_topic').get_parameter_value().string_value,
            10
        )
        self.debug_depth_image_publisher = self.create_publisher(
            Image,
            self.get_parameter('debug_depth_image_topic').get_parameter_value().string_value,
            10
        )
        self.get_logger().info(f"Subscribing to '{self.get_parameter('disparity_topic').get_parameter_value().string_value}'")
        self.get_logger().info(f"Subscribing to '{self.get_parameter('camera_info_topic').get_parameter_value().string_value}'")
        self.get_logger().info(f"Publishing clustered points to '{self.get_parameter('clustered_points_topic').get_parameter_value().string_value}'")
        self.get_logger().info(f"Publishing clustered markers to '{self.get_parameter('clustered_markers_topic').get_parameter_value().string_value}'")
        self.get_logger().info(f"Publishing debug depth image to '{self.get_parameter('debug_depth_image_topic').get_parameter_value().string_value}'")

    def camera_info_callback(self, msg):
        # Extract intrinsic parameters from the rectified projection matrix (p)
        # p matrix is 3x4: [fx' 0 cx' Tx; 0 fy' cy' Ty; 0 0 1 0]
        # For the left camera, Tx and Ty are usually 0.
        # fx, fy, cx, cy are in p[0], p[5], p[2], p[6] respectively.
        
        # Fix: msg.p changed to msg.p as per previous error.
        if len(msg.p) == 12 and np.sum(msg.p) != 0:
            self.fx = msg.p[0]
            self.fy = msg.p[5]
            self.cx = msg.p[2]
            self.cy = msg.p[6]
            self.get_logger().info(f"Camera intrinsics updated from {self.get_parameter('camera_info_topic').get_parameter_value().string_value}: fx={self.fx:.2f}, fy={self.fy:.2f}, cx={self.cx:.2f}, cy={self.cy:.2f}", once=True)
        else:
            self.get_logger().warn(f"Received CameraInfo with invalid P matrix from {self.get_parameter('camera_info_topic').get_parameter_value().string_value}. Please ensure camera is calibrated and rectified.", once=True)
            # Fallback to default parameters if P matrix is invalid
            self.fx = 525.0
            self.fy = 525.0
            self.cx = 319.5
            self.cy = 239.5
            self.get_logger().warn("Using hardcoded intrinsic parameters. Ensure these are correct!", once=True)


    def disparity_callback(self, msg):
        self.get_logger().debug('Received disparity image.')
        
        # Check if camera intrinsics are available
        if self.fx is None or self.fy is None or self.cx is None or self.cy is None:
            self.get_logger().warn("Camera intrinsics not yet received. Skipping disparity processing.", once=True)
            return

        # Prefer focal length and baseline from DisparityImage message itself
        focal_length_from_msg = msg.f 
        baseline_from_msg = msg.t     
        
        if focal_length_from_msg > 0:
            focal_length = focal_length_from_msg
        else:
            self.get_logger().warn("DisparityImage.f is 0 or invalid, using fx from CameraInfo for focal length.", once=True)
            focal_length = self.fx

        if baseline_from_msg > 0:
            self.baseline = baseline_from_msg
        else:
            self.get_logger().warn("DisparityImage.T is 0 or invalid. For accurate baseline, ensure your stereo pipeline publishes it correctly, or calculate from right camera's P matrix.", once=True)
            if self.baseline is None:
                 self.baseline = 0.12 # Fallback to a hardcoded baseline if all else fails.
                 self.get_logger().warn("Using hardcoded baseline (0.12m). This might be inaccurate!", once=True)

        if self.baseline is None or self.baseline <= 0:
            self.get_logger().error("Baseline is not set or invalid. Cannot convert disparity to depth accurately. Please ensure correct camera info or disparity message source.", once=True)
            return
            
        try:
            disparity_img_cv = self.bridge.imgmsg_to_cv2(msg.image, desired_encoding="passthrough")
        except Exception as e:
            self.get_logger().error(f"Failed to convert disparity image: {e}")
            return

        invalid_disparity_mask = (disparity_img_cv <= 0)
        disparity_img_cv[invalid_disparity_mask] = np.nan

        valid_disparity_mask = ~np.isnan(disparity_img_cv)
        depth_map = np.full_like(disparity_img_cv, np.nan, dtype=np.float32)
        
        EPSILON = 1e-6 
        depth_map[valid_disparity_mask] = (focal_length * self.baseline) / (disparity_img_cv[valid_disparity_mask] + EPSILON)
        
        try:
            display_depth = np.nan_to_num(depth_map, nan=0)
            if np.max(display_depth) > 0:
                display_depth = (display_depth / np.max(display_depth) * 255).astype(np.uint8)
            else:
                display_depth = np.zeros_like(display_depth, dtype=np.uint8)

            debug_depth_msg = self.bridge.cv2_to_imgmsg(display_depth, encoding="mono8")
            debug_depth_msg.header = msg.header 
            self.debug_depth_image_publisher.publish(debug_depth_msg)
        except Exception as e:
            self.get_logger().warn(f"Failed to publish debug depth image: {e}")

        processed_depth_map = self.preprocess_depth_image(
            depth_map,
            bilateral_filter_d=self.get_parameter('bilateral_filter_d').get_parameter_value().integer_value,
            bilateral_filter_sigma_color=self.get_parameter('bilateral_filter_sigma_color').get_parameter_value().double_value,
            bilateral_filter_sigma_space=self.get_parameter('bilateral_filter_sigma_space').get_parameter_value().double_value,
            fill_holes=self.get_parameter('fill_holes').get_parameter_value().bool_value,
            max_hole_size=self.get_parameter('max_hole_size').get_parameter_value().integer_value
        )

        points_3d, pixel_uv = self.depth_to_point_cloud_with_indices(
            processed_depth_map, self.fx, self.fy, self.cx, self.cy
        )

        if points_3d.shape[0] == 0:
            self.get_logger().info("No valid 3D points generated after processing. Skipping clustering.")
            self.publish_clustered_markers(np.array([]), np.array([]), msg.header) # Clear previous markers
            self.publish_clustered_point_cloud(np.array([]), np.array([]), msg.header) # Clear old points
            return

        # --- NEW: Adaptive Ground and Ceiling Filtering ---
        ground_inliers, ceiling_inliers = self.identify_ground_ceiling(
            self.segment_planes_ransac(points_3d),
            points_3d
        )

        # Create a mask for points that are NOT ground/ceiling
        all_inliers_to_filter = set(ground_inliers).union(set(ceiling_inliers))
        
        # Filter points for clustering
        filtered_points_indices = [i for i in range(len(points_3d)) if i not in all_inliers_to_filter]
        
        if not filtered_points_indices:
            self.get_logger().info("All points filtered as ground/ceiling. No objects to cluster.")
            self.publish_clustered_markers(np.array([]), np.array([]), msg.header) # Clear old markers
            self.publish_clustered_point_cloud(np.array([]), np.array([]), msg.header) # Clear old points
            return

        filtered_points_3d = points_3d[filtered_points_indices]
        
        # --- End New Filtering ---

        # 5. Cluster 3D Point Cloud using DBSCAN on filtered points
        cluster_labels, num_clusters = self.cluster_point_cloud_dbscan(
            filtered_points_3d, 
            eps=self.get_parameter('dbscan_eps').get_parameter_value().double_value,
            min_samples=self.get_parameter('dbscan_min_samples').get_parameter_value().integer_value
        )
        self.get_logger().info(f"Frame {msg.header.stamp.sec}.{msg.header.stamp.nanosec // 1000000:03d}: Detected {num_clusters} clusters (excluding noise) after filtering ground/ceiling.")

        # 6. Publish Clustered Points as PointCloud2 (Optional, for general visualization)
        self.publish_clustered_point_cloud(
            filtered_points_3d, # Publish filtered points with labels
            cluster_labels,
            msg.header
        )
        
        # 7. Publish Clustered Markers (NEW)
        self.publish_clustered_markers(
            filtered_points_3d, # Use filtered points for marker centroids/scales
            cluster_labels,
            msg.header
        )

    def preprocess_depth_image(self, depth_image,
                               bilateral_filter_d, bilateral_filter_sigma_color,
                               bilateral_filter_sigma_space, fill_holes, max_hole_size):
        """
        Applies robust preprocessing to a depth image.
        """
        depth_image_for_filter = np.nan_to_num(depth_image, nan=0.0).astype(np.float32)

        filtered_depth = cv2.bilateralFilter(depth_image_for_filter,
                                              bilateral_filter_d,
                                              bilateral_filter_sigma_color,
                                              bilateral_filter_sigma_space)
        
        filtered_depth[np.isnan(depth_image)] = np.nan

        if fill_holes:
            valid_mask = ~np.isnan(filtered_depth)
            valid_mask_uint8 = (valid_mask * 255).astype(np.uint8)
            holes_mask = cv2.bitwise_not(valid_mask_uint8)

            num_labels, labels_conn_comp, stats, centroids = cv2.connectedComponentsWithStats(holes_mask, 8, cv2.CV_32S)
            filled_depth = np.copy(filtered_depth)

            for i in range(1, num_labels):
                if stats[i, cv2.CC_STAT_AREA] < max_hole_size:
                    hole_region_mask = (labels_conn_comp == i)
                    dilated_hole_mask = cv2.dilate(hole_region_mask.astype(np.uint8),
                                                   np.ones((3, 3), np.uint8), iterations=1)
                    surrounding_valid_pixels = (dilated_hole_mask > 0) & valid_mask

                    if np.any(surrounding_valid_pixels):
                        median_val = np.nanmedian(filled_depth[surrounding_valid_pixels])
                        filled_depth[hole_region_mask] = median_val
                    else:
                        filled_depth[hole_region_mask] = np.nan
            return filled_depth
        return filtered_depth

    def depth_to_point_cloud_with_indices(self, depth_image, fx, fy, cx, cy):
        """
        Converts a depth image to a 3D point cloud (x, y, z coordinates)
        and returns the original (u,v) pixel coordinates for each point.
        """
        H, W = depth_image.shape
        points_3d_with_uv = []
        for v in range(H):
            for u in range(W):
                z = depth_image[v, u]
                if np.isnan(z) or z <= 0:
                    continue

                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points_3d_with_uv.append([x, y, z, u, v])

        if not points_3d_with_uv:
            return np.array([]), np.array([])
        
        points_3d_with_uv_array = np.array(points_3d_with_uv)
        points_3d = points_3d_with_uv_array[:, :3]
        pixel_uv = points_3d_with_uv_array[:, 3:5].astype(int)
        return points_3d, pixel_uv

    def cluster_point_cloud_dbscan(self, points, eps, min_samples):
        """
        Clusters 3D points using DBSCAN.
        """
        if points.shape[0] == 0:
            return np.array([]), 0

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)
        unique_labels = np.unique(labels)
        num_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        return labels, num_clusters

    def segment_planes_ransac(self, points):
        """
        Segments dominant planes using Open3D's RANSAC.
        Returns a list of (plane_model, inlier_indices) for detected planes.
        """
        min_plane_ransac_n = self.get_parameter('plane_ransac_n').get_parameter_value().integer_value
        if len(points) < min_plane_ransac_n:
            return []

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        plane_models = []
        # Create a boolean mask for points not yet assigned to a plane
        remaining_mask = np.ones(len(points), dtype=bool)
        
        max_plane_iters = self.get_parameter('max_plane_iterations').get_parameter_value().integer_value
        dist_threshold = self.get_parameter('plane_distance_threshold').get_parameter_value().double_value
        min_plane_points_thresh = self.get_parameter('min_plane_points').get_parameter_value().integer_value

        for i in range(5): # Try to find up to 5 dominant planes
            current_points_indices = np.where(remaining_mask)[0]
            if len(current_points_indices) < min_plane_ransac_n:
                break 

            temp_pcd = pcd.select_by_index(current_points_indices)
            
            if len(temp_pcd.points) < min_plane_ransac_n:
                break

            try:
                model, inliers_relative_to_temp_pcd = temp_pcd.segment_plane(
                    distance_threshold=dist_threshold,
                    ransac_n=min_plane_ransac_n,
                    num_iterations=max_plane_iters
                )
            except RuntimeError: 
                self.get_logger().debug("Open3D segment_plane failed to find a plane (RuntimeError).")
                break

            if len(inliers_relative_to_temp_pcd) < min_plane_points_thresh:
                continue # Plane too small to be significant

            # Map inlier indices back to the original point cloud's indices
            original_inlier_indices = current_points_indices[inliers_relative_to_temp_pcd]
            plane_models.append((model, original_inlier_indices))
            
            # Update remaining_mask to exclude points found in this plane
            remaining_mask[original_inlier_indices] = False

        return plane_models

    def identify_ground_ceiling(self, plane_models, points_3d):
        """
        Identifies ground and ceiling planes from a list of detected planes.
        Assumes camera Y-axis is roughly vertical (Y down) in the camera frame.
        Ground will be the lowest major horizontal plane.
        Ceiling will be the highest major horizontal plane.
        """
        ground_plane_inliers = []
        ceiling_plane_inliers = []
        
        horizontal_planes = [] # Store (plane_model, inlier_indices, avg_y_coord)

        plane_normal_threshold = self.get_parameter('plane_normal_threshold').get_parameter_value().double_value
        
        # Camera frame: X-right, Y-down, Z-forward.
        # A horizontal plane (ground/ceiling) will have a normal vector roughly (0, N_y, 0).
        # We check abs(dot_product(normal, (0,1,0))) > (1.0 - threshold)

        for model, inliers_indices in plane_models:
            a, b, c, d = model # A*x + B*y + C*z + D = 0
            normal = np.array([a, b, c])
            normal = normal / np.linalg.norm(normal) # Normalize

            # Check if plane is horizontal (normal is mostly along Y-axis in camera frame)
            dot_product_y_axis = np.abs(np.dot(normal, [0, 1, 0])) 

            if dot_product_y_axis > (1.0 - plane_normal_threshold): # Close to horizontal
                # Calculate average Y-coordinate of inlier points for this plane
                if len(inliers_indices) > 0:
                    avg_y = np.mean(points_3d[inliers_indices, 1]) # Y-coordinate in camera frame
                    horizontal_planes.append((model, inliers_indices, avg_y))
                else:
                    self.get_logger().debug("Skipping horizontal plane with no inliers.")


        # Sort horizontal planes by their average Y-coordinate (height)
        # Lowest Y means higher up in the camera frame (Y is down), but closer to camera if positive.
        # To get actual "height" relative to camera, it's more about absolute Y value.
        # For filtering, we typically want the lowest Y (most 'down' from camera center) as ground,
        # and highest Y (most 'up' from camera center) as ceiling.
        # So sorting by Y directly should work.
        horizontal_planes.sort(key=lambda x: x[2]) 

        ground_height_tolerance = self.get_parameter('ground_height_tolerance_m').get_parameter_value().double_value
        ceiling_height_tolerance = self.get_parameter('ceiling_height_tolerance_m').get_parameter_value().double_value

        if horizontal_planes:
            # Ground candidate: lowest horizontal plane (most positive Y, if Y is down)
            ground_candidate_model, ground_candidate_inliers, ground_avg_y = horizontal_planes[0]
            
            # Combine all horizontal planes within the ground tolerance
            for model, inliers, avg_y in horizontal_planes:
                if np.abs(avg_y - ground_avg_y) < ground_height_tolerance:
                    ground_plane_inliers.extend(inliers)
            self.get_logger().info(f"Identified potential ground plane(s) with lowest average Y: {ground_avg_y:.3f}m. Points: {len(set(ground_plane_inliers))}")

            # Ceiling candidate: highest horizontal plane
            if len(horizontal_planes) > 1:
                ceiling_candidate_model, ceiling_candidate_inliers, ceiling_avg_y = horizontal_planes[-1]
                
                # Combine all horizontal planes within the ceiling tolerance
                # Ensure ceiling is actually "above" ground (smaller Y in Y-down coord system)
                if ceiling_avg_y < ground_avg_y - 0.1: # Small margin to ensure ceiling is not ground
                    for model, inliers, avg_y in horizontal_planes:
                        if np.abs(avg_y - ceiling_avg_y) < ceiling_height_tolerance:
                            ceiling_plane_inliers.extend(inliers)
                    self.get_logger().info(f"Identified potential ceiling plane(s) with highest average Y: {ceiling_avg_y:.3f}m. Points: {len(set(ceiling_plane_inliers))}")
                else:
                    self.get_logger().debug("No distinct ceiling plane found significantly above ground.")
            
        return list(set(ground_plane_inliers)), list(set(ceiling_plane_inliers)) # Use set to remove duplicates

    def publish_clustered_point_cloud(self, points_3d, labels, header):
        """
        Publishes the clustered 3D points as a PointCloud2 message.
        Each point will have X, Y, Z coordinates and a cluster_id field.
        """
        if len(points_3d) == 0:
            msg = PointCloud2() # Create an empty message to clear RViz
            msg.header = header
            msg.height = 1
            msg.width = 0
            msg.is_dense = True
            msg.point_step = 0
            msg.row_step = 0
            msg.data = []
            self.clustered_points_publisher.publish(msg)
            return

        msg = PointCloud2()
        msg.header = header
        msg.height = 1 
        msg.width = points_3d.shape[0]
        msg.is_dense = False 
        
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='cluster_id', offset=12, datatype=PointField.INT32, count=1),
        ]
        msg.fields = fields
        msg.point_step = sum([_DATATYPES[f.datatype][1] for f in fields]) 
        msg.row_step = msg.point_step * msg.width

        buffer = []
        fmt = struct.Struct('<fffi') 
        
        for i in range(points_3d.shape[0]):
            x, y, z = points_3d[i]
            cluster_id = int(labels[i]) 
            buffer.append(fmt.pack(x, y, z, cluster_id))
            
        msg.data = b''.join(buffer)
        
        self.clustered_points_publisher.publish(msg)

    def publish_clustered_markers(self, points_3d, labels, header):
        """
        Publishes the clustered objects as a MarkerArray.
        Each cluster is represented by a CUBE marker at its centroid.
        """
        marker_array_msg = MarkerArray()
        unique_labels = np.unique(labels)
        
        marker_id = 0
        for cluster_id in unique_labels:
            if cluster_id == -1: # Skip noise
                continue

            cluster_points_indices = np.where(labels == cluster_id)[0]
            cluster_points = points_3d[cluster_points_indices]

            min_dbscan_samples = self.get_parameter('dbscan_min_samples').get_parameter_value().integer_value
            if len(cluster_points) < min_dbscan_samples:
                continue 

            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)

            # Calculate bounding box (min/max for scale)
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            scale = max_bounds - min_bounds
            
            # Ensure minimum scale so marker is visible
            min_visible_scale = 0.05 # 5 cm
            scale[scale < min_visible_scale] = min_visible_scale

            marker = Marker()
            marker.header = header
            marker.ns = "clusters"
            marker.id = marker_id
            marker.type = Marker.CUBE 
            marker.action = Marker.ADD

            marker.pose.position.x = centroid[0]
            marker.pose.position.y = centroid[1]
            marker.pose.position.z = centroid[2]
            
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0

            marker.scale.x = scale[0]
            marker.scale.y = scale[1]
            marker.scale.z = scale[2]

            # Assign a color based on cluster_id (using a colormap)
            norm_id = (cluster_id % 20) / 19.0 # Use modulo for 'tab20' colormap, 20 distinct colors
            color_rgba = plt.cm.get_cmap('tab20')(norm_id) # Returns (R, G, B, A)
            marker.color.r = float(color_rgba[0])
            marker.color.g = float(color_rgba[1])
            marker.color.b = float(color_rgba[2])
            marker.color.a = 0.6 # Semi-transparent for better visibility

            marker_array_msg.markers.append(marker)
            marker_id += 1
            
        # If no clusters are found, publish a DELETEALL marker to clear previous markers in RViz
        if not marker_array_msg.markers:
            delete_marker = Marker()
            delete_marker.header = header
            delete_marker.ns = "clusters" # Ensure namespace matches
            delete_marker.action = Marker.DELETEALL
            marker_array_msg.markers.append(delete_marker)

        self.clustered_markers_publisher.publish(marker_array_msg)

def main(args=None):
    rclpy.init(args=args)
    disparity_clusterer = DisparityClusterer()
    rclpy.spin(disparity_clusterer)
    disparity_clusterer.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()