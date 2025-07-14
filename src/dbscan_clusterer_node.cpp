#include "my_cluster_odom/DBSCANNode.hpp"

namespace dbscan_clusterer {

DBSCANNode::DBSCANNode() : Node("dbscan_node")
{
    // Declare parameters for DBSCAN and tracking
    this->declare_parameter<double>("dbscan_eps", 0.20);
    this->declare_parameter<int>("dbscan_min_pts", 10);
    this->declare_parameter<int>("n_stable_frames", 3);
    this->declare_parameter<int>("n_missed_frames", 20);
    this->declare_parameter<int>("max_lost_frames", 50);

    this->declare_parameter<double>("active_track_match_distance_threshold", 1.5);
    this->declare_parameter<double>("lost_track_position_threshold", 7.5);
    this->declare_parameter<double>("lost_track_dimension_threshold", 1.5);
    this->declare_parameter<double>("reid_position_weight", 1.0);
    this->declare_parameter<double>("reid_dimension_weight", 0.5);
    this->declare_parameter<double>("reid_combined_threshold", 7.0);
    this->declare_parameter<double>("voxel_leaf_size", 0.01);

    this->declare_parameter<double>("kalman_pos_noise_q", 0.1);
    this->declare_parameter<double>("kalman_vel_noise_q", 2.0);
    this->declare_parameter<double>("kalman_min_vel_noise_q", 0.05);
    this->declare_parameter<double>("kalman_vel_noise_q_decay_factor", 0.9);
    this->declare_parameter<double>("kalman_measurement_noise_r", 0.02);

    

    // Get parameters
    dbscan_eps_ = this->get_parameter("dbscan_eps").as_double();
    dbscan_min_pts_ = this->get_parameter("dbscan_min_pts").as_int();
    n_stable_frames_ = this->get_parameter("n_stable_frames").as_int();
    n_missed_frames_ = this->get_parameter("n_missed_frames").as_int();
    max_lost_frames_ = this->get_parameter("max_lost_frames").as_int();

    active_track_match_distance_threshold_ = this->get_parameter("active_track_match_distance_threshold").as_double();
    lost_track_position_threshold_ = this->get_parameter("lost_track_position_threshold").as_double();
    lost_track_dimension_threshold_ = this->get_parameter("lost_track_dimension_threshold").as_double();
    reid_position_weight_ = this->get_parameter("reid_position_weight").as_double();
    reid_dimension_weight_ = this->get_parameter("reid_dimension_weight").as_double();
    voxel_leaf_size_ = this->get_parameter("voxel_leaf_size").as_double();

    kalman_pos_noise_q_ = this->get_parameter("kalman_pos_noise_q").as_double();
    kalman_vel_noise_q_ = this->get_parameter("kalman_vel_noise_q").as_double();
    kalman_min_vel_noise_q_ = this->get_parameter("kalman_min_vel_noise_q").as_double();
    kalman_vel_noise_q_decay_factor_ = this->get_parameter("kalman_vel_noise_q_decay_factor").as_double();
    kalman_measurement_noise_r_ = this->get_parameter("kalman_measurement_noise_r").as_double();

    // Initialize helper classes
    cluster_refiner_ = std::make_unique<ClusterRefiner>(this->get_logger());
    cluster_tracker_ = std::make_unique<ClusterTracker>(
        this->get_logger(),
        kalman_pos_noise_q_, kalman_vel_noise_q_, kalman_min_vel_noise_q_, kalman_vel_noise_q_decay_factor_, kalman_measurement_noise_r_,
        active_track_match_distance_threshold_, lost_track_position_threshold_, lost_track_dimension_threshold_,
        reid_position_weight_, reid_dimension_weight_,
        n_stable_frames_, n_missed_frames_, max_lost_frames_, reid_combined_threshold_
    );

    // Subscribers and Publishers
    point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        "/points", 10, std::bind(&DBSCANNode::pointCloudCallback, this, std::placeholders::_1));

    cluster_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/clusters", 10);
    bounding_box_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/bounding_boxes", 10);
    cluster_data_publisher_ = this->create_publisher<my_cluster_odom::msg::ClusterArray>("/detected_clusters", 10);

    RCLCPP_INFO(this->get_logger(), "DBSCAN clustering node has started.");
    RCLCPP_INFO(this->get_logger(), "Parameters: eps=%.2f, minPts=%d", dbscan_eps_, dbscan_min_pts_);
    RCLCPP_INFO(this->get_logger(), "Tracking: N_STABLE_FRAMES=%d, N_MISSED_FRAMES=%d, MAX_LOST_FRAMES=%d", n_stable_frames_, n_missed_frames_, max_lost_frames_);
    RCLCPP_INFO(this->get_logger(), "Matching: Active_Dist=%.2f, Lost_Pos_Dist=%.2f, Lost_Dim_Diff=%.2f", active_track_match_distance_threshold_, lost_track_position_threshold_, lost_track_dimension_threshold_);
    RCLCPP_INFO(this->get_logger(), "Re-ID Weights: Pos=%.2f, Dim=%.2f", reid_position_weight_, reid_dimension_weight_);
    RCLCPP_INFO(this->get_logger(), "Kalman Filter (Adaptive Vel Noise): Pos_Q=%.2f, Initial_Vel_Q=%.2f, Min_Vel_Q=%.2f, Decay_Factor=%.2f, Meas_R=%.2f", 
                kalman_pos_noise_q_, kalman_vel_noise_q_, kalman_min_vel_noise_q_, kalman_vel_noise_q_decay_factor_, kalman_measurement_noise_r_);
}

void DBSCANNode::pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{   
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::fromROSMsg(*msg, *cloud);

    if (cloud->empty())
    {
        RCLCPP_WARN(this->get_logger(), "Received an empty point cloud.");
        return;
    }

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_y(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudFilters::applyAdaptiveYFilter(cloud, filtered_cloud_y, this->get_logger());
    if (filtered_cloud_y->empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_x(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudFilters::applyAdaptiveXFilter(filtered_cloud_y, filtered_cloud_x, this->get_logger());
    if (filtered_cloud_x->empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud_z(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudFilters::applyAdaptiveZFilter(filtered_cloud_x, filtered_cloud_z, this->get_logger());
    if (filtered_cloud_z->empty()) return;

    pcl::PointCloud<pcl::PointXYZ>::Ptr downsampled_cloud(new pcl::PointCloud<pcl::PointXYZ>());
    PointCloudFilters::applyVoxelGridFilter(filtered_cloud_z, downsampled_cloud, voxel_leaf_size_, this->get_logger());
    if (downsampled_cloud->empty()) return;

    RCLCPP_INFO(this->get_logger(), "Point cloud contains %lu points after filtering and downsampling.", downsampled_cloud->size());

    std::vector<std::vector<int>> clusters = cluster_refiner_->dbscanClustering(downsampled_cloud, static_cast<float>(dbscan_eps_), dbscan_min_pts_);
    std::vector<std::vector<int>> refined_clusters = cluster_refiner_->refineClusters(downsampled_cloud, clusters, dbscan_eps_, dbscan_min_pts_);
    
    // Process and get stable tracks
    std::vector<ClusterMemory> stable_tracks_for_pub = cluster_tracker_->processClusters(
        this->get_clock()->now(), downsampled_cloud, refined_clusters, active_tracks_, recently_lost_tracks_);

    // Publish visualizations and custom data
    publishClusterVisualizationsAndData(msg->header.frame_id, this->get_clock()->now(), 
                                        refined_clusters, downsampled_cloud, stable_tracks_for_pub,
                                        active_tracks_, recently_lost_tracks_);
}

void DBSCANNode::publishClusterVisualizationsAndData(
    const std::string& frame_id,
    rclcpp::Time current_frame_time,
    const std::vector<std::vector<int>>& detected_clusters, // Original detected clusters (pre-track association)
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& full_cloud, // The cloud after filtering/downsampling
    const std::vector<ClusterMemory>& stable_tracks, // Tracks verified as stable in this frame
    const std::vector<ClusterMemory>& active_tracks_current_state, // All active tracks (matched or missed)
    const std::vector<ClusterMemory>& recently_lost_tracks_current_state) // All recently lost tracks (re-identified or not)
{
    visualization_msgs::msg::MarkerArray marker_array;
    visualization_msgs::msg::MarkerArray bounding_box_array;
    my_cluster_odom::msg::ClusterArray cluster_data_array_msg;
    cluster_data_array_msg.header.stamp = current_frame_time;
    cluster_data_array_msg.header.frame_id = frame_id;

    // A map to quickly find the original cluster indices by track ID
    std::map<int, const std::vector<int>*> stable_track_cluster_map;
    // Iterate through current_detected_clusters to find the one associated with a stable track
    // This is a bit tricky since processClusters doesn't return the mapping.
    // We'll rely on the ClusterMemory.centroid matching the cluster_centroid.
    // A more robust way would be to pass the current_idx when creating/updating ClusterMemory in ClusterTracker.
    // For now, let's assume `stable_tracks` contain the Kalman-filtered centroid for publishing.

    // A more direct way: iterate over matched active/re-identified tracks during the tracking step
    // and store their original cluster indices, then pass that info here.
    // For simplicity of refactoring, we'll simplify the point cloud marker.
    // We will just publish bounding boxes and text for all active/lost tracks,
    // and for stable tracks, we'll try to find the actual point cloud.

    // Clear previous markers for all IDs
    // This assumes cluster_id_counter_ has been increasing, and we need to clear old IDs too.
    // A better approach for deleting markers is to publish a DELETEALL marker once,
    // or keep a set of all active IDs and delete only those that are no longer active/lost.
    // For simplicity, let's just make sure new markers overwrite old ones with the same ID.
    // To properly clear old markers, you'd need to send a DELETE command for IDs not present in the current frame.
    // For now, assume IDs are reused or the viz tool clears old markers.
    // Let's create a temporary marker to delete all previous clusters.
    visualization_msgs::msg::Marker delete_all_clusters_marker;
    delete_all_clusters_marker.header.frame_id = frame_id;
    delete_all_clusters_marker.header.stamp = current_frame_time;
    delete_all_clusters_marker.ns = "clusters";
    delete_all_clusters_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all_clusters_marker);

    visualization_msgs::msg::Marker delete_all_bboxes_marker;
    delete_all_bboxes_marker.header.frame_id = frame_id;
    delete_all_bboxes_marker.header.stamp = current_frame_time;
    delete_all_bboxes_marker.ns = "bounding_boxes";
    delete_all_bboxes_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    bounding_box_array.markers.push_back(delete_all_bboxes_marker);

    visualization_msgs::msg::Marker delete_all_text_marker;
    delete_all_text_marker.header.frame_id = frame_id;
    delete_all_text_marker.header.stamp = current_frame_time;
    delete_all_text_marker.ns = "cluster_ids";
    delete_all_text_marker.action = visualization_msgs::msg::Marker::DELETEALL;
    marker_array.markers.push_back(delete_all_text_marker);


    // Create a set of stable track IDs for quick lookup
    std::set<int> stable_track_ids;
    for(const auto& track : stable_tracks) {
        stable_track_ids.insert(track.id);
    }

    // Publish markers for stable tracks (full visibility, point clouds)
    for (const auto& cluster_mem : stable_tracks) {
        // Find the original detected cluster for point visualization based on proximity to Kalman centroid
        int original_cluster_idx = -1;
        double min_dist_to_detected_cluster = std::numeric_limits<double>::max();

        for(size_t i = 0; i < detected_clusters.size(); ++i) {
            Eigen::Vector3f detected_centroid = cluster_tracker_->computeCentroid(full_cloud, detected_clusters[i]);
            double dist = (cluster_mem.centroid - detected_centroid).norm();
            if (dist < min_dist_to_detected_cluster && dist < active_track_match_distance_threshold_) { // Reuse matching threshold
                min_dist_to_detected_cluster = dist;
                original_cluster_idx = static_cast<int>(i);
            }
        }

        if (original_cluster_idx != -1 && cluster_mem.missed_count == 0) { // Only show points if it was seen in this frame
            visualization_msgs::msg::Marker marker;
            marker.header.frame_id = frame_id;
            marker.header.stamp = current_frame_time;
            marker.ns = "clusters";
            marker.id = cluster_mem.id;
            marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
            marker.action = visualization_msgs::msg::Marker::ADD;
            marker.scale.x = marker.scale.y = marker.scale.z = 0.05;
            marker.color = cluster_mem.color;
            for (const auto &index : detected_clusters[original_cluster_idx])
            {
                geometry_msgs::msg::Point p;
                p.x = full_cloud->points[index].x;
                p.y = full_cloud->points[index].y;
                p.z = full_cloud->points[index].z;
                marker.points.push_back(p);
            }
            marker_array.markers.push_back(marker);
        }

        visualization_msgs::msg::Marker bbox_marker;
        bbox_marker.header.frame_id = frame_id;
        bbox_marker.header.stamp = current_frame_time;
        bbox_marker.ns = "bounding_boxes";
        bbox_marker.id = cluster_mem.id;
        bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
        bbox_marker.action = visualization_msgs::msg::Marker::ADD;
        bbox_marker.pose.position.x = cluster_mem.centroid.x();
        bbox_marker.pose.position.y = cluster_mem.centroid.y();
        bbox_marker.pose.position.z = cluster_mem.centroid.z();
        bbox_marker.scale.x = std::max(0.01f, cluster_mem.max_bounds.x() - cluster_mem.min_bounds.x());
        bbox_marker.scale.y = std::max(0.01f, cluster_mem.max_bounds.y() - cluster_mem.min_bounds.y());
        bbox_marker.scale.z = std::max(0.01f, cluster_mem.max_bounds.z() - cluster_mem.min_bounds.z());
        bbox_marker.color = cluster_mem.color;
        bbox_marker.color.a = (cluster_mem.missed_count == 0) ? 0.5f : 0.2f; // Faded if missed
        bounding_box_array.markers.push_back(bbox_marker);

        visualization_msgs::msg::Marker text_marker;
        text_marker.header.frame_id = frame_id;
        text_marker.header.stamp = current_frame_time;
        text_marker.ns = "cluster_ids";
        text_marker.id = cluster_mem.id;
        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
        text_marker.action = visualization_msgs::msg::Marker::ADD;
        text_marker.pose.position.x = cluster_mem.centroid.x();
        text_marker.pose.position.y = cluster_mem.centroid.y();
        text_marker.pose.position.z = cluster_mem.centroid.z() + 0.1f;
        text_marker.scale.z = 0.1;
        text_marker.color = cluster_mem.color;
        text_marker.color.a = (cluster_mem.missed_count == 0) ? 1.0f : 0.5f; // Faded text if missed
        text_marker.text = "ID: " + std::to_string(cluster_mem.id) +
                           "\nF: " + std::to_string(cluster_mem.frame_count) +
                           "\nM: " + std::to_string(cluster_mem.missed_count) +
                           ((cluster_mem.missed_count > 0 && cluster_mem.missed_count <= n_missed_frames_) ? " (Missed)" : "");
        marker_array.markers.push_back(text_marker);


        // Populate custom message for stable tracks
        my_cluster_odom::msg::Cluster cluster_msg;
        cluster_msg.id = cluster_mem.id;
        cluster_msg.centroid.x = cluster_mem.centroid.x();
        cluster_msg.centroid.y = cluster_mem.centroid.y();
        cluster_msg.centroid.z = cluster_mem.centroid.z();
        
        cluster_msg.dimensions.x = cluster_mem.max_bounds.x() - cluster_mem.min_bounds.x();
        cluster_msg.dimensions.y = cluster_mem.max_bounds.y() - cluster_mem.min_bounds.y();
        cluster_msg.dimensions.z = cluster_mem.max_bounds.z() - cluster_mem.min_bounds.z();

        cluster_msg.min_bounds.x = cluster_mem.min_bounds.x();
        cluster_msg.min_bounds.y = cluster_mem.min_bounds.y();
        cluster_msg.min_bounds.z = cluster_mem.min_bounds.z();

        cluster_msg.max_bounds.x = cluster_mem.max_bounds.x();
        cluster_msg.max_bounds.y = cluster_mem.max_bounds.y();
        cluster_msg.max_bounds.z = cluster_mem.max_bounds.z();

        cluster_msg.color = cluster_mem.color;
        cluster_msg.frame_count = cluster_mem.frame_count;
        cluster_msg.missed_count = cluster_mem.missed_count;

        for (int i = 0; i < 6; ++i) {
            cluster_msg.kalman_state[i] = static_cast<float>(cluster_mem.x_k(i));
        }
        for (int i = 0; i < 6; ++i) {
            for (int j = 0; j < 6; ++j) {
                cluster_msg.kalman_covariance[i * 6 + j] = static_cast<float>(cluster_mem.P_k(i, j));
            }
        }
        cluster_data_array_msg.clusters.push_back(cluster_msg);
    }

    // Publish markers for active tracks that are not yet stable (faded visibility)
    for (const auto& active_track : active_tracks_current_state) {
        if (stable_track_ids.count(active_track.id) == 0) { // If not already published as stable
            visualization_msgs::msg::Marker bbox_marker;
            bbox_marker.header.frame_id = frame_id;
            bbox_marker.header.stamp = current_frame_time;
            bbox_marker.ns = "bounding_boxes";
            bbox_marker.id = active_track.id;
            bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
            bbox_marker.action = visualization_msgs::msg::Marker::ADD;
            bbox_marker.pose.position.x = active_track.centroid.x();
            bbox_marker.pose.position.y = active_track.centroid.y();
            bbox_marker.pose.position.z = active_track.centroid.z();
            bbox_marker.scale.x = std::max(0.01f, active_track.max_bounds.x() - active_track.min_bounds.x());
            bbox_marker.scale.y = std::max(0.01f, active_track.max_bounds.y() - active_track.min_bounds.y());
            bbox_marker.scale.z = std::max(0.01f, active_track.max_bounds.z() - active_track.min_bounds.z());
            bbox_marker.color = active_track.color;
            bbox_marker.color.a = 0.3f; // Slightly faded for active but not stable
            bounding_box_array.markers.push_back(bbox_marker);

            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = frame_id;
            text_marker.header.stamp = current_frame_time;
            text_marker.ns = "cluster_ids";
            text_marker.id = active_track.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position.x = active_track.centroid.x();
            text_marker.pose.position.y = active_track.centroid.y();
            text_marker.pose.position.z = active_track.centroid.z() + 0.1f;
            text_marker.scale.z = 0.07; // Slightly smaller than stable text
            text_marker.color = active_track.color;
            text_marker.color.a = 0.7f; // Faded text
            text_marker.text = "ID: " + std::to_string(active_track.id) +
                               "\nF: " + std::to_string(active_track.frame_count);
            marker_array.markers.push_back(text_marker);
        }
    }


    // Publish very faded markers for recently lost tracks (only bounding box and text)
    // We iterate over `recently_lost_tracks_current_state` (the actual full list)
    // and filter for those that were NOT re-identified, or are still in the lost pool.
    // This is to avoid duplicating stable_tracks.
    // Also ensure not to duplicate active tracks that are not stable.
    std::set<int> active_and_stable_track_ids;
    for(const auto& track : stable_tracks) {
        active_and_stable_track_ids.insert(track.id);
    }
    for(const auto& track : active_tracks_current_state) {
        active_and_stable_track_ids.insert(track.id);
    }

    for (const auto& lost_track : recently_lost_tracks_current_state) {
        if (active_and_stable_track_ids.count(lost_track.id) == 0 && lost_track.frame_count >= n_stable_frames_) { // If it's not currently stable or active, and was stable enough to be visualized
            visualization_msgs::msg::Marker bbox_marker;
            bbox_marker.header.frame_id = frame_id;
            bbox_marker.header.stamp = current_frame_time;
            bbox_marker.ns = "bounding_boxes";
            bbox_marker.id = lost_track.id;
            bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
            bbox_marker.action = visualization_msgs::msg::Marker::ADD;
            bbox_marker.pose.position.x = lost_track.centroid.x();
            bbox_marker.pose.position.y = lost_track.centroid.y();
            bbox_marker.pose.position.z = lost_track.centroid.z();
            bbox_marker.scale.x = std::max(0.01f, lost_track.max_bounds.x() - lost_track.min_bounds.x());
            bbox_marker.scale.y = std::max(0.01f, lost_track.max_bounds.y() - lost_track.min_bounds.y());
            bbox_marker.scale.z = std::max(0.01f, lost_track.max_bounds.z() - lost_track.min_bounds.z());
            bbox_marker.color = lost_track.color;
            bbox_marker.color.a = 0.05f; // Very faded for lost tracks
            bounding_box_array.markers.push_back(bbox_marker);

            visualization_msgs::msg::Marker text_marker;
            text_marker.header.frame_id = frame_id;
            text_marker.header.stamp = current_frame_time;
            text_marker.ns = "cluster_ids";
            text_marker.id = lost_track.id;
            text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
            text_marker.action = visualization_msgs::msg::Marker::ADD;
            text_marker.pose.position.x = lost_track.centroid.x();
            text_marker.pose.position.y = lost_track.centroid.y();
            text_marker.pose.position.z = lost_track.centroid.z() + 0.1f;
            text_marker.scale.z = 0.05; // Smaller text for lost
            text_marker.color = lost_track.color;
            text_marker.color.a = 0.2f; // Faded text
            text_marker.text = "ID: " + std::to_string(lost_track.id) + " (Lost: " + std::to_string(lost_track.missed_count) + ")";
            marker_array.markers.push_back(text_marker);
        }
    }

    cluster_viz_publisher_->publish(marker_array);
    bounding_box_viz_publisher_->publish(bounding_box_array);
    cluster_data_publisher_->publish(cluster_data_array_msg);
}

} // namespace dbscan_clusterer

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<dbscan_clusterer::DBSCANNode>());
    rclcpp::shutdown();
    return 0;
}