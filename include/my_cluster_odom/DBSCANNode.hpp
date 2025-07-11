#ifndef DBSCAN_CLUSTERER_DBSCAN_NODE_HPP
#define DBSCAN_CLUSTERER_DBSCAN_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <random> // For random_engine in Color generation
#include <chrono> // For seeding random generator

// Custom message includes (replace 'my_cluster_odom' with your actual package name)
#include <my_cluster_odom/msg/cluster.hpp>
#include <my_cluster_odom/msg/cluster_array.hpp>

// Include our new classes and structures
#include "my_cluster_odom/Structures.hpp"
#include "my_cluster_odom/PointCloudFilters.hpp"
#include "my_cluster_odom/ClusterRefiner.hpp"
#include "my_cluster_odom/ClusterTracker.hpp"

namespace dbscan_clusterer {

class DBSCANNode : public rclcpp::Node
{
public:
    DBSCANNode(); // Constructor declaration

private:
    // Parameters
    double dbscan_eps_;
    int dbscan_min_pts_;
    int n_stable_frames_;
    int n_missed_frames_;
    int max_lost_frames_;
    double active_track_match_distance_threshold_;
    double lost_track_position_threshold_;
    double lost_track_dimension_threshold_;
    double reid_position_weight_;
    double reid_dimension_weight_;
    double kalman_pos_noise_q_;
    double kalman_vel_noise_q_;
    double kalman_min_vel_noise_q_;
    double kalman_vel_noise_q_decay_factor_;
    double kalman_measurement_noise_r_;

    // ROS 2 objects
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_viz_publisher_, bounding_box_viz_publisher_;
    rclcpp::Publisher<my_cluster_odom::msg::ClusterArray>::SharedPtr cluster_data_publisher_;

    // Instances of our new helper classes
    std::unique_ptr<ClusterRefiner> cluster_refiner_;
    std::unique_ptr<ClusterTracker> cluster_tracker_;

    // Track memory (managed by ClusterTracker, but stored here for passing)
    std::vector<ClusterMemory> active_tracks_;
    std::vector<ClusterMemory> recently_lost_tracks_;

    /**
     * @brief Callback function for incoming point cloud messages.
     * Orchestrates filtering, clustering, and tracking.
     * @param msg The shared pointer to the incoming PointCloud2 message.
     */
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg);

    /**
     * @brief Publishes visualization markers and custom cluster data messages.
     * @param frame_id The frame ID of the current point cloud.
     * @param current_frame_time The timestamp of the current frame.
     * @param detected_clusters The current frame's detected clusters (indices into original cloud).
     * @param full_cloud The full input point cloud for visualization purposes.
     * @param stable_tracks The vector of stable tracks to publish data and full markers for.
     * @param active_tracks_current_state The current state of active tracks (for faded visualization).
     * @param recently_lost_tracks_current_state The current state of recently lost tracks (for very faded visualization).
     */
    void publishClusterVisualizationsAndData(
        const std::string& frame_id,
        rclcpp::Time current_frame_time,
        const std::vector<std::vector<int>>& detected_clusters,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& full_cloud,
        const std::vector<ClusterMemory>& stable_tracks,
        const std::vector<ClusterMemory>& active_tracks_current_state,
        const std::vector<ClusterMemory>& recently_lost_tracks_current_state);
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_DBSCAN_NODE_HPP