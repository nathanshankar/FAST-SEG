#ifndef DBSCAN_CLUSTERER_CLUSTER_TRACKER_HPP
#define DBSCAN_CLUSTERER_CLUSTER_TRACKER_HPP

#include <vector>
#include <random>
#include <chrono>
#include <limits>
#include <algorithm>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/common.h> // For getMinMax3D

#include "my_cluster_odom/Structures.hpp" // Include the common structures

namespace dbscan_clusterer {

class ClusterTracker {
public:
    ClusterTracker(rclcpp::Logger logger,
                   double kalman_pos_noise_q,
                   double kalman_vel_noise_q,
                   double kalman_min_vel_noise_q,
                   double kalman_vel_noise_q_decay_factor,
                   double kalman_measurement_noise_r,
                   double active_track_match_distance_threshold,
                   double lost_track_position_threshold,
                   double lost_track_dimension_threshold,
                   double reid_position_weight,
                   double reid_dimension_weight,
                   int n_stable_frames,
                   int n_missed_frames,
                   int max_lost_frames,
                   double reid_combined_threshold);

    /**
     * @brief Processes a new set of detected clusters, performing tracking and updating track states.
     * @param current_frame_time The timestamp of the current frame.
     * @param cloud The full point cloud for computing centroids/bounds.
     * @param current_detected_clusters The clusters detected in the current frame (indices into 'cloud').
     * @param active_tracks The vector of currently active tracks (updated in place).
     * @param recently_lost_tracks The vector of recently lost tracks (updated in place).
     * @return A vector of ClusterMemory objects representing the tracks that are considered "stable" for publishing.
     */
    std::vector<ClusterMemory> processClusters(
        rclcpp::Time current_frame_time,
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<std::vector<int>>& current_detected_clusters,
        std::vector<ClusterMemory>& active_tracks,
        std::vector<ClusterMemory>& recently_lost_tracks);

    // Public for access by DBSCANNode for initialization/reset if needed
    Eigen::Vector3f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices);
    std_msgs::msg::ColorRGBA generateRandomColor();
    
    // For initializing the first Cluster ID
    int getNextClusterId() { return cluster_id_counter_++; }

private:
    rclcpp::Logger logger_;

    // Kalman Filter Parameters
    double kalman_pos_noise_q_;
    double kalman_vel_noise_q_; // Initial/Max value
    double kalman_min_vel_noise_q_;
    double kalman_vel_noise_q_decay_factor_;
    double kalman_measurement_noise_r_;

    // Tracking Parameters
    double active_track_match_distance_threshold_;
    double lost_track_position_threshold_;
    double lost_track_dimension_threshold_;
    double reid_position_weight_;
    double reid_dimension_weight_;
    int n_stable_frames_;
    int n_missed_frames_;
    int max_lost_frames_;

    double reid_combined_threshold_;

    std::default_random_engine rand_gen_;
    int cluster_id_counter_ = 1; // Starts from 1

    Eigen::MatrixXd R_; // Measurement noise covariance (constant)
    Eigen::MatrixXd H_; // Measurement matrix (constant)

    /**
     * @brief Initializes the Kalman filter for a new cluster.
     * @param cluster_mem The ClusterMemory struct to initialize.
     * @param initial_position The initial measured position of the cluster.
     * @param current_time The current time.
     */
    void initKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& initial_position, rclcpp::Time current_time);

    /**
     * @brief Predicts the next state of the Kalman filter.
     * @param cluster_mem The ClusterMemory struct to predict.
     * @param dt The time difference since the last update.
     */
    void predictKalmanFilter(ClusterMemory& cluster_mem, double dt);

    /**
     * @brief Updates the Kalman filter with a new measurement.
     * Also updates the adaptive process noise for velocity.
     * @param cluster_mem The ClusterMemory struct to update.
     * @param measurement The new measured position.
     * @param current_time The current time.
     */
    void updateKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& measurement, rclcpp::Time current_time);
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_CLUSTER_TRACKER_HPP