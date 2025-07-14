#ifndef DBSCAN_CLUSTERER_STRUCTURES_HPP
#define DBSCAN_CLUSTERER_STRUCTURES_HPP

#include <Eigen/Dense>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/rclcpp.hpp> // For rclcpp::Time

namespace dbscan_clusterer {

// Internal structure to hold cluster memory, including Kalman filter state
struct ClusterMemory {
    Eigen::Vector3f centroid; // This will store the Kalman-filtered centroid
    std_msgs::msg::ColorRGBA color;
    int id;
    int frame_count;
    int missed_count;
    Eigen::Vector4f min_bounds; // Store min/max for better matching/tracking (from measurement)
    Eigen::Vector4f max_bounds; // Store min/max for better matching/tracking (from measurement)

    Eigen::VectorXd x_k; // Kalman filter state: [px, py, pz, vx, vy, vz]'
    Eigen::MatrixXd P_k; // Kalman filter covariance
    rclcpp::Time last_update_time;
    
    // Adaptive Kalman Filter parameter for this specific track
    double current_kalman_vel_noise_q; // Per-track adaptive velocity noise
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_STRUCTURES_HPP