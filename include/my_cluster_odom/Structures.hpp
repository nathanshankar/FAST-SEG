#ifndef DBSCAN_CLUSTERER_STRUCTURES_HPP
#define DBSCAN_CLUSTERER_STRUCTURES_HPP

#include <Eigen/Dense>
#include <std_msgs/msg/color_rgba.hpp>
#include <rclcpp/rclcpp.hpp> // For rclcpp::Time

namespace dbscan_clusterer {

// Structure to hold information about a newly detected cluster (measurement)
// Used for data association before it becomes a tracked ClusterMemory object.
// Moved this definition BEFORE ClusterMemory
struct ClusterCandidate {
    Eigen::Vector3f centroid;
    Eigen::Vector3f dimensions; // dx, dy, dz
    Eigen::Vector4f min_bounds;
    Eigen::Vector4f max_bounds;
    int original_cloud_idx; // Index in the `current_detected_clusters` vector
};

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

    // Default constructor
    ClusterMemory() : 
        centroid(Eigen::Vector3f::Zero()), 
        color(), 
        id(0), 
        frame_count(0), 
        missed_count(0), 
        min_bounds(Eigen::Vector4f::Zero()), 
        max_bounds(Eigen::Vector4f::Zero()), 
        x_k(6), // Initialize with a size, e.g., 6 for [px,py,pz,vx,vy,vz]
        P_k(6, 6), // Initialize with a size, e.g., 6x6
        last_update_time(rclcpp::Time(0, 0, RCL_ROS_TIME)), 
        current_kalman_vel_noise_q(0.0)
    {
        x_k.setZero();
        P_k.setZero();
    }

    // Copy constructor for deep copy
    ClusterMemory(const ClusterMemory& other)
    :   centroid(other.centroid),
        color(other.color),
        id(other.id),
        frame_count(other.frame_count),
        missed_count(other.missed_count),
        min_bounds(other.min_bounds),
        max_bounds(other.max_bounds),
        x_k(other.x_k), // Deep copy of Eigen::VectorXd
        P_k(other.P_k), // Deep copy of Eigen::MatrixXd
        last_update_time(other.last_update_time),
        current_kalman_vel_noise_q(other.current_kalman_vel_noise_q)
    {}

    // Copy assignment operator for deep copy
    ClusterMemory& operator=(const ClusterMemory& other) {
        if (this == &other) {
            return *this; // Handle self-assignment
        }

        centroid = other.centroid;
        color = other.color;
        id = other.id;
        frame_count = other.frame_count;
        missed_count = other.missed_count;
        min_bounds = other.min_bounds;
        max_bounds = other.max_bounds;
        x_k = other.x_k; // Deep copy
        P_k = other.P_k; // Deep copy
        last_update_time = other.last_update_time;
        current_kalman_vel_noise_q = other.current_kalman_vel_noise_q;

        return *this;
    }
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_STRUCTURES_HPP