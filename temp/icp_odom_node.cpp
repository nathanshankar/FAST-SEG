#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp> // Added for publishing trajectory
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp> // Added for individual poses in the path
#include <sensor_msgs/msg/imu.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <tf2_ros/transform_broadcaster.h>
#include <Eigen/Dense>
#include <random>
#include <algorithm>
#include <cmath>

// Custom message includes (replace 'my_cluster_odom' with your actual package name)
#include <my_cluster_odom/msg/cluster.hpp>
#include <my_cluster_odom/msg/cluster_array.hpp>

class OdometryNode : public rclcpp::Node
{
public:
    OdometryNode() : Node("odometry_node")
    {
        // Odometry specific parameters
        this->declare_parameter<double>("static_object_velocity_threshold", 10.0);
        this->declare_parameter<int>("min_static_clusters_for_odom", 2);
        this->declare_parameter<int>("min_frames_for_odom_cluster", 5);

        // RANSAC parameters for odometry
        this->declare_parameter<int>("ransac_iterations", 100);
        this->declare_parameter<double>("ransac_inlier_threshold", 0.05);
        this->declare_parameter<int>("min_points_for_ransac_model", 2);

        // EMA Filter parameters
        this->declare_parameter<double>("odom_filter_alpha_translation", 0.7);
        this->declare_parameter<double>("odom_filter_alpha_rotation", 0.7);

        // IMU Integration parameters
        this->declare_parameter<double>("imu_yaw_weight", 0.5);
        this->declare_parameter<double>("imu_data_timeout_sec", 0.1);

        static_object_velocity_threshold_ = this->get_parameter("static_object_velocity_threshold").as_double();
        min_static_clusters_for_odom_ = this->get_parameter("min_static_clusters_for_odom").as_int();
        min_frames_for_odom_cluster_ = this->get_parameter("min_frames_for_odom_cluster").as_int();
        ransac_iterations_ = this->get_parameter("ransac_iterations").as_int();
        ransac_inlier_threshold_ = this->get_parameter("ransac_inlier_threshold").as_double();
        min_points_for_ransac_model_ = this->get_parameter("min_points_for_ransac_model").as_int();

        odom_filter_alpha_translation_ = this->get_parameter("odom_filter_alpha_translation").as_double();
        odom_filter_alpha_rotation_ = this->get_parameter("odom_filter_alpha_rotation").as_double();

        imu_yaw_weight_ = this->get_parameter("imu_yaw_weight").as_double();
        imu_data_timeout_sec_ = this->get_parameter("imu_data_timeout_sec").as_double();


        // Subscribers and Publishers
        cluster_data_subscriber_ = this->create_subscription<my_cluster_odom::msg::ClusterArray>(
            "/detected_clusters", 10, std::bind(&OdometryNode::clusterDataCallback, this, std::placeholders::_1));

        // IMU Subscriber
        imu_subscriber_ = this->create_subscription<sensor_msgs::msg::Imu>(
            "/imu", 10, std::bind(&OdometryNode::imuCallback, this, std::placeholders::_1));

        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry", 10);
        path_publisher_ = this->create_publisher<nav_msgs::msg::Path>("/trajectory", 10); // Path publisher
        tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*this);

        rand_gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());

        // Initialize current robot pose to origin
        current_robot_pose_.pose.pose.position.x = 0.0;
        current_robot_pose_.pose.pose.position.y = 0.0;
        current_robot_pose_.pose.pose.position.z = 0.0;
        current_robot_pose_.pose.pose.orientation.w = 1.0; // Identity quaternion
        current_robot_pose_.pose.pose.orientation.x = 0.0;
        current_robot_pose_.pose.pose.orientation.y = 0.0;
        current_robot_pose_.pose.pose.orientation.z = 0.0;
        last_odom_update_time_ = this->get_clock()->now();

        // Initialize filtered deltas
        filtered_delta_translation_global_ = Eigen::Vector3d::Zero();
        filtered_yaw_delta_ = 0.0;
        latest_imu_msg_ = nullptr; // Initialize IMU message pointer to null

        // Initialize path header
        robot_path_.header.frame_id = "odom";


        RCLCPP_INFO(this->get_logger(), "Odometry estimation node has started.");
        RCLCPP_INFO(this->get_logger(), "Odometry: Static_Vel_Thresh=%.2f, Min_Static_Clusters=%d, Min_Frames_For_Odom_Cluster=%d",
                    static_object_velocity_threshold_, min_static_clusters_for_odom_, min_frames_for_odom_cluster_);
        RCLCPP_INFO(this->get_logger(), "RANSAC: Iterations=%d, Inlier_Thresh=%.3f, Min_Points=%d", ransac_iterations_, ransac_inlier_threshold_, min_points_for_ransac_model_);
        RCLCPP_INFO(this->get_logger(), "Filter: Alpha_Translation=%.2f, Alpha_Rotation=%.2f", odom_filter_alpha_translation_, odom_filter_alpha_rotation_);
        RCLCPP_INFO(this->get_logger(), "IMU Fusion: Yaw_Weight=%.2f, Data_Timeout=%.2f sec", imu_yaw_weight_, imu_data_timeout_sec_);
    }

private:
    // Parameters
    double static_object_velocity_threshold_;
    int min_static_clusters_for_odom_;
    int min_frames_for_odom_cluster_;
    int ransac_iterations_;
    double ransac_inlier_threshold_;
    int min_points_for_ransac_model_;
    double odom_filter_alpha_translation_;
    double odom_filter_alpha_rotation_;
    double imu_yaw_weight_;
    double imu_data_timeout_sec_;

    rclcpp::Subscription<my_cluster_odom::msg::ClusterArray>::SharedPtr cluster_data_subscriber_;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr imu_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_publisher_; // Path publisher
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    geometry_msgs::msg::PoseWithCovarianceStamped current_robot_pose_; // Global pose of the robot
    rclcpp::Time last_odom_update_time_; // To calculate dt between odometry updates

    std::default_random_engine rand_gen_;

    // Store previous cluster data for odometry calculation
    std::vector<my_cluster_odom::msg::Cluster> previous_frame_clusters_;
    rclcpp::Time previous_frame_time_;

    // Store latest IMU data
    sensor_msgs::msg::Imu::SharedPtr latest_imu_msg_;

    // Variables for the EMA filter
    Eigen::Vector3d filtered_delta_translation_global_;
    double filtered_yaw_delta_;

    nav_msgs::msg::Path robot_path_; // Trajectory path

    /**
     * @brief Callback function for incoming IMU messages.
     * Stores the latest IMU data.
     * @param msg The shared pointer to the incoming Imu message.
     */
    void imuCallback(const sensor_msgs::msg::Imu::SharedPtr msg)
    {
        latest_imu_msg_ = msg;
        RCLCPP_DEBUG(this->get_logger(), "IMU: Received IMU data with timestamp: %f", msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);
    }

    /**
     * @brief Callback function for incoming ClusterArray messages.
     * @param msg The shared pointer to the incoming ClusterArray message.
     */
    void clusterDataCallback(const my_cluster_odom::msg::ClusterArray::SharedPtr msg)
    {
        RCLCPP_DEBUG(this->get_logger(), "Odometry: Received ClusterArray with timestamp: %f", msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);

        if (previous_frame_clusters_.empty()) {
            RCLCPP_INFO(this->get_logger(), "First cluster array received. Initializing previous frame data.");
            previous_frame_clusters_ = msg->clusters;
            previous_frame_time_ = msg->header.stamp;
            return;
        }

        calculateOdometry(msg->header.stamp, msg->header.frame_id, msg->clusters);

        // Update previous frame data for the next iteration
        previous_frame_clusters_ = msg->clusters;
        previous_frame_time_ = msg->header.stamp;
    }

    /**
     * @brief Calculates the odometry of the robot based on static clusters.
     * This function assumes a planar motion model (X-Z plane, rotation around Y-axis)
     * for odometry estimation.
     * @param current_frame_time The timestamp of the current frame.
     * @param frame_id The frame ID of the current data (e.g., "camera_infra1_optical_frame").
     * @param current_clusters The clusters detected in the current frame.
     */
    void calculateOdometry(rclcpp::Time current_frame_time, const std::string& frame_id, const std::vector<my_cluster_odom::msg::Cluster>& current_clusters)
    {
        if (previous_frame_clusters_.empty() || current_clusters.empty()) {
            RCLCPP_WARN(this->get_logger(), "Not enough cluster data to estimate odometry. Previous: %zu, Current: %zu.",
                        previous_frame_clusters_.size(), current_clusters.size());
            return;
        }

        // 1. Identify static candidates from previous_frame_clusters_ based on their predicted velocity
        //    AND filter by frame_count and missed_count.
        //    THEN, select the two clusters with the highest frame_count.
        std::vector<const my_cluster_odom::msg::Cluster*> potential_static_clusters;
        for (const auto& cluster : previous_frame_clusters_) {
            // Use the predicted velocity from Kalman filter (x_k[3], x_k[4], x_k[5])
            Eigen::Vector3d velocity(cluster.kalman_state[3], cluster.kalman_state[4], cluster.kalman_state[5]);

            // Apply new criteria: stable (frame_count) and not missed
            if (velocity.norm() < static_object_velocity_threshold_ &&
                cluster.frame_count >= min_frames_for_odom_cluster_ &&
                cluster.missed_count == 0) // Only use clusters that were seen in the previous frame
            {
                potential_static_clusters.push_back(&cluster);
            } else {
                RCLCPP_DEBUG(this->get_logger(), "Skipping previous cluster ID %d for odometry (vel: %.2f, frames: %d, missed: %d).",
                             cluster.id, velocity.norm(), cluster.frame_count, cluster.missed_count);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Found %zu potential static clusters after initial filtering.", potential_static_clusters.size());

        // Sort by frame_count in descending order to get the most stable clusters
        std::sort(potential_static_clusters.begin(), potential_static_clusters.end(),
                  [](const my_cluster_odom::msg::Cluster* a, const my_cluster_odom::msg::Cluster* b) {
                      return a->frame_count > b->frame_count;
                  });

        // Select the top 2 clusters (or fewer if not enough are available)
        std::vector<const my_cluster_odom::msg::Cluster*> selected_static_clusters;
        for (int i = 0; i < std::min(2, static_cast<int>(potential_static_clusters.size())); ++i) {
            selected_static_clusters.push_back(potential_static_clusters[i]);
            RCLCPP_INFO(this->get_logger(), "Selected static cluster ID %d with frame_count %d for odometry.",
                         potential_static_clusters[i]->id, potential_static_clusters[i]->frame_count);
        }

        // Now, proceed with odometry calculation using only `selected_static_clusters`
        if (selected_static_clusters.size() < static_cast<size_t>(min_static_clusters_for_odom_)) {
            RCLCPP_WARN(this->get_logger(), "Not enough selected static clusters (%zu) to estimate odometry. Need at least %d.",
                        selected_static_clusters.size(), min_static_clusters_for_odom_);
            return;
        }

        // 2. Match selected static clusters from previous frame to current frame clusters
        std::vector<Eigen::Vector2d> prev_2d_points_full; // Previous (predicted) X-Z positions of static clusters
        std::vector<Eigen::Vector2d> curr_2d_points_full; // Current (measured) X-Z positions of static clusters

        for (const auto& prev_static_cluster : selected_static_clusters) {
            // Use predicted position from previous frame for prev_points
            // The Kalman state is [px, py, pz, vx, vy, vz]. We use px, pz for 2D motion.
            Eigen::Vector3d predicted_pos_prev(prev_static_cluster->kalman_state[0],
                                               prev_static_cluster->kalman_state[1],
                                               prev_static_cluster->kalman_state[2]);

            // Find the corresponding cluster in the current frame by ID
            bool found_match = false;
            for (const auto& current_cluster : current_clusters) {
                // Only consider current clusters that are NOT missed (i.e., actually observed in this frame)
                if (current_cluster.id == prev_static_cluster->id && current_cluster.missed_count == 0) {
                    // We extract X and Z components for 2D RANSAC
                    prev_2d_points_full.push_back(Eigen::Vector2d(predicted_pos_prev.x(), predicted_pos_prev.z())); // X, Z
                    curr_2d_points_full.push_back(Eigen::Vector2d(current_cluster.centroid.x, current_cluster.centroid.z)); // X, Z
                    found_match = true;
                    break;
                }
            }
            if (!found_match) {
                RCLCPP_DEBUG(this->get_logger(), "Selected static cluster ID %d from previous frame was not matched in current frame (or was missed). Skipping for odometry.", prev_static_cluster->id);
            }
        }

        RCLCPP_INFO(this->get_logger(), "Found %zu matched and stable static points for RANSAC.", prev_2d_points_full.size());

        if (prev_2d_points_full.size() < static_cast<size_t>(min_points_for_ransac_model_)) {
            RCLCPP_WARN(this->get_logger(), "Not enough matched and stable static points (%zu) for RANSAC odometry. Need at least %d.",
                        prev_2d_points_full.size(), min_points_for_ransac_model_);
            return;
        }

        // --- Odometry Debug (Leave these, they are useful!) ---
        RCLCPP_INFO(this->get_logger(), "Number of prev_2d_points_full (matched static for RANSAC): %zu", prev_2d_points_full.size());

        // These messages will only appear if the node's logger level is DEBUG or lower.
        for (size_t k = 0; k < prev_2d_points_full.size(); ++k) {
            RCLCPP_DEBUG(this->get_logger(), "  Point %zu: Prev=(%.4f, %.4f), Curr=(%.4f, %.4f)",
                        k, prev_2d_points_full[k].x(), prev_2d_points_full[k].y(), // .y() here is the Z coordinate in 3D
                        curr_2d_points_full[k].x(), curr_2d_points_full[k].y()); // .y() here is the Z coordinate in 3D
        }
        // --- End Debug ---

        // 3. RANSAC to estimate 2D transformation (translation in X, Z + rotation around Y)
        Eigen::Vector2d best_translation_2d = Eigen::Vector2d::Zero();
        Eigen::Matrix2d best_rotation_2d = Eigen::Matrix2d::Identity();
        int max_inliers = 0;

        std::uniform_int_distribution<> distrib(0, prev_2d_points_full.size() - 1);

        for (int i = 0; i < ransac_iterations_; ++i) {
            std::vector<int> sample_indices;
            // Ensure we get unique sample indices
            while(sample_indices.size() < static_cast<size_t>(min_points_for_ransac_model_)) {
                int idx = distrib(rand_gen_);
                if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                    sample_indices.push_back(idx);
                }
            }

            // Centroids of sampled points
            Eigen::Vector2d p_prev_centroid_2d = Eigen::Vector2d::Zero();
            Eigen::Vector2d p_curr_centroid_2d = Eigen::Vector2d::Zero();

            for (int idx : sample_indices) {
                p_prev_centroid_2d += prev_2d_points_full[idx];
                p_curr_centroid_2d += curr_2d_points_full[idx];
            }
            p_prev_centroid_2d /= static_cast<double>(sample_indices.size());
            p_curr_centroid_2d /= static_cast<double>(sample_indices.size());

            Eigen::Matrix2d H_2d = Eigen::Matrix2d::Zero();
            for (int idx : sample_indices) {
                Eigen::Vector2d p_prev_centered_2d = prev_2d_points_full[idx] - p_prev_centroid_2d;
                Eigen::Vector2d p_curr_centered_2d = curr_2d_points_full[idx] - p_curr_centroid_2d;
                H_2d += p_prev_centered_2d * p_curr_centered_2d.transpose();
            }

            // SVD to find 2D rotation
            Eigen::JacobiSVD<Eigen::Matrix2d> svd_2d(H_2d, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2d R_est_2d = svd_2d.matrixV() * svd_2d.matrixU().transpose();

            // Check for reflection (2D determinant) and correct it
            if (R_est_2d.determinant() < 0) {
                Eigen::Matrix2d V_2d = svd_2d.matrixV();
                V_2d.col(1) *= -1; // Flip the second column of V
                R_est_2d = V_2d * svd_2d.matrixU().transpose();
            }

            Eigen::Vector2d t_est_2d = p_curr_centroid_2d - R_est_2d * p_prev_centroid_2d;

            // Calculate current inliers
            int current_inliers = 0;
            for (size_t k = 0; k < prev_2d_points_full.size(); ++k) {
                Eigen::Vector2d transformed_point_2d = R_est_2d * prev_2d_points_full[k] + t_est_2d;
                double distance = (transformed_point_2d - curr_2d_points_full[k]).norm();
                if (distance < ransac_inlier_threshold_) {
                    current_inliers++;
                }
            }

            if (current_inliers > max_inliers) {
                max_inliers = current_inliers;
                best_translation_2d = t_est_2d;
                best_rotation_2d = R_est_2d;
            }
        }

        RCLCPP_INFO(this->get_logger(), "RANSAC completed. Max inliers: %d, required: %d.",
                    max_inliers, min_points_for_ransac_model_);

        if (max_inliers < min_points_for_ransac_model_) {
            RCLCPP_WARN(this->get_logger(), "RANSAC failed to find enough inliers for odometry. Max inliers: %d, required: %d. Resetting filtered deltas.",
                        max_inliers, min_points_for_ransac_model_);
            // If RANSAC fails, reset filtered deltas to zero to prevent accumulation of bad data
            filtered_delta_translation_global_ = Eigen::Vector3d::Zero();
            filtered_yaw_delta_ = 0.0;
            return;
        }

        // Convert 2D transformation to 3D for odometry message
        // Assuming camera frame: X=Right, Y=Down, Z=Forward
        // Desired base_link/odom frame: X=Forward, Y=Left, Z=Up
        //
        // RANSAC's best_translation_2d.x() is movement along camera's X (Right)
        // RANSAC's best_translation_2d.y() is movement along camera's Z (Forward)
        //
        // To map to base_link:
        // Base_link X (Forward) = Camera Z
        // Base_link Y (Left)    = -Camera X
        // Base_link Z (Up)      = 0 (planar motion)
        Eigen::Vector3d best_translation_3d(best_translation_2d.y(),    // Camera Z (Forward) -> Base_link X (Forward)
                                            -best_translation_2d.x(),   // -Camera X (Right) -> Base_link Y (Left)
                                            0.0);                       // No Z motion in base_link (planar)

        // The 2D rotation matrix corresponds to a yaw rotation around the camera's Y-axis (Down).
        // To get the angle, we can use atan2 on the elements of the 2x2 rotation matrix.
        // R_2D = [cos(theta) -sin(theta); sin(theta) cos(theta)]
        // Here, the 2x2 matrix represents rotation in Camera X-Z plane.
        // Element (0,0) is cos(theta), (0,1) is -sin(theta), (1,0) is sin(theta), (1,1) is cos(theta).
        double pitch_delta_camera_yaw = atan2(best_rotation_2d(1,0), best_rotation_2d(0,0));

        // pitch_delta_camera_yaw is the rotation around camera's Y (which is 'yaw' in the camera's frame).
        // To map this to base_link's Z (yaw in ROS convention), we need to negate it.
        double yaw_delta_ransac_base_link = -pitch_delta_camera_yaw;

        double combined_yaw_delta = yaw_delta_ransac_base_link; // Start with RANSAC estimate

        // --- IMU Integration ---
        // Check if IMU data is available and fresh enough
        if (latest_imu_msg_ && (current_frame_time - rclcpp::Time(latest_imu_msg_->header.stamp)).seconds() < imu_data_timeout_sec_) {
            // IMU angular_velocity.y is rotation around camera's Y-axis (Down).
            // To map this to base_link's Z-axis (Up), we negate it.
            double imu_yaw_rate_base_link = -latest_imu_msg_->angular_velocity.y;
            double dt_odom_for_imu = (current_frame_time - last_odom_update_time_).seconds();

            if (dt_odom_for_imu > 0) {
                double imu_yaw_delta_base_link = imu_yaw_rate_base_link * dt_odom_for_imu;
                // Combine RANSAC yaw_delta with IMU yaw_delta using a weighted average
                combined_yaw_delta = (yaw_delta_ransac_base_link * (1.0 - imu_yaw_weight_)) + (imu_yaw_delta_base_link * imu_yaw_weight_);
                RCLCPP_DEBUG(this->get_logger(), "IMU Yaw Rate (base_link): %.4f, IMU Yaw Delta (base_link): %.4f, RANSAC Yaw Delta (base_link): %.4f, Combined Yaw Delta: %.4f",
                             imu_yaw_rate_base_link, imu_yaw_delta_base_link, yaw_delta_ransac_base_link, combined_yaw_delta);
            } else {
                RCLCPP_DEBUG(this->get_logger(), "IMU data available but dt_odom_for_imu is not positive. Skipping IMU fusion.");
            }
        } else {
            RCLCPP_DEBUG(this->get_logger(), "IMU data not available or too old for fusion. Using RANSAC yaw delta only.");
        }
        // --- End IMU Integration ---

        // --- Apply Simple Low-Pass Filter (EMA) to the estimated deltas ---
        // New estimate = alpha * current_measurement + (1 - alpha) * previous_filtered_estimate

        // Filter translation (best_translation_3d is already in base_link coordinates)
        filtered_delta_translation_global_.x() = odom_filter_alpha_translation_ * best_translation_3d.x() +
                                                  (1.0 - odom_filter_alpha_translation_) * filtered_delta_translation_global_.x();
        filtered_delta_translation_global_.y() = odom_filter_alpha_translation_ * best_translation_3d.y() +
                                                  (1.0 - odom_filter_alpha_translation_) * filtered_delta_translation_global_.y();
        filtered_delta_translation_global_.z() = odom_filter_alpha_translation_ * best_translation_3d.z() +
                                                  (1.0 - odom_filter_alpha_translation_) * filtered_delta_translation_global_.z();

        // Filter rotation (yaw) - now uses combined_yaw_delta, which is consistent with base_link's Z-axis yaw
        filtered_yaw_delta_ = odom_filter_alpha_rotation_ * combined_yaw_delta +
                              (1.0 - odom_filter_alpha_rotation_) * filtered_yaw_delta_;

        // Use the filtered deltas for odometry update
        // Rotation is now around the Z-axis of the odom frame (base_link's yaw axis)
        Eigen::Quaterniond delta_q_eigen_filtered = Eigen::Quaterniond(Eigen::AngleAxisd(filtered_yaw_delta_, Eigen::Vector3d::UnitZ()));

        // Get current robot orientation as an Eigen Quaternion
        Eigen::Quaterniond q_current_eigen(current_robot_pose_.pose.pose.orientation.w,
                                          current_robot_pose_.pose.pose.orientation.x,
                                          current_robot_pose_.pose.pose.orientation.y,
                                          current_robot_pose_.pose.pose.orientation.z);

        // Transform filtered relative translation from robot's *current* local frame (base_link) to the global odom frame.
        // The filtered_delta_translation_global_ already represents the delta in base_link coordinates (X=forward, Y=left, Z=up)
        // So, we rotate it by the current global robot orientation to get the delta in the global odom frame.
        Eigen::Vector3d final_delta_translation_global = q_current_eigen.toRotationMatrix() * filtered_delta_translation_global_;

        // Apply the translation
        current_robot_pose_.pose.pose.position.x += final_delta_translation_global.x();
        current_robot_pose_.pose.pose.position.y += final_delta_translation_global.y();
        current_robot_pose_.pose.pose.position.z += final_delta_translation_global.z();

        // Apply the filtered rotation
        q_current_eigen = q_current_eigen * delta_q_eigen_filtered;
        q_current_eigen.normalize(); // Ensure quaternion is normalized
        current_robot_pose_.pose.pose.orientation = tf2::toMsg(tf2::Quaternion(q_current_eigen.x(), q_current_eigen.y(), q_current_eigen.z(), q_current_eigen.w()));


        // --- Odometry Debug ---
        RCLCPP_INFO(this->get_logger(), "RANSAC Best Translation (Camera X, Z): (%.6f, %.6f)",
                    best_translation_2d.x(), best_translation_2d.y()); // .y() here is the Z component
        RCLCPP_INFO(this->get_logger(), "RANSAC Best Rotation (Camera Yaw around Y-axis): %.6f radians", pitch_delta_camera_yaw);
        RCLCPP_INFO(this->get_logger(), "RANSAC Yaw Delta (Base_link): %.6f radians", yaw_delta_ransac_base_link);

        RCLCPP_INFO(this->get_logger(), "Raw (Pre-Filtered) Delta Translation (Base_link X, Y, Z): (%.6f, %.6f, %.6f)",
                    best_translation_3d.x(), best_translation_3d.y(), best_translation_3d.z());

        RCLCPP_INFO(this->get_logger(), "Filtered Delta Translation (Base_link X, Y, Z): (%.6f, %.6f, %.6f)",
                    filtered_delta_translation_global_.x(), filtered_delta_translation_global_.y(), filtered_delta_translation_global_.z());
        RCLCPP_INFO(this->get_logger(), "Filtered Yaw Delta (Base_link): %.6f radians", filtered_yaw_delta_);
        RCLCPP_INFO(this->get_logger(), "Current Robot Pose (X, Y, Z): (%.6f, %.6f, %.6f)",
                    current_robot_pose_.pose.pose.position.x, current_robot_pose_.pose.pose.position.y, current_robot_pose_.pose.pose.position.z);
        RCLCPP_INFO(this->get_logger(), "Current Robot Orientation (W, X, Y, Z): (%.6f, %.6f, %.6f, %.6f)",
                    current_robot_pose_.pose.pose.orientation.w, current_robot_pose_.pose.pose.orientation.x,
                    current_robot_pose_.pose.pose.orientation.y, current_robot_pose_.pose.pose.orientation.z);
        RCLCPP_INFO(this->get_logger(), "----------------------");
        // --- End Debug ---

        // Publish Odometry message
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = current_frame_time;
        odom_msg.header.frame_id = "odom"; // Parent frame
        odom_msg.child_frame_id = "base_link"; // As requested by user

        odom_msg.pose.pose = current_robot_pose_.pose.pose; // This now holds the correctly oriented pose

        // Calculate and publish twist (velocity) using filtered deltas
        double dt_odom = (current_frame_time - last_odom_update_time_).seconds();
        if (dt_odom > 0) {
            // Twist should be in the child_frame_id (base_link)
            // The filtered_delta_translation_global_ is already the delta in base_link coordinates.
            odom_msg.twist.twist.linear.x = filtered_delta_translation_global_.x() / dt_odom;
            odom_msg.twist.twist.linear.y = filtered_delta_translation_global_.y() / dt_odom;
            odom_msg.twist.twist.linear.z = filtered_delta_translation_global_.z() / dt_odom;

            // Angular velocity is usually reported in the body frame (base_link)
            // filtered_yaw_delta_ is already the yaw delta around base_link's Z-axis.
            odom_msg.twist.twist.angular.x = 0.0;
            odom_msg.twist.twist.angular.y = 0.0;
            odom_msg.twist.twist.angular.z = filtered_yaw_delta_ / dt_odom; // Yaw rate around base_link's Z
        } else {
            odom_msg.twist.twist.linear.x = 0.0;
            odom_msg.twist.twist.linear.y = 0.0;
            odom_msg.twist.twist.linear.z = 0.0;
            odom_msg.twist.twist.angular.x = 0.0;
            odom_msg.twist.twist.angular.y = 0.0;
            odom_msg.twist.twist.angular.z = 0.0;
        }

        odometry_publisher_->publish(odom_msg);
        last_odom_update_time_ = current_frame_time;

        // Publish the TF transform
        geometry_msgs::msg::TransformStamped t;

        t.header.stamp = current_frame_time;
        t.header.frame_id = "odom"; // Parent frame
        t.child_frame_id = "base_link"; // As requested by user

        t.transform.translation.x = current_robot_pose_.pose.pose.position.x;
        t.transform.translation.y = current_robot_pose_.pose.pose.position.y;
        t.transform.translation.z = current_robot_pose_.pose.pose.position.z;
        t.transform.rotation = current_robot_pose_.pose.pose.orientation;

        tf_broadcaster_->sendTransform(t);
        // Add a debug message to explicitly show the timestamp of the published TF
        RCLCPP_DEBUG(this->get_logger(), "Odometry: Published TF odom->base_link at timestamp: %f", t.header.stamp.sec + t.header.stamp.nanosec * 1e-9);

        // --- Publish Trajectory (Path) message ---
        geometry_msgs::msg::PoseStamped pose_stamped;
        pose_stamped.header.stamp = current_frame_time;
        pose_stamped.header.frame_id = "odom";
        pose_stamped.pose = current_robot_pose_.pose.pose;
        robot_path_.poses.push_back(pose_stamped); // Add current pose to the path

        robot_path_.header.stamp = current_frame_time; // Update path header timestamp
        path_publisher_->publish(robot_path_); // Publish the entire path
        RCLCPP_DEBUG(this->get_logger(), "Odometry: Published Trajectory Path with %zu points at timestamp: %f",
                     robot_path_.poses.size(), robot_path_.header.stamp.sec + robot_path_.header.stamp.nanosec * 1e-9);
        // --- End Trajectory (Path) message ---
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdometryNode>());
    rclcpp::shutdown();
    return 0;
}