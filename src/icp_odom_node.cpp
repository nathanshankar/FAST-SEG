#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <geometry_msgs/msg/pose_with_covariance_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp> // For tf2::toMsg
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
        this->declare_parameter<float>("static_object_velocity_threshold", 0.1f); // m/s
        this->declare_parameter<int>("min_static_clusters_for_odom", 3);

        // RANSAC parameters for odometry
        this->declare_parameter<int>("ransac_iterations", 100);
        this->declare_parameter<float>("ransac_inlier_threshold", 0.2f); // meters
        this->declare_parameter<int>("min_points_for_ransac_model", 3); // Minimum 3 points for a more stable initial model

        static_object_velocity_threshold_ = this->get_parameter("static_object_velocity_threshold").as_double();
        min_static_clusters_for_odom_ = this->get_parameter("min_static_clusters_for_odom").as_int();
        ransac_iterations_ = this->get_parameter("ransac_iterations").as_int();
        ransac_inlier_threshold_ = this->get_parameter("ransac_inlier_threshold").as_double();
        min_points_for_ransac_model_ = this->get_parameter("min_points_for_ransac_model").as_int();

        cluster_data_subscriber_ = this->create_subscription<my_cluster_odom::msg::ClusterArray>(
            "/detected_clusters", 10, std::bind(&OdometryNode::clusterDataCallback, this, std::placeholders::_1));

        odometry_publisher_ = this->create_publisher<nav_msgs::msg::Odometry>("/odometry", 10);
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

        RCLCPP_INFO(this->get_logger(), "Odometry estimation node has started.");
        RCLCPP_INFO(this->get_logger(), "Odometry: Static_Vel_Thresh=%.2f, Min_Static_Clusters=%d", static_object_velocity_threshold_, min_static_clusters_for_odom_);
        RCLCPP_INFO(this->get_logger(), "RANSAC: Iterations=%d, Inlier_Thresh=%.2f, Min_Points=%d", ransac_iterations_, ransac_inlier_threshold_, min_points_for_ransac_model_);
    }

private:
    // Parameters
    float static_object_velocity_threshold_;
    int min_static_clusters_for_odom_;
    int ransac_iterations_;
    float ransac_inlier_threshold_;
    int min_points_for_ransac_model_;

    rclcpp::Subscription<my_cluster_odom::msg::ClusterArray>::SharedPtr cluster_data_subscriber_;
    rclcpp::Publisher<nav_msgs::msg::Odometry>::SharedPtr odometry_publisher_;
    std::unique_ptr<tf2_ros::TransformBroadcaster> tf_broadcaster_;

    geometry_msgs::msg::PoseWithCovarianceStamped current_robot_pose_; // Global pose of the robot
    rclcpp::Time last_odom_update_time_; // To calculate dt between odometry updates

    std::default_random_engine rand_gen_;

    // Store previous cluster data for odometry calculation
    std::vector<my_cluster_odom::msg::Cluster> previous_frame_clusters_;
    rclcpp::Time previous_frame_time_;

    /**
     * @brief Callback function for incoming ClusterArray messages.
     * @param msg The shared pointer to the incoming ClusterArray message.
     */
    void clusterDataCallback(const my_cluster_odom::msg::ClusterArray::SharedPtr msg)
    {
        RCLCPP_INFO(this->get_logger(), "Odometry: Received ClusterArray with timestamp: %f", msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9);

        if (previous_frame_clusters_.empty()) {
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
     * @param current_frame_time The timestamp of the current frame.
     * @param frame_id The frame ID of the current data (e.g., "camera_infra1_optical_frame").
     * @param current_clusters The clusters detected in the current frame.
     */
    void calculateOdometry(rclcpp::Time current_frame_time, const std::string& frame_id, const std::vector<my_cluster_odom::msg::Cluster>& current_clusters)
    {
        if (previous_frame_clusters_.empty() || current_clusters.empty()) {
            RCLCPP_WARN(this->get_logger(), "Not enough cluster data to estimate odometry.");
            return;
        }

        // 1. Identify static candidates from previous_frame_clusters_ based on their predicted velocity
        std::vector<const my_cluster_odom::msg::Cluster*> static_candidates;
        for (const auto& cluster : previous_frame_clusters_) {
            // Use the predicted velocity from Kalman filter (x_k[3], x_k[4], x_k[5])
            Eigen::Vector3d velocity(cluster.kalman_state[3], cluster.kalman_state[4], cluster.kalman_state[5]);
            if (velocity.norm() < static_object_velocity_threshold_) {
                static_candidates.push_back(&cluster);
            }
        }

        if (static_candidates.size() < min_static_clusters_for_odom_) {
            RCLCPP_WARN(this->get_logger(), "Not enough static clusters (%zu) to estimate odometry. Need at least %d.",
                        static_candidates.size(), min_static_clusters_for_odom_);
            return;
        }

        // 2. Match static candidates from previous frame to current frame clusters
        std::vector<Eigen::Vector2f> prev_2d_points_full; // Previous (predicted) positions of static clusters in 2D (X-Z plane of sensor)
        std::vector<Eigen::Vector2f> curr_2d_points_full; // Current (measured) positions of static clusters in 2D (X-Z plane of sensor)

        for (const auto& prev_static_cluster : static_candidates) {
            // Use predicted position from previous frame for prev_points
            // The Kalman state is [px, py, pz, vx, vy, vz]. We use px and pz for 2D motion.
            Eigen::Vector3d predicted_pos_prev(prev_static_cluster->kalman_state[0],
                                               prev_static_cluster->kalman_state[1],
                                               prev_static_cluster->kalman_state[2]);

            // Find the corresponding cluster in the current frame by ID
            bool found_match = false;
            for (const auto& current_cluster : current_clusters) {
                if (current_cluster.id == prev_static_cluster->id) {
                    // This is the matched current measurement for the static cluster
                    prev_2d_points_full.push_back(Eigen::Vector2f(predicted_pos_prev.x(), predicted_pos_prev.z())); // Sensor X, Sensor Z
                    curr_2d_points_full.push_back(Eigen::Vector2f(current_cluster.centroid.x, current_cluster.centroid.z)); // Sensor X, Sensor Z
                    found_match = true;
                    break;
                }
            }
            if (!found_match) {
                RCLCPP_DEBUG(this->get_logger(), "Static cluster ID %d from previous frame not found in current frame. Skipping for odometry.", prev_static_cluster->id);
            }
        }

        if (prev_2d_points_full.size() < min_points_for_ransac_model_) {
            RCLCPP_WARN(this->get_logger(), "Not enough matched static points (%zu) for RANSAC odometry. Need at least %d.",
                        prev_2d_points_full.size(), min_points_for_ransac_model_);
            return;
        }

        // 3. RANSAC to estimate 2D transformation (translation + rotation around Y-axis)
        // In 2D (X-Z plane of sensor), a point P_curr = R * P_prev + T
        Eigen::Vector2f best_translation = Eigen::Vector2f::Zero();
        float best_yaw = 0.0f;
        int max_inliers = 0;

        std::uniform_int_distribution<> distrib(0, prev_2d_points_full.size() - 1);
        std::normal_distribution<double> noise_dist(0.0, ransac_inlier_threshold_ / 3.0); // For generating noisy points for testing

        for (int i = 0; i < ransac_iterations_; ++i) {
            // Randomly select 2 pairs of points for 2D transformation (X-Z plane)
            // Need at least 2 points for a unique rigid body transformation. 3+ is better for robustness.
            if (prev_2d_points_full.size() < 2) {
                RCLCPP_WARN(this->get_logger(), "Not enough points for RANSAC (need at least 2).");
                break;
            }

            std::vector<int> sample_indices;
            while(sample_indices.size() < std::min((size_t)min_points_for_ransac_model_, prev_2d_points_full.size())) {
                int idx = distrib(rand_gen_);
                if (std::find(sample_indices.begin(), sample_indices.end(), idx) == sample_indices.end()) {
                    sample_indices.push_back(idx);
                }
            }

            Eigen::Matrix2f A = Eigen::Matrix2f::Zero();
            Eigen::Vector2f b = Eigen::Vector2f::Zero();

            // Centroids of sampled points
            Eigen::Vector2f p_prev_centroid = Eigen::Vector2f::Zero();
            Eigen::Vector2f p_curr_centroid = Eigen::Vector2f::Zero();

            for (int idx : sample_indices) {
                p_prev_centroid += prev_2d_points_full[idx];
                p_curr_centroid += curr_2d_points_full[idx];
            }
            p_prev_centroid /= sample_indices.size();
            p_curr_centroid /= sample_indices.size();

            Eigen::Matrix2f H = Eigen::Matrix2f::Zero();
            for (int idx : sample_indices) {
                Eigen::Vector2f p_prev_centered = prev_2d_points_full[idx] - p_prev_centroid;
                Eigen::Vector2f p_curr_centered = curr_2d_points_full[idx] - p_curr_centroid;
                H += p_prev_centered * p_curr_centered.transpose();
            }

            // SVD to find rotation
            Eigen::JacobiSVD<Eigen::Matrix2f> svd(H, Eigen::ComputeFullU | Eigen::ComputeFullV);
            Eigen::Matrix2f R_est = svd.matrixV() * svd.matrixU().transpose();

            // Check for reflection and correct it
            if (R_est.determinant() < 0) {
                Eigen::Matrix2f V = svd.matrixV();
                V(0, 1) *= -1;
                V(1, 1) *= -1;
                R_est = V * svd.matrixU().transpose();
            }

            Eigen::Vector2f t_est = p_curr_centroid - R_est * p_prev_centroid;

            // Calculate current inliers
            int current_inliers = 0;
            for (size_t k = 0; k < prev_2d_points_full.size(); ++k) {
                Eigen::Vector2f transformed_point = R_est * prev_2d_points_full[k] + t_est;
                double distance = (transformed_point - curr_2d_points_full[k]).norm();
                if (distance < ransac_inlier_threshold_) {
                    current_inliers++;
                }
            }

            if (current_inliers > max_inliers) {
                max_inliers = current_inliers;
                best_translation = t_est;
                best_yaw = std::atan2(R_est(1, 0), R_est(0, 0)); // Rotation matrix to yaw
            }
        }

        if (max_inliers < min_points_for_ransac_model_) {
            RCLCPP_WARN(this->get_logger(), "RANSAC failed to find enough inliers for odometry. Max inliers: %d, required: %d.",
                        max_inliers, min_points_for_ransac_model_);
            return;
        }

        // Update robot's pose based on the estimated transformation
        // The transformation (best_translation, best_yaw) is from previous_frame's sensor pose to current_frame's sensor pose
        // We need to apply this transformation in the 'odom' frame.

        // Get current robot orientation (yaw) in radians
        tf2::Quaternion q_current;
        tf2::fromMsg(current_robot_pose_.pose.pose.orientation, q_current);
        double roll, pitch, current_yaw;
        tf2::Matrix3x3(q_current).getRPY(roll, pitch, current_yaw);

        // Transform the 2D translation from sensor frame (X-Z plane) to odom frame (X-Y plane)
        // Note: X in sensor is X in odom, Z in sensor is Y in odom (assuming camera looks forward, ground is XZ plane)
        // This mapping depends on your camera's orientation relative to your robot's base_link.
        // Assuming sensor X -> robot X, sensor Z -> robot Y, sensor Y -> robot Z (vertical)
        // Assuming sensor X -> robot X, sensor Z -> robot Y for 2D ground plane motion
        // The Z (vertical) component is set to 0.0, assuming planar motion for this odometry source.
        Eigen::Vector3d delta_translation_robot_frame(best_translation.x(), best_translation.y(), 0.0);

        // Rotate delta_translation from robot's *current* frame to global odom frame
        Eigen::Matrix3d rotation_odom;
        rotation_odom = Eigen::AngleAxisd(current_yaw, Eigen::Vector3d::UnitZ()); // Rotation about Z-axis for yaw

        Eigen::Vector3d delta_translation = rotation_odom * delta_translation_robot_frame;

        // Apply the translation
        current_robot_pose_.pose.pose.position.x += delta_translation.x();
        current_robot_pose_.pose.pose.position.y += delta_translation.y();
        current_robot_pose_.pose.pose.position.z += delta_translation.z();

        // Apply the rotation (yaw)
        tf2::Quaternion delta_q;
        delta_q.setRPY(0, 0, best_yaw); // Rotation around Z-axis (yaw)
        q_current = q_current * delta_q; // Apply relative rotation
        q_current.normalize(); // Ensure quaternion is normalized
        current_robot_pose_.pose.pose.orientation = tf2::toMsg(q_current);

        // Publish Odometry message
        nav_msgs::msg::Odometry odom_msg;
        odom_msg.header.stamp = current_frame_time;
        odom_msg.header.frame_id = "odom"; // Parent frame
        odom_msg.child_frame_id = "base_link"; // As requested by user

        odom_msg.pose.pose = current_robot_pose_.pose.pose;

        // Calculate and publish twist (velocity)
        double dt_odom = (current_frame_time - last_odom_update_time_).seconds();
        if (dt_odom > 0) {
            odom_msg.twist.twist.linear.x = delta_translation.x() / dt_odom;
            odom_msg.twist.twist.linear.y = delta_translation.y() / dt_odom;
            odom_msg.twist.twist.linear.z = delta_translation.z() / dt_odom;
            odom_msg.twist.twist.angular.z = best_yaw / dt_odom; // Yaw rate
        } else {
            odom_msg.twist.twist.linear.x = 0.0;
            odom_msg.twist.twist.linear.y = 0.0;
            odom_msg.twist.twist.linear.z = 0.0;
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
        RCLCPP_INFO(this->get_logger(), "Odometry: Published TF odom->base_link at timestamp: %f", t.header.stamp.sec + t.header.stamp.nanosec * 1e-9);
    }
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<OdometryNode>());
    rclcpp::shutdown();
    return 0;
}