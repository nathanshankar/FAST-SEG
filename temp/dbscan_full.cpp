#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <visualization_msgs/msg/marker_array.hpp>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <queue>
#include <limits>
#include <random>
#include <chrono>
#include <algorithm> // For std::max, std::min
#include <cmath>

// For Kalman Filter
#include <Eigen/Dense>

// Custom message includes (replace 'my_cluster_odom' with your actual package name)
#include <my_cluster_odom/msg/cluster.hpp>
#include <my_cluster_odom/msg/cluster_array.hpp>

class DBSCANNode : public rclcpp::Node
{
public:
    DBSCANNode() : Node("dbscan_node")
    {
        // Declare parameters for DBSCAN and tracking
        this->declare_parameter<double>("dbscan_eps", 0.20); // Changed to double
        this->declare_parameter<int>("dbscan_min_pts", 10);
        this->declare_parameter<int>("n_stable_frames", 3);
        this->declare_parameter<int>("n_missed_frames", 20); // Frames before a track is moved to "lost" pool
        this->declare_parameter<int>("max_lost_frames", 200); // Example: keeps tracks in memory for longer

        this->declare_parameter<double>("active_track_match_distance_threshold", 1.5); // Changed to double
        this->declare_parameter<double>("lost_track_position_threshold", 7.5); // Changed to double
        this->declare_parameter<double>("lost_track_dimension_threshold", 1.5); // Changed to double
        this->declare_parameter<double>("reid_position_weight", 1.0); // Changed to double
        this->declare_parameter<double>("reid_dimension_weight", 0.5); // Changed to double

        this->declare_parameter<double>("kalman_pos_noise_q", 0.1); // Changed to double
        this->declare_parameter<double>("kalman_vel_noise_q", 2.0); // Changed to double
        this->declare_parameter<double>("kalman_min_vel_noise_q", 0.05); // NEW PARAMETER, Changed to double
        this->declare_parameter<double>("kalman_vel_noise_q_decay_factor", 0.9); // NEW PARAMETER, Changed to double
        this->declare_parameter<double>("kalman_measurement_noise_r", 0.02); // Changed to double

        // Get parameters (all using as_double() now)
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

        kalman_pos_noise_q_ = this->get_parameter("kalman_pos_noise_q").as_double();
        kalman_vel_noise_q_ = this->get_parameter("kalman_vel_noise_q").as_double();
        kalman_min_vel_noise_q_ = this->get_parameter("kalman_min_vel_noise_q").as_double(); // Get new parameter
        kalman_vel_noise_q_decay_factor_ = this->get_parameter("kalman_vel_noise_q_decay_factor").as_double(); // Get new parameter
        kalman_measurement_noise_r_ = this->get_parameter("kalman_measurement_noise_r").as_double();

        // Subscribers and Publishers
        point_cloud_subscriber_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/points", 10, std::bind(&DBSCANNode::pointCloudCallback, this, std::placeholders::_1));

        cluster_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/clusters", 10);
        bounding_box_viz_publisher_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/bounding_boxes", 10);
        cluster_data_publisher_ = this->create_publisher<my_cluster_odom::msg::ClusterArray>("/detected_clusters", 10); // Custom message publisher

        rand_gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());

        // Initialize constant Kalman filter matrices
        // R_ and H_ remain constant as they represent measurement model
        R_ = Eigen::MatrixXd::Identity(3, 3) * kalman_measurement_noise_r_; // Measurement noise

        H_ = Eigen::MatrixXd::Zero(3, 6);
        H_(0, 0) = 1.0;
        H_(1, 1) = 1.0;
        H_(2, 2) = 1.0;

        RCLCPP_INFO(this->get_logger(), "DBSCAN clustering node has started.");
        RCLCPP_INFO(this->get_logger(), "Parameters: eps=%.2f, minPts=%d", dbscan_eps_, dbscan_min_pts_);
        RCLCPP_INFO(this->get_logger(), "Tracking: N_STABLE_FRAMES=%d, N_MISSED_FRAMES=%d, MAX_LOST_FRAMES=%d", n_stable_frames_, n_missed_frames_, max_lost_frames_);
        RCLCPP_INFO(this->get_logger(), "Matching: Active_Dist=%.2f, Lost_Pos_Dist=%.2f, Lost_Dim_Diff=%.2f", active_track_match_distance_threshold_, lost_track_position_threshold_, lost_track_dimension_threshold_);
        RCLCPP_INFO(this->get_logger(), "Re-ID Weights: Pos=%.2f, Dim=%.2f", reid_position_weight_, reid_dimension_weight_);
        RCLCPP_INFO(this->get_logger(), "Kalman Filter (Adaptive Vel Noise): Pos_Q=%.2f, Initial_Vel_Q=%.2f, Min_Vel_Q=%.2f, Decay_Factor=%.2f, Meas_R=%.2f", 
                    kalman_pos_noise_q_, kalman_vel_noise_q_, kalman_min_vel_noise_q_, kalman_vel_noise_q_decay_factor_, kalman_measurement_noise_r_);
    }

private:
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

    int cluster_id_counter_ = 0;

    // Parameters (all changed to double)
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
    double kalman_vel_noise_q_; // Initial/Max value
    double kalman_min_vel_noise_q_; // NEW
    double kalman_vel_noise_q_decay_factor_; // NEW
    double kalman_measurement_noise_r_;

    std::vector<ClusterMemory> previous_clusters_; // Active tracks
    std::vector<ClusterMemory> recently_lost_clusters_; // Tracks that were missed, but not yet forgotten

    std::default_random_engine rand_gen_;

    // Q_ is now dynamic, R_ and H_ are constant
    Eigen::MatrixXd R_; // Measurement noise covariance
    Eigen::MatrixXd H_; // Measurement matrix

    // ROS 2 objects
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr point_cloud_subscriber_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr cluster_viz_publisher_, bounding_box_viz_publisher_;
    rclcpp::Publisher<my_cluster_odom::msg::ClusterArray>::SharedPtr cluster_data_publisher_; // Publisher for custom message

    /**
     * @brief Initializes the Kalman filter for a new cluster.
     * @param cluster_mem The ClusterMemory struct to initialize.
     * @param initial_position The initial measured position of the cluster.
     * @param current_time The current time.
     */
    void initKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& initial_position, rclcpp::Time current_time)
    {
        cluster_mem.x_k.resize(6);
        cluster_mem.x_k << initial_position.x(), initial_position.y(), initial_position.z(), 0.0, 0.0, 0.0; // Initial velocity is zero

        cluster_mem.P_k = Eigen::MatrixXd::Identity(6, 6);
        cluster_mem.P_k.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * 0.1; // Initial position uncertainty
        cluster_mem.P_k.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * 10.0; // Initial velocity uncertainty (high, because we don't know it)

        cluster_mem.last_update_time = current_time;
        cluster_mem.current_kalman_vel_noise_q = kalman_vel_noise_q_; // Initialize with max noise
    }

    /**
     * @brief Predicts the next state of the Kalman filter.
     * @param cluster_mem The ClusterMemory struct to predict.
     * @param dt The time difference since the last update.
     */
    void predictKalmanFilter(ClusterMemory& cluster_mem, double dt)
    {
        if (dt <= 0) { // Avoid issues with non-positive dt
            return;
        }

        Eigen::MatrixXd F(6, 6); // State transition matrix
        F << 1, 0, 0, dt, 0, 0,
             0, 1, 0, 0, dt, 0,
             0, 0, 1, 0, 0, dt,
             0, 0, 0, 1, 0, 0,
             0, 0, 0, 0, 1, 0,
             0, 0, 0, 0, 0, 1;

        // --- ADAPTIVE Q MATRIX ---
        // Create Q matrix dynamically for this specific track based on its current velocity noise
        Eigen::MatrixXd current_Q = Eigen::MatrixXd::Identity(6, 6);
        current_Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * kalman_pos_noise_q_; // Position noise (constant)
        current_Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * cluster_mem.current_kalman_vel_noise_q; // Adaptive Velocity noise

        cluster_mem.x_k = F * cluster_mem.x_k; // Predict state
        cluster_mem.P_k = F * cluster_mem.P_k * F.transpose() + current_Q; // Predict covariance
    }

    /**
     * @brief Updates the Kalman filter with a new measurement.
     * Also updates the adaptive process noise for velocity.
     * @param cluster_mem The ClusterMemory struct to update.
     * @param measurement The new measured position.
     * @param current_time The current time.
     */
    void updateKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& measurement, rclcpp::Time current_time)
    {
        Eigen::VectorXd z(3); // Measurement vector
        z << measurement.x(), measurement.y(), measurement.z();

        Eigen::VectorXd y = z - H_ * cluster_mem.x_k; // Measurement residual
        Eigen::MatrixXd S = H_ * cluster_mem.P_k * H_.transpose() + R_; // Innovation (or pre-fit residual) covariance
        Eigen::MatrixXd K = cluster_mem.P_k * H_.transpose() * S.inverse(); // Kalman gain

        cluster_mem.x_k = cluster_mem.x_k + K * y; // Update state estimate
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
        cluster_mem.P_k = (I - K * H_) * cluster_mem.P_k; // Update covariance estimate

        cluster_mem.last_update_time = current_time;

        // --- ADAPTIVE VELOCITY NOISE DECAY ---
        // When a track is updated (i.e., matched), reduce its velocity process noise
        // Ensure both arguments to std::max are of the same type (double)
        cluster_mem.current_kalman_vel_noise_q = std::max(kalman_min_vel_noise_q_,
                                                          cluster_mem.current_kalman_vel_noise_q * kalman_vel_noise_q_decay_factor_);
    }
    
    /**
     * @brief Callback function for incoming point cloud messages.
     * Performs filtering, DBSCAN clustering, and cluster tracking.
     * @param msg The shared pointer to the incoming PointCloud2 message.
     */
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {   
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received an empty point cloud.");
            return;
        }

        pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>());

        // --- Adaptive Y-axis (height) filter ---
        std::vector<float> ys_copy;
        ys_copy.reserve(cloud->size());
        for (const auto& p : cloud->points) ys_copy.push_back(p.y);
        if (ys_copy.empty()) { RCLCPP_WARN(this->get_logger(), "Y-coordinates empty, skipping Y-filter."); return; }
        
        // Use a more robust median-based approach or fixed height if ground/ceiling are known
        // For now, keeping your existing percentile logic
        std::nth_element(ys_copy.begin(), ys_copy.begin() + ys_copy.size() * 0.90, ys_copy.end());
        float ground_y = ys_copy[ys_copy.size() * 0.90];
        std::nth_element(ys_copy.begin(), ys_copy.begin() + ys_copy.size() * 0.15, ys_copy.end());
        float ceiling_y = ys_copy[ys_copy.size() * 0.15];

        pcl::PassThrough<pcl::PointXYZ> pass_y;
        pass_y.setInputCloud(cloud);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(ceiling_y + 0.25f, ground_y - 0.05f); // Small margin to avoid cutting off objects at boundary
        pass_y.filter(*filtered_cloud);

        if (filtered_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Point cloud empty after Y-axis filtering.");
            return;
        }

        // --- Adaptive X-axis filter ---
        std::vector<float> xs_copy;
        xs_copy.reserve(filtered_cloud->size());
        for (const auto& p : filtered_cloud->points) xs_copy.push_back(p.x);
        if (xs_copy.empty()) { RCLCPP_WARN(this->get_logger(), "X-coordinates empty, skipping X-filter."); return; }
        
        std::nth_element(xs_copy.begin(), xs_copy.begin() + xs_copy.size() * 0.10, xs_copy.end());
        float wall_min_threshold = xs_copy[xs_copy.size() * 0.10];
        std::nth_element(xs_copy.begin(), xs_copy.begin() + xs_copy.size() * 0.90, xs_copy.end());
        float wall_max_threshold = xs_copy[xs_copy.size() * 0.90];

        pcl::PassThrough<pcl::PointXYZ> pass_x;
        pass_x.setInputCloud(filtered_cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(wall_min_threshold + 0.05f, wall_max_threshold - 0.05f);
        pass_x.filter(*filtered_cloud);

        if (filtered_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Point cloud empty after X-axis filtering.");
            return;
        }

        // --- Adaptive Z-axis filter ---
        std::vector<float> zs_copy;
        zs_copy.reserve(filtered_cloud->size());
        for (const auto& p : filtered_cloud->points) zs_copy.push_back(p.z);
        if (zs_copy.empty()) { RCLCPP_WARN(this->get_logger(), "Z-coordinates empty, skipping Z-filter."); return; }
        
        std::nth_element(zs_copy.begin(), zs_copy.begin() + zs_copy.size() * 0.90, zs_copy.end());
        float wall_max_threshold_z = zs_copy[zs_copy.size() * 0.90];

        pcl::PassThrough<pcl::PointXYZ> pass_z;
        pass_z.setInputCloud(filtered_cloud);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(0.0f, wall_max_threshold_z - 0.05f); // Assuming objects are above ground (0.0)
        pass_z.filter(*filtered_cloud);

        if (filtered_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Point cloud empty after Z-axis filtering.");
            return;
        }

        // --- Voxel Grid Downsampling ---
        pcl::VoxelGrid<pcl::PointXYZ> vg;
        vg.setInputCloud(filtered_cloud);
        vg.setLeafSize(0.05f, 0.05f, 0.05f); // Keep this value reasonable for cluster definition
        vg.filter(*filtered_cloud);

        RCLCPP_INFO(this->get_logger(), "Point cloud contains %lu points after filtering and downsampling.", filtered_cloud->size());

        if (filtered_cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Point cloud empty after voxel grid downsampling.");
            return;
        }

        std::vector<std::vector<int>> clusters = dbscanClustering(filtered_cloud, static_cast<float>(dbscan_eps_), dbscan_min_pts_); // Cast eps to float
        std::vector<std::vector<int>> refined_clusters = refineClusters(filtered_cloud, clusters);
        processAndPublishClusters(msg->header.frame_id, filtered_cloud, refined_clusters);
    }

    /**
     * @brief Refines the detected clusters by filtering out small or degenerate clusters
     * and performing recursive sub-clustering for dense areas.
     * @param cloud The input point cloud.
     * @param clusters The initial clusters from DBSCAN.
     * @param recursion_level Current recursion depth.
     * @param max_recursion Maximum recursion depth for sub-clustering.
     * @return A vector of refined clusters (indices into the original cloud).
     */
    std::vector<std::vector<int>> refineClusters(
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
        const std::vector<std::vector<int>>& clusters,
        int recursion_level = 0, int max_recursion = 1)
    {
        std::vector<std::vector<int>> refined_clusters;

        const float MIN_CLUSTER_POINTS_THRESHOLD = 30; // Minimum points for a valid cluster
        const float MAX_DENSITY_FOR_SMALL_CLUSTER = 1000.0f; // Heuristic to break down dense clusters
        const float MIN_VOLUME_THRESHOLD = 0.01f; // Minimum volume for a valid cluster
        const float MIN_SIDE_LENGTH_THRESHOLD = 0.03f; // Minimum side length for a valid cluster

        for (const auto& cluster : clusters)
        {
            if (cluster.empty()) {
                RCLCPP_WARN(this->get_logger(), "Skipping empty cluster during refinement.");
                continue;
            }
            // Filter out very small clusters early
            if (cluster.size() < MIN_CLUSTER_POINTS_THRESHOLD) {
                continue;
            }

            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, cluster, min_pt, max_pt);

            float dx = max_pt.x() - min_pt.x();
            float dy = max_pt.y() - min_pt.y();
            float dz = max_pt.z() - min_pt.z();

            float volume = dx * dy * dz;
            if (volume < 1e-9f) volume = 1e-9f; // Avoid division by zero for density

            float density = static_cast<float>(cluster.size()) / volume;

            // Filter out degenerate clusters (flat, tiny, or too small volume)
            if (dx < MIN_SIDE_LENGTH_THRESHOLD || dy < MIN_SIDE_LENGTH_THRESHOLD || dz < MIN_SIDE_LENGTH_THRESHOLD)
                continue;
            if (volume < MIN_VOLUME_THRESHOLD)
                continue;

            // Heuristic to break down potentially merged dense clusters
            if (cluster.size() < MIN_CLUSTER_POINTS_THRESHOLD * 2 && density > MAX_DENSITY_FOR_SMALL_CLUSTER)
                continue;

            // Recursive clustering for less dense, larger clusters
            if (density < 50.0f && recursion_level < max_recursion) // Tune this density threshold
            {
                pcl::PointCloud<pcl::PointXYZ>::Ptr sub_cloud(new pcl::PointCloud<pcl::PointXYZ>());
                pcl::copyPointCloud(*cloud, cluster, *sub_cloud);
                if (sub_cloud->empty()) {
                    RCLCPP_WARN(this->get_logger(), "Sub-cloud empty after copying during refinement recursion. Skipping.");
                    continue;
                }

                float new_eps = static_cast<float>(dbscan_eps_) / 2.0f; // Use a smaller epsilon for sub-clustering
                auto sub_clusters = dbscanClustering(sub_cloud, new_eps, dbscan_min_pts_);

                // Map indices back to original cloud
                for (auto& sub : sub_clusters) {
                    std::vector<int> original_indices;
                    original_indices.reserve(sub.size());
                    for (auto& idx : sub) {
                        original_indices.push_back(cluster[idx]); // Get original index from parent cluster
                    }
                    sub = original_indices;
                }

                auto sub_refined = refineClusters(cloud, sub_clusters, recursion_level + 1, max_recursion);
                refined_clusters.insert(refined_clusters.end(), sub_refined.begin(), sub_refined.end());
            }
            else
            {
                refined_clusters.push_back(cluster);
            }
        }

        // --- Overlap Filtering (Remove contained bounding boxes) ---
        // This is important to ensure distinct objects, especially after sub-clustering
        std::vector<Eigen::Vector4f> mins, maxs;
        mins.reserve(refined_clusters.size());
        maxs.reserve(refined_clusters.size());

        for (const auto& cluster : refined_clusters)
        {
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, cluster, min_pt, max_pt);
            mins.push_back(min_pt);
            maxs.push_back(max_pt);
        }

        auto boxContains = [](const Eigen::Vector4f& minA, const Eigen::Vector4f& maxA,
                              const Eigen::Vector4f& minB, const Eigen::Vector4f& maxB) -> bool
        {
            const float EPS_CONTAIN = 1e-3f; // Small epsilon for floating point comparisons
            return (minA.x() <= minB.x() + EPS_CONTAIN && maxA.x() >= maxB.x() - EPS_CONTAIN) &&
                   (minA.y() <= minB.y() + EPS_CONTAIN && maxA.y() >= maxB.y() - EPS_CONTAIN) &&
                   (minA.z() <= minB.z() + EPS_CONTAIN && maxA.z() >= maxB.z() - EPS_CONTAIN);
        };

        std::vector<bool> keep(refined_clusters.size(), true);

        for (size_t i = 0; i < refined_clusters.size(); ++i)
        {
            if (!keep[i]) continue; // Already marked for removal

            for (size_t j = 0; j < refined_clusters.size(); ++j)
            {
                if (i == j || !keep[j]) continue; // Don't compare with self or already removed

                if (boxContains(mins[i], maxs[i], mins[j], maxs[j]))
                {
                    keep[j] = false; // Mark the *contained* box for removal
                }
                 // Also handle cases where A contains B and B contains A (shouldn't happen with proper unique clusters, but defensive)
                else if (boxContains(mins[j], maxs[j], mins[i], maxs[i]))
                {
                    keep[i] = false;
                    break; // i is contained, so no need to check other j's for this i
                }
            }
        }

        std::vector<std::vector<int>> filtered_clusters;
        for (size_t i = 0; i < refined_clusters.size(); ++i)
        {
            if (keep[i])
                filtered_clusters.push_back(refined_clusters[i]);
        }

        return filtered_clusters;
    }

    /**
     * @brief Performs DBSCAN clustering on the input point cloud.
     * @param cloud The input point cloud.
     * @param eps The maximum distance between two samples for one to be considered as in the neighborhood of the other.
     * @param minPts The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
     * @return A vector of vectors, where each inner vector contains the indices of points belonging to a cluster.
     */
    std::vector<std::vector<int>> dbscanClustering(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, float eps, int minPts)
    {
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<bool> visited(cloud->size(), false);
        std::vector<int> cluster_labels(cloud->size(), -1); // -1: noise, >=0: cluster ID
        std::vector<std::vector<int>> clusters;
        int current_cluster_id_dbscan = 0;

        for (size_t i = 0; i < cloud->size(); ++i)
        {
            if (visited[i]) continue;

            std::vector<int> neighbors;
            std::vector<float> distances;
            tree->radiusSearch(cloud->points[i], eps, neighbors, distances);

            if (neighbors.size() < static_cast<size_t>(minPts)) // Cast minPts to size_t
            {
                visited[i] = true; // Mark as visited, but it's a noise point (cluster_labels[i] remains -1)
                continue;
            }

            // Core point found, start a new cluster
            clusters.emplace_back();
            std::queue<int> neighbor_queue;
            neighbor_queue.push(i);
            visited[i] = true;
            cluster_labels[i] = current_cluster_id_dbscan;

            while (!neighbor_queue.empty())
            {
                int current = neighbor_queue.front();
                neighbor_queue.pop();
                clusters.back().push_back(current); // Add to current cluster

                tree->radiusSearch(cloud->points[current], eps, neighbors, distances);

                for (int idx : neighbors)
                {
                    if (!visited[idx])
                    {
                        visited[idx] = true;
                        cluster_labels[idx] = current_cluster_id_dbscan; // Assign to current cluster
                        
                        std::vector<int> new_neighbors;
                        std::vector<float> new_distances;
                        tree->radiusSearch(cloud->points[idx], eps, new_neighbors, new_distances);
                        if (new_neighbors.size() >= static_cast<size_t>(minPts)) { // Cast minPts to size_t
                            // If it's a core point, add its neighbors to the queue
                            neighbor_queue.push(idx);
                        }
                    }
                    // This 'else if' handles border points that might have been marked as noise (-1)
                    // and are now found to be reachable from a core point.
                    else if (cluster_labels[idx] == -1) {
                         cluster_labels[idx] = current_cluster_id_dbscan;
                    }
                }
            }
            current_cluster_id_dbscan++;
        }
        return clusters;
    }

    /**
     * @brief Generates a random color for visualization.
     * @return A std_msgs::msg::ColorRGBA with random RGB values and full alpha.
     */
    std_msgs::msg::ColorRGBA generateRandomColor()
    {
        std_msgs::msg::ColorRGBA color;
        std::uniform_real_distribution<> dist(0.0, 1.0);
        color.r = dist(rand_gen_);
        color.g = dist(rand_gen_);
        color.b = dist(rand_gen_);
        // Ensure some minimum brightness to be visible
        if (color.r + color.g + color.b < 0.8) {
            float sum_rgb = color.r + color.g + color.b;
            if (sum_rgb < 1e-6f) sum_rgb = 1e-6f; // Avoid division by zero
            float scale_factor = 0.8f / sum_rgb; // Cast 0.8 to float
            color.r *= scale_factor;
            color.g *= scale_factor;
            color.b *= scale_factor;
        }
        color.a = 1.0f;
        return color;
    }

    /**
     * @brief Computes the centroid of a cluster.
     * @param cloud The input point cloud.
     * @param indices The indices of points belonging to the cluster.
     * @return An Eigen::Vector3f representing the centroid.
     */
    Eigen::Vector3f computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices)
    {
        Eigen::Vector3f centroid(0, 0, 0);
        if (indices.empty()) return centroid;
        for (int idx : indices)
        {
            centroid.x() += cloud->points[idx].x;
            centroid.y() += cloud->points[idx].y;
            centroid.z() += cloud->points[idx].z;
        }
        centroid /= static_cast<float>(indices.size());
        return centroid;
    }

    /**
     * @brief Processes the detected clusters, performs tracking, and publishes visualization markers
     * and custom cluster data messages.
     * @param frame_id The frame ID of the current point cloud.
     * @param cloud The input point cloud.
     * @param clusters The detected clusters.
     */
    void processAndPublishClusters(const std::string &frame_id, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, std::vector<std::vector<int>> &clusters)
    {
        visualization_msgs::msg::MarkerArray marker_array;
        visualization_msgs::msg::MarkerArray bounding_box_array;
        my_cluster_odom::msg::ClusterArray cluster_data_array_msg;
        cluster_data_array_msg.header.stamp = this->get_clock()->now();
        cluster_data_array_msg.header.frame_id = frame_id;

        rclcpp::Time current_frame_time = this->get_clock()->now();

        // --- Step 1: Predict positions of active tracks ---
        // Increment missed_count for all active tracks and predict their next state.
        for (auto& prev_cluster : previous_clusters_) {
            prev_cluster.missed_count++;
            double dt = (current_frame_time - prev_cluster.last_update_time).seconds();
            if (dt < 0) { // Handle potential time jumps (e.g., bag file restarts)
                RCLCPP_WARN(this->get_logger(), "Negative dt detected for active track %d (%f). Clamping to 0.", prev_cluster.id, dt);
                dt = 0;
            }
            predictKalmanFilter(prev_cluster, dt);
        }

        // --- Step 2: Predict positions for recently lost tracks ---
        for (auto& lost_cluster : recently_lost_clusters_) {
            lost_cluster.missed_count++;
            double dt = (current_frame_time - lost_cluster.last_update_time).seconds();
            if (dt < 0) {
                RCLCPP_WARN(this->get_logger(), "Negative dt detected for lost track %d (%f). Clamping to 0.", lost_cluster.id, dt);
                dt = 0;
            }
            predictKalmanFilter(lost_cluster, dt);
        }


        // --- Step 3: Match current clusters with active tracks (previous_clusters_) ---
        std::vector<bool> current_cluster_matched(clusters.size(), false);
        std::vector<bool> previous_cluster_is_assigned(previous_clusters_.size(), false);

        std::vector<std::pair<int, int>> matches_found_active; // Stores {current_idx, previous_active_idx} pairs

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            if (clusters[i].empty() || clusters[i].size() < static_cast<size_t>(dbscan_min_pts_)) continue; // Only consider valid clusters

            Eigen::Vector3f current_centroid = computeCentroid(cloud, clusters[i]);

            int best_matched_prev_idx = -1;
            double min_dist = std::numeric_limits<double>::max(); // Use double for comparison

            for (size_t j = 0; j < previous_clusters_.size(); ++j)
            {
                if (previous_cluster_is_assigned[j]) {
                    continue; // This previous track has already been assigned
                }

                Eigen::Vector3f predicted_centroid_prev(static_cast<float>(previous_clusters_[j].x_k(0)), // Cast to float for Eigen::Vector3f
                                                        static_cast<float>(previous_clusters_[j].x_k(1)),
                                                        static_cast<float>(previous_clusters_[j].x_k(2)));

                double dist = (current_centroid - predicted_centroid_prev).norm(); // Calculation result will be float, promote for comparison

                if (dist < min_dist && dist < active_track_match_distance_threshold_)
                {
                    min_dist = dist;
                    best_matched_prev_idx = static_cast<int>(j);
                }
            }

            if (best_matched_prev_idx != -1)
            {
                matches_found_active.push_back({static_cast<int>(i), best_matched_prev_idx});
                current_cluster_matched[i] = true;
                previous_cluster_is_assigned[best_matched_prev_idx] = true;
            }
        }

        // --- Step 4: Re-identify unmatched current clusters with recently_lost_clusters_ ---
        std::vector<bool> lost_cluster_reidentified(recently_lost_clusters_.size(), false);
        std::vector<std::pair<int, int>> matches_found_reid; // Stores {current_idx, lost_cluster_idx} pairs

        for (size_t i = 0; i < clusters.size(); ++i)
        {
            if (current_cluster_matched[i] || clusters[i].empty() || clusters[i].size() < static_cast<size_t>(dbscan_min_pts_)) continue; // Skip if already matched or too small

            Eigen::Vector3f current_centroid = computeCentroid(cloud, clusters[i]);
            Eigen::Vector4f min_pt_current, max_pt_current;
            pcl::getMinMax3D(*cloud, clusters[i], min_pt_current, max_pt_current);

            double current_dx = static_cast<double>(max_pt_current.x() - min_pt_current.x());
            double current_dy = static_cast<double>(max_pt_current.y() - min_pt_current.y());
            double current_dz = static_cast<double>(max_pt_current.z() - min_pt_current.z());

            int best_reidentified_lost_idx = -1;
            double min_combined_score = std::numeric_limits<double>::max(); // Use a combined score for best match

            for (size_t j = 0; j < recently_lost_clusters_.size(); ++j)
            {
                if (lost_cluster_reidentified[j]) {
                    continue; // This lost track has already been re-identified
                }
                
                // --- Positional Check ---
                Eigen::Vector3f predicted_centroid_lost(static_cast<float>(recently_lost_clusters_[j].x_k(0)),
                                                        static_cast<float>(recently_lost_clusters_[j].x_k(1)),
                                                        static_cast<float>(recently_lost_clusters_[j].x_k(2)));
                double position_dist = (current_centroid - predicted_centroid_lost).norm();

                // --- Dimension/Size Check ---
                double lost_dx = static_cast<double>(recently_lost_clusters_[j].max_bounds.x() - recently_lost_clusters_[j].min_bounds.x());
                double lost_dy = static_cast<double>(recently_lost_clusters_[j].max_bounds.y() - recently_lost_clusters_[j].min_bounds.y());
                double lost_dz = static_cast<double>(recently_lost_clusters_[j].max_bounds.z() - recently_lost_clusters_[j].min_bounds.z());

                double dimension_diff = std::abs(current_dx - lost_dx) +
                                       std::abs(current_dy - lost_dy) +
                                       std::abs(current_dz - lost_dz);

                // --- Debugging Info ---
                RCLCPP_DEBUG(this->get_logger(), "Re-ID check: Current ID N/A (centroid: %.2f,%.2f,%.2f, dims: %.2f,%.2f,%.2f) vs. Lost ID %d (pred_pos: %.2f,%.2f,%.2f, stored_dims: %.2f,%.2f,%.2f)",
                            current_centroid.x(), current_centroid.y(), current_centroid.z(),
                            current_dx, current_dy, current_dz,
                            recently_lost_clusters_[j].id,
                            predicted_centroid_lost.x(), predicted_centroid_lost.y(), predicted_centroid_lost.z(),
                            lost_dx, lost_dy, lost_dz);
                RCLCPP_DEBUG(this->get_logger(), "  -> Position Dist: %.2f (Thresh: %.2f), Dimension Diff: %.2f (Thresh: %.2f)",
                            position_dist, lost_track_position_threshold_, dimension_diff, lost_track_dimension_threshold_);


                if (position_dist > lost_track_position_threshold_) {
                    RCLCPP_DEBUG(this->get_logger(), "  -> Rejected Lost ID %d: Positional distance too high.", recently_lost_clusters_[j].id);
                    continue; // Position too far, skip this lost track
                }
                if (dimension_diff > lost_track_dimension_threshold_) {
                    RCLCPP_DEBUG(this->get_logger(), "  -> Rejected Lost ID %d: Dimension difference too high.", recently_lost_clusters_[j].id);
                    continue; // Dimensions too different, skip
                }

                // --- Combine Scores (Weighted sum, configurable via parameters) ---
                double combined_score = (position_dist * reid_position_weight_) + (dimension_diff * reid_dimension_weight_); 

                if (combined_score < min_combined_score) {
                    min_combined_score = combined_score;
                    best_reidentified_lost_idx = static_cast<int>(j);
                }
            }

            if (best_reidentified_lost_idx != -1)
            {
                matches_found_reid.push_back({static_cast<int>(i), best_reidentified_lost_idx});
                current_cluster_matched[i] = true;
                lost_cluster_reidentified[best_reidentified_lost_idx] = true;
            } else {
                 RCLCPP_DEBUG(this->get_logger(), "  -> Current cluster %lu (pos: %.2f,%.2f,%.2f) failed to re-identify any lost track.",
                              i, current_centroid.x(), current_centroid.y(), current_centroid.z());
            }
        }


        // --- Step 5: Populate 'next_previous_clusters_state' and 'next_recently_lost_clusters_state' ---
        std::vector<ClusterMemory> next_previous_clusters_state;
        std::vector<ClusterMemory> next_recently_lost_clusters_state;

        // Process matched active clusters
        for (const auto& match : matches_found_active)
        {
            int current_idx = match.first;
            int prev_idx = match.second;

            Eigen::Vector3f current_centroid = computeCentroid(cloud, clusters[current_idx]);
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, clusters[current_idx], min_pt, max_pt);

            ClusterMemory updated_mem = previous_clusters_[prev_idx];
            updated_mem.min_bounds = min_pt; // Update bounds to current measurement
            updated_mem.max_bounds = max_pt;
            updated_mem.frame_count++;
            updated_mem.missed_count = 0; // Reset missed count

            updateKalmanFilter(updated_mem, current_centroid, current_frame_time);
            
            // --- MODIFICATION: Use Kalman-filtered position for centroid ---
            updated_mem.centroid.x() = static_cast<float>(updated_mem.x_k(0));
            updated_mem.centroid.y() = static_cast<float>(updated_mem.x_k(1));
            updated_mem.centroid.z() = static_cast<float>(updated_mem.x_k(2));
            // --- END MODIFICATION ---

            next_previous_clusters_state.push_back(updated_mem);

            // Publish markers for stable, active tracks
            if (updated_mem.frame_count >= n_stable_frames_)
            {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = current_frame_time;
                marker.ns = "clusters";
                marker.id = updated_mem.id;
                marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.scale.x = marker.scale.y = marker.scale.z = 0.05;
                marker.color = updated_mem.color;
                for (const auto &index : clusters[current_idx])
                {
                    geometry_msgs::msg::Point p;
                    p.x = cloud->points[index].x;
                    p.y = cloud->points[index].y;
                    p.z = cloud->points[index].z;
                    marker.points.push_back(p);
                }
                marker_array.markers.push_back(marker);

                visualization_msgs::msg::Marker bbox_marker;
                bbox_marker.header.frame_id = frame_id;
                bbox_marker.header.stamp = current_frame_time;
                bbox_marker.ns = "bounding_boxes";
                bbox_marker.id = updated_mem.id;
                bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
                bbox_marker.action = visualization_msgs::msg::Marker::ADD;
                // --- MODIFICATION: Use Kalman-filtered position for bounding box center ---
                bbox_marker.pose.position.x = static_cast<float>(updated_mem.x_k(0));
                bbox_marker.pose.position.y = static_cast<float>(updated_mem.x_k(1));
                bbox_marker.pose.position.z = static_cast<float>(updated_mem.x_k(2));
                // --- END MODIFICATION ---
                bbox_marker.scale.x = std::max(0.01f, max_pt.x() - min_pt.x());
                bbox_marker.scale.y = std::max(0.01f, max_pt.y() - min_pt.y());
                bbox_marker.scale.z = std::max(0.01f, max_pt.z() - min_pt.z());
                bbox_marker.color = updated_mem.color;
                bbox_marker.color.a = 0.5f;
                bounding_box_array.markers.push_back(bbox_marker);

                visualization_msgs::msg::Marker text_marker;
                text_marker.header.frame_id = frame_id;
                text_marker.header.stamp = current_frame_time;
                text_marker.ns = "cluster_ids";
                text_marker.id = updated_mem.id;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                // --- MODIFICATION: Use Kalman-filtered position for text marker ---
                text_marker.pose.position.x = static_cast<float>(updated_mem.x_k(0));
                text_marker.pose.position.y = static_cast<float>(updated_mem.x_k(1));
                text_marker.pose.position.z = static_cast<float>(updated_mem.x_k(2)) + 0.1f; // Place text slightly above filtered position
                // --- END MODIFICATION ---
                text_marker.scale.z = 0.1; // Text height
                text_marker.color = updated_mem.color;
                text_marker.color.a = 1.0f;
                text_marker.text = "ID: " + std::to_string(updated_mem.id) +
                                   "\nF: " + std::to_string(updated_mem.frame_count) +
                                   "\nM: " + std::to_string(updated_mem.missed_count);
                marker_array.markers.push_back(text_marker);
            }
        }

        // Process re-identified lost clusters
        for (const auto& match : matches_found_reid)
        {
            int current_idx = match.first;
            int lost_idx = match.second;

            Eigen::Vector3f current_centroid = computeCentroid(cloud, clusters[current_idx]);
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, clusters[current_idx], min_pt, max_pt);

            ClusterMemory reidentified_mem = recently_lost_clusters_[lost_idx];
            reidentified_mem.min_bounds = min_pt; // Update bounds to current measurement
            reidentified_mem.max_bounds = max_pt;
            reidentified_mem.frame_count++; // Continue frame count
            reidentified_mem.missed_count = 0; // Reset missed count

            updateKalmanFilter(reidentified_mem, current_centroid, current_frame_time);
            
            // --- MODIFICATION: Use Kalman-filtered position for centroid ---
            reidentified_mem.centroid.x() = static_cast<float>(reidentified_mem.x_k(0));
            reidentified_mem.centroid.y() = static_cast<float>(reidentified_mem.x_k(1));
            reidentified_mem.centroid.z() = static_cast<float>(reidentified_mem.x_k(2));
            // --- END MODIFICATION ---

            next_previous_clusters_state.push_back(reidentified_mem); // Add back to active tracks

            RCLCPP_INFO(this->get_logger(), "Track %d RE-IDENTIFIED (current frame count: %d).", reidentified_mem.id, reidentified_mem.frame_count);

            if (reidentified_mem.frame_count >= n_stable_frames_) {
                // Publish point cloud marker (new points)
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id;
                marker.header.stamp = current_frame_time;
                marker.ns = "clusters";
                marker.id = reidentified_mem.id;
                marker.type = visualization_msgs::msg::Marker::SPHERE_LIST;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.scale.x = marker.scale.y = marker.scale.z = 0.05;
                marker.color = reidentified_mem.color;
                for (const auto &index : clusters[current_idx])
                {
                    geometry_msgs::msg::Point p;
                    p.x = cloud->points[index].x;
                    p.y = cloud->points[index].y;
                    p.z = cloud->points[index].z;
                    marker.points.push_back(p);
                }
                marker_array.markers.push_back(marker);

                // Publish bounding box
                visualization_msgs::msg::Marker bbox_marker;
                bbox_marker.header.frame_id = frame_id;
                bbox_marker.header.stamp = current_frame_time;
                bbox_marker.ns = "bounding_boxes";
                bbox_marker.id = reidentified_mem.id;
                bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
                bbox_marker.action = visualization_msgs::msg::Marker::ADD;
                // --- MODIFICATION: Use Kalman-filtered position for bounding box center ---
                bbox_marker.pose.position.x = static_cast<float>(reidentified_mem.x_k(0));
                bbox_marker.pose.position.y = static_cast<float>(reidentified_mem.x_k(1));
                bbox_marker.pose.position.z = static_cast<float>(reidentified_mem.x_k(2));
                // --- END MODIFICATION ---
                bbox_marker.scale.x = std::max(0.01f, max_pt.x() - min_pt.x());
                bbox_marker.scale.y = std::max(0.01f, max_pt.y() - min_pt.y());
                bbox_marker.scale.z = std::max(0.01f, max_pt.z() - min_pt.z());
                bbox_marker.color = reidentified_mem.color;
                bbox_marker.color.a = 0.5f;
                bounding_box_array.markers.push_back(bbox_marker);

                // Publish text
                visualization_msgs::msg::Marker text_marker;
                text_marker.header.frame_id = frame_id;
                text_marker.header.stamp = current_frame_time;
                text_marker.ns = "cluster_ids";
                text_marker.id = reidentified_mem.id;
                text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                text_marker.action = visualization_msgs::msg::Marker::ADD;
                text_marker.pose.position.x = static_cast<float>(reidentified_mem.x_k(0)); // Cast Kalman state to float for ROS msg
                text_marker.pose.position.y = static_cast<float>(reidentified_mem.x_k(1));
                text_marker.pose.position.z = static_cast<float>(reidentified_mem.x_k(2)) + 0.1f;
                text_marker.scale.z = 0.1;
                text_marker.color = reidentified_mem.color;
                text_marker.color.a = 1.0f;
                text_marker.text = "ID: " + std::to_string(reidentified_mem.id) +
                                   "\nF: " + std::to_string(reidentified_mem.frame_count) +
                                   "\nM: " + std::to_string(reidentified_mem.missed_count) + " (Re-ID)";
                marker_array.markers.push_back(text_marker);
            }
        }

        // Process new clusters (unmatched current clusters after active and lost checks)
        for (size_t i = 0; i < clusters.size(); ++i)
        {
            if (!current_cluster_matched[i] && clusters[i].size() >= static_cast<size_t>(dbscan_min_pts_)) // Only consider new, valid clusters
            {
                Eigen::Vector3f current_centroid = computeCentroid(cloud, clusters[i]);
                Eigen::Vector4f min_pt, max_pt;
                pcl::getMinMax3D(*cloud, clusters[i], min_pt, max_pt);

                ClusterMemory new_mem;
                new_mem.color = generateRandomColor(); // Assign a new random color
                new_mem.id = cluster_id_counter_++;
                new_mem.frame_count = 1;
                new_mem.missed_count = 0;
                new_mem.min_bounds = min_pt;
                new_mem.max_bounds = max_pt;

                initKalmanFilter(new_mem, current_centroid, current_frame_time);

                // --- MODIFICATION: Use Kalman-filtered position for centroid for new tracks ---
                new_mem.centroid.x() = static_cast<float>(new_mem.x_k(0));
                new_mem.centroid.y() = static_cast<float>(new_mem.x_k(1));
                new_mem.centroid.z() = static_cast<float>(new_mem.x_k(2));
                // --- END MODIFICATION ---

                next_previous_clusters_state.push_back(new_mem);
                RCLCPP_INFO(this->get_logger(), "New Track %d created.", new_mem.id);
            }
        }

        // --- Step 6: Handle tracks that were NOT matched in this frame ---
        // Active tracks that were not matched
        for (size_t j = 0; j < previous_clusters_.size(); ++j)
        {
            if (!previous_cluster_is_assigned[j])
            {
                ClusterMemory missed_track = previous_clusters_[j];

                // For missed tracks, the current_kalman_vel_noise_q should increase to reflect growing uncertainty
                missed_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_, // Capped at initial max
                                                                    missed_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_); // Invert decay
                
                // --- MODIFICATION: Update missed_track.centroid to its predicted Kalman position ---
                missed_track.centroid.x() = static_cast<float>(missed_track.x_k(0));
                missed_track.centroid.y() = static_cast<float>(missed_track.x_k(1));
                missed_track.centroid.z() = static_cast<float>(missed_track.x_k(2));
                // --- END MODIFICATION ---

                // If missed active track is still within its tolerance, keep it in active list
                if (missed_track.missed_count <= n_missed_frames_)
                {
                    next_previous_clusters_state.push_back(missed_track); // Keep it active
                    
                    // If stable, publish its faded predicted markers (indicating it's still tracked but not seen)
                    if (missed_track.frame_count >= n_stable_frames_) {
                        visualization_msgs::msg::Marker bbox_marker;
                        bbox_marker.header.frame_id = frame_id;
                        bbox_marker.header.stamp = current_frame_time;
                        bbox_marker.ns = "bounding_boxes";
                        bbox_marker.id = missed_track.id;
                        bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
                        bbox_marker.action = visualization_msgs::msg::Marker::ADD;
                        bbox_marker.pose.position.x = static_cast<float>(missed_track.x_k(0)); // Cast to float
                        bbox_marker.pose.position.y = static_cast<float>(missed_track.x_k(1)); // Cast to float
                        bbox_marker.pose.position.z = static_cast<float>(missed_track.x_k(2)); // Cast to float
                        // Use stored bounds for scale, as current measurement is missing
                        bbox_marker.scale.x = std::max(0.01f, missed_track.max_bounds.x() - missed_track.min_bounds.x());
                        bbox_marker.scale.y = std::max(0.01f, missed_track.max_bounds.y() - missed_track.min_bounds.y());
                        bbox_marker.scale.z = std::max(0.01f, missed_track.max_bounds.z() - missed_track.min_bounds.z());
                        bbox_marker.color = missed_track.color;
                        bbox_marker.color.a = 0.2f; // Faded
                        bounding_box_array.markers.push_back(bbox_marker);

                        visualization_msgs::msg::Marker text_marker;
                        text_marker.header.frame_id = frame_id;
                        text_marker.header.stamp = current_frame_time;
                        text_marker.ns = "cluster_ids";
                        text_marker.id = missed_track.id;
                        text_marker.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
                        text_marker.action = visualization_msgs::msg::Marker::ADD;
                        text_marker.pose.position.x = static_cast<float>(missed_track.x_k(0));
                        text_marker.pose.position.y = static_cast<float>(missed_track.x_k(1));
                        text_marker.pose.position.z = static_cast<float>(missed_track.x_k(2)) + 0.1f;
                        text_marker.scale.z = 0.1;
                        text_marker.color = missed_track.color;
                        text_marker.color.a = 0.5f; // Faded text
                        text_marker.text = "ID: " + std::to_string(missed_track.id) +
                                           "\nF: " + std::to_string(missed_track.frame_count) +
                                           "\nM: " + std::to_string(missed_track.missed_count);
                        marker_array.markers.push_back(text_marker);

                        // Ensure the sphere list marker for this ID is removed if it was previously active.
                        // We only want to see the bounding box and text for missed tracks, not the points.
                        visualization_msgs::msg::Marker marker_delete_points;
                        marker_delete_points.header.frame_id = frame_id;
                        marker_delete_points.header.stamp = current_frame_time;
                        marker_delete_points.ns = "clusters";
                        marker_delete_points.id = missed_track.id;
                        marker_delete_points.action = visualization_msgs::msg::Marker::DELETE;
                        marker_array.markers.push_back(marker_delete_points);
                    }
                } else {
                    // Missed too many frames for an active track, move to recently_lost_clusters_
                    next_recently_lost_clusters_state.push_back(missed_track);
                    RCLCPP_INFO(this->get_logger(), "Track %d moved to recently lost (missed %d frames).", missed_track.id, missed_track.missed_count);
                    
                    // Delete its active markers so it doesn't show up in 'active' viz but rather in 'lost' viz later
                    visualization_msgs::msg::Marker marker_delete_points;
                    marker_delete_points.header.frame_id = frame_id;
                    marker_delete_points.header.stamp = current_frame_time;
                    marker_delete_points.ns = "clusters";
                    marker_delete_points.id = missed_track.id;
                    marker_delete_points.action = visualization_msgs::msg::Marker::DELETE;
                    marker_array.markers.push_back(marker_delete_points);

                    visualization_msgs::msg::Marker bbox_delete;
                    bbox_delete.header.frame_id = frame_id;
                    bbox_delete.header.stamp = current_frame_time;
                    bbox_delete.ns = "bounding_boxes";
                    bbox_delete.id = missed_track.id;
                    bbox_delete.action = visualization_msgs::msg::Marker::DELETE;
                    bounding_box_array.markers.push_back(bbox_delete);

                    visualization_msgs::msg::Marker text_delete;
                    text_delete.header.frame_id = frame_id;
                    text_delete.header.stamp = current_frame_time;
                    text_delete.ns = "cluster_ids";
                    text_delete.id = missed_track.id;
                    text_delete.action = visualization_msgs::msg::Marker::DELETE;
                    marker_array.markers.push_back(text_delete);
                }
            }
        }

        // Lost tracks that were not re-identified and are still within MAX_LOST_FRAMES
        for (size_t j = 0; j < recently_lost_clusters_.size(); ++j)
        {
            if (!lost_cluster_reidentified[j])
            {
                ClusterMemory lost_track = recently_lost_clusters_[j];

                // For lost tracks, the current_kalman_vel_noise_q should also increase
                lost_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_,
                                                                 lost_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_);
                
                // --- MODIFICATION: Update lost_track.centroid to its predicted Kalman position ---
                lost_track.centroid.x() = static_cast<float>(lost_track.x_k(0));
                lost_track.centroid.y() = static_cast<float>(lost_track.x_k(1));
                lost_track.centroid.z() = static_cast<float>(lost_track.x_k(2));
                // --- END MODIFICATION ---

                if (lost_track.missed_count <= max_lost_frames_)
                {
                    next_recently_lost_clusters_state.push_back(lost_track); // Keep it in the lost pool

                    // Visualize lost tracks even more faded
                    if (lost_track.frame_count >= n_stable_frames_) {
                        visualization_msgs::msg::Marker bbox_marker;
                        bbox_marker.header.frame_id = frame_id;
                        bbox_marker.header.stamp = current_frame_time;
                        bbox_marker.ns = "bounding_boxes";
                        bbox_marker.id = lost_track.id;
                        bbox_marker.type = visualization_msgs::msg::Marker::CUBE;
                        bbox_marker.action = visualization_msgs::msg::Marker::ADD;
                        bbox_marker.pose.position.x = static_cast<float>(lost_track.x_k(0));
                        bbox_marker.pose.position.y = static_cast<float>(lost_track.x_k(1));
                        bbox_marker.pose.position.z = static_cast<float>(lost_track.x_k(2));
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
                        text_marker.pose.position.x = static_cast<float>(lost_track.x_k(0));
                        text_marker.pose.position.y = static_cast<float>(lost_track.x_k(1));
                        text_marker.pose.position.z = static_cast<float>(lost_track.x_k(2)) + 0.1f;
                        text_marker.scale.z = 0.05; // Smaller text for lost
                        text_marker.color = lost_track.color;
                        text_marker.color.a = 0.2f; // Faded text
                        text_marker.text = "ID: " + std::to_string(lost_track.id) + " (Lost: " + std::to_string(lost_track.missed_count) + ")";
                        marker_array.markers.push_back(text_marker);
                        
                        // Ensure point cloud marker is deleted
                        visualization_msgs::msg::Marker marker_delete_points;
                        marker_delete_points.header.frame_id = frame_id;
                        marker_delete_points.header.stamp = current_frame_time;
                        marker_delete_points.ns = "clusters";
                        marker_delete_points.id = lost_track.id;
                        marker_delete_points.action = visualization_msgs::msg::Marker::DELETE;
                        marker_array.markers.push_back(marker_delete_points);
                    }
                } else {
                    // Track is permanently forgotten. Delete all its markers.
                    RCLCPP_INFO(this->get_logger(), "Track %d permanently forgotten (missed %d frames).", lost_track.id, lost_track.missed_count);
                    
                    visualization_msgs::msg::Marker marker_delete_cluster;
                    marker_delete_cluster.header.frame_id = frame_id;
                    marker_delete_cluster.header.stamp = current_frame_time;
                    marker_delete_cluster.ns = "clusters";
                    marker_delete_cluster.id = lost_track.id;
                    marker_delete_cluster.action = visualization_msgs::msg::Marker::DELETE;
                    marker_array.markers.push_back(marker_delete_cluster);

                    visualization_msgs::msg::Marker bbox_delete;
                    bbox_delete.header.frame_id = frame_id;
                    bbox_delete.header.stamp = current_frame_time;
                    bbox_delete.ns = "bounding_boxes";
                    bbox_delete.id = lost_track.id;
                    bbox_delete.action = visualization_msgs::msg::Marker::DELETE;
                    bounding_box_array.markers.push_back(bbox_delete);

                    visualization_msgs::msg::Marker text_delete;
                    text_delete.header.frame_id = frame_id;
                    text_delete.header.stamp = current_frame_time;
                    text_delete.ns = "cluster_ids";
                    text_delete.id = lost_track.id;
                    text_delete.action = visualization_msgs::msg::Marker::DELETE;
                    marker_array.markers.push_back(text_delete);
                }
            }
        }

        // Update the previous_clusters_ and recently_lost_clusters_ with the new states.
        previous_clusters_ = std::move(next_previous_clusters_state);
        recently_lost_clusters_ = std::move(next_recently_lost_clusters_state);

        // Populate and publish the custom ClusterArray message
        for (const auto& cluster_mem : previous_clusters_) {
            my_cluster_odom::msg::Cluster cluster_msg;
            cluster_msg.id = cluster_mem.id;
            // --- MODIFICATION: Publish Kalman-filtered centroid in the ROS message ---
            cluster_msg.centroid.x = cluster_mem.centroid.x();
            cluster_msg.centroid.y = cluster_mem.centroid.y();
            cluster_msg.centroid.z = cluster_mem.centroid.z();
            // --- END MODIFICATION ---
            
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

            // Copy Eigen states to fixed-size arrays for ROS message (ensure type consistency)
            for (int i = 0; i < 6; ++i) {
                cluster_msg.kalman_state[i] = static_cast<float>(cluster_mem.x_k(i)); // Cast to float
            }
            for (int i = 0; i < 6; ++i) {
                for (int j = 0; j < 6; ++j) {
                    cluster_msg.kalman_covariance[i * 6 + j] = static_cast<float>(cluster_mem.P_k(i, j)); // Cast to float
                }
            }
            cluster_data_array_msg.clusters.push_back(cluster_msg);
        }

        cluster_viz_publisher_->publish(marker_array);
        bounding_box_viz_publisher_->publish(bounding_box_array);
        cluster_data_publisher_->publish(cluster_data_array_msg); // Publish the custom message
    }

};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<DBSCANNode>());
    rclcpp::shutdown();
    return 0;
}