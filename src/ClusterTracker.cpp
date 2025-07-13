#include "my_cluster_odom/ClusterTracker.hpp"

namespace dbscan_clusterer {

ClusterTracker::ClusterTracker(rclcpp::Logger logger,
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
                               double reid_combined_threshold
                            )
    : logger_(logger),
      kalman_pos_noise_q_(kalman_pos_noise_q),
      kalman_vel_noise_q_(kalman_vel_noise_q),
      kalman_min_vel_noise_q_(kalman_min_vel_noise_q),
      kalman_vel_noise_q_decay_factor_(kalman_vel_noise_q_decay_factor),
      kalman_measurement_noise_r_(kalman_measurement_noise_r),
      active_track_match_distance_threshold_(active_track_match_distance_threshold),
      lost_track_position_threshold_(lost_track_position_threshold),
      lost_track_dimension_threshold_(lost_track_dimension_threshold),
      reid_position_weight_(reid_position_weight),
      reid_dimension_weight_(reid_dimension_weight),
      n_stable_frames_(n_stable_frames),
      n_missed_frames_(n_missed_frames),
      max_lost_frames_(max_lost_frames),
      reid_combined_threshold_(reid_combined_threshold) // <--- Added this line
{
    rand_gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());

    // Initialize constant Kalman filter matrices
    R_ = Eigen::MatrixXd::Identity(3, 3) * kalman_measurement_noise_r_; // Measurement noise

    H_ = Eigen::MatrixXd::Zero(3, 6);
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;
}

void ClusterTracker::initKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& initial_position, rclcpp::Time current_time)
{
    cluster_mem.x_k.resize(6);
    cluster_mem.x_k << initial_position.x(), initial_position.y(), initial_position.z(), 0.0, 0.0, 0.0; // Initial velocity is zero

    cluster_mem.P_k = Eigen::MatrixXd::Identity(6, 6);
    cluster_mem.P_k.block<3,3>(0,0) = Eigen::Matrix3d::Identity() * 0.1; // Initial position uncertainty
    cluster_mem.P_k.block<3,3>(3,3) = Eigen::Matrix3d::Identity() * 10.0; // Initial velocity uncertainty (high, because we don't know it)

    cluster_mem.last_update_time = current_time;
    cluster_mem.current_kalman_vel_noise_q = kalman_vel_noise_q_; // Initialize with max noise
}

void ClusterTracker::predictKalmanFilter(ClusterMemory& cluster_mem, double dt)
{
    if (dt <= 0) {
        return;
    }

    Eigen::MatrixXd F(6, 6);
    F << 1, 0, 0, dt, 0, 0,
         0, 1, 0, 0, dt, 0,
         0, 0, 1, 0, 0, dt,
         0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 1, 0,
         0, 0, 0, 0, 0, 1;

    Eigen::MatrixXd current_Q = Eigen::MatrixXd::Identity(6, 6);
    current_Q.block<3, 3>(0, 0) = Eigen::Matrix3d::Identity() * kalman_pos_noise_q_;
    current_Q.block<3, 3>(3, 3) = Eigen::Matrix3d::Identity() * cluster_mem.current_kalman_vel_noise_q;

    cluster_mem.x_k = F * cluster_mem.x_k;
    cluster_mem.P_k = F * cluster_mem.P_k * F.transpose() + current_Q;
}

void ClusterTracker::updateKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& measurement, rclcpp::Time current_time)
{
    Eigen::VectorXd z(3);
    z << measurement.x(), measurement.y(), measurement.z();

    Eigen::VectorXd y = z - H_ * cluster_mem.x_k;
    Eigen::MatrixXd S = H_ * cluster_mem.P_k * H_.transpose() + R_;
    Eigen::MatrixXd K = cluster_mem.P_k * H_.transpose() * S.inverse();

    cluster_mem.x_k = cluster_mem.x_k + K * y;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    cluster_mem.P_k = (I - K * H_) * cluster_mem.P_k;

    cluster_mem.last_update_time = current_time;

    cluster_mem.current_kalman_vel_noise_q = std::max(kalman_min_vel_noise_q_,
                                                      cluster_mem.current_kalman_vel_noise_q * kalman_vel_noise_q_decay_factor_);
}

Eigen::Vector3f ClusterTracker::computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices)
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

std_msgs::msg::ColorRGBA ClusterTracker::generateRandomColor()
{
    std_msgs::msg::ColorRGBA color;
    std::uniform_real_distribution<> dist(0.0, 1.0);
    color.r = dist(rand_gen_);
    color.g = dist(rand_gen_);
    color.b = dist(rand_gen_);
    if (color.r + color.g + color.b < 0.8) {
        float sum_rgb = color.r + color.g + color.b;
        if (sum_rgb < 1e-6f) sum_rgb = 1e-6f;
        float scale_factor = 0.8f / sum_rgb;
        color.r *= scale_factor;
        color.g *= scale_factor;
        color.b *= scale_factor;
    }
    color.a = 1.0f;
    return color;
}


std::vector<ClusterMemory> ClusterTracker::processClusters(
    rclcpp::Time current_frame_time,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<std::vector<int>>& current_detected_clusters,
    std::vector<ClusterMemory>& active_tracks,
    std::vector<ClusterMemory>& recently_lost_tracks)
{
    // Declare variables at the beginning of the function
    std::vector<ClusterMemory> next_active_tracks_state;
    std::vector<ClusterMemory> next_recently_lost_tracks_state;
    std::vector<ClusterMemory> stable_published_tracks; // Tracks that will be published as output

    // These variables need to be declared before they are used in the matching loops
    std::vector<bool> current_cluster_matched(current_detected_clusters.size(), false);
    std::vector<bool> active_track_is_assigned(active_tracks.size(), false);
    std::vector<std::pair<int, int>> matches_found_active;

    // --- Step 1: Predict positions of active tracks ---
    for (auto& prev_cluster : active_tracks) {
        prev_cluster.missed_count++;
        double dt = (current_frame_time - prev_cluster.last_update_time).seconds();
        if (dt < 0) {
            RCLCPP_WARN(logger_, "Negative dt detected for active track %d (%f). Clamping to 0.", prev_cluster.id, dt);
            dt = 0;
        }
        predictKalmanFilter(prev_cluster, dt);
    }

    // --- Step 2: Predict positions for recently lost tracks ---
    for (auto& lost_cluster : recently_lost_tracks) {
        lost_cluster.missed_count++;
        double dt = (current_frame_time - lost_cluster.last_update_time).seconds();
        if (dt < 0) {
            RCLCPP_WARN(logger_, "Negative dt detected for lost track %d (%f). Clamping to 0.", lost_cluster.id, dt);
            dt = 0;
        }
        predictKalmanFilter(lost_cluster, dt);
    }

    // --- Step 3: Match current clusters with active tracks ---
    for (size_t i = 0; i < current_detected_clusters.size(); ++i)
    {
        if (current_detected_clusters[i].empty()) continue; // Should have been filtered by refinement

        Eigen::Vector3f current_centroid = computeCentroid(cloud, current_detected_clusters[i]);

        int best_matched_active_idx = -1;
        double min_dist = std::numeric_limits<double>::max();

        for (size_t j = 0; j < active_tracks.size(); ++j)
        {
            if (active_track_is_assigned[j]) {
                continue;
            }

            Eigen::Vector3f predicted_centroid_active(static_cast<float>(active_tracks[j].x_k(0)),
                                                      static_cast<float>(active_tracks[j].x_k(1)),
                                                      static_cast<float>(active_tracks[j].x_k(2)));

            double dist = (current_centroid - predicted_centroid_active).norm();

            if (dist < min_dist && dist < active_track_match_distance_threshold_)
            {
                min_dist = dist;
                best_matched_active_idx = static_cast<int>(j);
            }
        }

        if (best_matched_active_idx != -1)
        {
            matches_found_active.push_back({static_cast<int>(i), best_matched_active_idx});
            current_cluster_matched[i] = true;
            active_track_is_assigned[best_matched_active_idx] = true;
        }
    }

    // --- Step 4: Re-identify unmatched current clusters with recently_lost_tracks_ ---
    std::vector<bool> lost_track_reidentified(recently_lost_tracks.size(), false);
    std::vector<std::pair<int, int>> matches_found_reid;

    for (size_t i = 0; i < current_detected_clusters.size(); ++i)
    {
        if (current_cluster_matched[i] || current_detected_clusters[i].empty()) continue;

        Eigen::Vector3f current_centroid = computeCentroid(cloud, current_detected_clusters[i]);
        Eigen::Vector4f min_pt_current, max_pt_current;
        pcl::getMinMax3D(*cloud, current_detected_clusters[i], min_pt_current, max_pt_current);

        double current_dx = static_cast<double>(max_pt_current.x() - min_pt_current.x());
        double current_dy = static_cast<double>(max_pt_current.y() - min_pt_current.y());
        double current_dz = static_cast<double>(max_pt_current.z() - min_pt_current.z());

        int best_reidentified_lost_idx = -1;
        double min_combined_score = std::numeric_limits<double>::max();

        for (size_t j = 0; j < recently_lost_tracks.size(); ++j)
        {
            if (lost_track_reidentified[j]) {
                continue;
            }
            
            Eigen::Vector3f predicted_centroid_lost(static_cast<float>(recently_lost_tracks[j].x_k(0)),
                                                    static_cast<float>(recently_lost_tracks[j].x_k(1)),
                                                    static_cast<float>(recently_lost_tracks[j].x_k(2)));
            double position_dist = (current_centroid - predicted_centroid_lost).norm();

            double lost_dx = static_cast<double>(recently_lost_tracks[j].max_bounds.x() - recently_lost_tracks[j].min_bounds.x());
            double lost_dy = static_cast<double>(recently_lost_tracks[j].max_bounds.y() - recently_lost_tracks[j].min_bounds.y());
            double lost_dz = static_cast<double>(recently_lost_tracks[j].max_bounds.z() - recently_lost_tracks[j].min_bounds.z());

            double dimension_diff = std::abs(current_dx - lost_dx) +
                                   std::abs(current_dy - lost_dy) +
                                   std::abs(current_dz - lost_dz);

            double weighted_reid_dist = (position_dist * reid_position_weight_) + (dimension_diff * reid_dimension_weight_);

            RCLCPP_DEBUG(logger_, "Re-ID check: Current ID N/A (centroid: %.2f,%.2f,%.2f, dims: %.2f,%.2f,%.2f) vs. Lost ID %d (pred_pos: %.2f,%.2f,%.2f, stored_dims: %.2f,%.2f,%.2f)",
                        current_centroid.x(), current_centroid.y(), current_centroid.z(),
                        current_dx, current_dy, current_dz,
                        recently_lost_tracks[j].id,
                        predicted_centroid_lost.x(), predicted_centroid_lost.y(), predicted_centroid_lost.z(),
                        lost_dx, lost_dy, lost_dz);
            RCLCPP_DEBUG(logger_, "  -> Weighted Re-ID Dist: %.2f (Thresh: %.2f)",
                        weighted_reid_dist, reid_combined_threshold_);

            // Refined re-identification condition
            if (weighted_reid_dist < reid_combined_threshold_ && weighted_reid_dist < min_combined_score) // <--- Modified this line
            {
                min_combined_score = weighted_reid_dist; // <--- Added this line
                best_reidentified_lost_idx = static_cast<int>(j); // <--- Added this line
            }
        }
        
        // After iterating through all lost tracks, if a best match was found
        if (best_reidentified_lost_idx != -1) { // <--- Modified this line
            matches_found_reid.push_back({static_cast<int>(i), best_reidentified_lost_idx}); // <--- Modified this line
            current_cluster_matched[i] = true;
            lost_track_reidentified[best_reidentified_lost_idx] = true; // <--- Modified this line
            RCLCPP_DEBUG(logger_, "Re-identified cluster (idx %lu) with lost track %d. Weighted dist: %.2f", i, recently_lost_tracks[best_reidentified_lost_idx].id, min_combined_score); // <--- Modified this line
        }
    }

    // --- Step 5: Populate 'next_active_tracks_state' and 'next_recently_lost_tracks_state' ---

    // Process matched active clusters
    for (const auto& match : matches_found_active)
    {
        int current_idx = match.first;
        int active_idx = match.second;

        Eigen::Vector3f current_centroid = computeCentroid(cloud, current_detected_clusters[current_idx]);
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, current_detected_clusters[current_idx], min_pt, max_pt);

        ClusterMemory updated_mem = active_tracks[active_idx];
        updated_mem.min_bounds = min_pt;
        updated_mem.max_bounds = max_pt;
        updated_mem.frame_count++;
        updated_mem.missed_count = 0;

        updateKalmanFilter(updated_mem, current_centroid, current_frame_time);
        
        updated_mem.centroid.x() = static_cast<float>(updated_mem.x_k(0));
        updated_mem.centroid.y() = static_cast<float>(updated_mem.x_k(1));
        updated_mem.centroid.z() = static_cast<float>(updated_mem.x_k(2));

        next_active_tracks_state.push_back(updated_mem);

        if (updated_mem.frame_count >= n_stable_frames_) {
            stable_published_tracks.push_back(updated_mem);
        }
    }

    // Process re-identified lost clusters
    for (const auto& match : matches_found_reid)
    {
        int current_idx = match.first;
        int lost_idx = match.second;

        Eigen::Vector3f current_centroid = computeCentroid(cloud, current_detected_clusters[current_idx]);
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, current_detected_clusters[current_idx], min_pt, max_pt);

        ClusterMemory reidentified_mem = recently_lost_tracks[lost_idx];
        reidentified_mem.min_bounds = min_pt;
        reidentified_mem.max_bounds = max_pt;
        reidentified_mem.frame_count++;
        reidentified_mem.missed_count = 0;

        updateKalmanFilter(reidentified_mem, current_centroid, current_frame_time);
        
        reidentified_mem.centroid.x() = static_cast<float>(reidentified_mem.x_k(0));
        reidentified_mem.centroid.y() = static_cast<float>(reidentified_mem.x_k(1));
        reidentified_mem.centroid.z() = static_cast<float>(reidentified_mem.x_k(2));

        next_active_tracks_state.push_back(reidentified_mem);
        RCLCPP_INFO(logger_, "Track %d RE-IDENTIFIED (current frame count: %d).", reidentified_mem.id, reidentified_mem.frame_count);

        if (reidentified_mem.frame_count >= n_stable_frames_) {
            stable_published_tracks.push_back(reidentified_mem);
        }
    }

    // Process new clusters
    for (size_t i = 0; i < current_detected_clusters.size(); ++i)
    {
        if (!current_cluster_matched[i])
        {
            Eigen::Vector3f current_centroid = computeCentroid(cloud, current_detected_clusters[i]);
            Eigen::Vector4f min_pt, max_pt;
            pcl::getMinMax3D(*cloud, current_detected_clusters[i], min_pt, max_pt);

            ClusterMemory new_mem;
            new_mem.color = generateRandomColor();
            new_mem.id = getNextClusterId();
            new_mem.frame_count = 1;
            new_mem.missed_count = 0;
            new_mem.min_bounds = min_pt;
            new_mem.max_bounds = max_pt;

            initKalmanFilter(new_mem, current_centroid, current_frame_time);

            new_mem.centroid.x() = static_cast<float>(new_mem.x_k(0));
            new_mem.centroid.y() = static_cast<float>(new_mem.x_k(1));
            new_mem.centroid.z() = static_cast<float>(new_mem.x_k(2));

            next_active_tracks_state.push_back(new_mem);
            RCLCPP_INFO(logger_, "New Track %d created.", new_mem.id);
        }
    }

    // --- Step 6: Handle tracks that were NOT matched in this frame ---
    // Active tracks that were not matched
    for (size_t j = 0; j < active_tracks.size(); ++j)
    {
        if (!active_track_is_assigned[j])
        {
            ClusterMemory missed_track = active_tracks[j];

            missed_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_,
                                                                missed_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_);
            
            missed_track.centroid.x() = static_cast<float>(missed_track.x_k(0));
            missed_track.centroid.y() = static_cast<float>(missed_track.x_k(1));
            missed_track.centroid.z() = static_cast<float>(missed_track.x_k(2));

            if (missed_track.missed_count <= n_missed_frames_)
            {
                next_active_tracks_state.push_back(missed_track);
                // If stable, we would mark it for publishing with faded markers by the node
                if (missed_track.frame_count >= n_stable_frames_) {
                    stable_published_tracks.push_back(missed_track); // Still publish its info even if missed
                }
            } else {
                next_recently_lost_tracks_state.push_back(missed_track);
                RCLCPP_INFO(logger_, "Track %d moved to recently lost (missed %d frames).", missed_track.id, missed_track.missed_count);
            }
        }
    }

    // Lost tracks that were not re-identified and are still within MAX_LOST_FRAMES
    for (size_t j = 0; j < recently_lost_tracks.size(); ++j)
    {
        if (!lost_track_reidentified[j])
        {
            ClusterMemory lost_track = recently_lost_tracks[j];

            lost_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_,
                                                             lost_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_);
            
            lost_track.centroid.x() = static_cast<float>(lost_track.x_k(0));
            lost_track.centroid.y() = static_cast<float>(lost_track.x_k(1));
            lost_track.centroid.z() = static_cast<float>(lost_track.x_k(2));

            if (lost_track.missed_count <= max_lost_frames_)
            {
                next_recently_lost_tracks_state.push_back(lost_track);
                if (lost_track.frame_count >= n_stable_frames_) {
                    stable_published_tracks.push_back(lost_track); // Still publish its info
                }
            } else {
                RCLCPP_INFO(logger_, "Track %d permanently forgotten (missed %d frames).", lost_track.id, lost_track.missed_count);
            }
        }
    }

    active_tracks = std::move(next_active_tracks_state);
    recently_lost_tracks = std::move(next_recently_lost_tracks_state);

    return stable_published_tracks;
}

} // namespace dbscan_clusterer