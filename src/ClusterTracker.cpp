#include "my_cluster_odom/ClusterTracker.hpp"
#include "my_cluster_odom/hungarian.hpp" // Include the Hungarian algorithm header
#include <map> // Required for std::map in processClusters
#include <sstream> // For building string streams for logging Eigen matrices

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
      reid_combined_threshold_(reid_combined_threshold)
{
    rand_gen_.seed(std::chrono::system_clock::now().time_since_epoch().count());

    // Initialize constant Kalman filter matrices
    R_ = Eigen::MatrixXd::Identity(3, 3) * kalman_measurement_noise_r_; // Measurement noise

    H_ = Eigen::MatrixXd::Zero(3, 6);
    H_(0, 0) = 1.0;
    H_(1, 1) = 1.0;
    H_(2, 2) = 1.0;

    RCLCPP_INFO(logger_, "ClusterTracker initialized with parameters:");
    RCLCPP_INFO(logger_, "  Kalman Q (pos/vel): %f / %f (min: %f, decay: %f)", kalman_pos_noise_q_, kalman_vel_noise_q_, kalman_min_vel_noise_q_, kalman_vel_noise_q_decay_factor_);
    RCLCPP_INFO(logger_, "  Kalman R (meas): %f", kalman_measurement_noise_r_);
    RCLCPP_INFO(logger_, "  Active Match Dist Threshold: %f", active_track_match_distance_threshold_);
    RCLCPP_INFO(logger_, "  Lost Pos/Dim Thresholds: %f / %f", lost_track_position_threshold_, lost_track_dimension_threshold_);
    RCLCPP_INFO(logger_, "  Re-ID Weights (pos/dim): %f / %f", reid_position_weight_, reid_dimension_weight_);
    RCLCPP_INFO(logger_, "  Re-ID Combined Threshold: %f", reid_combined_threshold_);
    RCLCPP_INFO(logger_, "  Stable/Missed/Max Lost Frames: %d / %d / %d", n_stable_frames_, n_missed_frames_, max_lost_frames_);
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

    RCLCPP_DEBUG(logger_, "KF Init for Track %d: x_k=[%f, %f, %f, %f, %f, %f]", cluster_mem.id,
                 cluster_mem.x_k(0), cluster_mem.x_k(1), cluster_mem.x_k(2),
                 cluster_mem.x_k(3), cluster_mem.x_k(4), cluster_mem.x_k(5));
    std::stringstream ss_pk;
    ss_pk << cluster_mem.P_k;
    RCLCPP_DEBUG(logger_, "KF Init P_k for Track %d:\n%s", cluster_mem.id, ss_pk.str().c_str());
}

void ClusterTracker::predictKalmanFilter(ClusterMemory& cluster_mem, double dt)
{
    if (dt <= 0) {
        RCLCPP_DEBUG(logger_, "KF Predict for Track %d: dt <= 0, skipping prediction.", cluster_mem.id);
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

    RCLCPP_DEBUG(logger_, "KF Predict for Track %d (dt=%f): x_k=[%f, %f, %f, %f, %f, %f]", cluster_mem.id, dt,
                 cluster_mem.x_k(0), cluster_mem.x_k(1), cluster_mem.x_k(2),
                 cluster_mem.x_k(3), cluster_mem.x_k(4), cluster_mem.x_k(5));
    std::stringstream ss_pk;
    ss_pk << cluster_mem.P_k;
    RCLCPP_DEBUG(logger_, "KF Predict P_k for Track %d:\n%s", cluster_mem.id, ss_pk.str().c_str());
}

void ClusterTracker::updateKalmanFilter(ClusterMemory& cluster_mem, const Eigen::Vector3f& measurement, rclcpp::Time current_time)
{
    Eigen::VectorXd z(3);
    z << measurement.x(), measurement.y(), measurement.z();

    Eigen::VectorXd y = z - H_ * cluster_mem.x_k;
    Eigen::MatrixXd S = H_ * cluster_mem.P_k * H_.transpose() + R_;
    Eigen::MatrixXd K = cluster_mem.P_k * H_.transpose() * S.inverse(); // S.inverse() can be problematic if S is singular

    cluster_mem.x_k = cluster_mem.x_k + K * y;
    Eigen::MatrixXd I = Eigen::MatrixXd::Identity(6, 6);
    cluster_mem.P_k = (I - K * H_) * cluster_mem.P_k;

    cluster_mem.last_update_time = current_time;

    cluster_mem.current_kalman_vel_noise_q = std::max(kalman_min_vel_noise_q_,
                                                      cluster_mem.current_kalman_vel_noise_q * kalman_vel_noise_q_decay_factor_);
    
    RCLCPP_DEBUG(logger_, "KF Update for Track %d (Measurement=[%f, %f, %f]):", cluster_mem.id, measurement.x(), measurement.y(), measurement.z());
    RCLCPP_DEBUG(logger_, "  Updated x_k=[%f, %f, %f, %f, %f, %f]",
                 cluster_mem.x_k(0), cluster_mem.x_k(1), cluster_mem.x_k(2),
                 cluster_mem.x_k(3), cluster_mem.x_k(4), cluster_mem.x_k(5));
    std::stringstream ss_pk;
    ss_pk << cluster_mem.P_k;
    RCLCPP_DEBUG(logger_, "  Updated P_k:\n%s", ss_pk.str().c_str());
}

Eigen::Vector3f ClusterTracker::computeCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices)
{
    Eigen::Vector3f centroid(0, 0, 0);
    if (indices.empty()) {
        RCLCPP_WARN(logger_, "Attempted to compute centroid for empty cluster indices.");
        return centroid;
    }
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

// New helper functions for cost calculation and Hungarian assignment
double ClusterTracker::computeEuclideanCost(const dbscan_clusterer::ClusterCandidate& cluster, const ClusterMemory& track) {
    Eigen::Vector3f predicted_centroid(static_cast<float>(track.x_k(0)),
                                       static_cast<float>(track.x_k(1)),
                                       static_cast<float>(track.x_k(2)));
    double cost = (cluster.centroid - predicted_centroid).norm();
    RCLCPP_DEBUG(logger_, "Euclidean Cost (Cluster %d to Track %d): Cluster_C=[%f,%f,%f], Track_Pred_C=[%f,%f,%f], Cost=%f",
                 cluster.original_cloud_idx, track.id,
                 cluster.centroid.x(), cluster.centroid.y(), cluster.centroid.z(),
                 predicted_centroid.x(), predicted_centroid.y(), predicted_centroid.z(),
                 cost);
    return cost;
}

double ClusterTracker::computeReIDCost(const dbscan_clusterer::ClusterCandidate& cluster, const ClusterMemory& track) {
    Eigen::Vector3f predicted_centroid(static_cast<float>(track.x_k(0)),
                                       static_cast<float>(track.x_k(1)),
                                       static_cast<float>(track.x_k(2)));
    double position_dist = (cluster.centroid - predicted_centroid).norm();

    double lost_dx = static_cast<double>(track.max_bounds.x() - track.min_bounds.x());
    double lost_dy = static_cast<double>(track.max_bounds.y() - track.min_bounds.y());
    double lost_dz = static_cast<double>(track.max_bounds.z() - track.min_bounds.z());
    Eigen::Vector3f lost_dimensions(lost_dx, lost_dy, lost_dz);

    double dimension_diff = (cluster.dimensions - lost_dimensions).norm();

    double combined_cost = (position_dist * reid_position_weight_) + (dimension_diff * reid_dimension_weight_);

    RCLCPP_DEBUG(logger_, "ReID Cost (Cluster %d to Lost Track %d): Pos_Dist=%f, Dim_Diff=%f, Combined_Cost=%f",
                 cluster.original_cloud_idx, track.id, position_dist, dimension_diff, combined_cost);
    RCLCPP_DEBUG(logger_, "  Cluster_C=[%f,%f,%f], Cluster_D=[%f,%f,%f]",
                 cluster.centroid.x(), cluster.centroid.y(), cluster.centroid.z(),
                 cluster.dimensions.x(), cluster.dimensions.y(), cluster.dimensions.z());
    RCLCPP_DEBUG(logger_, "  Track_Pred_C=[%f,%f,%f], Track_Lost_D=[%f,%f,%f]",
                 predicted_centroid.x(), predicted_centroid.y(), predicted_centroid.z(),
                 lost_dimensions.x(), lost_dimensions.y(), lost_dimensions.z());

    return combined_cost;
}

std::vector<std::pair<int, int>> ClusterTracker::performHungarianAssignment(
    const std::vector<std::vector<double>>& cost_matrix, double threshold)
{
    RCLCPP_DEBUG(logger_, "Performing Hungarian Assignment. Cost Matrix size: %zu x %zu", cost_matrix.size(), (cost_matrix.empty() ? 0 : cost_matrix[0].size()));
    if (!cost_matrix.empty()) {
        std::stringstream ss_cost_matrix;
        ss_cost_matrix << "Cost Matrix:\n";
        for (const auto& row : cost_matrix) {
            for (double val : row) {
                if (val == HungarianAlgorithm::INF) {
                    ss_cost_matrix << "INF\t";
                } else {
                    ss_cost_matrix << val << "\t";
                }
            }
            ss_cost_matrix << "\n";
        }
        RCLCPP_DEBUG(logger_, "%s", ss_cost_matrix.str().c_str());
    }

    HungarianAlgorithm hungarian;
    std::vector<int> assignments_flat = hungarian.Solve(cost_matrix);

    std::vector<std::pair<int, int>> matches;
    RCLCPP_DEBUG(logger_, "Hungarian Assignments (flat):");
    for (size_t i = 0; i < assignments_flat.size(); ++i) {
        int track_idx = assignments_flat[i];
        RCLCPP_DEBUG(logger_, "  Cluster_idx %zu assigned to Track_idx %d", i, track_idx);

        if (track_idx != -1 && i < cost_matrix.size() && static_cast<size_t>(track_idx) < cost_matrix[0].size()) {
            if (cost_matrix[i][track_idx] < threshold) {
                matches.emplace_back(static_cast<int>(i), track_idx);
                RCLCPP_DEBUG(logger_, "    -> Match accepted (Cost: %f < Threshold: %f)", cost_matrix[i][track_idx], threshold);
            } else {
                RCLCPP_DEBUG(logger_, "    -> Match rejected (Cost: %f >= Threshold: %f)", cost_matrix[i][track_idx], threshold);
            }
        } else {
            RCLCPP_DEBUG(logger_, "    -> No valid assignment or out of bounds for original matrix.");
        }
    }
    RCLCPP_DEBUG(logger_, "Total matches found by Hungarian (after thresholding): %zu", matches.size());
    return matches;
}


std::vector<ClusterMemory> ClusterTracker::processClusters(
    rclcpp::Time current_frame_time,
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<std::vector<int>>& current_detected_clusters,
    std::vector<ClusterMemory>& active_tracks,
    std::vector<ClusterMemory>& recently_lost_tracks)
{
    RCLCPP_INFO(logger_, "Processing new frame at time: %f", current_frame_time.seconds());
    RCLCPP_INFO(logger_, "  Active tracks count: %zu, Recently lost tracks count: %zu, Detected clusters count: %zu",
                active_tracks.size(), recently_lost_tracks.size(), current_detected_clusters.size());

    std::vector<ClusterMemory> next_active_tracks_state;
    std::vector<ClusterMemory> next_recently_lost_tracks_state;
    std::vector<ClusterMemory> stable_published_tracks;

    // --- Step 1: Predict positions of all tracks (active and recently lost) ---
    RCLCPP_DEBUG(logger_, "Step 1: Predicting positions of active tracks.");
    for (auto& prev_cluster : active_tracks) {
        prev_cluster.missed_count++;
        double dt = (current_frame_time - prev_cluster.last_update_time).seconds();
        if (dt < 0) {
            RCLCPP_WARN(logger_, "Negative dt detected for active track %d (%f). Clamping to 0.", prev_cluster.id, dt);
            dt = 0;
        }
        predictKalmanFilter(prev_cluster, dt);
    }

    RCLCPP_DEBUG(logger_, "Step 1: Predicting positions of recently lost tracks.");
    for (auto& lost_cluster : recently_lost_tracks) {
        lost_cluster.missed_count++;
        double dt = (current_frame_time - lost_cluster.last_update_time).seconds();
        if (dt < 0) {
            RCLCPP_WARN(logger_, "Negative dt detected for lost track %d (%f). Clamping to 0.", lost_cluster.id, dt);
            dt = 0;
        }
        predictKalmanFilter(lost_cluster, dt);
    }

    // --- Step 2: Prepare ClusterCandidates from current_detected_clusters ---
    RCLCPP_DEBUG(logger_, "Step 2: Preparing ClusterCandidates from detected clusters.");
    std::vector<dbscan_clusterer::ClusterCandidate> current_cluster_candidates;
    current_cluster_candidates.reserve(current_detected_clusters.size());
    for (size_t i = 0; i < current_detected_clusters.size(); ++i) {
        if (current_detected_clusters[i].empty()) {
            RCLCPP_WARN(logger_, "Detected cluster %zu is empty, skipping.", i);
            continue;
        }

        Eigen::Vector3f centroid = computeCentroid(cloud, current_detected_clusters[i]);
        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, current_detected_clusters[i], min_pt, max_pt);
        Eigen::Vector3f dimensions(max_pt.x() - min_pt.x(),
                                   max_pt.y() - min_pt.y(),
                                   max_pt.z() - min_pt.z());
        
        current_cluster_candidates.push_back({centroid, dimensions, min_pt, max_pt, static_cast<int>(i)});
        RCLCPP_DEBUG(logger_, "  ClusterCandidate %d: Centroid=[%f,%f,%f], Dimensions=[%f,%f,%f]",
                     static_cast<int>(i), centroid.x(), centroid.y(), centroid.z(),
                     dimensions.x(), dimensions.y(), dimensions.z());
    }
    RCLCPP_DEBUG(logger_, "  Total ClusterCandidates prepared: %zu", current_cluster_candidates.size());


    // Initialize flags for matched clusters and tracks
    std::vector<bool> current_cluster_matched(current_detected_clusters.size(), false);
    std::vector<bool> active_track_is_assigned(active_tracks.size(), false);
    std::vector<bool> lost_track_reidentified(recently_lost_tracks.size(), false);

    // --- Step 3: Global Nearest Neighbor (GNN) Matching with Active Tracks ---
    RCLCPP_INFO(logger_, "Step 3: Performing GNN matching with active tracks.");
    // Construct cost matrix for active tracks
    std::vector<std::vector<double>> cost_matrix_active(current_cluster_candidates.size(), std::vector<double>(active_tracks.size()));
    for (size_t i = 0; i < current_cluster_candidates.size(); ++i) {
        for (size_t j = 0; j < active_tracks.size(); ++j) {
            cost_matrix_active[i][j] = computeEuclideanCost(current_cluster_candidates[i], active_tracks[j]);
        }
    }

    // Perform Hungarian assignment for active tracks
    std::vector<std::pair<int, int>> matches_found_active = 
        performHungarianAssignment(cost_matrix_active, active_track_match_distance_threshold_);

    // Update matched flags based on active track assignments
    RCLCPP_DEBUG(logger_, "  Matches found for active tracks (%zu):", matches_found_active.size());
    for (const auto& match : matches_found_active) {
        current_cluster_matched[current_cluster_candidates[match.first].original_cloud_idx] = true;
        active_track_is_assigned[match.second] = true;
        RCLCPP_DEBUG(logger_, "    Cluster %d (original_idx %d) matched to Active Track %d",
                     match.first, current_cluster_candidates[match.first].original_cloud_idx, active_tracks[match.second].id);
    }

    // --- Step 4: Global Nearest Neighbor (GNN) Matching with Recently Lost Tracks (Re-identification) ---
    RCLCPP_INFO(logger_, "Step 4: Performing GNN matching with recently lost tracks (re-identification).");
    // Collect unmatched current clusters for re-identification
    std::vector<dbscan_clusterer::ClusterCandidate> unmatched_current_clusters;
    std::map<int, int> unmatched_original_idx_map; // Maps new index to original index
    for (size_t i = 0; i < current_cluster_candidates.size(); ++i) {
        if (!current_cluster_matched[current_cluster_candidates[i].original_cloud_idx]) {
            unmatched_current_clusters.push_back(current_cluster_candidates[i]);
            unmatched_original_idx_map[static_cast<int>(unmatched_current_clusters.size() - 1)] = current_cluster_candidates[i].original_cloud_idx;
        }
    }
    RCLCPP_DEBUG(logger_, "  Unmatched current clusters for re-identification: %zu", unmatched_current_clusters.size());

    // This vector will hold the final re-identified matches for processing in Step 5
    std::vector<std::pair<int, int>> matches_found_reid; 

    if (!unmatched_current_clusters.empty() && !recently_lost_tracks.empty()) {
        // Construct cost matrix for re-identification
        std::vector<std::vector<double>> cost_matrix_reid(unmatched_current_clusters.size(), std::vector<double>(recently_lost_tracks.size()));
        for (size_t i = 0; i < unmatched_current_clusters.size(); ++i) {
            for (size_t j = 0; j < recently_lost_tracks.size(); ++j) {
                cost_matrix_reid[i][j] = computeReIDCost(unmatched_current_clusters[i], recently_lost_tracks[j]);
            }
        }

        // Perform Hungarian assignment for re-identification
        std::vector<std::pair<int, int>> matches_found_reid_temp =
            performHungarianAssignment(cost_matrix_reid, reid_combined_threshold_);

        // Update matched flags based on re-identification assignments
        RCLCPP_DEBUG(logger_, "  Matches found for re-identification (%zu):", matches_found_reid_temp.size());
        for (const auto& match : matches_found_reid_temp) {
            int original_cluster_idx = unmatched_original_idx_map[match.first];
            current_cluster_matched[original_cluster_idx] = true;
            lost_track_reidentified[match.second] = true;

            // Store the re-identified matches in the consolidated list for Step 5
            matches_found_reid.emplace_back(original_cluster_idx, match.second);
            RCLCPP_DEBUG(logger_, "    Cluster %d (original_idx %d) re-identified to Lost Track %d",
                         match.first, original_cluster_idx, recently_lost_tracks[match.second].id);
        }
    } else {
        RCLCPP_INFO(logger_, "  Skipping re-identification: No unmatched clusters or no lost tracks.");
    }


    // --- Step 5: Populate 'next_active_tracks_state' and 'next_recently_lost_tracks_state' ---
    RCLCPP_INFO(logger_, "Step 5: Populating next track states.");

    // Process matched active clusters
    RCLCPP_DEBUG(logger_, "  Processing active matches.");
    for (const auto& match : matches_found_active)
    {
        const dbscan_clusterer::ClusterCandidate& current_candidate = current_cluster_candidates[match.first];
        ClusterMemory updated_mem = active_tracks[match.second];

        updated_mem.min_bounds = current_candidate.min_bounds;
        updated_mem.max_bounds = current_candidate.max_bounds;
        updated_mem.frame_count++;
        updated_mem.missed_count = 0;

        updateKalmanFilter(updated_mem, current_candidate.centroid, current_frame_time);
        
        updated_mem.centroid.x() = static_cast<float>(updated_mem.x_k(0));
        updated_mem.centroid.y() = static_cast<float>(updated_mem.x_k(1));
        updated_mem.centroid.z() = static_cast<float>(updated_mem.x_k(2));

        next_active_tracks_state.push_back(updated_mem);
        RCLCPP_DEBUG(logger_, "    Active Track %d updated. Frame count: %d", updated_mem.id, updated_mem.frame_count);

        if (updated_mem.frame_count >= n_stable_frames_) {
            stable_published_tracks.push_back(updated_mem);
            RCLCPP_DEBUG(logger_, "      Track %d is now stable.", updated_mem.id);
        }
    }

    // Process re-identified lost clusters
    RCLCPP_DEBUG(logger_, "  Processing re-identified lost tracks.");
    for (const auto& match : matches_found_reid)
    {
        const dbscan_clusterer::ClusterCandidate& current_candidate = current_cluster_candidates[match.first];
        ClusterMemory reidentified_mem = recently_lost_tracks[match.second];

        reidentified_mem.min_bounds = current_candidate.min_bounds;
        reidentified_mem.max_bounds = current_candidate.max_bounds;
        reidentified_mem.frame_count++; // Re-identified, so increment frame count
        reidentified_mem.missed_count = 0; // Reset missed count

        updateKalmanFilter(reidentified_mem, current_candidate.centroid, current_frame_time);
        
        reidentified_mem.centroid.x() = static_cast<float>(reidentified_mem.x_k(0));
        reidentified_mem.centroid.y() = static_cast<float>(reidentified_mem.x_k(1));
        reidentified_mem.centroid.z() = static_cast<float>(reidentified_mem.x_k(2));

        next_active_tracks_state.push_back(reidentified_mem);
        RCLCPP_INFO(logger_, "Track %d RE-IDENTIFIED (current frame count: %d).", reidentified_mem.id, reidentified_mem.frame_count);

        if (reidentified_mem.frame_count >= n_stable_frames_) {
            stable_published_tracks.push_back(reidentified_mem);
            RCLCPP_DEBUG(logger_, "      Re-identified Track %d is now stable.", reidentified_mem.id);
        }
    }

    // Process new clusters (those not matched to active or lost tracks)
    RCLCPP_INFO(logger_, "  Processing new clusters.");
    for (size_t i = 0; i < current_cluster_candidates.size(); ++i)
    {
        if (!current_cluster_matched[current_cluster_candidates[i].original_cloud_idx])
        {
            const dbscan_clusterer::ClusterCandidate& current_candidate = current_cluster_candidates[i];

            ClusterMemory new_mem;
            new_mem.color = generateRandomColor();
            new_mem.id = getNextClusterId();
            new_mem.frame_count = 1;
            new_mem.missed_count = 0;
            new_mem.min_bounds = current_candidate.min_bounds;
            new_mem.max_bounds = current_candidate.max_bounds;

            initKalmanFilter(new_mem, current_candidate.centroid, current_frame_time);

            new_mem.centroid.x() = static_cast<float>(new_mem.x_k(0));
            new_mem.centroid.y() = static_cast<float>(new_mem.x_k(1));
            new_mem.centroid.z() = static_cast<float>(new_mem.x_k(2));

            next_active_tracks_state.push_back(new_mem);
            RCLCPP_INFO(logger_, "New Track %d created.", new_mem.id);
        }
    }

    // --- Step 6: Handle tracks that were NOT matched in this frame ---
    RCLCPP_INFO(logger_, "Step 6: Handling unmatched tracks.");

    // Active tracks that were not assigned to any current cluster
    RCLCPP_DEBUG(logger_, "  Processing unassigned active tracks.");
    for (size_t j = 0; j < active_tracks.size(); ++j)
    {
        if (!active_track_is_assigned[j])
        {
            ClusterMemory missed_track = active_tracks[j]; // Create a copy to modify

            missed_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_,
                                                                missed_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_);
            
            missed_track.centroid.x() = static_cast<float>(missed_track.x_k(0));
            missed_track.centroid.y() = static_cast<float>(missed_track.x_k(1));
            missed_track.centroid.z() = static_cast<float>(missed_track.x_k(2));

            if (missed_track.missed_count <= n_missed_frames_)
            {
                next_active_tracks_state.push_back(missed_track);
                RCLCPP_DEBUG(logger_, "    Active Track %d missed (%d frames). Still active.", missed_track.id, missed_track.missed_count);
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
    RCLCPP_DEBUG(logger_, "  Processing unassigned recently lost tracks.");
    for (size_t j = 0; j < recently_lost_tracks.size(); ++j)
    {
        if (!lost_track_reidentified[j])
        {
            ClusterMemory lost_track = recently_lost_tracks[j]; // Create a copy to modify

            lost_track.current_kalman_vel_noise_q = std::min(kalman_vel_noise_q_,
                                                             lost_track.current_kalman_vel_noise_q / kalman_vel_noise_q_decay_factor_);
            
            lost_track.centroid.x() = static_cast<float>(lost_track.x_k(0));
            lost_track.centroid.y() = static_cast<float>(lost_track.x_k(1));
            lost_track.centroid.z() = static_cast<float>(lost_track.x_k(2));

            if (lost_track.missed_count <= max_lost_frames_)
            {
                next_recently_lost_tracks_state.push_back(lost_track);
                RCLCPP_DEBUG(logger_, "    Lost Track %d missed (%d frames). Still recently lost.", lost_track.id, lost_track.missed_count);
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

    RCLCPP_INFO(logger_, "Frame processing complete. New active tracks count: %zu, New recently lost tracks count: %zu, Stable tracks for publishing: %zu",
                active_tracks.size(), recently_lost_tracks.size(), stable_published_tracks.size());

    return stable_published_tracks;
}

} // namespace dbscan_clusterer
