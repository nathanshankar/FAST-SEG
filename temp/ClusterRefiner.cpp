#include "my_cluster_odom/ClusterRefiner.hpp"

#include <pcl/common/impl/io.hpp>
#include <pcl/common/common.h> // For pcl::getMinMax3D
#include <algorithm> // For std::sort, std::max, std::min
#include <cmath> // For std::abs
#include <queue> // For std::queue in DBSCAN
#include <vector> // Required for std::vector


namespace dbscan_clusterer {

ClusterRefiner::ClusterRefiner(rclcpp::Logger logger) : logger_(logger) {}

std::vector<std::vector<int>> ClusterRefiner::dbscanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float eps, int minPts)
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

        if (neighbors.size() < static_cast<size_t>(minPts))
        {
            visited[i] = true;
            continue;
        }

        clusters.emplace_back();
        std::queue<int> neighbor_queue;
        neighbor_queue.push(i);
        visited[i] = true;
        cluster_labels[i] = current_cluster_id_dbscan;

        while (!neighbor_queue.empty())
        {
            int current = neighbor_queue.front();
            neighbor_queue.pop();
            clusters.back().push_back(current);

            tree->radiusSearch(cloud->points[current], eps, neighbors, distances);

            for (int idx : neighbors)
            {
                if (!visited[idx])
                {
                    visited[idx] = true;
                    cluster_labels[idx] = current_cluster_id_dbscan;
                    
                    std::vector<int> new_neighbors;
                    std::vector<float> new_distances;
                    tree->radiusSearch(cloud->points[idx], eps, new_neighbors, new_distances);
                    if (new_neighbors.size() >= static_cast<size_t>(minPts)) {
                        neighbor_queue.push(idx);
                    }
                }
                else if (cluster_labels[idx] == -1) {
                     cluster_labels[idx] = current_cluster_id_dbscan;
                }
            }
        }
        current_cluster_id_dbscan++;
    }
    return clusters;
}

Eigen::Vector3f ClusterRefiner::getClusterCentroid(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, const std::vector<int>& indices)
{
    Eigen::Vector3f centroid = Eigen::Vector3f::Zero();
    if (indices.empty()) {
        return centroid;
    }
    for (int idx : indices) {
        centroid += cloud->points[idx].getVector3fMap();
    }
    return centroid / static_cast<float>(indices.size());
}

std::vector<std::vector<int>> ClusterRefiner::refineClusters(
    const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
    const std::vector<std::vector<int>>& clusters,
    double dbscan_eps, int dbscan_min_pts,
    int recursion_level, int max_recursion)
{
    std::vector<std::vector<int>> refined_clusters;

    const float MIN_CLUSTER_POINTS_THRESHOLD = 30;
    const float MAX_DENSITY_FOR_SMALL_CLUSTER = 1000.0f;
    const float MIN_VOLUME_THRESHOLD = 0.01f;
    const float MIN_SIDE_LENGTH_THRESHOLD = 0.03f;

    for (const auto& cluster : clusters)
    {
        if (cluster.empty()) {
            RCLCPP_WARN(logger_, "Skipping empty cluster during refinement.");
            continue;
        }
        if (cluster.size() < MIN_CLUSTER_POINTS_THRESHOLD) {
            continue;
        }

        Eigen::Vector4f min_pt, max_pt;
        pcl::getMinMax3D(*cloud, cluster, min_pt, max_pt);

        float dx = max_pt.x() - min_pt.x();
        float dy = max_pt.y() - min_pt.y();
        float dz = max_pt.z() - min_pt.z();

        float volume = dx * dy * dz;
        if (volume < 1e-9f) volume = 1e-9f;

        float density = static_cast<float>(cluster.size()) / volume;

        if (dx < MIN_SIDE_LENGTH_THRESHOLD || dy < MIN_SIDE_LENGTH_THRESHOLD || dz < MIN_SIDE_LENGTH_THRESHOLD)
            continue;
        if (volume < MIN_VOLUME_THRESHOLD)
            continue;

        if (cluster.size() < MIN_CLUSTER_POINTS_THRESHOLD * 2 && density > MAX_DENSITY_FOR_SMALL_CLUSTER)
            continue;

        if (density < 50.0f && recursion_level < max_recursion)
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr sub_cloud(new pcl::PointCloud<pcl::PointXYZ>());
            pcl::copyPointCloud(*cloud, cluster, *sub_cloud); 
            if (sub_cloud->empty()) {
                RCLCPP_WARN(logger_, "Sub-cloud empty after copying during refinement recursion. Skipping.");
                continue;
            }

            float new_eps = static_cast<float>(dbscan_eps) / 2.0f;
            auto sub_clusters = dbscanClustering(sub_cloud, new_eps, dbscan_min_pts);

            for (auto& sub : sub_clusters) {
                std::vector<int> original_indices;
                original_indices.reserve(sub.size());
                for (auto& idx : sub) {
                    original_indices.push_back(cluster[idx]);
                }
                sub = original_indices;
            }

            auto sub_refined = refineClusters(cloud, sub_clusters, dbscan_eps, dbscan_min_pts, recursion_level + 1, max_recursion);
            refined_clusters.insert(refined_clusters.end(), sub_refined.begin(), sub_refined.end());
        }
        else
        {
            refined_clusters.push_back(cluster);
        }
    }

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
        const float EPS_CONTAIN = 1e-3f;
        return (minA.x() <= minB.x() + EPS_CONTAIN && maxA.x() >= maxB.x() - EPS_CONTAIN) &&
               (minA.y() <= minB.y() + EPS_CONTAIN && maxA.y() >= maxB.y() - EPS_CONTAIN) &&
               (minA.z() <= minB.z() + EPS_CONTAIN && maxA.z() >= maxB.z() - EPS_CONTAIN);
    };

    std::vector<bool> keep(refined_clusters.size(), true);

    for (size_t i = 0; i < refined_clusters.size(); ++i)
    {
        if (!keep[i]) continue;

        for (size_t j = 0; j < refined_clusters.size(); ++j)
        {
            if (i == j || !keep[j]) continue;

            if (boxContains(mins[i], maxs[i], mins[j], maxs[j]))
            {
                keep[j] = false;
            }
            else if (boxContains(mins[j], maxs[j], mins[i], maxs[i]))
            {
                keep[i] = false;
                break;
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

} // namespace dbscan_clusterer