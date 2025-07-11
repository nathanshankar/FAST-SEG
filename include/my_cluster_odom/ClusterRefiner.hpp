#ifndef DBSCAN_CLUSTERER_CLUSTER_REFINER_HPP
#define DBSCAN_CLUSTERER_CLUSTER_REFINER_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/search/kdtree.h>
#include <pcl/common/common.h>
#include <queue>
#include <vector>
#include <limits> // For numeric_limits
#include <rclcpp/rclcpp.hpp> // For logging

namespace dbscan_clusterer {

class ClusterRefiner {
public:
    ClusterRefiner(rclcpp::Logger logger);

    /**
     * @brief Performs DBSCAN clustering on the input point cloud.
     * @param cloud The input point cloud.
     * @param eps The maximum distance between two samples for one to be considered as in the neighborhood of the other.
     * @param minPts The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
     * @return A vector of vectors, where each inner vector contains the indices of points belonging to a cluster.
     */
    std::vector<std::vector<int>> dbscanClustering(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, float eps, int minPts);

    /**
     * @brief Refines the detected clusters by filtering out small or degenerate clusters
     * and performing recursive sub-clustering for dense areas.
     * @param cloud The input point cloud.
     * @param clusters The initial clusters from DBSCAN.
     * @param dbscan_eps The DBSCAN epsilon parameter, used for sub-clustering.
     * @param dbscan_min_pts The DBSCAN minPts parameter.
     * @param recursion_level Current recursion depth.
     * @param max_recursion Maximum recursion depth for sub-clustering.
     * @return A vector of refined clusters (indices into the original cloud).
     */
    std::vector<std::vector<int>> refineClusters(
        const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
        const std::vector<std::vector<int>>& clusters,
        double dbscan_eps, int dbscan_min_pts,
        int recursion_level = 0, int max_recursion = 1);

private:
    rclcpp::Logger logger_;
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_CLUSTER_REFINER_HPP