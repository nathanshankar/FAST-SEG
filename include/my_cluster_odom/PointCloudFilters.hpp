#ifndef DBSCAN_CLUSTERER_POINT_CLOUD_FILTERS_HPP
#define DBSCAN_CLUSTERER_POINT_CLOUD_FILTERS_HPP

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <rclcpp/rclcpp.hpp> // For RCLCPP_WARN if using static methods with logger

namespace dbscan_clusterer {

class PointCloudFilters {
public:
    /**
     * @brief Applies adaptive Y-axis (height) filtering to the point cloud.
     * @param cloud_in The input point cloud.
     * @param cloud_out The filtered output point cloud.
     * @param logger The ROS logger for warnings.
     */
    static void applyAdaptiveYFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                     rclcpp::Logger logger);

    /**
     * @brief Applies adaptive X-axis filtering to the point cloud.
     * @param cloud_in The input point cloud.
     * @param cloud_out The filtered output point cloud.
     * @param logger The ROS logger for warnings.
     */
    static void applyAdaptiveXFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                     rclcpp::Logger logger);

    /**
     * @brief Applies adaptive Z-axis filtering to the point cloud.
     * @param cloud_in The input point cloud.
     * @param cloud_out The filtered output point cloud.
     * @param logger The ROS logger for warnings.
     */
    static void applyAdaptiveZFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                     rclcpp::Logger logger);

    /**
     * @brief Applies voxel grid downsampling to the point cloud.
     * @param cloud_in The input point cloud.
     * @param cloud_out The filtered output point cloud.
     * @param leaf_size The size of the voxel grid leaf.
     * @param logger The ROS logger for warnings.
     */
    static void applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                     pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                     float leaf_size, rclcpp::Logger logger);
                                     
};

} // namespace dbscan_clusterer

#endif // DBSCAN_CLUSTERER_POINT_CLOUD_FILTERS_HPP