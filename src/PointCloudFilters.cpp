#include "my_cluster_odom/PointCloudFilters.hpp"

namespace dbscan_clusterer {

void PointCloudFilters::applyAdaptiveYFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                             pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                             rclcpp::Logger logger)
{
    std::vector<float> ys_copy;
    ys_copy.reserve(cloud_in->size());
    for (const auto& p : cloud_in->points) ys_copy.push_back(p.y);
    if (ys_copy.empty()) {
        RCLCPP_WARN(logger, "Y-coordinates empty, skipping Y-filter.");
        cloud_out = cloud_in; // Ensure cloud_out is assigned, even if no filter
        return;
    }
    
    // Use a more robust median-based approach or fixed height if ground/ceiling are known
    // For now, keeping your existing percentile logic
    std::nth_element(ys_copy.begin(), ys_copy.begin() + ys_copy.size() * 0.90, ys_copy.end());
    float ground_y = ys_copy[ys_copy.size() * 0.90];
    std::nth_element(ys_copy.begin(), ys_copy.begin() + ys_copy.size() * 0.15, ys_copy.end());
    float ceiling_y = ys_copy[ys_copy.size() * 0.15];

    pcl::PassThrough<pcl::PointXYZ> pass_y;
    pass_y.setInputCloud(cloud_in);
    pass_y.setFilterFieldName("y");
    pass_y.setFilterLimits(ceiling_y + 0.25f, ground_y - 0.05f);
    pass_y.filter(*cloud_out);

    if (cloud_out->empty())
    {
        RCLCPP_WARN(logger, "Point cloud empty after Y-axis filtering.");
    }
}

void PointCloudFilters::applyAdaptiveXFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                             pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                             rclcpp::Logger logger)
{
    std::vector<float> xs_copy;
    xs_copy.reserve(cloud_in->size());
    for (const auto& p : cloud_in->points) xs_copy.push_back(p.x);
    if (xs_copy.empty()) {
        RCLCPP_WARN(logger, "X-coordinates empty, skipping X-filter.");
        cloud_out = cloud_in; // Ensure cloud_out is assigned, even if no filter
        return;
    }
    
    std::nth_element(xs_copy.begin(), xs_copy.begin() + xs_copy.size() * 0.10, xs_copy.end());
    float wall_min_threshold = xs_copy[xs_copy.size() * 0.10];
    // FIX: Changed ys_copy.size() to xs_copy.size()
    std::nth_element(xs_copy.begin(), xs_copy.begin() + xs_copy.size() * 0.90, xs_copy.end()); 
    float wall_max_threshold = xs_copy[xs_copy.size() * 0.90];

    pcl::PassThrough<pcl::PointXYZ> pass_x;
    pass_x.setInputCloud(cloud_in);
    pass_x.setFilterFieldName("x");
    pass_x.setFilterLimits(wall_min_threshold + 0.05f, wall_max_threshold - 0.05f);
    pass_x.filter(*cloud_out);

    if (cloud_out->empty())
    {
        RCLCPP_WARN(logger, "Point cloud empty after X-axis filtering.");
    }
}

void PointCloudFilters::applyAdaptiveZFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                             pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                             rclcpp::Logger logger)
{
    std::vector<float> zs_copy;
    zs_copy.reserve(cloud_in->size());
    for (const auto& p : cloud_in->points) zs_copy.push_back(p.z);
    if (zs_copy.empty()) {
        RCLCPP_WARN(logger, "Z-coordinates empty, skipping Z-filter.");
        cloud_out = cloud_in; // Ensure cloud_out is assigned, even if no filter
        return;
    }
    
    std::nth_element(zs_copy.begin(), zs_copy.begin() + zs_copy.size() * 0.90, zs_copy.end());
    float wall_max_threshold_z = zs_copy[zs_copy.size() * 0.90];

    pcl::PassThrough<pcl::PointXYZ> pass_z;
    pass_z.setInputCloud(cloud_in);
    pass_z.setFilterFieldName("z");
    pass_z.setFilterLimits(0.0f, wall_max_threshold_z - 0.05f);
    pass_z.filter(*cloud_out);

    if (cloud_out->empty())
    {
        RCLCPP_WARN(logger, "Point cloud empty after Z-axis filtering.");
    }
}

void PointCloudFilters::applyVoxelGridFilter(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_in,
                                             pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_out,
                                             float leaf_size, rclcpp::Logger logger)
{
    pcl::VoxelGrid<pcl::PointXYZ> vg;
    vg.setInputCloud(cloud_in);
    vg.setLeafSize(leaf_size, leaf_size, leaf_size);
    vg.filter(*cloud_out);

    if (cloud_out->empty())
    {
        RCLCPP_WARN(logger, "Point cloud empty after voxel grid downsampling.");
    }
}



} // namespace dbscan_clusterer