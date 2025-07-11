cmake_minimum_required(VERSION 3.8)
project(my_cluster_odom)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED) # For tf2::toMsg
find_package(tf2_eigen REQUIRED) # For tf2::eigenToMsg
find_package(pcl_ros REQUIRED) 
find_package(cv_bridge REQUIRED)

# For Eigen
find_package(Eigen3 CONFIG REQUIRED)

# For message generation
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(my_cluster_odom
  "msg/Cluster.msg"
  "msg/ClusterArray.msg"
  DEPENDENCIES std_msgs geometry_msgs 
)


# add_executable(dbscan_clusterer_node src/dbscan_clusterer_node.cpp)
# install(TARGETS dbscan_clusterer_node
#   DESTINATION lib/${PROJECT_NAME}
# )
# ament_target_dependencies(dbscan_clusterer_node
#   rclcpp sensor_msgs visualization_msgs pcl_conversions Eigen3 tf2_ros tf2_geometry_msgs tf2_eigen pcl_ros nav_msgs cv_bridge
#   rosidl_default_runtime # Add this line
#   # Also, add your custom message package here if it's external, but if it's in the same package, it's implicitly handled.
# )
# # Link against the generated message library for both executables
# target_link_libraries(dbscan_clusterer_node
#   my_cluster_odom__rosidl_typesupport_cpp # Link to the generated message library
# )

# Include directories
include_directories(
  include
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIR}
)

# Link directories (PCL libraries)
link_directories(
  ${PCL_LIB_DIRS}
)

# Add our custom library for clusterer components
add_library(${PROJECT_NAME}_core
  src/PointCloudFilters.cpp
  src/ClusterRefiner.cpp
  src/ClusterTracker.cpp
)

# Target include directories for the library
target_include_directories(${PROJECT_NAME}_core PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIR}
)

# Add ament dependencies for the core library
ament_target_dependencies(${PROJECT_NAME}_core
  rclcpp sensor_msgs visualization_msgs pcl_conversions Eigen3 tf2_ros tf2_geometry_msgs tf2_eigen pcl_ros nav_msgs cv_bridge
  rosidl_default_runtime
)

# Link libraries for our core library
target_link_libraries(${PROJECT_NAME}_core
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

# Add the main executable
add_executable(dbscan_clusterer_node src/dbscan_clusterer_node.cpp)

# Target include directories for the executable
target_include_directories(dbscan_clusterer_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${PCL_INCLUDE_DIRS}
  ${Eigen3_INCLUDE_DIR}
)

ament_target_dependencies(dbscan_clusterer_node
  rclcpp
  sensor_msgs
  visualization_msgs
  pcl_conversions
  rosidl_default_runtime
  # Also, add your custom message package here if it's external, but if it's in the same package, it's implicitly handled.
)

# Link libraries for the executable
target_link_libraries(dbscan_clusterer_node
  ${PROJECT_NAME}_core # Link our custom library
  my_cluster_odom__rosidl_typesupport_cpp
  ${PCL_LIBRARIES}
  ${Eigen3_LIBRARIES}
)

# Install rules
install(TARGETS
  ${PROJECT_NAME}_core
  dbscan_clusterer_node
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)


# add_executable(icp_odom_node src/icp_odom_node.cpp)
# install(TARGETS icp_odom_node
#   DESTINATION lib/${PROJECT_NAME}
# )
# ament_target_dependencies(icp_odom_node
#   rclcpp nav_msgs geometry_msgs tf2 tf2_ros tf2_geometry_msgs Eigen3 sensor_msgs
#   rosidl_default_runtime # Add this line
# )
# target_link_libraries(icp_odom_node
#   my_cluster_odom__rosidl_typesupport_cpp # Link to the generated message library
# )


if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
