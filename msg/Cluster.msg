# msg/Cluster.msg
# Represents a single detected cluster with its properties and Kalman filter state.

int32 id
geometry_msgs/Point centroid
geometry_msgs/Vector3 dimensions # dx, dy, dz of bounding box
geometry_msgs/Point min_bounds
geometry_msgs/Point max_bounds
std_msgs/ColorRGBA color
int32 frame_count
int32 missed_count
float64[6] kalman_state # [px, py, pz, vx, vy, vz]
float64[36] kalman_covariance # 6x6 covariance matrix, row-major