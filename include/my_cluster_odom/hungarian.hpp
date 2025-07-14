#pragma once
#include <vector>
#include <limits>
#include <algorithm>
#include <iostream> // For debugging, can remove later if not needed
#include <numeric>  // For std::iota (optional, but good for range-based loops)
#include <sstream>  // For stringstream to log Eigen matrices
#include <rclcpp/rclcpp.hpp> // For RCLCPP_DEBUG
#include <rcutils/logging_macros.h> // For RCUTILS_LOG_SEVERITY

class HungarianAlgorithm {
public:
    static constexpr double INF = std::numeric_limits<double>::max();

    /**
     * @brief Solves the assignment problem using the Kuhn-Munkres (Hungarian) algorithm.
     * Finds a minimum cost perfect matching in a bipartite graph.
     * @param cost_matrix The input cost matrix (rows: tasks/clusters, columns: workers/tracks).
     * cost_matrix[i][j] is the cost of assigning task i to worker j.
     * @return A vector `result` of size `n` (number of rows in original cost_matrix).
     * `result[i]` is the column index (worker/track index) assigned to row `i` (task/cluster index),
     * or -1 if row `i` is unassigned.
     */
    std::vector<int> Solve(const std::vector<std::vector<double>>& cost_matrix) {
        // Get a logger instance (assuming it's passed or accessible globally, or use a default)
        rclcpp::Logger logger = rclcpp::get_logger("HungarianAlgorithm");

        // Handle empty input matrix
        if (cost_matrix.empty()) {
            RCLCPP_DEBUG(logger, "Hungarian Solve: Empty cost_matrix. Returning empty assignments.");
            return {}; // No clusters, no assignments
        }
        if (cost_matrix[0].empty()) {
            RCLCPP_DEBUG(logger, "Hungarian Solve: Empty columns in cost_matrix. All clusters unassigned.");
            // All clusters are unassigned as there are no tracks to assign to.
            // Return a vector of -1 for each cluster.
            return std::vector<int>(cost_matrix.size(), -1);
        }

        int n = cost_matrix.size();    // Number of rows (clusters)
        int m = cost_matrix[0].size(); // Number of columns (tracks)
        int dim = std::max(n, m);      // Dimension for the square padded matrix

        // CRITICAL FIX: Handle the case where dim is 0 (i.e., both n and m are 0)
        // This prevents creating zero-sized vectors and accessing out-of-bounds indices.
        if (dim == 0) {
            RCLCPP_DEBUG(logger, "Hungarian Solve: Calculated dim is 0. Returning empty assignments.");
            return {}; // No clusters and no tracks, so no assignments possible.
        }

        RCLCPP_DEBUG(logger, "Hungarian Solve: n=%d, m=%d, dim=%d", n, m, dim);

        // Create a padded cost matrix (dim x dim)
        // Initialize with INF, then copy original costs
        std::vector<std::vector<double>> cost(dim, std::vector<double>(dim, INF));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                cost[i][j] = cost_matrix[i][j];
            }
        }
        // Log the padded cost matrix (only if DEBUG is enabled)
        #ifndef NDEBUG
        {
            std::stringstream ss_padded_cost;
            ss_padded_cost << "Padded Cost Matrix (" << dim << "x" << dim << "):\n";
            for (int i = 0; i < dim; ++i) {
                for (int j = 0; j < dim; ++j) {
                    if (cost[i][j] == INF) {
                        ss_padded_cost << "INF\t";
                    } else {
                        ss_padded_cost << cost[i][j] << "\t";
                    }
                }
                ss_padded_cost << "\n";
            }
            RCLCPP_DEBUG(logger, "%s", ss_padded_cost.str().c_str());
        }
        #endif


        // Kuhn-Munkres algorithm variables (0-indexed)
        std::vector<double> u(dim, 0.0); // Labels for rows (u_i)
        std::vector<double> v(dim, 0.0); // Labels for columns (v_j)
        std::vector<int> p(dim, -1);    // p[j] = row index matched to column j (assignment)
        std::vector<int> way(dim, -1);  // way[j] = previous column in augmenting path for column j
        std::vector<char> used(dim, 0); // Keep track of visited columns in current iteration (0 for false, 1 for true)

        // Main loop: Iterate for each row (task/cluster) to find an assignment
        for (int i = 0; i < n; ++i) { // Iterate over original rows
            RCLCPP_DEBUG(logger, "  Processing row i=%d", i);
            p[0] = i; // Current row (cluster) to find assignment for
            int j0 = 0; // Starting column for augmenting path search (dummy column 0)

            std::vector<double> minv(dim, INF); // Minimum slack for each column
            std::fill(used.begin(), used.end(), 0); // Reset 'used' flags for each iteration

            int iteration_count = 0; // Add iteration counter to detect infinite loops
            // Heuristic limit: Should not exceed dim * (dim + 1) iterations in worst case for a single row.
            // Using a slightly larger bound for safety.
            const int MAX_INNER_ITERATIONS = dim * (dim + 2); 

            // FIX: Declare delta outside the do-while loop for broader scope
            double delta = INF; 

            // Augmenting path search (Dijkstra-like)
            do {
                iteration_count++;
                if (iteration_count > MAX_INNER_ITERATIONS) {
                    RCLCPP_ERROR(logger, "HungarianAlgorithm: Potential infinite loop detected for row %d. Breaking inner loop.", i);
                    // Log state for debugging
                    std::stringstream ss_minv, ss_used, ss_u, ss_v, ss_p, ss_way;
                    for(double val : minv) ss_minv << val << " ";
                    for(char val : used) ss_used << (int)val << " ";
                    for(double val : u) ss_u << val << " ";
                    for(double val : v) ss_v << val << " ";
                    for(int val : p) ss_p << val << " ";
                    for(int val : way) ss_way << val << " ";

                    // FIX: Explicitly cast p[j0] to int
                    RCLCPP_ERROR(logger, "  minv: %s", ss_minv.str().c_str());
                    RCLCPP_ERROR(logger, "  used: %s", ss_used.str().c_str());
                    RCLCPP_ERROR(logger, "  u: %s", ss_u.str().c_str());
                    RCLCPP_ERROR(logger, "  v: %s", ss_v.str().c_str());
                    RCLCPP_ERROR(logger, "  p: %s", ss_p.str().c_str());
                    RCLCPP_ERROR(logger, "  way: %s", ss_way.str().c_str());
                    RCLCPP_ERROR(logger, "  j0: %d, i0: %d, delta: %f, j1: %d", j0, static_cast<int>(p[j0]), delta, j1);
                    break; // Exit the do-while loop to prevent full hang
                }

                used[j0] = 1; // Mark current column as visited (1 for true)
                int i0 = p[j0];  // Get the row associated with current column j0
                delta = INF; // Reset delta for each inner iteration (already declared)
                int j1 = -1; // Column with minimum slack (to extend path)

                // Log current state before finding min slack
                RCLCPP_DEBUG(logger, "    Inner loop iter %d: j0=%d, i0=%d (assigned to p[j0])", iteration_count, j0, i0);
                
                // Find minimum slack for unvisited columns
                for (int j = 0; j < dim; ++j) {
                    if (used[j] == 0) { // Check if not used (0 for false)
                        // Calculate slack: cost(i0, j) - u(i0) - v(j)
                        double cur = cost[i0][j] - u[i0] - v[j];
                        if (cur < minv[j]) {
                            minv[j] = cur;
                            way[j] = j0; // Store path: column j came from j0
                        }
                        if (minv[j] < delta) {
                            delta = minv[j];
                            j1 = j; // Update column with minimum slack
                        }
                    }
                }
                RCLCPP_DEBUG(logger, "    Min slack found: delta=%f, j1=%d", delta, j1);

                // If j1 is -1 here, it means no unvisited column was reachable or had finite slack.
                // This could indicate a problem with the graph structure or costs.
                if (j1 == -1) {
                    RCLCPP_ERROR(logger, "HungarianAlgorithm: j1 is -1, no unvisited column found with finite slack. Breaking inner loop.");
                    break; // Prevent infinite loop if no path can be extended
                }

                // Update labels (u and v) and slacks (minv)
                // This ensures u_i + v_j <= cost_ij for all (i,j)
                for (int j = 0; j < dim; ++j) {
                    if (used[j] == 1) { // Check if used (1 for true)
                        u[p[j]] += delta; // Update label for row p[j]
                        v[j] -= delta;    // Update label for column j
                    } else {
                        minv[j] -= delta; // Reduce slack for unvisited columns
                    }
                }
                j0 = j1; // Move to the column with minimum slack
            } while (p[j0] != -1); // Continue until an unmatched column is found (p[j0] == -1)
            RCLCPP_DEBUG(logger, "  Augmenting path search finished for row %d.", i);

            // Augment the path: update assignments (p) by backtracking
            do {
                int j1 = way[j0]; // Get the previous column in the path
                p[j0] = p[j1];    // Assign the row of j1 to j0
                j0 = j1;          // Move back along the path
            } while (j0 != 0); // Backtrack until the start of the path (dummy column 0)
            RCLCPP_DEBUG(logger, "  Assignments updated for row %d.", i);
        }

        // Extract results: result[row_idx] = col_idx
        // p[j] stores the row index (cluster index) that is assigned to column j (track index).
        std::vector<int> result(n, -1); // Initialize all original clusters as unassigned
        for (int j = 0; j < m; ++j) { // Iterate over original columns (tracks)
            // If column j is assigned to a row p[j], and that row is one of the original clusters (p[j] < n)
            if (p[j] != -1 && p[j] < n) {
                result[p[j]] = j; // Assign track_idx (j) to cluster_idx (p[j])
            }
        }
        RCLCPP_DEBUG(logger, "Hungarian Solve finished. Returning results.");
        return result;
    }
};
