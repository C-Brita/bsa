#ifndef BALLTREE_FIXED_H
#define BALLTREE_FIXED_H

/**
 * @file balltree.h
 * @brief Implementation of Ball Tree data structure for batch processing
 * 
 * This implementation creates complete binary trees for batches of points.
 * Each leaf node contains either one point (duplicated) or two points.
 * The tree is built recursively by selecting split dimensions based on
 * maximum spread of points along each dimension.
 */

#include <algorithm>
#include <cmath>
#include <cstdint>

using DataType = double;
using IndexType = long;
using MaskType = bool;

/**
 * @brief Find the dimension with maximum spread for splitting points
 * 
 * @param data          Pointer to the data array (num_points × num_features)
 * @param idx_array     Array of indices into data
 * @param start         Start index in idx_array
 * @param end           End index in idx_array (exclusive)
 * @param num_features  Number of features per point
 * @return              Index of the dimension with maximum spread
 */
IndexType find_split_dimension(
    const DataType* data,
    const IndexType* idx_array,
    IndexType start,
    IndexType end,
    IndexType num_features
) {
    const IndexType num_points = end - start;
    if (num_points <= 1) return 0;  // Handle edge case

    IndexType best_dim = 0;
    DataType max_spread = 0;
    
    // Find dimension with maximum spread
    for (IndexType dim = 0; dim < num_features; ++dim) {
        // Initialize with first point's value
        DataType min_val = data[idx_array[start] * num_features + dim];
        DataType max_val = min_val;
        
        // Find min and max values along this dimension
        for (IndexType i = start + 1; i < end; ++i) {
            DataType val = data[idx_array[i] * num_features + dim];
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        
        const DataType spread = max_val - min_val;
        if (spread > max_spread) {
            max_spread = spread;
            best_dim = dim;
        }
    }
    
    return best_dim;
}

/**
 * @brief Partition points around median value along given dimension
 * 
 * Uses std::nth_element to efficiently partition points such that
 * the point at split_index is in its sorted position, with smaller
 * values to the left and larger values to the right.
 * 
 * @param data          Pointer to data array
 * @param node_indices  Array of indices to partition
 * @param split_dim     Dimension along which to split
 * @param split_index   Position of the split
 * @param num_features  Number of features per point
 * @param num_points    Number of points to partition
 */
void partition_node_indices(
    const DataType* data,
    IndexType* node_indices,
    IndexType split_dim,
    IndexType split_index,
    IndexType num_features,
    IndexType num_points
) {
    // Lambda for comparing points along split dimension
    auto compare = [data, split_dim, num_features](IndexType a, IndexType b) {
        return data[a * num_features + split_dim] < data[b * num_features + split_dim];
    };
    
    std::nth_element(
        node_indices,
        node_indices + split_index,
        node_indices + num_points,
        compare
    );
}

/**
 * @brief Recursively build a ball tree
 * 
 * Creates a complete binary tree by recursively splitting points
 * along dimensions with maximum spread. Leaf nodes contain either
 * one point (duplicated) or two points.
 * 
 * @param data           Pointer to data array
 * @param idx_array      Working array of indices
 * @param output_indices Output array for tree structure
 * @param output_mask    Output array indicating dummy duplicates
 * @param idx_start      Start index in idx_array
 * @param idx_end        End index in idx_array (exclusive)
 * @param output_start   Start index in output_indices
 * @param num_features   Number of features per point
 * @param current_level  Current depth in tree
 * @param max_level      Maximum tree depth
 */
void build_tree(
    const DataType* data,
    IndexType* idx_array,
    IndexType* output_indices,
    MaskType* output_mask,
    IndexType idx_start,
    IndexType idx_end,
    IndexType output_start,
    IndexType num_features,
    IndexType current_level,
    IndexType max_level
) {
    const IndexType num_points = idx_end - idx_start;
    
    // Handle leaf nodes
    if (current_level == max_level) {
        if (num_points == 1) {
            // Single point - duplicate it
            const IndexType idx = idx_array[idx_start];
            output_indices[output_start] = idx;
            output_indices[output_start + 1] = idx;
            output_mask[output_start] = true;      // Original point
            output_mask[output_start + 1] = false; // Duplicate
        } else {
            // Two or more points - take first two
            output_indices[output_start] = idx_array[idx_start];
            output_indices[output_start + 1] = idx_array[idx_start + 1];
            output_mask[output_start] = true;     // Both points are original
            output_mask[output_start + 1] = true;
        }
        return;
    }
    
    // Find best splitting dimension and partition points
    const IndexType split_dim = find_split_dimension(
        data, idx_array, idx_start, idx_end, num_features
    );
    const IndexType mid_point = (idx_start + idx_end) / 2;
    
    if (num_points > 1) {
        partition_node_indices(
            data,
            idx_array + idx_start,
            split_dim,
            mid_point - idx_start,
            num_features,
            num_points
        );
    }
    
    // Calculate output position for left child's points
    const IndexType left_size = (1 << (max_level - current_level - 1)) * 2;
    
    // Recursively build left and right subtrees
    build_tree(
        data, idx_array, output_indices, output_mask,
        idx_start, mid_point, output_start,
        num_features, current_level + 1, max_level
    );
    
    build_tree(
        data, idx_array, output_indices, output_mask,
        mid_point, idx_end, output_start + left_size,
        num_features, current_level + 1, max_level
    );
}


/**
 * @brief Partitions leaves of a complete ball tree K times again assuming the leaves were subjected to an orthogonal transformation
 * 
 * Takes leaves of a complete ball tree (power of 2 size) and partitions them
 * K times, resulting in partitions of size N/2^K.
 * 
 * @param data              Pointer to data array
 * @param idx_array         Working array of indices (initialized to [0,1,2,...])
 * @param output_indices    Output array for the partitioned indices
 * @param idx_start         Start index in idx_array
 * @param idx_end           End index in idx_array (exclusive)
 * @param output_start      Start index in output_indices
 * @param num_features      Number of features per point
 * @param current_level   How many times data was partitioned
 * @param target_level    Number of partitioning steps
 */
void partition_ball_tree(
    const DataType* data,
    IndexType* idx_array,
    IndexType* output_indices,
    IndexType idx_start,
    IndexType idx_end,
    IndexType output_start,
    IndexType num_features,
    IndexType current_level,
    IndexType target_level
) {
    const IndexType num_points = idx_end - idx_start;
    
    // If we've reached the desired partitioning level or have only one point, stop recursion
    if (target_level == current_level || num_points <= 1) {
        // Copy the indices to the output
        for (IndexType i = 0; i < num_points; ++i) {
            output_indices[output_start + i] = idx_array[idx_start + i];
        }
        return;
    }
    
    // Find best splitting dimension and partition points
    const IndexType split_dim = find_split_dimension(
        data, idx_array, idx_start, idx_end, num_features
    );
    const IndexType mid_point = (idx_start + idx_end) / 2;
    
    if (num_points > 1) {
        partition_node_indices(
            data,
            idx_array + idx_start,
            split_dim,
            mid_point - idx_start,
            num_features,
            num_points
        );
    }
    
    // Calculate output position for left child's points
    const IndexType left_size = mid_point - idx_start;
    
    // Recursively build left and right subtrees
    partition_ball_tree(
        data, idx_array, output_indices,
        idx_start, mid_point, output_start,
        num_features, current_level + 1, target_level
    );
    
    partition_ball_tree(
        data, idx_array, output_indices,
        mid_point, idx_end, output_start + left_size,
        num_features, current_level + 1, target_level
    );
}


#endif // BALLTREE_FIXED_H