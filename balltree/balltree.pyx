#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
# distutils: language=c++
# cython: boundscheck=False
# cython: wraparound=False
# cython: initializedcheck=False
# cython: language_level=3

import math
import torch
import numpy as np
cimport numpy as cnp
from libcpp cimport bool
from libc.math cimport ceil, log2
from libc.stdlib cimport malloc, free
from cython.parallel cimport parallel, prange
from openmp cimport omp_get_max_threads, omp_set_num_threads

cnp.import_array()

ctypedef cnp.intp_t intp_t
ctypedef cnp.float64_t float64_t

cdef extern from "balltree.h" nogil:
    void build_tree(
        const float64_t* data,      # Input data array
        intp_t* idx_array,          # Working array for indices
        intp_t* output_indices,     # Output array for tree
        bool* output_mask,          # Output array for mask (dummy indicators)
        intp_t idx_start,           # Start index in working array
        intp_t idx_end,             # End index in working array
        intp_t output_start,        # Start index in output array
        intp_t num_features,        # Dimensionality of data
        intp_t current_level,       # Current tree level
        intp_t max_level            # Maximum tree depth
    )

    void partition_ball_tree(
        const float64_t* data,
        intp_t* idx_array,
        intp_t* output_indices,
        intp_t idx_start,
        intp_t idx_end,
        intp_t output_start,
        intp_t num_features,      
        intp_t current_level,       # number of partitions applied
        intp_t target_level,        # number of partitions to apply
    )

cdef struct BatchMetadata:
    intp_t start                    # Start index in original data array
    intp_t end                      # End index (exclusive) in original data array
    intp_t tree_offset              # Start index in output array for this tree
    intp_t tree_size                # Number of slots allocated for this tree
    intp_t num_leaves               # Number of leaves in the tree

cdef intp_t compute_tree_depth(intp_t num_points) nogil:
    """Compute the depth needed for a complete binary tree. """
    return <intp_t>ceil(log2(num_points)) - 1

cdef int build_batch_trees(
    const float64_t* data,          # Input data array
    BatchMetadata batch,            # Batch metadata
    intp_t* output,                 # Output array for tree structure
    bool* mask,                     # Output array for mask
    intp_t num_features             # Dimensionality of data
) nogil:
    """Process a single array of points to build a ball tree on.
    
    Thread-safe function that builds a complete binary tree for a subset of points.
    
    Parameters
    ----------
    data : float64_t*
        Pointer to the start of the data array
    batch : BatchMetadata
        Metadata for the current batch
    output : intp_t*
        Pointer to the output array
    num_features : intp_t
        Number of features in the data
        
    Returns
    -------
    int
        0 on success, negative values on error
    """
    cdef intp_t batch_size = batch.end - batch.start
    cdef intp_t max_level = compute_tree_depth(batch_size)
    
    # Allocate an intermediate array for indices
    cdef intp_t* idx_array = <intp_t*>malloc(batch_size * sizeof(intp_t))
    if idx_array == NULL:
        return -1
        
    # Initialize indices contiguouslly
    cdef intp_t i
    for i in range(batch_size):
        idx_array[i] = i
        
    build_tree(
        data + batch.start * num_features,  # Point to batch's data
        idx_array,                          # Temporary working array
        output + batch.tree_offset,         # Point to batch's output location
        mask + batch.tree_offset,           # Point to batch's mask location
        0,                                  # Start at beginning of batch
        batch_size,                         # Process whole batch
        0,                                  # Start at beginning of output
        num_features,                       
        0,                                  # Start at root level
        max_level                          
    )
    
    # Adjust indices to account for batch offset
    cdef intp_t num_leaves = 1 << max_level
    for i in range(num_leaves * 2):
        if i < batch.tree_size:
            output[batch.tree_offset + i] += batch.start
    
    free(idx_array)
    return 0

cdef int partition_batch_trees(
    const float64_t* data,         # Input data array
    BatchMetadata batch,           # Batch metadata
    intp_t target_level,           # Number of partitions to apply
    intp_t* output,                # Output array for tree structure
    intp_t num_features            # Dimensionality of data
) nogil:
    """Partition a single array of leaves of a complete binary tree multiple times.

    The function executes fixed number of partitioning operations such that in the resulting
    ball tree leaves contain >=2 points (while build_batch_trees goes until 2).
    
    Parameters
    ----------
    data : float64_t*
        Pointer to the start of the data array
    batch : BatchMetadata
        Metadata for the current batch
    target_level : intp_t
        Number of partitions to apply
    output : intp_t*
        Pointer to the output array
    num_features : intp_t
        Number of features in the data
        
    Returns
    -------
    int
        0 on success, negative values on error
    """
    cdef intp_t batch_size = batch.end - batch.start
    
    # Allocate an intermediate array for indices
    cdef intp_t* idx_array = <intp_t*>malloc(batch_size * sizeof(intp_t))
    if idx_array == NULL:
        return -1
        
    # Initialize indices contiguously
    cdef intp_t i
    for i in range(batch_size):
        idx_array[i] = i
    
    partition_ball_tree(
        data + batch.start * num_features,  # Point to batch's data
        idx_array,                          # Temporary working array
        output + batch.tree_offset,         # Point to batch's output location
        0,                                  # Start at beginning of batch
        batch_size,                         # Process whole batch
        0,                                  # Start at beginning of output
        num_features,
        0,                                  # Current level
        target_level                        # Target level
    )
    
    # Adjust indices to account for batch offset
    for i in range(batch_size):
        output[batch.tree_offset + i] += batch.start
    
    free(idx_array)
    return 0

def build_balltree_with_offsets(data: np.ndarray, batch_offsets: np.ndarray):
    """Build ball trees for multiple subarrays in parallel.
    
    For each subarray defined by batch_offsets, constructs a complete binary tree
    where each leaf node contains 2 points. Trees are built in parallel using OpenMP.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
    batch_offsets : ndarray of shape (num_batches + 1,)
        Indices that define the boundaries of subarrays in data.
        Each subarray i contains points from batch_offsets[i] to batch_offsets[i+1].
        
    Returns
    -------
    ndarray
        Flattened array containing all ball trees. Each leaf node occupies 2 slots,
        either containing two point indices or duplicating a single point index.
        Tree structure can be inferred from batch sizes.
        
    Notes
    -----
    - Input arrays must be contiguous
    - Builds complete binary trees with fixed depth based on subarray size
    - Thread-safe and GIL-free during tree construction
    
    Raises
    ------
    ValueError
        If input arrays have incorrect shapes or invalid batch offsets
    RuntimeError
        If tree construction fails
    """
    if not isinstance(data, np.ndarray) or not isinstance(batch_offsets, np.ndarray):
        raise ValueError("Both data and batch_offsets must be numpy arrays")

    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D array of shape (num_samples, num_features), "
            f"got shape {data.shape}"
        )

    if batch_offsets.ndim != 1:
        raise ValueError(
            f"Batch offsets must be 1D array of shape (num_batches + 1,), "
            f"got shape {batch_offsets.shape}"
        )
        
    if len(batch_offsets) < 2:
        raise ValueError("Batch offsets must contain at least 2 elements")
    
    cdef float64_t[:, ::1] data_arr = np.asarray(data, dtype=np.float64)
    cdef intp_t[::1] batch_offsets_arr = np.asarray(batch_offsets, dtype=np.intp)
    
    cdef intp_t num_batches = len(batch_offsets) - 1
    cdef intp_t num_features = data.shape[1]
    
    # Prepare meradata for each subarray
    cdef BatchMetadata[::1] batch_metadata = np.zeros(num_batches, dtype=[
        ('start', np.intp),
        ('end', np.intp),
        ('tree_offset', np.intp),
        ('tree_size', np.intp),
        ('num_leaves', np.intp)
    ])
    
    cdef intp_t total_size = 0
    cdef intp_t i, num_points
    cdef BatchMetadata* metadata
    
    for i in range(num_batches):
        # Get pointer to current batch metadata
        metadata = &batch_metadata[i]
        
        metadata.start = batch_offsets_arr[i]
        metadata.end = batch_offsets_arr[i + 1]
        
        if metadata.start >= metadata.end:
            raise ValueError(f"Invalid batch offsets at index {i}")
        if metadata.start < 0 or metadata.end > data.shape[0]:
            raise ValueError(f"Batch offset {i} out of bounds")
        
        # Calculate tree structure
        num_points = metadata.end - metadata.start
        metadata.num_leaves = 1 << compute_tree_depth(num_points)  # 2^depth
        metadata.tree_size = metadata.num_leaves * 2  # Each leaf gets 2 slots
        metadata.tree_offset = total_size
        
        total_size += metadata.tree_size
    
    # Allocate memory for the output arrays
    cdef intp_t[::1] output = np.empty(total_size, dtype=np.intp)
    cdef bool[::1] mask = np.empty(total_size, dtype=np.bool_)
    
    # Configure parallel processing of batches, equal chunks of compute per batch element
    cdef int num_threads = omp_get_max_threads()
    cdef int chunk_size = max(1, num_batches // num_threads)
    omp_set_num_threads(num_threads)

    cdef int ret
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(num_batches, schedule='static', chunksize=chunk_size):
            ret = build_batch_trees(
                &data_arr[0,0],
                batch_metadata[i],
                &output[0],
                &mask[0],
                num_features
            )
            if ret != 0:
                with gil:
                    raise RuntimeError(f"Error processing batch {i}: error code {ret}")
    
    return np.asarray(output), np.asarray(mask)

def partition_balltree_with_offsets(data: np.ndarray, batch_offsets: np.ndarray, target_level: int):
    """Partition ball trees for multiple subarrays in parallel.
    
    For each subarray defined by batch_offsets, partitions the points
    up to the specified target level.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
        Each subarray has length 2^k as it is a complete binary tree.
    batch_offsets : ndarray of shape (num_batches + 1,)
        Indices that define the boundaries of subarrays in data.
        Each subarray i contains points from batch_offsets[i] to batch_offsets[i+1].
    target_level : int
        Number of partitioning steps to apply.
        
    Returns
    -------
    ndarray
        Flattened array containing partitioned indices for all trees.
        
    Notes
    -----
    - Input arrays must be contiguous
    - Each subarray is a complete binary tree
    - Partitioning stops at the specified target level
    - Thread-safe and GIL-free during partitioning
    
    Raises
    ------
    ValueError
        If input arrays have incorrect shapes or invalid batch offsets
    RuntimeError
        If partitioning fails
    """
    if not isinstance(data, np.ndarray) or not isinstance(batch_offsets, np.ndarray):
        raise ValueError("Both data and batch_offsets must be numpy arrays")

    if data.ndim != 2:
        raise ValueError(
            f"Data must be 2D array of shape (num_samples, num_features), "
            f"got shape {data.shape}"
        )

    if batch_offsets.ndim != 1:
        raise ValueError(
            f"Batch offsets must be 1D array of shape (num_batches + 1,), "
            f"got shape {batch_offsets.shape}"
        )
        
    if len(batch_offsets) < 2:
        raise ValueError("Batch offsets must contain at least 2 elements")
    
    if target_level < 0:
        raise ValueError("Target level must be non-negative")
    
    cdef float64_t[:, ::1] data_arr = np.asarray(data, dtype=np.float64)
    cdef intp_t[::1] batch_offsets_arr = np.asarray(batch_offsets, dtype=np.intp)
    cdef intp_t target_level_val = target_level
    
    cdef intp_t num_batches = len(batch_offsets) - 1
    cdef intp_t num_features = data.shape[1]
    
    cdef BatchMetadata[::1] batch_metadata = np.zeros(num_batches, dtype=[
        ('start', np.intp),
        ('end', np.intp),
        ('tree_offset', np.intp),
        ('tree_size', np.intp),
        ('num_leaves', np.intp)
    ])
    
    cdef intp_t total_size = 0
    cdef intp_t i, num_points
    cdef BatchMetadata* metadata
    
    for i in range(num_batches):
        # Get pointer to current batch metadata
        metadata = &batch_metadata[i]
        
        metadata.start = batch_offsets_arr[i]
        metadata.end = batch_offsets_arr[i + 1]
        
        if metadata.start >= metadata.end:
            raise ValueError(f"Invalid batch offsets at index {i}")
        if metadata.start < 0 or metadata.end > data.shape[0]:
            raise ValueError(f"Batch offset {i} out of bounds")
        
        # Calculate number of leaves
        num_points = metadata.end - metadata.start
        
        metadata.tree_size = num_points  
        metadata.tree_offset = total_size
        metadata.num_leaves = num_points
        
        total_size += metadata.tree_size
    
    # Allocate memory for the output array
    cdef intp_t[::1] output = np.empty(total_size, dtype=np.intp)
    
    # Configure parallel processing of batches, equal chunks of compute per batch element
    cdef int num_threads = omp_get_max_threads()
    cdef int chunk_size = max(1, num_batches // num_threads)
    omp_set_num_threads(num_threads)

    cdef int ret
    
    with nogil, parallel(num_threads=num_threads):
        for i in prange(num_batches, schedule='static', chunksize=chunk_size):
            ret = partition_batch_trees(
                &data_arr[0,0],
                batch_metadata[i],
                target_level_val,
                &output[0],
                num_features
            )
            if ret != 0:
                with gil:
                    raise RuntimeError(f"Error processing batch {i}: error code {ret}")
    
    return np.asarray(output)

def build_balltree_with_idx(data: np.ndarray, batch_idx: np.ndarray):
    """Build ball trees for multiple subarrays in parallel using batch indices.
    
    Similar to build_balltree_with_offsets but takes batch indices instead of offsets.
    Batch indices must be contiguous integers starting from 0.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
    batch_idx : ndarray of shape (num_samples,)
        Integer array assigning each point to a batch.
        Must be contiguous integers starting from 0.
        
    Returns
    -------
    ndarray
        Flattened array containing all ball trees. Each leaf node occupies 2 slots,
        either containing two point indices or duplicating a single point index.
    """
    if not isinstance(batch_idx, np.ndarray):
        raise ValueError("batch_idx must be a numpy array")
        
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D array, got shape {batch_idx.shape}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError(
            f"batch_idx length ({batch_idx.shape[0]}) must match "
            f"number of data points ({data.shape[0]})"
        )
    
    if len(batch_idx) == 0:
        raise ValueError("batch_idx is empty")
    
    if batch_idx[0] != 0:
        raise ValueError("Batch indices must start from 0")
    
    n = len(batch_idx)
    
    # Compute offsets by finding where batch index changes
    change_points = np.where(batch_idx[1:n] != batch_idx[0:n-1])[0] + 1
    num_batches = batch_idx[n-1] + 1

    batch_offsets = np.zeros(num_batches + 1, dtype=np.intp)
    batch_offsets[1:num_batches] = change_points
    batch_offsets[num_batches] = n
    
    return build_balltree_with_offsets(data, batch_offsets)

def partition_balltree_with_idx(data: np.ndarray, batch_idx: np.ndarray, target_level: int):
    """Partition ball trees for multiple subarrays in parallel using batch indices.
    
    Similar to partition_balltree_with_offsets but takes batch indices instead of offsets.
    Batch indices must be contiguous integers starting from 0.
    
    Parameters
    ----------
    data : ndarray of shape (num_samples, num_features)
        Input data points. Must be contiguous and float64.
    batch_idx : ndarray of shape (num_samples,)
        Integer array assigning each point to a batch.
        Must be contiguous integers starting from 0.
    target_level : int
        Number of partitioning steps to apply.
        
    Returns
    -------
    ndarray
        Flattened array containing partitioned indices for all trees.
    """
    if not isinstance(batch_idx, np.ndarray):
        raise ValueError("batch_idx must be a numpy array")
        
    if batch_idx.ndim != 1:
        raise ValueError(f"batch_idx must be 1D array, got shape {batch_idx.shape}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError(
            f"batch_idx length ({batch_idx.shape[0]}) must match "
            f"number of data points ({data.shape[0]})"
        )
    
    if len(batch_idx) == 0:
        raise ValueError("batch_idx is empty")
    
    if batch_idx[0] != 0:
        raise ValueError("Batch indices must start from 0")
    
    n = len(batch_idx)
    
    # Compute offsets by finding where batch index changes
    change_points = np.where(batch_idx[1:n] != batch_idx[0:n-1])[0] + 1
    num_batches = batch_idx[n-1] + 1

    batch_offsets = np.zeros(num_batches + 1, dtype=np.intp)
    batch_offsets[1:num_batches] = change_points
    batch_offsets[num_batches] = n
    
    return partition_balltree_with_offsets(data, batch_offsets, target_level)

def build_balltree(data: torch.Tensor, batch_idx: torch.Tensor):
    """ Wrapper around 'build_balltree_with_idx' that handles torch Tensors. """

    if not isinstance(data, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
        raise ValueError("Inputs must be PyTorch tensors")
        
    if data.dim() != 2:
        raise ValueError(f"Data must be 2D tensor, got shape {tuple(data.shape)}")
        
    if batch_idx.dim() != 1:
        raise ValueError(f"batch_idx must be 1D tensor, got shape {tuple(batch_idx.shape)}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError("batch_idx length must match number of data points")

    device = data.device
    
    cdef cnp.ndarray data_np = data.detach().cpu().double().numpy()
    cdef cnp.ndarray batch_idx_np = batch_idx.detach().cpu().long().numpy()
    
    cdef tuple result = build_balltree_with_idx(data_np, batch_idx_np)
    
    return (
        torch.from_numpy(result[0]).to(device),
        torch.from_numpy(result[1]).to(device)
    )

def partition_balltree(data: torch.Tensor, batch_idx: torch.Tensor, target_level: int):
    """ Wrapper around 'partition_balltree_with_idx' that handles torch Tensors. """

    if not isinstance(data, torch.Tensor) or not isinstance(batch_idx, torch.Tensor):
        raise ValueError("Inputs must be PyTorch tensors")
        
    if data.dim() != 2:
        raise ValueError(f"Data must be 2D tensor, got shape {tuple(data.shape)}")
        
    if batch_idx.dim() != 1:
        raise ValueError(f"batch_idx must be 1D tensor, got shape {tuple(batch_idx.shape)}")
        
    if batch_idx.shape[0] != data.shape[0]:
        raise ValueError("batch_idx length must match number of data points")

    device = data.device
    
    cdef cnp.ndarray data_np = data.detach().cpu().double().numpy()
    cdef cnp.ndarray batch_idx_np = batch_idx.detach().cpu().long().numpy()

    cdef cnp.ndarray result = partition_balltree_with_idx(data_np, batch_idx_np, target_level)
    
    return torch.from_numpy(result).to(device)

def generate_rotation_matrix(angle: float, dim: int, device: torch.device):
    """ Generate a rotation matrix for the specified angle and dimensionality of the space.

    Angles are given in degrees. For 3D, it assumes that Euler angles are the same.
    """
    
    angle = math.radians(angle)
    c, s = math.cos(angle), math.sin(angle)
    if dim == 2:
        return torch.tensor([
            [c, -s], 
            [s,  c]
        ], device=device)
    elif dim == 3:
        return torch.tensor([
            [c*c,   s*c*(s-1),  s*(s+c*c)],
            [s*c,   s*s*s+c*c,  s*c*(s-1)],
            [-s,    s*c,        c*c      ]
        ], device=device)
    else:
        raise ValueError(f"Unsupported dimension: {dim}")

@torch.compiler.disable(recursive=False)
def build_balltree_with_rotations(data: torch.Tensor, batch_idx: torch.Tensor, strides: list, ball_sizes: list, angle: float = 45.):
    """ Builds the computational backbone of Erwin Transformer's layers.
    
        1) the main ball tree is built (used in attention and pooling/unpooling);
        2) its leaves are rotated to change partitions and enable cross-ball interaction;
        3) on top of those leaves, a tree is built up to leaf size of given ball size for the layer;
          - the tree is not built until leaf size 2 as the main one to save compute.
        4) leaves are coarsened according to the stride via mean pooling (as in the model).

    Parameters
    ----------
    data : torch.Tensor of shape (num_samples, num_features)
        Input data points.
    batch_idx : torch.Tensor of shape (num_samples,)
        Integer tensor assigning each point to a batch.
    strides : list of int
        Strides for pooling at each layer.
    ball_sizes : list of int
        Ball sizes for each layer.
    angle : float, optional
        Rotation angle in degrees.
        
    Returns
    -------
    tuple
        (tree_idx, tree_mask, [rot_tree_idx_1, rot_tree_idx_2, ...])
        where tree_idx and tree_mask are the original tree indices and mask,
        and rot_tree_idx_i are the rotated tree indices for each layer.
    """    
    assert len(strides) == len(ball_sizes) - 1, "Strides must be one less than ball sizes"
    dim = data.shape[1]
    num_layers = len(ball_sizes)
    
    # Build original tree
    tree_idx, tree_mask = build_balltree(data, batch_idx)

    if angle <= 0:
        rot_tree_indices = [None] * num_layers
    else:
        # Get leaves of the original tree
        leaves = data[tree_idx]
        current_batch_idx = batch_idx[tree_idx]
        
        # Number of partitions to do such that leaf balls contain necessary number of points
        target_partitions = [max(0, int(math.log2(leaves.shape[0] / bs))) for bs in ball_sizes]
        
        rotation_matrix = generate_rotation_matrix(angle, dim, data.device)
        rotated_leaves = torch.matmul(leaves, rotation_matrix)
        
        rot_tree_indices = []
        for i in range(num_layers):
            rot_tree_indices.append(
                partition_balltree(
                    rotated_leaves, 
                    current_batch_idx,
                    target_partitions[i]
                )
            )
            
            if i < num_layers - 1:
                rotated_leaves = rotated_leaves.view(-1, strides[i], dim).mean(axis=1)
                current_batch_idx = current_batch_idx[::strides[i]]

    return tree_idx, tree_mask, rot_tree_indices