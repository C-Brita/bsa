import sys
sys.path.append("../../")

import open3d as o3d
from balltree import build_balltree
from erwin.datasets import ShapenetCarDataset
from torch.utils.data import DataLoader
import torch
import random

train_dataset = ShapenetCarDataset(
        data_path="/home/scur2585/erwinxnsa/shapenetcar_data",
        split="train",
        knn=8,
    )

train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
    )

import numpy as np
import torch
import matplotlib.cm as cm
import open3d as o3d
import colorsys


def _to_numpy(arr):
    """torch.Tensor / np.ndarray / list ➜ (N, 3) float64 NumPy array."""
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Each point-set must have shape (N, 3)")
    return arr


def _high_contrast_palette(n):
    """
    Evenly spaced hues (HSV→RGB) with high sat/value.
    Produces very distinct, bright colours even for dozens of classes.
    """
    # Slight offset so the first colour isn’t always pure red
    golden_ratio = 0.61803398875
    base = -0.5        # random start hue each call
    hsv = [( (base + i * golden_ratio) % 1.0, 0.9, 0.95 ) for i in range(n)]
    return [colorsys.hsv_to_rgb(*h) for h in hsv]


def visualize_point_sets(point_sets, point_size=8, colours=[(0, 95 / 255, 115 / 255), (255 / 255, 217 / 255, 181/255)], window_w=1024, window_h=768):
    """
    Render each point-set in its own colour.

    Parameters
    ----------
    point_sets : list of (Ni,3) arrays / tensors
    point_size : float | int   – size of rendered points
    window_w, window_h : int   – Open3D window dimensions
    """
    if not isinstance(point_sets, (list, tuple)):
        raise TypeError("point_sets must be a list or tuple")

    clouds = [_to_numpy(ps) for ps in point_sets]
    n_sets = len(clouds)
    if n_sets == 0:
        raise ValueError("point_sets is empty")


    vis = o3d.visualization.Visualizer()
    vis.create_window(width=window_w, height=window_h)
    opt = vis.get_render_option()
    opt.background_color = np.array([1.0, 1.0, 1.0])
    opt.point_size = float(point_size)

    for pts, col in zip(clouds, colours):
        pc = o3d.geometry.PointCloud()
        S = np.array([0.7, 1.0, 1.0])
        pts_scaled = np.asarray(pts) * S - 5
        pc.points = o3d.utility.Vector3dVector(pts_scaled)
        pc.paint_uniform_color(col)
        vis.add_geometry(pc)

    vis.poll_events()
    vis.update_renderer()
    vis.run()
    vis.destroy_window()



for batch in train_loader:
    tree_idx, tree_mask = build_balltree(batch['node_positions'], batch['batch_idx'])
    grouped_points = batch['node_positions'][tree_idx]
    lone_point = grouped_points[:256]
    other_points = grouped_points[256:]
    visualize_point_sets([lone_point, other_points], point_size=8, colours=[(0, 95 / 255, 115 / 255), (255 / 255, 217 / 255, 181/255)])
    
    visualize_point_sets([grouped_points])

    l1 = [i for i in range(256)]
    l15 = [i+ j*32 + 2048 for i in range(64) for j in range(1, 5)]
    l2 = [i for i in range(4096) if (i not in l1 and i not in l15)]
    center_group_points = grouped_points.reshape(128, 32, 3).mean(1)
    second_visualize = grouped_points[l2]
    visualize_point_sets([grouped_points[l1], grouped_points[l15], second_visualize], point_size=8, colours=[(0, 95 / 255, 115 / 255), (1, 0, 0), (255 / 255, 217 / 255, 181/255)])
    visualize_point_sets([grouped_points[l1], grouped_points[l15], second_visualize], point_size=8, colours=[(0, 95 / 255, 115 / 255), (1, 0, 0), (181/255, 159/255, 48/255)])

    break