# Modified from https://github.com/kyonofx/mlcgmd/blob/main/graphwm/data/data.py.

import h5py
import math
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from functools import partial

from torch.utils.data import Dataset
from torch_cluster import knn_graph, radius_graph


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


def load_data(data_names, path):
    hf = h5py.File(path, "r")
    data = []
    for i in range(len(data_names)):
        d = np.array(hf.get(data_names[i]))
        data.append(d)
    hf.close()
    return data


@torch.no_grad()
def pos_to_acc(next_pos, pos_seq):
    """pos -> acc"""
    previous_pos = pos_seq[:, -1]
    previous_velocity = previous_pos - pos_seq[:, -2]
    next_velocity = next_pos - previous_pos
    return next_velocity - previous_velocity


@torch.no_grad()
def pos_to_vel(input_sequence):
    return input_sequence[:, 1:] - input_sequence[:, :-1]


@torch.no_grad()
def noise_augment(pos_seq, num_nodes, noise_begin=0.1, noise_end=0.01, noise_level=10):
    noise_sigmas = torch.linspace(math.log(noise_begin), math.log(noise_end), noise_level).exp().to(pos_seq.device)
    noise_level = torch.randint(0, len(noise_sigmas), (num_nodes.shape[0],), device=num_nodes.device)
    sigma = noise_sigmas[noise_level.repeat_interleave(num_nodes)].view(-1, 1, 1)
    pos_seq = pos_seq + torch.randn_like(pos_seq) * sigma
    return pos_seq


DILATION = 1


class MDDataset(Dataset):
    """
    Dataset class for molecular dynamics data.

    Args:
        directory: directory to data
        split: subset of data to use
        seq_len: number of frames of each sampled data point
        traj_len: length of the whole trajectory to be loaded
        cache_data: whether to cache data in memory
        knn: number of nearest neighbors to connect in the graph (if None, uses radius or bonds)
        radius: radius for connecting nodes in the graph (if None, uses knn or bonds)
    """

    def __init__(
        self,
        directory,
        split,
        seq_len,
        traj_len,
        cache_data=True,
        knn=None,
        radius=None,
    ):
        self.directory = Path(directory)
        self.split = split
        self.seq_len = seq_len

        with open(split, "r") as f:
            self.traj_index = f.read().splitlines()
        self.traj_index_paths = [self.directory / traj for traj in self.traj_index]

        self.n_rollouts = len(self.traj_index)
        self.traj_len = traj_len
        self.offset = self.traj_len - (self.seq_len + 1) * DILATION + 1

        if knn is not None:
            self.connectivity_fn = partial(knn_graph, k=knn, loop=True)
        elif radius is not None:
            self.connectivity_fn = partial(radius_graph, r=radius, loop=True)
        else:
            self.connectivity_fn = None

        self.pos_std = torch.tensor([21.4457, 31.7121, 25.5143])
        self.vel_std = torch.tensor([1.2809, 1.2735, 1.2792])
        self.acc_std = torch.tensor([1.7428, 1.7295, 1.7390])

        self.cached_data = self.cache_data() if cache_data else None

    def get_file_indices(self, idx):
        idx_rollout = idx // self.offset
        st_idx = idx % self.offset
        ed_idx = st_idx + (self.seq_len + 1) * DILATION
        return idx_rollout, st_idx, ed_idx

    def cache_data(self):
        """Cache data in memory for faster loading."""
        cached_data = {tidx: None for tidx in self.traj_index}
        for tidx, tpath in tqdm(
            zip(self.traj_index, self.traj_index_paths),
            total=len(self.traj_index),
            desc="Preloading data",
        ):
            node_type = load_data(["particle_type"], tpath / "ptype.h5")[0]
            bonds = load_data(["bond_indices"], tpath / "bond.h5")[0]
            positions = load_data(["position"], tpath / "position.h5")[0]

            # subtract mean
            positions -= positions.mean((0, 1), keepdims=True)

            reversed_bonds = np.concatenate([bonds[:, 1:], bonds[:, :1]], axis=1)
            bonds = np.concatenate([bonds, reversed_bonds], axis=0)

            cached_data[tidx] = {
                "node_type": torch.from_numpy(node_type).long(),
                "bonds": torch.from_numpy(bonds).long(),
                "positions": torch.from_numpy(positions).float().permute(1, 0, 2),
                "num_nodes": torch.LongTensor([len(node_type)]),
                "num_bonds": torch.LongTensor([len(bonds)]),
            }

        return cached_data

    def __len__(self):
        return self.n_rollouts * self.offset

    def __getitem__(self, idx):
        idx_rollout, st_idx, ed_idx = self.get_file_indices(idx)
        select_idx = np.arange(st_idx, ed_idx, DILATION, dtype=np.int64)

        if self.cached_data is None:
            raise NotImplementedError
        else:
            cached_traj = self.cached_data[self.traj_index[idx_rollout]]
            tensor_dict = {
                "node_type": cached_traj["node_type"],
                "positions": cached_traj["positions"][:, select_idx],
                "num_nodes": cached_traj["num_nodes"],
                "bonds": cached_traj["bonds"],
                "num_bonds": cached_traj["num_bonds"],
            }

        positions = tensor_dict["positions"]

        if self.split == "train":
            positions = noise_augment(positions, tensor_dict["num_nodes"])

        pos_seq = positions[:, :-1]
        next_pos = positions[:, -1]
        current_pos = pos_seq[:, -1] / self.pos_std

        vel_seq = pos_to_vel(pos_seq) / self.vel_std
        vel_seq = vel_seq.reshape(pos_seq.shape[0], -1)

        acc_target = pos_to_acc(next_pos, pos_seq)
        acc_target = acc_target / self.acc_std

        output_dict = {
            "vel_seq": vel_seq,  # N x (T * 3)
            "acc_target": acc_target,
            "node_positions": current_pos,  # N x 3
            "node_type": tensor_dict["node_type"],
            "num_nodes": tensor_dict["num_nodes"],
            "num_bonds": tensor_dict["num_bonds"],
            "edge_index": tensor_dict["bonds"],
        }

        return output_dict

    def collate_fn(self, batch: list):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])

        if self.connectivity_fn is not None:
            batch["edge_index"] = self.connectivity_fn(batch["node_positions"], batch["batch_idx"])
        else:
            index_offsets = torch.cumsum(batch["num_bonds"], -1)
            node_offsets = torch.cumsum(batch["num_nodes"], -1)
            for idx in range(len(index_offsets) - 1):
                batch["edge_index"][index_offsets[idx]:index_offsets[idx + 1]] += node_offsets[idx]
            batch["edge_index"] = batch["edge_index"].t().contiguous()

        return batch
