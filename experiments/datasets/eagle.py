# Modified from https://github.com/eagle-dataset/EagleMeshTransformer/blob/main/Dataloader/eagle.py.

import os.path
import random
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import one_hot


NODE_NORMAL = 0
NODE_INPUT = 4
NODE_OUTPUT = 5
NODE_WALL = 6
NODE_DISABLE = 2


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


def get_data(path, window_length, mode):
    # Time sampling is random during training, but set to a fix value during test and valid, to ensure repeatability.
    t = 0 if window_length == 990 else random.randint(0, 990 - window_length)
    t = 100 if mode != "train" and window_length != 990 else t
    data = np.load(os.path.join(path, "sim.npz"), mmap_mode="r")

    mesh_pos = data["pointcloud"][t : t + window_length].copy()

    cells = np.load("/" + os.path.join(path, f"triangles.npy"))
    cells = cells[t : t + window_length]

    Vx = data["VX"][t : t + window_length].copy()
    Vy = data["VY"][t : t + window_length].copy()

    Ps = data["PS"][t : t + window_length].copy()
    Pg = data["PG"][t : t + window_length].copy()

    velocity = np.stack([Vx, Vy], axis=-1)
    pressure = np.stack([Ps, Pg], axis=-1)
    node_type = data["mask"][t : t + window_length].copy()

    t = torch.arange(t, t + window_length)

    return mesh_pos, cells, node_type, t, velocity, pressure


def faces_to_edges(faces):
    edges = torch.cat([faces[:, :, :2], faces[:, :, 1:], faces[:, :, ::2]], dim=1)
    receivers, _ = torch.min(edges, dim=-1)
    senders, _ = torch.max(edges, dim=-1)
    packed_edges = torch.stack([senders, receivers], dim=-1).int()
    unique_edges = torch.unique(packed_edges, dim=1)
    unique_edges = torch.cat([unique_edges, torch.flip(unique_edges, dims=[-1])], dim=1)
    return unique_edges


class EagleDataset(Dataset):
    def __init__(
        self,
        data_path,
        mode="test",
        window_length=990,
        apply_onehot=True,
        normalize=True,
    ):
        """Eagle dataset
        :param data_path: path to the dataset
        :param window_length: length of the temporal window to sample the simulation
        :param apply_onehot: Encode node type as onehot vector, see global variables to see what is what
        useful for visualization
        :param normalize: center mean and std of velocity/pressure field
        """
        super().__init__()
        assert mode in ["train", "test", "valid"]

        self.window_length = window_length
        assert window_length <= 990, "window length must be smaller than 990"

        self.fn = data_path
        assert os.path.exists(self.fn), f"Path {self.fn} does not exist"

        self.apply_onehot = apply_onehot
        self.dataloc = []
        try:
            with open(f"Splits/{mode}.txt", "r") as f:
                for line in f.readlines():
                    self.dataloc.append(os.path.join(self.fn, line.strip()))
        except FileNotFoundError:
            with open(f"{data_path}/Splits/{mode}.txt", "r") as f:
                for line in f.readlines():
                    self.dataloc.append(os.path.join(self.fn, line.strip()))

        self.mode = mode
        self.do_normalization = normalize
        self.length = 990

    def __len__(self):
        return len(self.dataloc)

    def __getitem__(self, item):
        mesh_pos, faces, node_type, t, velocity, pressure = get_data(self.dataloc[item], self.window_length, self.mode)

        faces = torch.from_numpy(faces).long()
        mesh_pos = torch.from_numpy(mesh_pos).float()
        velocity = torch.from_numpy(velocity).float()
        pressure = torch.from_numpy(pressure).float()
        edges = faces_to_edges(faces)  # Convert triangles to edges (pairs of indices)
        node_type = torch.from_numpy(node_type).long()
        num_nodes = node_type.shape[1]
        num_edges = edges.shape[1]

        if self.apply_onehot:
            node_type = one_hot(node_type, num_classes=9).squeeze(-2)

        if self.do_normalization:
            velocity, pressure = self.normalize(velocity, pressure)

        output = {
            "mesh_pos": mesh_pos.permute(1, 0, 2),
            "edges": edges.permute(1, 0, 2).long(),
            "velocity": velocity.permute(1, 0, 2),
            "pressure": pressure.permute(1, 0, 2),
            "node_type": node_type.permute(1, 0, 2),
            "num_nodes": torch.LongTensor([num_nodes]),
            "num_edges": torch.LongTensor([num_edges]),
            "timestep": t[None],
        }

        return output

    def normalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
            std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure - mean) / std
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = (velocity - mean) / std
            velocity = velocity.reshape(velocity_shape)
        return velocity, pressure

    def denormalize(self, velocity=None, pressure=None):
        if pressure is not None:
            pressure_shape = pressure.shape
            mean = torch.tensor([-0.8322, 4.6050]).to(pressure.device)
            std = torch.tensor([7.4013, 9.7232]).to(pressure.device)
            pressure = pressure.reshape(-1, 2)
            pressure = (pressure * std) + mean
            pressure = pressure.reshape(pressure_shape)
        if velocity is not None:
            velocity_shape = velocity.shape
            mean = torch.tensor([-0.0015, 0.2211]).to(velocity.device).view(-1, 2)
            std = torch.tensor([1.7970, 2.0258]).to(velocity.device).view(-1, 2)
            velocity = velocity.reshape(-1, 2)
            velocity = velocity * std + mean
            velocity = velocity.reshape(velocity_shape)
        return velocity, pressure

    def collate_fn(self, batch: list):
        """
        Collate function for the dataset. Mainly handles batch offsets.
        """
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])

        if "edges" in batch.keys():
            index_offsets = torch.cumsum(batch["num_edges"], -1)
            node_offsets = torch.cumsum(batch["num_nodes"], -1)
            for idx in range(len(index_offsets) - 1):
                batch["edges"][index_offsets[idx]:index_offsets[idx + 1]] += node_offsets[idx]
            batch["edges"] = batch["edges"].permute(2, 1, 0)

        return batch
