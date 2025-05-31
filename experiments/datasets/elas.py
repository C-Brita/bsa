import os
import numpy as np
import torch
from torch_cluster import knn_graph


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


class ElasDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        data_path,
        split,
        knn=16,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.split = split
        self.knn = knn
        self.mean = torch.tensor(187.7388)
        self.std = torch.tensor(127.0829)
        self.pos_mean = torch.tensor([0.4998, 0.5003]).reshape(1,1,2)
        self.pos_std = torch.tensor([0.3217, 0.3216]).reshape(1,1,2)

        self.PATH_Sigma = data_path + '/elasticity/Meshes/Random_UnitCell_sigma_10.npy'
        self.PATH_XY = data_path + '/elasticity/Meshes/Random_UnitCell_XY_10.npy'

        s = torch.tensor(np.load(self.PATH_Sigma), dtype=torch.float).permute(1, 0)
        xy = torch.tensor(np.load(self.PATH_XY), dtype=torch.float).permute(2, 0, 1)

        if split == 'train':
            self.s = (s[:1000] - self.mean) / self.std
            self.xy = (xy[:1000] - self.pos_mean) / self.pos_std
        else:
            self.s = (s[-200:] - self.mean) / self.std
            self.xy = (xy[-200:] - self.pos_mean) / self.pos_std

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):
        output = {
            "node_positions": self.xy[idx],
            "target": self.s[idx],
            "num_nodes": torch.LongTensor([972]),
        }

        return output

    def collate_fn(self, batch):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])
        if self.knn:
            batch["edge_index"] = knn_graph(batch["node_positions"], k=self.knn, batch=batch["batch_idx"], loop=True)
        return batch