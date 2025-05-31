# Modified from https://github.com/ml-jku/UPT/blob/main/src/datasets/shapenet_car.py.

import os
import torch
from torch_cluster import knn_graph


def num_nodes_to_batch_idx(num_nodes):
    return torch.arange(len(num_nodes)).to(num_nodes.device).repeat_interleave(num_nodes)


class ShapenetCarDataset(torch.utils.data.Dataset):
    # generated with torch.randperm(889, generator=torch.Generator().manual_seed(0))[:189]
    TEST_INDICES = {
        550, 592, 229, 547, 62, 464, 798, 836, 5, 732, 876, 843, 367, 496,
        142, 87, 88, 101, 303, 352, 517, 8, 462, 123, 348, 714, 384, 190,
        505, 349, 174, 805, 156, 417, 764, 788, 645, 108, 829, 227, 555, 412,
        854, 21, 55, 210, 188, 274, 646, 320, 4, 344, 525, 118, 385, 669,
        113, 387, 222, 786, 515, 407, 14, 821, 239, 773, 474, 725, 620, 401,
        546, 512, 837, 353, 537, 770, 41, 81, 664, 699, 373, 632, 411, 212,
        678, 528, 120, 644, 500, 767, 790, 16, 316, 259, 134, 531, 479, 356,
        641, 98, 294, 96, 318, 808, 663, 447, 445, 758, 656, 177, 734, 623,
        216, 189, 133, 427, 745, 72, 257, 73, 341, 584, 346, 840, 182, 333,
        218, 602, 99, 140, 809, 878, 658, 779, 65, 708, 84, 653, 542, 111,
        129, 676, 163, 203, 250, 209, 11, 508, 671, 628, 112, 317, 114, 15,
        723, 746, 765, 720, 828, 662, 665, 399, 162, 495, 135, 121, 181, 615,
        518, 749, 155, 363, 195, 551, 650, 877, 116, 38, 338, 849, 334, 109,
        580, 523, 631, 713, 607, 651, 168,
    }

    def __init__(
        self,
        data_path,
        split,
        knn=16,
        seed=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.data_path = data_path
        self.split = split
        self.knn = knn
        self.seed = seed
        self.domain_min = torch.tensor([-2.0, -1.0, -4.5])
        self.domain_max = torch.tensor([2.0, 4.5, 6.0])
        self.mean = torch.tensor(-36.3099)
        self.std = torch.tensor(48.5743)

        # discover uris
        self.uris = []
        for i in range(9):
            param_uri = f"{self.data_path}/param{i}"
            for name in sorted(os.listdir(param_uri)):
                sample_uri = f"{param_uri}/{name}"
                if os.path.isdir(sample_uri):
                    self.uris.append(sample_uri)

        assert len(self.uris) == 889, f"found {len(self.uris)} uris instead of 889."
        # split into train/test uris
        if split == "train":
            train_idxs = [i for i in range(len(self.uris)) if i not in self.TEST_INDICES]
            self.uris = [self.uris[train_idx] for train_idx in train_idxs]
            assert len(self.uris) == 700
        elif split == "test":
            self.uris = [self.uris[test_idx] for test_idx in self.TEST_INDICES]
            assert len(self.uris) == 189
        else:
            raise NotImplementedError

    def __len__(self):
        return len(self.uris)

    # noinspection PyUnusedLocal
    def getitem_pressure(self, idx):
        p = torch.load(f"{self.uris[idx]}/pressure.th", weights_only=True)
        p -= self.mean
        p /= self.std
        return p

    def getitem_all_pos(self, idx):
        all_pos = torch.load(f"{self.uris[idx]}/mesh_points.th", weights_only=True)
        all_pos.sub_(self.domain_min).div_(self.domain_max - self.domain_min)
        all_pos = (all_pos - 0.4063) / 0.1531  # normalize
        return all_pos

    def __getitem__(self, idx):
        all_pos = self.getitem_all_pos(idx)
        pressure = self.getitem_pressure(idx)

        # Apply the same permutation to both tensors
        num_points = len(all_pos)
        perm_indices = torch.randperm(num_points)

        all_pos = all_pos[perm_indices]
        pressure = pressure[perm_indices]

        output = {
            "node_positions": all_pos,
            "target": pressure,
            "num_nodes": torch.LongTensor([num_points]),
        }

        return output

    def collate_fn(self, batch):
        batch = {k: torch.cat([d[k] for d in batch], dim=0) for k in batch[0].keys()}
        batch["batch_idx"] = num_nodes_to_batch_idx(batch["num_nodes"])
        if self.knn:
            batch["edge_index"] = knn_graph(batch["node_positions"], k=self.knn, batch=batch["batch_idx"], loop=True)
        return batch
