import torch
import torch.nn as nn
from torch.distributions.normal import Normal

NUM_NODE_TYPES = 4


class Embedding(nn.Module):
    def __init__(self, seq_len, out_dim):
        super().__init__()
        self.seq_embedding = nn.Linear((seq_len - 1) * 3, out_dim)
        self.ptype_embedding = nn.Embedding(NUM_NODE_TYPES, out_dim)

    def forward(self, flat_vel_seq, node_type):
        return self.seq_embedding(flat_vel_seq) + self.ptype_embedding(node_type)


class MDModel(nn.Module):
    def __init__(self, seq_len, dynamics_model):
        super().__init__()
        self.embedding_model = Embedding(seq_len=seq_len, out_dim=dynamics_model.in_dim)
        self.dynamics_model = dynamics_model
        self.proj_model = nn.Sequential(
            nn.Linear(dynamics_model.out_dim, dynamics_model.out_dim),
            nn.GELU(),
            nn.Linear(dynamics_model.out_dim, 6),
        )

    def forward(self, vel_seq, node_positions, node_type, batch_idx, **kwargs):
        node_features = self.embedding_model(vel_seq, node_type)
        acc_stats = self.proj_model(
            self.dynamics_model(
                node_features,
                node_positions,
                batch_idx=batch_idx,
                node_type=node_type,
                **kwargs,
            )
        )  # acc_stats: [N, 2*3] (mean, std)
        acc_mean, acc_std = torch.split(acc_stats, 3, dim=-1)
        acc_std = 1e-6 + nn.functional.softplus(acc_std)
        return acc_mean, acc_std

    def step(self, batch, prefix="train"):
        acc_mean, acc_std = self.forward(**batch)
        loss = (
            -Normal(acc_mean, acc_std).log_prob(batch["acc_target"]).mean()
        )  # log likelihood of the target acceleration
        with torch.no_grad():
            mse = nn.functional.mse_loss(acc_mean, batch["acc_target"])
        return {f"{prefix}/loss": loss, f"{prefix}/mse": mse}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
