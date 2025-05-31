import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.pos_embedding = nn.Linear(3, out_dim)

    def forward(self, pos):
        return self.pos_embedding(pos)


class CosmologyModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.main_model = main_model
        self.embedding_model = Embedding(main_model.in_dim)
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 3),
        )

    def forward(self, node_positions, **kwargs):
        node_features = self.embedding_model(node_positions)
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    def step(self, batch, prefix="train"):
        pred = self(batch["pos"], **batch)
        loss = ((pred - batch["target"]) ** 2).mean()
        return {f"{prefix}/loss": loss}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
