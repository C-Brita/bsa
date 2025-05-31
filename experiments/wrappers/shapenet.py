import torch
import torch.nn as nn


class Positional_Encoder(nn.Module):
    def __init__(self, num_features, dimensionality=3, sigma=1.):
        super().__init__()
        self.linear = nn.Linear(dimensionality, num_features // 2, bias=False)
        nn.init.normal_(self.linear.weight, 0.0, 1.0 / sigma)
        
    def forward(self, x):
        proj = self.linear(x)
        return torch.cat([torch.cos(proj), torch.sin(proj)], dim=-1)


class ShapenetCarModel(nn.Module):
    def __init__(self, main_model):
        super().__init__()
        self.pos_enc = Positional_Encoder(main_model.in_dim)
        self.main_model = main_model
        self.pred_head = nn.Sequential(
            nn.Linear(main_model.out_dim, main_model.out_dim),
            nn.GELU(),
            nn.Linear(main_model.out_dim, 1),
        )

        self.y_std_denorm = 48.5743

    def forward(self, node_positions, **kwargs):
        node_features = self.pos_enc(node_positions)
        return self.pred_head(self.main_model(node_features, node_positions, **kwargs))

    @torch.no_grad()
    def denorm_mse(self, pred, target):
        pred = pred * self.y_std_denorm
        target = target * self.y_std_denorm
        return ((pred - target) ** 2).mean()

    def step(self, batch, prefix="train"):
        pred = self(**batch).squeeze(-1)
        loss = ((pred - batch["target"]) ** 2).mean()
        mse = self.denorm_mse(pred, batch["target"])
        return {f"{prefix}/loss": loss, f"{prefix}/mse": mse}

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")
