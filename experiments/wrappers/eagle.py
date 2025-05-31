# Modified from: https://github.com/eagle-dataset/EagleMeshTransformer/blob/main/train_graphvit.py

import torch
import torch.nn as nn


NODE_INPUT = 4
NODE_WALL = 6
NODE_DISABLE = 2

MSE = nn.MSELoss()


def get_loss(velocity, pressure, output, state_hat, target):
    velocity = velocity[:, 1:]
    pressure = pressure[:, 1:]
    velocity_hat = state_hat[:, 1:, :2]

    rmse_velocity = torch.sqrt(((velocity - velocity_hat) ** 2).mean(dim=(-1)))
    loss_velocity = torch.mean(rmse_velocity)

    pressure_hat = state_hat[:, 1:, 2:]
    rmse_pressure = torch.sqrt(((pressure - pressure_hat) ** 2).mean(dim=(-1)))
    loss_pressure = torch.mean(rmse_pressure)

    loss = MSE(target[..., :2], output[..., :2]) + 0.1 * MSE(target[..., 2:], output[..., 2:])

    return loss, loss_pressure, loss_velocity


class Positional_Encoder(nn.Module):
    def __init__(self, num_features, pos_length=8):
        super().__init__()
        self.pos_start = 0
        self.pos_length = pos_length
        self.num_features = num_features

    def forward(self, pos):
        original_shape = pos.shape
        pos = pos.reshape(-1, original_shape[-1])
        index = torch.arange(self.pos_start, self.pos_start + self.pos_length, device=pos.device).float()
        freq = 2**index * torch.pi
        cos_feat = torch.cos(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        sin_feat = torch.sin(freq.view(1, 1, -1) * pos.unsqueeze(-1))
        embedding = torch.cat([cos_feat, sin_feat], dim=-1)
        embedding = embedding.view(*original_shape[:-1], -1)
        return embedding[..., : self.num_features]


class EagleModel(nn.Module):
    def __init__(self, main_model, denormalize_fn, use_pe=False):
        super().__init__()
        num_types = 9
        state_size = 4
        pos_enc_dim = 16

        self.use_pe = use_pe

        if use_pe:
            self.pos_encoder = Positional_Encoder(pos_enc_dim)
            self.embedding_model = nn.Linear(num_types + state_size + pos_enc_dim, main_model.in_dim)
        else:
            self.embedding_model = nn.Linear(num_types + state_size, main_model.in_dim)

        self.main_model = main_model
        self.pred_head = nn.Linear(main_model.out_dim, state_size)
        self.denormalize_fn = denormalize_fn

    def single_forward(self, node_state, node_pos, node_types, edges, batch_idx, **kwargs):
        if self.use_pe:
            node_features = torch.cat([node_state, node_types, self.pos_encoder(node_pos)], -1)
        else:
            node_features = torch.cat([node_state, node_types], -1)
        node_features = self.embedding_model(node_features)
        return self.pred_head(
            self.main_model(
                node_features, node_pos, batch_idx, edge_index=edges, **kwargs
            )
        )

    def forward(self, state, mesh_pos, edges, node_type, batch_idx, **kwargs):

        state_hat, output_hat = [state[:, 0]], []
        target = []

        for t in range(1, state.shape[1]):
            next_output = self.single_forward(
                state_hat[-1],
                mesh_pos[:, t - 1],
                node_type[:, t - 1],
                edges[:, t - 1],
                batch_idx=batch_idx,
                **kwargs,
            )
            next_state = state_hat[-1] + next_output

            target.append(state[:, t] - state_hat[-1])

            # Following MGN, we force the boundary conditions at each steps
            mask = torch.logical_or(node_type[:, t, NODE_INPUT] == 1, node_type[:, t, NODE_WALL] == 1)
            mask = torch.logical_or(mask, node_type[:, t, NODE_DISABLE] == 1)

            next_state[mask] = state[:, t][mask]

            state_hat.append(next_state)
            output_hat.append(next_output)

        velocity_hat = torch.stack(state_hat, dim=1)
        output_hat = torch.stack(output_hat, dim=1)

        target = torch.stack(target, dim=1)
        return velocity_hat, output_hat, target

    def step(self, batch, prefix="train"):
        velocity, pressure = batch["velocity"], batch["pressure"]
        state = torch.cat([velocity, pressure], dim=-1)

        state_hat, output, target = self(**batch, state=state)
        state_hat[..., :2], state_hat[..., 2:] = self.denormalize_fn(state_hat[..., :2], state_hat[..., 2:])
        velocity, pressure = self.denormalize_fn(velocity, pressure)

        loss, loss_pressure, loss_velocity = get_loss(velocity, pressure, output, state_hat, target)

        return {
            f"{prefix}/loss": loss,
            f"{prefix}/loss_presure": loss_pressure,
            f"{prefix}/loss_velocity": loss_velocity,
        }

    def training_step(self, batch):
        return self.step(batch, "train")

    @torch.no_grad()
    def validation_step(self, batch):
        return self.step(batch, "val")

    @torch.no_grad()
    def evaluation_step(self, batch):
        velocity, pressure = batch["velocity"], batch["pressure"]
        state = torch.cat([velocity, pressure], dim=-1)

        state_hat, _, _ = self(**batch, state=state, rebuild_tree=True)
        state_hat[..., :2], state_hat[..., 2:] = self.denormalize_fn(state_hat[..., :2], state_hat[..., 2:])
        velocity, pressure = self.denormalize_fn(velocity, pressure)

        velocity = velocity[:, 1:]
        pressure = pressure[:, 1:]
        velocity_hat = state_hat[:, 1:, :2]
        pressure_hat = state_hat[:, 1:, 2:]

        rmse_velocity = torch.sqrt((velocity - velocity_hat).pow(2).mean(dim=-1)).mean(0)
        rmse_pressure = torch.sqrt((pressure - pressure_hat).pow(2).mean(dim=-1)).mean(0)
        rmse_velocity = torch.cumsum(rmse_velocity, dim=0) / torch.arange(1, rmse_velocity.shape[0] + 1, device="cuda")
        rmse_pressure = torch.cumsum(rmse_pressure, dim=0) / torch.arange(1, rmse_pressure.shape[0] + 1, device="cuda")

        return rmse_velocity.cpu().detach(), rmse_pressure.cpu().detach()
