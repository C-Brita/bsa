import torch
import torch.nn as nn
import torch_cluster
from torch_scatter import scatter


def build_mlp(
    in_dim, dim, layers, out_dim, act_fn=nn.SiLU(), layer_norm=False, act_final=False
):
    mods = [nn.Linear(in_dim, dim), act_fn]
    for _ in range(layers - 1):
        mods += [nn.Linear(dim, dim), act_fn]
    mods += [nn.Linear(dim, out_dim)]
    if act_final:
        mods += [act_fn]
    if layer_norm:
        mods += [nn.LayerNorm(out_dim)]
    return nn.Sequential(*mods)


class PointNetConv(nn.Module):
    def __init__(self, nn, aggr):
        super().__init__()
        self.nn = nn
        self.aggr = aggr.lower()
        if self.aggr not in ['max', 'mean', 'sum']:
            raise ValueError("Aggregation method must be 'max' or 'mean' or 'sum'.")
    
    def forward(self, x, pos, edge_index):
        src, dst = edge_index
        x, x_c = x
        pos, pos_c = pos
        edge_features = self.nn(torch.cat([x[src], pos[src] - pos_c[dst]], dim=1))
        return scatter(edge_features, edge_index[1], dim=0, dim_size=x_c.shape[0], reduce=self.aggr)


class Downsample(torch.nn.Module):
    def __init__(self, ratio, r, nn, aggr):
        super().__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointNetConv(nn, aggr)

    def sample_and_group(self, x, pos, batch):
        fps_idx = torch_cluster.fps(pos, batch, ratio=self.ratio)
        new_x, new_pos, new_batch = x[fps_idx], pos[fps_idx], batch[fps_idx]
        row, col = torch_cluster.radius(pos, new_pos, self.r, batch, new_batch)
        return new_x, new_pos, new_batch, torch.stack([col, row], dim=0)

    def forward(self, x, pos, batch):
        x_c, pos_c, batch_c, edge_index = self.sample_and_group(x, pos, batch)
        x_c = self.conv((x, x_c), (pos, pos_c), edge_index)
        return x_c, pos_c, batch_c


class Upsample(torch.nn.Module):
    def __init__(self, k, nn):
        super().__init__()
        self.k = k
        self.nn = nn

    def knn_interpolate(self, x, pos_x, pos_y, batch_x, batch_y, num_workers=1):
        with torch.no_grad():
            assign_index = torch_cluster.knn(pos_x, pos_y, self.k, batch_x=batch_x, batch_y=batch_y, num_workers=num_workers)
            y_idx, x_idx = assign_index[0], assign_index[1]
            diff = pos_x[x_idx] - pos_y[y_idx]
            squared_distance = (diff * diff).sum(dim=-1, keepdim=True)
            weights = 1.0 / torch.clamp(squared_distance, min=1e-16)

        y = scatter(x[x_idx] * weights, y_idx, dim=0, dim_size=pos_y.size(0), reduce='sum')
        return y / scatter(weights, y_idx, dim=0, dim_size=pos_y.size(0), reduce='sum')
    
    def forward(self, x_c, pos_c, batch_c, res, pos, batch):
        x = self.knn_interpolate(x_c, pos_c, pos, batch_c, batch)
        return self.nn(torch.cat([x, res], dim=1)), pos, batch


class PointNetInteraction(torch.nn.Module):
    def __init__(self, in_dim, dim, layers, out_dim, aggr):
        super().__init__()
        # Input channels account for both `pos` and node features.
        self.up1_module = Downsample(0.5, 0.2, build_mlp(1*in_dim + 3, 1*dim, layers, 2*dim, layer_norm=True, act_final=True), aggr)
        self.up2_module = Downsample(0.25, 0.4, build_mlp(2*dim + 3, 2*dim, layers, 4*dim, layer_norm=True, act_final=True), aggr)
        self.up3_module = Downsample(0.125, 0.8, build_mlp(4*dim + 3, 4*dim, layers, 8*dim, layer_norm=True, act_final=True), aggr)
        self.down3_module = Upsample(32, build_mlp(8*dim + 4*dim, 4*dim, layers, 4*dim, layer_norm=True, act_final=True))
        self.down2_module = Upsample(16, build_mlp(4*dim + 2*dim, 2*dim, layers, 2*dim, layer_norm=True, act_final=True))
        self.down1_module = Upsample(8, build_mlp(2*dim + 1*dim + 3, 1*dim, layers, out_dim, layer_norm=True, act_final=True))

    def forward(self, nodes, coords, batch):
        up0_out = (nodes, coords, batch)
        up1_out = self.up1_module(*up0_out)
        up2_out = self.up2_module(*up1_out)
        up3_out = self.up3_module(*up2_out)
        
        down3_out = self.down3_module(*up3_out, *up2_out)
        down2_out = self.down2_module(*down3_out, *up1_out)
        down1_out = self.down1_module(*down2_out, *(torch.cat([nodes, coords], dim=1), coords, batch))

        return down1_out[0]
    

class PointNetPP(nn.Module):

    def __init__(self, in_dim, hidden_dim, mlp_layers):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = hidden_dim
        self.encoder = build_mlp(in_dim=in_dim, dim=hidden_dim, out_dim=hidden_dim, layers=mlp_layers)
        self.interaction = PointNetInteraction(hidden_dim, hidden_dim, mlp_layers, hidden_dim, 'mean')
        self.decoder = build_mlp(in_dim=hidden_dim, dim=hidden_dim, out_dim=hidden_dim, layers=mlp_layers)

    def forward(self, node_features, node_positions, batch_idx, **kwargs):
        node_features = self.encoder(node_features)
        output = self.interaction(node_features, node_positions, batch_idx)
        return self.decoder(output)