import torch
from torch import nn
from torch_scatter import scatter
import torch.nn.functional as F


class MLP(nn.Module):
    """A simple MLP."""
    def __init__(
        self,
        feature_sizes,
        activation: str = "gelu",
        activate_final: bool = False,
    ):
        super().__init__()
        self.feature_sizes = feature_sizes
        self.activation = getattr(F, activation)
        self.activate_final = activate_final
        
        self.layers = nn.ModuleList()
        for i in range(len(feature_sizes) - 1):
            layer = nn.Linear(feature_sizes[i], feature_sizes[i + 1])
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
            self.layers.append(layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1 or self.activate_final:
                x = self.activation(x)
        return x


class Message(nn.Module):
    def __init__(self, message_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.edge_mlp = MLP(
            feature_sizes=[message_dim] + [hidden_dim] * (layers - 1) + [out_dim],
            activation="gelu",
            activate_final=False
        )

    def forward(self, edge_features, node_features, receivers, senders):
        message_inputs = torch.cat([
            edge_features,
            node_features[receivers],
            node_features[senders]
        ], dim=1)
        return self.edge_mlp(message_inputs)


class AggregateUpdate(nn.Module):
    def __init__(self, reduce_fn, update_dim, hidden_dim, out_dim, layers):
        super().__init__()
        self.node_mlp = MLP(
            feature_sizes=[update_dim] + [hidden_dim] * (layers - 1) + [out_dim],
            activation="gelu",
            activate_final=False
        )
        self.reduce_fn = reduce_fn

    def forward(self, messages, node_features, receivers):
        receiver_agg = scatter(
            messages,
            receivers,
            dim=0,
            dim_size=node_features.size(0),
            reduce=self.reduce_fn,
        )
        return self.node_mlp(torch.cat([node_features, receiver_agg], dim=1))


class InteractionLayer(nn.Module):
    def __init__(
        self, reduce_fn, message_dim, hidden_dim, out_dim, layers
    ):
        super().__init__()
        self.message_fn = Message(message_dim, hidden_dim, hidden_dim, layers)
        self.update_fn = AggregateUpdate(
            reduce_fn, 2 * hidden_dim, hidden_dim, out_dim, layers
        )

    def forward(self, edge_features, node_features, receivers, senders):
        messages = self.message_fn(edge_features, node_features, receivers, senders)
        return self.update_fn(messages, node_features, receivers), messages


class MPNN(nn.Module):
    def __init__(
        self,
        node_dim,
        layers,
        latent_dim,
        mp_steps,
        reduce_fn="mean",
        dimensionality=3,
    ):
        super().__init__()
        self.in_dim = node_dim
        self.out_dim = latent_dim
        self.latent_dim = latent_dim
        
        # Node encoder
        self.node_encoder = MLP(
            feature_sizes=[node_dim, latent_dim],
            activation="gelu",
            activate_final=True
        )

        self.edge_encoder = MLP(
            feature_sizes=[dimensionality, latent_dim],
            activation="gelu",
            activate_final=True
        )
        
        # Message passing layers
        self.mp_steps = nn.ModuleList(
            [
                InteractionLayer(
                    reduce_fn=reduce_fn,
                    message_dim=latent_dim * 3,  # edge_features + 2 * node_features
                    hidden_dim=latent_dim,
                    out_dim=latent_dim,
                    layers=layers,
                )
                for _ in range(mp_steps)
            ]
        )
        
        self.node_norms = nn.ModuleList(
            [nn.LayerNorm(latent_dim) for _ in range(mp_steps)]
        )
        self.edge_norms = nn.ModuleList(
            [nn.LayerNorm(latent_dim) for _ in range(mp_steps)]
        )

    def forward(self, node_features, node_positions, batch_idx, **kwargs):
        assert all(k in kwargs for k in ['edge_index'])
        edge_index = kwargs['edge_index']
        
        # Encode node features
        node_features = self.node_encoder(node_features)
        edge_features = self.edge_encoder(node_positions[edge_index[1]] - node_positions[edge_index[0]])

        # Process
        for mp_layer, node_norm, edge_norm in zip(
            self.mp_steps, self.node_norms, self.edge_norms
        ):
            _node_features, _edge_features = mp_layer(
                edge_features,
                node_features,
                edge_index[1],
                edge_index[0],
            )
            node_features = node_norm(_node_features) + node_features
            edge_features = edge_norm(_edge_features) + edge_features

        return node_features, batch_idx