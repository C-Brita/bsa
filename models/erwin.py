from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_cluster
from einops import rearrange, reduce

from typing import Literal, List
from dataclasses import dataclass

from balltree import build_balltree_with_rotations


def scatter_mean(src: torch.Tensor, idx: torch.Tensor, num_receivers: int):
    """ 
    Averages all values from src into the receivers at the indices specified by idx.

    Args:
        src (torch.Tensor): Source tensor of shape (N, D).
        idx (torch.Tensor): Indices tensor of shape (N,).
        num_receivers (int): Number of receivers (usually the maximum index in idx + 1).
    
    Returns:
        torch.Tensor: Result tensor of shape (num_receivers, D).
    """
    result = torch.zeros(num_receivers, src.size(1), dtype=src.dtype, device=src.device)
    count = torch.zeros(num_receivers, dtype=torch.long, device=src.device)
    result.index_add_(0, idx, src)
    count.index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return result / count.unsqueeze(1).clamp(min=1)


class SwiGLU(nn.Module):
    """ W_3 SiLU(W_1 x) ⊗ W_2 x """
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)
    
    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))


class MPNN(nn.Module):
    """ 
    Message Passing Neural Network (see Gilmer et al., 2017).
        m_ij = MLP([h_i, h_j, pos_i - pos_j])       message
        m_i = mean(m_ij)                            aggregate
        h_i' = MLP([h_i, m_i])                      update

    """
    def __init__(self, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.message_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim + dimensionality, dim), 
                nn.GELU(), 
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])

        self.update_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * dim, dim), 
                nn.LayerNorm(dim)
            ) for _ in range(mp_steps)
        ])       

    def layer(self, message_fn: nn.Module, update_fn: nn.Module, h: torch.Tensor, edge_attr: torch.Tensor, edge_index: torch.Tensor):
        row, col = edge_index
        messages = message_fn(torch.cat([h[row], h[col], edge_attr], dim=-1))
        message = scatter_mean(messages, col, h.size(0))
        update = update_fn(torch.cat([h, message], dim=-1))
        return h + update
    
    @torch.no_grad()
    def compute_edge_attr(self, pos, edge_index):
        return pos[edge_index[0]] - pos[edge_index[1]]

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        edge_attr = self.compute_edge_attr(pos, edge_index)
        for message_fn, update_fn in zip(self.message_fns, self.update_fns):
            x = self.layer(message_fn, update_fn, x, edge_attr, edge_index)
        return x


class ErwinEmbedding(nn.Module):
    """ Linear projection -> MPNN."""
    def __init__(self, in_dim: int, dim: int, mp_steps: int, dimensionality: int = 3):
        super().__init__()
        self.mp_steps = mp_steps
        self.embed_fn = nn.Linear(in_dim, dim)
        self.mpnn = MPNN(dim, mp_steps, dimensionality)

    def forward(self, x: torch.Tensor, pos: torch.Tensor, edge_index: torch.Tensor):
        x = self.embed_fn(x)
        return self.mpnn(x, pos, edge_index) if self.mp_steps > 0 else x


@dataclass
class Node:
    """ Dataclass to store the hierarchical node information."""
    x: torch.Tensor
    pos: torch.Tensor 
    batch_idx: torch.Tensor
    tree_idx_rot: torch.Tensor | None = None
    children: Node | None = None


class BallPooling(nn.Module):
    """ 
    Pooling of leaf nodes in a ball (eq. 12):
        1. select balls of size 'stride'.
        2. concatenate leaf nodes inside each ball along with their relative positions to the ball center.
        3. apply linear projection and batch normalization.
        4. the output is the center of each ball endowed with the pooled features.
    """
    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(stride * in_dim + stride * dimensionality, out_dim)
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        if self.stride == 1: # no pooling
            return Node(x=node.x, pos=node.pos, batch_idx=node.batch_idx, children=node)

        with torch.no_grad():
            batch_idx = node.batch_idx[::self.stride]
            centers = reduce(node.pos, "(n s) d -> n d", 'mean', s=self.stride)
            pos = rearrange(node.pos, "(n s) d -> n s d", s=self.stride)
            rel_pos = rearrange(pos - centers[:, None], "n s d -> n (s d)")

        x = torch.cat([rearrange(node.x, "(n s) c -> n (s c)", s=self.stride), rel_pos], dim=1)
        x = self.norm(self.proj(x))

        return Node(x=x, pos=centers, batch_idx=batch_idx, children=node)


class BallUnpooling(nn.Module):
    """
    Ball unpooling (refinement; eq. 13):
        1. compute relative positions of children (from before pooling) to the center of the ball.
        2. concatenate the pooled features with the relative positions.
        3. apply linear projection and self-connection followed by batch normalization.
        4. the output is a refined tree with the same number of nodes as before pooling.
    """
    def __init__(self, in_dim: int, out_dim: int, stride: int, dimensionality: int = 3):
        super().__init__()
        self.stride = stride
        self.proj = nn.Linear(in_dim + stride * dimensionality, stride * out_dim)         
        self.norm = nn.BatchNorm1d(out_dim)

    def forward(self, node: Node) -> Node:
        with torch.no_grad():
            rel_pos = rearrange(node.children.pos, "(n m) d -> n m d", m=self.stride) - node.pos[:, None]
            rel_pos = rearrange(rel_pos, "n m d -> n (m d)")

        x = torch.cat([node.x, rel_pos], dim=-1)
        node.children.x = self.norm(node.children.x + rearrange(self.proj(x), "n (m d) -> (n m) d", m=self.stride))
        
        return node.children


class BallMSA(nn.Module):
    """ Ball Multi-Head Self-Attention (BMSA) module (eq. 8). """
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.ball_size = ball_size

        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.pe_proj = nn.Linear(dimensionality, dim)
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn((1, num_heads, 1, 1)))

    @torch.no_grad()
    def create_attention_mask(self, pos: torch.Tensor):
        """ Distance-based attention bias (eq. 10). """
        pos = rearrange(pos, '(n m) d -> n m d', m=self.ball_size)
        return self.sigma_att * torch.cdist(pos, pos, p=2).unsqueeze(1)

    @torch.no_grad()
    def compute_rel_pos(self, pos: torch.Tensor):
        """ Relative position of leafs wrt the center of the ball (eq. 9). """
        num_balls, dim = pos.shape[0] // self.ball_size, pos.shape[1]
        pos = pos.view(num_balls, self.ball_size, dim)
        return (pos - pos.mean(dim=1, keepdim=True)).view(-1, dim)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.pe_proj(self.compute_rel_pos(pos))
        q, k, v = rearrange(self.qkv(x), "(n m) (H E K) -> K n H m E", H=self.num_heads, m=self.ball_size, K=3)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=self.create_attention_mask(pos))
        x = rearrange(x, "n H m E -> (n m) (H E)", H=self.num_heads, m=self.ball_size)
        return self.proj(x)


class ErwinTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int, mlp_ratio: int, dimensionality: int = 3):
        super().__init__()
        self.ball_size = ball_size
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        self.BMSA = BallMSA(dim, num_heads, ball_size, dimensionality)
        self.swiglu = SwiGLU(dim, dim * mlp_ratio)

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        x = x + self.BMSA(self.norm1(x), pos)
        return x + self.swiglu(self.norm2(x))


class BasicLayer(nn.Module):
    def __init__(
        self,
        direction: Literal['down', 'up', None], # down: encoder, up: decoder, None: bottleneck
        depth: int,
        stride: int,
        in_dim: int,
        out_dim: int,
        num_heads: int,
        ball_size: int,
        mlp_ratio: int,
        rotate: bool,
        dimensionality: int = 3,

    ):
        super().__init__()
        hidden_dim = in_dim if direction == 'down' else out_dim

        self.blocks = nn.ModuleList([ErwinTransformerBlock(hidden_dim, num_heads, ball_size, mlp_ratio, dimensionality) for _ in range(depth)])
        self.rotate = [i % 2 for i in range(depth)] if rotate else [False] * depth

        self.pool = lambda node: node
        self.unpool = lambda node: node

        if direction == 'down' and stride is not None:
            self.pool = BallPooling(hidden_dim, out_dim, stride, dimensionality)
        elif direction == 'up' and stride is not None:
            self.unpool = BallUnpooling(in_dim, hidden_dim, stride, dimensionality)

    def forward(self, node: Node) -> Node:
        node = self.unpool(node)

        if len(self.rotate) > 1 and self.rotate[1]: # if rotation is enabled, it will be used in the second block
            assert node.tree_idx_rot is not None, "tree_idx_rot must be provided for rotation"
            tree_idx_rot_inv = torch.argsort(node.tree_idx_rot) # map from rotated to original

        for rotate, blk in zip(self.rotate, self.blocks):
            if rotate:
                node.x = blk(node.x[node.tree_idx_rot], node.pos[node.tree_idx_rot])[tree_idx_rot_inv]
            else:
                node.x = blk(node.x, node.pos)
        return self.pool(node)


class ErwinTransformer(nn.Module):
    """ 
    Erwin Transformer.

    Args:
        c_in (int): number of input channels.
        c_hidden (List): number of hidden channels for each encoder + bottleneck layer (reverse for decoder).
        ball_size (List): list of ball sizes for each encoder layer (reverse for decoder).
        enc_num_heads (List): list of number of heads for each encoder layer.
        enc_depths (List): list of number of ErwinTransformerBlock layers for each encoder layer.
        dec_num_heads (List): list of number of heads for each decoder layer.
        dec_depths (List): list of number of ErwinTransformerBlock layers for each decoder layer.
        strides (List): list of strides for each encoder layer (reverse for decoder).
        rotate (int): angle of rotation for cross-ball interactions; if 0, no rotation.
        decode (bool): whether to decode or not. If not, returns latent representation at the coarsest level.
        mlp_ratio (int): ratio of SWIGLU's hidden dim to a layer's hidden dim.
        dimensionality (int): dimensionality of the input data.
        mp_steps (int): number of message passing steps in the MPNN Embedding.

    Notes:
        - lengths of ball_size, enc_num_heads, enc_depths must be the same N (as it includes encoder and bottleneck).
        - lengths of strides, dec_num_heads, dec_depths must be N - 1.
    """
    def __init__(
        self,
        c_in: int,
        c_hidden: List,
        ball_sizes: List,
        enc_num_heads: List,
        enc_depths: List,
        dec_num_heads: List,
        dec_depths: List,
        strides: List,
        rotate: int,
        decode: bool = True,
        mlp_ratio: int = 4,
        dimensionality: int = 3,
        mp_steps: int = 3,
    ):
        super().__init__()
        assert len(enc_num_heads) == len(enc_depths) == len(ball_sizes)
        assert len(dec_num_heads) == len(dec_depths) == len(strides)
        assert len(strides) == len(ball_sizes) - 1
        
        self.rotate = rotate
        self.decode = decode
        self.ball_sizes = ball_sizes
        self.strides = strides

        self.embed = ErwinEmbedding(c_in, c_hidden[0], mp_steps, dimensionality)

        num_layers = len(enc_depths) - 1 # last one is a bottleneck
        
        self.encoder = nn.ModuleList()
        for i in range(num_layers):
            self.encoder.append(
                BasicLayer(
                    direction='down',
                    depth=enc_depths[i],
                    stride=strides[i],
                    in_dim=c_hidden[i],
                    out_dim=c_hidden[i + 1],
                    num_heads=enc_num_heads[i],
                    ball_size=ball_sizes[i],
                    rotate=rotate > 0,
                    mlp_ratio=mlp_ratio,
                    dimensionality=dimensionality,
                )
            )

        self.bottleneck = BasicLayer(
            direction=None,
            depth=enc_depths[-1],
            stride=None,
            in_dim=c_hidden[-1],
            out_dim=c_hidden[-1],
            num_heads=enc_num_heads[-1],
            ball_size=ball_sizes[-1],
            rotate=rotate > 0,
            mlp_ratio=mlp_ratio,
            dimensionality=dimensionality,
        )

        if decode:
            self.decoder = nn.ModuleList()
            for i in range(num_layers - 1, -1, -1):
                self.decoder.append(
                    BasicLayer(
                        direction='up',
                        depth=dec_depths[i],
                        stride=strides[i],
                        in_dim=c_hidden[i + 1],
                        out_dim=c_hidden[i],
                        num_heads=dec_num_heads[i],
                        ball_size=ball_sizes[i],
                        rotate=rotate > 0,
                        mlp_ratio=mlp_ratio,
                        dimensionality=dimensionality,
                    )
                )

        self.in_dim = c_in
        self.out_dim = c_hidden[0]
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0., std=0.02, a=-2., b=2.)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, node_features: torch.Tensor, node_positions: torch.Tensor, batch_idx: torch.Tensor, edge_index: torch.Tensor = None, tree_idx: torch.Tensor = None, tree_mask: torch.Tensor = None, radius: float = None, **kwargs):
        with torch.no_grad():
            # if not given, build the ball tree and radius graph
            if tree_idx is None and tree_mask is None:
                tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(node_positions, batch_idx, self.strides, self.ball_sizes, self.rotate)
            if edge_index is None and self.embed.mp_steps:
                assert radius is not None, "radius (float) must be provided if edge_index is not given to build radius graph"
                edge_index = torch_cluster.radius_graph(node_positions, radius, batch=batch_idx, loop=True)

        x = self.embed(node_features, node_positions, edge_index)

        node = Node(
            x=x[tree_idx],
            pos=node_positions[tree_idx],
            batch_idx=batch_idx[tree_idx],
            tree_idx_rot=None, # will be populated in the encoder
        )

        for layer in self.encoder:
            node.tree_idx_rot = tree_idx_rot.pop(0)
            node = layer(node)

        node.tree_idx_rot = tree_idx_rot.pop(0)
        node = self.bottleneck(node)

        if self.decode:
            for layer in self.decoder:
                node = layer(node)
            return node.x[tree_mask][torch.argsort(tree_idx[tree_mask])]

        return node.x, node.batch_idx