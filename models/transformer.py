from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from balltree import build_balltree_with_rotations, build_balltree

from erwinxnsa.nsa.lucidrains_native_sparse_attention import LucidrainsSparseAttention as NSA
from erwinxnsa.nsa.native_sparse_attention import NativeSparseAttention

def should_apply_block_residual(layer_idx: int, depth: int, block_size: int) -> bool:
    return (layer_idx + 1) % block_size == 0 or layer_idx + 1 == depth

class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, dim: int):
        super().__init__()
        self.w1 = nn.Linear(in_dim, dim)
        self.w2 = nn.Linear(in_dim, dim)
        self.w3 = nn.Linear(dim, in_dim)
    
    def forward(self, x: torch.Tensor):
        return self.w3(self.w2(x) * F.silu(self.w1(x)))

class GlobalAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dimensionality: int = 3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
    
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        
        self.position_proj = nn.Linear(dimensionality, dim)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """ Input shape: (B, n, D) """
        q, k, v = rearrange(self.qkv(x), "B n (K H D) -> K B H n D", K=3, H=self.num_heads, D=self.head_dim)

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            is_causal=False,
        )
        out = rearrange(attn_out, "B H n D -> B n (H D)")
        return self.out_proj(out)
    
class BallAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ball_size = ball_size
        
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.position_proj = nn.Linear(dimensionality, dim)
        
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn(1, num_heads, 1, 1))
        
    @torch.no_grad()
    def compute_relative_positions(self, pos: torch.Tensor):
        B, n, d = pos.shape
        m = n // self.ball_size                        
        pos = pos.view(B, m, self.ball_size, d)        
        rel = pos - pos.mean(dim=2, keepdim=True)      
        return rel.view(B, n, d)
    
    @torch.no_grad()
    def create_attention_bias(self, pos: torch.Tensor):
        B, n, d = pos.shape
        m = n // self.ball_size
       
        pos_flat = pos.view(B, m, self.ball_size, d).view(B * m, self.ball_size, d)
        dist = torch.cdist(pos_flat, pos_flat, p=2)                 
        return self.sigma_att * dist.unsqueeze(1)                  
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """ Input shape: (B, n, D) """
        B, n, _ = x.shape
        assert n % self.ball_size == 0, "n must be divisible by ball_size"
        m = n // self.ball_size
        
        x = x + self.position_proj(self.compute_relative_positions(pos))

        q, k, v = rearrange(self.qkv(x), "B (m b) (K H D) -> K (B m) H b D", 
                            b=self.ball_size, K=3, H=self.num_heads, D=self.head_dim)
        
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=self.create_attention_bias(pos),
            is_causal=False,
        )
        out = rearrange(attn_out, "(B m) H b D -> B (m b) (H D)", m=m, H=self.num_heads)
        return self.out_proj(out)
    
class LucidrainsSparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dimensionality: int = 3, **nsa_kwargs):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.pos_proj = nn.Linear(dimensionality, dim)
        
        self.attn = NSA(
            dim           = dim,
            dim_head      = dim // num_heads,
            heads         = num_heads,
            **nsa_kwargs,                     
        )
    
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.pos_proj(pos)
        x = self.attn(x)                   
        return x
    
class TransformerLayer(nn.Module):
    def __init__(
        self, dim: int, num_heads: int, mlp_ratio: int = 4, dimensionality: int = 3,
        ball_size: int = 256,
        attention_type: str = "ball", attention_kwargs: Optional[dict] = None
    ):
        super().__init__()
        self.norm1 = nn.RMSNorm(dim)
        self.norm2 = nn.RMSNorm(dim)
        
        if attention_type == "global":
            print('Using global attention')
            self.attn = GlobalAttention(dim, num_heads, dimensionality)
        elif attention_type == "ball":
            print('Using ball attention')
            ball_size = attention_kwargs.get("ball_size", 256)
            self.attn = BallAttention(dim, num_heads, ball_size, dimensionality)
        elif attention_type == "lucidrains_sparse":
            print('Using lucidrains sparse attention')
            self.attn = LucidrainsSparseAttention(dim, num_heads, dimensionality, **(attention_kwargs or {}))
        elif attention_type == "sparse":
            print('Using sparse attention')
            self.attn = NativeSparseAttention(
                dim,
                num_heads,
                ball_size,
                dimensionality=dimensionality,
                **(attention_kwargs or {})
            )
            self.attn = torch.compile(self.attn, backend="inductor", dynamic=False)
        else:
            raise ValueError(f"Unknown attention type: {attention_type}")
            
        self.mlp = SwiGLU(dim, dim * mlp_ratio)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.mlp(self.norm2(x))
        return x
    
@dataclass
class TransformerConfig:
    c_in: int           
    c_hidden: int        
    depth: int           
    num_heads: int     
    block_size: int = 6
    mlp_ratio: int = 4
    dimensionality: int = 3
    ball_size: int = 256
    should_rotate: bool = True
    rotation_angle: float = 45
    attention_type: str = "global"  # "global", "ball" or "sparse"
    attention_kwargs: Optional[dict] = None
    
class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.dimensionality = cfg.dimensionality
        
        self.embed = nn.Linear(cfg.c_in, cfg.c_hidden)

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    cfg.c_hidden,
                    cfg.num_heads,
                    cfg.mlp_ratio,
                    cfg.dimensionality,
                    cfg.ball_size,
                    cfg.attention_type,
                    cfg.attention_kwargs,
                )
                for _ in range(cfg.depth)
            ]
        )
        self.block_size = cfg.block_size
        
        self.in_dim = cfg.c_in
        self.out_dim = cfg.c_hidden
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, mean=0., std=0.02, a=-2., b=2.)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def _apply_rotation(self, x: torch.Tensor, pos: torch.Tensor, perm: torch.Tensor):
        inv_perm = torch.argsort(perm)  
        B, n, D = x.shape  
        x_flat = x.view(-1, D)[perm]  
        pos_flat = pos.view(-1, pos.size(-1))[perm]  
        rot_x = x_flat.view_as(x)  
        rot_pos = pos_flat.view_as(pos)  
        return rot_x, rot_pos, inv_perm  
                
    def forward(self, node_features: torch.Tensor, node_positions: torch.Tensor, batch_idx: torch.Tensor, **kwargs) -> torch.Tensor:
        with torch.no_grad():
            if self.cfg.should_rotate:
                tree_idx, tree_mask, tree_idx_rot = build_balltree_with_rotations(node_positions, batch_idx, [1], [self.cfg.ball_size, self.cfg.ball_size], self.cfg.rotation_angle)
            else:
                tree_idx, tree_mask = build_balltree(node_positions, batch_idx)
                tree_idx_rot = None
        x = self.embed(node_features)[tree_idx]
        pos = node_positions[tree_idx]
        
        batch_size = int(batch_idx.max().item()) + 1
        x = rearrange(x, "(B n) D -> B n D", B=batch_size)
        pos = rearrange(pos, "(B n) D -> B n D", B=batch_size)
        
        block_residual = x
        for layer_index, layer in enumerate(self.layers):
            should_rotate_layer = self.cfg.should_rotate and (layer_index % 2 == 1)
            
            # Forward rotation
            if should_rotate_layer:
                perm = tree_idx_rot[layer_index % 2]
                x, pos, inv_perm = self._apply_rotation(x, pos, perm)
                
            x = layer(x, pos)
            
            # Undo rotation
            if should_rotate_layer:
                x_flat = x.view(-1, x.size(-1))[inv_perm]
                pos_flat = pos.view(-1, pos.size(-1))[inv_perm]
                x = x_flat.view_as(x)
                pos = pos_flat.view_as(pos)
            
            if should_apply_block_residual(layer_index, self.cfg.depth, self.block_size):
                x = x + block_residual
                block_residual = x
            
        x = rearrange(x, "B n D -> (B n) D")
        
        x = x[tree_mask]
        x = x[torch.argsort(tree_idx[tree_mask])]
        
        return x