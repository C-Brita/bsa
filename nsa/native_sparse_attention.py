import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from copy import deepcopy   

class StaticGatedAttention(nn.Module):
    def __init__(self, num_heads: int):
        super().__init__()
        
        self.gates = nn.Parameter(torch.zeros(1, num_heads, 2, 1, 1))
        
    def forward(self, branches: torch.Tensor) -> torch.Tensor:
        ones = torch.ones_like(self.gates[:, :, :1])
        fused = torch.cat([ones, torch.sigmoid(self.gates)], 2)
        
        return (fused * branches).sum(dim=2)
    
class TokenGatedAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.to_strategy = nn.Sequential(
            nn.Linear(dim, 3 * num_heads),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, branches: torch.Tensor) -> torch.Tensor:
        B, H, N, S, D = branches.shape
        gates = self.to_strategy(x).view(B, N, H, 3).permute(0, 2, 1, 3)  
        weights = gates.unsqueeze(-1) 
        return (weights * branches).sum(dim=3)
            
class NativeSparseAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, ball_size: int, dimensionality: int = 3,
                 compress_block_size: int = 8, selected_blocks_number: int = 4, 
                 use_compress_mlp: bool = True, compress_mlp_expand_factor: float = 2.0,
                 use_group_selection: bool = True, group_selection_size: int = 8,
                 should_mask_blocks_in_ball: bool = True, use_token_gated_attention: bool = True,
                 use_coarse_q_attn: bool = True, use_coarse_q_importance: bool = True):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert not use_coarse_q_attn or use_coarse_q_importance, "If you enable coarse-Q attention you must also enable coarse-Q importance scoring"
        assert not use_coarse_q_attn or use_group_selection, "If you enable coarse-Q attention you must also enable group selection"
        
        # General parameters
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out_proj = nn.Linear(dim, dim)
        self.position_proj = nn.Linear(dimensionality, dim)
        
        # Local attention parameters
        self.ball_size = ball_size
        self.sigma_att = nn.Parameter(-1 + 0.01 * torch.randn(1, num_heads, 1, 1))
        
        # Coarse attention parameters
        self.compress_block_size = compress_block_size
        self.use_compress_mlp = use_compress_mlp
        
        self.use_coarse_q_attn = use_coarse_q_attn
        self.use_coarse_q_importance = use_coarse_q_importance
        
        if use_compress_mlp:
            compress_dimension = compress_block_size * (dim // num_heads)
            hidden_dim = int(compress_mlp_expand_factor * compress_dimension)
            
            self.k_compress_mlp = nn.Sequential(
                nn.Linear(compress_dimension,    hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim,  dim // num_heads),
            )
            self.v_compress_mlp = deepcopy(self.k_compress_mlp)
            
            if self.use_coarse_q_attn or self.use_coarse_q_importance:
                self.q_compress_mlp = deepcopy(self.k_compress_mlp)

        # Fine attention parameters
        self.selected_blocks_number = selected_blocks_number
        
        self.use_group_selection = use_group_selection
        self.group_selection_size = group_selection_size
        
        self.should_mask_blocks_in_ball = should_mask_blocks_in_ball
        
        # Gated attention
        self.use_token_gated_attention = use_token_gated_attention
        if use_token_gated_attention:
            self.gated_attention = TokenGatedAttention(dim, num_heads)
        else: 
            self.gated_attention = StaticGatedAttention(num_heads)
        
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
    
    def compute_local_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, pos: torch.Tensor):
        B, H, n, D = q.shape
        m = n // self.ball_size
        
        q = rearrange(q, 'B H (m b) D -> (B m) H b D', m=m, b=self.ball_size)
        k = rearrange(k, 'B H (m b) D -> (B m) H b D', m=m, b=self.ball_size)
        v = rearrange(v, 'B H (m b) D -> (B m) H b D', m=m, b=self.ball_size)
        
        local_attention = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask = self.create_attention_bias(pos),
            is_causal = False,
        )  
        return rearrange(local_attention, "(B m) H b D -> B H (m b) D", B=B, m=m) 
    
    def compute_importance_scores(self, q: torch.Tensor, k_block: torch.Tensor, scale: float, chunk_positions: int = 16_384):
        """Return sim  shape (B, H, n, num_blocks)  without allocating huge tensors."""
        B, H, n, Dh = q.shape

        if chunk_positions is None or n <= chunk_positions:
            return torch.einsum('B H i D, B H j D -> B H i j', q, k_block) * scale

        sims = []
        for q_chunk in q.split(chunk_positions, dim=2):
            sim = torch.einsum('B H i D, B H j D -> B H i j',
                               q_chunk, k_block) * scale
            sims.append(sim)
        return torch.cat(sims, dim=2)
        
    def compute_coarse_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        B, H, n, D = q.shape
        compressed_block_size = self.compress_block_size
        compressed_block_number = n // self.compress_block_size
        self.compressed_block_number = compressed_block_number
            
        k_blocks = k.view(B, H, compressed_block_number, compressed_block_size, D)
        v_blocks = v.view(B, H, compressed_block_number, compressed_block_size, D)
        
        if self.use_compress_mlp:
            coarse_k = self.k_compress_mlp(k_blocks.flatten(-2))
            coarse_v = self.v_compress_mlp(v_blocks.flatten(-2))
            
            if self.use_coarse_q_attn or self.use_coarse_q_importance:
                q_blocks = q.view(B, H, compressed_block_number, compressed_block_size, D)
                coarse_q = self.q_compress_mlp(q_blocks.flatten(-2))
            
        else:
            coarse_k = k_blocks.mean(dim=3)
            coarse_v = v_blocks.mean(dim=3)
            
            if self.use_coarse_q_attn or self.use_coarse_q_importance:
                q_blocks = q.view(B, H, compressed_block_number, compressed_block_size, D)
                coarse_q = q_blocks.mean(dim=3)

        if self.use_coarse_q_attn:
            coarse_attention = F.scaled_dot_product_attention(
                coarse_q, coarse_k, coarse_v,
                attn_mask=None,
                is_causal=False,
            )
            coarse_attention = coarse_attention.repeat_interleave(compressed_block_size, dim=2)
        else:
            coarse_attention = F.scaled_dot_product_attention(
                q, coarse_k, coarse_v,
                attn_mask=None,
                is_causal=False,
            )
        
        if self.use_coarse_q_attn or self.use_coarse_q_importance:
            importance_scores = self.compute_importance_scores(coarse_q, coarse_k, 1 / math.sqrt(D), chunk_positions=16_384)
        else:
            importance_scores = self.compute_importance_scores(q, coarse_k, 1 / math.sqrt(D), chunk_positions=16_384)
        
        return coarse_attention, importance_scores
    
    def compute_fine_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, importance_scores: torch.Tensor):
        B, H, n, D = q.shape
        compress_block_size = self.compress_block_size
        compress_block_number = n // compress_block_size
        blocks_per_ball = self.ball_size // compress_block_size
        use_block_level_importance_scores = self.use_coarse_q_attn or self.use_coarse_q_importance

        if self.should_mask_blocks_in_ball:
            if use_block_level_importance_scores: 
                query_block_index     = torch.arange(compress_block_number, device=q.device)  
                candidate_block_index = query_block_index                                      

                query_ball_index      = query_block_index    // blocks_per_ball               
                candidate_ball_index  = candidate_block_index// blocks_per_ball               

                overlap_mask = (candidate_ball_index.unsqueeze(0) == query_ball_index.unsqueeze(1))
                overlap_mask = overlap_mask.unsqueeze(0).unsqueeze(0)                         

            else:                                      
                query_pos            = torch.arange(n, device=q.device)                       
                candidate_block_idx  = torch.arange(compress_block_number, device=q.device)   

                query_ball_index     = query_pos            // self.ball_size                 
                candidate_ball_index = candidate_block_idx  // blocks_per_ball                

                overlap_mask = (candidate_ball_index.unsqueeze(0) ==
                                query_ball_index.unsqueeze(1))                                
                overlap_mask = overlap_mask.unsqueeze(0).unsqueeze(0)                         

            importance_scores = importance_scores.masked_fill(overlap_mask, float('-inf'))
            
        if self.use_group_selection: 
            group_size = self.group_selection_size
            assert n % group_size == 0, "n must be divisible by group_selection_size"
            group_number = n // group_size
            
            if use_block_level_importance_scores:
                assert self.group_selection_size == compress_block_size, (
                    f"group_selection_size ({self.group_selection_size}) must equal "
                    f"compress_block_size ({compress_block_size}) when using block-level importance scores."
                )
            else: 
                importance_scores = importance_scores.view(B, H, group_number, group_size, compress_block_number).mean(dim=3)
                
            top_group_blocks = torch.topk(importance_scores, self.selected_blocks_number, dim=-1).indices                
            top_blocks = top_group_blocks.repeat_interleave(group_size, dim=2)
        else:
            top_blocks = torch.topk(importance_scores, self.selected_blocks_number, dim=-1).indices
            
        top_blocks, _ = torch.sort(top_blocks, dim=-1)
        
        offsets = torch.arange(compress_block_size, device=q.device)  
        position_indices = (top_blocks.unsqueeze(-1) * compress_block_size + offsets).reshape(B, H, n, -1)

        m = position_indices.shape[-1]    
        
        idx = position_indices.reshape(B, H, n * m)            
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, -1, D)

        fine_k = k.gather(2, idx_expanded).view(B, H, n, m, D)
        fine_v = v.gather(2, idx_expanded).view(B, H, n, m, D)
        
        flash_q = q.unsqueeze(-2).reshape(-1, 1, D)
        flash_k = fine_k.reshape(-1, m, D)
        flash_v = fine_v.reshape(-1, m, D)
        
        fine_attention = F.scaled_dot_product_attention(
            flash_q, flash_k, flash_v, 
            attn_mask = None,
            is_causal = False
        )
        return fine_attention.squeeze(1).view(B, H, n, D)
        
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        """ Input shape: (B, n, D) """
        B, n, _ = x.shape
        assert n % self.ball_size == 0, "n must be divisible by ball_size"
        assert n % self.compress_block_size == 0, "n must be divisible by compress_block_size"
        
        x = x + self.position_proj(self.compute_relative_positions(pos))
        
        q, k, v = rearrange(self.qkv(x), "B n (K H D) -> K B H n D", K=3, H=self.num_heads, D=self.head_dim)
        
        coarse_attention, importance_scores = self.compute_coarse_attention(q, k, v)
        fine_attention = self.compute_fine_attention(q, k, v, importance_scores)
        local_attention = self.compute_local_attention(q, k, v, pos)
        
        if self.use_token_gated_attention:
            branches = torch.stack([local_attention, coarse_attention, fine_attention], dim=3)
            fused_attention = self.gated_attention(x, branches)
        else:
            branches = torch.stack([local_attention, coarse_attention, fine_attention], dim=2)
            fused_attention = self.gated_attention(branches)
        
        out = rearrange(fused_attention, 'B H n D -> B n (H D)')
        return self.out_proj(out)