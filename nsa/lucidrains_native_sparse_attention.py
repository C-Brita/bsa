from __future__ import annotations

from copy import deepcopy
from math import ceil
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, arange, stack, cat, tensor, Tensor
from torch.nn import Module, ModuleList

from local_attention import LocalAttention
from rotary_embedding_torch import RotaryEmbedding

import einx
from einops import einsum, repeat, rearrange, reduce, pack, unpack
from einops.layers.torch import Rearrange

from bsa.nsa.utils import (exists, default, round_down_mult, round_up_mult, divisible_by, 
                                        is_empty, max_neg_value, pad_at_dim, straight_through)

# b - batch
# h - heads
# qh - grouped query heads
# n - sequence (token level or compressed)
# w - windows, for fine or compressed
# i, j - query / key sequence
# d - feature dimension
# s - strategies

class BallAttention(nn.Module):
    def __init__(self, num_heads: int, head_dim: int, ball_size: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = head_dim
        self.ball_size = ball_size

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        B, H, n, Dh = q.shape
        assert n % self.ball_size == 0
        m = n // self.ball_size
        
        q_ball = rearrange(q, 'B H (m n) D -> (B m) H n D', n=self.ball_size)
        k_ball = rearrange(k, 'B H (m n) D -> (B m) H n D', n=self.ball_size)
        v_ball = rearrange(v, 'B H (m n) D -> (B m) H n D', n=self.ball_size)

        attn_out = F.scaled_dot_product_attention(
            q_ball, k_b, v_b,
            attn_mask=None, 
            is_causal=False
        )
        out = rearrange(attn_out, '(B m) H n D -> B H (m n) D', B=B, m=m)
        return out

def attend(q, k, v, return_sim = False, scale = None):
    scale = default(scale, q.shape[-1] ** -0.5)
    
    q_heads, k_heads = q.shape[1], k.shape[1]
    grouped_queries_number = q_heads // k_heads
    
    q = rearrange(q, 'b (h qh) ... -> b h qh ...', qh = grouped_queries_number)
    sim = einsum(q, k, 'b h qh i d, b h j d -> b h qh i j') * scale
    
    mask_value = max_neg_value(sim)
        
    attention = sim.softmax(dim = -1)
    attention_out = einsum(attention, v, 'b h qh i j, b h j d -> b h qh i d')
    attention_out = rearrange(attention_out, 'b h qh ... -> b (h qh) ...')
    
    if not return_sim:
        return attention_out
    
    sim = rearrange(sim, 'b h qh ... -> b (h qh) ...')
    return attention_out, sim

class LucidrainsSparseAttention(Module):
    def __init__(self, dim: int, dim_head: int, heads: int, sliding_window_size: int, 
                 compress_block_size: int, compress_block_sliding_stride: int, selection_block_size: int, selected_blocks_number: int,
                 kv_heads: int = None, memory_compressed_kv_number: int = 1,
                 compress_mlp: Module | None = None, compress_mlp_expand_factor: float = 1.0,
                 use_sliding_window: bool = True, ball_size: int = 256):
        super().__init__()
        
        kv_heads = default(kv_heads, heads)
        assert kv_heads <= heads and divisible_by(heads, kv_heads)
        
        self.heads = heads
        self.dim_head = dim_head
        self.kv_heads = kv_heads
        self.grouped_queries_number = heads // kv_heads
        
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads
        dim_kv_inner = dim_head * kv_heads
        
        self.norm = nn.RMSNorm(dim)
        self.rotary_emb = RotaryEmbedding(dim_head)
        
        qkv_split = (dim_inner, dim_kv_inner, dim_kv_inner)
        self.to_qkv = nn.Linear(dim, sum(qkv_split), bias = False)
        self.qkv_split = qkv_split
        
        self.use_sliding_window = use_sliding_window
        
        if use_sliding_window:
            self.sliding_window = LocalAttention(
                dim = dim_head,
                window_size = sliding_window_size,
                causal = False,
                exact_windowsize = True,
                autopad = True,
                use_rotary_pos_emb = False
            )
            self.sliding_window_size = sliding_window_size
        else: 
            self.ball_attn = BallAttention(
                num_heads = kv_heads,     
                head_dim  = dim_head,    
                ball_size = ball_size
            )
            self.ball_size = ball_size
            
        self.compress_block_size = compress_block_size
        self.compress_block_sliding_stride = compress_block_sliding_stride
        assert self.compress_block_size >= self.compress_block_sliding_stride, 'compress_block_size must be >= compress_block_sliding_stride'
        assert self.compress_block_sliding_stride > 0, 'compress_block_sliding_stride must be greater than 0'
        assert divisible_by(selection_block_size, self.compress_block_sliding_stride), f'selection_block_size {selection_block_size} must be divisible by compress_block_sliding_stride {self.compress_block_sliding_stride}'
        
        self.compression_split_window = nn.Sequential(
            Rearrange('b h n d -> (b h) d 1 n'),
            nn.ZeroPad2d(((compress_block_size - compress_block_sliding_stride), 0, 0, 0)),
            nn.Unfold(kernel_size=(1, self.compress_block_size), stride=(1, self.compress_block_sliding_stride)),
            Rearrange('(b h) (d n) w -> b h w n d', d=dim_head, h=kv_heads, n=self.compress_block_size)
        )
        
        assert memory_compressed_kv_number > 0
        self.memory_compressed_kv_number = memory_compressed_kv_number
        self.memory_compressed_kv = nn.Parameter(torch.zeros(2, kv_heads, memory_compressed_kv_number, dim_head))
        
        self.k_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))
        self.v_intrablock_positions = nn.Parameter(torch.zeros(kv_heads, self.compress_block_size, dim_head))

        if not exists(compress_mlp):
            compress_dim = self.compress_block_size * dim_head
            compress_mlp_dim_hidden = int(compress_mlp_expand_factor * compress_dim)
            
            compress_mlp = nn.Sequential(
                Rearrange('b h w n d -> b h w (n d)'),
                nn.Linear(compress_dim, compress_mlp_dim_hidden),
                nn.ReLU(),
                nn.Linear(compress_mlp_dim_hidden, dim_head),
            )
            
        self.k_compress = deepcopy(compress_mlp)
        self.v_compress = deepcopy(compress_mlp)
        
        self.selection_block_size = selection_block_size
        self.selected_blocks_number = selected_blocks_number
        
        strategy_combine_mlp = nn.Linear(dim, 3 * heads)
        nn.init.zeros_(strategy_combine_mlp.weight)
        strategy_combine_mlp.bias.data.copy_(tensor([-2., -2., 2.] * heads))
            
        self.to_strategy_combine = nn.Sequential(
            strategy_combine_mlp,
            nn.Sigmoid(),
            Rearrange('b n (h s) -> b h n s', h = heads)
        )
        
        self.split_heads = Rearrange('b n (h d) -> b h n d', d = dim_head)
        self.merge_heads = Rearrange('b h n d -> b n (h d)')
        self.combine_heads = nn.Linear(dim_inner, dim, bias = False)
        
    def forward(self, input: Tensor) -> Tensor:
        """ Input shape: (B, n, D) """
        batch, sequence_length, scale, heads, kv_heads, device = *input.shape[:2], self.scale, self.heads, self.kv_heads, input.device
        
        compress_divisible_sequence_length = round_down_mult(sequence_length, self.compress_block_sliding_stride)
        compressed_blocks_number = compress_divisible_sequence_length // self.compress_block_sliding_stride
        
        compress_overlap_length = self.compress_block_size - self.compress_block_sliding_stride
        has_compress_overlap = compress_overlap_length > 0
        
        fine_divisible_sequence_length = round_up_mult(sequence_length, self.selection_block_size)
        fine_blocks_number = fine_divisible_sequence_length // self.selection_block_size
        
        input = self.norm(input)
        
        qkv = self.to_qkv(input)
        q, k, v = qkv.split(self.qkv_split, dim = -1)
        q, k, v = map(self.split_heads, (q, k, v))
        
        k_compress_input, v_compress_input = k[..., :compress_divisible_sequence_length, :], v[..., :compress_divisible_sequence_length, :]
    
        if not is_empty(k_compress_input):
            k_compress_input = self.compression_split_window(k_compress_input)
            v_compress_input = self.compression_split_window(v_compress_input)
        
            k_compress_input = einx.add('b h w n d, h n d', k_compress_input, self.k_intrablock_positions)
            v_compress_input = einx.add('b h w n d, h n d', v_compress_input, self.v_intrablock_positions)
        else:
            k_compress_input, v_compress_input = tuple(t.reshape(batch, kv_heads, 0, self.compress_block_size, self.dim_head) for t in (k_compress_input, v_compress_input))

        compressed_q = q
        compressed_k = self.k_compress(k_compress_input) # Eq. (7) from the Native Sparse Attention paper
        compressed_v = self.v_compress(v_compress_input)
        
        # 1. Coarse attention over compressed keys and values
        memory_compressed_k, memory_compressed_v = repeat(self.memory_compressed_kv, 'kv ... -> kv b ...', b = batch)
        memory_compressed_kv_number = memory_compressed_k.shape[-2]
        
        compressed_attention_out, compressed_sim = attend(compressed_q, compressed_k, compressed_v, return_sim = True)

        q, k = self.rotary_emb.rotate_queries_with_cached_keys(q, k)
        
        # 2. Fine attention over selected based on compressed attention logits
        importance_scores = compressed_sim[..., memory_compressed_kv_number:]
        
        selected_blocks_number = min(self.selected_blocks_number, compressed_blocks_number)
        has_selected_kv_for_fine_attention = selected_blocks_number > 0
        
        importance_scores = reduce(importance_scores, 'b (h grouped_queries) ... -> b h ...', 'mean', grouped_queries = self.grouped_queries_number)
        fine_grouped_querie_number = self.grouped_queries_number
        
        if has_selected_kv_for_fine_attention:
            if self.compress_block_sliding_stride != self.selection_block_size:
                compress_per_fine_number = self.selection_block_size // self.compress_block_sliding_stride
                round_down_score_length = round_down_mult(importance_scores.shape[-1], compress_per_fine_number)
                importance_scores = importance_scores[..., :round_down_score_length]
                
                if not is_empty(importance_scores):
                    importance_scores = reduce(importance_scores, '... (j compress_per_fine_number) -> ... j', 'mean', compress_per_fine_number = compress_per_fine_number)
                    
                    i, j = importance_scores.shape[-2:]
                    q_sequence = arange(i, device = device) // self.selection_block_size
                    k_sequence = arange(j, device = device)
                    
                    block_diagonal_mask = einx.equal('i, j -> i j', q_sequence, k_sequence)
                    importance_scores = importance_scores.masked_fill(block_diagonal_mask, max_neg_value(compressed_sim))
            
            importance_scores = F.pad(importance_scores, (1, 0), value = -1e3)
            importance_scores = importance_scores.softmax(dim = -1)
            importance_scores = importance_scores[..., 1:]
            
        fine_q = q
        fine_k = k
        fine_v = v
        
        selected_blocks_number = min(selected_blocks_number, importance_scores.shape[-1])
        has_selected_kv_for_fine_attention = selected_blocks_number > 0
        
        remainder = fine_divisible_sequence_length - sequence_length
        pad_to_multiple = partial(pad_at_dim, pad = (0, remainder), dim = -2)
        
        if has_selected_kv_for_fine_attention:
            selected_importance_values, selected_block_indices = importance_scores.topk(selected_blocks_number, dim = -1)
            gates = straight_through(selected_importance_values, 1.0)
            
            fine_mask = selected_importance_values > 1e-10
            
            if sequence_length < fine_divisible_sequence_length:
                fine_k, fine_v, fine_q = map(pad_to_multiple, (fine_k, fine_v, fine_q))
                
                fine_mask = pad_at_dim(fine_mask, (0, remainder), value = False, dim = -2)
                selected_block_indices = pad_at_dim(selected_block_indices, (0, remainder), value = 0, dim = -2)
                
                if exists(gates):
                    gates = pad_at_dim(gates, (0, remainder), value = 0, dim = -2)
                    
            fine_mask = repeat(fine_mask, 'b h i w -> b h 1 i (w j)', j = self.selection_block_size)
            
            fine_k = rearrange(fine_k, 'b h (w n) d -> b h w n d', w = fine_blocks_number)
            fine_v = rearrange(fine_v, 'b h (w n) d -> b h w n d', w = fine_blocks_number)
            
            fine_k = repeat(fine_k, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
            fine_v = repeat(fine_v, 'b h w j d -> b h i w j d', i = selected_block_indices.shape[2])
            
            selected_block_indices = repeat(selected_block_indices, 'b h i sel -> b h i sel j d', j = fine_k.shape[-2], d = fine_k.shape[-1])
            fine_k = fine_k.gather(3, selected_block_indices)
            fine_v = fine_v.gather(3, selected_block_indices)
            
            fine_k = einx.multiply('b h i sel, b h i sel j d -> b h i sel j d', gates, fine_k)
            fine_k, fine_v = tuple(rearrange(t, 'b h i w j d -> b h i (w j) d') for t in (fine_k, fine_v))
            
            fine_q = rearrange(fine_q, 'b (h qh) ... -> b h qh ...', qh = fine_grouped_querie_number)
            fine_sim = einsum(fine_q, fine_k, 'b h qh i d, b h i j d -> b h qh i j') * self.scale
            
            mask_value = max_neg_value(fine_sim)
            fine_sim = fine_sim.masked_fill(~fine_mask, mask_value)
            
            fine_attention = fine_sim.softmax(dim = -1)
            fine_attention_out = einsum(fine_attention, fine_v, 'b h qh i j, b h i j d -> b h qh i d')
            fine_attention_out = rearrange(fine_attention_out, 'b h qh ... -> b (h qh) ...')
            fine_attention_out = fine_attention_out[..., :sequence_length, :]
        else:
            sequence_length = fine_k.shape[-2]
            fine_mask = None
            
            fine_k, fine_v, fine_q = map(pad_to_multiple, (fine_k, fine_v, fine_q))
            fine_q, fine_k, fine_v = tuple(rearrange(t, 'b h (w n) d -> (b w) h n d', n = self.selection_block_size) for t in (fine_q, fine_k, fine_v))
            
            fine_attention_out = attend(fine_q, fine_k, fine_v, return_sim = False)
            fine_attention_out = rearrange(fine_attention_out, '(b w) h n d -> b (h w) n d', b = batch)
            fine_attention_out = fine_attention_out[..., :sequence_length, :]
        
        local_attention_out = None
        if self.use_sliding_window:
            sliding_window_q = q 
            sliding_window_k = k
            sliding_window_v = v
    
            sliding_window_k, sliding_window_v = tuple(repeat(t, 'b h ... -> b (h w) ...', w = self.grouped_queries_number) for t in (sliding_window_k, sliding_window_v))            
            
            local_attention_out = self.sliding_window(sliding_window_q, sliding_window_k, sliding_window_v)
        else: 
            local_attention_q = q
            local_attention_k = k
            local_attention_v = v
            
            local_attention_out = self.ball_attn(local_attention_q, local_attention_k, local_attention_v)
    
        strategy_weighted_combine = self.to_strategy_combine(input)
        out = einsum(strategy_weighted_combine, stack([compressed_attention_out, fine_attention_out, local_attention_out]), 'b h n s, s b h n d -> b h n d')
        out = self.merge_heads(out)
        out = self.combine_heads(out)
        return out