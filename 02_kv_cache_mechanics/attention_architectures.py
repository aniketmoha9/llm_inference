#!/usr/bin/env python3
"""
Attention Architectures: MHA, MQA, GQA, and DeepSeek MLA.
Clean PyTorch implementations demonstrating tensor dimensions,
KV Cache memory scaling, and compression ratios.
"""

from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


class MultiHeadAttention(nn.Module):
    """Standard Multi-Head Attention (MHA): H_kv = H_q."""
    def __init__(self, d_model: int = 4096, num_heads: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), (k, v)


class MultiQueryAttention(nn.Module):
    """Multi-Query Attention (MQA): 1 KV head shared across all Q heads."""
    def __init__(self, d_model: int = 4096, num_heads: int = 32):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, self.head_dim, bias=False)  # Single KV head
        self.v_proj = nn.Linear(d_model, self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)  # [B, 1, S, head_dim]
        v = self.v_proj(x).view(B, S, 1, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        # Broadcast single KV head across all num_heads query heads
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), (k, v)


class GroupedQueryAttention(nn.Module):
    """Grouped-Query Attention (GQA): num_kv_heads groups (e.g. 8 KV heads for 32 Q heads)."""
    def __init__(self, d_model: int = 4096, num_heads: int = 32, num_kv_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.group_size = num_heads // num_kv_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None):
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        if kv_cache is not None:
            k_past, v_past = kv_cache
            k = torch.cat([k_past, k], dim=2)
            v = torch.cat([v_past, v], dim=2)

        # Repeat KV heads to match query heads: [B, num_heads, S, head_dim]
        k_expanded = k.repeat_interleave(self.group_size, dim=1)
        v_expanded = v.repeat_interleave(self.group_size, dim=1)

        scores = torch.matmul(q, k_expanded.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_expanded).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), (k, v)


class DeepSeekMLA(nn.Module):
    """
    Multi-Head Latent Attention (MLA) from DeepSeek-V2 / DeepSeek-V3 / DeepSeek-R1.
    Instead of caching H_kv x head_dim per layer, MLA projects KV into a compact
    latent vector c_kv of dimension d_c, plus a decoupled RoPE key vector k_rope.
    """
    def __init__(
        self,
        d_model: int = 4096,
        num_heads: int = 32,
        head_dim: int = 128,
        kv_latent_dim: int = 512,  # Compressed latent dimension d_c
        rope_head_dim: int = 64,   # Decoupled RoPE dimension
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.kv_latent_dim = kv_latent_dim
        self.rope_head_dim = rope_head_dim

        # Down-projection for KV into compressed latent space
        self.w_dkv = nn.Linear(d_model, kv_latent_dim, bias=False)
        # Decoupled RoPE Key projection
        self.w_kr = nn.Linear(d_model, rope_head_dim, bias=False)

        # Up-projection from latent vector into multi-head keys and values
        self.w_uk = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)
        self.w_uv = nn.Linear(kv_latent_dim, num_heads * head_dim, bias=False)

        # Query projection (includes content query + decoupled RoPE query)
        self.w_q = nn.Linear(d_model, num_heads * head_dim, bias=False)
        self.w_qr = nn.Linear(d_model, num_heads * rope_head_dim, bias=False)

        self.o_proj = nn.Linear(num_heads * head_dim, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mla_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        """
        MLA caches ONLY (c_kv, k_rope) instead of the full (K, V) tensors!
        c_kv shape:   [B, S, kv_latent_dim]
        k_rope shape: [B, S, rope_head_dim]
        """
        B, S, _ = x.shape

        # 1. Compress KV to latent vector and decoupled RoPE key
        c_kv = self.w_dkv(x)       # [B, S, kv_latent_dim]
        k_rope = self.w_kr(x)      # [B, S, rope_head_dim]

        if mla_cache is not None:
            c_kv_past, k_rope_past = mla_cache
            c_kv = torch.cat([c_kv_past, c_kv], dim=1)
            k_rope = torch.cat([k_rope_past, k_rope], dim=1)

        S_total = c_kv.shape[1]

        # 2. Up-project cached latent vector into multi-head K and V
        k_content = self.w_uk(c_kv).view(B, S_total, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.w_uv(c_kv).view(B, S_total, self.num_heads, self.head_dim).transpose(1, 2)

        # 3. Query projections
        q_content = self.w_q(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        q_rope = self.w_qr(x).view(B, S, self.num_heads, self.rope_head_dim).transpose(1, 2)
        k_rope_expanded = k_rope.unsqueeze(1).expand(-1, self.num_heads, -1, -1)

        # 4. Attention score = Content score + Decoupled RoPE score
        content_scores = torch.matmul(q_content, k_content.transpose(-2, -1))
        rope_scores = torch.matmul(q_rope, k_rope_expanded.transpose(-2, -1))
        scores = (content_scores + rope_scores) / ((self.head_dim + self.rope_head_dim) ** 0.5)

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(out), (c_kv, k_rope)


def print_kv_memory_comparison():
    print("=" * 86)
    print("  KV CACHE ARCHITECTURAL MEMORY COMPARISON (32 Layers, FP16)")
    print("=" * 86)

    layers = 32
    d_model = 4096
    num_heads = 32
    head_dim = 128
    precision_bytes = 2

    # Dimensions stored per token per layer:
    # MHA: 2 * num_heads * head_dim
    mha_elements = 2 * num_heads * head_dim
    # GQA (8 KV heads): 2 * 8 * head_dim
    gqa_elements = 2 * 8 * head_dim
    # MQA (1 KV head): 2 * 1 * head_dim
    mqa_elements = 2 * 1 * head_dim
    # DeepSeek MLA: kv_latent_dim (512) + rope_head_dim (64)
    mla_elements = 512 + 64

    seq_lengths = [2048, 8192, 32768, 131072]

    header = f"{'Architecture':<22} | {'Per-Token Elements':<18} | {'2K Context':<10} | {'8K Context':<10} | {'32K Context':<11} | {'128K Context':<12}"
    print(header)
    print("-" * len(header))

    configs = [
        ("MHA (LLaMA-1)", mha_elements),
        ("GQA-8 (LLaMA-3)", gqa_elements),
        ("MQA (Falcon)", mqa_elements),
        ("MLA (DeepSeek-V2/3/R1)", mla_elements),
    ]

    for name, elem_per_layer in configs:
        row = f"{name:<22} | {elem_per_layer:<18} | "
        for seq in seq_lengths:
            total_bytes = layers * elem_per_layer * seq * precision_bytes
            mb = total_bytes / (1024 * 1024)
            if mb < 1024:
                row += f"{mb:7.1f} MB | "
            else:
                row += f"{mb/1024:7.2f} GB | "
        print(row)
    print("=" * 86)


if __name__ == "__main__":
    print_kv_memory_comparison()
