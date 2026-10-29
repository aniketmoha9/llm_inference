#!/usr/bin/env python3
"""
Benchmark: Naive Autoregressive Generation vs. KV-Cached Generation.
Demonstrates the O(S^2) quadratic compute penalty of naive re-computation
versus O(1) step latency with KV Caching in PyTorch.
"""

import time
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# Device selection: auto-detect CUDA, Apple Silicon MPS, or fallback to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def sync_device():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()
    elif DEVICE.type == "mps":
        torch.mps.synchronize()


class SelfAttentionBlock(nn.Module):
    def __init__(self, d_model: int = 512, n_heads: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward_naive(self, x: torch.Tensor) -> torch.Tensor:
        """
        Naive forward pass: computes attention across all tokens in x from scratch.
        Input x: [batch, seq_len, d_model]
        """
        B, S, D = x.shape
        Q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Causal mask: allow attending only to previous positions
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        mask = torch.triu(torch.full((S, S), float("-inf"), device=x.device), diagonal=1)
        attn_scores = attn_scores + mask
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)  # [B, n_heads, S, head_dim]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.out_proj(out)

    def forward_with_cache(
        self,
        x_new: torch.Tensor,
        cached_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Cached forward pass: computes Q, K, V only for new token(s) at current step,
        appends K and V to cache, and attends only Q_new to K_cached.
        Input x_new: [batch, 1, d_model]
        """
        B, S_new, D = x_new.shape  # S_new is 1 during decode
        Q_new = self.q_proj(x_new).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
        K_new = self.k_proj(x_new).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)
        V_new = self.v_proj(x_new).view(B, S_new, self.n_heads, self.head_dim).transpose(1, 2)

        if cached_kv is not None:
            k_past, v_past = cached_kv
            K = torch.cat([k_past, K_new], dim=2)
            V = torch.cat([v_past, V_new], dim=2)
        else:
            K, V = K_new, V_new

        # Q_new attends to all keys up to current position: [B, n_heads, 1, S_total]
        attn_scores = torch.matmul(Q_new, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, S_new, D)
        return self.out_proj(out), (K, V)


def benchmark_generation(steps: int = 120, d_model: int = 512, n_heads: int = 8):
    print("=" * 76)
    print(f"  BENCHMARK: Naive Generation vs. KV-Cached Generation on {DEVICE}")
    print(f"  Config: d_model={d_model}, n_heads={n_heads}, steps={steps}")
    print("=" * 76)

    model = SelfAttentionBlock(d_model=d_model, n_heads=n_heads).to(DEVICE).eval()
    batch_size = 1

    # --- 1. Naive Generation (Recompute everything at each step) ---
    print("\n[1] Running Naive Generation (recomputing full prefix every token)...")
    history_tokens = torch.randn(batch_size, 1, d_model, device=DEVICE)
    naive_step_times = []

    start_naive = time.perf_counter()
    with torch.no_grad():
        for i in range(steps):
            step_start = time.perf_counter()
            _ = model.forward_naive(history_tokens)
            sync_device()
            step_end = time.perf_counter()
            naive_step_times.append((step_end - step_start) * 1000.0)

            # Generate next dummy token and concatenate
            next_token = torch.randn(batch_size, 1, d_model, device=DEVICE)
            history_tokens = torch.cat([history_tokens, next_token], dim=1)
    total_naive_time = time.perf_counter() - start_naive

    # --- 2. KV-Cached Generation ---
    print("[2] Running KV-Cached Generation (appending to cache, O(1) step)...")
    cached_kv = None
    current_token = torch.randn(batch_size, 1, d_model, device=DEVICE)
    cached_step_times = []

    start_cached = time.perf_counter()
    with torch.no_grad():
        for i in range(steps):
            step_start = time.perf_counter()
            out, cached_kv = model.forward_with_cache(current_token, cached_kv)
            sync_device()
            step_end = time.perf_counter()
            cached_step_times.append((step_end - step_start) * 1000.0)

            current_token = torch.randn(batch_size, 1, d_model, device=DEVICE)
    total_cached_time = time.perf_counter() - start_cached

    # --- Report Results ---
    print("\n--- STEP LATENCY COMPARISON (Selected Milestones) ---")
    print(f"{'Token Step':<12} | {'Naive Latency (ms)':<20} | {'KV-Cache Latency (ms)':<22} | {'Speedup':<10}")
    print("-" * 72)
    sample_steps = [1, 20, 40, 60, 80, 100, steps - 1]
    for s in sample_steps:
        if s < len(naive_step_times):
            naive_ms = naive_step_times[s]
            cache_ms = cached_step_times[s]
            speedup = naive_ms / max(cache_ms, 1e-4)
            print(f"Step {s+1:<7} | {naive_ms:<20.3f} | {cache_ms:<22.3f} | {speedup:<10.2f}x")

    print("-" * 72)
    print(f"Total Time for {steps} tokens:")
    print(f"  • Naive Total Time:    {total_naive_time * 1000.0:.2f} ms")
    print(f"  • KV-Cache Total Time: {total_cached_time * 1000.0:.2f} ms")
    print(f"  • Overall Speedup:     {total_naive_time / total_cached_time:.2f}x")
    print("=" * 76)


if __name__ == "__main__":
    benchmark_generation()
