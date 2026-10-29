#!/usr/bin/env python3
"""
Live Prefix / Prompt Caching Benchmark with Ollama.
Demonstrates how reusing a prompt prefix avoids recomputing KV caches,
slashing Time-To-First-Token (TTFT) and prefill latency.
"""

import json
from typing import Dict
import urllib.request
from ollama_profiler import profile_generation


SHARED_SYSTEM_PREFIX = """
You are a senior high-performance computing architect specializing in deep learning compilers,
GPU memory hierarchies, systolic arrays, tensor cores, and distributed inference serving systems.
Analyze technical problems with rigorous mathematical principles, citing arithmetic intensity,
SRAM versus HBM bandwidth, cache line sizes, roofline models, and kernel fusion techniques.
Context Knowledge:
- GPUs possess High Bandwidth Memory (HBM) with bandwidths ranging from 1.5 TB/s to 8 TB/s.
- SRAM on-chip caches (L1/Shared Memory) provide bandwidth exceeding 15 to 30 TB/s.
- The decode phase of transformer models is bottlenecked by the streaming of weights from HBM to SRAM.
- The KV cache avoids redundant key and value recomputation by storing past token representations.
- PagedAttention eliminates internal and external memory fragmentation using OS-style page tables.
- Continuous batching schedules at the token iteration boundary instead of static request boundaries.
- Speculative decoding leverages low-cost draft models to verify multiple tokens in one target forward pass.
"""


def test_prefix_caching(model: str = "bonsai-2:latest"):
    print("=" * 84)
    print(f"  LIVE CONTEXT / KV CACHE REUSE BENCHMARK ON OLLAMA ({model})")
    print("=" * 84)

    # --- Call 1: Full Prompt Prefill (Cold) ---
    print("\n[Run 1: Cold Start] Generating initial response with long system prompt...")
    payload_1 = {
        "model": model,
        "prompt": "Briefly list the primary benefits of KV caching.",
        "system": SHARED_SYSTEM_PREFIX,
        "stream": False,
        "options": {"num_predict": 25}
    }
    req_1 = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload_1).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_1) as resp:
        r1 = json.loads(resp.read().decode("utf-8"))

    p1_tokens = r1.get("prompt_eval_count", 0)
    p1_ms = r1.get("prompt_eval_duration", 0) / 1e6
    p1_tok_per_sec = (p1_tokens / (p1_ms / 1000.0)) if p1_ms > 0 else 0
    saved_context = r1.get("context", [])

    print(f"  • Prompt Tokens Processed:    {p1_tokens}")
    print(f"  • Prefill Latency:            {p1_ms:.2f} ms")
    print(f"  • Prefill Speed:              {p1_tok_per_sec:.2f} tok/s")
    print(f"  • KV Cache Context Size:      {len(saved_context)} tokens stored in VRAM")

    # --- Call 2: Turn 2 with Context Reused (Warm KV Cache) ---
    print("\n[Run 2: Follow-up Turn with Reused KV Cache] Sending follow-up prompt...")
    payload_2 = {
        "model": model,
        "prompt": "And what is the main trade-off?",
        "context": saved_context,  # Reusing the existing KV Cache!
        "stream": False,
        "options": {"num_predict": 25}
    }
    req_2 = urllib.request.Request(
        "http://localhost:11434/api/generate",
        data=json.dumps(payload_2).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req_2) as resp:
        r2 = json.loads(resp.read().decode("utf-8"))

    p2_tokens = r2.get("prompt_eval_count", 0)
    p2_ms = r2.get("prompt_eval_duration", 0) / 1e6
    p2_tok_per_sec = (p2_tokens / (p2_ms / 1000.0)) if p2_ms > 0 else 0

    print(f"  • New Prompt Tokens Processed: {p2_tokens}")
    print(f"  • Prefill Latency:             {p2_ms:.2f} ms")
    print(f"  • Prefill Speed:               {p2_tok_per_sec:.2f} tok/s")

    # --- Comparison Summary ---
    print("\n" + "-" * 84)
    print("  KV CACHE PERSISTENCE SUMMARY")
    print("-" * 84)
    print(f"{'Metric':<36} | {'Run 1 (Cold Prefill)':<20} | {'Run 2 (Reused KV Cache)':<20}")
    print("-" * 84)
    print(f"{'Tokens Evaluated':<36} | {p1_tokens:<20} | {p2_tokens:<20}")
    print(f"{'Prefill Latency':<36} | {p1_ms:<17.2f} ms | {p2_ms:<17.2f} ms")
    
    speedup = p1_ms / max(p2_ms, 0.01)
    print(f"{'TTFT / Prefill Acceleration':<36} | {'1.00x (Baseline)':<20} | {speedup:<17.2f} x faster")
    print("=" * 84)
    print("Key Takeaway: Passing the KV cache context avoids re-evaluating the hundreds")
    print("of previous prompt tokens, slashing prefill latency directly!")



if __name__ == "__main__":
    test_prefix_caching()
