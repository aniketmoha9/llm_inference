#!/usr/bin/env python3
"""
Ollama Inference Profiler & Telemetry Analyzer.
Profiles live models running in Ollama on Apple Silicon, measuring:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token / Inter-Token Latency)
- Prefill Throughput (Tokens/s) vs Decode Throughput (Tokens/s)
- Prefix/Prompt Cache hits
- Active VRAM and execution runner (Metal llamacpp vs MLX)
"""

import argparse
import json
import subprocess
import time
import urllib.request
from typing import Any, Dict, Optional


OLLAMA_API_URL = "http://localhost:11434/api/generate"


def get_loaded_model_info() -> Dict[str, str]:
    """Retrieves current VRAM usage, runner, and processor from 'ollama ps'."""
    try:
        res = subprocess.run(["ollama", "ps"], capture_output=True, text=True, check=True)
        lines = res.stdout.strip().split("\n")
        if len(lines) > 1:
            # Header: NAME ID SIZE PROCESSOR CONTEXT RUNNER UNTIL
            parts = lines[1].split()
            if len(parts) >= 6:
                return {
                    "name": parts[0],
                    "size": parts[2] + " " + parts[3] if len(parts) > 6 else parts[2],
                    "processor": parts[3] if len(parts) <= 6 else parts[4],
                    "runner": parts[5] if len(parts) > 6 else parts[5],
                }
    except Exception:
        pass
    return {"name": "Unknown", "size": "N/A", "processor": "N/A", "runner": "N/A"}


def profile_generation(
    model: str = "bonsai-2:latest",
    prompt: str = "Explain how GPU memory bandwidth affects LLM decoding in three concise bullet points.",
    max_tokens: int = 80,
    system_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    """Sends generation request to Ollama and extracts nanosecond-precision metrics."""
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.7,
        }
    }
    if system_prompt:
        payload["system"] = system_prompt

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(OLLAMA_API_URL, data=data, headers={"Content-Type": "application/json"})

    wall_start = time.perf_counter()
    with urllib.request.urlopen(req) as response:
        res_json = json.loads(response.read().decode("utf-8"))
    wall_duration = time.perf_counter() - wall_start

    # Convert Ollama nanosecond counters to human metrics
    load_ms = res_json.get("load_duration", 0) / 1e6
    prefill_tokens = res_json.get("prompt_eval_count", 0)
    prefill_cached_tokens = res_json.get("prompt_eval_cached_count", 0)
    prefill_ns = res_json.get("prompt_eval_duration", 1)
    prefill_ms = prefill_ns / 1e6
    prefill_tok_per_sec = (prefill_tokens / (prefill_ns / 1e9)) if prefill_ns > 0 else 0.0

    decode_tokens = res_json.get("eval_count", 0)
    decode_ns = res_json.get("eval_duration", 1)
    decode_ms = decode_ns / 1e6
    decode_tok_per_sec = (decode_tokens / (decode_ns / 1e9)) if decode_ns > 0 else 0.0
    tpot_ms = (decode_ms / decode_tokens) if decode_tokens > 0 else 0.0

    # TTFT = Load Duration + Prompt Eval Duration (prefill)
    ttft_ms = load_ms + prefill_ms

    ps_info = get_loaded_model_info()

    return {
        "model": model,
        "ps_info": ps_info,
        "wall_duration_s": wall_duration,
        "load_ms": load_ms,
        "prefill_tokens": prefill_tokens,
        "prefill_cached_tokens": prefill_cached_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tok_per_sec": prefill_tok_per_sec,
        "decode_tokens": decode_tokens,
        "decode_ms": decode_ms,
        "decode_tok_per_sec": decode_tok_per_sec,
        "tpot_ms": tpot_ms,
        "ttft_ms": ttft_ms,
        "response_text": res_json.get("response", "").strip(),
    }


def print_profile_report(metrics: Dict[str, Any]):
    print("=" * 82)
    print(f"  OLLAMA INFERENCE TELEMETRY REPORT: {metrics['model']}")
    print("=" * 82)

    ps = metrics["ps_info"]
    print(f"Runner Engine:        {ps.get('runner', 'llamacpp')} ({ps.get('processor', 'GPU')})")
    print(f"Resident VRAM Footprint: {ps.get('size', 'N/A')}")
    print(f"Model Load Latency:   {metrics['load_ms']:.2f} ms")
    print("-" * 82)

    print("--- 1. PREFILL PHASE (Prompt Evaluation / Compute Bound) ---")
    print(f"  • Prompt Tokens:             {metrics['prefill_tokens']} tokens")
    print(f"  • Cached Prefix Tokens:      {metrics['prefill_cached_tokens']} tokens")
    print(f"  • Prefill Latency:           {metrics['prefill_ms']:.2f} ms")
    print(f"  • Prefill Throughput:        {metrics['prefill_tok_per_sec']:.2f} tokens/sec")
    print(f"  • TTFT (Time To First Token):{metrics['ttft_ms']:.2f} ms")
    print("-" * 82)

    print("--- 2. DECODE PHASE (Autoregressive Generation / Memory Bandwidth Bound) ---")
    print(f"  • Generated Tokens:          {metrics['decode_tokens']} tokens")
    print(f"  • Generation Latency:        {metrics['decode_ms']:.2f} ms ({metrics['decode_ms']/1000.0:.2f} s)")
    print(f"  • Decode Throughput:         {metrics['decode_tok_per_sec']:.2f} tokens/sec")
    print(f"  • TPOT (Inter-Token Latency):{metrics['tpot_ms']:.2f} ms/token")
    print("-" * 82)

    ratio = metrics['prefill_tok_per_sec'] / max(metrics['decode_tok_per_sec'], 0.1)
    print(f"Key Inference Insight: Prefill is {ratio:.1f}x faster than Decode because GEMM")
    print("is compute-bound while single-stream decode is memory-bandwidth bound!")
    print("=" * 82)
    print(f"\nModel Output Snippet:\n{metrics['response_text'][:250]}...\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile Ollama Model Inference")
    parser.add_argument("--model", type=str, default="bonsai-2:latest", help="Ollama model name")
    parser.add_argument("--tokens", type=int, default=60, help="Max output tokens to generate")
    args = parser.parse_args()

    print(f"Profiling {args.model} via Ollama...")
    res = profile_generation(model=args.model, max_tokens=args.tokens)
    print_profile_report(res)
