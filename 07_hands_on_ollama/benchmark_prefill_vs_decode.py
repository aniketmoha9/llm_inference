#!/usr/bin/env python3
"""
Prefill vs. Decode Scaling Benchmark with Live Ollama Models.
Demonstrates how prompt length linearly scales prefill compute time,
while decode latency per token (TPOT) remains virtually flat.
"""

import json
import urllib.request
from typing import Dict, List


OLLAMA_URL = "http://localhost:11434/api/generate"


def run_prompt_test(model: str, prompt_text: str, max_tokens: int = 25) -> Dict[str, float]:
    payload = {
        "model": model,
        "prompt": prompt_text,
        "stream": False,
        "options": {
            "num_predict": max_tokens,
            "temperature": 0.2,
        }
    }
    req = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    prompt_tokens = data.get("prompt_eval_count", 0)
    prefill_ms = data.get("prompt_eval_duration", 1) / 1e6
    prefill_tps = (prompt_tokens / (prefill_ms / 1000.0)) if prefill_ms > 0 else 0

    decode_tokens = data.get("eval_count", 0)
    decode_ms = data.get("eval_duration", 1) / 1e6
    decode_tps = (decode_tokens / (decode_ms / 1000.0)) if decode_ms > 0 else 0
    tpot_ms = (decode_ms / decode_tokens) if decode_tokens > 0 else 0

    return {
        "prompt_tokens": prompt_tokens,
        "prefill_ms": prefill_ms,
        "prefill_tps": prefill_tps,
        "decode_tokens": decode_tokens,
        "decode_ms": decode_ms,
        "decode_tps": decode_tps,
        "tpot_ms": tpot_ms,
    }


def benchmark_scaling(model: str = "bonsai-2:latest"):
    print("=" * 86)
    print(f"  PREFILL vs. DECODE SCALING BENCHMARK ON OLLAMA ({model})")
    print("=" * 86)

    # Base sentence repeated to synthesize increasing prompt lengths
    base_snippet = (
        "In modern machine learning accelerators, memory bandwidth determines the throughput "
        "of autoregressive decoding, whereas parallel floating point units govern prompt evaluation. "
    )

    prompt_multipliers = [1, 5, 12, 25]
    results = []

    print(f"Testing {len(prompt_multipliers)} prompt sizes while holding output tokens fixed at ~25...\n")
    header = f"{'Prompt Tokens':<14} | {'Prefill (ms)':<14} | {'Prefill tok/s':<15} | {'Decode tok/s':<14} | {'TPOT (ms/tok)':<14}"
    print(header)
    print("-" * len(header))

    for m in prompt_multipliers:
        prompt_str = base_snippet * m + "\nSummarize the above key architectural bottleneck."
        res = run_prompt_test(model=model, prompt_text=prompt_str, max_tokens=25)
        results.append(res)
        print(
            f"{res['prompt_tokens']:<14} | "
            f"{res['prefill_ms']:<14.2f} | "
            f"{res['prefill_tps']:<15.1f} | "
            f"{res['decode_tps']:<14.1f} | "
            f"{res['tpot_ms']:<14.2f}"
        )

    print("-" * len(header))
    print("\nCore Architectural Observations:")
    print("  1. Prefill time scales with prompt length because it evaluates all input tokens.")
    print("  2. Decode speed (tok/s) and TPOT remain remarkably STABLE regardless of prompt size!")
    print("     Why? Because generating token n requires streaming the model weights once,")
    print("     which takes a fixed time governed by memory bandwidth, not prompt compute!")
    print("=" * 86)


if __name__ == "__main__":
    benchmark_scaling()
