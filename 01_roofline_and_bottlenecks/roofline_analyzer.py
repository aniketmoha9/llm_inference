#!/usr/bin/env python3
"""
Hardware Roofline & Inference Bottleneck Analyzer.
Calculates arithmetic intensity, prefill vs. decode characteristics,
and theoretical limits across modern hardware platforms (Apple Silicon, NVIDIA GPUs).
"""

from dataclasses import dataclass
from typing import Dict, List


@dataclass
class HardwareProfile:
    name: str
    peak_tflops_fp16: float       # Peak FP16 compute (Tensor / Vector)
    memory_bandwidth_gb_s: float  # DRAM/HBM/Unified memory bandwidth (GB/s)
    memory_capacity_gb: float     # Total VRAM or Unified Memory

    @property
    def machine_balance(self) -> float:
        """Critical arithmetic intensity (FLOPs / Byte)."""
        return (self.peak_tflops_fp16 * 1e12) / (self.memory_bandwidth_gb_s * 1e9)


HARDWARE_DATABASE: Dict[str, HardwareProfile] = {
    "apple_m_base": HardwareProfile("Apple M-Series (Base)", peak_tflops_fp16=15.0, memory_bandwidth_gb_s=150.0, memory_capacity_gb=16.0),
    "apple_m_max": HardwareProfile("Apple M-Series (Max)", peak_tflops_fp16=40.0, memory_bandwidth_gb_s=400.0, memory_capacity_gb=64.0),
    "apple_m_ultra": HardwareProfile("Apple M-Series (Ultra)", peak_tflops_fp16=80.0, memory_bandwidth_gb_s=800.0, memory_capacity_gb=128.0),
    "rtx_4090": HardwareProfile("NVIDIA RTX 4090", peak_tflops_fp16=165.0, memory_bandwidth_gb_s=1008.0, memory_capacity_gb=24.0),
    "nvidia_a100_80gb": HardwareProfile("NVIDIA A100 (80GB SXM4)", peak_tflops_fp16=312.0, memory_bandwidth_gb_s=2039.0, memory_capacity_gb=80.0),
    "nvidia_h100_sxm5": HardwareProfile("NVIDIA H100 (SXM5)", peak_tflops_fp16=989.0, memory_bandwidth_gb_s=3350.0, memory_capacity_gb=80.0),
    "nvidia_b200": HardwareProfile("NVIDIA B200", peak_tflops_fp16=2250.0, memory_bandwidth_gb_s=8000.0, memory_capacity_gb=192.0),
}


@dataclass
class ModelSpec:
    name: str
    params_billion: float
    num_layers: int
    hidden_dim: int
    num_heads: int
    num_kv_heads: int  # For GQA/MQA support
    head_dim: int

    def weight_bytes(self, precision_bits: int) -> float:
        bytes_per_param = precision_bits / 8.0
        return self.params_billion * 1e9 * bytes_per_param


LLAMA_3_8B = ModelSpec(
    name="LLaMA-3-8B",
    params_billion=8.03,
    num_layers=32,
    hidden_dim=4096,
    num_heads=32,
    num_kv_heads=8,  # GQA ratio = 4
    head_dim=128,
)

LLAMA_3_70B = ModelSpec(
    name="LLaMA-3-70B",
    params_billion=70.6,
    num_layers=80,
    hidden_dim=8192,
    num_heads=64,
    num_kv_heads=8,  # GQA ratio = 8
    head_dim=128,
)


def analyze_inference_step(
    hw: HardwareProfile,
    model: ModelSpec,
    batch_size: int,
    seq_len: int,
    precision_bits: int = 16,
    is_prefill: bool = False,
) -> Dict[str, float]:
    """
    Computes arithmetic intensity and execution bounds for a single forward pass.
    """
    bytes_per_elem = precision_bits / 8.0
    model_weight_bytes = model.weight_bytes(precision_bits)
    
    tokens = seq_len if is_prefill else 1
    total_tokens = batch_size * tokens

    # Linear layer FLOPs: ~2 FLOPs per parameter per token
    compute_flops = 2.0 * (model.params_billion * 1e9) * total_tokens

    # KV Cache bytes read during decode:
    # 2 (K and V) * num_layers * num_kv_heads * head_dim * seq_len * batch_size * bytes_per_elem
    kv_cache_bytes = 2 * model.num_layers * model.num_kv_heads * model.head_dim * seq_len * batch_size * bytes_per_elem

    # In decode, weights are read once per step, KV cache is read once
    total_bytes_transferred = model_weight_bytes + kv_cache_bytes

    # Arithmetic intensity (FLOPs / Byte)
    arithmetic_intensity = compute_flops / max(total_bytes_transferred, 1.0)

    # Time estimates
    time_compute_bound_s = compute_flops / (hw.peak_tflops_fp16 * 1e12)
    time_memory_bound_s = total_bytes_transferred / (hw.memory_bandwidth_gb_s * 1e9)
    actual_time_s = max(time_compute_bound_s, time_memory_bound_s)

    bottleneck = "Compute-Bound" if time_compute_bound_s > time_memory_bound_s else "Memory-Bandwidth Bound"

    tokens_per_sec = total_tokens / actual_time_s

    return {
        "tokens": total_tokens,
        "compute_gflops": compute_flops / 1e9,
        "memory_transferred_mb": total_bytes_transferred / (1024 * 1024),
        "arithmetic_intensity": arithmetic_intensity,
        "machine_balance": hw.machine_balance,
        "time_ms": actual_time_s * 1000.0,
        "tokens_per_sec": tokens_per_sec,
        "bottleneck": bottleneck,
    }


def print_comparison_table():
    print("=" * 88)
    print("  LLM INFERENCE ROOFLINE & BOTTLENECK ANALYSIS: LLaMA-3-8B (FP16 vs INT4)")
    print("=" * 88)

    model = LLAMA_3_8B

    print("\n--- 1. SINGLE-STREAM DECODE PHASE (Batch Size = 1, Generating 1 Token at Context = 2048) ---")
    header = f"{'Hardware':<26} | {'Precision':<9} | {'Intensity':<11} | {'Bottleneck':<15} | {'Speed (tok/s)':<13}"
    print(header)
    print("-" * len(header))

    for hw_key, hw in HARDWARE_DATABASE.items():
        for prec, prec_label in [(16, "FP16"), (4, "INT4")]:
            res = analyze_inference_step(hw, model, batch_size=1, seq_len=2048, precision_bits=prec, is_prefill=False)
            print(f"{hw.name:<26} | {prec_label:<9} | {res['arithmetic_intensity']:<11.2f} | {res['bottleneck']:<15} | {res['tokens_per_sec']:<13.1f}")
        print("-" * len(header))

    print("\n--- 2. PREFILL PHASE VS DECODE PHASE (Apple M-Series vs NVIDIA H100) ---")
    print(f"{'Platform':<24} | {'Phase':<12} | {'Tokens':<8} | {'Intensity (FLOP/B)':<20} | {'Bottleneck':<16} | {'Time (ms)':<10}")
    print("-" * 96)

    selected_hw = [HARDWARE_DATABASE["apple_m_base"], HARDWARE_DATABASE["nvidia_h100_sxm5"]]
    for hw in selected_hw:
        # Prefill: 1024 tokens
        prefill_res = analyze_inference_step(hw, model, batch_size=1, seq_len=1024, precision_bits=16, is_prefill=True)
        print(f"{hw.name:<24} | {'Prefill':<12} | {1024:<8} | {prefill_res['arithmetic_intensity']:<20.2f} | {prefill_res['bottleneck']:<16} | {prefill_res['time_ms']:<10.2f}")

        # Decode: 1 token (context 1024)
        decode_res = analyze_inference_step(hw, model, batch_size=1, seq_len=1024, precision_bits=16, is_prefill=False)
        print(f"{hw.name:<24} | {'Decode':<12} | {1:<8} | {decode_res['arithmetic_intensity']:<20.2f} | {decode_res['bottleneck']:<16} | {decode_res['time_ms']:<10.2f}")
        print("-" * 96)

    print("\n--- 3. BATCHING EFFECT: AMORTIZING WEIGHT READS ---")
    print("How batching transitions Decode from Memory-Bound to Compute-Bound on NVIDIA H100:")
    print(f"{'Batch Size':<12} | {'Intensity (FLOP/B)':<20} | {'Bottleneck':<16} | {'Total tok/s':<14} | {'TPOT (ms/token)':<15}")
    print("-" * 80)
    hw_h100 = HARDWARE_DATABASE["nvidia_h100_sxm5"]
    for b in [1, 4, 16, 64, 128, 256]:
        b_res = analyze_inference_step(hw_h100, model, batch_size=b, seq_len=1024, precision_bits=16, is_prefill=False)
        tpot_ms = b_res['time_ms'] # per batch step
        print(f"{b:<12} | {b_res['arithmetic_intensity']:<20.2f} | {b_res['bottleneck']:<16} | {b_res['tokens_per_sec']:<14.1f} | {tpot_ms:<15.2f}")
    print("=" * 88)


if __name__ == "__main__":
    print_comparison_table()
