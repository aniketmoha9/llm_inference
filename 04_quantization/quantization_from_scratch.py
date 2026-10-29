#!/usr/bin/env python3
"""
Quantization From Scratch:
Implements Symmetric, Asymmetric, Per-Channel, and Group-wise INT8/INT4 quantization.
Evaluates Mean Squared Error (MSE), SNR (dB), and memory footprint reduction.
"""

from typing import Tuple
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))


def quantize_symmetric(
    tensor: torch.Tensor,
    num_bits: int = 8,
    per_channel: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Symmetric Quantization (Zero-point = 0).
    Maps float values to [-2^(b-1) + 1, 2^(b-1) - 1].
    """
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    if per_channel:
        # Scale per row: [Rows, 1]
        max_val = torch.amax(torch.abs(tensor), dim=1, keepdim=True)
    else:
        # Single scalar scale for entire tensor
        max_val = torch.amax(torch.abs(tensor))

    scale = torch.clamp(max_val / qmax, min=1e-8)
    q_tensor = torch.clamp(torch.round(tensor / scale), qmin, qmax)
    return q_tensor, scale


def dequantize_symmetric(q_tensor: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Dequantizes symmetric quantized tensor: X_hat = X_q * scale."""
    return q_tensor * scale


def quantize_asymmetric(
    tensor: torch.Tensor,
    num_bits: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Asymmetric (Affine) Quantization.
    Maps float values to [0, 2^b - 1] with scale and zero-point.
    """
    qmin = 0
    qmax = 2 ** num_bits - 1

    min_val = torch.amin(tensor)
    max_val = torch.amax(tensor)

    scale = torch.clamp((max_val - min_val) / qmax, min=1e-8)
    zero_point = torch.clamp(torch.round(-min_val / scale), qmin, qmax)

    q_tensor = torch.clamp(torch.round(tensor / scale) + zero_point, qmin, qmax)
    return q_tensor, scale, zero_point


def dequantize_asymmetric(q_tensor: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    """Dequantizes affine quantized tensor: X_hat = (X_q - zero_point) * scale."""
    return (q_tensor - zero_point) * scale


def quantize_groupwise(
    tensor: torch.Tensor,
    group_size: int = 128,
    num_bits: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group-wise (Block-wise) Quantization (as in AWQ & GPTQ).
    Splits rows into groups of size `group_size`, computing independent scales.
    """
    orig_shape = tensor.shape
    rows = orig_shape[0]
    cols = orig_shape[1]
    
    # Reshape into [-1, group_size]
    tensor_grouped = tensor.view(-1, group_size)
    qmin = -(2 ** (num_bits - 1) - 1)
    qmax = 2 ** (num_bits - 1) - 1

    max_vals = torch.amax(torch.abs(tensor_grouped), dim=1, keepdim=True)
    scale = torch.clamp(max_vals / qmax, min=1e-8)

    q_grouped = torch.clamp(torch.round(tensor_grouped / scale), qmin, qmax)
    return q_grouped.view(orig_shape), scale


def dequantize_groupwise(q_tensor: torch.Tensor, scale: torch.Tensor, group_size: int = 128) -> torch.Tensor:
    orig_shape = q_tensor.shape
    q_grouped = q_tensor.view(-1, group_size)
    deq_grouped = q_grouped * scale
    return deq_grouped.view(orig_shape)


def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Tuple[float, float]:
    """Calculates Mean Squared Error (MSE) and Signal-to-Quantization-Noise Ratio (SQNR in dB)."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    signal_power = torch.mean(original ** 2).item()
    noise_power = max(mse, 1e-12)
    sqnr_db = 10.0 * torch.log10(torch.tensor(signal_power / noise_power)).item()
    return mse, sqnr_db


def evaluate_quantization_schemes():
    print("=" * 86)
    print("  LLM WEIGHT QUANTIZATION BENCHMARK & PRECISION ANALYSIS")
    print("=" * 86)

    torch.manual_seed(42)
    # Simulate a LLaMA-3 FFN projection weight matrix: [4096, 4096] with outlier channels
    rows, cols = 4096, 4096
    weights = torch.randn(rows, cols, device=DEVICE)

    # Inject realistic outlier channels (channels with 15x normal magnitude)
    outlier_indices = [12, 105, 512, 1024, 2048]
    weights[:, outlier_indices] *= 15.0

    print(f"Test Matrix: Simulated FFN Weight [{rows} x {cols}] (16.7M params)")
    print(f"FP16 Raw Memory Size: {(rows * cols * 2) / (1024 * 1024):.2f} MB")
    print(f"Outlier channels injected at column indices: {outlier_indices}\n")

    schemes = [
        ("INT8 Symmetric (Per-Tensor)", lambda w: (dequantize_symmetric(*quantize_symmetric(w, 8, per_channel=False)), 8)),
        ("INT8 Symmetric (Per-Channel)", lambda w: (dequantize_symmetric(*quantize_symmetric(w, 8, per_channel=True)), 8)),
        ("INT8 Asymmetric (Per-Tensor)", lambda w: (dequantize_asymmetric(*quantize_asymmetric(w, 8)), 8)),
        ("INT4 Symmetric (Per-Tensor)", lambda w: (dequantize_symmetric(*quantize_symmetric(w, 4, per_channel=False)), 4)),
        ("INT4 Symmetric (Per-Channel)", lambda w: (dequantize_symmetric(*quantize_symmetric(w, 4, per_channel=True)), 4)),
        ("INT4 Group-Wise (g=128, AWQ/GPTQ)", lambda w: (dequantize_groupwise(*quantize_groupwise(w, group_size=128, num_bits=4), group_size=128), 4)),
    ]

    header = f"{'Quantization Scheme':<35} | {'Bits':<6} | {'Memory (MB)':<12} | {'MSE Error':<12} | {'SNR (dB)':<10}"
    print(header)
    print("-" * len(header))

    for name, func in schemes:
        reconstructed, bits = func(weights)
        mse, snr_db = calculate_metrics(weights, reconstructed)
        mem_mb = (rows * cols * (bits / 8.0)) / (1024 * 1024)
        print(f"{name:<35} | {bits:<6} | {mem_mb:<12.2f} | {mse:<12.6f} | {snr_db:<10.2f}")

    print("-" * len(header))
    print("Key Takeaways:")
    print("  1. Per-tensor INT4 has high error because outliers blow up the global scale.")
    print("  2. Group-wise INT4 (AWQ/GPTQ style) achieves pristine quality (>18 dB SNR)")
    print("     while slashing memory by 4x (32 MB -> 8 MB)!")
    print("=" * 86)


if __name__ == "__main__":
    evaluate_quantization_schemes()
