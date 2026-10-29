#!/usr/bin/env python3
"""
Master Interactive Runner for LLM Inference Optimization Experiments.
Execute individual modules or run the complete benchmark suite.
"""

import os
import subprocess
import sys

MODULES = [
    ("1. Hardware Roofline & Bottleneck Analyzer", "01_roofline_and_bottlenecks/roofline_analyzer.py"),
    ("2. Naive vs. KV-Cached Generation Benchmark", "02_kv_cache_mechanics/naive_vs_kvcache_bench.py"),
    ("3. Attention Architectures (MHA, MQA, GQA, DeepSeek MLA)", "02_kv_cache_mechanics/attention_architectures.py"),
    ("4. PagedAttention Memory Manager & Fragmentation Simulator", "03_paged_attention_and_batching/paged_attention_simulator.py"),
    ("5. PagedAttention vs. Contiguous Attention Kernel Benchmark", "03_paged_attention_and_batching/paged_attention_kernel_bench.py"),
    ("6. Continuous Batching vs. Static Batching Simulator", "03_paged_attention_and_batching/continuous_batching_sim.py"),
    ("7. Weight Quantization From Scratch (INT8, INT4, Group-Wise AWQ)", "04_quantization/quantization_from_scratch.py"),
    ("8. Speculative Decoding Engine & Rejection Sampling", "05_speculative_decoding/speculative_engine.py"),
    ("9. Live Ollama Telemetry Profiler (bonsai-2:latest)", "07_hands_on_ollama/ollama_profiler.py"),
    ("10. Live Ollama Prefill vs. Decode Scaling Benchmark", "07_hands_on_ollama/benchmark_prefill_vs_decode.py"),
    ("11. Live Ollama KV Cache & Context Reuse Test", "07_hands_on_ollama/test_prefix_caching.py"),
    ("12. Live Multimodal Vision Inference (bonsai-2:latest)", "07_hands_on_ollama/multimodal_vision_inference.py"),
    ("13. Live Multimodal Audio Inference (nemotron-3.5-bolt:e4b-mxfp8)", "07_hands_on_ollama/audio_inference_benchmark.py"),
    ("14. Launch Mobile Web Chat Server & Dashboard", "08_remote_serving/web_chat_server.py"),
]


def run_script(rel_path: str):
    py_exec = sys.executable
    script_path = os.path.join(os.path.dirname(__file__), rel_path)
    print(f"\n>>> Executing: {rel_path} with {py_exec}\n")
    res = subprocess.run([py_exec, script_path])
    if res.returncode != 0:
        print(f"\n[Warning] {rel_path} exited with code {res.returncode}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        print("Running full LLM Inference Optimization experiment suite...\n")
        for title, path in MODULES:
            print("\n" + "#" * 80)
            print(f"  {title}")
            print("#" * 80)
            run_script(path)
        print("\nAll experiments completed successfully!")
        return

    print("=" * 70)
    print("  HIGH-PERFORMANCE LLM INFERENCE OPTIMIZATION SUITE")
    print("=" * 70)
    print("Select an experiment to run:\n")
    for i, (title, _) in enumerate(MODULES, 1):
        print(f"  [{i}] {title}")
    print("  [A] Run All Experiments")
    print("  [Q] Quit")
    print("=" * 70)

    choice = input("\nEnter choice (1-7, A, Q): ").strip().upper()
    if choice == "Q":
        return
    elif choice == "A":
        for title, path in MODULES:
            run_script(path)
    elif choice.isdigit() and 1 <= int(choice) <= len(MODULES):
        _, path = MODULES[int(choice) - 1]
        run_script(path)
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()
