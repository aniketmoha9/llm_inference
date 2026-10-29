# High-Performance LLM Inference & Optimization Engineering

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hardware Support](https://img.shields.io/badge/hardware-CUDA%20%7C%20Apple%20MPS%20%7C%20CPU-success.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end, code-first curriculum and benchmarking suite for understanding, optimizing, and serving Large Language Models (LLMs).

This repository bridges low-level hardware constraints (memory bandwidth, arithmetic intensity, SRAM vs HBM/DRAM) with state-of-the-art inference algorithms (KV Caching, GQA, DeepSeek MLA, PagedAttention), kernel-level principles (FlashAttention), inference scheduling (Continuous Batching, Chunked Prefill), quantization (INT8, INT4, AWQ), speculative decoding, and production serving architectures (vLLM, SGLang, TensorRT-LLM). It also includes a private streaming web server with real-time hardware telemetry and multimodal vision & audio inference.

---

## 🗺️ Master Curriculum & Inference Architecture

```
                                  [ User Request: Prompt Tokens ]
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │        Prefill Phase (Prompt)         │
                             │  • High Arithmetic Intensity (FLOP/B) │
                             │  • Compute-Bound (GEMM)               │
                             │  • Chunked Prefill / FlashAttention   │
                             └──────────────────┬────────────────────┘
                                                │
                                    [ Key & Value Tensors ]
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │       KV Cache Memory Manager         │
                             │  • PagedAttention (vLLM Block Table)  │
                             │  • MQA / GQA / MLA (DeepSeek)         │
                             │  • KV Cache Quantization (FP8 / INT4) │
                             └──────────────────┬────────────────────┘
                                                │
                             ┌──────────────────┴────────────────────┐
                             ▼                                       ▼
             ┌───────────────────────────────┐       ┌───────────────────────────────┐
             │     Decode Phase (Token n)    │       │     Speculative Decoding      │
             │  • Low Arithmetic Intensity   │  OR   │  • Fast Draft Model (K tokens)│
             │  • Memory Bandwidth Bound     │       │  • Target Model Parallel Verif│
             │  • FlashDecoding / Gemv       │       │  • Acceptance Rejection Step  │
             └───────────────┬───────────────┘       └───────────────┬───────────────┘
                             │                                       │
                             └──────────────────┬────────────────────┘
                                                │
                                                ▼
                             ┌───────────────────────────────────────┐
                             │   Continuous (Iteration) Batcher      │
                             │  • Dynamic Insert / Evict per step    │
                             │  • Maximize GPU Utilization & TPOT    │
                             └──────────────────┬────────────────────┘
                                                │
                                                ▼
                                    [ Output Token Stream ]
```

---

## 📁 Repository Structure & Modules

| Module | Core Topic | Key Concepts | Hands-On Script |
| :--- | :--- | :--- | :--- |
| [`01_roofline_and_bottlenecks`](./01_roofline_and_bottlenecks/) | **Hardware & Bottlenecks** | Prefill vs Decode, Arithmetic Intensity, Roofline Model, Memory Bandwidth Bound vs Compute Bound | [`roofline_analyzer.py`](./01_roofline_and_bottlenecks/roofline_analyzer.py) |
| [`02_kv_cache_mechanics`](./02_kv_cache_mechanics/) | **KV Cache & Attention** | $O(S^2)$ vs $O(1)$ step scaling, MHA, MQA, GQA, DeepSeek Multi-Head Latent Attention (MLA) | [`naive_vs_kvcache_bench.py`](./02_kv_cache_mechanics/naive_vs_kvcache_bench.py)<br>[`attention_architectures.py`](./02_kv_cache_mechanics/attention_architectures.py) |
| [`03_paged_attention_and_batching`](./03_paged_attention_and_batching/) | **Memory & Serving Scheduling** | Virtual memory paging for KV cache, Block tables, Zero internal/external fragmentation, Continuous Batching | [`paged_attention_simulator.py`](./03_paged_attention_and_batching/paged_attention_simulator.py)<br>[`paged_attention_kernel_bench.py`](./03_paged_attention_and_batching/paged_attention_kernel_bench.py)<br>[`continuous_batching_sim.py`](./03_paged_attention_and_batching/continuous_batching_sim.py) |
| [`04_quantization`](./04_quantization/) | **Model Compression** | Symmetric / Asymmetric INT8/INT4, Scales & Zero-points, Outlier channel mitigation, Group-Wise AWQ | [`quantization_from_scratch.py`](./04_quantization/quantization_from_scratch.py) |
| [`05_speculative_decoding`](./05_speculative_decoding/) | **Speculative Generation** | Draft model proposal ($\gamma$ tokens), Parallel target verification, Lossless rejection sampling | [`speculative_engine.py`](./05_speculative_decoding/speculative_engine.py) |
| [`06_production_engines_and_systems`](./06_production_engines_and_systems/) | **Production Serving Architectures** | Deep architectural review of vLLM, SGLang (RadixAttention), TensorRT-LLM, Prefill-Decode Disaggregation (Mooncake) | [System Architecture Guide](./06_production_engines_and_systems/README.md) |
| [`07_hands_on_ollama`](./07_hands_on_ollama/) | **Live Profiling & Multimodal** | Telemetry extraction (`prompt_eval`, `eval`), TTFT vs TPOT, KV prefix reuse, Vision projection, Native Audio speech reasoning | [`ollama_profiler.py`](./07_hands_on_ollama/ollama_profiler.py)<br>[`benchmark_prefill_vs_decode.py`](./07_hands_on_ollama/benchmark_prefill_vs_decode.py)<br>[`test_prefix_caching.py`](./07_hands_on_ollama/test_prefix_caching.py)<br>[`multimodal_vision_inference.py`](./07_hands_on_ollama/multimodal_vision_inference.py)<br>[`audio_inference_benchmark.py`](./07_hands_on_ollama/audio_inference_benchmark.py) |
| [`08_remote_serving`](./08_remote_serving/) | **Host & Telemetry Dashboard** | Self-hosted private server, SSE streaming, Mobile-responsive chat, Live hardware telemetry dashboard, Audio transcoding | [`web_chat_server.py`](./08_remote_serving/web_chat_server.py)<br>[Hosting Guide](./08_remote_serving/README.md) |

---

## ⚡ Quick Start: Environment & Reproduction

### 1. Clone Repository & Setup Virtual Environment

```bash
git clone https://github.com/aniketmoha9/llm_inference.git
cd llm_inference

# Create and activate Python virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies (pure PyTorch + NumPy)
pip install -r requirements.txt
```

### 2. Device Portability
All benchmark kernels automatically inspect and select the fastest available accelerator:
- **Apple Silicon (M1/M2/M3/M4/M5)**: Accelerated via Metal Performance Shaders (`mps`).
- **NVIDIA GPUs**: Accelerated via `cuda`.
- **x86/ARM CPUs**: Clean fallback to multithreaded CPU computation.

---

## 🧪 Running the Master Experiment Suite

You can run the interactive CLI menu to pick and execute any benchmark:

```bash
python3 run_all_experiments.py
```

Or run all core experiments end-to-end non-interactively:

```bash
python3 run_all_experiments.py --all
```

### Running Individual Module Benchmarks:

```bash
# 1. Hardware Roofline & Arithmetic Intensity Analyzer
python3 01_roofline_and_bottlenecks/roofline_analyzer.py

# 2. KV Cache Quadratic O(S^2) vs O(1) Benchmark
python3 02_kv_cache_mechanics/naive_vs_kvcache_bench.py

# 3. Attention Architectures & Memory Scaling (MHA, MQA, GQA, MLA)
python3 02_kv_cache_mechanics/attention_architectures.py

# 4. PagedAttention Memory Manager & Fragmentation Simulator
python3 03_paged_attention_and_batching/paged_attention_simulator.py

# 5. PagedAttention Kernel vs. Contiguous Attention Benchmark
python3 03_paged_attention_and_batching/paged_attention_kernel_bench.py

# 6. Continuous Batching vs. Static Batching Simulation
python3 03_paged_attention_and_batching/continuous_batching_sim.py

# 7. Weight Quantization From Scratch (INT8, INT4, Group-Wise AWQ)
python3 04_quantization/quantization_from_scratch.py

# 8. Speculative Decoding & Rejection Sampling Engine
python3 05_speculative_decoding/speculative_engine.py
```

---

## 📊 Summary of Key Empirical Findings

### 1. Roofline Model & Operational Regimes
* **Decode Phase**: Arithmetic Intensity $\approx 0.98\text{--}1.0\text{ FLOP/Byte}$. Memory-bandwidth bound. Every single token requires loading the entire model weights into registers ($O(N)$ memory traffic for $O(1)$ tokens).
* **Prefill Phase**: Arithmetic Intensity $\approx 1000+\text{ FLOP/Byte}$. Compute-bound GEMM. Model weights are reused across all prompt tokens in parallel.

### 2. The KV Cache
* **Naive Recomputation**: Step latency scales quadratically ($O(S^2)$). At step 250, step latency increases by **$11.6\times$**.
* **With KV Cache**: Step latency stays completely flat at $O(1)$ (~$1.4\text{ ms/token}$).

### 3. Attention Architectures (MHA vs. GQA vs. DeepSeek MLA)
For a 128k context window:
* **Multi-Head Attention (MHA)**: Requires **64.0 GB** of VRAM per sequence for KV cache alone.
* **Grouped-Query Attention (GQA - 8 KV heads)**: Compresses KV cache to **16.0 GB** ($4\times$ reduction).
* **Multi-Head Latent Attention (DeepSeek MLA)**: Compresses Key-Value vectors into a low-rank latent subspace ($d_c = 512$), reducing KV cache to **4.5 GB** (**$14.2\times$ reduction** vs MHA).

### 4. PagedAttention & Block Tables
* **Contiguous Memory Pre-allocation**: Wastes **60–80%** of GPU memory due to internal fragmentation, reservation waste, and external fragmentation.
* **PagedAttention**: Divides KV cache into non-contiguous physical pages (block size = 16 tokens). Memory waste drops to **< 2%**.
* **Mathematical Parity**: Paged gather-scatter attention kernel matches contiguous causal attention with exact mathematical equivalence (`0.00000000e+00` output error) while delivering **$6.56\times$ higher concurrent sequence capacity** in the same VRAM budget.

### 5. Continuous Batching
* **Static Batching**: Bound by the slowest request in the batch; GPU cores idle waiting for long outputs.
* **Continuous (Iteration-level) Batching**: Ingests new requests and evicts finished requests at every token step. Delivers **$1.66\times$ higher throughput** and **$2.31\times$ lower average request latency**.

### 6. Quantization (Per-Tensor vs. Group-Wise AWQ)
* **INT8 Symmetric**: Achieves **$38.8\text{ dB}$ SNR** and **$4.0\times$ memory reduction**.
* **Per-Tensor INT4**: Suffers severe accuracy degradation (**$1.02\text{ dB}$ SNR**) when activation/weight outliers exist.
* **Group-Wise INT4 (Group size = 64, AWQ-style)**: Recovers signal fidelity to **$13.83\text{ dB}$ SNR** ($19.0\times$ lower MSE) while maintaining **$7.1\times$ memory compression**.

### 7. Speculative Decoding
* Employs a small, memory-efficient Draft Model ($\gamma=3\text{--}4$ tokens) and verifies proposals concurrently in a single forward pass of the Target Model.
* Lossless rejection sampling preserves the target distribution while achieving **$1.8\times\text{--}2.2\times$ effective wall-clock acceleration**.

---

## 🌐 Private Remote Hosting & Live Telemetry Dashboard

Module [`08_remote_serving`](./08_remote_serving/) turns your machine into a 24/7 private AI inference server accessible globally from your mobile phone or another laptop:

```bash
python3 08_remote_serving/web_chat_server.py
```

### Features:
1. **Chat UI (`/`)**: Mobile-first, ChatGPT-style responsive interface with real-time Server-Sent Events (SSE) token streaming.
2. **Live Telemetry Dashboard (`/dashboard`)**:
   - Live Decode Throughput (tokens/sec) & TPOT (inter-token latency in ms)
   - Prefill Latency & Time-To-First-Token (TTFT)
   - Real-time Unified Memory & VRAM allocation gauge
   - Active loaded model & execution runner (`llamacpp Metal` vs `mlx`)
   - Connected client categorization (Local Wi-Fi vs Remote Tailscale 5G/VPN)
   - Live activity table with modality indicators (`🎵 Audio`, `🖼️ Vision`, `📝 Text`)
3. **Multimodal Vision & Audio Ingestion**:
   - Automatic client-side canvas downscaling ($\le 1024$px) to prevent context exhaustion.
   - Built-in on-the-fly audio transcoder (`ensure_compatible_audio`) that ingests mobile voice memos (`.m4a`, `.mp3`, `.caf`, `.wav`) and downmixes to 16kHz mono WAV in ~15 ms before feeding to `sound_encoder`.

---

## 📜 License

This repository is licensed under the [MIT License](LICENSE).
