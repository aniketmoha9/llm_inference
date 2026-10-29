# Module 07: Hands-On Profiling with Your Ollama Models

This module connects the inference optimization concepts directly to the real models downloaded in your local Ollama environment on your Apple M5 Mac (16 GB Unified Memory).

---

## 1. Your Local Model Portfolio

Running `ollama list` reveals 4 distinct models spanning architectures, sizes, and quantization levels:

| Model | Arch | Parameters | Quantization | Runner | VRAM Size | Purpose in Experiments |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `bonsai-2:latest` | Qwen 3.5 | **9.0B** | **Q2_0** (2-bit) | `llamacpp` (Metal) | **2.5 GB** | Extreme 2-bit quantization, full completion/chat |
| `nemotron-3.5-bolt:e4b-mxfp8` | Gemma 4 | **9.0B** | **mxfp8** (8-bit FP) | `mlx` (Apple Native) | **11.0 GB** | High-precision 8-bit, full completion/chat |
| `d3:latest` | Qwen 3.5 | **4.5B** | **mxfp8** (8-bit FP) | `mlx` | **5.6 GB** | Decision & vision classifier |
| `tev1:0.8b` | Qwen 3.5 | **0.75B** (752M) | **BF16** (16-bit) | `mlx` | **1.5 GB** | Ultra-compact unquantized classifier |

---

## 2. Key Hands-On Discoveries

### Discovery A: The Quantization Footprint Miracle (9B Q2_0 vs 9B mxfp8)
Notice that **both `bonsai-2` and `nemotron-3.5-bolt` have 9 Billion parameters**:
- `nemotron-3.5-bolt` (mxfp8 = 8 bits / 1 byte per parameter) takes **11 GB** of your 16 GB unified RAM.
- `bonsai-2` (Q2_0 = 2 bits / 0.25 bytes per parameter) takes only **2.5 GB**!
- That is a **4.4× reduction in memory** for the exact same parameter count, allowing a 9B model to easily run on low-memory hardware!

### Discovery B: Prefill vs. Decode Throughput in Real Time
Ollama exposes precise hardware performance counters in its API responses:
- `prompt_eval_count` & `prompt_eval_duration`: Nanoseconds spent in **Prefill Phase** (Prompt processing).
- `eval_count` & `eval_duration`: Nanoseconds spent in **Decode Phase** (Token generation).

In our profiling scripts, you will observe:
1. **Prefill speed**: Can reach **100–300+ tokens/sec** because all prompt tokens are processed concurrently via parallel matrix multiplications (GEMM).
2. **Decode speed**: Drops to **30–50 tokens/sec** because every single generated token requires streaming the entire model's weights across the unified memory bus (GEMV).

### Discovery C: Prompt / Prefix Caching Live
When you send a prompt with a large common prefix (e.g., a system prompt or a long article):
- **Call 1 (Cold cache)**: `prompt_eval_cached_count: 0`. The entire prompt is prefilled.
- **Call 2 (Warm cache)**: `prompt_eval_cached_count: 1000`. Ollama reuses the cached KV tensors from memory, reducing TTFT to almost **0 ms**!

---

## 3. Hands-On Profiling Tools in this Module

1. [`ollama_profiler.py`](./ollama_profiler.py):
   Profiles any model with custom prompts, computing TTFT, TPOT, Prefill tok/s, Decode tok/s, and memory usage.

2. [`benchmark_prefill_vs_decode.py`](./benchmark_prefill_vs_decode.py):
   Sweeps prompt length (50 to 1,000 tokens) to demonstrate how prefill scaling differs fundamentally from decode scaling.

3. [`test_prefix_caching.py`](./test_prefix_caching.py):
   Demonstrates KV cache prefix reuse in Ollama in real time.

4. [`multimodal_vision_inference.py`](./multimodal_vision_inference.py):
   Synthesizes a test image programmatically (pure Python, zero dependencies) and sends it through Ollama's vision projection layer (`bonsai-2:latest` or `nemotron-3.5-bolt`).

5. [`audio_inference_benchmark.py`](./audio_inference_benchmark.py):
   Evaluates native multimodal speech and acoustic reasoning using `nemotron-3.5-bolt:e4b-mxfp8`'s `sound_encoder`. Measures audio duration, audio prefill tokenization rate, and end-to-end question answering.
