# Module 01: Hardware Bottlenecks, Prefill vs. Decode, & The Roofline Model

To optimize LLM inference, you must first understand the fundamental physical constraints imposed by modern computer architecture: **compute throughput** (FLOPs/s) vs. **memory bandwidth** (Bytes/s).

---

## 1. The Two Distinct Phases of Inference

Generating text from a modern Transformer is not a uniform operation. It is split into two fundamentally different regimes:

```
[Prompt: "Explain general relativity"] 
           │
           ▼
┌───────────────────────────────────────┐
│         1. Prefill Phase              │
│  • Input: N prompt tokens             │
│  • Parallel: all tokens processed     │
│    simultaneously in attention & FFN  │
│  • Matrix × Matrix multiplications    │  ===> COMPUTE-BOUND
│    (GEMM: [B, N, D] x [D, D])         │       High Arithmetic Intensity
│  • Metric: Time-to-First-Token (TTFT) │
└──────────────────┬────────────────────┘
                   │ First generated token + initial KV Cache
                   ▼
┌───────────────────────────────────────┐
│         2. Decode Phase               │
│  • Input: 1 new token at a time       │
│  • Sequential: autoregressive loop    │
│  • Matrix × Vector multiplications    │  ===> MEMORY-BANDWIDTH BOUND
│    (GEMV: [B, 1, D] x [D, D])         │       Low Arithmetic Intensity
│  • Metric: Time-per-Output-Token      │
│    (TPOT / Inter-Token Latency)       │
└───────────────────────────────────────┘
```

---

## 2. Arithmetic Intensity & The Roofline Model

The **Roofline Model** defines whether an operation's execution speed is bottlenecked by the processor's compute units (ALUs/Tensor Cores) or by the memory bus streaming data from DRAM/HBM.

### Mathematical Formulation

$$\text{Arithmetic Intensity } (I) = \frac{\text{Floating Point Operations (FLOPs)}}{\text{Memory Transferred (Bytes)}}$$

Hardware has a **Machine Balance** / critical intensity:
$$I^* = \frac{\text{Peak Compute Performance (TFLOPs/s)}}{\text{Peak Memory Bandwidth (TB/s)}}$$

- If $I < I^*$: **Memory-Bandwidth Bound**. The ALUs spend most of their time idle, waiting for bytes to arrive from DRAM/HBM. Increasing compute speed won't help; only reducing memory transfers or increasing memory bandwidth will.
- If $I > I^*$: **Compute Bound**. The memory bus can feed data fast enough; execution time is determined purely by how fast the ALUs can multiply and add.

### Why Decode is Memory-Bandwidth Bound (GEMV)

Consider generating **1 single token** ($N=1$) with a batch size $B=1$ on a 7B parameter model in 16-bit precision ($\text{FP16} = 2\text{ bytes/parameter}$):

1. **Weights loaded from memory**:
   $$\text{Bytes} = 7 \times 10^9 \text{ parameters} \times 2 \text{ bytes} = 14 \text{ GB}$$
2. **Operations performed**:
   Each parameter undergoes 1 multiply and 1 accumulate per token ($2\text{ FLOPs/param}$):
   $$\text{FLOPs} = 2 \times 7 \times 10^9 = 14 \text{ GFLOPs}$$
3. **Arithmetic Intensity**:
   $$I = \frac{14 \text{ GFLOPs}}{14 \text{ GB}} = 1.0 \text{ FLOP / Byte}$$

Compare this to the hardware machine balance of an **NVIDIA H100 (SXM5)**:
- Peak FP16 Compute: $\approx 1000 \text{ TFLOPs/s}$ (dense)
- Peak HBM3 Bandwidth: $\approx 3.35 \text{ TB/s}$
$$I^* = \frac{1000 \times 10^{12}}{3.35 \times 10^{12}} \approx 298.5 \text{ FLOPs / Byte}$$

Because $I_{\text{decode}} = 1 \ll 298.5$, **the GPU compute cores are >99% idle during single-stream decode!** 

### Theoretical Single-Stream Decode Ceiling

$$\text{Max Tokens/sec} \le \frac{\text{Memory Bandwidth (Bytes/s)}}{\text{Model Parameters} \times \text{Bytes per parameter}}$$

| Hardware | Memory Bandwidth | 8B Model (FP16: 16 GB) | 8B Model (INT4: 4 GB) | 70B Model (INT4: 35 GB) |
| :--- | :--- | :--- | :--- | :--- |
| **Apple M-Series (Base/Pro)** | ~150 GB/s | 9.3 tok/s | 37.5 tok/s | 4.2 tok/s |
| **Apple M-Series (Max)** | ~400 GB/s | 25.0 tok/s | 100.0 tok/s | 11.4 tok/s |
| **NVIDIA A100 (80GB SXM4)** | 2,039 GB/s | 127.4 tok/s | 509.7 tok/s | 58.2 tok/s |
| **NVIDIA H100 (80GB SXM5)** | 3,350 GB/s | 209.3 tok/s | 837.5 tok/s | 95.7 tok/s |

> [!IMPORTANT]
> This single equation reveals why **Quantization (INT8 / INT4 / FP8)** makes LLMs run so much faster during the decode phase: it cuts the bytes loaded per token by $2\times$ or $4\times$, directly multiplying token generation speed!

---

## 3. Key Inference Metrics

When evaluating and optimizing inference servers, four metrics govern performance:

1. **TTFT (Time To First Token)**: Latency from request arrival to emitting token 1 (governed by the Prefill phase). Crucial for perceived responsiveness in chat.
2. **TPOT (Time Per Output Token)**: Latency per subsequent token (governed by the Decode phase). Needs to match or exceed human reading speed (~15–25 ms/tok).
3. **Throughput (Tokens/sec)**: Total number of tokens generated across all concurrent users per second. Governs server cost and infrastructure efficiency.
4. **Latency vs. Throughput Pareto Curve**: Increasing batch size increases total throughput (by amortizing weight reads across multiple queries), but past a saturation point, queuing delays explode TPOT and TTFT.
